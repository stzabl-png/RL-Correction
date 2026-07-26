"""让 DexMate 的手臂 + Sharpa 手用 IK 跟踪重建轨迹.

查看器(env_viewer --follow)和录像脚本(record_replay)共用这份实现,
避免两边各写一套后行为漂移。

驱动方式:
  臂  参考腕位姿 -> 机器人根系 -> DifferentialIKController(DLS) -> R/L_arm_j1..j7
  手  参考手指 qpos 按**关节名**映射到 DexMate 同名关节 (飞手就是 Sharpa 右手, 名字一致)
主交互手由 phase_* 自动判定, 所以左手 clip 会自动驱动左臂。
"""
from __future__ import annotations

import torch


class DexmateFollower:
    """需要 env 已完成 bimanual 摆放 (env._bimanual_res 存在)."""

    def __init__(self, env, log=print):
        from isaaclab.controllers import (DifferentialIKController,
                                          DifferentialIKControllerCfg)
        from isaaclab.utils.math import subtract_frame_transforms

        from rl_rebuild.correction import frames as F

        self.env = env
        self.log = log
        self.ok = False
        if getattr(env, "dexmate", None) is None or getattr(env, "_bimanual_res", None) is None:
            log("⚠ follow 需要 SHOW_DEXMATE=1 且 SHOW_HUMAN_TRAJ=1, 已跳过")
            return

        res = env._bimanual_res
        self.prim = res["primary"]                    # "right" / "left"
        P = "R" if self.prim == "right" else "L"
        self.dm = env.dexmate
        bn, jn = list(self.dm.body_names), list(self.dm.joint_names)

        self.arm = [jn.index(f"{P}_arm_j{i}") for i in range(1, 8) if f"{P}_arm_j{i}" in jn]
        ee_name = f"{self.prim}_hand_C_MC"
        self.ee_id = bn.index(ee_name)
        self.ee_jac = self.ee_id - 1                  # 固定基座: jacobian body 索引要 -1

        # 手指: 参考是飞手关节序(22) -> DexMate 同名关节
        self.hand_ids, self.hand_src = [], []
        for k, n in enumerate(env.hand.joint_names):
            if n in jn:
                self.hand_ids.append(jn.index(n))
                self.hand_src.append(k)

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=env.device)
        self.sub = subtract_frame_transforms

        J = res["joints"][self.prim]                  # 已摆放好的关节轨迹 (T,21,3)
        wq = F.sharpa_base_quat_from_joints(J)
        self.wp = torch.tensor(J[:, 0], dtype=torch.float32, device=env.device)
        self.wq = torch.tensor(wq, dtype=torch.float32, device=env.device)
        self.T = len(J)
        self._n = 0
        self.ok = True
        log(f"follow 已启用: {self.prim}臂 {len(self.arm)}关节 + 手 {len(self.hand_ids)}关节, "
            f"末端={ee_name}, 参考 {self.T} 帧")

    def step(self, k, log_every=60):
        """把臂驱到第 k%T 帧的参考腕位姿, 手指给参考 qpos. 返回位置误差(米)."""
        if not self.ok:
            return None
        dm, t = self.dm, k % self.T
        root = dm.data.root_state_w[:, 0:7]
        tgt_p, tgt_q = self.wp[t].unsqueeze(0), self.wq[t].unsqueeze(0)
        tp_b, tq_b = self.sub(root[:, 0:3], root[:, 3:7], tgt_p, tgt_q)
        self.ik.set_command(torch.cat([tp_b, tq_b], dim=-1))

        ee_w = dm.data.body_state_w[:, self.ee_id, 0:7]
        ep_b, eq_b = self.sub(root[:, 0:3], root[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
        # ⚠ 分两步切: [:, int, :, list] 会触发 "advanced index 被 slice 分隔" 规则,
        # 把 advanced 维提到最前 -> 形状错乱的 jacobian, IK 会**静默算错**.
        jf = dm.root_physx_view.get_jacobians()               # (N, B-1, 6, D)
        jac = jf[:, self.ee_jac, :, :][:, :, self.arm]        # (N, 6, len(arm))
        q = dm.data.joint_pos[:, self.arm]
        dm.set_joint_position_target(self.ik.compute(ep_b, eq_b, jac, q), joint_ids=self.arm)

        if self.hand_ids:
            fr = self.env.ref_finger[min(t, self.env.ref_finger.shape[0] - 1)]
            dm.set_joint_position_target(fr[self.hand_src].unsqueeze(0),
                                         joint_ids=self.hand_ids)
        err = float(torch.norm(tp_b - ep_b))
        self._n += 1
        if log_every and self._n % log_every == 1:
            self.log(f"[follow] 帧{t}/{self.T} 目标{[round(v,3) for v in tgt_p[0].tolist()]} "
                     f"末端{[round(v,3) for v in ee_w[0,:3].tolist()]} 误差{err*100:.1f}cm")
        return err
