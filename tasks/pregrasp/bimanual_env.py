"""双臂接近 env —— 一张网同时输出左右手动作。

## 架构: 不重构 env, 而是"换入换出侧状态, 同一份逻辑跑两遍"

`tasks/pregrasp/env.py` 的单边量被引用 100+ 处, 奖励是整块的。把它参数化 = 大重构 + 高翻车风险。
本模块沿用 `bimanual.py` 定的路子: **env 的单边逻辑一字不动**, 外面套一层侧状态容器,
调 env 的方法前把该侧状态换进 `self.*`, 调完换回。于是 `_pre_physics_step` /
`_get_rewards` / `_get_dones` 都可以**原样跑两遍**, 奖励/终止逻辑与单臂版**同一份代码**,
不会漂移(两套逻辑各自维护 = 台账里反复出问题的模式)。

代价: 换入换出必须覆盖**全部**单边量, 漏一个 = **静默串台**(A 手的动作写进 B 手的状态)。
`bimanual.SIDE_ATTRS` 是唯一真理来源, `assert_covered()` 每次构造都扫一遍。

## 前置验证 (bimanual_probe.py, 2026-08-17 实测全过)

  ① 两侧臂关节索引完全分开   右 [1,3,5,7,9,11,13] / 左 [0,2,4,6,8,10,12] —— **交错的**,
     所以动作/关节的对应**只能按 arm_jids 索引写, 不能按切片**。
  ② 两侧靶点相距 18.5cm, 各自离**自己**那个物体更近(13.5/12.8cm vs 25.9/16.9cm)
  ③ 两侧 IK 各自有解 (右 0.01cm / 左 0.36cm), 锚点用 env 实测的 arm_center(两侧共用)
  ④ assert_covered 扫不出漏网单边量

## 约定

* 动作 = [A侧 7 维, B侧 7 维] —— A = clip 的交互手(主手), B = 另一只手。
* 观测 = [A侧 obs, B侧 obs] 拼接。
* 奖励 = A + B (两手各自往自己目标靠近, 无耦合项; 会合/倾倒那种双手耦合任务
  需要额外的"关系项", 不在本模块 —— 见 bimanual.py 头部的 S1/S3 说明)。
* 终止 = A 或 B 任一触发 (任一只手撞桌/碰物体, 整个回合结束)。
"""
from __future__ import annotations

from collections.abc import Sequence

import os

import numpy as np
import torch

from tasks.pregrasp import bimanual as BM
from tasks.pregrasp.env import GraspTaskEnv


def _qmul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


class BimanualApproachEnv(GraspTaskEnv):
    """两只手各自接近自己的 GraspPose。动作 14 维 = 7(A) + 7(B)。"""

    def __init__(self, cfg, render_mode=None, **kw):
        # ---- A 侧: 完全按单臂路径构造 (场景/物体/先验都在这里建好) ----
        super().__init__(cfg, render_mode, **kw)
        self._A_name = cfg.hand_side
        self._B_name = "left" if self._A_name == "right" else "right"

        prior_b = getattr(cfg, "prior_b_npz", "")
        yaw_b = float(getattr(cfg, "prior_b_yaw_deg", -1.0))
        assert prior_b, "双臂 env 需要 cfg.prior_b_npz (另一只手的 GraspPose)"

        # ---- B 侧目标物体 = 场景里的第二个物体 (aux) ----
        aux_off = getattr(self, "aux_rel_offset_np", None)
        assert aux_off is not None, \
            "这条 clip 没有第二个物体 (aux) —— 双臂各抓各的无从谈起"
        _to0 = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                      device=self.device)
        A_obj = self.obj_init_pos.cpu().numpy().astype(np.float64)
        B_obj = A_obj + np.asarray(aux_off, np.float64)

        # ---- B 侧: 只换关节索引与靶点, 场景一字不动 ----
        # `_sig`/`_ep_sums` 构造时可能还不存在 —— 先建空的, 否则 SideState 的
        # `if hasattr(env, k)` 会跳过它们, 两侧就又共用了(而且静默)。
        for _k in ("_sig", "_ep_sums"):
            if not hasattr(self, _k):
                setattr(self, _k, {})
        self._A = BM.SideState(self._A_name, self)      # 先把 A 侧快照下来
        self._resolve_joint_ids(force_side=self._B_name)
        # ★★ 关节限位表必须按 B 侧重建 (2026-08-18 实锤): 左右臂限位是**镜像非对称**的
        #   (L_arm_j2 [-26,+89] vs R_arm_j2 [-89,+26]), _resolve_joint_ids 只换 jids
        #   不重建 lower/upper ⟹ B 快照抓到的是 A(右臂)的表 ⟹ 左臂 q_cmd 被夹在右臂
        #   限位上 —— 站姿 L_j2=+45° 从第一步就被削到 +26°, 恒差 19°, 左手永远到不了
        #   规划终点。零动作回放实测: L_j2 恒 -19.0°, 其余关节正常。
        _lim = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.arm_lower = _lim[..., 0][:, self.arm_jids]
        self.arm_upper = _lim[..., 1][:, self.arm_jids]
        # 碰撞/外壳的 body 索引也必须按 B 侧重建 —— 只换关节索引不换这些, 两只手会用
        # **同一条臂**的外壳算离桌间隙(实测 R/L 的 arm_table_gap 逐位相同)。
        _bn = list(self.hand.body_names)
        _P = "R" if self._B_name == "right" else "L"
        _O = "L" if _P == "R" else "R"
        self._build_collide_ids(_bn, _P, _O, _to0)
        # `table_bids` = 与桌/物做碰撞检查的**本侧**手部连杆 (手 + l7/l8/ee)
        self.table_bids = [i for i, n in enumerate(_bn)
                           if n.startswith(f"{self._B_name}_")
                           or n in (f"{_P}_arm_l7", f"{_P}_arm_l8", f"{_P}_ee")]
        # 手指 body 对也跟着换侧
        # ⚠ 手指列表必须与 env.py:202 **逐字一致** (我第一版照记忆写成 thumb/index, 错)
        self.cross_a = [_bn.index(f"{self._B_name}_{f}_DP")
                        for f in ("index", "middle", "ring")]
        self.cross_b = [_bn.index(f"{self._B_name}_{f}_DP")
                        for f in ("middle", "ring", "pinky")]
        zb = np.load(prior_b)
        a = np.radians(yaw_b)
        yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        oq_b = _qmul_np(yq, np.asarray(zb["canon_rot"], np.float64))
        from rl_rebuild.correction.kinematics import quat_to_R
        B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], np.float64) + B_obj
        B_gq = _qmul_np(oq_b, np.asarray(zb["grasp"][3:7], np.float64))
        _to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self._grasp_pos_w = _to(B_gp)
        self._grasp_quat_w = _to(B_gq)
        self.obj_init_pos = _to(B_obj)
        # 裁定B3: B 侧 PreGrasp = 抓姿沿"接触质心→腕"方向平移 3cm (掌心锚点),
        # 与 B_gp 同一条(raw)变换链
        _zbg = np.asarray(zb["grasp"], np.float64)
        _zbc = np.asarray(zb["contact_centroid"], np.float64)
        _ub = _zbg[:3] - _zbc
        _ub = _ub / max(np.linalg.norm(_ub), 1e-9)
        _pcm = float(getattr(cfg, "pregrasp_palm_cm", 5.0)) / 100.0
        self._pregrasp_w = [
            (quat_to_R(oq_b) @ (_zbg[:3] + _pcm * _ub) + B_obj,
             _qmul_np(oq_b, _zbg[3:7]))]
        if getattr(cfg, "pregrasp29", False):
            # PreGrasp29: B 侧靶点同样整体换成掌心 PreGrasp (与 A 侧口径一致)
            self._grasp_pos_w = _to(self._pregrasp_w[0][0])
            self._grasp_quat_w = _to(self._pregrasp_w[0][1])
        if getattr(cfg, "pregrasp_phase2", False):
            # Phase2 B 侧斜坡: 与 l5 分支同套路 (ArmIK + cuRobo 左手末帧当种子),
            # 腕 PreGrasp→B_gp 直线逐帧 IK; 指型终点 = zb 抓姿 22 关节。
            from rl_rebuild.correction.kinematics import ArmIK as _ArmIK
            from rl_rebuild.correction.kinematics import quat_to_R as _q2R
            _gqn2 = np.asarray(B_gq, np.float64)
            _gqn2 = _gqn2 / max(np.linalg.norm(_gqn2), 1e-12)
            _Rb2 = _q2R(_gqn2)
            _ikb2 = _ArmIK(self._B_name, anchor_link="arm_center",
                           anchor_T=self._anchor_T)
            _zr2 = np.load(cfg.curobo_ref_npz)
            _seed2 = np.asarray(_zr2["left_q"], np.float64)[-1]
            _pp0b = np.asarray(self._pregrasp_w[0][0], np.float64)
            _K2b = int(getattr(cfg, "phase2_steps", 50))
            _rows2, _errs2 = [], []
            for _k in range(_K2b):
                _al = _k / max(_K2b - 1, 1)
                _pt = (1 - _al) * _pp0b + _al * np.asarray(B_gp, np.float64)
                _rk = _ikb2.solve(_pt, _Rb2, q0=_seed2, iters=200)
                _seed2 = _rk["q"]
                _rows2.append(_rk["q"])
                _errs2.append(_rk["pos_err"])
            self._p2_arm = _to(np.stack(_rows2).astype(np.float32))
            _fg2 = np.clip(np.asarray(zb["grasp"], np.float64)[7:29][self._generic_perm],
                           self.dof_lower[0].cpu().numpy(),
                           self.dof_upper[0].cpu().numpy())
            self._p2_fin = _to(_fg2.astype(np.float32))
            self._p2_t = torch.zeros(self.num_envs, dtype=torch.long,
                                     device=self.device)
            # Phase2-RL: B 侧真抓姿靶点 (与 B 侧 _grasp_pos_w 同一条 raw 链/腕口径)
            self._g2_pos_w = _to(np.asarray(B_gp, np.float64))
            self._g2_quat_w = _to(_gqn2)
            # 第二段判据/里程碑状态按 B 侧独立预创建 (A 的已在父类建, 会被快照)
            self._g2_run = torch.zeros(self.num_envs, dtype=torch.long,
                                       device=self.device)
            self._g2_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            self._m1_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            self._m2_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            print(f"[phase2] B 侧就绪: 真抓姿靶点+抓姿指型 | (斜坡 {_K2b} 帧留档, "
                  f"IK 误差 max {max(_errs2)*100:.2f}cm)")
        # ★ B 侧的**物体**也要换成第二个刚体 —— 奖励里的 `_target_w()` 读的是
        #   `self.object.data.root_pos_w`, 不是 `_grasp_pos_w`。只换靶点不换物体,
        #   左手会拿"到瓶子的距离"当自己的奖励(2026-08-17 差点漏掉)。
        assert getattr(self, "aux", None) is not None, "aux 刚体不存在, B 侧没有目标物体"
        self.object = self.aux
        # 亲和点(接触质心, 物体局部系)也逐侧: 换物体不换它 = 靶点落在错的表面上
        zb_c = zb["contact_centroid"] if "contact_centroid" in zb.files else None
        if zb_c is not None:
            self.aff_local = _to(np.asarray(zb_c, np.float64))
        # ★ 起点池必须按 B 侧**重建** —— `_sp_init()` 在父类构造时跑过一次, 那时用的是
        #   **A 侧**的 arm_jids, 池里存的是 A 臂的关节角。直接给 B 用 = 把右臂的姿势
        #   套到左臂上 = 非法位形, 一解冻就被判死。
        #   2026-08-17 实测症状: 每回合都在第 5 步(= settle_steps)终止, A/B 各 64/64,
        #   于是 freeze_ctr 永远满、臂目标一次没更新过、两只手全程不动。
        if getattr(cfg, "start_pool", ""):
            self._sp_init()
            print(f"[bimanual] B 侧起点池已按 {self._B_name} 臂重建")
        # L5 (2026-08-17): B 侧先验派生量**定向重建** —— 不整体调 _load_grasp_prior
        # (它与主物体 mesh/clip entry/场景布局深度耦合, 且读 cfg.prior_yaw_deg=瓶的
        # 19.5° 去转杯的抓姿, 冒烟实测 IK 直接无解)。B 真正需要的只有五样:
        #   ① q_close 杯指模板  ② q_pregrasp(=cuRobo 左手末帧, 双重验收过的近点位形)
        #   ③ 左臂 IK 的抓姿关节  ④ 左臂抬升斜坡  ⑤ _fgate_dg 门控基准
        if getattr(cfg, "l5_couple", False) or (
                getattr(cfg, "pregrasp29", False)
                and not getattr(cfg, "approach_only", False)):
            # PG-A (2026-08-18晚): 物理抓稳版走完整 cand→微抬升链, B 侧五件套
            # (杯指模板/近点位形/左臂抓姿IK/抬升斜坡/门控基准) 同样必须重建
            from rl_rebuild.correction.kinematics import ArmIK
            self.obj_init_quat = _to(oq_b)
            _gqn = np.asarray(B_gq, np.float64)
            _gqn = _gqn / max(np.linalg.norm(_gqn), 1e-12)
            _Rb = quat_to_R(_gqn)
            _ikb = ArmIK(self._B_name, anchor_link="arm_center",
                         anchor_T=self._anchor_T)
            _zr = np.load(cfg.curobo_ref_npz)
            _q_near = np.asarray(_zr["left_q"], np.float64)[-1]
            # ⚠ 必须用 cuRobo 左手末帧(近点, 离抓位仅 5cm)当 IK 种子 —— 冷启动会
            # 掉进错误盆地 (冒烟实测 4.53cm; GUI 同位姿实测 0.36cm 可达)。
            # 验收看实测误差不看 ok 标志 (ok 绑着 pos_tol, 见 env.py:897 的教训)。
            _rg = _ikb.solve(np.asarray(B_gp, np.float64), _Rb,
                             q0=_q_near, iters=300, pos_tol=2e-4)
            _fk_near = _ikb.fk(_q_near)[0]
            assert np.isfinite(_rg["q"]).all() and _rg["pos_err"] < 0.01, (
                f"L5 B 侧抓姿 IK 失败 (err {_rg.get('pos_err', 1.0)*100:.2f}cm)\n"
                f"  诊断: FK(近点种子)={np.round(_fk_near, 3)} | "
                f"B_gp={np.round(np.asarray(B_gp, np.float64), 3)} | "
                f"|FK(近)−B_gp|={np.linalg.norm(_fk_near - np.asarray(B_gp)) * 100:.2f}cm "
                f"(应≈5cm; 大偏差=anchor_T 或坐标系错, 小偏差=IK 求解问题)\n"
                f"  A_obj={np.round(A_obj, 3)} aux_off={np.round(np.asarray(aux_off, np.float64), 3)} "
                f"B_obj={np.round(B_obj, 3)} oq_b={np.round(oq_b, 3)} yaw_b={yaw_b}")
            self._prior_q_grasp = _to(_rg["q"])
            self.q_pregrasp = _to(_q_near.astype(np.float32))
            _fsrc = (np.asarray(zb["close_anchor"], np.float64)
                     if "close_anchor" in zb.files
                     else np.asarray(zb["grasp"], np.float64)[7:29])
            _fb = np.clip(_fsrc[self._generic_perm],
                          self.dof_lower[0].cpu().numpy(),
                          self.dof_upper[0].cpu().numpy())
            self.q_close = _to(_fb)
            self._fgate_dg = float(np.linalg.norm(
                np.asarray(B_gp, np.float64) - np.asarray(B_obj, np.float64)))
            # 抬升斜坡 (与 env.py:936 同款: 收紧 pos_tol + 雅可比补步 + 实际抬升核验)
            lo_np = self.arm_lower[0].cpu().numpy().astype(np.float64)
            hi_np = self.arm_upper[0].cpu().numpy().astype(np.float64)
            lift, qw, _nf = [], _rg["q"].copy(), 0
            for _i in range(cfg.lift_steps + 1):
                _tgt = np.asarray(B_gp, np.float64) + np.array(
                    [0.0, 0.0, cfg.lift_height * _i / max(cfg.lift_steps, 1)])
                _r = _ikb.solve(_tgt, _Rb, q0=qw, iters=300, pos_tol=2e-4)
                if _r["ok"]:
                    qw = _r["q"].copy()
                elif _i > 0:
                    _nf += 1
                    _J = _ikb.jacobian(qw)
                    _dp = _tgt - _ikb.fk(qw)[0]
                    _dq = np.linalg.lstsq(_J[:3], _dp, rcond=None)[0]
                    qw = np.clip(qw + _dq, lo_np, hi_np)
                else:
                    qw = _r["q"].copy()
                lift.append(qw.copy())
            self.q_lift = _to(np.stack(lift))
            self.q_lift_delta = self.q_lift - self.q_lift[0:1]
            _lv = int(round(min(cfg.verify_lift_m
                                / (cfg.lift_height / max(cfg.lift_steps, 1)),
                                cfg.lift_steps)))
            _rise = float(_ikb.fk(lift[_lv])[0][2] - _ikb.fk(lift[0])[0][2])
            assert _rise >= 0.9 * cfg.verify_lift_m, (
                f"L5 B 侧抬升斜坡只抬得动 {_rise*1000:.2f}mm —— 左臂竖直 IK 病态")
            print(f"[bimanual] L5 B 侧定向重建: 抓姿IK {_rg['pos_err']*100:.2f}cm | "
                  f"近点=cuRobo 左手末帧 | 指模板 {os.path.basename(prior_b)} | "
                  f"抬升第{_lv}级实抬 {_rise*1000:.1f}mm (IK补步 {_nf}) | "
                  f"d_g={self._fgate_dg*100:.1f}cm | ⚠ 参与指分类沿用 A 侧(v1)")
        self._B = BM.SideState(self._B_name, self)      # 快照 B 侧
        # ★★ 架构级修正 (2026-08-17): SideState 存的是**引用**不是副本。父类里大量
        #     缓冲是**原地写** (`self.obj_start_pos[env_ids] = ...`), 两侧引用同一块
        #     内存时原地写互相覆盖 —— 字段确实在 SIDE_ATTRS 里、确实"换"了, 换的却是
        #     指向同一块内存的两个引用, 所以怎么补清单都没用。
        #     `bimanual.py` 原注释说"原地写两种方式都安全"是**错的**, 已一并更正。
        #     实测症状: 两侧的 obj_start_pos 都是杯子的 0.918m ⟹ A 把瓶子写到杯子
        #     位置上, 两物体重叠 ⟹ PhysX 爆炸弹开 ⟹ 双双飞到 1.25/1.52m ⟹ 判 thrown
        #     ⟹ 每步复位 ⟹ 永远冻结 ⟹ 两只手一步没动过。
        _cloned = 0
        for _k, _v in list(self._B.data.items()):
            if torch.is_tensor(_v):
                self._B.data[_k] = _v.clone()
                _cloned += 1
        print(f"[bimanual] B 侧张量已深拷贝 {_cloned} 项 (避免与 A 侧共享同一块缓冲)")
        self._resolve_joint_ids(force_side=self._A_name)  # 现场还原成 A 侧
        for k, v in self._A.data.items():                 # 把 A 侧的量写回去
            setattr(self, k, v)

        # ---- 自证: 两侧真的不同 (打印**最终生效值**, 不是"打算设成什么") ----
        print(f"[bimanual] A={self._A_name} 臂关节 {list(self._A.data['arm_jids'])}")
        print(f"[bimanual] B={self._B_name} 臂关节 {list(self._B.data['arm_jids'])}")
        _ov = set(map(int, self._A.data["arm_jids"])) & set(map(int, self._B.data["arm_jids"]))
        assert not _ov, f"两侧臂关节重叠 {sorted(_ov)} —— 会互相覆盖"
        _da = float(torch.norm(self._A.data["_grasp_pos_w"] - self._B.data["_grasp_pos_w"]))
        print(f"[bimanual] 两靶点相距 {_da*100:.1f}cm | A物体 {np.round(A_obj*100,1)} "
              f"B物体 {np.round(B_obj*100,1)} (cm)")
        assert _da > 0.05, f"两侧靶点只差 {_da*100:.1f}cm —— 八成串台了"
        # ★ 末端 body 必须分开 —— 不分开的后果是**静默串台的测量**: 动作各走各的,
        #   但 d_pos/腕位/臂间隙全用同一只手腕算。2026-08-17 冒烟实测抓到:
        #   R/L 的 d_pos 逐位相同(59.018)而 action_norm 不同(0.268 vs 0.000)。
        assert self._A.data["ee_id"] != self._B.data["ee_id"], (
            f"两侧末端 body 相同 (ee_id={self._A.data['ee_id']}) —— 测量会串台")
        assert self._A.data["object"] is not self._B.data["object"], \
            "两侧目标物体是同一个 —— 奖励会串台"
        print(f"[bimanual] 末端 body: A={self._A.data['ee_body']} B={self._B.data['ee_body']}")

        # ★ 动作缓冲**必须逐侧独立**, 且宽度 = 单侧动作维。
        #   bimanual.SHARED_OK 把 actions_buf/prev_actions 列为"共用"(理由: 动作向量是
        #   拼起来的整体), 但那样父类 `_pre_physics_step` 里
        #   `self.actions_buf.copy_(actions)` 就是 14 维缓冲 copy 7 维切片 —— 直接崩
        #   (2026-08-17 冒烟实测: "size of tensor a (14) must match b (7)")。
        #   逐侧各 7 维后, 每侧就是一个**忠实的单臂 env**, 拼接只发生在 obs/reward 层,
        #   每侧 obs 宽度也回到单臂的 139(而不是 146)。
        _n = int(cfg.action_space) // 2
        _N, _dev = self.num_envs, self.device
        for _sd in (self._A, self._B):
            _sd.data["actions_buf"] = torch.zeros(_N, _n, device=_dev)
            _sd.data["prev_actions"] = torch.zeros(_N, _n, device=_dev)
        print(f"[bimanual] 动作缓冲逐侧独立: 每侧 {_n} 维 (共用会让父类 copy_ 维度崩)")

        # ---- cuRobo 前馈参考: 逐侧接入 (2026-08-17 用户裁定) ----
        # 每侧把自己的整条规划(含"先右后左"的保持段)当 retract_path —— 前馈机制复用
        # env 现成的那套(零动作=沿参考走一步), 只是路径内容换成 cuRobo 的。
        _ref = getattr(cfg, "curobo_ref_npz", "")
        if _ref:
            _z = np.load(_ref, allow_pickle=True)
            _st = max(1, int(getattr(cfg, "curobo_ref_stride", 2)))
            _paths = {}
            for _sd in (self._A_name, self._B_name):
                _q = np.asarray(_z[f"{_sd}_q"], np.float32)[::_st].copy()
                _paths[_sd] = torch.tensor(_q, dtype=torch.float32, device=self.device)
            # 口径自证: 参考首帧必须 == 各自臂的站姿, 否则回合一开始就跳变
            for _tag, _side in (("A", self._A), ("B", self._B)):
                _nm = self._A_name if _tag == "A" else self._B_name
                _q_st = self.hand.data.default_joint_pos[0, _side.data["arm_jids"]]
                _d0 = float((_paths[_nm][0] - _q_st).abs().max())
                assert _d0 < 0.02, (f"{_tag} 侧参考首帧与站姿差 {np.degrees(_d0):.2f}° "
                                    f"—— 起点会跳变, 参考文件与场景不配套")
                _side.data["retract_path"] = _paths[_nm]
            self.retract_path = _paths[self._A_name]   # 当前现场是 A 侧
            print(f"[curobo_ref] ✅ 前馈已接: A({self._A_name}) "
                  f"{tuple(_paths[self._A_name].shape)} | B({self._B_name}) "
                  f"{tuple(_paths[self._B_name].shape)} | stride {_st} | 首帧=站姿 已验证")

        # ★ 防雷 (2026-08-18 深夜): B 侧 q_ref(人手参考)从未换侧重建 —— 任何走
        #   "人手首帧起步"的双臂配置都会把左臂摆成**右臂**的人手姿势 (PGA 实锤:
        #   j1 偏 121°, 腕在 1.9m 高空, 7M 步无效训练)。硬拦截: 双臂必须用
        #   退避起点族(retract_start) 或纯直接抓取起步。
        assert getattr(cfg, "retract_start", False) or \
            float(getattr(cfg, "direct_grasp_prob", 0.0)) >= 1.0, (
            "双臂环境必须开退避起点族 (approach_only 自动开; 完整任务加 --retract) "
            "—— B 侧 q_ref 未重建, 人手首帧起步 = 左臂乱摆 (2026-08-18 PGA 事故)")
        BM.assert_covered(self)
        _unc = BM.list_uncovered(self)
        if _unc:
            print(f"[bimanual] ⚠ 未覆盖的张量类字段 {len(_unc)} 个 (人工裁: 该分侧的补进 SIDE_ATTRS):")
            for _k, _t in _unc:
                print(f"           {_k:28s} {_t}")

    # ---------------------------------------------------------------- 双跑包装
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        n = actions.shape[1] // 2
        assert actions.shape[1] == 2 * n, f"动作维度 {actions.shape[1]} 不是偶数"
        _k = getattr(self, "_bi_dbg", 0)
        _dbg = _k < 400 and _k % 40 == 0      # 每 40 步打一次, 打 10 次
        for _tag, _side, _act in ((self._A_name[0].upper(), self._A, actions[:, :n]),
                                  (self._B_name[0].upper(), self._B, actions[:, n:])):
            with BM.use_side(self, _side):
                _t0 = self.arm_tgt.clone() if hasattr(self, "arm_tgt") else None
                super()._pre_physics_step(_act)
                if _dbg:
                    # 定点诊断: 光看 TB 的 action_norm 分不出"策略没出动作"还是
                    # "出了但没作用到臂上" —— 这里两个都量。
                    _dt = (float((self.arm_tgt - _t0).abs().max())
                           if _t0 is not None else float("nan"))
                    _d = float((self._anchor_w() - self._target_w()).norm(dim=1).mean())
                    print(f"[bi-dbg] 步{_k:4d} {_tag}: 动作 {float(_act.norm(dim=1).mean()):.4f} "
                          f"| 臂目标Δ {_dt*57.3:7.4f}° | 离目标 {_d*100:6.2f}cm "
                          f"| 冻结 {int((self.freeze_ctr > 0).sum()):3d} "
                          f"| 相位 {int(self.task_phase[0])}", flush=True)
        self._bi_dbg = _k + 1

    def _apply_action(self) -> None:
        with BM.use_side(self, self._A):
            super()._apply_action()
        with BM.use_side(self, self._B):
            super()._apply_action()

    def _get_observations(self) -> dict:
        # 父类里的 `_check_obs_dim` 拿 cfg.observation_space 比 —— 而那是**双侧**宽度。
        # 逐侧调用时必须临时换成单侧宽度, 否则每次都报 "146 != 292"。
        _full = self.cfg.observation_space
        self.cfg.observation_space = int(getattr(self.cfg, "_obs_single", _full // 2))
        try:
            with BM.use_side(self, self._A):
                oa = super()._get_observations()
            with BM.use_side(self, self._B):
                ob = super()._get_observations()
        finally:
            self.cfg.observation_space = _full
        out = dict(oa)
        for k in ("policy", "priv_info", "proprio_hist"):
            if k in oa and k in ob and torch.is_tensor(oa[k]):
                out[k] = torch.cat([oa[k], ob[k]], dim=-1)
        return out

    def _get_rewards(self) -> torch.Tensor:
        with BM.use_side(self, self._A):
            ra = super()._get_rewards()
        with BM.use_side(self, self._B):
            rb = super()._get_rewards()
        return ra + rb          # 无耦合项: 两手各自往自己目标去

    def _get_dones(self):
        with BM.use_side(self, self._A):
            ta, ua = super()._get_dones()
        with BM.use_side(self, self._B):
            tb, ub = super()._get_dones()
        _k = getattr(self, "_bi_dbg", 0)
        if _k < 400 and _k % 40 == 0:
            # 冻结恒为满 ⟹ 每步都在复位。这里查是**哪一侧、哪种**终止在触发。
            _br = []
            for _tag, _sd in ((self._A_name[0].upper(), self._A),
                              (self._B_name[0].upper(), self._B)):
                _sg = _sd.data.get("_sig") or {}
                _hit = {k: int(v.sum()) for k, v in _sg.items()
                        if hasattr(v, "dtype") and v.dtype == torch.bool and int(v.sum())}
                _br.append(f"{_tag}:{_hit}")
            for _tag, _sd in ((self._A_name[0].upper(), self._A),
                              (self._B_name[0].upper(), self._B)):
                _o = _sd.data.get("object")
                _os_ = _sd.data.get("obj_start_pos")
                if _o is not None:
                    _z = float((_o.data.root_pos_w[:, 2]
                                - self.scene.env_origins[:, 2]).mean())
                    _lim = self.cfg.table_top_z + self.cfg.max_obj_height
                    print(f"[bi-dbg]   {_tag} 物体高 {_z:.4f}m (thrown 线 {_lim:.4f}m) "
                          f"起始高 {float(_os_[:,2].mean()):.4f}m" if _os_ is not None
                          else f"[bi-dbg]   {_tag} 物体高 {_z:.4f}m", flush=True)
            print(f"[bi-dbg] 步{_k:4d} 终止: A {int(ta.sum()):3d} B {int(tb.sum()):3d} "
                  f"| 回合步 {int(self.episode_length_buf[0])}\n"
                  f"          触发项 {' || '.join(_br)}", flush=True)
        # ★ 合并 `_sig` 给下游 (确定性评测读 `raw._sig["newly_success"]`)。
        #   `use_side` 退出时会把 self._sig **还原成调用前的值**(空 dict), 所以两侧跑完
        #   外面读到的是空的 -> KeyError。2026-08-17 实测 C/D 在第一次评测时崩。
        #
        #   顺带定义**双臂的成功语义**(之前没定): 成功 = **两只手都到位**。
        #   不能用 `newly_success` 的或/与 —— 两只手可能在不同步到位, "newly" 的与是空的。
        #   所以按 arrived 的与算, 再自己维护 "首次达成" 的边沿。
        _sa = dict(self._A.data.get("_sig") or {})
        _sb = self._B.data.get("_sig") or {}
        for _k, _v in _sb.items():          # 布尔项取或(任一侧撞桌/碰物体都算)
            if _k in _sa and torch.is_tensor(_v) and _v.dtype == torch.bool:
                _sa[_k] = _sa[_k] | _v
            elif _k not in _sa:
                _sa[_k] = _v
        if getattr(self.cfg, "approach_only", False):
            _aa = self._A.data.get("arrived")
            _ab = self._B.data.get("arrived")
            if getattr(self.cfg, "pregrasp_phase2", False):
                # Phase2-RL (2026-08-18晚): 双臂"胜利"从"到位"推迟到"g2 达成"
                # (腕到真抓姿+指型贴近并保持), 与 env.py 侧内 success 口径一致 ——
                # 否则这里按 arrived 直接终止, 第二段被整个跳过(斜坡时代实测 43 步)。
                _ga = self._A.data.get("_g2_done")
                _gb = self._B.data.get("_g2_done")
                if _ga is not None:
                    _aa = _aa & _ga
                if _gb is not None:
                    _ab = _ab & _gb
        else:
            # L5 (裁定①): 各侧的"胜利" = 本回合已通过**微抬升验证** (succeeded 在
            # 侧内 _get_dones 里当步锁存, env.py:1673)。双臂成功 = 两侧都验证通过。
            # 先验证完的手抬着物体等另一只 —— 与接近段"先到的等"同一语义。
            _aa = self._A.data.get("succeeded")
            _ab = self._B.data.get("succeeded")
        if _aa is not None and _ab is not None:
            _both = _aa & _ab
            if not hasattr(self, "_bi_done_once"):
                self._bi_done_once = torch.zeros_like(_both)
            _sa["newly_success"] = _both & ~self._bi_done_once
            self._bi_done_once = self._bi_done_once | _both
        self._sig = _sa
        # ★ 终止语义 (2026-08-17 冒烟修正): 单手到位**不终止** —— 右手先到(约第81步,
        #   前馈时序)就触发 per-side success 终止, 回合结束时左手才刚开始动
        #   (实测: R arrive=1.0 而 L 最小 3.43cm 停滞)。参考里的"保持段"正是让先到的
        #   手停在原位等另一只 —— 所以: 失败(任一手)立即终止; 成功要**两手都到**。
        _t_fail = torch.zeros_like(ta)
        if _aa is not None and _ab is not None:
            _t_fail = (ta & ~_aa) | (tb & ~_ab)     # 各侧刨去"纯胜利"的终止 = 纯失败
            # "先赢后作恶"后门 (2026-08-17 用户从 ep800 录像揪出): 到位/验证过的手在
            # 等待期把物体推倒/掉落/抛飞/撞桌/卡死 —— 全部仍判失败。
            # L3 的教训: 只豁免"纯胜利"终止不够, 豁免会把该侧**所有**终止都吞掉,
            # 于是右手到位后推倒瓶子不受罚, 训练期推倒率一路涨到 100%,
            # 评测 99.8% 里混着大量"瓶子倒了的假成功"。
            # (pushed 自带 ~got_candidate 门控: 抓稳后的正常搬动不会触发, L5 安全)
            for _sg in (self._A.data.get("_sig") or {}, _sb):
                for _k2 in ("fell", "thrown", "toppled", "pushed", "table_crash", "stuck"):
                    _v2 = _sg.get(_k2)
                    if torch.is_tensor(_v2) and _v2.dtype == torch.bool:
                        _t_fail = _t_fail | _v2
            _term = _t_fail | _both
        else:
            _term = ta | tb
        return _term, (ua | ub)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """两侧各复位一次, 并把各自的日志**按侧加前缀**后合并。

        ⚠ 不加前缀会**静默丢掉一整只手的可观测性**: 父类把统计写进
        `self.extras["log"]` 的固定 key(`approach/d_pos_cm` 等), 跑两遍后
        **后一侧把前一侧整个覆盖**。2026-08-17 冒烟实测: TB 里只剩 B 侧的数,
        A 侧在干什么完全看不见 —— 而双臂最需要看的恰恰是"两只手是不是各动各的"。
        现在: `L/approach/d_pos_cm` 与 `R/approach/d_pos_cm` 各一条。
        """
        # ⚠ 父类是 `self.extras.update(log)` —— 把日志**平铺进 extras 顶层**,
        #   而 TB 读的正是顶层(ppo.py:452 只收顶层的标量)。只改 extras["log"] 无效
        #   (2026-08-17 实测: 改完 TB 里前缀一个没有)。
        per, flat = {}, {}
        if getattr(self, "_bi_done_once", None) is not None and env_ids is not None:
            self._bi_done_once[env_ids] = False     # 复位: 首次达成的边沿要清
        for tag, side in ((self._A_name[0].upper(), self._A),
                          (self._B_name[0].upper(), self._B)):
            # ★ B 侧复位不许重摆场景 (2026-08-18): B 的 self.object=aux, 重摆会把
            #   aux 的相对偏移二次应用 —— 杯子跑到 2×offset, 左手全系抓空气。
            self._bi_skip_scene = (side is self._B)
            with BM.use_side(self, side):
                super()._reset_idx(env_ids)
                _lg = self.extras.get("log", {}) or {}
                for _k in _lg:                       # 先撤掉父类刚平铺进去的无前缀键
                    self.extras.pop(_k, None)
                per[tag] = dict(_lg)
                flat.update({f"{tag}/{_k}": _v for _k, _v in _lg.items()})
        self._bi_skip_scene = False
        # 保留一份**两侧均值**在原 key 上: 下游(课程钩子/auto_stop)按原名取, 不能断
        _keys = set().union(*(d.keys() for d in per.values())) if per else set()
        for _k in _keys:
            _vs = [d[_k] for d in per.values() if _k in d and isinstance(d[_k], (int, float))]
            if _vs:
                flat[_k] = sum(_vs) / len(_vs)
        self.extras.update(flat)
        self.extras["log"] = flat
