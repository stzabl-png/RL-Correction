"""把手静态摆到 Dexonomy GraspPose 上看几何 (只渲染不跑物理, 类比 view_ref).

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.view_prior \
      --clip Grasp5 --grasp_prior tasks/pregrasp/priors/Grasp5.npz

  --pose grasp    (默认) 臂=抓握腕位 IK 解, 指=Dexonomy 抓握构型
  --pose pregrasp 臂=预抓位 IK 解, 指=张开
⚠ 手是被"摆"上去的, 不是物理撑住的 —— 看的是接触几何落点, 不是力.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp5")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0,
               help="≥0: 钉死物体 yaw (度), 与训练口径一致; <0 自搜")
p.add_argument("--pose", default="grasp", choices=("grasp", "pregrasp"))
p.add_argument("--second_prior", default="",
               help="第二只手的 GraspPose npz(摆到 aux 物体上) —— 双手同时看")
p.add_argument("--eye", default="0.55,-0.85,1.45")
p.add_argument("--lookat", default="-0.10,-0.08,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewprior")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.grasp_prior_npz = args.grasp_prior
if args.prior_yaw >= 0:
    cfg.prior_yaw_deg = float(args.prior_yaw)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0             # 看精确摆姿, 不抖
cfg.closure_init_max = 0.0
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1600, 900))
E = GraspTaskEnv(cfg)
E.reset()

W = E.scene.env_origins[0]
# 物体钉到精确初始位 (loader 已做 yaw 旋转)
pose = torch.cat([E.obj_init_pos + W, E.obj_init_quat]).unsqueeze(0)
E.object.write_root_pose_to_sim(pose)
E.object.write_root_velocity_to_sim(torch.zeros(1, 6, device=E.device))

# 手摆姿
q = E.hand.data.default_joint_pos.clone()
if args.pose == "grasp":
    q[:, E.arm_jids] = E._prior_q_grasp
    q[:, E.hand_jids] = E.q_close        # Dexonomy 抓握手型
else:
    q[:, E.arm_jids] = E.q_pregrasp
    q[:, E.hand_jids] = E.q_open
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E.hand.set_joint_position_target(q)
E.hand.write_data_to_sim()
E.sim.step(render=False)             # 推一步物理让 body 位姿刷新 (读数用)
E.hand.update(E.sim.get_physics_dt())
E.object.update(E.sim.get_physics_dt())
# 双物体螺旋场景 (2026-08-05): 上面那步物理里, 瞬移进物体的闭合手会产生去穿透
# 冲量把物体踢位移; 本脚本不走 env.step, 螺旋投影不会自动跑 —— 盖会留在原地
# 看起来像穿模. 把物体钉回去 + 按闭合相对位姿重摆盖.
if getattr(E, "screw_spec", None) is not None:
    E.object.write_root_pose_to_sim(pose)
    E.object.write_root_velocity_to_sim(torch.zeros(1, 6, device=E.device))
    E.object.update(E.sim.get_physics_dt())
    E._SA.reset_screw(E, E.object._ALL_INDICES)
    E.aux.update(E.sim.get_physics_dt())

# ---- 第二只手(2026-08-15 加): 把另一臂摆到 aux 物体的 GraspPose 上 ----
#   aux 的**绕轴自转是不可观测自由度**(P3: 回转体), 所以允许按 IK 可达性自搜它的 yaw
#   —— 这不改变任何可观测量, 与 D1"不许自搜主物体 yaw"不冲突(那条是因为主物体的
#   yaw 要与视频一致; 而 aux 的 yaw 本就没有视频依据)。
if args.second_prior and getattr(E, "aux", None) is not None:
    from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
    _side2 = "left" if cfg.hand_side == "right" else "right"
    _P2 = "R" if _side2 == "right" else "L"
    _jn = list(E.hand.joint_names)
    _aids2 = [i for i, n in enumerate(_jn) if n.startswith(f"{_P2}_arm")]
    _hids2 = [i for i, n in enumerate(_jn) if n.startswith(f"{_side2}_")]
    _bn = list(E.hand.body_names)
    _ac = _bn.index("arm_center")
    _T = np.eye(4)
    _T[:3, :3] = quat_to_R(E.hand.data.body_quat_w[0, _ac].cpu().numpy().astype(np.float64))
    _T[:3, 3] = (E.hand.data.body_pos_w[0, _ac] - W).cpu().numpy().astype(np.float64)
    _ik2 = ArmIK(_side2, anchor_link="arm_center", anchor_T=_T)
    _z2 = np.load(args.second_prior)
    _g2 = np.asarray(_z2["grasp"], np.float64).copy()
    # ★ 垫↔接触零位校准(与主手同一套): Dexonomy 接触标注在指尖极点、我们力垫在指腹,
    #   系统差 ~1.4cm。不补的话手会多陷进物体里 —— 用户在 GUI 里看到的左手穿模就是这个。
    from rl_rebuild.correction.ref_builders.replay_grasp import _urdf as _u2f
    from tasks.pregrasp.env import FINGERS as _FG
    from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER as _GJO
    _uu = _u2f()
    _qd2 = {n.replace("right_", f"{_side2}_"): float(v) for n, v in zip(_GJO, _g2[7:29])}
    _Th2 = np.eye(4)
    _Th2[:3, :3] = quat_to_R(_g2[3:7])
    _Th2[:3, 3] = _g2[:3]
    _pads2 = np.stack([_uu.link_pose(f"{_side2}_{f}_elastomer", _qd2, _Th2,
                                     f"{_side2}_hand_C_MC")[:3, 3] for f in _FG])
    _cts2 = np.asarray(_z2["contact_pos"], np.float64)
    _d22 = np.linalg.norm(_pads2[:, None, :] - _cts2[None], axis=-1)
    _own2 = _d22.argmin(axis=0)
    _act2 = np.array([bool(((_own2 == i) & (_d22[i] < 0.05)).any()) for i in range(5)])
    _sh2 = (_cts2[_d22.argmin(axis=1)][_act2] - _pads2[_act2]).mean(axis=0)
    if np.linalg.norm(_sh2) > 0.008:
        _g2[:3] += _sh2
        print(f"[view_prior] 第二只手 垫↔接触零位校准: 腕位平移 "
              f"{np.round(_sh2*100,2).tolist()}cm (|Δ|={np.linalg.norm(_sh2)*100:.2f}cm) "
              f"| 参与指 {int(_act2.sum())}/5")
    _ap = (E.aux.data.root_pos_w[0] - W).cpu().numpy().astype(np.float64)
    _aq = E.aux.data.root_quat_w[0].cpu().numpy().astype(np.float64)

    def _qm(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

    _best = (1e9, None, None, 0.0)
    for _yd in range(0, 360, 5):
        _y = np.radians(_yd)
        _qz = np.array([np.cos(_y/2), 0.0, 0.0, np.sin(_y/2)])
        _q2 = _qm(_qz, _aq)
        _wp = quat_to_R(_q2) @ _g2[:3] + _ap
        _r = _ik2.solve(_wp, quat_to_R(_qm(_q2, _g2[3:7])), iters=200)
        if _r["pos_err"] < _best[0]:
            _best = (float(_r["pos_err"]), _r["q"], _q2, float(_yd))
    _err, _q2arm, _q2obj, _yaw2 = _best
    print(f"[view_prior] 第二只手({_side2}) aux 自搜 yaw={_yaw2:.0f}° -> IK 位置误差 "
          f"{_err*100:.2f}cm {'✅' if _err < 0.02 else '⛔ >2cm, 该候选够不着'}")
    if _q2arm is not None:
        E.aux.write_root_pose_to_sim(torch.cat([
            torch.tensor(_ap, dtype=torch.float32, device=E.device) + W,
            torch.tensor(_q2obj, dtype=torch.float32, device=E.device)]).unsqueeze(0))
        E.aux.write_root_velocity_to_sim(torch.zeros(1, 6, device=E.device))
        q2 = E.hand.data.joint_pos.clone()
        q2[:, _aids2] = torch.tensor(_q2arm, dtype=q2.dtype, device=E.device)
        _n = min(len(_hids2), 22)
        q2[:, _hids2[:_n]] = torch.tensor(_g2[7:7+_n], dtype=q2.dtype, device=E.device)
        E.hand.write_joint_state_to_sim(q2, torch.zeros_like(q2))
        E.hand.set_joint_position_target(q2)
        E.hand.write_data_to_sim()
        E.sim.step(render=False)
        E.hand.update(E.sim.get_physics_dt())
        E.aux.update(E.sim.get_physics_dt())

pad_d = E._pad_dists()[0] * 1000
target = E._target_w()[0] - W
print("\n" + "=" * 70)
print(f"pose={args.pose} | 物体 {np.round((E.obj_init_pos).cpu().numpy(),4).tolist()}"
      f" | 对齐目标(接触质心) {np.round(target.cpu().numpy(),4).tolist()}")
print(f"五垫(elastomer 原点)到物体表面距离 mm: "
      f"{[round(float(v),1) for v in pad_d]}")
print("=" * 70)
print("[view_prior] 转视角看接触落点; 关窗口或 Ctrl-C 退出")
while app.is_running():
    E.sim.render()
E.close()
app.close()
