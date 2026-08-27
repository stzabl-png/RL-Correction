"""M1 物理可达性放音 (2026-08-27 拍板开工, 运行层纪律#19):

母带零策略播放(全 44 指关节+双臂逐帧驱动, 物体钳位静置), 播到交互首帧(完整抓形焊接位)
后定格 120 步, 实测:
  ① M1 触发了吗 / 第几步触发 —— 判据(拍板版, 去向心): 右手 >=4 垫 接触力>0.5N
     连续 candidate_hold_steps(4) 步; 左手以杯净接触力为聚合代理(本env左垫无独立传感器)
  ② 逐垫实际接触力(右手5垫) + 瓶/杯净接触力
结果人读 -> B_SmokeTest/REPORT.md, 判决入 DECISIONS.md。
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--hold", type=int, default=120)
p.add_argument("--lshift", default="", help="左腕再对中平移 'dx,dy[,dz]' (m), 实验校准用")
p.add_argument("--squeeze", type=int, default=0,
               help="1=定格时渐进收紧到 squeeze 位形(2026-08-27 用户裁定: 斜坡不瞬跳)")
p.add_argument("--ramp", type=int, default=60, help="收紧斜坡步数")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("smoke_m1")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
# 双手版 (2026-08-27): 左手五垫加装传感器, 过滤到杯(Aux) —— cfg 列表制, 建env前覆盖即可
from isaaclab.sensors import ContactSensorCfg
_LPADS = ["left_thumb_elastomer", "left_index_elastomer", "left_middle_elastomer",
          "left_ring_elastomer", "left_pinky_elastomer"]
cfg.contact_sensors = list(cfg.contact_sensors) + [
    ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}", history_length=1,
                     filter_prim_paths_expr=["/World/envs/env_.*/Aux"])
    for n in _LPADS]
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0


class SmokeEnv(GraspTaskEnv):
    """env 内部形状写死 5 传感器 —— 场景注册全部10个, 内部只暴露右5, 冒烟读全10."""

    def _setup_scene(self):
        super()._setup_scene()
        self._all_sensors = list(self._contact_sensors)
        self._contact_sensors = self._all_sensors[:5]


E = SmokeEnv(cfg)
E.reset()
for _ in range(90):
    E.scene.write_data_to_sim()
    E.sim.step(render=not args.headless)
    E.scene.update(E.sim.get_physics_dt())

z = np.load("tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v1.npz",
            allow_pickle=True)
jn = list(E.hand.joint_names)
fin = [str(n) for n in z["fin_names"]]
src = np.asarray(z["source"])
seg_end = int(np.where(src == 1)[0][0])          # 交互首行 = 完整抓形焊接位
print(f"[M1冒烟] 播放行 0..{seg_end} (approach+seam1), 然后定格 {args.hold} 步")

_objs0 = []
for _ob in (E.object, E.aux):
    _objs0.append((_ob, _ob.data.root_state_w.clone()))

q = E.hand.data.default_joint_pos.clone()


def set_row(t, write=True):
    for s, P in (("right", "R"), ("left", "L")):
        for i in range(7):
            q[:, jn.index(f"{P}_arm_j{i+1}")] = float(z[f"{s}_q"][t, i])
        for n, v in zip(fin, z[f"{s}_f"][t]):
            q[:, jn.index(n.replace("right_", f"{s}_"))] = float(v)
    if write:
        E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    for _ob, _st in _objs0:
        _ob.write_root_pose_to_sim(_st[:, :7])
        _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))


def pads():
    F = torch.cat([s_.data.force_matrix_w.view(1, 1, 3)
                   for s_ in E._all_sensors], dim=1).nan_to_num(0.0)
    f = F.norm(dim=-1)[0].cpu().numpy()           # (10,) 前5右手vs瓶, 后5左手vs杯
    return f[:5], f[5:]


def net(ob):
    try:
        f = ob.root_physx_view.get_net_contact_forces(E.sim.get_physics_dt())
        return float(torch.as_tensor(f).view(-1, 3).norm(dim=-1)[0])
    except Exception:
        return float("nan")


MINPADS, FTH, HOLD_N = int(cfg.success_min_pads), 0.5, int(cfg.candidate_hold_steps)
print(f"[M1冒烟] 判据(去向心版): 右手>= {MINPADS} 垫 力>{FTH}N 连续{HOLD_N}步")

if args.squeeze:
    zs_a = np.load("tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
    zs_b = np.load("tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")
    sq = {"right": np.asarray(zs_a["squeeze"], np.float64).reshape(-1)[7:29],
          "left": np.asarray(zs_b["squeeze"], np.float64).reshape(-1)[7:29]}
    print("[M1冒烟] squeeze 模式: 定格时手指压到 squeeze 位形")

for t in range(seg_end + 1):
    set_row(t)
    E.scene.write_data_to_sim()
    E.sim.step(render=not args.headless)
    E.scene.update(E.sim.get_physics_dt())

if args.lshift:
    from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
    _vs = [float(v) for v in args.lshift.split(",")]
    _dx, _dy = _vs[0], _vs[1]
    _dz = _vs[2] if len(_vs) > 2 else 0.0
    bn0 = list(E.hand.body_names)
    _wp0 = E.hand.data.body_pos_w[0, bn0.index("left_hand_C_MC")].cpu().numpy()
    _wq0 = E.hand.data.body_quat_w[0, bn0.index("left_hand_C_MC")].cpu().numpy()
    _tgt = _wp0.copy(); _tgt[0] += _dx; _tgt[1] += _dy; _tgt[2] += _dz
    _ikL = ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)
    _seed = np.array([float(q[0, jn.index(f"L_arm_j{i}")]) for i in range(1, 8)])
    _rL = _ikL.solve(_tgt - E.scene.env_origins[0].cpu().numpy(),
                     quat_to_R(_wq0), q0=_seed, iters=200)
    print(f"[M1冒烟] 左腕再对中 平移[{_dx*100:.1f},{_dy*100:.1f}]cm "
          f"IK误差{_rL['pos_err']*100:.2f}cm")
    _LQ = [float(v) for v in _rL["q"]]
    globals()["_LQ"] = _LQ

run = {"R": 0, "L": 0}
fire = {"R": -1, "L": -1}
hist = []
_LQ = globals().get("_LQ")     # lshift 块在上方已可能赋值, 勿盖 (2026-08-27 顺序bug)


_ARM_IDS = [jn.index(f"{P}_arm_j{i}") for P in ("R", "L") for i in range(1, 8)]


def set_hold(alpha=1.0):
    """2026-08-27 用户裁定'实心'体制: 物体钳位(=实心不动), **手指改位置目标驱动**
    (执行器有限刚度) —— 指碰壁即被接触力停住, 穿模物理消失, 压力=持续真值。
    手臂仍写死状态(刚性держ持), 只有手指走柔顺通道。"""
    set_row(seg_end, write=False)      # 只填目标, 不写死状态 (柔顺体制)
    global _LQ
    if _LQ is not None:
        for i in range(7):
            q[0, jn.index(f"L_arm_j{i+1}")] = _LQ[i]
    if args.squeeze:
        gsp = {"right": np.asarray(z["right_f"][seg_end], np.float64),
               "left": np.asarray(z["left_f"][seg_end], np.float64)}
        for s in ("right", "left"):
            fv = (1 - alpha) * gsp[s] + alpha * sq[s]     # 渐进收紧
            for n, v in zip(fin, fv):
                q[:, jn.index(n.replace("right_", f"{s}_"))] = float(v)
    # 臂: 写死状态(仅臂关节); 指: 只发位置目标, 由执行器柔性推进
    E.hand.write_joint_state_to_sim(q[:, _ARM_IDS], torch.zeros_like(q[:, _ARM_IDS]),
                                    joint_ids=_ARM_IDS)
    E.hand.set_joint_position_target(q)
    E.hand.write_data_to_sim()
    for _ob, _st in _objs0:
        _ob.write_root_pose_to_sim(_st[:, :7])
        _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))


first_contact = {"R": [None]*5, "L": [None]*5}
for k in range(args.hold):
    _a = min(1.0, k / max(args.ramp - 1, 1)) if args.squeeze else 1.0
    set_hold(_a)
    E.scene.write_data_to_sim()
    E.sim.step(render=not args.headless)
    E.scene.update(E.sim.get_physics_dt())
    fR, fL = pads()
    nR, nL = int((fR > FTH).sum()), int((fL > FTH).sum())
    hist.append((fR, fL, nR, nL))
    for tag, n_ in (("R", nR), ("L", nL)):
        run[tag] = run[tag] + 1 if n_ >= MINPADS else 0
        if run[tag] >= HOLD_N and fire[tag] < 0:
            fire[tag] = k
    for _i5 in range(5):
        if first_contact["R"][_i5] is None and fR[_i5] > FTH:
            first_contact["R"][_i5] = round(_a, 2)
        if first_contact["L"][_i5] is None and fL[_i5] > FTH:
            first_contact["L"][_i5] = round(_a, 2)
    if k % 10 == 0 or k == args.hold - 1:
        print(f"[M1冒烟] k{k:3d} α={_a:.2f}: 右垫N {np.round(fR, 1)} ({nR}垫) "
              f"| 左垫N {np.round(fL, 1)} ({nL}垫)", flush=True)

# 左手逐垫几何间隙: 垫体到杯轴径向距 - 杯半径3.5cm (负=压入)
bn = list(E.hand.body_names)
cup = E.aux.data.root_state_w[0].cpu().numpy()
FN = ("thumb", "index", "middle", "ring", "pinky")
_ctop = cup[2] + 0.066
print("[M1冒烟] 左垫高度-杯口(cm, 正=悬在杯口上方):", {
    f_: round(float((E.hand.data.body_pos_w[0, bn.index(f"left_{f_}_elastomer")]
                     .cpu().numpy()[2] - _ctop) * 100), 2) for f_ in FN})
print("[M1冒烟] 左垫→杯面间隙(cm, 负=压入):", {
    f_: round(float(np.linalg.norm(
        E.hand.data.body_pos_w[0, bn.index(f"left_{f_}_elastomer")].cpu().numpy()[:2]
        - cup[:2]) * 100 - 3.5), 2) for f_ in FN})
_pads_xy = np.stack([E.hand.data.body_pos_w[0, bn.index(f"left_{f_}_elastomer")]
                     .cpu().numpy()[:2] for f_ in FN])
_cen = _pads_xy.mean(0)
print(f"[M1冒烟] 左五垫质心 vs 杯心 水平偏差 = "
      f"{np.round((_cen - cup[:2]) * 100, 2)} cm (腕位反向平移即再对中)")

FN0 = ("thumb", "index", "middle", "ring", "pinky")
print("[M1冒烟] 首次接触的收紧深度α: 右", dict(zip(FN0, first_contact["R"])),
      "| 左", dict(zip(FN0, first_contact["L"])))
fRm = np.stack([h[0] for h in hist]).mean(0)
fLm = np.stack([h[1] for h in hist]).mean(0)
print(f"[M1冒烟] 定格均值: 右垫 {np.round(fRm, 2)} | 左垫 {np.round(fLm, 2)}")
for tag, nm in (("R", "右手vs瓶"), ("L", "左手vs杯")):
    if fire[tag] >= 0:
        print(f"[M1冒烟] ✅ M1({nm}) 触发 @定格第{fire[tag]}步 "
              f"(全程第{seg_end + 1 + fire[tag]}步)")
    else:
        print(f"[M1冒烟] ❌ M1({nm}) 未触发")
if fire["R"] >= 0 and fire["L"] >= 0:
    print(f"[M1冒烟] ✅✅ 双手 AND @定格第{max(fire.values())}步")
if not args.headless:
    print("[M1冒烟] GUI 模式: 定格保持, 关窗或 Ctrl-C 退出", flush=True)
    try:
        while app.is_running():
            set_hold(1.0)
            E.scene.write_data_to_sim()
            E.sim.step(render=True)
            E.scene.update(E.sim.get_physics_dt())
    except KeyboardInterrupt:
        pass
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
