"""合成完整轨迹 (手臂 cuRobo + 手指开合) 并独立核验.

核验用**自研 numpy FK** 重算 R_ee/hand_C_MC 与手上每个 link 的世界位姿 ——
与 cuRobo 是两套独立实现, 对得上才算数.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
SP = "/home/lyh/Project/RL_Correction/tools/grasp_design"
from rl_rebuild.correction.kinematics import Urdf, _T, DEFAULT_TORSO_DEG

TZ = 0.85
BASE_OFF = np.array([0.5, 0.0, 0.0])
N_SQUEEZE, N_RELEASE = 30, 20

import os
g = np.load(f"{SP}/{os.environ.get('GRIP_NPZ','grip_cfg.npz')}", allow_pickle=True)
a = np.load(f"{SP}/arm_traj.npz", allow_pickle=True)
q_open, q_grip = g["q_open"], g["q_grip"]
FJN = [str(s) for s in g["joint_names"]]
AJN = [str(s) for s in a["joint_names"]]
dt = float(a["dt"])
segs = [(k, a[k]) for k in a.files if k.startswith("seg")]
segs.sort(key=lambda t: int(t[0][3:].split("_")[0]))
print(f"手臂段: {[(k, v.shape[0]) for k, v in segs]}   dt={dt*1000:.1f}ms")

# ---- 拼接: 段间插入手指开合 (手臂静止) ----
Q_ARM, Q_FIN, PHASE = [], [], []


def push(arm_block, fin_block, tag):
    Q_ARM.append(np.asarray(arm_block)); Q_FIN.append(np.asarray(fin_block))
    PHASE.extend([tag] * len(arm_block))


def ramp(q0, q1, n):
    u = np.linspace(0, 1, n)
    u = (u * u * (3 - 2 * u))[:, None]
    return (1 - u) * q0[None] + u * q1[None]


# 7 段: P1接近 P2下降 P3提起 P4水平 P5下降 P6撤离 P7回家
# 夹紧在 P2 末 (到抓取位), 松开在 P5 末 (降回桌面高度) -> 持物段 = P3..P5.
# **按段名判定, 不按下标** —— 段数变过一次 (抬升从两段合成一段, 又加了回家段),
# 硬编码 range(2,5) 会整体错位, 表现为"还没夹紧就开始搬"或"举着物体回家".
SQUEEZE_AFTER, RELEASE_AFTER = "P2", "P5"
HELD_P = ("P3", "P4", "P5")


def tag_of(k):        # "seg3_P4" -> "P4"
    return k.split("_", 1)[1]


for i, (name, arm) in enumerate(segs):
    ph_i = tag_of(name)
    push(arm, np.repeat((q_grip if ph_i in HELD_P else q_open)[None], len(arm), 0), name)
    if ph_i == SQUEEZE_AFTER:   # 到抓取位: 五指同时向圆心夹紧 (手臂不动)
        push(np.repeat(arm[-1:], N_SQUEEZE, 0), ramp(q_open, q_grip, N_SQUEEZE), "SQUEEZE")
    if ph_i == RELEASE_AFTER:   # 降回桌面高度: 松开
        push(np.repeat(arm[-1:], N_RELEASE, 0), ramp(q_grip, q_open, N_RELEASE), "RELEASE")
missing = [p for p in (SQUEEZE_AFTER, RELEASE_AFTER) + HELD_P
           if p not in {tag_of(k) for k, _ in segs}]
if missing:
    raise SystemExit(f"❌ arm_traj.npz 里没有段 {missing} —— 和 plan_curobo.py 的 PLAN 对不上")

QA, QF = np.concatenate(Q_ARM), np.concatenate(Q_FIN)
print(f"合成轨迹 {len(QA)} 帧 / {len(QA)*dt:.2f}s")

# ---- 独立 FK 核验 ----
u = Urdf()
TORSO = {k: np.deg2rad(v) for k, v in DEFAULT_TORSO_DEG.items()}
BASE_T = _T(np.eye(3), np.array([-0.5, 0.0, 0.0]))
HB = "right_hand_C_MC"
CHK = [l for f in ("thumb", "index", "middle", "ring", "pinky")
       for s in ("PP", "MP", "DP", "elastomer", "fingertip")
       for l in [f"right_{f}_{s}"] if l in u.parent_joint]

lows, ee, LP = [], [], []
for t in range(len(QA)):
    q = dict(TORSO)
    q.update({n: float(v) for n, v in zip(AJN, QA[t])})
    q.update({n: float(v) for n, v in zip(FJN, QF[t])})
    Th = u.link_pose(HB, q, BASE_T)
    ee.append(Th[:3, 3])
    P = np.array([u.link_pose(l, q, Th, HB)[:3, 3] for l in CHK])
    LP.append(P)
    lows.append(P[:, 2].min())
ee, lows, LP = np.array(ee), np.array(lows), np.array(LP)

print(f"\n[核验] 手上最低 link 离桌: 全程最小 {(lows.min()-TZ)*100:+.2f}cm  "
      f"穿桌帧 {(lows < TZ).sum()}/{len(lows)}")

# 松手之后 (撤离 + 回家) 不许再蹭到已放下的物体 —— 这两段是纯关节插值/直线,
# cuRobo 的场景里根本没有物体, 只能在这里用 FK 兜底.
ph_all = np.array(PHASE)
if "obj_end" in a.files:
    OE = a["obj_end"]
    R_OBJ = 0.0386
    post = np.zeros(len(QA), bool)
    seen_rel = False
    for t, tag in enumerate(ph_all):
        seen_rel = seen_rel or tag == "RELEASE"
        post[t] = seen_rel and tag != "RELEASE"
    if post.any():
        dmin = np.linalg.norm(LP[post] - OE[None, None, :], axis=2).min()
        print(f"[核验] 松手后 (撤离+回家 {int(post.sum())} 帧) 手到已放下物体中心最近 "
              f"{dmin*100:.2f}cm  (物体半径 {R_OBJ*100:.2f}cm -> 净间隙 "
              f"{(dmin-R_OBJ)*100:+.2f}cm) {'✅' if dmin > R_OBJ + 0.005 else '❌ 会蹭到'}")
print(f"[核验] 手基座轨迹: 最低 {(ee[:,2].min()-TZ)*100:.2f}cm  最高 {(ee[:,2].max()-TZ)*100:.2f}cm")
ph = np.array(PHASE)
print(f"\n{'阶段':<26}{'帧':>6}{'时长s':>8}{'手最低-桌cm':>14}{'手基座末位置':>34}")
for tag in dict.fromkeys(PHASE):
    m = ph == tag
    print(f"{tag:<26}{m.sum():>6}{m.sum()*dt:>8.2f}{(lows[m].min()-TZ)*100:>14.2f}"
          f"{str(np.round(ee[m][-1],4).tolist()):>34}")

jump = np.degrees(np.abs(np.diff(QA, axis=0)).max(axis=1))
print(f"\n[核验] 手臂关节步间最大跳变 {jump.max():.2f}°/步 ({jump.max()/dt:.1f}°/s)")
if "q_home" in a.files:
    d_home = np.degrees(np.abs(QA[-1] - a["q_home"])).max()
    d_start = np.degrees(np.abs(QA[0] - a["q_home"])).max()
    print(f"[核验] 首帧 vs env 站姿 最大关节差 {d_start:.3f}°  "
          f"{'✅ 从初始位姿出发' if d_start < 0.01 else '❌ 起点不是 env 站姿'}")
    print(f"[核验] 末帧 vs env 站姿 最大关节差 {d_home:.3f}°  "
          f"{'✅ 已回到初始位姿' if d_home < 0.5 else '❌ 没回到初始位姿'}")
np.savez_compressed(f"{SP}/full_traj.npz", dt=dt, arm=QA, finger=QF,
                    arm_joints=np.array(AJN), finger_joints=np.array(FJN),
                    phase=ph, hand_base_pos=ee, hand_lowest_z=lows)
print(f"\n-> {SP}/full_traj.npz  ({len(QA)} 帧, 手臂 {QA.shape[1]} 关节 + 手指 {QF.shape[1]} 关节)")
