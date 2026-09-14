"""Sweep408 抓握力探针: 保持 GraspPose → 加力(合拢) → 向上提, 全程记录指垫压力。

回答的问题 (用户 2026-09-11): 右手拇指在柄左、食指在柄上、中指在柄右, 三指同时使劲, 能不能把扫把提起来?

  GUI:      bash gui_sweep408_lift.sh              (窗口起来后终端按 Enter 推进阶段)
  自检:     ... probe_lift.py --headless --no_wait  (不等 Enter, 直接跑完三阶段)

三阶段:
  A 保持   臂 = 母带第 0 行, 指 = GraspPose `grasp` 姿, 物体按物体听手摆好 (扫把头搁在桌上, 柄在张开的手里)。
          物理全开, 静置若干步让接触建立。→ 打印初始指垫力
  B 加力   指 grasp → `squeeze` 斜坡合拢, 逐步打印五指垫力 (这就是"三指使劲")
  C 提起   腕目标沿 +z 逐步抬 lift_cm, 每步记录 物体 z / 手 z / 掌系相对位姿漂移
          判据: 物体跟着上来 (Δ物体z ≈ Δ手z) = 提得起来; 物体留在桌上 = 提不起来
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--side", default="right", choices=("right", "left", "both"))
p.add_argument("--settle_steps", type=int, default=40, help="A 阶段静置步数")
p.add_argument("--squeeze_steps", type=int, default=60, help="B 阶段合拢斜坡步数")
p.add_argument("--lift_cm", type=float, default=10.0)
p.add_argument("--lift_steps", type=int, default=80)
p.add_argument("--no_wait", action="store_true", help="不等 Enter (自检用)")
p.add_argument("--render", action="store_true")
p.add_argument("--prior", default=None, help="换扫把先验 (只改指姿/标注, 不移动手 —— 见 --shift_cm)")
p.add_argument("--shift_cm", type=float, default=0.0,
               help="把**手**沿把手真实中心线朝头端移动多少 cm (握得更靠近扫把头 = 减小力臂)。"
                    "⚠ 平移先验的 p_oh 没用: 探针里臂与物体都取自母带第 0 行, p_oh 不参与摆放。")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
os.environ.setdefault("SHARPA_WANDB", "0")
os.environ.setdefault("POUR_OBJ_MASS", "0.15")
os.environ.setdefault("POUR_OBJ_FRIC", "1.0")
os.environ.setdefault("POUR_PAD_FRIC", "1.0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep408_lift")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import grip_env as GE  # noqa: E402
import task_config as TC  # noqa: E402

FING = ("thumb", "index", "middle", "ring", "pinky")
# ★SIDES = **实际动作**的手 (加力 + 抬臂), 不只是打印过滤。另一只手全程停在 grasp 姿、臂不动,
#   它的物体就搁在桌上不管 —— 用户 2026-09-11: "我只想看右手施加力 右手提起 不是双手"。
SIDES = (args.side,) if args.side != "both" else GE.SIDES


def wait(msg):
    print(f"\n>>> {msg}", flush=True)
    if args.no_wait:
        return
    try:
        input()
    except EOFError:
        pass


if args.prior:                      # 必须在 build_cfg 之前 —— task_config 是导入期读环境变量
    import os as _o
    TC.PRIOR_BROOM = _o.path.join(TC.REPO, "tasks/pregrasp/priors", args.prior + ".npz")
    GE.TC.PRIOR_BROOM = TC.PRIOR_BROOM
    print(f"[probe_lift] 扫把先验换成 {TC.PRIOR_BROOM}", flush=True)
E = GE.Sweep408GripEnv(GE.build_cfg(1))
E.reset()
dev = E.device
IK = {s: __import__("rl_rebuild.correction.kinematics", fromlist=["ArmIK"]).ArmIK(
    s, anchor_link="arm_center", anchor_T=E._anchor_T) for s in GE.SIDES}


def pads(side):
    return E._pad_force_mat(side)[0].cpu().numpy()


def show(tag, side):
    f = pads(side)
    dp, dr = E._rel(side)
    op = E._obj_pose(side)[0][0].cpu().numpy()
    hp = E._hand_pose(side)[0][0].cpu().numpy()
    print(f"  [{tag}] {side:5s} 指垫力 拇{f[0]:5.2f} 食{f[1]:5.2f} 中{f[2]:5.2f} 无名{f[3]:5.2f} 小{f[4]:5.2f} N"
          f" | >0.5N 的垫 {int((f > 0.5).sum())} | 物体z {op[2]:.4f} 手z {hp[2]:.4f}", flush=True)
    return f, op, hp


def step_phys(qfull, n):
    for _ in range(n):
        E.hand.set_joint_position_target(qfull)
        E.hand.write_data_to_sim()
        E.sim.step(render=bool(args.render))
        E.hand.update(E.dt); E.object.update(E.dt); E.aux.update(E.dt)
        for s_ in getattr(E, "_all_sensors", []):
            s_.update(E.dt)


# ---------- 握点前移: 把**手**沿把手中心线朝头端挪 ----------
ARM_Q0 = E.ref_arm[0].clone()
if abs(args.shift_cm) > 1e-9:
    import trimesh
    from rl_rebuild.correction import clips as _clips
    _mp = _clips.clip_entry(TC.CLIP)["mesh"]
    _V = np.asarray(trimesh.load(_mp, process=False).vertices, np.float64)
    _z0 = _V[:, 2].min()
    _C = [_V[(_V[:, 2] >= _z0 + i * 0.01) & (_V[:, 2] < _z0 + (i + 1) * 0.01)].mean(0) for i in range(11)
          if len(_V[(_V[:, 2] >= _z0 + i * 0.01) & (_V[:, 2] < _z0 + (i + 1) * 0.01)]) >= 8]
    _C = np.asarray(_C); _u, _s, _vt = np.linalg.svd(_C - _C.mean(0), full_matrices=False)
    _axL = _vt[0] * np.sign(_vt[0][2])                       # 把手中心线 (物体局部系)
    _R = GE._quat_to_mat(E.nominal["right"][1].unsqueeze(0))[0].cpu().numpy().astype(np.float64)
    _axW = _R @ _axL                                          # 换到世界系
    _ik = IK["right"]; _q0 = ARM_Q0[:7].cpu().numpy().astype(np.float64)
    _p, _Rw = _ik.fk(_q0)
    _r = _ik.solve(_p + _axW * (args.shift_cm / 100.0), _Rw, q0=_q0, iters=200)
    ARM_Q0[:7] = torch.tensor(_r["q"], dtype=torch.float32, device=dev)
    print(f"[probe_lift] 握点前移 {args.shift_cm:.1f}cm 沿把手中心线(世界系 {np.round(_axW,3)}); "
          f"腕 IK 误差 {_r['pos_err']*100:.2f}cm 姿态 {np.degrees(_r['rot_err']):.1f}°", flush=True)

# ---------- A 保持 ----------
q = E.hand.data.default_joint_pos.clone()
q[:, E.map_ids_t] = ARM_Q0 + E.arm_sag
for s in GE.SIDES:
    q[:, E.fid[s]] = E.f_grasp[s]
E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
E._pin()                                   # 物体按母带第 0 行摆好 (= 物体听手的结果)
step_phys(q, 4)
print(f"\n===== A 保持 GraspPose | 作用手 = {'/'.join(SIDES)} (另一只手全程不动) =====", flush=True)
step_phys(q, int(args.settle_steps))
for s in SIDES:
    show("A 静置后", s)
wait("按 Enter 开始【加力】(指 grasp→squeeze 合拢)")

# ---------- B 加力 ----------
print(f"\n===== B 加力: 指 grasp → squeeze, {args.squeeze_steps} 步 =====", flush=True)
N = int(args.squeeze_steps)
for i in range(1, N + 1):
    a = i / N
    qq = q.clone()
    for s in SIDES:                                  # 只有作用手合拢
        qq[:, E.fid[s]] = (1 - a) * E.f_grasp[s] + a * E.f_squeeze[s]
    step_phys(qq, 1)
    if i % max(N // 6, 1) == 0 or i == N:
        for s in SIDES:
            show(f"B {100*a:3.0f}%", s)
q = qq
base = {s: (E._obj_pose(s)[0][0, 2].item(), E._hand_pose(s)[0][0, 2].item()) for s in SIDES}
rel0 = {s: (E._rel(s)[0][0].clone(), E._rel(s)[1][0].clone()) for s in SIDES}
wait("按 Enter 开始【向上提】")

# ---------- C 提起 ----------
print(f"\n===== C 向上提 {args.lift_cm:.1f}cm / {args.lift_steps} 步 =====", flush=True)
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
seed = {s: ARM_Q0[(0 if s == "right" else 7):(7 if s == "right" else 14)].cpu().numpy().astype(np.float64)
        for s in SIDES}
tgt0 = {}
for s in SIDES:
    pp, RR = IK[s].fk(seed[s])
    tgt0[s] = (pp.copy(), RR.copy())
L = int(args.lift_steps)
for i in range(1, L + 1):
    dz = args.lift_cm / 100.0 * i / L
    qq = q.clone()
    for s in SIDES:                                  # 只有作用手的臂上抬, 另一只臂停在母带第 0 行
        r = IK[s].solve(tgt0[s][0] + np.array([0, 0, dz]), tgt0[s][1], q0=seed[s], iters=120)
        seed[s] = r["q"]
        sl = slice(0, 7) if s == "right" else slice(7, 14)
        qq[:, E.map_ids_t[sl]] = torch.tensor(r["q"], dtype=torch.float32, device=dev)
    step_phys(qq, 1)
    if i % max(L // 8, 1) == 0 or i == L:
        for s in SIDES:
            f, op, hp = show(f"C 抬{dz*100:4.1f}cm", s)
            d_obj = op[2] - base[s][0]; d_hand = hp[2] - base[s][1]
            rp, rq = E._rel(s)
            drift = float(torch.linalg.vector_norm(rp[0] - rel0[s][0])) * 100
            rot = float(torch.rad2deg(GE._qangle(rq[:1], rel0[s][1].unsqueeze(0))))
            print(f"         ↳ 物体升 {d_obj*100:+6.2f}cm / 手升 {d_hand*100:+6.2f}cm"
                  f" | 跟随率 {100*d_obj/max(d_hand,1e-6):5.1f}% | 掌系漂移 {drift:.2f}cm/{rot:.1f}°", flush=True)
print("\n===== 结论 =====", flush=True)
for s in SIDES:
    op = E._obj_pose(s)[0][0, 2].item(); hp = E._hand_pose(s)[0][0, 2].item()
    d_obj = op - base[s][0]; d_hand = hp - base[s][1]
    ok = d_obj > 0.7 * d_hand
    print(f"  {s:5s}: 手升 {d_hand*100:+.2f}cm, 物体升 {d_obj*100:+.2f}cm → "
          f"{'✅ 提得起来' if ok else '❌ 提不起来 (物体留在桌上/滑脱)'}", flush=True)
if args.render:
    print("\n(窗口仍开着, Ctrl-C 退出)", flush=True)
    while app.is_running():
        E.sim.step(render=True)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
