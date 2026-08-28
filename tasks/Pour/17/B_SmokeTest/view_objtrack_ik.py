"""物体轨迹反解IK查看器 (2026-08-29 用户构想):
Approach照谱 → squeeze稳物(物理全开) → 捕获抓变换 T = obj⁻¹∘wrist →
交互段: 手腕目标 = 物体参考行 ∘ T, 逐行ArmIK, 物体自由由物理携带 →
物体轨迹走完 → 融接回 Retreat。

用法(GUI观看): SHARPA_WANDB=0 POUR_SQUEEZE_FF=1 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
  $PY tasks/Pour/17/B_SmokeTest/view_objtrack_ik.py
加 --headless --selftest 只出数据报告。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--selftest", action="store_true", help="无GUI只报逐行跟踪偏差")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("objtrack_ik")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.setdefault("POUR_SQUEEZE_FF", "1")
import pour_env as PE
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R, R_to_quat

R_ = not args.headless
cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
E.force_entry = [0]          # 锁 t0 出生 (否则RSI随机缝点, 横扫撞飞物体)
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))

def stepper(n=1, render=R_):
    for _ in range(n * DECI):
        E.scene.write_data_to_sim()
        E.sim.step(render=render)
        E.scene.update(E.sim.get_physics_dt())

def set_row_target(r):
    tgt = E._ff_row(torch.tensor([r], device=dev))[0]
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    return tgt

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qrot(q, v):
    return quat_to_R(q) @ np.asarray(v, np.float64)

print("[objtrack] 第一幕: Approach 照谱 (0..189, 含缝1 squeeze 渐入)", flush=True)
for r in range(0, 190):
    set_row_target(r)
    stepper(1)
print("[objtrack] 第二幕: 站位 squeeze 稳物 30 步", flush=True)
for _ in range(30):
    set_row_target(190)
    stepper(1)
org = E.scene.env_origins[0].cpu().numpy()
# ★增量空间法 (dp_make_variants 同款, 免坐标系混叠):
# 腕目标[k] = T_k ∘ 腕模型系FK(当前q), T_k = 物体参考行k 相对首行的刚体运动
ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
ref_obj = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}
Nrow = E.PB.N_ROW
q_ik = {s: np.zeros((Nrow, 7)) for s in ("right", "left")}
q_seed, w0 = {}, {}
for s in ("right", "left"):
    qn = E.hand.data.joint_pos[0, E.map_ids_t[:7] if s == "right"
                               else E.map_ids_t[7:14]].cpu().numpy()
    q_seed[s] = qn.astype(np.float64)
    w0[s] = ik[s].fk(q_seed[s])
fail_ik = 0
for k in range(Nrow):
    for s in ("right", "left"):
        oi = side_obj[s]
        p0, q0_ = ref_obj[oi][0][:3], ref_obj[oi][0][3:7]
        pk, qk_ = ref_obj[oi][k][:3], ref_obj[oi][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tp = pk - Rk @ p0
        wp_t = Rk @ w0[s][0] + tp
        wR_t = Rk @ w0[s][1]
        r = ik[s].solve(wp_t, wR_t, q0=q_seed[s], iters=60)
        if r["pos_err"] > 0.01:
            fail_ik += 1
        q_ik[s][k] = r["q"]
        q_seed[s] = np.asarray(r["q"], np.float64)
print(f"[objtrack] IK 预解完毕 (增量空间法, >1cm 失败 {fail_ik}/{Nrow*2})", flush=True)
# 第三幕: 物体轨迹驱动 (手指钉 squeeze 形, 物体自由)
fin_hold = E._ff_row(torch.tensor([190], device=dev))[0][14:].clone()
dev_log = []
for k in range(Nrow):
    tgt = torch.zeros(58, device=dev)
    tgt[:7] = torch.tensor(q_ik["right"][k], device=dev, dtype=torch.float32)
    tgt[7:14] = torch.tensor(q_ik["left"][k], device=dev, dtype=torch.float32)
    tgt[14:] = fin_hold
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    stepper(1)
    bp = E.object.data.root_pos_w[0].cpu().numpy() - org
    dev_log.append(np.linalg.norm(bp - ref_obj[1][k][:3]))
    if k % 20 == 0:
        f = E._pads_f().norm(dim=-1)[0]
        print(f"[objtrack] k={k:3d} 瓶跟踪偏差={dev_log[-1]*100:.1f}cm "
              f"垫R={int((f[:5]>0.5).sum())} 垫L={int((f[5:]>0.5).sum())}", flush=True)
dl = np.array(dev_log) * 100
print(f"[objtrack] ★交互段瓶跟踪: 均值={dl.mean():.1f}cm P95={np.percentile(dl,95):.1f}cm "
      f"最大={dl.max():.1f}cm", flush=True)
# 第四幕: 融接 Retreat (25 步混回撤退首行, 后照谱)
q_now = E.hand.data.joint_pos[0, E.map_ids_t].cpu().numpy()
ret0 = E.ref58[E.RETREAT0].cpu().numpy()
for i in range(25):
    a = (i + 1) / 25.0
    mix = (1 - a) * q_now + a * ret0
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = torch.tensor(mix, device=dev, dtype=torch.float32)
    E.hand.set_joint_position_target(full)
    stepper(1)
for r in range(E.RETREAT0, E.T_ROW):
    set_row_target(r)
    stepper(1)
bp = E.object.data.root_pos_w[0].cpu().numpy() - org
cp = E.aux.data.root_pos_w[0].cpu().numpy() - org
print(f"[objtrack] 终态: 瓶距静置={np.linalg.norm(bp-ref_obj[1][0][:3])*100:.1f}cm "
      f"杯距静置={np.linalg.norm(cp-ref_obj[0][0][:3])*100:.1f}cm", flush=True)
if R_:
    print("[objtrack] GUI 保持, Ctrl-C 退出", flush=True)
    try:
        while app.is_running():
            stepper(1)
    except KeyboardInterrupt:
        pass
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
