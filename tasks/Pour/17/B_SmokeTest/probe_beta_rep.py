"""squeeze 剂量标定: 4 env 各带 β=1.0/1.5/2.0/3.0, 零动作过提升段, 看谁持住。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("beta")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)      # 前馈不带squeeze, 本探针自配剂量
import pour_env as PE

N = 4
BETA = [2.0, 2.0, 2.0, 2.0]
cfg = PE.build_cfg(num_envs=N)
E = PE.PourEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sqr = np.asarray(np.load("tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
sql = np.asarray(np.load("tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq = np.concatenate([sqr - ref[E.IA0, 14:36], sql - ref[E.IA0, 36:58]])
dsq_t = torch.tensor(dsq, dtype=torch.float32, device=dev)

def drive(rows_per_env, sq_scale):
    tgt = torch.zeros(N, 58, device=dev)
    for i in range(N):
        r = min(rows_per_env[i], E.T_ROW - 1)
        tgt[i] = E.ref58[r]
        tgt[i, 14:] += sq_scale[i] * dsq_t
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

# 第一幕: approach 全员照谱 (无squeeze)
for r in range(0, 166):
    drive([r]*N, [0.0]*N)
# 缝1: 各自剂量渐入 (25步斜坡) + 站稳30步
for i2, r in enumerate(range(166, 190)):
    a = (i2 + 1) / 24.0
    drive([r]*N, [b * a for b in BETA])
for _ in range(30):
    drive([190]*N, BETA)
org = E.scene.env_origins
d0 = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1).clone()
d_l0 = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
print("[beta] 站位垫数: " + " ".join(
    f"β{BETA[i]}:R{int((f[i,:5]>0.5).sum())}/L{int((f[i,5:]>0.5).sum())}"
    for i in range(N)), flush=True)
# 复验: 4env全β2.0, 播完整交互段190..324
for r in range(190, 325):
    drive([r]*N, BETA)
    if r % 20 == 0:
        wz = E.hand.data.body_pos_w[:, E.wid["R"], 2]
        bz_ = E.object.data.root_pos_w[:, 2]
        dd = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1)
        ff = E._pads_f().norm(dim=-1)
        print("[beta] 行%d " % r + " | ".join(
            f"β{BETA[i]}: 腕z={float(wz[i]-org[i,2]):.3f} 瓶z={float(bz_[i]-org[i,2]):.3f} "
            f"距Δ={float((dd[i]-d0[i])*100):+.1f}cm 垫R={int((ff[i,:5]>0.5).sum())}"
            for i in range(N)), flush=True)
# 双侧终判: 瓶(右)+杯(左), 且瓶回到静置附近站立
d_r = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1)
d_l2 = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1)
print("[beta] 终态324行 杯左手距Δ: " + " ".join(
    f"env{i}:{float((d_l2[i]-d_l0[i])*100):+.1f}cm" for i in range(N)), flush=True)
f = E._pads_f().norm(dim=-1)
for i in range(N):
    slip = float((d_r[i] - d0[i]) * 100)
    npr = int((f[i, :5] > 0.5).sum())
    bz = float(E.object.data.root_pos_w[i, 2] - org[i, 2])
    ok = slip < 3 and abs(bz - 0.959) < 0.05  # 324行=放回, 判回静置附近
    print(f"[beta] β={BETA[i]}: 滑移={slip:+.1f}cm 右垫={npr} 瓶z={bz:.3f} "
          f"-> {'✅提住了' if ok else '❌脱手'}", flush=True)
print("[beta] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
