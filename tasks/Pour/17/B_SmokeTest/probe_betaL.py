"""杯侧剂量标定: 右手锁β2.0, 左手扫β=1.0/1.5/2.0/2.5, 零动作播完整交互段。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("betaL")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import pour_env as PE

N = 4
BR = [2.0] * 4
BL = [1.0, 1.5, 2.0, 2.5]
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
dsq_r = torch.tensor(sqr - ref[E.IA0, 14:36], dtype=torch.float32, device=dev)
dsq_l = torch.tensor(sql - ref[E.IA0, 36:58], dtype=torch.float32, device=dev)

def drive(row, sR, sL):
    tgt = torch.zeros(N, 58, device=dev)
    r = min(row, E.T_ROW - 1)
    for i in range(N):
        tgt[i] = E.ref58[r]
        tgt[i, 14:36] += sR[i] * dsq_r
        tgt[i, 36:58] += sL[i] * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

for r in range(0, 166):
    drive(r, [0.0]*N, [0.0]*N)
for i2, r in enumerate(range(166, 190)):
    a = (i2 + 1) / 24.0
    drive(r, [b*a for b in BR], [b*a for b in BL])
for _ in range(30):
    drive(190, BR, BL)
org = E.scene.env_origins
dR0 = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1).clone()
dL0 = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
print("[betaL] 站位垫: " + " ".join(
    f"βL{BL[i]}:R{int((f[i,:5]>0.5).sum())}/L{int((f[i,5:]>0.5).sum())}" for i in range(N)), flush=True)
for r in range(190, 325):
    drive(r, BR, BL)
    if r % 20 == 0:
        dL = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1)
        cz = E.aux.data.root_pos_w[:, 2]
        ff = E._pads_f().norm(dim=-1)
        print(f"[betaL] 行{r} " + " | ".join(
            f"βL{BL[i]}: 杯距Δ={float((dL[i]-dL0[i])*100):+.1f}cm 杯z={float(cz[i]-org[i,2]):.3f} "
            f"垫L={int((ff[i,5:]>0.5).sum())}" for i in range(N)), flush=True)
dL = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1)
dR = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1)
f = E._pads_f().norm(dim=-1)
for i in range(N):
    sl = float((dL[i] - dL0[i]) * 100); sr_ = float((dR[i] - dR0[i]) * 100)
    cz = float(E.aux.data.root_pos_w[i, 2] - org[i, 2])
    ok = sl < 3 and abs(cz - 0.936 - (E.aux.data.root_pos_w[0,2]*0)) < 0.06
    print(f"[betaL] βL={BL[i]}: 杯滑移={sl:+.1f}cm 杯z={cz:.3f} 垫L={int((f[i,5:]>0.5).sum())} "
          f"(瓶滑移={sr_:+.1f}cm) -> {'✅杯保住' if sl < 3 else '❌杯丢'}", flush=True)
print("[betaL] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
