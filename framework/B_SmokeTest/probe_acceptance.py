"""v2 验收硬闸 (动态行号版): βR2.0/βL1.0 零动作照 env.ref58 播全链,
判: 瓶全程滑移<3cm; 杯放回站立(距静置<5cm 且 倾角回静置±15°); 双侧垫台账。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("v2gate")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import task_config as TC
import task_env as PE

N = 4
BR, BL = TC.BETA_R, TC.BETA_L
cfg = PE.build_cfg(num_envs=N)
E = PE.PourEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sqr = np.asarray(np.load(TC.PRIOR_MAIN)["squeeze"], np.float64).reshape(-1)[7:29]
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_r = torch.tensor(BR * (sqr - ref[E.IA0, 14:36]), dtype=torch.float32, device=dev)
dsq_l = torch.tensor(BL * (sql - ref[E.IA0, 36:58]), dtype=torch.float32, device=dev)
APP = E.APP_END

def drive(row, a):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[14:36] += a * dsq_r
    tgt[36:58] += a * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt.unsqueeze(0).expand(N, -1)
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

for r in range(0, APP):
    drive(r, 0.0)
for i, r in enumerate(range(APP, E.IA0)):
    drive(r, (i + 1) / max(E.IA0 - APP, 1))
for _ in range(30):
    drive(E.IA0, 1.0)
org = E.scene.env_origins
dR0 = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1).clone()
dL0 = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
print(f"[v2gate] 全链{E.T_ROW}行 IA=[{E.IA0},{E.IA1}] | 站位垫: " + " ".join(
    f"e{i}:R{int((f[i,:5]>0.5).sum())}/L{int((f[i,5:]>0.5).sum())}" for i in range(N)),
    flush=True)
slipR_max = torch.zeros(N, device=dev)
for r in range(E.IA0, E.IA1 + 1):
    drive(r, 1.0)
    dR = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1)
    slipR_max = torch.maximum(slipR_max, (dR - dR0).abs())
    if (r - E.IA0) % 25 == 0 or r > E.IA1 - 30:
        if (r - E.IA0) % 5 == 0:
            dL = (E.hand.data.body_pos_w[:, E.wid["L"]] - E.aux.data.root_pos_w).norm(dim=1)
            cz = E.aux.data.root_pos_w[:, 2]
            ff = E._pads_f().norm(dim=-1)
            print(f"[v2gate] 行{r} " + " | ".join(
                f"e{i}: 杯距Δ={float((dL[i]-dL0[i])*100):+.1f}cm 杯z={float(cz[i]-org[i,2]):.3f} "
                f"垫L={int((ff[i,5:]>0.5).sum())}" for i in range(N)), flush=True)
for r in range(E.IA1 + 1, E.T_ROW):
    drive(r, max(0.0, 1.0 - (r - E.IA1) / max(E.RETREAT0 - E.IA1, 1)))
for _ in range(20):
    drive(E.T_ROW - 1, 0.0)
cup, bot = E._read_objs()
restc = E.PB.rest[0]; restb = E.PB.rest[1]
def tilt_of(q, up):
    from progress_batch import _tilt
    return torch.rad2deg(_tilt(q, up))
tc = tilt_of(cup[:, 3:7], E.PB.upc); tc0 = tilt_of(restc[3:7].unsqueeze(0), E.PB.upc)[0]
tb = tilt_of(bot[:, 3:7], E.PB.up); tb0 = tilt_of(restb[3:7].unsqueeze(0), E.PB.up)[0]
npass = 0
for i in range(N):
    dc = float((cup[i, :3] - restc[:3]).norm() * 100)
    db = float((bot[i, :3] - restb[:3]).norm() * 100)
    okc = dc < 5 and abs(float(tc[i] - tc0)) < 15
    okb = float(slipR_max[i]) < 0.03 and db < 5 and abs(float(tb[i] - tb0)) < 15
    npass += int(okc and okb)
    print(f"[v2gate] e{i}: 瓶滑移峰={float(slipR_max[i])*100:.1f}cm 瓶距静置={db:.1f}cm "
          f"瓶倾差={float(tb[i]-tb0):+.0f}° | 杯距静置={dc:.1f}cm 杯倾差={float(tc[i]-tc0):+.0f}° "
          f"-> {'✅' if okc and okb else '❌'}", flush=True)
print(f"[v2gate] ★硬闸: {npass}/4 通过 (要求≥3)", flush=True)
print("[v2gate] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
