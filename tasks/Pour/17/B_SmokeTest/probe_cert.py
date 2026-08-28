"""认证显微探针: 1env g1出生零动作, 逐步打印认证机全量."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("cert")
app = AppLauncher(args).app
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"; os.environ["POUR_SQUEEZE_FF"] = "1"
os.environ["POUR_UNLOCK"] = "1,2,3"
import pour_env as PE
cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
gi = [i for i, e in enumerate(E.entries) if e[4] == "g1"][0]
E.force_entry = [gi]
E.reset()
z = torch.zeros(1, PE.ACT_DIM, device=E.device)
zb0 = zc0 = None
for t in range(140):
    obs, rew, term, trunc, _ = E.step(z)
    cup, bot = E._read_objs()
    f = E._pads_f().norm(dim=-1)
    wr = E.hand.data.body_pos_w[0, E.wid["R"], 2]
    if zb0 is None:
        zb0, zc0 = float(bot[0, 2]), float(cup[0, 2])
    ph, ct = int(E.PB.cert_phase[0]), int(E.PB.cert_t[0])
    if t % 2 == 0 or ph:
        print(f"[cert] t{t:3d} g1={int(E.PB.g1[0])} g2={int(E.PB.g2[0])} "
              f"ph={ph} ct={ct} try={int(E.PB.cert_try[0])} "
              f"瓶Δz={(float(bot[0,2])-zb0)*1000:+.1f}mm 杯Δz={(float(cup[0,2])-zc0)*1000:+.1f}mm "
              f"腕z={float(wr):.4f} 垫R={int((f[0,:5]>0.5).sum())} 垫L={int((f[0,5:]>0.5).sum())}",
              flush=True)
    if bool(E.PB.g2[0]):
        print("[cert] ✅ G2 立", flush=True); break
    if bool(term[0]) or bool(trunc[0]):
        print("[cert] 终止", flush=True); break
print("[cert] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
