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
import task_config as TC
import task_env as PE
cfg = PE.build_cfg(num_envs=8)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * 8   # 全员 t0: 生长握
E.reset()
z = torch.zeros(8, PE.ACT_DIM, device=E.device)
for t in range(430):
    obs, rew, term, trunc, _ = E.step(z)
    if t % 60 == 0:
        f = E._pads_f().norm(dim=-1)
        print(f"[certT0] t{t:3d} row={E.row.cpu().tolist()} "
              f"G1={E.PB.g1.int().cpu().tolist()} G2={E.PB.g2.int().cpu().tolist()} "
              f"try={E.PB.cert_try.cpu().tolist()} "
              f"垫L={[(f[i,:5]>0.5).sum().item() for i in range(8)]}", flush=True)
n1, n2 = int(E.PB.g1.sum()), int(E.PB.g2.sum())
print(f"[certT0] ★零动作t0: G1形成 {n1}/8 | 认证通过 {n2}/8", flush=True)
print("[cert] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
