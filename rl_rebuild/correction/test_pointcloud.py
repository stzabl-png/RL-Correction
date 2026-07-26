"""点云 -> PointNet 通路自检 (不训练, 只验形状/不变性/数值)."""
import argparse, torch
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args()
# GPU 独占槽位: 同一时刻只允许一个 Isaac 进程占 GPU (见 utils/gpu_guard.py).
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pctest")

app = AppLauncher(args).app

from rl_rebuild.correction import clips
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg
from rl_rebuild.algo.models.models import PointNetEncoder

cfg = SharpaCorrectionEnvCfg(); clips.configure_cfg(cfg, "pp0_anchor"); cfg.scene.num_envs = 8
env = SharpaCorrectionEnv(cfg); obs = env.reset()[0]
ok = True
def chk(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}")

print("\n=== obs_dict 结构 ===")
for k, v in obs.items(): print(f"  {k:14s} {tuple(v.shape)}")
chk("pointcloud 键存在", "pointcloud" in obs)
pc = obs["pointcloud"]
chk("形状 (N,P,3)", tuple(pc.shape) == (8, cfg.n_obj_points, 3), f"{tuple(pc.shape)}")
chk("policy 仍是 144 维 (点云没混进去)", obs["policy"].shape[1] == 144, f"{obs['policy'].shape[1]}")
chk("数值有限", bool(torch.isfinite(pc).all()))
chk("未被 clip_obs 削平", float(pc.abs().max()) < cfg.clip_obs - 1e-3, f"absmax={float(pc.abs().max()):.3f}")

print("\n=== 几何合理性 (腕系点云应落在手周围) ===")
d = pc.norm(dim=-1)
print(f"  点到腕距离: min {d.min()*100:.1f}cm  中位 {d.median()*100:.1f}cm  max {d.max()*100:.1f}cm")
chk("距离量级合理 (<60cm)", float(d.max()) < 0.6)
spread = (pc.max(dim=1).values - pc.min(dim=1).values).mean(0)
print(f"  点云包围盒边长 {[f'{x*100:.1f}cm' for x in spread.tolist()]}  (mesh AABB 7.6/18.1/5.0cm)")

print("\n=== PointNet 前向 ===")
enc = PointNetEncoder(in_dim=cfg.pc_in_dim, feat_dim=128).to(pc.device)
f1 = enc(pc)
chk("输出形状 (N,128)", tuple(f1.shape) == (8, 128), f"{tuple(f1.shape)}")
chk("输出有限", bool(torch.isfinite(f1).all()))
perm = torch.randperm(pc.shape[1], device=pc.device)
f2 = enc(pc[:, perm])
chk("置换不变 (打乱点序输出不变)", float((f1 - f2).abs().max()) < 1e-5,
    f"maxdiff={float((f1-f2).abs().max()):.2e}")
env.close(); app.close()
raise SystemExit(0 if ok else 1)
