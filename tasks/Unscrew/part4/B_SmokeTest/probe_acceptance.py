"""v2 验收硬闸 (Unscrew 版, 动态行号): βL 零动作照 env.ref58 播全链, 判:
  ① 左腕-瓶滑移全程 <3cm (瓶不脱手)
  ② 螺旋释放发生 (盖被拧开 —— 零动作下靠参考手指行, 过不了=RL 的课, 记档裁定)
  ③ 终态: 瓶距母带末行目标 <5cm/倾差<15°; 盖距末行目标 <8cm
过不了的段要么修参考, 要么 DECISIONS.md 记档"此段留给 RL"并得到裁定。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("v2gate")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402

N = 4
BL = TC.BETA_L
cfg = PE.build_cfg(num_envs=N)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_l = torch.tensor(BL * (sql - ref[E.IA0, 36:58]), dtype=torch.float32,
                     device=dev)
APP = E.APP_END


def drive(row, sL):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt.unsqueeze(0).expand(N, -1)
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
        E._SA.apply_screw(E)


for r in range(0, APP):
    drive(r, 0.0)
for i, r in enumerate(range(APP, E.IA0)):
    drive(r, (i + 1) / max(E.IA0 - APP, 1))
for _ in range(30):
    drive(E.IA0, 1.0)
org = E.scene.env_origins
dL0 = (E.hand.data.body_pos_w[:, E.wid["L"]]
       - E.object.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
print(f"[v2gate] 全链{E.T_ROW}行 IA=[{E.IA0},{E.IA1}] | 站位垫: " + " ".join(
    f"e{i}:L{int((f[i, :5] > 0.5).sum())}/R{int((f[i, 5:] > 0.5).sum())}"
    for i in range(N)), flush=True)
slipL_max = torch.zeros(N, device=dev)
for r in range(E.IA0, E.IA1 + 1):
    drive(r, 1.0)
    dL = (E.hand.data.body_pos_w[:, E.wid["L"]]
          - E.object.data.root_pos_w).norm(dim=1)
    slipL_max = torch.maximum(slipL_max, (dL - dL0).abs())
    if (r - E.IA0) % 20 == 0 or r > E.IA1 - 10:
        rel = (E.screw_has_depth & ~E.screw_engaged)
        print(f"[v2gate] 行{r} screw="
              f"{[round(float(v), 0) for v in torch.rad2deg(E.screw_angle).cpu().tolist()]}° "
              f"rel={rel.int().cpu().tolist()} "
              f"瓶滑={[round(float(v) * 100, 1) for v in (dL - dL0).cpu().tolist()]}cm",
              flush=True)
for r in range(E.IA1 + 1, E.T_ROW):
    drive(r, max(0.0, 1.0 - (r - E.IA1) / max(E.RETREAT0 - E.IA1, 1)))
for _ in range(20):
    drive(E.T_ROW - 1, 0.0)
bot, cap = E._read_objs()
endb, endc = E.PB.end[0], E.PB.end[1]
from progress_batch import _tilt  # noqa: E402
tb = torch.rad2deg(_tilt(bot[:, 3:7], E.PB.up[0]))
tb0 = torch.rad2deg(_tilt(endb[3:7].unsqueeze(0), E.PB.up[0]))[0]
rel = (E.screw_has_depth & ~E.screw_engaged)
npass = 0
for i in range(N):
    db = float((bot[i, :3] - endb[:3]).norm() * 100)
    dc = float((cap[i, :3] - endc[:3]).norm() * 100)
    okb = float(slipL_max[i]) < 0.03 and db < 5 and abs(float(tb[i] - tb0)) < 15
    okc = dc < 8
    okr = bool(rel[i])
    npass += int(okb and okc and okr)
    print(f"[v2gate] e{i}: 瓶滑移峰={float(slipL_max[i]) * 100:.1f}cm "
          f"瓶距末行={db:.1f}cm 倾差={float(tb[i] - tb0):+.0f}° | "
          f"盖距末行={dc:.1f}cm 释放={'✅' if okr else '❌'} "
          f"拧角={float(torch.rad2deg(E.screw_angle[i])):.0f}° "
          f"-> {'✅' if okb and okc and okr else '❌'}", flush=True)
print(f"[v2gate] ★硬闸: {npass}/{N} 通过 (要求≥3; 释放段过不了 → 记档裁定"
      f"'拧开留给 RL', 判 ①③ 即可)", flush=True)
print("[v2gate] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
