"""慢提起物理能力试验: seam1 出生, 零动作 squeeze, env0=全速 env1=半速 env2=1/3速。
判据: 提升段(k~16-30 对应行)结束时瓶是否仍被持住 (d4_r<5cm 且垫>=2)。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("slowlift")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.setdefault("POUR_SQUEEZE_FF", "1")
import pour_env as PE

N = 3
SPEED = [1, 2, 3]     # 每行重复次数: 1=全速 2=半速 3=1/3速
cfg = PE.build_cfg(num_envs=N)
E = PE.PourEnv(cfg)
E.reset()
dev = E.device
seam_i = [i for i, e in enumerate(E.entries) if e[4] == "seam1"][0]
E.force_entry = [seam_i] * N
E.reset()
# 手动驱动: 绕过 env 时钟, 每 env 按各自速率播 190..260 行 (含提升段)
DECI = int(getattr(E.cfg, "decimation", 12))
rows = {i: 190 for i in range(N)}
reps = {i: 0 for i in range(N)}
d0 = None
report = {}
for t in range(220*3):
    tgt = torch.stack([E._ff_row(torch.tensor([min(rows[i], 260)], device=dev))[0]
                       for i in range(N)])
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
    d_r = (E.hand.data.body_pos_w[:, E.wid["R"]]
           - E.object.data.root_pos_w).norm(dim=1)
    if d0 is None and t == 3:
        d0 = d_r.clone()
    f = E._pads_f().norm(dim=-1)
    for i in range(N):
        reps[i] += 1
        if reps[i] >= SPEED[i]:
            reps[i] = 0
            rows[i] += 1
        if rows[i] >= 240 and i not in report:      # 提升段(216=k26)早已过
            slip = float((d_r[i] - d0[i]) * 100)
            np_r = int((f[i, :5] > 0.5).sum())
            bz = float(E.object.data.root_pos_w[i, 2])
            report[i] = (slip, np_r, bz)
            print(f"[slowlift] {'全速' if SPEED[i]==1 else f'1/{SPEED[i]}速'} "
                  f"到行240: 滑移={slip:+.1f}cm 右垫={np_r} 瓶z={bz:.3f} "
                  f"-> {'✅持住' if slip < 5 and np_r >= 2 else '❌脱手'}", flush=True)
    if len(report) == N:
        break
print("[slowlift] 完毕")
try:
    _slot.release()
except Exception:
    pass
app.close()
