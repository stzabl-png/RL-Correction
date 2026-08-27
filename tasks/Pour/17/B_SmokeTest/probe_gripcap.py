"""握力定罪探针: env0=原质量(0.53kg)+力矩饱和检测 env1=半质量 env2=半质量+半速。"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("gripcap")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.setdefault("POUR_SQUEEZE_FF", "1")
import pour_env as PE

N = 3
SPEED = [1, 1, 2]
cfg = PE.build_cfg(num_envs=N)
E = PE.PourEnv(cfg)
E.reset()
dev = E.device
# 半质量: env1/env2 的瓶
mv = E.object.root_physx_view
m = mv.get_masses().clone()
m0 = float(m[0].sum())
m[1] *= 0.5; m[2] *= 0.5
mv.set_masses(m, torch.arange(N))
print(f"[gripcap] 瓶质量: env0={m0:.2f}kg env1/2={m0*0.5:.2f}kg", flush=True)
seam_i = [i for i, e in enumerate(E.entries) if e[4] == "seam1"][0]
E.force_entry = [seam_i] * N
E.reset()
fin_ids = E.map_ids_t[14:36]                        # 右手22指关节
DECI = int(getattr(E.cfg, "decimation", 12))
rows = {i: 190 for i in range(N)}; reps = {i: 0 for i in range(N)}
d0 = None; report = {}; tq_max = 0.0
# 力矩上限: 从执行器配置读
try:
    lim = E.hand.root_physx_view.get_dof_max_forces()[0][[int(j) for j in fin_ids]]
    lim_t = torch.as_tensor(np.asarray(lim), device=dev).abs().clamp(min=1e-6)
except Exception:
    lim_t = None
sat_hist = []
for t in range(500):
    tgt = torch.stack([E._ff_row(torch.tensor([min(rows[i], 260)], device=dev))[0]
                       for i in range(N)])
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
    if lim_t is not None:
        sat = (E.hand.data.applied_torque[0, fin_ids].abs() / lim_t)
        sat_hist.append(float(sat.max()))
    d_r = (E.hand.data.body_pos_w[:, E.wid["R"]] - E.object.data.root_pos_w).norm(dim=1)
    if d0 is None and t == 3:
        d0 = d_r.clone()
    f = E._pads_f().norm(dim=-1)
    for i in range(N):
        reps[i] += 1
        if reps[i] >= SPEED[i]:
            reps[i] = 0; rows[i] += 1
        if rows[i] >= 240 and i not in report:
            slip = float((d_r[i] - d0[i]) * 100)
            np_r = int((f[i, :5] > 0.5).sum())
            tag = ["原质量", "半质量", "半质量半速"][i]
            report[i] = 1
            print(f"[gripcap] {tag}: 滑移={slip:+.1f}cm 右垫={np_r} "
                  f"瓶z={float(E.object.data.root_pos_w[i,2]):.3f} "
                  f"-> {'✅持住' if slip < 5 and np_r >= 2 else '❌脱手'}", flush=True)
    if len(report) == N:
        break
if sat_hist:
    sh = np.array(sat_hist)
    print(f"[gripcap] env0 右指力矩饱和度: 峰值={sh.max():.2f} "
          f"P95={np.percentile(sh,95):.2f} (≥0.95 即封顶)", flush=True)
print("[gripcap] 完毕")
try:
    _slot.release()
except Exception:
    pass
app.close()
