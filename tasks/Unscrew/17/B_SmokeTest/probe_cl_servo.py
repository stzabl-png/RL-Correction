"""U9 右手闭环伺服代码路径探针: 强制触发, 看 Jacobian/限位/目标算得出且有限, 且腕朝目标收敛."""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import task_env as PE
N = 2
cfg = PE.build_cfg(num_envs=N); E = PE.UnscrewEnv(cfg)
assert E._right_cl, "需要 UNSCREW_RIGHT_CL=1"
E.force_entry = [2] * N   # g2 出生: 抓稳后, 盖仍咬合
E.reset()
from isaaclab.utils.math import quat_apply
for t in range(40):
    r = E.row.clamp(max=E.T_ROW - 1)
    E._cl_on[:] = True; E._cl_t += 1            # 强制触发 (不管触发条件)
    qff = E._right_cl_servo(r)
    assert qff is not None and torch.isfinite(qff).all(), "servo 输出非有限"
    lo = E.hand.data.joint_pos_limits[:, E.map_ids_t[:7], 0]; hi = E.hand.data.joint_pos_limits[:, E.map_ids_t[:7], 1]
    assert bool(((qff >= lo - 1e-6) & (qff <= hi + 1e-6)).all()), "servo 输出越限"
    full = E.hand.data.joint_pos.clone(); tgt = E.ref58[r].clone(); tgt[:, :7] = qff
    full[:, E.map_ids_t] = tgt; E.hand.set_joint_position_target(full)
    for _ in range(int(E.cfg.decimation)):
        E._SA.apply_screw(E); E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
    org = E.scene.env_origins
    cap_p = E.aux.data.root_pos_w - org; cap_q = E.aux.data.root_quat_w
    u = (E._cl_t.float() / E.CL_LADDER).clamp(0, 1).unsqueeze(1)
    p_local = E._cl_p_pre.unsqueeze(0) * (1 - u) + E._cl_p_grasp.unsqueeze(0) * u
    p_tgt = cap_p + quat_apply(cap_q, p_local.expand(N, 3))
    p_cur = E.hand.data.body_pos_w[:, E.wid["R"]] - org
    err = (p_tgt - p_cur).norm(dim=1)
    if t % 5 == 0 or t == 39:
        print(f"[cl] t{t:2d} 腕→目标 {[round(float(v)*100,2) for v in err]}cm | engaged {E.screw_engaged.tolist()} | 盖 {[round(float(v),3) for v in cap_p[0]]}", flush=True)
print(f"[cl] 末误差 {[round(float(v)*100,2) for v in err]}cm ({'收敛 ✅' if float(err.max()) < 0.03 else '⚠ 未到 3cm 内 (可达/碰撞?)'})", flush=True)
print("[cl] ★代码路径通过", flush=True)
app.close(); os._exit(0)
