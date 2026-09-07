"""横瓶段速度诊断：同一抓型/摩擦下比较每个母带行保持 1/2/3/4 个控制步。

目的不是生成正式参考，而是区分“抓型/摩擦不足”与“参考走得太快、PD 跟不上”。
四个环境从同一 P1 出发，分别慢放到交互 k=140；报告首次掉垫/滑移行、腕距
峰值和左臂最大关节跟踪误差。正式改动仍应通过重采样生成连续母带，不能把
zero-order hold 当最终轨迹。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_speed")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ["POUR_SQUEEZE_FF"] = "1"
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402

HOLDS = (1, 2, 3, 4)
N = len(HOLDS)
cfg = PE.build_cfg(num_envs=N)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))

sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_l = torch.tensor(np.clip(
    TC.BETA_L * (sql - ref[E.IA0, 36:58]),
    -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP),
    dtype=torch.float32, device=dev)


def drive(rows, squeeze):
    # rows 可为浮点：真正在线性插值后的关节路径上慢放。只重复整数行会先把
    # 20° 跳变完整发给 PD、再等待若干步，无法检验“消掉跳变”是否有效。
    rows_f = torch.tensor(rows, dtype=torch.float32, device=dev).clamp(
        0, E.T_ROW - 1)
    r0 = torch.floor(rows_f).long()
    r1 = torch.clamp(r0 + 1, max=E.T_ROW - 1)
    u = (rows_f - r0.float()).unsqueeze(1)
    tgt = ((1.0 - u) * E.ref58[r0] + u * E.ref58[r1]).clone()
    tgt[:, 36:58] += squeeze * dsq_l.unsqueeze(0)
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    E._update_screw_drive_gain()
    for _ in range(DECI):
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
    return tgt


APP, IA0 = E.APP_END, E.IA0
for r in range(APP):
    drive([r] * N, 0.0)
for i, r in enumerate(range(APP, IA0)):
    s = float(TC.seam_squeeze_profile((i + 1) / max(IA0 - APP, 1)))
    drive([r] * N, s)
for _ in range(30):
    drive([IA0] * N, 1.0)

d0 = (E.hand.data.body_pos_w[:, E.wid["L"]]
      - E.object.data.root_pos_w).norm(dim=1).clone()
f0 = E._pads_f().norm(dim=-1)
print("[speed] P1: " + " ".join(
    f"hold{h}:L{int((f0[i, :5] > 0.5).sum())}"
    for i, h in enumerate(HOLDS)), flush=True)

K_END = min(140, E.IA1 - E.IA0)
peak_slip = torch.zeros(N, device=dev)
peak_track = torch.zeros(N, device=dev)
lost_k = [None] * N
result = [None] * N
for tick in range(K_END * max(HOLDS) + 1):
    ks = [min(tick / h, K_END) for h in HOLDS]
    rows = [IA0 + k for k in ks]
    tgt = drive(rows, 1.0)
    d = (E.hand.data.body_pos_w[:, E.wid["L"]]
         - E.object.data.root_pos_w).norm(dim=1)
    slip = (d - d0).abs()
    peak_slip = torch.maximum(peak_slip, slip)
    q_act = E.hand.data.joint_pos[:, E.map_ids_t[7:14]]
    trk = torch.rad2deg((q_act - tgt[:, 7:14]).abs().amax(dim=1))
    peak_track = torch.maximum(peak_track, trk)
    f = E._pads_f().norm(dim=-1)
    pads = (f[:, :5] > 0.5).sum(dim=1)
    for i, h in enumerate(HOLDS):
        if lost_k[i] is None and ks[i] > 0 and (
                int(pads[i]) < 2 or float(slip[i]) > 0.03):
            lost_k[i] = round(ks[i], 2)
        if result[i] is None and tick >= K_END * h:
            result[i] = (float(slip[i] * 100), float(peak_slip[i] * 100),
                         int(pads[i]), float(peak_track[i]))
    if tick % 40 == 0:
        print("[speed] tick=%d " % tick + " | ".join(
            f"h{h}:k{ks[i]:.1f} slip={float(slip[i])*100:.1f}cm "
            f"L{int(pads[i])} trk={float(trk[i]):.1f}°"
            for i, h in enumerate(HOLDS)), flush=True)

for i, h in enumerate(HOLDS):
    slip, peak, pads, trk = result[i]
    print(f"[speed] hold={h} (effective stretch={3*h}x): "
          f"lost_k={lost_k[i]} end_slip={slip:.1f}cm peak={peak:.1f}cm "
          f"pads={pads} max_arm_track={trk:.1f}°", flush=True)

try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
os._exit(0)
