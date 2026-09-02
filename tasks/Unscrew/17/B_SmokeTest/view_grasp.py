"""Unscrew/17 GraspPose 目检 (§10 拍板①, 照 Pour L1-3 口径):
右臂+右手 = 初始站姿不动; 左臂+左手 = 直接摆到 LD227 站位 GraspPose (母带 IA0 行)。
瓶默认钉在静置位 (看抓法几何; --free 放开看物理), 每 2s 打印五垫力/瓶系径向/手最低点。
用法: bash tasks/Unscrew/17/B_SmokeTest/view_grasp.sh   (加 --squeeze 1.0 看加剂量后的合拢)
"""
import argparse, os, sys, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--squeeze", type=float, default=0.0, help="βL squeeze 剂量 (0=纯 GraspPose)")
p.add_argument("--free", action="store_true", help="瓶不钉 (看真实物理推挤)")
p.add_argument("--selftest", type=int, default=0, help=">0: 无头跑 N 步打读数退出")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "part4", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
import task_config as TC
import task_env as PE

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg); E.force_entry = [0]; E.reset()
dev = E.device; DECI = int(getattr(E.cfg, "decimation", 12))
org = E.scene.env_origins[0]
# ★钉位姿必须在任何自由物理步之前抓 (2026-09-02 修: 静置期没走螺纹装配, 盖自由落体掉进瓶口)
_pinb = torch.cat([E.object.data.root_pos_w, E.object.data.root_quat_w], dim=1).clone()
_pinc = torch.cat([E.aux.data.root_pos_w, E.aux.data.root_quat_w], dim=1).clone()
_zero6 = torch.zeros(1, 6, device=dev)
for _ in range(30):
    if not args.free:
        E.object.write_root_pose_to_sim(_pinb); E.object.write_root_velocity_to_sim(_zero6)
        E.aux.write_root_pose_to_sim(_pinc); E.aux.write_root_velocity_to_sim(_zero6)
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())

# 目标位形: 右 = 站姿行0; 左臂/左指 = 站位行 IA0 (= GraspPose 落位)
tgt = E.ref58[0].clone()
tgt[7:14] = E.ref58[E.IA0][7:14]
tgt[36:58] = E.ref58[E.IA0][36:58]
if args.squeeze > 0:
    sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
    dsq = np.clip(args.squeeze * (sql - E.ref58[E.IA0, 36:58].cpu().numpy()),
                  -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP)
    tgt[36:58] += torch.tensor(dsq, dtype=torch.float32, device=dev)
full = E.hand.data.default_joint_pos.clone()
full[0, E.map_ids_t] = tgt
E.hand.write_joint_state_to_sim(full, torch.zeros_like(full))   # 直接摆位 (无扫掠)
E.hand.set_joint_position_target(full)
print(f"[grasp目检] 左先验 = {os.path.basename(TC.PRIOR_AUX)} | 右手 = 初始站姿 | "
      f"瓶 {'自由' if args.free else '钉住'} | squeeze βL={args.squeeze}", flush=True)

_PADS = ("thumb", "index", "middle", "ring", "pinky")

def readout():
    f = E._pads_f().norm(dim=-1)[0]
    bp = E.object.data.root_pos_w[0] - org
    bq = E.object.data.root_quat_w[0]
    from isaaclab.utils.math import quat_apply
    up = quat_apply(bq.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
    tilt = float(torch.rad2deg(torch.acos(up[2].clamp(-1, 1))))
    lb = [i for i in E.hand_bids if E.hand.body_names[i].startswith("left")]
    hz = float(E.hand.data.body_pos_w[0, lb, 2].min() - org[2]) - 0.87
    pads = []
    for i, nm in enumerate(_PADS):
        pw = E.hand.data.body_pos_w[0, E._pad_bids[i]] - org
        r = float(((pw[:2] - bp[:2])).norm())
        pads.append(f"{nm} r{r*100:4.1f}cm F{float(f[i]):5.2f}N")
    print(f"[grasp目检] 左垫 {int((f[:5] > 0.5).sum())}/5 | " + " | ".join(pads)
          + f" | 瓶倾 {tilt:4.1f}° | 左手最低-桌 {hz*100:+.1f}cm", flush=True)

t0 = time.time(); n = 0
STEPS = args.selftest if args.selftest else 10**9
while n < STEPS:
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        if not args.free:
            E.object.write_root_pose_to_sim(_pinb); E.object.write_root_velocity_to_sim(_zero6)
            E.aux.write_root_pose_to_sim(_pinc); E.aux.write_root_velocity_to_sim(_zero6)
        E.scene.write_data_to_sim()
        E.sim.step(render=not args.headless)
        E.scene.update(E.sim.get_physics_dt())
    n += 1
    if time.time() - t0 > 2.0:
        readout(); t0 = time.time()
    if not args.selftest:
        time.sleep(0.02)
readout()
print("[grasp目检] done", flush=True)
app.close(); os._exit(0)
