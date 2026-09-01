"""右手捏盖**重定向**标定 (2026-08-31 用户裁定: "右手没有做重定向? 必须马上做")。

为什么必须有这一层: 左手有 GraspPose prior (Screw27_body 镜像) 把"人怎么抓"
翻译成这只手的关节角; **右手一直是空的** (βR=0), 手指直接用人手指流 —— 而人手
在拧盖窗内合拢中位只有 +1.0°, 配人手的腕位和指长够用, 搬到 SharpaWave 上
(腕位是按 盖心+reach·螺轴 算的、指长掌宽都不同) 就变成"手张在盖上方": 实测右垫
落在盖系轴向 +1.9~+8.7cm / 径向 4.3~9.3cm, 而盖只有半径 1.75cm、高 1.7cm。
零接触 => 真实螺纹副下扭矩传不进去 => 拧不动 (训练实测 ep_rew/screw 恒 0)。

本探针**量**出该合多少: 驱动到站位行后, 把 triad (拇/食/中) 按屈曲方向逐级合拢,
逐级报告 右垫接触数 / 指力 / 盖被顶动多少 / 瓶倾角, 取"接触够且不顶飞组件"的那档。
输出的度数回填 task_config.PINCH_R_DEG (换 clip / 换手都要复测)。

  UNSCREW_CLIP=32 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=0 $PY tasks/Unscrew/part4/B_SmokeTest/probe_pinch.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--levels", type=str, default="0,5,10,15,20,25,30,35,40",
               help="逐级合拢角度 (度)")
p.add_argument("--hold", type=int, default=25, help="每级保持步数")
p.add_argument("--no_pin", action="store_true",
               help="不钉瓶 (默认钉): 标定捏握要在**瓶立着**的前提下量, 否则缝1 那个"
                    "未解项 (合拢瞬态碰倒瓶, 台账 T2-6e) 会让基线本身就是倒的")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_probe_pinch")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ["POUR_SQUEEZE_FF"] = "1"
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import quat_to_R  # noqa: E402

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
BL = TC.BETA_L
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_l = torch.tensor(np.clip(BL * (sql - ref[E.IA0, 36:58]),
                             -TC.SQUEEZE_DELTA_CAP, TC.SQUEEZE_DELTA_CAP),
                     dtype=torch.float32, device=dev)

# ---- 右手 triad 的屈曲方向 (拇/食/中): 捏盖就是这三指合拢 ----
qr = np.load(os.path.join(TC.TAKE_DIR, "ref_qpos_right.npz"), allow_pickle=True)
FIN = [str(n) for n in qr["joint_names"]]
CURL = np.zeros(22)
for i, n in enumerate(FIN):
    if not n.startswith(("right_thumb", "right_index", "right_middle")):
        continue
    if n.endswith(("MCP_FE", "PIP", "IP")):
        CURL[i] = 1.0            # 主屈曲: 全额
    elif n.endswith("DIP"):
        CURL[i] = 0.5            # 末节: 半额 (指尖太勾会顶不到盖沿)
    elif n == "right_thumb_CMC_AA":
        CURL[i] = 1.0            # 拇指对掌: 没有它就形不成对握
print(f"[pinch] 参与合拢的关节 {int((CURL > 0).sum())} 个: "
      f"{[FIN[i] for i in np.flatnonzero(CURL)]}", flush=True)
CURL_T = torch.tensor(CURL, dtype=torch.float32, device=dev)


_pin_pose = torch.cat([E.object.data.root_pos_w.clone(),
                       E.object.data.root_quat_w.clone()], dim=1)
_pin_vel = torch.zeros(1, 6, device=dev)


def drive(row, sL, pinch_deg):
    tgt = E.ref58[min(row, E.T_ROW - 1)].clone()
    tgt[36:58] += sL * dsq_l                              # 左手 squeeze
    tgt[14:36] += float(np.radians(pinch_deg)) * CURL_T   # 右手 pinch
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    E._update_screw_drive_gain()
    for _ in range(DECI):
        if not args.no_pin:      # 盖由螺纹约束跟着瓶走, 钉瓶即钉住整个组件
            E.object.write_root_pose_to_sim(_pin_pose)
            E.object.write_root_velocity_to_sim(_pin_vel)
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())


APP, IA0 = E.APP_END, E.IA0
for r in range(0, APP):
    drive(r, 0.0, 0.0)
for i, r in enumerate(range(APP, IA0)):
    drive(r, float(TC.seam_squeeze_profile((i + 1) / max(IA0 - APP, 1))), 0.0)
for _ in range(15):
    drive(IA0, 1.0, 0.0)

org = E.scene.env_origins[0].cpu().numpy()
cap0 = E.aux.data.root_pos_w[0].cpu().numpy() - org
bot_q0 = E.object.data.root_quat_w[0].cpu().numpy()
print(f"\n[pinch] 站位基线: 盖 {np.round(cap0, 3)} | 瓶倾角 "
      f"{np.degrees(np.arccos(np.clip(quat_to_R(bot_q0)[2, 2], -1, 1))):.1f}°",
      flush=True)
# ---- 同不同一只手? 指垫(腕系) vs 先验接触点(抓取系) 直接对比 ----
import glob as _glob
_wq = E.hand.data.body_quat_w[0, E.wid["R"]].cpu().numpy()
_wp = E.hand.data.body_pos_w[0, E.wid["R"]].cpu().numpy() - org
_Rw = quat_to_R(_wq)
print("\n[对比] 本手指垫在**腕坐标系**里的位置 (cm):")
for _i, _nm in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
    _pw = E.hand.data.body_pos_w[0, E._pad_bids[5 + _i]].cpu().numpy() - org
    _l = _Rw.T @ (_pw - _wp)
    print(f"   {_nm:7s} {np.round(_l * 100, 1)}  |离腕 {np.linalg.norm(_l) * 100:5.1f}cm")
_cf_used = sorted(_glob.glob(os.path.join(TC.PRIOR_CAP_DIR, "*.npz")))
_zz = np.load(_cf_used[0])
_gp = np.asarray(_zz["grasp"], np.float64)[:3]
_gq = np.asarray(_zz["grasp"], np.float64)[3:7]
_Rg = quat_to_R(_gq / np.linalg.norm(_gq))
_cp = np.asarray(_zz["contact_pos"], np.float64)
_loc = (_Rg.T @ (_cp - _gp).T).T
print(f"[对比] 先验 {os.path.basename(_cf_used[0])} 的接触点在**抓取坐标系**里 (cm):")
print(f"   {len(_loc)} 点, 离腕 {np.linalg.norm(_loc, axis=1).min() * 100:.1f}"
      f"~{np.linalg.norm(_loc, axis=1).max() * 100:.1f}cm, 质心 "
      f"{np.round(_loc.mean(0) * 100, 1)}")
print("  合拢角  右垫  三指  最大指力   盖位移   瓶倾角   螺纹角")
best = None
for deg in [float(x) for x in args.levels.split(",")]:
    for _ in range(args.hold):
        drive(IA0, 1.0, deg)
    f = E._pads_f().norm(dim=-1)[0]
    n_pad = int((f[5:] > 0.5).sum())
    n_tri = int((f[5:][list(PE.SCREW_TRIAD)] > 0.5).sum())
    fmax = float(f[5:].max())
    cap = E.aux.data.root_pos_w[0].cpu().numpy() - org
    d_cap = float(np.linalg.norm(cap - cap0)) * 100
    bq = E.object.data.root_quat_w[0].cpu().numpy()
    tilt = float(np.degrees(np.arccos(np.clip(quat_to_R(bq)[2, 2], -1, 1))))
    scr = float(torch.rad2deg(E.screw_angle[0]).item())
    # 三指指垫在**盖坐标系**里的位置: 差在径向还是轴向, 一眼看到 (盖 r=1.75cm, h=1.7cm)
    Rc = quat_to_R(E.aux.data.root_quat_w[0].cpu().numpy())
    _pp = []
    for _i in list(PE.SCREW_TRIAD):
        _w = E.hand.data.body_pos_w[0, E._pad_bids[5 + _i]].cpu().numpy() - org
        _l = Rc.T @ (_w - cap)
        _pp.append(f"r{np.linalg.norm(_l[:2]) * 100:.1f}/z{_l[2] * 100:+.1f}")
    print(f"  {deg:5.0f}°  {n_pad}/5  {n_tri}/3  {fmax:7.2f}N  "
          f"{d_cap:6.2f}cm  {tilt:6.1f}°  {scr:6.1f}°  | 三指(盖系 cm) "
          + " ".join(_pp), flush=True)
    # 采纳判据: 三指 >=2 触盖, 指力不过分, 组件没被顶跑/顶倒
    _quiet = (d_cap < 2.0 and tilt < 20.0) if args.no_pin else True
    if best is None and n_tri >= 2 and fmax < 25.0 and _quiet:
        best = (deg, n_pad, n_tri, fmax)
if best:
    print(f"\n[pinch] ★建议 PINCH_R_DEG = {best[0]:.0f}° "
          f"(右垫 {best[1]}/5, 三指 {best[2]}/3, 峰值指力 {best[3]:.1f}N)",
          flush=True)
else:
    print("\n[pinch] ❌ 没有一档同时满足 [三指≥2 且 指力<25N 且 组件位移<2cm]"
          " —— 腕位/reach 还要调, 不是合拢量的问题", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
