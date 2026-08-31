"""左手抓握几何探针: 站位行到底"抓在哪"。

G1 要求左手 ≥3 垫触瓶。母带的左腕 IK 已经准到 0.1cm, 指令手型也确是 prior 的
grasp+squeeze, 可零动作回放到站位行仍然 0 垫接触 —— 那么问题只可能在"手型与
腕位姿的镜像是否自洽": prior 是**右手**抓姿, 腕位姿按 (x,-y,z)/(w,-x,y,-z)
镜像给左臂, 而指值是**按名**映射的。两者若不同源, 手就会合到瓶的另一侧/外侧。

本探针驱动到站位行 (含 βL squeeze), 然后逐垫报告:
  · 垫在**瓶坐标系**里的位置 (瓶轴 = z)
  · 到瓶轴的径向距离 (瓶半径 ~3.3cm; 径向 ≫ 半径 = 够不着, ≪ = 穿模)
  · 垫到瓶面的最近距离与实测接触力
并把 prior 自带的 contact_pos (右手约定) 与其镜像一并打出来做对照。

  UNSCREW_CLIP=32 SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=0 $PY tasks/Unscrew/part4/B_SmokeTest/probe_grasp.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--hold", type=int, default=40, help="站位行保持步数")
p.add_argument("--pin_bottle", action="store_true",
               help="把瓶钉在静置位 (分清'最终抓握位姿穿模' vs '进刀路径扫到')")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_probe_grasp")
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
dsq_l = torch.tensor(BL * (sql - ref[E.IA0, 36:58]), dtype=torch.float32,
                     device=dev)


_pin_pose = torch.cat([E.object.data.root_pos_w.clone(),
                       E.object.data.root_quat_w.clone()], dim=1)
_pin_vel = torch.zeros(1, 6, device=dev)


def drive(row, sL):
    tgt = E.ref58[min(row, E.T_ROW - 1)].clone()
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    E._update_screw_drive_gain()
    for _ in range(DECI):
        if args.pin_bottle:
            E.object.write_root_pose_to_sim(_pin_pose)
            E.object.write_root_velocity_to_sim(_pin_vel)
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())


APP, IA0 = E.APP_END, E.IA0
_org0 = E.scene.env_origins[0].cpu().numpy()


def _watch(tag, r):
    """逐行盯瓶: 倾角一超 15° 就点名 —— 机器段把瓶碰倒是这条链最常见的死法。"""
    _q = E.object.data.root_quat_w[0].cpu().numpy()
    _t = np.degrees(np.arccos(np.clip(quat_to_R(_q)[2, 2], -1, 1)))
    _p = E.object.data.root_pos_w[0].cpu().numpy() - _org0
    _f = E._pads_f().norm(dim=-1)[0]
    if _t > 15.0 and not getattr(_watch, "fired", False):
        _watch.fired = True
        print(f"[watch] ★瓶开始倾倒 @{tag} row={r} 倾角={_t:.1f}° pos={np.round(_p, 3)} "
              f"| 垫L={int((_f[:5] > 0.5).sum())} 垫R={int((_f[5:] > 0.5).sum())} "
              f"| 左指最大力 {float(_f[:5].max()):.2f}N 右指 {float(_f[5:].max()):.2f}N",
              flush=True)
    return _t


for r in range(0, APP):
    drive(r, 0.0)
    if r % 10 == 0 or r >= APP - 5:
        print(f"[watch] app row={r} 瓶倾角={_watch('app', r):.1f}°", flush=True)
for i, r in enumerate(range(APP, IA0)):
    drive(r, float(TC.seam_squeeze_profile((i + 1) / max(IA0 - APP, 1))))
    print(f"[watch] seam1 row={r} 瓶倾角={_watch('seam1', r):.1f}°", flush=True)
for _h in range(args.hold):
    drive(IA0, 1.0)
    if _h % 10 == 0:
        print(f"[watch] hold {_h} 瓶倾角={_watch('hold', IA0):.1f}°", flush=True)

org = E.scene.env_origins[0].cpu().numpy()
bp = E.object.data.root_pos_w[0].cpu().numpy() - org
bq = E.object.data.root_quat_w[0].cpu().numpy()
Rb = quat_to_R(bq)
f = E._pads_f().norm(dim=-1)[0].cpu().numpy()
names = ["thumb", "index", "middle", "ring", "pinky"]
print(f"\n[grasp] 瓶 pos={np.round(bp, 3)} 倾角="
      f"{np.degrees(np.arccos(np.clip(Rb[2, 2], -1, 1))):.1f}°", flush=True)
print(f"[grasp] 左腕 {np.round(E.hand.data.body_pos_w[0, E.wid['L']].cpu().numpy() - org, 3)}"
      f" | 右腕 {np.round(E.hand.data.body_pos_w[0, E.wid['R']].cpu().numpy() - org, 3)}")
# ---- 坐标系/跟踪体检: 命令 vs 实际 vs 离线 IK 的三方对账 ----
import json  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402
_q_cmd = E.ref58[IA0].cpu().numpy()
_q_act = E.hand.data.joint_pos[0, E.map_ids_t].cpu().numpy()
print("[体检] 臂关节 命令 vs 实际 (度):")
for _s, _sl in (("right", slice(0, 7)), ("left", slice(7, 14))):
    _c = np.degrees(_q_cmd[_sl]); _a = np.degrees(_q_act[_sl])
    print(f"   {_s:5s} 命令 {np.round(_c, 1)}")
    print(f"   {_s:5s} 实际 {np.round(_a, 1)}  最大差 {np.abs(_c - _a).max():.2f}°")
_bn = list(E.hand.body_names)
_ac = _bn.index("arm_center")
_ac_p = E.hand.data.body_pos_w[0, _ac].cpu().numpy() - org
_ac_q = E.hand.data.body_quat_w[0, _ac].cpu().numpy()
_rest = json.load(open(TC.REST_JSON))
_aT = np.asarray(_rest["anchor_T_right"], float)
print(f"[体检] arm_center 实测 {np.round(_ac_p, 4)} | env_rest 记录 "
      f"{np.round(_aT[:3, 3], 4)} | 差 {np.linalg.norm(_ac_p - _aT[:3, 3]) * 100:.2f}cm")
print(f"[体检] arm_center 姿态差 "
      f"{np.degrees(np.arccos(np.clip((np.trace(quat_to_R(_ac_q).T @ _aT[:3, :3]) - 1) * 0.5, -1, 1))):.2f}°")
for _s, _sl, _wk in (("right", slice(0, 7), "R"), ("left", slice(7, 14), "L")):
    _ik = ArmIK(_s, anchor_link="arm_center", anchor_T=_aT)
    _fk_act = _ik.fk(_q_act[_sl])[0]
    _fk_cmd = _ik.fk(_q_cmd[_sl])[0]
    _sim = E.hand.data.body_pos_w[0, E.wid[_wk]].cpu().numpy() - org
    print(f"[体检] {_s:5s} FK(实际)={np.round(_fk_act, 3)} 仿真读数={np.round(_sim, 3)} "
          f"差 {np.linalg.norm(_fk_act - _sim) * 100:.2f}cm | FK(命令)={np.round(_fk_cmd, 3)} "
          f"命令-实际 {np.linalg.norm(_fk_cmd - _fk_act) * 100:.2f}cm")

print("[grasp] 左垫 (瓶坐标系; 瓶轴=z):")
for i, nm in enumerate(names):
    wp = E.hand.data.body_pos_w[0, E._pad_bids[i]].cpu().numpy() - org
    lp = Rb.T @ (wp - bp)
    print(f"   {nm:7s} 瓶系 {np.round(lp, 3)} 径向 {np.linalg.norm(lp[:2]) * 100:5.1f}cm "
          f"轴向 {lp[2] * 100:+6.1f}cm | 力 {f[i]:.2f}N")
print("[grasp] 右垫 (盖坐标系):")
cp = E.aux.data.root_pos_w[0].cpu().numpy() - org
cq = E.aux.data.root_quat_w[0].cpu().numpy()
Rc = quat_to_R(cq)
for i, nm in enumerate(names):
    wp = E.hand.data.body_pos_w[0, E._pad_bids[5 + i]].cpu().numpy() - org
    lp = Rc.T @ (wp - cp)
    print(f"   {nm:7s} 盖系 {np.round(lp, 3)} 径向 {np.linalg.norm(lp[:2]) * 100:5.1f}cm "
          f"轴向 {lp[2] * 100:+6.1f}cm | 力 {f[5 + i]:.2f}N")
zp = np.load(TC.PRIOR_AUX)
cpos = np.asarray(zp["contact_pos"], np.float64)
print("[grasp] prior 记录的接触点 (右手约定, 物体系) 与镜像 (给左手的):")
for j in range(len(cpos)):
    m = np.array([cpos[j][0], -cpos[j][1], cpos[j][2]])
    print(f"   #{j} 原 {np.round(cpos[j], 3)} 径向 {np.linalg.norm(cpos[j][:2]) * 100:4.1f}cm"
          f" | 镜像 {np.round(m, 3)} 径向 {np.linalg.norm(m[:2]) * 100:4.1f}cm")
print(f"[grasp] 站位垫 L{int((f[:5] > 0.5).sum())}/5 R{int((f[5:] > 0.5).sum())}/5",
      flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
