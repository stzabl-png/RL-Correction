"""v2 训练可用性验收 (Unscrew 版, 动态行号)。

βL 零动作照 env.ref58 播全链，记录接触、滑移、释放与终态误差作为 correction
基线；这些量不要求 reference 自己成功。硬闸只检查状态有限、物理不发散，并把
世界指纹与 v2 MD5 原子写入 acceptance_v2.json，供远端训练复核。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
import json
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("v2gate")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ["POUR_SQUEEZE_FF"] = "1"
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402

N = 4
BL = TC.BETA_L
cfg = PE.build_cfg(num_envs=N)
import world_fingerprint as WF  # noqa: E402

assert os.path.abspath(PE.MASTER) == os.path.abspath(TC.REF_V2), (
    f"验收必须使用 reference_v2.npz，当前是 {PE.MASTER}")

E = PE.UnscrewEnv(cfg)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_l = torch.tensor(np.clip(BL * (sql - ref[E.IA0, 36:58]), -TC.SQUEEZE_DELTA_CAP,
                          TC.SQUEEZE_DELTA_CAP), dtype=torch.float32,
                     device=dev)
APP = E.APP_END


org = E.scene.env_origins
stable = torch.ones(N, dtype=torch.bool, device=dev)
max_pad_force = torch.zeros(N, device=dev)
max_obj_radius = torch.zeros(N, device=dev)
max_joint_abs = torch.zeros(N, device=dev)
max_screw_deg = torch.zeros(N, device=dev)
max_screw_tau_mNm = torch.zeros(N, device=dev)


def audit_state():
    """Only reject simulator corruption; contact quality is an RL objective."""
    global stable, max_pad_force, max_obj_radius, max_joint_abs
    global max_screw_deg, max_screw_tau_mNm
    q = E.hand.data.joint_pos
    pos = torch.cat([
        E.object.data.root_pos_w - org,
        E.aux.data.root_pos_w - org,
    ], dim=1)
    vel = torch.cat([
        E.object.data.root_lin_vel_w, E.object.data.root_ang_vel_w,
        E.aux.data.root_lin_vel_w, E.aux.data.root_ang_vel_w,
    ], dim=1)
    pf = E._pads_f().norm(dim=-1).amax(dim=1)
    finite = torch.isfinite(q).all(dim=1) & torch.isfinite(pos).all(dim=1) \
        & torch.isfinite(vel).all(dim=1) & torch.isfinite(pf)
    bounded = (pos.abs().amax(dim=1) < 5.0) & (q.abs().amax(dim=1) < 20.0) \
        & (vel.abs().amax(dim=1) < 1.0e4) & (pf < 1.0e5)
    stable &= finite & bounded
    max_pad_force = torch.maximum(max_pad_force, pf.nan_to_num(posinf=1.0e9))
    safe_pos = pos.nan_to_num(nan=1.0e9, posinf=1.0e9, neginf=-1.0e9)
    safe_q = q.nan_to_num(nan=1.0e9, posinf=1.0e9, neginf=-1.0e9)
    max_obj_radius = torch.maximum(
        max_obj_radius, safe_pos.view(N, 2, 3).norm(dim=2).amax(dim=1))
    max_joint_abs = torch.maximum(max_joint_abs, safe_q.abs().amax(dim=1))
    max_screw_deg = torch.maximum(
        max_screw_deg, torch.rad2deg(E.screw_angle).abs())
    if getattr(E, "_real_thread", False):
        max_screw_tau_mNm = torch.maximum(
            max_screw_tau_mNm, 1000.0 * E.screw_tau_ema.abs())


def drive(row, sL):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[:, E.map_ids_t] = tgt.unsqueeze(0).expand(N, -1)
    E.hand.set_joint_position_target(full)
    E._update_screw_drive_gain()
    for _ in range(DECI):
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
    audit_state()


for r in range(0, APP):
    drive(r, 0.0)
for i, r in enumerate(range(APP, E.IA0)):
    drive(r, float(TC.seam_squeeze_profile((i + 1) / max(E.IA0 - APP, 1))))
for _ in range(30):
    drive(E.IA0, 1.0)
dL0 = (E.hand.data.body_pos_w[:, E.wid["L"]]
       - E.object.data.root_pos_w).norm(dim=1).clone()
f = E._pads_f().norm(dim=-1)
station_pads_l = (f[:, :5] > 0.5).sum(dim=1).int().cpu().tolist()
station_pads_r = (f[:, 5:] > 0.5).sum(dim=1).int().cpu().tolist()
print(f"[v2gate] 全链{E.T_ROW}行 IA=[{E.IA0},{E.IA1}] | 站位垫: " + " ".join(
    f"e{i}:L{int((f[i, :5] > 0.5).sum())}/R{int((f[i, 5:] > 0.5).sum())}"
    for i in range(N)), flush=True)
slipL_max = torch.zeros(N, device=dev)
right_pads_peak = torch.zeros(N, dtype=torch.int32, device=dev)
with np.load(PE.MASTER, allow_pickle=True) as _timing_z:
    _right_start_k = int(np.asarray(_timing_z["right_start_k"]).item())
    _right_grasp_k = int(np.asarray(_timing_z["right_grasp_k"]).item())
for r in range(E.IA0, E.IA1 + 1):
    drive(r, 1.0)
    dL = (E.hand.data.body_pos_w[:, E.wid["L"]]
          - E.object.data.root_pos_w).norm(dim=1)
    slipL_max = torch.maximum(slipL_max, (dL - dL0).abs())
    # P1 右手本来就应远离瓶盖；只在原始数据的明显启动点之后统计接触。
    # “是否到盖”看 right_grasp_k 之后的峰值，而不是错误地检查站位行。
    if r - E.IA0 >= _right_start_k:
        _fr = E._pads_f().norm(dim=-1)
        right_pads_peak = torch.maximum(
            right_pads_peak, (_fr[:, 5:] > 0.5).sum(dim=1).to(torch.int32))
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
baseline_envs = []
reference_successes = 0
for i in range(N):
    db = float((bot[i, :3] - endb[:3]).norm() * 100)
    dc = float((cap[i, :3] - endc[:3]).norm() * 100)
    tilt_delta = float(tb[i] - tb0)
    screw_deg = float(torch.rad2deg(E.screw_angle[i]))
    okb = float(slipL_max[i]) < 0.03 and db < 5 and abs(float(tb[i] - tb0)) < 15
    okc = dc < 8
    okr = bool(rel[i])
    reference_successes += int(okb and okc and okr)
    success = bool(okb and okc and okr)
    baseline_envs.append({
        "env": i,
        "station_pads_l": station_pads_l[i],
        "station_pads_r": station_pads_r[i],
        "right_pads_peak_after_start": int(right_pads_peak[i]),
        "left_slip_peak_cm": round(float(slipL_max[i]) * 100, 4),
        "bottle_end_error_cm": round(db, 4),
        "bottle_tilt_error_deg": round(tilt_delta, 4),
        "cap_end_error_cm": round(dc, 4),
        "released": okr,
        "screw_deg": round(screw_deg, 4),
        "reference_success": success,
    })
    print(f"[v2gate] e{i}: 瓶滑移峰={float(slipL_max[i]) * 100:.1f}cm "
          f"瓶距末行={db:.1f}cm 倾差={tilt_delta:+.0f}° | "
          f"盖距末行={dc:.1f}cm 释放={'✅' if okr else '❌'} "
          f"拧角={screw_deg:.0f}° "
          f"-> {'✅' if okb and okc and okr else '❌'}", flush=True)
stable_envs = int(stable.sum().item())
ok = stable_envs >= 3
# ---- 可训练性台账 (T2-4 裁定下"绿灯"必须仍然有信息量) ----
# 只查"有限/不发散"的闸对**任何**母带都会亮绿 —— 包括瓶已经躺在地上的那种。
# reference 不必自己成功 (那是 RL correction 的活), 但下面这几条是"策略有没有
# 可能学会"的先决条件, 一律入账并显式点名; 不阻塞, 但绝不假装没看见。
_meta_v2 = {}
try:
    with np.load(PE.MASTER, allow_pickle=True) as _z:
        if "meta_v2" in _z.files:
            _meta_v2 = dict(kv.split("=", 1) for kv in str(_z["meta_v2"]).split(";")
                            if "=" in kv)
except Exception as _e:
    _meta_v2 = {"error": f"{type(_e).__name__}: {_e}"}
warn = []
if min(station_pads_l) < 3:
    warn.append(f"站位行左垫最少 {min(station_pads_l)}/5 < G1 所需 3 —— "
                f"左手没抓上瓶, G1 无从成形")
if int(right_pads_peak.min().item()) < 1:
    warn.append(f"右手从启动点 k={_right_start_k} 到任务末的接触峰值最少 "
                f"{int(right_pads_peak.min().item())}/5 —— 至少一个环境未碰到盖, "
                f"真实螺纹副下扭矩传不进去 (目标到盖 k={_right_grasp_k})")
_screw_max = float(max_screw_deg.max())
if _screw_max < 1.0:
    warn.append(f"零动作回放全程拧角 {_screw_max:.1f}° —— 参考本身一点没拧动")
_tau_max = (float(max_screw_tau_mNm.max())
            if getattr(E, "_real_thread", False) else float("nan"))
if (getattr(E, "_real_thread", False)
        and int(right_pads_peak.max().item()) < 1 and _tau_max > 5):
    warn.append(f"零接触却有 {_tau_max:.0f} mN·m 传入力矩 = 幻影扭矩通道 "
                f"(老台账 U40b/c/d 同款, 必须先修再训)")
for _k in ("frozen_r", "frozen_l"):
    if float(_meta_v2.get(_k, 0) or 0) > 40:
        warn.append(f"母带 {_k}={_meta_v2[_k]} 行是 IK 冻结行 (臂跟不上腕参考), "
                    f"残差界内难以救回")
baseline = {
    "reference_successes": reference_successes,
    "reference_success_required": False,
    "beta_l": BL,
    "envs": baseline_envs,
    "station_pads_l_min": int(min(station_pads_l)),
    "station_pads_r_min": int(min(station_pads_r)),
    "right_pads_peak_after_start": right_pads_peak.cpu().tolist(),
    "right_start_k": _right_start_k,
    "right_grasp_k": _right_grasp_k,
    "screw_deg_max": round(_screw_max, 3),
    "screw_tau_max_mNm": round(_tau_max, 3) if _tau_max == _tau_max else None,
    "reference_ik": _meta_v2,
    "warnings": warn,
}
if ok:
    receipt = {
        "schema": "unscrew_trainability_v1",
        "clip": TC.CLIP_ID,
        "reference_v2": os.path.abspath(PE.MASTER),
        "reference_v2_md5": TC.file_md5(PE.MASTER),
        "stable_envs": stable_envs,
        "per_env_stable": stable.int().cpu().tolist(),
        "max_pad_force_N": [round(float(v), 4) for v in max_pad_force.cpu()],
        "max_object_radius_m": [round(float(v), 4) for v in max_obj_radius.cpu()],
        "max_joint_abs_rad": [round(float(v), 4) for v in max_joint_abs.cpu()],
        "baseline": baseline,
        "num_envs": N,
        "world": WF.collect(E),
    }
    os.makedirs(os.path.dirname(TC.ACCEPTANCE_JSON), exist_ok=True)
    tmp_receipt = f"{TC.ACCEPTANCE_JSON}.tmp.{os.getpid()}"
    with open(tmp_receipt, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1, ensure_ascii=False)
    os.replace(tmp_receipt, TC.ACCEPTANCE_JSON)
    print(f"[v2gate] 验收凭据 -> {TC.ACCEPTANCE_JSON}", flush=True)

print(f"[v2gate] reference 零动作成功 {reference_successes}/{N}（仅诊断，不阻塞）",
      flush=True)
print(f"[v2gate] 站位垫 L{min(station_pads_l)}~{max(station_pads_l)}/5 "
      f"R{min(station_pads_r)}~{max(station_pads_r)}/5 | "
      f"右手启动后接触峰={right_pads_peak.cpu().tolist()} | 回放拧角峰 "
      f"{_screw_max:.1f}°"
      + (f" | 传入力矩峰 {_tau_max:.0f} mN·m" if _tau_max == _tau_max else ""),
      flush=True)
if warn:
    print("[v2gate] ⚠ 可训练性告警 (不阻塞, 但这些是策略能不能学会的先决条件):",
          flush=True)
    for _w in warn:
        print(f"[v2gate]   - {_w}", flush=True)
else:
    print("[v2gate] 可训练性告警: 无", flush=True)
print(f"[v2gate] ★训练稳定性: {stable_envs}/{N}（要求≥3） => {'✅' if ok else '❌'}", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
sys.stdout.flush()
os._exit(0 if ok else 1)
