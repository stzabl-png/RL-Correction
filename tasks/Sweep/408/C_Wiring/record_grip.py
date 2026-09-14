"""Sweep408 Stage-1 抓稳段 策略回放: 载 ckpt, num_envs 个回合并行 (env0 多机位录像), 确定性 mu, **逐步奖惩落盘**。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Sweep/408/C_Wiring/record_grip.py \
      --checkpoint logs/Sweep408_grip_s42/stage1_nn/last.pth --out logs/Sweep408_grip_s42/videos/last_r20 \
      --release_row 20 --headless --enable_cameras

产物 (前缀 --out):
  <out>_<cam>.mp4     env0 回放 (front/side/top 三机位), 1280x720, 20fps (1 帧 = 1 控制行)
  <out>_steps.csv     所有 env 逐步: 行号/钉住/放手/总奖 + 五项分量 (contact/hold/adv/bonus/act)
                      + 掌系漂移(位置/转角, 扫把与簸箕分列) + 认证/时钟 k/死亡种类 + 10 指垫力
  <out>_steps.npz     同上数组形式
  <out>_summary.json  每回合汇总 + 均值
物理规矩 = task_config.PHYS (与训练同源); 与 ckpt 同目录 world.json 核对物理/常量/IO 维, 不符即拒绝回放
(台账 Clean §5.23 的教训: 配方对不上时不报错、只给错数)。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", required=True, help="输出前缀 (不带扩展名)")
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--release_row", type=int, default=None, help="课程档位 (默认 = world.json 的 release_row_start)")
p.add_argument("--jitter", type=int, default=0)
p.add_argument("--no_video", action="store_true")
p.add_argument("--focal", type=float, default=24.0)
p.add_argument("--cams", default="front:-0.85,-0.45,0.60;side:0.75,-0.95,0.50;top:-0.30,-0.60,0.90",
               help="name:dx,dy,dz;... (相对两物体中点的偏移 m)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():                 # 物理规矩必须在 import env 之前
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep408_rec")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import grip_env as GE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = GE.SIDES                              # ("right","left") = (扫把, 簸箕)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
DIE_NAME = {0: "-", 1: "rel", 2: "drop", 3: "table"}

# ---- 世界核对 ----
ckpt = os.path.abspath(args.checkpoint)
wj = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "world.json")
world = json.load(open(wj)) if os.path.isfile(wj) else None
if world is None:
    print(f"[world] ⚠ 未找到 {wj}, 跳过核对", flush=True)
else:
    bad = []
    for k, v in world.get("physics", {}).items():
        if os.environ.get(k) != v:
            bad.append(f"{k}: 记录={v} 当前={os.environ.get(k)}")
    for k, v in world.get("stage1", {}).items():
        cur = getattr(TC, k, None)
        same = (abs(float(cur) - float(v)) < 1e-9) if isinstance(v, (int, float)) and cur is not None else (cur == v)
        if not same:
            bad.append(f"{k}: 记录={v} 当前={cur}")
    io_ = world.get("policy_io", {})
    if (io_.get("obs_dim"), io_.get("priv_dim"), io_.get("act_dim")) != (GE.OBS_DIM, GE.PRIV_DIM, GE.ACT_DIM):
        bad.append(f"policy_io: 记录={io_} 当前=({GE.OBS_DIM},{GE.PRIV_DIM},{GE.ACT_DIM})")
    # 母带比 **sha256** 不比路径 —— 路径跨机必然不同 (msc /home/yanghong vs 本机 /home/lyh),
    # 比路径会在跨机回放时假失败 (2026-09-11 踩过)。比 sha 既跨机可用, 也比比路径更强。
    _rs = world.get("reference", {}).get("sha256")
    if _rs:
        import hashlib
        _h = hashlib.sha256(open(TC.REFERENCE, "rb").read()).hexdigest()
        if _h != _rs:
            bad.append(f"母带 sha256: 记录={_rs[:16]}… 当前={_h[:16]}… ({os.path.basename(TC.REFERENCE)})")
    if bad and not os.environ.get("SWEEP408_IGNORE_WORLD"):
        print("[world] ✗ 与 ckpt 出生世界不符 (SWEEP408_IGNORE_WORLD=1 可跳过):\n  " + "\n  ".join(bad), flush=True)
        os._exit(2)
    print(f"[world] ✅ 匹配: 物理={world.get('physics')} arm={world.get('arm')} "
          f"母带={os.path.basename(TC.REFERENCE)} (sha 一致) "
          f"release_row_start={world.get('release_row_start')}", flush=True)

# ---- 环境 + 策略 ----
raw = GE.Sweep408GripEnv(GE.build_cfg(args.num_envs))
_rr = args.release_row if args.release_row is not None else (world or {}).get("release_row_start", TC.RELEASE_MAX)
raw.release_row_cur = int(_rr)
TC.RELEASE_JITTER = int(args.jitter)          # _reset_idx 每次从模块读
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_grip.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "sweep408_rec"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(ckpt)
agent.set_eval()
print(f"[record_grip] ckpt={ckpt} N={args.num_envs} release={raw.release_row_cur} jitter={TC.RELEASE_JITTER} "
      f"T_EP={raw.T_EP} 母带 {raw.T_REF} 行", flush=True)

# ---- 相机 (env0, 对准两物体中点) ----
_annots = {}
if not args.no_video:
    import omni.replicator.core as rep  # noqa: E402
    import omni.usd  # noqa: E402
    from pxr import Gf, UsdGeom  # noqa: E402
    org = raw._org()[0].detach().cpu().numpy()
    ctr = 0.5 * (raw.nominal["right"][0] + raw.nominal["left"][0]).detach().cpu().numpy() + org
    _st = omni.usd.get_context().get_stage()
    for spec in [c for c in args.cams.split(";") if c.strip()]:
        name, off = spec.split(":")
        eye = ctr + np.array([float(x) for x in off.split(",")])
        _cam = UsdGeom.Camera.Define(_st, f"/World/RecCam_{name}")
        _cam.CreateFocalLengthAttr().Set(float(args.focal))
        _m = Gf.Matrix4d(); _m.SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, ctr)), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(_cam).AddTransformOp().Set(_m.GetInverse())
        _rp = rep.create.render_product(f"/World/RecCam_{name}", (1280, 720))
        _a = rep.AnnotatorRegistry.get_annotator("rgb"); _a.attach(_rp); _annots[name] = _a
        print(f"[record_grip] 相机 {name}: eye={eye.round(3).tolist()} → {ctr.round(3).tolist()}", flush=True)
    for _ in range(12):                       # 预热 (首帧灰黑)
        raw.sim.render()
        for _a in _annots.values():
            _a.get_data()

# ---- 回放 ----
N = args.num_envs
T = int(raw.T_EP) + 2
cols = (["env", "step", "row", "release_row", "pinned", "released", "latched", "reward",
         "r_adv", "r_soft", "r_table", "r_contact", "r_bonus", "r_act", "c_broom", "c_pan",
         "dp_broom_cm", "dr_broom_deg", "dp_pan_cm", "dr_pan_deg", "within", "broom_ok", "pan_ok",
         "cert", "cert_run", "new_cert", "cert_timeout", "k", "clock_frac", "clock_adv",
         "died", "die_kind", "new_die", "success", "done",
         "act_abs_mean", "res_arm_max_rad", "res_fin_max_rad", "gap_broom_mm", "gap_pan_mm"]
        + [f"F_R_{f}" for f in FINGERS] + [f"F_L_{f}" for f in FINGERS])
rows = []
frames = {k: [] for k in _annots}
done_at = np.full(N, -1, dtype=int)
obs = env.reset()
f = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
with torch.no_grad():
    for t in range(T):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        tk = raw._tick
        d_now = dones.reshape(-1).bool().cpu().numpy() if torch.is_tensor(dones) else np.asarray(dones, bool).reshape(-1)
        cum = raw.cum_res
        rec = {
            "row": f(tk["row"]), "release_row": f(tk["release_row"]), "released": f(tk["released"]),
            "latched": f(tk["latched"]), "reward": f(tk["reward"]),
            "r_adv": f(tk["r_adv"]), "r_soft": f(tk["r_soft"]), "r_table": f(tk["r_table"]),
            "r_contact": f(tk["r_contact"]), "r_bonus": f(tk["r_bonus"]), "r_act": f(tk["r_act"]),
            "c_broom": f(tk["c_broom"]), "c_pan": f(tk["c_pan"]),
            "dp_broom_cm": f(tk["dp"]["right"]) * 100, "dr_broom_deg": np.degrees(f(tk["dr"]["right"])),
            "dp_pan_cm": f(tk["dp"]["left"]) * 100, "dr_pan_deg": np.degrees(f(tk["dr"]["left"])),
            "within": f(tk["within"]), "broom_ok": f(tk["broom_ok"]), "pan_ok": f(tk["pan_ok"]),
            "cert": f(tk["cert"]), "cert_run": f(tk["cert_run"]), "new_cert": f(tk["new_cert"]),
            "cert_timeout": f(tk["cert_timeout"]), "k": f(tk["k"]),
            "clock_frac": f(tk["k"]) / max(raw.T_REF - 1, 1), "clock_adv": f(tk["can"]),
            "died": f(tk["died"]), "die_kind": f(tk["die_kind"]), "new_die": f(tk["new_die"]),
            "success": f(tk["new_succ"]) + f(tk["clock_done"]) * 0, "done": d_now.astype(float),
            "act_abs_mean": f(act.abs().mean(1)),
            "res_arm_max_rad": f(cum[:, :14].abs().amax(1)), "res_fin_max_rad": f(cum[:, 14:].abs().amax(1)),
        }
        _g = tk.get("gaps") or {}
        rec["gap_broom_mm"] = f(_g["right"]) * 1000 if "right" in _g else np.full(N, np.nan)
        rec["gap_pan_mm"] = f(_g["left"]) * 1000 if "left" in _g else np.full(N, np.nan)
        FR, FL = f(tk["F"]["right"]), f(tk["F"]["left"])
        for i in range(N):
            if done_at[i] >= 0:
                continue                      # 该 env 已结束 (IsaacLab 已自动重置), 不记第二回合
            r = [i, t, int(rec["row"][i]), int(rec["release_row"][i]),
                 float(rec["row"][i] <= rec["release_row"][i])]
            r += [rec[c][i] for c in cols[5:40]]
            r += list(FR[i]) + list(FL[i])
            rows.append(r)
        if _annots and done_at[0] < 0:
            raw.sim.render()
            for k_, _a in _annots.items():
                d = _a.get_data()
                if d is not None and getattr(d, "size", 0):
                    frames[k_].append(np.asarray(d)[..., :3].astype(np.uint8))
        done_at = np.where(d_now & (done_at < 0), t, done_at)
        if (done_at >= 0).all():
            break

# ---- 落盘 ----
out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
arr = np.asarray(rows, dtype=np.float64)
with open(out + "_steps.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(cols)
    for r in rows:
        w.writerow([f"{x:.5g}" if isinstance(x, float) else x for x in r])
np.savez_compressed(out + "_steps.npz", cols=np.array(cols), data=arr, done_at=done_at,
                    checkpoint=ckpt, release_row_cur=raw.release_row_cur, t_ref=raw.T_REF)
ci = {c: j for j, c in enumerate(cols)}
eps = []
for i in range(N):
    m = arr[:, ci["env"]] == i
    if not m.any():
        continue
    e = arr[m]
    eps.append({
        "env": i, "len": int(m.sum()),
        "reward_sum": float(e[:, ci["reward"]].sum()),
        "cert": bool(e[:, ci["cert"]].max() > 0),
        "clock_frac_max": float(e[:, ci["clock_frac"]].max()),
        "success": bool(e[:, ci["success"]].max() > 0),
        "die_kind": DIE_NAME.get(int(e[:, ci["die_kind"]].max()), "?"),
        "dp_broom_max_cm": float(e[:, ci["dp_broom_cm"]].max()),
        "dr_broom_max_deg": float(e[:, ci["dr_broom_deg"]].max()),
        "dp_pan_max_cm": float(e[:, ci["dp_pan_cm"]].max()),
        "dr_pan_max_deg": float(e[:, ci["dr_pan_deg"]].max()),
        "r_adv": float(e[:, ci["r_adv"]].sum()), "r_soft": float(e[:, ci["r_soft"]].sum()),
        "r_table": float(e[:, ci["r_table"]].sum()), "r_contact": float(e[:, ci["r_contact"]].sum()),
        "r_bonus": float(e[:, ci["r_bonus"]].sum()),
    })
summ = {"checkpoint": ckpt, "release_row_cur": raw.release_row_cur, "jitter": TC.RELEASE_JITTER,
        "tape_rows": int(raw.T_REF), "episodes": eps,
        "mean": {k: float(np.mean([ep[k] for ep in eps])) for k in
                 ("reward_sum", "clock_frac_max", "dp_broom_max_cm", "dr_broom_max_deg",
                  "dp_pan_max_cm", "dr_pan_max_deg") if eps},
        "cert_rate": float(np.mean([ep["cert"] for ep in eps])) if eps else float("nan"),
        "success_rate": float(np.mean([ep["success"] for ep in eps])) if eps else float("nan")}
with open(out + "_summary.json", "w") as fj:
    json.dump(summ, fj, indent=1, ensure_ascii=False)
n_frames = 0
if _annots:
    import imageio
    for k_, fr in frames.items():
        if fr:
            imageio.mimsave(out + f"_{k_}.mp4", fr, fps=int(TC.CONTROL_HZ)); n_frames = len(fr)
print(f"[record_grip] → {out}_steps.csv/.npz  {out}_summary.json" + (f"  {len(_annots)}×mp4 ({n_frames} 帧)" if _annots else ""), flush=True)
print(f"[record_grip] 认证率={summ['cert_rate']:.3f} 成功率={summ['success_rate']:.3f} "
      f"时钟={summ['mean'].get('clock_frac_max', float('nan')):.3f} "
      f"扫把漂移 max {summ['mean'].get('dp_broom_max_cm', 0):.2f}cm/{summ['mean'].get('dr_broom_max_deg', 0):.1f}° "
      f"簸箕 {summ['mean'].get('dp_pan_max_cm', 0):.2f}cm/{summ['mean'].get('dr_pan_max_deg', 0):.1f}°", flush=True)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
