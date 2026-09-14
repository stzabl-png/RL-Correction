"""Clean/3 Stage-1 策略回放: 载 ckpt, num_envs 个回合并行 (env0 录像), 确定性 mu, **逐步奖惩落盘**。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/record_clean.py \\
      --checkpoint logs/Clean3_hold_s42/stage1_nn/last.pth --out logs/Clean3_hold_s42/videos/last_r35 \\
      --release_row 35 --headless --enable_cameras
产物 (前缀 --out):
  <out>_<cam>.mp4     env0 回放 (front/side/top 三机位, --cams 可改), 1280x720, 20 fps (1 帧 = 1 控制行)
  <out>_steps.csv     所有 env 逐步: 行号/钉住/总奖 + 五项分量 (contact/hold/bonus/act/cross) + 漂移/认证/掉落/成功
                      + 10 指垫力 + 相邻指尖侧向间距/3D 距离 (交叉判据原料)
  <out>_steps.npz     同上数组形式
  <out>_summary.json  每回合汇总 + 均值
物理规矩 = task_config.PHYS (与训练同源); 与 ckpt 同目录 world.json 核对物理/常量/IO 维, 不符即拒绝回放。
课程档位 --release_row 需手工给 (ckpt 不存; 训练日志 grep "[课程]" 取当前值), 默认 RELEASE_MAX。
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
p.add_argument("--release_row", type=int, default=None, help="课程档位 (默认 RELEASE_MAX)")
p.add_argument("--jitter", type=int, default=None, help="release 抖动行数 (默认 task_config.RELEASE_JITTER; 0=固定)")
p.add_argument("--no_video", action="store_true")
p.add_argument("--focal", type=float, default=24.0)
p.add_argument("--cams", default="front:-0.85,-0.45,0.60;side:0.75,-0.95,0.50;top:-0.30,-0.60,0.90",
               help="相机集 name:dx,dy,dz;... (相对两物体中点的偏移 m); 每个相机出 <out>_<name>.mp4 (仅 1 个时为 <out>.mp4)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():                 # 物理规矩必须在 import env 之前
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_rec")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import clean_env as CE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = CE.SIDES
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PAIRS = ("idx_mid", "mid_ring", "ring_pinky")

# ---- 世界核对 (与 ckpt 同目录 world.json): 物理三项 / stage1 常量 / 策略 IO 维 ----
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
    io = world.get("policy_io", {})
    if (io.get("obs_dim"), io.get("priv_dim"), io.get("act_dim")) != (CE.OBS_DIM, CE.PRIV_DIM, CE.ACT_DIM):
        bad.append(f"policy_io: 记录={io} 当前=({CE.OBS_DIM},{CE.PRIV_DIM},{CE.ACT_DIM})")
    if bad and not os.environ.get("CLEAN_IGNORE_WORLD"):
        print("[world] ✗ 与 ckpt 出生世界不符 (CLEAN_IGNORE_WORLD=1 可跳过):\n  " + "\n  ".join(bad), flush=True)
        os._exit(2)
    print(f"[world] ✅ 与 world.json 匹配: 物理={world.get('physics')} release_row_start={world.get('release_row_start')}", flush=True)

# ---- 环境 + 策略 ----
raw = CE.CleanHoldEnv(CE.build_cfg(args.num_envs))
raw.release_row_cur = int(args.release_row if args.release_row is not None else TC.RELEASE_MAX)
if args.jitter is not None:
    TC.RELEASE_JITTER = int(args.jitter)          # _reset_idx 每次从模块读
try:                                              # 用户规矩: 物体带图案纹理 (Clean3 无 SAM3D 档 → 程序图案)
    from rl_rebuild.correction.texture_objects import apply_textures
    apply_textures(raw, tex_dir=os.path.join(TC.REPO, "datasets", "clean_tableware", "3", "cache", "textures"),
                   names=(("object", "plate", "盘"), ("aux", "sponge", "海绵")))
except Exception as _te:
    print(f"[纹理] 跳过 ({_te})", flush=True)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_clean.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "clean3_rec"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(ckpt)
agent.set_eval()
print(f"[record_clean] ckpt={ckpt} N={args.num_envs} release_row_cur={raw.release_row_cur} jitter={TC.RELEASE_JITTER} "
      f"T_EP={TC.T_EP} cert={TC.CERT_POS*100:.1f}cm/{TC.CERT_ROT_DEG:.0f}°", flush=True)

# ---- 相机 (env0, 对准两物体中点; 多机位) ----
_annots = {}
if not args.no_video:
    import omni.replicator.core as rep  # noqa: E402
    import omni.usd  # noqa: E402
    from pxr import Gf, UsdGeom  # noqa: E402
    org = raw._org()[0].detach().cpu().numpy()
    ctr = 0.5 * (raw.nominal["left"][0] + raw.nominal["right"][0]).detach().cpu().numpy() + org
    _st = omni.usd.get_context().get_stage()
    for spec in [c for c in args.cams.split(";") if c.strip()]:
        name, off = spec.split(":"); off = np.array([float(x) for x in off.split(",")])
        eye = ctr + off
        _cam = UsdGeom.Camera.Define(_st, f"/World/RecCam_{name}")
        _cam.CreateFocalLengthAttr().Set(float(args.focal))
        _m = Gf.Matrix4d(); _m.SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, ctr)), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(_cam).AddTransformOp().Set(_m.GetInverse())
        _rp = rep.create.render_product(f"/World/RecCam_{name}", (1280, 720))
        _a = rep.AnnotatorRegistry.get_annotator("rgb"); _a.attach(_rp); _annots[name] = _a
        print(f"[record_clean] 相机 {name}: eye={eye.round(3).tolist()} → {ctr.round(3).tolist()} focal={args.focal}", flush=True)
    for _ in range(12):                      # 预热: 让纹理/材质加载完再开录 (首帧灰黑)
        raw.sim.render()
        for _a in _annots.values():
            _a.get_data()

# ---- 回放 ----
N = args.num_envs
T = int(TC.T_EP) + 2
cols = (["env", "step", "row", "release_row", "pinned", "released", "reward",
         "r_contact", "r_hold", "r_bonus", "r_act", "r_cross", "cross", "cross_any", "c_plate", "c_sponge",
         "dp_plate_cm", "dr_plate_deg", "dp_sponge_cm", "dr_sponge_deg", "within", "plate_ok", "sponge_ok",
         "cert", "cert_run", "new_cert", "dropped", "drop_plate", "drop_sponge", "new_drop", "success", "done",
         "act_abs_mean", "res_arm_max_rad", "res_fin_max_rad"]
        + [f"F_L_{f}" for f in FINGERS] + [f"F_R_{f}" for f in FINGERS]
        + [f"gapL_{q}_cm" for q in PAIRS] + [f"d3L_{q}_cm" for q in PAIRS]
        + [f"gapR_{q}_cm" for q in PAIRS] + [f"d3R_{q}_cm" for q in PAIRS])
rows = []
frames = {k: [] for k in _annots}
done_at = np.full(N, -1, dtype=int)
obs = env.reset()
with torch.no_grad():
    for t in range(T):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        tk = raw._tick
        d_now = dones.reshape(-1).bool().cpu().numpy() if torch.is_tensor(dones) else np.asarray(dones, bool).reshape(-1)
        gaps = {s: raw._finger_gaps(s) for s in SIDES}
        cum = raw.cum_res
        f = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
        rec = {
            "row": f(tk["row"]), "release_row": f(tk["release_row"]), "released": f(tk["released"]),
            "reward": f(tk["reward"]), "r_contact": f(tk["r_contact"]), "r_hold": f(tk["r_hold"]), "r_bonus": f(tk["r_bonus"]),
            "r_act": f(tk["r_act"]), "r_cross": f(tk["r_cross"]), "cross": f(tk["cross"]), "cross_any": f(tk["cross_any"]),
            "c_plate": f(tk["c_plate"]), "c_sponge": f(tk["c_sponge"]),
            "dp_plate_cm": f(tk["dp"]["left"]) * 100, "dr_plate_deg": np.degrees(f(tk["dr"]["left"])),
            "dp_sponge_cm": f(tk["dp"]["right"]) * 100, "dr_sponge_deg": np.degrees(f(tk["dr"]["right"])),
            "within": f(tk["within"]), "plate_ok": f(tk["plate_ok"]), "sponge_ok": f(tk["sponge_ok"]),
            "cert": f(tk["cert"]), "cert_run": f(tk["cert_run"]), "new_cert": f(tk["new_cert"]),
            "dropped": f(tk["dropped"]), "drop_plate": f(tk["drop_side"][:, 0]), "drop_sponge": f(tk["drop_side"][:, 1]),
            "new_drop": f(tk["new_drop"]), "success": f(tk["success"]), "done": d_now.astype(float),
            "act_abs_mean": f(act.abs().mean(1)), "res_arm_max_rad": f(cum[:, :14].abs().amax(1)), "res_fin_max_rad": f(cum[:, 14:].abs().amax(1)),
        }
        FL, FR = f(tk["F"]["left"]), f(tk["F"]["right"])
        gL, dL = f(gaps["left"][0]) * 100, f(gaps["left"][1]) * 100
        gR, dR = f(gaps["right"][0]) * 100, f(gaps["right"][1]) * 100
        for i in range(N):
            if done_at[i] >= 0:
                continue                                       # 该 env 已结束 (IsaacLab 已自动重置), 不记第二回合
            r = [i, t, int(rec["row"][i]), int(rec["release_row"][i]), float(rec["row"][i] <= rec["release_row"][i]), rec["released"][i], rec["reward"][i]]
            r += [rec[k][i] for k in cols[7:35]]
            if d_now[i]:                                       # 终止步: 指尖间距是重置后的读数, 记 NaN
                r += list(FL[i]) + list(FR[i]) + [float("nan")] * 12
            else:
                r += list(FL[i]) + list(FR[i]) + list(gL[i]) + list(dL[i]) + list(gR[i]) + list(dR[i])
            rows.append(r)
        if _annots and done_at[0] < 0:
            raw.sim.render()
            for k, _a in _annots.items():
                d = _a.get_data()
                if d is not None and getattr(d, "size", 0):
                    frames[k].append(np.asarray(d)[..., :3].astype(np.uint8))
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
                    checkpoint=ckpt, release_row_cur=raw.release_row_cur)
ci = {c: j for j, c in enumerate(cols)}
eps = []
for i in range(N):
    a = arr[arr[:, ci["env"]] == i]
    if len(a) == 0:
        continue
    rel = a[a[:, ci["released"]] > 0]
    cert_rows = a[a[:, ci["new_cert"]] > 0][:, ci["row"]]
    drop_rows = a[a[:, ci["new_drop"]] > 0][:, ci["row"]]
    eps.append({
        "env": i, "ep_len": int(a[-1, ci["row"]]), "release_row": int(a[0, ci["release_row"]]),
        "return": float(a[:, ci["reward"]].sum()),
        "sum": {k: float(a[:, ci[k]].sum()) for k in ("r_contact", "r_hold", "r_bonus", "r_act", "r_cross")},
        "success": bool(a[-1, ci["success"]]), "cert": bool(a[-1, ci["cert"]]), "cert_row": (int(cert_rows[0]) if len(cert_rows) else None),
        "dropped": bool(a[-1, ci["dropped"]]), "drop_plate": bool(a[-1, ci["drop_plate"]]), "drop_sponge": bool(a[-1, ci["drop_sponge"]]),
        "drop_row": (int(drop_rows[0]) if len(drop_rows) else None),
        "released_mean": ({k: float(rel[:, ci[k]].mean()) for k in ("dp_plate_cm", "dr_plate_deg", "dp_sponge_cm", "dr_sponge_deg", "within")} if len(rel) else None),
        "cross_frac": float(a[:, ci["cross_any"]].mean()),
        "cross_frac_released": (float(rel[:, ci["cross_any"]].mean()) if len(rel) else None),
        "gap_min_cm": {"L": [float(np.nanmin(a[:, ci[f"gapL_{q}_cm"]])) for q in PAIRS], "R": [float(np.nanmin(a[:, ci[f"gapR_{q}_cm"]])) for q in PAIRS]},
        "d3_min_cm": {"L": [float(np.nanmin(a[:, ci[f"d3L_{q}_cm"]])) for q in PAIRS], "R": [float(np.nanmin(a[:, ci[f"d3R_{q}_cm"]])) for q in PAIRS]},
    })
summary = {"checkpoint": ckpt, "release_row_cur": raw.release_row_cur, "jitter": int(TC.RELEASE_JITTER), "num_envs": N,
           "cert_thr": [TC.CERT_POS * 100, TC.CERT_ROT_DEG], "episodes": eps,
           "mean": {"success": float(np.mean([e["success"] for e in eps])), "cert": float(np.mean([e["cert"] for e in eps])),
                    "dropped": float(np.mean([e["dropped"] for e in eps])), "return": float(np.mean([e["return"] for e in eps])),
                    "cross_frac": float(np.mean([e["cross_frac"] for e in eps]))}}
with open(out + "_summary.json", "w") as fj:
    json.dump(summary, fj, indent=1, ensure_ascii=False)
n_frames = 0
if any(frames.values()):
    import imageio  # noqa: E402
    for k, fr in frames.items():
        if fr:
            imageio.mimsave(out + (".mp4" if len(frames) == 1 else f"_{k}.mp4"), fr, fps=int(TC.CONTROL_HZ)); n_frames = len(fr)
print(f"[record] {out} | 帧={n_frames}x{len(frames)}机位 | 逐步行={len(rows)} | success={summary['mean']['success']:.2f} "
      f"cert={summary['mean']['cert']:.2f} drop={summary['mean']['dropped']:.2f} return={summary['mean']['return']:.2f} "
      f"cross_frac={summary['mean']['cross_frac']:.2f} | done_at={done_at.tolist()}", flush=True)
for e in eps:
    print(f"[record]  env{e['env']}: len={e['ep_len']} rel={e['release_row']} ret={e['return']:.1f} {e['sum']} "
          f"succ={int(e['success'])} cert@{e['cert_row']} drop@{e['drop_row']} sponge_drop={int(e['drop_sponge'])} "
          f"rel_mean={e['released_mean']} cross={e['cross_frac']:.2f} gapmin R={np.round(e['gap_min_cm']['R'],2).tolist()} d3min R={np.round(e['d3_min_cm']['R'],2).tolist()}", flush=True)
sys.stdout.flush()
try:
    _slot.release()
except Exception:
    pass
os._exit(0)          # 不调 app.close(): Isaac 关闭会 hang (2026-09-08 实测两条卡在 close 里占显存不退)
