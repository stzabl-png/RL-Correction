"""Clean/3 Stage-2 策略回放: 载 ckpt, num_envs 个回合并行 (env0 三机位录像), 确定性 mu, 逐步奖惩落盘。
旗 (接触门/重锚) 与常量从 ckpt 同目录 world.json 的 stage2 还原; 课程档位 --release_row (默认 RELEASE_MIN=10)。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Clean/3/C_Wiring/record_task.py --checkpoint logs/Clean3_task_s42/stage1_nn/last.pth \\
      --out logs/Clean3_task_s42/videos/last_r10 --headless --enable_cameras
产物: <out>_{front,side,top}.mp4 (1280x720, 20fps=1 行/帧) · <out>_steps.csv/.npz (所有 env 逐步) · <out>_summary.json
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
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--release_row", type=int, default=None)
p.add_argument("--jitter", type=int, default=0)
p.add_argument("--no_video", action="store_true")
p.add_argument("--focal", type=float, default=24.0)
p.add_argument("--cams", default="front:-0.85,-0.45,0.60;side:0.75,-0.95,0.50;top:-0.30,-0.60,0.90")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
ckpt = os.path.abspath(args.checkpoint)
wj = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "world.json")
world = json.load(open(wj)) if os.path.isfile(wj) else {}
# ★旗必须在 import task_config 之前按 world.json 还原 (task_config 导入期读环境变量)
_s2 = world.get("stage2", {})
os.environ["CLEAN_S2_CLOCK_CONTACT"] = "1" if _s2.get("S2_CLOCK_NEEDS_CONTACT") else "0"
os.environ["CLEAN_S2_REANCHOR"] = "1" if _s2.get("S2_REANCHOR") else "0"
os.environ["CLEAN_S2_SOFT_REL"] = "1" if _s2.get("S2_SOFT_REL") else "0"
os.environ["CLEAN_S2_GATE_HOLD"] = "1" if _s2.get("S2_GATE_HOLD") else "0"
os.environ["CLEAN_S2_PUSH"] = os.environ.get("CLEAN_S2_PUSH_OVERRIDE", "0")   # 回放默认无推力 (鲁棒性回放传 CLEAN_S2_PUSH_OVERRIDE=1)
if _s2.get("S2_SOFT_W") is not None: os.environ["CLEAN_S2_SOFT_W"] = str(_s2["S2_SOFT_W"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in (world.get("physics") or TC.PHYS).items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_rec_task")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import clean_task_env as CE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = CE.SIDES
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# 世界核对 (None 记录 = 未验不拦)
bad = []
for k, v in _s2.items():
    if v is None or k == "S2_PUSH":
        continue                                                              # None=未验不拦; S2_PUSH 回放期有意关
    cur = getattr(TC, k, None)
    cur = json.loads(json.dumps(cur)) if cur is not None else None          # 元组/字典经 JSON 归一化后再比 (tuple≠list 假警报)
    same = (abs(float(cur) - float(v)) < 1e-9) if isinstance(v, (int, float)) and not isinstance(v, bool) and cur is not None else (cur == v)
    if not same:
        bad.append(f"{k}: 记录={v} 当前={cur}")
io = world.get("policy_io", {})
if io and (io.get("obs_dim"), io.get("priv_dim"), io.get("act_dim")) != (CE.OBS_DIM, CE.PRIV_DIM, CE.ACT_DIM):
    bad.append(f"policy_io 记录={io} 当前=({CE.OBS_DIM},{CE.PRIV_DIM},{CE.ACT_DIM})")
if bad and not os.environ.get("CLEAN_IGNORE_WORLD"):
    print("[world] ✗ 与 ckpt 出生世界不符:\n  " + "\n  ".join(bad), flush=True); os._exit(2)
print(f"[world] ✅ {wj}: recipe={world.get('recipe')} contact={TC.S2_CLOCK_NEEDS_CONTACT} reanchor={TC.S2_REANCHOR} 物理={world.get('physics')}", flush=True)

raw = CE.CleanTaskEnv(CE.build_cfg(args.num_envs))
raw.release_row_cur = int(args.release_row if args.release_row is not None else TC.RELEASE_MIN)
TC.RELEASE_JITTER = int(args.jitter)
try:
    from rl_rebuild.correction.texture_objects import apply_textures
    apply_textures(raw, tex_dir=os.path.join(TC.REPO, "datasets", "clean_tableware", "3", "cache", "textures"),
                   names=(("object", "plate", "盘"), ("aux", "sponge", "海绵")))
except Exception as _te:
    print(f"[纹理] 跳过 ({_te})", flush=True)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_clean.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "clean3_rec_task"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(ckpt); agent.set_eval()
print(f"[record_task] ckpt={ckpt} N={args.num_envs} release={raw.release_row_cur} T_EP={raw.T_EP} T_REF={raw.T_REF}", flush=True)

_annots = {}
if not args.no_video:
    import omni.replicator.core as rep  # noqa: E402
    import omni.usd  # noqa: E402
    from pxr import Gf, UsdGeom  # noqa: E402
    org = raw._org()[0].detach().cpu().numpy()
    ctr = 0.5 * (raw.nominal["left"][0] + raw.nominal["right"][0]).detach().cpu().numpy() + org
    _st = omni.usd.get_context().get_stage()
    for spec in [c for c in args.cams.split(";") if c.strip()]:
        name, off = spec.split(":"); eye = ctr + np.array([float(x) for x in off.split(",")])
        _cam = UsdGeom.Camera.Define(_st, f"/World/RecCam_{name}"); _cam.CreateFocalLengthAttr().Set(float(args.focal))
        _m = Gf.Matrix4d(); _m.SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, ctr)), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(_cam).AddTransformOp().Set(_m.GetInverse())
        _rp = rep.create.render_product(f"/World/RecCam_{name}", (1280, 720))
        _a = rep.AnnotatorRegistry.get_annotator("rgb"); _a.attach(_rp); _annots[name] = _a
    for _ in range(12):
        raw.sim.render()
        for _a in _annots.values():
            _a.get_data()

N = args.num_envs
cols = (["env", "step", "row", "release_row", "released", "certified", "k", "gate_ok", "can", "contact", "reward", "r_adv", "r_leash", "r_bonus", "r_act", "r_soft",
         "e_n_cm", "e_xy_cm", "dp_plate_cm", "dr_plate_deg", "dp_sponge_cm", "dr_sponge_deg", "within", "plate_tilt_deg", "gap_mm",
         "coverage", "travel_cm", "success", "died", "die_kind", "cross_any", "dq_re_deg", "done"]
        + [f"F_L_{f}" for f in FINGERS] + [f"F_R_{f}" for f in FINGERS])
rows = []; frames = {k: [] for k in _annots}; done_at = np.full(N, -1, dtype=int)
obs = env.reset()
f = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
with torch.no_grad():
    for t in range(raw.T_EP + 2):
        inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        tk = raw._tick
        d_now = (dones.reshape(-1) > 0).cpu().numpy() if torch.is_tensor(dones) else np.asarray(dones, bool).reshape(-1)
        rec = dict(row=f(tk["row"]), release_row=f(tk["release_row"]), released=f(tk["released"]), certified=f(tk["certified"]), k=f(tk["k"]),
                   gate_ok=f(tk["gate_ok"]), can=f(tk["can"]), contact=f(tk["sig"]["contact"]), reward=f(tk["reward"]), r_adv=f(tk["r_adv"]),
                   r_leash=f(tk["r_leash"]), r_bonus=f(tk["r_bonus"]), r_act=f(tk["r_act"]), r_soft=f(tk["r_soft"]), e_n_cm=f(tk["e_n"]) * 100, e_xy_cm=f(tk["e_xy"]) * 100,
                   dp_plate_cm=f(tk["dp"]["left"]) * 100, dr_plate_deg=np.degrees(f(tk["dr"]["left"])), dp_sponge_cm=f(tk["dp"]["right"]) * 100,
                   dr_sponge_deg=np.degrees(f(tk["dr"]["right"])), within=f(tk["within"]), plate_tilt_deg=np.degrees(f(tk["sig"]["plate_tilt"])),
                   gap_mm=f(tk["sig"]["gap_min"]) * 1000, coverage=f(tk["out"]["coverage"]), travel_cm=f(tk["out"]["travel"]) * 100,
                   success=f(tk["success"]), died=f(tk["died"]), die_kind=f(tk["die_kind"]), cross_any=f(tk["cross_any"]),
                   dq_re_deg=np.degrees(f(tk["dq_re"].abs().amax(1))), done=d_now.astype(float))
        FL, FR = f(tk["F"]["left"]), f(tk["F"]["right"])
        for i in range(N):
            if done_at[i] >= 0:
                continue
            rows.append([i, t] + [rec[c][i] for c in cols[2:33]] + list(FL[i]) + list(FR[i]))
        if _annots and done_at[0] < 0:
            raw.sim.render()
            for k, _a in _annots.items():
                d = _a.get_data()
                if d is not None and getattr(d, "size", 0):
                    frames[k].append(np.asarray(d)[..., :3].astype(np.uint8))
        done_at = np.where(d_now & (done_at < 0), t, done_at)
        if (done_at >= 0).all():
            break
out = os.path.abspath(args.out); os.makedirs(os.path.dirname(out), exist_ok=True)
arr = np.asarray(rows, dtype=np.float64)
with open(out + "_steps.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(cols)
    for r in rows:
        w.writerow([f"{x:.5g}" if isinstance(x, float) else x for x in r])
np.savez_compressed(out + "_steps.npz", cols=np.array(cols), data=arr, done_at=done_at, checkpoint=ckpt)
le = {k: v.cpu().numpy().tolist() for k, v in raw.last_ep.items()}
DIE = {0: "-", 1: "rel", 2: "tilt", 3: "plate_dev", 4: "drop", 5: "table"}
eps = [{k: le[k][i] for k in le} | {"die": DIE[int(le["die_kind"][i])]} for i in range(N)]
summ = {"checkpoint": ckpt, "release_row": raw.release_row_cur, "num_envs": N, "episodes": eps,
        "mean": {k: float(np.mean(le[k])) for k in ("cert", "success", "clock_frac", "coverage", "travel_cm", "relp_max_plate_cm", "relp_max_sponge_cm",
                                                     "relrot_max_plate_deg", "relrot_max_sponge_deg", "cross_frac", "ep_len")}}
with open(out + "_summary.json", "w") as fj:
    json.dump(summ, fj, indent=1, ensure_ascii=False)
n_frames = 0
if any(frames.values()):
    import imageio  # noqa: E402
    for k, fr in frames.items():
        if fr:
            imageio.mimsave(out + f"_{k}.mp4", fr, fps=int(TC.CONTROL_HZ)); n_frames = len(fr)
m = summ["mean"]
print(f"[record] {out} | 帧={n_frames}x{len(frames)} | 逐步行={len(rows)} | cert={m['cert']:.2f} success={m['success']:.2f} clock={m['clock_frac']:.2f} "
      f"cov={m['coverage']:.2f} travel={m['travel_cm']:.0f}cm relp P/S={m['relp_max_plate_cm']:.2f}/{m['relp_max_sponge_cm']:.2f}cm "
      f"rot P/S={m['relrot_max_plate_deg']:.1f}/{m['relrot_max_sponge_deg']:.1f}° | done_at={done_at.tolist()} die={[e['die'] for e in eps]}", flush=True)
sys.stdout.flush()
try: _slot.release()
except Exception: pass
os._exit(0)
