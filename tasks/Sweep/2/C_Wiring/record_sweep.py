"""Record one deterministic Sweep2 episode under project-root outputs_video/."""
from __future__ import annotations

import argparse
from collections import deque
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="", help="empty means zero-residual reference replay")
p.add_argument("--method", choices=("full", "wo_human", "wo_conf"), default="full")
p.add_argument("--out", default="", help="optional MP4 under project outputs_video/")
p.add_argument("--topdown_frames_dir", default="",
               help="optional directory under outputs_video/ for frames ending at first success")
p.add_argument("--success_context", type=int, default=12,
               help="number of consecutive top-down frames to retain through first success")
p.add_argument("--success_freeze_seconds", type=float, default=2.0,
               help="append an exact frozen terminal frame to the oblique MP4")
p.add_argument("--topdown_tail_on_failure", action="store_true",
               help="write the terminal context when the deterministic rollout fails")
p.add_argument("--trace", default="", help="optional rollout NPZ under project logs/")
p.add_argument("--seed", type=int, default=None, help="环境/torch 种子 (默认不动: 与历史录像一致); 成功率 <1 的策略换种子多录几条挑成功的做演示, 须在文件名里注明")
p.add_argument("--num_envs", type=int, default=1, help="场景里的环境数 (只拍/只记 env0). 单环境回合是完全确定的, 换 seed 无效; 多环境场景下 PhysX 接触求解顺序不同, env0 的结局会变 —— 用来捞成功回合做演示")
p.add_argument("--survey", action="store_true", help="普查模式: 不在 film_env 终止时停, 跑到所有 env 终止/超时, 末尾打印各 env 成败 (配 --num_envs, 找成功的 env 再拍)")
p.add_argument("--film_env", type=int, default=0, help="拍/记哪个 env (多环境场景下各 env 结局不同; 先不带 --out 跑一遍看末尾的各 env 成败, 再指定成功的那个拍)")
p.add_argument("--steps", type=int, default=0,
               help="0 records the complete reference plus a short terminal hold")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_record")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib  # noqa: E402
SE = importlib.import_module("sweep_grip_env" if os.environ.get("SWEEP_VARIANT") == "grip" else "sweep_env")  # noqa: E402
import csv  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
allowed = os.path.join(ROOT, "outputs_video")
assert args.out or args.topdown_frames_dir or args.trace, "provide --out and/or --topdown_frames_dir (or --trace for a trace-only pass)"
assert args.success_context > 0
out = os.path.abspath(args.out) if args.out else ""
if out:
    assert os.path.commonpath([allowed, out]) == allowed, f"video must be under {allowed}"
frames_dir = os.path.abspath(args.topdown_frames_dir) if args.topdown_frames_dir else ""
if frames_dir:
    assert os.path.commonpath([allowed, frames_dir]) == allowed, \
        f"top-down frames must be under {allowed}"
trace = os.path.abspath(args.trace) if args.trace else ""
if trace:
    logs_root = os.path.join(ROOT, "logs")
    assert os.path.commonpath([logs_root, trace]) == logs_root

_cfg = SE.build_cfg(args.num_envs, ablation_method=args.method)
if args.seed is not None:
    _cfg.seed = int(args.seed); SE.SweepEnv.seed(args.seed); print(f"[record_sweep] seed = {args.seed} (cfg.seed 同步; 静态 seed() 会被构造时的 cfg.seed 覆盖)", flush=True)
raw = SE.SweepEnv(_cfg)
# Keep the physical terminal state alive through rendering; the trace still uses
# tick["terminated"]/tick["timeout"] as the rollout boundary.
raw.suppress_terminal_reset = True
import hashlib as _hl
print(f"[record_sweep] reference = {SE.REFERENCE} sha256[:16]={_hl.sha256(open(SE.REFERENCE, 'rb').read()).hexdigest()[:16]}", flush=True)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent = None
if args.checkpoint:
    with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
        acfg = yaml.safe_load(f)
    acfg["algorithm"]["num_actors"] = 1
    agent = PPO(env, output_dir=os.path.join(ROOT, "logs", "_record_tmp"),
                full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
    agent.restore_test(args.checkpoint); agent.set_eval()
else:
    # Visual reconstruction audit must play the complete source path even when the
    # task clock would normally wait for physical cube contact.
    raw.force_replay = True

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
annot = None
if out:
    cam = UsdGeom.Camera.Define(stage, "/World/SweepRecCam")
    cam.CreateFocalLengthAttr().Set(18.0)
    # User-approved overview: robot front-left, slightly elevated, with the
    # upper body, both arms, and manipulation area visible.
    _o = raw.scene.env_origins[args.film_env].cpu().numpy().tolist()
    m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(1.25 + _o[0], -1.65 + _o[1], 1.55 + _o[2]),
                                   Gf.Vec3d(0.02 + _o[0], 0.0 + _o[1], 1.02 + _o[2]), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
    rp = rep.create.render_product("/World/SweepRecCam", (1280, 720))
    annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)

top_annot = None
if frames_dir:
    # Use Replicator's camera authoring path directly: its render product then
    # consumes the same position/look-at values without a USD matrix convention
    # conversion.  A tiny Y offset avoids a vertical look-at singularity while
    # keeping the optical center on the table center.
    _o = raw.scene.env_origins[args.film_env].cpu().numpy().tolist()
    top_cam = rep.create.camera(position=(0.0 + _o[0], -0.02 + _o[1], 2.20 + _o[2]),
                                look_at=(0.0 + _o[0], 0.0 + _o[1], 0.87 + _o[2]),
                                focal_length=18.0)
    top_rp = rep.create.render_product(top_cam, (1280, 720))
    top_annot = rep.AnnotatorRegistry.get_annotator("rgb"); top_annot.attach(top_rp)

K = int(args.film_env)
obs = env.reset(); frames = []; gmax = [0, 0, 0, 0]
top_frames = deque(maxlen=args.success_context)
success_step = None
rollout = {k: [] for k in ("obs", "priv_info", "actions", "rewards",
                            "rows", "gates", "success", "cube_pan",
                            "actor_mask", "cum_res", "next_obs", "next_priv_info",
                            "entered", "fully_inside", "deep_inside", "deep_margin",
                            "full_progress", "deep_progress",
                            "broom_assisted_progress")}
# 2026-09-14: 逐步奖惩分项 + 终止原因 + 握持量, 另落一份 csv (与 trace 同名 .csv), 给人看
step_rows = []
record_steps = args.steps if args.steps > 0 else raw.T + 40
with torch.no_grad():
    for t in range(record_steps):
        if agent is None:
            action = torch.zeros(1, SE.ACT_DIM, device=raw.device)
        else:
            action = agent.model.act_inference({
                "obs": agent.running_mean_std(obs["obs"]),
                "priv_info": obs["priv_info"]}).clamp(-1, 1)
        rollout["obs"].append(obs["obs"][K].cpu().numpy().copy())
        rollout["priv_info"].append(obs["priv_info"][K].cpu().numpy().copy())
        rollout["actor_mask"].append(obs["actor_mask"][K].cpu().numpy().copy())
        rollout["rows"].append(int(raw.row[K]))
        rollout["actions"].append(action[K].cpu().numpy().copy())
        obs, reward, done, info = env.step(action)
        tick = raw._tick_out
        rollout["rewards"].append(float(reward[K]))
        rollout["next_obs"].append(obs["obs"][K].cpu().numpy().copy())
        rollout["next_priv_info"].append(obs["priv_info"][K].cpu().numpy().copy())
        rollout["gates"].append(tick["gates"][K].cpu().numpy().copy())
        rollout["success"].append(bool(tick["success"][K]))
        rollout["cube_pan"].append(tick["signals"]["cube_pan"][K].cpu().numpy().copy())
        rollout["entered"].append(bool(tick["entered"][K]))
        rollout["fully_inside"].append(bool(tick["fully_inside"][K]))
        rollout["deep_inside"].append(bool(tick["deep_inside"][K]))
        rollout["deep_margin"].append(float(tick["deep_margin"][K]))
        rollout["full_progress"].append(float(tick["full_progress"][K]))
        rollout["deep_progress"].append(float(tick["deep_progress"][K]))
        rollout["broom_assisted_progress"].append(
            float(raw.broom_assisted_progress[K]))
        rollout["cum_res"].append(raw.cum_res[K].cpu().numpy().copy())
        # DirectRLEnv may reset progress before returning on terminal.
        for i in range(4): gmax[i] = max(gmax[i], int(tick["gates"][K, i]))
        terminal = bool((tick["terminated"] | tick["timeout"]).all()) if args.survey else bool(tick["terminated"][K] or tick["timeout"][K])
        sig = tick["signals"]; g = tick.get("grip")
        cube_z = float(raw.cube.data.root_pos_w[0, 2] - raw.cfg.table_top_z)
        row_rec = {"step": t, "row": int(raw.row[K]), "reward": float(reward[K]),
                   **{f"r_{k}": float(v[K]) for k, v in tick["reward_terms"].items()},
                   "gate1": int(tick["gates"][0, 0]), "gate2": int(tick["gates"][0, 1]), "gate3": int(tick["gates"][0, 2]), "gate4": int(tick["gates"][0, 3]),
                   "progress": float(sig["progress"][K]), "moved_cm": float(sig["moved"][K]) * 100, "broom_dist_cm": float(sig["broom_dist"][K]) * 100,
                   "cube_pan_x_cm": float(sig["cube_pan"][0, 0]) * 100, "cube_pan_y_cm": float(sig["cube_pan"][0, 1]) * 100, "cube_pan_z_cm": float(sig["cube_pan"][0, 2]) * 100,
                   "mouth_clr_mm": float(sig["mouth_clearance"][K]) * 1000, "pan_tilt_deg": float(sig["pan_tilt"][K]) * 57.2958, "cube_z_table_mm": cube_z * 1000,
                   "entered": int(tick["entered"][K]), "fully_inside": int(tick["fully_inside"][K]), "success": int(tick["success"][K]),
                   "terminated": int(tick["terminated"][K]), "timeout": int(tick["timeout"][K]),
                   "policy_active": int(t >= SE.SCRIPTED_PRELUDE_STEPS)}
        if g is not None:
            row_rec.update({"released": int(g["released"][K]), "certified": int(g["certified"][K]), "died": int(g["died"][K]),
                            "dp_broom_cm": float(g["dp_r"][K]) * 100, "dr_broom_deg": float(g["dr_r"][K]) * 57.2958,
                            "dp_pan_cm": float(g["dp_l"][K]) * 100, "dr_pan_deg": float(g["dr_l"][K]) * 57.2958})
        fail_cube = cube_z < -0.03
        row_rec["end_reason"] = ("success" if row_rec["success"] else "grip_die" if (g is not None and g["died"][K] and row_rec["terminated"]) else
                                 "cube_fell" if (row_rec["terminated"] and fail_cube) else "mouth_floor_fail" if row_rec["terminated"] else
                                 "timeout" if row_rec["timeout"] else "")
        step_rows.append(row_rec)
        if bool(tick["success"][K]) and success_step is None:
            success_step = t
        raw.sim.render()
        if annot is not None:
            data = annot.get_data()
            if data is not None and getattr(data, "size", 0):
                frames.append(np.asarray(data)[..., :3].astype(np.uint8))
        if top_annot is not None:
            top_data = top_annot.get_data()
            if top_data is not None and getattr(top_data, "size", 0):
                top_frames.append((t, np.asarray(top_data)[..., :3].astype(np.uint8)))
        if terminal:
            break
if out:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    freeze_frames = int(round(max(args.success_freeze_seconds, 0.0) * 20.0))
    if success_step is not None and frames and freeze_frames:
        frames.extend([frames[-1].copy() for _ in range(freeze_frames)])
    imageio.mimsave(out, frames, fps=20)
if frames_dir:
    if success_step is None and not args.topdown_tail_on_failure:
        raise RuntimeError("episode ended without success; no success-window frames written")
    os.makedirs(frames_dir, exist_ok=True)
    for frame_step, frame in top_frames:
        imageio.imwrite(os.path.join(frames_dir, f"frame_{frame_step:04d}.png"), frame)
if trace:
    os.makedirs(os.path.dirname(trace), exist_ok=True)
    np.savez_compressed(trace, **{k: np.asarray(v) for k, v in rollout.items()})
    csv_path = os.path.splitext(trace)[0] + "_steps.csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=list(step_rows[K].keys())); w.writeheader(); w.writerows(step_rows)
    ends = [r["end_reason"] for r in step_rows if r["end_reason"]]
    print(f"[record_sweep] 逐步账 → {csv_path} ({len(step_rows)} 步) 终止={ends[-1] if ends else 'running'} "
          f"gate最高={gmax} 累计奖 {sum(r['reward'] for r in step_rows):.2f}", flush=True)
print(f"[record_sweep] video={out or '-'} frames={len(frames)} "
      f"topdown={frames_dir or '-'} top_frames={len(top_frames)} "
      f"success_step={success_step} terminal_step={t} gates={gmax}")
try:
    _g = raw.progress.gates
    _ok = [int(i) for i in torch.nonzero(_g[:, 3]).reshape(-1).tolist()]
    print(f"[record_sweep] 各 env 成败 (gate4 锁存): 成功 {len(_ok)}/{_g.shape[0]} -> {_ok[:40]}", flush=True)
except Exception as _e:
    print(f"[record_sweep] 各 env 成败: 读取失败 {_e}", flush=True)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
