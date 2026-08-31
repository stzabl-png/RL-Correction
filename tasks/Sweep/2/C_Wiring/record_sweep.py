"""Record one deterministic Sweep2 episode under project-root outputs_video/."""
from __future__ import annotations

import argparse
from collections import deque
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="", help="empty means zero-residual reference replay")
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
import sweep_env as SE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
allowed = os.path.join(ROOT, "outputs_video")
assert args.out or args.topdown_frames_dir, "provide --out and/or --topdown_frames_dir"
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

raw = SE.SweepEnv(SE.build_cfg(1))
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
    m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(0.95, -1.15, 1.45),
                                   Gf.Vec3d(0.25, 0.0, 0.90), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
    rp = rep.create.render_product("/World/SweepRecCam", (1280, 720))
    annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)

top_annot = None
if frames_dir:
    # Use Replicator's camera authoring path directly: its render product then
    # consumes the same position/look-at values without a USD matrix convention
    # conversion.  A tiny Y offset avoids a vertical look-at singularity while
    # keeping the optical center on the table center.
    top_cam = rep.create.camera(position=(0.0, -0.02, 2.20),
                                look_at=(0.0, 0.0, 0.87),
                                focal_length=18.0)
    top_rp = rep.create.render_product(top_cam, (1280, 720))
    top_annot = rep.AnnotatorRegistry.get_annotator("rgb"); top_annot.attach(top_rp)

obs = env.reset(); frames = []; gmax = [0, 0, 0, 0]
top_frames = deque(maxlen=args.success_context)
success_step = None
rollout = {k: [] for k in ("obs", "priv_info", "actions", "rewards",
                            "rows", "gates", "success", "cube_pan",
                            "actor_mask", "cum_res", "next_obs", "next_priv_info",
                            "entered", "fully_inside", "deep_inside", "deep_margin",
                            "deep_progress", "broom_assisted_progress")}
record_steps = args.steps if args.steps > 0 else raw.T + 40
with torch.no_grad():
    for t in range(record_steps):
        if agent is None:
            action = torch.zeros(1, SE.ACT_DIM, device=raw.device)
        else:
            action = agent.model.act_inference({
                "obs": agent.running_mean_std(obs["obs"]),
                "priv_info": obs["priv_info"]}).clamp(-1, 1)
        rollout["obs"].append(obs["obs"][0].cpu().numpy().copy())
        rollout["priv_info"].append(obs["priv_info"][0].cpu().numpy().copy())
        rollout["actor_mask"].append(obs["actor_mask"][0].cpu().numpy().copy())
        rollout["rows"].append(int(raw.row[0]))
        rollout["actions"].append(action[0].cpu().numpy().copy())
        obs, reward, done, info = env.step(action)
        tick = raw._tick_out
        rollout["rewards"].append(float(reward[0]))
        rollout["next_obs"].append(obs["obs"][0].cpu().numpy().copy())
        rollout["next_priv_info"].append(obs["priv_info"][0].cpu().numpy().copy())
        rollout["gates"].append(tick["gates"][0].cpu().numpy().copy())
        rollout["success"].append(bool(tick["success"][0]))
        rollout["cube_pan"].append(tick["signals"]["cube_pan"][0].cpu().numpy().copy())
        rollout["entered"].append(bool(tick["entered"][0]))
        rollout["fully_inside"].append(bool(tick["fully_inside"][0]))
        rollout["deep_inside"].append(bool(tick["deep_inside"][0]))
        rollout["deep_margin"].append(float(tick["deep_margin"][0]))
        rollout["deep_progress"].append(float(tick["deep_progress"][0]))
        rollout["broom_assisted_progress"].append(
            float(raw.broom_assisted_progress[0]))
        rollout["cum_res"].append(raw.cum_res[0].cpu().numpy().copy())
        # DirectRLEnv may reset progress before returning on terminal.
        for i in range(4): gmax[i] = max(gmax[i], int(tick["gates"][0, i]))
        if bool(tick["success"][0]) and success_step is None:
            success_step = t
        # env.step() auto-resets a terminal environment before returning.  Rendering
        # here would capture the reset pose as the apparent success frame.  Keep the
        # last pre-terminal image instead (for the 15M regression this is frame 0231).
        if bool(done[0]):
            break
        raw.sim.render()
        if annot is not None:
            data = annot.get_data()
            if data is not None and getattr(data, "size", 0):
                frames.append(np.asarray(data)[..., :3].astype(np.uint8))
        if top_annot is not None:
            top_data = top_annot.get_data()
            if top_data is not None and getattr(top_data, "size", 0):
                top_frames.append((t, np.asarray(top_data)[..., :3].astype(np.uint8)))
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
print(f"[record_sweep] video={out or '-'} frames={len(frames)} "
      f"topdown={frames_dir or '-'} top_frames={len(top_frames)} "
      f"success_step={success_step} terminal_step={t} gates={gmax}")
try: _slot.release()
except Exception: pass
sys.stdout.flush()
os._exit(0)
