"""Replay the frozen Sweep2 experts and collect PPO-compatible transitions.

This script never renders or writes video.  Every output must live below logs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--expert25", required=True)
p.add_argument("--expert40", required=True)
p.add_argument("--failure", required=True)
p.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
assert args.headless

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_collect_transitions")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
LOGS = os.path.join(ROOT, "logs")
out_dir = os.path.abspath(args.out_dir)
assert os.path.commonpath([LOGS, out_dir]) == LOGS
os.makedirs(out_dir, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discounted_returns(rewards, done, gamma=0.99, scale=0.01):
    out = np.zeros_like(rewards, dtype=np.float32); g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = scale * float(rewards[i]) + gamma * g * (1.0 - float(done[i]))
        out[i] = g
    return out


raw = SE.SweepEnv(SE.build_cfg(1)); raw.force_replay = True
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
# All three accepted traces were generated from the archived v1 easy-start
# cube pose.  The reference NPZ was edited later, so recover this frozen world
# state explicitly instead of silently collecting in a different task.
failure_source = np.load(os.path.abspath(args.failure))
assert failure_source["obs"].shape[1] >= 141
raw.cube_start_ref[:] = torch.tensor(
    failure_source["obs"][0, 138:141], dtype=torch.float32, device=raw.device)


def collect(label, source, expect_entered, expect_deep_success):
    source = os.path.abspath(source)
    assert os.path.commonpath([LOGS, source]) == LOGS and os.path.isfile(source)
    z = np.load(source); actions = np.asarray(z["actions"], np.float32)
    rows = np.asarray(z["rows"], np.int64)
    assert actions.shape == (len(rows), SE.ACT_DIM)
    obs = env.reset(); data = {k: [] for k in (
        "obs", "priv_info", "actions", "rewards", "next_obs",
        "next_priv_info", "done", "success", "rows", "actor_mask",
        "cube_pan", "pan_clearance", "pan_tilt", "push_quality", "pan_quality")}
    entered_frame = -1; success_frame = -1
    with torch.no_grad():
        for i, (row, action_np) in enumerate(zip(rows, actions)):
            raw.row[:] = int(row)
            current = raw._get_observations()
            action = torch.tensor(action_np, device=raw.device).reshape(1, -1)
            nxt, reward, _, _ = env.step(action)
            tick = raw._tick_out; sig = tick["signals"]
            success = bool(tick["success"][0])
            if bool(tick["entered"][0]) and entered_frame < 0:
                entered_frame = i
            terminal = success or i == len(rows) - 1
            data["obs"].append(current["policy"][0].cpu().numpy())
            data["priv_info"].append(current["priv_info"][0].cpu().numpy())
            data["actor_mask"].append(current["actor_mask"][0].cpu().numpy())
            data["actions"].append(action_np); data["rewards"].append(float(reward[0]))
            data["next_obs"].append(nxt["obs"][0].cpu().numpy())
            data["next_priv_info"].append(nxt["priv_info"][0].cpu().numpy())
            data["done"].append(terminal); data["success"].append(success)
            data["rows"].append(row); data["cube_pan"].append(sig["cube_pan"][0].cpu().numpy())
            data["pan_clearance"].append(float(sig["mouth_clearance"][0]))
            data["pan_tilt"].append(float(sig["pan_tilt"][0]))
            data["push_quality"].append(float(tick["push_potential"][0]))
            data["pan_quality"].append(float(tick["pan_potential"][0]))
            obs = nxt
            if success:
                success_frame = i
                break
    assert (entered_frame >= 0) == expect_entered, (label, entered_frame)
    assert (success_frame >= 0) == expect_deep_success, (label, success_frame)
    arrays = {k: np.asarray(v) for k, v in data.items()}
    arrays["return_target"] = discounted_returns(arrays["rewards"], arrays["done"])
    target = os.path.join(out_dir, f"{label}_transitions.npz")
    np.savez_compressed(target, **arrays)
    print(f"[collect] {label}: n={len(arrays['rows'])} "
          f"entered_frame={entered_frame} deep_success_frame={success_frame} -> {target}")
    return {"source": source, "source_sha256": sha256(source), "output": target,
            "output_sha256": sha256(target), "samples": len(arrays["rows"]),
            "entered_frame": entered_frame, "success_frame": success_frame,
            "demo_role": ("near_success" if expect_entered else "failure")}


manifest = {
    "schema": 2, "obs_dim": SE.OBS_DIM, "priv_dim": SE.PRIV_DIM,
    "action_dim": SE.ACT_DIM, "scripted_prelude_steps": SE.SCRIPTED_PRELUDE_STEPS,
    "gamma": 0.99, "reward_scale": 0.01, "datasets": [
        collect("sweep2_entry25", args.expert25, True, False),
        collect("sweep2_entry40", args.expert40, True, False),
        collect("sweep2_canonical_failure", args.failure, False, False),
    ]}
with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
try: _slot.release()
except Exception: pass
sys.stdout.flush()
# Isaac can hang during plugin teardown after every artifact is durable.
os._exit(0)
