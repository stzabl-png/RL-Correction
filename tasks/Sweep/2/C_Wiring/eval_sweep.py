"""Deterministic fixed-start evaluation; acceptance requires >=512 episodes, >=50%."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--episodes", type=int, default=None)
p.add_argument("--protocol", choices=("final512", "curve50"), default="final512")
p.add_argument("--method", choices=("full", "wo_human", "wo_conf"), default="full")
p.add_argument("--variant_index", type=int, default=None,
               help="force one cube variant (0..4); requires SWEEP_CUBE_VARIANTS_NPZ")
p.add_argument("--world", default=None, help="training world.json; defaults beside checkpoint")
p.add_argument("--out", required=True, help="JSON path under project logs/")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
required = 512 if args.protocol == "final512" else 50
episodes = args.episodes if args.episodes is not None else required
assert episodes >= required, f"{args.protocol} requires at least {required} episodes"
assert args.num_envs >= 1
if args.variant_index is not None:
    assert 0 <= args.variant_index < 5
    assert os.environ.get("SWEEP_CUBE_VARIANTS_NPZ"), \
        "--variant_index requires SWEEP_CUBE_VARIANTS_NPZ"
    os.environ["SWEEP_CUBE_VARIANT_INDEX"] = str(args.variant_index)

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_eval")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
import world_fingerprint as WF  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

out = os.path.abspath(args.out)
assert os.path.commonpath([os.path.join(ROOT, "logs"), out]) == os.path.join(ROOT, "logs")
raw = SE.SweepEnv(SE.build_cfg(args.num_envs, ablation_method=args.method))
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
world_path = args.world or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)),
                                       "world.json")
with open(world_path) as handle:
    expected_world = json.load(handle)
expected_world.update({
    "task": ("Sweep2_cube_variants_fullinside" if SE.CUBE_VARIANTS
             else "Sweep2_fixed_cube_fullinside"),
    "policy_io": {"obs_dim": SE.OBS_DIM, "priv_dim": SE.PRIV_DIM,
                  "act_dim": SE.ACT_DIM},
    "time": {"control_dt_s": 0.05,
             "scripted_prelude_steps": SE.SCRIPTED_PRELUDE_STEPS},
    "success": {"definition": "fully_inside", "whole_cube_inside": True,
                "mouth_clearance_margin_m": 0.0, "immediate_termination": True,
                "recording_only_freeze_seconds": 2.0,
                "final_eval_episodes": 512, "required_rate": 0.50},
    "mouth_floor": {
        "penalty_start_clearance_m": SE.MOUTH_PENALTY_START_M,
        "penalty_span_m": SE.MOUTH_PENALTY_SPAN_M,
        "penalty_scale": SE.MOUTH_PENALTY_SCALE,
        "failure_clearance_m": SE.MOUTH_FAILURE_CLEARANCE_M,
    },
    "reference": {"path": SE.REFERENCE, "sha256": _sha256(SE.REFERENCE)},
})
if not SE.CUBE_VARIANTS:
    expected_world["cube_start_world_m"] = list(SE.SWEEP2_FIXED_CUBE_START)
pan_asset = SE.clips.clip_entry("Sweep2_broom")["secondary"]["mesh"]
expected_world["dustpan_asset"] = {"path": pan_asset, "sha256": _sha256(pan_asset)}
expected_world["method"] = {
    "name": raw.ablation.name,
    "human_shape": raw.ablation.human_shape,
    "confidence_mode": raw.ablation.confidence_mode,
    "warmup_role": raw.ablation.warmup_role,
}
WF.assert_match(world_path, expected_world,
                allow_legacy_full=(args.method == "full"))
with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
    acfg = yaml.safe_load(f)
acfg["algorithm"]["num_actors"] = args.num_envs
agent = PPO(env, output_dir=os.path.join(ROOT, "logs", "_eval_tmp"),
            full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint); agent.set_eval()
obs = env.reset()
successes, episode_lengths, done_n = 0, [], 0
gate_counts = [0, 0, 0, 0]
active_lengths = torch.zeros(args.num_envs, device=raw.device)
with torch.no_grad():
    while done_n < episodes:
        mu = agent.model.act_inference({"obs": agent.running_mean_std(obs["obs"]),
                                        "priv_info": obs["priv_info"]}).clamp(-1, 1)
        obs, reward, done, info = env.step(mu)
        active_lengths += 1
        for idx in torch.where(done)[0].tolist():
            if done_n >= episodes:
                break
            successes += int(raw._tick_out["success"][idx])
            for gate_i in range(4):
                gate_counts[gate_i] += int(raw._tick_out["gates"][idx, gate_i])
            episode_lengths.append(float(active_lengths[idx]))
            active_lengths[idx] = 0
            done_n += 1
rate = successes / episodes
report = {"checkpoint": os.path.abspath(args.checkpoint), "protocol": args.protocol,
          "method": args.method, "episodes": episodes,
          "parallel_envs": args.num_envs, "successes": successes,
          "success_rate": rate,
          "gate_counts": gate_counts,
          "gate_rates": [count / episodes for count in gate_counts],
          "variant_index": args.variant_index,
          "variant_id": (raw.cube_variant_ids[args.variant_index]
                         if args.variant_index is not None else None),
          "mean_length": sum(episode_lengths) / episodes,
          "seed": int(acfg.get("seed", 42)), "checkpoint_sha256": _sha256(args.checkpoint),
          "world": os.path.abspath(world_path),
          "acceptance_threshold": 0.50,
          "accepted": bool(rate >= 0.50) if args.protocol == "final512" else None}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f: json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
try: _slot.release()
except Exception: pass
sys.stdout.flush()
# Isaac Sim 5.1 may hang during plugin teardown after the report is durable.
# End at the same clean process boundary used by the training entrypoint.
os._exit(2 if args.protocol == "final512" and rate < 0.50 else 0)
