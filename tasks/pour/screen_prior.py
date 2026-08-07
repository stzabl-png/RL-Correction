"""Run Task-5 Gate-1/2 screening for one left/right prior pair.

Gate-1 is the side-specific arm IK performed while the environment loads each
prior.  Gate-2 uses the actual Task-5 robot, assets, materials, contact sensors,
and PhysX configuration.  Eight bounded finger-residual settings per side are
evaluated while nominal closure sweeps from open to fully closed.  Each setting
must then lift the object off the table and hold it; table support cannot
satisfy Gate-2.

The output is evidence for, not a replacement for, human Dexonomy preview
selection.  ``approve_grasps`` remains the only command that can mark a scene
ready for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from tasks.pour.core import CurriculumStage


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--left-prior", required=True)
parser.add_argument("--right-prior", required=True)
parser.add_argument("--left-template", required=True)
parser.add_argument("--right-template", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("pour-screen-prior")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tasks.pour.cfg import PourTaskCfg, configure_for_screening  # noqa: E402
from tasks.pour.env import PourTaskEnv  # noqa: E402


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _side_metadata(path: str | Path, expected: str) -> None:
    with np.load(path, allow_pickle=False) as data:
        required = {"grasp", "pregrasp", "contact_centroid", "hand_side"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{expected} prior missing {sorted(missing)}")
        encoded = np.asarray(data["hand_side"], dtype=np.uint8).tobytes().decode("utf-8")
        if encoded != expected:
            raise ValueError(f"prior hand_side={encoded!r}, expected {expected!r}")


def _force_vectors(env: PourTaskEnv, side: str) -> torch.Tensor:
    values = []
    for sensor in env._contact_sensors[side]:
        force = sensor.data.force_matrix_w
        if force is None:
            force = sensor.data.net_forces_w.unsqueeze(1)
        values.append(force.reshape(env.num_envs, -1, 3).sum(dim=1))
    # Sensor values are object-on-pad; flip to pad-on-object, matching the
    # calibrated convention used by the existing grasp screen.
    return -torch.stack(values, dim=1).nan_to_num(0.0)


def _quality(
    env: PourTaskEnv,
    side: str,
    object_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    force = _force_vectors(env, side)
    magnitude = force.norm(dim=-1).clamp(max=50.0)
    effective = ((magnitude - 0.2) / (3.0 - 0.2)).clamp(0.0, 1.0)
    tips = env.robot.data.body_pos_w[:, env.tip_body_ids[side]]
    center = object_position[:, None, :]
    inward = center - tips
    inward = inward / inward.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
    unit_force = force / magnitude[..., None].clamp(min=1.0e-6)
    centering = ((unit_force * inward).sum(dim=-1).clamp(min=0.0) * effective).sum(1) / 5.0
    total = magnitude.sum(dim=1)
    imbalance = force.sum(dim=1).norm(dim=1) / (total + 1.0e-6)
    torque = torch.cross(tips - center, force, dim=-1).sum(dim=1)
    torque_normalized = (
        torque.norm(dim=1) / ((total + 1.0e-6) * 0.04)
    ).clamp(max=2.0)
    quality = centering - 0.5 * imbalance - 0.5 * torque_normalized
    pads = effective.sum(dim=1)
    return quality, pads, imbalance, torque_normalized, magnitude


def _residual_targets(device: str) -> torch.Tensor:
    return torch.tensor(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.15, 0.00, 0.00, 0.00, 0.00],
            [-0.15, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.10, 0.10, 0.10],
            [0.00, -0.10, -0.10, -0.10, -0.10],
            [0.15, -0.05, -0.05, -0.05, -0.05],
            [-0.10, 0.10, 0.10, 0.10, 0.10],
            [0.10, -0.10, 0.10, -0.10, 0.10],
        ],
        dtype=torch.float32,
        device=device,
    )


def _record_for_side(
    side: str,
    *,
    template: str,
    prior_path: str,
    best: dict,
) -> dict:
    gate2 = bool(best["found"])
    return {
        "schema_version": 1,
        "hand_side": side,
        "object": "cup" if side == "left" else "bottle",
        "template": template,
        "prior_sha256": _sha256(prior_path),
        "gate1_pass": True,
        "gate2_pass": gate2,
        "joint_gate_pass": gate2,
        "pads_star": float(best["pads"]),
        "q_star": float(best["quality"]),
        "imbalance": float(best["imbalance"]),
        "torque_normalized": float(best["torque"]),
        "object_drift_m": float(best["drift"]),
        "object_lift_m": float(best["lift"]),
        "object_tilt_deg": float(best["tilt_deg"]),
        "stable_lift_pass": gate2,
        "setting_index": int(best["setting"]),
        "step": int(best["step"]),
        "criteria": {
            "pads_min": 4.0,
            "q_min_exclusive": 0.0,
            "drift_max_m": 0.03,
            "lift_min_m": 0.01,
            "tilt_max_deg": 30.0,
        },
    }


_side_metadata(args.left_prior, "left")
_side_metadata(args.right_prior, "right")
if args.steps < 70:
    raise SystemExit("--steps must be >=70 to cover closure, lift, and hold")

cfg = PourTaskCfg()
manifest = configure_for_screening(
    cfg,
    args.scene,
    left_prior=args.left_prior,
    right_prior=args.right_prior,
)
cfg.scene.num_envs = 16
cfg.seed = args.seed
cfg.curriculum_stage = int(CurriculumStage.SINGLE_GRASP)
cfg.start_jitter_position_m = 0.0
cfg.start_jitter_rotation_deg = 0.0
cfg.start_pool_size = 64
cfg.grasp_hold_steps = args.steps + 100
cfg.phase_timeout = (20, 100, args.steps + 100, 120, 120, 100, 20)
cfg.cup_tip_fail_deg = 179.0
cfg.max_spill_fraction = 1.0
cfg.episode_length_s = max(12.0, (args.steps + 200) / 20.0)

env = PourTaskEnv(cfg)
torch.manual_seed(args.seed)
env.reset()
env.single_side[:8] = 0
env.single_side[8:] = 1
targets = _residual_targets(env.device)
lift_start = 45
lift_height_m = 0.025
arm_scale = torch.tensor(
    cfg.arm_residual_max, dtype=torch.float32, device=env.device
) * float(cfg.arm_step_scale)
lift_target = {}
for side in ("left", "right"):
    ik = env.prior[side]["ik"]
    q_grasp = env.prior[side]["q_grasp"].astype(np.float64)
    position, rotation = ik.fk(q_grasp)
    solved = ik.solve(
        position + np.array([0.0, 0.0, lift_height_m]),
        rotation,
        q0=q_grasp,
        iters=240,
        pos_tol=2.0e-4,
    )
    if not solved["ok"] or not np.isfinite(solved["q"]).all():
        raise RuntimeError(f"{side} prior has no reachable Gate-2 lift pose")
    lift_target[side] = torch.tensor(
        solved["q"], dtype=torch.float32, device=env.device
    )
initial = {
    "left": env.scene.env_origins
    + torch.tensor(
        manifest.cup.initial_pose_wxyz[:3], dtype=torch.float32, device=env.device
    ),
    "right": env.scene.env_origins
    + torch.tensor(
        manifest.bottle.initial_pose_wxyz[:3], dtype=torch.float32, device=env.device
    ),
}
best = {
    side: {
        "found": False,
        "quality": -1.0e9,
        "pads": 0.0,
        "imbalance": 1.0e9,
        "torque": 1.0e9,
        "drift": 1.0e9,
        "lift": -1.0e9,
        "tilt_deg": 1.0e9,
        "setting": -1,
        "step": -1,
    }
    for side in ("left", "right")
}

for step in range(args.steps):
    action = torch.zeros(16, 26, device=env.device)
    for side_index, offset in ((0, 0), (1, 13)):
        rows = slice(side_index * 8, (side_index + 1) * 8)
        current = env.finger_delta[rows, side_index]
        action[rows, offset + 8 : offset + 13] = (
            (targets - current) / cfg.finger_delta_rate
        ).clamp(-1.0, 1.0)
        if step >= lift_start:
            side = "left" if side_index == 0 else "right"
            arm_error = lift_target[side] - env.arm_target[side][rows]
            action[rows, offset : offset + 7] = (
                arm_error / arm_scale
            ).clamp(-1.0, 1.0)
    env.step(action)
    if step < lift_start + 10:
        continue

    geometry = env._geometry()
    for side, rows, asset_name in (
        ("left", slice(0, 8), "cup"),
        ("right", slice(8, 16), "bottle"),
    ):
        asset = env.cup if side == "left" else env.bottle
        quality, pads, imbalance, torque, _ = _quality(
            env, side, asset.data.root_pos_w
        )
        displacement = asset.data.root_pos_w - initial[side]
        lift = displacement[:, 2]
        expected = torch.tensor(
            [0.0, 0.0, lift_height_m],
            dtype=torch.float32,
            device=env.device,
        )
        drift = (displacement - expected).norm(dim=1)
        tilt = torch.rad2deg(geometry[f"{asset_name}_tilt"])
        valid = (
            (pads >= 4.0)
            & (quality > 0.0)
            & (drift <= 0.03)
            & (lift >= 0.01)
            & (tilt <= 30.0)
        )
        local_valid = valid[rows]
        if not bool(local_valid.any()):
            continue
        local_quality = quality[rows].masked_fill(~local_valid, float("-inf"))
        index = int(local_quality.argmax().item())
        value = float(local_quality[index].item())
        if value <= best[side]["quality"]:
            continue
        absolute = index if side == "left" else index + 8
        best[side] = {
            "found": True,
            "quality": value,
            "pads": float(pads[absolute].item()),
            "imbalance": float(imbalance[absolute].item()),
            "torque": float(torque[absolute].item()),
            "drift": float(drift[absolute].item()),
            "lift": float(lift[absolute].item()),
            "tilt_deg": float(tilt[absolute].item()),
            "setting": index,
            "step": step,
        }

output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
reports = {
    "left": _record_for_side(
        "left", template=args.left_template, prior_path=args.left_prior, best=best["left"]
    ),
    "right": _record_for_side(
        "right", template=args.right_template, prior_path=args.right_prior, best=best["right"]
    ),
}
for side, report in reports.items():
    (output / f"{side}_screen.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
summary = {
    "demo_id": manifest.demo_id,
    "seed": args.seed,
    "steps": args.steps,
    "left_gate2_pass": reports["left"]["gate2_pass"],
    "right_gate2_pass": reports["right"]["gate2_pass"],
}
(output / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(f"[pour-screen] {summary} output={output}")
env.close()
app.close()
