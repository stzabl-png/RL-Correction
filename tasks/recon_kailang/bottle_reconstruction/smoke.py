"""Physics smoke test for the staged two-part water-bottle scene."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="water_bottle_twist_static")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--report", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("bottle_reconstruction_smoke")
app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from rl_rebuild.correction import clips, frames as F  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.env import (  # noqa: E402
    BottleReconstructionEnv,
)


def _geometry(env, asset, mesh_path: str) -> dict:
    vertices = F.load_obj_verts(mesh_path)
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    pos = asset.data.root_pos_w[0].detach().cpu().numpy() - origin
    quat = asset.data.root_quat_w[0].detach().cpu().numpy()
    world = F.rot_apply(np.repeat(quat[None], len(vertices), axis=0), vertices) + pos
    lo, hi = world.min(axis=0), world.max(axis=0)
    half = min(env.cfg.table_size[0], env.cfg.table_size[1]) / 2.0
    return {
        "pose_wxyz": np.concatenate([pos, quat]).tolist(),
        "aabb_min_m": lo.tolist(),
        "aabb_max_m": hi.tolist(),
        "aabb_center_m": (0.5 * (lo + hi)).tolist(),
        "bottom_gap_mm": float((lo[2] - env.cfg.table_top_z) * 1000.0),
        "table_margin_mm": float(
            (half - max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1])))
            * 1000.0
        ),
    }


def _physics(asset, prim_path: str) -> dict:
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    collision_count = 0
    material_paths = set()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_count += 1
        binding = prim.GetRelationship("material:binding:physics")
        if binding:
            material_paths.update(str(path) for path in binding.GetTargets())
    frictions = []
    for path in sorted(material_paths):
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(path))
        value = material.GetStaticFrictionAttr().Get()
        if value is not None:
            frictions.append(float(value))
    masses = asset.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
    inertias = asset.root_physx_view.get_inertias().detach().cpu().numpy()[0].reshape(-1)
    diagonal = inertias[[0, 4, 8]] if len(inertias) >= 9 else inertias[:3]
    gravity = root.GetAttribute("physxRigidBody:disableGravity")
    return {
        "rigid_body": bool(root.HasAPI(UsdPhysics.RigidBodyAPI)),
        "collision_prim_count": collision_count,
        "mass_kg": float(masses[0]),
        "inertia_diagonal": diagonal.tolist(),
        "disable_gravity": bool(gravity.Get()) if gravity else False,
        "static_frictions": frictions,
    }


def _placement_error(center_xy, trajectory, source_frame: int, source_length: int):
    aligned = 0 if source_length == 1 else int(round(
        source_frame * (len(trajectory) - 1) / (source_length - 1)
    ))
    wrist_xy = np.asarray(trajectory[aligned, :2], dtype=np.float64)
    error_mm = float(np.linalg.norm(np.asarray(center_xy) - wrist_xy) * 1000.0)
    return aligned, wrist_xy.tolist(), error_mm


def main() -> int:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    entry = clips.clip_entry(args.clip)
    secondary = entry.get("secondary")
    if not secondary:
        raise ValueError(f"{args.clip!r} is not a two-part bottle clip")
    clips.ensure_mesh_usd(
        secondary["mesh"], secondary["usd"], secondary["semantics"]
    )

    cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.rsi_prob = 0.0
    cfg.grasp_only = False
    cfg.show_dexmate = False
    cfg.show_human_traj = False

    env = BottleReconstructionEnv(cfg)
    failures: list[str] = []
    try:
        env.reset()
        initial_body = _geometry(env, env.object, entry["mesh"])
        initial_cap = _geometry(env, env.cap, secondary["mesh"])
        physics_body = _physics(env.object, "/World/envs/env_0/Object")
        physics_cap = _physics(env.cap, "/World/envs/env_0/Cap")
        source_length = int(np.load(entry["npz"], allow_pickle=True)["frames"].size)
        body_frame, body_wrist, body_error = _placement_error(
            initial_body["aabb_center_m"][:2], env.du.ref.track_wrist,
            entry["placement_frame"], source_length,
        )
        cap_frame, cap_wrist, cap_error = _placement_error(
            initial_cap["aabb_center_m"][:2], env.cap_ref.ref.track_wrist,
            secondary["placement_frame"], source_length,
        )

        robot_initial = env.hand.data.joint_pos[0].detach().clone()
        zero = torch.zeros(1, cfg.action_space, device=env.device)
        max_speed = {"body": 0.0, "cap": 0.0}
        nan_steps = done_steps = 0
        for _ in range(args.steps):
            obs, _, terminated, truncated, _ = env.step(zero)
            nan_steps += int(any(torch.isnan(value).any().item() for value in obs.values()))
            done_steps += int((terminated | truncated).any().item())
            max_speed["body"] = max(
                max_speed["body"], float(env.object.data.root_lin_vel_w.norm(dim=1).max())
            )
            max_speed["cap"] = max(
                max_speed["cap"], float(env.cap.data.root_lin_vel_w.norm(dim=1).max())
            )

        final_body = _geometry(env, env.object, entry["mesh"])
        final_cap = _geometry(env, env.cap, secondary["mesh"])
        final_speed = {
            "body": float(env.object.data.root_lin_vel_w[0].norm()),
            "cap": float(env.cap.data.root_lin_vel_w[0].norm()),
        }
        robot_drift = float(
            (env.hand.data.joint_pos[0] - robot_initial).abs().max()
        )

        def asset_checks(name, initial, final, physics, expected_mass, expected_mu):
            return {
                f"{name}_placement_xy_within_1mm": (
                    body_error if name == "body" else cap_error) < 1.0,
                f"{name}_initial_gap_2mm_within_1mm": abs(initial["bottom_gap_mm"] - 2.0) <= 1.0,
                f"{name}_final_table_contact": -3.0 <= final["bottom_gap_mm"] <= 3.0,
                f"{name}_inside_table": final["table_margin_mm"] >= 0.0,
                f"{name}_mass_matches": abs(physics["mass_kg"] - expected_mass) <= 1e-4,
                f"{name}_positive_inertia": all(value > 0.0 for value in physics["inertia_diagonal"]),
                f"{name}_rigid_body": physics["rigid_body"],
                f"{name}_collision": physics["collision_prim_count"] > 0,
                f"{name}_gravity": not physics["disable_gravity"],
                f"{name}_friction_matches": any(
                    abs(value - expected_mu) <= 1e-4
                    for value in physics["static_frictions"]
                ),
                f"{name}_no_explosive_motion": max_speed[name] < 1.0,
                f"{name}_settled": final_speed[name] < 0.05,
            }

        checks = {
            "observations_finite": nan_steps == 0,
            "no_early_reset": done_steps == 0,
            "robot_default_pose_stable": robot_drift < 0.1,
        }
        checks.update(asset_checks(
            "body", initial_body, final_body, physics_body,
            entry["semantics"].mass_kg, entry["semantics"].friction,
        ))
        checks.update(asset_checks(
            "cap", initial_cap, final_cap, physics_cap,
            secondary["semantics"].mass_kg, secondary["semantics"].friction,
        ))
        failures.extend(name for name, passed in checks.items() if not passed)
        report = {
            "clip": args.clip,
            "steps": args.steps,
            "placement": {
                "body": {
                    "interaction_source_frame": 15,
                    "placement_source_frame": entry["placement_frame"],
                    "placement_aligned_frame": body_frame,
                    "wrist_xy_m": body_wrist,
                    "error_mm": body_error,
                },
                "cap": {
                    "interaction_source_frame": 13,
                    "placement_source_frame": secondary["placement_frame"],
                    "placement_aligned_frame": cap_frame,
                    "wrist_xy_m": cap_wrist,
                    "error_mm": cap_error,
                },
            },
            "initial": {"body": initial_body, "cap": initial_cap},
            "final": {"body": final_body, "cap": final_cap},
            "physics": {"body": physics_body, "cap": physics_cap},
            "dynamics": {
                "max_speed_mps": max_speed,
                "final_speed_mps": final_speed,
                "robot_max_joint_drift_rad": robot_drift,
                "nan_steps": nan_steps,
                "done_steps": done_steps,
            },
            "checks": checks,
            "passed": not failures,
        }
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        print("\n[bottle-smoke-report]\n" + payload, flush=True)
        if args.report:
            path = Path(args.report).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        if failures:
            print("[bottle-smoke] FAIL: " + ", ".join(failures), file=sys.stderr)
            env.close()
            os._exit(1)
        print("[bottle-smoke] PASS", flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
