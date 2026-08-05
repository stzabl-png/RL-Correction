"""Physics smoke test for a staged static reconstruction.

The video wrist trajectory is used only to audit the object placement.  The
DexMate arms stay at the configured default pose; this entry point does not
replay the human trajectory or solve IK from it.

Example:
  PYTHONPATH=. python -m tasks.recon_kailang.static_reconstruction.smoke --headless \
      --clip task1_static_smoke --steps 80
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="task1_static_smoke")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--report", default="", help="Optional JSON report path.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("static_reconstruction_smoke")
app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from rl_rebuild.correction import clips, frames as F  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402


def _geometry_metrics(env: DexmateCorrectionEnv, vertices: np.ndarray) -> dict:
    """Return env-local AABB and table-gap measurements for env zero."""

    origin = env.scene.env_origins[0].detach().cpu().numpy()
    pos = env.object.data.root_pos_w[0].detach().cpu().numpy() - origin
    quat = env.object.data.root_quat_w[0].detach().cpu().numpy()
    world = F.rot_apply(np.repeat(quat[None], len(vertices), axis=0), vertices) + pos
    lo, hi = world.min(axis=0), world.max(axis=0)
    return {
        "pose_wxyz": np.concatenate([pos, quat]).tolist(),
        "aabb_min_m": lo.tolist(),
        "aabb_max_m": hi.tolist(),
        "aabb_center_m": (0.5 * (lo + hi)).tolist(),
        "bottom_gap_mm": float((lo[2] - env.cfg.table_top_z) * 1000.0),
        "table_margin_mm": float(
            (min(env.cfg.table_size[0], env.cfg.table_size[1]) / 2.0
             - max(abs(lo[0]), abs(hi[0]), abs(lo[1]), abs(hi[1]))) * 1000.0
        ),
    }


def _usd_physics_report(env: DexmateCorrectionEnv) -> dict:
    """Inspect the instantiated object, including its bound physics material."""

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Object")
    collision_prims = []
    approximations = []
    material_paths = set()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(str(prim.GetPath()))
        approximation = prim.GetAttribute("physics:approximation")
        if approximation and approximation.HasAuthoredValueOpinion():
            approximations.append(str(approximation.Get()))
        binding = prim.GetRelationship("material:binding:physics")
        if binding:
            material_paths.update(str(path) for path in binding.GetTargets())

    materials = []
    for path in sorted(material_paths):
        prim = stage.GetPrimAtPath(path)
        api = UsdPhysics.MaterialAPI(prim)
        materials.append({
            "path": path,
            "static_friction": api.GetStaticFrictionAttr().Get(),
            "dynamic_friction": api.GetDynamicFrictionAttr().Get(),
            "restitution": api.GetRestitutionAttr().Get(),
        })

    masses = env.object.root_physx_view.get_masses().detach().cpu().numpy().reshape(-1)
    inertias = env.object.root_physx_view.get_inertias().detach().cpu().numpy()[0].reshape(-1)
    inertia_diagonal = (
        inertias[[0, 4, 8]] if len(inertias) >= 9 else inertias[:3]
    )
    disable_gravity = root.GetAttribute("physxRigidBody:disableGravity")
    return {
        "rigid_body": bool(root.HasAPI(UsdPhysics.RigidBodyAPI)),
        "collision_prim_count": len(collision_prims),
        "collision_approximations": sorted(set(approximations)),
        "mass_kg": float(masses[0]),
        "inertia_diagonal": inertia_diagonal.tolist(),
        "disable_gravity": bool(disable_gravity.Get()) if disable_gravity else False,
        "materials": materials,
    }


def main() -> int:
    entry = clips.clip_entry(args.clip)
    if entry.get("source") != "static_reconstruction":
        raise ValueError(f"{args.clip!r} is not a static_reconstruction clip")
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("--num_envs and --steps must be positive")

    cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(cfg, args.clip)
    # AppLauncher and PhysX tensor views must use the same CUDA device.
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.rsi_prob = 0.0
    cfg.grasp_only = False
    cfg.show_dexmate = False
    cfg.show_human_traj = False

    env = DexmateCorrectionEnv(cfg)
    failures: list[str] = []
    try:
        obs, _ = env.reset()
        vertices = F.load_obj_verts(entry["mesh"])
        initial = _geometry_metrics(env, vertices)
        physics = _usd_physics_report(env)

        interaction_frame = int(env.du.ref.interaction_seg[0])
        wrist_xy = np.asarray(env.du.ref.track_wrist[interaction_frame, :2])
        center_xy = np.asarray(initial["aabb_center_m"][:2])
        placement_error_mm = float(np.linalg.norm(center_xy - wrist_xy) * 1000.0)

        initial_joint = env.hand.data.joint_pos[0].detach().clone()
        zero = torch.zeros(args.num_envs, cfg.action_space, device=env.device)
        nan_steps = 0
        done_steps = 0
        max_object_speed = 0.0
        for _ in range(args.steps):
            obs, _, terminated, truncated, _ = env.step(zero)
            nan_steps += int(any(torch.isnan(value).any().item() for value in obs.values()))
            done_steps += int((terminated | truncated).any().item())
            speed = env.object.data.root_lin_vel_w.norm(dim=1).max().item()
            max_object_speed = max(max_object_speed, float(speed))

        final = _geometry_metrics(env, vertices)
        final_speed = float(env.object.data.root_lin_vel_w[0].norm().item())
        robot_joint_drift = float(
            (env.hand.data.joint_pos[0] - initial_joint).abs().max().item()
        )
        xy_drift_mm = float(
            np.linalg.norm(
                np.asarray(final["aabb_center_m"][:2])
                - np.asarray(initial["aabb_center_m"][:2])
            ) * 1000.0
        )

        expected_mass = float(entry["semantics"].mass_kg)
        expected_friction = float(entry["semantics"].friction)
        material_friction = [
            material["static_friction"] for material in physics["materials"]
            if material["static_friction"] is not None
        ]
        checks = {
            "observations_finite": nan_steps == 0,
            "no_early_reset": done_steps == 0,
            "placement_xy_within_1mm": placement_error_mm < 1.0,
            "initial_gap_2mm_within_1mm": abs(initial["bottom_gap_mm"] - 2.0) <= 1.0,
            "final_table_contact": -3.0 <= final["bottom_gap_mm"] <= 3.0,
            "inside_table": final["table_margin_mm"] >= 0.0,
            "mass_matches_semantics": abs(physics["mass_kg"] - expected_mass) <= 1e-4,
            "positive_inertia": all(value > 0.0 for value in physics["inertia_diagonal"]),
            "rigid_body_enabled": physics["rigid_body"],
            "collision_enabled": physics["collision_prim_count"] > 0,
            "gravity_enabled": not physics["disable_gravity"],
            "friction_matches_semantics": any(
                abs(float(value) - expected_friction) <= 1e-4
                for value in material_friction
            ),
            "no_explosive_motion": max_object_speed < 1.0,
            "object_settled": final_speed < 0.05,
            "robot_default_pose_stable": robot_joint_drift < 0.1,
        }
        failures.extend(name for name, passed in checks.items() if not passed)
        report = {
            "clip": args.clip,
            "steps": args.steps,
            "interaction": {
                "hand": clips.interact_hand(args.clip),
                "aligned_frame": interaction_frame,
                "wrist_xy_m": wrist_xy.tolist(),
                "placement_error_mm": placement_error_mm,
            },
            "initial": initial,
            "final": final,
            "dynamics": {
                "xy_drift_mm": xy_drift_mm,
                "max_object_speed_mps": max_object_speed,
                "final_object_speed_mps": final_speed,
                "robot_max_joint_drift_rad": robot_joint_drift,
                "nan_steps": nan_steps,
                "done_steps": done_steps,
            },
            "physics": physics,
            "checks": checks,
            "passed": not failures,
        }
        payload = json.dumps(report, indent=2, ensure_ascii=False)
        print("\n[static-smoke-report]\n" + payload, flush=True)
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload + "\n", encoding="utf-8")
            print(f"[static-smoke] report: {report_path}", flush=True)
        if failures:
            print(
                "[static-smoke] FAIL: " + ", ".join(failures),
                file=sys.stderr, flush=True,
            )
            env.close()
            # Isaac shutdown can replace an uncaught exception's status with
            # zero. Exit explicitly so automation cannot accept a failed audit.
            os._exit(1)
        print("[static-smoke] PASS", flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
