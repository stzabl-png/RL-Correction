"""Record the reconstructed bottle/cap scene with both hand trajectories."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="water_bottle_twist_static")
parser.add_argument("--output", required=True)
parser.add_argument("--preview", default="")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--settle-tail", type=int, default=20)
parser.add_argument("--eye", default="0.75,1.15,1.55")
parser.add_argument("--lookat", default="-0.08,-0.04,0.90")
parser.add_argument("--resolution", default="1280,720")
parser.add_argument("--screw-demo", action="store_true")
parser.add_argument("--screw-torque-nm", type=float, default=0.003)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("bottle_reconstruction_record")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

from rl_rebuild.correction import clips, frames as F  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.env import (  # noqa: E402
    BottleReconstructionEnv,
)


def _vec3_array(points: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([
        Gf.Vec3f(*[float(value) for value in point]) for point in points
    ])


def _curve(stage, path: str, points: np.ndarray, color, width=0.003):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreatePointsAttr().Set(_vec3_array(points))
    curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(points)]))
    curve.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(points)))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curve.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def _sphere(stage, path: str, center: np.ndarray, color, radius: float):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    translate = UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp()
    translate.Set(Gf.Vec3d(*[float(value) for value in center]))
    return translate


def _mesh_center(mesh_path: str, pose: np.ndarray) -> np.ndarray:
    vertices = F.load_obj_verts(mesh_path)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return F.rot_apply(pose[None, 3:7], center[None])[0] + pose[:3]


def _source_to_aligned(source_frame: int, source_length: int, aligned_length: int):
    return 0 if source_length == 1 else int(round(
        source_frame * (aligned_length - 1) / (source_length - 1)
    ))


def _annotate(
        frame, aligned_frame, source_frame, source_length, body_error, cap_error,
        screw=None):
    image = np.ascontiguousarray(frame[..., :3].copy())
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 118), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    if screw is None:
        lines = [
            "Step4 bottle reconstruction | PCO-1810 body + cap | official DexMate asset",
            (f"replay {aligned_frame:03d} | source-video frame ~{source_frame:03d}/"
             f"{source_length - 1:03d} | body/cap XY error "
             f"{body_error:.3f}/{cap_error:.3f} mm"),
            "CYAN=left trajectory  MAGENTA=right trajectory  RED/GREEN=placement wrists  BLUE/YELLOW=asset centers",
            "Trajectories visualize placement evidence only; they do not command the robot in this QA run.",
        ]
    else:
        lines = [
            "Step4 Direct-GPU PCO-1810 screw mechanism | official DexMate asset",
            (f"mode={screw['mode']} | engaged={screw['engaged']} | "
             f"constraint angle={screw['turns']:.3f} turns | "
             f"axial travel={screw['travel_mm']:.3f} mm"),
            "Right-handed thread: 3.18 mm/turn, 2 constrained turns, automatic release at thread exit",
            "Applied torque is a mechanism audit only; it is not an RL policy or robot command.",
        ]
    for index, line in enumerate(lines):
        cv2.putText(
            image, line, (20, 25 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.60, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return image


def main() -> int:
    entry = clips.clip_entry(args.clip)
    secondary = entry.get("secondary")
    if not secondary:
        raise ValueError(f"{args.clip!r} is not a two-part bottle clip")
    clips.ensure_mesh_usd(
        secondary["mesh"], secondary["usd"], secondary["semantics"]
    )
    width, height = (int(value) for value in args.resolution.split(","))
    eye = tuple(float(value) for value in args.eye.split(","))
    lookat = tuple(float(value) for value in args.lookat.split(","))

    cfg = DexmateCorrectionEnvCfg()
    clips.configure_cfg(cfg, args.clip)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.rsi_prob = 0.0
    cfg.grasp_only = False
    cfg.show_dexmate = False
    cfg.show_human_traj = False
    cfg.viewer = ViewerCfg(
        eye=eye, lookat=lookat, origin_type="world", resolution=(width, height)
    )

    env = BottleReconstructionEnv(cfg, render_mode="rgb_array")
    try:
        env.reset()
        if args.screw_demo and env.screw_spec is None:
            raise ValueError("--screw-demo requires a screw-cap clip")
        if args.screw_demo and env.screw_spec.mode == "capture":
            axis_local = torch.zeros(1, 3, device=env.device)
            axis_local[:, 2] = 1.0
            from isaaclab.utils.math import quat_apply
            axis_w = quat_apply(env.object.data.root_quat_w, axis_local)
            cap_pos = env.object.data.root_pos_w + (
                env.screw_spec.closed_offset_m + env.screw_spec.travel_m
            ) * axis_w
            env.cap.write_root_pose_to_sim(torch.cat([
                cap_pos, env.object.data.root_quat_w
            ], dim=1))
            env.cap.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device)
            )
            env.apply_screw_constraint(integrate_angle=False)
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        left = np.asarray(env.du.ref.track_wrist[:, :3], dtype=np.float64) + origin
        right = np.asarray(env.cap_ref.ref.track_wrist[:, :3], dtype=np.float64) + origin
        with np.load(entry["npz"], allow_pickle=True) as replay:
            source_length = len(replay["frames"])
        body_frame = _source_to_aligned(
            entry["placement_frame"], source_length, len(left)
        )
        cap_frame = _source_to_aligned(
            secondary["placement_frame"], source_length, len(right)
        )
        body_center = _mesh_center(
            entry["mesh"], np.asarray(env.du.object_init_pose, dtype=np.float64)
        ) + origin
        cap_pose = np.concatenate([
            env.cap.data.root_pos_w[0].detach().cpu().numpy() - origin,
            env.cap.data.root_quat_w[0].detach().cpu().numpy(),
        ])
        cap_center = _mesh_center(secondary["mesh"], cap_pose) + origin
        body_error = float(np.linalg.norm(left[body_frame, :2] - body_center[:2]) * 1000)
        cap_error = float(np.linalg.norm(right[cap_frame, :2] - cap_center[:2]) * 1000)

        stage = omni.usd.get_context().get_stage()
        root = "/World/BottleReconstructionOverlay"
        moving_left = moving_right = None
        if args.screw_demo:
            # A small render-only marker makes rotation visible on the nearly
            # axisymmetric cap. It is a child of the cap and has no physics API.
            _sphere(
                stage, "/World/envs/env_0/Cap/AuditRotationMarker",
                np.array([0.015, 0.0, 0.012]), (1.0, 0.0, 0.0), 0.0025,
            )
        else:
            _curve(stage, root + "/LeftTrajectory", left, (0.0, 0.9, 1.0))
            _curve(stage, root + "/RightTrajectory", right, (1.0, 0.0, 0.8))
            _sphere(stage, root + "/BodyPlacementWrist", left[body_frame],
                    (1.0, 0.0, 0.0), 0.010)
            _sphere(stage, root + "/CapPlacementWrist", right[cap_frame],
                    (0.0, 1.0, 0.0), 0.010)
            _sphere(stage, root + "/BodyCenter", body_center,
                    (0.1, 0.3, 1.0), 0.009)
            _sphere(stage, root + "/CapCenter", cap_center,
                    (1.0, 0.8, 0.0), 0.009)
            moving_left = _sphere(stage, root + "/MovingLeft", left[0],
                                  (0.0, 1.0, 1.0), 0.007)
            moving_right = _sphere(stage, root + "/MovingRight", right[0],
                                   (1.0, 0.0, 0.8), 0.007)

        zero = torch.zeros(1, cfg.action_space, device=env.device)
        for _ in range(6):
            env.render()
        frames = []
        total = cfg.settle_steps + len(left) + args.settle_tail
        if args.screw_demo:
            total = max(total, cfg.settle_steps + 170)
        env.ep_total = max(env.ep_total, total + 2)
        done_steps = 0
        detached_tail = 0
        torque = torch.zeros(1, 3, device=env.device)
        if args.screw_demo:
            motion_sign = 1.0 if env.screw_spec.mode == "preengaged" else -1.0
            torque[:, 2] = motion_sign * args.screw_torque_nm
        for step in range(total):
            aligned = int(np.clip(step - cfg.settle_steps, 0, len(left) - 1))
            right_frame = int(np.clip(aligned, 0, len(right) - 1))
            if moving_left is not None:
                moving_left.Set(Gf.Vec3d(*[float(value) for value in left[aligned]]))
                moving_right.Set(Gf.Vec3d(*[float(value) for value in right[right_frame]]))
            if args.screw_demo and step >= cfg.settle_steps:
                env.apply_screw_constraint(extra_cap_torque_local=torque)
            _, _, terminated, truncated, _ = env.step(zero)
            done_steps += int((terminated | truncated).any().item())
            frame = env.render()
            if frame is not None:
                source_frame = int(round(
                    aligned * (source_length - 1) / max(len(left) - 1, 1)
                ))
                screw = None
                if args.screw_demo:
                    angle = float(env.screw_angle[0].item())
                    screw = {
                        "mode": env.screw_spec.mode,
                        "engaged": bool(env.screw_engaged[0].item()),
                        "turns": angle / (2.0 * np.pi),
                        "travel_mm": float(
                            env.screw_spec.direction * env.screw_spec.pitch_m
                            * angle / (2.0 * np.pi) * 1000.0
                        ),
                    }
                frames.append(_annotate(
                    np.asarray(frame), aligned, source_frame, source_length,
                    body_error, cap_error, screw,
                ))
            if (
                args.screw_demo
                and env.screw_spec.mode == "preengaged"
                and step >= cfg.settle_steps
                and not env.screw_engaged[0].item()
            ):
                detached_tail += 1
                if detached_tail >= 20:
                    break

        if not frames:
            raise RuntimeError("renderer returned no frames")
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        imageio.mimwrite(
            output, frames, fps=args.fps, codec="libx264", quality=8,
            macro_block_size=1,
        )
        preview = os.path.abspath(
            args.preview or os.path.splitext(output)[0] + "_preview.png"
        )
        preview_index = (
            min(cfg.settle_steps + 35, len(frames) - 1)
            if args.screw_demo
            else min(cfg.settle_steps + body_frame, len(frames) - 1)
        )
        imageio.imwrite(preview, frames[preview_index])
        report = {
            "clip": args.clip,
            "video": output,
            "preview": preview,
            "frames": len(frames),
            "fps": args.fps,
            "body_interaction_frame": 15,
            "body_placement_frame": entry["placement_frame"],
            "cap_interaction_frame": 13,
            "cap_placement_frame": secondary["placement_frame"],
            "body_xy_error_mm": body_error,
            "cap_xy_error_mm": None if args.screw_demo else cap_error,
            "done_steps": done_steps,
            "trajectory_drives_robot": False,
            "screw_demo": args.screw_demo,
        }
        with open(os.path.splitext(output)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print("[bottle-record] " + json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
