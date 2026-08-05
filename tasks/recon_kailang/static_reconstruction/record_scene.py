"""Record the Step4 scene with the video wrist trajectory overlaid.

The trajectory is visual evidence for object placement only.  It never drives
the DexMate robot in this static-reconstruction workflow.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="task1_static_smoke")
parser.add_argument("--output", required=True)
parser.add_argument("--preview", default="")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--settle_tail", type=int, default=20)
parser.add_argument("--eye", default="0.95,-1.15,1.55")
parser.add_argument("--lookat", default="0.0,0.0,0.90")
parser.add_argument("--resolution", default="1280,720")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("static_reconstruction_record")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from pxr import Gf, UsdGeom, Vt  # noqa: E402

from rl_rebuild.correction import clips, frames as F  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402


def _vec3_array(points: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(*[float(value) for value in point])
                          for point in points])


def _curve(stage, path: str, points: np.ndarray, color, width: float = 0.006):
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curve.CreatePointsAttr().Set(_vec3_array(points))
    curve.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(points)]))
    curve.CreateWidthsAttr().Set(Vt.FloatArray([width] * len(points)))
    curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    curve.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return curve


def _sphere(stage, path: str, center: np.ndarray, color, radius: float):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    translate = UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp()
    translate.Set(Gf.Vec3d(*[float(value) for value in center]))
    return translate


def _object_center(mesh_path: str, pose: np.ndarray) -> np.ndarray:
    vertices = F.load_obj_verts(mesh_path)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return F.rot_apply(pose[None, 3:7], center[None])[0] + pose[:3]


def _annotate(frame: np.ndarray, ref_frame: int, interaction_frame: int,
              xy_error_mm: float) -> np.ndarray:
    image = np.ascontiguousarray(frame[..., :3].copy())
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 112), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, image)
    lines = [
        "Step4 static reconstruction | phone on table | official DexMate asset",
        (f"video wrist frame {ref_frame:03d} | first interaction {interaction_frame:03d} "
         f"| initial wrist/object XY error {xy_error_mm:.3f} mm"),
        "CYAN=approach  ORANGE=after interaction  RED=interaction wrist  GREEN=initial object center",
        "Human wrist is a placement/visualization marker only; it does not command the robot.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(image, line, (20, 25 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> int:
    entry = clips.clip_entry(args.clip)
    if entry.get("source") != "static_reconstruction":
        raise ValueError(f"{args.clip!r} is not a static_reconstruction clip")
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

    env = DexmateCorrectionEnv(cfg, render_mode="rgb_array")
    try:
        env.reset()
        origin = env.scene.env_origins[0].detach().cpu().numpy()
        wrist = np.asarray(env.du.ref.track_wrist[:, :3], dtype=np.float64) + origin
        interaction_frame = int(env.du.ref.interaction_seg[0])
        object_center = _object_center(
            entry["mesh"], np.asarray(env.du.object_init_pose, dtype=np.float64)
        ) + origin
        wrist_contact = wrist[interaction_frame]
        xy_error_mm = float(np.linalg.norm(
            wrist_contact[:2] - object_center[:2]) * 1000.0)

        stage = omni.usd.get_context().get_stage()
        root = "/World/StaticReconstructionOverlay"
        if interaction_frame >= 1:
            _curve(stage, root + "/Approach", wrist[:interaction_frame + 1],
                   (0.0, 0.9, 1.0))
        if interaction_frame < len(wrist) - 1:
            _curve(stage, root + "/AfterInteraction", wrist[interaction_frame:],
                   (1.0, 0.45, 0.0))
        _curve(stage, root + "/XYAlignment",
               np.stack([wrist_contact, object_center]), (1.0, 1.0, 1.0), 0.003)
        _sphere(stage, root + "/InteractionWrist", wrist_contact,
                (1.0, 0.0, 0.0), 0.017)
        _sphere(stage, root + "/ObjectCenter", object_center,
                (0.0, 1.0, 0.0), 0.017)
        moving = _sphere(stage, root + "/MovingWrist", wrist[0],
                         (1.0, 1.0, 0.0), 0.013)

        zero = torch.zeros(1, cfg.action_space, device=env.device)
        for _ in range(6):
            env.render()

        frames = []
        total_steps = cfg.settle_steps + env.du.ref.L + args.settle_tail
        done_steps = 0
        for step in range(total_steps):
            ref_frame = int(np.clip(step - cfg.settle_steps, 0, env.du.ref.L - 1))
            moving.Set(Gf.Vec3d(*[float(value) for value in wrist[ref_frame]]))
            _, _, terminated, truncated, _ = env.step(zero)
            done_steps += int((terminated | truncated).any().item())
            frame = env.render()
            if frame is not None:
                frames.append(_annotate(
                    np.asarray(frame), ref_frame, interaction_frame, xy_error_mm
                ))

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
        imageio.imwrite(preview, frames[min(
            cfg.settle_steps + interaction_frame, len(frames) - 1
        )])
        report = {
            "clip": args.clip,
            "video": output,
            "preview": preview,
            "frames": len(frames),
            "fps": args.fps,
            "interaction_frame": interaction_frame,
            "initial_xy_error_mm": xy_error_mm,
            "done_steps": done_steps,
            "trajectory_drives_robot": False,
        }
        report_path = os.path.splitext(output)[0] + ".json"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print("[static-record] " + json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
