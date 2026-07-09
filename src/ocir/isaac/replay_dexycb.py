#!/usr/bin/env python3
"""Replay selected DexYCB object poses and MANO mesh tracks in Isaac Sim."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir.isaac.sim_cli import apply_mode_defaults, run_sim_cli_main
from ocir.sim.isaac_server import DEFAULT_OCIR_DATA_ROOT
if "ocir.dexycb.mano_model" in sys.modules:
    importlib.reload(sys.modules["ocir.dexycb.mano_model"])
from ocir.dexycb.mano_model import load_mano_betas, load_mano_model, pose_m_to_vertices_and_joints


DEFAULT_ISAACSIM_MODE = os.environ.get("OCIR_ISAACSIM_MODE", "webrtc").lower()
DEFAULT_MANIFEST = DEFAULT_OCIR_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json"
DEFAULT_OUT_DIR = DEFAULT_OCIR_DATA_ROOT / "testing/dexycb/replay"
OCIR_ARTIFACT_DIRS = {"raw_data", "processed_data", "testing"}


CAMERA_TO_ISAAC = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_DEXYCB_REPLAY {message}", flush=True)


def resolve_artifact_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in OCIR_ARTIFACT_DIRS:
        return DEFAULT_OCIR_DATA_ROOT / path
    return path


def safe_prim_component(name: str, prefix: str = "prim") -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"{prefix}_{safe}"
    return safe


def normalize_quat_wxyz(q: Iterable[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=float)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return normalize_quat_wxyz([0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s])
    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        s = math.sqrt(max(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2], 1e-12)) * 2.0
        q = [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s]
    elif idx == 1:
        s = math.sqrt(max(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2], 1e-12)) * 2.0
        q = [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s]
    else:
        s = math.sqrt(max(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1], 1e-12)) * 2.0
        q = [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s]
    return normalize_quat_wxyz(q)


@dataclass(frozen=True)
class DexYCBFrameMapper:
    rotation: np.ndarray
    translation: np.ndarray
    scene_frame: str
    camera: str
    extrinsics: str | None = None

    def points_to_isaac(self, points: np.ndarray, z_offset: float = 0.0) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        out = points @ self.rotation.T + self.translation
        out[..., 2] += float(z_offset)
        return out

    def pose_to_isaac(self, pose_y: np.ndarray, z_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        rot_cam = np.asarray(pose_y[:, :3], dtype=float)
        pos_cam = np.asarray(pose_y[:, 3], dtype=float)
        rot = self.rotation @ rot_cam
        pos = self.rotation @ pos_cam + self.translation
        pos[2] += float(z_offset)
        return pos, matrix_to_quat_wxyz(rot)

    def report(self) -> dict:
        return {
            "scene_frame": self.scene_frame,
            "camera": self.camera,
            "extrinsics": self.extrinsics,
            "rotation": np.asarray(self.rotation, dtype=float).tolist(),
            "translation": np.asarray(self.translation, dtype=float).tolist(),
        }


def load_dexycb_frame_mapper(manifest: dict, sequence: dict) -> DexYCBFrameMapper:
    camera = str(sequence["canonical_camera"])
    meta = yaml.safe_load(Path(sequence["meta"]).read_text(encoding="utf-8"))
    extrinsics_name = str(meta["extrinsics"])
    extrinsics_path = Path(manifest["selected_root"]) / "calibration" / f"extrinsics_{extrinsics_name}" / "extrinsics.yml"
    extrinsics_data = yaml.full_load(extrinsics_path.read_text(encoding="utf-8"))["extrinsics"]
    if camera not in extrinsics_data:
        raise KeyError(f"camera {camera} missing from {extrinsics_path}")

    cam_to_rig = np.asarray(extrinsics_data[camera], dtype=float).reshape(3, 4)
    cam_rotation = cam_to_rig[:, :3]
    cam_translation = cam_to_rig[:, 3]
    if "apriltag" not in extrinsics_data:
        raise KeyError(f"apriltag transform missing from {extrinsics_path}")
    tag_to_rig = np.asarray(extrinsics_data["apriltag"], dtype=float).reshape(3, 4)
    tag_rotation = tag_to_rig[:, :3]
    tag_translation = tag_to_rig[:, 3]
    camera_to_tag_rotation = tag_rotation.T @ cam_rotation
    camera_to_tag_translation = tag_rotation.T @ (cam_translation - tag_translation)
    return DexYCBFrameMapper(
        camera_to_tag_rotation,
        camera_to_tag_translation,
        scene_frame="apriltag",
        camera=camera,
        extrinsics=extrinsics_name,
    )


def camera_pose_to_isaac(
    pose_y: np.ndarray,
    z_offset: float = 0.0,
    frame_mapper: DexYCBFrameMapper | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if frame_mapper is not None:
        return frame_mapper.pose_to_isaac(pose_y, z_offset=z_offset)
    rot_cam = np.asarray(pose_y[:, :3], dtype=float)
    pos_cam = np.asarray(pose_y[:, 3], dtype=float)
    rot = CAMERA_TO_ISAAC @ rot_cam @ CAMERA_TO_ISAAC.T
    pos = CAMERA_TO_ISAAC @ pos_cam
    pos[2] += float(z_offset)
    return pos, matrix_to_quat_wxyz(rot)


def camera_points_to_isaac(
    points: np.ndarray,
    z_offset: float = 0.0,
    frame_mapper: DexYCBFrameMapper | None = None,
) -> np.ndarray:
    if frame_mapper is not None:
        return frame_mapper.points_to_isaac(points, z_offset=z_offset)
    points = np.asarray(points, dtype=float)
    out = points @ CAMERA_TO_ISAAC.T
    out[..., 2] += float(z_offset)
    return out


def add_material(stage, path: str, color: tuple[float, float, float]):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def set_xform_pose(prim, position: Iterable[float], quat_wxyz: Iterable[float]) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    q = list(quat_wxyz)
    ops = list(xform.GetOrderedXformOps())
    translate = None
    orient = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient = op
    if translate is None:
        translate = xform.AddTranslateOp()
    if orient is None:
        orient = xform.AddOrientOp()
    translate.Set(Gf.Vec3d(*[float(v) for v in position]))
    if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))
    else:
        orient.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))


def create_mesh_prim(stage, path: str, mesh_path: Path, material, color_index: int):
    import trimesh
    from pxr import Gf, UsdGeom, Vt

    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    xform = UsdGeom.Xform.Define(stage, path)
    mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*[float(x) for x in v]) for v in vertices])
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).astype(int).tolist()))
    mesh.CreateDoubleSidedAttr(True)
    bind_material(mesh.GetPrim(), material)
    return {
        "path": path,
        "prim": xform.GetPrim(),
        "mesh": str(mesh_path),
        "num_vertices": int(vertices.shape[0]),
        "num_faces": int(faces.shape[0]),
        "vertices": vertices,
        "faces": faces,
        "color_index": color_index,
    }


def transformed_vertices(vertices: np.ndarray, pose_y: np.ndarray, frame_mapper: DexYCBFrameMapper | None = None) -> np.ndarray:
    pos, quat = camera_pose_to_isaac(pose_y, z_offset=0.0, frame_mapper=frame_mapper)
    rot = quat_to_matrix(quat)
    return vertices @ rot.T + pos


def quat_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def create_table(stage, center_xy: np.ndarray, size_xy: np.ndarray, top_z: float):
    from pxr import Gf, UsdGeom

    table_mat = add_material(stage, "/World/Materials/SupportTable", (0.45, 0.42, 0.36))
    cube = UsdGeom.Cube.Define(stage, "/World/SupportTable")
    thickness = 0.04
    cube.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(top_z - thickness / 2.0)))
    cube.AddScaleOp().Set(Gf.Vec3f(float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0, thickness / 2.0))
    bind_material(cube.GetPrim(), table_mat)
    return {"path": "/World/SupportTable", "top_z": float(top_z), "size_xy": size_xy.astype(float).tolist()}


def create_mano_mesh(stage, path: str, faces: np.ndarray):
    from pxr import Gf, UsdGeom, Vt

    mat = add_material(stage, "/World/Materials/ManoMesh", (0.78, 0.62, 0.52))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, -10.0)] * 778)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(np.asarray(faces, dtype=np.int64).reshape(-1).astype(int).tolist()))
    mesh.CreateDoubleSidedAttr(True)
    bind_material(mesh.GetPrim(), mat)
    return mesh


def update_mano_mesh(mesh, vertices_xyz: np.ndarray | None) -> None:
    from pxr import Gf

    if vertices_xyz is None:
        points = np.zeros((778, 3), dtype=float)
        points[:, 2] = -10.0
    else:
        points = np.asarray(vertices_xyz, dtype=float)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*[float(x) for x in xyz]) for xyz in points])


def load_sequence_mano(manifest: dict, sequence: dict) -> dict:
    side = str(sequence.get("mano_side", "right")).lower()
    model_key = "MANO_LEFT.pkl" if side == "left" else "MANO_RIGHT.pkl"
    model_path = manifest["mano_models"][model_key]
    calib = sequence.get("mano_calib")
    if isinstance(calib, list):
        calib_name = calib[0]
    else:
        calib_name = calib
    if not calib_name:
        raise ValueError(f"sequence {sequence['sequence_id']} has no mano_calib entry")
    selected_root = Path(manifest["selected_root"])
    betas_path = selected_root / "calibration" / f"mano_{calib_name}" / "mano.yml"
    return {
        "side": side,
        "model_path": str(model_path),
        "betas_path": str(betas_path),
        "model": load_mano_model(model_path, side=side),
        "betas": load_mano_betas(betas_path),
    }


def make_record_camera(stage, args: argparse.Namespace, center: np.ndarray, bbox_min: np.ndarray | None = None, bbox_max: np.ndarray | None = None):
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom

    eye = np.array(args.camera_eye, dtype=float)
    target = np.array(args.camera_target, dtype=float)
    if args.auto_camera:
        if bbox_min is not None and bbox_max is not None:
            bbox_center = (np.asarray(bbox_min, dtype=float) + np.asarray(bbox_max, dtype=float)) / 2.0
            bbox_extent = np.asarray(bbox_max, dtype=float) - np.asarray(bbox_min, dtype=float)
            radius = float(max(np.max(bbox_extent), 0.30))
            target = bbox_center + np.array([0.0, 0.0, 0.03])
            eye = target + np.array([0.85 * radius, -1.85 * radius, 0.85 * radius])
        else:
            target = center + np.array([0.0, 0.0, 0.08])
            eye = target + np.array([0.0, -1.4, 0.55])
    camera_path = "/World/RecordCamera"
    cam = UsdGeom.Camera.Define(stage, camera_path)
    cam.CreateFocalLengthAttr(float(args.camera_focal_length))
    cam.CreateHorizontalApertureAttr(float(args.camera_horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())
    camera = Camera(prim_path=camera_path, name="dexycb_record_camera", resolution=(args.width, args.height))
    camera.initialize()
    set_camera_view(eye=eye, target=target, camera_prim_path=camera_path)
    return camera, {"eye": eye.astype(float).tolist(), "target": target.astype(float).tolist()}


def capture_camera_png(app, camera, path: Path) -> None:
    import cv2

    rgb = None
    for _ in range(30):
        app.update()
        rgb = camera.get_rgb(device="cpu")
        if rgb is not None:
            break
    if rgb is None:
        raise RuntimeError("camera returned no RGB")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_video(frame_paths: list[Path], out_path: Path, fps: int) -> bool:
    import cv2

    if not frame_paths:
        return False
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        writer.write(image)
    writer.release()
    return True


def read_label(sequence: dict, frame_id: int) -> dict[str, np.ndarray]:
    path = Path(sequence["path"]) / sequence["canonical_camera"] / f"labels_{frame_id:06d}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def valid_joints(label: dict[str, np.ndarray]) -> np.ndarray | None:
    joints = np.asarray(label["joint_3d"][0], dtype=float)
    if not np.isfinite(joints).all() or np.any(joints < -0.5):
        return None
    return joints


def compute_scene_alignment(
    sequence: dict,
    object_infos: list[dict],
    first_label: dict,
    frame_mapper: DexYCBFrameMapper | None = None,
) -> dict:
    all_points = []
    pose_y = first_label["pose_y"]
    for idx, info in enumerate(object_infos):
        if idx >= len(pose_y):
            continue
        all_points.append(transformed_vertices(info["vertices"], pose_y[idx], frame_mapper=frame_mapper))
    if not all_points:
        return {"z_offset": 0.0, "table_top_z": 0.0, "center": [0.0, 0.75, 0.1], "table_size_xy": [0.8, 0.8]}
    pts = np.concatenate(all_points, axis=0)
    min_z = float(np.min(pts[:, 2]))
    z_offset = -min_z + 0.001
    pts[:, 2] += z_offset
    mn = np.min(pts[:, :2], axis=0)
    mx = np.max(pts[:, :2], axis=0)
    center = np.mean(pts, axis=0)
    size_xy = np.maximum(mx - mn + 0.45, np.array([0.75, 0.75]))
    return {
        "z_offset": float(z_offset),
        "table_top_z": 0.0,
        "center": center.astype(float).tolist(),
        "table_center_xy": ((mn + mx) / 2.0).astype(float).tolist(),
        "table_size_xy": size_xy.astype(float).tolist(),
        "initial_object_bbox_min": pts.min(axis=0).astype(float).tolist(),
        "initial_object_bbox_max": pts.max(axis=0).astype(float).tolist(),
    }


def setup_stage(app):
    import omni.usd
    from pxr import UsdGeom

    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Materials")
    UsdGeom.Xform.Define(stage, "/World/Objects")
    UsdGeom.Xform.Define(stage, "/World/MANO")
    return stage


def replay_sequence(app, args: argparse.Namespace, manifest: dict, sequence: dict, progress=None) -> dict:
    import omni.timeline
    from pxr import UsdLux

    seq_id = sequence["sequence_id"]
    out_dir = Path(args.out_dir).expanduser() / seq_id
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    stage = setup_stage(app)
    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(450.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(180.0)

    material_colors = [
        (0.9, 0.25, 0.2),
        (0.2, 0.55, 0.95),
        (0.2, 0.8, 0.35),
        (0.95, 0.75, 0.2),
        (0.7, 0.45, 0.9),
    ]
    materials = [add_material(stage, f"/World/Materials/Object_{i}", color) for i, color in enumerate(material_colors)]
    object_infos = []
    for idx, model in enumerate(sequence["ycb_models"]):
        mesh_path = Path(model["mesh"])
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        path = f"/World/Objects/{safe_prim_component(model['name'], prefix='object')}_{idx}"
        object_infos.append(create_mesh_prim(stage, path, mesh_path, materials[idx % len(materials)], idx))

    frame_ids = sequence["frame_ids"][: int(args.max_frames) if int(args.max_frames) > 0 else None]
    frame_mapper = load_dexycb_frame_mapper(manifest, sequence)
    first_label = read_label(sequence, frame_ids[0])
    alignment = compute_scene_alignment(sequence, object_infos, first_label, frame_mapper=frame_mapper)
    z_offset = float(alignment["z_offset"])
    table_report = create_table(
        stage,
        np.asarray(alignment.get("table_center_xy", [0.0, 0.75]), dtype=float),
        np.asarray(alignment["table_size_xy"], dtype=float),
        float(alignment["table_top_z"]),
    )
    center = np.asarray(alignment["center"], dtype=float)
    mano = load_sequence_mano(manifest, sequence)
    mano_mesh = create_mano_mesh(stage, "/World/MANO/mesh", mano["model"].faces)
    mano_root_error = None
    mano_joint_error = None
    camera_bbox_min = np.asarray(alignment.get("initial_object_bbox_min"), dtype=float)
    camera_bbox_max = np.asarray(alignment.get("initial_object_bbox_max"), dtype=float)
    for frame_id in frame_ids[: min(20, len(frame_ids))]:
        maybe_label = read_label(sequence, frame_id)
        maybe_joints = valid_joints(maybe_label)
        if maybe_joints is None:
            continue
        hand_points = camera_points_to_isaac(maybe_joints, z_offset=z_offset, frame_mapper=frame_mapper)
        camera_bbox_min = np.minimum(camera_bbox_min, hand_points.min(axis=0))
        camera_bbox_max = np.maximum(camera_bbox_max, hand_points.max(axis=0))
        try:
            _, mano_joints_cam = pose_m_to_vertices_and_joints(mano["model"], maybe_label["pose_m"][0], mano["betas"])
            joint_errors = np.linalg.norm(mano_joints_cam - maybe_joints, axis=1)
            mano_root_error = float(joint_errors[0])
            mano_joint_error = {
                "mean_m": float(np.mean(joint_errors)),
                "max_m": float(np.max(joint_errors)),
            }
        except Exception:
            mano_root_error = None
            mano_joint_error = None
        break
    camera, camera_report = make_record_camera(
        stage,
        args,
        center,
        bbox_min=camera_bbox_min,
        bbox_max=camera_bbox_max,
    )
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        app.update()

    captured: list[Path] = []
    trace = []
    for step_idx, frame_id in enumerate(frame_ids):
        if progress is not None and (step_idx == 0 or step_idx % max(1, int(args.progress_every)) == 0 or step_idx == len(frame_ids) - 1):
            progress(f"sequence {seq_id}: replay frame {step_idx + 1}/{len(frame_ids)}")
        label = read_label(sequence, frame_id)
        pose_y = label["pose_y"]
        for obj_idx, info in enumerate(object_infos):
            if obj_idx >= len(pose_y):
                continue
            pos, quat = camera_pose_to_isaac(pose_y[obj_idx], z_offset=z_offset, frame_mapper=frame_mapper)
            set_xform_pose(info["prim"], pos, quat)
        joints = valid_joints(label)
        try:
            if joints is None or not np.isfinite(label["pose_m"]).all() or np.allclose(label["pose_m"], 0.0):
                update_mano_mesh(mano_mesh, None)
            else:
                mano_vertices_cam, mano_joints_cam = pose_m_to_vertices_and_joints(mano["model"], label["pose_m"][0], mano["betas"])
                update_mano_mesh(mano_mesh, camera_points_to_isaac(mano_vertices_cam, z_offset=z_offset, frame_mapper=frame_mapper))
        except Exception as exc:
            if progress is not None:
                progress(f"sequence {seq_id}: MANO mesh update failed at frame {frame_id}: {type(exc).__name__}: {exc}")
            update_mano_mesh(mano_mesh, None)
        app.update()
        if step_idx % max(1, int(args.capture_every)) == 0 or step_idx == len(frame_ids) - 1:
            frame_path = frame_dir / f"frame_{len(captured):04d}.png"
            capture_camera_png(app, camera, frame_path)
            captured.append(frame_path)
        target_idx = sequence.get("ycb_grasp_ind", 0) or 0
        target_pos, _ = camera_pose_to_isaac(pose_y[int(target_idx)], z_offset=z_offset, frame_mapper=frame_mapper)
        trace.append(
            {
                "frame_id": int(frame_id),
                "target_object_position": target_pos.astype(float).tolist(),
                "hand_valid": joints is not None,
                "mano_pose_m_available": bool(np.isfinite(label["pose_m"]).all()),
            }
        )
        if args.step_delay > 0:
            time.sleep(float(args.step_delay))

    stage_path = out_dir / "scene.usd"
    stage.GetRootLayer().Export(str(stage_path.resolve()))
    video_path = out_dir / "replay.mp4"
    screenshot = out_dir / "screenshot.png"
    video_ok = write_video(captured, video_path, int(args.fps))
    if captured:
        import cv2

        mid = cv2.imread(str(captured[len(captured) // 2]))
        if mid is not None:
            cv2.imwrite(str(screenshot), mid)
    report = {
        "ok": True,
        "sequence_id": seq_id,
        "mode": "replay",
        "object_replayed": True,
        "object_motion_source": "dexycb_pose_y_kinematic_replay",
        "hand_replayed": True,
        "hand_visualization": "mano_mesh",
        "mano_pose_m_recorded": True,
        "full_mano_mesh_rendered": True,
        "full_mano_mesh_note": "Official MANO pickle is loaded through an OCIR NumPy LBS path matching DexYCB's MANO PCA convention: pose_m[:3] is global axis-angle, pose_m[3:48] is 45 PCA hand-pose coefficients, and pose_m[48:51] is translation.",
        "mano_side": mano["side"],
        "mano_side_source": "DexYCB sequence meta mano_sides",
        "mano_root_joint_error_m": mano_root_error,
        "mano_joint_error_m": mano_joint_error,
        "mano_model": mano["model_path"],
        "mano_betas": mano["betas"].astype(float).tolist(),
        "canonical_camera": sequence["canonical_camera"],
        "frame_mapper": frame_mapper.report(),
        "frames_total": len(frame_ids),
        "frames_captured": len(captured),
        "video_written": bool(video_ok),
        "video": str(video_path) if video_ok else None,
        "screenshot": str(screenshot) if screenshot.exists() else None,
        "stage": str(stage_path),
        "object_count": len(object_infos),
        "objects": [{k: v for k, v in info.items() if k not in {"prim", "vertices", "faces"}} for info in object_infos],
        "table": table_report,
        "alignment": alignment,
        "camera": camera_report,
        "trace": trace,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_replay_batch(app, args: argparse.Namespace, progress=None) -> dict:
    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    selected_ids = set(args.sequences or [])
    sequences = [seq for seq in manifest["sequences"] if not selected_ids or seq["sequence_id"] in selected_ids]
    if not sequences:
        raise ValueError(f"no sequences selected; requested={sorted(selected_ids)}")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for sequence in sequences:
        log(f"starting replay for {sequence['sequence_id']}")
        reports.append(replay_sequence(app, args, manifest, sequence, progress=progress))
    summary = {
        "ok": True,
        "task": "dexycb_replay",
        "manifest": str(Path(args.manifest).expanduser()),
        "out_dir": str(out_dir),
        "sequence_count": len(reports),
        "sequences": [report["sequence_id"] for report in reports],
        "videos": [report["video"] for report in reports if report.get("video")],
        "reports": [str(out_dir / report["sequence_id"] / "report.json") for report in reports],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(f"wrote summary {out_dir / 'summary.json'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["webrtc", "local"], default=DEFAULT_ISAACSIM_MODE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sequences", nargs="*", default=[])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--step-delay", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[0.0, -0.25, 0.5])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.0, 0.75, 0.1])
    parser.add_argument("--auto-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-focal-length", type=float, default=45.0)
    parser.add_argument("--camera-horizontal-aperture", type=float, default=38.0)
    parser.add_argument("--hold-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hold-open-seconds", type=float, default=0.0)
    parser.add_argument("--livestream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reuse-instance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--request-timeout", type=float, default=0.0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--shutdown-server", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def normalize_paths(args: argparse.Namespace) -> None:
    args.manifest = resolve_artifact_path(args.manifest)
    args.out_dir = resolve_artifact_path(args.out_dir)
    if int(args.capture_every) <= 0:
        raise ValueError("--capture-every must be positive")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive")


def normalize_replay_args(params: dict) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def run_replay_task(app, args: argparse.Namespace, progress) -> dict:
    global log
    old_log = log

    def task_log(message: str) -> None:
        old_log(message)
        progress(message)

    log = task_log
    try:
        return run_replay_batch(app, args, progress=progress)
    finally:
        log = old_log


def register_sim_tasks(registry) -> None:
    registry.register(
        "dexycb_replay",
        run=run_replay_task,
        normalize=normalize_replay_args,
        description="Replay selected DexYCB object poses and MANO mesh tracks.",
    )


def main() -> int:
    args = parse_args()
    return run_sim_cli_main(
        args,
        log=log,
        run_batch=run_replay_batch,
        task="dexycb_replay",
        log_prefix="OCIR_DEXYCB_REPLAY",
        restart_hint=(
            "Restart it with this task module, for example:\n"
            "  scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py "
            "--task-module scripts/isaac/replay_dexycb.py"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
