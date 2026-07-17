#!/usr/bin/env python3
"""Visualize a synthesized floating-hand grasp pose in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir.grasp_synthesis.assets import ASSET_ROOT, DEFAULT_SHARPA_WAVE_RIGHT_CONFIG, load_sharpa_wave_right
from ocir.grasp_synthesis.object_surface import ObjectSurface

# The persistent server hot-reloads THIS module per job, but plain
# from-imports would silently rebind against the stale helper modules left
# in sys.modules (e.g. an old write_video kept encoding mp4v video long
# after the H.264 change landed on disk). Refresh the shared helpers first,
# before any from-import below -- or in the task modules reloaded by
# register_sim_tasks -- binds names from them.
import importlib as _importlib

from ocir.isaac import replay_dexycb as _replay_dexycb_module
from ocir.isaac import sim_cli as _sim_cli_module

_importlib.reload(_replay_dexycb_module)
_importlib.reload(_sim_cli_module)

from ocir.isaac.replay_dexycb import camera_pose_to_isaac, load_dexycb_frame_mapper, quat_to_matrix, read_label
from ocir.sim.control_client import request_json, server_is_running, submit_job
from ocir.sim.isaac_server import DEFAULT_OCIR_DATA_ROOT, launch_simulation_app


DEFAULT_ISAACSIM_MODE = os.environ.get("OCIR_ISAACSIM_MODE", "webrtc").lower()
DEFAULT_OUT_DIR = DEFAULT_OCIR_DATA_ROOT / "testing/grasp_synthesis/isaac_visualization"
DEFAULT_MANIFEST = DEFAULT_OCIR_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json"
SERVER_ONLY_FIELDS = {
    "mode",
    "reuse_instance",
    "livestream",
    "hold_open",
    "hold_open_seconds",
    "hold_open_until_closed",
    "control_host",
    "control_port",
    "request_timeout",
    "status",
    "shutdown_server",
}
OCIR_ARTIFACT_DIRS = {"raw_data", "processed_data", "testing"}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_GRASP_ISAAC {message}", flush=True)


def resolve_artifact_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in OCIR_ARTIFACT_DIRS:
        return DEFAULT_OCIR_DATA_ROOT / path
    return path


def load_grasp_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    action = np.asarray(data.get("action"), dtype=float)
    if action.shape[0] < 29:
        raise ValueError(f"{path} does not contain a 29D floating-hand grasp action")
    return data


def normalize_np(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def axis_angle_to_matrix_np(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize_np(np.asarray(axis, dtype=float).reshape(1, 3))[0]
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = normalize_np(np.asarray(quat, dtype=float).reshape(1, 4))[0]
    w, x, y, z = quat
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def action_root_transform_np(action: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, 3] = np.asarray(action[:3], dtype=float)
    out[:3, :3] = quat_wxyz_to_matrix_np(np.asarray(action[3:7], dtype=float))
    return out


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def origin_matrix(elem) -> np.ndarray:
    out = np.eye(4, dtype=float)
    origin = elem.find("origin")
    if origin is not None:
        if origin.attrib.get("xyz"):
            out[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=float)
        if origin.attrib.get("rpy"):
            out[:3, :3] = rpy_to_matrix(np.fromstring(origin.attrib["rpy"], sep=" ", dtype=float))
    return out


def link_transforms_from_urdf(urdf_path: Path, base_link: str, joint_order: list[str], qpos: np.ndarray) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    q_map = {name: float(qpos[i]) for i, name in enumerate(joint_order)}
    children = {}
    for elem in root.findall("joint"):
        parent = elem.find("parent").attrib["link"]
        child = elem.find("child").attrib["link"]
        axis = elem.find("axis")
        axis_xyz = np.fromstring(axis.attrib.get("xyz", "0 0 1"), sep=" ", dtype=float) if axis is not None else np.asarray([0.0, 0.0, 1.0])
        children.setdefault(parent, []).append(
            (
                child,
                elem.attrib.get("type", "fixed"),
                origin_matrix(elem),
                axis_xyz,
                elem.attrib["name"],
            )
        )

    transforms = {base_link: np.eye(4, dtype=float)}
    stack = [base_link]
    while stack:
        parent = stack.pop()
        for child, joint_type, origin, axis, joint_name in children.get(parent, []):
            motion = np.eye(4, dtype=float)
            if joint_type in {"revolute", "continuous"}:
                motion[:3, :3] = axis_angle_to_matrix_np(axis, q_map.get(joint_name, 0.0))
            transforms[child] = transforms[parent] @ origin @ motion
            stack.append(child)
    return transforms


def find_object_mesh(surface: ObjectSurface, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    return surface.object_mesh_path if surface.object_mesh_path.exists() else None


def sequence_by_id(manifest: dict, sequence_id: str) -> dict:
    for sequence in manifest.get("sequences", []):
        if sequence.get("sequence_id") == sequence_id:
            return sequence
    raise KeyError(f"sequence_id not found in manifest: {sequence_id}")


def infer_sequence_id(surface: ObjectSurface, requested: str | None) -> str | None:
    if requested:
        return requested
    value = surface.metadata.get("sequence_id")
    if value is None:
        return None
    return str(value)


def object_transform_from_raw_data(surface: ObjectSurface, args: argparse.Namespace, points_object: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return raw-data world object transform and lifted object points.

    The grasp optimizer records hand root poses in the canonical YCB object
    frame. DexYCB `pose_y` also maps canonical YCB object coordinates into a
    camera frame. We reuse the replay pipeline to map that camera pose into the
    Isaac world, then lift the object so its lowest point sits on the tabletop.
    """

    fallback = {
        "mode": "canonical_identity_fallback",
        "reason": None,
        "sequence_id": None,
        "frame_id": None,
        "object_index": None,
        "lift_z": None,
    }
    if not args.use_raw_object_pose:
        fallback["reason"] = "--no-use-raw-object-pose"
    elif args.manifest is None or not Path(args.manifest).exists():
        fallback["reason"] = f"manifest missing: {args.manifest}"
    else:
        sequence_id = infer_sequence_id(surface, args.sequence_id)
        if sequence_id is None:
            fallback["reason"] = "no sequence_id in surface artifact and --sequence-id was not provided"
        else:
            manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            sequence = sequence_by_id(manifest, sequence_id)
            frame_ids = [int(frame_id) for frame_id in sequence["frame_ids"]]
            frame_id = int(args.frame_id) if args.frame_id is not None else frame_ids[0]
            object_index = int(surface.metadata.get("object_index", sequence.get("ycb_grasp_ind", 0) or 0))
            frame_mapper = load_dexycb_frame_mapper(manifest, sequence)
            label = read_label(sequence, frame_id)
            pose_y = np.asarray(label["pose_y"], dtype=float)[object_index]
            pos, quat = camera_pose_to_isaac(pose_y, z_offset=0.0, frame_mapper=frame_mapper)
            rot = quat_to_matrix(quat)
            raw_points = points_object @ rot.T + pos[None, :]
            lift_z = float(args.tabletop_z) - float(raw_points[:, 2].min())
            pos = pos + np.array([0.0, 0.0, lift_z], dtype=float)
            points_world = points_object @ rot.T + pos[None, :]
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = rot
            transform[:3, 3] = pos
            report = {
                "mode": "dexycb_raw_pose_y",
                "manifest": str(Path(args.manifest)),
                "sequence_id": sequence_id,
                "frame_id": int(frame_id),
                "object_index": int(object_index),
                "object_name": str(surface.metadata.get("object_name", sequence["ycb_models"][object_index].get("name"))),
                "canonical_camera": sequence.get("canonical_camera"),
                "frame_mapper": frame_mapper.report(),
                "raw_position_before_lift": np.asarray(pos - np.array([0.0, 0.0, lift_z], dtype=float), dtype=float).tolist(),
                "quaternion_wxyz": np.asarray(quat, dtype=float).tolist(),
                "lift_z": lift_z,
                "tabletop_z": float(args.tabletop_z),
            }
            return transform, points_world, report

    transform = np.eye(4, dtype=float)
    object_offset = np.array([0.0, 0.0, float(args.tabletop_z) - float(points_object[:, 2].min())], dtype=float)
    transform[:3, 3] = object_offset
    points_world = points_object + object_offset[None, :]
    fallback["lift_z"] = float(object_offset[2])
    fallback["tabletop_z"] = float(args.tabletop_z)
    return transform, points_world, fallback


def add_material(stage, path: str, color: tuple[float, float, float], roughness: float = 0.55):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def set_prim_visibility(stage, prim_path: str, visible: bool) -> None:
    """Toggle a prim's (and its descendants') render visibility.

    Used to keep an overlay (e.g. the anchored-BODex ghost hand or affordance
    heatmap) present in the interactive/exported scene while excluding it
    from the saved screenshot photos -- USD visibility is inherited, so
    hiding a group's root Xform hides everything under it in one call.
    """

    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


def _vt_vec3f_array(values: np.ndarray):
    """(N,3) numpy -> Vt.Vec3fArray, zero-copy when the USD build supports it.

    The per-element ``Gf.Vec3f`` Python loop this replaces dominated
    visualization time (the Sharpa hand's visual meshes alone are ~414k
    vertices per rendered hand)."""

    from pxr import Gf, Vt

    values = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1, 3))
    from_numpy = getattr(Vt.Vec3fArray, "FromNumpy", None)
    if from_numpy is not None:
        return from_numpy(values)
    return Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in values])


def define_mesh(stage, path: str, vertices: np.ndarray, faces: np.ndarray, material) -> dict:
    from pxr import UsdGeom, Vt

    vertices = np.asarray(vertices, dtype=float)
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(_vt_vec3f_array(vertices))
    from_numpy = getattr(Vt.IntArray, "FromNumpy", None)
    if from_numpy is not None:
        mesh.CreateFaceVertexCountsAttr(from_numpy(np.full((faces.shape[0],), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(from_numpy(faces.reshape(-1)))
    else:
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).astype(int).tolist()))
    mesh.CreateDoubleSidedAttr(True)
    bind_material(mesh.GetPrim(), material)
    return {"path": path, "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])}


def define_points(stage, path: str, points: np.ndarray, colors: np.ndarray, width: float) -> dict:
    from pxr import UsdGeom, Vt

    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=float)
    prim = UsdGeom.Points.Define(stage, path)
    prim.CreatePointsAttr(_vt_vec3f_array(points))
    from_numpy = getattr(Vt.FloatArray, "FromNumpy", None)
    if from_numpy is not None:
        prim.CreateWidthsAttr(from_numpy(np.full((points.shape[0],), float(width), dtype=np.float32)))
    else:
        prim.CreateWidthsAttr(Vt.FloatArray([float(width)] * points.shape[0]))
    prim.CreateDisplayColorAttr(_vt_vec3f_array(colors))
    prim.GetDisplayColorAttr().SetMetadata("interpolation", "vertex")
    return {"path": path, "points": int(points.shape[0]), "width": float(width)}


def create_table(stage, bbox_min: np.ndarray, bbox_max: np.ndarray, tabletop_z: float, margin: float) -> dict:
    from pxr import Gf, UsdGeom

    material = add_material(stage, "/World/Materials/Table", (0.45, 0.42, 0.36))
    center_xy = (bbox_min[:2] + bbox_max[:2]) / 2.0
    size_xy = np.maximum(bbox_max[:2] - bbox_min[:2] + 2.0 * float(margin), np.asarray([0.35, 0.35], dtype=float))
    thickness = 0.04
    cube = UsdGeom.Cube.Define(stage, "/World/Table")
    cube.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(tabletop_z - thickness / 2.0)))
    cube.AddScaleOp().Set(Gf.Vec3f(float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0, thickness / 2.0))
    bind_material(cube.GetPrim(), material)
    return {"path": "/World/Table", "center_xy": center_xy.astype(float).tolist(), "size_xy": size_xy.astype(float).tolist(), "top_z": float(tabletop_z)}


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
    UsdGeom.Xform.Define(stage, "/World/Object")
    UsdGeom.Xform.Define(stage, "/World/SharpaHand")
    return stage


#: Three mutually-orthogonal camera view directions (unit vectors, world/
#: object frame) with a compatible "up" vector each -- a straight-down "top"
#: view needs a different up than the two horizontal views, since up can't be
#: parallel to the view direction -- `set_camera_view` (the Isaac Sim
#: viewport helper used for all repositioning, see `point_camera`) assumes a
#: fixed world-up of (0, 0, 1) and would hit that degenerate case for an
#: exactly vertical "top" view, so its direction carries a tiny (~1 degree)
#: tilt instead of the true (0, 0, 1); visually indistinguishable from
#: straight-down but keeps the look-at well-defined.
def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    return v / np.linalg.norm(v)


ORTHOGONAL_VIEWS: tuple[tuple[str, np.ndarray], ...] = (
    ("front", _unit([0.0, -1.0, 0.0])),
    ("side", _unit([1.0, 0.0, 0.0])),
    ("top", _unit([0.0, 0.02, 1.0])),
)


def compute_focus_bbox(
    primary_points: np.ndarray, secondary_points: np.ndarray, max_secondary_ratio: float = 1.6
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box to frame the camera on: the object ("primary") extended to
    include the hand ("secondary") only if that doesn't blow the shot out.

    A hand from a poorly-converged grasp can end up far from the object; if
    naively fitting the camera to the full scene, that single outlier would
    force the camera to zoom out until the object is a speck. If the combined
    bbox would be more than ``max_secondary_ratio`` times the object's own
    extent, frame on the object alone (modestly padded) instead.
    """

    obj_min, obj_max = primary_points.min(axis=0), primary_points.max(axis=0)
    if secondary_points is None or secondary_points.size == 0:
        return obj_min, obj_max
    obj_extent = float(np.max(np.maximum(obj_max - obj_min, 1e-6)))
    combined = np.concatenate([primary_points, secondary_points], axis=0)
    comb_min, comb_max = combined.min(axis=0), combined.max(axis=0)
    comb_extent = float(np.max(np.maximum(comb_max - comb_min, 1e-6)))
    if comb_extent <= max_secondary_ratio * obj_extent:
        return comb_min, comb_max
    pad = 0.15 * obj_extent
    return obj_min - pad, obj_max + pad


def build_camera(stage, app, args: argparse.Namespace, bbox_min: np.ndarray, bbox_max: np.ndarray):
    """Create the (single, reused) record camera and return the (target,
    radius) its orthogonal views are framed around.

    Positioning it is left entirely to ``point_camera``/`set_camera_view`
    (see there): ``Camera.initialize()`` sets up its own translate/orient
    xform ops on this prim, so hand-rolling a raw ``Gf.Matrix4d`` transform op
    here would either go stale or collide with those.
    """

    from isaacsim.sensors.camera import Camera
    from pxr import Gf, UsdGeom

    center = (bbox_min + bbox_max) / 2.0
    extent = np.maximum(bbox_max - bbox_min, 0.05)
    radius = max(float(np.max(extent)), 0.30)
    target = center + np.asarray(args.camera_target_offset, dtype=float)

    prim_path = "/World/RecordCamera"
    cam = UsdGeom.Camera.Define(stage, prim_path)
    cam.CreateFocalLengthAttr(float(args.camera_focal_length))
    cam.CreateHorizontalApertureAttr(float(args.camera_horizontal_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera = Camera(prim_path=prim_path, name="grasp_visualization_camera", resolution=(int(args.width), int(args.height)))
    camera.initialize()
    return camera, prim_path, target, radius


def point_camera(app, camera_prim_path: str, target: np.ndarray, eye: np.ndarray) -> None:
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=eye, target=target, camera_prim_path=camera_prim_path)
    for _ in range(10):
        app.update()


def capture_camera_rgb(app, camera) -> np.ndarray:
    rgb = None
    for _ in range(45):
        app.update()
        rgb = camera.get_rgb(device="cpu")
        if rgb is not None:
            break
    if rgb is None:
        raise RuntimeError("camera returned no RGB")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def capture_camera_png(app, camera, path: Path) -> None:
    import cv2

    rgb = capture_camera_rgb(app, camera)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def capture_orthogonal_composite(
    app, camera, camera_prim_path: str, target: np.ndarray, radius: float, args: argparse.Namespace, out_dir: Path, stem: str
) -> tuple[Path, dict]:
    """Render the 3 orthogonal views and lay them out as one 1-row x 3-column image."""

    import cv2

    distance = radius * float(args.camera_distance_scale)
    panels = []
    view_reports = {}
    for idx, (name, direction) in enumerate(ORTHOGONAL_VIEWS):
        eye = target + direction * distance
        point_camera(app, camera_prim_path, target, eye)
        bgr = cv2.cvtColor(capture_camera_rgb(app, camera), cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if idx < len(ORTHOGONAL_VIEWS) - 1:
            bgr = cv2.copyMakeBorder(bgr, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panels.append(bgr)
        view_reports[name] = {"eye": eye.astype(float).tolist()}
    composite = np.concatenate(panels, axis=1)
    path = out_dir / f"{stem}.png"
    cv2.imwrite(str(path), composite)
    camera_report = {
        "target": target.astype(float).tolist(),
        "radius": float(radius),
        "distance_scale": float(args.camera_distance_scale),
        "layout": "1x3_front_side_top",
        "views": view_reports,
    }
    return path, camera_report


def visualize_grasp(app, args: argparse.Namespace, progress=None) -> dict:
    import omni.timeline
    import trimesh
    from pxr import UsdLux

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    grasp = load_grasp_json(Path(args.grasp_json))
    surface = ObjectSurface.from_sequence_dir(args.sequence_dir)
    asset = load_sharpa_wave_right(args.asset_config)
    action = np.asarray(grasp["action"], dtype=float)
    qpos = action[7:]
    if qpos.shape[0] != len(asset.config["joint_order"]):
        raise ValueError(f"grasp has {qpos.shape[0]} joints but Sharpa asset expects {len(asset.config['joint_order'])}")
    limits = asset.config["joint_limits"]
    lower = np.asarray([limits[name][0] for name in asset.config["joint_order"]], dtype=float)
    upper = np.asarray([limits[name][1] for name in asset.config["joint_order"]], dtype=float)
    qpos = np.clip(qpos, lower, upper)
    action = action.copy()
    action[7:] = qpos

    object_scale = float(grasp.get("object_scale", 1.0))
    if object_scale <= 0:
        raise ValueError(f"grasp_json object_scale must be positive, got {object_scale}")
    points_object = np.asarray(surface.points_object_frame, dtype=float) * object_scale
    world_from_object, points_world, object_pose_report = object_transform_from_raw_data(surface, args, points_object)
    stage = setup_stage(app)
    UsdLux.DistantLight.Define(stage, "/World/KeyLight").CreateIntensityAttr(450.0)
    UsdLux.DomeLight.Define(stage, "/World/FillLight").CreateIntensityAttr(180.0)

    object_material = add_material(stage, "/World/Materials/Object", (0.95, 0.72, 0.16))
    object_mesh_path = find_object_mesh(surface, args.object_mesh)
    object_report = {"mesh": None}
    object_mesh_vertices_world = points_world
    if object_mesh_path is not None:
        loaded = trimesh.load(str(object_mesh_path), force="mesh", process=False)
        vertices_object = np.asarray(loaded.vertices, dtype=float) * object_scale
        object_mesh_vertices_world = vertices_object @ world_from_object[:3, :3].T + world_from_object[:3, 3][None, :]
        faces = np.asarray(loaded.faces, dtype=np.int64)
        object_report = define_mesh(stage, "/World/Object/Mesh", object_mesh_vertices_world, faces, object_material)
        object_report["mesh"] = str(object_mesh_path)
        object_report["scale"] = object_scale

    cloud_report = {"enabled": False}
    if args.show_object_points:
        point_color = np.tile(np.asarray([0.25, 0.55, 0.85]), (points_world.shape[0], 1))
        cloud_report = define_points(
            stage,
            "/World/Object/SurfacePoints",
            points_world,
            point_color,
            float(args.point_width),
        )
        cloud_report["enabled"] = True

    hand_material = add_material(stage, "/World/Materials/SharpaHand", (0.82, 0.84, 0.88))
    elastomer_material = add_material(stage, "/World/Materials/SharpaElastomer", (0.08, 0.36, 0.85))
    # The synthesized grasp action is expressed in the canonical object frame,
    # so compose raw/replay object pose first, then the optimized hand root.
    root_tf = world_from_object @ action_root_transform_np(action)
    link_tfs = link_transforms_from_urdf(asset.urdf_path, asset.config.get("base_link", "right_hand_C_MC"), list(asset.config["joint_order"]), qpos)
    root = ET.parse(asset.urdf_path).getroot()
    hand_mesh_reports = []
    for link_elem in root.findall("link"):
        link_name = link_elem.attrib["name"]
        link_tf = link_tfs.get(link_name)
        if link_tf is None:
            continue
        for visual_idx, visual in enumerate(link_elem.findall("visual")):
            mesh_elem = visual.find("geometry/mesh")
            if mesh_elem is None or not mesh_elem.attrib.get("filename"):
                continue
            filename = mesh_elem.attrib["filename"]
            if filename.startswith("package://"):
                package_path = filename[len("package://") :]
                package, _, rest = package_path.partition("/")
                mesh_path = asset.urdf_path.parent.parent / package / rest
            else:
                mesh_path = asset.urdf_path.parent / filename
            if not mesh_path.exists():
                continue
            loaded = trimesh.load_mesh(str(mesh_path), process=False)
            vertices = np.asarray(loaded.vertices, dtype=float)
            if not vertices.size:
                continue
            if mesh_elem.attrib.get("scale"):
                vertices = vertices * np.fromstring(mesh_elem.attrib["scale"], sep=" ", dtype=float)
            tf = root_tf @ link_tf @ origin_matrix(visual)
            vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=float)], axis=1)
            world_vertices = (tf @ vertices_h.T).T[:, :3]
            faces = np.asarray(loaded.faces, dtype=np.int64)
            material = elastomer_material if "elastomer" in mesh_path.name.lower() else hand_material
            safe_link = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in link_name)
            hand_mesh_reports.append(
                define_mesh(stage, f"/World/SharpaHand/{safe_link}_{visual_idx}", world_vertices, faces, material)
            )

    object_points = np.concatenate([points_world, object_mesh_vertices_world], axis=0)
    hand_points_list = []
    for prim_path in [report["path"] for report in hand_mesh_reports]:
        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute("points")
        if attr:
            hand_points_list.append(np.asarray([[p[0], p[1], p[2]] for p in attr.Get()], dtype=float))
    hand_points = np.concatenate(hand_points_list, axis=0) if hand_points_list else np.zeros((0, 3))

    bbox_min, bbox_max = compute_focus_bbox(object_points, hand_points, float(args.camera_focus_max_ratio))
    if args.show_table:
        table_report = create_table(stage, bbox_min, bbox_max, float(args.tabletop_z), float(args.table_margin))
    else:
        table_report = {"enabled": False, "path": None, "top_z": float(args.tabletop_z)}
    camera, camera_prim_path, cam_target, cam_radius = build_camera(stage, app, args, bbox_min, bbox_max)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        app.update()

    screenshot, camera_report = capture_orthogonal_composite(
        app, camera, camera_prim_path, cam_target, cam_radius, args, out_dir, "isaac_grasp"
    )
    stage_path = out_dir / "scene.usd"
    stage.GetRootLayer().Export(str(stage_path.resolve()))

    if args.hold_open:
        if progress is not None:
            progress(f"grasp visualization ready; holding viewport for {args.hold_open_seconds}s")
        deadline = time.monotonic() + float(args.hold_open_seconds)
        while time.monotonic() < deadline:
            app.update()
            time.sleep(1.0 / 60.0)

    report = {
        "ok": True,
        "task": "grasp_pose_visualization",
        "out_dir": str(out_dir),
        "grasp_json": str(args.grasp_json),
        "sequence_dir": str(args.sequence_dir),
        "asset_config": str(asset.config_path),
        "stage": str(stage_path),
        "screenshot": str(screenshot),
        "object": object_report,
        "object_pose": object_pose_report,
        "object_scale": object_scale,
        "object_points": cloud_report,
        "hand_mesh_count": len(hand_mesh_reports),
        "table": table_report,
        "camera": camera_report,
        "tabletop_z": float(args.tabletop_z),
        "world_from_object": world_from_object.astype(float).tolist(),
        "score": float(grasp.get("score", np.nan)),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["webrtc", "local"], default=DEFAULT_ISAACSIM_MODE)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only used with --mode local. Defaults to a real on-screen window, since local mode's "
        "purpose is a standalone launch on a headed machine (no persistent Isaac server involved).",
    )
    parser.add_argument("--stream-ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--frame-id", type=int, default=None)
    parser.add_argument("--use-raw-object-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grasp-json", type=Path, default=None)
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_SHARPA_WAVE_RIGHT_CONFIG)
    parser.add_argument("--object-mesh", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tabletop-z", type=float, default=0.8)
    parser.add_argument("--table-margin", type=float, default=0.18)
    parser.add_argument("--show-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-object-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point-width", type=float, default=0.004)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--camera-distance-scale",
        type=float,
        default=1.5,
        help="Camera distance from the focus target, as a multiple of the scene's bounding radius. Applied "
        "identically to all 3 orthogonal views (closer than the pre-multi-view single-oblique-shot default).",
    )
    parser.add_argument(
        "--camera-focus-max-ratio",
        type=float,
        default=1.6,
        help="Cap on how far the hand may extend the object-centered framing: if including the hand would "
        "expand the bounding box beyond this multiple of the object's own extent (e.g. a stray hand pose "
        "from a failed grasp), frame on the object alone instead of zooming out to fit both.",
    )
    parser.add_argument("--camera-target-offset", type=float, nargs=3, default=[0.0, 0.0, 0.04])
    parser.add_argument("--camera-focal-length", type=float, default=45.0)
    parser.add_argument("--camera-horizontal-aperture", type=float, default=38.0)
    parser.add_argument("--hold-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hold-open-seconds", type=float, default=10.0)
    parser.add_argument(
        "--hold-open-until-closed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only used with --mode local: after the timed hold, keep the Isaac window open until it is "
        "closed manually instead of exiting. Off by default so batch standalone runs proceed to the "
        "next sequence on their own.",
    )
    parser.add_argument("--livestream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reuse-instance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--request-timeout", type=float, default=0.0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--shutdown-server", action="store_true")
    return parser


def apply_mode_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "webrtc":
        if args.livestream is False:
            parser.error("--mode webrtc cannot be combined with --no-livestream")
        if args.reuse_instance is False:
            parser.error("--mode webrtc cannot be combined with --no-reuse-instance")
        args.livestream = True
        args.reuse_instance = True
        return
    if args.mode == "local":
        if args.livestream is True:
            parser.error("--mode local cannot be combined with --livestream")
        if args.reuse_instance is True:
            parser.error("--mode local cannot be combined with --reuse-instance")
        args.livestream = False
        args.reuse_instance = False
        return
    parser.error(f"unsupported --mode {args.mode!r}; use 'webrtc' or 'local'")


def normalize_paths(args: argparse.Namespace) -> None:
    args.grasp_json = resolve_artifact_path(args.grasp_json)
    args.sequence_dir = resolve_artifact_path(args.sequence_dir)
    args.asset_config = resolve_artifact_path(args.asset_config)
    args.object_mesh = resolve_artifact_path(args.object_mesh)
    args.manifest = resolve_artifact_path(args.manifest)
    args.out_dir = resolve_artifact_path(args.out_dir)
    if args.status or args.shutdown_server:
        return
    if args.grasp_json is None:
        raise ValueError("--grasp-json is required")
    if args.sequence_dir is None:
        raise ValueError("--sequence-dir is required")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive")
    if float(args.tabletop_z) <= 0.0:
        raise ValueError("--tabletop-z must be positive")
    if float(args.point_width) <= 0.0:
        raise ValueError("--point-width must be positive")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def normalize_visualization_args(params: dict) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def submit_job_to_server(args: argparse.Namespace) -> int:
    status = request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0)
    registered = {task.get("name") for task in status.get("registered_tasks", [])}
    if "grasp_pose_visualization" not in registered:
        print(
            "OCIR_GRASP_ISAAC control server is running, but it did not register task "
            "'grasp_pose_visualization'. Restart it with scripts/isaac/visualize_grasp.py "
            "in the --task-module list.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    params = {}
    for key, value in vars(args).items():
        if key in SERVER_ONLY_FIELDS:
            continue
        params[key] = str(value) if isinstance(value, Path) else value
    code, _ = submit_job(
        host=args.control_host,
        port=args.control_port,
        task="grasp_pose_visualization",
        params=params,
        request_timeout=float(args.request_timeout),
    )
    return code


def run_visualization_task(app, args: argparse.Namespace, progress) -> dict:
    global log
    old_log = log

    def task_log(message: str) -> None:
        old_log(message)
        progress(message)

    log = task_log
    try:
        return visualize_grasp(app, args, progress=progress)
    finally:
        log = old_log


def register_sim_tasks(registry) -> None:
    registry.register(
        "grasp_pose_visualization",
        run=run_visualization_task,
        normalize=normalize_visualization_args,
        description="Visualize a synthesized floating-hand grasp pose with the object in Isaac Sim.",
    )
    # The anchored-BODex variant and both trajectory physics simulations
    # register through this module (the server's default --task-module) so a
    # hot-reloading persistent server exposes them without a restart; reload
    # keeps their source fresh across job submissions.
    import importlib

    from ocir.full_traj import simulate_full_traj
    from ocir.isaac import simulate_grasp_traj, visualize_anchored_grasp

    importlib.reload(visualize_anchored_grasp).register_anchored_task(registry)
    importlib.reload(simulate_grasp_traj).register_sim_tasks(registry)
    importlib.reload(simulate_full_traj).register_sim_tasks(registry)


def main() -> int:
    args = parse_args()
    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_GRASP_ISAAC no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_GRASP_ISAAC no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.mode == "local":
        # The timed hold (--hold-open / --hold-open-seconds) happens inside
        # visualize_grasp; afterwards the app closes so batch standalone runs
        # continue to the next sequence, unless --hold-open-until-closed asks
        # to keep the window alive for manual inspection.
        app = launch_simulation_app(args)
        try:
            visualize_grasp(app, args)
            if args.hold_open_until_closed:
                log("holding Isaac window open until it is closed manually (--hold-open-until-closed)")
                while app.is_running():
                    app.update()
                    time.sleep(1.0 / 60.0)
        finally:
            app.close()
        return 0
    if server_is_running(args.control_host, args.control_port):
        return submit_job_to_server(args)
    print(
        "OCIR_GRASP_ISAAC no OCIR Isaac control server is running at "
        f"{args.control_host}:{args.control_port}. Start the persistent Isaac Sim server first.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
