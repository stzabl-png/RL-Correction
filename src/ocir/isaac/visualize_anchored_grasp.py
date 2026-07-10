#!/usr/bin/env python3
"""Visualize an anchored-BODex grasp in Isaac Sim with its human-demo context.

Renders everything the base ``visualize_grasp`` scene has (object mesh, hand,
optional table), plus the anchored-pipeline context needed to eyeball
similarity to the human demonstration:

- object surface points colored by the sequence's affordance heatmap
  (``affordance.npz``) instead of uniform blue;
- the human MANO hand at the seed's anchor frame, as a point cloud from
  ``human_demo.npz``;
- a translucent "ghost" of the retargeted anchor pose (the grasp record's
  ``anchor_action``) so the optimized grasp can be compared against where the
  human hand was.

Both execution modes of the base script are supported and behave the same
way:

- ``--mode webrtc`` (default): submit an ``anchored_grasp_visualization`` job
  to the persistent Isaac control server. The task is registered through
  ``visualize_grasp.register_sim_tasks`` (the server's default task module),
  so a running server picks it up via hot reload without a restart.
- ``--mode local``: launch a one-shot local Isaac Sim window (pop-up
  interface for a headed machine, no server involved).

Pure-BODex records (without anchor/demo fields) degrade gracefully: the
overlays that lack data are skipped and reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir.grasp_synthesis.anchored_bodex.demo_data import (
    AFFORDANCE_FILENAME,
    DEMO_FILENAME,
    resolve_sequence_file,
)
from ocir.grasp_synthesis.assets import load_sharpa_wave_right
from ocir.grasp_synthesis.object_surface import ObjectSurface
from ocir.isaac.visualize_grasp import (
    SERVER_ONLY_FIELDS,
    action_root_transform_np,
    add_material,
    bind_material,
    apply_mode_defaults,
    build_camera,
    build_parser as build_base_parser,
    capture_orthogonal_composite,
    compute_focus_bbox,
    create_table,
    define_mesh,
    define_points,
    find_object_mesh,
    link_transforms_from_urdf,
    load_grasp_json,
    normalize_paths,
    object_transform_from_raw_data,
    origin_matrix,
    set_prim_visibility,
    setup_stage,
)
from ocir.sim.control_client import request_json, server_is_running, submit_job
from ocir.sim.isaac_server import launch_simulation_app

TASK_NAME = "anchored_grasp_visualization"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_ANCHORED_ISAAC {message}", flush=True)


def affordance_colors(values: np.ndarray) -> np.ndarray:
    """Heatmap color ramp without a matplotlib dependency: dark red for
    rarely-contacted points blending to bright yellow at peak contact."""

    v = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)[:, None]
    cold = np.asarray([0.55, 0.08, 0.05], dtype=float)[None, :]
    hot = np.asarray([1.0, 0.9, 0.1], dtype=float)[None, :]
    return (1.0 - v) * cold + v * hot


def add_translucent_material(stage, path: str, color: tuple[float, float, float], opacity: float):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def expand_to_full_joint_order(
    active_action: np.ndarray, active_joint_names: list[str], full_joint_order: list[str]
) -> np.ndarray:
    """[pos, quat, q(active order)] -> [pos, quat, q(full order)], passives 0."""

    active_action = np.asarray(active_action, dtype=float)
    full = np.zeros((7 + len(full_joint_order),), dtype=float)
    full[:7] = active_action[:7]
    index = {name: i for i, name in enumerate(active_joint_names)}
    for i, name in enumerate(full_joint_order):
        if name in index:
            full[7 + i] = active_action[7 + index[name]]
    return full


def render_hand(stage, asset, action: np.ndarray, world_from_object: np.ndarray, prim_root: str, materials: dict) -> list[dict]:
    """Pose the URDF's visual meshes at ``action`` (full joint order) under
    ``prim_root``; same mesh-resolution rules as the base visualizer."""

    import trimesh

    joint_order = list(asset.config["joint_order"])
    limits = asset.config["joint_limits"]
    lower = np.asarray([limits[name][0] for name in joint_order], dtype=float)
    upper = np.asarray([limits[name][1] for name in joint_order], dtype=float)
    qpos = np.clip(np.asarray(action[7:], dtype=float), lower, upper)
    root_tf = world_from_object @ action_root_transform_np(np.asarray(action, dtype=float))
    link_tfs = link_transforms_from_urdf(asset.urdf_path, asset.config.get("base_link", "right_hand_C_MC"), joint_order, qpos)
    root = ET.parse(asset.urdf_path).getroot()
    reports = []
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
            material = materials["elastomer"] if "elastomer" in mesh_path.name.lower() else materials["hand"]
            safe_link = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in link_name)
            reports.append(define_mesh(stage, f"{prim_root}/{safe_link}_{visual_idx}", world_vertices, faces, material))
    return reports


def _sequence_npz(sequence_dir: Path, key: str, default_name: str) -> Path | None:
    meta_path = sequence_dir / "sequence.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    try:
        return resolve_sequence_file(sequence_dir, meta, key, default_name)
    except FileNotFoundError:
        return None


def visualize_anchored_grasp(app, args: argparse.Namespace, progress=None) -> dict:
    import omni.timeline
    import trimesh
    from pxr import UsdGeom, UsdLux

    def phase(message: str) -> None:
        log(message)
        if progress is not None:
            progress(message, phase="run")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    phase("loading grasp record and sequence inputs")
    grasp = load_grasp_json(Path(args.grasp_json))
    sequence_dir = Path(args.sequence_dir)
    surface = ObjectSurface.from_sequence_dir(sequence_dir)
    asset = load_sharpa_wave_right(args.asset_config)
    action = np.asarray(grasp["action"], dtype=float)

    object_scale = float(grasp.get("object_scale", 1.0))
    points_object = np.asarray(surface.points_object_frame, dtype=float) * object_scale
    phase("resolving object world pose")
    world_from_object, points_world, object_pose_report = object_transform_from_raw_data(surface, args, points_object)

    phase("setting up USD stage")
    stage = setup_stage(app)
    UsdGeom.Xform.Define(stage, "/World/AnchorHand")
    UsdGeom.Xform.Define(stage, "/World/DemoHand")
    UsdLux.DistantLight.Define(stage, "/World/KeyLight").CreateIntensityAttr(450.0)
    UsdLux.DomeLight.Define(stage, "/World/FillLight").CreateIntensityAttr(180.0)

    phase("building object mesh")
    object_material = add_material(stage, "/World/Materials/Object", (0.95, 0.72, 0.16))
    object_mesh_path = find_object_mesh(surface, args.object_mesh)
    object_report = {"mesh": None}
    vertices = points_world
    if object_mesh_path is not None:
        loaded = trimesh.load(str(object_mesh_path), force="mesh", process=False)
        vertices_object = np.asarray(loaded.vertices, dtype=float) * object_scale
        vertices = vertices_object @ world_from_object[:3, :3].T + world_from_object[:3, 3][None, :]
        faces = np.asarray(loaded.faces, dtype=np.int64)
        object_report = define_mesh(stage, "/World/Object/Mesh", vertices, faces, object_material)
        object_report["mesh"] = str(object_mesh_path)

    # --- Affordance-colored object surface points -------------------------
    phase("building affordance point cloud")
    affordance_report = {"enabled": False, "reason": None}
    if args.show_affordance:
        aff_path = _sequence_npz(sequence_dir, "affordance", AFFORDANCE_FILENAME)
        if aff_path is None:
            affordance_report["reason"] = f"no {AFFORDANCE_FILENAME} in sequence dir"
        else:
            with np.load(aff_path, allow_pickle=False) as data:
                aff_points = np.asarray(data["points_object_frame"], dtype=float) * object_scale
                heat = np.asarray(data["heatmap"], dtype=float)
            # Only the actual contact region: a full-surface blanket of
            # zero-heat points just hides the object and the heatmap.
            contacted = heat > 0.0
            if not contacted.any():
                affordance_report["reason"] = "heatmap is all zeros"
            else:
                aff_world = (
                    aff_points[contacted] @ world_from_object[:3, :3].T + world_from_object[:3, 3][None, :]
                )
                affordance_report = define_points(
                    stage,
                    "/World/Object/AffordancePoints",
                    aff_world,
                    affordance_colors(heat[contacted]),
                    float(args.point_width),
                )
                affordance_report.update(
                    {
                        "enabled": True,
                        "source": str(aff_path),
                        "contact_points": int(contacted.sum()),
                        "max_heat": float(heat.max()),
                    }
                )

    # --- Human demo hand (MANO vertices) at the anchor frame --------------
    # Shown in both the saved photos and the live/exported scene.
    phase("building demo hand point cloud")
    demo_report = {"enabled": False, "reason": None}
    demo_points = np.zeros((0, 3))
    anchor_frame_index = grasp.get("anchor_frame_index")
    if args.demo_frame is not None:
        anchor_frame_index = int(args.demo_frame)
    if args.show_demo_hand:
        demo_path = _sequence_npz(sequence_dir, "human_demo", DEMO_FILENAME)
        if demo_path is None:
            demo_report["reason"] = f"no {DEMO_FILENAME} in sequence dir"
        elif anchor_frame_index is None:
            demo_report["reason"] = "record has no anchor_frame_index and --demo-frame not given"
        else:
            with np.load(demo_path, allow_pickle=False) as data:
                hand_vertices = np.asarray(data["hand_vertices_object"], dtype=float)
                valid = np.asarray(data["valid_mask"], dtype=bool)
            idx = int(anchor_frame_index)
            if idx < 0 or idx >= hand_vertices.shape[0] or not valid[idx]:
                demo_report["reason"] = f"demo frame index {idx} is out of range or invalid"
            else:
                verts = hand_vertices[idx] @ world_from_object[:3, :3].T + world_from_object[:3, 3][None, :]
                color = np.tile(np.asarray([0.18, 0.72, 0.32]), (verts.shape[0], 1))
                demo_report = define_points(stage, "/World/DemoHand/Vertices", verts, color, float(args.demo_point_width))
                demo_report.update({"enabled": True, "frame_index": idx, "source": str(demo_path)})
                demo_points = verts

    # --- Anchor ghost hand (retargeted human pose, pre-optimization) ------
    # Kept in the live/exported scene, but hidden from the saved photos (see
    # the visibility toggle below) -- the photos should show only the object,
    # the final grasp, and the human demo point cloud.
    phase("building anchor ghost hand")
    anchor_report = {"enabled": False, "reason": None}
    ghost_mesh_reports: list[dict] = []
    if args.show_anchor_hand:
        anchor_action = grasp.get("anchor_action")
        anchor_joint_names = grasp.get("optimized_joint_names")
        if anchor_action is None or anchor_joint_names is None:
            anchor_report["reason"] = "record has no anchor_action/optimized_joint_names (not an anchored-BODex record)"
        else:
            full_anchor = expand_to_full_joint_order(
                np.asarray(anchor_action, dtype=float), list(anchor_joint_names), list(asset.config["joint_order"])
            )
            ghost = add_translucent_material(stage, "/World/Materials/AnchorGhost", (0.25, 0.5, 0.95), float(args.anchor_opacity))
            ghost_mesh_reports = render_hand(
                stage, asset, full_anchor, world_from_object, "/World/AnchorHand",
                {"hand": ghost, "elastomer": ghost},
            )
            anchor_report = {"enabled": True, "mesh_count": len(ghost_mesh_reports), "opacity": float(args.anchor_opacity)}

    # --- Optimized grasp hand ---------------------------------------------
    phase("building optimized grasp hand")
    hand_material = add_material(stage, "/World/Materials/SharpaHand", (0.82, 0.84, 0.88))
    elastomer_material = add_material(stage, "/World/Materials/SharpaElastomer", (0.08, 0.36, 0.85))
    hand_mesh_reports = render_hand(
        stage, asset, action, world_from_object, "/World/SharpaHand",
        {"hand": hand_material, "elastomer": elastomer_material},
    )

    phase("framing camera")
    object_points = np.concatenate([points_world, vertices], axis=0)
    hand_points_list = []
    for prim_path in [report["path"] for report in hand_mesh_reports + ghost_mesh_reports]:
        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute("points")
        if attr:
            hand_points_list.append(np.asarray([[p[0], p[1], p[2]] for p in attr.Get()], dtype=float))
    if demo_points.size:
        hand_points_list.append(demo_points)
    hand_points = np.concatenate(hand_points_list, axis=0) if hand_points_list else np.zeros((0, 3))

    bbox_min, bbox_max = compute_focus_bbox(object_points, hand_points, float(args.camera_focus_max_ratio))
    if args.show_table:
        table_report = create_table(stage, bbox_min, bbox_max, float(args.tabletop_z), float(args.table_margin))
    else:
        table_report = {"enabled": False, "path": None, "top_z": float(args.tabletop_z)}
    camera, camera_prim_path, cam_target, cam_radius = build_camera(stage, app, args, bbox_min, bbox_max)

    phase("rendering and capturing screenshot")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        app.update()

    # The saved photos show only the object, the final grasp, and the human
    # demo point cloud -- the ghost hand and affordance heatmap stay in the
    # interactive/exported scene (toggleable there) but are hidden here.
    photo_hidden_prims = []
    if anchor_report.get("enabled"):
        photo_hidden_prims.append("/World/AnchorHand")
    if affordance_report.get("enabled"):
        photo_hidden_prims.append("/World/Object/AffordancePoints")
    for prim_path in photo_hidden_prims:
        set_prim_visibility(stage, prim_path, False)

    screenshot, camera_report = capture_orthogonal_composite(
        app, camera, camera_prim_path, cam_target, cam_radius, args, out_dir, "isaac_anchored_grasp"
    )

    for prim_path in photo_hidden_prims:
        set_prim_visibility(stage, prim_path, True)
    for _ in range(5):
        app.update()

    phase("exporting stage")
    stage_path = out_dir / "scene.usd"
    stage.GetRootLayer().Export(str(stage_path.resolve()))

    if args.hold_open:
        if progress is not None:
            progress(f"anchored grasp visualization ready; holding viewport for {args.hold_open_seconds}s")
        deadline = time.monotonic() + float(args.hold_open_seconds)
        while time.monotonic() < deadline:
            app.update()
            time.sleep(1.0 / 60.0)

    report = {
        "ok": True,
        "task": TASK_NAME,
        "out_dir": str(out_dir),
        "grasp_json": str(args.grasp_json),
        "sequence_dir": str(sequence_dir),
        "stage": str(stage_path),
        "screenshot": str(screenshot),
        "object": object_report,
        "object_pose": object_pose_report,
        "affordance_points": affordance_report,
        "demo_hand": demo_report,
        "anchor_hand": anchor_report,
        "hand_mesh_count": len(hand_mesh_reports),
        "table": table_report,
        "camera": camera_report,
        "world_from_object": world_from_object.astype(float).tolist(),
        "score": float(grasp.get("score", np.nan)),
        "rank_score": grasp.get("rank_score"),
        "affordance_coverage": grasp.get("affordance_coverage"),
        "pose_similarity": grasp.get("pose_similarity"),
        "anchor_frame_id": grasp.get("anchor_frame_id"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = __doc__
    parser.add_argument("--show-affordance", action=argparse.BooleanOptionalAction, default=True, help="Color object surface points by the affordance heatmap.")
    parser.add_argument("--show-demo-hand", action=argparse.BooleanOptionalAction, default=True, help="Render the human MANO hand vertices at the anchor frame.")
    parser.add_argument("--show-anchor-hand", action=argparse.BooleanOptionalAction, default=True, help="Render a translucent ghost of the retargeted anchor pose.")
    parser.add_argument("--anchor-opacity", type=float, default=0.35)
    parser.add_argument("--demo-point-width", type=float, default=0.003)
    parser.add_argument("--demo-frame", type=int, default=None, help="Override the demo frame index (defaults to the record's anchor_frame_index).")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def normalize_task_args(params: dict) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def run_task(app, args: argparse.Namespace, progress) -> dict:
    return visualize_anchored_grasp(app, args, progress=progress)


def register_anchored_task(registry) -> None:
    """Called from ``visualize_grasp.register_sim_tasks`` (the server's
    default task module), so a hot-reloading persistent server exposes this
    task without any restart or task-module-list change."""

    registry.register(
        TASK_NAME,
        run=run_task,
        normalize=normalize_task_args,
        description="Visualize an anchored-BODex grasp with affordance heatmap, human demo hand, and anchor ghost.",
    )


def submit_job_to_server(args: argparse.Namespace) -> int:
    status = request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0)
    registered = {task.get("name") for task in status.get("registered_tasks", [])}
    hot_reload = bool(status.get("hot_reload_tasks", False))
    if TASK_NAME not in registered and not hot_reload:
        print(
            f"OCIR_ANCHORED_ISAAC control server did not register task {TASK_NAME!r} and hot reload is off. "
            "Restart it (scripts/sim/start_isaacsim_server.py) with an up-to-date checkout.",
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
        task=TASK_NAME,
        params=params,
        request_timeout=float(args.request_timeout),
    )
    return code


def main() -> int:
    args = parse_args()
    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_ANCHORED_ISAAC no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_ANCHORED_ISAAC no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.mode == "local":
        # Timed hold happens inside visualize_anchored_grasp; close afterwards
        # so batch standalone runs proceed (see visualize_grasp.main).
        app = launch_simulation_app(args)
        try:
            visualize_anchored_grasp(app, args)
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
        "OCIR_ANCHORED_ISAAC no OCIR Isaac control server is running at "
        f"{args.control_host}:{args.control_port}. Start the persistent Isaac Sim server first, "
        "or pass --mode local for a one-shot pop-up window.",
        file=sys.stderr,
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
