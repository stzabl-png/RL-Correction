"""
ObjectPoseStage - per-frame object pose. Two backends:

  backend="pp_ros" (default): FoundationPosePP-ROS — FoundationPose++ tracking
    (2D mask-centroid + 6D Kalman) on top of the Isaac ROS TensorRT
    FoundationPoseTrackingNode. Real-time TensorRT refine. PP-ROS is a TRACKER
    (no register), so frame-0 init pose is obtained from the in-process FP
    register (the "local" backend, one frame). The tracker runs inside the
    isaac_ros_dev_container via docker exec; scene/masks/out are staged under the
    container's mounted workspace, then poses are read back.

  backend="local": the original in-process FoundationPose (flat estimater from
    the V2AP-proven copy) doing register@frame0 + track_one per frame. Also used
    by pp_ros for the frame-0 register.

Consumes : ctx.objects["mesh_path","mesh_mask","mesh_frame_idx"],
           ctx.objects["object_mask"] (per-frame, for pp_ros 2D track),
           ctx.depth (N,H,W m), ctx.intrinsics (3,3), ctx.frames/ctx.video_path
Produces : ctx.objects["pose"]      (M,4,4) ob_in_cam from reg frame onward
           ctx.objects["pose_dir"]  dir with ob_in_cam/ (+ track_vis/ for local)
           ctx.objects["pose_reg_idx"], ["pose_backend"]
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

from ego_pipeline.context import EgoContext
from ego_pipeline.utils.object_io import (
    build_fp_scene_dir, build_pp_scene_dir, extract_frames,
)
from .base import PipelineStage
from ego_pipeline.repo_paths import FOUNDATIONPOSE_ROOT, ISAAC_ROS_WS

# V2AP-proven FoundationPose copy (flat import, verified in biv2ap Phase 0).
DEFAULT_FP_ROOT = str(FOUNDATIONPOSE_ROOT)

# FoundationPosePP-ROS / Isaac ROS runtime. The container mounts ONLY
# ISAAC_WS_HOST -> ISAAC_WS_CONT, so anything the tracker reads/writes must live
# under ISAAC_WS_HOST (host) which the container sees at ISAAC_WS_CONT.
ISAAC_WS_HOST = os.environ.get("ISAAC_ROS_WS_HOST",
                               str(ISAAC_ROS_WS))
ISAAC_WS_CONT = "/workspaces/isaac_ros-dev"
PP_CONTAINER = os.environ.get("FPPP_CONTAINER", "isaac_ros_dev_container")
PP_OVERLAY = f"{ISAAC_WS_CONT}/pp_overlay/install/setup.bash"
PP_REFINE_ENGINE = os.environ.get(
    "FPPP_REFINE_ENGINE",
    f"{ISAAC_WS_CONT}/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan")
PP_SCRATCH_HOST = os.path.join(ISAAC_WS_HOST, "biv2ap_pp")


def _h2c(host_path: str) -> str:
    """Map a host path under ISAAC_WS_HOST to its container path."""
    return os.path.abspath(host_path).replace(ISAAC_WS_HOST, ISAAC_WS_CONT, 1)


class ObjectPoseStage(PipelineStage):
    def __init__(self, backend: str = "pp_ros", fp_root: str = DEFAULT_FP_ROOT,
                 shorter_side: int = 480, est_iter: int = 5, track_iter: int = 2,
                 debug: int = 1, dt: float = 0.05, container: str = PP_CONTAINER):
        self.backend = backend
        self.fp_root = fp_root
        self.shorter_side = shorter_side
        self.est_iter = est_iter
        self.track_iter = track_iter
        self.debug = debug
        self.dt = dt
        self.container = container

    def name(self) -> str:
        return "object_pose"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if ctx.depth is None:
            missing.append("depth")
        if ctx.intrinsics is None:
            missing.append("intrinsics")
        if not ctx.objects or not ctx.objects.get("mesh_path"):
            missing.append("objects['mesh_path'] (run ObjectMeshStage first)")
        # frame-0 register always uses in-process FP (also the 'local' backend).
        if not os.path.isfile(os.path.join(self.fp_root, "estimater.py")):
            missing.append(f"FoundationPose at {self.fp_root}")
        if self.backend == "pp_ros":
            if ctx.objects is None or ctx.objects.get("object_mask") is None:
                missing.append("objects['object_mask'] (per-frame, for pp_ros 2D track)")
            r = subprocess.run(["docker", "exec", self.container, "true"],
                               capture_output=True)
            if r.returncode != 0:
                missing.append(f"Isaac ROS container '{self.container}' not reachable")
            if not os.path.isdir(ISAAC_WS_HOST):
                missing.append(f"Isaac ROS workspace mount {ISAAC_WS_HOST}")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        frames = ctx.frames if ctx.frames is not None else extract_frames(ctx.video_path)
        mask0 = ctx.objects["mesh_mask"]
        mesh_path = ctx.objects["mesh_path"]

        # Register where the mesh was reconstructed and track forward from there.
        reg_idx = int(ctx.objects.get("mesh_frame_idx", 0))
        frames_s = frames[reg_idx:]
        depth_s = ctx.depth[reg_idx:]

        scene_dir = os.path.join(ctx.output_dir, "objects", "fp_scene")
        out_dir = os.path.join(ctx.output_dir, "objects", "fp_pose")
        os.makedirs(out_dir, exist_ok=True)
        build_fp_scene_dir(frames_s, depth_s, ctx.intrinsics, mask0, scene_dir,
                           shorter_side=self.shorter_side)

        if self.backend == "local":
            poses = self._run_fp(mesh_path, scene_dir, out_dir)
        elif self.backend == "pp_ros":
            poses = self._run_pp_ros(ctx, frames_s, depth_s, mask0, mesh_path,
                                     scene_dir, out_dir, reg_idx)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        ctx.objects["pose"] = poses
        ctx.objects["pose_dir"] = out_dir
        ctx.objects["pose_reg_idx"] = reg_idx
        ctx.objects["pose_backend"] = self.backend
        print(f"  Object pose [{self.backend}]: {len(poses)} frames from "
              f"reg_idx={reg_idx} -> {out_dir}/ob_in_cam")
        return ctx

    # ── pp_ros: in-process register@frame0 -> stage PP scene -> docker tracker ──
    def _run_pp_ros(self, ctx, frames_s, depth_s, mask0, mesh_path,
                    scene_dir, out_dir, reg_idx):
        # 1) frame-0 init pose from in-process FP register (scaled metric mesh).
        P0 = self._run_fp(mesh_path, scene_dir, out_dir, register_only=True)
        print(f"    [pp_ros] frame-0 init pose from in-process FP register "
              f"(z={P0[2,3]:.3f}m)")

        # 2) stage PP scene + masks + init + metric mesh under the mounted dir.
        tag = os.path.basename(os.path.normpath(ctx.output_dir)) or "run"
        host_root = os.path.join(PP_SCRATCH_HOST, tag)
        # The container (root) writes out/ files; clean via docker so the host
        # (non-root) isn't blocked by root-owned leftovers.
        subprocess.run(["docker", "exec", self.container, "rm", "-rf", _h2c(host_root)])
        os.makedirs(host_root, exist_ok=True)
        masks_s = ctx.objects["object_mask"][reg_idx:]
        scene_h, masks_h = build_pp_scene_dir(frames_s, depth_s, ctx.intrinsics,
                                              masks_s, host_root,
                                              shorter_side=self.shorter_side)
        init_h = os.path.join(host_root, "init", "000000.txt")
        os.makedirs(os.path.dirname(init_h), exist_ok=True)
        np.savetxt(init_h, P0, fmt="%.6f")
        mesh_h = os.path.join(host_root, "mesh.obj")
        self._bake_scaled_mesh(mesh_path, mesh_h)
        pp_out_h = os.path.join(host_root, "out")

        # 3) drive the Isaac TensorRT tracker in the container.
        self._drive_pp_tracker(_h2c(mesh_h), _h2c(scene_h), _h2c(masks_h),
                               _h2c(init_h), _h2c(pp_out_h),
                               os.path.join(host_root, "launch.log"))

        # 4) read poses back; mirror into the stage out_dir.
        pose_files = sorted(glob.glob(os.path.join(pp_out_h, "ob_in_cam", "*.txt")))
        if not pose_files:
            raise RuntimeError(f"pp_tracker produced no poses under {pp_out_h}")
        os.makedirs(os.path.join(out_dir, "ob_in_cam"), exist_ok=True)
        poses = []
        for p in pose_files:
            P = np.loadtxt(p).reshape(4, 4)
            poses.append(P)
            np.savetxt(os.path.join(out_dir, "ob_in_cam", os.path.basename(p)),
                       P, fmt="%.6f")
        return np.stack(poses)

    def _bake_scaled_mesh(self, mesh_path, out_path):
        """Write a metric copy of the SAM3D mesh (scale.json applied) for the
        Isaac TensorRT node, which loads the mesh file as-is (no scaling)."""
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
        if os.path.exists(scale_json):
            sf = float(json.load(open(scale_json)).get("scale_factor", 1.0))
            if abs(sf - 1.0) > 0.01:
                mesh.vertices = mesh.vertices * sf
        mesh.export(out_path)

    def _drive_pp_tracker(self, mesh_c, scene_c, masks_c, init_c, out_c, launch_log_h):
        src = ("source /opt/ros/jazzy/setup.bash; "
               f"source {ISAAC_WS_CONT}/install/setup.bash; "
               f"source {PP_OVERLAY}; "
               "export LD_LIBRARY_PATH=$(ls -d /opt/ros/jazzy/share/*/gxf/lib "
               "2>/dev/null|tr '\\n' ':')$LD_LIBRARY_PATH")

        def dx(cmd, **kw):
            return subprocess.run(["docker", "exec", self.container, "bash", "-lc", cmd], **kw)

        dx("pkill -9 -f component_container_mt 2>/dev/null; true")
        time.sleep(1)
        # launch TensorRT node (attached background process; log streamed to host)
        launch_cmd = (f"{src}; cd {ISAAC_WS_CONT}; stdbuf -oL -eL ros2 launch "
                      f"foundationpose_pp_ros fp_pp_track.launch.py "
                      f"mesh_file_path:={mesh_c} refine_engine_file_path:={PP_REFINE_ENGINE}")
        node = subprocess.Popen(["docker", "exec", self.container, "bash", "-lc", launch_cmd],
                                stdout=open(launch_log_h, "w"), stderr=subprocess.STDOUT)
        try:
            if not self._wait_node_ready(launch_log_h, timeout=300):
                raise RuntimeError(f"PP-ROS TensorRT node failed to start; see {launch_log_h}")
            time.sleep(2)
            print("    [pp_ros] TensorRT node ready; running pp_tracker ...")
            track_cmd = (f"{src}; ros2 run foundationpose_pp_ros pp_tracker "
                         f"--scene {scene_c} --masks {masks_c} --init_pose {init_c} "
                         f"--out {out_c} --dt {self.dt}")
            dx(track_cmd, check=True)
            # container runs as root; hand results back to the host user.
            dx(f"chown -R {os.getuid()}:{os.getgid()} {os.path.dirname(out_c)} 2>/dev/null; true")
        finally:
            dx("pkill -9 -f component_container_mt 2>/dev/null; true")
            node.terminate()
            try:
                node.wait(timeout=10)
            except subprocess.TimeoutExpired:
                node.kill()

    @staticmethod
    def _wait_node_ready(log_h, timeout=300):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(log_h):
                txt = open(log_h, errors="ignore").read()
                if "Node was started" in txt:
                    return True
                if any(k in txt for k in ("Traceback", "Failed to load",
                                          "terminate called", "[ERROR]")):
                    return False
            time.sleep(2)
        return False

    # ── local backend: V2AP batch_obj_pose_ego.run_fp (register + track) ──────
    # register_only=True returns just the frame-0 pose (used to seed pp_ros).
    def _run_fp(self, mesh_path, scene_dir, out_dir, register_only=False):
        if self.fp_root not in sys.path:
            sys.path.insert(0, self.fp_root)
        import json as _json
        import trimesh
        import torch
        import imageio
        import nvdiffrast.torch as dr
        from estimater import (FoundationPose, ScorePredictor, PoseRefinePredictor,
                               set_seed, draw_posed_3d_box, draw_xyz_axis)
        from datareader import YcbineoatReader

        set_seed(0)
        scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()

        mesh = trimesh.load(mesh_path)
        if len(mesh.faces) > 5000:
            import fast_simplification
            ratio = 1.0 - min(5000 / len(mesh.faces), 0.9999)
            v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=ratio)
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

        # SAM3D meshes are ~unit-cube normalized; apply scale.json if present.
        scale_json = os.path.join(os.path.dirname(mesh_path), "scale.json")
        if os.path.exists(scale_json):
            sf = float(_json.load(open(scale_json)).get("scale_factor", 1.0))
            if abs(sf - 1.0) > 0.01:
                mesh.vertices *= sf
        else:
            d = mesh.bounding_sphere.primitive.radius * 2
            if d > 0.8:
                print(f"    No scale.json, mesh diameter={d:.3f}m — pose may be non-metric")

        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                             mesh=mesh, scorer=scorer, refiner=refiner,
                             debug_dir=out_dir, debug=self.debug, glctx=glctx)
        reader = YcbineoatReader(video_dir=scene_dir, shorter_side=None, zfar=np.inf)

        os.makedirs(os.path.join(out_dir, "ob_in_cam"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "track_vis"), exist_ok=True)

        poses = []
        n = len(reader.color_files)
        for i in range(n):
            color, depth = reader.get_color(i), reader.get_depth(i)
            if i == 0:
                mask = reader.get_mask(0).astype(bool)
                pose = est.register(K=reader.K, rgb=color, depth=depth,
                                    ob_mask=mask, iteration=self.est_iter)
                if register_only:
                    torch.cuda.empty_cache()
                    return pose.reshape(4, 4)
            else:
                pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                     iteration=self.track_iter)
            poses.append(pose.reshape(4, 4))
            np.savetxt(os.path.join(out_dir, "ob_in_cam", f"{reader.id_strs[i]}.txt"),
                       pose.reshape(4, 4), fmt="%.6f")
            if self.debug >= 1:
                cp = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=cp, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=cp, scale=0.1, K=reader.K,
                                    thickness=3, transparency=0, is_input_rgb=True)
                imageio.imwrite(os.path.join(out_dir, "track_vis", f"{reader.id_strs[i]}.png"), vis)
            torch.cuda.empty_cache()

        return np.stack(poses)
