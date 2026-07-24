"""
ObjectMeshStage - object MESH via SAM 3D Objects (single-image reconstruction).

Picks the best frame from the object mask track, then runs the repo's
`generate_mesh_sam3d.py` as a subprocess. SAM3D needs `LIDRA_SKIP_INIT=1` and
`SPCONV_ALGO=native` (Blackwell sm_120) set *before* spconv import, so it is run
as a child process (still inside biv2ap) rather than imported in-process.

Consumes : ctx.objects["object_mask"]  (N,H,W) binary per-frame object mask
           ctx.frames or ctx.video_path (RGB re-extracted if needed)
Produces : ctx.objects["mesh_path"]       canonical .obj
           ctx.objects["mesh_frame_idx"]  frame used for reconstruction
           ctx.objects["mesh_mask"]       (H,W) mask of that frame (for FP register)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np

from ego_pipeline.context import EgoContext
from ego_pipeline.utils.object_io import (
    extract_frames, pick_best_frame, write_sam3d_input,
    backproject_mask, pointcloud_diameter,
)
from .base import PipelineStage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SAM3D_REPO = os.path.join(BIV2AP_DIR, "third_party", "sam-3d-objects")


class ObjectMeshStage(PipelineStage):
    def __init__(self, sam3d_repo: str = SAM3D_REPO,
                 config: str = "checkpoints/hf/pipeline.yaml",
                 seed: int = 42, hand_mesh_path: str | None = None,
                 with_mesh_postprocess: bool = True, with_texture_baking: bool = False):
        self.sam3d_repo = sam3d_repo
        self.config = config
        self.seed = seed
        self.hand_mesh_path = hand_mesh_path
        self.with_mesh_postprocess = with_mesh_postprocess
        self.with_texture_baking = with_texture_baking

    def name(self) -> str:
        return "object_mesh"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if not ctx.objects or ctx.objects.get("object_mask") is None:
            missing.append("objects['object_mask'] (run ObjectMaskStage first)")
        if not os.path.isfile(os.path.join(self.sam3d_repo, "generate_mesh_sam3d.py")):
            missing.append(f"SAM3D repo at {self.sam3d_repo}")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        masks = ctx.objects["object_mask"]
        frames = ctx.frames if ctx.frames is not None else extract_frames(ctx.video_path)

        idx = pick_best_frame(masks)
        print(f"  Best reconstruction frame: {idx}")
        mask0 = np.asarray(masks[idx])

        work_dir = os.path.join(ctx.output_dir, "objects", "sam3d_input")
        image_path, masks_dir = write_sam3d_input(frames[idx], mask0, work_dir)

        cmd = [
            sys.executable, "generate_mesh_sam3d.py",
            "--image_path", os.path.abspath(image_path),
            "--masks_dir", os.path.abspath(masks_dir),
            "--config", self.config,
            "--seed", str(self.seed),
            "--with_mesh_postprocess", str(self.with_mesh_postprocess),
            "--with_texture_baking", str(self.with_texture_baking),
        ]
        if self.hand_mesh_path:
            cmd += ["--with_hand_mesh", "--hand_mesh_path", os.path.abspath(self.hand_mesh_path)]

        env = dict(os.environ, LIDRA_SKIP_INIT="1", SPCONV_ALGO="native")
        print(f"  Running SAM3D: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=self.sam3d_repo, env=env, check=True)

        # generate_mesh_sam3d.py writes <masks_dir>/<obj>/<obj>.obj
        objs = sorted(glob.glob(os.path.join(masks_dir, "**", "*.obj"), recursive=True))
        objs = [o for o in objs if "plane" not in os.path.basename(o).lower()]
        if not objs:
            raise RuntimeError(f"SAM3D produced no .obj under {masks_dir}")
        mesh_path = objs[0]
        print(f"  Mesh -> {mesh_path}")

        ctx.objects["mesh_path"] = mesh_path
        ctx.objects["mesh_frame_idx"] = int(idx)
        ctx.objects["mesh_mask"] = mask0

        # Metric scaling: SAM3D meshes are unit-normalized. Estimate scale.json
        # from depth+mask so ObjectPoseStage (FoundationPose) gets a metric mesh.
        if ctx.depth is not None and ctx.intrinsics is not None:
            self._write_scale_json(mesh_path, mask0, ctx.depth[idx], ctx.intrinsics)
        return ctx

    def _write_scale_json(self, mesh_path, mask, depth_frame, K):
        import trimesh
        pts = backproject_mask(mask, depth_frame, K)
        d_real = pointcloud_diameter(pts)
        mesh = trimesh.load(mesh_path)
        d_mesh = float(mesh.bounding_sphere.primitive.radius * 2)
        if d_real <= 0 or d_mesh <= 0:
            print(f"    Scale: cannot estimate (d_real={d_real:.3f} d_mesh={d_mesh:.3f}) — skipping")
            return
        sf = d_real / d_mesh
        info = {"scale_factor": sf, "d_real_m": d_real, "d_mesh_raw": d_mesh,
                "n_pts": int(len(pts)), "method": "megasam_depth_mask_egopipe"}
        with open(os.path.join(os.path.dirname(mesh_path), "scale.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"    Scale: d_real={d_real:.3f}m d_mesh={d_mesh:.3f} -> ×{sf:.4f}")
