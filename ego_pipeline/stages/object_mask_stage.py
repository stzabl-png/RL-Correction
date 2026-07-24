"""
ObjectMaskStage - per-frame 2D mask of the manipulated object (HV2RD Step 3).

Pipeline (P1 = EgoHOS-seeded SAM2, the option chosen in DEPLOYMENT_PLAN):

  1. dump ctx frames to a contiguous 00000.jpg.. dir                  [biv2ap]
  2. EgoHOS 1st-order object inference -> per-frame obj1 label maps    [ISOLATED
     run via `conda run -n egohos ...` as a subprocess; EgoHOS is mmcv1.6/         egohos env]
     mmseg0.24/torch1.10 and will NOT run on Blackwell sm_120.
  3. convert obj1 label maps -> sparse instance seed masks            [biv2ap]
  4. SAM2 video propagation (fwd+bwd) seeds -> DENSE per-frame masks  [biv2ap]
  5. (optional) MANO-based left/right relabel of object components    [biv2ap]

Only step 2 needs the isolated egohos env; everything else runs in biv2ap.
The EgoHOS step is the single remaining external dependency: when the egohos env
+ `third_party/egohos` clone + checkpoints are not present, this stage raises a
clear error pointing at EGOHOS_AUTODL_SETUP.md. Pre-computed EgoHOS outputs can
also be supplied via `egohos_outputs_dir` to skip the subprocess entirely.

Produces : ctx.objects["object_mask"]    (N,H,W) uint8 binary union object mask
           ctx.objects["instance_mask"]  (N,H,W) uint8 1=left,2=right,3=both
"""
from __future__ import annotations

import glob
import os
import subprocess

import numpy as np
from PIL import Image

from ego_pipeline.context import EgoContext
from ego_pipeline.utils.object_io import (
    dump_frames_dir, egohos_label_to_instances, propagate_masks_sam2,
    project_hand_mask, relabel_lr, extract_frames,
)
from .base import PipelineStage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# SAM2 (already installed in biv2ap from DexImit-Open/third_party/Grounded-SAM-2).
DEFAULT_SAM2_CKPT = os.path.join(
    BIV2AP_DIR, "DexImit-Open", "ckpts", "sam2", "sam2.1_hiera_large.pt")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# EgoHOS isolated env + repo (Phase 2 / ISOLATED — created per EGOHOS_AUTODL_SETUP.md).
DEFAULT_EGOHOS_REPO = os.path.join(BIV2AP_DIR, "third_party", "egohos")
DEFAULT_EGOHOS_ENV = "egohos"


class ObjectMaskStage(PipelineStage):
    def __init__(
        self,
        sam2_ckpt: str = DEFAULT_SAM2_CKPT,
        sam2_cfg: str = DEFAULT_SAM2_CFG,
        egohos_env: str = DEFAULT_EGOHOS_ENV,
        egohos_repo: str = DEFAULT_EGOHOS_REPO,
        egohos_outputs_dir: str | None = None,
        min_area: int = 200,
        relabel_with_mano: bool = False,
    ):
        self.sam2_ckpt = sam2_ckpt
        self.sam2_cfg = sam2_cfg
        self.egohos_env = egohos_env
        self.egohos_repo = egohos_repo
        self.egohos_outputs_dir = egohos_outputs_dir
        self.min_area = min_area
        self.relabel_with_mano = relabel_with_mano

    def name(self) -> str:
        return "object_mask"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if not os.path.isfile(self.sam2_ckpt):
            missing.append(f"SAM2 ckpt at {self.sam2_ckpt}")
        # EgoHOS is the isolated dependency; allowed to be absent only if the
        # caller supplies pre-computed outputs.
        if self.egohos_outputs_dir is None and not os.path.isdir(self.egohos_repo):
            missing.append(
                f"EgoHOS repo at {self.egohos_repo} (isolated env; see "
                "DEPLOYMENT_PLAN Phase 2 / EGOHOS_AUTODL_SETUP.md), or pass "
                "egohos_outputs_dir=<precomputed obj1 label maps>")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        frames = ctx.frames if ctx.frames is not None else extract_frames(ctx.video_path)
        H, W = frames[0].shape[:2]
        work = os.path.join(ctx.output_dir, "objects")
        frame_dir, n = dump_frames_dir(frames, os.path.join(work, "rgb_all"))

        # ── Step 2 (ISOLATED): EgoHOS obj1 label maps ────────────────────────
        egohos_dir = self.egohos_outputs_dir or self._run_egohos(frame_dir, work)

        # ── Step 3: obj1 label maps -> sparse instance seeds ─────────────────
        seed_masks: list[np.ndarray | None] = [None] * n
        n_seed = 0
        for p in sorted(glob.glob(os.path.join(egohos_dir, "*.png"))):
            fi = int(os.path.splitext(os.path.basename(p))[0])
            if 0 <= fi < n:
                lab = np.asarray(Image.open(p))
                seed_masks[fi] = egohos_label_to_instances(lab, min_area=self.min_area)
                n_seed += 1
        print(f"  EgoHOS seeds: {n_seed}/{n} frames")

        # ── Step 4: SAM2 propagation -> dense instance masks ─────────────────
        dense = propagate_masks_sam2(frame_dir, n, seed_masks, H, W,
                                     self.sam2_cfg, self.sam2_ckpt)

        # ── Step 5 (optional): MANO L/R relabel ──────────────────────────────
        if self.relabel_with_mano:
            dense = self._relabel_with_mano(ctx, dense, H, W)

        ctx.objects = ctx.objects or {}
        ctx.objects["instance_mask"] = dense
        ctx.objects["object_mask"] = (dense > 0).astype(np.uint8)
        cov = int((dense > 0).any(axis=(1, 2)).sum())
        print(f"  Object mask: {cov}/{n} frames have object foreground")
        return ctx

    # ── ISOLATED seam: run EgoHOS inference in its own conda env ─────────────
    def _run_egohos(self, frame_dir: str, work: str) -> str:
        out_dir = os.path.join(work, "egohos_obj1")
        os.makedirs(out_dir, exist_ok=True)
        # The exact EgoHOS image/obj1 entry script + flags must match the cloned
        # repo (EGOHOS_AUTODL_SETUP.md §3 warns the script name varies, e.g.
        # pred_all_obj1.sh / mmseg_inference). Wrapper script keeps that contract
        # in one place so only this seam changes when the env is built.
        runner = os.path.join(self.egohos_repo, "run_obj1_infer.py")
        if not os.path.isfile(runner):
            raise RuntimeError(
                f"EgoHOS runner not found: {runner}\n"
                "  This is the ISOLATED Phase-2 step. Create the egohos env and "
                "an obj1 inference wrapper per EGOHOS_AUTODL_SETUP.md, OR pass "
                "egohos_outputs_dir=<dir of obj1 label-map PNGs> to ObjectMaskStage.")
        cmd = ["conda", "run", "-n", self.egohos_env, "python", runner,
               "--images", frame_dir, "--out", out_dir]
        print(f"  Running EgoHOS (isolated env '{self.egohos_env}'): {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return out_dir

    # ── MANO L/R relabel using camera-frame MANO vertices from ctx ──────────
    def _relabel_with_mano(self, ctx, dense, H, W):
        verts = self._mano_cam_verts(ctx)
        if verts is None:
            print("  [relabel] MANO camera-frame vertices unavailable - keeping "
                  "EgoHOS/SAM2 instance ids (object union is unaffected)")
            return dense
        K = np.asarray(ctx.intrinsics)
        left_v, right_v = verts  # each (N, V, 3) in camera frame
        out = np.zeros_like(dense)
        for i in range(len(dense)):
            fg = dense[i] > 0
            if not fg.any():
                continue
            Lm = project_hand_mask(left_v[i], K, H, W)
            Rm = project_hand_mask(right_v[i], K, H, W)
            out[i] = relabel_lr(fg, Lm, Rm, min_area=self.min_area)
        return out

    def _mano_cam_verts(self, ctx):
        """Per-frame camera-frame MANO vertices (left, right) or None.

        Not yet wired: ctx stores MANO as world-space axis-angle params, so this
        needs a MANO forward-kinematics pass + world->cam transform. Left as a
        hook because object_mask/mesh/pose only consume the union mask; L/R
        instance identity is a non-critical refinement (see relabel_lr_with_mano).
        """
        return None
