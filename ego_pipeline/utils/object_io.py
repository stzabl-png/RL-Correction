"""
Object-branch IO helpers (the `io_standard` adapter).

Bridges the in-memory EgoContext (K.npy / depth.npz / video frames) to the
on-disk layouts that the reused third-party tools expect:

  - SAM3D `generate_mesh_sam3d.py`  : an RGB png + a masks_dir of binary masks
  - FoundationPose `YcbineoatReader` : <scene>/{rgb,depth,masks}/*.png + cam_K.txt

No second copy of the dataset is kept on disk beyond these scratch scene dirs.
Ported from V2AP `batch_obj_pose_ego.py` (scene builder) + HV2RD STEP_5
`select_frame.py` (best-frame heuristic).
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import label, distance_transform_edt

# EgoHOS-seeded SAM2 keeps instances separate: 1=left-hand obj, 2=right-hand obj,
# 3=both-hands obj. Object foreground for mesh/pose is any id > 0 (the union).
INSTANCE_VALUES = (1, 2, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Frames: ego_pipeline's ViPEStage discards RGB, so re-extract from the video.
# Returns (N, H, W, 3) uint8 RGB, index-aligned with ctx.depth/ctx.poses.
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.stack(frames)


# ─────────────────────────────────────────────────────────────────────────────
# Best-frame selection for single-image SAM3D reconstruction.
# Ported VERBATIM-in-spirit from HV2RD STEP_5 `select_frame.select_best_frame`
# (the improved mean±band heuristic — NOT the old "pick largest overall" one,
#  which select_frame's docstring warns biases toward over-segmented masks):
#   1. drop the extreme top/bottom `trim` fraction of areas as outliers,
#   2. take the mean area of the rest (a clean, typical mask),
#   3. keep frames whose area is within mean*(1±band),
#   4. among those prefer a single-blob (frag>=frag_thresh), non-border mask;
#      pick the largest area within that typical band.
# masks: (N, H, W) bool/uint8 stack, or None entries allowed.
# Returns the chosen frame index.
# ─────────────────────────────────────────────────────────────────────────────
def pick_best_frame(masks, trim: float = 0.10, band: float = 0.15,
                    frag_thresh: float = 0.90) -> int:
    rows = []  # {idx, area, frag, border}
    for i, m in enumerate(masks):
        if m is None:
            continue
        mb = np.asarray(m) > 0
        area = int(mb.sum())
        if area == 0:
            continue
        lab, n = label(mb)
        largest = max((lab == k).sum() for k in range(1, n + 1)) if n else 0
        frag = largest / area if area else 0.0
        border = bool(mb[0].any() or mb[-1].any() or mb[:, 0].any() or mb[:, -1].any())
        rows.append({"idx": i, "area": area, "frag": frag, "border": border})

    if not rows:
        raise RuntimeError("pick_best_frame: no non-empty masks")

    areas = np.array(sorted(r["area"] for r in rows))
    lo_q, hi_q = np.quantile(areas, [trim, 1.0 - trim])   # drop extremes
    core = areas[(areas >= lo_q) & (areas <= hi_q)]
    mean = float(core.mean()) if len(core) else float(areas.mean())
    band_lo, band_hi = mean * (1 - band), mean * (1 + band)

    in_band = [r for r in rows if band_lo <= r["area"] <= band_hi] or rows
    clean = [r for r in in_band if r["frag"] >= frag_thresh and not r["border"]]
    pool = clean or in_band
    return max(pool, key=lambda r: r["area"])["idx"]


# ─────────────────────────────────────────────────────────────────────────────
# Build a YCBInEOAT-format scene dir for FoundationPose from EgoContext data.
#   frames : (N,H,W,3) uint8 RGB        (index-aligned with depth)
#   depth  : (N,H,W)  float32 meters
#   K      : (3,3) intrinsics at depth/frame resolution
#   mask0  : (H,W) binary mask for the register frame (frame 0 of the scene)
# All resized to shorter_side; K scaled accordingly. Returns scene_dir.
# ─────────────────────────────────────────────────────────────────────────────
def build_fp_scene_dir(frames, depth, K, mask0, scene_dir: str,
                       shorter_side: int = 480) -> str:
    N = min(len(frames), len(depth))
    H0, W0 = frames[0].shape[:2]
    scale = shorter_side / min(H0, W0)
    Hf, Wf = int(round(H0 * scale)), int(round(W0 * scale))

    Kf = K.astype(np.float64).copy()
    sx, sy = Wf / W0, Hf / H0
    Kf[0, 0] *= sx; Kf[0, 2] *= sx
    Kf[1, 1] *= sy; Kf[1, 2] *= sy

    for sub in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)

    for i in range(N):
        sid = f"{i:06d}"
        rgb = cv2.resize(frames[i], (Wf, Hf))
        cv2.imwrite(os.path.join(scene_dir, "rgb", f"{sid}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d = cv2.resize(depth[i], (Wf, Hf))
        d_mm = (d * 1000).clip(0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(scene_dir, "depth", f"{sid}.png"), d_mm)

    m = cv2.resize(np.asarray(mask0).astype(np.uint8), (Wf, Hf),
                   interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(scene_dir, "masks", "000000.png"), (m > 0).astype(np.uint8) * 255)

    np.savetxt(os.path.join(scene_dir, "cam_K.txt"), Kf, fmt="%.6f")
    return scene_dir


# ─────────────────────────────────────────────────────────────────────────────
# Build a FoundationPosePP-ROS scene dir (Isaac ROS pp_tracker input format):
#   <root>/scene/{K.txt, rgb/%06d.png, depth/%06d.npy(float32 m)}
#   <root>/masks/%06d.png   (per-frame binary object mask; centroid = 2D track pt)
# Resized to shorter_side with K scaled (matches the proven hoi4d_fp res). Unlike
# build_fp_scene_dir, depth is .npy in METRES and masks are per-frame (the PP
# tracker needs a mask every frame, not just frame 0). Returns (scene, masks).
# ─────────────────────────────────────────────────────────────────────────────
def build_pp_scene_dir(frames, depth, K, masks, scene_root: str,
                       shorter_side: int = 480):
    N = min(len(frames), len(depth), len(masks))
    H0, W0 = frames[0].shape[:2]
    scale = shorter_side / min(H0, W0)
    Hf, Wf = int(round(H0 * scale)), int(round(W0 * scale))
    # Isaac ROS gxf::VideoBuffer requires EVEN width/height (odd -> node crashes).
    Hf -= Hf % 2; Wf -= Wf % 2

    Kf = np.asarray(K, np.float64).copy()
    sx, sy = Wf / W0, Hf / H0
    Kf[0, 0] *= sx; Kf[0, 2] *= sx
    Kf[1, 1] *= sy; Kf[1, 2] *= sy

    scene = os.path.join(scene_root, "scene")
    masks_dir = os.path.join(scene_root, "masks")
    rgb_d = os.path.join(scene, "rgb"); depth_d = os.path.join(scene, "depth")
    for d in (rgb_d, depth_d, masks_dir):
        os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(scene, "K.txt"), Kf, fmt="%.6f")

    for i in range(N):
        sid = f"{i:06d}"
        rgb = cv2.resize(np.asarray(frames[i]), (Wf, Hf))
        cv2.imwrite(os.path.join(rgb_d, f"{sid}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d = cv2.resize(np.asarray(depth[i]), (Wf, Hf)).astype(np.float32)
        np.save(os.path.join(depth_d, f"{sid}.npy"), d)
        m = cv2.resize((np.asarray(masks[i]) > 0).astype(np.uint8), (Wf, Hf),
                       interpolation=cv2.INTER_NEAREST) * 255
        cv2.imwrite(os.path.join(masks_dir, f"{sid}.png"), m)
    return scene, masks_dir


# ─────────────────────────────────────────────────────────────────────────────
# Write a single RGB frame + its binary object mask in the layout SAM3D
# `generate_mesh_sam3d.py --image_path <png> --masks_dir <dir>` expects.
# Returns (image_path, masks_dir).
# ─────────────────────────────────────────────────────────────────────────────
def write_sam3d_input(frame_rgb, mask, work_dir: str, obj_name: str = "object"):
    os.makedirs(work_dir, exist_ok=True)
    masks_dir = os.path.join(work_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    image_path = os.path.join(work_dir, "frame.png")
    cv2.imwrite(image_path, cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR))
    mb = (np.asarray(mask) > 0).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(masks_dir, f"{obj_name}.png"), mb)
    return image_path, masks_dir


# ─────────────────────────────────────────────────────────────────────────────
# Dump ctx frames as a contiguous 00000.jpg.. dir (the layout SAM2's
# init_state(video_path=...) and EgoHOS both consume). Returns (frame_dir, N).
# ─────────────────────────────────────────────────────────────────────────────
def dump_frames_dir(frames, frame_dir: str) -> tuple[str, int]:
    os.makedirs(frame_dir, exist_ok=True)
    n = len(frames)
    for i in range(n):
        cv2.imwrite(os.path.join(frame_dir, f"{i:05d}.jpg"),
                    cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2BGR))
    return frame_dir, n


# ─────────────────────────────────────────────────────────────────────────────
# EgoHOS obj1 label map -> instance mask (1=left obj, 2=right obj, 3=both obj).
# Ported from HV2RD STEP_3 `egohos_to_masks.convert_label_map`.
#
# Two EgoHOS output conventions exist (egohos_to_masks.py docstring resolves it):
#   RAW_OBJ1 (default): the raw obj1 pass is 4-class {0 bg,1 left,2 right,3 both}
#                       -> remap {1:1, 2:2, 3:3}.
#   MERGED_0_8        : the README's merged hands+obj viz is 0-8, objects at 3/4/5
#                       -> remap {3:1, 4:2, 5:3}.
# Pick the one matching whatever the EgoHOS inference script actually writes.
# ─────────────────────────────────────────────────────────────────────────────
EGOHOS_REMAP_RAW_OBJ1 = {1: 1, 2: 2, 3: 3}
EGOHOS_REMAP_MERGED_0_8 = {3: 1, 4: 2, 5: 3}


def egohos_label_to_instances(label_map, remap=EGOHOS_REMAP_RAW_OBJ1,
                              min_area: int = 0) -> np.ndarray:
    label_map = np.asarray(label_map)
    out = np.zeros(label_map.shape, dtype=np.uint8)
    for src, dst in remap.items():
        m = label_map == src
        if min_area and 0 < int(m.sum()) < min_area:
            continue
        out[m] = dst
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 video propagation seeded by sparse instance-encoded masks.
# Ported from HV2RD STEP_3 `egohos_sam2.py` (_read_seeds / _store / run_sequence):
# every seed frame becomes a SAM2 box-prompt conditioning frame; SAM2 propagates
# forward then backward (backward only fills frames not already covered).
#   seed_masks : list[(H,W) uint8 instance mask or None]  (sparse; EgoHOS seeds)
#   frame_dir  : contiguous 00000.jpg.. dir (from dump_frames_dir)
# Returns (N,H,W) uint8 dense instance masks (values in {0,1,2,3}).
# ─────────────────────────────────────────────────────────────────────────────
def propagate_masks_sam2(frame_dir, n_frames, seed_masks, H, W,
                         sam2_cfg, sam2_ckpt, device="cuda"):
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    # collect per-instance seed boxes from every frame where the instance appears
    seeds = {v: [] for v in INSTANCE_VALUES}
    for fi, sm in enumerate(seed_masks):
        if sm is None:
            continue
        arr = np.asarray(sm)
        for v in INSTANCE_VALUES:
            box = _mask_to_box(arr == v)
            if box is not None:
                seeds[v].append((fi, box))
    seeds = {v: lst for v, lst in seeds.items() if lst}
    if not seeds:
        return np.zeros((n_frames, H, W), dtype=np.uint8)

    predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt, device=device)
    dtype = torch.bfloat16
    dense = [np.zeros((H, W), dtype=np.uint8) for _ in range(n_frames)]

    autocast = (torch.autocast("cuda", dtype=dtype) if device == "cuda"
                else _nullctx())
    with torch.inference_mode(), autocast:
        state = predictor.init_state(video_path=frame_dir,
                                     offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        for v, dets in seeds.items():
            for (fi, box) in dets:
                predictor.add_new_points_or_box(state, frame_idx=fi, obj_id=v, box=box)
        for fi, obj_ids, logits in predictor.propagate_in_video(state):
            _store_sam2(dense, fi, obj_ids, logits, H, W)
        for fi, obj_ids, logits in predictor.propagate_in_video(state, reverse=True):
            _store_sam2(dense, fi, obj_ids, logits, H, W, fill_only=True)

    return np.stack(dense)


def _mask_to_box(mask):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _store_sam2(dense, fi, obj_ids, logits, H, W, fill_only=False):
    for k, oid in enumerate(obj_ids):
        v = int(oid)
        m = (logits[k] > 0.0).squeeze().detach().cpu().numpy()
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = m.astype(bool)
        if fill_only and dense[fi][m].any():
            continue
        dense[fi][m] = v


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ─────────────────────────────────────────────────────────────────────────────
# MANO-based LEFT/RIGHT relabel of object components (optional refinement).
# Ported from HV2RD STEP_5 `relabel_lr_with_mano.py` (project_hand_mask / relabel).
# EgoHOS L/R is noisy; MANO knows hands reliably, so discard EgoHOS's L/R and
# re-assign each object connected-component to the nearest projected hand.
# ─────────────────────────────────────────────────────────────────────────────
def project_hand_mask(verts_cam, K, H, W, dilate=15):
    """Project camera-frame MANO vertices and fill their convex hull -> mask."""
    v = np.asarray(verts_cam)
    v = v[v[:, 2] > 1e-3]
    if len(v) < 3:
        return np.zeros((H, W), bool)
    uv = (np.asarray(K) @ v.T).T
    uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
    m = np.zeros((H, W), np.uint8)
    hull = cv2.convexHull(uv)
    cv2.fillConvexPoly(m, hull, 1)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m.astype(bool)


# ─────────────────────────────────────────────────────────────────────────────
# Metric scale estimation for SAM3D meshes (unit-normalized -> metric), so
# FoundationPose can match the mesh against metric depth. Mirrors V2AP
# estimate_obj_scale (method "megasam_depth_mask"): back-project the object mask
# with real depth+K -> metric point cloud -> real diameter; scale = real/mesh.
# ─────────────────────────────────────────────────────────────────────────────
def backproject_mask(mask, depth, K):
    """Object-mask pixels + depth -> (M,3) metric camera-frame points."""
    ys, xs = np.where(np.asarray(mask) > 0)
    d = np.asarray(depth)[ys, xs]
    ok = np.isfinite(d) & (d > 0)
    xs, ys, d = xs[ok], ys[ok], d[ok]
    K = np.asarray(K)
    X = (xs - K[0, 2]) / K[0, 0] * d
    Y = (ys - K[1, 2]) / K[1, 1] * d
    return np.stack([X, Y, d], axis=1)


def pointcloud_diameter(pts, pct=(2, 98)):
    """Robust metric diameter (bbox diagonal) after trimming depth outliers."""
    if len(pts) < 10:
        return 0.0
    lo, hi = np.percentile(pts[:, 2], pct)
    pts = pts[(pts[:, 2] >= lo) & (pts[:, 2] <= hi)]
    if len(pts) < 10:
        return 0.0
    return float(np.linalg.norm(pts.max(0) - pts.min(0)))


def relabel_lr(obj_fg, Lmask, Rmask, min_area=200, touch_thr=25):
    """Assign each object component to left(1)/right(2)/both(3) by hand proximity."""
    out = np.zeros(obj_fg.shape, np.uint8)
    dL = distance_transform_edt(~Lmask) if Lmask.any() else None
    dR = distance_transform_edt(~Rmask) if Rmask.any() else None
    lab, n = label(obj_fg)
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < min_area:
            continue
        dl = dL[c].min() if dL is not None else 1e9
        dr = dR[c].min() if dR is not None else 1e9
        l_touch, r_touch = dl <= touch_thr, dr <= touch_thr
        if l_touch and r_touch:
            out[c] = 3
        elif l_touch:
            out[c] = 1
        elif r_touch:
            out[c] = 2
        else:
            out[c] = 1 if dl <= dr else 2
    return out
