"""Light-touch outlier cleaning for fused world trajectories (hands + object).

This is reconstruction, not a physics solver: the goal is only to keep isolated
outliers (e.g. an object flung far away for a few frames when depth momentarily
breaks, then snapping back) from dragging the overall trajectory off. We do NOT
try to recover the true pose — downstream RL correction handles fine fixes. We:

  1. detect outlier frames with robust statistics (deviation from a rolling-median
     baseline + MAD threshold), plus, for the object, a cross-modal check against
     the grasping hand (a held object should stay near a hand);
  2. replace them with information from neighbours: short gaps are interpolated
     (linear translation, slerp rotation), long gaps / sequence ends are held at
     the nearest good frame (object, which always needs a pose) or left invalid
     (hands, which carry a validity mask downstream);
  3. emit a per-frame confidence in [0,1] (1 real, 0.5 interpolated, 0.3 held,
     0 ignored) so RL knows which frames to trust.

All functions are pure numpy/scipy so this runs under any env that has scipy
(the fuse step's env does).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CONF_REAL = 1.0
CONF_INTERP = 0.5
CONF_HELD = 0.3
CONF_IGNORED = 0.0


@dataclass
class CleanConfig:
    median_window: int = 5          # rolling-median baseline window (odd)
    mad_k: float = 3.5              # outlier if residual > median + k * MAD
    min_residual_m: float = 0.03   # floor so sub-3cm jitter is never flagged
    max_interp_gap: int = 6        # runs up to this many frames are interpolated
    jump_floor_m: float = 0.20     # a step this large is a teleport, not real motion


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Per-axis rolling median over axis 0 (reflect-padded)."""
    if x.shape[0] < 3 or window < 3:
        return x.copy()
    r = window // 2
    pad = np.pad(x, ((r, r), (0, 0)), mode="reflect")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = np.median(pad[i:i + window], axis=0)
    return out


def _robust_threshold(vals: np.ndarray, k: float, floor: float) -> float:
    if vals.size == 0:
        return np.inf
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    return max(med + k * mad, med + floor, floor)


def _detect_outliers(trans: np.ndarray, valid: np.ndarray, cfg: CleanConfig) -> np.ndarray:
    """Flag frames whose position deviates from a robust local baseline."""
    T = trans.shape[0]
    flag = np.zeros(T, dtype=bool)
    if valid.sum() < 3:
        return flag
    # baseline over nearest-filled positions so a spike doesn't poison the window
    filled = trans.copy()
    good = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        filled[i] = trans[good[np.argmin(np.abs(good - i))]]
    baseline = _rolling_median(filled, cfg.median_window)
    res = np.linalg.norm(filled - baseline, axis=1)
    thr = _robust_threshold(res[valid], cfg.mad_k, cfg.min_residual_m)
    flag[valid] = res[valid] > thr
    return flag


def _excursion_outliers(trans: np.ndarray, valid: np.ndarray, cfg: CleanConfig) -> np.ndarray:
    """Flag multi-frame 'flew out and came back' excursions via velocity jump pairs.

    A transient glitch (e.g. depth breaking for several frames) shows up as a big
    step OUT then a big step BACK, with the position returning near where it left.
    A genuine move (place the object down and release) steps out but does NOT return,
    so it is left untouched. This catches plateaus the rolling-median baseline misses
    without flagging legitimate sustained motion.
    """
    T = trans.shape[0]
    flag = np.zeros(T, dtype=bool)
    if valid.sum() < 3:
        return flag
    filled = trans.copy()
    good = np.where(valid)[0]
    for i in np.where(~valid)[0]:
        filled[i] = trans[good[np.argmin(np.abs(good - i))]]
    step = np.linalg.norm(np.diff(filled, axis=0), axis=1)  # step[i] = |pos[i+1]-pos[i]|
    # excursion boundaries are teleport-sized steps only, so ordinary fast motion
    # between the out- and back-jumps does not break the pairing.
    thr = max(_robust_threshold(step, cfg.mad_k, cfg.min_residual_m), cfg.jump_floor_m)
    jumps = np.where(step > thr)[0]                          # jump between i and i+1
    for a, b in zip(jumps[:-1], jumps[1:]):
        exc = np.arange(a + 1, b + 1)                        # displaced frames
        if exc.size == 0 or exc.size > cfg.max_interp_gap:
            continue
        pre, post = filled[a], filled[b + 1] if b + 1 < T else filled[a]
        anchor_line = (pre + post) / 2.0
        dev = np.linalg.norm(filled[exc] - anchor_line, axis=1).max()
        returned = np.linalg.norm(post - pre)               # small => came back
        if dev > thr and returned < 0.5 * dev:
            flag[exc] = True
    return flag


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end) runs where mask is True."""
    runs, i, T = [], 0, mask.shape[0]
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def clean_pose_stream(
    trans: np.ndarray,
    rot: np.ndarray | None,
    valid: np.ndarray,
    cfg: CleanConfig,
    *,
    require_full: bool,
    extra_outliers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Clean one trajectory. Returns (trans, rot, new_valid, confidence).

    require_full=True (object): every frame gets a pose (interp or held).
    require_full=False (hand): long/edge outliers are left invalid for downstream.
    rot may be axis-angle (T,3) or None; if given it is slerped on repaired frames.
    """
    T = trans.shape[0]
    trans = trans.astype(np.float64).copy()
    valid = valid.astype(bool).copy()
    conf = np.where(valid, CONF_REAL, CONF_IGNORED).astype(np.float32)

    outlier = _detect_outliers(trans, valid, cfg) | _excursion_outliers(trans, valid, cfg)
    if extra_outliers is not None:
        outlier = outlier | (extra_outliers & valid)
    outlier &= valid
    conf[outlier] = CONF_IGNORED

    good = valid & ~outlier
    new_valid = good.copy()
    rot_obj = Rotation.from_rotvec(rot.astype(np.float64)) if rot is not None else None
    rot_out = rot.astype(np.float64).copy() if rot is not None else None

    for a, b in _runs(~good):
        left = a - 1 if a - 1 >= 0 and good[a - 1] else None
        right = b if b < T and good[b] else None
        idx = np.arange(a, b)
        if left is not None and right is not None and (b - a) <= cfg.max_interp_gap:
            w = (idx - left) / (right - left)
            trans[idx] = (1 - w)[:, None] * trans[left] + w[:, None] * trans[right]
            if rot_obj is not None:
                sl = Slerp([left, right], Rotation.concatenate([rot_obj[left], rot_obj[right]]))
                rot_out[idx] = sl(idx).as_rotvec()
            new_valid[idx] = True
            conf[idx] = CONF_INTERP
        else:
            anchor = left if left is not None else right
            if anchor is not None and require_full:
                trans[idx] = trans[anchor]
                if rot_obj is not None:
                    rot_out[idx] = rot_out[anchor]
                new_valid[idx] = True
                conf[idx] = CONF_HELD
            # else (hand, no anchor / too long): leave invalid, conf stays 0

    return trans, rot_out, new_valid, conf


def clean_object_world(
    ob_world: np.ndarray,
    hand_trans: np.ndarray | None,
    hand_valid: np.ndarray | None,
    cfg: CleanConfig,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean an object pose stream (T,4,4). Returns (cleaned (T,4,4), confidence (T,))."""
    ob = np.asarray(ob_world, dtype=np.float64).copy()
    T = ob.shape[0]
    if valid is None:
        valid = np.ones(T, dtype=bool)
    trans = ob[:, :3, 3]
    rot = Rotation.from_matrix(ob[:, :3, :3]).as_rotvec()
    t2, r2, _nv, conf = clean_pose_stream(
        trans, rot, valid, cfg, require_full=True
    )
    ob[:, :3, 3] = t2
    ob[:, :3, :3] = Rotation.from_rotvec(r2).as_matrix()
    return ob, conf


def clean_hand_payload(
    hand_trans: np.ndarray,
    hand_rot: np.ndarray,
    hand_valid: np.ndarray,
    cfg: CleanConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clean both hands (2,T,·). Returns (hand_trans, hand_rot, hand_valid, hand_confidence)."""
    ht = np.asarray(hand_trans, dtype=np.float64).copy()
    hr = np.asarray(hand_rot, dtype=np.float64).copy()
    hv = np.asarray(hand_valid).astype(bool).copy()
    conf = np.zeros(hv.shape, dtype=np.float32)
    for h in range(ht.shape[0]):
        t2, r2, nv, c = clean_pose_stream(
            ht[h], hr[h], hv[h], cfg, require_full=False
        )
        ht[h], hr[h], hv[h], conf[h] = t2, r2, nv, c
    return ht, hr.astype(np.float32), hv.astype(np.float32), conf
