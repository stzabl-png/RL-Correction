"""Demo-trajectory analysis for anchored BODex: grasp window, wrist frames,
and per-sequence hand-contact roles.

Pure NumPy (no torch/CUDA) so it can run and be inspected offline.

Keypoint order (from ``mano_model.pose_m_to_vertices_and_joints``):
index 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 pinky
(fingertips at 4/8/12/16/20, MCPs at 1/5/9/13/17).

MANO skinning part ids (``vertex_part_ids``): 0 wrist/palm, 1-3 index,
4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb (distal part last per finger).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ocir.grasp_synthesis.anchored_bodex.affordance import Affordance
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo

KP_WRIST = 0
KP_INDEX_MCP = 5
KP_MIDDLE_MCP = 9
KP_RING_MCP = 13

#: Canonical contact-role order. One-to-one, order-preserving with
#: ``bodex_curobo_v2.solver.SHARPA_CONTACT_POINTS``.
ROLE_ORDER = (
    "pinky_tip",
    "pinky_pad",
    "ring_pad",
    "ring_tip",
    "middle_pad",
    "middle_tip",
    "index_pad",
    "index_tip",
    "thumb_pad",
    "thumb_tip",
    "palm",
)

#: MANO skinning part ids contributing to each role.
ROLE_TO_MANO_PARTS: dict[str, tuple[int, ...]] = {
    "index_tip": (3,),
    "index_pad": (1, 2),
    "middle_tip": (6,),
    "middle_pad": (4, 5),
    "pinky_tip": (9,),
    "pinky_pad": (7, 8),
    "ring_tip": (12,),
    "ring_pad": (10, 11),
    "thumb_tip": (15,),
    "thumb_pad": (13, 14),
    # Part 0 covers palm + back of hand; proximity alone disambiguates (the
    # hand is ~2-3 cm thick, so dorsal vertices never fall within the role
    # contact threshold while the palmar side touches).
    "palm": (0,),
}

DEFAULT_ROLE_CONTACT_THRESHOLD_M = 0.008
DEFAULT_ROLE_MIN_VERTICES = 5
DEFAULT_ROLE_MIN_FRAME_FRACTION = 0.3


def wrist_frame_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Build a 4x4 wrist pose in the keypoints' frame.

    Origin = wrist joint; x-axis toward the middle MCP (finger direction);
    the palm normal (index-MCP x ring-MCP cross product) seeds the y-axis via
    Gram-Schmidt. The residual rotation to the robot hand's base_link lives in
    the MANO-transfer calibration yaml -- this convention just has to stay
    fixed and match the calibration script.
    """

    keypoints = np.asarray(keypoints, dtype=float)
    origin = keypoints[KP_WRIST]
    x_raw = keypoints[KP_MIDDLE_MCP] - origin
    n_raw = np.cross(keypoints[KP_INDEX_MCP] - origin, keypoints[KP_RING_MCP] - origin)
    x = x_raw / max(np.linalg.norm(x_raw), 1e-9)
    y = n_raw - np.dot(n_raw, x) * x
    y = y / max(np.linalg.norm(y), 1e-9)
    z = np.cross(x, y)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = origin
    return pose


@dataclass(frozen=True)
class DemoGraspAnalysis:
    pickup_frame_index: int
    grasp_frame_index: int
    window_indices: np.ndarray           # frame indices in the grasp window
    seed_frame_indices: np.ndarray       # contact frames usable for seeding
    wrist_poses_object: np.ndarray       # (T,4,4), NaN rows for invalid frames
    active_roles: tuple[str, ...]
    role_contact_stats: dict[str, dict] = field(repr=False)
    report: dict = field(repr=False)

    @property
    def grasp_wrist_pose_object(self) -> np.ndarray:
        return self.wrist_poses_object[self.grasp_frame_index]

    @property
    def grasp_keypoints_object(self) -> np.ndarray:
        return np.asarray(self.report["grasp_keypoints_object"], dtype=float)


def _detect_pickup_frame(
    demo: HumanDemo,
    affordance: Affordance,
    *,
    move_threshold_m: float,
    rot_threshold_rad: float,
    hold_frames: int,
) -> tuple[int, str]:
    valid = demo.valid_indices
    n_ref = max(1, int(round(0.2 * valid.shape[0])))
    ref_frames = valid[:n_ref]
    ref_pos = np.median(demo.object_pose_camera[ref_frames, :3, 3], axis=0)
    ref_rots = demo.object_pose_camera[ref_frames, :3, :3]
    ref_rot = ref_rots[ref_rots.shape[0] // 2]

    moved = np.zeros((demo.num_frames,), dtype=bool)
    for i in valid:
        pose = demo.object_pose_camera[i]
        trans = np.linalg.norm(pose[:3, 3] - ref_pos)
        cos_angle = np.clip((np.trace(ref_rot.T @ pose[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        moved[i] = trans > move_threshold_m or np.arccos(cos_angle) > rot_threshold_rad

    for i in valid:
        stop = min(i + hold_frames, demo.num_frames)
        window = moved[i:stop]
        if window.shape[0] >= hold_frames and window.all():
            return int(i), "object_motion"

    # Object never moves persistently: fall back to peak hand-object contact.
    if affordance.frame_contact_count.max() > 0:
        return int(np.argmax(affordance.frame_contact_count)), "max_contact_fallback"
    raise ValueError("cannot detect a pickup frame: object never moves and hand never contacts it")


def analyze_demo(
    demo: HumanDemo,
    affordance: Affordance,
    *,
    move_threshold_m: float = 0.015,
    rot_threshold_rad: float = np.deg2rad(15.0),
    hold_frames: int = 3,
    window_before: int = 8,
    window_after: int = 4,
    role_contact_threshold_m: float = DEFAULT_ROLE_CONTACT_THRESHOLD_M,
    role_min_vertices: int = DEFAULT_ROLE_MIN_VERTICES,
    role_min_frame_fraction: float = DEFAULT_ROLE_MIN_FRAME_FRACTION,
) -> DemoGraspAnalysis:
    if str(demo.mano_side).lower() != "right":
        raise ValueError(
            f"anchored BODex currently supports right-hand demos only (got mano_side="
            f"{demo.mano_side!r}); the Sharpa Wave asset in this repo is a right hand"
        )

    pickup, pickup_mode = _detect_pickup_frame(
        demo,
        affordance,
        move_threshold_m=move_threshold_m,
        rot_threshold_rad=rot_threshold_rad,
        hold_frames=hold_frames,
    )

    usable = demo.valid_mask & affordance.frame_in_contact
    seed_frames = np.flatnonzero(usable)
    if seed_frames.size == 0:
        raise ValueError("no demo frames are simultaneously valid and in hand-object contact")

    lo, hi = pickup - window_before, pickup + window_after
    window = seed_frames[(seed_frames >= lo) & (seed_frames <= hi)]
    if window.size == 0:
        # Contact phase never overlaps the detected pickup window (e.g. late
        # detection); use the contact frames nearest the pickup instead.
        order = np.argsort(np.abs(seed_frames - pickup))
        window = np.sort(seed_frames[order[: min(window_before + window_after + 1, seed_frames.size)]])

    grasp_frame = int(window[np.argmax(affordance.frame_contact_count[window])])

    wrist_poses = np.full((demo.num_frames, 4, 4), np.nan)
    for i in demo.valid_indices:
        wrist_poses[i] = wrist_frame_from_keypoints(demo.hand_joints_object[i])

    # Per-role contact statistics over the grasp window.
    object_points = affordance.points_object_frame
    part_ids = demo.vertex_part_ids
    role_masks = {role: np.isin(part_ids, ROLE_TO_MANO_PARTS[role]) for role in ROLE_ORDER}
    role_frame_hits = {role: 0 for role in ROLE_ORDER}
    role_min_dist = {role: np.inf for role in ROLE_ORDER}
    for i in window:
        vertices = demo.hand_vertices_object[i]
        dist = np.sqrt(
            np.min(np.sum((vertices[:, None, :] - object_points[None, :, :]) ** 2, axis=2), axis=1)
        )
        for role in ROLE_ORDER:
            role_dist = dist[role_masks[role]]
            role_min_dist[role] = min(role_min_dist[role], float(role_dist.min()))
            if int(np.count_nonzero(role_dist <= role_contact_threshold_m)) >= role_min_vertices:
                role_frame_hits[role] += 1

    stats = {
        role: {
            "contact_frame_fraction": role_frame_hits[role] / float(window.size),
            "min_distance_m": role_min_dist[role],
            "num_vertices": int(role_masks[role].sum()),
        }
        for role in ROLE_ORDER
    }
    active = tuple(
        role for role in ROLE_ORDER if stats[role]["contact_frame_fraction"] >= role_min_frame_fraction
    )

    fallback_reason = None
    finger_tips_active = {r for r in active if r.endswith("_tip")}
    if len(active) < 3:
        fallback_reason = f"only {len(active)} active roles"
    elif "thumb_tip" not in finger_tips_active and "thumb_pad" not in active:
        fallback_reason = "thumb never in contact (no opposition)"
    if fallback_reason is not None:
        active = ROLE_ORDER

    report = {
        "pickup_mode": pickup_mode,
        "pickup_frame_id": int(demo.frame_ids[pickup]),
        "grasp_frame_id": int(demo.frame_ids[grasp_frame]),
        "window_frame_ids": [int(demo.frame_ids[i]) for i in window],
        "num_seed_frames": int(seed_frames.size),
        "role_fallback_reason": fallback_reason,
        "grasp_keypoints_object": demo.hand_joints_object[grasp_frame].tolist(),
    }
    return DemoGraspAnalysis(
        pickup_frame_index=pickup,
        grasp_frame_index=grasp_frame,
        window_indices=window,
        seed_frame_indices=seed_frames,
        wrist_poses_object=wrist_poses,
        active_roles=active,
        role_contact_stats=stats,
        report=report,
    )
