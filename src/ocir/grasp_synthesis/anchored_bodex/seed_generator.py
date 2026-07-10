"""Anchored seed generation: retargeted human contact-frame poses, relaxed
out of contact, plus Halton jitter.

Unlike bodex_curobo_v2's surface-normal-facing ``HeurGraspSeedGenerator``,
no object-surface points are sampled at all. The seed pool is built from the
demo frames where the human hand contacts the object (``frame_in_contact``
from the cached affordance), each retargeted to a full Sharpa action:

1. sample n frames from the contact set (window frames weighted higher);
2. batch-retarget the unique frames (fingertip-fit IK, :mod:`retarget`);
3. **relax**: open the flexion joints by ``relax_flexion`` and pull the wrist
   back ``relax_standoff`` along the palm approach axis (base-frame +X, the
   same convention as the BODex seeder), so a seed never starts in contact or
   penetration -- a bad starting point for the staged contact optimization;
4. jitter with a dedicated Halton generator; seed #0 stays the unjittered
   relaxed grasp-frame pose.

Each seed keeps its own *un-relaxed* retargeted action as the anchor for the
annealed pose prior and pose-similarity metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ocir.grasp_synthesis.anchored_bodex.affordance import Affordance
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import DemoGraspAnalysis
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.anchored_bodex.retarget import HandFitter, RetargetResult
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost import HaltonGenerator
from ocir.grasp_synthesis.bodex_curobo_v2.seed_generator import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)

#: Flexion joints opened by the relaxation step (FE/PIP/DIP/IP curl joints;
#: AA/abduction joints and the pinky CMC are left untouched).
RELAX_JOINT_SUFFIXES = ("_FE", "_PIP", "_DIP", "_IP")


def relax_joint_mask(joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    return np.array(
        [any(name.endswith(suffix) for suffix in RELAX_JOINT_SUFFIXES) for name in joint_names],
        dtype=bool,
    )


@dataclass
class AnchoredSeedGenerator:
    """Builds [pos(3), quat_wxyz(4), joints] seeds from demo contact frames.

    ``build(n)`` must be called once (it samples frames and runs the batched
    IK); ``get_samples(n)`` then returns the seeds in the rollout's expected
    format, and ``ref_actions`` holds the matching per-seed anchors.
    """

    fitter: HandFitter
    demo: HumanDemo
    affordance: Affordance
    analysis: DemoGraspAnalysis
    relax_flexion: float = 0.15        # rad subtracted from flexion joints
    relax_standoff: float = 0.015      # m pulled back along the approach axis
    jitter_pos: float = 0.01           # m, per-axis
    jitter_rot_deg: float = 10.0       # deg, per-axis
    jitter_joint: float = 0.08         # rad, per flexion joint
    window_weight: float = 3.0         # sampling weight for grasp-window frames
    seed: int = 1312

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._seeds: torch.Tensor | None = None
        self._ref_actions: torch.Tensor | None = None
        self._frame_indices: np.ndarray | None = None
        self._retarget: RetargetResult | None = None

    @property
    def ref_actions(self) -> torch.Tensor:
        if self._ref_actions is None:
            raise RuntimeError("AnchoredSeedGenerator.build() must be called first")
        return self._ref_actions

    @property
    def seed_frame_indices(self) -> np.ndarray:
        if self._frame_indices is None:
            raise RuntimeError("AnchoredSeedGenerator.build() must be called first")
        return self._frame_indices

    @property
    def retarget_result(self) -> RetargetResult:
        if self._retarget is None:
            raise RuntimeError("AnchoredSeedGenerator.build() must be called first")
        return self._retarget

    def _sample_frames(self, n: int) -> np.ndarray:
        frames = self.analysis.seed_frame_indices
        weights = np.ones(frames.shape[0], dtype=float)
        weights[np.isin(frames, self.analysis.window_indices)] = self.window_weight
        weights /= weights.sum()
        grasp = self.analysis.grasp_frame_index
        if n <= 1:
            return np.array([grasp], dtype=np.int64)
        replace = n - 1 > frames.shape[0]
        rest = self._rng.choice(frames, size=n - 1, replace=replace, p=weights)
        # Seed #0 is always the grasp frame itself.
        return np.concatenate([[grasp], np.sort(rest)]).astype(np.int64)

    def build(self, n: int) -> None:
        frame_choice = self._sample_frames(n)
        unique_frames, inverse = np.unique(frame_choice, return_inverse=True)
        result = self.fitter.fit_frames(
            self.analysis.wrist_poses_object[unique_frames],
            self.demo.hand_joints_object[unique_frames],
            unique_frames,
        )
        device_cfg = self.fitter.device_cfg
        ref_unique = device_cfg.to_device(result.ref_actions.astype(np.float32))
        inverse_t = torch.as_tensor(inverse, device=ref_unique.device, dtype=torch.long)
        ref_actions = ref_unique[inverse_t]  # (n, 7+J) per-seed anchors

        joint_names = self.fitter.joint_names
        n_joints = len(joint_names)
        relax_mask = device_cfg.to_device(relax_joint_mask(joint_names).astype(np.float32))

        # Relax: open flexion joints, clamp to limits.
        relaxed = ref_actions.clone()
        relaxed[:, 7:] = torch.clamp(
            relaxed[:, 7:] - self.relax_flexion * relax_mask.view(1, -1),
            self.fitter.joint_lower.view(1, -1),
            self.fitter.joint_upper.view(1, -1),
        )
        # Relax: pull the wrist back along the palm approach axis (base +X).
        from curobo._src.geom.transform import torch_quaternion_to_matrix

        rot = torch_quaternion_to_matrix(relaxed[:, 3:7])
        approach = rot[..., :, 0]  # base-frame +X in world coordinates
        relaxed[:, :3] = relaxed[:, :3] - self.relax_standoff * approach

        # Jitter (dedicated Halton stream): 3 pos + 3 euler + n_joints.
        jitter_rot = math.radians(self.jitter_rot_deg)
        low = [-self.jitter_pos] * 3 + [-jitter_rot] * 3 + [-self.jitter_joint] * n_joints
        high = [self.jitter_pos] * 3 + [jitter_rot] * 3 + [self.jitter_joint] * n_joints
        halton = HaltonGenerator(6 + n_joints, device_cfg, up_bounds=high, low_bounds=low, seed=self.seed)
        jitter = halton.get_samples(n, bounded=True)
        jitter[0] = 0.0  # seed #0 = unjittered relaxed grasp-frame pose

        jitter_mat = euler_angles_to_matrix(torch.flip(jitter[:, 3:6], dims=[-1]), "ZYX")
        final_rot = rot @ jitter_mat
        final_pos = relaxed[:, :3] + jitter[:, :3]
        final_q = torch.clamp(
            relaxed[:, 7:] + jitter[:, 6:] * relax_mask.view(1, -1),
            self.fitter.joint_lower.view(1, -1),
            self.fitter.joint_upper.view(1, -1),
        )
        seeds = torch.cat([final_pos, matrix_to_quaternion(final_rot), final_q], dim=-1)

        self._seeds = seeds
        self._ref_actions = ref_actions
        self._frame_indices = frame_choice
        self._retarget = result

    def get_samples(self, num_samples: int) -> torch.Tensor:
        if self._seeds is None or self._seeds.shape[0] != num_samples:
            self.build(num_samples)
        return self._seeds
