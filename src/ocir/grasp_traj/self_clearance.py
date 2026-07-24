"""Self-clearance correction of the trajectory's INITIAL posture (frame 0).

The Isaac simulation initializes the hand articulation by teleporting the
joints to frame 0 of the trajectory. The retargeting IK has no
self-collision term, and the near-open start posture commands straight
parallel fingers whose designed true-mesh gaps are only 0.05-1 mm -- under
the collider approximation that is actual interpenetration (the collision
spheres read the finger-base pairs at -2.7 mm; the PhysX convex hulls
overlap too). With hand self-collision ON, initializing INSIDE contact makes
PhysX resolve the overlap impulsively on the first step and the fingers blow
up / wedge. Only the initialization is dangerous: once the sim runs,
commanded targets that bring fingers close together merely produce steady,
drive-absorbable contact forces.

:class:`SelfClearanceTuner` therefore corrects the frame-0 posture only: it
spreads the finger SPREAD joints (the four MCP abductions, pinky CMC, thumb
CMC/MCP abductions -- never the wrist, never the flexion profile) until
every non-adjacent collision-sphere pair keeps at least ``buffer_m`` of
surface clearance, or the posture's anatomical maximum when the buffer is
out of reach. The caller (``generator.py``) decays the correction back to
the original trajectory over a short window so the commanded targets stay
continuous, and guards the affected frames against losing object clearance.

**Metric calibration** (flat-posture probe vs true-mesh measurements): the
spheres are inflated ~1.7 mm past the mesh on the binding finger-base pairs,
so sphere clearance 0.0 corresponds to ~1.5-1.8 mm of true mesh gap (and
non-overlapping PhysX hulls). The anatomical CEILING of the spread subspace
is only ~+0.1 mm sphere clearance at full fan (measured): finger-base /
webbing spheres barely move under abduction, exactly like a human hand whose
finger bases stay adjacent no matter how wide it spreads. The default buffer
is therefore **0.0** -- demanding millimetres of positive sphere clearance is
geometrically impossible, and satisfying PhysX's rest-offset preference
(1+1 mm with the baked asset) additionally requires the collider-level fix
(reduced rest offsets / finer decomposition), not posture tuning.

**Solver**: greedy coordinate ascent with batched line scans over the 7
spread joints. Gradient methods (Adam, L-BFGS, projected GD -- all tried)
reliably tangle fingers through each other or stall on the relu contact
landscape; an exhaustive line scan per coordinate cannot tangle and needs no
step size. Among scan values satisfying the buffer the one nearest the
original joint value is chosen (minimal perturbation); while unsatisfied it
climbs to each coordinate's maximum.

**Flexion is never touched**: staggering flexion would move sphere CENTERS
apart while the actual parallel finger surfaces stay side by side -- it games
the sphere metric without separating the hand.

Note the palm<->thumb_MC pair is excluded here by URDF-adjacency derivation
(bridged by the collider-less CMC_VL virtual link) exactly as it is
FilteredPairsAPI-exempted in the simulation asset -- the exclusion sets agree.
"""

from __future__ import annotations

import numpy as np
import torch

from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.assets import SharpaWaveAsset
from ocir.grasp_synthesis.anchored_bodex.guidance import build_self_collision_pairs
from ocir.grasp_synthesis.clearance import ClearanceChecker

#: Sphere-metric self-clearance the initial posture must keep. 0.0 = sphere
#: surfaces touching = ~1.5-1.8 mm true mesh gap at the binding finger-base
#: pairs (see module docstring); positive values are anatomically unreachable
#: at open postures (+0.1 mm ceiling at full spread). Negative disables.
DEFAULT_SELF_CLEARANCE_BUFFER_M = 0.0

#: Joint-name substrings identifying the spread subspace.
SPREAD_JOINT_TOKENS = ("_MCP_AA", "_CMC_AA", "pinky_CMC")


class SelfClearanceTuner:
    def __init__(
        self,
        asset: SharpaWaveAsset,
        device_cfg: DeviceCfg,
        checker: ClearanceChecker | None = None,
        *,
        buffer_m: float = DEFAULT_SELF_CLEARANCE_BUFFER_M,
        scan_values: int = 61,
        sweeps: int = 3,
        tolerance_m: float = 1e-4,
    ):
        self.device_cfg = device_cfg
        self.checker = checker if checker is not None else ClearanceChecker(asset, device_cfg)
        self.buffer_m = float(buffer_m)
        self.scan_values = int(scan_values)
        self.sweeps = int(sweeps)
        self.tolerance_m = float(tolerance_m)

        names = list(self.checker.sphere_link_names)
        pairs_i, pairs_j = build_self_collision_pairs(asset.urdf_path, names)
        device = device_cfg.device
        self._pair_i = torch.tensor(pairs_i, device=device, dtype=torch.long)
        self._pair_j = torch.tensor(pairs_j, device=device, dtype=torch.long)
        radii = self.checker.radii
        self._pair_radii = radii.index_select(0, self._pair_i) + radii.index_select(0, self._pair_j)
        self.pair_link_names = [(names[i], names[j]) for i, j in zip(pairs_i, pairs_j)]

        limits = asset.config["joint_limits"]
        self.joint_order = list(asset.config["joint_order"])
        self.spread_idx = [
            i for i, n in enumerate(self.joint_order) if any(t in n for t in SPREAD_JOINT_TOKENS)
        ]
        self._spread_bounds = [tuple(limits[self.joint_order[i]]) for i in self.spread_idx]

    def pair_clearances(self, joints: torch.Tensor) -> torch.Tensor:
        """(F, J) finger joints -> (F, P) signed sphere-surface clearance per
        non-adjacent pair. Base pose is irrelevant to self-clearance, so the
        FK runs at the identity root."""

        f = joints.shape[0]
        root = torch.zeros((f, 7), device=joints.device, dtype=joints.dtype)
        root[:, 3] = 1.0  # identity quat (wxyz)
        centers = self.checker.sphere_world_positions(torch.cat([root, joints], dim=-1))  # (F,N,3)
        ci = centers.index_select(1, self._pair_i)
        cj = centers.index_select(1, self._pair_j)
        return torch.linalg.norm(ci - cj, dim=-1) - self._pair_radii.view(1, -1)

    def min_clearances(self, joints: np.ndarray) -> np.ndarray:
        """(F, J) -> (F,) worst-pair sphere clearance per frame."""

        with torch.no_grad():
            c = self.pair_clearances(self.device_cfg.to_device(np.asarray(joints, dtype=np.float32)))
        return c.min(dim=-1).values.cpu().numpy()

    def worst_pairs(self, joints_frame: np.ndarray, top_k: int = 5) -> list[dict]:
        """Worst offending pairs of a single posture (below the buffer)."""

        with torch.no_grad():
            c = self.pair_clearances(
                self.device_cfg.to_device(np.asarray(joints_frame, dtype=np.float32)[None])
            )[0].cpu().numpy()
        order = np.argsort(c)[:top_k]
        return [
            {"links": list(self.pair_link_names[int(k)]), "min_clearance_m": float(c[int(k)])}
            for k in order
            if c[int(k)] < self.buffer_m
        ]

    def solve_posture(self, q: np.ndarray) -> tuple[np.ndarray, float]:
        """Greedy coordinate ascent over the spread joints of one posture.

        Per coordinate: batched line scan over its full range; among values
        whose worst-pair clearance meets the buffer, the one nearest the
        original joint value is chosen (minimal perturbation); while no value
        qualifies, the clearance-maximizing value (a nearest-original
        tolerance band here would stall the ascent -- early
        single-coordinate gains are tiny). Returns
        ``(tuned_posture, achieved_min_clearance)``; the posture is returned
        untouched when it already satisfies the buffer."""

        q_orig = np.asarray(q, dtype=np.float64)
        q_work = q_orig.copy()
        threshold = self.buffer_m - self.tolerance_m
        start = float(self.min_clearances(q_work[None])[0])
        if start >= threshold:
            return q_work, start
        for _ in range(self.sweeps):
            changed = False
            for k, i in enumerate(self.spread_idx):
                lo, hi = self._spread_bounds[k]
                vals = np.unique(np.concatenate([np.linspace(lo, hi, self.scan_values), [q_orig[i], q_work[i]]]))
                vals = vals[(vals >= lo) & (vals <= hi)]
                batch = np.tile(q_work[None], (vals.size, 1))
                batch[:, i] = vals
                scores = self.min_clearances(batch)
                ok = np.nonzero(scores >= threshold)[0]
                if ok.size:
                    pick = ok[np.argmin(np.abs(vals[ok] - q_orig[i]))]
                else:
                    pick = int(np.argmax(scores))
                if abs(vals[pick] - q_work[i]) > 1e-9:
                    changed = True
                q_work[i] = vals[pick]
            if not changed:
                break
        achieved = float(self.min_clearances(q_work[None])[0])
        return q_work, achieved
