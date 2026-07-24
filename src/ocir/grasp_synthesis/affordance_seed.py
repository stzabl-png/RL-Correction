"""Affordance-guided grasp seeding.

Predicts a per-point "expected grasp area" heatmap for an object with OCIR's
**vendored** affordance model (a Sonata point-transformer encoder + MLP head,
``ocir.affordance``; checkpoint at ``assets/affordance/model.pt``), then
restricts the frozen ``bodex_curobo_v2`` seed pool to the high-affordance region
-- so grasp synthesis initializes only where a human would actually grab the
object, instead of over the whole surface.

The affordance backbone needs ``sonata`` + ``spconv-cu128`` + ``torch_scatter``,
which the grasp-synthesis env does not have, so prediction runs as a subprocess
in a dedicated affordance conda env (see ``envs/affordance-requirements.txt``);
synthesis then consumes the cached ``affordance.npz`` in-process.

Config via environment variables (with defaults):
  OCIR_AFFORDANCE_ENV                conda env with the sonata stack (default: deximit)
  OCIR_AFFORDANCE_CKPT               checkpoint (default: <repo>/assets/affordance/model.pt)
  OCIR_AFFORDANCE_EXTRA_PYTHONPATH   extra paths for the predict subprocess
                                     (only needed if ``sonata`` is importable via a
                                     checkout rather than pip-installed in the env)

``affordance.npz`` schema (written by ``ocir.affordance.predict``):
  points_raw (N,3) surface samples in the object-mesh frame
  heatmap    (N,)  per-point affordance in [0,1]
  points     (N,3) unit-sphere-normalized samples; center (3,); scale ()
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[3]
AFFORDANCE_ENV = os.environ.get("OCIR_AFFORDANCE_ENV", "deximit")
AFFORDANCE_CKPT = os.environ.get("OCIR_AFFORDANCE_CKPT", str(REPO_ROOT / "assets" / "affordance" / "model.pt"))


def predict_affordance(
    mesh_path: str | Path,
    out_dir: str | Path,
    *,
    n_points: int = 2048,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Run the AffordanceModel on ``mesh_path`` (subprocess into its conda env)
    and return the path to the written ``affordance.npz``. Cached: if the npz
    already exists and ``force`` is False, prediction is skipped."""

    out_dir = Path(out_dir)
    npz = out_dir / "affordance.npz"
    if npz.exists() and not force:
        return npz

    ckpt = Path(AFFORDANCE_CKPT)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"affordance checkpoint not found at {ckpt}. It ships via git-lfs at "
            f"assets/affordance/model.pt; run `git lfs pull`, or set OCIR_AFFORDANCE_CKPT."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean PYTHONPATH: only OCIR's src (so `ocir.affordance` imports) plus any
    # OCIR_AFFORDANCE_EXTRA_PYTHONPATH. We must NOT inherit the caller's
    # PYTHONPATH -- when synthesis runs under env_isaacsim its activate hook puts
    # Isaac Sim's bundled numpy/torch on PYTHONPATH, which would shadow the
    # affordance env's own torch / spconv / torch_scatter in this subprocess.
    src_root = REPO_ROOT / "src"
    extra = os.environ.get("OCIR_AFFORDANCE_EXTRA_PYTHONPATH", "")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src_root}:{extra}" if extra else str(src_root)

    conda_exe = os.environ.get("CONDA_EXE") or "conda"
    cmd = [
        conda_exe, "run", "--no-capture-output", "-n", AFFORDANCE_ENV, "python",
        "-m", "ocir.affordance.predict",
        "--mesh", str(Path(mesh_path).resolve()),
        "--ckpt", str(ckpt.resolve()),
        "--n", str(int(n_points)),
        "--seed", str(int(seed)),
        "--out", str(out_dir.resolve()),
    ]
    print(f"[affordance] predicting expected grasp area -> {npz} (env={AFFORDANCE_ENV})", flush=True)
    subprocess.run(cmd, env=env, check=True)
    if not npz.exists():
        raise RuntimeError(f"affordance prediction did not write {npz}")
    return npz


def load_affordance_region(
    mesh_path: str | Path,
    affordance_npz: str | Path,
    *,
    threshold: float = 0.5,
    min_points: int = 16,
    top_fraction_fallback: float = 0.25,
    world_up_object: np.ndarray | None = None,
    min_normal_up: float = -0.2,
    approach_dir_object: np.ndarray | None = None,
    approach_cone_cos: float = 0.174,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return ``(points, normals, info)`` for the high-affordance region:
    surface points (projected onto the mesh) with outward face normals, ready to
    seed the grasp optimizer. Falls back to the top ``top_fraction_fallback`` of
    points if the threshold selects fewer than ``min_points``.

    When ``world_up_object`` (the table "up" in the object frame) is given, seed
    points whose outward normal points DOWNWARD (``normal.up < min_normal_up``)
    are dropped, so the hand only approaches the object from above/the side --
    never from below the table. Without this the optimizer freely picks
    under-the-object grasps whose wrist ends up beneath the tabletop.

    When ``approach_dir_object`` (the human hand's approach direction in the
    object frame, from the pre-contact wrist PATH -- never the unreliable hand
    orientation) is given, seed points are further restricted to the side the
    hand actually approached from: a point survives only if its outward normal
    faces back toward the incoming hand, ``normal . (-approach) >=
    approach_cone_cos``. The default ``0.174`` is ``cos(80 deg)`` -- a wide cone
    that keeps the whole near hemisphere and only drops the clearly-unreachable
    far side (e.g. the underside of an apple grabbed top-down). Applied after the
    table-up filter and, like it, skipped if fewer than ``min_points`` survive
    (never narrow the region down to nothing on a noisy direction)."""

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    centroid = np.asarray(mesh.vertices, dtype=float).mean(axis=0)

    aff = np.load(str(affordance_npz))
    points_raw = np.asarray(aff["points_raw"], dtype=np.float64)
    heat = np.asarray(aff["heatmap"], dtype=np.float64)

    mask = heat > float(threshold)
    used_fallback = False
    if int(mask.sum()) < int(min_points):
        used_fallback = True
        k = max(int(min_points), int(top_fraction_fallback * len(heat)))
        idx = np.argsort(-heat)[:k]
        mask = np.zeros(len(heat), dtype=bool)
        mask[idx] = True

    sel = points_raw[mask]
    closest, _, face_id = trimesh.proximity.closest_point(mesh, sel)
    normals = np.asarray(mesh.face_normals)[face_id].astype(np.float64)
    inward = np.sum(normals * (closest - centroid[None, :]), axis=1) < 0
    normals[inward] *= -1.0

    # drop under-the-object seeds (down-facing normals) so the hand never
    # approaches from below the table
    dropped_down = 0
    if world_up_object is not None:
        up = np.asarray(world_up_object, dtype=np.float64).reshape(3)
        up = up / (np.linalg.norm(up) + 1e-12)
        keep = (normals @ up) >= float(min_normal_up)
        if int(keep.sum()) >= int(min_points):
            dropped_down = int((~keep).sum())
            closest = closest[keep]
            normals = normals[keep]

    # narrow to the side the human hand approached from: keep points whose
    # outward normal faces back toward the incoming hand (approach direction).
    dropped_cone = 0
    approach_used = None
    region_before_cone = int(len(closest))
    if approach_dir_object is not None:
        a = np.asarray(approach_dir_object, dtype=np.float64).reshape(3)
        a = a / (np.linalg.norm(a) + 1e-12)
        keep = (normals @ (-a)) >= float(approach_cone_cos)
        if int(keep.sum()) >= int(min_points):
            dropped_cone = int((~keep).sum())
            closest = closest[keep]
            normals = normals[keep]
            approach_used = np.round(a, 4).tolist()

    info = {
        "affordance_npz": str(affordance_npz),
        "threshold": float(threshold),
        "region_points": int(len(closest)),
        "total_points": int(len(heat)),
        "heatmap_max": float(heat.max()),
        "heatmap_mean": float(heat.mean()),
        "used_top_fraction_fallback": used_fallback,
        "dropped_down_facing": int(dropped_down),
        "approach_dir_object": approach_used,
        "approach_cone_cos": float(approach_cone_cos) if approach_dir_object is not None else None,
        "approach_cone_applied": approach_used is not None,
        "region_points_before_cone": region_before_cone,
        "dropped_by_cone": int(dropped_cone),
        "region_bounds_min": np.round(closest.min(0), 4).tolist(),
        "region_bounds_max": np.round(closest.max(0), 4).tolist(),
    }
    return closest.astype(np.float64), normals.astype(np.float64), info


def approach_cone_mask(
    mesh_path: str | Path,
    points: np.ndarray,
    approach_dir_object: np.ndarray,
    *,
    cone_cos: float = 0.174,
) -> np.ndarray:
    """Boolean keep-mask over ``points`` (object frame): ``True`` where the
    mesh's outward normal nearest the point faces back toward the incoming hand,
    ``normal . (-approach) >= cone_cos``. Narrows a grasp region to the side the
    human hand approached from -- the same wide-cone test ``load_affordance_region``
    applies, factored out so the anchored pipeline can reuse it on its affordance
    guidance points (which carry no precomputed normals)."""

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    centroid = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    closest, _, face_id = trimesh.proximity.closest_point(mesh, pts)
    normals = np.asarray(mesh.face_normals)[face_id].astype(np.float64)
    inward = np.sum(normals * (closest - centroid[None, :]), axis=1) < 0
    normals[inward] *= -1.0
    a = np.asarray(approach_dir_object, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    return (normals @ (-a)) >= float(cone_cos)


# Fingertip-only contact subset (the 5 distal-phalanx tips of SHARPA_CONTACT_POINTS)
# for TopDown fingertip/pinch grasps of small or flat objects -- no pads, no palm,
# so the optimizer stops trying to envelop a thin object with the whole hand.
FINGERTIP_CONTACT_POINTS = [
    "right_pinky_DP/1",
    "right_ring_DP/1",
    "right_middle_DP/1",
    "right_index_DP/1",
    "right_thumb_DP/1",
]
# Matched force-closure pressure constraints over the 5 fingertip indices
# (0 pinky, 1 ring, 2 middle, 3 index, 4 thumb): preserves the thumb-vs-fingers
# opposition structure of the 11-point default.
FINGERTIP_PRESSURE_CONSTRAINTS = [
    [[0, 1, 2, 3, 4], 1.0],
    [[4], 0.5],
    [[0, 1, 2, 3], 0.7],
    [[0, 1], 0.4],
    [[2, 3], 0.5],
]


def install_fingertip_contacts():
    """Monkeypatch the frozen solver's module-level contact-point set to the
    fingertip-only subset (with matched pressure constraints), so the optimizer
    builds a precision/pinch grasp instead of a whole-hand envelope. Call BEFORE
    ``solve_sharpa_bodex``. Returns the patched solver module."""

    import copy
    import ocir.grasp_synthesis.bodex_curobo_v2.solver as solver_mod

    solver_mod.SHARPA_CONTACT_POINTS = list(FINGERTIP_CONTACT_POINTS)
    ge = copy.deepcopy(solver_mod.DEFAULT_GE_PARAM)
    ge["pressure_constraints"] = [[list(g), float(c)] for g, c in FINGERTIP_PRESSURE_CONSTRAINTS]
    solver_mod.DEFAULT_GE_PARAM = ge
    return solver_mod


def install_table_penalty(table_up_object: np.ndarray, table_offset: float, weight: float):
    """Monkeypatch the frozen ``SharpaBodexRollout`` to add a hand-table collision
    penalty *during optimization* (the frozen core has none -- it only avoids the
    table via seed filtering + post-hoc selection). Each contact sphere's lowest
    point below the object-frame tabletop plane is penalized ``relu(depth)^2``,
    summed, times ``weight`` -- the same energy the anchored rollout uses. Call
    BEFORE ``solve_sharpa_bodex``. Returns the patched solver module."""

    import torch
    import ocir.grasp_synthesis.bodex_curobo_v2.solver as solver_mod

    up_np = np.asarray(table_up_object, dtype=np.float64).reshape(3)
    up_np = (up_np / (np.linalg.norm(up_np) + 1e-12)).astype(np.float32)
    off = float(table_offset)
    w = float(weight)
    orig = solver_mod.SharpaBodexRollout._costs_and_constraints

    def _patched(self, state, opt_progress: float = 0.0):
        costs = orig(self, state, opt_progress)
        up = getattr(self, "_afford_table_up", None)
        if up is None:
            up = self.device_cfg.to_device(up_np)
            self._afford_table_up = up
        robot_spheres, _ = self._fk_contact_inputs(state.position)
        centers = robot_spheres[:, 0, :, :3]        # (B, n_contacts, 3)
        radii = robot_spheres[:, 0, :, 3]           # (B, n_contacts)
        height = torch.matmul(centers, up)          # (B, n_contacts)
        depth = torch.relu(off - (height - radii))  # meters below the tabletop
        table_cost = w * (depth * depth).sum(dim=-1)  # (B,)
        costs.costs.add(table_cost.view(-1, 1, 1), "table_penalty")
        return costs

    solver_mod.SharpaBodexRollout._costs_and_constraints = _patched
    return solver_mod


def install_affordance_seed_sampler(points: np.ndarray, normals: np.ndarray, *, seed: int = 0):
    """Monkeypatch ``bodex_curobo_v2.solver.sample_surface_points_and_normals``
    so the (frozen) solver builds its seed pool from the affordance region
    instead of the whole surface. Honors the ``inflate`` argument. Returns the
    patched solver module (call BEFORE ``solve_sharpa_bodex``)."""

    import ocir.grasp_synthesis.bodex_curobo_v2.solver as solver_mod

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    rng = np.random.default_rng(seed)

    def _affordance_sampler(mesh, num, inflate, convex_hull, seed=0):  # noqa: ARG001
        idx = rng.choice(len(points), size=int(num), replace=True)
        pts = points[idx].copy()
        nrm = normals[idx].copy()
        if inflate:
            pts = pts + float(inflate) * nrm
        return pts.astype(np.float64), nrm.astype(np.float64)

    solver_mod.sample_surface_points_and_normals = _affordance_sampler
    return solver_mod
