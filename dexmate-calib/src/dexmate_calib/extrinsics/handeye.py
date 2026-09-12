"""Eye-to-hand calibration of a fixed external camera against the robot base.

Problem
-------
A ChArUco board is rigidly attached to a robot link (``link``); a fixed camera
(``cam``) observes it while the robot is moved through several poses.  For view
``i`` the robot provides ``T_base_link_i`` (forward kinematics) and the camera
provides 2D observations of known board corners.  The unknowns are

* ``X = T_base_cam``   – the pose of the camera in the robot base frame (goal), and
* ``Y = T_link_board`` – the pose of the board on the link (nuisance, unknown at
  mounting time).

For every view the rigid chain must close::

    T_base_link_i @ Y == X @ T_cam_board_i

Approach
--------
1. Per-view PnP gives ``T_cam_board_i``.
2. ``cv2.calibrateRobotWorldHandEye`` (AX = ZB) provides a closed-form initial
   ``X`` and ``Y``.
3. Non-linear refinement minimises the *reprojection error* of all board corners
   over the 12 parameters of ``X`` and ``Y`` (left-perturbation on SE(3),
   Levenberg–Marquardt, Huber weights), so the result does not inherit per-view
   PnP noise.
4. Views whose reprojection RMS stays above the threshold are dropped and the
   refinement is repeated; leave-one-view-out re-solves report stability.

Only numpy and OpenCV are used; no robot or camera SDK is imported here so the
solver is fully testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dexmate_calib.geometry.transforms import (
    inv_T,
    pose_error,
    project_rotation,
    rotation_angle_deg,
    rt_to_T,
    se3_exp,
    so3_log,
)


@dataclass
class HandEyeView:
    name: str
    object_points: np.ndarray
    """(N, 3) board-frame corner coordinates in metres."""
    image_points: np.ndarray
    """(N, 2) observed pixel coordinates in the (distorted) image."""
    T_base_link: np.ndarray
    """4x4 pose of the mounting link in the robot base frame (from FK)."""
    T_cam_board_pnp: np.ndarray | None = None
    """4x4 per-view PnP board pose in the camera frame; filled by :func:`solve_hand_eye`."""
    T_cam_board_candidates: list[np.ndarray] | None = None
    """All PnP candidates (planar IPPE gives up to two); the chosen one is ``T_cam_board_pnp``."""

    def __post_init__(self) -> None:
        self.object_points = np.asarray(self.object_points, dtype=np.float64).reshape(-1, 3)
        self.image_points = np.asarray(self.image_points, dtype=np.float64).reshape(-1, 2)
        self.T_base_link = np.asarray(self.T_base_link, dtype=np.float64).reshape(4, 4)
        if len(self.object_points) != len(self.image_points):
            raise ValueError(f"{self.name}: object/image point count mismatch")


@dataclass
class HandEyeSolution:
    T_base_cam: np.ndarray
    T_link_board: np.ndarray
    rms_px: float
    view_reports: list[dict]
    inlier_views: list[str]
    rejected_views: list[dict]
    initialisation: dict
    refinement: dict
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "T_base_cam": self.T_base_cam.tolist(),
            "T_link_board": self.T_link_board.tolist(),
            "rms_reprojection_error_px": self.rms_px,
            "inlier_views": list(self.inlier_views),
            "rejected_views": list(self.rejected_views),
            "initialisation": self.initialisation,
            "refinement": self.refinement,
            "diagnostics": self.diagnostics,
            "view_reports": self.view_reports,
        }


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-contrib-python is required") from exc
    return cv2


def _pose_to_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))
    return rvec.reshape(3, 1), np.ascontiguousarray(T[:3, 3]).reshape(3, 1)


def pnp_board_pose_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> list[np.ndarray]:
    """Return candidate ``T_cam_board`` poses for one planar view, best first.

    Planar targets are solved with IPPE, which yields up to two solutions (the classic
    "flip" ambiguity, significant for small or single-tag targets); each is LM-refined,
    kept if the board is in front of the camera and sorted by reprojection RMS.  Falls
    back to the iterative solver when IPPE fails.
    """
    cv2 = _cv2()
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if len(obj) < 4:
        raise ValueError("PnP needs at least 4 board corners")
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    planar = np.allclose(obj[:, 0, 2], obj[0, 0, 2])
    if planar:
        try:
            _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
            raw = list(zip(rvecs, tvecs))
        except cv2.error:
            raw = []
    if not raw and len(obj) >= 6:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            raw = [(rvec, tvec)]
    if not raw:
        raise ValueError("solvePnP failed")
    scored: list[tuple[float, np.ndarray]] = []
    for rvec, tvec in raw:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        T = rt_to_T(R, tvec)
        if T[2, 3] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        rms = _rms_from_residuals(img.reshape(-1, 2) - projected.reshape(-1, 2))
        scored.append((rms, T))
    if not scored:
        raise ValueError("PnP placed the board behind the camera")
    scored.sort(key=lambda item: item[0])
    # Drop near-duplicate solutions (IPPE sometimes returns the same pose twice).
    unique: list[np.ndarray] = []
    for _, T in scored:
        if all(pose_error(T, U)[1] > 5e-4 or pose_error(T, U)[0] > 0.05 for U in unique):
            unique.append(T)
    return unique


def pnp_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    """Best single ``T_cam_board`` for one view (see :func:`pnp_board_pose_candidates`)."""
    return pnp_board_pose_candidates(object_points, image_points, camera_matrix, dist_coeffs)[0]


def project_board(
    object_points: np.ndarray,
    T_cam_board: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    cv2 = _cv2()
    rvec, tvec = _pose_to_rvec_tvec(T_cam_board)
    D = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        D,
    )
    return projected.reshape(-1, 2)


def predicted_T_cam_board(
    T_base_cam: np.ndarray, T_base_link: np.ndarray, T_link_board: np.ndarray
):
    return inv_T(T_base_cam) @ T_base_link @ T_link_board


def view_residuals(
    view: HandEyeView,
    T_base_cam: np.ndarray,
    T_link_board: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
) -> np.ndarray:
    """(N, 2) reprojection residuals ``observed - predicted`` for one view."""
    T_cam_board = predicted_T_cam_board(T_base_cam, view.T_base_link, T_link_board)
    if T_cam_board[2, 3] <= 0.0:
        # Board behind the camera: penalise strongly but keep finite.
        return np.full_like(view.image_points, 1e4)
    projected = project_board(view.object_points, T_cam_board, camera_matrix, dist_coeffs)
    return view.image_points - projected


def stacked_residuals(views, X, Y, K, D) -> np.ndarray:
    return np.concatenate([view_residuals(v, X, Y, K, D).reshape(-1) for v in views])


def _rms_from_residuals(res: np.ndarray) -> float:
    res = res.reshape(-1, 2)
    if len(res) == 0:
        return float("nan")
    return math.sqrt(float(np.mean(np.sum(res**2, axis=1))))


# --------------------------------------------------------------------------- init


def _vec(M: np.ndarray) -> np.ndarray:
    """Column-major vectorisation (matches vec(AXB) = (B^T kron A) vec(X))."""
    return np.asarray(M, dtype=np.float64).reshape(9, order="F")


def _unvec(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).reshape(3, 3, order="F")


def solve_ax_yb_linear(
    A_list: list[np.ndarray], B_list: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``A_i @ Y = X @ B_i`` (Li et al. 2010, Kronecker formulation).

    Rotations: ``R_A_i R_Y = R_X R_B_i`` gives the homogeneous linear system
    ``[(I kron R_A_i), -(R_B_i^T kron I)] [vec R_Y; vec R_X] = 0`` whose null vector is
    recovered by SVD and projected onto SO(3).  Translations then follow from the
    linear least-squares problem ``R_A_i t_Y - t_X = R_X t_B_i - t_A_i``.
    """
    if len(A_list) < 3:
        raise ValueError("Need at least 3 pose pairs for the closed-form solution")
    I3 = np.eye(3)
    rows = []
    for A, B in zip(A_list, B_list):
        RA, RB = A[:3, :3], B[:3, :3]
        rows.append(np.hstack([np.kron(I3, RA), -np.kron(RB.T, I3)]))
    M = np.vstack(rows)
    _, _, Vt = np.linalg.svd(M)
    null = Vt[-1]
    RY_raw, RX_raw = _unvec(null[:9]), _unvec(null[9:])
    if np.linalg.det(RY_raw) < 0.0:
        RY_raw, RX_raw = -RY_raw, -RX_raw
    RY, RX = project_rotation(RY_raw), project_rotation(RX_raw)
    if np.linalg.det(RX) < 0.0:  # pragma: no cover - only for wildly inconsistent data
        raise ValueError("Closed-form rotation is improper; pose pairs are inconsistent")
    # Translations.
    lhs, rhs = [], []
    for A, B in zip(A_list, B_list):
        lhs.append(np.hstack([A[:3, :3], -I3]))
        rhs.append(RX @ B[:3, 3] - A[:3, 3])
    sol, *_ = np.linalg.lstsq(np.vstack(lhs), np.concatenate(rhs), rcond=None)
    tY, tX = sol[:3], sol[3:]
    return rt_to_T(RX, tX), rt_to_T(RY, tY)


def initial_hand_eye(views: list[HandEyeView]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Closed-form initialisation candidates ``(method, X, Y)`` for ``A_i Y = X B_i``.

    ``A_i = T_base_link_i`` (FK) and ``B_i = T_cam_board_i`` (per-view PnP).  Only the
    numpy Kronecker solver is used; OpenCV's ``calibrateRobotWorldHandEye`` is not
    exposed in every wheel (missing in opencv-contrib-python 5.0), so we do not depend
    on it.
    """
    usable = [v for v in views if v.T_cam_board_pnp is not None]
    if len(usable) < 3:
        raise ValueError("Need at least 3 views with PnP poses for initialisation")
    A_list = [v.T_base_link for v in usable]
    B_list = [v.T_cam_board_pnp for v in usable]
    X, Y = solve_ax_yb_linear(A_list, B_list)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("Closed-form hand-eye initialisation produced non-finite values")
    return [("kronecker_li2010", X, Y)]


def _pose_chain_rms(views, X, Y) -> tuple[float, float]:
    """Mean rotation/translation gap between FK-predicted and PnP board poses."""
    rot, trans = [], []
    for v in views:
        if v.T_cam_board_pnp is None:
            continue
        pred = predicted_T_cam_board(X, v.T_base_link, Y)
        r_deg, t_m = pose_error(pred, v.T_cam_board_pnp)
        rot.append(r_deg)
        trans.append(t_m)
    return float(np.mean(rot)), float(np.mean(trans))


def _resolve_pnp_ambiguities(views: list[HandEyeView], X: np.ndarray, Y: np.ndarray) -> int:
    """Pick, per view, the PnP candidate closest to the chain prediction. Returns #changes."""
    changed = 0
    for v in views:
        cands = v.T_cam_board_candidates or []
        if len(cands) < 2:
            continue
        pred = predicted_T_cam_board(X, v.T_base_link, Y)

        def score(T, _pred=pred):
            r_deg, t_m = pose_error(_pred, T)
            return r_deg + 100.0 * t_m  # 1 deg ~ 1 cm

        best = min(cands, key=score)
        if best is not v.T_cam_board_pnp and score(best) < score(v.T_cam_board_pnp) - 1e-9:
            v.T_cam_board_pnp = best
            changed += 1
    return changed


# ------------------------------------------------------------------------- refine


def _apply_params(params: np.ndarray, X0: np.ndarray, Y0: np.ndarray):
    return se3_exp(params[:6]) @ X0, se3_exp(params[6:]) @ Y0


def _huber_weights(res: np.ndarray, delta_px: float) -> np.ndarray:
    """Per-residual-component IRLS weights from the per-corner residual norm."""
    r2 = res.reshape(-1, 2)
    norms = np.linalg.norm(r2, axis=1)
    w = np.ones_like(norms)
    mask = norms > delta_px
    w[mask] = delta_px / norms[mask]
    return np.repeat(w, 2)


def refine_hand_eye(
    views: list[HandEyeView],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    X0: np.ndarray,
    Y0: np.ndarray,
    *,
    huber_px: float = 2.0,
    max_iterations: int = 60,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Levenberg–Marquardt refinement of ``(X, Y)`` on reprojection residuals."""
    X, Y = X0.copy(), Y0.copy()
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = dist_coeffs
    lam = 1e-3
    eps = 1e-6
    history: list[float] = []

    def cost_of(res: np.ndarray, w: np.ndarray) -> float:
        return float(np.sum(w * res * res))

    res = stacked_residuals(views, X, Y, K, D)
    w = _huber_weights(res, huber_px)
    cost = cost_of(res, w)
    history.append(_rms_from_residuals(res))
    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        # Numeric Jacobian around the current linearisation point (params = 0).
        J = np.zeros((res.size, 12))
        for k in range(12):
            dp = np.zeros(12)
            dp[k] = eps
            Xp, Yp = _apply_params(dp, X, Y)
            Xm, Ym = _apply_params(-dp, X, Y)
            J[:, k] = (
                stacked_residuals(views, Xp, Yp, K, D) - stacked_residuals(views, Xm, Ym, K, D)
            ) / (2.0 * eps)
        JtW = J.T * w
        H = JtW @ J
        g = JtW @ res
        improved = False
        for _ in range(12):
            A = H + lam * np.diag(np.diag(H) + 1e-12)
            try:
                delta = -np.linalg.solve(A, g)  # Gauss-Newton step on r(d) ~ r0 + J d
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            Xn, Yn = _apply_params(delta, X, Y)
            res_n = stacked_residuals(views, Xn, Yn, K, D)
            w_n = _huber_weights(res_n, huber_px)
            cost_n = cost_of(res_n, w_n)
            if cost_n < cost:
                X, Y, res, w, cost = Xn, Yn, res_n, w_n, cost_n
                lam = max(lam / 3.0, 1e-9)
                improved = True
                history.append(_rms_from_residuals(res))
                if float(np.linalg.norm(delta)) < tolerance:
                    converged = True
                break
            lam *= 10.0
        if not improved or converged:
            converged = converged or not improved
            break
    info = {
        "iterations": iterations,
        "converged": bool(converged),
        "huber_px": huber_px,
        "rms_px_history": [round(h, 5) for h in history],
        "final_rms_px": _rms_from_residuals(res),
    }
    return X, Y, info


# -------------------------------------------------------------------- top level


def _view_report(view, X, Y, K, D) -> dict:
    res = view_residuals(view, X, Y, K, D)
    norms = np.linalg.norm(res, axis=1)
    report = {
        "view": view.name,
        "corners": len(view.object_points),
        "rms_px": float(math.sqrt(np.mean(norms**2))),
        "max_px": float(np.max(norms)),
    }
    if view.T_cam_board_pnp is not None:
        pred = predicted_T_cam_board(X, view.T_base_link, Y)
        r_deg, t_m = pose_error(pred, view.T_cam_board_pnp)
        report["pnp_rotation_gap_deg"] = r_deg
        report["pnp_translation_gap_mm"] = t_m * 1000.0
        report["board_distance_m"] = float(view.T_cam_board_pnp[2, 3])
    return report


def motion_diversity(views: list[HandEyeView]) -> dict:
    """Rotation/translation span of the link poses; small spans make X unobservable."""
    if len(views) < 2:
        return {"views": len(views), "max_pairwise_rotation_deg": 0.0, "translation_span_m": 0.0}
    Ts = [v.T_base_link for v in views]
    max_rot = 0.0
    axes = []
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            dR = Ts[i][:3, :3].T @ Ts[j][:3, :3]
            ang = rotation_angle_deg(dR)
            max_rot = max(max_rot, ang)
            if ang > 1e-3:
                w = so3_log(dR)
                axes.append(w / np.linalg.norm(w))
    positions = np.array([T[:3, 3] for T in Ts])
    span = float(np.max(np.ptp(positions, axis=0))) if len(positions) else 0.0
    axis_rank = int(np.linalg.matrix_rank(np.array(axes), tol=0.2)) if axes else 0
    return {
        "views": len(views),
        "max_pairwise_rotation_deg": float(max_rot),
        "rotation_axis_rank": axis_rank,
        "translation_span_m": span,
    }


def solve_hand_eye(
    views: list[HandEyeView],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    *,
    huber_px: float = 2.0,
    max_view_rms_px: float = 3.0,
    min_views: int = 8,
    max_rejection_rounds: int = 5,
    leave_one_out: bool = True,
    method: str = "reprojection",
) -> HandEyeSolution:
    """Full pipeline: PnP → closed-form init → refine → outlier rejection → LOO.

    ``method="reprojection"`` (default) refines on corner reprojection error after the
    Kronecker closed-form initialisation, with IPPE flip resolution and per-view outlier
    rejection.  ``method="robotcamcalib"`` runs the verbatim RobotCamCalib solver
    (:mod:`dexmate_calib.extrinsics.robotcamcalib`) on the same detections and FK poses:
    bundle PnP (ITERATIVE + LM), alternating LS init, pose-residual Gauss-Newton with
    Huber; no flip resolution or rejection rounds.  Per-view reprojection statistics are
    reported for both.
    """
    if method not in ("reprojection", "robotcamcalib"):
        raise ValueError("method must be 'reprojection' or 'robotcamcalib'")
    if method == "robotcamcalib":
        return _solve_hand_eye_robotcamcalib(
            views, camera_matrix, dist_coeffs, min_views=min_views, leave_one_out=leave_one_out
        )
    if len(views) < min_views:
        raise ValueError(f"Need at least {min_views} views, got {len(views)}")
    K = np.asarray(camera_matrix, dtype=np.float64)
    D = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)

    rejected: list[dict] = []
    active: list[HandEyeView] = []
    for v in views:
        try:
            if v.T_cam_board_pnp is None:
                v.T_cam_board_candidates = pnp_board_pose_candidates(
                    v.object_points, v.image_points, K, D
                )
                v.T_cam_board_pnp = v.T_cam_board_candidates[0]
            elif v.T_cam_board_candidates is None:
                v.T_cam_board_candidates = [v.T_cam_board_pnp]
            active.append(v)
        except ValueError as exc:
            rejected.append({"view": v.name, "reason": f"pnp_failed: {exc}"})
    if len(active) < min_views:
        raise ValueError(f"Only {len(active)} views survived PnP; need {min_views}")

    diversity = motion_diversity(active)

    # Closed-form initialisation; then resolve planar PnP flip ambiguities against the
    # chain prediction and re-initialise until the candidate choice is stable.
    flips_total = 0
    for _ in range(4):
        candidates = initial_hand_eye(active)
        scored = []
        for name, X, Y in candidates:
            rms = _rms_from_residuals(stacked_residuals(active, X, Y, K, D))
            scored.append((rms, name, X, Y))
        scored.sort(key=lambda item: item[0])
        init_rms, init_method, X, Y = scored[0]
        flips = _resolve_pnp_ambiguities(active, X, Y)
        flips_total += flips
        if flips == 0:
            break
    init_rot_gap, init_trans_gap = _pose_chain_rms(active, X, Y)
    initialisation = {
        "method": init_method,
        "rms_px": init_rms,
        "mean_pnp_rotation_gap_deg": init_rot_gap,
        "mean_pnp_translation_gap_mm": init_trans_gap * 1000.0,
        "pnp_flips_resolved": flips_total,
        "candidates": {name: rms for rms, name, _, _ in scored},
    }

    def refine(subset, X_init, Y_init, **kw):
        return refine_hand_eye(subset, K, D, X_init, Y_init, huber_px=huber_px, **kw)

    refine_info: dict = {}
    for round_index in range(max_rejection_rounds + 1):
        X, Y, refine_info = refine(active, X, Y)
        refine_info.setdefault("method", method)
        rms_per_view = np.array([_view_report(v, X, Y, K, D)["rms_px"] for v in active])
        median = float(np.median(rms_per_view))
        mad = float(np.median(np.abs(rms_per_view - median))) * 1.4826
        threshold = max(max_view_rms_px, median + 3.0 * mad)
        bad = [i for i, r in enumerate(rms_per_view) if r > threshold]
        if not bad or len(active) - len(bad) < min_views or round_index == max_rejection_rounds:
            break
        # Drop the worst views only, at most 20% per round, so we don't overshoot.
        bad.sort(key=lambda i: -rms_per_view[i])
        bad = bad[: max(1, len(active) // 5)]
        for i in sorted(bad, reverse=True):
            v = active.pop(i)
            rejected.append(
                {
                    "view": v.name,
                    "reason": "reprojection_outlier",
                    "rms_px": float(rms_per_view[i]),
                    "threshold_px": float(threshold),
                }
            )

    reports = [_view_report(v, X, Y, K, D) for v in active]
    final_rms = _rms_from_residuals(stacked_residuals(active, X, Y, K, D))

    diagnostics = {"motion_diversity": diversity}
    if leave_one_out and len(active) > min_views:
        loo_rot, loo_trans, loo_rms = [], [], []
        for i in range(len(active)):
            subset = active[:i] + active[i + 1 :]
            Xi, Yi, _ = refine(subset, X, Y, max_iterations=30)
            r_deg, t_m = pose_error(X, Xi)
            loo_rot.append(r_deg)
            loo_trans.append(t_m)
            held = _view_report(active[i], Xi, Yi, K, D)
            loo_rms.append(held["rms_px"])
        diagnostics["leave_one_out"] = {
            "T_base_cam_rotation_spread_deg": {
                "median": float(np.median(loo_rot)),
                "max": float(np.max(loo_rot)),
            },
            "T_base_cam_translation_spread_mm": {
                "median": float(np.median(loo_trans) * 1000.0),
                "max": float(np.max(loo_trans) * 1000.0),
            },
            "held_out_view_rms_px": {
                "median": float(np.median(loo_rms)),
                "p90": float(np.percentile(loo_rms, 90)),
                "max": float(np.max(loo_rms)),
            },
        }

    return HandEyeSolution(
        T_base_cam=X,
        T_link_board=Y,
        rms_px=final_rms,
        view_reports=reports,
        inlier_views=[v.name for v in active],
        rejected_views=rejected,
        initialisation=initialisation,
        refinement=refine_info,
        diagnostics=diagnostics,
    )


def _solve_hand_eye_robotcamcalib(
    views: list[HandEyeView],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None,
    *,
    min_views: int,
    leave_one_out: bool,
) -> HandEyeSolution:
    """RobotCamCalib pipeline on dexmate-calib inputs; see ``extrinsics/robotcamcalib.py``."""
    from dexmate_calib.extrinsics import robotcamcalib as rcc

    K = np.asarray(camera_matrix, dtype=np.float64)
    D = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    rejected: list[dict] = []
    active: list[HandEyeView] = []
    for v in views:
        try:
            v.T_cam_board_pnp = rcc.estimate_board_pose_bundle_pnp(
                v.object_points, v.image_points, K, D
            )
            v.T_cam_board_candidates = [v.T_cam_board_pnp]
            active.append(v)
        except (RuntimeError, ValueError, _cv2().error) as exc:
            rejected.append({"view": v.name, "reason": f"pnp_failed: {exc}"})
    if len(active) < min_views:
        raise ValueError(f"Only {len(active)} views survived PnP; need {min_views}")

    A = [v.T_base_link for v in active]
    B = [v.T_cam_board_pnp for v in active]
    # Report the LS initialisation exactly as calibrate_cammount_and_tag_prob computes it
    # (X_TagmountTag = I -> Y -> X -> Y); the solver recomputes it internally.
    Xt = np.eye(4)
    Yc = rcc._solve_Y_given_X(A, B, Xt)
    Xt = rcc._solve_X_given_Y(A, B, Yc)
    Yc = rcc._solve_Y_given_X(A, B, Xt)
    init_X, init_Y = Yc, Xt  # X_CammountCam = T_base_cam, X_TagmountTag = T_link_board
    init_rms = _rms_from_residuals(stacked_residuals(active, init_X, init_Y, K, D))
    init_rot_gap, init_trans_gap = _pose_chain_rms(active, init_X, init_Y)

    X, Y, info = rcc.solve_third_view(A, B)
    reports = [_view_report(v, X, Y, K, D) for v in active]
    final_rms = _rms_from_residuals(stacked_residuals(active, X, Y, K, D))
    diagnostics = {"motion_diversity": motion_diversity(active)}
    if leave_one_out and len(active) > min_views:
        loo_rot, loo_trans, loo_rms = [], [], []
        for i in range(len(active)):
            subset = active[:i] + active[i + 1 :]
            Xi, Yi, _ = rcc.solve_third_view(
                [v.T_base_link for v in subset], [v.T_cam_board_pnp for v in subset]
            )
            r_deg, t_m = pose_error(X, Xi)
            loo_rot.append(r_deg)
            loo_trans.append(t_m)
            loo_rms.append(_view_report(active[i], Xi, Yi, K, D)["rms_px"])
        diagnostics["leave_one_out"] = {
            "T_base_cam_rotation_spread_deg": {
                "median": float(np.median(loo_rot)),
                "max": float(np.max(loo_rot)),
            },
            "T_base_cam_translation_spread_mm": {
                "median": float(np.median(loo_trans) * 1000.0),
                "max": float(np.max(loo_trans) * 1000.0),
            },
            "held_out_view_rms_px": {
                "median": float(np.median(loo_rms)),
                "p90": float(np.percentile(loo_rms, 90)),
                "max": float(np.max(loo_rms)),
            },
        }
    return HandEyeSolution(
        T_base_cam=X,
        T_link_board=Y,
        rms_px=final_rms,
        view_reports=reports,
        inlier_views=[v.name for v in active],
        rejected_views=rejected,
        initialisation={
            "method": "robotcamcalib_alternating_ls",
            "rms_px": init_rms,
            "mean_pnp_rotation_gap_deg": init_rot_gap,
            "mean_pnp_translation_gap_mm": init_trans_gap * 1000.0,
            "pnp_flips_resolved": 0,
        },
        refinement={
            "method": "robotcamcalib",
            "iterations": int(info["iters"]),
            "converged": bool(info["iters"] < 200),
            "closure_rot_deg": {
                "mean": info["rot_err_deg_mean"],
                "median": info["rot_err_deg_med"],
                "max": info["rot_err_deg_max"],
            },
            "closure_trans_mm": {
                "mean": info["trans_err_mean"] * 1000.0,
                "median": info["trans_err_med"] * 1000.0,
                "max": info["trans_err_max"] * 1000.0,
            },
            "final_rms_px": final_rms,
        },
        diagnostics=diagnostics,
    )
