"""Verbatim port of the RobotCamCalib hand-eye solver (``method="robotcamcalib"``).

Everything between the ``BEGIN VERBATIM`` / ``END VERBATIM`` markers is copied from
``RobotCamCalib/extr_calib.py`` (Lie helpers, ``_wahba``, ``_solve_Y_given_X``,
``_solve_X_given_Y``, ``block_diag``, ``calibrate_cammount_and_tag_prob``) and
``RobotCamCalib/apriltag_board.py`` (the bundle-PnP solve) without algorithmic changes,
so results are numerically the same as that repository would produce.  Only formatting
that ruff enforces (whitespace, unused names) was touched.

What is adapted here is the *input side*: correspondences come from dexmate-calib board
profiles/detectors (ChArUco or AprilTag), ``K``/``D`` come from the session's camera
calibration (Kinect factory intrinsics; RobotCamCalib passed ``D=None`` because its
RealSense stream was already rectified), and ``X_WorldTagmount`` comes from Dexmate
URDF forward kinematics with ``X_WorldCammount = I`` (camera mount = robot base).

Role mapping (RobotCamCalib third-view example -> dexmate-calib)::

    A_i = X_WorldCammount^-1 X_WorldTagmount = T_base_link_i
    B_i = X_CamTag                          = T_cam_board_i
    X_TagmountTag                           = T_link_board (Y in handeye.py)
    X_CammountCam                           = T_base_cam   (X in handeye.py)

Not part of RobotCamCalib and therefore *not* applied when this method is selected:
IPPE flip resolution, per-view outlier rejection rounds and the Kronecker init.  The
leave-one-out diagnostic (if requested) simply re-runs this solver on subsets.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# BEGIN VERBATIM (RobotCamCalib/extr_calib.py)
# ---------------------------------------------------------------------------


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def so3_log(R):
    # numerically stable log
    cos_th = (np.trace(R) - 1.0) / 2.0
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = np.arccos(cos_th)
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0])
    w_hat = (R - R.T) / (2 * np.sin(th))
    return np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) * th


def se3_exp(dw, dq):
    R = so3_exp(dw)
    # first-order adequate for small steps; Jacobian could be added if needed
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = dq
    return T


def se3_log(T):
    w = so3_log(T[:3, :3])
    q = T[:3, 3]
    return w, q


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# --- Huber on Mahalanobis^2 ---
def huber_weight_mahal(r2, delta2):
    w = np.ones_like(r2)
    mask = r2 > delta2
    w[mask] = np.sqrt(delta2 / r2[mask])
    return w


def compose_left(T, xi):
    # left-multiplicative update: T_new = Exp(xi) * T
    dw = xi[:3]
    dq = xi[3:]
    return se3_exp(dw, dq) @ T


def pack_r(w, p):
    return np.hstack([w, p])  # 6,


def _wahba(Rs_src, Rs_tgt):
    H = np.zeros((3, 3))
    for Rsrc, Rtgt in zip(Rs_src, Rs_tgt):
        H += Rtgt @ Rsrc.T
    U, _S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _solve_Y_given_X(A_list, B_list, X):
    RX = X[:3, :3]
    tX = X[:3, 3]
    R_targets = []
    R_sources = []
    for A, B in zip(A_list, B_list):
        R_targets.append(A[:3, :3] @ RX)
        R_sources.append(B[:3, :3])
    RY = _wahba(R_sources, R_targets)
    # t: A tX + a ≈ RY b + tY
    M = []
    v = []
    for A, B in zip(A_list, B_list):
        M.append(np.eye(3))
        v.append((A[:3, :3] @ tX + A[:3, 3]) - (RY @ B[:3, 3]))
    M = np.vstack(M)
    v = np.hstack(v)
    tY, *_ = np.linalg.lstsq(M, v, rcond=None)
    T = np.eye(4)
    T[:3, :3] = RY
    T[:3, 3] = tY
    return T


def _solve_X_given_Y(A_list, B_list, Y):
    RY = Y[:3, :3]
    tY = Y[:3, 3]
    R_sources = []
    R_targets = []
    for A, B in zip(A_list, B_list):
        R_sources.append(A[:3, :3])
        R_targets.append(RY @ B[:3, :3])
    RX = _wahba(R_sources, R_targets)
    # A tX + a ≈ RY b + tY
    M = []
    v = []
    for A, B in zip(A_list, B_list):
        M.append(A[:3, :3])
        v.append((RY @ B[:3, 3] + tY) - A[:3, 3])
    M = np.vstack(M)
    v = np.hstack(v)
    tX, *_ = np.linalg.lstsq(M, v, rcond=None)
    T = np.eye(4)
    T[:3, :3] = RX
    T[:3, 3] = tX
    return T


def block_diag(blocks):
    # simple block-diagonal constructor
    r = sum(b.shape[0] for b in blocks)
    c = sum(b.shape[1] for b in blocks)
    out = np.zeros((r, c))
    i = j = 0
    for B in blocks:
        rr, cc = B.shape
        out[i : i + rr, j : j + cc] = B
        i += rr
        j += cc
    return out


# ---------- PROBABILISTIC MLE (Config-3) with GN + numerical Jacobians ----------
def calibrate_cammount_and_tag_prob(
    X_CamTag_list: np.ndarray,  # B_i (n,4,4)
    X_WorldCammount_list: np.ndarray,  # (n,4,4)
    X_WorldTagmount_list: np.ndarray,  # (n,4,4)
    Sigma_w_list: list[np.ndarray] | None = None,  # rot cov of B_i
    Sigma_p_list: list[np.ndarray] | None = None,  # trans cov of B_i
    max_iters: int = 200,
    huber_delta_rot_deg: float = 3.0,
    huber_delta_trans: float = 0.01,
    eps_dx: float = 1e-8,
    eps_stop_deg: float = 1e-6,
    eps_stop_trans: float = 1e-8,
    damping: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = X_CamTag_list.shape[0]
    assert X_CamTag_list.shape == (n, 4, 4)
    assert X_WorldCammount_list.shape == (n, 4, 4)
    assert X_WorldTagmount_list.shape == (n, 4, 4)

    # Build A_i (trusted) and B_i (noisy)
    A_list = [inv_T(X_WorldCammount_list[i]) @ X_WorldTagmount_list[i] for i in range(n)]
    B_list = [X_CamTag_list[i] for i in range(n)]

    # Cov defaults
    if Sigma_w_list is None:
        Sigma_w_list = [(np.radians(3.0) ** 2) * np.eye(3) for _ in range(n)]
    if Sigma_p_list is None:
        Sigma_p_list = [(0.01**2) * np.eye(3) for _ in range(n)]
    SwI = [np.linalg.inv(Sw) for Sw in Sigma_w_list]
    SpI = [np.linalg.inv(Sp) for Sp in Sigma_p_list]

    # Simple LS init (same as before)
    X = np.eye(4)
    Y = _solve_Y_given_X(A_list, B_list, X)
    X = _solve_X_given_Y(A_list, B_list, Y)
    Y = _solve_Y_given_X(A_list, B_list, X)

    # Robust params
    delta_w2 = (np.radians(huber_delta_rot_deg)) ** 2
    delta_p2 = (huber_delta_trans) ** 2

    # parameter vector is [xi_X (6), xi_Y (6)] but we update via left-mult on (X,Y) directly
    it = 0
    for it in range(max_iters):
        # assemble residuals and block-diagonal weight inverses
        r_all = []
        W_inv_blocks = []
        for i, (A, B, Sw, Sp) in enumerate(zip(A_list, B_list, SwI, SpI)):
            # residual transform: E_i = X^{-1} A^{-1} Y B  (should be ~ Identity)
            Ei = inv_T(X) @ inv_T(A) @ Y @ B
            w_i, p_i = se3_log(Ei)

            # Mahalanobis^2 per part
            r2w = w_i.T @ Sw @ w_i
            r2p = p_i.T @ Sp @ p_i
            ww = huber_weight_mahal(np.array([r2w]), delta_w2)[0]
            wp = huber_weight_mahal(np.array([r2p]), delta_p2)[0]

            # stack 6x1 residual (apply sqrt weights inside W_inv later)
            r_i = pack_r(w_i, p_i)
            r_all.append(r_i)

            # Build W^{-1} per sample (6x6), using robust weights
            # We put W^{-1} = diag( ww*Sw, wp*Sp )
            Wi_inv = np.block([[ww * Sw, np.zeros((3, 3))], [np.zeros((3, 3)), wp * Sp]])
            W_inv_blocks.append(Wi_inv)

        r_all = np.hstack(r_all)  # (6n,)
        W_inv = block_diag(W_inv_blocks)  # (6n x 6n)

        # Numerically build J (6n x 12): columns for [δx(6), δy(6)]
        J = np.zeros((6 * n, 12))

        # finite-diff step size
        h = eps_dx

        # columns 0..5: effect of δx on residuals
        for k in range(6):
            xi = np.zeros(6)
            xi[k] = h
            Xp = compose_left(X, xi)  # left-mult perturb
            col = []
            for i, (A, B) in enumerate(zip(A_list, B_list)):
                Ei_p = inv_T(Xp) @ inv_T(A) @ Y @ B
                w_p, p_p = se3_log(Ei_p)
                Ei = inv_T(X) @ inv_T(A) @ Y @ B
                w_0, p_0 = se3_log(Ei)
                dr = pack_r(w_p - w_0, p_p - p_0) / h
                col.append(dr)
            J[:, k] = np.hstack(col)

        # columns 6..11: effect of δy on residuals
        for k in range(6):
            xi = np.zeros(6)
            xi[k] = h
            Yp = compose_left(Y, xi)
            col = []
            for i, (A, B) in enumerate(zip(A_list, B_list)):
                Ei_p = inv_T(X) @ inv_T(A) @ Yp @ B
                w_p, p_p = se3_log(Ei_p)
                Ei = inv_T(X) @ inv_T(A) @ Y @ B
                w_0, p_0 = se3_log(Ei)
                dr = pack_r(w_p - w_0, p_p - p_0) / h
                col.append(dr)
            J[:, 6 + k] = np.hstack(col)

        # Solve Gauss–Newton normal equations with damping (Levenberg)
        # (J^T W^{-1} J + λI) Δ = - J^T W^{-1} r
        JT_Wi = J.T @ W_inv
        H = JT_Wi @ J
        g = JT_Wi @ r_all
        H += damping * np.eye(12)
        try:
            delta = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(H) @ g

        # split and update
        dx = delta[:6]
        dy = delta[6:]
        X = compose_left(X, dx)
        Y = compose_left(Y, dy)

        if verbose:
            rm = np.sqrt(np.mean(r_all**2))
            print(
                f"[it {it:02d}] |delta_x|={np.linalg.norm(dx):.2e}, |delta_y|={np.linalg.norm(dy):.2e}, rmse={rm:.3e}"
            )

        # stopping
        if (
            np.linalg.norm(dx[:3]) < np.radians(eps_stop_deg)
            and np.linalg.norm(dx[3:]) < eps_stop_trans
            and np.linalg.norm(dy[:3]) < np.radians(eps_stop_deg)
            and np.linalg.norm(dy[3:]) < eps_stop_trans
        ):
            break

    # report residuals on Ai X ≈ Y Bi  (identity if perfect)
    rot_err_deg, trans_err = [], []
    for A, B in zip(A_list, B_list):
        E = inv_T(A @ X) @ (Y @ B)
        wE, qE = se3_log(E)
        rot_err_deg.append(np.degrees(np.linalg.norm(wE)))
        trans_err.append(np.linalg.norm(qE))

    info = {
        "iters": it + 1,
        "rot_err_deg_mean": float(np.mean(rot_err_deg)),
        "rot_err_deg_med": float(np.median(rot_err_deg)),
        "rot_err_deg_max": float(np.max(rot_err_deg)),
        "trans_err_mean": float(np.mean(trans_err)),
        "trans_err_med": float(np.median(trans_err)),
        "trans_err_max": float(np.max(trans_err)),
    }

    X_CammountCam = Y
    X_TagmountTag = X
    return X_CammountCam, X_TagmountTag, info


# ---------------------------------------------------------------------------
# BEGIN VERBATIM (RobotCamCalib/apriltag_board.py::estimate_board_pose_bundle_pnp),
# with the correspondence collection replaced by dexmate-calib detections.
# ---------------------------------------------------------------------------


def estimate_board_pose_bundle_pnp(obj_pts, img_pts, K, D=None):
    """Estimate X_CamBoard from all detected board corners with one bundle PnP solve."""
    import cv2

    obj_pts = np.asarray(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    dist_coeffs = (
        np.zeros((5, 1), dtype=np.float64) if D is None else np.asarray(D, dtype=np.float64)
    )
    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        np.asarray(K, dtype=np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed for AprilTag board.")

    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts,
        img_pts,
        np.asarray(K, dtype=np.float64),
        dist_coeffs,
        rvec,
        tvec,
    )
    R_cam_board, _ = cv2.Rodrigues(rvec)
    X_cam_board = np.eye(4)
    X_cam_board[:3, :3] = R_cam_board
    X_cam_board[:3, 3] = tvec.reshape(3)
    return X_cam_board


# ---------------------------------------------------------------------------
# END VERBATIM
# ---------------------------------------------------------------------------


def solve_third_view(
    T_base_link_list: list[np.ndarray],
    T_cam_board_list: list[np.ndarray],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run RobotCamCalib's third-view configuration on dexmate-calib inputs.

    Returns ``(T_base_cam, T_link_board, info)``.  ``kwargs`` are forwarded unchanged to
    :func:`calibrate_cammount_and_tag_prob` (``huber_delta_rot_deg``, ``huber_delta_trans``,
    ``max_iters`` ...).
    """
    n = len(T_base_link_list)
    if n != len(T_cam_board_list) or n == 0:
        raise ValueError("Need the same non-zero number of robot and camera poses")
    X_CamTag = np.asarray(T_cam_board_list, dtype=np.float64).reshape(n, 4, 4)
    X_WorldCammount = np.repeat(np.eye(4)[None], n, axis=0)  # camera mount = robot base
    X_WorldTagmount = np.asarray(T_base_link_list, dtype=np.float64).reshape(n, 4, 4)
    X_CammountCam, X_TagmountTag, info = calibrate_cammount_and_tag_prob(
        X_CamTag, X_WorldCammount, X_WorldTagmount, **kwargs
    )
    return X_CammountCam, X_TagmountTag, info
