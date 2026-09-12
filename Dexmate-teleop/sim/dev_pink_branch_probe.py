"""Pink local-optimum branch probe (no Isaac, CPU only).

Round 10 observation: tiny differences in the starting configuration swing the
converged per-motion dev by tens of mm.  This probe quantifies that: for each
demo delta, run PinkVegaIK open-loop for 10s from home perturbed by uniform
noise of magnitude EPS, record the final EE error and final q, and cluster the
final q's to count distinct local-optimum branches.

Run from MagicDexMate/sim with .venv-isaac python.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

DELTAS = {  # chest-frame deltas, right hand; left mirrors y
    "fwd 25cm": np.array([0.25, 0.0, 0.0]),
    "lat 30cm": np.array([0.0, -0.30, 0.0]),
    "up 30cm": np.array([0.0, 0.0, 0.30]),
}
TICKS = 600  # 10s @ 60Hz: past convergence, isolates branch choice
SEEDS = 20
EPS_LEVELS = [0.02, 0.10]  # rad, uniform per-joint perturbation of home


def quat_wxyz(R):
    q = pin.Quaternion(R)
    return [q.w, q.x, q.y, q.z]


def make_targets(ik):
    """Targets computed from the *unperturbed* home FK: identical across seeds."""
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)
    chest = ik._chest
    cp, cq = chest.translation.copy(), quat_wxyz(chest.rotation)
    tgt = {}
    for h, sgn in (("right", 1.0), ("left", -1.0)):
        T0 = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
        tgt[h] = (T0.translation.copy(), quat_wxyz(T0.rotation), sgn)
    return cp, cq, tgt


def run_case(label, d_r, eps, seed):
    ik = PinkVegaIK(dt=1 / 60)
    cp, cq, base = make_targets(ik)
    tgt_pos = {}
    chest_R = ik._chest.rotation
    for h in ("right", "left"):
        t0p, t0q, sgn = base[h]
        d = d_r.copy()
        d[1] *= sgn
        tp = t0p + chest_R @ d
        ik.set_target_chest(h, cp, cq, tp, t0q)
        tgt_pos[h] = tp
    # perturb the start AFTER targets are fixed
    rng = np.random.default_rng(seed)
    q = ik.q_home.copy() + rng.uniform(-eps, eps, size=ik.q_home.shape)
    lo = ik.model.lowerPositionLimit + 1e-3
    hi = ik.model.upperPositionLimit - 1e-3
    q = np.clip(q, lo, hi)
    ik.config.q = q
    ik.config.update()
    for _ in range(TICKS):
        ik.solve()
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)
    errs = {h: np.linalg.norm(ik.ee_pos(h) - tgt_pos[h]) * 1000 for h in ("right", "left")}
    return errs, ik.config.q.copy()


def cluster(qs, tol=0.15):
    """Greedy clustering of final configurations by max-abs joint distance."""
    reps, counts = [], []
    for q in qs:
        for i, r in enumerate(reps):
            if np.max(np.abs(q - r)) < tol:
                counts[i] += 1
                break
        else:
            reps.append(q)
            counts.append(1)
    return counts


def main():
    for eps in EPS_LEVELS:
        print(f"\n===== eps = {eps:.2f} rad, {SEEDS} seeds, {TICKS} ticks =====")
        for label, d_r in DELTAS.items():
            R, L, qs = [], [], []
            for s in range(SEEDS):
                errs, q = run_case(label, d_r, eps, s)
                R.append(errs["right"])
                L.append(errs["left"])
                qs.append(q)
            counts = cluster(qs)
            print(
                f"  {label:>9s}: R min/med/max {np.min(R):6.1f}/{np.median(R):6.1f}/{np.max(R):6.1f}mm"
                f"  L {np.min(L):6.1f}/{np.median(L):6.1f}/{np.max(L):6.1f}mm"
                f"  branches={len(counts)} sizes={sorted(counts, reverse=True)}"
            )
    # reference: unperturbed home start
    print("\n===== reference: exact home start =====")
    for label, d_r in DELTAS.items():
        errs, _ = run_case(label, d_r, 0.0, 0)
        print(f"  {label:>9s}: R {errs['right']:6.1f}mm L {errs['left']:6.1f}mm")


if __name__ == "__main__":
    main()
