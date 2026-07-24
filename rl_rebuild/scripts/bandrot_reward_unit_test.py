# BandRot reward unit test (OFFLINE, pure numpy; no sim, no torch, no Isaac boot).
#
# Proves the anti-jitter BAND rotation reward in
#   rl_rebuild/tasks/inhand_rotate/sharpa_wave_graspxl_bandrot.py  (_BandRotMixin._get_rewards)
# has the intended properties, by mirroring its per-step math EXACTLY in numpy:
#
#   ema     <- ema_alpha*ema + (1-ema_alpha)*omega_signed
#   mag     =  clip((|ema| - floor) / (omax - floor), 0, 1)
#   band    =  band_scale * mag * sign(ema)
#   r_rot   =  band * clip(1 + gain*max(0, mean_signed_revs_per_ep), 1, cap)
#   cum    <-  cum + ema*dt
#   level   =  floor(cum / sparse_theta) ;  gained = max(0, level - ratchet)
#   sparse  =  sparse_bonus * gained ;  ratchet <- max(ratchet, level)
#
# Checks (must all PASS):
#   (1) band == 0 for a CONSTANT omega = 0.037 rad/s (the rewarded-jitter attractor) at every step.
#   (2) band > 0 AND strictly increasing for constant omega in [0.15, 3.0].
#   (3) a +-0.5 rad/s SQUARE-WAVE fed through the EMA yields band ~ 0 (jitter cancels).
#   (4) sparse bonus fires on NET cum-angle progress (constant +omega) and NEVER on oscillation.
#   (5) adaptive_mult == 1 at 0 revs, increases with mean revs, clamps to cap.
#
# Run:  <sharpa-python> rl_rebuild/scripts/bandrot_reward_unit_test.py   (plain numpy is enough)
import math
import numpy as np

# ---- config (mirror sharpa_wave_graspxl_bandrot.SharpaWaveGraspXLBandRotCfg) ----
BAND_SCALE = 3.0
BAND_OMEGA_FLOOR = 0.15
BAND_OMEGA_MAX = 3.0
EMA_ALPHA = 0.9
ADAPTIVE_WROT_GAIN = 0.5
ADAPTIVE_WROT_CAP = 3.0
SPARSE_BONUS = 5.0
SPARSE_THETA = 0.3
DT = 12.0 / 240.0            # step_dt = decimation(12) * sim.dt(1/240) = 0.05 s
JITTER = 0.037               # the observed "rewarded jitter attractor" signed speed


def band_of_ema(ema):
    """Band value for a given (already-smoothed) signed speed ema. Mirrors the env exactly."""
    mag = np.clip((abs(ema) - BAND_OMEGA_FLOOR) / max(BAND_OMEGA_MAX - BAND_OMEGA_FLOOR, 1e-6), 0.0, 1.0)
    return BAND_SCALE * mag * np.sign(ema)


def adaptive_mult(mean_signed_revs_per_ep):
    return float(np.clip(1.0 + ADAPTIVE_WROT_GAIN * max(0.0, mean_signed_revs_per_ep), 1.0, ADAPTIVE_WROT_CAP))


def run_traj(omega_seq, mean_revs=0.0):
    """Simulate the per-step band pipeline over a sequence of RAW omega_signed values.
    Returns dict with per-step arrays band, r_rot, sparse, and the final cum_angle."""
    ema = 0.0
    cum = 0.0
    ratchet = math.floor(cum / SPARSE_THETA)
    mult = adaptive_mult(mean_revs)
    bands, r_rots, sparses, emas = [], [], [], []
    for w in omega_seq:
        ema = EMA_ALPHA * ema + (1.0 - EMA_ALPHA) * w
        b = band_of_ema(ema)
        r = b * mult
        # integrate only the SUPRA-FLOOR signed speed (mirrors _BandRotMixin): sub-floor drift/jitter
        # contributes exactly 0 to cum_angle, so it can never leak a sparse bonus.
        omega_eff = np.sign(ema) * max(abs(ema) - BAND_OMEGA_FLOOR, 0.0)
        cum = cum + omega_eff * DT
        level = math.floor(cum / SPARSE_THETA)
        gained = max(0, level - ratchet)
        sp = SPARSE_BONUS * gained
        ratchet = max(ratchet, level)
        bands.append(b); r_rots.append(r); sparses.append(sp); emas.append(ema)
    return dict(band=np.array(bands), r_rot=np.array(r_rots), sparse=np.array(sparses),
                ema=np.array(emas), cum=cum)


def main():
    checks = []
    print("==== BANDROT REWARD UNIT TEST (offline numpy) ====")
    print(f"  cfg: band_scale={BAND_SCALE} floor={BAND_OMEGA_FLOOR} max={BAND_OMEGA_MAX} "
          f"ema_alpha={EMA_ALPHA} theta={SPARSE_THETA} bonus={SPARSE_BONUS} dt={DT}")

    # (1) constant jitter 0.037 -> band == 0 at every step ---------------------
    T = 400
    jit = run_traj([JITTER] * T)
    max_abs_band_jit = float(np.max(np.abs(jit["band"])))
    max_abs_ema_jit = float(np.max(np.abs(jit["ema"])))
    print(f"\n(1) constant omega={JITTER}: max|ema|={max_abs_ema_jit:.4f} (< floor {BAND_OMEGA_FLOOR}), "
          f"max|band|={max_abs_band_jit:.6f}, total_sparse={jit['sparse'].sum():.1f}")
    checks.append(("jitter 0.037 -> band == 0 (every step)", max_abs_band_jit == 0.0))
    checks.append(("jitter 0.037 -> sparse == 0", jit["sparse"].sum() == 0.0))

    # (2) band > 0 and strictly increasing for omega in [0.15, 3.0] ------------
    omegas = [0.20, 0.5, 1.0, 2.0, 3.0]
    steady_bands = [band_of_ema(w) for w in omegas]  # steady-state ema == constant w
    print("\n(2) steady-state band vs omega (in [floor, max]):")
    for w, b in zip(omegas, steady_bands):
        print(f"      omega={w:4.2f} -> band={b:.4f}")
    print(f"      omega=floor({BAND_OMEGA_FLOOR}) -> band={band_of_ema(BAND_OMEGA_FLOOR):.4f} (boundary == 0)")
    checks.append(("band > 0 for all omega in [0.20, 3.0]", all(b > 0.0 for b in steady_bands)))
    checks.append(("band strictly increasing in omega", all(np.diff(steady_bands) > 0.0)))
    checks.append(("band saturates to band_scale at omega_max", abs(steady_bands[-1] - BAND_SCALE) < 1e-9))
    checks.append(("band == 0 exactly at the floor", band_of_ema(BAND_OMEGA_FLOOR) == 0.0))

    # (3) +-0.5 square-wave through the EMA -> band ~ 0 ------------------------
    sq = [0.5 if (t % 2 == 0) else -0.5 for t in range(T)]
    osc = run_traj(sq)
    max_abs_band_osc = float(np.max(np.abs(osc["band"][50:])))  # ignore warm-up
    max_abs_ema_osc = float(np.max(np.abs(osc["ema"][50:])))
    print(f"\n(3) +-0.5 square-wave: steady max|ema|={max_abs_ema_osc:.4f} (< floor), "
          f"max|band|={max_abs_band_osc:.6f}, mean_band={osc['band'][50:].mean():.6f}")
    checks.append(("+-0.5 oscillation -> band ~ 0 (|band| < 1e-6)", max_abs_band_osc < 1e-6))

    # (4) sparse bonus: fires on NET progress, NEVER on oscillation ------------
    prog = run_traj([0.5] * T)  # constant +0.5 rad/s -> steady net rotation
    n_bonus_prog = int(round(prog["sparse"].sum() / SPARSE_BONUS))
    expected_levels = math.floor(prog["cum"] / SPARSE_THETA)
    n_bonus_osc = int(round(osc["sparse"].sum() / SPARSE_BONUS))
    print(f"\n(4) sparse bonus:")
    print(f"      constant +0.5: cum_angle={prog['cum']:.3f} rad, bonuses fired={n_bonus_prog} "
          f"(expected floor(cum/theta)={expected_levels}), total_sparse={prog['sparse'].sum():.1f}")
    print(f"      +-0.5 oscillation: cum_angle={osc['cum']:.4f} rad, bonuses fired={n_bonus_osc}, "
          f"total_sparse={osc['sparse'].sum():.1f}")
    checks.append(("net +progress fires >=1 sparse bonus", n_bonus_prog >= 1))
    checks.append(("sparse count == floor(cum_angle/theta) on net progress", n_bonus_prog == expected_levels))
    checks.append(("oscillation fires 0 sparse bonuses", osc["sparse"].sum() == 0.0))

    # (5) adaptive_mult monotonic + clamped -----------------------------------
    m0 = adaptive_mult(0.0); m2 = adaptive_mult(2.0); m_big = adaptive_mult(100.0)
    print(f"\n(5) adaptive_mult: revs=0 -> {m0:.3f}, revs=2 -> {m2:.3f}, revs=100 -> {m_big:.3f} (cap {ADAPTIVE_WROT_CAP})")
    checks.append(("adaptive_mult == 1.0 at 0 revs", abs(m0 - 1.0) < 1e-9))
    checks.append(("adaptive_mult increases with revs", m2 > m0))
    checks.append(("adaptive_mult clamps to cap", abs(m_big - ADAPTIVE_WROT_CAP) < 1e-9))
    # adaptivity actually raises r_rot for the same spin
    r_lo = band_of_ema(0.5) * adaptive_mult(0.0)
    r_hi = band_of_ema(0.5) * adaptive_mult(2.0)
    checks.append(("adaptive_mult raises r_rot for same spin", r_hi > r_lo))

    # ---- summary ----
    print("\n==== RESULTS ====")
    ok = True
    for desc, c in checks:
        print(f"   [{'PASS' if c else 'FAIL'}] {desc}")
        ok = ok and bool(c)
    print("\n==================== SUMMARY ====================")
    print(f"  OVERALL: {'PASS -> band reward has the intended anti-jitter properties' if ok else 'FAIL -> retune the band reward BEFORE training'}")
    print("================================================")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
