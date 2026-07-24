# Finger-GAITING reward unit test (OFFLINE, pure numpy; no sim, no torch, no Isaac boot).
#
# Proves the 5 gait terms in
#   rl_rebuild/tasks/inhand_rotate/sharpa_wave_graspxl_bandgait.py  (_BandGaitMixin._term_g{1..5}_*)
# have the intended shape, by mirroring their per-step math EXACTLY in numpy on 3 synthetic single-env
# trajectories:
#   GAITING : >=3 fingers cycle on/off (each releases >=k_off steps then re-contacts & sustains),
#             joints stay MID-range, sustained net rotation (omega>rot_thresh), object held.
#   ROLLING : thumb+middle planted the whole time, others off, joints drift to their LIMITS,
#             sustained net rotation, object held.  (the current 2-finger roll we want to beat)
#   FLUTTER : fingers toggle every 2 steps, NO net rotation (omega=0), object DROPPED (held=0).
#             (a degenerate "farm" attempt -- must earn ~0 for every term)
#
# PASS criteria (all must hold):
#   * G1..G5 each score  sum(GAITING) > sum(ROLLING)
#   * G1..G5 each score  |sum(FLUTTER)| ~ 0   (held/rotating gating kills it)
#
# Run:  <sharpa-python> rl_rebuild/scripts/gait_reward_unit_test.py   (plain numpy is enough)
import numpy as np

# ---- params: MIRROR SharpaWaveBandGaitCylinderCfg defaults (sharpa_wave_graspxl_bandgait.py) ----
CONTACT_THRESH = 0.1
ROT_THRESH = 0.15
HELD_DISP = 0.04
# G1 re-contact
G1_W, G1_K_OFF, G1_M_SUS, G1_CAP = 1.0, 5, 5, 1.0
# G2 load-share
G2_W, G2_CAP = 0.5, 1.0
# G3 contact-age
G3_W, G3_T, G3_CAP = 0.5, 25.0, 1.0
# G4 joint-headroom
G4_W, G4_CAP = 0.5, 1.0
# G5 recovery
G5_W, G5_WINDOW, G5_BETA, G5_DROP, G5_CAP = 1.0, 15, 0.8, 1.0, 1.0

N_F = 5
# fixed fingertip unit directions (object at origin), planar; thumb opposes the finger cluster.
# [thumb, index, middle, ring, pinky] angles (deg): thumb=180 opposed to 340/20/60/100.
_ANG = np.deg2rad(np.array([180.0, 340.0, 20.0, 60.0, 100.0]))
FTIP_DIR = np.stack([np.cos(_ANG), np.sin(_ANG), np.zeros(N_F)], axis=-1)   # (5,3) unit


# =====================================================================================
# numpy references -- mirror the torch env term-by-term (single env; eng is (5,) in {0,1})
# =====================================================================================
class GaitRef:
    def __init__(self):
        self.g1_off = np.zeros(N_F); self.g1_on = np.zeros(N_F); self.g1_pend = np.zeros(N_F)
        self.g3_age = np.zeros(N_F)
        self.g5_ema = 0.0; self.g5_win = 0.0

    def g1(self, eng, n_eng, rotating, held):
        was_off_enough = (self.g1_off >= G1_K_OFF).astype(float)
        reengage = eng * was_off_enough
        self.g1_pend = np.maximum(self.g1_pend, reengage)
        self.g1_on = np.where(eng > 0.5, self.g1_on + 1.0, 0.0)
        self.g1_off = np.where(eng > 0.5, 0.0, self.g1_off + 1.0)
        self.g1_pend = self.g1_pend * eng
        fire = self.g1_pend * (self.g1_on >= G1_M_SUS).astype(float)
        self.g1_pend = self.g1_pend * (1.0 - fire)
        return min(G1_W * fire.sum(), G1_CAP) * held * rotating

    def g2(self, eng, n_eng, rotating, held):
        share = np.clip((n_eng - 2.0) / 2.0, 0.0, 1.0)
        return min(G2_W * share, G2_CAP) * held

    def g3(self, eng, n_eng, rotating, held):
        self.g3_age = np.where(eng > 0.5, self.g3_age + 1.0, 0.0)
        over = np.clip((self.g3_age - G3_T) / G3_T, 0.0, 1.0)
        pen = min(G3_W * over.sum(), G3_CAP)
        return -pen * held

    def g4(self, eng, n_eng, rotating, held, norm):
        # norm: (22,) normalized joint position in ~[-1,1] (0 mid-range, +-1 at a limit)
        headroom = np.clip(1.0 - norm * norm, 0.0, 1.0)
        return min(G4_W * headroom.mean(), G4_CAP) * rotating * held

    def g5(self, eng, n_eng, rotating, held):
        release = 1.0 if (self.g5_ema - n_eng) >= G5_DROP else 0.0
        self.g5_win = G5_WINDOW if release > 0.5 else max(self.g5_win - 1.0, 0.0)
        self.g5_ema = G5_BETA * self.g5_ema + (1.0 - G5_BETA) * n_eng
        d = FTIP_DIR - 0.0                                   # object at origin -> d == FTIP_DIR
        dn = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-6)
        w = eng / (n_eng + 1e-6)
        mean_dir = (dn * w[:, None]).sum(0)
        opposition = np.clip(1.0 - np.linalg.norm(mean_dir), 0.0, 1.0)
        grasp_q = (n_eng / 5.0) * opposition
        in_window = 1.0 if self.g5_win > 0.0 else 0.0
        return min(G5_W * grasp_q * in_window, G5_CAP) * held


# =====================================================================================
# synthetic trajectories -> per-step dict(eng(5), omega, held01, norm(22))
# =====================================================================================
def traj_gaiting(T=200):
    steps = []
    # staggered per-finger on/off: period 20, on 14 (>=m_sus), off 6 (>=k_off), phase = f*4.
    # EVERY finger cycles (none planted) -> G1 fires per cycle, G2 keeps n_eng at 3-4, G3 stays ~0
    # (each on-run 14 < T=25), G4 headroom ~1 (joints mid-range).
    # A periodic 2-step REGRIP DIP (grip briefly reduced to 2 fingers) creates the release event G5 needs
    # (n_engaged drops >=1 below the smoothed count) -- a legitimate gaiting regrip, not a farm.
    for t in range(T):
        eng = np.zeros(N_F)
        for f in range(N_F):
            ph = (t + f * 4) % 20
            eng[f] = 1.0 if ph < 14 else 0.0   # 14 on / 6 off
        if t > 0 and (t % 25) < 2:              # regrip dip: hand off to thumb+index for 2 steps
            eng = np.array([1.0, 1.0, 0.0, 0.0, 0.0])   # n_eng = 2 (clear release vs the ~3.5 EMA)
        norm = np.zeros(22) + 0.02 * np.sin(t / 5.0)   # joints stay mid-range
        steps.append(dict(eng=eng, omega=0.30, held01=1.0, norm=norm))
    return steps


def traj_rolling(T=200):
    steps = []
    for t in range(T):
        eng = np.zeros(N_F); eng[0] = 1.0; eng[2] = 1.0   # thumb + middle planted, others off
        drift = min(t / 50.0, 0.98)                        # joints ramp to their limits
        norm = np.full(22, drift)
        steps.append(dict(eng=eng, omega=0.30, held01=1.0, norm=norm))
    return steps


def traj_flutter(T=200):
    steps = []
    for t in range(T):
        eng = np.ones(N_F) if (t % 2 == 0) else np.zeros(N_F)  # all fingers toggle every step
        norm = np.zeros(22)
        steps.append(dict(eng=eng, omega=0.0, held01=0.0, norm=norm))  # NO rotation, DROPPED
    return steps


def score(traj):
    """Return summed reward over the trajectory for each of G1..G5 (fresh state per term)."""
    refs = {k: GaitRef() for k in ["g1", "g2", "g3", "g4", "g5"]}
    tot = {k: 0.0 for k in refs}
    for s in traj:
        eng = s["eng"]; n_eng = float(eng.sum())
        rotating = 1.0 if abs(s["omega"]) > ROT_THRESH else 0.0
        held = 1.0 if s["held01"] > 0.5 else 0.0            # synthetic disp already binarized via held01
        tot["g1"] += refs["g1"].g1(eng, n_eng, rotating, held)
        tot["g2"] += refs["g2"].g2(eng, n_eng, rotating, held)
        tot["g3"] += refs["g3"].g3(eng, n_eng, rotating, held)
        tot["g4"] += refs["g4"].g4(eng, n_eng, rotating, held, s["norm"])
        tot["g5"] += refs["g5"].g5(eng, n_eng, rotating, held)
    return tot


def main():
    G = score(traj_gaiting()); R = score(traj_rolling()); F = score(traj_flutter())
    names = {"g1": "G1 recontact", "g2": "G2 loadshare", "g3": "G3 contactage",
             "g4": "G4 headroom", "g5": "G5 recovery"}
    print("==== FINGER-GAITING REWARD UNIT TEST (offline, numpy) ====")
    print(f"{'term':16s} {'GAITING':>10s} {'ROLLING':>10s} {'FLUTTER':>10s}   "
          f"{'GAIT>ROLL':>9s} {'FLUT~0':>7s}")
    ok = True
    for k in ["g1", "g2", "g3", "g4", "g5"]:
        g, r, f = G[k], R[k], F[k]
        gr = g > r
        fl = abs(f) < 1e-6
        ok = ok and gr and fl
        print(f"{names[k]:16s} {g:10.3f} {r:10.3f} {f:10.3f}   "
              f"{('PASS' if gr else 'FAIL'):>9s} {('PASS' if fl else 'FAIL'):>7s}")
    print("----------------------------------------------------------")
    print(f"OVERALL: {'PASS -- every term scores GAITING>ROLLING and FLUTTER~0' if ok else 'FAIL'}")
    print("==========================================================")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
