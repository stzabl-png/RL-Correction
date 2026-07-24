# D0 — Reward unit test + Phi-sanity (OFFLINE, no sim, no training).
#
# Purpose: prove on paper that the *minimal* adjust-then-rotate reward
#   r_t = w_rot * clip(omega . axis, -0.5, 0.5)            # objective (dominant)
#       + w_phi * [ gamma*Phi(s') - Phi(s) ]              # ONE potential-based readiness term (Ng 1999)
#       - w_e   * energy(tau, qdot)                        # small, global (anti-squeeze + sim2real)
#       - w_lin * ||v_obj||                                # small, anti-translate/roll
#       - drop_penalty on (debounced) drop                # outcome-based termination
#   (the pose-deviation penalty toward the pinch is DELETED)
# ranks the DESIRED behavior (adjust pinch->enclosing, then sustained rotation)
# strictly above every documented hack (squeeze, translate, one-time-then-drop, do-nothing).
#
# Phi (rotation-readiness), axis-conditioned, single scalar in [0,1], FORCE-MAGNITUDE-INDEPENDENT
# (so squeezing cannot inflate it):
#   engaged fingers   = those with contact force > f_thresh
#   spread            = 1 - || mean_i( dir_i ) ||  over engaged fingers' directions in the plane _|_ axis
#                       (opposing/distributed contacts -> ~1 ; one-sided pinch -> ~0 ; <2 engaged -> 0)
#   coverage          = clip((n_engaged - 1)/4, 0, 1)      (1 finger -> 0 ; 5 fingers -> 1)
#   Phi               = 0.5*spread + 0.5*coverage
#
# Run:  <sharpa-python> rl_rebuild/scripts/d0_reward_unit_test.py
import numpy as np

# ---- weights (the minimal reward) -------------------------------------------
W_ROT   = 2.5      # rotation objective (matches repo rotate scale); must dominate steady state
W_PHI   = 3.0      # readiness-potential gain (guides pinch->enclosing; small vs sustained rotation)
W_E     = 0.01     # global energy (anti-squeeze + sim2real); small
W_LIN   = 0.5      # object linear-velocity penalty (anti-translate/roll)
DROP_PEN = 10.0    # drop penalty; must exceed any gain it offsets (reward-design P drop)
GAMMA   = 0.99
F_THRESH = 0.1     # N, contact counted as engaged (matches env 0.1 cutoff)
AXIS = np.array([0.0, 0.0, 1.0])
R_OBJ = 0.04       # object radius (m), for contact placement

# ---- Phi --------------------------------------------------------------------
def phi(contacts_xy_angle_deg, forces):
    """contacts_xy_angle_deg: (5,) angle of each fingertip contact around the axis (deg);
       forces: (5,) contact force magnitude per finger. Returns Phi in [0,1]."""
    forces = np.asarray(forces, float)
    eng = forces > F_THRESH
    n = int(eng.sum())
    coverage = np.clip((n - 1) / 4.0, 0.0, 1.0)
    if n < 2:
        spread = 0.0
    else:
        ang = np.deg2rad(np.asarray(contacts_xy_angle_deg, float)[eng])
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)        # unit dirs in plane _|_ axis
        resultant = np.linalg.norm(dirs.mean(axis=0))
        spread = 1.0 - resultant
    return 0.5 * spread + 0.5 * coverage

# ---- per-step reward --------------------------------------------------------
def step_reward(s, s_next):
    """s, s_next: dicts with keys angles, forces, omega_axis, v_obj, torque, dropped."""
    r_rot = W_ROT * np.clip(s_next["omega_axis"], -0.5, 0.5)
    r_phi = W_PHI * (GAMMA * phi(s_next["angles"], s_next["forces"]) - phi(s["angles"], s["forces"]))
    r_e   = -W_E * float(np.sum(np.asarray(s_next["torque"], float) ** 2))
    r_lin = -W_LIN * float(s_next["v_obj"])
    r_drop = -DROP_PEN if s_next["dropped"] else 0.0
    return dict(rot=r_rot, phi=r_phi, energy=r_e, linvel=r_lin, drop=r_drop,
                total=r_rot + r_phi + r_e + r_lin + r_drop)

# ---- canonical single states (for Phi-sanity) -------------------------------
PINCH   = dict(angles=[10, 0, 0, 0, 0],      forces=[1, 0, 0, 0, 0])              # 1 finger (matches D0a: n=1 at t0)
PINCH2  = dict(angles=[10, 40, 0, 0, 0],     forces=[1, 1, 0, 0, 0])              # 2 same-side fingers
OPP2    = dict(angles=[0, 180, 0, 0, 0],     forces=[1, 0, 0, 0, 1])              # 2 opposed fingers
ENC4    = dict(angles=[0, 90, 180, 270, 0],  forces=[1, 1, 1, 1, 0])              # 4 enclosing
ENC5    = dict(angles=[0, 72, 144, 216, 288],forces=[1, 1, 1, 1, 1])             # 5 enclosing
SQUEEZE = dict(angles=[0, 72, 144, 216, 288],forces=[8, 8, 8, 8, 8])             # enclosing but crushing
GRAZE   = dict(angles=[0, 72, 144, 216, 288],forces=[.02, .02, .02, .02, .02])    # transient/below-threshold

def sanity():
    rows = [("pinch(1)", PINCH), ("pinch2(same-side)", PINCH2), ("opposed-2", OPP2),
            ("enclose-4", ENC4), ("enclose-5", ENC5), ("squeeze-5", SQUEEZE), ("graze(transient)", GRAZE)]
    print("==== Phi-SANITY (single states) ====")
    vals = {}
    for name, s in rows:
        v = phi(s["angles"], s["forces"]); vals[name] = v
        print(f"  Phi[{name:18s}] = {v:.3f}")
    checks = [
        ("enclose-5 > pinch(1)",            vals["enclose-5"] > vals["pinch(1)"] + 0.5),
        ("enclose-5 > pinch2(same-side)",   vals["enclose-5"] > vals["pinch2(same-side)"] + 0.3),
        ("squeeze == enclose (force-indep)",abs(vals["squeeze-5"] - vals["enclose-5"]) < 1e-6),
        ("graze ~ 0 (transient ignored)",   vals["graze(transient)"] < 0.05),
        ("enclose-5 >= enclose-4",          vals["enclose-5"] >= vals["enclose-4"]),
        ("opposed-2 < enclose-4",           vals["opposed-2"] < vals["enclose-4"]),
    ]
    ok = True
    for desc, c in checks:
        print(f"   [{'PASS' if c else 'FAIL'}] {desc}"); ok = ok and c
    return ok

# ---- trajectory builders (for the reward unit test) -------------------------
def st(angles, forces, omega, v_obj, torque, dropped=False):
    return dict(angles=angles, forces=forces, omega_axis=omega, v_obj=v_obj, torque=torque, dropped=dropped)

def lerp_grasp(t, T):
    """interpolate pinch(1 finger) -> enclosing(5 fingers) over T steps; returns (angles, forces)."""
    frac = t / max(T - 1, 1)
    n = 1 + int(round(4 * frac))                                   # 1..5 fingers engage
    ang = [0, 72, 144, 216, 288]
    forces = [1.0 if i < n else 0.0 for i in range(5)]
    return ang, forces

def traj_desired(T=30, conv=10):
    s = []
    for t in range(T):
        if t < conv:
            a, f = lerp_grasp(t, conv)
            s.append(st(a, f, omega=0.0, v_obj=0.01, torque=[0.5]*5))      # repositioning fingers, small torque
        else:
            s.append(st([0,72,144,216,288], [1]*5, omega=0.5, v_obj=0.01, torque=[0.4]*5))  # sustained rotation
    return s

def traj_donothing(T=30):
    return [st([10,0,0,0,0], [1,0,0,0,0], omega=0.0, v_obj=0.0, torque=[0.05]*5) for _ in range(T)]

def traj_squeeze(T=30, conv=10):
    s = []
    for t in range(T):
        if t < conv:
            a, f = lerp_grasp(t, conv); s.append(st(a, f, 0.0, 0.01, [0.5]*5))
        else:
            s.append(st([0,72,144,216,288], [8]*5, omega=0.0, v_obj=0.01, torque=[3.0]*5))  # crush, no rotation
    return s

def traj_translate(T=30, conv=10):
    s = []
    for t in range(T):
        if t < conv:
            a, f = lerp_grasp(t, conv); s.append(st(a, f, 0.0, 0.01, [0.5]*5))
        else:
            s.append(st([0,72,144,216,288], [1]*5, omega=0.0, v_obj=0.30, torque=[0.4]*5))   # translate, no rotation
    return s

def traj_onetime(T=30):
    """shear the pinch -> brief rotation -> drop (the documented one-time hack)."""
    s = []
    for t in range(T):
        if t < 4:
            s.append(st([10,0,0,0,0], [1,0,0,0,0], omega=0.5, v_obj=0.05, torque=[1.0]*5))   # forced shear from pinch
        elif t == 4:
            s.append(st([10,0,0,0,0], [0,0,0,0,0], omega=0.2, v_obj=0.20, torque=[0.5]*5, dropped=True))
        else:
            s.append(st([10,0,0,0,0], [0,0,0,0,0], omega=0.0, v_obj=0.0, torque=[0.0]*5))
    return s

def episode_return(traj):
    """discounted return + per-term breakdown; stops accumulating after a drop."""
    s0 = st([10,0,0,0,0], [1,0,0,0,0], 0.0, 0.0, [0.05]*5)         # all episodes start at the pinch
    prev = s0; G = 0.0; terms = dict(rot=0., phi=0., energy=0., linvel=0., drop=0.); g = 1.0
    for s_next in traj:
        r = step_reward(prev, s_next)
        G += g * r["total"]
        for k in terms: terms[k] += g * r[k]
        g *= GAMMA; prev = s_next
        if s_next["dropped"]:
            break
    return G, terms

def reward_unit_test():
    print("\n==== REWARD UNIT TEST (discounted episode return) ====")
    trajs = [("DESIRED adjust->rotate", traj_desired()), ("do-nothing(pinch)", traj_donothing()),
             ("squeeze-hack", traj_squeeze()), ("translate-hack", traj_translate()),
             ("one-time-then-drop", traj_onetime())]
    results = {}
    print(f"  {'trajectory':24s} {'return':>8s}  | {'rot':>7s} {'phi':>6s} {'energy':>7s} {'linvel':>7s} {'drop':>6s}")
    for name, tr in trajs:
        G, t = episode_return(tr); results[name] = G
        print(f"  {name:24s} {G:8.2f}  | {t['rot']:7.2f} {t['phi']:6.2f} {t['energy']:7.2f} {t['linvel']:7.2f} {t['drop']:6.1f}")
    best = max(results, key=results.get)
    desired = results["DESIRED adjust->rotate"]
    checks = [(f"DESIRED is the unique argmax (best={best})", best == "DESIRED adjust->rotate")]
    for name in results:
        if name != "DESIRED adjust->rotate":
            checks.append((f"DESIRED > {name}", desired > results[name] + 1e-6))
    ok = True
    print()
    for desc, c in checks:
        print(f"   [{'PASS' if c else 'FAIL'}] {desc}"); ok = ok and c
    return ok

if __name__ == "__main__":
    a = sanity()
    b = reward_unit_test()
    print("\n==================== SUMMARY ====================")
    print(f"  Phi-sanity:        {'PASS' if a else 'FAIL'}")
    print(f"  reward unit test:  {'PASS' if b else 'FAIL'}")
    print(f"  OVERALL:           {'PASS -> reward ranks target behavior highest; safe to implement' if (a and b) else 'FAIL -> retune reward BEFORE training'}")
    print("================================================")
    import sys; sys.exit(0 if (a and b) else 1)
