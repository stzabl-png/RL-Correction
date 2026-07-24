# CPU unit test for the REGRIP-sweep reward logic (no isaaclab import; mirrors the per-step math of
# sharpa_wave_graspxl_regrip_gait.py (R-G1) and sharpa_wave_graspxl_rgtb.py (R-D5) the same way
# gait_reward_unit_test.py mirrors bandgait — if those files change, update the mirrored blocks here).
# Run: python rl_rebuild/scripts/regrip_sweep_unit_test.py
import torch

K_OFF, M_SUS = 5, 5


def run_g1(force_traj, active_traj, held=1.0, thr=0.1):
    """Mirror of _RegripGaitMixin._term_g1_recontact (de-rotationed, settle-frozen). force_traj (T,5)."""
    T, F = force_traj.shape
    off = torch.zeros(F); on = torch.zeros(F); pend = torch.zeros(F)
    total = 0.0
    for t in range(T):
        eng = (force_traj[t] > thr).float()
        act = float(active_traj[t])
        reeng = eng * (off >= K_OFF).float() * act
        pend = torch.maximum(pend, reeng)
        on_new = torch.where(eng > 0.5, on + 1.0, torch.zeros_like(on))
        off_new = torch.where(eng > 0.5, torch.zeros_like(off), off + 1.0)
        on = act * on_new + (1 - act) * on
        off = act * off_new + (1 - act) * off
        pend = pend * eng
        fire = pend * (on >= M_SUS).float() * act
        pend = pend * (1.0 - fire)
        total += float((1.0 * fire.sum()).clamp(0.0, 1.0) * held * act)
    return total


def run_rgtb(force_traj, active_traj, held=1.0, on_thr=0.15, off_thr=0.05, baseline=2.0, cap=3.0):
    """Mirror of SharpaWaveGraspXLRegripD5RGTBSphereEnv._get_rewards' RGTB block. Returns
    (event_total, milestone_total, n_replace, n_accrete)."""
    T, F = force_traj.shape
    eng_state = torch.zeros(F); off = torch.zeros(F); on = torch.zeros(F)
    pend = torch.zeros(F); pend_prior = torch.zeros(F); ever = torch.zeros(F)
    started = False; pinch = torch.zeros(F); token = 0.0; level = baseline; paid = 0.0
    sus3 = 0.0; milestone = 0.0
    ev_tot, ms_tot, n_rep, n_acc = 0.0, 0.0, 0, 0
    for t in range(T):
        f = force_traj[t]
        act = float(active_traj[t])
        eng_state = torch.where(f > on_thr, torch.ones_like(eng_state),
                                torch.where(f < off_thr, torch.zeros_like(eng_state), eng_state))
        n_deb = float(eng_state.sum())
        if act > 0.5 and not started:
            pinch = eng_state.clone(); started = True
        qualify = eng_state * (off >= K_OFF).float() * act
        new_pend = (qualify > 0.5) & (pend < 0.5)
        prior = torch.maximum(ever, pinch).clamp(0, 1)
        pend_prior = torch.where(new_pend, prior, pend_prior)
        ever = torch.maximum(ever, eng_state * act)
        pend = torch.maximum(pend, qualify)
        on_new = torch.where(eng_state > 0.5, on + 1.0, torch.zeros_like(on))
        off_new = torch.where(eng_state > 0.5, torch.zeros_like(off), off + 1.0)
        on = act * on_new + (1 - act) * on
        off = act * off_new + (1 - act) * off
        pend = pend * eng_state
        fire = pend * (on >= M_SUS).float() * act
        pend = pend * (1.0 - fire)
        released_pinch = float((pinch * (off >= K_OFF).float()).amax())
        token = max(token, released_pinch * act)
        fired_any = bool(fire.amax() > 0.5)
        can_pay = (token > 0.5) and (n_deb > level) and (paid < cap) and (held > 0.5) and (act > 0.5) and fired_any
        if can_pay:
            ev_tot += 1.0; level = n_deb; paid += 1.0
            if float((fire * pend_prior).amax()) > 0.5:
                n_rep += 1
            else:
                n_acc += 1
        sus3 = (sus3 + 1.0) if (n_deb >= 3.0 and act > 0.5 and held > 0.5) else (sus3 if act < 0.5 else 0.0)
        hit = 1.0 if (sus3 >= 10 and milestone < 0.5) else 0.0
        ms_tot += hit * 5.0
        milestone = max(milestone, hit)
    return ev_tot, ms_tot, n_rep, n_acc


def traj(T, spec):
    """Build (T,5) force trajectory from {finger: [(t0,t1,force),...]}."""
    x = torch.zeros(T, 5)
    for fi, spans in spec.items():
        for (a, b, v) in spans:
            x[a:b, fi] = v
    return x


def main():
    T = 120
    active = torch.ones(T); active[:20] = 0.0        # 20-step settle window
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        fails += 0 if cond else 1

    print("== R-G1 regrip (de-rotationed re-contact) ==")
    # genuine regrip: thumb+index pinched through settle; index releases 30-40, re-places 40+;
    # middle recruits at 60 after being off >=5 active steps. NO rotation anywhere.
    g = traj(T, {0: [(0, T, 1.0)], 1: [(0, 30, 1.0), (40, T, 1.0)], 2: [(60, T, 1.0)]})
    r = run_g1(g, active)
    check("genuine release->re-place + recruit fires (no rotation gate)", r >= 2.0, f"total={r}")
    # threshold flicker: finger toggles every step -> on-streak never reaches M_SUS
    fl = traj(T, {3: [(t, t + 1, 1.0) for t in range(20, T, 2)]})
    r = run_g1(fl, active)
    check("flicker earns 0", r == 0.0, f"total={r}")
    # settle-frozen: contact pattern entirely inside settle earns 0
    st = traj(T, {2: [(5, 18, 1.0)]})
    r = run_g1(st, torch.zeros(T))
    check("settle window earns 0", r == 0.0, f"total={r}")

    print("== R-D5 RGTB (release token + ratchet + forensics) ==")
    # (a) genuine: pinch {T,I}; index releases (25-35), re-places 35+ (token; re-place fire at n=2 NOT
    # > baseline -> no pay); middle recruits at 60 -> n=3 > 2 -> pays, accretion; ring at 80 -> n=4 pays.
    a = traj(T, {0: [(0, T, 1.0)], 1: [(0, 25, 1.0), (35, T, 1.0)],
                 2: [(60, T, 1.0)], 3: [(80, T, 1.0)]})
    ev, ms, rep, acc = run_rgtb(a, active)
    check("token+ratchet pays exactly on n=3 and n=4 recruits", ev == 2.0, f"events={ev}")
    check("milestone pays once", ms == 5.0, f"milestone={ms}")
    check("forensics: both paid events are first-touch accretion", (rep, acc) == (0, 2), f"rep={rep} acc={acc}")
    # (b) accretion WITHOUT any pinch release -> token stays 0 -> nothing pays
    b = traj(T, {0: [(0, T, 1.0)], 1: [(0, T, 1.0)], 2: [(60, T, 1.0)], 3: [(80, T, 1.0)]})
    ev, ms, rep, acc = run_rgtb(b, active)
    check("no release token -> no event pay (milestone independent)", ev == 0.0, f"events={ev}")
    check("milestone still pays on sustained n>=3", ms == 5.0, f"milestone={ms}")
    # (c) dual-threshold flicker at 0.08/0.12 N never toggles the debounced state
    c = torch.zeros(T, 5); c[:, 0] = 1.0; c[:, 1] = 1.0
    c[20:, 2] = torch.tensor([0.12 if (t % 2 == 0) else 0.08 for t in range(20, T)])
    ev, ms, rep, acc = run_rgtb(c, active)
    check("0.08/0.12N flicker: no events, no milestone", (ev, ms) == (0.0, 0.0), f"ev={ev} ms={ms}")
    # (d) re-place classification: index (pinch member) release->re-place while middle+ring already on
    d = traj(T, {0: [(0, T, 1.0)], 1: [(0, 30, 1.0), (45, T, 1.0)],
                 2: [(22, T, 1.0)], 3: [(24, T, 1.0)]})
    ev, ms, rep, acc = run_rgtb(d, active)
    check("pinch-finger re-place classified as re-place", rep >= 1, f"rep={rep} acc={acc} ev={ev}")

    print(("ALL PASSED" if fails == 0 else f"{fails} FAILURES"))
    raise SystemExit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
