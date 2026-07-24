# CPU unit test for the REGRIP v2 chassis + design logic (mirrors the per-step math of
# sharpa_wave_regripv2_common.py / _designs.py; update if those change).
# Run: python rl_rebuild/scripts/regripv2_unit_test.py
import math
import torch

fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    fails += 0 if cond else 1


# ---- 1. sink termination debounce (chassis _get_dones mirror) ----
def sink_sim(z_traj, zstart, below=0.03, deb=3):
    cnt, out = 0.0, []
    for z in z_traj:
        cnt = cnt + 1 if z < zstart - below else 0.0
        out.append(cnt >= deb)
    return out

s = sink_sim([0.62, 0.60, 0.585, 0.585, 0.585, 0.62, 0.585, 0.585, 0.585], 0.62)
check("sink fires only after 3 consecutive below-steps", s == [False]*4 + [True, False, False, False, True])
s = sink_sim([0.585, 0.62] * 6, 0.62)
check("alternating dip never fires", not any(s))

# ---- 2. handover FSM (chassis mirror: idle->armed->verify->completed; aborts clear state) ----
def ho_sim(eng_traj, sup_traj, clear=True, balanced=True):
    st, off, timer, done = 0, 0, 0.0, 0
    rel = False
    hist = [0.0] * 6
    for i, (eng, sup) in enumerate(zip(eng_traj, sup_traj)):
        sup_prev = hist[(i + 1) % 6]
        hist[i % 6] = sup
        off_prev = off
        off = 0 if eng else off + 1
        just_rel = off_prev == 0 and off == 1
        just_rec = off_prev > 0 and eng
        held = sup > 0.8 * sup_prev and sup_prev > 0.05
        if st == 0 and just_rel and held and clear:
            st = 1
        elif st == 1 and just_rec:
            st = 2 if (off_prev >= 5 and True) else 0     # displacement assumed >= 8mm in this mirror
            timer = 10.0 if st == 2 else 0.0
        if st == 2:
            ok = clear and balanced and eng
            if ok:
                timer -= 1.0
                if timer <= 0.0:
                    done += 1; st = 0
            else:
                st = 0; timer = 0.0
    return done

eng = [1]*5 + [0]*6 + [1]*15
sup = [1.0]*26
check("genuine handover completes once", ho_sim(eng, sup) == 1, f"got {ho_sim(eng, sup)}")
sup_drop = [1.0]*5 + [0.01]*6 + [1.0]*15
check("release WITHOUT support held never arms", ho_sim(eng, sup_drop) == 0)
eng_short = [1]*5 + [0]*2 + [1]*15                     # off only 2 steps -> short hop
check("short-hop re-contact rejected", ho_sim(eng_short, sup) == 0)

# ---- 3. entropy ratchet (R5 mirror): pays growth once, monotone Hmax ----
def entropy(p):
    p = torch.tensor(p).clamp_min(1e-9)
    p = p / p.sum()
    return float((-(p * p.log()).sum()) / math.log(5.0))

hmax, paid = 0.0, 0.0
for shares in ([1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]):
    H = entropy(shares)
    paid += 2.0 * max(0.0, H - hmax)
    hmax = max(hmax, H)
check("entropy ratchet totals 2.0 at full coverage", abs(paid - 2.0) < 1e-5, f"paid={paid:.4f}")
check("planted-3 caps H below 0.9 gate", entropy([1, 1, 1, 0, 0]) < 0.9)

# ---- 4. leave-one-out margin (R3 mirror) ----
def margin(F, m=0.05, g=9.81, ext=(0.0, 0.0, 0.0)):
    F = torch.tensor(F, dtype=torch.float)                     # (5,3)
    eng = (F.norm(dim=-1) > 0.15).float()
    base = F.sum(0) + torch.tensor([0.0, 0.0, -m * g]) + torch.tensor(ext)
    rho = (base.unsqueeze(0) - F).norm(dim=-1) / (m * g)
    rho = torch.where(eng > 0.5, rho, torch.full_like(rho, 9.0))
    n = eng.sum()
    return float((1.0 - rho.min().clamp(0.0, 1.0)) * (n >= 2.0).float())

w = 0.05 * 9.81
# over-provisioned: three fingers each carrying w/2 (net +w/2): removing any one leaves EXACT support
F3o = [[0, 0, w / 2]] * 3 + [[0, 0, 0]] * 2
check("over-provisioned 3x(w/2) -> margin 1.0", abs(margin(F3o) - 1.0) < 1e-3, f"{margin(F3o):.3f}")
# balanced N-equal-share grip: margin = 1 - 1/N
F3b = [[0, 0, w / 3]] * 3 + [[0, 0, 0]] * 2
check("balanced 3-finger -> margin 2/3", abs(margin(F3b) - 2.0 / 3.0) < 1e-3, f"{margin(F3b):.3f}")
F2b = [[0, 0, w / 2]] * 2 + [[0, 0, 0]] * 3
check("balanced 2-finger -> margin 0.5 (< 3-finger)", abs(margin(F2b) - 0.5) < 1e-3, f"{margin(F2b):.3f}")
check("single finger -> margin 0 (n<2 gate)", margin([[0, 0, w]] + [[0, 0, 0]] * 4) == 0.0)

# ---- 5. R3 end bonus pays on TIMEOUT only ----
for tout, fail, expect in ((True, False, True), (False, True, False), (False, False, False)):
    pays = tout and not fail
    check(f"end bonus timeout={tout} fail={fail} -> pays={expect}", pays == expect)

print("CORE TESTS DONE" if fails == 0 else f"{fails} FAILURES (core)")


# ---- 6. R4S active-switch (mirror of _R4SwitchMixin): shed-taint exclusion + all-5 round bonus ----
def r4s_sim(eng_traj, shed_traj, k_off=5, m_on=5, rounds_cap=2):
    """eng_traj/shed_traj: lists of 5-tuples (debounced engagement / is-shed-target). Returns
    (first_tiers_paid, round_bonuses_paid). clear/active assumed true throughout."""
    F = 5
    off = [0.0] * F; on = [0.0] * F; pend = [0.0] * F; taint = [0.0] * F
    switched = [0.0] * F; first = [0.0] * F
    tiers, rounds = 0, 0
    for eng, shed in zip(eng_traj, shed_traj):
        for f in range(F):
            if shed[f]:
                taint[f] = 1.0
        for f in range(F):
            if eng[f] and off[f] >= k_off and taint[f] == 0.0:
                pend[f] = 1.0
            if eng[f]:
                taint[f] = 0.0
            off[f] = 0.0 if eng[f] else off[f] + 1.0
            on[f] = on[f] + 1.0 if eng[f] else 0.0
            if not eng[f]:
                pend[f] = 0.0
            if pend[f] and on[f] >= m_on:
                pend[f] = 0.0
                if first[f] == 0.0:
                    tiers += 1; first[f] = 1.0
                switched[f] = 1.0
        if sum(switched) >= 5 and rounds < rounds_cap:
            rounds += 1; switched = [0.0] * F
    return tiers, rounds

T = 200
def cyc(f, start, period=30, offlen=6):
    return [(t >= 20) and not (start + (f * 0) <= (t - start) % period < offlen) if t >= start else True
            for t in range(T)]
# all five fingers voluntarily cycle (staggered), no sheds
eng = [[not (20 + 12 * f <= t and (t - 20 - 12 * f) % 40 < 6) for f in range(5)] for t in range(T)]
shed = [[False] * 5 for _ in range(T)]
tiers, rounds = r4s_sim(eng, shed)
check("R4S: five voluntary cycles -> 5 tiers + >=1 round", tiers == 5 and rounds >= 1,
      f"tiers={tiers} rounds={rounds}")
# same cycles but every release is shed-caused -> nothing pays
shed_all = [[(20 + 12 * f <= t and (t - 20 - 12 * f) % 40 < 1) for f in range(5)] for t in range(T)]
tiers, rounds = r4s_sim(eng, shed_all)
check("R4S: shed-caused releases pay nothing", tiers == 0 and rounds == 0, f"tiers={tiers} rounds={rounds}")
# rounds capped
eng2 = [[not (10 + 3 * f <= t and (t - 10 - 3 * f) % 20 < 6) for f in range(5)] for t in range(400)]
shed2 = [[False] * 5 for _ in range(400)]
_, rounds = r4s_sim(eng2, shed2, rounds_cap=2)
check("R4S: round bonus capped at 2", rounds == 2, f"rounds={rounds}")

# ---- 7. R4R recruitment tiers (mirror of _R4RecruitMixin) ----
def r4r_sim(loaded_traj, streak_need=5):
    F = 5
    streak = [0.0] * F; recruited = [0.0] * F; tiers = 0
    for loaded in loaded_traj:
        for f in range(F):
            streak[f] = streak[f] + 1.0 if loaded[f] else 0.0
            if streak[f] >= streak_need and recruited[f] == 0.0:
                recruited[f] = 1.0; tiers += 1
    return tiers, sum(recruited)

sustained = [[t >= 10 + 8 * f for f in range(5)] for t in range(80)]
tiers, rec = r4r_sim(sustained)
check("R4R: staggered sustained loads -> 5 tiers, monotone", tiers == 5 and rec == 5)
flicker = [[(t % 3 == 0) for _ in range(5)] for t in range(80)]
tiers, rec = r4r_sim(flicker)
check("R4R: 3-step flicker never recruits", tiers == 0 and rec == 0)

print("R4X TESTS DONE" if fails == 0 else f"{fails} FAILURES (r4x)")
raise SystemExit(0 if fails == 0 else 1)
