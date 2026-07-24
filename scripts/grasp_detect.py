#!/usr/bin/env python3
"""
Grasp / release detection from reconstructed trajectories (single- & two-hand).

Core idea (depth-bias robust):
  The reconstructed hand-object RELATIVE position r(t)=H(t)-O(t) carries a large,
  slowly-varying depth bias, so its ABSOLUTE magnitude |r| is NOT reliable.
  Time derivatives cancel a (locally) constant bias, so we key off motion:

    (A) COUPLING  : during contact hand & object move together -> relative speed
                    |vH-vO| small AND relative vector r(t) locally stable
                    (low windowed std). Bias-invariant.
    (B) OBJECT ACTIVITY : a passive object only moves when held, so object speed
                    |vO| above the resting floor => grasped. Covers the carrying
                    phase where the object rotates in hand and (A) breaks.

  contact(t) = A OR B  ->  morphology  ->  [grasp_onset, release] segments.

Two-hand handling:
  Object activity says "SOME hand holds it", not which. We attribute activity to
  the hand(s) whose relative speed is (near-)minimal among the valid hands:
      activity_h = active & ( spRel_h <= max(margin*min_spRel, T_rel) )
  - one valid hand   -> reduces exactly to single-hand behaviour
  - single-hand carry -> the idle hand has large spRel and is excluded
  - bimanual grasp   -> both hands near the min -> both fire
  - handover         -> attribution switches between hands over time

All trajectories are in gravity_z_up_world (object_ob_in_world & hand_trans).
"""
import argparse, json
import numpy as np

HAND_NAME = {0: "left", 1: "right"}


def smooth(x, w=5):
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, i], k, "same") for i in range(x.shape[1])], 1)


def rollstd(r, w=7):
    n = len(r); out = np.zeros(n)
    for i in range(n):
        a = max(0, i - w); b = min(n, i + w + 1)
        out[i] = np.linalg.norm(r[a:b].std(0))
    return out


def mask_to_segs(m):
    s = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            s.append([int(i), int(j)]); i = j + 1
        else:
            i += 1
    return s


def morph(m, close_gap=8, min_len=5):
    m = m.copy()
    for a, b in mask_to_segs(~m):                    # fill short holds inside a grasp
        if b - a + 1 <= close_gap and a > 0 and b < len(m) - 1:
            m[a:b + 1] = True
    for a, b in mask_to_segs(m):                     # drop too-short blips
        if b - a + 1 < min_len:
            m[a:b + 1] = False
    return m


def load_tracks(npz_path):
    """Return object centre (F,3) and per-hand {trans (F,3), valid (F,)} at object fps."""
    d = np.load(npz_path, allow_pickle=True)
    O = d["object_ob_in_world"][:, :3, 3]
    n = len(O)
    ht = d["hand_trans"]                              # (H, 2F, 3)
    hv = d["hand_valid"]                              # (H, 2F)
    hp = d["hand_pose"]                               # (H, 2F, 45) MANO joint axis-angles
    step = max(1, ht.shape[1] // n)                  # hands at 2x time res -> downsample
    hands = {}
    for h in range(ht.shape[0]):
        pose = hp[h][::step][:n].reshape(n, 15, 3)
        hands[h] = dict(trans=ht[h][::step][:n],
                        valid=hv[h][::step][:n] > 0.5,
                        closure=np.linalg.norm(pose, axis=2).sum(1))  # finger-flexion magnitude
    return O, hands, n


def detect(npz_path, T_rel=0.010, T_rig=0.020, T_obj=0.004,
           activity_margin=2.0, close_gap=10, min_len=5, min_valid=10,
           min_activity=8, trim_to_motion=0, trim_margin=5, closure_pct=0,
           closure_anchor_pct=50):
    # Defaults = Round-5 tuned config (5-fold CV seg-F1 0.849, locked-test 0.836, gap ~0).
    """min_activity>0 enables ACTIVITY ANCHORING: a morphed segment is kept only
    if the object genuinely moves inside it (>= min_activity frames with spO>T_obj).
    Kills both-static edge false-positives and coupling-only jitter fragments."""
    O_raw, hands, n = load_tracks(npz_path)
    O = smooth(O_raw)
    vO = np.gradient(O, axis=0)
    spO = np.linalg.norm(vO, axis=1)
    active = spO > T_obj

    # per-hand kinematic features
    feats = {}
    for h, hd in hands.items():
        if hd["valid"].sum() < min_valid:
            continue
        H = smooth(hd["trans"])
        vH = np.gradient(H, axis=0)
        feats[h] = dict(valid=hd["valid"],
                        spRel=np.linalg.norm(vH - vO, axis=1),
                        rig=rollstd(H - O),
                        closure=smooth(hd["closure"][:, None])[:, 0])

    if not feats:
        return {}, dict(spO=spO)

    # relative-speed minimum across valid hands (for activity attribution)
    valid_hands = list(feats)
    spRel_stack = np.stack([feats[h]["spRel"] for h in valid_hands])           # (Hv, n)
    valid_stack = np.stack([feats[h]["valid"] for h in valid_hands])
    spRel_masked = np.where(valid_stack, spRel_stack, np.inf)
    min_spRel = spRel_masked.min(0)                                            # (n,)

    results = {}
    for i, h in enumerate(valid_hands):
        f = feats[h]
        coupling = (f["spRel"] < T_rel) & (f["rig"] < T_rig)
        # attribute object activity to hand(s) that track the object best
        thr = np.maximum(activity_margin * min_spRel, T_rel)
        activity_h = active & (f["spRel"] <= thr)
        contact = (coupling | activity_h) & f["valid"]
        if closure_pct > 0:                                        # hand-closed AND-gate
            c = f["closure"]
            c_thr = np.percentile(c[f["valid"]], closure_pct) if f["valid"].any() else 0
            contact = contact & (c > c_thr)
        contact = morph(contact, close_gap, min_len)
        segs = mask_to_segs(contact)
        if min_activity > 0:                                       # anchoring: object-moves OR hand-closed
            c = f["closure"]
            c_thr = np.percentile(c[f["valid"]], closure_anchor_pct) if (closure_anchor_pct and f["valid"].any()) else np.inf
            kept = []
            for s in segs:
                obj_moved = (spO[s[0]:s[1] + 1] > T_obj).sum() >= min_activity
                hand_closed = (c[s[0]:s[1] + 1] > c_thr).sum() >= min_activity   # rescues obj-recon failures
                if obj_moved or hand_closed:
                    kept.append(s)
            segs = kept
        if trim_to_motion:                                         # snap edges to object motion
            trimmed = []
            for a, b in segs:
                mv = np.where(spO[a:b + 1] > T_obj)[0]
                if len(mv) == 0:
                    trimmed.append([a, b]); continue
                na = max(a, a + mv[0] - trim_margin)
                nb = min(b, a + mv[-1] + trim_margin)
                trimmed.append([int(na), int(nb)])
            segs = trimmed
        results[h] = segs
    return results, dict(spO=spO, feats=feats)


def evaluate(segs, gt, n=300):
    g = np.zeros(n, bool); p = np.zeros(n, bool)
    for a, b in gt: g[a:b + 1] = True
    for a, b in segs: p[a:b + 1] = True
    inter = (g & p).sum(); uni = (g | p).sum()
    return dict(IoU=round(inter / max(1, uni), 3),
                precision=round(inter / max(1, p.sum()), 3),
                recall=round(inter / max(1, g.sum()), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--gt", help="grasp_annotation.json for evaluation")
    a = ap.parse_args()
    results, _ = detect(a.npz)
    gt = json.load(open(a.gt))["annotations"] if a.gt else None
    for h in sorted(results):
        name = HAND_NAME.get(h, str(h))
        print(f"[{name}] grasp segments [onset, release]:", results[h])
        if gt is not None and gt.get(name):
            print(f"       GT: {gt[name]}   eval: {evaluate(results[h], gt[name])}")
