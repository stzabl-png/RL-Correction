#!/usr/bin/env python3
"""
Self-optimization harness for grasp detection (INNER loop = automatic param search).

Protocol (overfitting-safe):
  1. Discover every annotated clip, map to its reconstruction npz.
  2. Split: hold out a LOCKED test set (never used for tuning) by clip-id hash.
  3. K-fold CV on the remaining (train) clips: search parameters, pick the config
     with the best mean validation score (default: segment-F1).
  4. Refit nothing (thresholds ARE the model) -> evaluate the chosen config ONCE
     on the locked test set. Report train-CV vs test to expose overfitting.

The OUTER loop (structural changes: new features/rules) is driven by a human/model
reading the per-clip failure report this prints — not by this script.

Usage:
  python3 grasp_tune.py                 # smoke test on whatever is annotated
  python3 grasp_tune.py --trials 300 --test-frac 0.2 --folds 5
"""
import argparse, glob, json, os, hashlib, itertools
import numpy as np
from grasp_detect import detect, HAND_NAME
from grasp_metrics import full_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# search space for the INNER loop (physically-motivated ranges, kept small on purpose)
SPACE = dict(
    T_rel=[0.004, 0.005, 0.006, 0.008, 0.010],
    T_rig=[0.015, 0.020, 0.025, 0.030],
    T_obj=[0.003, 0.004, 0.005, 0.006],
    activity_margin=[1.3, 1.5, 2.0],
    close_gap=[6, 8, 10],
    min_len=[4, 5, 6],
    min_activity=[0, 3, 5, 8],
    closure_anchor_pct=[0, 50, 65, 80],  # MANO hand-closed OR-anchor (rescues obj-recon fails)
    # REVERTED (kept in detector, out of search): trim_to_motion (R3 overfit) and
    # closure_pct AND-gate (R4: CV never selected it, hurt recall).
)


def load_dataset():
    anns = sorted(glob.glob(os.path.join(ROOT, "Data/**/grasp_annotation.json"), recursive=True))
    ds = []
    for ann in anns:
        rel = ann.split("HOI4D_release/")[1].split("/align_rgb")[0]
        npz = os.path.join(ROOT, "Output/ReconstructOutput/hoi4d", rel, "world_fused.npz")
        if not os.path.exists(npz):
            continue
        gt = json.load(open(ann))["annotations"]
        ds.append(dict(id=rel, npz=npz, gt=gt))
    return ds


def clip_score(clip, params, tol=5):
    """Mean segment-F1 over annotated hands of one clip; also returns frame_iou."""
    results, _ = detect(clip["npz"], **params)
    name2idx = {v: k for k, v in HAND_NAME.items()}
    f1s, ious, detail = [], [], {}
    for side, segs_gt in clip["gt"].items():
        if not segs_gt:
            continue
        det = results.get(name2idx.get(side), [])
        m = full_metrics(det, segs_gt, 300, tol=tol)
        f1s.append(m["seg_f1"]); ious.append(m["frame_iou"])
        detail[side] = dict(det=det, gt=segs_gt, **{k: round(v, 3) for k, v in m.items() if isinstance(v, float)})
    return (np.mean(f1s) if f1s else 0.0, np.mean(ious) if ious else 0.0, detail)


def split_locked_test(ds, test_frac):
    """Deterministic hold-out by clip-id hash (stable across runs, no Math.random)."""
    test = [c for c in ds if (int(hashlib.md5(c["id"].encode()).hexdigest(), 16) % 100) < test_frac * 100]
    train = [c for c in ds if c not in test]
    return train, test


def kfold(items, k):
    idx = list(range(len(items)))
    return [[items[j] for j in idx if j % k == f] for f in range(k)]


def mean_f1(clips, params):
    return float(np.mean([clip_score(c, params)[0] for c in clips])) if clips else 0.0


def search(train, folds, trials):
    """Grid if small, else random sample; pick config with best mean CV F1."""
    import random
    keys = list(SPACE)
    all_combos = list(itertools.product(*[SPACE[k] for k in keys]))
    if len(all_combos) <= trials:
        combos = all_combos
    else:
        combos = random.Random(42).sample(all_combos, trials)
    fs = kfold(train, min(folds, max(1, len(train))))
    best, best_cv = None, -1
    for combo in combos:
        params = dict(zip(keys, combo))
        val = []
        for f in range(len(fs)):
            val_clips = fs[f]
            if not val_clips:
                continue
            val.append(mean_f1(val_clips, params))   # thresholds have no fit step -> just eval on val fold
        cv = float(np.mean(val)) if val else 0.0
        if cv > best_cv:
            best_cv, best = cv, params
    return best, best_cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--target", type=float, default=0.95)
    a = ap.parse_args()

    ds = load_dataset()
    print(f"annotated clips: {len(ds)}")
    if len(ds) < 8:
        print("!! WARNING: too few clips for a trustworthy split — this is a SMOKE TEST only.")
    train, test = split_locked_test(ds, a.test_frac)
    print(f"train(tune)={len(train)}  locked-test={len(test)}")

    best, best_cv = search(train, a.folds, a.trials)
    print("\nbest params:", best)
    print(f"train CV seg-F1 = {best_cv:.3f}")

    # one-shot locked test-set evaluation
    if test:
        test_f1 = mean_f1(test, best)
        print(f"LOCKED-TEST seg-F1 = {test_f1:.3f}  (this is the honest number)")
        gap = best_cv - test_f1
        print(f"overfit gap (CV - test) = {gap:+.3f}  {'<-- watch' if gap > 0.1 else 'ok'}")
        report = test_f1
    else:
        report = best_cv

    print("\n---- per-clip failure report (drives the OUTER/structural loop) ----")
    for c in ds:
        f1, iou, detail = clip_score(c, best)
        flag = "" if f1 >= a.target else "  <-- below target"
        print(f"{c['id']:42s} F1={f1:.2f} IoU={iou:.2f}{flag}")
        for side, d in detail.items():
            print(f"      [{side}] det={d['det']} gt={d['gt']} onsetMAE={d.get('onset_mae')} relMAE={d.get('release_mae')}")

    print(f"\n==> {'REACHED' if report >= a.target else 'NOT reached'} target {a.target}: score={report:.3f}")


if __name__ == "__main__":
    main()
