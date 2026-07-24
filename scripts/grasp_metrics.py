#!/usr/bin/env python3
"""
Evaluation metrics for grasp/release detection.

Primary   : segment-level F1 with boundary tolerance (does each grasp get found?)
Secondary : boundary median error in frames (how tight are onset/release?)
Also kept : frame-level IoU / precision / recall (strict overlap).

Segment matching: greedy by temporal IoU; a (pred, gt) pair matches if their
IoU >= min_iou OR both boundaries fall within `tol` frames.
"""
import numpy as np


def _mask(segs, n):
    m = np.zeros(n, bool)
    for a, b in segs:
        m[a:b + 1] = True
    return m


def frame_metrics(pred, gt, n):
    g, p = _mask(gt, n), _mask(pred, n)
    inter = (g & p).sum(); uni = (g | p).sum()
    return dict(frame_iou=inter / max(1, uni),
                frame_prec=inter / max(1, p.sum()),
                frame_rec=inter / max(1, g.sum()))


def _seg_iou(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    uni = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / max(1, uni)


def segment_metrics(pred, gt, tol=5, min_iou=0.5):
    """Greedy match; returns F1 + boundary errors over matched pairs."""
    matched_gt, matched_pred = set(), set()
    onset_err, release_err = [], []
    pairs = []
    for i, pr in enumerate(pred):
        for j, gtx in enumerate(gt):
            iou = _seg_iou(pr, gtx)
            near = abs(pr[0] - gtx[0]) <= tol and abs(pr[1] - gtx[1]) <= tol
            if iou >= min_iou or near:
                pairs.append((iou, i, j))
    for iou, i, j in sorted(pairs, reverse=True):
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i); matched_gt.add(j)
        onset_err.append(abs(pred[i][0] - gt[j][0]))
        release_err.append(abs(pred[i][1] - gt[j][1]))
    tp = len(matched_gt); fp = len(pred) - tp; fn = len(gt) - tp
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(seg_f1=f1, seg_prec=prec, seg_rec=rec, tp=tp, fp=fp, fn=fn,
                onset_mae=float(np.median(onset_err)) if onset_err else None,
                release_mae=float(np.median(release_err)) if release_err else None)


def full_metrics(pred, gt, n, tol=5, min_iou=0.5):
    m = segment_metrics(pred, gt, tol, min_iou)
    m.update(frame_metrics(pred, gt, n))
    return m
