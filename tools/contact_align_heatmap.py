#!/usr/bin/env python
"""Align the retargeted SharpaWave hand onto the object it grasps, then map the contact.

Why: monocular reconstruction gets the hand's DEPTH wrong by centimetres, so the
retargeted hand either floats in front of the object or sinks into it. The image
however shows exactly where the human hand pressed (hand mask touching object mask),
and that observation carries no depth ambiguity. So: use what is seen (2D contact band)
to fix what is guessed (3D depth), by one rigid translation of the hand.

Stages:
  check    [this file, done]  unify frames, FK the hand, extract the observed contact
                              region, and render a 4-panel diagnostic figure
  align    [next]             solve one 3-DoF translation putting the pads on that region
  heatmap  [next]             per-vertex contact weight on the object mesh (object-local)

Usage:
  python tools/contact_align_heatmap.py <ReconstructOutput/take-dir> [--side right]
      [--frame N] [--video path.mp4] [--out-dir DIR] [--stage check]

Runs in the `biv2ap` conda env (numpy / trimesh / scipy / matplotlib / opencv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
from contact import frames as F                                        # noqa: E402
from contact import observe2d as O                                     # noqa: E402
from contact import align as A                                         # noqa: E402
from contact import heatmap as HM                                      # noqa: E402
from contact import urdf_fk, viz                                       # noqa: E402
from repo_paths import DATA_ROOT, RECON_OUTPUT                         # noqa: E402


def guess_video(recon_dir: Path):
    """<...>/ReconstructOutput/<dataset>/<rel...>/<take> -> Data/**/<rel...>/<take>.mp4"""
    try:
        rel = recon_dir.resolve().relative_to(Path(RECON_OUTPUT).resolve())
    except ValueError:
        return None
    tail = Path(*rel.parts[1:]) if len(rel.parts) > 1 else rel        # drop dataset name
    hits = sorted(Path(DATA_ROOT).glob(f"**/{tail}.mp4"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="take dir under ReconstructOutput")
    ap.add_argument("--retarget-dir", type=Path, default=None,
                    help="matching RetargetOutput take dir (default: mirrored path)")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--frame", type=int, default=None,
                    help="frame to analyse (default: first annotated contact frame)")
    ap.add_argument("--annotation", type=Path, default=None)
    ap.add_argument("--auto-frame", action="store_true",
                    help="pick the frame where the hand covers the most of the object "
                         "(use when grasp_annotation.json is missing)")
    ap.add_argument("--best-frame-in-interval", action="store_true",
                    help="(now the default; accepted for compatibility)")
    ap.add_argument("--interval-start", action="store_true",
                    help="use the annotated interval's FIRST frame instead of its "
                         "best-evidence frame. That frame is the moment contact begins, so "
                         "the hand barely covers the object and the silhouette term has "
                         "almost nothing to work with -- measured 0.11-0.33 bite IoU there "
                         "versus 0.61-0.89 at the best frame. Kept only for comparison.")
    ap.add_argument("--video", type=Path, default=None, help="mp4 for the figure background")
    ap.add_argument("--gt-hdf5", type=Path, default=None,
                    help="EgoDex GT hdf5, OFFLINE DIAGNOSTIC ONLY (default: <video>.hdf5 if present)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <RetargetOutput take>/contact")
    ap.add_argument("--stage", default="align", choices=["check", "align", "heatmap"])
    ap.add_argument("--ignore-quality", action="store_true",
                    help="run even on a take that take_quality_gate.py marked unusable")
    ap.add_argument("--no-qpos", action="store_true",
                    help="run without ref_qpos.npz (observation-only: no SharpaWave hand)")
    ap.add_argument("--dilate-px", type=int, default=9, help="hand-mask dilation for the contact band")
    ap.add_argument("--falloff-px", type=int, default=25, help="soft falloff around the band")
    ap.add_argument("--n-pad-pts", type=int, default=4000)
    ap.add_argument("--n-hand-pts", type=int, default=12000)
    ap.add_argument("--n-surface-pts", type=int, default=120000, help="object surface samples")
    ap.add_argument("--n-search-hand", type=int, default=2500, help="hand pts used during the search")
    ap.add_argument("--n-search-pad", type=int, default=900)
    ap.add_argument("--search-scale", type=float, default=0.2, help="image downscale for the search")
    ap.add_argument("--gap", type=float, default=0.004, help="target pad<->surface gap (m)")
    ap.add_argument("--w-sil", type=float, default=1.0)
    ap.add_argument("--w-pen", type=float, default=1.5)
    ap.add_argument("--w-touch", type=float, default=0.7,
                    help="do not raise this to fight the pads being pushed off the object: "
                         "tested at 1.5 with pen-metric=pmean and it backfired, because a "
                         "strong must-touch plus a strong must-not-penetrate pins the hand to "
                         "the nearest OUTER surface, which is the camera-facing one -- i.e. "
                         "exactly the surface proven untouched (takes 1 and 18 went to 100%% "
                         "negative-evidence violation)")
    ap.add_argument("--w-neg", type=float, default=3.0,
                    help="weight on the negative-evidence term. At the original 0.5 this "
                         "term contributed a median 9%% of the total energy (3%% on some "
                         "takes), so 'do not put pads where the camera proves there is no "
                         "hand' -- a hard constraint -- was cheaper to violate than a 3%% "
                         "change in silhouette overlap. 3.0 was measured better than 0.5 on "
                         "8 takes (violations 4/8 -> 1/8, median IoU 0.599 -> 0.617); it is "
                         "NOT a tuned optimum, only a tested improvement")
    ap.add_argument("--single-start", action="store_true",
                    help="one search from one seed (the old behaviour); by default the "
                         "search is seeded at several depths along the viewing ray, which "
                         "is the direction the silhouette term cannot resolve")
    ap.add_argument("--pen-metric", default="pmean", choices=["mean", "max", "pmean", "q95"],
                    help="how the penetration term scores a pose. 'mean' over all hand points "
                         "dilutes the violation ~10-19x because almost none penetrate; 'max' is "
                         "a wall the search just retreats from, losing contact (measured: IoU "
                         "0.599 -> 0.450). 'pmean' averages over the penetrating points only, "
                         "which keeps a gradient without the dilution: on 9 takes it cut median "
                         "penetration 0.93 -> 0.51 cm (max 2.50 -> 1.35), removed every "
                         "negative-evidence violation, and raised takes meeting "
                         "IoU>=0.5 & pen<=1cm & neg<=10%% from 3/9 to 5/9, at a 0.026 cost in "
                         "median IoU. NOTE pmean is ~10-19x larger than mean, so w_pen is not "
                         "comparable across the two")
    ap.add_argument("--params", default=None,
                    help="skip the search and use this transform: 'tx,ty,tz' in metres "
                         "(optionally +',rx,ry,rz' rotvec). Use to re-render or to reuse "
                         "a solve from a previous run.")
    ap.add_argument("--sigma", type=float, default=0.008, help="heatmap kernel width (m)")
    ap.add_argument("--no-veto", action="store_true",
                    help="do NOT zero the heat on surface the camera proves is untouched")
    ap.add_argument("--allow-rot", action="store_true",
                    help="after the translation solve, also search a small rotation about "
                         "the pad centroid, and report both (translation stays the default)")
    args = ap.parse_args()

    take = F.load_take(args.take, args.retarget_dir, args.side, args.annotation, args.video,
                       require_qpos=not args.no_qpos)
    if take.video_path is None:
        take.video_path = guess_video(take.recon_dir)
    out_dir = Path(args.out_dir) if args.out_dir else take.retarget_dir / "contact"
    out_dir.mkdir(parents=True, exist_ok=True)

    dq = take.recon_dir / "data_quality.json"
    if dq.exists():
        q = json.loads(dq.read_text())
        if q.get("verdict") == "rejected" and not args.ignore_quality:
            print(f"[quality] take {q['take']} was marked UNUSABLE by take_quality_gate.py:")
            for r in q.get("reasons", []):
                print(f"[quality]   - {r}")
            print("[quality] skipping. Pass --ignore-quality to run anyway.")
            return 2

    print(f"[take]   {take.recon_dir}")
    print(f"[take]   retarget: {take.retarget_dir}")
    print(f"[take]   Tv={take.Tv}  side={take.side}  intervals={take.intervals}")
    print(f"[take]   video: {take.video_path}")

    # ---- stage 0: frames --------------------------------------------------
    R_sw, t_sw = take.scene_from_world
    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_sw) - 1) / 2, -1, 1))))
    print(f"[frame]  table-scene <- recon-world recovered: t={np.round(t_sw, 4).tolist()} m, "
          f"rotation {rot_deg:.4f} deg")
    print(f"[frame]  fit residual max = {take.fit_residual_mm:.4f} mm", end="")
    if take.fit_residual_mm > 1.0:
        print("  <-- TOO LARGE: ref_qpos.npz and replay_world.npz disagree")
        return 1
    print("  OK")

    import trimesh
    mesh = trimesh.load(take.obj_mesh_path, force="mesh", process=False)
    print(f"[mesh]   {take.obj_mesh_path.name}: {len(mesh.vertices)} verts, "
          f"extents {np.round(mesh.extents, 3).tolist()} m")

    if args.frame is not None:
        f = args.frame
    elif args.interval_start and take.intervals:
        f = take.f0
        print(f"[frame]  using the interval's first frame {f} (--interval-start)")
    else:
        scan = take.contact_frames() if (take.intervals and not args.auto_frame) else None
        f, npx, _ = O.best_evidence_frame(take, mesh, scan, log=print)
        where = "inside the annotated interval" if scan is not None else "over the whole take"
        print(f"[frame]  picked frame {f} {where}: largest observed bite = {npx} px"
              + (f" (interval starts at {take.f0})" if take.intervals else ""))
    if not take.hand_valid[f]:
        print(f"[warn]   hand is INVALID at frame {f} (qpos was hole-filled from a neighbour)")
    if not take.obj_valid[f]:
        print(f"[warn]   object pose is invalid at frame {f}")

    sanity = F.object_projection_iou(take, f, mesh)
    print(f"[sanity] object projection vs SAM2 mask: IoU={sanity['iou']:.3f} "
          f"({sanity['n_projected']} verts, median depth {sanity['mean_depth_m']*100:.1f} cm)")
    if sanity["iou"] < 0.5:
        print("[sanity] WARNING: low IoU -- mesh/pose/K/camera convention may disagree")

    # ---- stage 1: SharpaWave forward kinematics ---------------------------
    T_obj = take.obj_T_world[f]
    obj_w = np.asarray(mesh.vertices) @ T_obj[:3, :3].T + T_obj[:3, 3]
    pad_pts = palm_pts = all_pts = None
    d_surf = np.array([np.nan])
    tip_err = np.array([np.nan])
    if take.finger_qpos is None:
        print("[fk]     ref_qpos.npz absent -> skipping SharpaWave FK "
              "(observation-only run; the hand cannot be drawn or aligned)")
    else:
        chain = urdf_fk.load_hand(take.side)
        poses = chain.fk_world(take.qpos_dict(f), take.wrist_pos[f], take.wrist_quat_wxyz[f])
        pad_links = chain.link_names(urdf_fk.PAD_SUFFIXES)
        palm_links = chain.link_names(urdf_fk.PALM_SUFFIXES)
        pad_pts, pad_link_id, _ = chain.sample_points(poses, pad_links, n=args.n_pad_pts, seed=0)
        palm_pts, _, _ = chain.sample_points(poses, palm_links, n=args.n_pad_pts, seed=2)
        all_pts, _, _ = chain.sample_points(poses, n=args.n_hand_pts, seed=1)
        print(f"[fk]     {len(chain.links)} links, {len(pad_links)} pads {pad_links}")
        print(f"[fk]     hand span {np.round(all_pts.max(0) - all_pts.min(0), 3).tolist()} m "
              f"(SharpaWave is ~0.20 m long)")

        # FK cross-check: SharpaWave fingertips should sit within a few cm of the MANO
        # fingertips they were retargeted from. Large gaps mean the FK/qpos/wrist
        # conventions disagree -- a bug in us, not in the reconstruction.
        tip_links = [f"{take.side}_{n}_fingertip"
                     for n in ("thumb", "index", "middle", "ring", "pinky")]
        tip_fk = np.stack([poses[n][:3, 3] for n in tip_links])
        tip_err = np.linalg.norm(tip_fk - take.mano_joints[f][[4, 8, 12, 16, 20]], axis=1)
        print(f"[fk]     fingertip FK vs MANO: {np.round(tip_err * 100, 1).tolist()} cm "
              f"(a few cm = normal retarget residual)")
        if tip_err.max() > 0.06:
            print("[fk]     WARNING: fingertips too far from MANO -- check FK / qpos conventions")

        # where the hand is relative to the object right now (the error to be fixed)
        from scipy.spatial import cKDTree
        tree = cKDTree(obj_w[:: max(1, len(obj_w) // 60000)])
        d_surf, _ = tree.query(pad_pts)
        print(f"[fk]     pad->object-surface distance: min {d_surf.min()*100:.1f} cm, "
              f"median {np.median(d_surf)*100:.1f} cm  |  pad->object-centre min "
              f"{np.linalg.norm(pad_pts - T_obj[:3, 3], axis=1).min()*100:.1f} cm")

    # ---- health: is the reconstructed hand even on the right pixels? -------
    health = F.hand_reprojection_health(take, take.contact_frames() if take.intervals else None)
    print(f"[health] reconstructed hand joints landing inside the observed hand mask: "
          f"mean {health['mean']*100:.0f}% over {health['n_frames']} contact frames "
          f"({health['good_frames']} frames > 50%)")
    if health["n_frames"] and health["mean"] < 0.2:
        print("[health] WARNING: the reconstructed hand does NOT reproject onto the video's hand.")
        print("[health]          The error is not depth-only, so ONE translation for the whole")
        print("[health]          trajectory cannot fix it -- align per contact frame instead.")

    gt = None
    if args.gt_hdf5 or (take.video_path and take.video_path.with_suffix(".hdf5").exists()):
        gt_path = args.gt_hdf5 or take.video_path.with_suffix(".hdf5")
        gt = F.egodex_gt_wrist(gt_path, take.side, f)
        if gt:
            print(f"[gt]     EgoDex GT wrist (offline check only): px {np.round(gt['px']).astype(int).tolist()}, "
                  f"depth {gt['depth_m']*100:.1f} cm  vs reconstruction depth "
                  f"{np.linalg.norm(take.wrist_pos[f] - take.c2w[f][:3, 3])*100:.1f} cm")

    # ---- stage 2: what the video says about where the hand is -------------
    ev = O.hand_occlusion_evidence(take, f, mesh, dilate_px=args.dilate_px,
                                   falloff_px=args.falloff_px)
    print(f"[obs2d]  object hidden by the hand: {ev['bite_px']} px "
          f"({ev['occlusion_ratio']*100:.1f}% of its silhouette), {ev['n_occluded']} verts")
    print(f"[obs2d]  object plainly visible (provably NOT touched): {ev['free_px']} px, "
          f"{ev['n_free']} verts")
    if ev["bite_px"] < 200:
        print("[obs2d]  WARNING: the hand barely occludes the object at this frame -- "
              "the silhouette term will be weak. Pick a frame deeper into the grasp.")

    # ---- stage 3: solve one translation -----------------------------------
    aligned = before = None
    if args.stage in ("align", "heatmap") and all_pts is not None:
        import trimesh as _tm
        geom = A.ObjectGeometry(mesh, T_obj, n_samples=args.n_surface_pts)
        if geom.occ is None:
            print(f"[align]  no voxel occupancy ({geom.occ_error}) -> penetration term disabled")
        img_lo = A.ImageEvidence.build(take, f, ev, scale=args.search_scale)
        img_hi = A.ImageEvidence.build(take, f, ev, scale=1.0)

        # cheap point sets for the search, full ones for the reported numbers
        rng = np.random.default_rng(0)
        s_hand = all_pts[rng.choice(len(all_pts), min(args.n_search_hand, len(all_pts)), replace=False)]
        si = rng.choice(len(pad_pts), min(args.n_search_pad, len(pad_pts)), replace=False)

        search = A.TranslationAligner(geom, img_lo, s_hand, pad_pts[si], pad_link_id[si],
                                      gap=args.gap, w_sil=args.w_sil, w_pen=args.w_pen,
                                      w_touch=args.w_touch, w_neg=args.w_neg,
                                      pen_metric=args.pen_metric)
        final = A.TranslationAligner(geom, img_hi, all_pts, pad_pts, pad_link_id,
                                     gap=args.gap, w_sil=args.w_sil, w_pen=args.w_pen,
                                     w_touch=args.w_touch, w_neg=args.w_neg,
                                     pen_metric=args.pen_metric)

        if args.params:
            params = np.array([float(v) for v in args.params.split(",")], dtype=np.float64)
            if params.size not in (3, 6):
                raise SystemExit("--params needs 3 (translation) or 6 (translation+rotvec) numbers")
            print(f"[align]  using the supplied transform, search skipped: "
                  f"t={np.round(params[:3]*100, 1).tolist()} cm"
                  + (f", rot={np.degrees(np.linalg.norm(params[3:])):.1f} deg" if params.size == 6 else ""))
            res = dict(t=params[:3], params=params)
        else:
            t0 = search.initial_guess(ev, obj_w)
            print(f"[align]  initial guess from the occluded surface patch: "
                  f"{np.round(t0*100, 1).tolist()} cm (|t0|={np.linalg.norm(t0)*100:.1f} cm)")
            if args.single_start:
                res = search.solve(t0)
            else:
                res = search.solve_multistart(search.ray_seeds(t0, take.c2w[f][:3, 3]))
        params = res["params"]
        rot_res = None
        if args.allow_rot and not args.params:
            print("[align]  --allow-rot: refining with a rotation about the pad centroid")
            rot_res = search.solve_rigid(res["t"])
            params = rot_res["params"]

        t = params[:3]
        before = final.terms(np.zeros(3))
        aligned = final.terms(params)
        aligned["params"] = params
        for k in ("winning_seed", "seed_energies", "seed_energy_spread"):
            if k in res:
                aligned[k] = res[k]
        aligned["t"], aligned["rotvec"] = t, params[3:] if params.size == 6 else np.zeros(3)
        before["t"], before["rotvec"] = np.zeros(3), np.zeros(3)
        trans_only = final.terms(res["t"]) if args.allow_rot else None
        for st, pp in ((before, np.zeros(3)), (aligned, params)):
            st["bite_pred_full"] = img_hi.render_bite(final.apply(pp, all_pts))
            st["layers"] = [(final.apply(pp, all_pts), (0.78, 0.78, 0.80), 0.7, "SharpaWave surface"),
                            (final.apply(pp, palm_pts), (1.0, 0.62, 0.10), 1.2, "palm"),
                            (final.apply(pp, pad_pts), (1.0, 0.15, 0.15), 2.0, "fingertip pads")]
        print(f"[align]  translation = {np.round(t*100, 1).tolist()} cm "
              f"(|t| = {np.linalg.norm(t)*100:.1f} cm)")
        cam_dir = take.wrist_pos[f] - take.c2w[f][:3, 3]
        cam_dir /= np.linalg.norm(cam_dir)
        print(f"[align]  decomposition: {np.dot(t, cam_dir)*100:+.1f} cm along the viewing ray, "
              f"{np.linalg.norm(t - np.dot(t, cam_dir)*cam_dir)*100:.1f} cm across it")
        print(f"[align]  bite IoU        {before['bite_iou']:.3f} -> {aligned['bite_iou']:.3f}")
        print(f"[align]  pad->surface    {before['pad_min_cm']:.1f} -> {aligned['pad_min_cm']:.1f} cm "
              f"(per pad link: {aligned['pad_link_min_cm']})")
        print(f"[align]  max penetration {aligned['pen_max_cm']:.2f} cm")
        print(f"[align]  pads sitting on provably-untouched surface: {aligned['neg']*100:.0f}%")
        if trans_only is not None:
            print(f"[align]  COMPARISON  translation only : bite IoU {trans_only['bite_iou']:.3f}, "
                  f"pad->surface {trans_only['pad_min_cm']:.1f} cm, "
                  f"penetration {trans_only['pen_max_cm']:.2f} cm")
            print(f"[align]  COMPARISON  + rotation {np.degrees(np.linalg.norm(aligned['rotvec'])):.1f} deg"
                  f" : bite IoU {aligned['bite_iou']:.3f}, "
                  f"pad->surface {aligned['pad_min_cm']:.1f} cm, "
                  f"penetration {aligned['pen_max_cm']:.2f} cm")
        if aligned["bite_iou"] < 0.35:
            print("[align]  NOTE: low bite IoU after translation -- the hand's ORIENTATION is")
            print("[align]        probably wrong too, which no translation can repair.")

    # ---- stage 4: contact heatmap on the object ---------------------------
    hm = pads_aligned = None
    if args.stage == "heatmap" and aligned is not None:
        pads_aligned = final.apply(aligned["params"] if "params" in aligned else aligned["t"],
                                   pad_pts)
        hm = HM.contact_heatmap(
            mesh,
            [dict(frame=f, T_obj=T_obj, pads=pads_aligned, link_id=pad_link_id,
                  free=None if args.no_veto else ev["free"], weight=1.0)],
            sigma=args.sigma)
        print(f"[heat]   sigma {args.sigma*1000:.0f} mm | hot verts (w>0.5) {hm['n_hot']} "
              f"= {hm['hot_area_frac']*100:.2f}% of the surface")
        print(f"[heat]   vetoed by direct observation: {hm['n_vetoed']} verts "
              f"({'disabled' if args.no_veto else 'enabled'})")
        per_link = HM.summarize_by_link(hm, pad_links)
        print(f"[heat]   per pad: {per_link}")
        ply = HM.export_ply(mesh, hm["weight"], out_dir / f"contact_heatmap_frame{f:04d}_{take.side}.ply")
        np.savez_compressed(
            out_dir / f"contact_heatmap_frame{f:04d}_{take.side}.npz",
            vertex_weight=hm["weight"].astype(np.float32),
            vertex_distance_m=hm["distance_m"].astype(np.float32),
            nearest_pad_link=hm["nearest_link"].astype(np.int16),
            pad_link_names=np.array(pad_links),
            vetoed=hm["vetoed"], occluded=ev["occluded"],
            vertices=np.asarray(mesh.vertices, np.float32),
            faces=np.asarray(mesh.faces, np.int32),
            sigma=args.sigma, frames=np.array(hm["frames"]),
            transform=aligned["params"] if "params" in aligned else aligned["t"],
            side=take.side, frame_of_reference="object_local")
        print(f"[out]    {ply}")

    # ---- figures ----------------------------------------------------------
    extra = [(gt["px"], (1.0, 0.0, 0.9), "EgoDex GT wrist (offline check)")] if gt else []
    fk_info = (("SharpaWave pad -> object surface: min %.1f cm, median %.1f cm   |   "
                % (d_surf.min() * 100, np.median(d_surf) * 100) if all_pts is not None else
                "no ref_qpos.npz -> SharpaWave hand not shown   |   ")
               + f"hand hides {ev['occlusion_ratio']*100:.0f}% of the object"
               + f"   |   recon hand inside its own hand mask: {health['mean']*100:.0f}%")
    layers = [] if all_pts is None else [
        (all_pts, (0.78, 0.78, 0.80), 0.7, "SharpaWave surface (FK)"),
        (palm_pts, (1.0, 0.62, 0.10), 1.2, "palm"),
        (pad_pts, (1.0, 0.15, 0.15), 2.0, "fingertip pads (elastomer)")]
    layers.append((take.mano_joints[f], (0.1, 0.9, 1.0), 26, "reconstructed MANO joints"))
    png = viz.figure_step12(take, ev, mesh, hand_pts=layers,
                            out_path=out_dir / f"stage12_frame{f:04d}_{take.side}.png",
                            fk_info=fk_info, sanity=sanity, extra_px=extra)
    outputs = [png]
    if hm is not None:
        outputs.append(viz.figure_heatmap(
            take, ev, mesh, hm, pads_aligned,
            out_dir / f"stage4_heatmap_frame{f:04d}_{take.side}.png", link_names=pad_links))
    if aligned is not None:
        outputs.append(viz.figure_align(
            take, ev, mesh, before, aligned,
            out_dir / f"stage3_align_frame{f:04d}_{take.side}.png", extra_px=extra))

    report = dict(
        take=str(take.recon_dir), retarget=str(take.retarget_dir), side=take.side,
        frame=int(f), Tv=take.Tv, intervals=take.intervals,
        scene_shift_m=np.asarray(t_sw).tolist(),
        scene_rotation_deg=rot_deg, frame_fit_residual_mm=take.fit_residual_mm,
        object_mask_iou=sanity["iou"], object_median_depth_m=sanity["mean_depth_m"],
        sharpa_hand_available=all_pts is not None,
        fingertip_fk_vs_mano_cm=np.round(tip_err * 100, 2).tolist(),
        pad_to_surface_min_cm=float(d_surf.min() * 100),
        pad_to_surface_median_cm=float(np.median(d_surf) * 100),
        hand_reprojection_health=health,
        egodex_gt=dict(px=np.round(gt["px"], 1).tolist(), depth_m=gt["depth_m"],
                       confidence=gt["confidence"]) if gt else None,
        occlusion=dict(bite_px=ev["bite_px"], free_px=ev["free_px"],
                       ratio=ev["occlusion_ratio"], occluded_verts=ev["n_occluded"],
                       free_verts=ev["n_free"]),
        figures=[str(p) for p in outputs],
    )
    if aligned is not None:
        report["alignment"] = dict(
            translation_m=aligned["t"].tolist(),
            translation_cm=np.round(aligned["t"] * 100, 2).tolist(),
            magnitude_cm=float(np.linalg.norm(aligned["t"]) * 100),
            bite_iou_before=before["bite_iou"], bite_iou_after=aligned["bite_iou"],
            pad_min_cm_before=before["pad_min_cm"], pad_min_cm_after=aligned["pad_min_cm"],
            pad_link_min_cm=aligned["pad_link_min_cm"],
            penetration_max_cm=aligned["pen_max_cm"],
            pads_on_free_surface=aligned["neg"],
            energy_terms={k: aligned[k] for k in ("total", "sil", "pen", "touch", "neg")},
            weights=dict(sil=args.w_sil, pen=args.w_pen, touch=args.w_touch,
                         neg=args.w_neg, pen_metric=args.pen_metric,
                         multistart=not args.single_start),
            winning_seed=aligned.get("winning_seed"),
            seed_energies=aligned.get("seed_energies"),
            seed_energy_spread=aligned.get("seed_energy_spread"),
            rotation_deg=float(np.degrees(np.linalg.norm(aligned["rotvec"]))),
            rotation_rotvec=aligned["rotvec"].tolist(),
            translation_only=None if trans_only is None else dict(
                bite_iou=trans_only["bite_iou"], pad_min_cm=trans_only["pad_min_cm"],
                penetration_max_cm=trans_only["pen_max_cm"], pads_on_free_surface=trans_only["neg"]),
            note="the reconstructed hand was displaced by far more than a depth error, so "
                 "this translation RE-PLACES the hand from 2D occlusion + object geometry "
                 "rather than nudging the original trajectory",
        )

    if hm is not None:
        report["heatmap"] = dict(
            sigma_m=args.sigma, veto_enabled=not args.no_veto,
            hot_vertices=hm["n_hot"], hot_area_fraction=hm["hot_area_frac"],
            vetoed_vertices=hm["n_vetoed"],
            per_pad_hot_vertices=HM.summarize_by_link(hm, pad_links),
            frame_of_reference="object_local")

    stage_tag = "4" if hm is not None else ("3" if aligned is not None else "12")
    rp = out_dir / f"stage{stage_tag}_frame{f:04d}_{take.side}.json"
    rp.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / f"evidence_frame{f:04d}_{take.side}.npz",
                        occluded=ev["occluded"], free=ev["free"],
                        rim_weight=ev["rim_weight"].astype(np.float32),
                        visible=ev["visible"], frame=f, side=take.side,
                        translation=np.zeros(3) if aligned is None else aligned["t"],
                        rotvec=np.zeros(3) if aligned is None else aligned["rotvec"])

    for o in outputs:
        print(f"\n[out]    {o}")
    print(f"[out]    {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
