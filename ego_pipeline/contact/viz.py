"""Diagnostic figures. Everything here is for a human to eyeball -- no pipeline logic."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .observe2d import project_points


def figure_align(take, ev, mesh, before, after, out_path, extra_px=()):
    """Stage 3: did the translation reproduce the occlusion the video shows?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = ev["frame"]
    H, W = ev["obj_mask"].shape
    img = load_video_frame(take.video_path, f)

    fig = plt.figure(figsize=(17, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    for col, (tag, st) in enumerate((("before", before), ("after", after))):
        ax = fig.add_subplot(gs[0, col])
        _bg(ax, img, (H, W), f"({'ab'[col]}) {tag} alignment: observed bite (red) vs "
                             f"SharpaWave's bite (cyan)   IoU={st['bite_iou']:.3f}")
        overlay_mask(ax, ev["obj_mask"], (0.3, 0.6, 1.0), 0.22)
        overlay_mask(ax, ev["bite2d"], (1.0, 0.15, 0.15), 0.60)
        overlay_mask(ax, st["bite_pred_full"], (0.1, 0.95, 1.0), 0.45)

        ax = fig.add_subplot(gs[1, col])
        _bg(ax, img, (H, W), f"({'cd'[col]}) SharpaWave {tag}: pads red, surface grey")
        overlay_mask(ax, ev["hand_mask"], (0.2, 1.0, 0.2), 0.16)
        overlay_mask(ax, ev["obj_mask"], (0.3, 0.6, 1.0), 0.28)
        for pts, col_, size, lbl in st["layers"]:
            uvh, zh, inbh = project_points(pts, take.c2w[f], take.K, (H, W))
            m = inbh & (zh > 0)
            ax.scatter(uvh[m, 0], uvh[m, 1], s=size, c=[col_], linewidths=0, label=lbl)
        for px, col_, lbl in extra_px:
            ax.scatter([px[0]], [px[1]], s=200, marker="X", c=[col_], edgecolors="k",
                       linewidths=1.2, label=lbl, zorder=5)
        if col == 1:
            ax.legend(loc="upper left", fontsize=8, framealpha=0.65)

    a = after
    fig.suptitle(
        f"{take.recon_dir.name} | {take.side} hand | frame {f} | translation "
        f"{np.round(a['t']*100, 1).tolist()} cm (|t|={np.linalg.norm(a['t'])*100:.1f} cm)\n"
        f"bite IoU {before['bite_iou']:.3f} -> {a['bite_iou']:.3f}   |   "
        f"pad->surface min {before['pad_min_cm']:.1f} -> {a['pad_min_cm']:.1f} cm   |   "
        f"max penetration {a['pen_max_cm']:.1f} cm   |   "
        f"pads on plainly-visible surface {a['neg']*100:.0f}%", fontsize=10.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def figure_heatmap(take, ev, mesh, hm, pads_world, out_path, link_names=()):
    """Stage 4: the contact heatmap on the object, plus where it came from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .heatmap import colorize

    f = ev["frame"]
    H, W = ev["obj_mask"].shape
    img = load_video_frame(take.video_path, f)
    V = np.asarray(mesh.vertices)
    w = hm["weight"]
    step = max(1, len(V) // 40000)
    Vs, ws = V[::step], w[::step]
    rgb = colorize(ws) / 255.0

    fig = plt.figure(figsize=(17, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # three orbit views of the object, coloured by heat
    for i, (elev, azim) in enumerate(((22, -60), (22, 40), (70, -60))):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        order = np.argsort(ws)                                  # hot points drawn last
        ax.scatter(Vs[order, 0], Vs[order, 1], Vs[order, 2], c=rgb[order],
                   s=np.where(ws[order] > 0.05, 5.0, 1.0), linewidths=0, depthshade=False)
        r = np.abs(V).max()
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=5)
        ax.set_title(f"contact heat, object-local  (view {i+1})", fontsize=9)

    # the aligned hand on the video frame
    ax = fig.add_subplot(gs[1, 0])
    _bg(ax, img, (H, W), "aligned SharpaWave pads on the video frame")
    overlay_mask(ax, ev["obj_mask"], (0.3, 0.6, 1.0), 0.28)
    uvh, zh, inbh = project_points(pads_world, take.c2w[f], take.K, (H, W))
    m = inbh & (zh > 0)
    ax.scatter(uvh[m, 0], uvh[m, 1], s=2.0, c=[(1.0, 0.15, 0.15)], linewidths=0)

    # heat projected back onto the image
    ax = fig.add_subplot(gs[1, 1])
    _bg(ax, img, (H, W), "heat projected back into the image (veto region in blue)")
    sel = ev["visible"] & ev["in_bounds"]
    uv = ev["uv"][sel]
    ax.scatter(uv[ev["free"][sel], 0], uv[ev["free"][sel], 1], s=0.5,
               c=[(0.25, 0.55, 1.0)], linewidths=0)
    hot = sel & (w > 0.05)
    uvh2 = ev["uv"][hot]
    ax.scatter(uvh2[:, 0], uvh2[:, 1], s=2.0, c=colorize(w[hot]) / 255.0, linewidths=0)

    # per-finger breakdown
    ax = fig.add_subplot(gs[1, 2])
    from .heatmap import summarize_by_link
    per = summarize_by_link(hm, link_names)
    names = [n.split("_")[1] if "_" in n else n for n in per]
    ax.barh(names, list(per.values()), color="#d1495b")
    ax.set_xlabel("hot vertices (weight > 0.5)", fontsize=9)
    ax.set_title("which pad owns the contact", fontsize=9)
    ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{take.recon_dir.name} | {take.side} | frames {hm['frames']} | sigma "
        f"{hm['sigma']*1000:.0f} mm | hot verts {hm['n_hot']} "
        f"({hm['hot_area_frac']*100:.1f}% of the surface) | "
        f"vetoed by direct observation {hm['n_vetoed']}", fontsize=10.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def load_video_frame(video_path, f: int):
    import cv2
    if video_path is None or not Path(video_path).exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    cap.release()
    return img[:, :, ::-1].copy() if ok else None


def _bg(ax, img, hw, title):
    H, W = hw
    ax.imshow(img if img is not None else np.full((H, W, 3), 40, np.uint8))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def overlay_mask(ax, mask, color, alpha=0.35):
    rgba = np.zeros(mask.shape + (4,))
    rgba[..., :3] = color
    rgba[..., 3] = mask * alpha
    ax.imshow(rgba)


def figure_step12(take, obs, mesh, hand_pts, out_path, fk_info=None, sanity=None,
                  extra_px=()):
    """4-panel report for stages 1-2: frames unified, hand FK'd, contact region observed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = obs["frame"]
    H, W = obs["obj_mask"].shape
    img = load_video_frame(take.video_path, f)

    fig = plt.figure(figsize=(17, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) masks + contact band -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _bg(ax, img, (H, W), f"(a) frame {f}: hand mask (green) / object mask (blue) / "
                         f"object hidden by the hand = 'bite' (red, {obs['bite_px']} px, "
                         f"{obs['occlusion_ratio']*100:.0f}% of the object)")
    overlay_mask(ax, obs["hand_mask"], (0.2, 1.0, 0.2), 0.25)
    overlay_mask(ax, obs["obj_mask"], (0.3, 0.6, 1.0), 0.35)
    overlay_mask(ax, obs["bite2d"], (1.0, 0.15, 0.15), 0.75)

    # (b) observed contact region on the object -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _bg(ax, img, (H, W), f"(b) object surface evidence: hidden behind the hand "
                         f"(orange, {obs['n_occluded']}) vs plainly visible = provably "
                         f"NOT touched (blue, {obs['n_free']})")
    sel = obs["visible"] & obs["in_bounds"]
    uv = obs["uv"][sel]
    ax.scatter(uv[obs["free"][sel], 0], uv[obs["free"][sel], 1], s=0.6,
               c=[(0.25, 0.55, 1.0)], linewidths=0, label="free (no contact possible)")
    ax.scatter(uv[obs["occluded"][sel], 0], uv[obs["occluded"][sel], 1], s=1.2,
               c=[(1.0, 0.55, 0.05)], linewidths=0, label="occluded (contact may be here / behind)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.65)

    # (c) SharpaWave hand where the retarget currently puts it -----------------
    ax = fig.add_subplot(gs[1, 0])
    _bg(ax, img, (H, W), "(c) where the reconstruction puts the hand, vs where the video shows it")
    overlay_mask(ax, obs["hand_mask"], (0.2, 1.0, 0.2), 0.18)
    overlay_mask(ax, obs["obj_mask"], (0.3, 0.6, 1.0), 0.30)
    for pts, col, size, lbl in hand_pts:
        uvh, zh, inbh = project_points(np.atleast_2d(pts), take.c2w[f], take.K, (H, W))
        m = inbh & (zh > 0)
        ax.scatter(uvh[m, 0], uvh[m, 1], s=size, c=[col], linewidths=0, label=lbl)
    for px, col, lbl in extra_px:
        ax.scatter([px[0]], [px[1]], s=220, marker="X", c=[col], edgecolors="k",
                   linewidths=1.2, label=lbl, zorder=5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.65)

    # (d) the same region, on the object mesh in its own frame ------------------
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    V = np.asarray(mesh.vertices)
    step = max(1, len(V) // 30000)
    Vs = V[::step]
    occ, fre = obs["occluded"][::step], obs["free"][::step]
    rest = ~(occ | fre)
    ax.scatter(Vs[rest, 0], Vs[rest, 1], Vs[rest, 2], c="0.85", s=1.0,
               linewidths=0, depthshade=False)
    ax.scatter(Vs[fre, 0], Vs[fre, 1], Vs[fre, 2], c=[(0.25, 0.55, 1.0)], s=2.0,
               linewidths=0, depthshade=False)
    ax.scatter(Vs[occ, 0], Vs[occ, 1], Vs[occ, 2], c=[(1.0, 0.55, 0.05)], s=4.0,
               linewidths=0, depthshade=False)
    r = np.abs(V).max()
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("(d) object-local frame: orange = hidden by the hand,\n"
                 "blue = provably untouched, grey = unobserved (back side)", fontsize=10)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=6)

    l1 = (f"{take.recon_dir.name} | {take.side} hand | frame {f}/{take.Tv} | "
          f"scene->world residual {take.fit_residual_mm:.3f} mm")
    if sanity:
        l1 += f" | object-mask IoU {sanity['iou']:.3f}"
    fig.suptitle(l1 + ("\n" + fk_info if fk_info else ""), fontsize=10.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
