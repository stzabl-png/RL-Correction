"""Stage 3: one 3-DoF translation that puts the SharpaWave hand on the object.

Each degree of freedom is driven by the evidence that actually constrains it:

  perpendicular to the viewing ray (2 DoF)   the hand's occlusion of the object. Shift
      the hand 1 cm sideways and the "bite" it takes out of the object silhouette moves
      visibly, so this pins the image-plane position hard.
  along the viewing ray (1 DoF)   the object's 3D geometry. A silhouette barely changes
      when you slide along the ray (only by a 1/z scale), so it cannot fix depth;
      "touch the surface, do not penetrate it" can, because the object pose is trusted.

Plus one negative term: object surface that the camera plainly sees uncovered cannot be
in contact, so pads must not claim to touch there.

We match the bite ON THE OBJECT rather than the whole hand silhouette on purpose: the
EgoDex hand masks include the entire forearm, which the SharpaWave hand does not have,
so a full-silhouette IoU would be biased by an arm that cannot be modelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observe2d import project_points, splat_mask


# --------------------------------------------------------------- object geometry
class ObjectGeometry:
    """Distance-to-surface and inside/outside for the object at one frame.

    Both are precomputed lookups (KD-tree over surface samples, voxel occupancy in the
    object's own frame) because the search below evaluates them thousands of times.
    """

    def __init__(self, mesh, T_world, n_samples=200_000, pitch=0.004, seed=0):
        import trimesh
        from scipy.spatial import cKDTree

        self.T = np.asarray(T_world, dtype=np.float64)
        pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
        self.surface_world = pts @ self.T[:3, :3].T + self.T[:3, 3]
        self.tree = cKDTree(self.surface_world)

        self.occ = None
        try:                                       # SAM3D meshes are usually watertight
            vg = mesh.voxelized(pitch=pitch).fill()
            self.occ = np.asarray(vg.matrix, dtype=bool)
            self.occ_inv = np.linalg.inv(np.asarray(vg.transform, dtype=np.float64))
        except Exception as e:                     # non-watertight -> no penetration term
            self.occ_error = repr(e)

    def distance(self, P_world) -> np.ndarray:
        return self.tree.query(P_world)[0]

    def inside(self, P_world) -> np.ndarray:
        if self.occ is None:
            return np.zeros(len(P_world), bool)
        Pl = (P_world - self.T[:3, 3]) @ self.T[:3, :3]          # world -> object local
        idx = np.rint(Pl @ self.occ_inv[:3, :3].T + self.occ_inv[:3, 3]).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.occ.shape)), axis=1)
        out = np.zeros(len(P_world), bool)
        out[ok] = self.occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out


# --------------------------------------------------------------- image evidence
@dataclass
class ImageEvidence:
    """Observed masks + object depth, downscaled so the search stays cheap."""
    K: np.ndarray
    c2w: np.ndarray
    hw: tuple
    proj_obj: np.ndarray
    obj_depth: np.ndarray
    bite_obs: np.ndarray
    free_obs: np.ndarray
    scale: float
    splat_r: int = 1
    splat_close: int = 5

    @classmethod
    def build(cls, take, f, ev, scale=0.25):
        import cv2
        H, W = ev["proj_obj"].shape
        h, w = int(round(H * scale)), int(round(W * scale))
        K = take.K.copy()
        K[:2] *= scale

        def down_b(m):
            return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        d = ev["obj_depth"].copy()
        d[~np.isfinite(d)] = 1e6
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        # the splat footprint must track the resolution, otherwise a fixed 1-px radius
        # leaves holes at full res and the rendered bite (and its IoU) is under-reported
        return cls(K=K, c2w=take.c2w[f], hw=(h, w), proj_obj=down_b(ev["proj_obj"]),
                   obj_depth=d, bite_obs=down_b(ev["bite2d"]), free_obs=down_b(ev["free2d"]),
                   scale=scale, splat_r=max(1, int(round(5 * scale))),
                   splat_close=max(3, int(round(25 * scale)) | 1))

    def render_bite(self, P_world, depth_eps=0.004) -> np.ndarray:
        """Which object pixels this hand point cloud would hide."""
        uv, z, inb = project_points(P_world, self.c2w, self.K, self.hw)
        ok = inb & (z > 0)
        if not ok.any():
            return np.zeros(self.hw, bool)
        # only points genuinely in front of the object's front surface occlude it
        front = np.zeros(len(P_world), bool)
        front[ok] = z[ok] < self.obj_depth[uv[ok, 1], uv[ok, 0]] + depth_eps
        m = splat_mask(uv, ok & front, self.hw, radius=self.splat_r, close=self.splat_close)
        return m & self.proj_obj

    def free_hits(self, P_world) -> np.ndarray:
        uv, z, inb = project_points(P_world, self.c2w, self.K, self.hw)
        ok = inb & (z > 0)
        out = np.zeros(len(P_world), bool)
        out[ok] = self.free_obs[uv[ok, 1], uv[ok, 0]]
        return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


# --------------------------------------------------------------- the solver
class TranslationAligner:
    """Solves for a translation; optionally also a small rotation about the pad centroid.

    Rotation is OFF by default and exists as a controlled comparison: if translation
    alone plateaus at a mediocre bite IoU, the question "is the hand's orientation also
    wrong?" should be answered with a number, not a guess.
    """

    def __init__(self, geom: ObjectGeometry, img: ImageEvidence, hand_pts, pad_pts,
                 pad_link_id, gap=0.004, contact_band=0.008,
                 w_sil=1.0, w_pen=1.5, w_touch=0.7, w_neg=0.5, pen_metric="mean"):
        self.g, self.img = geom, img
        self.hand, self.pad, self.pid = hand_pts, pad_pts, pad_link_id
        self.pivot = pad_pts.mean(0)
        self.gap, self.band = gap, contact_band
        self.w = dict(sil=w_sil, pen=w_pen, touch=w_touch, neg=w_neg)
        self.pen_metric = pen_metric
        self.n_eval = 0

    def pose(self, params) -> tuple:
        """params = t (3,) or [t, rotvec] (6,) -> (R, t). Rotation is about self.pivot."""
        p = np.asarray(params, dtype=np.float64)
        if p.size == 3:
            return np.eye(3), p
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(p[3:]).as_matrix(), p[:3]

    def apply(self, params, P) -> np.ndarray:
        R, t = self.pose(params)
        return (P - self.pivot) @ R.T + self.pivot + t

    def terms(self, params) -> dict:
        self.n_eval += 1
        hand, pad = self.apply(params, self.hand), self.apply(params, self.pad)

        e_sil = 1.0 - iou(self.img.render_bite(hand), self.img.bite_obs)

        d_hand = self.g.distance(hand)
        pen = np.where(self.g.inside(hand), d_hand, 0.0)
        # 'mean' spreads the violation over every hand point, and since only ~1% of them
        # ever penetrate, it dilutes a real 1.7 cm intrusion into a near-zero number --
        # raising w_pen then fights the dilution rather than the physics. 'max' scores the
        # worst point, which is what "must not interpenetrate" actually means.
        if self.pen_metric == "max":
            e_pen = float((pen.max() / 0.01) ** 2)
        elif self.pen_metric == "pmean":
            # mean over the points that actually penetrate: keeps a usable gradient (unlike
            # 'max', which is a wall the optimiser just retreats from, losing contact) while
            # not being diluted by the ~99% of hand points that never touch (unlike 'mean')
            hit = pen[pen > 0]
            e_pen = float(np.mean((hit / 0.01) ** 2)) if hit.size else 0.0
        elif self.pen_metric == "q95":
            e_pen = float((np.quantile(pen, 0.95) / 0.01) ** 2)
        else:
            e_pen = float(np.mean((pen / 0.01) ** 2))

        d_pad = self.g.distance(pad)
        # one representative distance per pad link: a grasp needs SOME pads on the
        # surface, not all of them, so take each link's closest point.
        per_link = np.array([d_pad[self.pid == i].min() for i in np.unique(self.pid)])
        e_touch = float(np.mean((np.maximum(0.0, per_link - self.gap) / 0.02) ** 2))

        touching = d_pad < self.band
        e_neg = float((touching & self.img.free_hits(pad)).sum() / max(touching.sum(), 1))

        total = sum(self.w[k] * v for k, v in
                    dict(sil=e_sil, pen=e_pen, touch=e_touch, neg=e_neg).items())
        return dict(total=total, sil=e_sil, pen=e_pen, touch=e_touch, neg=e_neg,
                    bite_iou=1.0 - e_sil, pen_max_cm=float(pen.max() * 100),
                    pad_min_cm=float(d_pad.min() * 100),
                    pad_link_min_cm=(per_link * 100).round(2).tolist())

    def energy(self, params) -> float:
        return self.terms(params)["total"]

    def _grid_descend(self, best_p, best_e, slots, span, step, max_restarts=3):
        """Coordinate-block grid search over the 3 parameters listed in `slots`."""
        g = np.arange(-span, span + 1e-9, step)
        for _ in range(max_restarts):
            base = best_p.copy()
            for a in g:
                for b in g:
                    for c in g:
                        p = base.copy()
                        p[list(slots)] += np.array([a, b, c])
                        e = self.energy(p)
                        if e < best_e - 1e-9:
                            best_e, best_p = e, p
            if np.allclose(best_p, base):
                break
        return best_p, best_e

    # -- search ---------------------------------------------------------------
    def initial_guess(self, ev, mesh_V_world) -> np.ndarray:
        """Put the pads where the video says the hand is: at the object surface the
        hand hides, pushed out along the surface normal by roughly a finger radius."""
        occl = ev["occluded"]
        if occl.sum() < 20:
            target = self.g.T[:3, 3]                                # fall back to centre
        else:
            target = mesh_V_world[occl].mean(0)
            out = target - self.g.T[:3, 3]
            n = np.linalg.norm(out)
            target = target + (out / n) * 0.02 if n > 1e-6 else target
        return target - self.pad.mean(0)

    # ── 试过并**否决**的做法: 三维平移 ICP 种子 (2026-08-13) ──
    # 想法: 最终解里沿视线只占 2.4cm、垂直于视线占 15.3cm, 所以"只解深度"只优化了 14%,
    #       不如反复"查指垫最近表面点 -> 整体挪过去", 三维一起给好起点。
    # 结果: **更差**, 已删除实现。同一用例(clip0 左手×瓶身 f67, 正确手位):
    #       指垫到面 0.004 -> 0.4 cm;  2D 咬合 IoU 1.000 -> 0.000;  食指垫差 2.91cm;
    #       评估次数只省 18% (3941->3212), 耗时几乎不变 (246->231s)。
    # 原因: 纯 ICP 只管"指垫贴表面", **完全不看 2D 证据** —— 它把手拉出去 21.4cm,
    #       停在一个"指垫确实贴着表面、但投影根本不在物体上"的位置, 优化器再也拉不回来。
    #       要三维一起解, 必须把轮廓项也纳入目标, 那就不是廉价的最近点迭代了。

    def hand_depth_seed(self, t0, cam_pos, span=0.12, step=0.002, log=None):
        """沿视线**解**出深度, 而不是撒点碰运气。→ (seed, 诊断)

        `ray_seeds` 的注释说清了病因: 轮廓项对深度是盲的, 沿射线串着几个几乎等价的
        能量盆地, 所以要撒 5 个起点各跑一遍完整优化 —— 5 倍开销买一个"别掉错盆地"。

        但深度这条轴上我们其实**有测量**: 接触帧上手正贴着物体, 而手是米制可信通道
        (EgoDex 设备追踪; ARCTIC 上 HaWoR 锚后 74mm 也远好于物体深度)。
        于是把"撞运气"换成"直接找让指垫最贴合表面的那个偏移":

            对每个候选 s: d_i = 指垫点(平移 t0 + s·d) 到物体表面的距离
            取使 **中位距离**最小的 s  —— 中位而非均值: 被遮挡/没参与接触的垫点
            会拖偏均值, 中位对它们不敏感。

        只做 KDTree 最近点查询, 不渲染, 一次扫描 ~120 个候选也只要毫秒级。
        返回的 seed 交给原来的优化器继续精修, 垂直于视线的两个自由度仍由 2D 证据管
        —— 那两个本来就准, 不该动。
        """
        import numpy as _np
        t0 = _np.asarray(t0, dtype=_np.float64)
        d = self.pad.mean(0) + t0 - _np.asarray(cam_pos, dtype=_np.float64)
        n = _np.linalg.norm(d)
        if n < 1e-9:
            return t0, {"ok": False, "why": "指垫与相机重合, 视线方向不定"}
        d = d / n
        ss = _np.arange(-span, span + 1e-9, step)
        med = _np.array([float(_np.median(self.g.distance(self.pad + t0 + s * d))) for s in ss])
        k = int(_np.argmin(med))
        seed = t0 + ss[k] * d
        info = {"ok": True, "s_m": float(ss[k]), "median_gap_m": float(med[k]),
                "median_gap_at_t0_m": float(med[int(len(ss) // 2)]),
                "n_candidates": int(len(ss))}
        if log:
            log(f"[align]  手定深度: 沿视线移 {ss[k]*100:+.1f}cm -> 指垫到面中位 "
                f"{med[k]*1000:.1f}mm (t0 处 {med[len(ss)//2]*1000:.1f}mm), "
                f"扫了 {len(ss)} 个候选")
        return seed, info

    def ray_seeds(self, t0, cam_pos, depths=(-0.06, -0.03, 0.03, 0.06)) -> list:
        """t0 plus copies displaced ALONG THE VIEWING RAY.

        The silhouette term is depth-blind (sliding along the ray barely changes what the
        hand covers), so the energy landscape has several near-equivalent basins strung
        out along that ray, and a single-start search picks whichever one it happens to
        fall into. Measured: take 11 moved from IoU 0.543 to 0.392 on a weight change that
        could not affect it, which is basin-hopping, not a real difference. Seeding
        explicitly along the ambiguous direction turns that lottery into a search.
        """
        d = self.pad.mean(0) + t0 - np.asarray(cam_pos, dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-12)
        return [np.asarray(t0, dtype=np.float64)] + [t0 + s * d for s in depths]

    def solve_multistart(self, seeds, coarse=(0.10, 0.025), keep=2,
                         fine=((0.03, 0.008), (0.008, 0.002)), log=print) -> dict:
        """Coarse pass from every seed, then refine only the most promising ones.

        Refining all of them would cost one full search each; refining the best `keep`
        costs ~30% over a single-start run and still escapes the wrong basin.
        """
        coarse_res = []
        for i, s in enumerate(seeds):
            t, e = self._grid_descend(np.asarray(s, dtype=np.float64), self.energy(s),
                                      (0, 1, 2), coarse[0], coarse[1], max_restarts=2)
            coarse_res.append((e, t, i))
        coarse_res.sort(key=lambda r: r[0])
        if log:
            log("[align]  coarse energies by seed: "
                + ", ".join(f"#{i}={e:.4f}" for e, _, i in sorted(coarse_res, key=lambda r: r[2])))

        best_e, best_t, best_i = coarse_res[0]
        for e0, t0_, i in coarse_res[:keep]:
            t, e = t0_, e0
            for span, step in fine:
                t, e = self._grid_descend(t, e, (0, 1, 2), span, step, max_restarts=2)
            if e < best_e:
                best_e, best_t, best_i = e, t, i
        spread = coarse_res[-1][0] - coarse_res[0][0]
        if log:
            log(f"[align]  winner = seed #{best_i}, E={best_e:.4f}, "
                f"seed spread {spread:.4f} ({self.n_eval} evals)")
        out = self.terms(best_t)
        out.update(t=best_t, params=best_t, rotvec=np.zeros(3),
                   winning_seed=int(best_i), seed_energy_spread=float(spread),
                   seed_energies=[float(e) for e, _, _ in sorted(coarse_res, key=lambda r: r[2])])
        return out

    def solve(self, t0, schedule=((0.10, 0.025), (0.03, 0.008), (0.008, 0.002)),
              max_restarts=3, log=print) -> dict:
        """Coarse-to-fine grid search. Derivative-free on purpose: the silhouette IoU is
        piecewise constant, so gradients are meaningless and the 40 cm initial error
        needs a search with a real basin, not a local step."""
        best_t = np.asarray(t0, dtype=np.float64)
        best_e = self.energy(best_t)
        for span, step in schedule:
            g = np.arange(-span, span + 1e-9, step)
            for _ in range(max_restarts):         # let the optimum walk out of the box
                base = best_t.copy()
                for dx in g:
                    for dy in g:
                        for dz in g:
                            t = base + np.array([dx, dy, dz])
                            e = self.energy(t)
                            if e < best_e - 1e-9:
                                best_e, best_t = e, t
                if np.allclose(best_t, base):
                    break
            if log:
                log(f"[align]  span +-{span*100:.0f} cm / step {step*100:.1f} cm -> "
                    f"E={best_e:.4f}  |t|={np.linalg.norm(best_t)*100:.1f} cm  "
                    f"({self.n_eval} evals)")
        best = self.terms(best_t)
        best["t"] = best_t
        best["params"] = best_t
        best["rotvec"] = np.zeros(3)
        return best

    def solve_rigid(self, t_init, rounds=((np.radians(45), np.radians(15), 0.03, 0.008),
                                          (np.radians(15), np.radians(5), 0.01, 0.0025),
                                          (np.radians(5), np.radians(1.5), 0.004, 0.001)),
                    log=print) -> dict:
        """Translation + small rotation, by alternating grid descent on each block.

        A 6-D grid is out of reach, but the two blocks barely interact once the hand is
        already on the object: rotation reshapes the occlusion, translation re-seats it.
        """
        p = np.concatenate([np.asarray(t_init, dtype=np.float64), np.zeros(3)])
        e = self.energy(p)
        for r_span, r_step, t_span, t_step in rounds:
            p, e = self._grid_descend(p, e, (3, 4, 5), r_span, r_step)
            p, e = self._grid_descend(p, e, (0, 1, 2), t_span, t_step)
            if log:
                log(f"[align]  rot +-{np.degrees(r_span):.0f}deg/{np.degrees(r_step):.1f} "
                    f"trans +-{t_span*100:.1f}cm/{t_step*100:.2f} -> E={e:.4f} "
                    f"(rot {np.degrees(np.linalg.norm(p[3:])):.1f}deg, {self.n_eval} evals)")
        best = self.terms(p)
        best["t"], best["rotvec"], best["params"] = p[:3], p[3:], p
        return best
