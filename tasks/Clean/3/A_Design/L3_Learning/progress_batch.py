"""Clean/3 判据单一来源 (向量化, 无 Isaac 依赖): 覆盖率 × 行程 + 盘保持.

所有几何在**盘规范系** (z-up, 原点=盘心; 由盘输入系位姿 × canon_rot⁻¹ 得到) 里算:
  海绵规范系: x 7.5cm 宽, y 13.3cm 长, z 3.8cm 厚; 擦盘面 = -z 面, 离原点 face_offset (1.91cm)。
  盘顶面径向剖面 top(r): r<4.5cm 平底 -0.73cm, 4.5~8.5cm 斜坡到 +1.22cm。
"擦到" = 海绵足迹采样点沿 -z 偏 face_offset 后离盘面 < contact_tol 且 r<=rim。
成功 (台账 §5.2 A4) = 覆盖率 >= cover_min ∧ 行程 >= travel_min ∧ 盘倾角 < tilt_max ∧ 盘位移 < dev_max ∧ 双物未掉。
cover_min / travel_min 由 A0 零动作放音标定 (L5-27 铁则: 参考自己做得到), 见 DECISIONS §5.5。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CleanGeometry:
    plate_rim_r: float = 0.085
    sponge_half_x: float = 0.0375
    sponge_half_y: float = 0.0665
    sponge_face_offset: float = 0.0191
    contact_tol: float = 0.005
    cell: float = 0.01                      # 覆盖栅格 1cm
    grid_half: float = 0.09                 # 栅格 [-9,9]cm -> 18×18
    # 盘顶面径向剖面 (输入系 y = 规范系 z), 与 build_reference 同源 (mesh 实测)
    prof_r: tuple = (0.0025, 0.0075, 0.0125, 0.0175, 0.0225, 0.0275, 0.0325, 0.0375, 0.0425,
                     0.0475, 0.0525, 0.0575, 0.0625, 0.0675, 0.0725, 0.0775, 0.0825, 0.0875)
    prof_y: tuple = (-0.0075, -0.0074, -0.0073, -0.0073, -0.0073, -0.0073, -0.0072, -0.0072,
                     -0.0071, -0.0066, 0.0019, 0.0064, 0.0075, 0.0084, 0.0094, 0.0104, 0.0115, 0.0122)
    # 成功判据 (2026-09-07 定案: 参考自身几何成绩 覆盖 0.565 / 行程 87cm 的 80%; 碟心不算, 用户裁定 §5.6)
    cover_min: float = 0.45
    travel_min: float = 0.70                # m, 接触中的海绵中心面内行程 (**成功门槛**)
    # 行程奖励的归一化分母, **与成功门槛解耦**: 门槛按 take 重定时 (L5-27 铁则"参考自己做得到")
    # 不希望奖励斜率跟着变 —— 那是一次改两个参数, 判读表会失效 (台账"一次只改一个参数")。
    # 恒为 take3 定线时的 0.70; 默认路径下 travel_min == travel_norm, 与改动前逐位相同。
    travel_norm: float = 0.70
    tilt_max_deg: float = 15.0
    plate_dev_max: float = 0.03             # m, 盘心离参考/初始位置

    @classmethod
    def from_reference(cls, ref, **over):
        """按**母带实测**几何构造 (盘顶面剖面 + 海绵擦盘面偏移)。

        上面那套默认值是 take 3 的盘 (⌀18×2.4cm 正放浅碟) 实测值。换 take/换资产后必须跟着换 ——
        2026-09-10 take18 (倒扣盘, 厚 1.9cm 且圈足被削平) 实测: 判据仍拿 take3 剖面量,
        零动作 gap 报 −16.7mm ("海绵陷进盘里 1.7cm"), 覆盖/接触/落座全部失真。
        母带本来就存了 `plate_top_profile_r/y` 与 `sponge_face_offset` (hold_env 早就是这么读的),
        这里把判据也接到同一处。缺键的老母带退回默认 ⟹ take3 冠军线逐位不变。
        """
        import numpy as _np
        z = ref if hasattr(ref, "files") else _np.load(ref, allow_pickle=True)
        d = cls()
        kw = {}
        if "plate_top_profile_r" in z.files:
            pr = _np.asarray(z["plate_top_profile_r"], float).ravel()
            py = _np.asarray(z["plate_top_profile_y"], float).ravel()
            fo = float(z["sponge_face_offset"]) if "sponge_face_offset" in z.files else d.sponge_face_offset
            # 同一块盘就沿用已发布的默认值: 上面那串是 take3 盘的实测值**手抄进代码时四舍五入过**
            # (剖面差 ≤0.05mm, face_off 0.0191 vs 实测 0.018934 差 0.17mm)。这 0.2mm 若跟着母带走,
            # take3 冠军存档的判据尺子就被挪了一下, 与已发布成绩不再严格可比 (台账"改共享代码必跑对照")。
            # ⟹ 差 <0.5mm 判为同一块盘, 走默认; 真换了盘 (take18 倒扣盘差 13mm) 才整套换成母带实测。
            # 逐项比, 只换真的不一样的那一项 (take18 换的是盘, 海绵仍是 take3 那块 ⟹ face_off 不该跟着动)
            if not (len(pr) == len(d.prof_r)
                    and _np.abs(pr - _np.asarray(d.prof_r)).max() < 5e-4
                    and _np.abs(py - _np.asarray(d.prof_y)).max() < 5e-4):
                kw["prof_r"] = tuple(float(v) for v in pr)
                kw["prof_y"] = tuple(float(v) for v in py)
            if abs(fo - d.sponge_face_offset) >= 5e-4:
                kw["sponge_face_offset"] = fo
        # 成功门槛可按 take 重定 (L5-27 铁则: 参考自己做得到)。take 8 的参考行程只有 59cm,
        # 沿用 take3 那条 70cm 会让 success 结构性为 0 ⟹ 按同口径 (参考自身 80%) 给 47cm。
        # 默认不传 = 原值, take3/18 线逐位不变; 奖励斜率走 travel_norm 不受影响。
        import os as _os
        if _os.environ.get("CLEAN_TRAVEL_MIN_CM"):
            kw["travel_min"] = float(_os.environ["CLEAN_TRAVEL_MIN_CM"]) / 100.0
        if _os.environ.get("CLEAN_COVER_MIN"):
            kw["cover_min"] = float(_os.environ["CLEAN_COVER_MIN"])
        kw.update(over)
        return cls(**kw)


def quat_conjugate(q):
    out = q.clone(); out[..., 1:] = -out[..., 1:]; return out


def quat_mul(a, b):
    w1, x1, y1, z1 = a.unbind(-1); w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def quat_apply(q, v):
    qw, qv = q[..., :1], q[..., 1:]
    return v + 2.0 * torch.cross(qv, torch.cross(qv, v, dim=-1) + qw * v, dim=-1)


class CleanSignals:
    """把世界系位姿翻译成盘规范系信号 (逐步调用)."""

    def __init__(self, num_envs, device, q_ci, geometry: CleanGeometry = CleanGeometry()):
        self.g = geometry; self.N = int(num_envs); self.dev = torch.device(device)
        self.q_ci = torch.as_tensor(q_ci, dtype=torch.float32, device=self.dev).expand(self.N, 4)
        g = geometry
        fx = np.arange(-g.sponge_half_x + 0.0075, g.sponge_half_x, 0.01)
        fy = np.arange(-g.sponge_half_y + 0.0065, g.sponge_half_y, 0.01)
        fxx, fyy = np.meshgrid(fx, fy, indexing="ij")
        self.foot = torch.tensor(np.stack([fxx.ravel(), fyy.ravel(), -g.sponge_face_offset * np.ones(fxx.size)], 1),
                                 dtype=torch.float32, device=self.dev)     # (F,3) 擦盘面采样点 (海绵规范系)
        n = int(round(2 * g.grid_half / g.cell))
        gx = np.arange(-g.grid_half, g.grid_half, g.cell) + g.cell / 2
        cx, cy = np.meshgrid(gx, gx, indexing="ij")
        self.disk = torch.tensor((cx**2 + cy**2) <= g.plate_rim_r**2, device=self.dev)
        self.n_cell = n
        self.prof_r = torch.tensor(g.prof_r, dtype=torch.float32, device=self.dev)
        self.prof_y = torch.tensor(g.prof_y, dtype=torch.float32, device=self.dev)

    def top_at(self, r):
        rr = r.clamp(self.prof_r[0], self.prof_r[-1])
        idx = torch.searchsorted(self.prof_r, rr.reshape(-1)).clamp(1, len(self.prof_r) - 1)
        r0, r1 = self.prof_r[idx - 1], self.prof_r[idx]; y0, y1 = self.prof_y[idx - 1], self.prof_y[idx]
        return (y0 + (y1 - y0) * (rr.reshape(-1) - r0) / (r1 - r0)).reshape(r.shape)

    def seat_height(self, xy, psi):
        """平放海绵 (面内 yaw ψ) 中心在盘规范系 xy 时的落座高度 (原点 z): 足迹最高支撑点 + face_offset.
        刚体海绵跨在碟心平底与斜坡缘之间, 由最高的支撑点决定高度 (13.3cm 长, 碟心平底只有 9cm)."""
        c, s_ = torch.cos(psi), torch.sin(psi)
        fx, fy = self.foot[:, 0], self.foot[:, 1]
        px = xy[:, 0:1] + c[:, None] * fx[None] - s_[:, None] * fy[None]
        py = xy[:, 1:2] + s_[:, None] * fx[None] + c[:, None] * fy[None]
        r = torch.sqrt(px**2 + py**2)
        return self.top_at(r).amax(dim=1) + self.g.sponge_face_offset

    def __call__(self, plate_pos, plate_quat, sponge_pos, sponge_quat):
        """plate_*/sponge_*: 输入系刚体位姿 (世界). 返回盘规范系信号 dict."""
        N, F = self.N, self.foot.shape[0]
        pqc = quat_mul(plate_quat, quat_conjugate(self.q_ci))
        sqc = quat_mul(sponge_quat, quat_conjugate(self.q_ci))
        p_sc = quat_apply(quat_conjugate(pqc), sponge_pos - plate_pos)
        q_rel = quat_mul(quat_conjugate(pqc), sqc)
        pts = quat_apply(q_rel[:, None, :].expand(-1, F, -1).reshape(-1, 4),
                         self.foot[None].expand(N, -1, -1).reshape(-1, 3)).reshape(N, F, 3) + p_sc[:, None, :]
        rp = torch.linalg.vector_norm(pts[:, :, :2], dim=2)
        gap = pts[:, :, 2] - self.top_at(rp)
        touch = (gap < self.g.contact_tol) & (rp <= self.g.plate_rim_r)
        up = quat_apply(pqc, torch.tensor([0., 0., 1.], device=self.dev).expand(N, 3))
        tilt = torch.acos(up[:, 2].clamp(-1, 1))
        return dict(sponge_in_plate=p_sc, sponge_r=torch.linalg.vector_norm(p_sc[:, :2], dim=1),
                    gap_min=gap.amin(dim=1), touch=touch, contact=touch.any(dim=1),
                    foot_pts=pts, plate_tilt=tilt, plate_up=up)


class CleanProgressBatch:
    """覆盖率 (earn-only) + 行程 (接触中累计) + 保持判据; 里程碑 gates: [接触, 覆盖过半, 覆盖达标, 成功]."""

    def __init__(self, num_envs, device, signals: CleanSignals, geometry: CleanGeometry = CleanGeometry()):
        self.g = geometry; self.S = signals; self.N = int(num_envs); self.dev = torch.device(device)
        n = signals.n_cell
        self.cover = torch.zeros(self.N, n, n, dtype=torch.bool, device=self.dev)
        self.travel = torch.zeros(self.N, device=self.dev)
        self.prev_xy = torch.zeros(self.N, 2, device=self.dev)
        self.has_prev = torch.zeros(self.N, dtype=torch.bool, device=self.dev)
        self.prev_cover = torch.zeros(self.N, device=self.dev)
        self.gates = torch.zeros(self.N, 4, dtype=torch.bool, device=self.dev)
        self.plate_ref = torch.zeros(self.N, 3, device=self.dev)
        self.disk_n = self.S.disk.sum().float()

    def reset(self, env_ids, plate_ref_pos=None):
        self.cover[env_ids] = False; self.travel[env_ids] = 0; self.has_prev[env_ids] = False
        self.prev_cover[env_ids] = 0; self.gates[env_ids] = False
        if plate_ref_pos is not None:
            self.plate_ref[env_ids] = plate_ref_pos

    def coverage(self):
        return (self.cover & self.S.disk[None]).sum(dim=(1, 2)).float() / self.disk_n

    def step(self, sig, plate_pos, dropped):
        g = self.g
        pts, touch = sig["foot_pts"], sig["touch"]
        idx = torch.floor((pts[:, :, :2] + g.grid_half) / g.cell).long().clamp(0, self.S.n_cell - 1)
        flat = (idx[:, :, 0] * self.S.n_cell + idx[:, :, 1])
        env_idx = torch.arange(self.N, device=self.dev)[:, None].expand_as(flat)
        cov_flat = self.cover.view(self.N, -1)
        cov_flat[env_idx[touch], flat[touch]] = True
        xy = sig["sponge_in_plate"][:, :2]
        d = torch.linalg.vector_norm(xy - self.prev_xy, dim=1) * (self.has_prev & sig["contact"]).float()
        self.travel += d; self.prev_xy = xy.clone(); self.has_prev[:] = True
        cov = self.coverage()
        cov_delta = (cov - self.prev_cover).clamp_min(0.0); self.prev_cover = torch.maximum(self.prev_cover, cov)
        tilt_ok = sig["plate_tilt"] < np.radians(g.tilt_max_deg)
        dev_ok = torch.linalg.vector_norm(plate_pos - self.plate_ref, dim=1) < g.plate_dev_max
        hold_ok = tilt_ok & dev_ok & ~dropped
        old = self.gates.clone()
        self.gates[:, 0] |= sig["contact"]
        self.gates[:, 1] |= cov >= 0.5 * g.cover_min
        self.gates[:, 2] |= cov >= g.cover_min
        self.gates[:, 3] |= self.gates[:, 2] & (self.travel >= g.travel_min) & hold_ok
        new_gate = self.gates & ~old
        reward = 10.0 * cov_delta + 0.5 * d * sig["contact"].float() / max(g.travel_norm, 1e-6) * 10.0
        reward = reward + 0.5 * new_gate[:, 0] + 1.0 * new_gate[:, 1] + 4.0 * new_gate[:, 2] + 12.0 * new_gate[:, 3]
        return {**sig, "coverage": cov, "cov_delta": cov_delta, "travel": self.travel.clone(),
                "tilt_ok": tilt_ok, "dev_ok": dev_ok, "hold_ok": hold_ok,
                "gates": self.gates.clone(), "new_gate": new_gate, "success": self.gates[:, 3].clone(),
                "task_reward": reward}
