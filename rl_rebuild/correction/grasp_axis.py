"""对握轴候选生成 — 从物体 mesh 算出"该从哪儿夹、夹哪个方向", 纯几何.

替代"瞄 affordance 表面重心"的做法. 一对对握点同时给出三样东西:
  中点 = **受力中心**(在材料内部, 合拢点该去这里; 瞄表面会把物体推开)
  连线 = **夹持轴**(定手的朝向)
  长度 = 夹持宽度(要落在 Sharpa 夹口 5.06cm 的合理区间)

## 两个必须的坐标系步骤 (漏了会算出物理上不可行的解)

1. **转到 rest_q** —— env 里物体被重新平放 (stable pose, 修正重建的 32.5° 倾斜).
   在重建姿态下采样, 算出来的轴摆到桌上就不对.
2. **桌面筛选** —— 甜甜圈平躺时, 几何上最省力的夹法是"上下夹环壁"(宽 2.4cm),
   而那需要一根手指伸到物体**底下**, 那里是桌面. 必须筛掉近竖直的轴.
   实测: 不筛的话最优解 100% 落在这一类 (2026-07-28).

摩擦锥用指尖↔物体的**实际**摩擦: SuperGrip 3.0×3.0 = 9.0 (correction_env._setup_scene),
不是 clip 语义里的 0.5 —— 那是**桌子**的材质.

  $PY -m rl_rebuild.correction.grasp_axis --clip Grasp2 --mu 3.0 1.0 0.5
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rl_rebuild.correction import clips
from rl_rebuild.correction import frames as F
from rl_rebuild.correction.ref_builders.replay_grasp import (
    _affordance_target, _flat_rest_quat, grasp_center_local)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TABLE_Z = 0.85


def sample_pairs(clip, mu, n_sample=6000, n_seed=3000, seed=0, max_width=0.12):
    """-> list of dict(mid, axis, width, aff_dist, tilt_deg), 已转到 rest 姿态."""
    import trimesh
    e = clips.clip_entry(clip)
    m = trimesh.load(e["mesh"], process=True)
    verts = F.load_obj_verts(e["mesh"])
    c = verts.mean(0)
    rest_q = _flat_rest_quat(verts)

    P, fid = trimesh.sample.sample_surface(m, n_sample, seed=seed)
    N = m.face_normals[fid]
    P = P - c
    P = F.rot_apply(np.broadcast_to(rest_q, (len(P), 4)), P)
    N = F.rot_apply(np.broadcast_to(rest_q, (len(N), 4)), N)
    aff = F.rot_apply(rest_q[None], (_affordance_target(e["affordance"]) - c)[None])[0]

    half = np.arctan(mu)
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.choice(len(P), min(n_seed, len(P)), replace=False):
        d = P - P[i]
        L = np.linalg.norm(d, axis=1)
        # ⚠ 上限不能按"合拢终态跨度"设. 手张开时拇指→四指有 13.3cm (限位 16.4cm),
        #   合拢终态的 5.06cm 是**终点**不是**能张多大**. 早先按 5cm 截断, 导致
        #   跨直径 (7.7cm) 的解一个都采不到, 而且静默返回空 (2026-07-28 踩过).
        ok = (L > 0.015) & (L < max_width)
        if not ok.any():
            continue
        u = d[ok] / L[ok, None]
        # ⚠ 对握条件: 连线必须在两端都**指向材料内部**, 即 u ≈ −n_i 且 −u ≈ −n_j.
        #   2026-07-28 踩过: 两端符号写得不一致 (一端 u·n_i, 一端 (−u)·n_j),
        #   结果把教科书式的径向对握 (u·n_i=−0.993) 判成不合格 173°, 反而选中
        #   "一端朝外一端朝内"的斜穿线段. 可视化才看出来.
        a1 = np.arccos(np.clip(-(u @ N[i]), -1, 1))
        a2 = np.arccos(np.clip((u * N[ok]).sum(1), -1, 1))
        g = (a1 < half) & (a2 < half)
        for j in np.flatnonzero(ok)[g]:
            ax = P[j] - P[i]
            w = float(np.linalg.norm(ax))
            ax = ax / w
            mid = (P[i] + P[j]) / 2
            out.append(dict(mid=mid, axis=ax, width=w,
                            aff_dist=float(np.linalg.norm(mid - aff)),
                            tilt=float(np.degrees(np.arcsin(abs(ax[2])))),
                            p1=P[i], p2=P[j]))
    return out, aff


def filter_chain(cands, max_tilt=30.0, w_lo=0.02, w_hi=0.045, aff_max=0.02):
    """逐级筛选, 返回 {阶段名: 剩余数} 与最终候选."""
    stages = {"对握采样": len(cands)}
    c = [x for x in cands if x["tilt"] <= max_tilt]
    stages[f"轴倾角≤{max_tilt:.0f}°"] = len(c)
    c = [x for x in c if w_lo <= x["width"] <= w_hi]
    stages[f"夹持宽{w_lo*100:.0f}~{w_hi*100:.0f}cm"] = len(c)
    c = [x for x in c if x["aff_dist"] <= aff_max]
    stages[f"离affordance≤{aff_max*100:.0f}cm"] = len(c)
    c.sort(key=lambda x: x["aff_dist"])
    return stages, c


def ik_check(cands, obj_world_xy, hand="right", hover=0.03, topn=20):
    """对前 topn 个候选做 IK 校验 (用缓存的 arm_center 锚点; 见 check_placement 的告警)."""
    from rl_rebuild.correction.kinematics import ArmIK
    b = json.load(open(os.path.join(_REPO, "DEXMATE_BODY_POSES.json")))["bodies"]
    aT = np.eye(4)
    aT[:3, 3] = b["arm_center"]
    ik = ArmIK(hand, anchor_link="arm_center", anchor_T=aT)
    gc = grasp_center_local(hand)
    ok = 0
    for x in cands[:topn]:
        # 目标: 合拢中心落在中点正上方 hover; 朝向暂用"手指指向 -Z"的最简取法
        tgt = np.array([obj_world_xy[0] + x["mid"][0],
                        obj_world_xy[1] + x["mid"][1],
                        TABLE_Z + 0.0152 + x["mid"][2] + hover])
        R = np.eye(3)
        r = ik.solve(tgt - R @ gc, R)
        x["ik_ok"] = bool(r.get("ok"))
        ok += x["ik_ok"]
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="Grasp2")
    ap.add_argument("--mu", type=float, nargs="+", default=[3.0, 1.0, 0.5])
    ap.add_argument("--ik", action="store_true", help="额外跑 IK 校验(用缓存锚点)")
    a = ap.parse_args()
    for mu in a.mu:
        cands, aff = sample_pairs(a.clip, mu)
        stages, kept = filter_chain(cands)
        print(f"\n{'='*66}\nμ = {mu}   摩擦锥半角 {np.degrees(np.arctan(mu)):.1f}°")
        for k, v in stages.items():
            print(f"  {k:<22} {v:>9,}")
        if kept:
            b0 = kept[0]
            print(f"  最优: 宽 {b0['width']*100:.2f}cm  倾角 {b0['tilt']:.1f}°  "
                  f"离affordance {b0['aff_dist']*100:.2f}cm  轴 {np.round(b0['axis'],2)}")
            if a.ik:
                n = ik_check(kept, (0.0, 0.0))
                print(f"  IK 校验前 20 个候选: {n}/20 可达")
        else:
            print("  ⚠ 全部被筛掉")
