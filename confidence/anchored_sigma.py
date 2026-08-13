#!/usr/bin/env python3
"""锚后误差预测 σ + 物体参考轨迹的 σ 融合 —— 给 RL 用的逐帧不确定度。

============================ 它解决什么问题 ============================

RL 摆放物体时不信重建给的绝对位置：`ref_builders/replay_grasp.py` 第③步"物体听手"，
把**交互开始帧**的物体 XY 对齐到手的抓取锚点。那一刻的偏移被归零，所以**绝对位置误差
里的系统性偏移根本进不了仿真**。RL 承受的只是"从锚点出发之后轨迹怎么发散"（锚后误差）。

而 `conf_pos` 是拿绝对误差验证的，换到锚后口径后几乎失去预测力（帧级 rho 从 −0.32 掉到
−0.22，11 条里 3 条方向相反）——它的判据全是图像内轮廓比对，锚定后剩下的是相对形变。

============================ 定标（55 条 ARCTIC grab，7 subject） ============================

    σ_物体(mm) = 0.876 × 该帧离锚点位移(mm) + 1.2
    σ_手  (mm) = 0.440 × 该帧离锚点位移(mm) + 6.7

位移一律在**我们自己的重建世界系**里量 —— 这是运行时唯一能拿到的量。

**留一 subject 交叉验证**（每个 subject 轮流当验证集，参数只来自其余 subject）：

| | 物体 | 手 |
|---|---|---|
| 锚后误差中位 | 119mm | 74mm |
| 预测误差（运行时口径） | **40mm** | 33mm |
| 常数基线 | 66mm | 38mm |
| 胜率 | 6/7 subject | 6/7 |
| （对照）对齐口径 | 18mm | 31mm |

★ **物体侧 18→40mm 的落差全部来自世界尺度不确定**：相机 Umeyama 尺度在 55 条上
跨 **0.15–1.94**(对齐残差仅 8mm, 说明尺度是真的差这么多, 不是拟合不好)。
手侧 31→33mm 几乎不动 —— **手是米制正确的, 没有继承 ViPE 的尺度错**
(实测手幅度比中位 1.16, 与 1/s 的相关仅 −0.12)。
⇒ **修 ViPE 的世界尺度能把物体侧预测误差砍掉一半**, 比在下游打补丁值钱。

⚠ **手侧的 σ 收益很小**：常数基线 38mm 已逼近位移法 31mm —— 手的误差**基本均匀**，
不随位移变化，逐帧加权几乎没有意义。**真正值得逐帧加权的是物体侧**（66 → 18mm）。

⚠ **按物体类别给不同 k 已测试并放弃**：类别在新视频上未知；网格几何量（最长轴/最短轴/
长宽比）对 k 的相关全在 ±0.26 内（n=55 不显著）。且排除 5 条近静止 take 后 k 的标准差
从 0.71 塌到 0.24 —— 此前"类别能解释方差"主要是那几个离群值撑起来的。用单一常数。

============================ σ 融合：比现行做法更准 ============================

现行 `replay_grasp.py` 的物体参考**完全由腕合成**（位置 = 锚点 + 腕相对位移，朝向恒定），
理由是"相对量，对重建绝对偏移免疫"。实测（55 条 / 7979 接触帧，靶 = ARCTIC 真值物体位移）：

| 物体参考来源 | 中位 | p75 | p90 |
|---|---|---|---|
| 只用重建物体位移 | 116mm | 212 | 333 |
| 只用腕位移（现行） | 114mm | 181 | 249 |
| **σ 逆方差融合** | **94mm** | **157** | **213** |
| σ 硬切换 | 99mm | 166 | 229 |

两个来源**互补**（物体赢 31 条、腕赢 24 条），融合比现行**中位好 18%、p90 好 15%**，
也好过硬切换。融合权重 `w = σ_hand² / (σ_obj² + σ_hand²)`。

============================ 用法边界（务必读） ============================

1. ⚠⚠ **σ 只能由参考轨迹离线算，绝不能用仿真里物体实际走的距离。**
   σ 随位移增长；若用仿真中的实际位移，策略会发现"把物体推远 → 权重变小 → 惩罚消失"，
   直接学会甩开物体。这是标准的奖励黑客路径。本工具输出的是**离线固定的时间曲线**。
2. **σ 只管位置，不管朝向。** 旋转误差中位约 53°，锚定完全不改善；现行 track_object 的
   朝向是恒定值。要跟踪朝向必须另做可信度。
3. **低置信段的权重要留下限。** 权重趋 0 时该段没有任何信号，策略可能学会扔掉物体。
   `--weight-floor` 默认 0.15；或在 RL 侧保留一项不加权的"保持握住"约束。
4. **这不是挑帧器。** 按 σ 挑"误差最小的帧"会退化成"只留物体还没动的帧"
   （实测保留 25% 时运动覆盖率仅 52%），那些帧对 RL 恰恰最没信息。
5. **适用域**：全部证据来自**接触起始帧之后**的桌面抓取。抓握之前的物体误差没有数据；
   铰接（ARCTIC `*_use_*`）没有测过。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ---- 定标常数：55 条 ARCTIC grab，留一 subject 交叉验证，**运行时口径** ----
# ⚠ 回归量必须是"我们自己世界系里量的位移"(运行时唯一能拿到的), 不是"对齐到真值后的位移"。
#   用对齐后位移拟合会得到虚高的 18mm —— 那建立在"运行时能拿到真值做对齐"这个不成立的前提上。
K_OBJ, FLOOR_OBJ = 0.876, 1.2      # 物体: 预测误差 40mm (常数基线 66mm), 6/7 subject
K_HAND, FLOOR_HAND = 0.440, 6.7    # 手:   预测误差 33mm (常数基线 38mm), 6/7 — 收益很小
CALIB = "arctic15 55 takes / 7 subjects, leave-one-subject-out, runtime caliber 2026-08-12"
FP_ONSET_OFFSET = 10               # frame_plan 的 fp_register_frame = onset + 10


def mesh_centroid(scene: Path, oid: str) -> np.ndarray | None:
    """物体参考点 = 网格顶点质心(物体局部系)。整条 take 用同一点, 与真值对拍口径一致。"""
    for p in (scene / "objects" / oid / "object_mesh_scaled_final.obj",
              scene / "object_mesh_scaled_final.obj"):
        if p.is_file():
            v = np.array([[float(x) for x in ln.split()[1:4]]
                          for ln in p.read_text().splitlines() if ln.startswith("v ")])
            return v.mean(0) if len(v) else None
    return None


def find_anchor(scene: Path, n: int) -> tuple[int, str]:
    """锚点 = RL 摆放所用的**交互开始帧**。三档来源, 越靠前越权威。

    ⚠ confidence 是第 9 步, 此时 `replay_world.npz`(RL 真正读的 phase_ 数组)还不存在,
    只能用这三档等价物。实测 v17A 的 onset 与接触检测起始帧中位只差 1 帧(10/11 条在 3 帧内)。
    """
    ca = scene / "contact_auto.json"
    if ca.is_file():
        ann = json.loads(ca.read_text()).get("annotations", {})
        st = [sg[0] for sd in ("left", "right") for sg in (ann.get(sd) or [])]
        if st:
            return int(min(st)), "contact_auto"
    cands = [scene / "frame_plan.json"]
    try:
        cands += list((scene.parents[2] / "interim").glob(
            f"*/{scene.parent.name}__{scene.name}/sam2_object/frame_plan.json"))
    except IndexError:
        pass
    for fp in cands:
        if fp.is_file():
            o = json.loads(fp.read_text()).get("objects", {}).get("object_0", {})
            f = o.get("fp_register_frame")
            if f is not None:
                return max(int(f) - FP_ONSET_OFFSET, 0), "frame_plan(fp-10)"
    return int(0.3 * n), "fallback(30%)"        # 与 replay_grasp 的兜底同规则


def hand_at(ht: np.ndarray, hv: np.ndarray, n: int):
    """手流重采样到物体/相机的帧轴。

    ⚠ 两者帧率可能不同: ARCTIC 15fps 视频里 `hand_*` 长度是 `object_*` 的 **2 倍**
    (手按 30fps 标注帧索引)。直接用视频帧号取手会取到错误时刻且**不报任何错**。
    这里按比例重采样, 长度相同时退化为恒等。
    """
    nh = ht.shape[1]
    idx = np.round(np.linspace(0, nh - 1, n)).astype(int) if nh != n else np.arange(n)
    return ht[:, idx], hv[:, idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True, help="take 目录(含 world_fused.npz)")
    ap.add_argument("--out", type=Path, default=None, help="额外写一份到该目录")
    ap.add_argument("--weight-floor", type=float, default=0.15,
                    help="奖励权重下限; 0 会让低置信段完全无信号(策略可能扔掉物体)")
    ap.add_argument("--tolerance-mm", type=float, default=50.0,
                    help="任务位置容差 τ。权重 = τ²/(τ²+σ²): σ=τ 时 0.5, σ=3τ 时 0.1。"
                         "**别用 take 内的 min(σ) 做归一** —— min 落在锚点(σ≈7.6mm), "
                         "会把几乎所有帧压到下限。用 τ 才可解释且跨视频可比。")
    a = ap.parse_args(argv)

    z = np.load(a.scene / "world_fused.npz", allow_pickle=True)
    if "object_ob_in_world_all" in z.files:
        W = np.asarray(z["object_ob_in_world_all"], float)              # (N,T,4,4)
        ids = [str(s) for s in z["object_ids"]]
        valid = (np.asarray(z["object_pose_valid_by_frame"], bool)
                 if "object_pose_valid_by_frame" in z.files else np.ones(W.shape[:2], bool))
    else:
        W = np.asarray(z["object_ob_in_world"], float)[None]
        ids, valid = ["object_0"], np.ones((1, len(z["object_ob_in_world"])), bool)

    n = W.shape[1]
    anchor, src = find_anchor(a.scene, n)
    ht, hv = (np.asarray(z["hand_trans"], float), np.asarray(z["hand_valid"]).astype(bool)) \
        if "hand_trans" in z.files else (None, None)
    if ht is not None:
        ht, hv = hand_at(ht, hv, n)

    out: dict = {"schema_version": "anchored_sigma_v2", "n_frames": int(n),
                 "anchor_frame": int(anchor), "anchor_source": src,
                 "k_obj": K_OBJ, "floor_obj_mm": FLOOR_OBJ,
                 "k_hand": K_HAND, "floor_hand_mm": FLOOR_HAND,
                 "weight_floor": a.weight_floor, "tolerance_mm": a.tolerance_mm, "calibration": CALIB,
                 "note": "逐帧噪声尺度 + σ 融合参考; 非挑帧器; σ 必须离线算, 详见 docstring",
                 "objects": {}}
    arr: dict = {}

    for i, oid in enumerate(ids):
        c = mesh_centroid(a.scene, oid)
        if c is None:
            out["objects"][oid] = {"error": "找不到网格, 跳过"}
            continue
        p = np.einsum("tij,j->ti", W[i, :, :3, :3], c) + W[i, :, :3, 3]      # 世界系质心 (T,3)
        ok = valid[i] if valid.ndim == 2 else valid
        ai = anchor if (0 <= anchor < n and ok[anchor]) else int(np.argmax(ok))

        d_obj = p - p[ai]                                                    # 物体离锚位移 (m)
        s_obj = K_OBJ * np.linalg.norm(d_obj, axis=1) * 1000 + FLOOR_OBJ     # mm

        # ---- 手侧: 取接触期有效帧最多的那只手 ----
        # ⚠ 必须带着有效掩码往下传: hand_valid=False 的帧里 hand_trans 是陈旧/填充值,
        #   照用会污染融合(实测 p90 从 213mm 劣化到 249mm)。无效帧退回"只用物体"。
        d_hand = s_hand = m_hand = None
        if ht is not None:
            best = -1
            for h in range(ht.shape[0]):
                m = hv[h] & np.isfinite(ht[h]).all(1)
                if m[ai:].sum() > best and m[ai]:
                    best, d_hand, m_hand = m[ai:].sum(), ht[h] - ht[h][ai], m
            if d_hand is not None:
                s_hand = K_HAND * np.linalg.norm(d_hand, axis=1) * 1000 + FLOOR_HAND

        # ---- σ 逆方差融合 (实测比"只用腕"中位好 18%、p90 好 15%) ----
        if d_hand is not None:
            w = (s_hand ** 2 / (s_obj ** 2 + s_hand ** 2))
            w[~m_hand] = 1.0                       # 手无效 -> 该帧只用物体
            s_h = np.where(m_hand, s_hand, np.inf)
            d_fused = w[:, None] * d_obj + (1 - w)[:, None] * np.nan_to_num(d_hand)
            s_fused = np.sqrt(1.0 / (1.0 / s_obj ** 2 + 1.0 / s_h ** 2))
            w = w[:, None]
        else:
            w = np.ones((n, 1))
            d_fused, s_fused = d_obj, s_obj

        # 奖励权重: 相对**任务容差 τ** 归一 (σ=τ→0.5, σ=3τ→0.1), 再夹下限。
        # τ 而不是 take 内 min(σ): 后者落在锚点(σ≈floor), 会把几乎所有帧压到下限。
        tau2 = a.tolerance_mm ** 2
        w_rew = np.clip(tau2 / (tau2 + s_fused ** 2), a.weight_floor, 1.0)
        for k_, v_ in (("sigma_obj_mm", s_obj), ("sigma_fused_mm", s_fused),
                       ("disp_obj_mm", np.linalg.norm(d_obj, axis=1) * 1000),
                       ("fuse_weight_obj", w[:, 0]), ("reward_weight", w_rew),
                       ("ref_disp_fused_m", d_fused)):
            arr[f"{oid}__{k_}"] = v_
        if s_hand is not None:
            arr[f"{oid}__sigma_hand_mm"] = s_hand

        f = np.isfinite(s_fused)
        out["objects"][oid] = {
            "anchor_frame_used": int(ai),
            "sigma_obj_mm_median": round(float(np.median(s_obj[f])), 1),
            "sigma_fused_mm_median": round(float(np.median(s_fused[f])), 1),
            "sigma_fused_mm_p90": round(float(np.percentile(s_fused[f], 90)), 1),
            "reward_weight_median": round(float(np.median(w_rew[f])), 3),
            "hand_available": s_hand is not None,
            "n_valid": int(f.sum()),
        }

    (a.scene / "traj_sigma.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    np.savez(a.scene / "traj_sigma.npz", anchor_frame=anchor, **arr)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{'__'.join(a.scene.parts[-2:])}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1))
    for oid, r in out["objects"].items():
        if "error" in r:
            print(f"[anchored_sigma] {oid}: {r['error']}")
        else:
            print(f"[anchored_sigma] {oid}: 锚点 f{r['anchor_frame_used']} ({src})  "
                  f"σ物体 {r['sigma_obj_mm_median']}mm → σ融合 {r['sigma_fused_mm_median']}mm "
                  f"(p90 {r['sigma_fused_mm_p90']})  奖励权重中位 {r['reward_weight_median']}"
                  f"{'' if r['hand_available'] else '  ⚠无手数据, 未融合'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
