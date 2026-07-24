"""Layer 1 —— 2D 手-物协同运动闸(co-motion gate),零重建、几乎零成本。

问题:HOI-DETR 的 hf-link 只是"手-物关联/靠近",手搁在物体旁也高置信触发,
不能当抓取用(见记忆 hoi-detr-env-run)。本模块用**主动协同运动**把真抓从
"靠近/静态搁着"里分出来:真抓时手实际去搬动物体、物体作为刚体跟着手动。

输入:
  - v17A 第 2 步产物 video_mask_sequence.json(SAM2 固定 ID 实例的逐帧 centroid_xy)
  - recon take 的逐帧左右手 mask(left_hand_0/right_hand_0.png,与实例同 1080x1920 系)
输出:每个(手 side × 实例 object_id)的 co-motion 指标 + 是否"真抓"判定;
      并可写成 contact_auto.json(带物体实例身份),供 phase.auto 消费。

判据(active co-motion):
  真抓 = 手在≥min_active_frames 帧里明显移动(>move_px),且这些帧里物体跟随
         (物体均速>follow_px 且 手物位移方向余弦相关>corr_thr)。
  纯"静态搁着"(手几乎不动)因缺乏主动操作证据而被剔除 —— 代价是"静握不动"
  这种也会被剔(假阴),那类残余歧义交给 Layer 2(VLM)。

局限:见上;co-motion 只在有运动时判别,全程静止的握持无法与静止搁置区分。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HANDS = ("left", "right")


@dataclass
class CoMotionConfig:
    move_px: float = 10.0          # 手"明显移动"的逐帧位移阈值(px/帧)
    min_active_frames: int = 8     # 需要多少帧手明显移动才算有主动操作证据
    follow_px: float = 8.0         # 主动帧里物体被认为"跟随"的均速阈值(px/帧)
    corr_thr: float = 0.5          # 手/物位移方向余弦相关阈值
    min_pixels: int = 30           # mask 面积下限(小于视为空)


def _hand_centroid(take_dir: Path, side: str, i: int, min_pixels: int):
    p = take_dir / "masks" / "hands" / "frames" / f"frame_{i:06d}_masks" / f"{side}_hand_0.png"
    if not p.is_file():
        return None
    try:
        from PIL import Image
        m = np.asarray(Image.open(p).convert("L")) > 127
    except Exception:
        return None
    if int(m.sum()) < min_pixels:
        return None
    ys, xs = np.where(m)
    return np.array([xs.mean(), ys.mean()])


def _instance_centroids(manifest: dict) -> dict[str, dict[int, np.ndarray]]:
    """从 video_mask_sequence.json 取每个固定 ID 实例的逐帧质心(status=accepted)。"""
    out: dict[str, dict[int, np.ndarray]] = {oid: {} for oid in manifest.get("object_ids", [])}
    for f in manifest.get("frames", []):
        i = int(f["frame_idx"])
        for oid, ov in f.get("objects", {}).items():
            if ov.get("status") != "accepted":
                continue
            c = (ov.get("metrics") or {}).get("centroid_xy")
            if c:
                out.setdefault(oid, {})[i] = np.asarray(c, dtype=float)
    return out


def _pair_metrics(hand_ts: dict[int, np.ndarray], obj_ts: dict[int, np.ndarray],
                  cfg: CoMotionConfig) -> dict | None:
    fr = sorted(set(hand_ts) & set(obj_ts))
    if len(fr) < 8:
        return None
    H = np.array([hand_ts[i] for i in fr])
    O = np.array([obj_ts[i] for i in fr])
    rel_std = float(np.linalg.norm((O - H).std(axis=0)))
    dH, dO = [], []
    idx = {f: k for k, f in enumerate(fr)}
    for a, b in zip(fr[:-1], fr[1:]):
        if b - a == 1:  # 只用连续帧算速度,避免跨空帧放大
            dH.append(H[idx[b]] - H[idx[a]])
            dO.append(O[idx[b]] - O[idx[a]])
    if len(dH) < 3:
        return None
    dH = np.array(dH); dO = np.array(dO)
    hs = np.linalg.norm(dH, axis=1); os_ = np.linalg.norm(dO, axis=1)
    active = hs > cfg.move_px                      # 手明显移动的帧
    mv = (hs > 3) & (os_ > 3)
    corr = float(np.mean((dH[mv] * dO[mv]).sum(1)
                         / (np.linalg.norm(dH[mv], axis=1) * np.linalg.norm(dO[mv], axis=1) + 1e-6))) \
        if int(mv.sum()) >= 3 else float("nan")
    follow = float(np.mean(os_[active])) if int(active.sum()) > 0 else 0.0
    active_corr = float(np.mean((dH[active] * dO[active]).sum(1)
                                / (np.linalg.norm(dH[active], axis=1) * np.linalg.norm(dO[active], axis=1) + 1e-6))) \
        if int(active.sum()) >= 3 else float("nan")
    return {
        "coexist_frames": len(fr), "rel_std": rel_std, "corr": corr,
        "hand_mean_speed": float(hs.mean()), "active_frames": int(active.sum()),
        "obj_speed_when_active": follow, "active_corr": active_corr,
        "frame_span": [int(fr[0]), int(fr[-1])],
    }


def _is_grasp(m: dict, cfg: CoMotionConfig) -> bool:
    if m is None:
        return False
    ac = m["active_corr"]
    return (m["active_frames"] >= cfg.min_active_frames
            and m["obj_speed_when_active"] >= cfg.follow_px
            and not np.isnan(ac) and ac >= cfg.corr_thr)


def evaluate(instance_dir, take_dir, cfg: CoMotionConfig | None = None) -> dict:
    """对一条 take 评估每(手 × 实例)的 co-motion,返回判定 + 指标。

    instance_dir: v17A 第 2 步 INSTANCE_OUTPUT(含 video_mask_sequence/video_mask_sequence.json)
    take_dir:     recon take 目录(含 masks/hands/frames/.../left_hand_0.png)
    """
    cfg = cfg or CoMotionConfig()
    instance_dir, take_dir = Path(instance_dir), Path(take_dir)
    vms = instance_dir / "video_mask_sequence" / "video_mask_sequence.json"
    manifest = json.loads(vms.read_text())
    obj_ts = _instance_centroids(manifest)
    hand_ts = {s: {i: c for i in range(_max_frame(manifest) + 1)
                   if (c := _hand_centroid(take_dir, s, i, cfg.min_pixels)) is not None}
               for s in HANDS}

    pairs, grasps = {}, {}
    for side in HANDS:
        for oid, ots in obj_ts.items():
            m = _pair_metrics(hand_ts[side], ots, cfg)
            pairs[f"{side}|{oid}"] = m
            if _is_grasp(m, cfg):
                grasps.setdefault(side, []).append({"object_id": oid, "frame_span": m["frame_span"],
                                                    "active_corr": m["active_corr"]})
    return {"video": manifest.get("video"), "object_ids": manifest.get("object_ids", []),
            "config": asdict(cfg), "pairs": pairs, "grasps": grasps}


def _max_frame(manifest: dict) -> int:
    return max((int(f["frame_idx"]) for f in manifest.get("frames", [])), default=0)


def to_contact_auto(result: dict, num_frames: int, fps: float | None = None) -> dict:
    """把 grasp 判定转成 contact_auto.json(带物体实例身份);段=被抓实例的 frame_span。"""
    from .types import segments_to_dense  # 复用
    ann = {"left": [], "right": []}
    inst = {"left": [], "right": []}
    for side, gl in result.get("grasps", {}).items():
        for g in gl:
            ann[side].append([int(g["frame_span"][0]), int(g["frame_span"][1])])
            inst[side].append(g["object_id"])
    return {
        "num_frames": int(num_frames), "fps": fps, "source": "auto",
        "method": "comotion_gate_v1", "params": result.get("config", {}),
        "annotations": ann, "grasped_instances": inst,
    }


def _fmt(result: dict) -> str:
    lines = [f"video: {result['video']}", f"实例: {result['object_ids']}"]
    lines.append("(手|实例)          共存 rel_std  corr  手均速 主动帧 主动时物速 主动corr  判定")
    for k, m in result["pairs"].items():
        if m is None:
            lines.append(f"{k:22s} 帧太少"); continue
        g = "✅真抓" if _is_grasp(m, CoMotionConfig(**result["config"])) else "❌剔除"
        lines.append(f"{k:22s} {m['coexist_frames']:3d} {m['rel_std']:6.1f} {m['corr']:+.2f} "
                     f"{m['hand_mean_speed']:5.1f} {m['active_frames']:4d} {m['obj_speed_when_active']:6.1f} "
                     f"{m['active_corr']:+.2f}   {g}")
    for side, gl in result.get("grasps", {}).items():
        for g in gl:
            lines.append(f"  -> {side} 抓 {g['object_id']} @ 帧{g['frame_span']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Layer1 co-motion gate: 真抓 vs 靠近/搁着")
    ap.add_argument("instance_dir", help="v17A 第2步 INSTANCE_OUTPUT")
    ap.add_argument("take_dir", help="recon take 目录(含 masks/hands)")
    ap.add_argument("--write", metavar="contact_auto.json", default=None, help="写 contact_auto.json 到此路径")
    ap.add_argument("--num-frames", type=int, default=None)
    args = ap.parse_args()
    res = evaluate(args.instance_dir, args.take_dir)
    print(_fmt(res))
    if args.write:
        n = args.num_frames or (_max_frame(json.loads((Path(args.instance_dir) / "video_mask_sequence" / "video_mask_sequence.json").read_text())) + 1)
        Path(args.write).write_text(json.dumps(to_contact_auto(res, n), ensure_ascii=False, indent=1))
        print("wrote", args.write)
