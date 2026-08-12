#!/usr/bin/env python3
"""尺度三路共识融合(最终层):L_extent(多帧点云跨度) x L_hand(手长锚点)
x L_typical(类别常识尺寸)。取代 scale_sanity_qwen 的"出界才 clamp"。

三路误差来源独立:extent 错在 mask/深度,hand 错在目测比例,typical 错在
物体不典型/认错。融合规则:
  1. Qwen 锚点先仲裁 extent 的两个聚合口径(median_all vs median_topclean,
     四视频验证无单一赢家):取更接近 geomean(L_hand, L_typical) 的那个。
  2. 三路两两比值均在 [1/TOL, TOL] -> consensus,直接用 L_extent(唯一"量出来"的)。
  3. L_extent 离群而两锚点互相一致 -> geomean(L_hand, L_typical),corrected_by_prior。
  4. 锚点互相打架 -> extent 与其一致者同侧则用 extent(partial_prior),
     否则 extent + low_confidence。
  5. identity confidence 低 -> 丢弃 L_typical(手长不依赖认对物体)。
铰接物体口径:prompt 明确要求按"画面中当前构型"报尺寸(摊开的笔记本按摊开算)。

一次 Qwen 调用;其余纯本地。输出 <run>/scale_fuse/<object_id>.json,
并对 sam3d_scale mesh 写等比修正副本 + correction_factor。

用法(sam3 env):
  python scale_fuse_qwen.py --run runs/s01_ketchup_grab_01 \
      --dataset arctic --video-id s01_ketchup_grab_01 [--hand-cm 18.5] [--tol 1.6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SD = Path(__file__).resolve().parent
RECON = SD.parent / "ego_pipeline/Reconstruction"
V17A = SD.parent / "experimental/hoi_detr_v17a"
sys.path.insert(0, str(V17A))
from experiments.hoi_detr.qwen_client import build_user_content, call_qwen  # noqa: E402

CROP_LONG_SIDE = 1024

SYSTEM = (
    "你是 3D 重建管线的尺度审核员。依据画面里人手与物体的相对大小以及你对该类物体的"
    "常识,估计物体的真实尺寸。人手(腕根到中指尖)长度约 18-19cm。"
    "只输出要求的 JSON,不要多余文字。"
)

PROMPT = """图1:手与物体接触/抓握的帧(手和物体距离相机深度相近,像素大小可直接比较)。
图2:物体的 mask 叠加图(红色区域即目标物体,请以这个物体为准)。
请估计目标物体的**最长跨度**。注意:一律按物体在**画面中的当前构型**估计
(例如摊开的笔记本按摊开后的跨度,而非合拢状态)。
只输出 JSON:
{
  "object_description": "目标物体是什么(一句话)",
  "identity_confidence": "high/medium/low",  // 你对认出这是什么物体的把握
  "current_configuration": "物体当前构型/状态(一句话,如'完全摊开'/'直立放置')",
  "hand_lengths": {"low": 数值, "high": 数值},   // 当前构型最长跨度 = 多少个手长,保守区间
  "typical_size_cm": {"low": 数值, "high": 数值}, // 这类物体在该构型下最长跨度通常多少厘米
  "reason": "判断依据(一句话)"
}"""


def parse_json(text: str) -> dict:
    import re
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON: {text[:200]}")
    return json.loads(match.group(0))


def geomean(lo: float, hi: float) -> float:
    return float(np.sqrt(max(lo, 1e-6) * max(hi, 1e-6)))


def ratio_ok(a: float, b: float, tol: float) -> bool:
    r = a / b
    return 1.0 / tol <= r <= tol


def mesh_longest_extent(obj_path: Path) -> float:
    vs = [[float(x) for x in l.split()[1:4]] for l in open(obj_path) if l.startswith("v ")]
    v = np.asarray(vs)
    return float((v.max(0) - v.min(0)).max())


def save_crop(frame: np.ndarray, path: Path, mb: np.ndarray | None = None) -> None:
    img = frame.copy()
    if mb is not None:
        img[mb] = (0.6 * img[mb] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    s = CROP_LONG_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--object-id", default=None)
    ap.add_argument("--recon-object", default="object_0")
    ap.add_argument("--hand-cm", type=float, default=18.5)
    ap.add_argument("--tol", type=float, default=1.6)
    args = ap.parse_args()
    run = args.run.resolve()
    interim = RECON / "data/interim" / args.dataset / args.video_id

    final = json.loads((run / "final_selection.json").read_text())
    valid = {o: e for o, e in final.items() if e.get("final_frame") is not None}
    targets = {o: e for o, e in valid.items() if e.get("interaction_target")}
    object_id = args.object_id or next(iter(targets or valid))
    report = json.loads((run / "frame_selection_v21/report_v2.json").read_text())["objects"][object_id]
    extent = json.loads((run / "scale_extent_v1" / f"{object_id}.json").read_text())

    # ── Qwen 锚点(一次调用) ──
    detail = report.get("occ_detail") or {}
    contact_f = (int(max(detail, key=lambda k: detail[k].get("contact", 0) + detail[k].get("occ_hull", 0)))
                 if detail else int(final[object_id]["final_frame"]))
    recon_f = int(final[object_id]["final_frame"])
    manifest = json.loads((run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())
    mask_path = {int(fr["frame_idx"]): fr["objects"][object_id]["mask"]
                 for fr in manifest["frames"]
                 if object_id in fr.get("objects", {}) and fr["objects"][object_id].get("mask")}
    out_dir = run / "scale_fuse"
    out_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(next(run.glob("*.mp4"))))

    def frame_at(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        assert ok, idx
        return fr

    p1 = out_dir / f"{object_id}_contact_f{contact_f:06d}.jpg"
    p2 = out_dir / f"{object_id}_recon_f{recon_f:06d}_overlay.jpg"
    save_crop(frame_at(contact_f), p1)
    save_crop(frame_at(recon_f), p2,
              cv2.imread(mask_path[recon_f], cv2.IMREAD_GRAYSCALE) > 0)
    cap.release()
    resp = call_qwen(SYSTEM, build_user_content(PROMPT, [p1, p2]))
    est = parse_json(resp.content)

    H = args.hand_cm / 100.0
    L_hand = geomean(float(est["hand_lengths"]["low"]), float(est["hand_lengths"]["high"])) * H
    L_typical = geomean(float(est["typical_size_cm"]["low"]), float(est["typical_size_cm"]["high"])) / 100.0
    conf = str(est.get("identity_confidence", "low")).lower()
    anchors = [L_hand] + ([L_typical] if conf in ("high", "medium") else [])
    anchor_mid = float(np.exp(np.mean(np.log(anchors))))

    # ── 规则1:锚点仲裁 extent 聚合口径 ──
    agg = extent["aggregate"]
    cand = {k: agg[k] for k in ("median_all_m", "median_topclean_m") if agg.get(k)}
    extent_key = min(cand, key=lambda k: abs(np.log(cand[k] / anchor_mid)))
    L_extent = float(cand[extent_key])

    # ── 规则2-5:共识融合 ──
    tol = args.tol
    hand_typ_agree = conf in ("high", "medium") and ratio_ok(L_hand, L_typical, tol)
    extent_vs_anchor = ratio_ok(L_extent, anchor_mid, tol)
    if extent_vs_anchor:
        L_final, verdict = L_extent, "consensus" if (hand_typ_agree or conf == "low") else "extent_with_partial_prior"
    elif hand_typ_agree or conf == "low":
        L_final, verdict = anchor_mid, "corrected_by_prior"
    else:
        L_final, verdict = L_extent, "low_confidence_extent"

    result = {
        "object_id": object_id,
        "qwen": est,
        "qwen_raw": resp.content,
        "contact_frame": contact_f,
        "recon_frame": recon_f,
        "L_hand_m": round(L_hand, 4),
        "L_typical_m": round(L_typical, 4),
        "identity_confidence": conf,
        "anchor_mid_m": round(anchor_mid, 4),
        "extent_aggregator_chosen": extent_key,
        "extent_candidates_m": cand,
        "L_extent_m": round(L_extent, 4),
        "tol": tol,
        "verdict": verdict,
        "L_final_m": round(L_final, 4),
    }

    # ── 对 sam3d_scale mesh 写修正副本 ──
    objdir = interim / "sam3d_scale/objects" / args.recon_object
    mesh = next((p for p in (objdir / "object_mesh_scaled_final.obj",
                             objdir / "object_mesh_scaled_stage1.obj") if p.exists()), None)
    if mesh is not None:
        L_geo = mesh_longest_extent(mesh)
        factor = L_final / L_geo
        result.update(L_geo_m=round(L_geo, 4), correction_factor=round(factor, 4),
                      source_mesh=str(mesh))
        corrected = out_dir / f"{object_id}_mesh_fused.obj"
        with open(mesh) as src, open(corrected, "w") as dst:
            for line in src:
                if line.startswith("v "):
                    parts = line.split()
                    xyz = [float(x) * factor for x in parts[1:4]]
                    dst.write(f"v {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
                              + (" " + " ".join(parts[4:]) if len(parts) > 4 else "") + "\n")
                else:
                    dst.write(line)
        result["fused_mesh"] = str(corrected)

    (out_dir / f"{object_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[{object_id}] extent={L_extent:.3f}m({extent_key}) hand={L_hand:.3f}m "
          f"typical={L_typical:.3f}m(conf={conf}) -> {verdict} L_final={L_final:.3f}m"
          + (f" (geo={result.get('L_geo_m')}m x{result.get('correction_factor')})" if mesh else ""))
    print("->", out_dir / f"{object_id}.json")


if __name__ == "__main__":
    main()
