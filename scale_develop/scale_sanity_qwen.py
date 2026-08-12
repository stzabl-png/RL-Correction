#!/usr/bin/env python3
"""scale 护栏(手锚点版):对 sam3d_scale 的几何尺度做常识性校验。

原理:成人手长(腕根到中指尖)≈18.5cm、方差±15%,是画面里唯一已知尺寸的参照物。
让 Qwen 在"手与物体接触的帧"上(同深度,像素比≈真实比)估计物体最长边是几个手长,
得到常识区间;几何尺度在区间的 α 倍护栏内则原样放行(几何精度高于目测,不动它),
出界才 clamp 到边缘并打标 —— 首要目标是拦离谱,第一戒律是别误伤。

用法(sam3 env):
  python scale_sanity_qwen.py --run runs/s01_ketchup_grab_01 \
      --scaled-mesh <object_mesh_scaled_final.obj> --scale-metadata <scale_metadata.json> \
      [--alpha 2.0] [--hand-cm 18.5]
输出: --run 下 scale_guardrail/<object>.json + (若修正) corrected mesh 副本。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

V17A = Path(__file__).resolve().parent.parent / "experimental/hoi_detr_v17a"
sys.path.insert(0, str(V17A))
from experiments.hoi_detr.qwen_client import build_user_content, call_qwen  # noqa: E402

HAND_VAR = 0.15      # 手长方差 ±15%
CROP_LONG_SIDE = 1024

SYSTEM = (
    "你是 3D 重建管线的尺度审核员。依据画面里人手与物体的相对大小估计物体的真实尺寸。"
    "人手(腕根到中指尖)长度约 18-19cm。只输出要求的 JSON,不要多余文字。"
)

PROMPT = """图1:手与物体接触/抓握的帧(手和物体距离相机深度相近,像素大小可直接比较)。
图2:物体的 mask 叠加图(红色区域即目标物体,请以这个物体为准)。
请估计:目标物体的**最长边**大约是多少个"手长"(1 手长 = 腕根到中指尖 ≈ 18-19cm)。
只输出 JSON:
{
  "object_description": "目标物体是什么(一句话)",
  "hand_lengths": {"low": 数值, "high": 数值},   // 物体最长边 = 多少个手长,给保守区间
  "typical_size_cm": {"low": 数值, "high": 数值}, // 交叉验证:这类物体最长边通常多少厘米
  "confidence": "high/medium/low",
  "reason": "判断依据(一句话)"
}"""


def parse_json(text: str) -> dict:
    import re
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON: {text[:200]}")
    return json.loads(match.group(0))


def mesh_longest_extent(obj_path: Path) -> float:
    vs = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(vs)
    return float((v.max(0) - v.min(0)).max())


def pick_contact_frame(report_obj: dict) -> int:
    """从 v2.1 report 里挑接触分最高的帧(抓握帧,手物同深度,最适合目测倍数)。"""
    detail = report_obj.get("occ_detail") or {}
    if not detail:
        return int(report_obj["chosen_frame"])
    return int(max(detail, key=lambda k: detail[k].get("contact", 0) + detail[k].get("occ_hull", 0)))


def save_crop(frame: np.ndarray, path: Path, mb: np.ndarray | None = None) -> None:
    img = frame.copy()
    if mb is not None:
        img[mb] = (0.6 * img[mb] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    scale = CROP_LONG_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="scale_develop 的视频 run 目录")
    ap.add_argument("--object-id", default=None, help="默认取 final_selection 里第一个非 spurious track")
    ap.add_argument("--scaled-mesh", required=True, type=Path, help="sam3d_scale 输出的 final mesh(米)")
    ap.add_argument("--scale-metadata", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--hand-cm", type=float, default=18.5)
    args = ap.parse_args()
    run = args.run.resolve()

    final = json.loads((run / "final_selection.json").read_text())
    object_id = args.object_id or next(
        oid for oid, e in final.items() if e.get("final_frame") is not None)
    report = json.loads((run / "frame_selection_v21/report_v2.json").read_text())["objects"][object_id]
    manifest = json.loads(
        (run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())

    out_dir = run / "scale_guardrail"
    out_dir.mkdir(exist_ok=True)
    video = next(run.glob("*.mp4"))
    cap = cv2.VideoCapture(str(video))

    def frame_at(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        assert ok, idx
        return fr

    def mask_at(idx: int) -> np.ndarray:
        for fr in manifest["frames"]:
            if int(fr["frame_idx"]) == idx:
                return cv2.imread(str(fr["objects"][object_id]["mask"]), cv2.IMREAD_GRAYSCALE) > 0
        raise KeyError(idx)

    # ── 几何尺度:直接量最终 mesh 的最长边(米) ──
    L_geo = mesh_longest_extent(args.scaled_mesh)

    # ── Qwen 常识区间:接触帧全图 + 重建帧 mask 叠加 ──
    contact_f = pick_contact_frame(report)
    recon_f = int(final[object_id]["final_frame"])
    p1 = out_dir / f"{object_id}_contact_f{contact_f:06d}.jpg"
    p2 = out_dir / f"{object_id}_recon_f{recon_f:06d}_overlay.jpg"
    save_crop(frame_at(contact_f), p1)
    save_crop(frame_at(recon_f), p2, mask_at(recon_f))
    resp = call_qwen(SYSTEM, build_user_content(PROMPT, [p1, p2]))
    est = parse_json(resp.content)

    H = args.hand_cm / 100.0
    k_lo, k_hi = float(est["hand_lengths"]["low"]), float(est["hand_lengths"]["high"])
    L_hand = float(np.sqrt(max(k_lo, 1e-3) * max(k_hi, 1e-3))) * H
    band = (L_hand / args.alpha, L_hand * args.alpha)

    result = {
        "object_id": object_id,
        "L_geo_m": round(L_geo, 4),
        "qwen": est,
        "qwen_raw": resp.content,
        "contact_frame": contact_f,
        "recon_frame": recon_f,
        "L_hand_anchor_m": round(L_hand, 4),
        "hand_range_m": [round(k_lo * H * (1 - HAND_VAR), 4), round(k_hi * H * (1 + HAND_VAR), 4)],
        "guard_band_m": [round(band[0], 4), round(band[1], 4)],
        "alpha": args.alpha,
    }
    if band[0] <= L_geo <= band[1]:
        result.update(verdict="pass", L_final_m=round(L_geo, 4), correction_factor=1.0)
    else:
        L_final = min(max(L_geo, band[0]), band[1])
        factor = L_final / L_geo
        result.update(verdict="corrected", L_final_m=round(L_final, 4),
                      correction_factor=round(factor, 4))
        # 修正后 mesh 副本(顶点等比缩放)
        corrected = out_dir / f"{object_id}_mesh_corrected.obj"
        with open(args.scaled_mesh) as src, open(corrected, "w") as dst:
            for line in src:
                if line.startswith("v "):
                    parts = line.split()
                    xyz = [float(x) * factor for x in parts[1:4]]
                    dst.write(f"v {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"
                              + (" " + " ".join(parts[4:]) if len(parts) > 4 else "") + "\n")
                else:
                    dst.write(line)
        result["corrected_mesh"] = str(corrected)
    if args.scale_metadata and args.scale_metadata.exists():
        result["scale_metadata"] = json.loads(args.scale_metadata.read_text())

    cap.release()
    (out_dir / f"{object_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[{object_id}] L_geo={L_geo:.3f}m anchor={L_hand:.3f}m "
          f"band=[{band[0]:.3f},{band[1]:.3f}] -> {result['verdict']}"
          f" L_final={result['L_final_m']}m")


if __name__ == "__main__":
    main()
