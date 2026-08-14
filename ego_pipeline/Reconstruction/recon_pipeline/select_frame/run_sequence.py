#!/usr/bin/env python3
"""选帧(第 4 步) —— 决定 **SAM3D 重建帧 / sam3d_scale 尺度参考帧**。

契约(见 _common/frame_plan.py): 往 `frame_plan.json` 的 `objects.<oid>.sam3d_frame`
写一个帧号。消费方 sam3d(:558)/sam3d_scale(:333) 已接线, null 或 mask 空则回退 prompt 帧。
★ 不碰 `fp_register_frame`(固定=交互开始帧+10, auto_label 写入)。

实现 = agent/v17a-scale-develop 验证过的三层选帧器(核心在 selector_core.py):
  0. track 过滤(手物连接/位移/帧数)  1. 几何粗筛  2. SAM2 像素级遮挡精排
  3. Qwen 终审(VLM 不可达自动退回几何 top-1, 不会让管线崩)
验证: ketchup f591 / 扫把 f295 / microwave f618(凹槽 track 被判 spurious)。

输入:
  * sam2_object step 目录: label_prompt.json / frame_plan.json / video_segmentation/masks
  * v17A 产物(video_mask_sequence.json + hoi_detr_probe/detections.json):
    默认在 experimental/hoi_detr_v17a/data/interim/<ds>/<vid>/ 下自动探测
    (instance_pipeline_v17a/ 或 instance/ 两种布局), --v17a-dir 可显式指定。
    ⚠ 多物体必须是**视频级** video_mask_sequence.json(CLAUDE.md 第 6 条)。
    缺失时保持 sam3d_frame=null(下游回退 prompt 帧), 本步骤照常完成。

v17A track id(object_0001) -> 管线 object id(object_0) 映射:
  1) sam2_object/v17a_import.json 的 source_object_id(adapter 布局)
  2) label_prompt.json 的 provenance(多物体自动标注)
  3) 兜底: prompt 帧上 v17A mask 与 sam2_object mask 的 IoU 匹配(>=0.3)

产物(select_frame step 目录): report.json(含 ranked 干净帧列表, 供 sam3d_scale
的多帧跨度估计复用) + final_frame_<oid>_f*.jpg + qwen_audit/。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent.parent
STEP_DIR_SELF = Path(__file__).resolve().parent
for p in (RECON_ROOT, STEP_DIR_SELF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _common.frame_plan import load_frame_plan, write_frame_plan  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402

STEP = "select_frame"
REPO_ROOT = RECON_ROOT.parent.parent.parent  # ego_pipeline/Reconstruction/recon_pipeline -> 仓库根
V17A_ROOT = REPO_ROOT / "experimental" / "hoi_detr_v17a"


def _find_v17a_inputs(v17a_dir: Path | None, dataset: str, video_id: str) -> tuple[Path | None, Path | None]:
    roots = [v17a_dir] if v17a_dir else [
        V17A_ROOT / "data" / "interim" / dataset / video_id,
    ]
    for root in roots:
        if root is None or not root.is_dir():
            continue
        manifest = None
        for rel in ("instance_pipeline_v17a/video_mask_sequence/video_mask_sequence.json",
                    "instance/video_mask_sequence/video_mask_sequence.json",
                    "video_mask_sequence/video_mask_sequence.json"):
            if (root / rel).is_file():
                manifest = root / rel
                break
        detections = root / "hoi_detr_probe" / "detections.json"
        if manifest and detections.is_file():
            return manifest, detections
    return None, None


def _sam2_mask(obj_dir: Path, frame_idx: int, object_id: str):
    import cv2
    p = obj_dir / "video_segmentation" / "masks" / f"frame_{frame_idx:06d}_masks" / f"{object_id}.png"
    if not p.is_file():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def _map_tracks(obj_dir: Path, prompt_objects: list, manifest: dict) -> dict[str, str]:
    """返回 {pipeline_oid: v17a_oid}。"""
    import cv2
    import numpy as np

    mapping: dict[str, str] = {}
    # 1) adapter 布局
    imp = obj_dir / "v17a_import.json"
    if imp.is_file():
        try:
            data = json.loads(imp.read_text())
            src, dst = data.get("source_object_id"), data.get("object_id")
            if src and dst:
                mapping[dst] = src
        except (OSError, json.JSONDecodeError):
            pass
    # 2) label_prompt provenance(多物体自动标注)
    try:
        lp = json.loads((obj_dir / "label_prompt.json").read_text())
        for o in (lp.get("provenance") or {}).get("objects", []):
            pid, vid_ = o.get("object_id"), o.get("v17a_object_id") or o.get("source_object_id")
            if pid and vid_:
                mapping.setdefault(pid, vid_)
    except (OSError, json.JSONDecodeError):
        pass
    # 3) IoU 兜底
    v17a_ids = list(manifest.get("object_ids") or [])
    by_frame = {int(f["frame_idx"]): f for f in manifest.get("frames") or []}
    for obj in prompt_objects:
        pid = obj.object_id
        if pid in mapping:
            continue
        pm = _sam2_mask(obj_dir, int(obj.frame_idx), pid)
        if pm is None:
            if len(v17a_ids) == 1:
                mapping[pid] = v17a_ids[0]
            continue
        best, best_iou = None, 0.0
        fr = by_frame.get(int(obj.frame_idx))
        for voi in v17a_ids:
            rec = ((fr or {}).get("objects") or {}).get(voi)
            if not rec or not rec.get("mask"):
                continue
            vm = cv2.imread(str(rec["mask"]), cv2.IMREAD_GRAYSCALE)
            if vm is None:
                continue
            vm = vm > 0
            inter = float(np.logical_and(pm, vm).sum())
            union = float(np.logical_or(pm, vm).sum())
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = voi, iou
        if best is not None and best_iou >= 0.3:
            mapping[pid] = best
        elif len(v17a_ids) == 1:
            mapping[pid] = v17a_ids[0]
    return mapping


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--v17a-dir", type=Path, default=None,
                    help="v17A 产物目录(含 hoi_detr_probe/ 与 video_mask_sequence), 默认自动探测")
    ap.add_argument("--no-vlm", action="store_true", help="跳过 Qwen 终审, 用几何 top-1")
    ap.add_argument("--top", type=int, default=6)
    args, _extra = ap.parse_known_args(argv)

    step_dir = interim_step_dir(args.dataset, args.video_id, STEP)
    if is_step_complete(step_dir, STEP) and not args.force:
        return 0
    step_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import cv2

    import selector_core as sc
    from sam2_object.sam2_object_common import (
        DEFAULT_SAM2_CHECKPOINT, DEFAULT_SAM2_MODEL_CFG, load_label_prompt)

    obj_dir = interim_step_dir(args.dataset, args.video_id, "sam2_object")
    prompt = load_label_prompt(obj_dir)
    manifest_path, detections_path = _find_v17a_inputs(args.v17a_dir, args.dataset, args.video_id)
    if manifest_path is None:
        print(f"[select_frame] {args.video_id}: 找不到 v17A manifest/detections "
              f"(--v17a-dir 可指定) —— sam3d_frame 保持 null, 下游回退 prompt 帧。", flush=True)
        write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                              extra={"status_detail": "v17a_inputs_missing"})
        return 0

    manifest = json.loads(manifest_path.read_text())
    detections = json.loads(detections_path.read_text())
    hands = sc.hand_boxes_by_frame(detections)
    cap = cv2.VideoCapture(str(args.video))

    # ── 第 0 层: track 过滤 ──
    track_stats = sc.filter_tracks(manifest, detections)
    for oid, st in track_stats.items():
        print(f"[select_frame] {oid}: frames={st['n_frames']} hand_link="
              f"{st['hand_link_fraction']:.0%} disp={st['max_centroid_disp_diagonals']:.3f}"
              f" -> {st['verdict']}", flush=True)

    # ── 第 1+2 层: 两段式选帧(只对 keep 的 track) ──
    reports: dict[str, dict] = {}
    for oid in manifest.get("object_ids") or []:
        if (track_stats.get(oid) or {}).get("verdict", "keep") != "keep":
            continue
        rep = sc.select_frames_for_track(
            manifest, oid, cap, hands,
            cache_dir=step_dir / "hand_masks",
            sam2_cfg=DEFAULT_SAM2_MODEL_CFG, sam2_ckpt=DEFAULT_SAM2_CHECKPOINT,
            top_k=args.top)
        if rep is not None:
            reports[oid] = rep
            print(f"[select_frame] {oid}: accepted={rep['n_accepted']} stage1={rep['n_stage1']}"
                  f" geometric_top1=f{rep['chosen_frame']} ({rep['hand_mode']})", flush=True)

    # ── 第 3 层: Qwen 终审 + 主体仲裁 ──
    qwen_call = None
    if not args.no_vlm:
        qwen_call, why = sc.make_qwen_caller(V17A_ROOT)
        if qwen_call is None:
            print(f"[select_frame] VLM 不可用({why}) -> geometric_fallback", flush=True)
    final = sc.arbitrate(reports, track_stats, manifest, cap, step_dir / "qwen_audit", qwen_call)
    for oid, tf in track_stats.items():
        if tf.get("verdict", "keep") != "keep" and oid not in final:
            final[oid] = {"final_frame": None, "source": tf["verdict"],
                          "skip_reconstruction": True, "track_filter": tf}

    # ── 映射 v17A oid -> 管线 oid, 写 frame_plan(保留 fp_register_frame 等既有字段) ──
    mapping = _map_tracks(obj_dir, prompt.objects, manifest)  # {pipeline: v17a}
    plan = load_frame_plan(obj_dir)
    plan_objects = dict(plan.get("objects") or {})
    written = {}
    for obj in prompt.objects:
        pid = obj.object_id
        voi = mapping.get(pid)
        entry = dict(plan_objects.get(pid) or {})
        chosen, why = None, "no_v17a_track_match"
        if voi and voi in final:
            e = final[voi]
            if e.get("final_frame") is not None:
                cand = int(e["final_frame"])
                m = _sam2_mask(obj_dir, cand, pid)
                if m is not None and m.any():
                    chosen, why = cand, f"select_frame_v21+{e['source']}"
                else:
                    why = f"chosen_f{cand}_mask_empty_in_sam2_object"
            else:
                why = e.get("source", "no_final_frame")
        entry["sam3d_frame"] = chosen
        entry["sam3d_source"] = why + (f" (v17a={voi})" if voi else "")
        plan_objects[pid] = entry
        written[pid] = {"sam3d_frame": chosen, "why": why, "v17a_track": voi}
        print(f"[select_frame] frame_plan {pid}: sam3d_frame={chosen} ({why})", flush=True)
    write_frame_plan(obj_dir, plan_objects)

    # ── 产物: report + 最终帧可视化 ──
    report_out = {
        "video": str(args.video),
        "v17a_manifest": str(manifest_path),
        "track_filter": track_stats,
        "objects": {oid: {**rep, "final": final.get(oid, {})} for oid, rep in reports.items()},
        "pipeline_mapping": mapping,
        "frame_plan_written": written,
    }
    (step_dir / "report.json").write_text(
        json.dumps(report_out, ensure_ascii=False, indent=1, default=str))
    import numpy as np
    for pid, w in written.items():
        if w["sam3d_frame"] is None:
            continue
        ff = w["sam3d_frame"]
        img = sc.read_frame(cap, ff).copy()
        m = _sam2_mask(obj_dir, ff, pid)
        if m is not None:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 4)
        cv2.putText(img, f"SAM3D_FRAME {pid} f{ff}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
        cv2.imwrite(str(step_dir / f"final_frame_{pid}_f{ff:06d}.jpg"), img)
    cap.release()

    write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra={"status_detail": "ok", "frame_plan": written,
                                 "vlm": "on" if qwen_call else "off"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
