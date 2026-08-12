#!/usr/bin/env python3
"""分段版 sam3_hands:SAM3 视频推理会把全视频张量整体搬上 GPU(~12.6MB/帧),
长视频(数千帧)直接 OOM。本脚本把视频切成 <=chunk 帧的段,逐段跑管线原版
segment_hand_on_video(输出约定 100% 一致),再把局部帧号的 mask 合并回全局布局,
最后补写 sam3_hands_complete.json + legacy hand_masks_complete.json。

每段的 prompt 帧从 HOI-DETR detections 里选:双手总分最高的帧;整段无手检出
(<min-score)则全段写零 mask(手不在画面,语义正确)。

用法(sam3 env, CUDA_VISIBLE_DEVICES 选卡):
  python sam3_hands_chunked.py --video sweep.mp4 --detections detections.json \
      --step-dir <interim>/sam3_hands --video-id 2_sweep [--chunk 1100] [--sam3-version sam3]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

RECON_PIPELINE = Path(__file__).resolve().parent.parent / "ego_pipeline" / "Reconstruction" / "recon_pipeline"
sys.path.insert(0, str(RECON_PIPELINE))
sys.path.insert(0, str(RECON_PIPELINE / "_legacy" / "sam3"))
import common as sam3_common  # noqa: E402  _legacy/sam3/common.py 原代码


def hand_scores_by_frame(detections_json: Path) -> dict[int, list[float]]:
    data = json.loads(detections_json.read_text())
    out: dict[int, list[float]] = {}
    for fr in data.get("frames") or []:
        s = sorted((float(d["score"]) for d in fr.get("detections") or []
                    if d.get("class_name") == "hand"), reverse=True)
        out[int(fr["frame_idx"])] = s[:2]
    return out


def pick_prompt_frame(scores: dict[int, list[float]], lo: int, hi: int, min_score: float) -> int | None:
    best, best_val = None, 0.0
    for i in range(lo, hi):
        s = [x for x in scores.get(i, []) if x >= min_score]
        if not s:
            continue
        val = sum(s) + (1.0 if len(s) >= 2 else 0.0)  # 强烈偏好双手同框
        if val > best_val:
            best, best_val = i, val
    return best


def write_chunk_video(cap: cv2.VideoCapture, lo: int, hi: int, fps: float, path: Path) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(lo, hi):
        ok, fr = cap.read()
        if not ok:
            break
        vw.write(fr)
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--detections", required=True, type=Path)
    ap.add_argument("--step-dir", required=True, type=Path)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--chunk", type=int, default=1100)
    ap.add_argument("--sam3-version", default="sam3")
    ap.add_argument("--min-hand-score", type=float, default=0.5)
    ap.add_argument("--prob-thresh", type=float, default=0.5)
    args = ap.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    assert cap.isOpened(), args.video
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scores = hand_scores_by_frame(args.detections)

    seg_dir = args.step_dir / args.video_id / "video_segmentation"
    masks_dir = seg_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    predictor = sam3_common.build_predictor(version=args.sam3_version)
    hands = {"left_hand_0": "left hand", "right_hand_0": "right hand"}
    agg = {oid: {"obj_id": oid, "text_prompt": prompt, "prompt_used": prompt,
                 "prompts_tried": [prompt], "num_frames": n, "detected": False,
                 "detected_frames": 0, "output_prob_thresh": args.prob_thresh,
                 "prompt_score": None, "tracked_video": None}
           for oid, prompt in hands.items()}

    chunks = [(s, min(s + args.chunk, n)) for s in range(0, n, args.chunk)]
    tmp_root = Path(tempfile.mkdtemp(prefix="sam3_chunks_"))
    try:
        for ci, (lo, hi) in enumerate(chunks):
            prompt_frame = pick_prompt_frame(scores, lo, hi, args.min_hand_score)
            print(f"[chunk {ci}] frames [{lo},{hi}) prompt_frame={prompt_frame}", flush=True)
            if prompt_frame is None:
                for g in range(lo, hi):
                    d = masks_dir / f"frame_{g:06d}_masks"
                    d.mkdir(exist_ok=True)
                    for oid in hands:
                        cv2.imwrite(str(d / f"{oid}.png"), np.zeros((h, w), np.uint8))
                continue
            # 断点续跑:该段该手的首尾 mask 都已在全局目录 → 上次已完成,跳过
            def chunk_done(oid: str) -> bool:
                return all((masks_dir / f"frame_{g:06d}_masks" / f"{oid}.png").is_file()
                           for g in (lo, hi - 1))

            todo = {oid: prompt for oid, prompt in hands.items() if not chunk_done(oid)}
            if not todo:
                print(f"[chunk {ci}] 已完成,跳过", flush=True)
                continue
            chunk_mp4 = tmp_root / f"chunk_{ci:02d}.mp4"
            if not chunk_mp4.exists():
                write_chunk_video(cap, lo, hi, fps, chunk_mp4)
            for oid, prompt in todo.items():
                out_dir = tmp_root / f"chunk_{ci:02d}_{oid}"
                r = sam3_common.segment_hand_on_video(
                    predictor,
                    video_path=chunk_mp4,
                    text_prompt=prompt,
                    obj_id=oid,
                    output_dir=out_dir,
                    frame_idx=prompt_frame - lo,
                    save_overlays=False,
                    save_videos=False,
                    overlay_fps=24,
                    output_prob_thresh=args.prob_thresh,
                )
                det = int(r.get("detected_frames") or 0)
                agg[oid]["detected_frames"] += det
                agg[oid]["detected"] = agg[oid]["detected"] or bool(r.get("detected"))
                ps = r.get("prompt_score")
                if ps is not None:
                    prev = agg[oid]["prompt_score"]
                    agg[oid]["prompt_score"] = max(prev, ps) if prev is not None else ps
                for local in range(hi - lo):
                    src = out_dir / "masks" / f"frame_{local:06d}_masks" / f"{oid}.png"
                    dst_dir = masks_dir / f"frame_{lo + local:06d}_masks"
                    dst_dir.mkdir(exist_ok=True)
                    if src.is_file():
                        shutil.move(str(src), str(dst_dir / f"{oid}.png"))
                    else:
                        cv2.imwrite(str(dst_dir / f"{oid}.png"), np.zeros((h, w), np.uint8))
                shutil.rmtree(out_dir, ignore_errors=True)
                print(f"[chunk {ci}] {oid}: detected {det}/{hi-lo}", flush=True)
            chunk_mp4.unlink(missing_ok=True)
    finally:
        try:
            predictor.shutdown()
        except Exception:
            pass
        shutil.rmtree(tmp_root, ignore_errors=True)

    # 统一从盘上重算 detected_frames(续跑时内存统计不完整)
    for oid in hands:
        cnt = 0
        for g in range(n):
            p = masks_dir / f"frame_{g:06d}_masks" / f"{oid}.png"
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
            if m is not None and m.any():
                cnt += 1
        agg[oid]["detected_frames"] = cnt
        agg[oid]["detected"] = cnt > 0

    assert any(v["detected"] for v in agg.values()), "两只手在全视频都没检出,不写完成标记"

    legacy = {"status": "complete", "sequence": args.video_id, "hands": agg,
              "backend": "sam3_hands_chunked", "chunk_size": args.chunk,
              "sam3_version": args.sam3_version}
    (args.step_dir / args.video_id / "hand_masks_complete.json").write_text(json.dumps(legacy, indent=2))

    completion = {"status": "complete", "step": "sam3_hands", "dataset": "sweep_dustpan",
                  "video_id": args.video_id,
                  "masks_dir": str(masks_dir), "hands": agg,
                  "sam3_version": args.sam3_version, "sam3_requested_version": args.sam3_version,
                  "backend": "sam3_hands_chunked", "chunk_size": args.chunk}
    (args.step_dir / "sam3_hands_complete.json").write_text(json.dumps(completion, indent=2))
    for oid, v in agg.items():
        print(f"[done] {oid}: detected_frames={v['detected_frames']}/{n} best_prompt_score={v['prompt_score']}")


if __name__ == "__main__":
    main()
