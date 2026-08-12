#!/usr/bin/env python3
"""Qwen 终审(第三段):对 v2.1 的 top-1 做分项验证,不合格则从时间 NMS 后的
top-K 里让 Qwen 比选。同时兼任虚假实例过滤(is_discrete_object)。

判定规则在本脚本里(不让 Qwen 下总结论):硬性项 = is_discrete_object / occlusion /
completeness / mask_quality 全 pass 才 accept;view/sharpness 只记录。
兜底链:API 失败重试 2 次 → 回退几何 top-1(source=geometric_fallback);
top-K 全拒 → 几何 top-1 + low_confidence 标记。prompt/响应原文全部落盘可审计。

前置过滤(track_filter.json,由 filter_tracks.py 产出):static_background /
insufficient_track 的 track 直接跳过,不消耗 Qwen 调用。
主体仲裁:多个 track 通过时标 interaction_target——hand_link_fraction 差距悬殊
(≥0.5 vs <0.25)直接判,否则用各自接触帧让 Qwen 指认被操作的物体。

用法(sam3 env): python qwen_final_arbiter.py --run runs/s01_ketchup_grab_01 [--k 6]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

V17A = Path(__file__).resolve().parent.parent / "experimental/hoi_detr_v17a"
sys.path.insert(0, str(V17A))
from experiments.hoi_detr.qwen_client import build_user_content, call_qwen, make_client  # noqa: E402

CROP_EXPAND = 1.5
CROP_LONG_SIDE = 1024
HARD_CRITERIA = ("occlusion", "completeness", "mask_quality")

SYSTEM = (
    "你是 3D 重建管线的选帧审核员。用户会给你视频帧(原图裁剪 + 物体mask叠加图),"
    "评估该帧是否适合作为单图 3D 重建的参考帧。严格按要求输出 JSON,不要输出多余文字。"
)

VERIFY_PROMPT = """图1是视频帧的裁剪,图2是同一裁剪上的物体 mask 叠加(红色半透明区域+轮廓线)。
请先描述 mask 圈住的东西,然后逐项评估。只输出如下 JSON:
{
  "object_description": "mask圈住的东西是什么(一句话)",
  "is_discrete_object": true/false,   // 独立物体或物体的一个部件(如翻盖/盖子/把手)都算 true;仅当是桌面/背景/贴纸,或把多个不相干物体混在一个mask里时为 false
  "occlusion":    {"pass": true/false, "reason": "物体可见表面是否几乎无手或他物遮挡"},
  "completeness": {"pass": true/false, "reason": "物体是否完整在画面内,未被画幅裁切"},
  "mask_quality": {"pass": true/false, "reason": "mask是否贴合物体:没漏掉物体部分,也没把手/桌面/别的物体包进来"},
  "view_informative": {"pass": true/false, "reason": "该视角是否体现物体三维形状(非完全正对的退化平面视角)"},
  "sharpness":    {"pass": true/false, "reason": "是否清晰无明显运动模糊"}
}"""

FALLBACK_PROMPT = """以下 {n} 张图是同一物体在不同帧的 mask 叠加裁剪,标号 {labels}(与图片顺序一一对应)。
请选出最适合做单图 3D 重建参考帧的一张:优先无遮挡、物体完整、mask 贴合、视角有三维信息、清晰。
只输出如下 JSON:
{{
  "choice": "标号字母",
  "reason": "选它的理由(一句话)",
  "rejected": {{"标号": "不选的主因(短语)", ...}}
}}"""

TARGET_PROMPT = """以下 {n} 张图分别是同一段"人手操作物体"视频中 {n} 个不同物体 track 的
mask 叠加(红色区域,标号 {labels},与图片顺序一一对应,均取各自与手接触最多的帧)。
请判断哪个标号是**手主要抓握/操作的目标物体**(而非桌面、支撑台、背景或大件家具)。
只输出如下 JSON:
{{"target": "标号字母", "reason": "判断依据(一句话)"}}"""


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in response: {text[:200]}")
    return json.loads(match.group(0))


def crop_pair(frame: np.ndarray, mb: np.ndarray, out_raw: Path, out_overlay: Path) -> None:
    ys, xs = np.nonzero(mb)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = (x1 - x0 + 1) * CROP_EXPAND / 2, (y1 - y0 + 1) * CROP_EXPAND / 2
    half = max(half_w, half_h, 80)
    X0, X1 = max(0, int(cx - half)), min(frame.shape[1], int(cx + half))
    Y0, Y1 = max(0, int(cy - half)), min(frame.shape[0], int(cy + half))
    raw = frame[Y0:Y1, X0:X1]
    over = raw.copy()
    sub = mb[Y0:Y1, X0:X1]
    over[sub] = (0.6 * over[sub] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (0, 0, 255), 3)
    for img, path in ((raw, out_raw), (over, out_overlay)):
        scale = CROP_LONG_SIDE / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        cv2.imwrite(str(path), img)


def load_mask_for(manifest: dict, object_id: str, frame_idx: int) -> np.ndarray:
    for frame in manifest["frames"]:
        if int(frame["frame_idx"]) == frame_idx:
            obj = (frame.get("objects") or {}).get(object_id)
            return cv2.imread(str(obj["mask"]), cv2.IMREAD_GRAYSCALE) > 0
    raise KeyError(f"{object_id} f{frame_idx}")


def qwen_call_logged(system: str, content, log: list, tag: str, retries: int = 2) -> dict | None:
    client = make_client()
    for attempt in range(retries + 1):
        try:
            resp = call_qwen(system, content, client=client)
            log.append({"tag": tag, "attempt": attempt, "response": resp.content})
            return parse_json(resp.content)
        except Exception as exc:  # API 或解析失败都走重试
            log.append({"tag": tag, "attempt": attempt, "error": repr(exc)})
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    run = args.run.resolve()

    report = json.loads((run / "frame_selection_v21/report_v2.json").read_text())
    manifest = json.loads((run / "instance/video_mask_sequence/video_mask_sequence.json").read_text())
    video = next(run.glob("*.mp4"))
    cap = cv2.VideoCapture(str(video))
    audit = run / "qwen_audit"
    audit.mkdir(exist_ok=True)

    def frame_at(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        assert ok, idx
        return fr

    tf_path = run / "track_filter.json"
    track_filter = json.loads(tf_path.read_text())["tracks"] if tf_path.exists() else {}

    final = {}
    for object_id, obj in report["objects"].items():
        tf = track_filter.get(object_id)
        if tf and tf.get("verdict", "keep") != "keep":
            final[object_id] = {"final_frame": None, "source": tf["verdict"],
                                "skip_reconstruction": True, "track_filter": tf}
            print(f"[{object_id}] 前置过滤 -> {tf['verdict']} (无 Qwen 调用)")
            continue
        cands = (obj.get("candidates_nms") or obj.get("ranked") or [])[: args.k]
        if not cands:
            final[object_id] = {"final_frame": None, "source": "no_candidates"}
            continue
        top1 = cands[0]
        log: list = []
        entry = {"geometric_top1": top1, "candidates": cands}

        # ── 阶段V:分项验证 top-1 ──
        raw_p = audit / f"{object_id}_f{top1:06d}_raw.jpg"
        over_p = audit / f"{object_id}_f{top1:06d}_overlay.jpg"
        crop_pair(frame_at(top1), load_mask_for(manifest, object_id, top1), raw_p, over_p)
        verify = qwen_call_logged(SYSTEM, build_user_content(VERIFY_PROMPT, [raw_p, over_p]), log, "verify")
        entry["verify"] = verify

        if verify is None:
            entry.update(final_frame=top1, source="geometric_fallback")
        elif not verify.get("is_discrete_object", True):
            entry.update(final_frame=None, source="spurious_track", skip_reconstruction=True)
        elif all((verify.get(c) or {}).get("pass") for c in HARD_CRITERIA):
            entry.update(final_frame=top1, source="qwen_accept")
        else:
            # ── 阶段F:top-K 比选 ──
            labels = [chr(ord("A") + n) for n in range(len(cands))]
            paths = []
            for label, idx in zip(labels, cands):
                p = audit / f"{object_id}_cand{label}_f{idx:06d}.jpg"
                crop_pair(frame_at(idx), load_mask_for(manifest, object_id, idx),
                          audit / f"{object_id}_cand{label}_raw_unused.jpg", p)
                paths.append(p)
            prompt = FALLBACK_PROMPT.format(
                n=len(cands), labels=", ".join(f"{l}=帧{i}" for l, i in zip(labels, cands)))
            pick = qwen_call_logged(SYSTEM, build_user_content(prompt, paths), log, "fallback")
            if pick and pick.get("choice") in labels:
                entry.update(
                    final_frame=cands[labels.index(pick["choice"])],
                    source="qwen_override", fallback=pick,
                )
            else:
                entry.update(final_frame=top1, source="geometric_fallback",
                             low_confidence=True, fallback=pick)
        if tf:
            entry["track_filter"] = tf
        entry["qwen_log"] = log
        final[object_id] = entry
        print(f"[{object_id}] top1=f{top1} -> {entry['source']}"
              f" final=f{entry.get('final_frame')}"
              + (f" ({verify.get('object_description')})" if verify else ""))

    # ── 主体仲裁:多个 track 通过时标 interaction_target ──
    passers = [oid for oid, e in final.items() if e.get("final_frame") is not None]
    if len(passers) == 1:
        final[passers[0]]["interaction_target"] = True
    elif len(passers) > 1:
        fracs = {oid: (track_filter.get(oid) or {}).get("hand_link_fraction", 0.0)
                 for oid in passers}
        ranked = sorted(passers, key=lambda o: -fracs[o])
        if fracs[ranked[0]] >= 0.5 and all(fracs[o] < 0.25 for o in ranked[1:]):
            winner, how = ranked[0], f"hand_link_margin({fracs[ranked[0]]:.0%})"
        else:
            labels = [chr(ord("A") + n) for n in range(len(ranked))]
            paths = []
            for label, oid in zip(labels, ranked):
                detail = report["objects"][oid].get("occ_detail") or {}
                cf = (int(max(detail, key=lambda k: detail[k].get("contact", 0)
                              + detail[k].get("occ_hull", 0)))
                      if detail else int(final[oid]["final_frame"]))
                p = audit / f"target_{label}_{oid}_f{cf:06d}.jpg"
                crop_pair(frame_at(cf), load_mask_for(manifest, oid, cf),
                          audit / f"target_{label}_{oid}_raw_unused.jpg", p)
                paths.append(p)
            log: list = []
            prompt = TARGET_PROMPT.format(
                n=len(ranked), labels=", ".join(f"{l}={o}" for l, o in zip(labels, ranked)))
            pick = qwen_call_logged(SYSTEM, build_user_content(prompt, paths), log, "target")
            if pick and pick.get("target") in labels:
                winner, how = ranked[labels.index(pick["target"])], "qwen_target"
            else:
                winner, how = ranked[0], "hand_link_fallback"
            final[winner].setdefault("qwen_log", []).extend(log)
        for oid in passers:
            final[oid]["interaction_target"] = oid == winner
        final[winner]["target_source"] = how
        print(f"主体仲裁: {winner} ({how}), 其余 {[o for o in passers if o != winner]} 降为次要")

    cap.release()
    (run / "final_selection.json").write_text(json.dumps(final, ensure_ascii=False, indent=2))
    print("final ->", run / "final_selection.json")


if __name__ == "__main__":
    main()
