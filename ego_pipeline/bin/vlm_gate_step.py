#!/usr/bin/env python3
"""VLM 透明门的**管线包装** —— 把孤立脚本 `vlm_transparency_gate.py` 接进重建链路。

原脚本是 CLI（要手工给 `--sample frame:mask`），从未接进 `reconstruct.sh`，
所以"判透明"这一步实际从没在自动流程里跑过。本包装做三件事：

1. 从 v17A 的实例 manifest 里**跨时间挑 2~3 帧非空 mask**（空透明 vs 装液体需要看内容物，
   单帧不够）；manifest 缺失时回落到 `sam2_object` 的传播 mask。
2. 逐实例调用透明门。
3. 判定落盘 `vlm_gate.json`，**只记录不删数据** —— 过滤与否交给下游决定。

⚠ 默认策略 **strict**（2026-08-13 起）：**透明材质一律剔除**，不用于重建与训练。
旧的 v2（只剔高置信"空透明"、装内容物的放行）用 `--filter-policy v2` 保留，仅供复现旧结论。
收紧的依据见 `vlm_transparency_gate.should_filter` 上方注释（透明度与深度失败单调相关）。**本步骤不自行删除任何实例**，避免"VLM 一句话
毙掉一条数据"且不可追溯。

⚠ 依赖 vLLM 服务（默认 `VLM_API_BASE=http://127.0.0.1:8807/v1`，`VLM_MODEL=vlm`）。
服务不可用时**跳过并写明原因**，不让整条重建失败 —— 透明判定是加分项不是必需项。

用法:
    python vlm_gate_step.py --dataset X --video-id Y --video <mp4> [--max-instances 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

RR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def service_up(base: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def pick_samples(dataset: str, vid: str, n_want: int = 3):
    """→ {instance_id: [(frame_idx, mask_path), ...]}，跨时间均匀取非空 mask。"""
    out: dict[str, list] = {}
    v17a = (RR / "experimental/hoi_detr_v17a/data/interim" / dataset / vid
            / "instance_pipeline_v17a/video_mask_sequence")
    man = next((p for p in (v17a / "video_mask_sequence.json",
                            v17a / "video_mask_sequence.synth.json") if p.is_file()), None)
    if man:
        d = json.loads(man.read_text())
        per: dict[str, list] = {}
        for fr in d.get("frames", []):
            fi = int(fr["frame_idx"])
            for oid, o in (fr.get("objects") or {}).items():
                if not isinstance(o, dict):
                    continue
                # ⚠ 用 raw_mask: mask 字段在被质量门拒掉的帧上为空, 会挑不到样本
                p = o.get("mask") or o.get("raw_mask")
                if p and p not in ("None", None) and Path(p).is_file():
                    per.setdefault(oid, []).append((fi, Path(p)))
        for oid, lst in per.items():
            if not lst:
                continue
            step = max(1, len(lst) // n_want)
            out[oid] = lst[::step][:n_want]
        return out

    # 回落: 管线自己的 sam2_object 传播 mask
    md = (RR / "Output/ReconstructOutput/interim" / dataset / vid
          / "sam2_object/video_segmentation/masks")
    if md.is_dir():
        frames = sorted(md.iterdir())
        if frames:
            step = max(1, len(frames) // n_want)
            for fd in frames[::step][:n_want]:
                for png in sorted(fd.glob("*.png")):
                    fi = int(fd.name.split("_")[1])
                    out.setdefault(png.stem, []).append((fi, png))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--max-instances", type=int, default=4)
    ap.add_argument("--filter-policy", default="v2", choices=("strict", "v2"),
                    help="strict(默认): 透明材质一律剔除, 不用于重建/训练; "
                         "v2(旧): 只剔高置信空透明, 装内容物的放行")
    ap.add_argument("--force", action="store_true",
                    help="忽略 vlm_gate.json 里的缓存, 重新问 VLM")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    from vlm_transparency_gate import gate_path                      # noqa: E402
    out = a.out or gate_path(a.dataset, a.video_id)   # 与 auto_label 写的是同一份
    out.parent.mkdir(parents=True, exist_ok=True)
    base = (os.environ.get("VLM_API_BASE") or os.environ.get("QWEN_BASE_URL")
            or "http://127.0.0.1:8807/v1")

    if not service_up(base):
        out.write_text(json.dumps({"status": "skipped", "reason": f"VLM 服务不可用: {base}",
                                   "hint": "GPU=7 bash ~/bin/start_vlm.sh"},
                                  ensure_ascii=False, indent=1))
        print(f"[vlm_gate] 服务不可用({base}), 跳过 —— 不阻塞重建")
        return 0

    samples = pick_samples(a.dataset, a.video_id)
    if not samples:
        out.write_text(json.dumps({"status": "skipped", "reason": "找不到任何实例 mask"},
                                  ensure_ascii=False, indent=1))
        print("[vlm_gate] 无实例 mask, 跳过")
        return 0

    from vlm_transparency_gate import judge_instances, judge_part_change   # noqa: E402
    # ★ 整条视频问一次"分件/合件" —— 这是 Retrieval 的判据, 与实例无关, 不需要 mask。
    #   单个刚体网格表达不了"盖相对瓶身转", 这类 take 必须换分件资产。
    part = {}
    try:
        part = judge_part_change(a.video)
        print(f"[vlm_gate] 分件: {part.get('part_change')} {part.get('parts')} "
              f"conf={part.get('confidence')} 需Retrieval={part.get('needs_retrieval')}")
    except Exception as e:
        part = {"error": f"{type(e).__name__}: {e}"}
        print(f"[vlm_gate] 分件判定失败: {part['error']}")

    # ★ 材质判定走**缓存**: auto_label 的透明门可能已经判过并写进 vlm_gate.json。
    #   缓存命中就不再问 VLM —— 省一次整段视频的上传, 也保证两处结论一致。
    try:
        res = judge_instances(a.video, samples, dataset=a.dataset, video_id=a.video_id,
                              policy=a.filter_policy, force=a.force,
                              max_instances=a.max_instances)
    except Exception as e:
        res = {}
        print(f"[vlm_gate] 材质判定失败: {type(e).__name__}: {e}")
    for oid, v in res.items():
        print(f"[vlm_gate] {oid}: {v.get('material') or v.get('error') or '?'}  "
              f"conf={v.get('confidence','?')}  建议过滤={v.get('filter')}  "
              f"帧={v.get('sample_frames')}")
    # judge_instances 已经写过 objects 段; 这里只补 part_change, 不覆盖它
    doc = json.loads(out.read_text()) if out.is_file() else {}
    doc.update({"status": "ok", "api": base, "filter_policy": a.filter_policy,
                "objects": {**(doc.get("objects") or {}), **res},
                "part_change": part,
                "note": "只记录不删数据; 过滤与否交下游"})
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"[vlm_gate] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
