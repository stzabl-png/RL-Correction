#!/usr/bin/env python3
"""把我们管线的 interim 布局适配成 scale_develop 选帧器要的 `--run` 布局(全用软链)。

选帧器(`select_frame_v2.py` / `qwen_final_arbiter.py`)是在另一台机器上按它自己的
`runs/<id>/` 目录写的:
    runs/<id>/<video>.mp4
    runs/<id>/hoi_detr_probe/detections.json
    runs/<id>/instance/video_mask_sequence/video_mask_sequence.json
我们的管线产出在:
    experimental/hoi_detr_v17a/data/interim/<dataset>/<video_id>/hoi_detr_probe/detections.json
    experimental/hoi_detr_v17a/data/interim/<dataset>/<video_id>/instance_pipeline_v17a/
        video_mask_sequence/video_mask_sequence.json      ← 中间目录名不同
视频则在 dataset root(或 preflight 镜像)里。

只建软链、不拷贝:mask 序列可能上万个 png, 复制既慢又浪费; 而且软链让选帧器读到的
永远是管线的最新产物, 不会出现"适配目录里是旧快照"这种静默不一致。

⚠ 选帧器还会读 `frame_selection_baseline/report.json` 来画对比图。那是它评测基线用的,
不是选帧本身的输入 —— 缺失时本脚本会造一个空壳, 让选帧器能跑完(对比图里就没有基线那一列)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RR = Path(__file__).resolve().parents[1]
V17A_INTERIM = RR / "experimental" / "hoi_detr_v17a" / "data" / "interim"


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink() if dst.is_symlink() else None
        if dst.exists() and not dst.is_symlink():
            return                                  # 真实目录/文件, 不动
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def synth_video_manifest(inst_dir: Path) -> Path | None:
    """视频级 manifest 缺失时, 由各 ready 的 episode 合成一份。

    为什么需要:选帧器读的是视频级 manifest, 而它由"跨周期视觉身份链接"产出 —— 那一步
    经常失败(arctic 两条 take 都失败: laptop `failed_global_identity_linking`, 匹配分
    0.584 vs 次高 0.308 不够决定性)。失败时逐 episode 的 mask 完好, 只是没串起来。

    合成同时修掉一个更实际的缺陷:降级路径是"取**最后一个** ready episode 的最早 accepted 帧",
    实测 ketchup 因此被困在一个**只有 6 帧**的 episode(347-352)里, 而 episode_00(138帧)、
    episode_01(157帧) 全被跳过。整条重建的网格/尺度/位姿锚点都押在那 6 帧的首帧上。
    合并后选帧器能在全片的 accepted 帧里挑。

    ⚠ 假设:跨 episode 的同名 instance 是同一个物体。这本该由身份链接确认, 而它恰恰失败了。
    所以这里打印各 episode 的 mask 面积中位数供人眼核对 —— 面积量级差很多就说明合并错了。
    (当前只对单物体 take 使用, 与降级路径"只注册主实例"的既有假设一致。)
    """
    eps = sorted(inst_dir.glob("episode_*/sequence_attempt_*/mask_sequence.json"))
    frames, ids, per_ep = [], [], []
    for p in eps:
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("status") != "ready":
            continue
        fr = m.get("frames") or []
        frames.extend(fr)
        ids.extend(m.get("object_ids") or [])
        n_acc = sum(1 for f in fr for o in (f.get("objects") or {}).values()
                    if o.get("status") == "accepted")
        per_ep.append((p.parts[-3], len(fr), n_acc))
    if not frames:
        return None
    frames.sort(key=lambda f: int(f["frame_idx"]))
    out = inst_dir / "video_mask_sequence" / "video_mask_sequence.synth.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema_version": "persistent_video_mask_sequence_v1_synth",
         "object_ids": sorted(set(ids)),
         "provenance": {"synthesized_from": [e[0] for e in per_ep],
                        "note": "视频级 manifest 缺失(身份链接失败), 由 ready 的 episode 合并; "
                                "假设跨 episode 同名 instance 是同一物体"},
         "frames": frames}, ensure_ascii=False, indent=1))
    print(f"  合成视频级 manifest: {len(frames)} 帧, ids={sorted(set(ids))}")
    for name, n, acc in per_ep:
        print(f"    {name}: {n} 帧, accepted 条目 {acc}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True, help="源 mp4")
    ap.add_argument("--run-root", type=Path, default=RR / "Output" / "frame_select" / "runs")
    a = ap.parse_args()

    src = V17A_INTERIM / a.dataset / a.video_id
    det = src / "hoi_detr_probe" / "detections.json"
    inst = src / "instance_pipeline_v17a"
    man = inst / "video_mask_sequence" / "video_mask_sequence.json"
    if not det.is_file():
        print(f"X 缺 HOI-DETR detections: {det}", file=sys.stderr)
        return 2
    if not man.is_file():
        man = synth_video_manifest(inst)
        if man is None:
            print(f"X 既无视频级 manifest, 也没有任何 ready 的 episode: {inst}", file=sys.stderr)
            return 2
    if not a.video.is_file():
        print(f"X 视频不存在: {a.video}", file=sys.stderr)
        return 2

    run = (a.run_root / a.video_id).resolve()
    run.mkdir(parents=True, exist_ok=True)
    link(a.video.resolve(), run / a.video.name)
    link(det.parent.resolve(), run / "hoi_detr_probe")
    # 选帧器按 run/instance/video_mask_sequence/video_mask_sequence.json 读;
    # 合成文件叫 .synth.json, 这里在 run 下建真实目录 + 软链到具体文件, 保留"哪份是合成的"可见
    inst_link = run / "instance" / "video_mask_sequence"
    inst_link.mkdir(parents=True, exist_ok=True)
    link(man.resolve(), inst_link / "video_mask_sequence.json")

    base = run / "frame_selection_baseline" / "report.json"
    if not base.exists():
        ids = json.loads(man.read_text()).get("object_ids") or []
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text(json.dumps(
            {"objects": {o: {"chosen_frame": None,
                             "note": "占位: 未跑 eval_baseline_frame_selection.py; "
                                     "选帧器只用它画对比图, 不影响选帧结果"} for o in ids}},
            ensure_ascii=False, indent=1))
        print(f"  (造了基线占位, {len(ids)} 个物体)")

    print(f"run 目录就绪: {run}")
    for c in sorted(run.iterdir()):
        tgt = f" -> {c.resolve()}" if c.is_symlink() else ""
        print(f"  {c.name}{tgt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
