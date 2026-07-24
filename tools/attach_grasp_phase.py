#!/usr/bin/env python3
"""attach_grasp_phase.py — 把「抓取/接触阶段」逐帧标签贴到已有的重建产物上。

给一条 take(或直接给 npz),读阶段 provider(人工 grasp_annotation.json,
或将来同学的自动接触检测),把逐帧阶段写进:
  - world_fused.npz : hand_phase (2,Th) int8 [0]=左 1]=右(对齐 hand_trans),
                      obj_phase (Tv,) int8,phase_source,phase_meta
  - replay_world.npz: phase_left/right (Tv,) int8,phase_obj (Tv,),phase_source,phase_meta

不重跑重建 —— 事后补标注/换自动检测都用这个工具就地更新 npz。

用法:
  python tools/attach_grasp_phase.py <take目录 | world_fused.npz | replay_world.npz> \
      [--annotation grasp_annotation.json] [--provider auto|manual|none] \
      [--targets world,replay] [--dry-run]

  # 例:HOI4D,标注在 Data 视频目录,重建输出在 Output —— 显式给标注路径
  python tools/attach_grasp_phase.py \
      Output/ReconstructOutput/hoi4d/ZY20210800003/H3/C5/N11/S58/s01/T1 \
      --annotation Data/HOI4D/HOI4D_release/ZY20210800003/H3/C5/N11/S58/s01/T1/align_rgb/grasp_annotation.json

在任意有 numpy 的环境(如 hawor)跑即可。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# 让本脚本能 import ego_pipeline/phase
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "ego_pipeline"))
from phase import PhaseTracks, load_phase  # noqa: E402


def _resolve_targets(target: Path):
    """返回 {'world': path|None, 'replay': path|None, 'take_dir': dir}。"""
    if target.is_dir():
        take = target
        world = take / "world_fused.npz"
        replay = next(iter(sorted(take.glob("replay*world*.npz"))), take / "replay_world.npz")
        return {
            "world": world if world.is_file() else None,
            "replay": replay if replay.is_file() else None,
            "take_dir": take,
        }
    # 单个 npz
    name = target.name.lower()
    kind = "world" if "world_fused" in name else "replay"
    return {kind: target, ("replay" if kind == "world" else "world"): None,
            "take_dir": target.parent}


def _num_frames(paths) -> int:
    for p in (paths.get("world"), paths.get("replay")):
        if p and p.is_file():
            d = np.load(p, allow_pickle=True)
            if "num_frames" in d.files:
                return int(d["num_frames"])
            for k in ("frames", "joints_right", "obj_pose"):
                if k in d.files:
                    return int(np.asarray(d[k]).shape[0])
    raise RuntimeError("无法确定 num_frames(Tv)")


def _rewrite_npz(path: Path, updates: dict, *, compressed: bool, dry_run: bool):
    """就地把 updates 合并进 npz(保留原字段)。写临时文件再原子替换。"""
    d = np.load(path, allow_pickle=True)
    payload = {k: d[k] for k in d.files}
    added = {k: (v.shape if hasattr(v, "shape") else type(v).__name__) for k, v in updates.items()}
    payload.update(updates)
    if dry_run:
        print(f"  [dry-run] {path.name} 将写入: {added}")
        return
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=str(path.parent))
    os.close(fd)
    (np.savez_compressed if compressed else np.savez)(tmp, **payload)
    saved = tmp if tmp.endswith(".npz") else tmp + ".npz"  # savez 会补 .npz
    os.replace(saved, path)
    print(f"  ✔ {path.name} 写入: {list(updates)}")


def _phase_meta_str(tracks: PhaseTracks) -> str:
    return json.dumps({"source": tracks.source, "fps": tracks.fps, **tracks.meta},
                      ensure_ascii=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="take 目录 或 world_fused.npz / replay_world.npz")
    ap.add_argument("--annotation", default=None, help="显式 grasp_annotation.json 路径")
    ap.add_argument("--provider", default="auto",
                    help="auto(默认,auto->manual 回退) | manual | none")
    ap.add_argument("--targets", default="world,replay",
                    help="要写哪些产物,逗号分隔(默认 world,replay)")
    ap.add_argument("--dry-run", action="store_true", help="只打印,不写文件")
    args = ap.parse_args()

    paths = _resolve_targets(Path(args.target).resolve())
    want = {t.strip() for t in args.targets.split(",") if t.strip()}
    Tv = _num_frames(paths)

    tracks = load_phase(paths["take_dir"], num_frames=Tv, prefer=args.provider,
                        annotation_path=args.annotation)
    if tracks is None:
        print(f"[attach] 无可用阶段信息(provider={args.provider},take={paths['take_dir']});"
              "人工标注请先跑 tools/annotate_grasp_frames.py,或用 --annotation 指定。")
        return 1
    print(f"[attach] Tv={Tv}  {tracks.summary()}")
    meta = _phase_meta_str(tracks)

    if "replay" in want and paths.get("replay"):
        _rewrite_npz(paths["replay"], {
            "phase_left": tracks.left, "phase_right": tracks.right,
            "phase_obj": tracks.obj,
            "phase_source": np.array(tracks.source), "phase_meta": np.array(meta),
        }, compressed=True, dry_run=args.dry_run)

    if "world" in want and paths.get("world"):
        d = np.load(paths["world"], allow_pickle=True)
        Th = int(np.asarray(d["hand_trans"]).shape[1]) if "hand_trans" in d.files else Tv
        _rewrite_npz(paths["world"], {
            "hand_phase": tracks.to_hand_timeline(Th),   # (2,Th) 对齐 hand_trans
            "obj_phase": tracks.obj,                     # (Tv,) 对齐 object_ob_in_world
            "phase_source": np.array(tracks.source), "phase_meta": np.array(meta),
        }, compressed=False, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
