#!/usr/bin/env python3
"""资产检索 —— VLM 判 `needs_retrieval` 时, 用分件 CAD 顶替 SAM3D 重建。

============================ 为什么 ============================

重建产出的是**单个刚体网格**。若被操作物体在视频里会一分为多(瓶子→瓶身+瓶盖)或多合一,
单刚体在原理上表达不了这个动作 —— 拧盖的本质就是**盖相对瓶身转**。

实测(clip 0, screw_unscrew_bottle_cap):

| 物体表示 | conf_pos | conf_rot | 旋转 |
|---|---|---|---|
| 单件 SAM3D 网格(24.7cm) | 50 | 23 | **弃用** |
| CAD 瓶身 + CAD 瓶盖(19.7 + 1.7cm) | **82** | **34** | **可用** |

VLM 对本任务 29/29 条都判 `needs_retrieval=true`(separates 14 / combines 12 / both 3,
全部 confidence=high) —— 也就是**没有一条**能用单件重建正确表达。

============================ 顺带省掉两步 ============================

`fp_pose` 与 `fuse` 只从 `sam3d_scale/objects/<oid>/object_mesh_scaled_final.obj` 取网格
(见 `fp_common.scaled_mesh_path` 与 `fuse/run_sequence.py:563`), 该目录的其它产物下游都不读。
所以把 CAD 放进这个位置, 就能**同时跳过 `sam3d` 和 `sam3d_scale`**。
CAD 本身是米制的, 也不需要尺度估计。

============================ 部件怎么分配给实例 ============================

★ **按 mask 面积排序配对, 不按 object_id 顺序。**
`v17a_multi_object_prompt.py` 是按"accepted 帧里的峰值面积"给实例排序的, 而实测
29 条里有 6 条把**瓶盖**排成了 object_0(瓶身大部分帧被质量门拒掉, 峰值面积反而低)。
所以 object_0 不保证是大件。这里改用**整段视频的 mask 面积中位数**排序 —— 它对
个别帧的质量门抖动不敏感。

判据可靠性: 记录最大件/次大件的面积比。比值太小(<2)说明两件在图像上差不多大,
排序不可靠, 写进 `retrieval.json` 让下游知道。

用法:
    python retrieve_assets.py --dataset egodex_auto --video-id <task>__<n> --task <task>
    # 命中返回 0 并落盘 retrieval.json; 未命中返回 3(调用方应回退到重建)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

RR = Path(__file__).resolve().parents[2]
REGISTRY = RR / "ego_pipeline/Retargeting/assets/retrieval/registry.json"


def interim_dir(dataset: str, video_id: str) -> Path:
    root = Path(os.environ.get("RECON_INTERIM_ROOT",
                               RR / "Output/ReconstructOutput/interim"))
    return root / dataset / video_id


def instance_mask_areas(itm: Path) -> dict[str, float]:
    """→ {object_id: 面积中位数(px)}，取自管线自己的 sam2_object 传播 mask。"""
    import cv2
    md = itm / "sam2_object/video_segmentation/masks"
    per: dict[str, list[int]] = {}
    if not md.is_dir():
        return {}
    for fd in sorted(md.iterdir()):
        for png in sorted(fd.glob("*.png")):
            m = cv2.imread(str(png), 0)
            if m is not None:
                per.setdefault(png.stem, []).append(int((m > 0).sum()))
    return {k: float(np.median(v)) for k, v in per.items() if v}


def identify_parts(itm: Path, video: Path | None, oids: list[str],
                   part_names: list[str]) -> dict[str, str] | None:
    """→ {object_id: part_name}；VLM 不可用或全部指认失败时返回 None(调用方应拒绝装配)。"""
    if video is None or not Path(video).is_file():
        print(f"[retrieval] 没有视频({video}), 无法看图指认")
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from vlm_transparency_gate import judge_part_identity      # noqa: E402
        from vlm_gate_step import pick_samples                     # noqa: E402
    except Exception as e:
        print(f"[retrieval] 载入 VLM 指认失败: {e}")
        return None
    samples = pick_samples(itm.parent.name, itm.name)
    out = {}
    for oid in oids:
        sm = samples.get(oid) or _masks_for(itm, oid)
        if not sm:
            continue
        try:
            v = judge_part_identity(Path(video), sm, part_names)
        except Exception as e:
            print(f"[retrieval] {oid} 指认失败: {type(e).__name__}: {e}")
            return None
        print(f"[retrieval] VLM 指认 {oid} -> {v.get('part')} "
              f"(conf={v.get('confidence')}) {str(v.get('evidence'))[:60]}")
        if v.get("part") in part_names:
            out[oid] = v["part"]
    return out or None


def _masks_for(itm: Path, oid: str, n_want: int = 3) -> list:
    """回退取样: 直接从 sam2_object 的传播 mask 里跨时间取 3 帧。"""
    md = itm / "sam2_object/video_segmentation/masks"
    got = []
    if md.is_dir():
        fds = sorted(md.iterdir())
        for fd in fds[::max(1, len(fds) // n_want)][:n_want]:
            p = fd / f"{oid}.png"
            if p.is_file():
                got.append((int(fd.name.split("_")[1]), p))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--task", default=None,
                    help="任务名(默认从 video_id 的 '<task>__<n>' 切出)")
    ap.add_argument("--video", type=Path, default=None,
                    help="原视频; 排序配对不可靠时用它做 VLM 看图指认")
    ap.add_argument("--force", action="store_true",
                    help="即使 VLM 没判 needs_retrieval 也强制使用资产")
    a = ap.parse_args()

    task = a.task or a.video_id.rsplit("__", 1)[0]
    itm = interim_dir(a.dataset, a.video_id)
    out = itm / "retrieval.json"

    def bail(reason: str, code: int = 3) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"status": "skipped", "reason": reason, "task": task},
                                  ensure_ascii=False, indent=1))
        print(f"[retrieval] 不使用资产: {reason}")
        return code

    # ---- 1. VLM 说要不要 ----
    gate = itm / "vlm_gate.json"
    need, pc = None, {}
    if gate.is_file():
        g = json.loads(gate.read_text())
        pc = g.get("part_change") or {}
        need = pc.get("needs_retrieval")
    if not a.force:
        if need is None:
            return bail("没有 vlm_gate.json 的分件判定(先跑 vlm_gate_step)")
        if not need:
            return bail(f"VLM 判 part_change={pc.get('part_change')}, 不需要分件资产")

    # ---- 2. 库里有没有 ----
    if not REGISTRY.is_file():
        return bail(f"缺资产表 {REGISTRY}")
    reg = json.loads(REGISTRY.read_text())
    entry = (reg.get("assets") or {}).get(task)
    if not entry:
        return bail(f"资产库里没有任务 '{task}' 的条目 —— 回退到 SAM3D 重建")
    parts = sorted(entry["parts"], key=lambda p: -max(p["extent_cm"]))
    missing = [p["name"] for p in parts if not (RR / p["mesh"]).is_file()]
    if missing:
        return bail(f"资产文件缺失: {missing}")

    # ---- 3. 实例按 mask 面积排序, 与部件按尺寸排序配对 ----
    areas = instance_mask_areas(itm)
    if not areas:
        return bail("没有 sam2_object 的 mask, 无法把部件分配给实例")
    order = sorted(areas, key=lambda k: -areas[k])
    if len(order) < len(parts):
        print(f"[retrieval] ⚠ 实例只有 {len(order)} 个但资产有 {len(parts)} 件 —— "
              f"只装配前 {len(order)} 件。缺的那件很可能是 v17A 没发现(小/深色/常被握住)")
    n = min(len(order), len(parts))
    ratio = (areas[order[0]] / areas[order[1]]) if len(order) > 1 else float("nan")

    # ★ 排序配对只在"实例数 == 部件数 且 面积比够大"时才可靠。否则必须看图指认 ——
    #   实测 clip 2 只有一个实例且它是**瓶盖**(4696px / 3.8cm), 按排序会被无条件装成
    #   bottle_body。装错部件不会报错, 只会让下游拿着错的几何去做接触和 RL。
    by_vlm = None
    if len(order) != len(parts) or (len(order) > 1 and ratio < 2.0):
        why = ("实例数 %d != 部件数 %d" % (len(order), len(parts))
               if len(order) != len(parts) else "面积比 %.2f < 2" % ratio)
        print(f"[retrieval] 排序配对不可靠({why}) -> 改用 VLM 看图指认部件")
        by_vlm = identify_parts(itm, a.video, order, [p["name"] for p in parts])
        if by_vlm is None:
            return bail(f"排序不可靠({why})且 VLM 指认不可用 —— 拒绝乱装, 回退到重建")

    dst_root = itm / "sam3d_scale" / "objects"
    pmap = {p["name"]: p for p in parts}
    assign = []
    pairs = ([(oid, pmap[nm]) for oid, nm in by_vlm.items() if nm in pmap]
             if by_vlm else [(order[k], parts[k]) for k in range(n)])
    if not pairs:
        return bail("VLM 没能把任何实例指认成已知部件 —— 回退到重建")
    for oid, part in pairs:
        d = dst_root / oid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RR / part["mesh"], d / "object_mesh_scaled_final.obj")
        assign.append({"object_id": oid, "part": part["name"],
                       "mask_area_median_px": areas[oid],
                       "extent_cm": part["extent_cm"],
                       "mesh": part["mesh"]})
        print(f"[retrieval] {oid} (mask 中位 {areas[oid]:.0f}px) <- {part['name']} "
              f"{part['extent_cm']}cm")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "ok", "task": task, "asset_source": entry.get("source"),
        "matched_by": entry.get("matched_by", "task_name"),
        "vlm_part_change": pc.get("part_change"), "vlm_parts": pc.get("parts"),
        "assignment": assign,
        "assign_method": ("VLM 看图指认" if by_vlm else "mask 面积中位数降序 <-> 部件最大边降序"),
        "area_ratio_top2": None if np.isnan(ratio) else round(float(ratio), 2),
        "assign_reliable": bool(len(order) > 1 and ratio >= 2.0),
        "skipped_steps": ["sam3d", "sam3d_scale"],
        "note": "CAD 本身米制, 无需尺度估计; fp_pose/fuse 只从该路径取网格",
    }, ensure_ascii=False, indent=1))
    if len(order) > 1 and ratio < 2.0:
        print(f"[retrieval] ⚠ 最大两件的 mask 面积比仅 {ratio:.2f} (<2) —— "
              f"按面积排序不可靠, 部件可能装反, 已记进 retrieval.json")
    print(f"[retrieval] ✓ 使用资产, 跳过 sam3d/sam3d_scale -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
