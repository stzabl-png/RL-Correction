#!/usr/bin/env python3
"""Step: contact —— confidence 之后提取手↔物接触点(2D 证据修 3D 深度), 第 10 步。

纯 CPU 步骤(不进 GPU_STEPS)。薄包装, 真正的库在 ego_pipeline/contact/ 与 tools/,
设计说明见 tools/contact_align_heatmap.py 顶部与 ego_pipeline/contact/__init__.py。

链路(每条 take, 全幂等):
  1) bridge  recon_to_replay      world_fused → replay_world.npz     hawor  ~15s
  2) detect  phase.detect         2D mask 邻接 → 接触区间            hawor  ~30s/物体
     contact_auto.json 保持全物体并集(phase.auto 消费的旧契约不动);
     多物体 take 另出逐物体 contact_auto_<oid>.json —— 判"哪只手在摸哪个物体"
     必须逐物体: 并集会把右手摸瓶也算成"在接触", 杯子就被错提了
  3) 每个物体 × 每只有接触区间的手:
     make_ref_qpos --hand S       DexPilot → ref_qpos_S.npz(物体无关) MagicDexMate venv ~1s
     contact_align_heatmap --object oid  对齐+热度图(区间中点单帧)     biv2ap ~7min
     (不用 --auto-frame: 全片扫描 ~7s/帧 = 20 分钟级, 批量不可接受)
  4) 完成标记写 final 目录(带各手结果摘要), interim 清理后仍在

没有接触区间的手直接跳过(如 pour 右手握的瓶子没被重建, 不该对着杯子 mesh 提)。
依赖缺失(MagicDexMate venv 未部署, 例如 UCB 初次)时: 写 skipped 标记不挡批量,
日志醒目提示, 补部署后 --force 重跑即可。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent.parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.paths import final_video_dir, is_step_complete, write_step_completion  # noqa: E402

STEP = "contact"
REPO_ROOT = RECON_ROOT.parent.parent.parent          # Reconstruct_and_Retarget 仓库根
EGO = REPO_ROOT / "ego_pipeline"


def _find_mdm_py() -> Path:
    """dex_retargeting+pinocchio 的解释器: 本地=MagicDexMate venv;
    UCB=轻量 conda env dexretarget(2026-08-10 建, 复制本地 editable 源+pin2.7)。"""
    cands = [os.environ.get("MAGICDEX_PYTHON"),
             REPO_ROOT / "third_party" / "MagicDexMate" / ".venv-isaac" / "bin" / "python",
             Path.home() / "miniconda3" / "envs" / "dexretarget" / "bin" / "python"]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return Path(str(cands[1]))       # 缺省返回本地路径, 上层按"缺依赖"处理


MDM_PY = _find_mdm_py()


def _hawor_py() -> str:
    py = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
    return py if Path(py).is_file() else sys.executable


def _run(tag: str, cmd: list[str], cwd: Path | None = None) -> int:
    print(f"[contact] {tag}: {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None).returncode


def pick_frame(intervals: list[list[int]]) -> int:
    """最长接触区间的中点帧。"""
    s, e = max(intervals, key=lambda ab: ab[1] - ab[0])
    return int((s + e) // 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)            # 协议兼容; 本步纯 CPU
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")      # 诊断图本来就总是产出
    ap.add_argument("--no-heatmap", action="store_true",
                    help="跳过 4/4 对齐+热度图(每手~9min 的单帧优化)。接触语义走 "
                         "mano_contact(C1 全手探头)时热度图仅是选择器 fallback/RL 可视化, "
                         "GraspPose 已携带接触信息 —— 批量提取用这个开关提速 10x+")
    args, _extra = ap.parse_known_args(argv)

    scene = final_video_dir(args.dataset, args.video_id)
    if not (scene / "world_fused.npz").is_file():
        print(f"[contact] X fuse 未完成: {scene}", file=sys.stderr)
        return 2
    if is_step_complete(scene, STEP) and not args.force:
        print(f"[contact] 已完成, 跳过 {scene} (--force 重跑)")
        return 0

    if not MDM_PY.is_file():
        print(f"[contact] ⚠⚠ MagicDexMate venv 缺失({MDM_PY}) — 本 take 接触提取跳过, "
              f"部署后 --force 重跑。", flush=True)
        write_step_completion(scene, STEP, dataset=args.dataset, video_id=args.video_id,
                              extra={"contact": "skipped_missing_deps", "missing": str(MDM_PY)})
        return 0

    py = _hawor_py()
    if not (scene / "replay_world.npz").is_file():
        rc = _run("1/4 bridge", [py, EGO / "bridge" / "recon_to_replay.py",
                                 "--in", scene, "--cpu"])
        if rc:
            return rc

    import numpy as np                                     # hawor env 自带
    with np.load(scene / "world_fused.npz", allow_pickle=True) as z:
        obj_ids = ([str(x) for x in np.asarray(z["object_ids"]).tolist()]
                   if "object_ids" in z.files else ["object_0"])

    # 并集版 contact_auto.json: phase.auto 的旧契约, 始终产出
    if not (scene / "contact_auto.json").is_file():
        rc = _run("2/4 接触区间检测(并集)", [py, "-m", "phase.detect", scene], cwd=EGO)
        if rc:
            return rc

    summary: dict = {"objects": {}}
    done_qpos: set[str] = set()
    for oid in obj_ids:
        ca = scene / ("contact_auto.json" if len(obj_ids) == 1
                      else f"contact_auto_{oid}.json")
        if not ca.is_file():
            rc = _run(f"2/4 接触区间检测({oid})",
                      [py, "-m", "phase.detect", scene, "--object-id", oid,
                       "--out-name", ca.name], cwd=EGO)
            if rc:
                return rc
        ann = json.loads(ca.read_text())["annotations"]
        osum: dict = {}
        for side in ("left", "right"):
            ivs = ann.get(side) or []
            if not ivs:
                osum[side] = {"status": "no_contact_interval"}
                print(f"[contact] {oid}/{side}: 无接触区间, 跳过")
                continue
            frame = pick_frame(ivs)
            qp = scene / f"ref_qpos_{side}.npz"
            if side not in done_qpos and not qp.is_file():
                rc = _run(f"3/4 {side} 重定向", [MDM_PY, "-m", "contact.make_ref_qpos",
                                                scene, "--hand", side, "--out", qp], cwd=EGO)
                if rc:
                    return rc
            done_qpos.add(side)
            if args.no_heatmap:
                osum[side] = {"status": "heatmap_skipped", "frame": frame, "intervals": ivs}
                print(f"[contact] {oid}/{side}: --no-heatmap, 跳过对齐+热度图")
                continue
            cmd = ["conda", "run", "--no-capture-output", "-n", "biv2ap", "python",
                   REPO_ROOT / "tools" / "contact_align_heatmap.py", scene,
                   "--retarget-dir", scene, "--side", side, "--frame", frame,
                   "--object", oid, "--stage", "heatmap"]
            if args.video.is_file():
                cmd += ["--video", args.video]
            rc = _run(f"4/4 {oid}/{side} 对齐+热度图 @f{frame}", cmd)
            if rc:
                return rc
            sfx = "" if oid == "object_0" else f"_{oid}"
            res = scene / "contact" / f"stage4_frame{frame:04d}_{side}{sfx}.json"
            row: dict = {"status": "ok", "frame": frame, "intervals": ivs}
            try:
                d = json.loads(res.read_text())
                row.update({k: d[k] for k in ("hot_verts", "hot_frac", "per_pad") if k in d})
            except Exception:                                 # 摘要缺失不挡完成
                row["summary"] = "unparsed"
            osum[side] = row
        summary["objects"][oid] = osum

    write_step_completion(scene, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra=summary)
    brief = "; ".join(
        f"{oid} " + ", ".join(f"{sd}:{v['status']}" + (f"@f{v['frame']}" if "frame" in v else "")
                              for sd, v in osum.items())
        for oid, osum in summary["objects"].items())
    print(f"[contact] ✓ {args.video_id}: {brief}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
