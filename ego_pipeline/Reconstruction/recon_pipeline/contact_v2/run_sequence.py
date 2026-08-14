#!/usr/bin/env python3
"""接触点提取(第 10 步) —— 从重建里**测量**手在物体上碰到了哪里。

    1/4  replay_world.npz 陈旧/缺失 -> 重做 bridge (手的 MANO 网格来源)
    2/4  接触区间检测 phase.detect  -> contact_auto.json(并集) + contact_auto_<oid>.json
    3/4  接触点提取 contact.extract_v2                      纯 CPU, 整条 take 2~5 秒
    4/4  给 GraspPose Agent 的交接件 contact.grasp_prompt

============================ 与旧 contact 步的关系 ============================

旧步 `contact/run_sequence.py` **仍在仓里但不再默认启用**(不在 run_pipeline 的 STEPS 里,
用 `--steps contact_align` 可单独调)。两者不是快慢之分, 是**答的不是同一个问题**:

  旧: 搜索一个 3-DoF 平移把 **SharpaWave 机器手**摆到物体上, 能量项里 `w_touch` 主动
      把手往表面拉 —— 即"假设一定有抓握"然后解出深度。结果必然几何合理, 但那是优化器
      摆出来的, 不是视频里的证据。单帧 ~7min(约 1200 次能量评估)。
      ★ 手位修正后它的 bite IoU 从 1.000 掉到 0.000 —— 之前的满分是两个错误相消
        (手偏 16cm, 对齐器推回 15.5cm)。所以它不能当测量用。

  新: 不搜索, 逐帧量一次。整个稳定窗(11~20 帧)一起统计, 2~5 秒。测不出来就如实报失败。

需要"几何合理的抓握先验"时, 用 `extract_v2 --object-follows-hand`(产物走 contact_prior_*,
不写 GRASP 阶段标签、不给 RL 当监督) —— 同样是假设, 但代价从 1200 次搜索降到一次解析平移。

============================ 产物 ============================

    <take>/contact/contact_v2_<oid>_<side>.npz   接触热度(+.ply 可直接看)
    <take>/contact/contact_v2_summary.json       **所有**尝试过的组合(含失败)
    <take>/contact/contact_auto_grasp.json       逐帧阶段标签 FREE/CONTACT/GRASP
                                                 phase.load_phase() 直读, 下游零改动
    <take>/contact/grasp_prompt.{md,json}        给 GraspPose Agent
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

REPO_ROOT = RECON_ROOT.parent.parent.parent
EGO = REPO_ROOT / "ego_pipeline"
STEP = "contact"


def _hawor_py() -> str:
    py = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
    return py if Path(py).is_file() else sys.executable


def _run(tag: str, cmd: list[str], cwd: Path | None = None) -> int:
    print(f"[contact] {tag}: {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None).returncode


def _stale(dst: Path, *srcs: Path) -> bool:
    """dst 比任一源旧(或缺失) -> 需要重做。

    ★ 只判"存在"会静默用过期数据。实测踩过两次: fuse 修好手部变换后, take 的
      replay_world.npz 没人重生成, 于是接触提取读的是修复前的手 —— 投影到图上离真实的手
      379~432px(正常 32~69px), 而**不报任何错**, 结果被误判成"物体位姿差"。
    """
    if not dst.is_file():
        return True
    m = dst.stat().st_mtime
    return any(s.is_file() and s.stat().st_mtime > m + 1.0 for s in srcs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)          # 协议兼容; 本步纯 CPU
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    args, extra = ap.parse_known_args(argv)

    scene = final_video_dir(args.dataset, args.video_id)
    if not (scene / "world_fused.npz").is_file():
        print(f"[contact] X fuse 未完成: {scene}", file=sys.stderr)
        return 2
    if is_step_complete(scene, STEP) and not args.force:
        print(f"[contact] 已完成, 跳过 {scene} (--force 重跑)")
        return 0

    py = _hawor_py()

    # ---- 1/4 手的 MANO 网格 ----
    if _stale(scene / "replay_world.npz", scene / "world_fused.npz"):
        why = "缺失" if not (scene / "replay_world.npz").is_file() else "比 world_fused.npz 旧"
        print(f"[contact] replay_world.npz {why} -> 重做 bridge")
        if _run("1/4 bridge", [py, EGO / "bridge" / "recon_to_replay.py",
                               "--in", scene, "--out", scene, "--cpu"]):
            return 1

    import numpy as np                                     # hawor env 自带
    with np.load(scene / "world_fused.npz", allow_pickle=True) as z:
        oids = ([str(x) for x in np.asarray(z["object_ids"]).tolist()]
                if "object_ids" in z.files else ["object_0"])

    # ---- 2/4 接触区间 ----
    # 并集版是 phase.auto 的旧契约, 始终产出; 多物体时**必须**再出逐物体版 ——
    # 并集会把"右手在摸瓶盖"也算成"右手在摸瓶身"(实测差 37 帧)。
    if not (scene / "contact_auto.json").is_file():
        if _run("2/4 接触区间(并集)", [py, "-m", "phase.detect", scene], cwd=EGO):
            return 1
    if len(oids) > 1:
        for oid in oids:
            ca = scene / f"contact_auto_{oid}.json"
            if not ca.is_file():
                if _run(f"2/4 接触区间({oid})",
                        [py, "-m", "phase.detect", scene, "--object-id", oid,
                         "--out-name", ca.name], cwd=EGO):
                    return 1

    # ---- 3/4 提取 ----
    if _run("3/4 接触点提取", [py, "-m", "ego_pipeline.contact.extract_v2", scene] + extra,
            cwd=REPO_ROOT):
        return 1

    # ---- 4/4 交接件 ----
    if _run("4/4 GraspPose 交接件",
            [py, "-m", "ego_pipeline.contact.grasp_prompt", scene], cwd=REPO_ROOT):
        return 1

    sm = scene / "contact" / "contact_v2_summary.json"
    summary = json.loads(sm.read_text()) if sm.is_file() else {"attempts": []}
    write_step_completion(scene, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra={"extractor": "extract_v2", **summary})
    ok = [a for a in summary.get("attempts", []) if a["status"] == "ok"]
    brief = ", ".join(f"{a['object_id']}×{a['side']}(对生{a['opposition']:.2f})" for a in ok) \
        or "无可用抓握"
    print(f"[contact] ✓ {args.video_id}: {len(ok)}/{len(summary.get('attempts', []))} 可用 —— {brief}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
