#!/usr/bin/env python3
"""VLM 门(第 5 步) —— 判材质(透明?) 与 分件(需要 Retrieval?)。纯 CPU + 远端 VLM 服务。

============================ 位置为什么在这 ============================

必须夹在 `sam2_object` 之后、`sam3d` 之前:

  * 之后: 判材质要逐实例的 mask 采样, 而实例是 v17A/SAM2 出的;
  * 之前: **分件判定决定下一步是"重建网格"还是"取 CAD 资产"**。放到最后就来不及了 ——
    单个刚体网格在原理上表达不了"盖相对瓶身转", 拧瓶盖这类动作必须换分件资产。

============================ 结果落在哪 ============================

    <interim>/<dataset>/<video_id>/vlm_gate.json
      objects.<inst>.material/verdict/filter   材质与是否建议剔除
      part_change.needs_retrieval              下一步 retrieval 读它

★ 这份文件是**唯一真相**: auto_label 的透明门与本步读写同一份, 谁先跑谁写, 后来者读缓存。
  以前两处各问一遍 VLM, 既花两倍(每次要整段视频 base64 上传), 又可能给出互相矛盾的结论。

============================ 服务不可达时 ============================

**不阻塞重建, 但必须留痕。** 写 status=skipped 并说明原因, 而不是默默当成"不透明"放行 ——
默默放行会让透明物体混进训练集且没人知道。下游看到 skipped 应当自己决定要不要用这条。

服务在 UCB 8 卡机的 GPU7(Qwen3.5-27B-Int4)。本地跑需要先开隧道, 见
`tools/vlm_tunnel.sh`。端点用 VLM_API_BASE 或 QWEN_BASE_URL(两个名字等价)。
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

from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402

REPO_ROOT = RECON_ROOT.parent.parent.parent
EGO = REPO_ROOT / "ego_pipeline"
STEP = "vlm_gate"


def _hawor_py() -> str:
    py = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
    return py if Path(py).is_file() else sys.executable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)          # 协议兼容; 本步不占本地卡
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--filter-policy", default="strict", choices=("strict", "v2"))
    args, extra = ap.parse_known_args(argv)

    step_dir = interim_step_dir(args.dataset, args.video_id, STEP)
    if is_step_complete(step_dir, STEP) and not args.force:
        print(f"[vlm_gate] 已完成, 跳过 {args.video_id} (--force 重跑)")
        return 0

    cmd = [_hawor_py(), EGO / "bin" / "vlm_gate_step.py",
           "--dataset", args.dataset, "--video-id", args.video_id,
           "--video", args.video, "--filter-policy", args.filter_policy]
    if args.force:
        cmd.append("--force")
    print(f"[vlm_gate] {' '.join(map(str, cmd))}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT)).returncode
    if rc:
        # ★ 非致命: VLM 是**旁路证据**, 挂了不该拖垮重建。但要写明"没判过"。
        print(f"[vlm_gate] ⚠ 判定失败(rc={rc}) —— 记为未判定, 不阻塞重建", flush=True)
        write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                              extra={"vlm_gate": "failed", "returncode": rc})
        return 0

    sys.path.insert(0, str(EGO / "bin"))
    from vlm_transparency_gate import gate_path                       # noqa: E402
    gp = gate_path(args.dataset, args.video_id)
    doc = json.loads(gp.read_text()) if gp.is_file() else {"status": "missing"}
    objs = doc.get("objects") or {}
    pc = doc.get("part_change") or {}
    summary = {
        "gate_json": str(gp), "gate_status": doc.get("status"),
        "n_instances": len(objs),
        "n_filtered": sum(1 for v in objs.values() if v.get("filter")),
        "needs_retrieval": pc.get("needs_retrieval"),
        "part_change": pc.get("part_change"),
    }
    write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra=summary)
    print(f"[vlm_gate] ✓ {args.video_id}: {summary['n_instances']} 实例, "
          f"建议剔除 {summary['n_filtered']}, 需Retrieval={summary['needs_retrieval']} "
          f"({summary['part_change']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
