#!/usr/bin/env python3
"""VLM 检索判定(第 5 步) —— 判分件: 这条 take 要不要走资产库 Retrieval。纯 CPU + 远端 VLM。

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

============================ 服务不可达时: **终止本条并报错** ============================

用户 2026-08-14 裁定: VLM 服务不可达 -> 本条 take 直接失败, 不允许"跳过继续"。

理由: 这一步的产物决定下一步是**重建网格**还是**取 CAD 资产**。判不出来时继续跑,
等于默默按"不需要资产"处理 —— 而分件物体用单刚体网格在原理上就表达不了
(clip0 实测单件 conf_rot 23 旋转弃用, 换 CAD 双件后 34 可用)。
错的默认比没有结果更糟: 数据看起来齐全, 没人知道它是猜的。

批量里一条失败不影响其余(队列逐条记 failed)。

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
STEP = "vlm_retrieval"


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
        print(f"[vlm_retrieval] 已完成, 跳过 {args.video_id} (--force 重跑)")
        return 0

    cmd = [_hawor_py(), EGO / "bin" / "vlm_gate_step.py",
           "--dataset", args.dataset, "--video-id", args.video_id,
           "--video", args.video, "--filter-policy", args.filter_policy]
    if args.force:
        cmd.append("--force")
    print(f"[vlm_retrieval] {' '.join(map(str, cmd))}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT)).returncode
    if rc:
        print(f"[vlm_retrieval] X 判定失败(rc={rc}) —— **本条终止**, 不写完成标记", file=sys.stderr)
        return 1

    sys.path.insert(0, str(EGO / "bin"))
    from vlm_transparency_gate import gate_path                       # noqa: E402
    gp = gate_path(args.dataset, args.video_id)
    doc = json.loads(gp.read_text()) if gp.is_file() else {"status": "missing"}
    # ★ 服务不可达时 vlm_gate_step 会写 status=skipped 并返回 0(它自己不做裁决)。
    #   裁决在这里: 判不出分件就终止, 不允许按"不需要资产"的默认往下跑。
    if doc.get("status") != "ok":
        print(f"[vlm_retrieval] X {args.video_id}: 分件未判定({doc.get('reason') or doc.get('status')})"
              f" —— **本条终止**。VLM 服务在 UCB 8 卡机 GPU7, 本地需先开隧道:\n"
              f"      ./tools/vlm_tunnel.sh\n"
              f"    判不出来就继续 = 默默按'不需要资产'处理, 而分件物体用单刚体网格"
              f"表达不了(clip0 单件 conf_rot 23 弃用 -> CAD 双件 34 可用)。",
              file=sys.stderr)
        return 1
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
    print(f"[vlm_retrieval] ✓ {args.video_id}: {summary['n_instances']} 实例, "
          f"建议剔除 {summary['n_filtered']}, 需Retrieval={summary['needs_retrieval']} "
          f"({summary['part_change']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
