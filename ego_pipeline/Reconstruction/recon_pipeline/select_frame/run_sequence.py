#!/usr/bin/env python3
"""选帧(第 4 步) —— 决定 **SAM3D 重建帧 / sam3d_scale 尺度参考帧**。

╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠ 本步骤是**占位**, 等杜邦(7mare)接入 —— 当前实现只是"什么都不做"。          ║
║    未接入时全链照常跑通(sam3d/sam3d_scale 回退到标注 prompt 帧), 但选帧质量   ║
║    直接决定物体位姿与尺度的上限, 属于**硬性缺口**而非可选优化。              ║
╚══════════════════════════════════════════════════════════════════════════════╝

============================ 你(杜邦)只需要做一件事 ============================

往 `frame_plan.json` 的 `objects.<oid>.sam3d_frame` 写一个帧号, **别的都不用改**。

  文件位置: `<interim>/<dataset>/<video_id>/sam2_object/frame_plan.json`
  写法    : `_common.frame_plan.write_plan()` / 直接改 json 都行, schema 见
            `_common/frame_plan.py` 头注(schema_version = frame_plan_v1)

消费方**已经接好线**, 不需要你动:
  * `sam3d/run_sequence.py:558`        读 sam3d_frame 当重建帧
  * `sam3d_scale/run_sequence.py:333`  读同一个值当尺度参考帧(两步必须同帧)
  值为 null 时两步都回退到标注 prompt 帧; 该帧 mask 为空时也回退并打日志 ——
  **计划永远只是建议**, 你写错了不会让管线崩, 只会退回旧行为。

★ **不要碰 `fp_register_frame`。** 用户 2026-08-14 明确: FoundationPose 注册帧固定为
  **交互开始帧 + 10**(auto_label 已自动算好写入), 不走选帧器。三个"帧"各司其职:
      重建帧/尺度帧  <- 你的选帧器(要"看得最清楚、遮挡最少"的一帧)
      FP 注册帧      <- 交互帧+10  (要"物体已被拿稳、姿态代表全片"的一帧)

============================ 你的代码在哪 ============================

`agent/v17a-scale-develop` 分支的 `scale_develop/select_frame_v2.py`(279 行, 两段式):
  第一段 几何粗筛: 贴边/碎裂/时间窗相对面积/凸度/填充率/Laplacian 清晰度 -> 前 60
  第二段 像素级遮挡精排: SAM2 以 HOI-DETR 手框为 prompt 出手 mask, 算
      occ = |手∩物体凸包|/|凸包| + 0.25×|手∩边界环|/|环|
      (凸包项天然区分手在物体前/后 —— 手在后不算遮挡)
  3 视频验证: ketchup f514->f162, phone f234->f278, scissors f514->f446

接入时要改的两处(与主线已收敛的约定):
  1. **env**: 你用的 `hoidetr` / `sam3` 两个环境, 主线已统一为 `codetr`(SAM3+SAM2+
     HOI-DETR 都在里面) 与 `biv2ap`(SAM3D/FP)。`HV2RD` env 已于 2026-08-14 删除。
     环境分工权威在 `run_batch_queue.STEP_ENVS`。
  2. **输入路径**: 用 `_common.paths.interim_step_dir(dataset, video_id, "sam2_object")`
     取 label_prompt/frame_plan; v17A 的 mask manifest 在
     `experimental/hoi_detr_v17a/data/interim/<dataset>/<video_id>/instance_pipeline_v17a/`
     ⚠ 多物体必须读**视频级** `video_mask_sequence.json`, 不是 episode 级
     (见仓库 CLAUDE.md 第 6 条, 喂错会让多物体静默塌成单物体)

============================ 位置为什么在这 ============================

排在 `sam2_object` 之后(要它的逐帧 mask)、`sam3d` 之前(要在重建前给出帧号)。
本步骤纯 CPU + 一次 SAM2 图像模式推理, 不占大显存。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent.parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402

STEP = "select_frame"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    args, _extra = ap.parse_known_args(argv)

    step_dir = interim_step_dir(args.dataset, args.video_id, STEP)
    if is_step_complete(step_dir, STEP) and not args.force:
        return 0

    # ── 占位: 不写 sam3d_frame, 下游回退到 prompt 帧 ──────────────────────────
    print(f"[select_frame] {args.video_id}: **未接入**(杜邦的选帧器待接) —— "
          f"sam3d_frame 保持 null, sam3d/sam3d_scale 回退到标注 prompt 帧。"
          f"接入指引见本文件头注。", flush=True)
    write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra={"status_detail": "not_implemented",
                                 "owner": "7mare(杜邦)",
                                 "contract": "写 frame_plan.json 的 objects.<oid>.sam3d_frame",
                                 "source_branch": "agent/v17a-scale-develop",
                                 "source_file": "scale_develop/select_frame_v2.py"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
