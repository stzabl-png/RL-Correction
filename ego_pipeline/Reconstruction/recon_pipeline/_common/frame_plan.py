"""frame_plan.json —— 逐物体的"关键帧计划"契约(2026-08-11)。

背景: 最适合 FP 配准的帧 ≠ 最适合 SAM3D 重建的帧。
  * FP 配准帧: 同事实测**交互开始帧 + 10** 效果好(物体已被拿稳、遮挡姿态代表全片) ——
    auto_label 从 v17A 的 interaction_onset_frame 自动算好写入;
  * SAM3D 重建帧: 杜邦在做专门的选帧器 —— **sam3d_frame 留空(null)给他填**,
    null 时各步回退现行为(标注 prompt 帧)。他只需往本文件写数字, 不用改任何步骤代码。

文件位置: sam2_object step 目录下 `frame_plan.json`(与 label_prompt.json 同处)。
消费者: fp_pose(fp_register_frame) / sam3d + sam3d_scale(sam3d_frame, 两步必须同帧)。
安全性: 计划帧上的 mask 为空时消费方回退 prompt 帧并打日志 —— 计划永远只是建议。

schema:
{
 "schema_version": "frame_plan_v1",
 "objects": {
   "object_0": {
     "fp_register_frame": 13,  "fp_source": "interaction_onset(3)+10",
     "sam3d_frame": null,      "sam3d_source": "reserved(dubang 选帧器待接; null=用 prompt 帧)"
   }
 }
}
"""
from __future__ import annotations

import json
from pathlib import Path

FRAME_PLAN_FILENAME = "frame_plan.json"
SCHEMA = "frame_plan_v1"


def frame_plan_path(sam2_object_dir: Path) -> Path:
    return Path(sam2_object_dir) / FRAME_PLAN_FILENAME


def load_frame_plan(sam2_object_dir: Path) -> dict:
    p = frame_plan_path(sam2_object_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def planned_frame(plan: dict, object_id: str, key: str) -> int | None:
    """None = 计划缺席/留空 → 调用方用自己的默认(prompt 帧)。"""
    v = ((plan.get("objects") or {}).get(object_id) or {}).get(key)
    return int(v) if v is not None else None


def write_frame_plan(sam2_object_dir: Path, objects: dict) -> Path:
    p = frame_plan_path(sam2_object_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema_version": SCHEMA, "objects": objects},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return p
