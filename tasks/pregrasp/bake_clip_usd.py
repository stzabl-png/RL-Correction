"""replay_grasp clip 的 USD 显式烘焙 (标准前置步).

replay_grasp 源的 clip, correction_env 假定 object USD "已就位" (correction_env.py:399),
view_curobo_plan / build_approach 也直接 load USD —— 都不烘. 本工具按 clip 定义把
主物体的**视觉 USD** (objects/object_*.usd, 默认凸分解) 和次物体的**物理 USD**
(cache/object_*.usd, 128 hull + shrink_wrap, 治瓶颈幻影锥壳) 一次烘好.

双物体 clip (pour 瓶+杯) 跑两遍即可覆盖全部 4 个 USD:
  bake_clip_usd.py --clip Pour31_bottle   # 瓶视觉 objects/1 + 杯物理 cache/0
  bake_clip_usd.py --clip Pour31_cup      # 杯视觉 objects/0 + 瓶物理 cache/1

用带 isaacsim 的解释器跑 (需 Kit).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("bake_clip_usd")
app = AppLauncher(args).app

import os  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402

e = clips.clip_entry(args.clip)
assert e["source"] == "replay_grasp", f"本工具只处理 replay_grasp clip, {args.clip} 是 {e['source']}"

# 主物体: 视觉 USD (entry['usd'] = objects/object_*.usd), 默认凸分解.
print(f"[bake] {args.clip} 主物体视觉 -> {e['usd']}")
clips.ensure_mesh_usd(e["mesh"], e["usd"], e["semantics"])

# 次物体: 物理 USD (sec['usd'] = cache/object_*.usd), 128 hull + shrink_wrap.
sec = e["secondary"]
print(f"[bake] {args.clip} 次物体物理 -> {sec['usd']} (128 hull + shrink_wrap)")
clips.ensure_mesh_usd(sec["mesh"], sec["usd"], sec["semantics"],
                      max_convex_hulls=int(sec.get("usd_convex_hulls", 128)),
                      shrink_wrap=bool(sec.get("usd_shrink_wrap", True)))

print("[bake] 完成")
app.close()
