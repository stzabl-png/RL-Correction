# Dexonomy（SharpaWave 定制版）

本仓库在上游 Dexonomy（RSS 2025 typed grasp 合成）基础上接入了 SharpaWave 右手，并打通了
"物体 mesh → GraspPose 合成 → Isaac PhysX 验证"的完整闭环。

**做抓取生成/验证任务前，先读 `tools/GRASP_PIPELINE_MANUAL.md`** —— 一键脚本用法、
各步骤输入输出、数据格式、坐标系约定和历史教训都在里面。

速查：
- 一键管线：`tools/grasp_pipeline.sh <mesh.obj> <名字>`（dexonomy 环境，仓库根目录执行）
- 最终交付物：`output/<名>_sharpa_wave/isaac_succ/`（成功抓取 npy + 视频 + 报告）
- 环境：合成用 conda env `dexonomy`；Isaac 走 ocir 仓库的包装脚本（env_isaacsim），
  服务器必须 `--mode local` 启动
- 用户已确认的约束：不要"虎口朝下"的抓取（管线 step 4.5 自动过滤）
