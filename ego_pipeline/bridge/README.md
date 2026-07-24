# bridge (Reconstruct_and_Retarget 自己的胶水)

把 Reconstruction 的输出转成 Retargeting 的输入，并串起端到端。整体设计见 `../INTEGRATION_PLAN.md`。

- `recon_to_replay.py`（P2，待写）— `world_fused.npz` → `replay_world.npz`（hawor 环境跑 MANO + 时间轴对齐 +
  物体位姿四元数化）；复用 `../Retargeting/scripts/hawor_to_joints.py` 的 MANO 逻辑。
- `run_ego2robot.py`（P3，待写）— 端到端编排：Reconstruction → bridge → Retargeting，含中间产物清理。

回归基准：`third_party/MagicDexMate/RetargetInput/HOI4D/hoi4d_recon_samples/<vid>/replay_world.npz`。
