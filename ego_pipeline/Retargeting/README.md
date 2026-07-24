# Retargeting (vendored from MagicDexMate)

Recon 产物 → SharpaWave 22-DOF qpos / Isaac 仿真。代码 vendor 自 `Reconstruct_and_Retarget/third_party/MagicDexMate`
（你自己的 retarget 仓库），由 `../sync/sync_retarget.sh` 刷新。输入来自 `../Reconstruction` 的产物
（经 `../bridge` 转成 `replay_world.npz` + `object.usd`）。整体设计见 `../INTEGRATION_PLAN.md`。

## 内容（vendor 的部分）
- `magicdexmate/` 核心包 · `sim/` 仿真 · `scripts/` 工具(obj_to_usd 等) · `configs/` retarget 配置
- `assets/robots/hands/` SharpaWave URDF+mesh（release 需要）

## 环境
`.venv`（teleop）/ `.venv-isaac`（Isaac）仍由 MagicDexMate 那边的 uv 管理（绝对路径），不 vendor。
首次启动 Isaac 加 `OMNI_KIT_ACCEPT_EULA=YES`。

## 刷新
```bash
../sync/sync_retarget.sh            # 预览
../sync/sync_retarget.sh --apply    # 落地
```
