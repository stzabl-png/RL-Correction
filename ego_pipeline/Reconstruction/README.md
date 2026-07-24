# Reconstruction (vendored from HV2RD)

第一视角 mp4 → `world_fused.npz` + `object_mesh_scaled_final.obj`。
代码 vendor 自 Jiakai 的 `HumanVideo2RobotData`（HV2RD）。**HV2RD 是与 Jiakai 的 git 交流桥梁**，
本目录是 Reconstruct_and_Retarget 内的纯代码副本。整体设计见 `../INTEGRATION_PLAN.md`。

## 内容
- `recon_pipeline/` — pipeline 代码（由 `../sync/sync_recon.sh` 与 HV2RD 双向同步，纯代码）。
- `third_party/` — 6 个模型仓库，由 `setup/init_submodules.sh` 按 HV2RD 的 pinned 提交克隆（gitignore）。
- `weights/` — 由 `setup/setup_weights.sh` 拷贝本机已有权重（gitignore，见 `WEIGHTS.md`）。
- `setup/` — `init_submodules.sh` / `build_exts.sh` / `setup_weights.sh`（P1）。

## 与 HV2RD 同步
```bash
cd /home/lyh/Project/HumanVideo2RobotData && git pull   # 拉 Jiakai 更新
../sync/sync_recon.sh pull            # 预览 → 加 --apply 落地
# 反向：改完本地 recon_pipeline 后
../sync/sync_recon.sh push --apply    # 回灌 HV2RD → 在 HV2RD 里 git commit && git push
```

## 运行（P1 起）
环境 cu128(.venv) / HV2RD / hawor / biv2ap 为机器级 conda 环境，按名复用。
输入扫 `Reconstruct_and_Retarget/Data/**/*.mp4`，输出镜像到 `Reconstruct_and_Retarget/Output/ReconstructOutput/<dataset>/<嵌套 take 路径>/`
（结构与原数据集一致；retarget 产物在并行的 `Output/RetargetOutput/<同路径>/`）。
