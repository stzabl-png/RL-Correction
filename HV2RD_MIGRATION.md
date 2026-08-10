# HV2RD → Reconstruct_and_Retarget 收编迁移（2026-08-10）

> 用户裁定：不再使用 HumanVideo2RobotData（HV2RD），一切归入 R&R；收编完成后删除 HV2RD。
> 本文档是执行台账：先写清方案，执行时逐项打勾，坑记录在末尾。

## 迁移前的家底（2026-08-10 侦察）

```
HV2RD (34GB, git 仓库, remote=jiaka1chen/HumanVideo2RobotData, 本地 11 项未提交)
├── recon_pipeline/   1.7MB  ★实际在跑的权威版(比 R&R vendor 份多 fuse 轨迹清洗、
│                            hawor SAM3 补帧、confidence 第9步等提交)
├── third_party/      32GB   sam-3d-objects 13G / vipe 8.4G / HOI-DETR 5.6G / hawor 3.6G
│                            / sam2 926M / FoundationPose 273M / mmcv 145M / sam3 75M
├── data/             1.1MB  object_labels(标注缓存,要迁) / video_lists / interim(残渣)
└── README.md

R&R = GitHub stzabl-png/RL-Correction 的工作区, 分支 Step2_NoisyRecon
  - vendor 的 recon_pipeline 已落后 HV2RD (10 文件分叉, HV2RD 新)
  - 同事(开朗)的更新在 origin/agent/v17a-auto-mask-reconstruction:
    sam3d_scale(SAM3D尺度为准,FP降级诊断) + sam3_hands(--checkpoint) +
    recon_kailang 适配器 + v17A experimental 升级
  - R&R/third_party 与 HV2RD/third_party 撞名的 3 个目录(hawor/vipe/sam-3d-objects)
    是不同用途的 checkout(R&R 侧是旧 benchmark 工具用的), 互不干扰
```

## 方案（8 个阶段）

1. **快照提交**：R&R 现有 23 项未提交改动先原样入库（删 .bak 垃圾），迁移不和旧账混在一起
2. **拉平远端**：pull 拿到 confidence 提交；merge 同事的 agent/v17a 分支（无文件重叠，应干净）
3. **代码合并**：用 HV2RD 版整体覆盖 vendor 的 recon_pipeline，再恢复同事改的两个文件
   （sam3_hands/sam3d_scale 两份 copy 里原本逐字节相同，patch 可无损换底）
4. **资产迁移**：`mv` HV2RD 的 third_party/ 与 data/ 到 `ego_pipeline/Reconstruction/` 下
   （同文件系统秒完成；REPO_ROOT=ego_pipeline/Reconstruction，相对布局天然吻合，不用改代码）；
   拆掉子模块的 .git 指针（母仓要删，留着就是悬空引用）；.gitignore 挡住 32GB 资产
5. **指针修复**：repo_paths.sh/py 把 RECON_PIPELINE 指回仓内；修所有硬编码
   HumanVideo2RobotData 的文件（label.sh / sync_recon.sh / build_db.sh / init_submodules.sh /
   _legacy/*/common.py / object_io.py / urdf_fk.py / object_mask_stage.py …grep 为准）
6. **验证**：py_compile 全量；vipe uv venv 迁移后可 import；FoundationPose/sam2/sam3 按新路径
   可加载；`reconstruct.sh --dry-run` 全步骤成链；confidence 步冒烟
7. **归档+删除**：HV2RD 的 git 历史打 bundle + 未提交改动打 patch 存进 R&R docs/hv2rd_archive/
   （几 MB 的保险，Jiakai 的远端仓库也还在）；确认无进程、无残余引用后 `rm -rf`
8. **收尾**：提交推送 Step2_NoisyRecon；更新 RL_Correction 侧文档与记忆

## 执行记录

（执行时逐项补充）
