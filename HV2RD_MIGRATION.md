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

## 执行记录（2026-08-10 全部完成）

1. ✅ 快照提交 23 项本地改动（contact/v17a 工具集等），删 3 个 .bak
2. ✅ merge origin/Step2_NoisyRecon（confidence 工具链）+ origin/agent/v17a-auto-mask-reconstruction（开朗），两路均无冲突
3. ✅ rsync HV2RD/recon_pipeline 覆盖 vendor 份 → 恢复开朗的 sam3_hands/sam3d_scale/README 三个文件
   （sam3_hands/sam3d_scale 两份 copy 原本逐字节相同，换底无损；README 的 HV2RD 版无独有内容）
4. ✅ mv third_party(32GB)/data(1.1MB) → ego_pipeline/Reconstruction/ 下（同盘秒完成；
   .gitignore 原有规则已覆盖）；5 个子模块 .git 指针拆除，pin 存 docs/hv2rd_archive/submodule_pins.txt
5. ✅ 指针修复：repo_paths.sh/.py 摘除 HV2RD_ROOT（RECON_PIPELINE 远端本已指仓内）；
   删 sync_recon.sh（两份 copy 同步器，使命结束）；build_db.sh、RL_Correction/step1 paths.py
   指向新 third_party；init_submodules.sh 改读固化 pin
6. ✅ 验证：py_compile 全过；vipe editable 安装的烤死路径已修（__editable__*_finder.py 的
   MAPPING）且 uv run 加 --no-sync（不加会联网重解析卡死）；sam2/FoundationPose 新位置
   import 通过；reconstruct.sh --dry-run 9 步成链、分层 video_id 正确
   （★merge 曾把 EGODEX_ROOT 默认改成同事机器的 Data/EgoDex，已改回本机 V2AP 路径）；
   confidence 步幂等跳过正常
7. ✅ 归档：hv2rd_full_history.bundle(24MB 全历史) + 未提交 patch + status/log 快照
   → docs/hv2rd_archive/；确认无进程占用后 rm -rf HV2RD
8. ✅ 提交推送 Step2_NoisyRecon

## 迁移后的坑备忘

- vipe 必须 `uv run --no-sync`（step_launcher/run_batch_queue 已内置）；若将来真要
  uv sync，先备份 .venv —— sync 可能按 pyproject 重装 8.4GB 环境
- editable 安装（vipe）换路径要修 site-packages 的 `__editable__*_finder.py`
- 子模块已转普通目录：升级上游版本时按 submodule_pins.txt 的 commit 对齐
- 同事机器如依赖 HV2RD_ROOT / Data/EgoDex 布局，用同名环境变量覆盖，别改默认值
