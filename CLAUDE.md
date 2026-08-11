# Reconstruct_and_Retarget

一条 egocentric 视频 → 重建(相机/双手 MANO/物体 mesh+6DoF) → 轨迹可信度打分 → 供 RL 用。
GitHub = stzabl-png/RL-Correction 的 Step2_NoisyRecon 分支(本目录就是其工作区)。

## 先读这些, 再下结论

| 主题 | 权威文档 |
|---|---|
| 怎么跑重建(9 步全链) | `ego_pipeline/reconstruct.sh` 头部注释 |
| 远程跑法(默认!)/同步/拉回 | `ego_pipeline/Reconstruction/docs/REMOTE_RUNBOOK.md` |
| 分步耗时与设备边界 | `ego_pipeline/Reconstruction/docs/RUNTIME_LEDGER.md` |
| 可信度评分/σ 用法 | `RL_Correction/steps/step2_reconstruction/CONFIDENCE_GUIDE.md`(仓内快照 `confidence/`) |
| HV2RD 收编史(已删除) | `HV2RD_MIGRATION.md` |

## 立刻会犯错的几件事

1. **物体标注默认全自动, 人工点选只是 fallback。** `label_object.py` 的 "Interactive"
   docstring 和 run_pipeline 的 `--label-mode` 报错都只描述 fallback 路线——**不要**据此
   推断"每条视频要人工标注 2 分钟"。自动路线: `ego_pipeline/bin/auto_label_v17a.py`
   (reconstruct.sh 标注缺失时自动调; 机制=v17A 选帧取内切极点只写 label_prompt,
   SAM2 自己传播全片 —— **不要**改回直接搬 v17A mask, 选帧质量门会把覆盖砍成几帧)。
2. **跑数据一律远程 UCB 8 卡**(yanghong@169.229.192.185), 本地 4080S 16GB 只做开发——
   sam3d/sam3d_scale/fp_pose 三步峰值 26~27GB, 本地必 OOM。见 REMOTE_RUNBOOK。
3. **远程选卡只认 `--gpu N` 传参**: 步骤脚本会覆写 CUDA_VISIBLE_DEVICES, 外部 export 无效。
4. SAM3_VERSION=sam3 已在 reconstruct.sh 固化(sam3.1 三种卡全有问题), 别改回。
5. vipe 必须 `uv run --no-sync`(已内置); 真要 uv sync 先备份 .venv。
6. **v17A 两份 manifest 别混**(2026-08-10 踩过: 喂错让多物体静默塌成单物体):
   - **episode 级** `mask_sequence.json` (`persistent_mask_sequence_v1`): 单实例路径用;
     开朗的 mask 搬运 adapter 也只认它(他 README 的视频级示例有误)。
   - **视频级** `video_mask_sequence.json` (`persistent_video_mask_sequence_v1`):
     **多物体必须用这份** —— 注册出来的部件只出现在这里。clip4 实测视频级
     `object_0001`+`object_0002`(盖在 f59 注册), 而同一条 clip 的 episode 级只有
     `instance_0001`。入口: `auto_label_v17a.py --instance all`
     (或环境变量 `AUTO_LABEL_INSTANCE=all`)。
7. 从单个文件的 docstring 推断能力边界之前, 先 grep `tools/` 和 `ego_pipeline/bin/`——
   这个仓的自动化入口多数在这两处。
