> # ⚠ 这是**快照,不是权威源** —— 2026-08-16 加
>
> 本目录的 `*.py` 是从 **`/home/lyh/Project/Reconstruct_and_Retarget/confidence/`** 拷来的
> 一份快照(**两个仓是独立的**,各有各的 `.git`)。它**从未被提交过**,在 git 里显示为未跟踪,
> 因而看起来像"有人正在改还没提交"—— 不是。
>
> **它现在已经过期。** 例:重建侧 2026-08-15 夜改过 `pose_projection_check.py`
> (加 `--object`、订正内参注释),本快照没有。
>
> **要读实现、要跑脚本,一律去权威源**;本目录只当"这条链有哪些环节"的索引看。
> 唯一的例外是 `hand_audit.py` —— 那是 RL 侧自己写的诊断工具,权威源里没有。
>
> (这个坑与 2026-08-15 那一串同族:**两处同名内容、不同步,而读的人不知道自己读的是哪一份。**
>  见 `docs/POUR_TRAINING_DESIGN.md` §15.9~15.11。)

# step2_reconstruction —— 三维重建 + 轨迹可信度

## 已就位:物体轨迹可信度链路 (2026-08-10 定稿)

打分(pose_audit) → CT 第二观察员(cotracker_consistency) → 平滑(rts_smoother) →
take 裁决(take_manifest)。**评分标准、smooth/filtered/σ 用法、消费规则全在
`CONFIDENCE_GUIDE.md`**,下游取数入口是 `poseqa/TAKE_MANIFEST.json`。
confidence 已注册为重建 pipeline 第 9 步(fuse 之后自动跑),接入方案见
`PIPELINE_INTEGRATION.md`。

## 重建本体:在 R&R 仓内(HV2RD 已收编删除)

2026-08-10 起唯一权威在
`Reconstruct_and_Retarget/ego_pipeline/Reconstruction/recon_pipeline/`
(= GitHub RL-Correction 的 Step2_NoisyRecon 分支)。两份 copy 分叉的历史问题已终结:
HV2RD 实跑版为底 + 开朗的 sam3d_scale/sam3_hands 补丁合体后入仓,third_party(32GB)
与 data 同步迁入 `ego_pipeline/Reconstruction/` 下,HV2RD 仓库已归档删除
(全过程与坑见 R&R 根目录 `HV2RD_MIGRATION.md`,git 历史 bundle 在
`ego_pipeline/Reconstruction/docs/hv2rd_archive/`)。

启动入口不变:`ego_pipeline/reconstruct.sh`(批量) / `recon_pipeline/run_pipeline.py`(顺序)。
