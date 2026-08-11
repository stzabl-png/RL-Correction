# confidence 作为重建 pipeline 的最后一步（已实施）

> 2026-08-10。目标：一条数据 fuse 完成 → 立刻自动跑可信度评分 → 落 manifest。
> 状态：**已实施并验收**（§6 全部 6 步完成）。实施中发现并修掉的坑：
> ① marker extras 里的 manifest `status` 键会覆盖完成标记的 `status:"complete"`
>    导致幂等失效 → 改名 `manifest_status`；
> ② 断点续跑缺口：fuse 完成但 confidence 未跑的视频会被 `_final_complete` 整条跳过
>    → 批量队列在该分支只补跑 confidence（interim 已删也不会重跑重建）；
> ③ confidence 的 `_step_done` 必须查 final 目录（interim 会被 cleanup 删掉）；
> ④ 同物体自动判定在尺度拟合不可靠时退回形状比对（12_scene_v17aframes 案例）。
> 存量 33 条已回填 completion marker，批量队列不会重算。

## 0. 现状勘察结论（决定方案的四个事实）

1. **权威 pipeline 只有一份**：实施当日在 `$HV2RD_ROOT/recon_pipeline`。
   （★2026-08-10 晚已收编：HV2RD 删除，权威版迁入
   `Reconstruct_and_Retarget/ego_pipeline/Reconstruction/recon_pipeline/`，
   本文所有 HV2RD 路径按此对应；迁移台账见 R&R 根目录 HV2RD_MIGRATION.md。）
2. **两个入口都要挂**：
   - `run_pipeline.py`——单机顺序跑，`STEPS` 元组最后一项是 `fuse`；
   - `run_batch_queue.py`——批量队列（8 卡机在用），`AUTO_STEPS` + `STEP_ENVS` +
     `DEFAULT_STEP_GPU_MEM_MB` 三处注册，按显存预算调度 worker。
3. **步骤协议现成**：每步一个 `run_sequence.py`，收 `--dataset --video-id --video --gpu
   [--force]`；输出目录 `final_video_dir(dataset, video_id)`（`video_id` 按 `__` 拆层级）；
   幂等靠 `completion_marker` / `is_step_complete` / `write_step_completion`。
   fuse 完成后场景目录里有 `world_fused.npz + masks/ + objects/ + reconstruction_complete.json`
   ——正好是 confidence 工具链需要的全部输入。
4. **环境零新增**：`STEP_ENVS["fuse"]=="hawor"`，我们的四件工具（cc/audit/rts/manifest）
   也全跑在 hawor env。CoTracker 需 GPU ≈6.5GB。

## 1. 新步骤 `confidence`（挂在 fuse 之后）

新建 `recon_pipeline/confidence/run_sequence.py`（HV2RD 侧薄包装，≈100 行）：

```
输入:  --dataset --video-id --video --gpu [--force] [--visualize]
前置:  scene = final_video_dir(dataset, video_id)
       未见 reconstruction_complete.json → 报错退出(fuse 没完成)
       is_step_complete(scene, "confidence") 且非 --force → 跳过
执行(依次, 都调 RL_Correction 侧的既有工具):
  1) cotracker_consistency.py --scene $scene --video $video --no-viz   # GPU 1~2 分
  2) pose_audit.py --scene $scene --ct-dir $POSEQA/cc                  # ★需新增单take模式
  3) rts_smoother.py --scene $scene --audit $POSEQA/pose_audit.json
  4) take_manifest.py --audit ... --out $POSEQA/TAKE_MANIFEST.json     # 全量再生, 秒级
  (--visualize 时追加 conf_viz.py)
收尾:  write_step_completion(..., extra={conf_pos_median, conf_rot_median,
       rotation_usable, refuted_frames})   # 批量摘要里直接能看到分数
```

工具位置与数据根都走环境变量（沿用 repo_paths 风格，可覆盖）：

```
RL_CONF_TOOLS ?= /home/lyh/Project/RL_Correction/steps/step2_reconstruction
POSEQA_ROOT   ?= $RR_ROOT/Data/VideoPrior/poseqa
```

视频路径**直接用 job 传入的 `--video`**，不再走 `resolve_video()` 的猜路径逻辑
（那是给离线补跑用的，pipeline 里源视频本来就是已知的）。

## 2. 两个入口的注册改动

`run_pipeline.py`：`STEPS` 末尾加 `"confidence"`；`STEP_SCRIPTS` 加一行。

`run_batch_queue.py`：
```python
AUTO_STEPS += ("confidence",)
STEP_SCRIPTS["confidence"] = RECON_ROOT / "confidence" / "run_sequence.py"
STEP_ENVS["confidence"] = "hawor"
GPU_STEPS |= {"confidence"}
DEFAULT_STEP_GPU_MEM_MB["confidence"] = 7000     # CoTracker 实测 ~6.5GB
```

## 3. RL_Correction 侧需要的两个改动（实施前提）

1. **`pose_audit.py` 加 `--scene` 单 take 增量模式**。现在只有全库扫描（33 条 2.6 分钟，
   线性增长，批量场景不可接受）。增量 = 只审这条，读-改-写 `pose_audit.json` 里对应 entry。
2. **共享 json 加文件锁（fcntl.flock）**：批量队列多 worker 并发跑 confidence 时,
   `pose_audit.json` 与 `TAKE_MANIFEST.json` 是共享读改写点，必须锁；
   `cc/`、`rts/` 是按 take 分文件的，天然无冲突。

## 4. 人工门怎么办（pipeline 自动化的边界）

- **透明物体排除**（2026-08-10 用户裁定）：在**重建开始之前**由 VLM 看材质直接过滤，
  透明的根本不进重建流水线 —— confidence 步不用管材质。manifest 的 EXCLUDE 名单只
  保留给存量（video 3 家族）。
- **同视频多重建择优**：`take_manifest.py` 自动裁决——“同 task 目录 + 同源视频前缀的组，
  conf 最高者 active，其余 deselected”，人工名单只留 override。

## 5. 运行成本与调度

单 take 增量：CT 1~2 分（GPU）+ audit 单条 ~5 秒 + rts <10 秒 + manifest 秒级
≈ **2~3 分钟/条**，相对上游（vipe 33GB/sam3d 22GB、几十分钟）可忽略。
7GB 显存预算意味着批量机上 confidence 可以和小步骤并发，不挤大步骤。

## 6. 实施顺序（每步可独立验证）

1. `pose_audit.py --scene` 增量模式 + flock（RL_Correction 侧，离线可测：
   增量结果必须与全库重扫的该 take entry 逐字段一致）
2. `take_manifest.py` 同视频自动择优 + flock + `needs_review`
3. `confidence/run_sequence.py` 包装（HV2RD 侧）
4. 两入口注册
5. 冒烟：挑一条已完成 take，`run_pipeline.py --steps confidence --video-id ...`，
   校验分数与现库一致、marker 落盘、二次运行跳过、`--force` 重跑
6. 存量回填：给现有 33 条写 confidence completion marker（跑一个回填脚本），
   避免批量队列把老数据全部重算

## 7. 明确不做的

- 不动 R&R vendor 的 recon_pipeline 死代码（等迁移合并时一起处理）
- 不在 pipeline 里跑可视化（conf_viz/cc 视频只在 `--visualize` 时出，批量默认关）
- 不自动改 EXCLUDE/DESELECT 人工名单（自动裁决只加不减，人工裁定永远优先）
