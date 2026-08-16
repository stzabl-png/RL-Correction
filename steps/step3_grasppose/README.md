# step3_grasppose —— 抓取合成与优化

给 Step2 重建出来的物体网格 + 接触点云，合成一批 SharpaWave 能真做出来的 GraspPose，
排序后交给 Step4。**算法本体是 Dexonomy（RSS 2025），不训练**：撒点 → Adam 匹配模板接触点
→ MuJoCo 物理精修 → 力封闭 QP。

## 一条命令

```bash
python steps/step3_grasppose/run_take.py --take <重建take目录>
```

一次跑完这条 take 该抓的**全部** (物体 × 手)，产出 `<take>/grasp_pose_plan.json` 给 Step4。
新机器先跑一次 `bash steps/step3_grasppose/setup_workdir.sh`（建 `assets/object` 软链 +
`pip install -e`，两步都不能省——`dexrun` 是 console script，只设 `PYTHONPATH` 没用）。

## 七步链路

| 步 | 做什么 | 关键点 |
|---|---|---|
| ① 摆放 | `tools/upright_from_recon.py` | 取重建**开头 10 帧**位姿中位数 → 吸附最近的稳定静置姿态。不能用抓握窗：那段人已经把东西拿起来了（pour/17 实测瓶已离桌 12.7cm） |
| ② 粗筛 | `tools/coarse_filter_templates.py` | 只砍**几何上不可能**的（接触指数 ±1、模板隐含直径/物体直径比）。★不拿 VLM 的 `contact_depth`/`palm_contact` 做硬过滤 |
| ③ 查先验 | `tools/template_prior.py` | 按 `(形状｜物体接触带直径档｜空心)` 取历史表现排序，含 ε-greedy 探索位 |
| ④ 低 epoch 扫描 | `SCAN_EPOCH=3` | 逐个试，覆盖率达到该档位历史最高 ×0.85 就停 |
| ⑤ 精生成 | `GEN_EPOCH=10 GEN_PTS=4096` | 比上游默认 `50×1024×n_final10` 快 2.6 倍且质量更好 |
| ⑥ 评分排序 | `tools/rank_by_coverage.py` | `0.4×覆盖率@1cm + 0.3×高度匹配 + 0.3×离桌余量`；硬闸：虎口朝下 / 手指伸进空腔 / 离桌余量 |
| ⑦ 回写先验 | | 实测结果写回 `assets/template_prior.json`，用得越多越准 |

pour/17 瓶子端到端约 13 分钟。

## Step2 → Step3 接口

`run_take.py` 从 take 目录读这些，**只取数值**：

| 文件 | 取什么 |
|---|---|
| `contact/grasp_prompt.json` | `grasps[]` = 工作清单；`contact_region.height_pct_median` = 评分的参考接触高度；`rejected[]` 原样记进产物 |
| `confidence_complete.json` | `manifest_status` / `rotation_usable` / `position_grade` |
| `vlm_grasp.json` | `answer.<side>.object_shape`（先验档位）、`n_contact_fingers`（粗筛指数） |

**⚠ 绝不用产物里烘死的绝对路径。** `grasp_prompt.json` 的 `mesh` / `contact_cloud_npz`、
`confidence_complete.json` 的 `poseqa_root` 写的都是产出那台机器的路径（pour/17 实测是
`/home/yanghong/...`）。路径一律由传入的 take 目录重建。

**confidence 闸**
- `manifest_status != active` → 整条 take 跳过（被 `take_manifest.py` 择优淘汰或硬排除）
- `rotation_usable == false` → **不能拿重建朝向摆物体**，降级为概率最高的稳定静置姿态，
  产物标 `placement_source: stable_pose_only`，让 Step4 知道这个摆放没有视频依据
- `position_grade == "poor"` → 照跑，标 `low_confidence`

## 会立刻踩到的坑

1. **`dexrun` 必须在 PATH 里**。`take_grasp_pipeline.sh` 里那句
   `export PATH="$DEXO_ENV/bin:$PATH"` 不能删 —— 少了它，`|| true` 会吞掉
   `command not found`，表现是每个模板静默产出 0 个候选、`output/scan_*` 目录压根没被创建，
   查起来像"扫描目录凭空消失"。
2. **判物体空心用「凸分解块总体积 / 凸包体积」，不能用 `mesh.volume`**。SAM3D 出的网格常常
   不水密（实测瓶子 `is_watertight=False`），按体积比会把实心瓶子误判成空心，进而查错先验档位。
3. **先验的档位键只能由物体决定**。别用"模板隐含直径/物体直径"这个比值 —— 它每个模板各不
   相同，同一个物体会落进多个档位，查询时无从取舍。
4. **配额不能单独放大**。`n_final` 调大而采样池不动，会从被拒的候选里随机抓来凑数
   （实测 `5×8192×n_final100` 力封闭从 25 掉到 9）。要放大就三个一起放大。
5. **手模型用 `sharpa_wave_v2`**（官方 mujoco_menagerie 版，掌部 32 块凸分解）。旧版只有
   8 块粗凸包（体积 213.6 vs 57.0 cm³），会把大量候选误判成"手掌撞物体"：同条件杯子力封闭
   397/500 → 496/500。

## 目录

```
run_take.py            ★入口: Step2 take → 全部 (物体×手)
paths.py               RR_ROOT / STEP3_OBJ_ROOT / DEXO_ENV 出口
setup_workdir.sh       新机器初始化
tools/                 七步链路的实现, 核心 take_grasp_pipeline.sh
dexonomy/              Dexonomy 上游包(vendor) + 我们的改动
assets/hand/           SharpaWave 左右手 v1/v2 + 上游其余手型(LFS)
assets/template_prior.json    模板经验先验
assets/object          → 软链, 11G 不入库, 见 paths.STEP3_OBJ_ROOT
select_grasp_template.py      VLM×几何双证人选模板(现降级为给先验提供 shape/指数)
```

## 不在这里的

- **Isaac PhysX 验证**：还在 `Dexonomy/isaac/`，下一轮迁入。
- **BODex / cuRobo 路线**：已废弃。历史在 tag `archive/step3-bodex`
  （原 `Step3_GraspPose_Optimization` 分支）。
