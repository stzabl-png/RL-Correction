# 选帧器 + 尺度估计接入记录(基于 d4ca855, 2026-08-14, 杜邦)

> 对应 handoff: `HANDOFF_SELECTFRAME_SCALE_2026-08-14.md`。
> 主提交 `3cb7625`(B/C 两个硬性缺口), 另有本文第 5 节的三处兼容小修。
> 验证: ketchup(与 agent 分支逐项一致) + watercup(全新视频端到端)。

## 1. 接了什么

| 缺口 | 实现 | 文件 |
|---|---|---|
| B 选帧(原空占位) | 三层选帧器: track 过滤(手物连接/位移/帧数) → 几何粗筛+SAM2 像素遮挡精排 → Qwen 终审+主体仲裁 | `recon_pipeline/select_frame/run_sequence.py` + `selector_core.py`(新) |
| C 尺度(原单帧 PCA) | 三路共识融合: L_extent(多帧点云跨度, **不信朝向**) × L_hand(手长锚点) × L_typical(类别常识) | `recon_pipeline/sam3d_scale/scale_fusion.py`(新) + `run_sequence.py` 挂点 |

**契约完全按 handoff**:
- 选帧只写 `frame_plan.objects.<oid>.sam3d_frame`(不碰 `fp_register_frame`);
  消费方 sam3d/:558、sam3d_scale/:333 的既有接线未动。
- 尺度保留几何路径当对照: metadata 里 `scale_geometric` / `scale_fused` /
  `scale_verdict` 三字段, 几何 mesh 存 `object_mesh_scaled_geometric.obj`;
  融合成立时 `object_mesh_scaled_final.obj` 改写为融合尺度(下游 fp_pose/fuse 无感知)。
- v17A track id → 管线 oid: `v17a_import.json` → label_prompt provenance → prompt 帧
  mask IoU, 三级映射。
- 选帧的 `report.json`(ranked 干净帧)被 sam3d_scale 的多帧跨度复用(同一套质量判据)。

**尺度口径**: 一律"最长边"。点云侧 = PCA 第一主轴 P2–P98 鲁棒跨度;
mesh 侧 = AABB 最长单边; Qwen 锚点 prompt 问的也是最长跨度。**不是**包围盒对角线。

## 2. 运行方式: 零变化

13 步队列(`run_batch_queue` / `reconstruct.sh`)原样跑, `select_frame` 与
`sam3d_scale` 自动生效。手动单步:

```bash
# 选帧(codetr env; v17A 产物默认自动探测, 也可 --v17a-dir 指定)
python recon_pipeline/select_frame/run_sequence.py \
    --dataset <ds> --video-id <vid> --video <mp4> --gpu 0
# 尺度(biv2ap env; 融合自动跑; 显存 <48GB 的机器可加 --skip-fp-diagnostic)
python recon_pipeline/sam3d_scale/run_sequence.py \
    --dataset <ds> --video-id <vid> --video <mp4> --gpu 0 --depth-scale 1.0
```

前提: `DASHSCOPE_API_KEY`(或 `QWEN_BASE_URL` 指向自部署 vLLM + `QWEN_LOCAL_VLLM=1`),
模型可用 `QWEN_MODEL` 覆盖(当前验证于 qwen3.8-max)。

## 3. 降级路径(设计为"永远不崩")

| 缺什么 | 行为 |
|---|---|
| v17A manifest/detections | `sam3d_frame=null`, 下游回退 prompt 帧(=旧行为) |
| SAM2 | 遮挡精排退回手框版(v2.0) |
| VLM 不可达 | 选帧用几何 top-1(`geometric_fallback`); 尺度保持几何版, `scale_verdict` 记原因 |
| 计划帧 mask 为空 | 消费方回退 prompt 帧并打日志(同事已有机制) |

首次远程跑完看一眼 `scale_verdict` 与 `sam3d_source` 即可确认没有静默降级。
(未验证点: `biv2ap` env 是否有 `openai` 包 —— 没有则尺度融合降级为纯几何。)

## 4. 验证结果

**ketchup(s01_ketchup_grab_01, GT 21.3cm)** — 与 agent 分支逐项一致:
- 选帧: 几何 top-1 f162 → Qwen 否决(遮挡/mask 混手指) → 比选改判 **f591**;
  桌面 track(3 帧)被 `insufficient_track` 零成本拦截
- 尺度: 几何 34.5cm(1.62×) → 融合 **20.5cm(0.96×, consensus)**

**watercup(左杯, 全新视频)** — 全链端到端 + 锚点纠偏路径实证:
- 选帧: top-1 f132 被否决 → **f19**; sam3d/sam3d_scale 日志确认 `参考帧 3 → 19`
- 尺度: 几何 9.1cm 与点云跨度 9.6cm 互洽但均偏小(vipe 深度整体低估 ~2.3×,
  和 2026-08-05 扫把 8.3cm 同款病), 两个深度无关锚点互洽(22.1/21.9cm)
  → `corrected_by_prior`, **22.0cm**(此值本质来自常识锚点, 待实物量杯高检验)
- 产物位置: `data/interim/local/watercup/`(frame_plan / select_frame 报告与审计 /
  sam3d_scale metadata 与双 mesh); 上游 v17A 产物在 `scale_develop/runs/watercup/`
- 最终 mesh AABB = 13.9 × **22.0** × 13.7 cm(几何对照 5.7 × 9.1 × 5.7, 等比 ×2.425)

### 单标量尺度的语义边界(watercup 暴露, 重要)

尺度融合校准的是**一个标量**(等比缩放): 只有最长边(此例=杯高)被三路信号直接约束,
其余维度的比例**完全继承 SAM3D mesh 的形状**。watercup 直径 13.8cm 即
"SAM3D 认为高:径=1.6:1" 推出, 未被独立校准, 目测偏粗(实物约 2:1 以上)。
若下游对次轴尺寸敏感(抓取宽度!), 后续可做**各向异性校准**: extent 计算里
`spans_m` 三轴数据已经在算(`scale_extent` 的 per-frame spans), 缺的只是
"次轴与 mesh 次轴比对 + 径向修正"的规则, 但那会破坏"单标量不动形状"的现约定,
需要先和主线对齐。

## 5. 顺手修的三处兼容(未含在 3cb7625)

1. `experimental/hoi_detr_v17a/experiments/hoi_detr/run_sequence.py` 两处
   `unlink(missing_ok=True)` → py3.7 兼容写法(本地 hoidetr env; codetr 不受影响)
2. `recon_kailang/v17_mask_adapter/import_v17a_masks.py` schema 检查接受视频级
   `persistent_video_mask_sequence_v1`(多物体必用视频级, CLAUDE.md 第 6 条;
   与 agent 分支同款补丁)
3. `sam3d_scale --skip-fp-diagnostic`(在 3cb7625 内): FP register 1080p 需 ~47GB,
   其产物 `foundationpose_ref_pose.txt` 全仓无消费者, 显存紧张的机器可跳

## 6. 踩坑记录(重要)

- **prompt 消融事故**: 移植融合 prompt 时删了 JSON 字段的行内注释, Qwen 立刻把
  `hand_lengths`(应为"几个手长")按厘米填, 锚点变 3.4m。已恢复验证版原文并在
  `scale_fusion.py` 头部立牌。印证 handoff 第 5 节"prompt 任何变化都要重验"。
- 本地验证 env: 本机无 codetr/biv2ap, 用 sam3/foundationpose 等价跑通;
  代码不绑 env 名, 远程按 `STEP_ENVS` 走。

## 7. 剩余验收(handoff 第 7 节, 需远程机)

- laptop / 扫把对拍; `egodex_auto/pour/17` object_0 靶子(conf 52/5.5, 44 帧证伪)
- CAD(十万面)与 SAM3D(七十万面)两种网格各验一次
- Qwen 五处客户端收口(D 节)未动 —— 本次只保证新增调用走统一 `qwen_client`
  且 `enable_thinking=False`
