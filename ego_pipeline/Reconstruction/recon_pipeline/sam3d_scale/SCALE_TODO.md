# 尺度估计的硬性缺口 —— 等杜邦(7mare)接入

> ⚠ 这不是可选优化。现行 `_estimate_scale_given_orientation` 是**单帧单标量 PCA 主轴比**,
> 实测系统性偏大: ketchup 1.60×、laptop 3.07×、扫把 ~2×。尺度错 = 物体大小错 =
> 接触点位置错 = 抓取姿态错，整条下游跟着错。

## 现状（本步骤在做什么）

`sam3d_scale/run_sequence.py`
- 输入: `sam3d` 的原始网格 + 参考帧的 mask + ViPE 深度(`--depth-scale` 转米制)
- 做法: 在**单帧**上把 mask 反投影成点云, 与网格做 PCA 主轴比 → 一个标量缩放
- 输出: `objects/<oid>/object_mesh_scaled_final.obj`(fp_pose/fuse 只读这一个)
  + `scale_fpalign_scale_metadata.json`(scale_stage1 / scale_final / warnings)

## 杜邦已有的东西（`agent/v17a-scale-develop` 分支 `scale_develop/`）

| 文件 | 作用 |
|---|---|
| `scale_extent_v1.py` | **第 1 层**: 去朝向化的多帧跨度比 |
| `scale_fuse_qwen.py` | **最终层**: 三路共识融合(跨度 × 手长锚点 × 类别常识), 一次 Qwen 调用 |
| `scale_sanity_qwen.py` | 手锚点护栏(对现行几何尺度做常识校验) |
| `compare_scale_gt.py` | 三方对比: 原版 vs 护栏修正 vs ARCTIC GT |
| `filter_tracks.py` / `qwen_final_arbiter.py` | track 级背景过滤 + 终审主体仲裁 |

**核心洞察(他的)**: **不信朝向**。扫把案例证明"朝向错 = 尺度错" —— 观测长度 18.4cm
是对的, 朝向歪导致算出 40cm。所以第 1 层直接用 mask 反投影点云的 PCA 鲁棒跨度
(P2–P98)当最长边, 不经过 SAM3D/FP 的朝向。

**实测**(他的三条):

| 视频 | 原版 | 融合后 | 判定 |
|---|---|---|---|
| ketchup | 1.60× | **0.96×** | consensus |
| laptop | 3.07× | **1.15×** | corrected_by_prior |
| 扫把 | ~2× | **≈1×** | consensus |

值得学的一个细节: laptop 被 Qwen 认成"木质砧板"(还是 high confidence), 但尺寸反而准
—— **手长锚点不依赖类别认对**, 这是双锚点设计的容错。

## 接入时的约定（与主线已收敛的部分）

1. **保留原几何路径当对照**。两者不一致时把差异**记进产物**, 不要静默替换 ——
   否则以后没人知道哪个数是怎么来的。建议字段:
   `scale_geometric` / `scale_fused` / `scale_verdict`(consensus|corrected_by_prior|…)
2. **env**: 你用的 `hoidetr`/`sam3` 已废, 主线是 `codetr`(SAM3+SAM2+HOI-DETR) 与
   `biv2ap`(SAM3D/FP/本步骤)。`HV2RD` env 已于 2026-08-14 删除。
   权威在 `run_batch_queue.STEP_ENVS`。
3. **Qwen 调用要收口**。目前全仓有**五处**独立客户端(你的终审仲裁/尺度融合/尺度护栏
   + `ego_pipeline/bin/vlm_transparency_gate.py` + `RL_Correction/steps/step3_grasppose/
   vlm_grasp_schema.py`)。收口时**必须带上**这四条硬约束(GraspPose 侧标定出来的):
   - `enable_thinking: False` —— 开着从 1.7s/42 tokens 变成 24.6s/600 tokens 还被截断
   - 结构化输出用 **xgrammar 后端 + `disable_any_whitespace`**, 两个键要一起给;
     否则模型会无限吐换行烧光 max_tokens(实测 2423 字符里 2116 个换行), 且 `auto` 后端会拒
   - schema 内容不动(v3.2): `role` 枚举叫 `support`; 问"几根手指接触"不问手指身份;
     **不注入任务名**(防作弊式评测)
   - prompt 模板**任何变化**都要拿 screw27 重验左右手(真值: 右手扶瓶、左手拧盖) ——
     实测同一条视频 temperature=0, 多加一个无关问题就 10/10 稳定答错
4. **选帧影响本步**: 尺度参考帧 = `frame_plan.sam3d_frame`(与 sam3d 同帧), 由
   `select_frame` 步骤给。选帧质量直接决定这里的上限。

## 验收

接入后至少要在这几条上对拍(都有对照):
- 你的三条: ketchup(GT 21.3cm) / laptop / 扫把
- 主线的: `egodex_auto/pour/17` —— 现状 object_0 conf_pos 52/conf_rot 5.5/44 帧被证伪,
  object_1 83/50/0 帧; 尺度改好后 object_0 应有改善
