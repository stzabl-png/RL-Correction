# 交接：选帧器 + 尺度估计 接入主线（给杜邦 / 7mare）

> 分支 `Step2_NoisyRecon`，写于 2026-08-14。
> 你的工作在 `agent/v17a-scale-develop`（最后提交 08-12 00:36，比主线落后 ~80 个提交）。
> **主线已经把位子和接线都留好了，你只需要把算法搬进来。**

---

## 0. 一句话

主线现在是 **13 步**，其中 `select_frame` 是**空占位**、`sam3d_scale` 用的还是老的单帧 PCA。
这两处是**硬性缺口**（不是可选优化）：选帧决定重建/尺度参考帧的质量上限，尺度错则物体大小
错、接触点位置错、抓取姿态错，整条下游跟着错。

---

## 1. 当前全链（`184d40c`）

```
【前置】reconstruct.sh 循环内，队列之外
  P1 video_preflight   砍掉开头连续暗帧(曝光预热)，硬链接不转码       hawor
  P2 HOI-DETR 检测     --hoi-only 可单独跑并出评审视频               codetr
  P3 实例发现 + SAM2 传播 → label_prompt.json                        codetr
  P4 材质门(逐实例问 VLM)  ★透明 → 整条 exit 3 终止                  hawor
  P5 写 frame_plan.json：fp_register_frame = 交互开始帧 + 10          —

【队列】13 步
   1 vipe            相机位姿 + 深度                                  cu128
   2 sam3_hands      SAM3 手部分割                                    codetr
   3 sam2_object     SAM2 物体 mask 全片传播                          codetr
   4 select_frame    ★★ 你的位子(现为空占位)                          codetr
   5 hawor           单目手部重建(EgoDex 路线用 ARKit 替代)           hawor
   6 vlm_retrieval   分件判定 → needs_retrieval                       hawor
   7 retrieval       命中 → 替 8/9 写完成标记并跳过                    hawor
   8 sam3d           物体三维重建，读 frame_plan.sam3d_frame          biv2ap
   9 sam3d_scale     尺度估计，同帧 ★★ 你的第二处                      biv2ap
  10 fp_pose         FoundationPose，读 frame_plan.fp_register_frame  biv2ap
  11 fuse            融合到统一世界系                                 hawor
  12 confidence      轨迹可信度打分                                   hawor
  13 contact         接触点提取(测量式)                               hawor
```

权威在 `run_batch_queue.py` 的 `AUTO_STEPS` / `STEP_SCRIPTS` / `STEP_ENVS`，别凭记忆。

---

## 2. B：选帧器 —— 你只需要写一个数字

### 契约

往 `frame_plan.json` 的 `objects.<oid>.sam3d_frame` 写一个帧号，**别的都不用改**。

```
文件：<interim>/<dataset>/<video_id>/sam2_object/frame_plan.json
schema：_common/frame_plan.py 头注（frame_plan_v1）
```

消费方**已经接好线**：

| 消费者 | 位置 | 用途 |
|---|---|---|
| `sam3d/run_sequence.py` | :558 | 重建帧 |
| `sam3d_scale/run_sequence.py` | :333 | 尺度参考帧（**必须与 sam3d 同帧**）|

`null` 时两步都回退到标注 prompt 帧；你写的那帧 mask 为空时也回退并打日志——
**计划永远只是建议，你写错不会让管线崩，只会退回旧行为。**

### ★ 不要碰 `fp_register_frame`

用户 2026-08-14 明确：FoundationPose 注册帧固定为**交互开始帧 + 10**，`auto_label` 已
自动算好写入，不走选帧器。三个"帧"各司其职：

| 帧 | 由谁定 | 要什么样的帧 |
|---|---|---|
| 重建帧 / 尺度参考帧 | **你的选帧器** | 看得最清楚、遮挡最少 |
| FP 注册帧 | 交互帧 + 10（已实现） | 物体已被拿稳、姿态代表全片 |

### 落点

`ego_pipeline/Reconstruction/recon_pipeline/select_frame/run_sequence.py`
（现为占位，头注里有完整接入指引）。它排在 `sam2_object` 之后、`hawor` 之前，
纯 CPU + 一次 SAM2 图像模式推理，不占大显存。

### 你已有的东西

`scale_develop/select_frame_v2.py`（279 行，两段式）。三视频验证：
ketchup f514→f162、phone f234→f278、scissors f514→f446，全部避开抓握重遮挡帧。

---

## 3. C：尺度估计

详见 `recon_pipeline/sam3d_scale/SCALE_TODO.md`。要点：

- 现行 `_estimate_scale_given_orientation` 是**单帧单标量 PCA 主轴比**，系统性偏大
  （你实测 ketchup 1.60× / laptop 3.07× / 扫把 ~2×）
- 你的三路共识融合实测 0.96× / 1.15× / ≈1×
- 核心洞察"**不信朝向**"（扫把：观测长度 18.4cm 是对的，朝向歪导致算出 40cm）
- **约定：保留原几何路径当对照**，两者不一致时把差异记进产物，不要静默替换 ——
  否则以后没人知道哪个数是怎么来的。建议字段
  `scale_geometric` / `scale_fused` / `scale_verdict`

---

## 4. 你必须改的两处环境差异

### 4.1 conda env 已收敛

| 你用的 | 主线现在用 |
|---|---|
| `hoidetr` | **`codetr`**（SAM3 + SAM2 + HOI-DETR 都在里面） |
| `sam3` | **`codetr`** |
| （HV2RD） | **已于 2026-08-14 删除**（本地 + UCB 各回收 12GB） |

完整分工（权威在 `run_batch_queue.STEP_ENVS`）：

```
cu128    vipe
codetr   sam3_hands / sam2_object / select_frame / HOI-DETR v17A / label.sh
biv2ap   sam3d / sam3d_scale / fp_pose
hawor    hawor / vlm_retrieval / retrieval / fuse / confidence / contact
```

`codetr` 里我们补装了 `sam3`（仓内 `third_party/sam3` 的 editable，`--no-deps`）与
`ftfy==6.1.1`，装前装后 torch/mmcv/transformers/numpy 版本逐项核对未变。

### 4.2 输入路径

- 用 `_common.paths.interim_step_dir(dataset, video_id, "sam2_object")` 取
  `label_prompt.json` / `frame_plan.json`
- v17A 的 mask manifest 在
  `experimental/hoi_detr_v17a/data/interim/<dataset>/<video_id>/instance_pipeline_v17a/`
- ⚠ **多物体必须读视频级 `video_mask_sequence.json`**，不是 episode 级
  （仓库 `CLAUDE.md` 第 6 条；喂错会让多物体静默塌成单物体）

---

## 5. D：Qwen 调用收口的四条硬约束

全仓目前有**五处**独立 VLM 客户端：你的终审仲裁 / 尺度融合 / 尺度护栏 +
`ego_pipeline/bin/vlm_transparency_gate.py` + `RL_Correction/steps/step3_grasppose/
vlm_grasp_schema.py`。收口时**必须带上**这四条（GraspPose 侧踩坑标定出来的）：

1. **`enable_thinking: False` 必须保留** —— 开着从 1.7s/42 tokens 变成 24.6s/600 tokens
   还被截断
2. **结构化输出用 xgrammar 后端 + `disable_any_whitespace`，两个键要一起给** ——
   否则模型会无限吐换行烧光 max_tokens（实测 2423 字符里 2116 个换行），
   且 `auto` 后端会直接拒
3. **schema 内容不动（v3.2）**：`role` 枚举必须叫 `support`；问"几根手指接触"不问
   手指身份；**不注入任务名**（防作弊式评测）
4. **prompt 模板任何变化都要拿 screw27 重验左右手**（真值：右手扶瓶、左手拧盖）——
   实测同一条视频 temperature=0，多加一个无关问题就 10/10 稳定答错

VLM 服务在 UCB 8 卡机 GPU7；本地跑 `./tools/vlm_tunnel.sh` 开隧道。
端点用 `VLM_API_BASE` 或 `QWEN_BASE_URL`（等价）。

---

## 6. 主线这两天改了什么（可能影响你）

| 改动 | 影响 |
|---|---|
| `vlm_gate` → **`vlm_retrieval`** | 步骤改名；**VLM 不可达 = 本条终止并报错**，不再跳过继续 |
| `depth_scale` 硬闸 `[0.01, 10]` | 超出即终止本条（clip 21/22 实测算出 208/156，ViPE 深度塌了） |
| 透明规则 = **strict** | 用户 2026-08-13 裁定：**透明材质一律不用于重建与训练**，不看置信度、不看装没装东西。旧的 v2（只滤"空透明"）**已作废** |
| `contact` 换 `extract_v2` | 一条 take 2~8 秒（旧的 8 分钟/手）；旧对齐器保留为 `contact_align`，不在默认 STEPS |
| `frame_plan.json` | 新增契约文件，就是给你的接口 |
| `--record` | 已删（无头录像的 snap_cam 位姿设不上去，产的是空场景） |

---

## 7. 接入后的验收

至少要在这几条上对拍（都有对照数据）：

| take | 现状 | 期望 |
|---|---|---|
| 你的 ketchup（GT 21.3cm）/ laptop / 扫把 | 1.60× / 3.07× / ~2× | 接近 1× |
| `egodex_auto/pour/17` object_0（杯） | conf_pos 52 / conf_rot 5.5 / **44 帧被证伪**；物体体检 **0.290（可疑）** | 应有改善 |
| `egodex_auto/pour/17` object_1（瓶） | 83 / 50 / 0 帧；体检 0.556 | 不应变差 |

`pour/17` 的 object_0 是很好的靶子：**两套互不相关的判据**（confidence 与新加的物体
mask 体检）同时指出它的位姿不可信，而它很可能就是选帧选到了重遮挡帧。

### 一条硬性验收纪律

**改完在两种网格上各验一次**：CAD（十万面级）和 SAM3D 重建（**七十万面级**）。
2026-08-14 一晚上有两次翻车都是"小网格上好好的、大网格上 OOM 或拖死"
（`mesh.contains` 在 13.5 万面上 2.9s→254.4s，在 68.8 万面上直接被 OOM 杀掉）。

---

## 8. 协作规矩

1. **文件归属**：`ego_pipeline/*` 归主线；`Dexonomy/tools/*`、`tools/arctic_eval/*`、
   `RL_Correction/steps/step3_grasppose/*` 归 GraspPose 侧。要改对方的先说一声
2. **改共享文件用精确替换，不要整文件读出→写回** —— 2026-08-14 因此发生过一次
   静默覆盖，同一件事留下两份实现、其中一份成了死代码（白建一次体素表，
   screw/0 22.8s→7.0s）
3. **提交前 `git status` 看清楚工作区里有没有别人的东西**，别一把 `git add` 扫进去
4. **先 rebase 到 `Step2_NoisyRecon` 再动手** —— 你那条分支已落后 ~80 个提交。
   或者只把 `scale_develop/` 这个独立目录 cherry-pick 过来（你只动了 v17A 的 2 个
   文件共 33 行，冲突面很小）

---

## 9. 一句提醒

你那个"**不信朝向**"的判断（扫把：观测长度对、朝向歪导致尺度错 2 倍）和主线这两天
反复撞到的是同一类问题——**单目下沿视线方向的量不可观测，而所有 2D 体检都会照常通过**。
主线在 `pour/17` 上踩过一次：手离物体 810mm，其中 **99% 在深度方向**，面内只有 98mm，
于是手部体检、物体体检、投影 IoU 全部正常，唯独结果是错的。

所以接尺度时，凡是"沿视线方向"的量都要额外找第二个证人（你的手长锚点就是这个作用）。
