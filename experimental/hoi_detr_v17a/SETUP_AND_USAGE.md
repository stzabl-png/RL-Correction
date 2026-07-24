# HOI-DETR / v17A 自动交互分割 —— 打包说明与集成指南

> **状态:实验性,尚未接入主 Pipeline。** 当前 `Reconstruct_and_Retarget` 主流程的
> **物体 Mask 和抓取 Phase 仍是人工标注**(`--web` 点标物体 + `tools/annotate_grasp_frames.py` 标接触)。
> 本目录是 Jiakai 的 v17A「纯视觉自动交互物体分割」代码副本,打包随仓库 release,
> **由接手同学负责后续集成**。集成前请先读完本文件。

---

## 0. 这是什么 / 不是什么

- **是**:一条纯视觉、无需文本提示的**交互物体分割**管线。HOI-DETR 检测「手 / firstobject / secondobject」框 + 手-物 link,SAM2 做零件级 mask 传播,输出带**固定实例 ID** 的逐帧 mask + interaction 关键帧。
- **不是**:它**不做左右手判定、不做 pose estimation、不判"真抓 vs 靠近"**。HOI-DETR 只有单个 `hand` 类,无 handedness;它的 hf-link 只是"手-物关联/靠近",手搁在旁边也会高置信触发(实测过假阳性)。
- **来源**:`github.com/jiaka1chen/HumanVideo2RobotData` 分支 **`STEP_3_V17`**(v17A),本副本对应 commit 见 `SOURCE_COMMIT.txt`(`be0791b`)。后续以该分支为准,拉更新回来覆盖 `experiments/`。

---

## 1. 需要的外部资产(都不入 git,自行下载)

| 资产 | 获取方式 |
|---|---|
| HOI-DETR 源码 | `git clone https://github.com/AhmadDarKhalil/HOI-DETR.git`,用 commit `1b367292f3833afd64a204bd4d9d84519541d035`(= 当前 main) |
| HOI-DETR checkpoint `epoch_5.pth` | **已公开、无需授权**:`wget -O checkpoints/epoch_5.pth https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth`(5,855,053,598 字节;SHA-256 `4708fd0ddc5c3d386bad67c31152de58676840b911f6d01219e0092a603277d3`,下完务必校验) |
| SAM2 源码 | 子模块 `third_party/sam2`(facebookresearch/sam2) |
| SAM2 checkpoint | `third_party/sam2/checkpoints/sam2.1_hiera_large.pt`,config `configs/sam2.1/sam2.1_hiera_l.yaml` |

---

## 2. 环境搭建(在 RTX 5090 / Blackwell sm_120 上实测跑通)

关键难点:HOI-DETR 依赖 **mmcv-full 1.7.2**(2023 老库),Blackwell 无预编译轮子,**必须从源码按 sm_120 编**。以下是踩平后的配方。

```bash
# 1) 建环境(克隆一个已有 torch2.11+cu128 的环境最省事;需 py3.10 + torch 支持 sm_120)
conda create -n codetr --clone <某个已有 torch2.11+cu128 环境>   # 或自建 py3.10 + torch>=2.7(cu128)
conda activate codetr

# 2) 【坑A】setuptools 必须降到 60.2.0(新版走 PEP517 隔离构建,mmcv setup.py 会因缺 pkg_resources 失败)
pip install --force-reinstall --no-deps --ignore-installed "setuptools==60.2.0" wheel
python -c "import setuptools; assert setuptools.__version__=='60.2.0'"   # 确认真降下来(conda 混装可能残留 egg-info)

# 3) 从源码编 mmcv 1.7.2(sm_120)
git clone --depth 1 --branch v1.7.2 https://github.com/open-mmlab/mmcv.git
cd mmcv
TORCH_CUDA_ARCH_LIST="12.0" MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS=8 python setup.py develop
python -c "from mmcv.ops import RoIAlign; import mmcv; print('mmcv', mmcv.__version__)"   # 应打印 1.7.2
cd ..

# 4) mmdet 2.25.3 = HOI-DETR 仓库自带 fork,用 legacy develop(PEP660 pip editable 装不上)
cd <HOI-DETR>
python setup.py develop --no-deps      # 其 __init__ 的 mmcv 上限已被上游改成 2.7.0,1.7.2 合规,无需再改断言
cd -

# 5) 【坑B】--no-deps 会漏 terminaltables(mmdet 运行时必需)
pip install terminaltables    # tensorboard 可不装

# 6) sam2(跳过可选 CUDA ext,避免 sm_120 编译出岔)
cd third_party/sam2 && SAM2_BUILD_CUDA=0 python setup.py develop --no-deps && cd -

# 7) 自检:三件套同环境可导入
python -c "from mmcv.ops import RoIAlign; from sam2.sam2_image_predictor import SAM2ImagePredictor; import mmdet; print('ok', mmdet.__version__)"
```

**【坑C】preflight「modified tracked files」**:runner 会要求 HOI-DETR checkout `git diff --quiet` 干净,但该 repo 把大量 `.pyc`/`egg-info` 纳入了版本管理,`import` 会重写它们导致变脏。修:
```bash
cd <HOI-DETR>
git ls-files '*.pyc' '*.egg-info/*' | xargs git update-index --skip-worktree
```
并且**跑的时候加** `PYTHONDONTWRITEBYTECODE=1`。

**代码位置**:v17A 用绝对 import `experiments.hoi_detr.*`,所以必须从**含 `experiments/` 的目录**运行,即本目录 `experimental/hoi_detr_v17a/`。

---

## 3. 运行(三步)

```bash
cd experimental/hoi_detr_v17a        # 这里有 experiments/hoi_detr/
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VIDEO=<绝对路径.mp4>  DATASET=egodex  VIDEO_ID=<唯一id>
export HOI_DETR_ROOT=<HOI-DETR>  HOI_CHECKPOINT=$HOI_DETR_ROOT/checkpoints/epoch_5.pth
export SAM2_ROOT=<repo>/third_party/sam2  SAM2_CHECKPOINT=$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt
export INSTANCE_OUTPUT=data/interim/$DATASET/$VIDEO_ID/instance_pipeline_v17a

# 步骤1:HOI-DETR 逐帧检测 -> detections.json(手/物框 + hf/fs link)
conda run -n codetr python -m experiments.hoi_detr.run_sequence \
  --dataset "$DATASET" --video-id "$VIDEO_ID" --video "$VIDEO" --gpu 0 \
  --hoi-detr-root "$HOI_DETR_ROOT" --checkpoint "$HOI_CHECKPOINT" --checkpoint-authorized \
  --source-revision 1b367292f3833afd64a204bd4d9d84519541d035 \
  --checkpoint-revision 85719ac7bf20b8b67e26206faddf0d9582052046 --frame-stride 1 --visualize

# 步骤2:episode 分割 + SAM2 传播 -> video_mask_sequence.json(固定实例 ID + 逐帧 mask)
conda run -n codetr python -m experiments.hoi_detr.run_instance_video_segmentation \
  --video "$VIDEO" --detections "data/interim/$DATASET/$VIDEO_ID/hoi_detr_probe/detections.json" \
  --output-dir "$INSTANCE_OUTPUT" --sam2-root "$SAM2_ROOT" --checkpoint "$SAM2_CHECKPOINT" \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml --gpu 0

# 步骤3(可选):诊断预览视频
conda run -n codetr python -m experiments.hoi_detr.render_mask_sequence_preview \
  --manifest "$INSTANCE_OUTPUT/video_mask_sequence/video_mask_sequence.json" \
  --video "$VIDEO" --output "$INSTANCE_OUTPUT/diagnostic_review.mp4" --watermark v17A
```

**关键输出**:
- `hoi_detr_probe/detections.json` —— 逐帧 `hand/firstobject/secondobject` 框 + hf/fs link(无 handedness)。
- `.../video_mask_sequence/video_mask_sequence.json` —— **下游正式消费**的固定 ID manifest:`frames[].objects[object_000N].metrics.centroid_xy / area_pixels / bbox_diagonal`、`raw_mask` 路径、`status`。实例 mask 只在 interaction 窗口内公开。
- SAM2 post-processing 那句 warning(连通域 CUDA ext)因 `SAM2_BUILD_CUDA=0` 跳过,官方说可忽略,不影响 mask。

**实例编号规则**:每 episode 重置、按接触先后顺序编号(第一个接触的物体=0001),多物体场景可能出现 0001/0002 顺序翻转,这是规则固有现象,不是 bug。

---

## 4. 集成到主 Pipeline 的建议(给接手同学)

主流程当前:重建时 `--web` 人工点标物体 → mask;`tools/annotate_grasp_frames.py` 人工标接触区间 → `grasp_annotation.json` → `attach_grasp_phase.py` 写进 `world_fused.npz` 的 `hand_phase`。两个可替换的接入点已留好:

1. **替换人工 Mask**:v17A 的 `video_mask_sequence.json` 逐帧实例 mask 可替换重建里的物体 mask 来源。注意它是 object-centric、多实例、只在交互窗口内有 mask;要对齐到重建的帧索引和物体身份。

2. **替换人工 Phase(左右手接触)**:接入点 `ego_pipeline/phase/auto.py`(消费 `contact_auto.json`,`load_phase(prefer=auto)` 会自动切),契约见该文件。**但 HOI-DETR 的 hf-link ≠ 真抓**,直接用会误判(手搁旁边也触发)。本仓已实现两条更可靠的路子,建议在其上集成:
   - `ego_pipeline/phase/detect.py` —— recon-only,2D 手 mask ∩ 物体 mask 邻接判接触,天然分左右手(不依赖 v17A)。
   - `ego_pipeline/phase/comotion_gate.py`(**Layer 1**)—— 读 v17A 步骤2 的 `video_mask_sequence.json` 实例 centroid + recon 左右手 mask,按**主动协同运动**(手实际搬动物体、物体作为刚体跟随:active_corr>0.5 且手移动≥8帧且物体跟随)判"真抓 vs 靠近/搁着",输出带**物体实例身份**的 `contact_auto.json`。实测能剔除 hf-link 的假阳性。
   - **残余**:全程静止的"静握不动"co-motion 分不出(会假阴)→ 规划中的 **Layer 2 = VLM 验证器**,只对静态歧义段调用。
   - 左右手信号来源:recon 侧 SAM3 `left_hand_0/right_hand_0` mask + HaWoR L+R MANO(HOI-DETR 自身不带)。

3. **左右手 × 物体实例**:v17A 给"何时、哪个物体实例被交互",recon 给"那只手是左是右",两者叠加 = 逐帧 × 左右手 × 接触物体实例。`comotion_gate.py` 已把这条链走通(在 basic_pick_place/0 上验证:右手抓 object_0002,左手搁杯被正确剔除)。

---

## 5. 依赖与边界速查

- Python 3.10 · PyTorch ≥2.7(cu128,需 sm_120)· mmcv-full 1.7.2(源码编)· mmdet 2.25.3(HOI-DETR fork)· SAM2。
- checkpoint/权重/视频/mask/运行输出**都不入 git**。
- HOI-DETR checkpoint 单卡显存峰值较高;5090(32GB)可跑单条。
- 本副本是快照;权威源是 `HumanVideo2RobotData` 分支 `STEP_3_V17`,更新从那边拉。
