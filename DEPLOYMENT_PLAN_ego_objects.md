# Reconstruct_and_Retarget — Ego Pipeline 物体分支部署计划 (Step3 + Step5)

> 目标：把 GitHub `HumanVideo2RobotData` 的 STEP_3(物体2D mask) + STEP_5(物体3D重建)
> 接进本地 `ego_pipeline`，让 **ViPE / HaWoR / SAM3D / FoundationPose** 全部在
> **`biv2ap`** 单环境跑通；不扰乱本机其它 15 个 conda 环境与已部署模型。
> 日期：2026-06-23

## 环境拓扑（定稿）

| 环境 | 角色 | 跑什么 |
|---|---|---|
| **`biv2ap`** (主) | torch2.11+cu128, sm_120 | ViPE、HaWoR、SAM3D、FoundationPose、SAM2 传播、全部 glue + 新 stage |
| **`egohos`** (新建·隔离·CPU) | torch1.11cpu + mmcv-full1.6 + mmseg0.24 | **仅** EgoHOS 检测器，子进程调用，不接触 biv2ap |

**不扰乱本地的硬规则**：只往 `biv2ap` 装少量纯 glue 包；`sam-3d-objects`/FoundationPose 只引用不重装；
新建的 `egohos` 完全隔离；`base` 与其它环境零改动；新代码只落在 `ego_pipeline/`。

## 本机现状（已实测 2026-06-23）

- biv2ap 已具备 SAM3D 全栈：spconv 2.3.6 + kaolin 0.17 + warp 1.14 + pytorch3d + nvdiffrast (全 ✅)
- ViPE `import vipe` ✅；FoundationPose 代码在 `DexImit-Open/third_party/FoundationPose-plus-plus/FoundationPose`
- SAM3D 两份：`third_party/sam-3d-objects`(submodule) + `DexImit-Open/ckpts/sam3d`
- SAM2 权重在 `DexImit-Open/ckpts/sam2/sam2.1_hiera_large.pt`，但 `import sam2` 缺失 → 需装 sam2 包
- EgoHOS 本机无（隔离环境新建）

## Phase 0 结果（已验证 2026-06-23）

biv2ap 内四个模型 **全部 import 通过**：

| 模型 | 状态 | biv2ap 内调用方式（已实测） |
|---|---|---|
| ViPE | ✅ | `import vipe`（`third_party/vipe` on path） |
| HaWoR | ✅ | chdir `third_party/hawor` 后 `from scripts.scripts_test_video...`（**无需独立 hawor env**） |
| SAM3D | ✅ | `sys.path += third_party/sam-3d-objects`，**必须设 `LIDRA_SKIP_INIT=1 SPCONV_ALGO=native`**，import `sam3d_objects.pipeline.*` |
| FoundationPose | ✅ | **用 V2AP 已验证副本** `/home/lyh/Project/V2AP/thirdparty/foundationpose` on path，`from estimater import ...`（flat） |

- sam2 已装进 biv2ap（源码 `DexImit-Open/third_party/Grounded-SAM-2`，`pip install -e . --no-deps --no-build-isolation` + `SAM2_BUILD_CUDA=0`），torch 保持 2.11.0+cu128 不变。
- **FP 副本决策**：本地 `DexImit-Open/.../FoundationPose-plus-plus` 用相对 import（`from .Utils import *`），不能 flat 加载；V2AP 副本用 flat import 且已在 biv2ap 验证 → ObjectPoseStage 指向 V2AP 副本（或把它拷进 Reconstruct_and_Retarget/third_party）。
- SAM3D 的 `sam3d_objects/__init__.py` 在无 `LIDRA_SKIP_INIT` 时会 `import sam3d_objects.init`（该 submodule 不存在）→ 故运行期必须带该环境变量。

## Phase 1 结果（已验证 2026-06-23）

- **SAM3D 实跑通过**：`demo.py` 在 biv2ap/5090 上端到端重建，~62s，输出 57MB `splat.ply`，final_iou 0.924（含自带 pose 对齐 + render-and-compare）。
  - 运行：`LIDRA_SKIP_INIT=1 SPCONV_ALGO=native python demo.py`（在 `third_party/sam-3d-objects/`）。
  - 权重已就位 `checkpoints/hf/pipeline.yaml`；`_target_: sam3d_objects` 已正确，**无需 V2AP 那种改名**。
  - 31GB 5090 单进程即可，**GitHub STEP_5 的两遍 VRAM hack 不需要**。
  - 正式入口用 `generate_mesh_sam3d.py --image_path <png> --masks_dir <dir>`（支持 `--with_hand_mesh --hand_mesh_path <hawor.obj>` 直接吃 HaWoR 手网格、`--json_path` 批量、postprocess/texture）——比 GitHub `run_sam3d.py` 更强，ObjectMeshStage 直接调它。
- **FoundationPose 运行依赖齐全**：transformations / fast_simplification / nvdiffrast / pytorch3d / warp 在 biv2ap 全 ok；FP 已在 V2AP/biv2ap 验证过（记忆：60 poses/7.4s）。FP 的实跑 smoke 并入 Phase 4 端到端（需 SAM3D mesh + mask + depth 作输入）。

## GitHub 复用清单（定稿 2026-06-23，来自 jiaka1chen/HumanVideo2RobotData）

仓库每个 step 是独立 branch：`STEP_3` / `STEP_5` / `main`。逐文件核对后的决策：

**✅ 复用（移植进 biv2ap，非隔离）**
| 文件 | branch | 移植到 |
|---|---|---|
| `step3_mask/egohos_sam2.py` | STEP_3 | `object_io.propagate_masks_sam2`（SAM2 fwd+bwd 传播，核心） |
| `step3_mask/egohos_to_masks.py` | STEP_3 | `object_io.egohos_label_to_instances`（纯 numpy 后处理，非隔离） |
| `step5_recon/relabel_lr_with_mano.py` | STEP_5 | `object_io.project_hand_mask` + `relabel_lr` |
| `step5_recon/select_frame.py` | STEP_5 | `object_io.pick_best_frame`（**改用 mean±band 版**，原 drop_pct 取最大是被废弃的偏过分割写法） |
| `step3_mask/eval_miou.py` | STEP_3 | Phase 4 验证时移植 |

**🔒 隔离（唯一需 egohos env）**：EgoHOS 推理本体（`third_party/egohos` + `EGOHOS_AUTODL_SETUP.md`，mmcv1.6/mmseg0.24/torch1.10），产出 obj1 label maps。
注意标签约定：raw obj1 是 4 类 {0,1,2,3}（remap {1:1,2:2,3:3}）；README 的 0-8 是 hands+obj 合并可视化（remap {3:1,4:2,5:3}）。`egohos_label_to_instances` 两者都支持。

**❌ 不复用**：`step5_recon/run_sam3d.py`（16GB 两遍 VRAM hack，5090 单进程用 `generate_mesh_sam3d.py` 取代）、`align_and_render.py`（ICP，用 FoundationPose 取代，仅可借 backproject/trim_outliers 做可视化）、`hands23_infer.py`（P2 备选检测器，仅 EgoHOS 不行时回退）、`io_standard.py`（磁盘 layout 读取器，已用内存版 `object_io`）、`depth_refine.py`/`mano_videos.py`/`compare_handseg.py`。

## 进度（2026-06-23 晚 — 崩溃恢复后）

- ✅ Phase 3 接线全部完成：`context.py`(objects 字段) + `pipeline.py`(with_objects 挂 3 stage) + `stages/__init__.py`(导出)。`make_default_pipeline` 现为 7 stage，biv2ap 内 import 全通过。
- ✅ `object_io.py` 补齐复用 helper：`dump_frames_dir` / `egohos_label_to_instances` / `propagate_masks_sam2` / `project_hand_mask` / `relabel_lr`；`pick_best_frame` 修为 mean±band。纯 python helper 全过 self-test。
- ✅ `object_mask_stage.py` 已建（frame dump → EgoHOS 子进程 seam → label 转换 → SAM2 传播 → union object_mask；MANO relabel 为可选 hook）。
- ✅ **隔离件 EgoHOS 已建好并验证（2026-06-23 晚）** —— 见下方 Phase 2 完成记录。
- ⏳ MANO L/R relabel：`_mano_cam_verts` 留 hook（需 MANO FK + world→cam）。object_mask/mesh/pose 只吃 union，不阻塞主链。

## Phase 2 完成（2026-06-23 晚 — EgoHOS 隔离环境 + ObjectMaskStage 端到端）

- **egohos conda env**（CPU 隔离）：python3.9 + torch1.11.0+cpu + torchvision0.12+cpu + mmcv-full1.6.0(cpu/torch1.11 预编译 wheel) + mmseg0.24.1(本地 `third_party/egohos/mmsegmentation` -e 装)。
  - **关键坑**：mmcv-full1.6 装完会把 numpy 顶到 2.0（ABI 冲突）→ 必须最后 `pip install "numpy<2"`（锁 1.26.4）。
- **EgoHOS 仓库**：clone 到 `third_party/egohos`；checkpoints 5.37GB（gdown work_dirs.zip）解到 `mmsegmentation/work_dirs/`，含 obj1 三阶段 ckpt：`seg_twohands_ccda` / `twohands_to_cb_ccda` / `twohands_cb_to_obj1_ccda`。
- **obj1 三阶段链**：predict_image 跑 twohands→cb→obj1；obj1 模型 in_channels=5（RGB+twohands+cb），mmseg 改过的 `encode_decode` 从 `dirname(dirname(img))/pred_twohands`、`/pred_cb` 读旁路通道 → 所以工作目录必须 `<base>/{images,pred_twohands,pred_cb,pred_obj1}/`。obj1 输出 raw 4 类 {0,1,2,3}（=EGOHOS_REMAP_RAW_OBJ1），实测 ✅。
- **wrapper**：`third_party/egohos/run_obj1_infer.py`（`--images <dir> --out <dir>`，CPU，3 阶段，**SyncBN→BN** `revert_sync_batchnorm` 修复 CPU 推理）。3 帧×3 阶段 ~10s。
- **端到端验证**：`tools/test_object_mask_hoi4d.py` 跑 C3 序列 20 帧 @540×960 —— EgoHOS 子进程(egohos CPU)种子 20/20 → SAM2 传播(biv2ap GPU) → dense object_mask 20/20 帧有前景，instance ids {1,2,3} 齐全；overlay 显示 mask 准确贴合被操作物体（毛巾）。**ObjectMaskStage 跨双环境跑通。**

## FoundationPosePP-ROS 接入（2026-06-24 — 替换 ObjectPoseStage 后端）

把用户的结合版物体位姿模型 **FoundationPosePP-ROS**（FoundationPose++ 2D tracker+6D Kalman ⊕ Isaac ROS TensorRT refine）接成 ObjectPoseStage 后端。见 [[v2ap-recon-team]]。

- 包复制到 `third_party/FoundationPosePP-ROS`（用户 branch 的源）。**它是 tracker 不做注册**，frame-0 init pose 用现有 in-process FP register 一帧得到（Q1 定）。
- **运行约束**：必须在 `isaac_ros_dev_container`（root）里跑；容器**只挂载 V2AP `isaac_ros_ws → /workspaces/isaac_ros-dev`**，Reconstruct_and_Retarget 不可见。故包 colcon-build 到 `isaac_ros_ws/pp_overlay`（隔离 overlay，避开主 workspace 的 isaac_ros_test 悬挂依赖），scene/masks/out staging 到 `isaac_ros_ws/biv2ap_pp/<tag>/`（容器可见），跑完 chown 回 host uid，poses 读回 Reconstruct_and_Retarget。
- `ObjectPoseStage(backend="pp_ros"|"local")`，pp_ros 为默认：① in-process FP 注册 reg 帧 → P0；② `build_pp_scene_dir`（depth=.npy 米、逐帧 mask、**偶数宽高**——奇数会让 gxf::VideoBuffer 崩）；③ 烘焙 metric mesh（scale.json ×scale 后导出，Isaac 节点按原样加载不缩放）；④ docker exec 启 TensorRT 节点 + pp_tracker；⑤ 读回 ob_in_cam。
- **踩坑**：(a) colcon 主 workspace 报 isaac_ros_test 缺失 → 用隔离 overlay 构建；(b) ros2 launch 输出全缓冲，timeout+pipe 丢日志 → 用 stdbuf + 写挂载文件轮询；(c) 奇数宽高 855 → 节点崩 0 refine → 强制偶数；(d) 容器 root 写的 out 文件 host 删不掉 → docker exec rm -rf 清理 + chown 回 1000。
- **验证**：Phase A 单测 Toycar 60 帧 59 refine ~4s(~15fps)；全链 C3（mask→mesh→pp_ros）20/20 refine，z 0.76→0.88m，3D box overlay 准确跟随物体。测试脚本 `tools/test_object_branch_full_hoi4d.py --pose-backend pp_ros`。

## Phase 4 全链验证（2026-06-24 — mask→mesh→pose 端到端跑通）

`tools/test_object_branch_full_hoi4d.py`：C3 序列 30 帧 @328×584，**全部由新 stage 现算**（不用预算 mask/mesh）：
- ObjectMask：30/30 帧有物体前景（EgoHOS egohos-CPU 子进程 → SAM2 biv2ap-GPU）。
- ObjectMesh：SAM3D 在 best frame 9 重建 `object.obj`；**深度反投影估尺度** d_real=0.391m / d_mesh=1.205 → scale.json ×0.3248（metric 缩放生效）。
- ObjectPose：FP 在 reg_idx=9（=重建帧）注册并前向 track 21 帧，z∈[0.76,0.88]m（metric 合理）；track_vis 显示 3D box 准确套住物体并跟随。
- **两处链路 bug 在此修复**：① ObjectPoseStage 改为在 `mesh_frame_idx` 注册并前向 track（原来把 best-frame mask 错放到 frame0）；② ObjectMeshStage 新增 `_write_scale_json`（backproject mask+depth→真实直径，否则 SAM3D 单位归一 mesh 让 FP 非 metric）。helper `backproject_mask`/`pointcloud_diameter` 在 object_io。
- 结论：**整个 object branch 在 biv2ap 端到端跑通并产 metric 位姿。** 三 stage + 隔离件全部就绪。

## Phase 4 部分验证（2026-06-23 晚 — 预算 mask 跑通 ObjectPoseStage）

用 V2AP 预算资产绕过 EgoHOS/SAM3D，直测 biv2ap 的 **ObjectPoseStage (FoundationPose)**：
- 资产来源：V2AP `obj_recon_input/egocentric/<seq>/0.png`(手动 register mask) + `obj_meshes/hoi4d/<seq>/{mesh.ply,scale.json}` + `egocentric_depth/hoi4d/<seq>/{depth.npz,K.npy}` + HOI4D extracted_images(MegaSAM 子采样到 60 帧)。全部归一化到 depth 分辨率(328×584)后塞进 EgoContext。
- 测试脚本：`tools/test_object_pose_hoi4d.py`（可复用，`--seq` 选序列）。
- 结果：3/3 序列在 biv2ap/5090 端到端跑通，各出 60 帧 ob_in_cam + track_vis；位姿 metric 合理（z≈0.7–1.0m）。frame0 注册准确落在标注物体上（toy car / towel）。
- 已知现象：长 egocentric 序列后段 FP track 漂移——这是 FP 固有限制，与 V2AP `batch_obj_pose_ego.run_fp` 同源同行为，非集成缺陷。ObjectPoseStage 与该脚本逐行对齐。
- 结论：**FoundationPose 分支在 biv2ap 集成成功。** SAM3D(ObjectMeshStage) 已在 Phase 1 单独验证(demo iou 0.924)；待 EgoHOS 隔离件就位后跑 mask→mesh→pose 全链 Phase 4。

## 最终架构

```
ViPE → HaWoR → CoordUnify → Smooth
   └─ ObjectMaskStage   Step3: EgoHOS(子进程,egohos env) → SAM2传播(biv2ap) → MANO relabel左右手
   └─ ObjectMeshStage   Step5: select_frame + run_sam3d(两遍VRAM hack, biv2ap)
   └─ ObjectPoseStage   FoundationPose(biv2ap) —— 替代 GitHub 的 align_and_render ICP
```
物体位姿用 FoundationPose（canonical）；GitHub `align_and_render.py` 的 ICP 不移植，只借其
深度反投影点云做 FP 可视化校验。所有产物写入 `ctx.objects`。

## 分阶段

### Phase 0 — 基线快照（零改动）
- biv2ap 内逐一 import 验证 ViPE / HaWoR / SAM3D / FoundationPose。
- 确认 `HaWoRStage` 在 biv2ap 真跑通（本机有独立 `hawor` env，需确认 biv2ap 里 HaWoR 依赖齐全）。
- 装 sam2 进 biv2ap（`pip install -e` DexImit 内 sam2 源码，不动 ckpt）。

### Phase 1 — SAM3D + FoundationPose 固化进 biv2ap
- SAM3D：套用 v2ap-sam3d-5090 的 4 个 fix（`SPCONV_ALGO=native`、hydra `_target_` 改名、
  kaolin fallback 签名、权重路径 `checkpoints/hf/pipeline.yaml`）。biv2ap 已有 spconv/kaolin，预计直跑。
- FoundationPose：`pip install transformations fast_simplification` 进 biv2ap 让 estimater 可 import + mesh 抽稀。
- 各跑 1 个 HOI4D 序列 smoke 通过。

### Phase 2 — 物体 MASK（EgoHOS 隔离方案）
1. 新建隔离 `egohos` env（CPU，torch1.11+mmcv1.6+mmseg0.24），clone EgoHOS 到 `third_party/egohos` 并下权重
   （参考 GitHub STEP_3 `EGOHOS_AUTODL_SETUP.md`）。
2. 移植 STEP_3 的 `egohos_to_masks.py` + `egohos_sam2.py`。
3. `ObjectMaskStage`（壳在 biv2ap）：
   - 把 ctx 帧落到临时 standard-layout 目录；
   - 子进程 `conda run -n egohos python egohos_to_masks.py …` → seed masks；
   - 回 biv2ap：SAM2 传播补全整段 → 全帧物体 mask；
   - `relabel_lr_with_mano.py` 用 ctx 的 MANO 重判左右手 → instance ids；
   - 写 `ctx.objects["masks"]`。

### Phase 3 — 接进 ego_pipeline
- `io_standard` 薄适配器：EgoContext(K.npy/depth.npz/帧) ↔ Step3/5 的 `sam3d_inputs/` 视图，不落第二份数据。
- `context.py` 把 `objects` 字段从 `# Future` 改成正式结构（masks/mesh/pose）。
- `make_default_pipeline` 末尾挂三个新 stage。

### Phase 4 — HOI4D 验证（本机有 Data/HOI4D）
- mask：`eval_miou.py` 复测 mIoU。
- mesh+pose：1 序列端到端 smoke，输出 ctx.objects + FP 的 ob_in_cam/track_vis。
