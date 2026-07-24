# Ego_Pipeline 整合规划：自包含的 Reconstruction + Retargeting

> **目标**：把 Reconstruct_and_Retarget 打造成一个**自包含、可独立运行、可随论文 release** 的数据处理项目。
> 从别人的项目里 vendor「我们需要的」代码与资产进来：Recon 来自 Jiakai 的 HV2RD
> (`HumanVideo2RobotData`)，Retarget 来自 `third_party/MagicDexMate`。
> `ego_pipeline` 自己能从 mp4 端到端跑出 MANO 轨迹 + 物体 Mesh/位姿，Retargeting 的输入直接来自 Reconstruction。
>
> **HV2RD 只作为与 Jiakai 的 git 交流桥梁**（拉他的更新 / 推我们对 Recon 的改动），不参与 Reconstruct_and_Retarget 的运行。
> 制定 2026-06-27。本文件是规划。

---

## 0. 原则

- **vendor 而非软链**：Reconstruct_and_Retarget 里是真实文件副本，纳入 Reconstruct_and_Retarget 自己的 git，无 symlink。
- **只留需要的文件**：第三方仓库里与本链路无关的东西不进来（测试样本、其它分支、各自的 venv 等）。
- **重资产分层**：代码 vendor；模型仓库用 pinned submodule；权重用 `weights/`(gitignore)+下载/拷贝脚本；
  HOI4D 数据（217G，已在 `Reconstruct_and_Retarget/Data/HOI4D`）永远走 `--dataset-root` 外部引用。
- **改动回流**：对 Recon 代码的任何修改，经 `sync/` 脚本回灌 HV2RD → 按 Jiakai 格式 push，避免搞乱上游。

---

## 1. 目标目录结构

```
Reconstruct_and_Retarget/ego_pipeline/
├── Reconstruction/                 # 自包含重建（vendor 自 HV2RD）
│   ├── recon_pipeline/             # 纯 pipeline 代码（1.1MB，sync 自 HV2RD）
│   ├── third_party/                # 6 个模型仓库：pinned git submodule（同 HV2RD 的 .gitmodules）
│   │     vipe/ sam2/ sam3/ hawor/ sam-3d-objects/ FoundationPose/
│   ├── weights/                    # gitignore；setup_weights.sh 拷贝/下载（见 §5）
│   ├── setup/                      # init_submodules.sh / build_exts.sh / setup_weights.sh
│   └── README.md                   # 与 HV2RD 的对应关系 + 环境/运行说明
│
├── Retargeting/                    # 自包含 retarget（vendor 自 third_party/MagicDexMate）
│   ├── magicdexmate/               # 核心包（sources/retarget/sinks/skeleton）
│   ├── sim/  scripts/  configs/    # 仿真/工具/retarget 配置
│   ├── assets/robots/hands/        # SharpaWave URDF+mesh（release 需要 → 入库）
│   └── README.md
│
├── bridge/                         # Reconstruct_and_Retarget 自己的胶水（Retarget 输入 ← Recon 输出）
│   ├── recon_to_replay.py          # world_fused.npz → replay_world.npz（hawor env）
│   └── run_ego2robot.py            # 端到端编排：Recon → bridge → Retarget
│
├── sync/                           # 与上游的 vendoring 同步（HV2RD 为 GitHub 桥）
│   ├── sync_recon.sh               # recon_pipeline ⇄ HV2RD（pull/push，白名单 rsync + diff 审阅）
│   └── sync_retarget.sh            # MagicDexMate → Retargeting（单向 vendor + diff 审阅）
│
├── _legacy_stages/                 # 旧 stage 实现归档（vipe/hawor/object_*，被 Reconstruction 取代）
└── INTEGRATION_PLAN.md
```

---

## 2. 数据流契约（核心，不变）

```
 第一视角 mp4
   │  Reconstruction/recon_pipeline/run_batch_queue.py（多 conda env 自动切换）
   ▼
 Reconstruction 输出 <vid>/world_fused.npz + object_mesh_scaled_final.obj
   │  bridge/recon_to_replay.py (hawor env：MANO 前向 + 时间轴对齐 + 四元数化)
   │  Retargeting/scripts/obj_to_usd.py (.venv-isaac：.obj → .usd)
   ▼
 <vid>/replay_world.npz + <vid>/object.usd
   │  Retargeting/sim/retarget_isaacsim.py (.venv-isaac)
   ▼
 SharpaWave 22-DOF qpos / Isaac 仿真回放 / 数据采集
```

### 2.1 `world_fused.npz`（frame=gravity_z_up_world，+Z 朝上）
`c2w`(300,4,4) · `object_ob_in_cam/in_world`(300,4,4) · `object_frame_indices`(300) ·
`hand_trans/rot`(2,600,3) · `hand_pose`(2,600,45) · `hand_betas`(2,600,10) · `hand_valid`(2,600)。
手 axis0 `0=left,1=right`；物体 300@15fps，手 600@30fps（`hawor_time=2*video_frame`）。

### 2.2 `replay_world.npz`（bridge 产出，retarget 消费）
`joints_left/right`(300,21,3 世界米制, **OpenPose/MediaPipe 21 序**) · `valid_left/right`(300) ·
`obj_pose`(300,7 `[x,y,z,qw,qx,qy,qz]` 世界) · 可选 `mano_verts_right`(300,778,3)/`mano_faces`(1538,3) ·
`frames`(300) · `fps`(=15)。

> 利好：HaWoR/MANO 21 序 == MagicDexMate 期望序；`retarget/frames.to_mano` 每帧重估坐标系、
> 对世界系/单位免疫 → 关节对上即可，**无需坐标对齐/重索引**。

---

## 3. 自包含运行的关键：环境与扩展绑定

### 3.1 各步骤环境与已部署模型

| 环境 | numpy/torch/py | 服务步骤 | 已部署模型/组件 |
|---|---|---|---|
| **cu128 + ViPE `.venv`** | numpy 2.2.6 / torch 2.9.0+cu128 / 独立 uv | vipe | ViPE（相机/深度/重力，原生 CUDA 扩展） |
| **HV2RD** | 1.26.4 / 2.11.0+cu128 / py3.11 | sam3_hands, sam2_object, label | SAM3(jiaka fork, `build_sam3_predictor`)、SAM2(`_C` cp311) |
| **hawor** | 1.26.4 / 2.11.0+cu128 / py3.10 | hawor, fuse | HaWoR(MANO/手部世界系)、joblib/trimesh |
| **biv2ap** | 1.26.4 / 2.11.0+cu128 / py3.10 | sam3d, sam3d_scale, fp_pose | **SAM3D/Fast-SAM3D**(`sam3d_objects`→Reconstruct_and_Retarget/third_party/Fast-SAM3D)、**MoGe** 1.0、**GeoCalib** 1.0、**FoundationPose**(mycpp 已编译)、pytorch3d 0.7.8 / kaolin 0.17 / nvdiffrast 0.4 / diff_gaussian_rasterization / gsplat / spconv-cu120 / xatlas / pymeshfix / utils3d / lietorch 0.3 / warp 1.14 |

### 3.2 能否合并成一个环境？

- **ViPE 不能并入**：numpy 2.x + torch 2.9，与其余 numpy 1.26 + torch 2.11 根本冲突 → 永远独立 uv venv。
- **SAM2/SAM3/HaWoR 理论可并入 biv2ap**（torch/numpy 一致），但**不无害**：HV2RD 是 py3.11 而 biv2ap 是 py3.10
  （sam2 `_C` 需为 cp310 重编、sam3 重装）；且 biv2ap 为 SAM3D 钉死了 timm/roma/scikit-image 等版本，
  与 HaWoR/SAM3 的需求可能冲突。**结论：维持多环境隔离**（3 conda + 1 venv），不合并——稳健、可复现、符合 release 规范。

### 3.3 扩展绑定

Reconstruction 要在 Reconstruct_and_Retarget 内独立跑，需让原生扩展/可编辑安装**绑定到 Reconstruct_and_Retarget 的 third_party 副本**
（当前它们绑在 HV2RD 路径上）：

| 组件 | 现状（绑 HV2RD） | 自包含做法 |
|---|---|---|
| sam2 `_C`（在 HV2RD conda env，editable→HV2RD/third_party/sam2） | 跑会 import HV2RD 的 sam2 | 在 BiV2AP/third_party/sam2 重新 `pip install -e . --no-build-isolation` |
| FoundationPose `mycpp`（HV2RD 路径编译） | .so 在 HV2RD | 在 BiV2AP 副本 `build_all_conda.sh` 重编 |
| ViPE uv `.venv`（HV2RD/third_party/vipe） | venv 在 HV2RD | 在 BiV2AP/third_party/vipe `uv sync`（CUDA_HOME=/usr/local/cuda） |
| conda envs（cu128/HV2RD/hawor/biv2ap） | 机器级、按名调用 | **复用，不重建**（与仓库路径无关）；只需重绑上面三个扩展 |

> conda 环境是机器级、靠名字 `conda run -n` 调用的，与代码所在仓库无关 → 直接复用，不必重建。
> 真正要重做的只有「3 个路径绑定的原生扩展」，由 `Reconstruction/setup/build_exts.sh` 一键完成。
> 权重/数据用 §5 的脚本指向已有副本，**不重复下载**。

---

## 4. 桥接器（唯一新写的核心代码）

`bridge/recon_to_replay.py`（hawor env）：
1. left/right 取 `hand_{trans,rot,pose,betas}` → MANO 前向 → 778 顶点 + 21 关节(OpenPose 序, 世界米制)。
   **复用** `Retargeting/scripts/hawor_to_joints.py` 的 MANO 逻辑（输入改成读 `world_fused.npz`）。
2. 手 600@30 → 300@15 对齐（`joints[2i]`，或有插值索引键时线性插值）；`valid[i]=hand_valid[side,2i]`。
3. `object_ob_in_world`(300,4,4) → `obj_pose`(300,7)（旋转矩阵→四元数 scalar-first）。已同在 z-up 世界，无需转坐标。
4. 写 `replay_world.npz`（§2.2）。网格→USD 用 `obj_to_usd.py`。

> 回归基准：`third_party/MagicDexMate/RetargetInput/HOI4D/hoi4d_recon_samples/<vid>/replay_world.npz`
> 是人工产出的样本，可逐字段比对验证新桥接器。

---

## 5. Input / Output / Data 落位（通用扫描 + 结构镜像）

**约定：`Data/` 只放输入**（不放可视化/测试 mp4），因此输入可安全地"自动扫全部 mp4"。

### 5.0 通用输入发现（取代 HOI4D 专用）

- 递归 `Data/**/*.mp4`；每个 mp4：`dataset` = 它在 `Data/` 下**第一层目录名**，
  `rel_path` = 相对 `Data/<dataset>/` 的完整层级路径。
- **`.egoignore`**（放在 `Data/` 根或各数据集根，gitignore 风格 glob）：排除不想处理的 mp4。
- 不再绑定 hoi4d；以后新增数据集只要丢进 `Data/<新名字>/...` 即被自动识别，无需改代码。

### 5.1 目录布局（输出镜像输入结构）

```
Reconstruct_and_Retarget/
├── Data/                                 # ★ Input：仅输入；自动扫 **/*.mp4（.egoignore 可排除）
│   ├── HOI4D/HOI4D_release/ZY.../align_rgb/image.mp4
│   └── <其它数据集>/.../*.mp4
└── Output/
    ├── ReconstructOutput/
    │   ├── interim/<dataset>/<flat_id>/     # 中间产物（gitignore；成功后自动删，见 §5.3）
    │   └── <dataset>/<嵌套 take 路径>/      # ★ Reconstruct：镜像来源结构
    │         world_fused.npz · object_mesh_scaled_final.obj
    └── RetargetOutput/
        └── <dataset>/<同一嵌套 take 路径>/  # ★ Retarget：与 recon 同结构（bridge 写入）
              replay_world.npz · object.usd                # run_retarget.sh 的输入
# 例：take id ZY20210800001__H1__C11__N07__S185__s02__T2
#   → Output/ReconstructOutput/hoi4d/ZY20210800001/H1/C11/N07/S185/s02/T2/
#   → Output/RetargetOutput/hoi4d/ZY20210800001/H1/C11/N07/S185/s02/T2/
# 嵌套由 RECON_FINAL_NESTED=1 开启（扁平 id A__B__C -> A/B/C；默认关，对其它消费者透明）
```

### 5.2 需要的代码改动（无害，可上游 HV2RD）

1. `_common/dataset.py`：加**通用 auto 模式**——`rglob("*.mp4")` + `.egoignore` 过滤；
   `VideoJob.rel_path` 保留完整层级；保留并不破坏现有 `--dataset hoi4d` 路径。
2. `_common/paths.py`：artifact 路径从扁平 `<dataset>/<video_id>` 改为**嵌套 `<dataset>/<rel_path>`**
   （`interim_step_dir` / `final`）——嵌套天然避免跨数据集重名（HOI4D 一堆 `image.mp4` 不撞）；
   并把 `INTERIM_ROOT`/`FINAL_ROOT` 改为环境变量可配（`RECON_INTERIM_ROOT`/`RECON_FINAL_ROOT`，
   默认仍 `REPO_ROOT/data/...`）→ 产物统一进 `Reconstruct_and_Retarget/Output/ego_pipeline/`，`Reconstruction/` 保持纯代码无 data 目录。

### 5.3 处理模型与自动清理（现状）

- recon `run_batch_queue.py` 是**并行批处理队列**（CPU 打标签 + 内存感知 GPU worker 槽），
  `--gpu-ids × --workers-per-gpu` 决定并发；每个 worker 槽内单视频按 8 步顺序跑完再取下一个
  （`--gpu-ids 0 --workers-per-gpu 1` ≈ 逐条）。
- **已自带中间产物清理**：`_cleanup_successful_video` 在最终校验通过后删 interim、只留小的
  `object_label_backup`（`--keep-interim` 才保留）。
- **缺口**：recon 止步于 `world_fused.npz`，**无 replay 脚本**。video→retarget 输入的最后一跳由
  `bridge/recon_to_replay.py` 补齐，并在 `run_ego2robot.py` 里把 bridge 的中间产物一并纳入清理。

### 5.4 权重（自包含但不重复下载）

`Reconstruction/setup/setup_weights.sh`：把本机已有权重**拷贝**（非软链）到位；release 时改为公开源下载。
来源（均已存在，见 [[local-ml-assets]]）：SAM2 large、SAM3(HF cache)、SAM3D(Reconstruct_and_Retarget do-as-i-do)、
FoundationPose(V2AP)、MANO(Reconstruct_and_Retarget HaWoR _DATA)。weights 体量大 → `weights/` 进 `.gitignore`，
仓库只留脚本 + `WEIGHTS.md`（来源/校验和），符合 release 规范。

---

## 6. 同步工作流（HV2RD = 唯一 GitHub 出入口）

```
拉 Jiakai：  cd HV2RD && git pull  →  sync/sync_recon.sh pull  （rsync→BiV2AP，审 diff，BiV2AP 提交）
推 BiV2AP：  改 Reconstruction/recon_pipeline  →  sync/sync_recon.sh push  （rsync→HV2RD，审 diff）
             →  cd HV2RD && git commit && git push   （按 Jiakai 格式上传）
```
rsync 白名单只含 `recon_pipeline/`，排除 `third_party/ data/ weights/ .venv *.pt`，保证两边只交换纯代码。
Retarget 侧 `sync_retarget.sh` 单向 vendor（MagicDexMate 是你自己的仓库，按需取 magicdexmate/sim/scripts/configs/assets）。

---

## 7. 实施路线图

- **P0 脚手架**：建 `ego_pipeline/{Reconstruction,Retargeting,bridge,sync}`；写 `sync_*.sh`；
  vendor recon_pipeline 代码 + MagicDexMate 需要的部分；init third_party submodules。
- **P1 自包含运行**：`setup/{init_submodules,build_exts,setup_weights}.sh` 跑通；
  在 Reconstruct_and_Retarget 内 dry-run + 单视频 smoke（vipe→…→fuse）出 `world_fused.npz`。
- **P2 桥接器**：`recon_to_replay.py`，用 `hoi4d_recon_samples` 回归；obj→usd。
- **P3 端到端**：`run_ego2robot.py` 串 Recon→bridge→Retarget；`retarget_isaacsim.py --mode render` 回放一段。
- **P4 收尾**：旧 stage 归档到 `_legacy_stages/`；补 README/WEIGHTS.md；跑 `test_seq.txt` 三段验收。

---

## 8. 已确认的决策

1. **执行位置**：Reconstruct_and_Retarget 要能**独立跑**重建（不只是消费产物）。→ §3 自包含环境绑定。
2. **Retargeting 落位**：vendor 进 `ego_pipeline/Retargeting`，输入来自 Reconstruction；
   release 需要的 assets（URDF/mesh）入库。
3. **落位方式**：vendor 真实副本（无软链），third_party 用 pinned submodule。
4. **HV2RD 角色**：仅 git 交流桥梁，不参与 Reconstruct_and_Retarget 运行。
5. **旧 ego_pipeline stages**：归档（被 Reconstruction 取代）。
```
```

> 待定（实测一份 `world_fused.npz` 后敲定）：手→物体时间轴用 2i 抽取还是插值（取决于完整 npz 是否带 hawor 插值索引键）。

---

## 9. 目标用法 & 调度器作为全项目基础设施

### 9.1 两条命令（已实现，在 `ego_pipeline/`）

```bash
cd /home/lyh/Project/Reconstruct_and_Retarget/ego_pipeline
./reconstruct.sh 10        # 命令1：遍历 Data 重建前 10 条 → Output/ego_pipeline/<dataset>/<vid>/
                           #   (开 HTTP 标注页；产物 world_fused.npz + object_mesh；中间产物自动清理)
./retarget.sh   10         # 命令2：把前 10 条 recon 产物转成仿真输入
                           #   → 每个 take 补 replay_world.npz + object.usd
# 常用：./reconstruct.sh 10 --skip-label --force   ./retarget.sh 10 --skip-usd
```

**实现状态**：
- ✅ `reconstruct.sh` → 调 HV2RD 引擎（third_party/扩展/conda 环境已部署），`RECON_FINAL_ROOT`/`RECON_INTERIM_ROOT`
  重定向到 `Reconstruct_and_Retarget/Output/ego_pipeline`；dry-run 验证发现/限流/输出路径正确。
- ✅ `bridge/recon_to_replay.py`（hawor 环境）`world_fused.npz → replay_world.npz`；
  对样本回归：obj_pose/valid/fps **完全一致**，joints 有效帧 ≤18mm（均值 6.5mm，重采样方法差异，retarget 无感）。
- ✅ `bridge/retarget.py` 批处理：扫 Output → 每条跑 bridge(+ `obj_to_usd.py` via `.venv-isaac`，`--skip-usd` 可关)。
- ⏳ 真实 10 条端到端尚未实跑（重建需人工标注 + 数 GPU·小时）；机制已验证，可随时开跑。
- 🔜 通用多数据集扫描（非 HOI4D）+ 嵌套 rel_path 输出：`Data/` 现仅 HOI4D，待加新数据集时补 per-dataset id 规则（§5.0/§5.2）。

### 9.2 把 Hybrid batch runner 抽成 Reconstruct_and_Retarget 共享调度器

**目标**：一套「硬件自适应的并行批处理引擎」贯通整个 Reconstruct_and_Retarget，任意机器自动发挥最大效率。

| 现状 | 说明 |
|---|---|
| ✅ 已有地基 | `GpuReservation`（线程安全、per-GPU 显存预算、per-step 显存成本）+ `_worker_slots` + work queue |
| ❌ 缺硬件探测 | 无 GPU 数/空闲显存/CPU 核 自动探测；`--gpu-ids` 手动传、显存预算写死 43000MB |

**改造方案**：
1. **硬件自动探测**：`pynvml`/`nvidia-smi` → GPU 数 + 各卡实时空闲显存；`os.cpu_count()` → CPU 核。
2. **自动并发度**：`workers_per_gpu = 空闲显存 ÷ per-step 显存成本`（成本表已有雏形），异构卡分别算。
3. **抽成共享库** `Reconstruct_and_Retarget/common/scheduler.py`：通用契约 = work-items + 每阶段 (env, GPU/CPU 成本)；
   recon / retarget / 未来管线统一 import。
4. **保持环境隔离**：调度器只管"在哪个 env、用多少卡/显存跑哪一步"，各步仍 `conda run -n <env>` / `.venv`。

**现实边界**：单机自适应很现实（补探测即可）；多节点/集群会上一个复杂度台阶，且 per-step 显存成本需按真实数据标定。
建议先做「抽共享库 + 单机硬件自动探测」（投入小收益大），多节点后置。
