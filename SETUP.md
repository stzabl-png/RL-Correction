# 环境与依赖准备(clone 后必读)

本仓库**只提交第一方代码**(约 42MB)。数据、模型权重、外部大型依赖都不入 git,需按本文准备。

---

## 1. 路径:不需要改代码

所有路径统一走 **`ego_pipeline/repo_paths.py`**(shell 用 `repo_paths.sh`):

- **`RR_ROOT`(本仓根)从文件位置自动推导** —— clone 到任何机器、任何目录都能直接跑,**无需配置**。
- 外部依赖保留开发机默认值,都可用**同名环境变量**覆盖:

```bash
export HV2RD_ROOT=/your/path/HumanVideo2RobotData     # 重建 recon_pipeline 上游
export V2AP_ROOT=/your/path/V2AP                       # FoundationPose / isaac_ros_ws
export EGODEX_ROOT=/your/path/egodex                   # EgoDex 数据集
export A2G_ROOT=/your/path/Affordance2Grasp            # EgoDex 原始数据 / hawor _DATA
export HAWOR_PYTHON=/your/conda/envs/hawor/bin/python
```

排查当前解析结果:
```bash
python ego_pipeline/repo_paths.py     # 打印所有根路径,并标注 [不存在]
```

---

## 2. `third_party/`(不入 git,必须自行准备)

`.gitignore` 排除了 `third_party/`(本机约 88GB)。主流程实际依赖的组件:

| 组件 | 用途 | 来源 |
|---|---|---|
| **MagicDexMate** | ⭐ retarget + Isaac Sim 回放(SharpaWave 手) | 内部仓库,向作者索取 |
| **hawor** | 手部 MANO 重建 | `github.com/ThunderVVV/HaWoR` |
| **vipe** | 相机位姿 / 深度 | `github.com/nv-tlabs/vipe` |
| **sam-3d-objects** | 物体单视图 mesh(SAM3D) | `github.com/facebookresearch/sam-3d-objects` |
| **FoundationPose / FoundationPosePP-ROS** | 物体 6D pose 跟踪 | `github.com/NVlabs/FoundationPose` |
| **sam2 / Fast-SAM3D** | 物体 mask 传播 | `github.com/facebookresearch/sam2` |
| megasam / da3 / video_depth / SpaTrackerV2 | 深度/轨迹对比实验(非必需) | 各自上游 |
| dex-retargeting / BODex_upstream / Articulation_Bodex | 抓取合成相关(见 Step3 分支) | 各自上游 |

> `ego_pipeline/Reconstruction/setup/init_submodules.sh` 可辅助拉取(路径已走 `repo_paths.sh`,
> 用 `HV2RD_ROOT` 指向你的 HumanVideo2RobotData)。

**注意**:`ego_pipeline/Retargeting/assets/` 下已 **vendored** 一份 SharpaWave 手资产(urdf/usd/meshes),
retarget 用这份即可,不依赖 `third_party/MagicDexMate/assets`。

---

## 3. 数据与产物

- **`Data/`、`Output/` 不入 git**。数据集获取见 `Step1_DataInput` 分支(HuggingFace `UCBProject/RL_Correction`)。
- 产物目录:`Output/ReconstructOutput/<dataset>/<嵌套路径>/`、`Output/RetargetOutput/...`,
  可用 `RR_DATA_ROOT` / `RR_OUTPUT_ROOT` 改到别处。

> ⚠️ 历史遗留:本仓 2026-07 由 `Bi-V2AP` 改名为 `Reconstruct_and_Retarget`。
> **改名前生成的旧产物**,其 json/npz 元数据里仍烘着旧的 `/home/lyh/Project/Bi-V2AP/...` 绝对路径
> (二进制 npz 无法批量替换)。请只用命令行显式传入的 take 目录,不要信产物内部的绝对路径。

---

## 4. 运行

见 **`USAGE.md`**(重建 → retarget → Isaac Sim 回放 三步全流程)。

抓取接触阶段(grasp phase)当前为**人工标注**:
`tools/annotate_grasp_frames.py` 标区间 → `tools/attach_grasp_phase.py` 写进 npz。
自动化方案(`ego_pipeline/phase/comotion_gate.py`、`experimental/hoi_detr_v17a/`)已打包但**尚未接入**。

---

## 5. 第三方资产授权提示

`ego_pipeline/Retargeting/assets/robots/hands/sharpa_wave/` 下的 **SharpaWave 机械手 USD/STL/URDF 来自 MagicDexMate**,
属第三方硬件资产,随附 `LICENSE.txt` / `NOTICE.txt` / `ATTRIBUTION.md`。
**对外分发或添加协作者前,请确认其授权范围。**
