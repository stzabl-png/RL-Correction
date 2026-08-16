# Data Engine —— 分步结构

把"重建出来不可用"的人手视频数据,修正为仿真验证过的可用交互轨迹。

```
video ─▶ Step1 感知 ─▶ Step2 重建 ─▶ Step3 抓取 ─▶ Step4 RL ─▶ Step5 判定 ─▶ D_high-quality
```

| 目录 | 职责 | 输入 → 输出 | 状态 |
|---|---|---|---|
| `step1_perception/` | 输入与感知 | video → 物体 mask + 双手 mask + 逐帧接触(分左右手) + 选定重建帧 | **整理中** |
| `step2_reconstruction/` | 三维重建 | 上述 + video → 深度/相机 → 物体 mesh → 人手轨迹 → 物体位姿 → 统一坐标系 | 待迁入(现在 `Reconstruct_and_Retarget`) |
| `step3_grasppose/` | 抓取合成与优化 | mesh(重建 or retrieval) → GraspPose + PhysX 验证 | **生成+优化已迁入**(2026-08-15); PhysX 验证栈仍在 `Dexonomy/isaac/` |
| `step4_rl/` | RL 残差修正 | 重建数据 + prior → 修正后轨迹 | 待迁入(现在本仓 `rl_rebuild/` + `tasks/`) |
| `step5_success_tracker/` | 成功判定 | RL 输出 → 训练是否正确 / 动作是否合理 → `D_verified` → 去重 | **尚未设计** |

## 为什么要这个结构

原来 5 个 Step 是 GitHub 上**互不相干的平行分支**,各配一个本地目录
(`Reconstruct_and_Retarget` / `HumanVideo2RobotData` / `ocir-grasp-synthesis` /
`Dexonomy` / `RL_Correction`)。后果:`git diff Step2 Step4` 毫无意义、分支之间无法合并、
跨 step 引用只能靠一堆环境变量(`RR_ROOT` / `OCIR_ROOT` / `HV2RD_ROOT` …)、
推一次代码要切五个目录。

目标形态是**单仓 + 目录分步**:一次 push,跨 step 引用变成仓内相对路径,
同事 clone 一次拿到全流程。

## 迁移约定

- **逐 step 迁入,迁一个验证一个**。未迁入的 step 保持上游仓可用,不制造半损状态。
- **代码入仓,重资产不入**:`third_party/`、模型权重、视频、重建产物一律走环境变量,
  沿用 `rl_rebuild/correction/paths.py` 的写法(每个 step 自带 `paths.py`)。
- **迁入期间上游副本不动**,等该 step 验证通过再删上游副本、改为引用本仓。
- conda 环境**不因合仓而合并**:8 个环境照旧,每步仍 `conda run -n <env>`。

> HV2RD → R&R 的合并**在 git 上已经完成**(`Step2_NoisyRecon` 分支的
> `ego_pipeline/Reconstruction/recon_pipeline/` 就是它),只有本机工作区还是分开的。
