# Pour 三视频 × 三训练臂 ckpt 打包 (2026-09-08)

九条 run = pour17 / pour25 / pour31 三条视频 × Base / NH / NHNC 三条训练臂, 全部 seed 51 (pour17 三条为 PFX_abl* 系列), 20M 步, 1024 env, 冠军配方 + 硬物理 (0.5kg / μ1 / 指垫 μ1, 有意覆写 G-A 规矩)。每条目录: `last.pth` (评测用的 ckpt) · `world.json` (出生世界指纹: 旗子/物理/参考 md5, 回放评测必须按它还原) · `<run>_tex.mp4` (带纹理确定性回放, 1 env t0) · `eval10_s2026.log` (三视频×三臂 10 回合开扰动测评) · `eval512/` (512 回合 t0 评测, 有则附)。

## 三条臂

| 臂 | 含义 | 旗子 |
|---|---|---|
| Base | 人手参考 + 物体参考 (HYB), 接触起始黄窗 | `POUR_VARIANT=HYB POUR_TIER_FLOOR_START=auto` |
| NH | 撤人手参考, 只留物体参考 (OBJ), 黄窗 | `POUR_VARIANT=OBJ POUR_TIER_FLOOR_START=auto` |
| NHNC | NH 再拍平置信度 (无分档), 臂偏差/皮筋常量 | `POUR_VARIANT=OBJ POUR_CONF_FLAT=1 POUR_DEV_ARM_FLAT=0.08 POUR_LEASH_FLAT=0.05` |

共用环境变量见 `common/launch_flags.env`。

## 对应关系与结果

eval10 = `eval_pour.py --num_envs 10 --episodes 10 --seed 2026`, 开扰动 `POUR_OBJ_JITTER=0.03`; **用户裁定的判据 = G3 精确倒水, 不要求放回**。★成功 = G3倒水 ∧ 放回 是旧主判据, 表里一并列出。eval512 = 512 回合 t0 口径 (同开扰动)。

| 视频 | 臂 | run | 步数 | eval10 G3倒水 | eval10 放回 | eval10 ★成功 | eval512 G3倒水 | eval512 ★成功 | 参考母带 (md5) |
|---|---|---|---|---|---|---|---|---|---|
| pour17 | Base | `PFX_ablP0` | 20.0M | 1.0000 | 0.0000 | 0.0000 | - | - | `pour17_reference_v3.npz` (1ee52f77) |
| pour17 | NH | `PFX_ablNH` | 20.0M | 0.0000 | 0.0000 | 0.0000 | - | - | `pour17_reference_v3.npz` (1ee52f77) |
| pour17 | NHNC | `PFX_ablNHNC` | 20.0M | 0.3000 | 0.0000 | 0.0000 | - | - | `pour17_reference_v3.npz` (1ee52f77) |
| pour25 | Base | `PFX_P25s51` | 20.0M | 1.0000 | 0.0000 | 0.0000 | 0.9434 | 0.0000 | `pour25_reference_v5.npz` (956c488c) |
| pour25 | NH | `PFX_P25ablNH_s51` | 20.0M | 0.0000 | 0.0000 | 0.0000 | 0.2832 | 0.0039 | `pour25_reference_v5.npz` (956c488c) |
| pour25 | NHNC | `PFX_P25ablNHNC_s51` | 20.0M | 0.1000 | 0.0000 | 0.0000 | 0.0176 | 0.0000 | `pour25_reference_v5.npz` (956c488c) |
| pour31 | Base | `PFX_P31s51` | 20.0M | 1.0000 | 0.0000 | 0.0000 | 0.9707 | 0.0449 | `pour31_reference_v5.npz` (8df00660) |
| pour31 | NH | `PFX_P31ablNH_s51` | 20.0M | 1.0000 | 0.0000 | 0.0000 | 0.9434 | 0.0000 | `pour31_reference_v5.npz` (8df00660) |
| pour31 | NHNC | `PFX_P31ablNHNC_s51` | 20.0M | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `pour31_reference_v5.npz` (8df00660) |

合计 (eval10, G3 倒水口径): **Base 30/30 · NH 10/30 · NHNC 4/30**; 放回九条全 0/10。pour17 三条的 512 回合评测在 Denso 训练机上做过 (G3倒水 Base 0.996 / NH 0.002 / NHNC 0.488, 记录于 2026-09-07 的消融汇总), 日志留在 Denso 未随包。

## 怎么用

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python   # 或 Denso: ~/miniconda3/envs/isaac/bin/python
source common/launch_flags.env
# 评测 (例: pour31 Base). 把本包的 <run>/ 目录放回 logs/<run>/ (含 stage1_nn/last.pth 与 world.json), 脚本按 world.json 还原物理与旗子:
env $COMMON $ARM_Base CUDA_VISIBLE_DEVICES=0 POUR_REF_NPZ=tasks/Pour/31/A_Design/L2_Reference/pour31_reference_v5.npz \
  $PY tasks/Pour/31/C_Wiring/eval_pour.py --checkpoint logs/PFX_P31s51/stage1_nn/last.pth --num_envs 10 --episodes 10 --seed 2026 --headless
# 录像 (纹理, 1 env):
env $COMMON $ARM_Base CUDA_VISIBLE_DEVICES=0 POUR_REF_NPZ=... $PY tasks/Pour/31/C_Wiring/record_pour.py --checkpoint logs/PFX_P31s51/stage1_nn/last.pth --out out.mp4 --headless --enable_cameras
```

目录约定: 仓里 `logs/<run>/stage1_nn/last.pth` + `logs/<run>/world.json`; 本包每条目录整体拷回即可 (`mkdir -p logs/<run>/stage1_nn && cp last.pth logs/<run>/stage1_nn/ && cp world.json logs/<run>/`)。

## 依赖 (不在包内)

- 代码: `tasks/Pour/{17,25,31}/C_Wiring/` (pour_env / eval_pour / record_pour / world_fingerprint), `rl_rebuild/`, `tasks/pregrasp/`。
- 数据: `datasets/pour17`, `datasets/pour25`, `datasets/pour31` (USD 自包含 + `cache/textures` SAM3D 纹理), `tasks/pregrasp/priors/{aag_pour17_*, Pour25_*, Pour31_*_thumbfix, anchor_T_*}`。
- 参考母带 `common/pour{17,25,31}_reference_v{3,5}.npz` 已附 (world.json 里钉了 md5, 不一致会被 `world_fingerprint.assert_match` 拒绝)。
- PPO 网络配置 `common/ppo_pour.yaml` (三任务同一份, md5 0870403b)。

## 记录
- 训练: Denso 4×2080Ti, 2026-09-05~07, 发射脚本 `denso_ablation.sh` / `denso_p25_ablation.sh` / `denso_p31.sh` / `denso_p31_ablation.sh`。
- 纹理录像批: `denso_record9.sh` 2026-09-07 23:12~23:50 UTC。
- 打包: 2026-09-08 本地 `exports/pour_3x3_ckpts_20260908/`。
