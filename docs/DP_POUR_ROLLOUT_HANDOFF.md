# Pour 三视频 Base ckpt → DP 数据采集 交接 (2026-09-08)

给用我们的 RL ckpt 在 sim 里 rollout 采 DP 数据的同学。三条视频 (pour17 / pour25 / pour31) 各一条 Base 策略, eval10 (10 回合开扰动) 各 10/10 精确倒水。
分支/包里包含跑通 rollout 所需的**全部**代码、数据、先验、母带与 ckpt; 缺的只有 Isaac 栈本身 (见 §1)。

## 0. 三条 ckpt 与对应关系

| 视频 | run | ckpt (包内路径) | 母带 (POUR_REF_NPZ) | 旗 | eval10 G3 | eval512 G3 |
|---|---|---|---|---|---|---|
| pour17 | PFX_ablP0 | `exports/pour_3x3_ckpts_20260908/pour17/Base_PFX_ablP0/last.pth` | `tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz` | `$ARM_Base` | 10/10 | 0.996 |
| pour25 | PFX_P25s51 | `.../pour25/Base_PFX_P25s51/last.pth` | `tasks/Pour/25/A_Design/L2_Reference/pour25_reference_v5.npz` | `$ARM_Base` | 10/10 | 0.943 |
| pour31 | PFX_P31s51 | `.../pour31/Base_PFX_P31s51/last.pth` | `tasks/Pour/31/A_Design/L2_Reference/pour31_reference_v5.npz` | `$ARM_Base` | 10/10 | 0.971 |

每条 ckpt 同目录的 `world.json` 是它的出生世界指纹 (物理、旗、母带 md5、判据摘要)。**回放前脚本会按它核对, 不符直接拒跑** —— 这是有意的, 别绕过。
NH / NHNC 两条臂也在包里, 但 NH 在 pour17 是 0/10, 不要拿来采数据。

## 1. 机器准备 (一次)

- Isaac Sim 5.1 + IsaacLab_MagicSim + conda `isaac` 环境: 照 `docs/DEPLOY_NEW_MACHINE.md` §1。**clone 前先装 git-lfs** (§2A), 母带/先验/USD/纹理都是 LFS 对象。
- 环境变量 (每个 shell): `OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=<repo>`; 多卡共享机再加 `RL_ISAAC_NO_GUARD=1`;
  臂 IK 的 URDF: `VEGA_URDF=<你机器上的 vega_1p_sharpa.urdf>` (Denso/msc 都在 `~/data/vega_urdf/`; 不设会去找 lyh 本机绝对路径)。
- 世界一致性自检 (必做): `SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Pour/17/C_Wiring/smoke_zero.py --headless` 跑零动作,
  横幅里必须有 `experience file: ...headless.kit`、`难度覆写 物体: 质量=0.500kg 摩擦=1.00/1.00`、`指垫摩擦覆写 = 1`。
  ⚠ 我们发现 msc 8 卡机的 IsaacLab 版本让臂下垂标定差一个量级 (另一个世界), ckpt 在那里表现不同; 你的机器先用 `eval_pour.py` 跑 10 回合对一下 §3 的数字。

## 2. 旗与命令 (与训练/评测一字不差)

```bash
source exports/pour_3x3_ckpts_20260908/common/launch_flags.env    # 定义 COMMON / ARM_Base / ARM_NH / ARM_NHNC
PY=<conda isaac>/bin/python
# 评测 (eval10 协议: 10 env × 1 回合, t0 出生, 开扰动 ±3cm 瓶杯同移无 yaw, seed 2026)
env $COMMON $ARM_Base CUDA_VISIBLE_DEVICES=0 POUR_REF_NPZ=tasks/Pour/31/A_Design/L2_Reference/pour31_reference_v5.npz \
  $PY tasks/Pour/31/C_Wiring/eval_pour.py --checkpoint logs/PFX_P31s51/stage1_nn/last.pth --num_envs 10 --episodes 10 --seed 2026 --headless
# 带纹理录像 (1 env, 相机 1280x720)
env $COMMON $ARM_Base CUDA_VISIBLE_DEVICES=0 POUR_REF_NPZ=... $PY tasks/Pour/31/C_Wiring/record_pour.py --checkpoint ... --out out.mp4 --headless --enable_cameras
```
ckpt 目录约定: 脚本从 `--checkpoint` 的上两级找 `world.json`, 所以把包里的 `<run>/` 整体放成 `logs/<run>/stage1_nn/last.pth` + `logs/<run>/world.json`。
COMMON 里 `POUR_OBJ_JITTER=0.03` 就是采集/评测的扰动; 改种子 = 另一组扰动。

## 3. 用来对账的参考数字 (Denso 2080Ti, 2026-09-07)

eval10 (seed 2026, G3 精确倒水口径, 不要求放回): Base pour17 10/10, pour25 10/10, pour31 10/10; 放回九条全 0/10。
零动作/确定性回放里瓶子在手内的相对运动 (探针): 倒水达成时位移 1.1~1.6cm、长轴倾斜 4~19°; 倒完后 pour17 多数脱手, pour31 能走完时钟。
你的机器上如果 eval10 差出 2 个回合以上, 先怀疑世界 (§1 自检), 不要怀疑 ckpt。

## 4. rollout 骨架 (照 `record_pour.py`, 每步能读到什么)

```python
import world_fingerprint as WF; WF.restore_physics_env(WF.world_json_of(ckpt))   # 先还原物理再建环境
cfg = PE.build_cfg(num_envs=N); raw = PE.PourEnv(cfg)
apply_textures(raw, tex_dir="datasets/pour31/cache/textures", names=(("object","bottle","瓶"),("aux","cup","杯")))  # 真实 SAM3D 纹理
env = GymStyleEnvWrapper(raw, clip_actions=1.0); agent = PPO(...); agent.restore_test(ckpt); agent.set_eval()
WF.assert_match(raw, world_json, strict=True)          # 世界闸
obs = env.reset()
for t in range(T):
    mu = agent.model.act_inference({"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]})
    obs, rew, dones, infos = env.step(torch.clamp(mu, -1, 1))
    # 可读: raw.hand.data.joint_pos / joint_vel (全 DOF), raw.q_cmd 或 set_joint_position_target 的目标 (58 维=臂14+指44, 见 pour_env._pre_physics_step),
    #      raw.object / raw.aux 的 root_pos_w / root_quat_w (瓶/杯), raw._pads_f() 指垫力, raw.PB.g1/g2/g3 (接触/认证/倒水 门, 逐步取最大),
    #      raw.PB.k (母带时钟行), raw.row (回合行)。相机: record_pour.py 里的 /World/RecCam 定义方式。
```
入库判据 = `raw.PB.g3` 曾为 1 (G3 精确倒水: 倾角≥90° ∧ 壶口投影落进杯口盘 ∧ 连续 25 步); 不要求放回。
⚠ IsaacLab 在 step() 内对终止 env 当场自动重置 —— 终局量要逐步取最大/在 done 那一步之前读 (record_pour 有注释)。
⚠ 录完/采完不要调 `app.close()` (会挂死占显存), 直接 `os._exit(0)`。

## 5. sim 物体出生位置 (base 系, 母带第 0 行; 三条视频不同, 来自重建的 XY 刚体约定)

| 视频 | 杯 obj_0 | 瓶 obj_1 |
|---|---|---|
| pour17 | (-0.14, 0.17, 0.94) | (-0.13, 0.00, 0.96) |
| pour25 | (-0.30, -0.17, 0.95) | (-0.35, -0.37, 0.99) |
| pour31 | (-0.16, -0.02, 0.94) | (-0.15, -0.17, 0.96) |
桌面 z=0.87。改出生区 = 改任务几何 (approach 与交互母带都要重锚), 不是改个数字的事。

## 6. 本次交接包含的文件 (分支 `pour_dp_release_20260908`)

代码: `rl_rebuild/` · `tasks/pregrasp/{env.py,cfg.py,finger_residual_bound.json,priors/}` · `tasks/Pour/17/{C_Wiring,A_Design/L3_Learning,A_Design/L2_Reference}` · `tasks/Pour/25` · `tasks/Pour/31`
数据 (LFS): `datasets/pour17` (+`cache/textures`) · `datasets/pour25` · `datasets/pour31` (各含 `objects/*.usd`, `keyframes.json`, `scene_layout.json`, `cache/textures`)
ckpt: `exports/pour_3x3_ckpts_20260908/` (九条 last.pth + world.json + 母带 + 旗 + eval 日志 + 纹理录像; README/MANIFEST.md5 在内)
文档: 本文 · `docs/DEPLOY_NEW_MACHINE.md` · `docs/DP_AAGF_RUNBOOK.md` (上一轮 AAG-F 采集流程, 世界指纹对账那一步同样适用) · `docs/MANUAL.md`
