# 接近路径规划 · 标准流程 (2026-09-05 定稿)

> 给"物体 clip + 每只手的 GraspPose",一条命令产出从对称站姿到 GraspPose 的
> **双手同时接近**参考。pour25 定稿并端到端验证(复现一致 0.0000 rad)。

## 一条命令

```bash
cd /home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/pregrasp/build_approach.py \
  --clip   <ClipName> \
  --prior_a <右/主手 GraspPose.npz> --yaw_a <度> \
  --prior_b <左/副手 GraspPose.npz> --yaw_b <度> \
  --out    <输出 Approach_xxx.npz>
```

输入只有三样:**物体(clip,含 primary+secondary 摆位与 USD)**、**每只手的 GraspPose
(prior npz + 绕物体轴 yaw)**。主手 = clip 的 `hand_side`(通常右)抓 primary 物体;
副手抓 secondary。thumbfix prior 两阶段通用(cuRobo 阶段锁指只用腕位)。

产出 `Approach_xxx.npz` = **165 帧级**:cuRobo 站姿→PreGrasp0(叠加成双手同时)+ 成形梯
PreGrasp0→Grasp(手指逐级合拢)。行格式 right_q/left_q(T,7)+right_f/left_f(T,22)+seg 表。

## 内置标准配方(一般不用改,均可 `--` 覆盖)

| 旗 | 值 | 为什么 |
|---|---|---|
| `--joint` | 0 | 先分别单臂规划再**叠加成同时**(joint=1 联合规划两物挤一侧常几何无解) |
| `--own_obstacle` | 1 | 接近段瓶杯**都常开障碍**(默认 V2AP 会排除本手目标 ⟹ 臂从物体里穿过) |
| `--sphere_buffer` | 0.008 | cuRobo 碰撞球实测比真实 mesh 瘦 0.4~0.8cm/link,充胖补欠近似 |
| `--sphere_buffer_links` | arm | **只充臂 link**(全局充胖会把 PreGrasp0 判死——手指必须贴物) |

只用 cuRobo 到 **PreGrasp0**,丢弃它的 grasp-reach(那段规划时世界为空、会穿物),
最后一段贴合交给成形梯(沿 prior 设计的接近方向下探)。依据见
`memory/pour25-approach-curobo-fixes.md`。

## 看效果

脚本结束会打印现成的 GUI 命令,或手动:

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/pregrasp/view_approach.py \
  --approach <Approach_xxx.npz> --clip <ClipName> \
  --prior_a <右手 GraspPose.npz> --yaw_a <度> [--max_frame N]
```

按 Enter 播放;`--max_frame N` 停在第 N 帧分级检查(cuRobo 段 0~80,成形各级梯
≈101/111/119/127/135/145,完整抓握 164)。

## 组成脚本(编排器自动串起来,一般不单独调)

- `build_approach.py` — ★标准入口(编排三阶段 + 自动杀 Isaac 关闭卡死进程)
- `view_curobo_plan.py` — cuRobo 双手规划(带 own_obstacle / sphere_buffer 旗)
- `curobo_plan_worker.py` — cuRobo 子进程(warp 与 Isaac 冲突,必须独立进程)
- `build_motion.py` — pregrasp 叠加 plan → 接成形梯 → Approach.npz
- `view_approach.py` — GUI 播放器

## 排查(哪段撞就看哪段)

用 `--max_frame` 逐级停;若某臂 link 穿物体:先确认是不是 `own_obstacle=1`(接近段本手
目标要留障碍);再看是不是 cuRobo 球欠近似(调 `--sphere_buffer` / `--sphere_buffer_links`,
臂穿→充臂,别充全局)。手指在 PreGrasp0/成形末尾贴物是**正常抓握接触**,不是 bug。
