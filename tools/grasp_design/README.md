# grasp_design — 手工设计一条能抓起来的开环抓取轨迹

2026-07-28 那次会话在 `/tmp` scratchpad 里做的一整套离线设计脚本 + 中间产物,
原目录 `/tmp/claude-1000/-home-lyh/ac318f9c-.../scratchpad` 会被系统清掉, 故搬到此处持久化。
脚本里原先写死的 scratchpad 路径已统一改成本目录绝对路径。

目的: 在 RL 修正之前, 先用纯几何/IK 手工造出一条**开环就能抓起物体**的参考轨迹,
作为 RL correction 的 sanity baseline (物体真的被抓离桌面 >5cm)。

## 跑法

所有脚本都 `sys.path.insert(0, "/home/lyh/Project/RL_Correction")`, 用 MagicSim 的 venv:

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
cd /home/lyh/Project/RL_Correction

# 纯 numpy 离线阶段 (不开 Isaac)
$PY tools/grasp_design/facts.py            # 物体尺寸/位置、手指长度、交互段位移
$PY tools/grasp_design/sweep_depth.py      # 合拢深度扫描 (太深戳桌 / 太浅夹不住)
$PY tools/grasp_design/design_fingertip6.py  # 指尖环抓构型 v3 (变量=右臂7关节+开/合指角) -> grip_cfg_v6.npz
$PY tools/grasp_design/contact_check.py    # 胶垫到真实 mesh 表面的严格接触判定
$PY tools/grasp_design/plan_curobo.py      # cuRobo 规划 pick-transport-place 手臂轨迹 -> arm_traj.npz
$PY tools/grasp_design/export_traj.py      # 合成手臂+手指完整轨迹, 自研 FK 独立核验 -> full_traj.npz

# Isaac 回放验证 (GUI, 实时, 带跟随物体的标记球, 循环 5 遍)
SHARPA_WANDB=0 PYTHONPATH=. $PY tools/grasp_design/replay_zero.py --realtime --mark --loop 5
# 无头快跑
SHARPA_WANDB=0 PYTHONPATH=. $PY tools/grasp_design/replay_zero.py --headless
```

`replay_zero.py` 走的是**绝对关节轨迹**直灌 `env.hand.set_joint_position_target`,
绕开 `_pre_physics_step` 的参考/残差机制, 只驱动右臂 7 + 右手 22 关节。
用 `GRIP_NPZ=grip_cfg_v5.npz` 环境变量换抓取构型。

## 文件

| 文件 | 说明 |
|---|---|
| `facts.py` | 四组事实数据: 物体尺寸/位置、指长与合拢指尖相对腕位置、交互段手物位移 |
| `design_traj.py` | 最早的分段参考轨迹 (approach/descend/close/lift) + 离线可行性核验 → `traj_P.npy` |
| `sweep_depth.py` | 合拢深度扫描 |
| `_fastmod.py` | 指尖环抓求解的公共模块 (预编译链快速 FK + least_squares 残差) |
| `design_fingertip.py`…`3.py` | v1/v2: 变量=腕位姿, 解出来手臂够不到 (位置差 10.4cm, 3 关节顶限位) → `grip_cfg.npz` |
| `design_fingertip4.py` / `5.py` | 中间版本 → `grip_cfg_v4.npz` / `grip_cfg_v5.npz` |
| `design_fingertip6.py` | **v3 当前版**: 变量改成右臂关节角, 腕位姿由 FK 生成, 天然可达 → `grip_cfg_v6.npz` |
| `contact_check.py` | 胶垫到真实 mesh 顶点/表面距离 (圆环是环面, 圆柱近似会高估接触) |
| `plan_curobo.py` | cuRobo v2 规划整条手臂轨迹; 注意坐标系 `p_curobo = p_env + [0.5,0,0]` |
| `export_traj.py` | 拼手臂+手指, 用自研 numpy FK 独立复核 cuRobo 结果 |
| `replay_zero.py` | Isaac 零动作开环回放, 判据: 离桌 >5cm 的步数 |
| `*.log` | 各阶段当时的运行输出 (`fp2`…`fp6` = design_fingertip 各版, `replay.log`, `curobo.log`) |
| `*.npz` / `*.npy` | 中间产物, 见上表箭头; `replay_log.npz` = 回放时的物体/手轨迹 |

注: `design_fingertip6.py` 写的是 `grip_cfg_v6.npz`, 但那次会话跑到一半 (`fp6.log` 只有 51 字节),
本目录里没有 v6 产物; `replay_zero.py` 默认读 `grip_cfg.npz`。
