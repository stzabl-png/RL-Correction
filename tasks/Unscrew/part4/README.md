# Unscrew/part4 —— 拧瓶盖任务实例 (V5 框架, 2026-08-29 实例化)

框架身 = `steps/step4_rl/framework` (Pour17 v5 已验收版)。数据 =
`datasets/unscrew_bottle` (egodex_part4 拧瓶盖 18 条重建, 17 条可用;
物体逐帧 conf_pos/conf_rot + 人手逐帧置信度)。

## 一句话

**数据引擎**: 每次用一条重建数据, 在置信度门控的双参考体制 (物体轨迹三档皮筋
/时钟门 + 人手指形指引×人手conf) 下, 用 RL 残差修正从**粗糙重建轨迹恢复一条
物理可行的好轨迹** (左手抓瓶提起转平 → 右手三指慢拧开盖 → 盖放回母带终点 →
撤退), 再批量导出演示。`UNSCREW_CLIP=<n>` 选条 (默认 32=榜首)。

## 分工 (2026-08-29 用户确认)

- **cuRobo 只管机器段** (残差冻结照谱): Approach 站姿→站位腕靶 / Retreat 回站姿。
- **缝1 起的整个交互段 = RL**: 参考=置信度物体轨迹+置信度人手指流; 拧盖动作
  无任何照谱成分 (驱动增益要求真实指尖接触, r_screw 奖真实转角, G3=物理 detach)。

## 与 Pour17 实例的差异 (全部落账 A_Design/DECISIONS.md)

| 件 | Pour17 | Unscrew |
|---|---|---|
| 物体角色 | object=瓶(右手) aux=杯(左手) | object=瓶(**左手**) aux=盖(右手, 螺旋 preengaged) |
| G1 | 双手各≥3/5垫 | 只判左手 (右手时序上后进场) |
| G3 | 倾角≥90°×口口距 hold25 | **拧开释放** (螺旋 detach 锁存) |
| placed 目标 | 静置位 | **母带末行** (盖 rest≠end) |
| 主档 tmix | 主物体 min(pos,rot) | **双物体短板** |
| 形状指引 | 臂 Δq cos | **右手指 Δq cos × 人手conf档** (腕平移是死数据) |
| 新增 | — | 螺旋块 obs 4 维 / r_screw / 三指分级慢拧驱动 |

## 跑法 (顺序不可乱; PY=~/miniforge3/envs/isaac/bin/python)

```bash
bash tasks/Unscrew/part4/C_Wiring/data_engine.sh 32     # 1-6 步自动 (probe_rest →
                                                        # v1 → [cuRobo] → 自检 → 冒烟)
# 后续人工判读节点 (probe_beta β标定 / v2 / 硬闸 / 训练 / 评测) 见脚本尾部打印
```

已验证 (2026-08-29, clip32): scene_layout 17 条全出 / v1 母带 273 行 /
**判据自检家族五件全绿** (放音收入 38 / 批量逐位一致 / RSI 四点位 / 变体反向 / 体制)。

## 已知欠账 (先读 DECISIONS T1 节再动手)

1. **本机无 MagicSim 定制版 cuRobo** (`curobo.motion_planner` fork, 非 PyPI):
   装法 = 从有 MagicSim 的机器拷 `Third_Party/curobo` + pip install -e 进 isaac
   解释器 + `MAGICSIM_ROOT=<路径>`。没装之前机器段是 smoothstep 占位 (只够冒烟)。
2. 离线右臂 IK 达标率低 (URDF 锚系统差): probe_rest 实测锚重跑可修一半,
   终解是 build_reference v2 (Isaac 内站位捕获)。
3. βL=2.0 是抄 Pour17 的初值, 必须 probe_beta 复标。
4. Isaac 侧冒烟 (smoke_zero/probe_*) 尚未在本机跑过 —— 本次交付验证到离线链全绿。

## 环境旗 (框架沿革保留 POUR_* 前缀, 勿改名)

`UNSCREW_CLIP` 选条 | `UNSCREW_TURNS` 拧开圈数覆写 | `POUR_VARIANT` HYB/OBJ |
`POUR_BETA_L/R` | `POUR_REF_NPZ` 母带覆写 | `POUR_UNLOCK` 冒烟出生点直开 |
`POUR_NO_D6` 1-env 回放 | `POUR_SQUEEZE_FF/BONUS_NOW/BONUS_DIST` 与 Pour 同义
