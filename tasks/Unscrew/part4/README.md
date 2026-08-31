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

离线已验证 (2026-08-30, clip32): scene_layout 17 条全出 / 审计 v1 273 行 /
**判据自检家族五件全绿** (放音收入 38 / 批量逐位一致 / RSI 全点位 / 变体反向 /
体制 9 项)。严格位置/姿态双门槛下，离线右臂 IK 仅 2%（全行中位
14.61cm/19.7°），左臂 56%；当前入库 v1 仍是 offline-rest + smoothstep
脚手架，训练入口会拒绝它。必须完成真实静置、cuRobo 双机器段、v2 重铸及硬闸
后再发射。

规划产物绑定交互几何摘要与 `env_rest.json` 哈希，旧规划不能串用；物理硬闸
≥3/4 通过才原子写 `acceptance_v2.json`（绑定 v2 全文件 MD5 + 完整世界指纹）。
训练入口会先于 IsaacLab 导入做母带/凭据预检，建环境后再核对现场世界。

## 已知欠账 (先读 DECISIONS T1 节再动手)

1. ~~本机无 cuRobo~~ **已解决 (2026-08-30)**: `curobo.motion_planner` 就是
   NVlabs/curobo 新版主线, 已装 (~/WorkSpace/curobo) + 机器人配置自动生成
   (`datasets/vega_urdf/vega_1p_sharpa_curobo.yml`), 规划冒烟全通 —— 详见
   DECISIONS T1-3b。跑 plan_machine_segs 即可出机器段。
2. 离线右臂 IK 达标率低 (URDF 锚系统差): probe_rest 实测锚重跑可修一半,
   终解是 build_reference v2 (Isaac 内站位捕获)。
3. βL=2.0 是抄 Pour17 的初值, 必须 probe_beta 复标。
4. Isaac 侧冒烟尚未完成；2026-08-30 启动链/D6 已修至场景初始化阶段，但两张
   GPU 被既有任务持续占满，真实 probe_rest/smoke_zero/probe_* 待空卡续跑。

## 环境旗 (框架沿革保留 POUR_* 前缀, 勿改名)

`UNSCREW_CLIP` 选条 | `UNSCREW_TURNS` 拧开圈数覆写 | `POUR_VARIANT` HYB/OBJ |
`POUR_BETA_L/R` | `POUR_REF_NPZ` 母带覆写 | `POUR_UNLOCK` 冒烟出生点直开 |
`POUR_NO_D6` 1-env 回放 | `POUR_SQUEEZE_FF/BONUS_NOW/BONUS_DIST` 与 Pour 同义
