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
# 后续节点 (probe_beta 基线诊断 / v2 / 稳定性闸 / 训练 / 评测) 见脚本尾部打印
```

clip32 已验证：17 条自包含数据入库；probe 实测静置；cuRobo Approach/Retreat；
v1 全链 316 行；**判据自检家族五件全绿**；`smoke_zero --steps 800`
完成（静置误差 <5mm、obs=(4,507)、机器段死线 0 误触）。probe_beta 四档
β=1/1.5/2/3 的零动作参考均未持住；按 T2-4 用户裁定，这是 RL correction
基线而非母带失败，正式 βL=1.0 不做 prior 外推。

规划产物绑定交互几何摘要与 `env_rest.json` 哈希，旧规划不能串用。
v2 允许保存 IK 坏行诊断；训练可用性闸要求 ≥3/4 环境全链数值有限且不发散，
再原子写 `acceptance_v2.json`（绑定 v2 全文件 MD5 + 完整世界指纹）。
训练入口会先于 IsaacLab 导入做母带/凭据预检，建环境后再核对现场世界。

> **只需要 RL 环境 (物理建模 + 奖励设置) 的人, 直接读
> [`docs/ENV_UNSCREW_V5.md`](../../../docs/ENV_UNSCREW_V5.md)** —— 文件职责、
> 螺纹副物理、Gate/护送判据、人手→机械手重定向、改完怎么自检, 一份讲完。
> 要跑训练的另见 [`docs/HANDOFF_20260901_unscrew_v5.md`](../../../docs/HANDOFF_20260901_unscrew_v5.md)。

## 当前收尾状态 (2026-08-31 T2-6 之后)

**这一轮修掉的三层病** (详见 DECISIONS T2-6):
1. **参考臂行不可达** —— 右臂对自己的腕目标中位差 44.7cm、100/103 行贴限位。
   根因是把"判据看不见的自由度"照抄人手: 盖/瓶绕自身轴的自转在数据里不可信
   (用户明示: 盖旋转置信度低)、在判据里不可判 (`placed` 只看轴倾角), 却被
   当成硬参考。改成解出来: 右腕抓握自转扫描 + 脱离后限速漂移、盖姿态跟手走、
   瓶自转脱离后冻结; 再修 IK 三处自伤 (失败解当种子/跨分支顶替/平滑平均分支)。
   结果 右臂 12.6%→77%、左臂 66%→100%, 持瓶朝向 hold_yaw=-35° 入库。
2. **假摩擦** —— 旧口径"接触门 × ω 阻尼"让轻碰即空转 (碰一下瓶盖就自己脱落)。
   按老方法线 U40 移植真实螺纹副: 咬合期球形重惯量 + PhysX 摩擦锥传扭 +
   静锁/库仑/粘滞 + 反作用扭矩回瓶身 + 零接触真值门。新增 `probe_thread.py`。
3. **判据只拦得住"甩下去"** —— 补齐 U41 的另两件: 持盖运输步数、下放降幅峰值。
   判据摘要 schema 3→4, 变体自检 5→7 件。



1. v2 已重铸为 316 行（81/25/103/25/82），MD5
   `0d78d12af9c426a629f96c570ae88983`；IK 坏行与 15 个关键窗坏行已入
   `meta_v2` 作为 correction 基线，不阻塞训练。
2. `acceptance_v2.json` 已绑定上述 MD5 和完整世界指纹；4/4 环境全链状态、
   速度和力均有限且不发散。零动作成功 0/4 按用户裁定只作诊断。
3. HYB/OBJ 均完成 4-env、32-step rollout + 一次 PPO 更新，checkpoint 全张量
   有限；RSI 解锁冒烟还验证两种变体的交互分项台账非零。首轮无认证样本的
   `sr/cert_pass=NaN` 仍保留“无分母”语义，但已在日志边界跳过，并以 0 更新
   课程 EMA，避免状态永久污染。长期任务成功率只认独立 eval。
4. 跨服务器部署见仓库根目录 `docs/DEPLOY_UNSCREW.md`；训练不需要安装
   cuRobo，已规划机器段随 Git LFS 分支一起部署。

## 环境旗 (框架沿革保留 POUR_* 前缀, 勿改名)

`UNSCREW_CLIP` 选条 | `UNSCREW_TURNS` 拧开圈数覆写 | `POUR_VARIANT` HYB/OBJ |
`POUR_BETA_L/R` | `POUR_REF_NPZ` 母带覆写 | `POUR_UNLOCK` 冒烟出生点直开 |
`POUR_NO_D6` 1-env 回放 | `POUR_SQUEEZE_FF/BONUS_NOW/BONUS_DIST` 与 Pour 同义
