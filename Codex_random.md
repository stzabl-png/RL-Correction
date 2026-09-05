# Sweep2 Cube-Start Generalization

更新时间：2026-09-03。本文件只记录任务2：保持当前Sweep2 reconstruction、双臂reference、GraspPose、物理世界、Gate、reward和完整方法不变，只改变cube初始位置与颜色，训练一条覆盖5个固定配置的共享策略。它不记录主消融，也不记录未来4条新Sweep reconstruction。

## 1. 实验边界

- 任务1主消融独立运行；本任务不得修改或重启其session、checkpoint和录像。
- 任务2的5个配置共享同一条重建工具/双臂轨迹，只改变cube。
- 颜色仅用于录像辨识，不进入191维Actor observation，也不改变质量、摩擦或碰撞属性。
- 任务3等待用户提供4条新Sweep数据对应的GraspPose；它和本任务不是同一种扩展。

## 2. 五个固定配置

原始cube中心为`[-0.0259767957, -0.1788897067, 0.8830000162] m`。由reference首帧簸箕姿态建立桌面内pan-local坐标轴：保留原中心，沿局部横向左右各移1 mm，沿局部纵向前后各移5 mm。“前”指朝簸箕入口方向；只投影到世界XY，世界Z不变。

| variant | color | world XY offset (mm) | cube start world XYZ (m) |
|---|---|---:|---|
| `center` | red | (0.000, 0.000) | (-0.0259768, -0.1788897, 0.8830000) |
| `left_1mm` | blue | (-0.923, -0.385) | (-0.0268998, -0.1792744, 0.8830000) |
| `right_1mm` | green | (+0.923, +0.385) | (-0.0250537, -0.1785050, 0.8830000) |
| `front_5mm` | yellow | (+2.033, -4.568) | (-0.0239436, -0.1834577, 0.8830000) |
| `back_5mm` | purple | (-2.033, +4.568) | (-0.0280100, -0.1743218, 0.8830000) |

正式产物：

```text
tasks/Sweep/2/A_Design/L2_Reference/build_cube_start_variants.py
tasks/Sweep/2/A_Design/L2_Reference/sweep2_cube_cross5_v1.json
tasks/Sweep/2/A_Design/L2_Reference/sweep2_cube_cross5_v1.npz
```

每个variant是固定配置，不是episode内连续随机jitter。训练按`env_id mod 5`分配，1024环境数量为205/205/205/205/204。

## 3. 几何与物理验证

静态几何检查5/5通过：前80个scripted steps内扫把工作点至cube中心最小距离33.45–36.64 mm；最近接触发生在reference第423–425行，距离4.67–9.59 mm；接触窗口工作点速度0.075–0.078 m/s。

物理验证采用同批5环境、零residual、完整reference强制回放。reset坐标回读误差为0.0 mm（5/5），证明环境正确读取manifest位置。

绝对的“前80步cube不得移动”不适用于本reference：原始中心本身会在scripted prefix被工具推动。正式判据是四个偏移配置不得比同批中心产生更严重的击飞：前缀位移、前缀速度、整段最大速度不超过中心的1.15倍，且偏移配置单步位移不超过30 mm。

| variant | prelude displacement | prelude speed | trajectory speed | max one-step move | result |
|---|---:|---:|---:|---:|---|
| `center` | 43.73 mm | 1.082 m/s | 2.463 m/s | 22.95 mm | baseline |
| `left_1mm` | 24.85 mm | 0.329 m/s | 1.060 m/s | 27.88 mm | pass |
| `right_1mm` | 19.32 mm | 0.311 m/s | 0.929 m/s | 15.21 mm | pass |
| `front_5mm` | 22.82 mm | 0.389 m/s | 1.102 m/s | 27.48 mm | pass |
| `back_5mm` | 21.31 mm | 0.696 m/s | 0.990 m/s | 26.38 mm | pass |

四个新配置没有引入比中心更严重的直接撞飞。报告：`logs/Sweep2_cube_cross5_validation_20260903/report.json`。

历史5 mm圆周方案出现最高180.52 mm前缀位移，已拒绝。短暂生成的0.5 mm圆周方案也已被cross5取代，不得使用旧ring5产物训练。

## 4. 训练协议

- 一条shared policy覆盖5个配置；若最终分位置评测显示明显相互干扰，再考虑单配置policy。
- 完整方法：reference confidence ON、human shaping ON。
- 无Actor BC、无离线Critic预热；前10个PPO epochs只做在线critic-only更新。
- 1024 environments，seed 42，训练至24M agent steps。
- checkpoint每3M保存；deterministic video每6M生成。
- `world.json`记录manifest路径/hash、分配规则、reference hash和空`warmup_transitions`。
- 最终报告五位置宏平均和每个位置的Gate4/fully-inside。

## 5. 运行台账

训练已于2026-09-04正常达到24M并自行退出，没有Traceback、CUDA OOM或人为中止：

```text
tmux:      sweep2_cube_cross5_full_gpu0_20260903（已正常结束）
run:       logs/Sweep2_cube_cross5_full_nooffline_seed42_20260903/
artifact:  Sweep2CubeCross5FullNoOffline__20260903_policy
final:     24,018,944 actual agent steps
manifest:  SHA-256 0d27138593491cf690d1d1ebc50c77ff29badb4a8445011c0f960f9dfba601cf
```

`world.json`确认`cube_variants.enabled=true`、分配方式为`env_id_mod_5`、method为`full`且`warmup_transitions=[]`。3M至24M的八个checkpoint均已生成。

## 6. 训练窗口结果

下表是各checkpoint保存时的训练窗口统计，不是固定回合独立评测；窗口包含五种配置的混合episode，因此目前只能解释为共享策略总体曲线，不能当作每个位置的单独成功率。

| checkpoint | completed episodes | Gate3 | Gate4 | fully-inside | success mouth clearance |
|---:|---:|---:|---:|---:|---:|
| 3M | 115 | 32.17% | 32.17% | 32.17% | 5.67 mm |
| 6M | 63 | 52.38% | 49.21% | 50.79% | 3.70 mm |
| 9M | 87 | 68.97% | 66.67% | 66.67% | 4.39 mm |
| 12M | 101 | 61.39% | 60.40% | 61.39% | 5.38 mm |
| 15M | 91 | 68.13% | 67.03% | 67.03% | 7.60 mm |
| 18M | 98 | 74.49% | 72.45% | 72.45% | 5.35 mm |
| 21M | 101 | 72.28% | 71.29% | 71.29% | 2.88 mm |
| 24M | 95 | 84.21% | **81.05%** | 83.16% | 5.49 mm |

Operational success仍严格采用Gate4；个别窗口的累计`fully_inside`统计略高于Gate4，不能用来替换Gate4结果。共享策略从3M的32.17%提升到24M的81.05%，说明同一策略联合学习五个小范围cube起点在总体上可行。12M和21M有小幅回落，整体仍呈上升趋势，24M为当前最高Gate4窗口。

6M、12M和18M录像已经生成。原autorecorder随训练父进程退出而漏过刚保存的24M节点，随后在GPU1通过独立单环境tmux `sweep2_cube_cross5_record24_gpu1_20260904`完成补录。24M录像在step 289达到真实Gate4终态，`gates=[1,1,1,1]`，录像、top-down terminal frames和`rollout.npz`均已落盘。

## 7. 24M四个改变位置的固定评测

用户指定不重复评测默认中心，只评测四个发生位移的配置。四组使用完全相同的24M checkpoint、deterministic mean action、seed 42和Gate4成功定义；每组通过`SWEEP_CUBE_VARIANT_INDEX`强制50个环境全部使用同一个固定位置，并完成恰好50个episodes。

```text
checkpoint: logs/checkpoints/Sweep2CubeCross5FullNoOffline__20260903_policy_0024M/checkpoint.pth
checkpoint SHA-256: 8508cf06ae5d879b97b9bde8f36918a2e876f208803dd37f07196708c52a6a22
manifest SHA-256: 0d27138593491cf690d1d1ebc50c77ff29badb4a8445011c0f960f9dfba601cf
results: logs/Sweep2_cube_cross5_eval50_24M_20260904/
```

| variant | offset | success | Gate1 | Gate2 | Gate3 | Gate4 | mean episode steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cube_left` | pan-local left 1 mm | 42/50 = **84%** | 100% | 94% | 86% | 84% | 333.74 |
| `cube_right` | pan-local right 1 mm | 45/50 = **90%** | 100% | 100% | 92% | 90% | 316.58 |
| `cube_front` | pan-local front 5 mm | 40/50 = **80%** | 100% | 88% | 80% | 80% | 337.74 |
| `cube_back` | pan-local back 5 mm | 38/50 = **76%** | 100% | 96% | 76% | 76% | 340.26 |
| four-position aggregate | 200 episodes | 165/200 = **82.5%** | 100% | 94.5% | 83.5% | 82.5% | — |

四个改变位置全部明显超过50%，范围为76%–90%，宏平均/合并平均均为82.5%（四组episode数相同）。该结果排除了“训练窗口81.05%完全由默认中心贡献”的解释，说明一条共享策略确实能覆盖这四个已训练的小范围位置变化。

主要差异发生在Gate2→Gate3：右移1 mm最好，Gate2为100%、Gate4为90%；后移5 mm虽然Gate2为96%，但Gate3/Gate4降至76%，说明它通常能产生真实扫动，困难主要在把方块送进入口，而不是场景ready或首次接触。四组Gate3到Gate4最多只损失2个百分点，表明一旦形成浅进入，绝大多数episode也能完成whole-footprint fully-inside。

该评测证明的是对训练分布内四个固定位置的覆盖，不是对任意连续位置或未见位置的泛化。默认中心未纳入本轮200-episode结果；训练窗口81.05%与固定评测82.5%口径不同，前者是混合训练episode窗口，后者是四个强制位置各50次deterministic rollout，不能把两者当作同一统计量。未经明确批准，不更改Gate、reward、reference或物理参数来迁就困难位置。
