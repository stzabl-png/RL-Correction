# Clean 擦盘子 — 母带 (L2)

**当前可训练的仍只有 take 3 一条。** take 18 已按"倒扣盘变体"重做出一条 `clean18_reference_v1.npz`
(2026-09-10, 台账 §5.16), 几何两关全过但**零动作冒烟未过 (die 0.917), 未发车**。take 1 目检出局。
take 18 与 take 3 **资产分家** (盘用 take18 自己的倒扣网格 + 自己的 GraspPose; 布仍用 take3 的),
⟹ 原来"多条母带共用一套资产混训"的设想对 take 18 **作废**, 它是独立的变体任务。
每条母带贡献**运动**: 盘的位置轨 + 海绵在盘系里的面内擦拭图案。

## 两条带

| 母带 | 运动来源 | 行数 (20Hz) | T_EP | 覆盖率 | 行程 | 接触行 | 双臂稠密 IK | 腕距 min |
|---|---|---|---|---|---|---|---|---|
| `clean3_reference_v1.npz` | take 3 | 424 (21.2s) | 661 | 0.569 | 87cm | 1.00 | 右 1.00 / 左 1.00 | 10.7cm |
| `clean18_reference_v1.npz` | take 18 | 465 (23.2s) | 714 | **1.000** | 167cm | 1.00 | 右 1.00 / 左 1.00 | 11.6cm |

⚠ take 18 那行的覆盖/行程**不与 take 3 同尺**: 它擦的是倒扣盘的外底 (近乎平面, y∈[0.53,1.23]cm),
take 3 擦的是碟内面 (碟形, y∈[−0.75,1.22]cm)。判据几何已按母带走 (`CleanGeometry.from_reference`),
但两条带量的是两块不同的面, 成绩不可直接并排。
⚠⚠ **take 18 上覆盖率这把尺子基本失去区分度**: 平面上 13.2cm 海绵够得着碟心, 参考自身覆盖就已经
**1.000 (碟心 60/60, 缘带 156/156)**, 而 A4 线才 0.45 —— take 3 上参考自身只有 0.569 且碟心 0/60。
take 18 线要判"擦得好不好", 得另找有头寸的量 (例如行程、接触行占比、或把格子调细)。

**take 1 已出局** (2026-09-10 目检): 视频里右手全程握的是**黄色洗洁精瓶**在往盘上挤, 不是擦拭。
母带移到 `archive/clean_take1_soap/`, 数据仍留在 `datasets/clean_tableware/1/` 备将来做挤压类任务。

**take 18 = 倒扣盘变体** (2026-09-10 重做, 台账 §5.13/§5.14/§5.16): 视频里盘是倒扣的 (擦外底),
现在用 **take 18 自己的盘网格** (⌀18cm×1.9cm, 擦拭面带真圈足) + 为倒扣盘重新合成的 GraspPose
`Clean18_plate_left.npz` (`27_Quadpod__s14_17_6`); 洗碗布仍用 take 3 的资产与先验。
**不要加 `--plate_flip`** —— 那是给"翻 take 3 资产"写的; take 18 自己的网格在规范系里天然就是倒扣。
**选候选要看方位角**: env 起手会把盘按视频 yaw 0° 摆好核 IK (阈值 2cm), 候选的腕方位角必须落在
take 3 那一带 (≈161°); 用户初选的 006 是 −49°, 在那里 IK 差 21cm 被闸拦下 (母带侧转 plate_yaw 补偿
**没用**, env 脚手架不吃母带的 plate_yaw)。
**当前卡在第三关**: 零动作 die 0.917 (全是 die_rel, 掉落/倾覆 0), 海绵相对锁存转 21.1° 越过 20° 死线,
盘自己在左手里转 12.2°。诊断与下一步见台账 §5.16。

A4 达标线 = 覆盖率 ≥0.45 ∧ 行程 ≥70cm, 两条都过。碟心 0/60 格两条都碰不到
(13.2cm 海绵放不进 ⌀18cm 浅碟的 9cm 平底), 这是资产几何决定的, 不是母带问题。

`clean3_reference_v2/v3.npz` 是 take 3 的旧变体 (v2 = 沉降重锚, v3 = 带抓稳前奏, Stage-1 用),
不在多母带集合里。

## 复现命令

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
B=tasks/Clean/3/A_Design/L2_Reference/build_reference.py

# take 3 (原版, 2026-09-07 定案)
env PYTHONPATH=. $PY $B --take 3 --assets_take 3 \
  --plate_motion_scale 0.5 --psi_mode const --plate_yaw_deg 0 --sponge_yaw_deg -130 \
  --output tasks/Clean/3/A_Design/L2_Reference/clean3_reference_v1.npz

# take 18 倒扣盘变体 (2026-09-10 重做: 盘用 take18 自己的资产+先验, 布仍用 take3 的)
env PYTHONPATH=. $PY $B --take 18 --assets_take 18 --sponge_assets_take 3 \
  --plate_prior tasks/pregrasp/priors/Clean18_plate_left.npz \
  --sponge_prior tasks/pregrasp/priors/Clean3_sponge_right.npz \
  --plate_motion_scale 0.5 --psi_mode const --plate_yaw_deg 0 --sponge_yaw_deg 60 \
  --output tasks/Clean/3/A_Design/L2_Reference/clean18_reference_v1.npz
```

`--take` 决定运动来源 (rts 物轨 + replay_world 手轨 + scene_layout 落盘位置),
`--assets_take` 决定**盘**网格, `--sponge_assets_take` 决定**布**网格 (默认跟 `--assets_take`)。

`--sponge_xy_recenter` / `--sponge_xy_scale` 是 2026-09-10 为 take 1 加的面内图案归心/缩幅旋钮,
take 1 出局后当前两条带都不用它们 (默认恒等), 留着给将来"图案偏出盘面"的 take。
⚠ 它们能把不贴盘的轨迹硬掰成擦拭图案 —— **先确认这条 take 的动作真的是擦, 再用**,
take 1 的教训就是用它掩盖了"这根本不是擦盘"这个事实。

**海绵面内 yaw 偏置怎么定 (三条判据, 缺一不可)**: `--psi_mode const` 把重建的 ψ(t) 换成中位常量,
偏置必须同时满足 ① 双臂稠密 IK `ok_ratio` = 1.00 ② **双腕间距 min ≥ 10cm** ③ 零动作不掉不死。
`--geometry_only` 一次把三项里的前两项 (可达 + 腕距) 和覆盖率一起打出来。

⚠ **只按可达 + 覆盖选会选出"两手叠在胸口"的解** —— take 18 首版取 yaw=15, 可达 1.00、覆盖 0.634,
但腕距 min 只有 3.2cm, GUI 里双臂扭曲重叠在机器人胸口。yaw 全周扫描 (步长 30°) 后:
| yaw | 可达(右) | 腕距 min | 零动作 die_rel | 海绵手内滑移 |
|---|---|---|---|---|
| 15 (首版) | 1.00 | 3.2cm ✗ | 0.500 | 1.58cm |
| **60 (定版)** | 1.00 | **11.0cm** | **0.000** | **0.30cm** |
| 75 | 1.00 | 13cm | 0.727 | 2.3cm |
| 90 | 1.00 | 15.2cm | 0.933 ✗ | 2.54cm |
腕距越大不等于越好: yaw 越往 90 转, 海绵越是被**横着拖**而不是顺长轴拖, 抓握力矩变大, 零动作就滑。
⟹ 取"腕距刚够 (≈ take 3 的 10.7cm 同级) 且零动作最稳"的那一档。

⚠ 稀疏可达扫描每 20 源帧采一次, **会漏行**: 曾有配置稀疏全通而稠密 IK 在第 400 行失败。
以稠密 IK 的 `ok_ratio` 为准。

**闸已内置**: `--min_wrist_gap_cm` (默认 10.0) 在稠密 IK 之后核双腕间距, 不过就**拒绝写盘**并提示重扫 yaw。
用 `--min_wrist_gap_cm 0` 可强行放行 (只打印不拦)。已验: 首版 yaw=15 被拒 (min 2.6cm 无文件产出),
定版 yaw=60 放行且与已发布母带逐位一致。

## 拿哪条训练

用 `CLEAN_REF_NPZ` 指定, 不传就是 take 3:

```bash
# take 18 资产分家 ⟹ clip 与盘先验必须跟母带一起换 (只换 CLEAN_REF_NPZ 会拿 take3 的盘去装 take18 的带)
bash tasks/Clean/3/C_Wiring/local_clean3_task.sh Clean18_taskS_s42 42 \
  CLEAN_CLIP=Clean18_plate \
  CLEAN_PRIOR_PLATE=/home/lyh/Project/RL_Correction/tasks/pregrasp/priors/Clean18_plate_left.npz \
  CLEAN_REF_NPZ=tasks/Clean/3/A_Design/L2_Reference/clean18_reference_v1.npz \
  CLEAN_S2_SOFT_REL=1
```

零动作冒烟 (发车前必跑):

```bash
env OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=. RL_ISAAC_NO_GUARD=1 \
    POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0 \
    CLEAN_REF_NPZ=<母带> $PY tasks/Clean/3/C_Wiring/smoke_task.py --headless --num_envs 8 --modes zero
```

零动作实测 (8 env, 2026-09-10 定版): take 3 认证 0.125 / 死亡 0.188 / 掉落 0;
take 18 认证 0.023 / **死亡 0.000** / 掉落 0 / 海绵手内滑移峰值 0.30cm。
两条都不掉物体。认证率低是零动作的固有现象 (盘在认证窗内沉降不到 1cm/5°), take 3 同样只有 0.125,
它照样训到 16/16 —— 判读要看死亡率与手内滑移, 不是零动作认证率。
