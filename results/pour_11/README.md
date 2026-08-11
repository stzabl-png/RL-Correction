# pour/11 —— 十步全链首个完整样例（2026-08-10）

EgoDex `test/pour/11.mp4`（169 帧 1080p，右手持透明茶瓶向左手扶的灰塑料杯倒水）。
**全程零人工**：v17A 自动标注（`--instance all` 注册双物体）→ 8 步重建（UCB A6000）
→ confidence 逐物体打分 → contact 逐物体×逐手接触提取。本目录是完整产物快照；
复现命令一条：`./reconstruct.sh <视频> --dataset egodex --root <root> --gpu-ids=<卡>`
（AUTO_LABEL_INSTANCE=all），跑法见 `ego_pipeline/Reconstruction/docs/REMOTE_RUNBOOK.md`。

## 成绩单（confidence, 逐物体）

| 物体 | conf_pos | conf_rot | CT反驳 | 尺度 | 裁决 |
|---|---|---|---|---|---|
| object_0 茶瓶(右手) | **87 (good)** | **46 (可用)** | 1 帧 | 可靠 | active |
| object_1 灰杯(左手) | 50 (mixed) | 7 (弃用) | 22 帧 | 可靠(mesh 偏大) | active |

**透明规则 v2 的实证依据**：装茶色液体的透明瓶 FP 表现良好（据此规则改为"空透明才过滤"）；
杯子的低分是多物体标注轮次选帧产出过大 mesh（16×26cm vs 真实约 8×14）所致，与材质无关
——同一只杯在单物体轮次曾 93/30.5。液面下降会让瓶身后段显透明（外观非刚体），本条未致命。

## 接触（contact, 2D 修 3D）

| 组合 | 帧 | 热区 | 主要指垫 |
|---|---|---|---|
| 瓶 × 右手 | f93 | 6069 顶点 (3.2%) | ring/middle/thumb 握带 |
| 杯 × 左手 | f85 | 3892 顶点 (1.2%) | middle/ring 扶带 |
| 瓶×左手 / 杯×右手 | — | 逐物体区间检测自动判零接触跳过 | |

## 目录

```
world_fused.npz        ★相机c2w+K / 双物体6DoF逐帧 / 双手MANO (含 *_all 多物体字段)
objects/*/object_mesh_scaled_final.obj   两物体 mesh (SAM3D 尺度)
masks/                 双手+双物体逐帧 mask
replay_world.npz       双手21关节世界系 + obj_pose_all
ref_qpos_{left,right}.npz  SharpaWave 22-qpos (DexPilot)
contact/               ★接触热度 ply/npz(顶点权重) + 诊断图 + stage4 json
contact_auto*.json     接触区间(并集 + 逐物体)
poseqa/                逐物体打分: audit 逐帧明细 / rts 平滑轨迹+σ / cc CT一致性 / manifest 裁决
*_complete.json        三个完成标记(含分数摘要)
```

RL 消费规则见 `confidence/CONFIDENCE_GUIDE.md` §3（位置 σ 分层、旋转双门、自由轴）。
未收录：interim 中间产物（可再生）、take 级重复 mesh。
