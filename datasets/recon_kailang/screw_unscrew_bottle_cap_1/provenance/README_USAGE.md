# screw_unscrew_bottle_cap/1 重建数据 · 使用说明

EgoDex 第一视角"拧开铝制水瓶盖"。左手握瓶身、右手拧下瓶盖并放到桌面。
104 帧 / 30fps / 世界系 `gravity_z_up_world`（重力对齐、米制，EgoDex 设备标定给定）。
重做于 2026-08-24；旧版（单物体、52/52）已存快照。

## 分数 —— 全 screw 库目前最好的一条

| 物体 | 身份 | 位置 | 旋转 | 证伪 | 判定 |
|---|---|---|---|---|---|
| object_0 | 瓶身（左手持） | **84.0 good** | **70.5 good** | **0/104** | **双 good** |
| object_1 | 瓶盖（右手拧） | **92.0 good** | 48.5 poor | 3/104 | 位置极佳 |

对照：此前最好的 23 号是 盖 83/70、瓶身 65/65。本条两件的位置分都更高。
高可信帧内更好：瓶身 67 帧中位 **88**，瓶盖 86 帧中位 **94**。

瓶身 `rotation_free_axes=[2]` —— 绕瓶轴自转不可观测（可观测度 0.011，轴对称+CAD 无纹理），
这是**正确的自由轴标注**，RL 在该维自由探索即可，不要当成缺陷。

## 从 52/52 单物体到现在，做对了三件事

**1. 走上 CAD 分件路线（此前从未真正生效）**
VLM 早就判 `separates` + `needs_retrieval=true`，但这条数据一直用的是 SAM3D 单物体网格
（155738 顶点、重建帧 f7）。根因是 framescan 精修轮带 `--force` 重跑 sam3d，把 retrieval
放好的分件 CAD 覆盖掉，而 PROVENANCE 还照抄 retrieval.json 声称"已用 CAD"——2026-08-22
修复（commit 916e04c）后本条才第一次吃到分件资产。
现分配：object_0←bottle_body、object_1←bottle_cap，面积比 6.74，`assign_reliable: true`。

**2. 把左手从瓶身 mask 里排除（conf 44 → 84）**
v17A 的单点标注让 SAM2 把**整只左手**圈进了瓶身 mask（中位 73277px，按深度反推"宽度"
10cm 而 CAD 只有 6.5cm）。手是非刚体、又与瓶相对运动，混进刚体 mask 会同时污染网格拟合
与逐帧打分。修法：f95 处三个负点 (875,600)/(838,668)/(880,730) 压在手上；mask 降到 44222px。

**3. 瓶盖注册帧从 f95 挪到 f40（conf 35 → 92，证伪 36 → 3）**
病因不是遮挡（盖的遮挡中位仅 0.01），而是注册帧离"盖还拧在瓶上"那段（f3–35）太远，
向后跟踪漂移到**穿进瓶身**（failure mode `penetrating`）。实测各候选：

| 盖注册帧 | IoU | conf | 证伪 |
|---|---|---|---|
| f20 | 0.413 | 50 / 25 | 25 |
| f30 | 0.362 | 51.5 / 11 | 52 |
| **f40** | 0.662 | **92.0 / 48.5** | **3** |
| f60 | 0.670 | 87.0 / 49.5 | 14 |
| f70 | 0.593 | 35 / 23 | 26 |
| f95 | 0.608 | 35 / 18 | 36 |

★ f20 是盘面正对相机、mask 面积全片最大（9094px）的一帧，按直觉最该适合注册，**实测反而
差很多** —— 这条链上的选帧只能实测、不能推理。f40 三次复跑结果完全一致（92.0/48.5/证伪3），
不是抽签运气。

## CAD 尺寸：本条无需缩放

深度+内参实测本瓶 **19.2–20.0cm 高**，资产库 CAD 是 19.7cm —— 吻合。
（注意：clip 27 那只瓶实测 26.5cm，需把同一套 CAD 放大 ×1.28。**两条不是同一个瓶子**，
所以"资产库 CAD 按任务名硬指定"这件事本身是有风险的，用之前应逐条验尺寸。）

## 逐帧可信标志

`object_valid_measured.npz`：
- `trustworthy (2,104)` 遮挡<0.15 且未被 CoTracker 证伪
  - object_0 可信段：f0–20、f48–50、f61–103（67 帧）
  - object_1 可信段：f0–19、f21–25、f27–35、f52–78 等（86 帧）
- `occlusion`、`conf_pos_per_frame` 原始逐帧量

本条 take 级中位已经很高，逐帧标志主要用于挑最干净的片段做精细用途。

## 文件

| 路径 | 内容 |
|---|---|
| `world_fused.npz` | 相机 + 双手 MANO + 两物体 6DoF |
| `object_valid_measured.npz` | 逐帧可信标志 |
| `objects/object_{0,1}/` | 分件 CAD（bottle_body / bottle_cap） |
| `poseqa/rts_*.npz` | RTS 平滑轨迹 + 逐帧 σ |
| `contact/`, `contact_auto_*.json` | 接触点云与手-物区间 |
| `retarget/` | `object_{0,1}.usd`、`ref_qpos_left/right.npz`（104/104 帧有效）、`replay_world.npz` |
| `conf_screw_unscrew_bottle_cap_1.mp4` | 双物体可信度叠加视频 |

Isaac 回放：

```bash
cd /home/lyh/Project/Reconstruct_and_Retarget && \
OMNI_KIT_ACCEPT_EULA=YES third_party/MagicDexMate/.venv-isaac/bin/python \
  ego_pipeline/Retargeting/sim/retarget_isaacsim.py \
  --traj results/screw_1_better/retarget/replay_world.npz \
  --object-usd results/screw_1_better/retarget/object_0.usd \
               results/screw_1_better/retarget/object_1.usd --mode render
```
