# screw_unscrew_bottle_cap/1 · Isaac 可复现包

EgoDex 第一视角"拧开铝制水瓶盖"的完整重建 + 分件 CAD + Isaac 回放器。
拿到这个包，只要你有 Isaac Sim 的 Python 环境，就能看到和我们这边**一模一样**的回放。

104 帧 / 30fps / 世界系 `gravity_z_up_world`（重力对齐、米制）。

## 一、先看效果（唯一需要你自己准备的是 Isaac）

```bash
cd <解包目录>/screw_1_isaac_bundle

OMNI_KIT_ACCEPT_EULA=YES <你的 isaac python> viewer/sim/retarget_isaacsim.py \
  --traj   data/retarget/replay_world.npz \
  --object-usd data/retarget/object_0.usd data/retarget/object_1.usd \
  --mode render
```

`<你的 isaac python>` 例如 `.../MagicDexMate/.venv-isaac/bin/python`。

**操作**：窗口打开后停在第 0 帧（SharpaWave 双手 + 瓶身 + 瓶盖就位）→ **ENTER 播放，
再按 ENTER 重播，`q`+ENTER 退出**。手和物体的轨迹会画成 3D 线。

常用开关：`--loop` 循环；`--no-traj` 关轨迹线；`--mode physics` 改成物理仿真
（物体第 0 帧摆好后交给物理引擎，能不能抓住由接触决定）。

### 为什么这个包能跨机器复现
- 两个 `object_*.usd` 的**网格已内嵌**（USDC 二进制，无外部引用），不依赖我们这边的路径；
- `viewer/` 自带手部 USD（`assets/robots/hands/sharpa_wave/`）与 `magicdexmate` 重定向包，
  脚本按 `REPO=dirname(sim/)` 找它们，目录结构已保持一致；
- 唯一的外部依赖是 **Isaac Sim 运行时本身**（体积太大，无法随包分发）。

## 二、这条数据的质量

| 物体 | 身份 | 位置 | 旋转 | 证伪 | 判定 |
|---|---|---|---|---|---|
| object_0 | 瓶身（左手持） | **84.0 good** | **70.5 good** | 0/104 | 双 good |
| object_1 | 瓶盖（右手拧） | **92.0 good** | 48.5 poor | 3/104 | 位置极佳 |

这是我们 screw 任务库里目前最好的一条（此前最好是 23 号：盖 83/70、瓶身 65/65）。

### ⚠ 用之前必须知道的一件事：拧的"转角"没有被捕捉到

瓶盖的 conf_rot 是**分段**的，而且分得很不巧：

| 阶段 | conf_rot 中位 |
|---|---|
| f0–35 盖在瓶上、正在拧 | 31 |
| f36–51 刚拧下、握在手中 | 22 |
| f52+ 盖静置桌面 | **81** |

即"盖子不动时旋转很准，恰恰在拧的过程中最不准"。更直接的证据：把盖相对瓶身绕瓶轴的
转角解出来，f0→f35 累计只有 **-109°**，而真实拧开一个瓶盖通常要 360°–720°。

原因是几何决定的：盖的三轴可观测度 [0.324, 0.346, **0.176**]，最弱的那根就是拧的轴 ——
光滑圆盘绕自身对称轴转，单目 RGB 上几乎没有可追踪特征。**这不是参数没调好，是原理限制。**

⇒ 可放心用：瓶身完整 6DoF、瓶盖的**位置**轨迹、双手 ARKit 米制轨迹。
⇒ 不要用：从瓶盖视觉位姿去读"拧了多少圈"。要这个信号，更现实的做法是从**手指转动**推
（手部数据是设备级真值，MANO 拟合关节误差左 5.9mm / 右 5.6mm）。

瓶身 `rotation_free_axes=[2]`：绕瓶轴自转不可观测（可观测度 0.011），已被正确标为自由轴，
RL 在该维自由探索即可，不是缺陷。

## 三、目录说明

```
data/                       重建产物(整条 take)
  world_fused.npz             相机 + 双手 MANO + 两物体 6DoF
  object_valid_measured.npz   逐帧可信标志(遮挡<0.15 且未被证伪)
  objects/object_{0,1}/       分件 CAD 网格(拟合尺度后)
  masks/ contact/ poseqa/     mask、接触点云与手-物区间、RTS 平滑轨迹+逐帧 σ
  README_USAGE.md             详细数据说明(含本条是怎么调出来的)
  conf_screw_unscrew_bottle_cap_1.mp4  双物体可信度叠加视频(投影轮廓 vs mask + 分数带)
  retarget/
    replay_world.npz          ← Isaac 回放输入
    object_0.usd object_1.usd ← Isaac 物体资产(网格已内嵌)
    ref_qpos_left/right.npz   SharpaWave 双手 22 维关节参考(104/104 帧有效)

cad/                        retrieval 用的资产库原始 CAD
  bottle_body.obj (6.5×6.5×19.7cm)  bottle_cap.obj (3.5×3.5×1.7cm)
  registry_excerpt.json       资产库条目节选(含尺寸告警)

viewer/                     Isaac 回放器(自带依赖, 不需要克隆我们的仓库)
  sim/retarget_isaacsim.py
  magicdexmate/               重定向包
  assets/robots/hands/sharpa_wave/{left,right}/   SharpaWave 手部 USD
```

## 四、关于这里的 CAD（重要）

物体网格**不是重建出来的**，是资产库的分件 CAD：VLM 判定本条 `part_change=separates`
（瓶身与瓶盖会分离），单个刚体网格在原理上表达不了"盖相对瓶身转"，所以走 retrieval 换 CAD。

⚠ **资产库 CAD 的绝对尺寸不可信**，用前要逐条验：本瓶深度+内参实测 19.2–20.0cm，与 CAD 的
19.7cm 吻合，无需缩放；但同一套 CAD 用在 clip 27 那只瓶上时实测 26.5cm，需放大 ×1.28。
两条不是同一个瓶子 —— 而 retrieval 目前是"按任务名硬指定"资产的。

## 五、复现/追溯

- 逐帧分数与判据出处：`data/poseqa/`、`data/README_USAGE.md`
- 本条的调优过程（三处修复各值多少分、各候选帧的实测对比）都写在 `data/README_USAGE.md`
- 数据来源：EgoDex test split `screw_unscrew_bottle_cap/1.mp4`；相机/内参/重力/手均为
  设备标定给定（非视觉估计），深度与物体位姿为重建
