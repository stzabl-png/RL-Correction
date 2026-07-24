# 两套训练设定 A / B

本仓库有**两套并存、互不干扰**的抓取训练设定。它们的分界不是实现细节，而是
**"抓姿合成器(BODex/GraspPose)对这个物体到底管不管用"**。

| | **设定 A** | **设定 B** |
|---|---|---|
| 一句话 | RL 学习 **GraspPose 能处理**的物体 | RL 学习 **GraspPose 处理不了**的物体 |
| 物体特征 | **普遍偏大**，可被整手包络/力封闭 | **普遍偏小、偏扁**，只能指尖捏取 |
| 抓姿来源 | BODex/cuRobo 规划出可用抓姿 | **BODex 抓不出**（包络与桌面几何互斥） |
| RL 的角色 | 在已有可行抓姿轨迹上做**残差修正** | 从 PreGrasp 出发**自己学会指尖抓取** |
| 代表 clip | `Grasp0`(pp0) | `Grasp2`(甜甜圈, 7.7×2.6cm) |

---

## 设定 A — cuRobo-anchor 残差修正

**适用**：BODex/cuRobo 能给出可行抓姿的物体（普遍较大、适合整手包络）。

```
参考骨干 = cuRobo close 轨迹 (腕基本不动, 手指按规划合拢)
RL 输出  = 28 维残差 (腕 ±2cm / ±0.05rad, 手指 ±0.1rad) 叠加在参考上
奖励     = 乘性核心 lift × grip × falling + shaping(approach/contact/lift/imit) + 正则
```

- producer：`producers/ocir.py`(`load_ocir`)，读 OCIR 四类源(ocir_sequence / grasp_pose / curobo_traj)
- clip 注册：`clips.py` 中 `_td_ocir(...)`，variant = `anchor`(cuRobo 骨干) / `human`(重建人手骨干)
- `loose_grip_factor = 0.3`（允许一定的整手握持）
- **状态**：`Grasp0` 单物体已跑通，确定性评测 **99.9%**（详见 `MANUAL.md`）

---

## 设定 B — affordance 引导的指尖抓取

**适用**：BODex 抓不出的物体（普遍小/扁；包络与桌面互斥，指尖捏取又不适合 BODex）。

**设计定调**：指尖抓取是**目的本身**。能用整手 cage 握住的物体，设定 A 的 GraspPose
早就能完成；设定 B 存在的意义就是让 RL 学会 GraspPose 覆盖不到的小/扁物体的指尖抓取。
因此 cage/scoop 是要压制的退化行为，成功判据保留"≥2 指尖 elastomer 接触"。

```
参考骨干 = 重建人手腕 path (腕位姿可信) , re-anchor 到 affordance 区 + 抓取窗内冻结腕
手指参考 = 傻瓜匀速合拢斜坡 (不用 GraspPose, 不用重建手指 pose)
RL 输出  = 同 A 的 28 维残差
奖励     = A 的全部 + 饱和 affordance 项 (只奖励指尖落在高 affordance 点)
```

- producer：`producers/replay_grasp.py`(`load_replay_grasp`)
- clip 注册：`clips.py` 中 `_replay_grasp(...)`，`source="replay_grasp"`
- affordance：逐点热图 `affordance.npz`(`points_raw` (P,3) 物体系 + `heatmap` (P,) ∈[0,1])
- `lam_afford = 0.4`（饱和 `tanh`，防止"所有手指堆到同一侧"）
- `loose_grip_factor = 0.05`（压制 cage/scoop，逼指尖抓）

### 数据可信度（设定 B 的关键前提）
上游重建里三类量的可信度**不同**，设定 B 的设计完全建立在这个判断上：

| 量 | 可信 | 用途 |
|---|---|---|
| 腕位姿（位置+朝向） | ✅ | 定 PreGrasp、算掌心、re-anchor 的锚 |
| 物体 6DoF track | ❌ | **弃用**（实测相对手漂移 10–15cm） |
| 手指 pose | ❌ | **弃用**（换成傻瓜合拢斜坡） |

### re-anchor（设定 B 的核心手法）
物体轨迹不可信 → **从可信的腕反推物体该在哪**：

```
接触窗首帧的腕位姿(可信)
  → 掌心 = 腕 + 9cm·手系+z (与 env _palm_pos 同约定)
  → 物体放到 affordance 加权重心处 (环/接触区, 不是几何质心)
  → 以物体为锚, 手-物整体落桌; 整手上抬 hover_gap 留间隙
  → 抓取窗内冻结腕 (只合手指), 抬升由 grasp_only 硬编码接管
```

---

## 两套如何隔离

设定 B 的所有新增都是**默认关 / 按 clip 门控**，设定 A 零影响：

- `RewardWeights.lam_afford` 默认 `0.0`，仅当 clip 挂了 `affordance` 时由 env 置为 `cfg.lam_afford`
- `loose_grip_afford`、`_tip_affordance()`、affordance 加载：均只在有 affordance 数据时生效
- `clips.load_data_unit` 按 `source` 分派：`replay_grasp` → 设定 B，其余 → 设定 A
- 唯一全局改动是观测加了 `.nan_to_num(0)`（兜 NaN，对设定 A 无害）

## 新增一个物体

- **设定 A**：`clips.py` 里加 `_td_ocir(dataset, obj, grasp_file=...)`，需要 OCIR 四类源齐全
- **设定 B**：`clips.py` 里加 `_replay_grasp(part, name, aff_obj)`，需要
  重建 mesh + retarget `replay_world.npz` + affordance `affordance.npz`

---

## 现状与已知边界

- **设定 A**：`Grasp0` 99.9% 已跑通；边界见 `MANUAL.md` §7（GraspPose 候选全 `ok=false`、
  摩擦模型不匹配等——当前结果是绕过而非解决这些问题）。
- **设定 B**：**管线已全部跑通**（producer / re-anchor / 冻结腕 / affordance 奖励 / 训练可运行），
  但首个对象 `Grasp2`(甜甜圈) **尚未训练成功**：零残差基线能形成拇指+小指对置接触
  (0.81/5)，但抬升期物体被留在桌上——平放薄环的顶抓需要手指探到环下或勾进洞，
  当前"通用合拢 + 固定腕"产生的是侧碰而非可提起的握持。方法框架成立，对象选择待调整。
