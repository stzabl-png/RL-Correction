# ckpt 交接协议(世界契约)

> 立此协议的事故: 站姿 USD 换版后某 ckpt 回放 **0/64**,而 obs **348 维一字未变**、
> 程序零报错。我们叫这类问题「**同维异义**」——形状没变、含义变了,所有自动检查全绿。

## 一句话规矩

**ckpt 只能在它出生的世界里回放。交接 ckpt 必须连同世界的身份证一起交。**

## 为什么

训练出来的策略是一张「输入数字 → 输出数字」的查表。它执行时**不重新认识场景**,
学到的一切都锚在一套固定几何上:肩膀/手臂基座在哪、桌面多高、物体多重多滑、
参考轨迹长什么样。换了世界,同一个物体在策略眼里的数字就变了,它照旧记忆输出,
手就伸到错误的位置——**而且不会报错**。

## 机制(已落码,不靠人记)

| 环节 | 做什么 | 在哪 |
|---|---|---|
| 训练开始 | 采集完整世界指纹写 `<run>/world.json` | `train_pour.py` → `world_fingerprint.write()` |
| 回放/评测 | 读 ckpt 同级的 `world.json`,与当前 env 比对,**关键项不符直接拒跑(退出码 11)** | `record_pour.py` / `eval_pour.py` → `WF.assert_match()` |
| 强制跳过 | `POUR_IGNORE_WORLD=1`(**只在你明确知道后果时用**) | 环境变量 |

### 关键项(不符 = ckpt 在这台机器上无效)

```
robot.usd_md5                              站姿/机器人 USD
robot.controlled_joint_names_in_order      受控关节名与顺序
reference.md5                              母带
policy_io.obs_dim / act_dim                观测/动作维度
time.control_dt_s / decimation             控制步
table.table_top_z_m                        桌面高度
objects.object_{0,1}.mass_kg / static_friction   物体质量与摩擦
```

### 提示项(不符通常不致命,但会打印)

physics_dt / solver_type / env_spacing / friction_curriculum / obj_jitter_xy /
self_collision / 垫力阈值 / 垫数阈值

## 交接一个 ckpt 要给对方什么

```
<run>/stage1_nn/last.pth        权重
<run>/world.json                ★世界身份证(缺了就没法核对)
母带 npz(按 world.json 里的 md5 取那一版)
机器人 USD(按 world.json 里的 usd_md5 取那一份)
物体 USD(world.json 记了每个物体实际加载的路径)
```

对方跑之前**什么都不用手工核对**——`record_pour.py` / `eval_pour.py` 会自动比对并
在不匹配时拒跑,并逐项打印「出生世界 vs 当前世界」。

## 三条踩过的坑

1. **`world.json` 里写 `"default"` 等于没写**。旧版只记 `usd: "default"`,那只说明
   「没有覆写」,不说明用了哪个文件。现在记**实际路径 + md5**。
2. **没有绝对的「病文件」,只有相对权重的版本错配**。同一份旧站姿 USD,对新任务是错的,
   对那批旧权重却是**唯一正确的世界**。所以旧世界要存档,不要删
   (`assets/vega_1p_sharpa_fixedtorso_stance0803.usd` 是 AAG-F 的出生世界,**永久禁删**)。
3. **母带换代同样会让 ckpt 失效**。母带决定参考轨迹与判据行数;换母带 = 换世界。
