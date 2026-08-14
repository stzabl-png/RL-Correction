# 重建+打分 运行台账

> 记每次代表性运行的分步耗时、设备与踩坑。新条目往下追加。

## 2026-08-10 · pour/11 全链首跑（v17A 自动标注 → 9 步重建 → confidence）

**设备**
- 本地：RTX 4080 SUPER 16GB (sm_89) + i7-14700K —— 从本日起**只做开发**，跑数据一律远程
- 远程：UCB 8×A6000 48GB (sm_86, yanghong@169.229.192.185, 共享机)
- 视频：169 帧 1080p（EgoDex test/pour/11，倒水：右手冰茶瓶→左手灰塑料杯）
- 物体注册：v17A 自动发现 2 实例 → 选杯子(instance_0001, 帧3)；冰茶瓶半透明按规则不注册

**分步耗时（成功路径）**

| 步骤 | 耗时 | 设备 | 说明 |
|---|---|---|---|
| v17A HOI-DETR 检测 | 38 s | 本地 4080S | 5.8GB ckpt, 含 SHA 校验 |
| v17A SAM2 实例传播 | 88 s | 本地 4080S | 2 实例全 166 帧 mask |
| adapter 转格式 | 5 s | 本地 CPU | 代替人工标注 |
| vipe | 177 s | 本地 4080S | |
| sam3_hands | 62 s | 本地 4080S | 必须 SAM3_VERSION=sam3 |
| sam2_object | 0 | — | 被 adapter 产物替代, 跳过 |
| hawor | 53 s | 本地 4080S | |
| sam3d | 93 s | A6000 | 本地 16GB 必 OOM(峰值 ~26GB) |
| sam3d_scale | 51 s | A6000 GPU5 | 本地必 OOM(FP 姿态诊断峰值 ~27GB); 新语义(SAM3D 尺度为准)已生效 scale=0.199, 0 警告 |
| fp_pose | 38 s | A6000 GPU5 | 同上 FP register 峰值超 16GB |
| fuse | ~30 s | A6000 CPU | |
| confidence | 40 s | 本地 4080S | cc+audit×2+rts+manifest |
| 传输(4 往返) | ~1 min | 千兆内网 | interim 87M↑ + hawor 21M↑ + 产物 ~300M↓ |

**纯计算合计 ≈ 10 分钟**；本次实际墙钟 ≈ 32 分钟，差额全是 OOM 折返与诊断（见坑）。
以后按"数据全远程"跑法，预期单条 **≈ 12 分钟**（远程排队另计）。

**结果**：`egodex/test/pour/11` 入库 —— conf_pos 93.0(good) / conf_rot 30.5(可用) /
refuted 4 帧 / 尺度可靠 / 自由轴[1]（杯子回转对称轴, rot_obs 0.012, 正确识别为不可观测）。
全库 34 takes → active 30, 旋转可用 13。

**本次踩掉的坑（已修/已固化）**
1. sam3.1 在 4080S(Ada) 检不出手 → reconstruct.sh 固化 `SAM3_VERSION:=sam3`（三种卡都该用 sam3）
2. **16GB 显存边界 = sam3d / sam3d_scale / fp_pose 三步**（FP 252 候选 × 1080p warp 峰值 26~27GB）→ 跑数据全远程的直接依据
3. **步骤脚本会 `os.environ["CUDA_VISIBLE_DEVICES"]=str(--gpu)` 强制覆写** → 远程选卡必须 `--gpu N` 直传，外部 export 无效（曾三连 OOM 在最挤的 GPU0 上）
4. UCB 共享机礼仪：GPU7=vLLM 专占；root 的 rl_rebuild 训练不能动；邻居显存会瞬时波动，选卡留 ≥7GB 余量
5. 开朗 adapter 只认 episode 级 mask_sequence.json（schema `persistent_mask_sequence_v1`），他文档里写的视频级 json 是错的
6. conda 环境里 editable/egg-link 烤死路径已全量修复（HV2RD 收编后遗症, 25 处 6 个 env）

**待办**
- ✅ UCB 代码树已同步为权威版(2026-08-10 深夜, git 化+符号链接布局, 见 REMOTE_RUNBOOK.md)
- v17A 多实例注册（--instance all）与 VLM 透明过滤前置，未接

## 2026-08-10 · contact 第 10 步入链(pour/11 冒烟)

设备: 本地 i7-14700K 纯 CPU。区间检测(2D mask 邻接): 166 帧接触段秒出;
单手对齐+热度图 @区间中点: **~8 分钟/手**(12k 次能量评估为主)。
语义验证: 右手(握未重建的瓶)自动零区间跳过, 左手 [3,168]→f85 提取 —— 无需人工指定手/帧。
对比: --auto-frame 全片扫描 ~21 分钟, 批量禁用, 只在诊断时用。

## 2026-08-10 · pour/11 双物体重跑(不过滤透明) → 透明规则 v2

远程 GPU5 全链(v17A --instance all 注册瓶+杯) + 本地入库/接触。**逐物体成绩**:
- 茶瓶(object_0, 右手): conf_pos 87(good)/conf_rot 46(可用)/refuted 1 —— **装深色液体的
  透明瓶 FP 表现良好**, 据此规则改 v2: 只过滤"空透明"
- 灰杯(object_1, 左手): 50(mixed)/7(弃用)/refuted 22 —— 多物体标注轮次选帧出了过大 mesh
  (16×26cm, 真实约 8×14), 是 mesh 质量问题非材质; 单物体轮次同一杯曾 93/30.5
接触(多物体闭环首跑): 瓶×右手 @f93 热区 6069 顶点(3.2%, ring/middle/thumb);
杯×左手 @f85 热区 3892(1.2%, middle/ring); 交叉组合(瓶×左/杯×右)零区间自动跳过 ✓
途中修三 bug: confidence 快照半新半旧(--object 不认, 教训:刷新快照必须整目录)、
入队短路缺口(后处理步失败的视频被永久跳过)、层级 root 重摆。
液面下降实证: f93 倒空段瓶身显透明 —— pour 类外观非刚体, 本条未致命(rot 46)但批量留意。

## 2026-08-14 · contact 换 extract_v2: 8 分钟/手 → 18.9 秒/条, 默认改为**开**

设备: 本地 i7-14700K 与 UCB A6000 机, 均纯 CPU(本步不占卡)。

| | 旧 `contact/`(对齐+热度图) | 新 `contact_v2/`(extract_v2) |
|---|---|---|
| 每个"物体×手" | **~8 分钟**(约 1200 次能量评估) | **1.3~1.7 秒** |
| 一条 take | 4 组合 ≈ 28~44 分钟 | **2~5 秒**(UCB 实测 18.9 秒含起环境) |
| 处理帧数 | **1 帧**(区间中点) | **11~20 帧**(整个稳定窗) |
| 性质 | 搜索: 假设有抓握, 把 SharpaWave 手摆上去 | 测量: 逐帧量一次, 测不出就报失败 |

★ 两者不是快慢之分, 是**答的不是同一个问题**。旧的靠 `w_touch` 能量项主动把手拉上表面,
  结果必然几何合理但那是优化器摆的; 手位修正后它的 bite IoU 已从 1.000 掉到 0.000
  (旧的满分是两个错误相消: 手偏 16cm, 对齐器推回 15.5cm)。旧的保留为 `contact_align`,
  不在默认 STEPS 里。

据此 `reconstruct_egodex.sh` 的 contact 默认从**关**改为**开**(要关用 `NO_CONTACT=1`)。
默认关会让下游分不清"这条没有接触"和"这条没跑过接触" —— 而接触点是 GraspPose 选模板
要用的 Video Prior。
