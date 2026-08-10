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
- UCB 代码树是 2026-08-01 旧版，待同步为合并后权威版（本次远程跑靠手工推单文件补丁）
- v17A 多实例注册（--instance all）与 VLM 透明过滤前置，未接
