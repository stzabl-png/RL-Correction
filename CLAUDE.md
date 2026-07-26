# RL_Correction

**动手之前先读 `docs/MANUAL.md`**（怎么跑、模型结构、踩过的坑）。
**两套训练设定 A/B 的分界见 `docs/TRAINING_SETUPS_A_B.md`**（A=GraspPose 能处理的物体/普遍偏大；
B=GraspPose 处理不了的小/扁物体，靠 affordance 学指尖抓取）。
当前进度快照见 `docs/HANDOFF_*.md`（取时间戳最新的那份）。

## 一句话
PPO 训练**残差策略**：在参考手部轨迹上叠加小幅修正，让飞手 SharpaWave 在 IsaacSim 里抓起桌上物体。
`pp0` 单物体已跑通（冠军 run = `logs/Grasp0`，**带点云**，确定性评测 **99.9%**）。
命名约定：run = `Grasp<物体号>`（0 对应 pp0）；点云已是默认输入，不再单独标 tag。

## 立刻会犯错的几件事

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python    # 唯一带 isaacsim+isaaclab 的解释器
```

1. **所有命令加 `SHARPA_WANDB=0`**，否则 wandb 弹交互式登录卡死进程。
2. **成功率只认 `eval_policy.py`**。训练期 TensorBoard 的 `success_rate_t0` 分母是"这一步恰好
   reset 的 env 数"，单点非 0 即 1，不可信。
3. **改动作界 / sigma 之前先读 `MANUAL.md` §6.1**。残差界 × sigma = 每步扰动量，
   这个任务的容忍度是**毫米级**；放大过就是接触率 0.1%、物体被打飞。
4. **`ppo.py` 里那几个 `.detach()` 别删**（§6.4），删了立刻泄漏几个 GB 直至 OOM。
5. **杀 Isaac 进程后要复查**：可能训练循环停了但卡在关闭流程里仍占显存，需要 `kill -9`；
   杀完**等 10 秒**再起下一个，否则触发 carb mutex 崩溃。
6. **`ls | tail` 是字典序**（`ep_900` 排 `ep_3000` 后），取最新 ckpt 用 `last.pth`。
7. **⚡ 别让两个 Isaac 同时满载 —— 会把整机电源打断电**（2026-07-22 14:46 / 2026-07-24 16:15
   各断过一次，journal 里**没有任何** shutdown/panic/OOM/thermal 记录，是电源 OCP 瞬断）。
   本机 RTX 5090(600W) + i7-14700K(主板把 PL1 解到 253W)，供电余量极薄。已加防护：
   - `rl_rebuild/utils/gpu_guard.py`：**所有 10 个 Isaac 入口**启动前先抢 flock 独占槽位，
     同一时刻只有一个 Isaac 进程能上 GPU；拿不到就排队等，不会并发挤上去。
   - 录像让出协议：`autorecord.sh` 先落 PAUSE 标记 → 训练在 **epoch 边界**（刚存完 ckpt）
     释放槽位挂起 → 录像独占跑完并**完全退出** → 静置 10s → 删标记 → 训练拿回槽位继续。
     训练不丢进度，只是墙钟变长。
   - `train.py` 的 `--num_envs` 有软上限 **1024**（断电那几个 run 实际跑的是 2048，功耗翻倍）；
     确实要更大规模用 `RL_MAX_ENVS=2048` 显式覆盖。
   - `--video_every` 默认 **2000**（原 1000）。
   逃生阀：`RL_ISAAC_NO_GUARD=1` 整个关掉（**只在电源问题确认解决后用**）。

## 当前结论的边界
这 `99.9%` 建立在几个**绕过去而非解决**的问题之上，别当作已解决（详见 `MANUAL.md` §7）：
- pp0 的 8 个 GraspPose 候选**全部 `ok=false`**，策略是自己找到了可行抓法
- GraspPose 的 11 个规划接触点**没有一个是 elastomer**，而只有 elastomer 有高摩擦
- 人手重建腕位误差约 20cm，`pp0_human` 骨干当前不可用
