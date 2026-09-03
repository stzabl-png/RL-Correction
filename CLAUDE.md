# RL_Correction

**动手之前先读 `docs/MANUAL.md`**（怎么跑、模型结构、踩过的坑）。
**两套训练设定 A/B 的分界见 `docs/TRAINING_SETUPS_A_B.md`**（A=GraspPose 能处理的物体/普遍偏大；
B=GraspPose 处理不了的小/扁物体，靠 affordance 学指尖抓取）。
当前进度快照见 `docs/HANDOFF_*.md`（取时间戳最新的那份）。
**要改设计（残差界/奖励/课程）之前先读 `docs/DESIGN_LOOP.md`**：每条设计都配了可证伪信号，
改动上线前先把假设和证伪条件写进台账；训完读 TensorBoard 逐项账本 (ep_rew/* + term/*)
判读是哪条假设被数据推翻。一次只改一个参数，否则判读表失效。

**2026-07-30 起当前任务 = `tasks/pregrasp/`**（PreGrasp 稳定抓握+微抬升验证,确定性评测
99.95%,全过程见台账）。入口: `tasks.pregrasp.{train,eval,record,play,smoke}`。
旧 correction 任务的入口/诊断脚本已清理（git 历史可寻），`rl_rebuild` 只余引擎+被继承的底层 env。

## 一句话
PPO 训练**残差策略**：在参考手部轨迹上叠加小幅修正，让 DexMate+SharpaWave 在 IsaacSim
里完成 端到端 Approach+Pick（对称站姿→接近→抓稳→抬升）。**冠军配方已复现盖章**
（2026-08-04，端到端确定性 99.99%/100.00%/93.93% 三 seed，配方见 `MANUAL.md` §8）。
**全部成功成果收录在 `results/part2/basic_pick_place/`**（按数据集物体号组织，含 ckpt/
TB/评测/录像；旧 `logs/Grasp0` 等已迁入，见该目录 README 的新旧对照表）。

## 立刻会犯错的几件事

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python    # 唯一带 isaacsim+isaaclab 的解释器
```

1. **所有命令加 `SHARPA_WANDB=0`**，否则 wandb 弹交互式登录卡死进程。
2. **成功率只认确定性评测 `tasks/pregrasp/eval.py`**。训练期 TensorBoard 的成功率
   带探索噪声和(旧任务)分母问题;pregrasp 任务还要注意 TB 口径含 c0 随机化脚手架, 会**低估**。
3. **改动作界 / sigma 之前先读 `MANUAL.md` §6.1**。残差界 × sigma = 每步扰动量，
   这个任务的容忍度是**毫米级**；放大过就是接触率 0.1%、物体被打飞。
4. **`ppo.py` 里那几个 `.detach()` 别删**（§6.4），删了立刻泄漏几个 GB 直至 OOM。
5. **杀 Isaac 进程后要复查**：可能训练循环停了但卡在关闭流程里仍占显存，需要 `kill -9`；
   杀完**等 10 秒**再起下一个，否则触发 carb mutex 崩溃。
   **5b. 起过的 Isaac 进程也要回头看它退没退**（2026-08-31 补）。上面这条只覆盖
   "我主动杀之后"，不覆盖"起完就不管"。一次性脚本（`build_ref_v5` / `probe_*` /
   `eval_pour` / `smoke_*`）跑完常卡在 Isaac 关闭流程里**不退**，继续占显存和
   `gpu_guard` 的 flock。当晚一次踩到：`build_ref_v5` 正常 10 分钟的活挂了
   **9 小时 48 分**，一直占着本机 GPU，害我误判"造带很慢"。同一晚在 taitan 和
   Denso 也各遇到一次。**判据：非训练的 Isaac 进程 `etimes > 1800` 一律当僵尸清。**
   守夜脚本 `scratchpad/watchdog.sh` 已内置这条巡检。
6. **`ls | tail` 是字典序**（`ep_900` 排 `ep_3000` 后），取最新 ckpt 用 `last.pth`。
   没有 `last.pth` 时用 **`ls ep_*.pth | sort -V | tail -1`**（`sort -V` 按版本号排）。
   实测：`ls|tail -1` 在 10 个 ckpt 里取到的是 **2M** 那个，不是 13M 那个。
   **6b. `pgrep -f` / `ps|grep <模式>` 会匹配到"提到这个模式的进程"，不只是目标进程**
   （2026-08-31 一夜踩了 **5 次**，其中 2 次杀掉了我自己的 ssh 会话，退出码 255）。
   四种变体都遇到过：① 命令匹配自己 ② 父 shell 的命令行里含着子进程要 grep 的字符串
   ③ 写脚本的 heredoc 里含着脚本自己要 grep 的模式 ④ **远端 shell 的命令行里含着
   目标名**（`ssh host 'ps|grep probe_placed'` —— 这条 ssh 自己就叫 probe_placed）。
   `[p]robe` 方括号技巧**只防①**，防不住②③④。
   **可靠写法：按 `comm` 过滤，bash 永远匹配不上：**
   ```bash
   ps -u $USER -o pid=,comm=,cmd= | awk '$2 ~ /^python/ && /train_pour\.py/ {print $1}'
   ```
   或者干脆记下 PID 按 PID 操作。
7. **训练必须加 `--headless`**。`train.py` 不会自己设（不像 `dexmate_baseline.py` 那些脚本），
   漏了就加载 `isaaclab.python.kit` 而不是 `.headless.kit`，1024 env 开渲染会在建场景阶段
   被 OOM killer 杀掉（退出码 137，日志停在 `[setup] object 刚体化`，run 目录都来不及建）。
   核验方法：日志里 grep `experience file`，必须是 `.headless.kit`。
7. **⚡ 电源红线（2026-07-31 起已放宽）**：本机 GPU 已从 RTX 5090(600W) 换成
   **RTX 4080 SUPER 16GB(320W)**，配 i7-14700K(主板把 PL1 解到 253W)。旧卡时代两次整机
   瞬断（2026-07-22 14:46 / 2026-07-24 16:15，journal 里**没有任何** shutdown/panic/OOM/
   thermal 记录，是电源 OCP）不再适用于新卡。**新的约束是显存 16GB，不是功耗。**
   下面的防护仍然保留（避免两个 Isaac 抢 GPU 仍然有意义，只是不再是断电风险）：
   - `rl_rebuild/utils/gpu_guard.py`：**所有 10 个 Isaac 入口**启动前先抢 flock 独占槽位，
     同一时刻只有一个 Isaac 进程能上 GPU；拿不到就排队等，不会并发挤上去。
   - 录像让出协议：`autorecord.sh` 先落 PAUSE 标记 → 训练在 **epoch 边界**（刚存完 ckpt）
     释放槽位挂起 → 录像独占跑完并**完全退出** → 静置 10s → 删标记 → 训练拿回槽位继续。
     训练不丢进度，只是墙钟变长。
   - `train.py` 的 `--num_envs` 软上限 **1024**（当年为限功耗设的）；换卡后瓶颈变成 16GB
     显存，要更大规模用 `RL_MAX_ENVS=2048` 覆盖，**但先确认显存够**（5090 是 32GB）。
   - `--video_every` 默认 **2000**（原 1000）。
   逃生阀：`RL_ISAAC_NO_GUARD=1` 整个关掉。
   吞吐参考：5090 约 2600~3600 FPS，4080S 约 1800~2400 —— 墙钟要乘 1.5。
8. **两条通用设计（2026-08-31 用户裁定，适用所有任务，台账 L5-34/L5-36）**：
   - **G-A 物理规矩：物体 0.1kg、物体摩擦 5、指垫摩擦 5**（multiply 合成手↔物 25）。
     默认值在 `rl_rebuild/correction/env/correction_env.py` 的 `PHYS_RULE`（导入期 setdefault，
     基类 `__init__` 覆写主体物，pour 补杯），**不要靠发射脚本手填**——L5-31~33 三批就是
     这么静默丢回硬物理的（瓶 0.53/μ3.0、杯 0.15/μ0.5、指垫 3.0）。核验：训练日志 grep
     `难度覆写`，必须有 `质量=0.100kg 摩擦=5.00/5.00` 两行 + `指垫摩擦覆写 = 5.0`。
   - **G-B 接触起始黄窗：交互行 `[0, r_lift+3]` 档位上限黄**（`tier=min(tier,1)`，只放松不收紧），
     `r_lift` = 参考物体首次抬升 ≥ CERT_RISE 的行，**按母带算不写死**（v3 = 0~18 行）。
     实现 `rl_rebuild/correction/tier_floor.py`，`POUR_TIER_FLOOR_START=auto|<n>|0`；
     对照开关 `POUR_TIER_SHUFFLE=<seed>`（主对照，替代拍平）/ `POUR_TIER_REVERSE=1`。
     核验：日志 grep `G-B 接触起始黄窗: 交互行 0~18`。
   - 回放/探针脚本一律走 `world_fingerprint.restore_physics_env()` 按 ckpt 的 world.json 还原
     物理；档位三项进指纹 CRITICAL。新老世界成绩**不可比**（L5-13 判读限制）。
   - 当前计划与数据在 `ImproveBase/`（PLAN.md / runs/ / launch_logs/ / probes/）。

## 当前结论的边界
这 `99.9%` 建立在几个**绕过去而非解决**的问题之上，别当作已解决（详见 `MANUAL.md` §7）：
- pp0 的 8 个 GraspPose 候选**全部 `ok=false`**，策略是自己找到了可行抓法
- GraspPose 的 11 个规划接触点**没有一个是 elastomer**，而只有 elastomer 有高摩擦
- 人手重建腕位误差约 20cm，`pp0_human` 骨干当前不可用
