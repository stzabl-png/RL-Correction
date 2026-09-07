# 交接: Unscrew V5 (clip17) — 拧盖 (v61) 与拔盖 (PULL_v1) 两条线的上手说明

**分支** `unscrew_v5_handoff_20260907` · **时间** 2026-09-07 · **状态** 两条线都在训
（源机 2×TITAN RTX：GPU0 = v60 对照 + PULL_v1，GPU1 = v61），本文档面向**第二个人**：
看懂设计 → 在自己机器上把训练跑起来 → 一起改。全史在
`tasks/Unscrew/part4/A_Design/DECISIONS.md`（本轮 = T2-32 ~ T2-40），上一份交接
`docs/HANDOFF_20260901_unscrew_v5.md` 已被本文档取代（那时还是 clip32）。

---

## 0. 一分钟读懂任务

DexMate + SharpaWave 双臂五指手，IsaacLab，**残差 PPO**：策略在一条 451 行的参考
母带上叠加小幅修正（臂 5mm/步、指按标定界），完成闭环：

初始站姿 → 接近 → 左手抓瓶举升（G1/G2）→ 转平 → 右手拇食双指捏盖 → **打开盖**（G3）
→ 盖放平 + 撤退归位（G4）。

"打开盖"有两种物理，**其余一切逐字相同**（同母带、同双手路径、同奖励、同观测维数）：

| | 拧盖 `screw`（默认） | 拔盖 `pull`（T2-40） |
|---|---|---|
| 装配物理 | 真实螺纹摩擦副：指尖扭矩 > 0.04 N·m 解锁，准静态转动，转满 30° 脱扣 | 轴向塞盖副：指尖轴向力 > 2 N 解锁，准静态滑出，拔满 15 mm 脱扣 |
| 代码 | `tasks/pregrasp/screw_assembly.py` `_thread_friction_step/_finish` | 同文件 `_pull_step/_pull_finish`（同构，逐条照搬防幻影规则） |
| 开关 | `UNSCREW_CAP_MODE=screw`（不设即此） | `UNSCREW_CAP_MODE=pull` + `UNSCREW_PULL_*` |
| `screw_angle` 语义 | 螺纹转角 | 拔出进度（拔满 = 满角），下游零改动 |
| 验收凭据 | `L2_Reference/17/acceptance_v2.json` | `L2_Reference/17/acceptance_v2_pull.json` |
| 世界指纹 | `assembly.cap_mode=screw` | `assembly.cap_mode=pull`（两种 ckpt 互斥） |

抓姿都是用户手调的**拇食双指**（`UNSCREW_CAP_GRASP=user_thumb_index`），不是三指。

---

## 1. 目录地图（只列你会碰的）

```
tasks/Unscrew/part4/
  A_Design/DECISIONS.md            ★设计台账: 每条改动 = 假设 + 证伪针 + 判读, 改之前先读 T2-32~T2-40
  A_Design/L2_Reference/17/        母带 reference_v2.npz (md5 04c2beb1) + 两份验收凭据 + 旧带备份
  A_Design/L3_Learning/progress_batch.py   G 链状态机 (G1~G4, 时钟, RSI 出生表, cert 认证)
  B_SmokeTest/build_reference.py   母带重铸 (T2-39: 右臂外壳离桌约束在这里)
  B_SmokeTest/probe_acceptance.py  零残差全链稳定性门 + 签验收凭据 (换世界/换母带后必跑)
  B_SmokeTest/calib_arm_ff.py      逐行稳态漂移探针 (诊断工具, --iters 0)
  B_SmokeTest/record_reference.py  母带逐行回放录像 (硬写位姿, 不算碰撞)
  C_Wiring/task_env.py             环境: 奖励通道 / 观测 / RSI / 模式开关
  C_Wiring/task_config.py          单一参数源 (clip / 抓姿 / 凭据路径 / 预检)
  C_Wiring/world_fingerprint.py    世界指纹 (ckpt ↔ 世界绑定, 对不上拒跑)
  C_Wiring/train_task.py           训练入口 (预检 → 指纹 → PPO)
  C_Wiring/record_task.py          策略录像 (终止步前读闸门, 见 §6 伪证铁律)
  C_Wiring/launch_remote.sh        ★可移植发射器 (自动找 Isaac Python / 资产 / 预检)
  C_Wiring/recipe_v61.sh           ★当前配方 (source 即可, MODE=pull 切拔盖)
  D_Handoff_20260907/              本次交接的数据: 三个 run 的 world.json + TensorBoard + 最新 ckpt + 发射脚本 + 母带回放视频
tasks/pregrasp/screw_assembly.py   装配物理 (拧/拔两种)
tasks/pregrasp/priors/Screw27_cap_candidates/user_thumb_index.npz   用户双指盖抓姿先验
```

---

## 2. 在你的机器上跑起来（照抄）

前提：带 isaacsim + isaaclab 的 Python（源机是 `/home/feiyang/isaacsim/python.sh`；
`launch_remote.sh` 会自动探测 `~/miniforge3/envs/isaac/bin/python` 等，探不到就
`export PY=/path/to/isaac/python`）。**所有命令都要 `SHARPA_WANDB=0`**（发射器已设）。

```bash
git clone -b unscrew_v5_handoff_20260907 https://github.com/stzabl-png/RL-Correction.git
cd RL-Correction
git lfs install && git lfs pull        # 母带/先验 (*.npz) 与 USD 资产走 LFS, 不拉就是指针文件, 发射器会报 "LFS 文件仍是指针"
# 1) 拧盖 (= v61 配方)
source tasks/Unscrew/part4/C_Wiring/recipe_v61.sh
tasks/Unscrew/part4/C_Wiring/launch_remote.sh Unscrew17HYB_v61_yourname 0 42 HYB 17
# 2) 拔盖 (= PULL_v1 配方)   —— 新开一个 shell, 别在上面那个 shell 里继续 source
MODE=pull source tasks/Unscrew/part4/C_Wiring/recipe_v61.sh
tasks/Unscrew/part4/C_Wiring/launch_remote.sh Unscrew17PULL_v1_yourname 1 42 HYB 17
# 日志 logs/<NAME>.out, pid 在 logs/<NAME>.pid, TensorBoard 在 logs/<NAME>/stage1_tb
```

发射器会先跑**母带预检 + 验收凭据核对 + 世界指纹**，任何一项对不上都直接退出并
打印原因——这是设计如此，不要绕过（`UNSCREW_ALLOW_UNVERIFIED_REF=1` 只给调试用）。
常见拒跑：
- `验收凭据未绑定当前 reference_v2.npz` → 母带或世界变了，重跑
  `probe_acceptance.py --headless`（带同一套 recipe 环境变量；拔盖模式下它自动写
  `acceptance_v2_pull.json`）。
- `assembly.cap_mode: 记录=screw 当前=pull` → 你在用拧盖世界的 ckpt/凭据跑拔盖（或反之）。
- `experience file` 不是 `.headless.kit` → 漏了 `--headless`（发射器带了）。

源机上的原始发射脚本（绝对路径版，逐字配方）在 `D_Handoff_20260907/launch_*.sh`。

冒烟（可选，几分钟，4 env）：`recipe` 之后跑
`B_SmokeTest/probe_acceptance.py --headless`（稳定 ≥3/4 才算过）。

---

## 3. 当前实验设计（读 TB 之前必须知道的）

**奖励 = 母带跟随 + 一串"连续 earn-only 棘轮工资"（过程）+ G 链奖金（终点）**，
每个通道在 TensorBoard 里都有独立账本 `ep_rew/<name>`，判读靠逐项对账，不看总分。

| 通道 | 含义 | 台账 |
|---|---|---|
| `adv/leash/ms` | 沿母带前进 / 物体离母带轨迹的皮带 / 里程 | 早期 |
| `wage`, `cwage` | G1 持握工资；**举升认证窗内 z 上升棘轮**（治好了 cert 通过率衰减） | T2-32 |
| `capap` | 右垫到盖心 12→2 cm 连续接近坡（出生基线免费额，深出生不白领） | T2-34 |
| `capw` | 右垫触盖计件（离散稀有事件，单独不够，配合坡用） | T2-33 |
| `pinchw` | **捏且转/捏且拔**：≥2 右垫在盖 **且** `screw_angle` 当步爬过回合新高 (>0.05°) 才发 0.05/步，不封顶 | T2-36 |
| `screw` | K×Δθ 拧转/拔出势（双向，回退扣分） | 早期 |
| `bonus/gate*` | G1 +5, G2 +8, G3 +10, G4 +15 | 早期 |

**G 链**：G1 = 左手 3/5 垫持握 10 步；G2 = 举升认证（两物体 +5 mm、腕-瓶滑移 <8 mm）；
G3 = 装配脱扣（拧满 30° / 拔满 15 mm）；G4 = 盖放稳 + 回站姿。时钟（母带行推进）
钉在 G2 之前，按物体进度（5 cm/45°）推进。

**RSI 出生表** `['t0','g1','g2','g3','ret','c']`：25% 回合从对准态 c 出生（双臂冻结，
+1 观测旗）。⚠ g3/ret/c 出生**天生**盖已脱扣、`screw_angle` = 满角——见 §6。

**成功率口径**：`sr_t0/*` 是从 t0 出生的真口径；`sr/*` 含 RSI 谱系（`sr/gate4` 有值
不代表闭环成功，`sr_t0/gate4` 才是）。**确定性评测只认 `record_task.py`/eval**。

---

## 4. 三条线的现状与判读针（2026-09-07 14:20 快照）

| run | 世界 | 母带 | 进度 | 角色 |
|---|---|---|---|---|
| `Unscrew17HYB_v60` | 拧盖 | 旧带 (f526f23f, 右前臂穿桌) | ~20M | 几何变量的对照 |
| `Unscrew17HYB_v61` | 拧盖 | **新带** (04c2beb1, T2-39 抬腕离桌) | ~2M | 主线 |
| `Unscrew17PULL_v1` | **拔盖** | 新带 | 刚起 | 预案线 |

**T2-39 的针（v61 vs v60）**：① `diag/cap_any`（触盖率）10M 内应比 v60 同龄段
（0.002~0.004）高一个量级；② `ep_rew/pinchw` 出现持续流水；③ `sr/gate3` 持续非零；
④ cap_any 涨而 pinchw/screw 仍 0 → 瓶颈转到指尖力/摩擦；⑤ `sr/cert_pass`、`sr_t0/gate2`
不得低于 v60。

**T2-40 的针（PULL_v1）**：① 拔盖世界的 `sr/gate3` 应远早于拧盖世界脱零；② 若拔盖也
gate3≡0 → 瓶颈在"够到并捏住"，与拧无关；③ 拔力反作用回瓶身不得打掉 cert_pass/gate2。

三条线约 10M（源机 ~5 h）做第一次判读。读法：`D_Handoff_20260907/<run>/stage1_tb`
或源机 `logs/<run>/stage1_tb`，脚本示例：

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/Unscrew17HYB_v61/stage1_tb', size_guidance={'scalars': 0}); ea.Reload()
for t in ['diag/cap_any','ep_rew/pinchw','ep_rew/screw','sr/gate3','sr/cert_pass','sr_t0/gate2','sr_t0/align']:
    ev = ea.Scalars(t); print(t, [(round(e.step/1e6,1), round(e.value,4)) for e in ev[-5:]])
```

---

## 5. 本轮为什么这么改（三件事，都有数据背书，细节在台账）

1. **T2-36 捏且转工资**：v59 在 23M 冲顶后触盖率退潮 70 倍——盖子三通道都是一次性
   收入，cert 链收益走高后碰盖的死线风险盖过残值。改成"捏且转才发、不封顶"。
2. **T2-37 → T2-39 右前臂穿桌**：以为是控制器 5 cm 漂移（方案 A 前馈），探针证伪：
   自由空间双臂跟踪母带 ~2 mm；真因是母带右臂 IK 在行 212~228 把**前臂**解进桌面
   1.8~3 cm，PhysX 顶臂 3 cm 把手带离盖。修法：重铸母带时加外壳离桌 ≥1.2 cm 约束；
   零空间抬肘无解（旋肘圆极大 −1.3 cm），最终**只抬腕**（≤45 mm，关节改变 ≤10°）。
   验收：稳态漂移全行 0.2~0.4 cm，c 出生手-盖距离 5.4→5.3 cm（旧 5.2→8.8）。
3. **T2-40 拔盖**：用户预案，作为"拧"这个动作是否是唯一瓶颈的对照实验。

---

## 6. 铁律（两次被咬过的坑，不要再犯）

- **伪证铁律**：IsaacLab 在 `step()` 内对终止 env **当场重生**。终止步之后读到的
  闸门/角度/画面全是**下一回合的出生态**；抽到 g3/ret/c 谱系就会"看见"盖已拧下、
  `screw_angle` 恰好满角。任何录像/探针/评测：先判 done，**done 之后一律不读不渲染**；
  "拧开/拔开"的视频必须附逐步爬升的角度/拔出量自证。
- **一次只改一个变量**，改之前把假设 + 证伪针写进 `DECISIONS.md`，训完按通道对账。
- **世界指纹与验收凭据是防呆，不是障碍**：对不上就是世界变了，重签而不是绕过。
- 别用 `pkill -f <字符串>` 杀 Isaac（字符串在自己命令行里就自杀）；按 pid 杀，
  杀完等 10 s 再起下一个 Isaac 进程（carb mutex）。Isaac 脚本结束常卡在
  `app.close()`，按 pid 收尾即可。
- 训练必须 `--headless`；`SHARPA_WANDB=0`；ckpt 取 `last.pth`（`ls | tail` 是字典序）。

---

## 7. 协作约定（两个人一起改）

- 在本分支上开各自的分支 `unscrew_v5_<name>_<topic>`，回合并到本分支。
- 每个改动 = `DECISIONS.md` 新增一条 `T2-xx`（假设 / 实装 / 验证 / 预注册针），
  run 名带 `T2-xx` 号；对照 run 不动。
- 改母带 → 重跑 `build_reference.py` → `probe_acceptance.py` 重签凭据 → 旧带备份到
  `L2_Reference/17/_backup_<why>_<date>/`。
- 改装配物理 → 同时更新 `world_fingerprint.py` 的 CRITICAL 项（旧记录补默认值），
  并跑拧/拔两种冒烟（拧盖回归不能变数字）。
- 拧盖与拔盖**共用一份代码**，模式开关是唯一分叉点；别复制环境。
