# 交接: Unscrew V5 (clip32) — 右手重定向打通后的剩余验证

**分支** `unscrew_v5` · **时间** 2026-09-01 · **状态** 代码与机器段产物已就绪，
**缺 v2 母带 + 验收凭据**（两步都在目标机上跑，不需要 cuRobo）。

本机 GPU 被其他任务占满，故把剩余三步交接出去。全史见
`tasks/Unscrew/part4/A_Design/DECISIONS.md`（T2-6 ~ T2-11 是这一轮）。

---

## 一、这一轮改了什么（为什么必须重做 v2 与凭据）

1. **右手抓握重定向**（T2-10，最关键）：右手过去没有抓取先验（βR=0，指形直接用
   人手指流），结果**全程碰不到盖** → 真实螺纹副下扭矩传不进去 → `ep_rew/screw`
   恒 0（首训 8.5M 步实测零信号）。现在接上 `Screw27_cap_candidates`（镜像到右手
   + 物体系原点修正 + 闭环对准 + 捏合量），零动作实测：
   **右垫 2/5、三指 2/3、指力 7.58N、螺纹角 13.7°**，再合 10° 即拧满脱扣。
2. **拧开角 0.75 圈(270°) → 30°**（T2-11，用户裁定：已破封的松盖）。
   `assembly.turns` 是世界指纹的 **CRITICAL** 项 ⇒ 旧验收凭据必然对不上。
3. **螺纹物理换成 U45 准静态**（合并 `unscrew_v5_u44fix`）：
   `ω=(|τ|−kinetic)⁺/b`、`I_eff 5e-3→5e-4`、安全夹 4.0→2.5。飞轮已死
   （撤力滑行 7.3°→1.2°），"戳转"物理上失效。
4. **机器段真正走到操作位置**（T2-8 的能力这轮生效）：Approach/Retreat 各 **161 行**
   = 满障碍腿(81) + 贴物腿(81，目标物体排除出碰撞世界)。缝1 因此退化成
   **"臂不动、只合手指"**，原先手写的 16cm 笛卡尔进刀（碰倒瓶的来源）退役。
   母带全链 **476 行**（app161/seam1 25/ia103/seam2 25/ret162）。
5. **课程死锁修法**（T2-9）：渐进 RSI 的解锁要求"**t0 口径**的 G1 ≥30%"，而 t0
   够不着 G1 ⇒ 永不解锁 ⇒ 100% 回合都是 t0。发射时带 `POUR_UNLOCK=1,2,3` 直接
   开满出生点（t0 仍占 20%）。

---

## 二、目标机上要跑的三步（照抄即可）

前置：
```bash
git fetch origin && git checkout unscrew_v5 && git pull && git lfs pull
export PY=/path/to/isaac/python          # 脚本会自动找 miniforge3/miniconda3 下的 isaac 环境
bash tasks/Unscrew/part4/C_Wiring/verify_deploy.sh 32   # 这一步现在会报"缺 v2"——正常, 下面就是补它
```

统一环境变量（下面每条命令都带）：
```bash
COMMON="CUDA_VISIBLE_DEVICES=0 TMPDIR=$HOME/tmp UNSCREW_CLIP=32 SHARPA_WANDB=0 \
        OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=."
```

### 步骤 1 — 重铸 v2 母带（约 10~15 分钟，不需要 cuRobo）
```bash
env $COMMON $PY -u tasks/Unscrew/part4/B_SmokeTest/build_reference.py --headless
```
**通过标准**：打印 `[v2] 已写 ... reference_v2.npz md5=...`；
`交互IK` 的"关键窗坏行"个位数、冻结行 R<40/L<10 属正常。
⚠ 跑完进程可能卡在 `app.close()`（本仓库已知病）——产物落盘后 `kill -9` 即可，
杀完等 10 秒再起下一个。

### 步骤 2 — 出验收凭据（约 10 分钟）
```bash
env $COMMON $PY -u tasks/Unscrew/part4/B_SmokeTest/probe_acceptance.py --headless
```
**通过标准**：`★训练稳定性: 4/4（要求≥3） => ✅`，并写出
`A_Design/L2_Reference/32/acceptance_v2.json`（绑定新 v2 的 MD5 + 完整世界指纹）。
下面的"可训练性告警"是**诊断**不阻塞；这一轮**期望看到它比上一轮好**：
右手不再是 `站位垫 R0/5`、`回放拧角 0.0°`。

### 步骤 3 — 零动作冒烟（约 15 分钟，可与训练并行前先做）
```bash
env $COMMON POUR_SQUEEZE_FF=1 $PY -u tasks/Unscrew/part4/C_Wiring/smoke_zero.py \
    --steps 600 --headless
```
**通过标准**：
- `A 静置对账` 瓶/盖 均 <5mm（铁则）
- `机器段死线误触 = 0`（铁则；范围=cuRobo 背书的 `row<APP_END`）
- `缝1 交接段死线 = N` 是**记账项**，不阻塞
- 本机中途读数已看到：站位出生的 env `垫L=4`、**零动作下 `screw` 自己在涨**
  （10.1°/5.1°，满 30° 脱扣）——目标机上应能复现

### 步骤 4 — 发射训练
```bash
POUR_UNLOCK=1,2,3 bash tasks/Unscrew/part4/C_Wiring/launch_remote.sh \
    UnscrewHYB_v45 0 42 HYB 32
POUR_UNLOCK=1,2,3 bash tasks/Unscrew/part4/C_Wiring/launch_remote.sh \
    UnscrewOBJ_v45 1 42 OBJ 32
```
⚠ **`POUR_UNLOCK=1,2,3` 不能省**——否则会复现 T2-9 的课程死锁（8.5M 步零信号）。

---

## 三、判读针（训练开跑后盯这几条，已预登记在 DECISIONS）

| 针 | 期望 | 破了说明什么 |
|---|---|---|
| `sr/gate1` (H6.7) | +2M 内破 20% | 问题不在出生点而在交互段本身 |
| `diag/screw_tau_mNm` + `diag/cap_any` (H6.2) | 零接触步上力矩≈0 | **幻影力矩通道**，立刻停训 |
| `ep_rew/screw` | 应显著 >0（这轮的核心指标） | 右手又没抓住 |
| `sr_t0/gate1` | 随训练上升 | t0 出生点学不会自己合拢（T2-6e 的未解项） |
| `diag/carry_steps` | release 后能涨到 ≥10 | 策略拧开就撒手（U41② 会拦住 placed） |

**成功率只认独立评测** `tasks/Unscrew/part4/C_Wiring/eval_task.py`，训练期 TB 的
`sr/gate4` 被 `ret` 出生点污染（那个出生点预置 placed，只要撤退就算 G4）——
看无偏口径 `sr_t0/*`。

---

## 四、已知未解 / 下一轮候选

1. **t0 出生点在缝1 合拢瞬态会碰倒瓶**（T2-6e）。铁则范围已收回 cuRobo 背书那段；
   缝1 单独记账。这一轮机器段改成端到端到操作位置后**可能已经缓解**（缝1 不再
   有 16cm 横扫），目标机的冒烟可以复核 `缝1 交接段死线` 是否降到 0。
2. **右手三指只到 2/3**。拇指/中指还差约 1cm 收拢，交给 RL 手指残差（±68.8° 总量）。
   若训练里 `diag/n_triad` 长期 <1，再回头调 `CAP_PINCH_DEG` / 换候选
   （`UNSCREW_CAP_GRASP=<前缀>` 可钉死某个候选；`probe_capgrasp --fit --curl` 复测）。
3. **Retreat 的第二腿**已能规划（本轮 161 行），此前的"Approach 倒放"兜底不再触发。

## 五、这一轮新增的工具（复现/复测都靠它们，别猜）

| 工具 | 用途 |
|---|---|
| `B_SmokeTest/probe_capgrasp.py` | 遍历盖抓取候选，自解腕 IK + 套指形 + 钉瓶，**直接量接触**；`--fit` 闭环对准、`--curl` 逐级捏合、`--pick` 钉候选 |
| `B_SmokeTest/probe_pinch.py` | 站位行逐级捏合，报三指接触/指力/组件扰动 + 实际腕位对账 |
| `B_SmokeTest/probe_grasp.py` | 左手抓握几何 + 三方对账（命令/实际/离线 IK）+ 逐行盯瓶 + `--pin_bottle` |
| `B_SmokeTest/probe_thread.py` | 真实螺纹副五段物理探针（静置/阈下/阈上稳态/回锁/脱扣） |

标定值都在 `C_Wiring/task_config.py`，每个都带"是怎么量出来的"的注释：
`CAP_GRASP_PICK` / `CAP_GRASP_TRIM`（**盖坐标系**）/ `CAP_PINCH_DEG` /
`PRIOR_RADIAL_TRIM` / `HAND_DROP` / `HOLD_YAW_BY_CLIP` / `SEAM_MOVE_FRAC`。
换 clip 或换手都要用上面的探针复测。
