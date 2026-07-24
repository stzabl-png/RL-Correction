# 实现框架（可编辑骨架 · 我们一起填）

> 把 [设计总览](PHASE1_RESIDUAL_CORRECTION_DESIGN.md) 落成"模块 + 数据契约 + 函数签名"。
> 这是**骨架草稿**：签名/字段先占位，`TODO`/`❓开放点` 标出要一起定的地方。确认结构后再生成真 `.py` stub。
> 约定：所有位姿 = 位置(3)+四元数wxyz(4)=7；米/弧度/z-up；帧对齐见 [FRAME_ALIGNMENT.md](FRAME_ALIGNMENT.md)。

---

## 0. 概念流水线（5 阶段 · 用户框架 · 丰富版）

**① 视频重建**　输入=视频 → 输出=物体 & 人手 mesh 轨迹（noisy）
- Reconstruct_and_Retarget Path B：ViPE(度量深度+SLAM) → HaWoR(MANO 手) → FoundationPose(物体6D)。
- ⚠️ 输出是**人手 MANO** mesh，不是 Sharpa。

**② 数据准备**（备齐一个 case + 对齐到同一坐标系）
1. Noisy 重建①（=①的输出）
2. **重定向 ①→Sharpa**（magicdexmate，MANO→22指）　⚠️**原框架漏了**：RL 吃 Sharpa 手，人手 mesh 必须先重定向
3. 优化抓取位姿②（Affordance → OCIR/BODex sharpa_right，干净 grasp+接触）
4. 可行抓取轨迹③（cuRobo sharpa_right，规划到②）
5. 坐标系对齐（四系统→env 系；都 z-up/wxyz，主要是重锚+四元数序+帧率，见 FRAME_ALIGNMENT.md）

**③ PPO on-policy 残差 RL**：像"人"① + 可行(②③) + 物体特征
- 残差锚在③；追踪①；物体点云当物体特征。
- ⚠️ **"像人"必须分通道**：只追①的**可信通道**（物体粗轨迹+粗手运动），**不追**噪声大的手指细节（交②③）。
- **3.1 非对称 critic**：Actor 看 obs，Critic 拿 sim 真值（穿模/物体真速度/摩擦）✅
- **3.2 手写 Grasping Reward**，后期 LLM 生成（reward-反馈迭代更对口 **Eureka** ＞ RoboGen[生成任务/场景]）
  - **3.2.1 分段 reward**：轨迹切段（接近→抓取→抬升→操作），每段一个 reward+靶子（接近追①手/腕、抓取追②、抬升/操作追①物体）✅；⚠️段边界靠**完成度事件**判（接触建立=抓取完成、物体离面=抬升完成）
- **3.3 I/O**：输入=重建①+可行③+物体点云 → 输出=一条修正轨迹（并行多条）✅
  - 补：动作=28-DOF 残差PD、腕=wrench-PD、RSI复位、物理松弛课程、硬ET、成功=撤支撑测试

**④ 输出判成功**
- 输出轨迹**本就是 sim rollout 结果**，直接读 sim 参数（穿模/滑移/撤支撑）判简单任务　⚠️不用"再在 sim 里跑一遍"（除非 replay 存档轨迹二次验证）
- 复杂任务：**渲染 sim 画面 + 数值参数 → VLM 判断**　⚠️不是"encoder→latent→VLM"（VLM 吃图/文，不吃任意 latent）

**⑤ 基于判断更新 Grasping Reward**（闭环，Eureka 式）
- VLM 判断 → 调 reward 权重/项 → 重训。
- ⚠️ **很贵的外循环**（每次更新=重训一次策略）→ **Phase-3 元层**；且按**任务类**更新、**非每条轨迹**。MVP 先手写 reward 跑 ①-④，⑤后置。

### ⚠️ 指出不对（汇总）
1. 漏**重定向**（②）：MANO→Sharpa 是①进 RL 的必经步。
2. **"像人"要分通道**：噪声参考不能整条贴追，只追可信通道，手指细节交②③（上轮核心结论）。
3. **物体点云**：单物体 MVP 其实不需要（几何烤进权重），是**跨物体泛化**才用的"物体特征"；现在放也行但属多余复杂度。
4. **④ VLM 吃 latent 不对** → 渲染图像+数值参数喂 VLM。
5. **④"在 sim 里运行获取参数"**：修正本身就在 sim 跑，参数当场有，不必再跑。
6. **⑤ 闭环贵**：Phase-3 元层，按任务类离线做，别按样本做。
7. 参考：reward-反馈迭代看 **Eureka**（RoboGen 是生成任务/场景）。

---

## A. 模块布局（proposed，可改）

```
rl_rebuild/correction/
  schema.py            # 数据契约: FrameConvention / RefTrajectory / GraspTarget / DataUnit / OutputTrajectory
  frames.py            # B1: 单一坐标约定 + 变换 + 002aa185 单测
  producers/
    recon_biv2ap.py    # ① video HOI recon (Path B: world_fused.npz) -> 手MANO + 物体6D
    retarget_mdm.py    # ①-手指: magicdexmate 封装 -> Sharpa 22 关节
    grasp_ocir.py      # ② OCIR/BODex(sharpa_right) -> GraspTarget (+接触)
    traj_curobo.py     # ③ cuRobo(sharpa_right) 规划到 ②grasp -> 可行接近轨迹(锚点)
  assemble.py          # 把 producers 对齐到 env 系, 拼成 DataUnit
  env/
    correction_env.py      # RL 修正 env (改/继承 SharpaWaveGraspXLEnv)
    correction_env_cfg.py
    reward.py              # 乘法 imitation×接触 reward
  harvest.py           # B7: 训练好的策略 rollout(t=0 完整) -> OutputTrajectory
  train.py             # 复用 rl_rebuild/scripts/train.py 的 PPO, 换 task
```

复用（不重写）：`rl_rebuild/algo/ppo`(PPO)、`algo/models`(v3net)、`tasks/inhand_rotate/sharpa_wave_env.py`(力矩PD/触觉/浮动手基座)、`graspxl/grasp.py`(四元数)。

---

## B. 数据契约（dataclass 草稿 — 先定这个，后面都挂在它上）

```python
# schema.py
from dataclasses import dataclass
import numpy as np

@dataclass
class FrameConvention:
    up_axis: str = "z"          # z-up
    quat_order: str = "wxyz"    # scalar-first
    units: str = "m"            # 米/弧度
    # 权威 TARGET = IsaacLab env-local (world - env_origins), 锚 hand_init_pose

@dataclass
class GraspTarget:               # ② OCIR/BODex, 抓取段"可信目标"
    finger_q: np.ndarray         # (22,) 目标手指构型 (SHARPA_USD 序, 弧度)
    wrist_pose: np.ndarray       # (7,) 目标手根位姿 (env 系)
    contact_fingers: np.ndarray  # (5,) bool 哪几指该接触   ← MVP 用这个(标量力可判)
    contact_points: np.ndarray | None = None   # (P,3) env 系, 可选(critic/精修用)
    contact_normals: np.ndarray | None = None  # (P,3)
    grasp_phase_frame: int = 0   # ③ 轨迹里"到达抓取"的帧号  ❓相位对应见 F

@dataclass
class RefTrajectory:             # 对齐后的参考束 (长度 L, 共享相位时钟=③)
    # ③ cuRobo 可行轨迹 = 残差锚点 + RSI 源
    anchor_wrist:  np.ndarray    # (L,7)
    anchor_finger: np.ndarray    # (L,22)
    # ① 视频可信通道 = 松追踪目标 (保留"那次操作" b)
    track_object:  np.ndarray    # (L,7)   物体粗轨迹 (Reconstruct_and_Retarget Path B)  ← 主追踪
    track_wrist:   np.ndarray    # (L,7)   手腕粗运动 (松追踪+接近引导)
    # ①-手指(magicdexmate): 低可信, 默认不追, 存着备用/诊断
    human_finger:  np.ndarray | None = None   # (L,22)
    grasp: GraspTarget = None    # ② 抓取段目标
    # ❓相位时钟: 用 ③ 帧索引; ① 已重采样到同一 L; 完成度门控见 §2.6

@dataclass
class DataUnit:                  # env 每个 case 的完整输入
    object_id: str
    mesh_path: str
    object_init_pose: np.ndarray # (7,)
    ref: RefTrajectory
    goal_object_pose: np.ndarray # (7,)  lift 目标  ❓delta-z vs 6D, 见 F
    fps: float
    frame: FrameConvention

@dataclass
class OutputTrajectory:          # 收割的数据产品 (数据引擎输出)
    hand_root_pose: np.ndarray   # (T,7)
    finger_q:       np.ndarray   # (T,22)
    object_pose:    np.ndarray   # (T,7)
    timestamps:     np.ndarray   # (T,)  重采样到视频时间轴
    contacts:       dict | None  # {points,normals,forces} per frame, 可选副产物
    success: bool                # 撤支撑测试通过
    metrics: dict                # {penetration_iv, slip, object_fidelity_to_gt, contact_iou_ocir, ...}
    meta: dict                   # {object_id, source_video, frame, fps}
```

---

## C. 各模块骨架（签名 + TODO）

```python
# frames.py  (B1 — 一切的前提, 先写 + 单测)
def recon_to_env(pose_world, anchor, yaw, env_origin) -> np.ndarray:
    """Reconstruct_and_Retarget world_fused(gravity_z_up_world) -> IsaacLab env-local.
    重心化(减anchor)+re-yaw(摆进hand_init_pose)+加env_origin; 手&物体同一变换保HOI几何."""
    # TODO: 实现; anchor=物体质心或t0抓取腕位; yaw=R_z(θ)
def xyzw_to_wxyz(q): ...          # Reconstruct_and_Retarget HaWoR handoff 是 xyzw
def resample(traj, src_fps, dst_fps): ...   # 手30->物体/视频15
def bodex_to_env(robot_pose, obj_scene_pose, obj_env_pose, env_origin) -> np.ndarray:
    """BODex world -> env; 变 object-相对 T_obj_hand=inv(P_obj)@P_hand; +90°about-X 对账."""
def selftest_002aa185():
    """断言: 一条 replay_world 的手+物体落进 env 后, 手指↔物体几何关系不变(穿模量一致)."""
    # TODO
```

```python
# producers/recon_biv2ap.py   ①
def load_recon(seq_dir) -> dict:
    """读 world_fused.npz/replay_world.npz(Path B, 非 ego_pipeline).
    返回 {hand_trans(2,N,3), hand_rot_aa(2,N,3), object_pose(M,4x4), fps_hand=30, fps_obj=15, side[0]=左[1]=右}."""
    # TODO: 只取右手[1]; object 4x4 -> 7(wxyz)

# producers/retarget_mdm.py   ①-手指
def retarget(hand_keypoints_21) -> np.ndarray:   # (22,) per frame
    """magicdexmate: 21关键点(只用腕+5指尖) -> Sharpa 22指关节(SDK序). 逐帧.
    ❗欠约束(中节/外展/CMC 优化器猜) = 不可信来源之一."""
    # TODO: 封装 magicdexmate.retarget; 重排 SDK->USD 序

# producers/grasp_ocir.py     ②
def gen_grasp(mesh_path, affordance=None) -> GraspTarget:
    """OCIR/BODex(sharpa_right) 出干净 grasp+接触.
    MVP: 先用 poses/n4_pose1.npy 顶替(002aa185); OCIR clone 后替换."""
    # TODO: 读 BODex .npy(stage=1) -> GraspTarget; 或读 n4_pose1

# producers/traj_curobo.py    ③ (锚点)
def plan_traj(mesh_path, grasp: GraspTarget, start_pose) -> tuple:
    """cuRobo(sharpa_right) 规划 open->grasp(->lift) 可行接近轨迹.
    返回 (anchor_wrist(L,7), anchor_finger(L,22)). = 残差锚 + RSI 源."""
    # TODO: 驱动 cuRobo; ❓MVP 可先授权直线插值顶替(见 F)
```

```python
# assemble.py
def build_data_unit(seq_dir, object_id, mesh_path) -> DataUnit:
    recon  = load_recon(seq_dir)                      # ①
    fingers= [retarget(kp) for kp in recon_keypoints] # ①-手指
    grasp  = gen_grasp(mesh_path)                     # ②
    anchor = plan_traj(mesh_path, grasp, start)       # ③
    # -> 全部经 frames.py 对齐到 env 系; ① 重采样到 ③ 的 L; 拼 RefTrajectory
    # TODO: 相位对齐 ❓(F); 组装 DataUnit
```

```python
# env/correction_env.py   (继承/改 SharpaWaveGraspXLEnv)
class CorrectionEnv(SharpaWaveGraspXLEnv):
    def _pre_physics_step(self, a):        # a=(N,28): [腕pos3,腕rot3, 指22]
        # 残差锚在 ③: wrist_tgt = anchor_wrist[phase] ⊕ scale_w·a[:6]
        #             q_finger_tgt = anchor_finger[phase] + scale_f·a[6:]
        # 腕 SE(3) wrench-PD -> set_external_force_and_torque; clamp F_max/T_max
        # 指 力矩PD (复用父类)
        ...  # TODO
    def _get_observations(self):
        # 相对量: (anchor - 当前)在{t,t+1,t+5} + (track_object-obj) + (②grasp-当前)
        # + 本体感觉/腕位姿速度/触觉/手系重力 ; 归一化交 RunningMeanStd
        ...  # TODO
    def _get_rewards(self):
        return compute_correction_reward(self, ...)   # reward.py
    def _reset_idx(self, ids):
        # RSI: 随机相位 t <- 相位受限(早期接触前); 写手/物体=③[t]; 速度=③差分; 物体动态
        ...  # TODO
    def _get_dones(self):
        # 硬ET: 掉落/穿模/腕越界/接触力=0持续K步 ; 成功=撤支撑测试
        ...  # TODO
```

```python
# env/reward.py   乘法 imitation×接触
def compute_correction_reward(env, ref, priv):
    r_hand   = exp(-k_h * hand_track_err)      # 追 ②/③(抓取严格) 或 ①粗手(操作松)
    r_object = exp(-k_o * object_track_err)    # 追 ①物体粗轨迹, k_o << k_h (松)
    r_contact= contact_match(force, target_set) # D-Grasp r_c: 达成接触集 + min(力,5·m·g)
    r_imit   = r_hand * r_object * r_contact    # 乘法(掉接触->归零)
    r_reg    = -(w_pen*penetration + w_ov*obj_vel + w_sm*smooth + w_ws*wrist_oob)
    return r_imit + r_reg                       # ❓乘法/加法组合方式 (F)
    # 物理松弛课程: 重力/摩擦/阈值 由 env.curriculum 控
```

```python
# harvest.py   (B7 数据引擎输出)
def harvest(policy, data_unit) -> OutputTrajectory:
    # 从 t=0 完整 rollout(确定性 mu); 记录 手/物体/接触 per frame
    # 撤支撑测试 -> success; 重采样到视频时间轴 + 低通(alpha=0.4)
    # 算 metrics(穿模改善/滑移/物体对GT保真/接触IoU); 写 OutputTrajectory
    ...  # TODO
```

---

## D. 一个 case 的数据流（端到端）

```
seq_dir(视频重建) ─► load_recon ─┐
                                 ├─ retarget ─► ①手指
mesh ─► gen_grasp(②) ─► plan_traj(③锚点)                    frames.py 对齐
        └─ recon物体/粗手 = ①可信通道 ─────────────────────► build_data_unit ─► DataUnit
                                                                                   │
DataUnit ─► CorrectionEnv(残差锚③, 追①物体+②抓取, 乘法reward, RSI, ET) ─PPO─► policy
                                                                                   │
policy ─► harvest(t=0 rollout + 撤支撑测试 + 重采样) ─► OutputTrajectory(干净可行数据)
```

---

## E. 对应表（设计 § ↔ 模块 ↔ 契约 ↔ 状态）

| 设计 § | 流水线阶段 | 模块 | 输入→输出契约 | 状态 |
|---|---|---|---|---|
| §5,§FRAME | 帧对齐 | `frames.py` | 各系统位姿 → env 系 | B1 先写 |
| §1,§5 ① | 视频重建 | `producers/recon_biv2ap.py` | seq_dir → 手+物体(Path B) | 代码在,未接 |
| §1,§5 ①指 | 重定向 | `producers/retarget_mdm.py` | 21kp → 22关节 | 已可用 |
| §1,§5 ② | 抓取优化 | `producers/grasp_ocir.py` | mesh(+affordance) → GraspTarget | OCIR未clone,n4顶替 |
| §1,§5 ③ | 可行规划 | `producers/traj_curobo.py` | mesh+② → 锚点轨迹 | cuRobo配置在,未接 |
| §3 | 组装 | `assemble.py` | producers → DataUnit | 待写 |
| §2.1-2.7 | RL 修正 | `env/correction_env.py`+`reward.py` | DataUnit → policy | 待写 |
| §2.6算法 | 训练 | 复用 `scripts/train.py` | env → ckpt | 复用 |
| §4 | 收割 | `harvest.py` | policy+DataUnit → OutputTrajectory | 待写 |

---

## F. ❓开放点（填的时候一起定）

1. **相位时钟 & ①↔③对齐**：③(cuRobo)定相位, ①重采样到同 L? 还是完成度事件驱动? 抓取段/操作段怎么切?
2. **grasp_phase_frame**：③ 轨迹里"到达抓取"的帧怎么标(接触建立? 到 ②pose?)。
3. **goal_object_pose**：lift = delta-z 还是 6D 目标? 哪个帧算成功?
4. **reward 组合**：`r_imit(乘) + r_reg(加)` vs 全乘? k_h/k_o/接触权重初值?
5. **traj_curobo MVP 顶替**：先授权直线插值(open→n4)顶 ③, 还是一开始就真接 cuRobo?
6. **①-手指用不用**：低可信默认不追; 要不要在接近段轻追/或只当诊断?
7. **DataUnit 落盘格式**：单个 .npz? 字段命名? 便于 Reconstruct_and_Retarget pipeline 直接产出对齐。
8. **object_init_pose 来源**：① 首帧物体位姿 vs ② 抓取时物体位姿。
```
