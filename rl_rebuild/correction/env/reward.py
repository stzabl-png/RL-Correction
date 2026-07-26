"""Reward v3 — 乘性任务核 + 可退火 shaping 环 + 常驻正则 (设计见 M1 清单).

纯函数实现: 只吃张量, 不碰 env/仿真 — 可离线单测 (python -m ...reward 直接跑).
所有权重经 RewardWeights 传入且可在训练中改写 (退火 = 训练循环改 λ, 状态触发).
每项独立返回, env 逐项写入 extras["log"] — 查 reward hacking 的账本.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RewardWeights:
    # ---- 永久核心 ----
    w_core: float = 2.0
    lift_target: float = 0.10        # 满分抬升高度 (成功判据同款 10cm)
    min_contacts: int = 2            # 握持门槛: 至少几根指尖接触
    loose_grip_factor: float = 0.3   # 抬升但接触不足时的乘性折扣 (防手背铲/腕抡)
    # ---- shaping 环 (退火对象; 训练循环按成功率 EMA 衰减这些 λ) ----
    lam_approach: float = 0.5
    lam_contact: float = 0.3
    lam_lift: float = 0.5
    lam_traj: float = 0.2            # 物体粗轨迹引导 (纪要: 小权重, 成功后 → 0)
    lam_imit: float = 0.1            # 人手模仿 (更弱, 随 λ_traj 同步退火)
    sigma_approach: float = 0.10     # exp 核宽度 (m)
    sigma_traj: float = 0.10
    # ---- affordance: 奖励指尖落在高 affordance 点 (=人手接触区/环). 饱和防"堆一侧".
    #      默认 0 = 关 (Grasp0 等不受影响); Grasp2 等小/扁物体训练时开.
    lam_afford: float = 0.0
    afford_sat: float = 0.5          # tanh 饱和尺度: Σ(指尖接触×afford)/sat
    # ---- cuRobo 预抓取路点引导 (只接近段生效; PPO 按 sr_ema 退火 lam_curobo→0) ----
    lam_curobo: float = 0.0          # 默认关; Exp1 由 train.py 置初值 (如 0.5), 再随能力退火
    sigma_curobo: float = 0.10       # 路点 exp 核宽 (m)
    # ---- 常驻正则 (不退火) ----
    w_act: float = 0.005             # 动作幅度
    w_rate: float = 0.01             # 动作变化率 (平滑)
    w_wrench: float = 0.02           # 腕 wrench 用量 (省力)
    w_spike: float = 0.5             # 物体速度尖峰 (禁暴力)
    spike_vel: float = 1.5           # 超过此速度 (m/s) 算尖峰
    # ---- D-Grasp 三件套 ----
    term_clamp: float = -10.0        # 逐项下限
    total_clip: float = -2.0         # 总分下限


def compute_reward(
    *,
    # ---- 当前状态 ----
    obj_pos: torch.Tensor,           # (N,3) 物体位置 (桌面局部系, 已减 env_origin)
    obj_linvel: torch.Tensor,        # (N,3)
    obj_rest_z: float,               # 物体静置时的中心高度 (标定常数)
    palm_pos: torch.Tensor,          # (N,3) 掌心位置
    contacts: torch.Tensor,          # (N,5) 指尖接触 (bool/0-1)
    finger_q: torch.Tensor,          # (N,22)
    wrist_pos: torch.Tensor,         # (N,3)
    # ---- 参考 (当前帧) ----
    ref_obj_pos: torch.Tensor,       # (N,3)
    ref_finger_q: torch.Tensor,      # (N,22)
    ref_wrist_pos: torch.Tensor,     # (N,3)
    # ---- 动作/控制 ----
    action: torch.Tensor,            # (N,28) 本步残差动作 (已归一化 [-1,1])
    prev_action: torch.Tensor,       # (N,28)
    wrench_norm: torch.Tensor,       # (N,) 本步腕 wrench 用量 (|F|/F_max + |T|/T_max)
    # ---- 相位掩码 ----
    active: torch.Tensor,            # (N,) bool; False=静置前奏 (任务/引导分不发, 正则照扣)
    # ---- 跨步状态 (env 持有, 本函数就地更新; reset 时 env 负责重置) ----
    prev_palm_dist: torch.Tensor,    # (N,) 上一步掌心-物体距离; reset 时=当前距离
    lift_hw: torch.Tensor,           # (N,) 抬升高水位 (0..1); reset 时=0
    w: RewardWeights,
    finger_weight: torch.Tensor | None = None,   # (22,) 关节权重 (远端指节 4x), None=均匀
    curobo_wp: torch.Tensor | None = None,       # (N,3) cuRobo 预抓取路点(桌面局部系); None=不用
    in_approach: torch.Tensor | None = None,     # (N,) bool 接近段掩码 (路点引导只在此段付)
    tip_afford: torch.Tensor | None = None,      # (N,5) 每指尖最近 affordance 点的 heatmap [0,1]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (total (N,), 逐项 dict). 全向量化, 无循环.

    反年金设计 (GR00T 冠军教训):
    - approach = 势函数差分: 靠近付正、悬停零、远离付负 — 站岗无收入
    - lift shaping = 高水位棘轮: 只付新高度纪录, 保持/回爬旧高一分不付
    """
    N = obj_pos.shape[0]
    act = active.float()

    # ================= 第一层: 永久核心 (乘性, 防偏科) =================
    lift = ((obj_pos[:, 2] - obj_rest_z) / w.lift_target).clamp(0.0, 1.0)
    n_contact = contacts.float().sum(dim=1)
    grip_gate = torch.where(n_contact >= w.min_contacts,
                            torch.ones(N, device=obj_pos.device),
                            torch.full((N,), w.loose_grip_factor, device=obj_pos.device))
    # 掉落惩罚因子: 快速下坠 (v_z < -0.3 m/s) 指数衰减 — 抓不稳松手立刻失分
    falling = torch.exp(-4.0 * (-obj_linvel[:, 2] - 0.3).clamp(min=0.0))
    r_task = lift * grip_gate * falling * act

    # ================= 第二层: shaping 环 (退火对象) =================
    # 靠近: 势函数差分 (Ng 1999, 策略不变性). 靠近正 / 停零 / 离开负.
    d_palm = (palm_pos - obj_pos).norm(dim=1)
    r_approach = ((prev_palm_dist - d_palm) / w.sigma_approach) * act
    prev_palm_dist.copy_(d_palm)                               # 就地更新状态
    r_contact = (n_contact / 5.0) * act
    # 抬升: 高水位棘轮. 只付 (当前高度 − 历史最高), 保持/掉落/回爬旧高均为零.
    r_lift_sh = (lift - lift_hw).clamp(min=0.0) * act
    lift_hw.copy_(torch.maximum(lift_hw, lift * act))          # 就地更新高水位
    r_traj = torch.exp(-(obj_pos - ref_obj_pos).norm(dim=1) / w.sigma_traj) * act
    # 手模仿: 手指 MAE (可加权, 远端 4x) + 腕位置误差. 圆柱对称 → 物体姿态不打分,
    # 腕姿态误差已由 wrench-PD 闭环, 此处只约束位置通道防漂.
    fq_err = (finger_q - ref_finger_q).abs()
    if finger_weight is not None:
        fq_err = fq_err * finger_weight
    r_imit = torch.exp(-(2.0 * (wrist_pos - ref_wrist_pos).norm(dim=1)
                         + 0.5 * fq_err.mean(dim=1))) * act

    # ================= 第三层: 常驻正则 (负项, 不退火) =================
    p_act = -w.w_act * action.square().sum(dim=1)
    p_rate = -w.w_rate * (action - prev_action).square().sum(dim=1)
    p_wrench = -w.w_wrench * wrench_norm.square()
    p_spike = -w.w_spike * (obj_linvel.norm(dim=1) - w.spike_vel).clamp(min=0.0).square()

    terms = {
        "task": w.w_core * r_task,
        "approach": w.lam_approach * r_approach,
        "contact": w.lam_contact * r_contact,
        "lift": w.lam_lift * r_lift_sh,
        "traj": w.lam_traj * r_traj,
        "imit": w.lam_imit * r_imit,
        "pen_act": p_act,
        "pen_rate": p_rate,
        "pen_wrench": p_wrench,
        "pen_spike": p_spike,
    }
    # cuRobo 预抓取路点引导: 接近段内奖励腕靠近 cuRobo 的 close-起点位姿.
    # 势"到位"信号(非路径模仿) -> 教"快速够到预抓取位", 不印 cuRobo motion; 按能力退火.
    if curobo_wp is not None:                     # lam=0(退火完成)时仍留项, 保持日志键稳定
        gate = (in_approach.float() if in_approach is not None else act)
        r_curobo = torch.exp(-(wrist_pos - curobo_wp).norm(dim=1) / w.sigma_curobo) * gate
        terms["curobo"] = w.lam_curobo * r_curobo
    # affordance: 指尖落在高 affordance 点才付, 饱和 (1 个好接触≈拿大头, 再堆指头几乎不加分).
    # 目的: 逼 RL 用指尖抓小/扁物体的接触区, 而非整手 cage (cage 是 BODex 的活).
    if tip_afford is not None:
        r_afford = torch.tanh((contacts.float() * tip_afford).sum(dim=1) / w.afford_sat) * act
        terms["afford"] = w.lam_afford * r_afford
    # D-Grasp 三件套: 逐项 clamp -> 求和 -> 全局 clip
    # NaN 防护: 物体物理偶发发散会让状态量变 NaN -> reward NaN -> advantage/loss NaN ->
    # 梯度裁剪也救不回来(裁剪 NaN 仍是 NaN) -> 权重 NaN -> 训练崩. 必须在这里截断.
    terms = {k: v.nan_to_num(0.0) for k, v in terms.items()}
    total = sum(t.clamp(min=w.term_clamp) for t in terms.values())
    total = total.clamp(min=w.total_clip).nan_to_num(w.total_clip)
    return total, terms


# ======================= 离线单测 (无需 Isaac) =======================
if __name__ == "__main__":
    torch.manual_seed(0)
    N = 4
    w = RewardWeights()
    rest_z = 0.92

    # 跨步状态: 时序测试共享这两个 buffer (env 里也是这么持有的)
    prev_dist = torch.full((N,), 0.40)
    lift_hw = torch.zeros(N)

    def step(palm_xyz, obj_z=rest_z, contacts=0.0, active=True, **kv):
        base = dict(
            obj_pos=torch.tensor([[0.0, 0.0, obj_z]] * N),
            obj_linvel=torch.zeros(N, 3), obj_rest_z=rest_z,
            palm_pos=torch.tensor([palm_xyz] * N),
            contacts=torch.full((N, 5), float(contacts)),
            finger_q=torch.zeros(N, 22), wrist_pos=torch.zeros(N, 3),
            ref_obj_pos=torch.tensor([[0.0, 0.0, rest_z]] * N),
            ref_finger_q=torch.zeros(N, 22), ref_wrist_pos=torch.zeros(N, 3),
            action=torch.zeros(N, 28), prev_action=torch.zeros(N, 28),
            wrench_norm=torch.zeros(N),
            active=torch.full((N,), active, dtype=torch.bool),
            prev_palm_dist=prev_dist, lift_hw=lift_hw, w=w,
        )
        base.update(kv)
        return compute_reward(**base)

    # ---- 时序 1: approach 势函数差分 (prev_dist 从 0.40 开始) ----
    _, t1 = step([0.0, 0.0, rest_z + 0.30])        # 靠近: 0.40 -> 0.30
    _, t2 = step([0.0, 0.0, rest_z + 0.30])        # 悬停: 0.30 -> 0.30
    _, t3 = step([0.0, 0.0, rest_z + 0.38])        # 离开: 0.30 -> 0.38

    # ---- 时序 2: lift 高水位棘轮 (hw 从 0 开始, 手已就位+满接触) ----
    near = [0.0, 0.0, rest_z + 0.02]
    _, l1 = step(near, obj_z=rest_z + 0.05, contacts=1)   # 首抬 5cm: 付 0->0.5
    _, l2 = step(near, obj_z=rest_z + 0.05, contacts=1)   # 保持: 0
    _, l3 = step(near, obj_z=rest_z + 0.00, contacts=1)   # 掉回桌面: 0
    _, l4 = step(near, obj_z=rest_z + 0.05, contacts=1)   # 回爬旧高: 0 (不重付)
    _, l5 = step(near, obj_z=rest_z + 0.08, contacts=1)   # 新高 8cm: 付 0.5->0.8

    # ---- 单步场景 (与旧版相同的核心检验) ----
    lift_hw.zero_(); prev_dist.fill_(0.40)
    r_grasp, tg = step([0.0, 0.0, rest_z + 0.12], obj_z=rest_z + 0.10, contacts=1)
    lift_hw.zero_(); prev_dist.fill_(0.40)
    r_shovel, _ = step([0.0, 0.0, rest_z + 0.12], obj_z=rest_z + 0.10, contacts=0)
    lift_hw.zero_(); prev_dist.fill_(0.40)
    r_settle, _ = step([0.3, 0.3, 1.2], active=False)
    lift_hw.zero_(); prev_dist.fill_(0.40)
    r_wild, _ = step([0.3, 0.3, 1.2], action=torch.rand(N, 28) * 2 - 1,
                     prev_action=-torch.rand(N, 28), wrench_norm=torch.full((N,), 1.5),
                     obj_linvel=torch.tensor([[3.0, 0.0, 0.0]] * N))

    lam = w.lam_lift
    checks = [
        ("靠近付正 (势差分)", t1["approach"][0] > 0.2),
        ("悬停零年金", abs(t2["approach"][0].item()) < 1e-6),
        ("离开付负", t3["approach"][0] < -0.2),
        ("首次抬升付款 (0->0.5)", abs(l1["lift"][0].item() - lam * 0.5) < 1e-4),
        ("保持原高零利息", abs(l2["lift"][0].item()) < 1e-6),
        ("掉落不付款", abs(l3["lift"][0].item()) < 1e-6),
        ("回爬旧高不重付", abs(l4["lift"][0].item()) < 1e-6),
        ("新高只付增量 (0.5->0.8)", abs(l5["lift"][0].item() - lam * 0.3) < 1e-4),
        ("真抓取核心分满", abs(tg["task"][0].item() - w.w_core) < 1e-4),
        ("铲抬 < 真抓取 (乘性门控)", r_grasp[0] - r_shovel[0] > 0.8),
        ("静置段无任务/引导分", r_settle[0] <= 0.0),
        ("暴力乱挥被罚且有下限", (r_wild[0] < 0) and (r_wild[0] >= w.total_clip)),
    ]
    print(f"{'检验':<30} 结果")
    ok = True
    for name, passed in checks:
        ok &= bool(passed)
        print(f"{name:<32} {'PASS' if passed else 'FAIL'}")
    print(f"\n真抓={r_grasp[0]:.3f} 铲抬={r_shovel[0]:.3f} 静置={r_settle[0]:.3f} 乱挥={r_wild[0]:.3f}")
    raise SystemExit(0 if ok else 1)
