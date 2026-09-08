"""v5 母带生成 (L5-1 拍板): 交互段(190..324)由物体轨迹反解IK重铸。
第一幕 Approach照谱+缝1剂量渐入(βR=2.0/βL=1.0) → 站稳30步 → 捕获增量空间锚 →
逐行IK(135行×双臂) + 5mm认证行IK → 缝2重融接 → 写 pour17_reference_v2.npz。
Approach/Retreat/物体轨迹/conf 原样; 人手行存 human_* 供 P-HYB 红档参考。"""
import argparse, hashlib, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("build_v5")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
# ★L5-37: V1 输入可配置 (POUR_REF_V1IN) —— 位置泛化把 v1 位置变体升级成 v5(273帧) 格式。
#   默认原 v1 (幂等). 输入必须是"未扩张的 v1 格式"(135交互), 否则重跑会再扩张一次。
V1 = os.environ.get("POUR_REF_V1IN") or "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v1.npz"
# ★L5-11 #4: 生成器必须从 v1 读物体轨迹 —— env 默认加载 v2, 重跑一次就多扩张一次
# (实测 135->163->191). 显式覆写 MASTER, 保证幂等。
os.environ["POUR_REF_NPZ"] = os.path.abspath(V1)
import pour_env as PE
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

BR, BL = 2.0, 1.0
OUT = os.environ.get("POUR_REF_OUT") or "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v2.npz"
# ★L5-31 消融旗 POUR_REF_NOCONF: 造"没有置信度"的母带 (flat 臂用)。
#   用户裁定 flat 连母带一起重造 —— 因为置信度不只在运行时用, 它在**母带生成时**
#   就参与了朝向平滑: `_smooth_quat` 拿 conf>=40 挑锚点, 不可信帧由两侧 SLERP 插值,
#   再按转速上限迭代降级(降谁也按 conf 挑)。只关运行时等于只消融了一半。
#   A 版 = 完全跳过这道平滑, 原始重建朝向直接进 IK。
#   ⚠ 预期风险(L5-11 实测): 重建在倒水段长轴摆 44~57°/帧, 会被翻译成 72°/帧 的
#   关节跳, 而残差界只有几度。**A 版很可能过不了"关节连续性"那道出厂检查** ——
#   过不了本身就是结论(置信度平滑是母带可造性的必要条件), 那时退 B 版
#   (保留转速上限, 只是不用 conf 挑锚点)。
NOCONF = os.environ.get("POUR_REF_NOCONF") == "1"
# ★L5-31 用户裁定: 连续性硬闸是**我们自己流水线的质量关**, 不是物理定律。
#   消融要问的是"没有置信度这一整套会怎样" —— 拿我们自己的闸把原始重建拦下来,
#   等于没做这个消融。POUR_REF_ALLOW_JUMP=1 只放行**连续性**这一道, 其余三道
#   (峰值保全 / IK 精度 / 末态-判据一致性)照常把关。
ALLOW_JUMP = os.environ.get("POUR_REF_ALLOW_JUMP") == "1"
# ★L5-32 噪声消融 (2026-08-31): 往物体**朝向**注入噪声, 模拟"重建质量更差"。
#   为什么只注朝向、不注位置 —— 实测 v1 真重建段(source=1, 135帧)的高频残差:
#     位置  RMS 0.01~0.36mm (随平滑窗)   朝向  瓶 7.03° / 杯 5.16°
#   位置轨迹在上游**已经被平滑过**, 它的残差量不到的是"感知噪声"而是"平滑残余",
#   拿它当基准会低估好几个数量级。朝向的残差才是真的 —— 而且整套 conf 机制
#   (conf_rot / CONF_TRUST=40 / _smooth_quat) 本来就**只管朝向**, 说明设计者
#   当初也是这么判断的。所以噪声消融定为**纯朝向**, 并在台账里写明这个限制。
#   档位: 1× = 7°(实测) · 3× = 21°。注在 _smooth_quat **之前** —— 让置信度机制
#   有机会把它清掉, 这样"1× 被吸收 / 3× 吸收不掉"本身就是一个结果。
NOISE_MULT = float(os.environ.get("POUR_REF_NOISE_MULT", "0"))
NOISE_SEED = int(os.environ.get("POUR_REF_NOISE_SEED", "17"))
# 实测: v1 真重建段(source=1, 135帧) 朝向的高频残差 RMS (Savitzky-Golay 窗9)
NOISE_BASE_DEG = {1: 7.03, 0: 5.16}      # 1=瓶 0=杯
ALLOW_FEATLOSS = os.environ.get("POUR_REF_ALLOW_FEATLOSS") == "1"
if NOCONF:
    print("[v5] " + "=" * 66, flush=True)
    print("[v5] ★POUR_REF_NOCONF=1: 不用置信度做任何朝向平滑 —— 原始重建直接进 IK",
          flush=True)
    print("[v5]   这就是「没有置信度」的真实条件: 不挑帧、不修轨迹。", flush=True)
    print("[v5] " + "=" * 66, flush=True)
if ALLOW_JUMP:
    print("[v5] ⚠★POUR_REF_ALLOW_JUMP=1: **连续性硬闸只报不拦**。", flush=True)
    print("[v5]   已知后果: 关节跳变会超过残差界(几度), 策略在那几行结构上跟不上,",
          flush=True)
    print("[v5]   时钟大概率卡住 —— 这是消融的**预期结果**, 不是故障。", flush=True)
    print("[v5]   判读时必须说清: 失败机制是'参考跳变超出残差能力', 不是'RL 学不会'。",
          flush=True)
cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sqr = np.asarray(np.load("tasks/pregrasp/priors/Pour25_bottle_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
sql = np.asarray(np.load("tasks/pregrasp/priors/Pour25_cup_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_r = torch.tensor(sqr - ref[E.IA0, 14:36], dtype=torch.float32, device=dev)
dsq_l = torch.tensor(sql - ref[E.IA0, 36:58], dtype=torch.float32, device=dev)

def drive(row, sR, sL):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[14:36] += sR * dsq_r
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

print("[v5] 第一幕: Approach+缝1 (βR=2.0/βL=1.0)", flush=True)
for r in range(0, 166):
    drive(r, 0.0, 0.0)
for i, r in enumerate(range(166, 190)):
    a = (i + 1) / 24.0
    drive(r, BR * a, BL * a)
for _ in range(30):
    drive(190, BR, BL)
f = E._pads_f().norm(dim=-1)[0]
print(f"[v5] 站位垫: R{int((f[:5]>0.5).sum())}/L{int((f[5:]>0.5).sum())}", flush=True)

ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
ref_obj_raw = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}
Nrow_raw = E.PB.N_ROW
# ---- 放回窗时间扩张 (L5-3): 交互行 72..100 (=全链262..290) 二倍细分 ----
# 母带放回 = 回正旋转55° + 10cm/s 下降复合, v2硬闸实证 3垫握被拧脱; 扩张后速率减半
# 窗口表: (起, 止, 细分倍数)。倍数 n = 该窗每行摊成 n 行。
# ★L5-11 新增倒水转动窗: 行40~52 是腕部奇异区, IK 需要 40.8°/帧 的关节运动
#   (腕目标只转8.2°, 但雅可比病态)。放慢 8 倍 -> ~5°/帧, 回到可执行范围。
#   放回窗 72~100 沿用 2 倍 (治放回俯冲甩脱)。
# ★pour25: 快转/奇异区实测在交互 24~46 行(杯朝向重建噪声 25°/行@26 + 瓶转25-71行),
#   非 pour17 的 40~53。8x 盖住它治抓握段跳变超界; 46~末 2x 治倾倒保持&放回甩脱。
WINDOWS = [(24, 46, 8), (46, 103, 2)]
frac_rows = []
for k in range(Nrow_raw):
    frac_rows.append(float(k))
    for (a, b, n) in WINDOWS:
        if a <= k < b:
            for j in range(1, n):
                frac_rows.append(k + j / n)
            break
frac_rows = np.array(frac_rows)
Nrow = len(frac_rows)
def _lerp_track(tr):
    lo = np.floor(frac_rows).astype(int)
    hi = np.minimum(lo + 1, Nrow_raw - 1)
    a = (frac_rows - lo)[:, None]
    pos = tr[lo, :3] * (1 - a) + tr[hi, :3] * a
    q0, q1 = tr[lo, 3:7], tr[hi, 3:7]
    sgn = np.sign((q0 * q1).sum(axis=1, keepdims=True)); sgn[sgn == 0] = 1
    q = q0 * (1 - a) + q1 * sgn * a
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    return np.concatenate([pos, q], axis=1)
# ★L5-11 #2 主修复: 物体朝向按置信度平滑 + 转速上限
# 病因: 重建在倒水段朝向估计不稳(位置只动0.3cm、长轴摆44~57°/帧), conf_rot 13~39
# 全在红档 —— 系统自己知道不可信, 我们却照单全收喂给 IK, 被翻译成 72°/帧 的关节跳。
ROT_CAP = np.radians(10.0)          # 转速上限 10°/帧 = 200°/秒 (人倒水的合理上限)
CONF_TRUST = 40.0                   # conf_rot 低于此 = 不可信, 由两侧可信帧插值

def _qslerp(q0, q1, a):
    q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)

def _qang(q0, q1):
    return 2 * np.arccos(min(1.0, abs(float(np.dot(
        q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))))

def _smooth_quat(Q, conf, tag, use_conf=True):
    """可信帧当锚, 不可信帧 SLERP; 再迭代压掉超过转速上限的残余跳变。

    ★L5-31 `use_conf=False` = B 版消融: **保留转速上限, 但不用置信度挑锚点**。
      置信度在本函数里出现两次: ① 初始 trust 掩膜 ② 违反转速上限时"降级哪一端"。
      B 版把①改成"全部当锚"(等于不做基于可信度的插值), ②改成纯几何判据
      (降级相邻转速更大的那一端)。这样就把"conf 挑锚点"和"有没有转速上限"
      两件事分开了。
      由来: A 版(完全不平滑)实测过不了连续性硬闸 —— 右臂 j5 在腕奇异区跳 31.65°
      (>25° 上限), 而腕目标只动 0.71cm/5.70°。左臂最大仅 6.94°, 越限只有 11 行。
      ⟹ 置信度平滑的实际作用是**压住奇异区那几行**, 不是整体去噪。
    """
    n = len(Q)
    trust = (conf >= CONF_TRUST) if use_conf else np.ones(n, dtype=bool)
    trust[0] = trust[-1] = True                    # 两端必须是锚
    for _it in range(200 if not use_conf else 40):
        idx = np.where(trust)[0]
        out = Q.copy()
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a <= 1:
                continue
            for j in range(a + 1, b):
                out[j] = _qslerp(Q[a], Q[b], (j - a) / (b - a))
        rate = np.array([_qang(out[i], out[i + 1]) for i in range(n - 1)])
        if rate.max() <= ROT_CAP:
            break
        # 最快的那一跳: 把置信度较低的那一端降级为不可信, 重新插值
        j = int(rate.argmax())
        cand = [k for k in (j, j + 1) if trust[k] and 0 < k < n - 1]
        if not cand:
            break
        if use_conf:
            trust[min(cand, key=lambda k: conf[k])] = False
        else:
            # ★纯几何降级, 且**每轮批量降**, 不是一帧一帧降。
            #   第一版每轮只降一帧、上限 40 轮 —— 而 conf 版一开始就把 conf<40 的
            #   上百帧一次性标为不可信。两者力度差一个数量级, 那样比出来的"不用
            #   conf 就造不出母带"是我的实现太弱, 不是 conf 不可替代。
            #   现在改成: 每轮把**所有**相邻转速越限的内点一起降级, 力度对齐。
            _bad = [k for k in range(1, n - 1)
                    if trust[k] and max(rate[k - 1], rate[k]) > ROT_CAP]
            if not _bad:
                trust[max(cand, key=lambda k: max(rate[max(k - 1, 0)],
                                                  rate[min(k, n - 2)]))] = False
            else:
                for k in _bad:
                    trust[k] = False
    print(f"[v5] {tag} 朝向平滑: 锚点 {int(trust.sum())}/{n}, "
          f"转速 中位={np.degrees(np.median(rate)):.2f}° 最大={np.degrees(rate.max()):.2f}° "
          f"(上限 {np.degrees(ROT_CAP):.0f}°)", flush=True)
    return out

def _task_features(P, Q, oi):
    """任务关键特征: 用于"平滑有没有把任务改掉"的前后对账。
    倾角峰值(倒水靠它) / 高度行程(提起靠它) / 水平行程(搬运靠它)。"""
    up = np.array([0.0, 1.0, 0.0])
    tl = []
    for q in Q:
        w, x, y, z = q / np.linalg.norm(q)
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        v = R @ up
        tl.append(np.degrees(np.arccos(np.clip(v[2] / max(np.linalg.norm(v), 1e-9),
                                               -1, 1))))
    tl = np.array(tl)
    return {"倾角峰值": float(tl.max()),
            "倾角行程": float(tl.max() - tl.min()),
            "高度行程": float((P[:, 2].max() - P[:, 2].min()) * 100),
            "水平行程": float(np.linalg.norm(P[:, :2] - P[0, :2], axis=1).max() * 100)}

_z1 = np.load(V1, allow_pickle=True)
_rows1 = np.where(np.asarray(_z1["source"]) == 1)[0]
def _tilt_to(q, up_world):
    """物体自身 up 轴(局部 +y)在世界系与给定参考轴的夹角(弧度)。"""
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    v = R @ np.array([0.0, 1.0, 0.0])
    c = float(np.dot(v, up_world) / (np.linalg.norm(v) * np.linalg.norm(up_world)))
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


HOME_K = 30              # 末态归位窗 (原始行数); 5.6cm/30行 = 1.9mm/行, 极缓


def _home_end(traj, tag, K=HOME_K):
    """把交互段末尾 K 行的物体位姿平滑送回**它自己的第 0 行**。

    为什么目标是"第 0 行"而不是母带第 0 行: 判据 self.rest 与 env.rest_pose
    **都取交互段首行**(实测确认), 仿真也把物体放在那里。
    为什么不是瞬跳: 母带原本在交互/机器段边界一行之内从 4.08cm 跳回 0 ——
    带子能瞬移, 真瓶子不行。
    权重用余弦缓入, 首末导数为 0, 不给关节连续性硬闸添乱。
    """
    out = traj.copy()
    n = len(out)
    K = min(K, n - 1)
    p_t, q_t = out[0, :3].copy(), out[0, 3:7].copy()
    d0 = float(np.linalg.norm(out[-1, :3] - p_t))
    for i in range(n - K, n):
        a = 0.5 * (1 - np.cos(np.pi * (i - (n - K) + 1) / K))     # 0->1 余弦
        out[i, :3] = (1 - a) * out[i, :3] + a * p_t
        out[i, 3:7] = _qslerp(out[i, 3:7], q_t, a)
    d1_ = float(np.linalg.norm(out[-1, :3] - p_t))
    print(f"[v5] 末态归位 {tag}: 末行距首行 {d0*100:.2f}cm -> {d1_*100:.2f}cm "
          f"(窗 {K} 行, 峰值 {d0/K*1000:.1f}mm/行)", flush=True)
    return out


FEAT_TOL = 0.05          # 关键特征允许的相对损失 (5%)
_feat_bad = 0
for oi, nm in ((0, "杯"), (1, "瓶")):
    _cf = np.asarray(_z1[f"conf_rot_{oi}"], np.float64)[_rows1]
    _before = _task_features(ref_obj_raw[oi][:, :3], ref_obj_raw[oi][:, 3:7], oi)
    if NOISE_MULT > 0:
        # ★注入点必须在 `_before` **之后**、平滑之前:
        #   在 _before 之前注 → 出厂检查比的是"加噪 vs 加噪+平滑", 量的是**平滑器
        #   清掉了多少**, 而不是任务退化了多少。我第一版就写错在这, 表现为杯子的
        #   倾角峰值从本该的 ~0° 变成 66°, 检查当场红。
        #   在 _before 之后注 → 比的是"干净 vs 加噪后", 正是我们要报的量。
        # ★第 0 行绝不加噪: 它就是 PourProgressBatch 的 `rest`, 是 placed 判据的
        #   唯一参照。动了它, 噪声母带与 v3 就不是同一把尺, 本轮"换成纯绝对判据
        #   以便跨臂比较"的全部意义当场作废。(末行由 _home_end 强制等于首行。)
        _sig = NOISE_MULT * NOISE_BASE_DEG[oi]
        _rng = np.random.default_rng(NOISE_SEED + oi)
        _n = len(ref_obj_raw[oi])
        _ax = _rng.normal(size=(_n, 3))
        _ax /= np.linalg.norm(_ax, axis=1, keepdims=True)
        _an = np.radians(_rng.normal(0.0, _sig, _n))
        _an[0] = 0.0
        _h = _an / 2.0
        _dq = np.concatenate([np.cos(_h)[:, None], _ax * np.sin(_h)[:, None]], 1)
        _q = ref_obj_raw[oi][:, 3:7]
        _w1, _v1 = _q[:, :1], _q[:, 1:]
        _w2, _v2 = _dq[:, :1], _dq[:, 1:]
        _nw = _w1 * _w2 - (_v1 * _v2).sum(1, keepdims=True)
        _nv = _w1 * _v2 + _w2 * _v1 + np.cross(_v1, _v2)
        _qn = np.concatenate([_nw, _nv], 1)
        _qn /= np.linalg.norm(_qn, axis=1, keepdims=True)
        _dev = np.degrees([_qang(_q[i], _qn[i]) for i in range(_n)])
        ref_obj_raw[oi][:, 3:7] = _qn
        print(f"[v5] ★{nm} 注入朝向噪声 {NOISE_MULT:.0f}× = sigma {_sig:.2f}° "
              f"(实测该物体的重建高频残差 {NOISE_BASE_DEG[oi]:.2f}°) seed={NOISE_SEED + oi}: "
              f"实际偏离 中位 {np.median(_dev):.2f}° 最大 {_dev.max():.2f}° "
              f"| 第0行偏离 {_dev[0]:.4f}° (必须 0, 它是 placed 的 rest)", flush=True)
    if NOCONF:
        _rt = np.array([_qang(ref_obj_raw[oi][i, 3:7], ref_obj_raw[oi][i + 1, 3:7])
                        for i in range(len(ref_obj_raw[oi]) - 1)])
        print(f"[v5] {nm} 朝向**完全未平滑**: 原始转速 中位={np.degrees(np.median(_rt)):.2f}° "
              f"最大={np.degrees(_rt.max()):.2f}° (基线版会压到 {np.degrees(ROT_CAP):.0f}° 以下)",
              flush=True)
    else:
        ref_obj_raw[oi][:, 3:7] = _smooth_quat(
            ref_obj_raw[oi][:, 3:7].copy(), _cf, nm, use_conf=True)
    # ★L5-27 末态归位: 参考的交互段**末行必须回到它自己的首行**, 否则
    #   "完美跟随参考"与 placed 判据(≤M3_POS 3cm)直接矛盾。
    #   实测 v2: 瓶末行距首行 5.60cm, 末尾连续<3cm 的行数 = 0 —— 参考从未把瓶
    #   送进容差, placed 永远立不了, G4 因此封顶。杯 2.04cm 合格但判据是 AND。
    #   归位放在平滑之后、峰值检查取 _after 之前, 让峰值保全检查一并覆盖它:
    #   若归位把任务特征(倾角峰值/高度行程/水平行程)改坏了, 出厂就会红。
    ref_obj_raw[oi] = _home_end(ref_obj_raw[oi], nm)
    _after = _task_features(ref_obj_raw[oi][:, :3], ref_obj_raw[oi][:, 3:7], oi)
    # ★L5-11 峰值保全检查: 平滑是"不信重建", 但不能顺手把任务本身改掉。
    # 本次实测倾角峰值 111.3° 前后不变 —— 但换个任务(如拧瓶盖)关键动作可能恰好
    # 落在低置信帧, 那时平滑会把任务特征一起抹掉, 且只会在训练几小时后才暴露。
    _msg = []
    for k in _before:
        b, a = _before[k], _after[k]
        rel = (b - a) / max(abs(b), 1e-9)
        flag = "★损失" if rel > FEAT_TOL else ""
        _msg.append(f"{k} {b:.1f}->{a:.1f}({-rel*100:+.1f}%){flag}")
        if rel > FEAT_TOL:
            _feat_bad += 1
    print(f"[v5] {nm} 关键特征对账: " + " | ".join(_msg), flush=True)
if _feat_bad:
    if ALLOW_FEATLOSS:
        # ★噪声母带的**预期结果**, 不是故障: 注了噪声, 任务特征本来就会退化。
        #   这道检查原本是防"平滑顺手把任务改坏"的意外; 这里退化是**处理本身**。
        #   但必须把实际损失记进 meta 和台账 —— 它是这条消融"到底注了多重"的
        #   真实量度, 比 sigma 更有意义(sigma 是输入, 特征损失是输出)。
        print(f"[v5] ⚠★POUR_REF_ALLOW_FEATLOSS=1: {_feat_bad} 项关键特征损失 "
              f">{FEAT_TOL*100:.0f}%, **只报不拦**。", flush=True)
        print("[v5]   噪声消融的预期结果。判读时必须用上面的'关键特征对账'说明"
              "这条母带的任务被改到什么程度。", flush=True)
        print("[v5]   ★但仍须单独核验: 加噪后的母带自身还能不能满足 G3_pour "
              "连续 25 行 —— 过不了就是 L5-27 重演(判据比参考自身还严)。", flush=True)
    else:
        print(f"[v5] ★峰值保全检查未过: {_feat_bad} 项关键特征损失 "
              f">{FEAT_TOL*100:.0f}% —— 平滑改掉了任务本身, 不写母带", flush=True)
        raise SystemExit(3)
else:
    print("[v5] ✅ 峰值保全检查通过", flush=True)

ref_obj = {oi: _lerp_track(ref_obj_raw[oi]) for oi in (0, 1)}
floor_map = np.floor(frac_rows).astype(int)          # 新交互行 -> 原交互行
print(f"[v5] 时间扩张: 交互 {Nrow_raw} -> {Nrow} 行 (窗 {WINDOWS})", flush=True)
q_ik = {s: np.zeros((Nrow, 7)) for s in ("right", "left")}
cert = {}
q_seed, w0 = {}, {}
for s in ("right", "left"):
    sl = slice(0, 7) if s == "right" else slice(7, 14)
    q_seed[s] = E.hand.data.joint_pos[0, E.map_ids_t[sl]].cpu().numpy().astype(np.float64)
    w0[s] = ik[s].fk(q_seed[s])
fail_ik, err_max = 0, 0.0
rng = np.random.default_rng(17)
for s in ("right", "left"):
    best = None
    for trial in range(12):
        q0t = q_seed[s] if trial == 0 else q_seed[s] + rng.normal(0, 0.03, 7)
        r5 = ik[s].solve(w0[s][0] + np.array([0, 0, 0.015]), w0[s][1], q0=q0t, iters=300)
        if best is None or r5["pos_err"] < best["pos_err"]:
            best = r5
        if best["pos_err"] < 0.001:
            break
    cert[s] = np.asarray(best["q"], np.float64)
    print(f"[v5] 认证行IK {s}: pos_err={best['pos_err']*1000:.2f}mm", flush=True)
# ★L5-10 IK 连续性修复: 目标细分连续跟踪 (原来每行独立解 -> 冗余零空间换支,
# 实测 v2 母带 行234->236 肩关节跳 42°+72°/帧, 策略残差只有4.6°根本无力跟随)
MAX_DQ = np.radians(3.0)      # 每行关节变化上限
SUB_MAX = 32                  # 细分上限
UP_LOCAL = np.array([0.0, 1.0, 0.0])    # 瓶/杯的长轴(局部系)

def _axis_only_R(q0_, qk_):
    """★L5-11 #3: 只取"把长轴从 a0 转到 ak"的最小旋转, 丢掉绕长轴的自转分量。
    轴对称物体绕自身长轴转多少不可观测, 照抄会把估计抖动变成手腕的无谓自转。"""
    a0 = quat_to_R(q0_) @ UP_LOCAL
    ak = quat_to_R(qk_) @ UP_LOCAL
    a0 = a0 / np.linalg.norm(a0); ak = ak / np.linalg.norm(ak)
    v = np.cross(a0, ak)
    c = float(np.dot(a0, ak))
    sn = float(np.linalg.norm(v))
    if sn < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / sn
    th = np.arctan2(sn, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def _tgt(s_, oi, k):
    p0, q0_ = ref_obj[oi][0][:3], ref_obj[oi][0][3:7]
    pk, qk_ = ref_obj[oi][k][:3], ref_obj[oi][k][3:7]
    Rk = _axis_only_R(q0_, qk_)
    return Rk @ w0[s_][0] + (pk - Rk @ p0), Rk @ w0[s_][1]

def _slerp_solve(s_, oi, k, q_from):
    """★L5-11: 小步长跟随 —— 腕部奇异(j5过零, j4/j6转轴对齐)处求解器会在等价构型
    间跳(实测腕目标只动1.5cm/8°, j5却跳42°)。把单步上限压到 1.1°/迭代, 逼它从上一帧
    的解沿局部分支慢慢挪, 就跳不过去了。收敛不了才退回默认步长。"""
    tp1, tR1 = _tgt(s_, oi, k)
    r = ik[s_].solve(tp1, tR1, q0=q_from, iters=600, step_clip=0.02)
    dq = float(np.abs(np.asarray(r["q"]) - q_from).max())
    if dq <= MAX_DQ and r["pos_err"] <= 0.01:
        return r, 1
    if r["pos_err"] > 0.01:                       # 小步长没够着 -> 放开步长重试
        r = ik[s_].solve(tp1, tR1, q0=q_from, iters=200)
        dq = float(np.abs(np.asarray(r["q"]) - q_from).max())
        if dq <= MAX_DQ and r["pos_err"] <= 0.01:
            return r, 1
    tp0, tR0 = _tgt(s_, oi, max(k - 1, 0))
    n_sub = 2
    while n_sub <= SUB_MAX:
        q_cur, ok = q_from.copy(), True
        worst = 0.0
        for j in range(1, n_sub + 1):
            a = j / n_sub
            pj = (1 - a) * tp0 + a * tp1
            Rj = tR0 @ _rot_interp(tR0, tR1, a)
            rj = ik[s_].solve(pj, Rj, q0=q_cur, iters=120)
            d = float(np.abs(np.asarray(rj["q"]) - q_cur).max())
            worst = max(worst, d)
            q_cur = np.asarray(rj["q"], np.float64)
        if worst <= MAX_DQ and rj["pos_err"] <= 0.01:
            return rj, n_sub
        n_sub *= 2
    return rj, n_sub          # 尽力而为, 由后续硬闸报警

def _rot_interp(R0, R1, a):
    """返回 dR 使 R0 @ dR = 插值姿态 (轴角线性插值)."""
    dR = R0.T @ R1
    w = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if w < 1e-8:
        return np.eye(3)
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                   dR[1, 0] - dR[0, 1]]) / (2 * np.sin(w))
    th = w * a
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

sub_tot = 0
for k in range(Nrow):
    for s in ("right", "left"):
        oi = side_obj[s]
        r, nsub = _slerp_solve(s, oi, k, q_seed[s])
        sub_tot += nsub - 1
        if r["pos_err"] > 0.01:
            fail_ik += 1
        err_max = max(err_max, float(r["pos_err"]))
        q_ik[s][k] = r["q"]
        q_seed[s] = np.asarray(r["q"], np.float64)
print(f"[v5] IK 细分总次数={sub_tot} (0=全部一次过)", flush=True)

# ---- ★L5-33 方案C (用户裁定): IK 之后平滑**关节**, 物体轨迹保持带噪 ----
#   由来: 噪声母带 v1 版把噪声一路带进关节, 抓握段逐行跳变 65% 超残差界(32.21),
#   测的变成"参考跳变超出残差能力"而不是"参考有噪声能不能修" —— 问错了问题。
#   正确建模: 重建噪声在**物体轨迹**里(策略看到的参考、皮筋、时钟都吃它),
#   而关节前馈本来就该平滑(真实管线也会这么做)。
#   ★只平滑 IK 关节列; obj_pos/obj_quat 一个字不动 —— 那是消融的本体。
JOINT_SMOOTH = os.environ.get("POUR_REF_JOINT_SMOOTH") == "1"
if JOINT_SMOOTH:
    from scipy.signal import savgol_filter as _sg
    for s_ in ("right", "left"):
        _before_js = q_ik[s_].copy()
        _light = _sg(q_ik[s_], 9, 2, axis=0)
        # ★抓握段(前 40 行)加重平滑并交叉融接。理由: 那一段人手几乎不动,
        #   行间"运动"基本全是噪声→IK 的产物, 重平滑不伤任务; 而倒水段有真运动,
        #   只能用轻窗。首版单一窗9 实测右臂抓握段 P95 5.63° 仍超线(4.58°),
        #   第五道检查当场拒写 —— 检查干了它该干的事。
        _q = _light.copy()
        for _w in (21, 31, 45):
            _heavy = _sg(q_ik[s_], _w, 2, axis=0)
            _q = _light.copy()
            _q[:40] = _heavy[:40]
            _fade = np.linspace(1.0, 0.0, 10)[:, None]        # 行40~49 交叉融接
            _q[40:50] = _fade * _heavy[40:50] + (1 - _fade) * _light[40:50]
            _dg = np.degrees(np.abs(np.diff(_q[:40], axis=0))).max(axis=1)
            if np.percentile(_dg, 95) <= np.degrees(0.08):
                print(f"[v5] ★{s_} 抓握段重平滑收敛于窗{_w}", flush=True)
                break
        q_ik[s_][:] = _q
        _dev_js = np.degrees(np.abs(q_ik[s_] - _before_js)).max()
        print(f"[v5] ★{s_} 关节列已平滑 (轻窗9 + 抓握段重窗): 与未平滑最大偏 "
              f"{_dev_js:.2f}°", flush=True)
    print("[v5] ★方案C: 物体轨迹列**保持带噪**, 仅关节列平滑", flush=True)

# ---- ★L5-33 第五道出厂检查: 抓握段逐行跳变 (这次就是漏了它才白跑两条线) ----
#   抓握/认证发生在交互段最前面(约前 40 行), 那里的跳变超残差界 = 策略结构上
#   跟不上 = G2 永远过不去 = 后面全部白搭。P95 <= 0.08 rad(黄档界)。
_g_bad = 0
for s_ in ("right", "left"):
    _dg = np.degrees(np.abs(np.diff(q_ik[s_][:40], axis=0))).max(axis=1)
    _p95, _mx = np.percentile(_dg, 95), _dg.max()
    ok_g = _p95 <= np.degrees(0.08)
    _amax = int(_dg.argmax())
    _top = np.argsort(_dg)[-5:][::-1]
    print(f"[v5] 第五道·抓握段跳变 {s_}: P95 {_p95:.2f}° 最大 {_mx:.2f}° "
          f"(线 {np.degrees(0.08):.2f}°) {'✅' if ok_g else '❌'}", flush=True)
    print(f"[v5]   诊断 {s_}: 最大跳在交互第{_amax}行 (0=onset/缝口); "
          f"前5大跳行={_top.tolist()} 值={np.round(_dg[_top],1).tolist()}°", flush=True)
    _g_bad += 0 if ok_g else 1
if _g_bad:
    print("[v5] ★第五道出厂检查未过: 抓握段跳变超残差界 —— 不写母带", flush=True)
    raise SystemExit(5)

# ★生成后硬闸: 关节连续性 (v2 母带就是死在这里没查)
_bad = 0
for s in ("right", "left"):
    dd = np.degrees(np.abs(np.diff(q_ik[s], axis=0)))
    mx = dd.max(axis=1)
    p95 = float(np.percentile(mx, 95))
    print(f"[v5] {s} 逐行最大关节跳变: 中位={np.median(mx):.2f}° "
          f"P95={p95:.2f}° 最大={mx.max():.2f}°", flush=True)
    # ★闸门口径 (L5-11 修正): 原 3°/5° 是我拍的, 连真人做的动作都过不了
    # (人手行实测 P95 8.03°/帧、最大 24.04°/帧)。改按"人做得到"定线。
    if mx.max() > 25.0 or p95 > 8.0:
        _bad += 1
        _w = np.where(mx > 5.0)[0]
        print(f"[v5]   {s} 越限行(交互行号): {_w[:12].tolist()}", flush=True)
        for _b in _w[:4]:
            _j = int(dd[_b].argmax())
            print(f"[v5]     行{_b}->{_b+1}: j{_j+1} 跳 {dd[_b,_j]:.1f}°; "
                  f"腕目标位移 {np.linalg.norm(_tgt(s,side_obj[s],_b+1)[0]-_tgt(s,side_obj[s],_b)[0])*100:.2f}cm; "
                  f"腕目标转角 {np.degrees(np.arccos(np.clip((np.trace(_tgt(s,side_obj[s],_b)[1].T@_tgt(s,side_obj[s],_b+1)[1])-1)/2,-1,1))):.2f}°",
                  flush=True)
if _bad:
    if ALLOW_JUMP:
        print("[v5] ⚠连续性硬闸**未过但被放行** (ALLOW_JUMP=1): "
              "要求 P95<8° 且 最大<25°。这条母带带着已知的关节跳变, "
              "**只能用于消融对照, 不得当作正式母带**。", flush=True)
    else:
        print("[v5] ★连续性硬闸未过 (要求 P95<8° 且 最大<25°, 按人手行实测定线) —— 不写母带",
              flush=True)
        raise SystemExit(2)
else:
    print("[v5] ✅ 连续性硬闸通过", flush=True)
print(f"[v5] 交互IK: >1cm 失败 {fail_ik}/{Nrow*2} 最大误差={err_max*100:.2f}cm", flush=True)

d1 = dict(np.load(V1, allow_pickle=True))
# ★锚点从 V1 seg_lens 推导 (pour25 交互 103≠pour17 135, 不能写死 324/350)。
#   seg=[approach,seam1,interact,seam2,retreat]; IA0=approach+seam1, IA1=旧交互末帧, RET0=撤离首帧。
_sl1 = [int(x) for x in d1["seg_lens"]]
IA0 = _sl1[0] + _sl1[1]
IA1 = IA0 + _sl1[2] - 1
RET0 = IA0 + _sl1[2] + _sl1[3]
print(f"[v5] 锚点(从seg_lens推): IA0={IA0} IA1={IA1} RET0={RET0} seg={_sl1}", flush=True)
SEAM2, NRET = RET0 - IA1 - 1, len(d1["right_q"]) - RET0
Tn = IA0 + Nrow + SEAM2 + NRET          # 新全链长
ia_rows = IA0 + np.arange(Nrow)
def _splice(colv, ia_val, tail0=RET0 - SEAM2):
    pre = np.asarray(colv)[:IA0]
    tail = np.asarray(colv)[IA1 + 1:]
    return np.concatenate([pre, ia_val, tail], axis=0)
v2r = _splice(d1["right_q"], q_ik["right"])
v2l = _splice(d1["left_q"], q_ik["left"])
v2rf = _splice(d1["right_f"], np.asarray(d1["right_f"])[IA0 + floor_map])
v2lf = _splice(d1["left_f"], np.asarray(d1["left_f"])[IA0 + floor_map])
# 缝2 重融接 (新交互末行 → retreat 首行)
RET0n = IA0 + Nrow + SEAM2
for i in range(SEAM2):
    a = (i + 1) / (SEAM2 + 1)
    rr = IA0 + Nrow + i
    v2r[rr] = (1-a) * v2r[IA0 + Nrow - 1] + a * v2r[RET0n]
    v2l[rr] = (1-a) * v2l[IA0 + Nrow - 1] + a * v2l[RET0n]
    v2rf[rr] = (1-a) * v2rf[IA0 + Nrow - 1] + a * v2rf[RET0n]
    v2lf[rr] = (1-a) * v2lf[IA0 + Nrow - 1] + a * v2lf[RET0n]
# 物体轨迹/conf/source/frame_of_row 列重拼 (交互段=扩张后; 物体列用母带系原值插值)
obj_cols = {}
for oi in (0, 1):
    raw = np.concatenate([np.asarray(d1[f"obj_pos_{oi}"], np.float64),
                          np.asarray(d1[f"obj_quat_{oi}"], np.float64)], axis=1)
    ia_raw = raw[IA0:IA1 + 1]
    lo = floor_map; hi = np.minimum(lo + 1, Nrow_raw - 1)
    a = (frac_rows - lo)[:, None]
    # ★L5-27: 位置列必须写**归位后**的同一条轨迹。原来这里重读 v1 原始值,
    #   归位后就会出现"IK 按归位轨迹解、判据按原始轨迹打分"的自相矛盾 ——
    #   与上面朝向列那条注释是同一个道理。
    pos = ref_obj[oi][:, :3].copy()
    assert len(pos) == Nrow, (len(pos), Nrow)
    _unused_pos = ia_raw[lo, :3] * (1 - a) + ia_raw[hi, :3] * a
    q0, q1 = ia_raw[lo, 3:7], ia_raw[hi, 3:7]
    sgn = np.sign((q0 * q1).sum(axis=1, keepdims=True)); sgn[sgn == 0] = 1
    q = q0 * (1 - a) + q1 * sgn * a
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    obj_cols[f"obj_pos_{oi}"] = _splice(d1[f"obj_pos_{oi}"], pos)
    # ★平滑后的朝向写回 npz: 判据(皮筋/时钟门/G3)与 IK 必须用同一条轨迹, 否则
    #   前馈按平滑轨迹走, 判据却按原始跳变轨迹打分 —— 自相矛盾。
    q_sm = ref_obj[oi][:, 3:7].copy()
    q_sm = q_sm / np.linalg.norm(q_sm, axis=1, keepdims=True)
    obj_cols[f"obj_quat_{oi}"] = _splice(d1[f"obj_quat_{oi}"], q_sm)
    for cn in (f"conf_pos_{oi}", f"conf_rot_{oi}"):
        obj_cols[cn] = _splice(d1[cn], np.asarray(d1[cn])[IA0 + floor_map])
src_new = _splice(d1["source"], np.ones(Nrow, np.int8))
fr_new = _splice(d1["frame_of_row"],
                 np.asarray(d1["frame_of_row"])[IA0 + floor_map])
segl1 = [int(v) for v in d1["seg_lens"]]
segl1[2] = Nrow
assert sum(segl1) == Tn, (segl1, Tn)
with open(V1, "rb") as fh:
    parent_md5 = hashlib.md5(fh.read()).hexdigest()[:8]
hum_r = _splice(d1["right_q"], np.asarray(d1["right_q"])[IA0 + floor_map])
hum_l = _splice(d1["left_q"], np.asarray(d1["left_q"])[IA0 + floor_map])
hum_rf = _splice(d1["right_f"], np.asarray(d1["right_f"])[IA0 + floor_map])
hum_lf = _splice(d1["left_f"], np.asarray(d1["left_f"])[IA0 + floor_map])
out = dict(d1)
out.update(obj_cols, source=src_new, frame_of_row=fr_new,
           seg_lens=np.asarray(segl1, np.int64),
           right_q=v2r, left_q=v2l, right_f=v2rf, left_f=v2lf,
           human_right_q=hum_r, human_left_q=hum_l,
           human_right_f=hum_rf, human_left_f=hum_lf,
           cert_arm7_right=cert["right"], cert_arm7_left=cert["left"],
           meta_v5=np.array([f"gen=L5-1;parent_v1_md5={parent_md5};betaR={BR};betaL={BL};"
                             f"obj_rest_z=0.959;ik=delta_space;date=2026-08-27;"
                             f"up_local=0.0,1.0,0.0;"
                             f"noise_mult={NOISE_MULT};noise_seed={NOISE_SEED};"
                             f"noise_base_deg={NOISE_BASE_DEG};"
                             f"featloss_allowed={int(ALLOW_FEATLOSS)};"
                             f"conf_smooth={'off' if NOCONF else 'on'};"
                             f"continuity_gate={'bypassed' if ALLOW_JUMP else 'passed'};"
                             f"joint_smooth={int(JOINT_SMOOTH)}"]))
# ═══ ★第四道出厂检查 (L5-27): 末态-判据一致性 ═══
import sys as _sys                                                    # noqa: E402
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "A_Design", "L3_Learning"))
from progress import endstate_consistency as _endchk                  # noqa: E402
_e4 = []
for oi, nm in ((0, "杯"), (1, "瓶")):
    _P = np.asarray(obj_cols[f"obj_pos_{oi}"], np.float64)[IA0:IA0 + Nrow]
    _Q = np.asarray(obj_cols[f"obj_quat_{oi}"], np.float64)[IA0:IA0 + Nrow]
    _ok4, _inf = _endchk(_P, _Q)
    print(f"[v5] 末态-判据一致性 {nm}: 末行距首行={_inf['end_pos_cm']:.2f}cm "
          f"(限{_inf['lim_pos_cm']:.0f}) 末行倾角={_inf['end_tilt_deg']:.1f}° "
          f"(限{_inf['lim_tilt_deg']:.0f}) 末尾连续达标={_inf['tail_ok_rows']}行 "
          f"(需{_inf['need_hold']})", flush=True)
    if not _ok4:
        _e4 += [f"{nm}: {m}" for m in _inf["bad"]]
if _e4:
    print("[v5] ★第四道出厂检查未过 (末态-判据一致性) —— 不写母带:", flush=True)
    for _m in _e4:
        print(f"[v5]     {_m}", flush=True)
    raise SystemExit(4)
print("[v5] ✅ 第四道出厂检查通过 (末态-判据一致性)", flush=True)

np.savez(OUT, **out)
with open(OUT, "rb") as fh:
    print(f"[v5] 已写 {OUT} md5={hashlib.md5(fh.read()).hexdigest()[:8]}", flush=True)
print("[v5] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
