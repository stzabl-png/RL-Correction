"""Tip_Pinch 模板校准: 给 2 指捏取候选烘一个"内压合拢锚点" (2026-08-05).

背景 (screw 27 盖任务, 三个左手 Tip_Pinch 候选实测):
  ① Dexonomy 的接触标在 *_fingertip 极点, 名义抓姿 = 指尖**零距贴**盖壁 (零力);
  ② 模板 c/δ 只能沿 open→grasp 方向动, 而沿该方向加深时**拇指径向外掠**
     (+5~+12mm), 捏不拢 —— 深度扫描/腕位扫描 16 组全部零接触, 结构性死局;
  ③ 接触带在盖下沿 (z≈0.45cm), 装配场景里被瓶肩挡住 (Dexonomy 是对单独放
     在桌上的盖生成的).

修法 (对着圆柱对称性做, 接触几何不变):
  a. 腕位/接触整体沿盖轴上移 dz, 把接触带挪到盖壁上半段 (默认 z=1.1cm);
  b. 对每根参与指, 逐关节坐标下降解一个"指尖目标在盖壁**内侧** press_mm 处"
     的关节位形 → 存成 ``close_anchor`` (22,); c=1 时 PD 顶着壁内目标 = 持续
     内压力, 与五指模板 c>1 深捏同机理. RL 的每指 δ 沿该方向 = 压力大小旋钮.

用**系统 python3** 跑 (纯 URDF FK, 无 Isaac):
  python3 tasks/pregrasp/calib_tip_pinch.py \
      --prior tasks/pregrasp/priors/Screw27_cap_candidates/31_5.npz \
      --out   tasks/pregrasp/priors/Screw27_cap_candidates/31_5_calib.npz
"""
import argparse

import numpy as np

from rl_rebuild.correction.kinematics import quat_to_R
from rl_rebuild.correction.ref_builders.replay_grasp import (
    GENERIC_JOINT_ORDER,
    _urdf,
)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def tip_pos(u, zrow, q22, finger, side, link="fingertip"):
    qd = {n.replace("right_", f"{side}_"): float(v)
          for n, v in zip(GENERIC_JOINT_ORDER, q22)}
    Th = np.eye(4)
    Th[:3, :3] = quat_to_R(zrow[3:7])
    Th[:3, 3] = zrow[:3]
    return u.link_pose(f"{side}_{finger}_{link}", qd, Th,
                       f"{side}_hand_C_MC")[:3, 3]


def solve_finger(u, zrow, q22, finger, target, side, iters=120, link="fingertip"):
    """坐标下降: 只动该指的关节, 让指定 link 逼近 target (物体系)."""
    jids = [i for i, n in enumerate(GENERIC_JOINT_ORDER) if f"_{finger}_" in n]
    q = q22.copy()
    lo = np.full(22, -2.6)
    hi = np.full(22, 2.6)      # SharpaWave 关节界近似; env 下发时还会按 dof 钳
    best = np.linalg.norm(tip_pos(u, zrow, q, finger, side, link) - target)
    step = 0.12
    for it in range(iters):
        improved = False
        for j in jids:
            for s in (+step, -step):
                t = q.copy()
                t[j] = np.clip(t[j] + s, lo[j], hi[j])
                d = np.linalg.norm(tip_pos(u, zrow, t, finger, side, link) - target)
                if d < best - 1e-6:
                    q, best, improved = t, d, True
        if not improved:
            step *= 0.5
            if step < 0.004:
                break
    return q, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--side", default="left")
    p.add_argument("--wall_z", type=float, default=0.011,
                   help="接触带目标高度 (物体系 m); 盖高 0.017")
    p.add_argument("--press_mm", type=float, default=6.0,
                   help="目标探进盖壁内侧的深度")
    p.add_argument("--link", default="fingertip", choices=["fingertip", "elastomer"],
                   help="解哪个 link 到目标: fingertip=指尖极点 (低摩擦 DP 区, "
                        "传感器盲); elastomer=胶垫原点 (高摩擦垫脸, 传感器认这个). "
                        "实测垫原点距垫面约 5mm, elastomer 模式的目标半径按此补偿")
    p.add_argument("--fingers", default="thumb,index")
    args = p.parse_args()

    z = dict(np.load(args.prior))
    zg = z["grasp"].astype(np.float64).copy()
    cts = z["contact_pos"].astype(np.float64).copy()
    u = _urdf()

    # a) 沿盖轴上移: 接触带 → wall_z (圆柱壁平移对称, 接触几何不变)
    dz = args.wall_z - float(cts[:, 2].mean())
    zg[2] += dz
    cts[:, 2] += dz
    zpre = z["pregrasp"].astype(np.float64).copy()
    zpre[:, 2] += dz
    print(f"[calib] 接触带上移 dz={dz*1000:+.1f}mm -> z̄={cts[:, 2].mean()*100:.2f}cm "
          f"(盖高 1.7cm, 避开瓶肩)")

    # b) 逐指解内压锚点: 目标 = 该指最近接触点方向上的半径目标点
    #    fingertip 模式: 半径 = 壁(1.75cm) − press_mm
    #    elastomer 模式: 垫**原点**半径 = 壁 + 5mm(原点到垫面) − press_mm
    q_anchor = zg[7:29].copy()
    pad_face_off = 0.005
    for f in args.fingers.split(","):
        t0 = tip_pos(u, zg, zg[7:29], f, args.side, args.link)
        d = np.linalg.norm(cts - t0, axis=1)
        c = cts[d.argmin()]
        radial = c[:2] / max(np.linalg.norm(c[:2]), 1e-9)
        wall_r = float(np.linalg.norm(c[:2]))
        if args.link == "elastomer":
            tgt_r = wall_r + pad_face_off - args.press_mm / 1000.0
        else:
            tgt_r = wall_r - args.press_mm / 1000.0
        target = np.array([radial[0] * tgt_r, radial[1] * tgt_r, args.wall_z])
        q_anchor, err = solve_finger(u, zg, q_anchor, f, target, args.side,
                                     link=args.link)
        r_now = np.linalg.norm(tip_pos(u, zg, q_anchor, f, args.side,
                                       args.link)[:2])
        print(f"[calib] {f}[{args.link}]: 目标半径 {tgt_r*100:.2f}cm, "
              f"解残差 {err*1000:.1f}mm, 实际半径 {r_now*100:.2f}cm (壁 {wall_r*100:.2f})")
        assert err < 0.006, f"{f} 内压锚点解不到位 ({err*1000:.1f}mm), 候选几何不适配"

    # c) 非参与指满卷收拢 (GENERIC_CLOSED 拳式), 收出捏取平面 —— 实测钉在
    #    Dexonomy 原收拢位时中/环指 (无传感器的连杆) 会先于拇食指压到盖/颈,
    #    把整套装配推走 (pushed 终止), 且力对判据不可见.
    from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_CLOSED
    used = set(args.fingers.split(","))
    tucked = [f for f in FINGERS if f not in used]
    for j, n in enumerate(GENERIC_JOINT_ORDER):
        if any(f"_{f}_" in n for f in tucked):
            q_anchor[j] = float(GENERIC_CLOSED[j])
    # 收拢后离线核验: 非参与指的各节到盖轴的最小水平距离
    qd = {n.replace("right_", f"{args.side}_"): float(v)
          for n, v in zip(GENERIC_JOINT_ORDER, q_anchor)}
    Th = np.eye(4)
    Th[:3, :3] = quat_to_R(zg[3:7])
    Th[:3, 3] = zg[:3]
    worst = 1e9
    for f in tucked:
        for link in ("MP", "DP", "elastomer"):
            try:
                pos = u.link_pose(f"{args.side}_{f}_{link}", qd, Th,
                                  f"{args.side}_hand_C_MC")[:3, 3]
            except Exception:
                continue
            r = float(np.linalg.norm(pos[:2]))
            zh = float(pos[2])
            # 只在盖/颈高度带 (z -1~3cm) 里查径向余量
            if -0.01 < zh < 0.03:
                worst = min(worst, r - 0.0175)
    msg = f"{worst*100:.1f}cm" if worst < 1e8 else "不在高度带 (安全)"
    print(f"[calib] 非参与指 {tucked} 满卷收拢, 盖高度带内最小径向余量: {msg}")
    if worst < 0.008 and worst < 1e8:
        print("[calib] ⚠ 收拢后仍逼近盖壁 (<8mm), 训练时留意 pushed 率")

    z["grasp"] = zg
    z["pregrasp"] = zpre
    z["contact_pos"] = cts
    z["contact_centroid"] = cts.mean(axis=0)
    z["close_anchor"] = q_anchor          # (22,) GENERIC 序, c=1 的合拢锚点
    np.savez(args.out, **z)
    print(f"[calib] -> {args.out}")


if __name__ == "__main__":
    main()
