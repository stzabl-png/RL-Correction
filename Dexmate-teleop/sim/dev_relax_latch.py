#!/usr/bin/env python3
"""`--relax-at-limit` 的自锁回归:肩关节顶限位**不得**触发朝向放松。

    cd <仓库根目录> && .venv/bin/python sim/dev_relax_latch.py

2026-08-10 查清的失效(此前记作「对机器负载极度敏感,根因未知」):

  放松触发条件读的是**全部 7 个关节**的最小 slack,而这个机制要治的失效只有
  腕部 j6/j7 被掌心朝向推到限位这一种(实测里顶死的就是它们,workflow 里也
  写明「决定朝向能否达到的只有最后两个」)。于是存在一个自锁:

    任何扰动让**肩肘**(j2/j3/j4)短暂贴限位 -> 触发,把这条臂的朝向权重
    压到 1% -> 失去朝向约束的手臂被 3 自由度位置任务放任游荡,肩关节持续
    贴死 -> slack 恒 0 -> 放松永不解除。

  三次「负载下变糟」的跑,主力顶限位关节全是 j2(肩):base_guarded2 是
  L_arm_j2 4275 帧(实时倍率 2.37,根本没怎么慢),guard_loaded 是 R_arm_j2
  4279 帧、单段连续 35.4 秒;而空载好跑里 j2 只有零星几帧。锁在哪条臂,
  哪条腕的误差就翻倍(34.6mm / 53.2mm),另一条纹丝不动。**负载不是根因,
  只是把双稳态系统踢进吸收态的扰动源** —— 这就是为什么对 slack 做 EMA
  平滑(治单帧噪声的)治不了它,也是为什么同配置两跑能一次 2.57%、
  一次 65.06%。

修法:slack 只看 j6/j7。肩肘顶限位是位置/冗余的事,放松朝向既不对症,还把
唯一能把手臂拉回来的约束拿掉了。

本测试不依赖 Isaac,直接驱动 PinkVegaIK:
  1. 【肩不触发】把 R_arm_j2 摆到限位上,解一步,朝向权重必须**纹丝不动**;
     —— 旧代码在这里失败(权重被压到 1%),这就是自锁的入口。
  2. 【锁进去要能出来】从 j2 贴限位出发跟踪一个可达目标 3 秒,结束时权重
     必须恢复、j2 必须离开限位、朝向误差必须收敛。
  3. 【腕仍要触发】把 R_arm_j6 摆到限位上,解一步,权重必须确实被压下去
     —— 防止修法把机制本身修没了;再跟踪可达目标,权重要能自行恢复
     (posture 把 j6 拉回量程中央,放松自解除,这正是它和自锁的区别)。
  4. 【饱和时不许贴墙】持续命令一个超出腕量程的朝向,放松必须触发,且
     结束时腕关节必须停在「贴限位」判据(1°)之外 —— 放松曲线带 2° 地板
     就是为这一条:没有地板时平衡点在 slack 约 0.5°,放松开着、指标却纹丝
     不动(2026-08-10 空载实测 13.64%,与基线 14.15% 无差别)。
  5. 【位置够不着时不许磨墙】(--relax-pos-at-limit)持续命令一个超出臂展
     的位置,结束时肩肘腕都必须停在判据之外,末端仍要尽量接近(不许因此
     缩回去)。动机:clip_headwaist 上腕部残余贴限位的 94% 与同臂肩肘同时
     发生 —— 位置任务在够不着时借腕关节抓毫米,把整条手臂压在限位上磨。
  6. 【位置插值不许影响正常跟踪】开着 --relax-pos-at-limit 跟踪可达目标,
     位置误差必须与不开时一样收敛到毫米级。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

ORI0 = 0.5          # teleop 默认 --pink-ori-cost
RELAX = 0.01        # 推荐档
MARGIN = 3.0


def build(pos: bool = False) -> PinkVegaIK:
    return PinkVegaIK(dt=1.0 / 50.0, orientation_cost=ORI0, substeps=3,
                      relax_at_limit=RELAX, relax_margin_deg=MARGIN,
                      relax_pos_at_limit=pos)


def ee_pose(ik: PinkVegaIK, hand: str):
    ik.refresh_fk()
    m = ik.data.oMf[ik.model.getFrameId(EE_FRAME[hand])]
    return m.translation.copy(), m.rotation.copy()


def ori_weight(ik: PinkVegaIK, hand: str) -> float:
    return float(np.mean(ik.frame_tasks[hand].cost[3:]))


def ang_deg(Ra, Rb) -> float:
    return float(np.degrees(np.abs(pin.log3(Ra.T @ Rb)).max()))


def slack_deg(ik: PinkVegaIK, joint: str) -> float:
    i = ik.pin_names.index(joint)
    q = ik.config.q[i]
    return float(np.degrees(min(q - ik.model.lowerPositionLimit[i],
                                ik.model.upperPositionLimit[i] - q)))


def pin_joint(ik: PinkVegaIK, joint: str):
    """把一个关节摆到它的下限上(模拟「游荡到限位」),其余关节保持默认。"""
    i = ik.pin_names.index(joint)
    q = ik.config.q.copy()
    q[i] = ik.model.lowerPositionLimit[i]
    ik.config.q = q
    ik.config.update()


def track(ik: PinkVegaIK, tgt: dict, steps: int):
    for _ in range(steps):
        for hand, (p, quat) in tgt.items():
            ik.set_target(hand, p, quat)
        ik.solve()


def quat_wxyz(R) -> np.ndarray:
    q = pin.Quaternion(R)
    return np.array([q.w, q.x, q.y, q.z])


def main() -> int:
    bad = 0

    # 可达目标 = 默认姿势自己的末端位姿(两只手都设,避免另一只手乱跑搅局)
    ik = build()
    tgt = {}
    for hand in EE_FRAME:
        p, R = ee_pose(ik, hand)
        tgt[hand] = (p, quat_wxyz(R))
    R_home = {h: ee_pose(build(), h)[1] for h in EE_FRAME}

    # ---- 1. 肩关节顶限位,朝向权重必须纹丝不动 --------------------------
    ik = build()
    pin_joint(ik, "R_arm_j2")
    track(ik, tgt, 1)
    w = ori_weight(ik, "right")
    ok = abs(w - ORI0) < 1e-9
    print(f"[1] R_arm_j2(肩)贴限位解一步:朝向权重 {w:.4f} (满权重 {ORI0})"
          f"  {'✅ 未触发' if ok else '❌ 被触发了 —— 这就是自锁的入口'}")
    bad += 0 if ok else 1

    # ---- 2. 从肩贴限位出发,3 秒内必须爬出来 ----------------------------
    track(ik, tgt, 149)
    w = ori_weight(ik, "right")
    s = slack_deg(ik, "R_arm_j2")
    _, R = ee_pose(ik, "right")
    e = ang_deg(R, R_home["right"])
    ok = w > 0.9 * ORI0 and s > MARGIN and e < 5.0
    print(f"[2] 跟踪可达目标 3 秒后:权重 {w:.4f}  j2 距限位 {s:.1f}°"
          f"  朝向误差 {e:.1f}°  {'✅ 出来了' if ok else '❌ 锁死在里面'}")
    bad += 0 if ok else 1

    # ---- 3. 腕关节顶限位仍要触发,且能自行解除 --------------------------
    ik = build()
    pin_joint(ik, "R_arm_j6")
    track(ik, tgt, 1)
    w1 = ori_weight(ik, "right")
    trig = w1 < 0.5 * ORI0
    print(f"[3a] R_arm_j6(腕)贴限位解一步:朝向权重 {w1:.4f}"
          f"  {'✅ 触发了' if trig else '❌ 没触发 —— 机制被修没了'}")
    bad += 0 if trig else 1
    track(ik, tgt, 149)
    w2 = ori_weight(ik, "right")
    s = slack_deg(ik, "R_arm_j6")
    ok = w2 > 0.9 * ORI0 and s > MARGIN
    print(f"[3b] 跟踪可达目标 3 秒后:权重 {w2:.4f}  j6 距限位 {s:.1f}°"
          f"  {'✅ 自行解除' if ok else '❌ 解除不了'}")
    bad += 0 if ok else 1

    # ---- 4. 不可达朝向:放松要触发,腕不许停在墙上 ----------------------
    ik = build()
    p, R = ee_pose(ik, "right")
    axis = R[:, 0]                       # EE 红色轴(垂直掌面,见 record_of_relationship)
    R_bad = pin.exp3(np.radians(150.0) * axis) @ R
    tgt_bad = dict(tgt)
    tgt_bad["right"] = (p, quat_wxyz(R_bad))
    ik.relax_stat["n"] = 0
    track(ik, tgt_bad, 300)
    fired = ik.relax_stat["n"] > 0
    smin = min(slack_deg(ik, f"R_arm_j{i}") for i in (6, 7))
    ok = fired and smin > 1.0
    print(f"[4] 命令超量程朝向 6 秒:放松触发 {ik.relax_stat['n']} 步  "
          f"腕关节距限位最近 {smin:.1f}°  "
          f"{'✅ 没贴墙' if ok else ('❌ 停在墙上' if fired else '❌ 放松根本没触发(前提不成立,换轴/角度)')}")
    bad += 0 if ok else 1

    # ---- 5. 位置够不着:肩肘腕都不许磨墙,末端仍要尽量伸过去 -------------
    ik = build(pos=True)
    p, R = ee_pose(ik, "right")
    # 方向有讲究:朝 −x 伸的够不着停在**臂展边界奇异点**上,全臂离限位都
    # 27° 以上,根本不顶限位(实测)。顶限位的够不着是**关节量程**先到头的
    # 方向 —— 往身体对侧+下,不开插值时 R_arm_j2 被压到 0.0°(实测),与
    # clip_headwaist 里 j2 贴下限的真实失效同型。
    # 朝向目标给**当前朝向**(每步刷新):真人横扫时掌向跟着手转,纯位置
    # 不可达才是 clip 数据里的主情形;「位置和朝向同时不可达」是另一个
    # 场景,见第 7 条。
    out = p + np.array([0.0, +0.5, -0.6])
    for _ in range(300):
        _, R_now = ee_pose(ik, "right")
        ik.set_target("right", out, quat_wxyz(R_now))
        ik.set_target("left", tgt["left"][0], tgt["left"][1])
        ik.solve()
    fired = ik.relax_stat_pos["n"] > 0
    smin = min(slack_deg(ik, f"R_arm_j{i}") for i in range(1, 8))
    pe, _ = ee_pose(ik, "right")
    reach = float(np.linalg.norm(pe - p))     # 相对起点伸出了多远
    ok = fired and smin > 1.0 and reach > 0.05
    print(f"[5] 命令超臂展位置 6 秒(掌向随手):插值触发 {ik.relax_stat_pos['n']} 步  "
          f"全臂距限位最近 {smin:.1f}°  末端伸出 {reach*1000:.0f}mm  "
          f"{'✅ 不磨墙且仍在伸' if ok else ('❌' if fired else '❌ 没触发(前提不成立)')}")
    bad += 0 if ok else 1

    # ---- 7. 位置和朝向同时不可达:记录现状,只要求不锁死 -----------------
    # 这个复合场景下 j2 可能被朝向任务压在限位上(朝向触发只看 j6/j7,是
    # 防自锁的刻意取舍)。这里只要求:需求回到可达范围后 3 秒内全臂离墙、
    # 误差收敛 —— 即「顶过之后能回来」,这正是用户抱怨的反面。
    ik = build(pos=True)
    tgt_both = dict(tgt)
    tgt_both["right"] = (out, quat_wxyz(R))       # 位置超臂展 + 掌向钉死 home
    track(ik, tgt_both, 300)
    smin_dur = min(slack_deg(ik, f"R_arm_j{i}") for i in range(1, 8))
    track(ik, tgt, 150)                           # 需求回到可达
    smin_after = min(slack_deg(ik, f"R_arm_j{i}") for i in range(1, 8))
    _, R_end = ee_pose(ik, "right")
    e = ang_deg(R_end, R_home["right"])
    ok = smin_after > MARGIN and e < 5.0
    print(f"[7] 位置+朝向同时不可达(期间最紧 {smin_dur:.1f}°),需求回可达 3 秒后:"
          f"全臂距限位最近 {smin_after:.1f}°  朝向误差 {e:.1f}°  "
          f"{'✅ 能回来' if ok else '❌ 回不来 —— 这才是要修的'}")
    bad += 0 if ok else 1

    # ---- 6. 开着位置插值,正常跟踪不许受影响 ----------------------------
    ik = build(pos=True)
    track(ik, tgt, 150)
    pe, _ = ee_pose(ik, "right")
    err = float(np.linalg.norm(pe - tgt["right"][0])) * 1000
    ok = err < 5.0
    print(f"[6] 开位置插值跟踪可达目标 3 秒:位置误差 {err:.1f}mm  "
          f"{'✅ 不受影响' if ok else '❌ 正常跟踪被搞坏了'}")
    bad += 0 if ok else 1

    print(f"\n{'✅ 全部通过' if bad == 0 else f'❌ {bad} 项失败'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
