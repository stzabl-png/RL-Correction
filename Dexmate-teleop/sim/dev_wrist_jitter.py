#!/usr/bin/env python3
"""腕部微振荡量尺:量 j6/j7 的「小幅来回抖」,回答用户看到的手腕抖动来自哪里。

    .venv/bin/python sim/dev_wrist_jitter.py 关节CSV [关节CSV ...]

与既有尺子的分工:`regress_teleop` 的「>5° 跳变」抓大跳,这里抓**方向来回翻转
的小幅振荡**(0.1° 到 5° 之间的反向摆动)——肉眼看到的"抖"正是这种。

对每份 CSV、对 L/R 的 j6、j7(以及对照组肩部 j2)报:
  - 反转率:指令序列每秒方向反转多少次(幅度落在微抖带内才计);
  - 反转幅度:这些来回摆动的平均/95 分位幅度(度);
  - 实际执行侧(got_* 列,若有)同样报一份 —— 区分「指令就在抖」与
    「指令平稳、物理在抖」。

尺子自检(--self-check,不需要任何 CSV):
  ① 纯匀速斜坡必须报 0 反转;② 叠加 0.5° 10Hz 正弦的斜坡必须报出接近
  2×10=20 次每秒的反转率、幅度接近 1°;③ 每 2 秒一次的 10° 台阶(大跳)
  不许计入微抖带。三条全过,出的数才可信。
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

CTRL_HZ = 50.0
BAND = (0.1, 5.0)          # 微抖带:反转幅度在此区间才算"抖"
JOINTS = ["L_arm_j6", "L_arm_j7", "R_arm_j6", "R_arm_j7", "R_arm_j2"]


def reversals(q_deg: np.ndarray) -> tuple[float, float, float]:
    """返回 (每秒反转次数, 平均幅度, p95 幅度)。

    做法:把序列分成单调段,相邻段交界即一次方向反转;该次反转的幅度取
    后一段的行程。只统计幅度落在微抖带内的反转。
    """
    d = np.diff(q_deg)
    d = d[np.abs(d) > 1e-9]                 # 完全不动的帧不构成方向信息
    if len(d) < 2:
        return 0.0, 0.0, 0.0
    sign = np.sign(d)
    seg_end = np.where(np.diff(sign) != 0)[0]
    amps = []
    prev = 0
    runs = np.split(d, seg_end + 1)
    for run in runs[1:]:                    # 每个新段的开始 = 一次反转
        amp = float(np.abs(run.sum()))
        if BAND[0] <= amp <= BAND[1]:
            amps.append(amp)
    dur_s = len(q_deg) / CTRL_HZ
    if not amps:
        return 0.0, 0.0, 0.0
    return len(amps) / dur_s, float(np.mean(amps)), float(np.percentile(amps, 95))


def analyse(path: pathlib.Path):
    rows = list(csv.DictReader(open(path)))
    print(f"\n== {path.name}({len(rows)} 帧) ==")
    print(f"{'关节':<10}{'指令:次/秒':>10}{'均幅°':>8}{'p95°':>8}"
          f"{'执行:次/秒':>12}{'均幅°':>8}")
    for jn in JOINTS:
        ccol, acol = f"cmd_{jn}", f"got_{jn}"
        if ccol not in rows[0]:
            continue
        c = np.degrees(np.array([float(r[ccol]) for r in rows]))
        rc = reversals(c)
        line = f"{jn:<10}{rc[0]:>10.1f}{rc[1]:>8.2f}{rc[2]:>8.2f}"
        if acol in rows[0] and rows[0][acol]:
            a = np.degrees(np.array([float(r[acol]) for r in rows]))
            ra = reversals(a)
            line += f"{ra[0]:>12.1f}{ra[1]:>8.2f}"
        print(line)


def self_check() -> int:
    t = np.arange(0, 10, 1 / CTRL_HZ)
    ramp = 3.0 * t                                        # 匀速斜坡
    r1 = reversals(ramp)
    ok1 = r1[0] == 0.0
    print(f"[自检1] 匀速斜坡反转率 {r1[0]:.1f}(应为 0){'✅' if ok1 else '❌'}")
    wob = ramp * 0 + 0.5 * np.sin(2 * np.pi * 10 * t)     # 0.5° 10Hz 摆动
    r2 = reversals(wob)
    ok2 = 15 <= r2[0] <= 25 and 0.5 <= r2[1] <= 1.5
    print(f"[自检2] 0.5°/10Hz 摆动:反转率 {r2[0]:.1f}(应≈20)"
          f" 均幅 {r2[1]:.2f}°(应≈1){'✅' if ok2 else '❌'}")
    steps = np.repeat(np.arange(5) * 10.0, 100)           # 每 2 秒 10° 台阶
    r3 = reversals(steps)
    ok3 = r3[0] == 0.0
    print(f"[自检3] 10° 台阶(大跳)反转率 {r3[0]:.1f}(应为 0,不入微抖带)"
          f"{'✅' if ok3 else '❌'}")
    return 0 if (ok1 and ok2 and ok3) else 1


def main() -> int:
    if "--self-check" in sys.argv or len(sys.argv) < 2:
        rc = self_check()
        if len(sys.argv) < 2:
            return rc
        if rc:
            return rc
    for p in sys.argv[1:]:
        if p == "--self-check":
            continue
        analyse(pathlib.Path(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
