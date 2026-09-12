#!/usr/bin/env python3
"""「关节转到限位,之后动作全不准」—— 把这个失效量出来。

    cd <仓库根目录> && .venv/bin/python sim/dev_limit_lock.py 关节CSV [关节CSV ...]

用户报的现象是:做幅度稍大或者胳膊肘旋转的动作时,机器人自己把关节转到限位或
别的很奇怪的姿态,**导致接下来的动作都不准了**。「接下来」这三个字是关键 ——
它说明这不是一次性的误差,而是一个**进去了就出不来**的状态。

所以要量的不是「有没有碰到限位」,而是:
  1. 有多少帧至少有一个手臂关节贴在限位上(1 度以内);
  2. **最长的一段连续贴限位有多久** —— 这才对应「接下来都不准」;
  3. 贴住的是哪些关节。零空间那三个(j1/j3/j5)贴住,和腕部关节贴住,是两回事。

关节角从回归台留下的 `cmd_*` 列读(那是**指令**,即求解器解出来的),限位从
URDF 读。两者都不是我编的。
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.environ.get(
    "VEGA_URDF",
    str(pathlib.Path.home() / "Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf"))
NEAR_DEG = 1.0        # 距限位多近算「贴住」
CTRL_HZ = 50.0        # 回归台跑的控制频率,用来把帧数换算成秒


def urdf_limits(path: str) -> dict[str, tuple[float, float]]:
    root = ET.parse(path).getroot()
    out = {}
    for j in root.iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def analyse(csv_path: pathlib.Path, lim: dict) -> dict:
    # 文件还在被写就不要读。2026-08-08 我在这里栽过一次:回归还在跑,我读到
    # 半截 CSV(3865 帧,完整是 6889),据此差点报出一个相反的结论。半截文件
    # 读起来和完整文件一模一样,没有任何东西会提示你。
    age = time.time() - csv_path.stat().st_mtime
    if age < 10.0:
        raise SystemExit(
            f"{csv_path.name} 在 {age:.0f} 秒前还被写过,可能正在生成中。"
            f"等它写完再来 —— 半截文件不会报错,只会给出错的答案。")
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0] if c.startswith("cmd_") and "_arm_j" in c]
    names = [c[len("cmd_"):] for c in cols]
    missing = [n for n in names if n not in lim]
    if missing:
        raise SystemExit(f"URDF 里没有这些关节的限位:{missing[:4]} —— 尺子不对,先修尺子")
    q = np.array([[float(r[c]) for c in cols] for r in rows])
    lo = np.array([lim[n][0] for n in names])
    hi = np.array([lim[n][1] for n in names])
    near = np.radians(NEAR_DEG)
    pinned = (q <= lo + near) | (q >= hi - near)          # (帧, 关节)
    any_pinned = pinned.any(axis=1)
    per_joint = pinned.sum(axis=0)
    worst = int(np.argmax(per_joint))
    return {
        "帧数": len(q),
        "贴限位的帧占比%": float(100 * any_pinned.mean()),
        "最长连续贴限位(秒)": longest_run(any_pinned) / CTRL_HZ,
        "贴得最多的关节": names[worst],
        "该关节贴住帧数": int(per_joint[worst]),
        "涉及的关节": [names[i] for i in np.argsort(-per_joint) if per_joint[i] > 0][:5],
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    lim = urdf_limits(URDF)
    print(f"[尺子] 从 URDF 读到 {len(lim)} 个关节限位;"
          f"贴限位判据 = 距上下限 {NEAR_DEG}° 以内")
    # 自检:限位必须是 lower < upper,且手臂关节都在
    arm = [n for n in lim if "_arm_j" in n]
    assert len(arm) == 14, f"手臂关节应有 14 个,读到 {len(arm)}"
    assert all(lim[n][0] < lim[n][1] for n in arm), "有关节的上下限反了,尺子坏了"
    print(f"[尺子] 自检通过:14 个手臂关节,上下限方向正确 "
          f"(例 R_arm_j5 {np.degrees(lim['R_arm_j5'][0]):.0f}° 到 "
          f"{np.degrees(lim['R_arm_j5'][1]):.0f}°)")

    for p in sys.argv[1:]:
        r = analyse(pathlib.Path(p), lim)
        print(f"\n{pathlib.Path(p).name}")
        print(f"  帧数 {r['帧数']}   贴限位帧占比 {r['贴限位的帧占比%']:.2f}%   "
              f"最长连续 {r['最长连续贴限位(秒)']:.1f} 秒")
        if r["涉及的关节"]:
            print(f"  涉及:{'、'.join(r['涉及的关节'])}(最多的是 "
                  f"{r['贴得最多的关节']},{r['该关节贴住帧数']} 帧)")
        else:
            print("  没有任何关节贴到限位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
