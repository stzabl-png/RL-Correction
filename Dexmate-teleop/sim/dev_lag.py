#!/usr/bin/env python3
"""量「操作者动了之后,指令过多久才跟上」—— 滤波器到底滞后多少毫秒。

    cd <仓库根目录> && .venv/bin/python sim/dev_lag.py trace.csv [trace2.csv ...]

第 4 条(真机跟不上人)拆成两半:**慢**是速度上限,物理的;**迟**是滞后,
软件能拿回来。此前只知道两级 one-euro(末端 1.0 Hz、关节 6.0 Hz)「大概 90 到
215 毫秒」,那是按一阶低通估算的,从来没量过。

`--trace-csv` 里同时有 `human_*`(操作者手腕相对腰)和 `cmd_*`(解算器解出来的
末端目标)。两者在不同的坐标系里、尺度也不同,所以**不比位置,比速度大小**:
速度大小是标量、与坐标系无关,滞后表现为两条曲线的时间平移。取互相关的峰值
位置就是滞后。

自检(必须过,否则数字不可信)
  * 把 human 自己和自己错开一个已知的帧数,量出来必须等于那个帧数;
  * 峰值相关系数太低(<0.3)说明两条曲线根本不像,此时的滞后没有意义,会报出来。
"""
from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np


def load(path: pathlib.Path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for h in ("right", "left"):
        r = [x for x in rows if x["hand"] == h]
        if len(r) < 50:
            continue
        t = np.array([float(x["t"]) for x in r])
        hu = np.array([[float(x[f"human_{a}"]) for a in "xyz"] for x in r])
        cm = np.array([[float(x[f"cmd_{a}"]) for a in "xyz"] for x in r])
        ok = np.isfinite(hu).all(axis=1) & np.isfinite(cm).all(axis=1)
        out[h] = (t[ok], hu[ok], cm[ok])
    return out


def speed(t, p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t)
    return d / np.maximum(dt, 1e-9)


def lag_ms(t, a, b, max_ms=600.0):
    """b 相对 a 滞后多少毫秒(正 = b 在后)。返回 (滞后, 峰值相关系数)。"""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan"), 0.0
    a /= a.std()
    b /= b.std()
    dt = float(np.median(np.diff(t)))
    k = int(max_ms / 1000.0 / dt)
    best, bl = -2.0, 0
    for s in range(0, k + 1):
        n = len(a) - s
        if n < 50:
            break
        c = float(np.dot(a[:n], b[s:s + n]) / n)
        if c > best:
            best, bl = c, s
    return bl * dt * 1000.0, best


def report(path: pathlib.Path):
    data = load(path)
    if not data:
        print(f"{path.name}: 没有可用的行(需要 human_* 与 cmd_* 都是有限值)")
        return
    print(f"\n{path.name}")
    for h, (t, hu, cm) in data.items():
        sh, sc = speed(t, hu), speed(t, cm)
        ms, c = lag_ms(t[1:], sh, sc)
        note = "" if c >= 0.3 else "   <-- 相关性太低,这个滞后不可信"
        print(f"  {h:5s} 滞后 {ms:6.1f} ms   峰值相关 {c:.2f}   "
              f"{len(t)} 帧 / {t[-1] - t[0]:.0f} 秒{note}")
        if h == "right" and c >= 0.3:
            ascii_plot(t[1:], sh, sc, ms)
    # 自检:把 human 自己错开已知帧数,量出来必须等于它
    t, hu, _ = next(iter(data.values()))
    sh = speed(t, hu)
    dt = float(np.median(np.diff(t)))
    for shift in (3, 8):
        ms, c = lag_ms(t[1:][shift:], sh[shift:], np.roll(sh, shift)[shift:])
        want = shift * dt * 1000.0
        ok = abs(ms - want) < dt * 1000.0 * 1.5
        print(f"  [自检] 人为错开 {shift} 帧({want:.0f} ms)-> 量得 {ms:.0f} ms "
              f"{'✓' if ok else '✗ 尺子坏了'}")


def ascii_plot(t, a, b, ms, width=72, rows=9, secs=6.0):
    """把两条速度曲线画成字符图 —— 滞后要能**看见**,不能只是一个数字。

    `.` 是操作者,`#` 是指令。两条形状相同、指令整体靠右,右移的格数乘以每格
    的秒数就是滞后。这是同一件事的第二种表达,和互相关那个数互为佐证。
    """
    m = t <= t[0] + secs
    t, a, b = t[m], a[m], b[m]
    if len(t) < width:
        return
    idx = np.linspace(0, len(t) - 1, width).astype(int)
    a, b = a[idx], b[idx]
    hi = max(a.max(), b.max()) or 1.0
    grid = [[" "] * width for _ in range(rows)]
    for x in range(width):
        for arr, ch in ((a, "."), (b, "#")):
            y = int((rows - 1) * (1 - arr[x] / hi))
            grid[max(0, min(rows - 1, y))][x] = ch
    print(f"    前 {secs:.0f} 秒的速度大小(. 操作者 / # 指令),"
          f"每格 {secs / width * 1000:.0f} ms,滞后约 {ms:.0f} ms")
    for r in grid:
        print("    |" + "".join(r))
    print("    +" + "-" * width)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for a in sys.argv[1:]:
        report(pathlib.Path(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
