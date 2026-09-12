#!/usr/bin/env python3
"""拍手测试:量出手和手臂两路数据之间还剩多少时间偏移。

    # 录一段「双手快速合拢拍一下」(页面上开始新一段 -> 开始录制 -> 拍手 -> 停止)
    cd <仓库根目录>
    .venv/bin/python scripts/check_clap_sync.py data/sessions/<名称>

**为什么必须做这一步**:各路数据现在都打主机墙上时钟,所以**戳**是对齐的;但
各路的采集延迟不同(手套→重定向 与 PICO→producer→consumer 不是一条路径),
**戳对齐不等于动作对齐**。合并本身不会报错,错了也看不出来 —— 只会把一份
「看起来同步、其实差几十毫秒」的数据固化下来,那比两个分开的文件更糟。

拍击瞬间是一个两路都能看见的尖锐事件:
  - 手那一路:手指关节角出现拐点(合拢到最紧再回弹);
  - 手臂那一路:两只手的指令末端间距出现极小值。
两个拐点的时刻之差就是残余偏移。

判据
  |偏移| < 20 ms  ——  半个控制周期,可以直接合并;
  偏移是常数     ——  记进 attrs,离线补偿即可;
  偏移在漂       ——  **不要合并**,先查哪一路的时间戳不可信。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arm_gap(session: pathlib.Path):
    """手臂那一路:两只手指令末端的间距随时间。"""
    import msgpack
    p = session / "arm.msgpack"
    if not p.exists():
        raise SystemExit(f"没有 {p.name} —— 手臂那一路没录上,这个测试做不了")
    t, gap = [], []
    with open(p, "rb") as fh:
        for m in msgpack.Unpacker(fh, raw=False):
            a = m.get("arm") or {}
            if len(a) != 2 or "_t" not in m:
                continue
            try:
                r = np.asarray(a["right"]["cmd"], float)
                l = np.asarray(a["left"]["cmd"], float)
            except (KeyError, TypeError):
                continue
            t.append(float(m["_t"]))
            gap.append(float(np.linalg.norm(r - l)))
    return np.asarray(t), np.asarray(gap)


def hand_curl(session: pathlib.Path):
    """手那一路:全部手指关节角之和(拍击时手会张开/合拢,出现拐点)。"""
    import h5py
    files = sorted(session.glob("*_right.h5")) + sorted(session.glob("*_left.h5"))
    if not files:
        raise SystemExit("没有手部 HDF5 —— 手那一路没录上,这个测试做不了")
    f = h5py.File(files[0], "r")
    side = "right" if files[0].name.endswith("_right.h5") else "left"
    t = np.asarray(f["hand_timestamp"])
    q = np.asarray(f[f"{side}_hand_target_joint_positions"])
    f.close()
    return t, q.sum(axis=1)


def _subsample_peak(t, y, idx):
    """峰值的亚采样位置:在极值点两侧各取一点拟合抛物线。

    不做这一步的话,分辨率就是采样周期本身 —— 手那一路 20 Hz 即 50 ms,
    比这个脚本自己的 20 ms 判据还粗,测试永远不可能有把握地通过。
    (合成素材实测:不插值 50 ms / 真值 40 ms;插值后见下方自检。)
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(t[idx])
    a, b, c = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(t[idx])
    delta = 0.5 * (a - c) / den                    # 落在 [-0.5, 0.5]
    dt = float(t[idx + 1] - t[idx - 1]) / 2.0
    return float(t[idx]) + float(np.clip(delta, -0.5, 0.5)) * dt


def sharpest(t, y, exclude=0.5):
    """最尖锐的拐点时刻:|一阶差分| 的最大值处(两端各去掉一点,避免起止效应),
    再做亚采样插值。"""
    if len(t) < 5:
        raise SystemExit("样本太少,量不了")
    d = np.abs(np.gradient(y, t))
    m = (t > t[0] + exclude) & (t < t[-1] - exclude)
    idx = int(np.argmax(np.where(m, d, -np.inf)))
    return _subsample_peak(t, d, idx), float(d[idx])


def self_check() -> int:
    """合成一段真实偏移已知(40 ms)的数据,量出来必须对得上。

    这把尺子的分辨率原本等于采样周期(手 20 Hz = 50 ms),**比它自己的 20 ms
    判据还粗**,那样的测试永远不可能有把握地通过。加了亚采样插值之后才够用,
    这条自检就是锁住这一点的。
    """
    TRUE = 0.040
    t_arm = np.arange(0, 12, 1 / 50.0)
    gap = np.minimum(0.02 + 0.45 * np.abs(np.sin(0.35 * t_arm)),
                     np.abs(t_arm - 6.0) * 1.2 + 0.015)
    t_h = np.arange(0, 12, 1 / 20.0)
    curl = 1.2 / (1 + np.exp(-14 * (t_h - (6.0 + TRUE))))
    i = int(np.argmin(np.where((t_arm > 0.5) & (t_arm < 11.5), gap, np.inf)))
    ta = _subsample_peak(t_arm, -gap, i)
    th, _ = sharpest(t_h, curl)
    got = (th - ta) * 1000.0
    err = abs(got - TRUE * 1000.0)
    ok = err < 5.0
    print(f"[自检] 合成偏移 {TRUE*1000:.0f} ms -> 量得 {got:.1f} ms,误差 {err:.1f} ms "
          f"{'✓' if ok else '✗ 尺子不准'}")
    print(f"[自检] 采样周期 手 {1000/20:.0f} ms / 手臂 {1000/50:.0f} ms —— "
          f"没有亚采样插值的话分辨率就是 50 ms,粗于本脚本 20 ms 的判据")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--self-check", action="store_true",
                    help="用一段**答案已知**的合成数据验这把尺子(真实偏移 40 ms),"
                         "不需要任何录制")
    args = ap.parse_args()
    if args.self_check:
        return self_check()
    if not args.session:
        raise SystemExit("要给一个录制目录,或者用 --self-check")
    d = pathlib.Path(args.session)

    ta, ga = arm_gap(d)
    th, qh = hand_curl(d)
    print(f"手臂 {len(ta)} 帧 {ta[-1]-ta[0]:.2f} 秒;手 {len(th)} 帧 {th[-1]-th[0]:.2f} 秒")

    # 手臂那一路取间距的极小值(拍击 = 两手最近),手那一路取拐点
    m = (ta > ta[0] + 0.5) & (ta < ta[-1] - 0.5)
    i_arm = int(np.argmin(np.where(m, ga, np.inf)))
    t_arm = _subsample_peak(ta, -ga, i_arm)      # 极小值 = -ga 的极大值
    t_hand, sharp = sharpest(th, qh)
    off = (t_hand - t_arm) * 1000.0
    res = max(float(np.median(np.diff(ta))), float(np.median(np.diff(th)))) * 1000
    print(f"手臂间距最小处 t={t_arm:.3f}(间距 {ga.min()*1000:.0f} mm)")
    print(f"两路的采样周期上限 {res:.0f} ms —— 亚采样插值之后才谈得上 20 ms 判据")
    print(f"手关节拐点     t={t_hand:.3f}(变化率 {sharp:.2f} rad/s)")
    print(f"\n残余偏移 = {off:+.1f} ms(手相对手臂)")
    if abs(off) < 20:
        print("判定:小于 20 ms,可以直接合并。")
        return 0
    print("判定:超过 20 ms。**在查清之前不要把合并结果当同步数据。**\n"
          "  下一步:多录几段,看这个偏移是常数还是在漂 —— 常数就记进 attrs "
          "离线补偿,漂就说明某一路的时间戳不可信。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
