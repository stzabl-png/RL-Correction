#!/usr/bin/env python3
"""把一次录制的各路数据按主机墙上时钟合并成一份。

    cd <仓库根目录>
    .venv/bin/python scripts/merge_episode.py data/sessions/<名称>
    .venv/bin/python scripts/merge_episode.py data/sessions/<名称> --out merged.h5

一次录制的目录里是各路各写各的:

    arm.msgpack            手臂指令与实际关节(consumer 的 --pub-state 流)
    robot.msgpack          dexmate 真机各关节(位置/速度/电流)
    pico.msgpack           PICO 原始帧(含 body24)
    glove_{left,right}.msgpack   wuji 手套原始骨架
    <名称>_{left,right}.h5       Sharpa 手关节 + 触觉

**离线合并,不在线合并**:在线合并会把两个进程绑在一起,而它们现在只有广播、
没有等待,任何一条都能单独重启。合并放在事后做,做错了还能重做。

对齐口径:**一律用主机墙上时钟**,各路的字段名不同但含义相同:
    arm      _t          (录制进程收到那一帧的时刻)
    robot    t_wall_us
    pico     t_us        (producer 读到那一帧的时刻)
    glove    t_wall_us   (另有 t_us = 手套设备时钟,不参与对齐)
    hand h5  hand_timestamp / tactile_timestamp

⚠ **手臂那一路的 `t_us` 是仿真时间,不是墙上时间**,实测是墙钟的 2 倍且倍率
逐次都在变。这里用的是 `_t`,不是 `t_us`。

⚠ **戳对齐不等于动作对齐**。各路的采集延迟不同(手套→重定向 与 PICO→producer
→consumer 不是一条路径),共同的墙钟只保证「记录时刻」对得上。真正的偏移要用
拍手测试量出来,在那之前不要把合并结果当成同步数据。本脚本会把各路的
起止时刻和帧率打出来,以及两两之间的时间重叠,方便一眼看出哪一路没录上。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_msgpack_stream(path: pathlib.Path, t_key: str):
    """读一条拼接的 msgpack 流,返回 (时间数组[秒], 帧列表)。"""
    import msgpack
    rows = []
    with open(path, "rb") as fh:
        for m in msgpack.Unpacker(fh, raw=False):
            rows.append(m)
    if not rows:
        return np.zeros(0), []
    if t_key not in rows[0]:
        raise SystemExit(
            f"{path.name} 里没有 {t_key} 字段(有的是 {sorted(rows[0])[:6]} …)。"
            f"这一路是用旧版本录的,没有墙上时钟,不能参与对齐。")
    t = np.array([float(r[t_key]) for r in rows])
    if t_key.endswith("_us"):
        t /= 1e6
    return t, rows


def describe(name, t):
    if len(t) == 0:
        return f"  {name:12s} 空"
    dur = t[-1] - t[0]
    hz = (len(t) - 1) / dur if dur > 0 else 0.0
    return (f"  {name:12s} {len(t):6d} 帧  {dur:7.2f} 秒  {hz:6.1f} Hz  "
            f"起 {t[0]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="data/sessions/ 下的某个目录")
    ap.add_argument("--out", default="", help="输出 HDF5,默认写进该目录的 merged.h5")
    ap.add_argument("--rate", type=float, default=50.0,
                    help="合并后的公共时间轴频率")
    args = ap.parse_args()

    d = pathlib.Path(args.session)
    if not d.is_dir():
        raise SystemExit(f"没有这个目录:{d}")

    streams: dict[str, tuple] = {}
    spec = [("arm", "arm.msgpack", "_t"),
            ("robot", "robot.msgpack", "t_wall_us"),
            ("pico", "pico.msgpack", "t_us"),
            ("glove_left", "glove_left.msgpack", "t_wall_us"),
            ("glove_right", "glove_right.msgpack", "t_wall_us"),
            # 点云:每帧一个 npz 文件,时间戳单独一条流。合并只带索引和戳,
            # 不复制点(一段几百 MB)—— 要点云时按 index 去对应目录取,单位
            # 是**毫米**(见 unit 字段)。
            ("cloud", "cloud_t.msgpack", "t_wall_us")]
    # 多相机(2026-08-11 起):每台相机一条流,按序列号命名
    # cloud_t_<序列号>.msgpack + cloud_<序列号>/NNNNNN.npz;单相机旧命名
    # (上面那条 "cloud")保留兼容。
    for tf in sorted(d.glob("cloud_t_*.msgpack")):
        serial = tf.name[len("cloud_t_"):-len(".msgpack")]
        spec.append((f"cloud_{serial}", tf.name, "t_wall_us"))
    print(f"[合并] {d}")
    for name, fn, tk in spec:
        p = d / fn
        if not p.exists():
            print(f"  {name:12s} 没有这一路")
            continue
        t, rows = read_msgpack_stream(p, tk)
        streams[name] = (t, rows)
        print(describe(name, t))

    # 手部 HDF5(两条独立长度的流:关节与触觉)
    hands = {}
    for h5 in sorted(d.glob("*_left.h5")) + sorted(d.glob("*_right.h5")):
        import h5py
        side = "left" if h5.name.endswith("_left.h5") else "right"
        f = h5py.File(h5, "r")
        hands[side] = f
        print(describe(f"hand_{side}", np.asarray(f["hand_timestamp"])))
        if "tactile_timestamp" in f:
            print(describe(f"tac_{side}", np.asarray(f["tactile_timestamp"])))

    if not streams and not hands:
        raise SystemExit("这个目录里没有任何可合并的数据")

    # 公共时间轴 = 各路都覆盖到的那一段
    starts, ends = [], []
    for t, _ in streams.values():
        if len(t):
            starts.append(t[0]); ends.append(t[-1])
    for f in hands.values():
        t = np.asarray(f["hand_timestamp"])
        if len(t):
            starts.append(t[0]); ends.append(t[-1])
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        print("\n⚠ 各路的时间区间没有交集 —— 说明有一路没有和别的同时录。"
              "起止见上表。不写输出。")
        return 1
    n = max(1, int((t1 - t0) * args.rate))
    grid = t0 + np.arange(n) / args.rate
    print(f"\n[合并] 公共区间 {t1 - t0:.2f} 秒,{n} 行 @ {args.rate:.0f} Hz")

    import h5py
    out = pathlib.Path(args.out) if args.out else d / "merged.h5"
    with h5py.File(out, "w") as g:
        g.create_dataset("t_wall", data=grid)
        g.attrs["session"] = d.name
        g.attrs["note"] = ("最近邻对齐,口径为主机墙上时钟。戳对齐不等于动作"
                           "对齐,残余偏移需用拍手测试量。")
        for name, (t, rows) in streams.items():
            if not len(t):
                continue
            idx = np.searchsorted(t, grid).clip(0, len(t) - 1)
            g.create_dataset(f"{name}/index", data=idx)
            g.create_dataset(f"{name}/t_wall", data=t[idx])
            # 原始帧是异构 dict,整段以 JSON 存一份,索引指过去
            g.create_dataset(f"{name}/rows_json",
                             data=np.array([json.dumps(r, default=str)
                                            for r in rows], dtype=h5py.string_dtype()))
        for side, f in hands.items():
            t = np.asarray(f["hand_timestamp"])
            idx = np.searchsorted(t, grid).clip(0, len(t) - 1)
            g.create_dataset(f"hand_{side}/index", data=idx)
            g.create_dataset(f"hand_{side}/t_wall", data=t[idx])
            for k in f:
                if k.startswith(("tactile_", f"{side}_hand_tactile_")):
                    continue      # 触觉留在原文件里,按 tac_index 取,不复制
                try:
                    g.create_dataset(f"hand_{side}/{k}", data=np.asarray(f[k])[idx])
                except Exception:                          # noqa: BLE001
                    pass
            if "tactile_timestamp" in f:
                tt = np.asarray(f["tactile_timestamp"])
                if len(tt):
                    g.create_dataset(f"hand_{side}/tactile_index",
                                     data=np.searchsorted(tt, grid).clip(0, len(tt) - 1))
    for f in hands.values():
        f.close()
    print(f"[合并] 已写入 {out}({out.stat().st_size / 1e6:.1f} MB)")
    print("[合并] 触觉没有复制进来:它是这里面最大的一块,原文件里按 "
          "hand_<side>/tactile_index 取即可(读取走 "
          "magicdexmate.recording.read_tactile)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
