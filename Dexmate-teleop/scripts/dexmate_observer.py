#!/usr/bin/env python3
"""只读回显真机姿态:连接机器人,读实测关节角,发布到 zmq。**不发送任何指令。**

    ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_observer.py

存在的理由(2026-08-10,真机现场):控制台页面刚打开时,Viser 里的机器人摆
在默认姿势,而真机在它自己的姿势上 —— 操作者看到的第一眼就是错的。这个进程
把真机实测姿态发出来,镜像在仿真流缺席时用它画,页面一打开就与真机一致。

为什么是独立进程:dexcontrol 只能在它自己的 venv 里跑(zenoh 一族依赖,
不能装进控制台的 .venv)。控制台探测到机器人广播就自动启动本进程,
「连接手臂」(桥接)启动时自动停掉 —— 桥接自己会读状态,不需要第二个
zenoh 客户端。

发布格式(msgpack,CONFLATE 友好,每帧全量):
    {"t_wall_us": int, "q": {关节名: 弧度}}
关节名与 dexcontrol / URDF 相同(桥接的 goal 名字校验依赖这一点,同一来源)。
"""
from __future__ import annotations

import argparse
import time

import msgpack
import zmq


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pub", default="tcp://*:5590")
    ap.add_argument("--hz", type=float, default=15.0)
    args = ap.parse_args()

    try:
        from dexcontrol.robot import Robot
    except ImportError as e:
        raise SystemExit(
            f"[observer] 无法 import dexcontrol ({e})\n"
            "请用装了它的解释器运行:\n"
            "  ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_observer.py")

    print("[observer] 正在连接机器人(只读,不下发任何指令)...", flush=True)
    bot = Robot()
    print("[observer] 已连接,开始发布实测姿态", flush=True)

    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.bind(args.pub)
    n, n_empty = 0, 0
    try:
        while True:
            q: dict[str, float] = {}
            for comp in ("left_arm", "right_arm", "head", "torso"):
                try:
                    q.update({k: float(v) for k, v in
                              bot.get_joint_pos_dict(comp).items()})
                except Exception:                          # noqa: BLE001
                    pass
            if q:
                pub.send(msgpack.packb(
                    {"t_wall_us": int(time.time() * 1e6), "q": q},
                    use_bin_type=True))
                n += 1
                if n == 1:
                    print(f"[observer] 首帧:{len(q)} 个关节", flush=True)
                elif n % 900 == 0:                         # 每分钟一行心跳
                    print(f"[observer] 已发布 {n} 帧", flush=True)
            else:
                n_empty += 1
                if n_empty in (1, 50):
                    print("[observer] 连接着但读不到任何关节 <-- 没有在回显",
                          flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[observer] 退出,共发布 {n} 帧", flush=True)
        bot.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
