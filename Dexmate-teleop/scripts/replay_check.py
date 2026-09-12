#!/usr/bin/env python3
"""录制一段遥操作,先在浏览器里预览,确认后再让真机重放同一段动作。

第一次连接真机时,直接实时遥操作意味着一边是没验证过的硬件链路,一边是随时
在变的人的动作,出了问题分不清是哪一边。这个程序把两者拆开:动作先录下来,
不接机器人反复看,确认无误之后再原样重放给机器人。重放的内容与预览的内容是
同一份数据,不存在"预览时是这样、上机变成那样"的可能。

录的是发给机器人的指令关节角本身(仿真程序 --pub-state 里的 cmd 字段),
也就是 dexmate_bridge.py 平时下发的那一份,因此重放与实时运行走的是同一条
下游通路。

三个阶段
    预览      循环播放,离合标记置空。桥接程序要求离合非空才会下发,所以
              这个阶段即便桥接程序开着、开关也开着,机器人依然不会动作。
    同步      操作者在页面上打开开关后进入。持续送出第一帧,让桥接程序按
              它自己的慢速同步把机器人开到起始姿势。
    重放      按录制时的节奏播放一遍,机器人跟随。播完自动回到预览。

开关用的是浏览器页面里已有的「驱动真机」,不额外做界面。页面关闭或开关关闭
即视为撤销,程序立刻退回预览并把离合标记置空。

用法(先 cd 到 <仓库根目录>)
    # 1. 录制。一边正常遥操作,一边录下要发给机器人的指令
    .venv/bin/python scripts/replay_check.py --record logs/take_0001.msgpack

    # 2. 停掉实时遥操作(本程序要占用同一个地址),然后回放
    .venv/bin/python scripts/replay_check.py --play logs/take_0001.msgpack

    # 3. 需要机器人动作时,另开一个终端运行桥接程序
    ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_bridge.py --live

    # 只想看一遍,完全不接机器人
    .venv/bin/python scripts/replay_check.py --preview logs/take_0001.msgpack
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import msgpack
import zmq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.control_link import ControlSubscriber  # noqa: E402


def record(args) -> int:
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 10000)      # 录制要每一帧,不能像显示那样只取最新
    sock.connect(args.sub)
    out = pathlib.Path(args.record)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[录制] 订阅 {args.sub}")
    print(f"[录制] 写入 {out}")
    print("[录制] 正常遥操作即可,结束时按 Ctrl-C")
    n = n_cmd = n_eng = 0
    t0 = None
    try:
        with open(out, "wb") as fh:
            while True:
                try:
                    m = msgpack.unpackb(sock.recv(zmq.NOBLOCK), raw=False)
                except zmq.Again:
                    time.sleep(0.002)
                    continue
                now = time.time()
                t0 = t0 or now
                m["_t"] = now                # 主机时钟,重放按它还原节奏
                fh.write(msgpack.packb(m, use_bin_type=True))
                n += 1
                n_cmd += bool(m.get("cmd"))
                n_eng += bool(m.get("engaged"))
                if n % 200 == 0:
                    print(f"\r[录制] {n} 帧,{now - t0:.1f} 秒", end="", flush=True)
    except KeyboardInterrupt:
        pass

    dur = (time.time() - t0) if t0 else 0.0
    print(f"\n[录制] 结束:{n} 帧,{dur:.1f} 秒 -> {out}")
    # 一段没有指令或从未闭合离合的录制无法用于驱动机器人,当场说明,
    # 而不是等到回放时才发现机器人不动。
    if n == 0:
        print("[录制] 未收到任何数据。仿真程序是否在运行,--pub-state 地址是否一致。")
        return 1
    print(f"[录制] 其中含指令的 {n_cmd} 帧,离合闭合的 {n_eng} 帧")
    if n_cmd == 0:
        print("[录制] 没有一帧带指令,这段无法用于驱动机器人。仿真程序版本可能过旧。")
        return 1
    if n_eng == 0:
        print("[录制] 没有一帧的离合是闭合的,这段无法用于驱动机器人。"
              "录制时手臂需要处于跟随状态。")
        return 1
    return 0


def load(path: pathlib.Path) -> list[dict]:
    with open(path, "rb") as fh:
        return list(msgpack.Unpacker(fh, raw=False))


def play(args) -> int:
    path = pathlib.Path(args.play or args.preview)
    frames = load(path)
    if not frames:
        print(f"[回放] {path} 是空的")
        return 1
    dur = frames[-1].get("_t", 0) - frames[0].get("_t", 0)
    n_eng = sum(1 for f in frames if f.get("engaged"))
    print(f"[回放] {path}:{len(frames)} 帧,{dur:.1f} 秒,其中离合闭合 {n_eng} 帧")
    if n_eng == 0:
        print("[回放] 这段录制没有一帧离合是闭合的,机器人不会动作。")
        if args.play:
            return 1

    pub = zmq.Context.instance().socket(zmq.PUB)
    try:
        pub.bind(args.pub)
    except zmq.ZMQError as e:
        print(f"[回放] 无法绑定 {args.pub}:{e}")
        print("[回放] 实时遥操作可能仍在运行。回放需要占用同一个地址,请先停止它。")
        return 1
    print(f"[回放] 发布于 {args.pub},浏览器页面可直接观看")

    preview_only = bool(args.preview)
    ctl = None if preview_only else ControlSubscriber(args.control)
    if preview_only:
        print("[回放] 仅预览,不会驱动机器人")
    else:
        print("[回放] 当前为预览,离合标记置空,机器人不会动作")
        print("[回放] 确认动作无误后,在页面的「驱动真机」处打开开关")

    def send(frame: dict, engaged: bool):
        m = dict(frame)
        m.pop("_t", None)
        m["engaged"] = frame.get("engaged", []) if engaged else []
        m["t_us"] = int(time.time() * 1e6)
        try:
            pub.send(msgpack.packb(m, use_bin_type=True), zmq.NOBLOCK)
        except zmq.Again:
            pass

    state, i, t_state = "preview", 0, time.time()
    n_pass = 0
    armed_prev = True   # 视开关初始为开,避免程序刚起来就把已经打开的开关当成一次确认
    try:
        while True:
            now = time.time()
            armed = bool(ctl and (ctl.latest(now) or {}).get("enabled"))
            # 按上升沿触发,而不是看开关当前的状态。开关一直开着时,重放一遍
            # 结束后不再自动开始下一遍,否则机器人会在无人确认的情况下反复
            # 重放。要再放一遍,把开关关掉再打开。
            rising = armed and not armed_prev
            armed_prev = armed

            if state == "preview":
                send(frames[i], engaged=False)
                i = (i + 1) % len(frames)
                if rising:
                    state, t_state, i = "sync", now, 0
                    print(f"\n[回放] 开关已打开。送出起始姿势,等待机器人同步,"
                          f"预留 {args.sync_hold:.0f} 秒")
                    print("[回放] 机器人此时正在缓慢移动到这段动作的起始姿势")

            elif state == "sync":
                send(frames[0], engaged=True)
                if not armed:
                    state, t_state = "preview", now
                    print("\n[回放] 开关已关闭,退回预览")
                elif now - t_state >= args.sync_hold:
                    state, t_state, i = "run", now, 0
                    print("[回放] 开始重放")

            elif state == "run":
                if not armed:
                    send(frames[i], engaged=False)
                    state, t_state = "preview", now
                    print("\n[回放] 开关已关闭,已停止重放并退回预览")
                    continue
                send(frames[i], engaged=True)
                i += 1
                if i >= len(frames):
                    n_pass += 1
                    print(f"[回放] 第 {n_pass} 遍重放结束,已退回预览。"
                          "需要再放一遍时,把开关关掉再打开")
                    state, t_state, i = "preview", now, 0

            # 还原录制时的节奏。预览与重放用同一套时间轴,两者速度一致。
            if state in ("preview", "run") and i > 0:
                dt = frames[i].get("_t", 0) - frames[i - 1].get("_t", 0)
                time.sleep(min(max(dt, 0.0), 0.2))
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[回放] 已停止")
    finally:
        for _ in range(5):                  # 收尾时明确置空离合
            send(frames[0], engaged=False)
            time.sleep(0.02)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", metavar="文件", help="录制一段遥操作")
    g.add_argument("--play", metavar="文件", help="预览,确认后重放给机器人")
    g.add_argument("--preview", metavar="文件", help="只预览,不接机器人")
    ap.add_argument("--sub", default="tcp://127.0.0.1:5583",
                    help="录制时订阅的地址,即仿真程序的 --pub-state")
    ap.add_argument("--pub", default="tcp://*:5583",
                    help="回放时发布的地址。与仿真程序相同,因此页面与桥接程序不必改动")
    ap.add_argument("--control", default="tcp://127.0.0.1:5584",
                    help="页面开关所在的频道")
    ap.add_argument("--sync-hold", type=float, default=20.0,
                    help="打开开关后送出起始姿势的时长,单位为秒。"
                         "桥接程序在这段时间内把机器人缓慢开到起始姿势")
    args = ap.parse_args()
    return record(args) if args.record else play(args)


if __name__ == "__main__":
    sys.exit(main())
