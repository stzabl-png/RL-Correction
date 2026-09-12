#!/usr/bin/env python3
"""真机姿态只读回显的免硬件测试(dexmate_observer -> mirror 那条线)。

    cd <仓库根目录> && .venv/bin/python sim/dev_robot_echo.py

三条断言(每条都能失败):
  1. 仿真流缺席时,假 observer 发布的关节角要被镜像画出来,状态行要写明
     「显示真机实测姿态」;
  2. 仿真流一来,回显必须**立即让位**(画的是仿真的 q,状态行回到 ✅)——
     真机回显只是开场画面,不许和指令链路抢;
  3. 仿真流再断 1.5 秒后,回显要接回来。

端口全部用非默认值,不与正在跑的控制台/仿真冲突。
"""
from __future__ import annotations

import pathlib
import sys
import time

import msgpack
import numpy as np
import viser
import zmq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from viser_isaac_mirror import Mirror  # noqa: E402

ROBOT_PUB = "tcp://127.0.0.1:6590"
SIM_PUB = "tcp://127.0.0.1:6583"


def drawn(mirror, joint: str) -> float:
    cfg = mirror.urdf.cfg
    return float(cfg[mirror.names.index(joint)])


def pump(mirror, seconds: float):
    t0 = time.time()
    while time.time() - t0 < seconds:
        mirror.tick()
        time.sleep(0.02)


def main() -> int:
    ctx = zmq.Context.instance()
    rob = ctx.socket(zmq.PUB)
    rob.bind(ROBOT_PUB)
    sim = ctx.socket(zmq.PUB)
    sim.bind(SIM_PUB)

    server = viser.ViserServer(port=8198, verbose=False)
    mirror = Mirror(server, sub=SIM_PUB, control="tcp://127.0.0.1:6584",
                    hand_right="", hand_left="", pico="",
                    robot_state=ROBOT_PUB)
    bad = 0

    # --- 1. 仿真缺席,回显接管 -------------------------------------------
    t0 = time.time()
    while time.time() - t0 < 3.0:
        rob.send(msgpack.packb(
            {"t_wall_us": int(time.time() * 1e6),
             "q": {"L_arm_j4": -1.0, "R_arm_j1": 0.7}}, use_bin_type=True))
        mirror.tick()
        time.sleep(0.02)
    v = drawn(mirror, "L_arm_j4")
    ok = abs(v - (-1.0)) < 1e-6 and "真机" in mirror.g_state.value
    print(f"[1] 仿真缺席:画出的 L_arm_j4={v:+.3f}(期望 -1.000)  "
          f"状态行「{mirror.g_state.value}」  {'✅' if ok else '❌ 回显没生效'}")
    bad += 0 if ok else 1

    # --- 2. 仿真流一来,回显让位 -----------------------------------------
    t0 = time.time()
    while time.time() - t0 < 1.0:
        rob.send(msgpack.packb(
            {"t_wall_us": int(time.time() * 1e6),
             "q": {"L_arm_j4": -1.0}}, use_bin_type=True))
        sim.send(msgpack.packb(
            {"t_us": 0, "q": {"L_arm_j4": 0.5}, "cmd": {}}, use_bin_type=True))
        mirror.tick()
        time.sleep(0.02)
    v = drawn(mirror, "L_arm_j4")
    ok = abs(v - 0.5) < 1e-6 and "✅" in mirror.g_state.value
    print(f"[2] 仿真恢复:画出的 L_arm_j4={v:+.3f}(期望 +0.500,仿真的值)  "
          f"状态行「{mirror.g_state.value}」  {'✅ 让位了' if ok else '❌ 回显抢了链路'}")
    bad += 0 if ok else 1

    # --- 3. 仿真再断,回显接回 -------------------------------------------
    t0 = time.time()
    while time.time() - t0 < 2.2:      # > 1.5s 断流判定
        rob.send(msgpack.packb(
            {"t_wall_us": int(time.time() * 1e6),
             "q": {"L_arm_j4": -1.0}}, use_bin_type=True))
        mirror.tick()
        time.sleep(0.02)
    v = drawn(mirror, "L_arm_j4")
    ok = abs(v - (-1.0)) < 1e-6 and "真机" in mirror.g_state.value
    print(f"[3] 仿真再断:画出的 L_arm_j4={v:+.3f}(期望回到 -1.000)  "
          f"{'✅ 接回来了' if ok else '❌ 没接回'}")
    bad += 0 if ok else 1

    print(f"\n{'✅ 全部通过' if bad == 0 else f'❌ {bad} 项失败'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
