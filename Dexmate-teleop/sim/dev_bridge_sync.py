#!/usr/bin/env python3
"""桥接程序的同步阶段能不能收敛 —— 用一个会动的假 consumer 逼它。

    cd ~/dexmate/MagicDexMate && .venv/bin/python sim/dev_bridge_sync.py

要证的事情
    同步阶段以 --sync-speed(默认 8 度每秒)从机器人当前位置走向操作者当前
    姿势。它的退出判据是「还差多远」小于 --sync-tol。而 goal 里除了双臂还有
    头的三个关节,戴着头显的人头一直在动,速度远高于 8 度每秒 —— 于是这个
    距离永远收不到阈值以下,状态机永远离不开同步档,双臂就以 8 度每秒一直爬。
    这正是「真机运动速度过慢,远不如遥操作人员」的机制。

    修法是同步阶段追一张不动的快照,走到之后再看操作者移动了多少:移动得少
    就转入跟随,移动得多就再走一趟。快照不动,所以每一趟必定收敛。

三条断言,每条都能对着旧代码失败
    1. 头持续转动时,桥接必须进入跟随。旧代码在这一条上会一直停在同步档。
    2. 进入跟随所花的时间必须与「初始差值 / 同步速度」相符 —— 证明它是慢速
       走完的,不是把安全档跳过去了。
    3. 进入跟随后,下发的头关节角必须跟着指令动 —— 证明跟随是真的在跟。

这个脚本自己也要能失败:把 --sync-speed 调到比头还快(例如 200),第 1 条会
以另一种方式通过(直接追上),第 2 条会因为用时远小于预期而失败。
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys
import time

import msgpack
import numpy as np
import zmq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from magicdexmate.control_link import ControlPublisher  # noqa: E402
from magicdexmate.home_pose import ARM_HOME  # noqa: E402

HOME_R_ARM_J1 = ARM_HOME["R_arm_j1"]   # 判据取自同一份常量,不另抄一个数

ARMS = [f"{s}_arm_j{i}" for s in ("L", "R") for i in range(1, 8)]
HEAD = [f"head_j{i}" for i in (1, 2, 3)]

OFFSET_DEG = 20.0        # 假设机器人初始比指令差这么多
SYNC_SPEED = 8.0         # 与桥接默认一致
HEAD_AMP_DEG = 25.0      # 头的摆幅
HEAD_HZ = 0.4            # 头的频率 -> 峰值角速度 2*pi*0.4*25 = 63 度每秒,远快于 8
RUN_S = 48.0
SESSION = "_dev_bridge_sync"     # 录制目录名,由广播传给桥接
# 第 4 条(2026-08-10 真机复盘后加):跟随中目标突然跳一大步(= 输入流打嗝
# 恢复后操作者已经移走)时,追赶必须以跟随限速匀速进行,不许瞬移。真机现场
# 的「卡住后猛甩」正是旧默认 150 度每秒 ≈ 硬件极限在追赶。步长要够大
# (120°),让 1 秒一行的周期读数能分辨"匀速走了两秒"与"一秒内到位"。
STEP_DEG = 120.0
STEP_T = 8.0    # 放在首次同步完成后的干净跟随段里。不能放在回合归位之后:归位要把
                 # 肘关节(默认位 ±126°)按 8 度每秒走 15.8 秒单程、往返 32 秒,台架
                 # 时长内跟随档回不来 —— 前两版分别把阶跃插进回程(被当成快照后的移动)
                 # 和排在归位后(永远等不到跟随),断言都只能读到 0
FOLLOW = float(os.environ.get("DEV_FOLLOW_SPEED", "60"))


def main() -> int:
    peak = 2 * np.pi * HEAD_HZ * HEAD_AMP_DEG
    print(f"[台架] 头的峰值角速度 {peak:.0f} 度每秒,同步速度 {SYNC_SPEED:.0f} 度每秒 "
          f"({peak / SYNC_SPEED:.1f} 倍)。追活动目标必然收不敛。")

    # ⛔ 安全:2026-08-10 真机现场事故。本台架此前把假指令绑在生产端口
    # :5583/:5584 上,并广播 enabled=True —— 当时用户有一个已连接真机的桥接
    # 正订阅这两个端口,把台架的合成信号(头 ±25° 摆动、手臂走向 0)当成真
    # 指令执行,机器人被驱动,用户按了急停。两个教训:①订阅者不占端口绑定,
    # 「端口被占会自动失败」的假设对订阅者不成立;②任何会发布指令的测试
    # 必须用与生产完全不同的端口。因此这里改用 16583/16584,被测桥接用
    # 参数指到同一对端口;另外开跑前检查真机进程,在就拒绝运行(双保险)。
    _live = subprocess.run(["pgrep", "-af", "dexmate_bridge"],
                           capture_output=True, text=True).stdout
    _live = [ln for ln in _live.splitlines() if "--live" in ln]
    if _live:
        print("[台架] ⛔ 检测到已连接真机的桥接进程,拒绝运行:")
        for ln in _live:
            print("   ", ln)
        return 1
    SUB = "tcp://127.0.0.1:16583"
    CTL = "tcp://127.0.0.1:16584"
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.bind("tcp://*:16583")
    ctl = ControlPublisher("tcp://*:16584")

    target = sys.argv[1] if len(sys.argv) > 1 else "scripts/dexmate_bridge.py"
    print(f"[台架] 被测程序 {target}(专用端口 {SUB} / {CTL},与生产隔离)")
    proc = subprocess.Popen(
        [sys.executable, "-u", target,
         "--dry-run", "--sub", SUB, "--control", CTL,
         "--sim-start-offset", str(OFFSET_DEG),
         "--follow-speed", str(FOLLOW),
         "--sync-speed", str(SYNC_SPEED), "--duration", str(RUN_S + 2)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0)                       # 让订阅建立,否则前几帧会丢

    t0 = time.time()
    while time.time() - t0 < RUN_S:
        t = time.time() - t0
        # 头在 t=12 后定住(取 t=12 时刻的正弦值,连续无跳变)。持续摆动的
        # 头会让「快照后此间移动 >15° 再走一趟」的检查在回合归位后反复不过
        # (第一次同步能过是相位巧合:2.5 秒恰好一个摆动周期,净位移≈0),
        # 桥接困在同步趟里,第 4 条的跟随态追赶就永远测不到。真人头部不会
        # 无休止正弦摆动,定住不损失第 1 至 3 条的判别力(它们用 t<7 的数据)。
        head = np.radians(HEAD_AMP_DEG * np.sin(2 * np.pi * HEAD_HZ * min(t, 12.0)))
        cmd = {n: 0.0 for n in ARMS}
        if t > STEP_T:
            # 目标突跳:模拟输入流打嗝恢复后操作者已经移走 STEP_DEG
            cmd["L_arm_j1"] = float(np.radians(STEP_DEG))
        cmd["head_j1"] = float(head)
        cmd["head_j2"] = float(head * 0.5)
        cmd["head_j3"] = 0.0
        pub.send(msgpack.packb({"t_us": int(t * 1e6), "cmd": cmd,
                                "engaged": ["right", "left"], "mode": "arms+head"},
                               use_bin_type=True), zmq.NOBLOCK)
        # 中间那一段打开录制,验证桥接是跟着广播开关的,而不是一直在写
        ctl.send({"enabled": True,
                  "recording": 4.0 < t < 10.0,
                  "session": SESSION,
                  # 阶跃验完后开一段新的:桥接应当开始回默认姿势并走到(0b 只验
                  # 归位本身;肘 126° 的回程 15.8 秒,台架不等它)
                  "epoch": 1 if t > 14.0 else 0})   # 放在阶跃追赶(t=8 起约 2 秒)之后
        time.sleep(0.02)

    proc.wait(timeout=20)
    out = proc.stdout.read() if proc.stdout else ""
    print("\n".join("    " + ln for ln in out.strip().splitlines()[-14:]))

    ok = True

    # 0. 录制:页面广播 recording + session 时,真机关节要落到那个目录里
    sess_dir = ROOT / "data" / "sessions" / SESSION
    if f"开始录真机关节" in out and "停止录真机关节" in out:
        print(f"[通过] 第 0 条:录制随广播开关,目录 {sess_dir.name}"
              f"(试运行读不到真机,所以帧数为 0 是对的)")
    else:
        print("[失败] 第 0 条:广播里带了 recording 与 session,桥接却没有开始录制。")
        ok = False

    # 0b. 新的一段:必须先回默认姿势,而且真的走到了。
    # 判据用 R_arm_j1 在整个过程中的**峰值**,不是某一个时刻的读数 —— 周期
    # 读数每秒才打一次,到达默认位之后它立刻就开始往操作者姿势走,按时刻取
    # 必然读到一个已经偏离的值(第一版就是这么误判的)。
    # 假 consumer 全程把手臂指令为 0,初始偏移 +20°,所以没有归位这一步,
    # R_arm_j1 的峰值只能是 20°;归位了才会到默认值 28.6°。两者可区分。
    want = np.degrees(HOME_R_ARM_J1)
    peak = max((float(v) for v in re.findall(r"R_arm_j1=\s*([-+\d.]+)", out)),
               default=0.0)
    if "回默认姿势" not in out:
        print("[失败] 第 0b 条:广播了新的段号,桥接没有开始回默认姿势。")
        ok = False
    elif "已回到默认姿势" not in out:
        print(f"[失败] 第 0b 条:开始回默认姿势了,但没走到(峰值 {peak:.1f}°,"
              f"目标 {want:.1f}°)。")
        ok = False
    elif abs(peak - want) > 1.5:
        print(f"[失败] 第 0b 条:走完了,但 R_arm_j1 峰值 {peak:.1f}° 不等于默认值 "
              f"{want:.1f}° —— 它去的不是默认位。")
        ok = False
    else:
        print(f"[通过] 第 0b 条:新的一段先回了默认姿势(R_arm_j1 峰值 {peak:.1f}°,"
              f"默认值 {want:.1f}°;不归位的话峰值只会是初始偏移 {OFFSET_DEG:.0f}°)")

    # 1. 必须进入跟随
    if "同步完成,开始跟随" not in out:
        print("\n[失败] 第 1 条:头一直在动,桥接始终没有离开同步档。"
              "双臂会以 8 度每秒一直爬,这就是真机跟不上人的机制。")
        ok = False
    else:
        print("\n[通过] 第 1 条:进入了跟随。")

    # 2. 用时必须与「初始差值 / 同步速度」相符
    m = re.search(r"following\s+t=\s*([\d.]+)s", out)
    first = re.search(r"syncing\s+t=\s*([\d.]+)s", out)
    want = OFFSET_DEG / SYNC_SPEED
    if m and first:
        took = float(m.group(1)) - float(first.group(1))
        lo, hi = want * 0.5, want * 2.0 + 1.5
        verdict = "通过" if lo <= took <= hi else "失败"
        ok &= verdict == "通过"
        print(f"[{verdict}] 第 2 条:同步用时约 {took:.1f} 秒,"
              f"预期 {want:.1f} 秒(允许 {lo:.1f} 至 {hi:.1f})。")
    else:
        print("[失败] 第 2 条:读不到同步与跟随两个阶段的时间戳。")
        ok = False

    # 3. 跟随阶段头必须在动
    vals = [float(v) for v in re.findall(r"following.*?head_j1=\s*([-+\d.]+)", out)]
    if len(vals) >= 3 and (max(vals) - min(vals)) > 5.0:
        print(f"[通过] 第 3 条:跟随阶段 head_j1 走了 {max(vals) - min(vals):.1f}°,"
              f"确实在跟。")
    else:
        print(f"[失败] 第 3 条:跟随阶段 head_j1 只走了 "
              f"{(max(vals) - min(vals)) if vals else 0:.1f}°,没有在跟。")
        ok = False

    # 4. 目标突跳后的追赶必须受限速约束:贴着限速匀速走,不许瞬移,也不许不走
    pts = [(float(t_), float(v)) for t_, v in
           re.findall(r"following\s+t=\s*([\d.]+)s.*?L_arm_j1=\s*([-+\d.]+)", out)]
    pts.sort()
    speeds = [(abs(v2 - v1) / max(t2 - t1, 1e-6))
              for (t1, v1), (t2, v2) in zip(pts, pts[1:]) if t2 > t1]
    vmax = max(speeds, default=0.0)
    final = pts[-1][1] if pts else 0.0
    cap = FOLLOW * 1.3 + 3.0
    if vmax <= cap and final > STEP_DEG - 3.0 and vmax >= FOLLOW * 0.5:
        print(f"[通过] 第 4 条:{STEP_DEG:.0f}° 突跳后按限速追赶 —— 实测最大 "
              f"{vmax:.0f} 度每秒(限速 {FOLLOW:.0f},允许至 {cap:.0f}),已走到 "
              f"{final:.1f}°。")
    else:
        why = (f"实测最大 {vmax:.0f} 度每秒超过限速 {FOLLOW:.0f} 的允许上限 "
               f"{cap:.0f} —— 这就是真机上「猛甩」的机制" if vmax > cap else
               f"没走到位(终值 {final:.1f}°/{STEP_DEG:.0f}°)或没在动"
               f"(最大 {vmax:.0f} 度每秒)")
        print(f"[失败] 第 4 条:{why}。")
        ok = False

    print("\n结论:" + ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
