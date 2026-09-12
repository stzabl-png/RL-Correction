#!/usr/bin/env python3
"""Drive the REAL Dexmate Vega from the teleop consumer's commanded joints.

    PICO -> producer -> consumer (mapping + Pink IK) -> joint targets
                                                          |
                                          Isaac (simulated)  +  THIS (hardware)

Isaac was never in the mapping path - the law is numpy and the IK is pinocchio,
both on the CPU. It is the simulated executor. This is the real one, and it
takes the same commands off the same zmq stream the mirror viewer uses, so the
robot and the viewer cannot disagree about what was asked for.

SEPARATE PROCESS ON PURPOSE. dexcontrol must not be installed into the Isaac
venv: it pulls zenoh, dexcomm and dexbot_utils, and that environment is
load-bearing and fragile (numpy pinned to 1.26, onnxruntime pinned away from
CUDA 13). Run this one where dexcontrol already lives:

    ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_bridge.py
    ~/anaconda3/envs/dexmate/bin/python  scripts/dexmate_bridge.py

THREE STATES

  off        Nothing is sent. The state on startup, and whenever the mirror
             viewer's switch is off, the browser is closed, the stream goes
             quiet, or the E-stop is engaged.
  syncing    The robot moves SLOWLY from wherever it is to where the operator
             currently is, at a fixed --sync-speed, however far that is.
             THE TORSO IS PART OF THIS ONLY IN arms+head+waist. In the other
             two modes the bridge never reads, syncs or commands it - the
             operator sets it by hand, and it MUST already be at the pose the
             mapping assumes, because the arm targets are built from that chest
             pose and a torso anywhere else makes every one of them wrong,
             silently. See goal_of().
  following  Normal teleop.

  It drops back to `syncing` whenever the arm clutch opens and closes again:
  the IK's redundant degree of freedom can settle on a different branch after a
  re-engage, and on hardware that means the elbow sweeping a long way through
  space even though the hand does not. A per-tick slew limit bounds speed, not
  travel; only re-syncing bounds travel.

WHAT IS NOT PROTECTED - know this before switching it on
  * Pink has no collision model and the sim runs with self-collisions disabled,
    so the mapping can command the two arms into each other. --min-hand-gap is
    a blunt guard on the commanded wrists, not a collision check.
  * The Sharpa hands are not in the mapping and are never commanded.
  * The chassis is not commanded. Park it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import msgpack
import numpy as np
import zmq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from magicdexmate.control_link import ControlSubscriber  # noqa: E402
from magicdexmate.home_pose import home_for  # noqa: E402
# (no TORSO_HOME import: the bridge never synthesises a torso pose - it either
# relays the consumer's commanded one or leaves the torso alone entirely.)

ARM = {"left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
       "right_arm": [f"R_arm_j{i}" for i in range(1, 8)]}
HEAD = [f"head_j{i}" for i in (1, 2, 3)]
TORSO = [f"torso_j{i}" for i in (1, 2, 3)]
# 手臂各关节的硬件速度上限,单位 rad/s。与 URDF 及 V2AP 的 DEXMATE_DEFAULT_ARM
# _VEL_LIMITS 逐位一致。速度前馈发出去的值一律钳在这以内。
ARM_VMAX = np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7])


class Sink:
    """The robot, or a printing stand-in for it."""

    def __init__(self, dry_run: bool, components: set[str]):
        self.dry = dry_run
        self.components = components
        self.bot = None
        if dry_run:
            print("[bridge] 试运行模式,不连接机器人,仅打印将要下发的指令")
            return
        try:
            from dexcontrol.robot import Robot
        except ImportError as e:
            raise SystemExit(
                f"[bridge] 无法 import dexcontrol ({e})\n"
                "请用装了它的解释器运行,例如:\n"
                "  ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_bridge.py\n"
                "  ~/anaconda3/envs/dexmate/bin/python scripts/dexmate_bridge.py")
        print("[bridge] 正在连接机器人 ...")
        self.bot = Robot()
        print("[bridge] 已连接")

    def read(self) -> dict[str, float]:
        if self.bot is None:
            return {}
        out = {}
        for comp in ("left_arm", "right_arm", "head", "torso"):
            try:
                out.update(self.bot.get_joint_pos_dict(comp))
            except Exception as e:                        # noqa: BLE001
                print(f"[bridge] 无法读取 {comp} 的关节位置:{e}")
        return out

    def read_state(self) -> dict:
        """真机的实测关节量。位置一定有,速度与电流有就带上。

        录真机本体数据只能在这里做:它要一个 dexcontrol 连接,而这个进程已经
        有一个了。另起一个进程就是第二个 zenoh 客户端去订同一批话题,没有必要。
        """
        if self.bot is None:
            return {}
        out: dict = {}
        for comp in ("left_arm", "right_arm", "head", "torso"):
            if comp not in self.components:
                continue
            try:
                c = getattr(self.bot, comp)
                names = c.get_joint_name()
                st = c.get_joint_state()          # (N,3) 位置/速度/电流或力矩
                for i, n in enumerate(names):
                    out[n] = [float(x) for x in st[i]]
            except Exception:                             # noqa: BLE001
                try:
                    out.update({n: [float(v)] for n, v in
                                self.bot.get_joint_pos_dict(comp).items()})
                except Exception:                         # noqa: BLE001
                    pass
        return out

    def limits(self) -> dict[str, tuple[float, float]]:
        """The ROBOT's own limits. The sim's come from the USD and are not
        authoritative for hardware."""
        if self.bot is None:
            return {}
        out = {}
        for comp in ("left_arm", "right_arm", "head", "torso"):
            try:
                c = getattr(self.bot, comp)
                names = c.get_joint_name()
                # dexcontrol >=0.5.0: property is `joint_pos_limit` (singular),
                # shape (N,2) rows=[lo,hi]. The old plural name never existed on
                # the SDK, so this branch silently returned {} and the bridge's
                # own clamp below was dead code.
                lim = getattr(c, "joint_pos_limit", None)
                if lim is None:
                    lim = getattr(c, "joint_pos_limits", None)   # legacy naming
                if lim is None:
                    continue
                lim = np.asarray(lim, dtype=float)
                if lim.ndim == 2 and lim.shape[1] == 2 and lim.shape[0] == len(names):
                    lo, hi = lim[:, 0], lim[:, 1]                # (N,2)
                elif lim.ndim == 2 and lim.shape[0] == 2:
                    lo, hi = lim[0], lim[1]                      # legacy (lo[], hi[])
                else:
                    print(f"[bridge] {comp} 限位形状异常 {lim.shape},忽略")
                    continue
                for i, n in enumerate(names):
                    out[n] = (float(lo[i]), float(hi[i]))
            except Exception:                             # noqa: BLE001
                pass
        return out

    def estopped(self) -> bool:
        if self.bot is None:
            return False
        try:
            return bool(self.bot.estop.is_software_estop_enabled())
        except Exception:                                 # noqa: BLE001
            return False

    def plan(self, q: dict[str, float], include_torso: bool):
        """这一拍要下发给哪些部件。试运行与真机调用同一个函数,所以试运行
        打印出来的就是真机会发的那一份 —— 先前 write() 在试运行下直接返回,
        这段过滤逻辑一次都没被执行过,试运行因此证明不了"只发了手臂"。"""
        out = []
        for comp, names in ARM.items():
            if comp in self.components and all(n in q for n in names):
                out.append((comp, names))
        if "head" in self.components and all(n in q for n in HEAD):
            out.append(("head", HEAD))
        # 躯干要同时满足两个独立条件:被列进 --components,且控制模式是
        # arms+head+waist。另外两档由操作者手动摆放,不读不发。
        if include_torso and all(n in q for n in TORSO):
            out.append(("torso", TORSO))
        return out

    def write(self, q: dict[str, float], include_torso: bool,
              qd: dict[str, float] | None = None):
        """下发这一拍的指令。qd 非空时,手臂走带速度前馈的接口。

        位置环 PID 追一个运动中的目标,稳态滞后正比于「速度 / Kp」—— 目标停下
        来时误差为零,动得越快落后越多,主观上就是「慢的时候还行、快了跟不上」。
        前馈把这一项直接抵掉,而且不用抬刚度(抬刚度会同时放大碰撞时的力)。
        头和躯干没有这个接口,仍走纯位置。
        """
        sending = self.plan(q, include_torso)
        if self.bot is None:
            return sending                      # 试运行:算了但不发
        for comp, names in sending:
            c = getattr(self.bot, comp)
            if qd is not None and comp in ARM and all(n in qd for n in names):
                v = np.clip([qd[n] for n in names], -ARM_VMAX, ARM_VMAX)
                c.set_joint_pos_vel([q[n] for n in names], v.tolist())
            else:
                c.set_joint_pos([q[n] for n in names])
        return sending

    def pid(self, scale: float):
        """读出手臂位置环的 P 倍率;scale>0 时先设置再读回。

        出厂默认值此前从没被任何人看过一眼 —— 而「真机跟不上」最直接的嫌疑就是
        它偏低。设置后再读回,是为了让「设进去了」有读数,而不是靠调用没抛异常。
        """
        if self.bot is None:
            return
        for comp in ("left_arm", "right_arm"):
            if comp not in self.components:
                continue
            try:
                c = getattr(self.bot, comp)
                if scale > 0:
                    r = c.set_pid([float(scale)] * 7)
                    if not r.get("success"):
                        print(f"[bridge] {comp} 设置 P 倍率失败:{r.get('message')}")
                got = c.get_pid()
                p = got.get("p")
                print(f"[bridge] {comp} 位置环 P 倍率 {p}"
                      + ("(出厂默认,未改动)" if scale <= 0 else f"(已设为 {scale}）"))
            except Exception as e:                        # noqa: BLE001
                print(f"[bridge] {comp} 读取 P 倍率失败:{e}")

    def close(self):
        if self.bot is not None:
            self.bot.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", default="tcp://127.0.0.1:5583",
                    help="仿真程序 --pub-state 的地址")
    ap.add_argument("--control", default="tcp://127.0.0.1:5584",
                    help="浏览器页面的开关频道。超过 1.5 秒未收到消息即停用,"
                         "使关闭页面等同于让机器人停止,而非按最后一条指令继续运行")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="默认值。不连接机器人,仅打印将要下发的指令")
    ap.add_argument("--live", dest="dry_run", action="store_false",
                    help="实际驱动机器人")
    ap.add_argument("--components", default="left_arm,right_arm,head",
                    help="需要跟随的部件。躯干仅在 --control-mode arms+head+waist "
                         "下驱动;另外两档中不读取、不初始化、也不下发,"
                         "由操作者手动摆放")
    ap.add_argument("--hz", type=float, default=100.0,
                    help="指令下发频率。dexcontrol 需要高频重复调用,上限 500Hz")
    ap.add_argument("--sync-speed", type=float, default=8.0,
                    help="同步阶段每个关节的最大角速度,单位为度每秒。无论距离多远"
                         "都按此速度移动,因此初始姿态相差多少都是安全的")
    ap.add_argument("--sync-tol", type=float, default=1.5, help="同步完成的判定阈值,单位为度")
    ap.add_argument("--sync-regap", type=float, default=15.0,
                    help="单位为度。同步走到快照姿势后,若操作者已经移动超过此值,"
                         "再慢速走一趟;否则转入跟随")
    ap.add_argument("--follow-speed", type=float, default=60.0,
                    help="跟随阶段每个关节的最大角速度,单位为度每秒。硬件自身上限为"
                         "137 度每秒(j1/j2)与 155 度每秒(j3 至 j7)。默认 60 ≈ 硬件的"
                         " 0.4 倍,与参考实现(V2AP/T-Rex 的 DEXMATE_VEL_LIMIT_SCALE"
                         "=0.4)一致 —— 2026-08-10 实测:旧默认 150 等于按硬件极限"
                         "速度追赶,输入流每次中断恢复后机器人猛甩 30° 左右补上落差"
                         "(72 分钟里 568 次落后告警、30 次停用),操作者报「危险」。"
                         "落后的读数仍会打出来;确实想更快时显式给这个参数")
    ap.add_argument("--min-hand-gap", type=float, default=0.08,
                    help="单位为米。两手的指令位置近于此值时不再下发。求解器没有"
                         "碰撞模型,仿真的自碰撞检测也处于关闭状态,上游没有任何环节拦截")
    ap.add_argument("--no-feedforward", dest="feedforward", action="store_false",
                    help="手臂改用纯位置指令。默认走 set_joint_pos_vel,把这一拍的"
                         "指令速度一并发下去 —— 位置环追运动中的目标必然滞后,滞后量"
                         "正比于速度除以 Kp,前馈把它抵掉。若真机出现抖动就关掉它")
    ap.add_argument("--arm-pid-scale", type=float, default=0.0,
                    help="手臂位置环 P 增益相对出厂值的倍率,允许 0.1 至 4。"
                         "0(默认)= 不改动,只读出当前值。抬高会让跟随更紧,"
                         "同时也会放大碰撞时产生的力,所以不要随手调大")
    ap.add_argument("--sim-start-offset", type=float, default=0.0,
                    help="单位为度,仅试运行有效。把「机器人当前位置」假设成与首帧"
                         "指令相差这么多度。试运行读不到真机,否则会从指令位置起步,"
                         "同步阶段等于不存在 —— 这个参数让它可以被离线验证")
    ap.add_argument("--lag-warn", type=float, default=25.0,
                    help="指令落后目标超过这个角度(度)就提示一次。限速是安全"
                         "需要,落后是它的必然结果,但操作者需要知道自己跑赢了"
                         "机器人,而不是觉得「手感不对」")
    ap.add_argument("--rec-hz", type=float, default=50.0,
                    help="录真机关节的频率。录制由页面的开关触发,写进页面指定的"
                         "那个子目录,与手臂指令、手部数据同一个目录")
    ap.add_argument("--stale-ms", type=float, default=200.0)
    ap.add_argument("--duration", type=float, default=0.0)
    args = ap.parse_args()

    comps = {c.strip() for c in args.components.split(",") if c.strip()}
    bad = comps - {"left_arm", "right_arm", "head", "torso"}
    if bad:
        raise SystemExit(f"[bridge] 无法识别的部件:{sorted(bad)}")

    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(args.sub)
    ctl = ControlSubscriber(args.control)
    print(f"[bridge] 订阅指令 {args.sub}")
    print(f"[bridge] 订阅开关 {args.control}")
    print(f"[bridge] 跟随部件 {sorted(comps)}   "
          f"{'试运行' if args.dry_run else '真机'}")
    print(f"[bridge] 下发 {args.hz:.0f} 次每秒;同步 {args.sync_speed:.0f} 度每秒,"
          f"跟随 {args.follow_speed:.0f} 度每秒(硬件上限 137 至 155 度每秒)")

    sink = Sink(args.dry_run, comps)
    lim = sink.limits()
    if lim:
        print(f"[bridge] 已从机器人读取 {len(lim)} 个关节限位,以此为准,不采用仿真的限位")
    print(f"[bridge] 手臂速度前馈:{'开(set_joint_pos_vel)' if args.feedforward else '关(纯位置)'}")
    sink.pid(args.arm_pid_scale)

    state = "off"                      # off | syncing | following
    q_now: dict[str, float] = {}
    prev_engaged: list = []
    last_msg = 0.0
    held: dict | None = None
    period = 1.0 / args.hz
    tol = np.radians(args.sync_tol)
    # 两个阶段的上限都是「度每秒」,而且乘的是**实测**的这一拍耗时,不是标称周期。
    # 此前跟随阶段的上限直接就是每拍的角度:改 --hz 会连带改变跟随速度(同步速度
    # 却不受影响),而循环只要跑不满标称频率,机器人就会悄悄变慢,没有任何读数
    # 会显示这件事。按实测时间换算之后,速度对循环抖动免疫。
    w_sync = np.radians(args.sync_speed)
    w_follow = np.radians(args.follow_speed)
    regap = np.radians(args.sync_regap)
    t_prev = 0.0
    sync_goal: dict[str, float] | None = None   # 同步阶段的快照目标,见下
    sync_pass = 0
    epoch = 0                    # 遥操作分段;变化就先回默认姿势
    homing = False               # 当前这趟慢速同步是「回默认位」而不是「追操作者」
    t0 = time.time()
    n_sent = 0
    n_sat = 0                    # 跟随阶段被限速削掉的关节步数
    n_step = 0
    n_tick = 0                   # 用来报实测循环频率,而不是相信 --hz
    n_lag = 0.0                  # 本秒内指令落后目标的最大角度
    t_lagwarn = 0.0
    t_print = 0.0
    t_probe = 0.0
    t_hz = 0.0
    rec_fh = None                # 真机关节录制文件
    rec_n = 0
    t_rec = 0.0
    why = {"未收到消息": 0, "仿真程序版本过旧": 0, "离合未闭合": 0,
           "数据过期": 0, "开关处于关闭状态": 0, "急停": 0, "双手距离过近": 0,
           "无法读取机器人当前位置": 0}

    def goal_of(cmd: dict, mode: str) -> dict:
        """Where we are heading.

        The torso is INCLUDED ONLY IN arms+head+waist. In arms and arms+head it
        is not part of the mapping - the operator positions it by hand - so the
        bridge must not read it, sync it, or command it. Touching it there
        would move the thing both arms hang off, for no reason.
        """
        g = {n: v for n, v in cmd.items()
             if mode == "arms+head+waist" or n not in TORSO}
        return g

    try:
        while True:
            tick = time.time()
            n_tick += 1
            if not t_hz:
                t_hz = tick
            if args.duration and tick - t0 > args.duration:
                break
            try:
                held = msgpack.unpackb(sock.recv(zmq.NOBLOCK), raw=False)
                last_msg = tick
            except zmq.Again:
                pass  # HOLD the previous one - see `fresh` just below.

            # This loop runs faster than the consumer publishes, so most ticks
            # have no new message. Re-reading `cmd` off a missing message would
            # empty it and read as "the stream died": the bridge would disarm
            # on every other tick, and re-arming restarts `syncing`, so the arms
            # would crawl at --sync-speed in stutters and never actually follow.
            # --stale-ms is the thing that decides the data is too old, not the
            # accident of which tick a packet landed on.
            fresh = bool(last_msg) and (tick - last_msg) * 1000.0 < args.stale_ms
            m = held if fresh else None

            cmd = (m or {}).get("cmd") or {}
            engaged = (m or {}).get("engaged") or []
            mode = (m or {}).get("mode") or "arms"
            drive_torso = mode == "arms+head+waist"
            if not last_msg:
                why["未收到消息"] += 1
            elif not fresh:
                why["数据过期"] += 1
            elif not cmd:
                why["仿真程序版本过旧"] += 1
            elif not engaged:
                why["离合未闭合"] += 1

            ctl_state = ctl.latest(tick)
            want_on = bool(ctl_state and ctl_state.get("enabled"))
            if not want_on:
                why["开关处于关闭状态"] += 1

            # 录真机本体数据。页面广播 recording + session,写进和手臂指令、
            # 手部数据同一个目录。与「是否驱动真机」无关 —— 机器人被别的方式
            # 摆动时同样值得记录。
            rec_on = bool(ctl_state and ctl_state.get("recording")
                          and ctl_state.get("session"))
            if rec_on and rec_fh is None:
                d = ROOT / "data" / "sessions" / str(ctl_state["session"])
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    rec_fh = open(d / "robot.msgpack", "ab")
                    rec_n = 0
                    print(f"[bridge] 开始录真机关节 -> {d / 'robot.msgpack'}"
                          + ("(试运行,读不到真机,只会写出 0 帧)"
                             if sink.bot is None else ""))
                except OSError as e:
                    print(f"[bridge] 无法建立录制文件:{e}")
                    rec_on = False
            elif not rec_on and rec_fh is not None:
                rec_fh.close()
                rec_fh = None
                print(f"[bridge] 停止录真机关节,共 {rec_n} 帧")
            if rec_fh is not None and tick - t_rec >= 1.0 / args.rec_hz:
                t_rec = tick
                st = sink.read_state()
                if st:
                    # 只打墙上时钟。仿真那一路的 t_us 是仿真时间,与这里不同源,
                    # 六路数据唯一能对齐的共同口径就是主机墙钟。
                    rec_fh.write(msgpack.packb(
                        {"t_wall_us": int(tick * 1e6), "joints": st},
                        use_bin_type=True))
                    rec_n += 1

            stop = None
            if sink.estopped():
                stop = "急停被按下"
                why["急停"] += 1
            arm_t = (m or {}).get("arm") or {}
            if args.min_hand_gap > 0 and len(arm_t) == 2:
                pr = np.asarray(arm_t["right"]["cmd"], float)
                pl = np.asarray(arm_t["left"]["cmd"], float)
                gap = float(np.linalg.norm(pr - pl))
                if gap < args.min_hand_gap:
                    stop = f"两手指令位置相距仅 {gap * 1000:.0f}mm"
                    why["双手距离过近"] += 1

            ok = bool(want_on and cmd and engaged and fresh and stop is None)

            if not ok:
                if state != "off":
                    print("[bridge] 已停用:" + (stop or "开关关闭或数据中断"))
                state = "off"
                time.sleep(period)
                continue

            # 目标先钳进机器人自己的限位,再参与任何距离计算。钳制放在这里,
            # 快照与收敛判据用的就都是可达的目标(见 goal_of 下方那条注释)。
            live = goal_of(cmd, mode)
            if lim:
                live = {n: (float(np.clip(v, lim[n][0], lim[n][1])) if n in lim else v)
                        for n, v in live.items()}

            if state == "off":
                goal0 = live
                q_meas = sink.read()
                missing = [n for n in goal0 if n not in q_meas]
                if sink.bot is not None and missing:
                    # NEVER fall back to "assume we are already there". A joint
                    # that could not be read would default to its own goal,
                    # report zero distance to it, and get stepped straight onto
                    # the target in a single tick - the exact fast unplanned
                    # motion that `syncing` exists to prevent. Not knowing where
                    # the robot is means not moving it.
                    why["无法读取机器人当前位置"] += 1
                    print(f"[bridge] 无法读取 {len(missing)} 个关节的当前位置"
                          f"({', '.join(missing[:4])}{' ...' if len(missing) > 4 else ''})"
                          ",因此不予启用。慢速同步需要先确定机器人当前所处的位置")
                    time.sleep(period)
                    continue
                q_now = q_meas or {n: v + np.radians(args.sim_start_offset)
                                   for n, v in goal0.items()}
                state = "syncing"
                sync_goal, sync_pass = None, 0
                prev_engaged = list(engaged)
                print(f"[bridge] 模式 {mode}"
                      + ("(躯干由映射驱动)" if drive_torso else
                         "(躯干不动作,也不读取,由操作者手动摆放)"))

            if set(engaged) != set(prev_engaged):
                print("[bridge] 离合状态变化,重新进行慢速同步")
                state, sync_goal = "syncing", None
                prev_engaged = list(engaged)

            # 新的一段:先把机器人**物理地**送回默认姿势,再开始跟随。参考实现
            # 每个回合都这么做,坏构型因此活不过一个回合。走的是同一套慢速同步
            # 机制,只是目标换成固定的默认位而不是操作者当前姿势。
            _e = int((ctl_state or {}).get("epoch") or 0)
            if _e != epoch:
                epoch = _e
                home = home_for(comps)
                if home:
                    state, sync_goal = "syncing", dict(home)
                    sync_pass = 0
                    homing = True
                    far = max((abs(np.degrees(v - q_now.get(n, v)))
                               for n, v in home.items()), default=0.0)
                    print(f"[bridge] 第 {_e} 段开始:回默认姿势,相差最大 {far:.1f}°,"
                          f"按 {args.sync_speed} 度每秒移动,预计 "
                          f"{far / max(args.sync_speed, 1e-6):.1f} 秒")

            if state == "syncing" and sync_goal is None:
                # 同步阶段追的是一张快照,不是活动目标。追活动目标时收敛判据
                # 永远不成立:戴着头显的人,头几乎不可能不动,而头的三个关节
                # 也在 goal 里,于是 worst 始终大于阈值,状态机永远离不开同步
                # 档,双臂就以 --sync-speed 一直爬。快照是不动的,所以这一趟
                # 必定在 far/--sync-speed 秒内走完。
                sync_goal = dict(live)
                sync_pass += 1
                far = max((abs(np.degrees(v - q_now.get(n, v)))
                           for n, v in sync_goal.items()), default=0.0)
                print(f"[bridge] 慢速同步第 {sync_pass} 趟:相差最大的关节为 {far:.1f}°,"
                      f"按 {args.sync_speed} 度每秒移动,预计需要 "
                      f"{far / max(args.sync_speed, 1e-6):.1f} 秒,期间请保持姿势不变")

            goal = sync_goal if state == "syncing" else live
            # 这一拍实际过去了多久。上限 4 倍周期,免得某次卡顿(GC、日志刷盘)
            # 换来一次超大的步长 —— 限速器的意义就在于任何一拍都不许走太远。
            dt = min(max(tick - t_prev, 0.0), 4.0 * period) if t_prev else period
            t_prev = tick
            step = (w_sync if state == "syncing" else w_follow) * dt
            out, qd, worst, sat = {}, {}, 0.0, 0
            for n, v in goal.items():
                cur = q_now.get(n, v)
                d = v - cur
                if abs(d) > step:
                    sat += 1
                nxt = cur + float(np.clip(d, -step, step))
                if n in lim:
                    nxt = float(np.clip(nxt, lim[n][0], lim[n][1]))
                out[n] = nxt
                # 这一拍指令实际走了多远除以实际用时,就是指令速度。它天生被上面
                # 的限速钳过,所以有界且与位置指令严格自洽 —— 不是另算一份估计。
                qd[n] = (nxt - cur) / max(dt, 1e-6)
                worst = max(worst, abs(d))
            q_now = out
            if state == "following":
                n_sat += sat
                n_step += len(goal)
                # 指令落后目标多少。限速是安全需要,落后是它的必然结果 ——
                # 但操作者不知道自己正在跑赢机器人,只会觉得「手感不对」。
                # 持续落后就说一声,别让他猜。
                lag_deg = max(n_lag, np.degrees(worst))
                n_lag = lag_deg
                if lag_deg > args.lag_warn and tick - t_lagwarn > 3.0:
                    t_lagwarn = tick
                    print(f"[bridge] 指令落后目标 {lag_deg:.0f}°,已被限速削掉 "
                          f"{100.0 * n_sat / max(1, n_step):.0f}% 的关节步 —— "
                          f"你比机器人快,慢一点它才跟得上")
            sending = sink.write(out,
                                 include_torso=drive_torso and "torso" in comps,
                                 qd=qd if args.feedforward else None)
            if n_sent == 0:
                # 第一次真正下发时把清单打出来。问"确保只发手臂了吗"要有读数
                # 回答,而不是靠读代码。
                _desc = "、".join(f"{c}({len(n)} 个关节)" for c, n in sending)
                print(f"[bridge] 本次实际下发的部件:{_desc or '无'}")
                _skip = [c for c in ("left_arm", "right_arm", "head", "torso")
                         if c not in [x[0] for x in sending]]
                print(f"[bridge] 不下发:{'、'.join(_skip) or '无'}")
            n_sent += 1

            if state == "syncing" and worst < tol:
                if homing:
                    # 刚回到默认姿势。下面那段会照常再慢速走一趟到操作者当前
                    # 姿势 —— 这正是我们要的顺序:先回已知姿势,再从那里出发。
                    homing = False
                    print("[bridge] 已回到默认姿势")
                # 走到快照姿势了。操作者在这期间移动了多少,决定接下来是转入
                # 跟随,还是再慢速走一趟 —— 转入跟随的那一刻剩多大的差,就会
                # 以跟随速度走多远,所以这个差必须先量出来。
                gap = max((abs(np.degrees(v - q_now.get(n, v)))
                           for n, v in live.items()), default=0.0)
                if np.radians(gap) > regap:
                    print(f"[bridge] 已走到快照姿势,但操作者此间移动了 {gap:.1f}°,"
                          f"超过 {args.sync_regap}°,再慢速走一趟")
                    sync_goal = None
                else:
                    state, sync_goal = "following", None
                    print(f"[bridge] 同步完成,开始跟随(操作者此间移动 {gap:.1f}°)")

            if tick - t_print >= 1.0:
                # 试运行和真机都打。此前这段被 args.dry_run 挡着,于是真机跑的
                # 时候一条周期读数都没有 —— 而「真机跟不跟得上」正是只有真机在
                # 跑时才能量到的东西。
                t_print = tick
                shown = " ".join(f"{n}={np.degrees(out[n]):+6.1f}"
                                 for n in ("R_arm_j1", "L_arm_j1", "head_j1",
                                           "torso_j3") if n in out)
                extra = f" 实测 {n_tick / max(1e-6, tick - t_hz):3.0f} 次每秒"
                n_tick, t_hz = 0, tick
                if state == "following":
                    extra += f" 限速占比 {100.0 * n_sat / max(1, n_step):4.1f}%"
                    n_sat = n_step = 0
                if sink.bot is not None and tick - t_probe >= 2.0:
                    # 指令与实测之差。限速占比回答「是不是我自己限住了」,这一项
                    # 回答「机器人有没有跟上我发的指令」,两者缺一不可:两个都小
                    # 才说明慢的原因在更上游。
                    t_probe = tick
                    meas = sink.read()
                    lag = [(n, abs(np.degrees(v - meas[n])))
                           for n, v in out.items() if n in meas]
                    if lag:
                        dn, dv = max(lag, key=lambda kv: kv[1])
                        extra += f" 真机滞后 {dv:5.1f}°({dn})"
                if state == "following":
                    extra += f" 指令落后 {n_lag:4.0f}°"
                n_lag = 0.0
                print(f"[bridge] {state:9s} t={tick - t0:5.1f}s 已发 {n_sent} "
                      f"最远 {np.degrees(worst):5.1f}°{extra}  {shown}")
            time.sleep(max(0.0, period - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()

    print(f"[bridge] 已结束,共下发 {n_sent} 条指令")
    if n_sent == 0:
        print("[bridge] 未下发任何指令。各周期被拦下的原因如下:")
        for k, v in why.items():
            if v:
                print(f"    {k:16s} {v}")
        if why["仿真程序版本过旧"]:
            print("    -> 仿真程序只发送了 q,没有 cmd,说明其启动早于该字段的加入,需要重启。")
        if why["开关处于关闭状态"]:
            print("    -> 浏览器页面(http://localhost:8086)中的开关尚未打开。")
    return 0 if n_sent else 1


if __name__ == "__main__":
    sys.exit(main())
