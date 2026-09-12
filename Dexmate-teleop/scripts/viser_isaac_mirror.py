#!/usr/bin/env python3
"""Mirror Isaac's ACTUAL joint positions in Viser. No mapping, no IK, no drift.

Isaac renders the bare USD, which has no hands, so palm orientation cannot be
judged there. viser_mapping_preview.py draws the hands - but it gets there by
re-running the mapping and the IK itself, and two implementations of one
mapping drift. That has already misled the operator twice in one day: once the
preview was defaulted to a different --control-mode than the running consumer
(so the torso moved in the browser and not in Isaac, read as "the waist is not
locked"), and once to a different mapping law and scale.

This viewer cannot drift, because it computes nothing. The consumer publishes
the joint positions Isaac actually has - not the commanded targets, so a PD
that is not keeping up shows up here too - and this only poses the URDF with
them and draws the Sharpa hands on top.

The display + switches live in the Mirror class so that vega_console.py can
embed the SAME implementation instead of growing a second one - the drift rule
above applies to viewers as much as to mappings.

    consumer:  sim/teleop_vega_pico.py ... --pub-state tcp://127.0.0.1:5583
    viewer:    .venv/bin/python scripts/viser_isaac_mirror.py

Usage:
    .venv/bin/python scripts/viser_isaac_mirror.py
    .venv/bin/python scripts/viser_isaac_mirror.py --sub tcp://127.0.0.1:5583
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import json
import msgpack
import numpy as np
import viser
import viser.extras
import yourdfpy
import zmq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.palm_fix import (  # noqa: E402
    FINGERS_IN_EE, PALM_FIX, PALM_IN_EE)
from magicdexmate.control_link import ControlPublisher  # noqa: E402
from magicdexmate.retarget.mapping import sharpa_sdk_joint_names  # noqa: E402

_SHARPA = pathlib.Path(__file__).resolve().parents[1] / "assets/vega_1_sharpa.urdf"
BARE = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
DEFAULT_URDF = _SHARPA if _SHARPA.exists() else BARE
EE_LINKS = {"right": "R_ee", "left": "L_ee"}


class Mirror:
    """The robot view + the operator's switches, on an existing viser server.

    tick() is non-blocking; the caller owns the loop and the sleep. Counters
    for smoke tests: self.n_arm (arm frames drawn), self.hand_state["n"].
    """

    # SMPL-24 的父关节表(与 viser_body_review.py 同一份来源)。画操作者骨架用。
    PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
              18, 19, 20, 21]

    def __init__(self, server: viser.ViserServer,
                 sub: str = "tcp://127.0.0.1:5583",
                 control: str = "tcp://127.0.0.1:5584",
                 hand_right: str = "tcp://127.0.0.1:5556",
                 hand_left: str = "",
                 urdf_path: pathlib.Path = DEFAULT_URDF,
                 extra_state=None, real_buttons=None,
                 pico: str = "tcp://127.0.0.1:5581",
                 robot_state: str = "tcp://127.0.0.1:5590",
                 cam_view: str = "tcp://127.0.0.1:5591"):
        self.server = server
        # extra keys merged into every switch broadcast (the console adds
        # hand_enabled / recording through this)
        self.extra_state = extra_state

        self.urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True,
                                       build_scene_graph=True)
        self.names = [j for j, jo in self.urdf.joint_map.items()
                      if jo.type != "fixed"]

        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)      # always the freshest frame
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(sub)

        server.scene.set_up_direction("+z")
        self.vurdf = viser.extras.ViserUrdf(server, self.urdf,
                                            root_node_name="/robot")
        server.scene.add_grid("/grid", width=3.0, height=3.0)
        g = server.gui
        self.g_state = g.add_text("链路", "等待手臂程序 …")
        self.g_gaze = g.add_text("头看哪", "-")
        self.g_lean = g.add_text("胸前倾", "-")
        self.g_palm = g.add_text("掌心 R/L", "-")
        # The consumer publishes its own commanded EE target next to the
        # achieved joint state, so this readout costs no recomputation -
        # deriving the target here would make this a second implementation
        # again, which is the one thing this viewer exists not to be.
        self.g_track = g.add_text("跟随状态", "-")
        self.g_pinned = g.add_text("顶在限位的关节", "-")
        self.g_hand = g.add_text("Sharpa 手", "-")
        self.g_cam = g.add_text("相机画面", "无(相机进程未运行)")
        self.g_pico = g.add_text("操作者", "等待 PICO …")
        self.g_pico_palm = g.add_text("操作者掌心", "-")

        # --- the operator's switches. They live here because these are
        # decisions made WHILE WEARING the headset. Republished every tick; a
        # subscriber that stops hearing this for 1.5 s treats it as OFF, so
        # closing the browser stops the robot rather than leaving it running
        # on the last thing it was told.
        with g.add_folder("真机"):
            if real_buttons is not None:
                real_buttons(g)     # the console's connect/disconnect buttons
            self.g_enable = g.add_checkbox("驱动手臂(Dexmate)", False)
            self.g_bridge = g.add_text("手臂状态", "未连接")
            self.g_hand_en = g.add_checkbox("驱动真手(Sharpa)", False)
            self.g_sharpa = g.add_text("真手状态", "未连接")
            g.add_markdown("打开后机器人先缓慢移动到当前姿势再跟随。"
                           "关闭页面即停止。")
        with g.add_folder("控制模式"):
            self.g_mode = g.add_dropdown(
                "mode", ["arms", "arms+head", "arms+head+waist"],
                initial_value="arms+head+waist")
            g.add_markdown("仅 `arms+head+waist` 驱动躯干,另外两档由人工摆放。")
        with g.add_folder("防抖动"):
            self.g_smooth = g.add_slider("强度 (Hz,越小越稳越钝)",
                                         0.0, 20.0, 0.5, 6.0)

        self.ctl = ControlPublisher(control)
        print(f"[mirror] 开关频道 {control}")

        # --- the HAND stream. Separate pipeline, separate glove, separate
        # port - JSON (not msgpack): a `hello` every 2 s with joint names +
        # crc32, `qpos` at 60 Hz. NO CONFLATE: conflate would drop essentially
        # every hello (measured 713 qpos, 0 hello over 12 s) and the crc check
        # would silently never run.
        self.hand_socks = {}
        self.hand_state = {"q": {}, "crc": {}, "names": {}, "n": 0, "last": 0.0}
        for h, addr in (("right", hand_right), ("left", hand_left)):
            if not addr:
                continue
            s_ = zmq.Context.instance().socket(zmq.SUB)
            s_.setsockopt(zmq.SUBSCRIBE, b"")
            s_.setsockopt(zmq.RCVHWM, 200)
            s_.connect(addr)
            self.hand_socks[h] = s_
            self.hand_state["names"][h] = sharpa_sdk_joint_names(h)
            print(f"[mirror] 订阅 {h} 手 {addr}")

        print(f"[mirror] 订阅 {sub}")
        # 操作者那一侧。页面此前只显示机器人,看不到「人是怎么映射过去的」。
        # 直接订 PICO producer,不经过 consumer —— 两者是独立的两条流,人这边
        # 停了机器人照常显示,反之亦然。
        self.pico_sock = None
        self.n_pico = 0
        self.n_palm = 0
        # PICO 是 y 上、viser 是 z 上。位置换轴用的是 [-z,-x,y],
        # 写成矩阵,朝向要用同一个基变换(R' = M R Mᵀ)才自洽。
        self._PICO_TO_VISER = np.array([[0., 0., -1.],
                                        [-1., 0., 0.],
                                        [0., 1., 0.]])
        if pico:
            self.pico_sock = zmq.Context.instance().socket(zmq.SUB)
            self.pico_sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.pico_sock.setsockopt(zmq.CONFLATE, 1)
            self.pico_sock.connect(pico)
            print(f"[mirror] 订阅 PICO {pico}(显示操作者骨架)")
        # 真机实测姿态(dexmate_observer.py 只读发布)。仿真流缺席时用它画,
        # 页面一打开就与真机一致,而不是摆默认姿势。仿真流一来就让位 ——
        # 仿真在跑时,操作者要看的是指令链路,不是回显。
        self.robot_sock = None
        self.robot_q: dict = {}
        self.robot_last = 0.0
        self.n_robot = 0
        if robot_state:
            self.robot_sock = zmq.Context.instance().socket(zmq.SUB)
            self.robot_sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.robot_sock.setsockopt(zmq.CONFLATE, 1)
            self.robot_sock.connect(robot_state)
            print(f"[mirror] 订阅真机回显 {robot_state}(仿真未运行时显示)")
        # 相机画面流(kinect_pointcloud --pub-view)。外参没标,点云与机器人
        # 不在同一坐标系,所以摆在机器人**旁边**的固定位置看内容;标完外参再
        # 谈对齐。相机光轴系(x右 y下 z前)转到 viser 的 z 上:z->x、x->-y、
        # y->-z,前方指向 +x,直立不倒挂。
        self.cam_sock = None
        self.n_cam = 0
        self._cam_last = 0.0
        # 多相机:按序列号各一份点云与计数,新序列号出现时在旁边再排一台的
        # 位置(外参都没标,并排看内容即可)
        self._cam_pcs: dict = {}
        self._cam_stat: dict = {}
        self._CAM_TO_VISER = np.array([[0.0, 0.0, 1.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]])
        self._CAM_OFFSET = np.array([1.2, 1.2, 0.8])
        if cam_view:
            self.cam_sock = zmq.Context.instance().socket(zmq.SUB)
            self.cam_sock.setsockopt(zmq.SUBSCRIBE, b"")
            self.cam_sock.setsockopt(zmq.CONFLATE, 1)
            self.cam_sock.connect(cam_view)
            print(f"[mirror] 订阅相机画面 {cam_view}")
        self.n_arm = 0
        self.last = 0.0
        # Held across ticks so a tick with no arm message still draws the arm
        # WHERE IT ACTUALLY IS instead of snapping 20 joints to zero.
        self.arm_q: dict = {}

    # -- internals ------------------------------------------------------------
    def _pump_hands(self):
        for h, s_ in self.hand_socks.items():
            while True:
                try:
                    msg = json.loads(s_.recv(zmq.NOBLOCK).decode())
                except zmq.Again:
                    break
                except Exception:                          # noqa: BLE001
                    break
                if msg.get("type") == "hello":
                    if self.hand_state["crc"].get(h) is None:
                        got = msg.get("joint_names") or []
                        mine = sharpa_sdk_joint_names(h)
                        if list(got) != list(mine):
                            print(f"[mirror] ⚠ {h} 手关节顺序与本地不一致,"
                                  f"改用手套程序报的顺序({len(got)} 个)")
                        else:
                            print(f"[mirror] {h} 手关节顺序已确认 "
                                  f"({len(got)} 个, crc {msg.get('crc')})")
                    self.hand_state["crc"][h] = msg.get("crc")
                    self.hand_state["names"][h] = msg.get("joint_names")
                elif msg.get("type") == "qpos":
                    if self.hand_state["crc"].get(h) is not None \
                            and msg.get("crc") != self.hand_state["crc"][h]:
                        continue          # joint order changed under us
                    nm = self.hand_state["names"].get(h) or []
                    qp = msg.get("qpos") or []
                    if len(nm) == len(qp):
                        self.hand_state["q"].update(dict(zip(nm, qp)))
                        self.hand_state["n"] += 1
                        self.hand_state["last"] = time.time()

    def _pump_pico(self):
        """画操作者的 body24 骨架。位置按 PICO 原始坐标(y 轴朝上)转成
        viser 的 z 轴朝上,并整体挪到机器人旁边 —— 不做任何映射,这里要显示的
        就是「人摆成什么样」,映射对不对由机器人那一侧回答。"""
        if self.pico_sock is None:
            return
        msg = None
        while True:
            try:
                msg = msgpack.unpackb(self.pico_sock.recv(zmq.NOBLOCK), raw=False)
            except zmq.Again:
                break
        if msg is None:
            return
        b = msg.get("body24")
        if not b:
            return
        p = np.array([[f[0], f[1], f[2]] for f in b], dtype=np.float32)
        # PICO 是 y 上;viser 场景是 z 上。只换轴,不缩放。
        p = np.stack([-p[:, 2], -p[:, 0], p[:, 1]], axis=1)
        p[:, 1] += 1.2                      # 挪到机器人左侧,两个不重叠
        seg = np.array([[p[par], p[j]] for j, par in enumerate(self.PARENT)
                        if par >= 0], dtype=np.float32)
        col = np.full((len(seg), 2, 3), (90, 200, 255), dtype=np.uint8)
        self.server.scene.add_line_segments("/pico/bones", seg, col, line_width=4.0)
        self.server.scene.add_point_cloud(
            "/pico/joints", p,
            np.full((len(p), 3), 235, dtype=np.uint8), point_size=0.02)

        # 掌心朝哪 —— 只画骨架的话看不出这个,而掌心朝向恰恰是这条映射的关键。
        # 计算与 viser_body_review.py 完全同一份(PALM_FIX 把 SMPL 腕系搬到
        # R_ee,掌法向 = +ee.x、手指 = +ee.z),不另写第二份。
        try:
            from scipy.spatial.transform import Rotation as _R
            st, ep, ef = [], [], []
            for hand, j in (("left", 20), ("right", 21)):
                q = np.asarray(b[j][3:], dtype=np.float64)      # xyzw
                if abs(np.linalg.norm(q) - 1.0) > 1e-3:
                    continue                                    # 没有有效朝向
                Rw = _R.from_quat(q).as_matrix()
                Rw = self._PICO_TO_VISER @ Rw @ self._PICO_TO_VISER.T
                R_ee = Rw @ PALM_FIX[hand]
                st.append(p[j])
                ep.append(p[j] + 0.20 * (R_ee @ PALM_IN_EE))
                ef.append(p[j] + 0.20 * (R_ee @ FINGERS_IN_EE))
            if st:
                self.server.scene.add_arrows(
                    "/pico/palm", np.stack([np.array(st), np.array(ep)], axis=1),
                    np.array([[245, 245, 245]] * len(st), dtype=np.uint8),
                    shaft_radius=0.007, head_radius=0.018, head_length=0.04)
                self.server.scene.add_arrows(
                    "/pico/fingers", np.stack([np.array(st), np.array(ef)], axis=1),
                    np.array([[255, 210, 60]] * len(st), dtype=np.uint8),
                    shaft_radius=0.007, head_radius=0.018, head_length=0.04)
                self.n_palm += 1
                # 用词说出来,不只画箭头 —— 戴着头显时读一行字比看箭头快
                def _dir(v):
                    ax = ("前", "后", "左", "右", "上", "下")
                    k = int(np.argmax(np.abs(v)))
                    return ax[2 * k + (0 if v[k] > 0 else 1)]
                self.g_pico_palm.value = " / ".join(
                    f"{'左' if i == 0 else '右'}掌{_dir(np.asarray(ep[i]) - np.asarray(st[i]))}"
                    f"·指{_dir(np.asarray(ef[i]) - np.asarray(st[i]))}"
                    for i in range(len(st)))
        except Exception:                                       # noqa: BLE001
            pass
        self.n_pico += 1
        self.g_pico.value = f"✅ 骨架 {self.n_pico} 帧"

    def _broadcast(self):
        # EVERY tick, arm stream up or not: the Sharpa enable and the
        # recording flag must reach their listeners even when the arm
        # consumer is down - the pipelines are independent.
        state = {"enabled": bool(self.g_enable.value),
                 "hand_enabled": bool(self.g_hand_en.value),
                 "mode": self.g_mode.value,
                 "smooth": float(self.g_smooth.value)}
        if self.extra_state is not None:
            state.update(self.extra_state())
        self.ctl.send(state)

    def _draw(self, q: dict):
        arr = np.array([q.get(j, 0.0) for j in self.names])
        self.vurdf.update_cfg(arr)
        self.urdf.update_cfg(arr)

    # -- one non-blocking iteration -------------------------------------------
    def tick(self):
        self._pump_hands()
        self._pump_pico()
        self._broadcast()
        if self.hand_socks:
            age = (time.time() - self.hand_state["last"]
                   if self.hand_state["last"] else 999)
            self.g_hand.value = (f"✅ {self.hand_state['n']} 帧" if age < 1.0
                                 else "⚠ 无数据,手套程序未运行")
        if self.robot_sock is not None:
            try:
                r = msgpack.unpackb(self.robot_sock.recv(zmq.NOBLOCK), raw=False)
                self.robot_q = r.get("q") or self.robot_q
                self.robot_last = time.time()
                self.n_robot += 1
            except zmq.Again:
                pass
        if self.cam_sock is not None:
            # CONFLATE 每次只留最新一条;多相机各 5Hz 交替发,单个 tick 只收
            # 一条也够(页面 50Hz tick,远快于来流)。
            try:
                c = msgpack.unpackb(self.cam_sock.recv(zmq.NOBLOCK), raw=False)
                serial = str(c.get("serial", "cam"))
                pts = (np.frombuffer(c["xyz_mm"], dtype=np.int16)
                       .reshape(-1, 3).astype(np.float32) / 1000.0)
                cols = np.frombuffer(c["rgb"], dtype=np.uint8).reshape(-1, 3)
                if serial not in self._cam_pcs:
                    idx = len(self._cam_pcs)
                    off = self._CAM_OFFSET + np.array([0.0, -1.6 * idx, 0.0])
                    self._cam_pcs[serial] = {
                        "off": off,
                        "pc": self.server.scene.add_point_cloud(
                            f"/camera_view/{serial}",
                            points=pts @ self._CAM_TO_VISER.T + off,
                            colors=cols, point_size=0.012)}
                    print(f"[mirror] 相机 {serial} 的画面已加入场景")
                else:
                    h = self._cam_pcs[serial]
                    h["pc"].points = pts @ self._CAM_TO_VISER.T + h["off"]
                    h["pc"].colors = cols
                self._cam_stat[serial] = self._cam_stat.get(serial, 0) + 1
                self.n_cam += 1
                self._cam_last = time.time()
            except zmq.Again:
                pass
            if self._cam_last:
                age = time.time() - self._cam_last
                if age < 2.0:
                    per = " · ".join(f"{s} {n}帧"
                                     for s, n in self._cam_stat.items())
                    self.g_cam.value = f"✅ {len(self._cam_pcs)} 台:{per}"
                else:
                    self.g_cam.value = f"⚠ 断流 {age:.0f}s(相机进程停了?)"
        try:
            m = msgpack.unpackb(self.sock.recv(zmq.NOBLOCK), raw=False)
        except zmq.Again:
            m = None
            if time.time() - self.last > 1.5 and self.last:
                self.g_state.value = (f"⚠ 断流 {time.time() - self.last:.0f}s,"
                                      "仿真程序可能已停止")
            elif not self.last:
                self.g_state.value = "等待手臂程序 …"
        if m is None:
            # 仿真流缺席(还没启动,或已断流 1.5 秒以上)时,用真机实测姿态
            # 回显 —— 页面一打开就与真机一致。仿真流一恢复,下一帧就让位。
            sim_absent = (not self.last) or (time.time() - self.last > 1.5)
            if sim_absent and self.robot_sock is not None \
                    and time.time() - self.robot_last < 1.0:
                self._draw({**self.robot_q, **self.hand_state["q"]})
                self.g_state.value = (f"⚠ 仿真未运行 — 显示真机实测姿态"
                                      f"(只读,{self.n_robot} 帧)")
            elif self.hand_state["q"] or self.arm_q:
                # hands only: HOLD the last arm pose, animate the fingers
                self._draw({**self.arm_q, **self.hand_state["q"]})
            return

        self.arm_q = m.get("q") or self.arm_q
        q = {**self.arm_q, **self.hand_state["q"]}
        self.last, self.n_arm = time.time(), self.n_arm + 1
        self._draw(q)
        self.g_state.value = f"✅ {self.n_arm} 帧"

        arm = m.get("arm") or {}
        if arm:
            res = {h: float(np.linalg.norm(np.array(v["ee"])
                                           - np.array(v["cmd"])) * 1000)
                   for h, v in arm.items()}
            w = max(res.values())
            self.g_track.value = (
                f"✓ 跟随正常 ({w:.0f}mm)" if w < 30 else
                f"⚠ 偏差 {w:.0f}mm,接近饱和" if w < 80 else
                f"✗ 偏差 {w:.0f}mm,该方向已无法跟随")
            seg = [[np.array(v["ee"]), np.array(v["cmd"])]
                   for h, v in arm.items() if res[h] > 5.0]
            if seg:
                self.server.scene.add_arrows(
                    "/gap", np.asarray(seg),
                    np.array([[255, 40, 40]] * len(seg), dtype=np.uint8),
                    shaft_radius=0.006, head_radius=0.014, head_length=0.03)
            else:
                self.server.scene.add_arrows(
                    "/gap", np.zeros((1, 2, 3)),
                    np.array([[255, 40, 40]], dtype=np.uint8),
                    shaft_radius=0.001, head_radius=0.001, head_length=0.001)

        # a joint parked on its limit has no DoF left: this is what "the IK
        # locks up" and "the two arms diverge" look like from outside
        hit = []
        for jn, jo in self.urdf.joint_map.items():
            if "_arm_j" not in jn or jn not in q or jo.limit is None:
                continue
            v = q[jn]
            if (v - jo.limit.lower < np.radians(1.0)
                    or jo.limit.upper - v < np.radians(1.0)):
                hit.append(jn.replace("_arm_", ""))
        self.g_pinned.value = ", ".join(hit) if hit else "无"

        try:
            C = np.array(self.urdf.get_transform("arm_center",
                                                 self.urdf.base_link))
            self.g_lean.value = \
                f"{np.degrees(np.arccos(np.clip(C[2, 2], -1, 1))):.0f}°"
            H = np.array(self.urdf.get_transform("vega_1_head_l3",
                                                 self.urdf.base_link))
            gz = H[:3, 0]
            self.g_gaze.value = (
                f"方位 {np.degrees(np.arctan2(gz[1], gz[0])):+.0f}°  "
                f"仰角 {np.degrees(np.arcsin(np.clip(gz[2], -1, 1))):+.0f}°")
            starts, ends_p, ends_f, dirs = [], [], [], []
            for h, ln in EE_LINKS.items():
                T = np.array(self.urdf.get_transform(ln, self.urdf.base_link))
                p = T[:3, 3]
                starts.append(p)
                ends_p.append(p + 0.22 * (T[:3, :3] @ PALM_IN_EE))
                ends_f.append(p + 0.22 * (T[:3, :3] @ FINGERS_IN_EE))
                d = T[:3, :3] @ PALM_IN_EE
                ax = [("前", [1, 0, 0]), ("后", [-1, 0, 0]), ("左", [0, 1, 0]),
                      ("右", [0, -1, 0]), ("上", [0, 0, 1]), ("下", [0, 0, -1])]
                dirs.append(max(ax, key=lambda kv: float(np.dot(d, kv[1])))[0])
            self.g_palm.value = f"{dirs[0]} / {dirs[1]}"
            self.server.scene.add_arrows(
                "/palm", np.stack([np.array(starts), np.array(ends_p)], axis=1),
                np.array([[245, 245, 245]] * 2, dtype=np.uint8),
                shaft_radius=0.008, head_radius=0.020, head_length=0.045)
            self.server.scene.add_arrows(
                "/fingers", np.stack([np.array(starts), np.array(ends_f)], axis=1),
                np.array([[255, 210, 60]] * 2, dtype=np.uint8),
                shaft_radius=0.008, head_radius=0.020, head_length=0.045)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", default="tcp://127.0.0.1:5583")
    ap.add_argument("--control", default="tcp://127.0.0.1:5584")
    ap.add_argument("--hand-right", default="tcp://127.0.0.1:5556",
                    help="右手手套数据流,空字符串表示不订阅")
    ap.add_argument("--hand-left", default="")
    ap.add_argument("--headless-seconds", type=float, default=0.0,
                    help="无界面自检:订阅 N 秒后打印读数并退出,"
                         "两条数据流都没有数据时返回非零")
    ap.add_argument("--urdf", type=pathlib.Path, default=DEFAULT_URDF)
    ap.add_argument("--port", type=int, default=8086)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    mirror = Mirror(server, sub=args.sub, control=args.control,
                    hand_right=args.hand_right, hand_left=args.hand_left,
                    urdf_path=args.urdf)
    _port = getattr(server, "get_port", lambda: args.port)()
    print(f"[mirror] open http://localhost:{_port}")

    t_start = time.time()
    while True:
        if args.headless_seconds and time.time() - t_start > args.headless_seconds:
            n, hn = mirror.n_arm, mirror.hand_state["n"]
            print(f"[mirror] headless smoke: 手臂 {n} 帧 / 手 {hn} 帧  "
                  f"链路={mirror.g_state.value}  手={mirror.g_hand.value}  "
                  f"跟随={mirror.g_track.value}  限位={mirror.g_pinned.value}  "
                  f"胸前倾={mirror.g_lean.value}  视线={mirror.g_gaze.value}  "
                  f"掌心={mirror.g_palm.value}")
            return 0 if (n > 0 or hn > 0) else 1
        mirror.tick()
        time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
