#!/usr/bin/env python
"""Viser joint teach bench: pose Vega by hand and record what each joint does.

Robot and sliders live in one browser window, driven straight off the URDF -
no Isaac, no GPU, no second process. Posing a robot is pure kinematics, and
the URDF is the same kinematics the USD was generated from (verified to 0.5 mm;
they differ only by a fixed base transform), so physics buys nothing here and
costs a 40 s startup plus the GPU the teleop replays want.

Why it exists
-------------
Rather than guessing mapping parameters and checking the result, the operator
authors the answer: for each STATIC action in the catalogue they pose the robot
the way it ought to look, and save. Paired with the tracker capture of the same
action, the mapping stops being a guess and becomes a fit, with these poses as
ground truth.

  logs/robot_action_poses.json   action slug -> authored joints + EE poses
  logs/joint_jog_notes.jsonl     free-text observations, one JSON per line

Run (one terminal, from the MagicDexMate root):
    .venv/bin/python scripts/viser_joint_jog.py          # browser: :8081
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import viser
import viser.extras
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
from magicdexmate.actions import (  # noqa: E402
    FREEFORM, HEAD_WAIST_LABELS, SESSION_PM_LABELS, STATIC_LABELS, slug_of)

_SHARPA = pathlib.Path(__file__).resolve().parents[1] / "assets/vega_1_sharpa.urdf"
# prefer the hand-fitted model when it has been built - what the real robot
# wears - and fall back to the bare arm so this still runs without it
DEFAULT_URDF = _SHARPA if _SHARPA.exists() else pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
# the "tuck" home the scene and the Pink solver both start from (vega_scene.py
# init_state / pink_vega_ik.HOME_ARM) - open where the robot actually sits,
# not at an all-zeros pose that never occurs in the rig
HOME = {
    **TORSO_HOME,
    "R_arm_j1": 0.5, "R_arm_j2": -0.3, "R_arm_j3": 0.0, "R_arm_j4": -2.2,
    "R_arm_j5": -0.4, "R_arm_j6": 0.0, "R_arm_j7": 0.0,
    "L_arm_j1": -0.5, "L_arm_j2": 0.3, "L_arm_j3": 0.0, "L_arm_j4": -2.2,
    "L_arm_j5": 0.4, "L_arm_j6": 0.0, "L_arm_j7": 0.0,
}
SKIP = "_wheel_j"                  # base is parked; never posed
# Vega's own joints, by name. On the Sharpa-fitted model 44 more actuated
# joints arrive from the hands; they render, but they are not what this bench
# poses, and 44 extra sliders would bury the 20 that matter.
POSABLE = ("torso_j", "_arm_j", "head_j")
GROUPS = [
    ("torso 躯干", lambda n: n.startswith("torso")),
    ("right arm 右臂", lambda n: n.startswith("R_arm")),
    ("left arm 左臂", lambda n: n.startswith("L_arm")),
    ("head 头", lambda n: n.startswith("head")),
]
EE_LINKS = {"right": "R_ee", "left": "L_ee"}
# The arm chain, drawn so "which link is the EE" is answerable by looking.
# R_ee is not a joint: R_arm_j7 is the last moving one, then two fixed hops
# (l7 -> l8, then l8 -> R_ee rotated 90 deg about z, zero translation). So
# R_ee sits exactly on R_arm_l8 - the arm's end flange, where a hand bolts on -
# and the human counterpart of it is the WRIST, not the hand.
CHAIN = {
    "right": ["vega_1_R_arm_l1", "R_arm_l2", "R_arm_l3", "R_arm_l4",
              "R_arm_l5", "vega_1_R_arm_l6", "R_arm_l7", "R_arm_l8", "R_ee"],
    "left": ["vega_1_L_arm_l1", "L_arm_l2", "L_arm_l3", "L_arm_l4",
             "L_arm_l5", "vega_1_L_arm_l6", "L_arm_l7", "L_arm_l8", "L_ee"],
}
CHAIN_COL = {"right": (255, 170, 60), "left": (80, 200, 255)}
REF_LINK = "arm_center"            # chest; the mapping's robot-side anchor
# Last link of the head chain. Its +x is the gaze direction and its +z is the
# top of the head - established by sweeping each joint and watching which way
# the link turns, not by reading the URDF axes (those are in each joint's own
# frame, and at the leaning working pose they do not mean what they look like).
HEAD_TIP = "vega_1_head_l3"
MIRROR_FLIP = ("1", "2", "5")      # left-arm joints whose sign mirrors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=pathlib.Path, default=DEFAULT_URDF)
    ap.add_argument("--port", type=int, default=8081,
                    help="8081, not 8080: the trace viewer owns 8080 and both "
                         "benches are useful at once")
    ap.add_argument("--notes", type=pathlib.Path,
                    default=pathlib.Path("logs/joint_jog_notes.jsonl"))
    ap.add_argument("--poses", type=pathlib.Path,
                    default=pathlib.Path("logs/robot_action_poses.json"))
    ap.add_argument("--no-meshes", action="store_true",
                    help="skip visual meshes (faster load, ugly robot)")
    args = ap.parse_args()
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    args.poses.parent.mkdir(parents=True, exist_ok=True)

    if not args.urdf.exists():
        raise SystemExit(f"URDF not found: {args.urdf}")
    print(f"[bench] loading {args.urdf}")
    urdf = yourdfpy.URDF.load(str(args.urdf), load_meshes=not args.no_meshes,
                              build_scene_graph=True)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")
    server.scene.add_frame("/base", axes_length=0.25, axes_radius=0.006)

    # ViserUrdf mirrors the kinematic tree as nested scene nodes under
    # /robot/visual/..., so anything parented to a link node inherits that
    # link's exact transform. Drawing markers at separately-computed world
    # coordinates instead only LOOKS aligned - it is a second copy of the
    # kinematics that can silently disagree with what is rendered.
    parent_of = {j.child: j.parent for j in urdf.joint_map.values()}

    def node_path(link: str) -> str:
        chain, cur = [], link
        while cur in parent_of and cur != urdf.base_link:
            chain.append(cur)
            cur = parent_of[cur]
        return "/robot/visual/" + "/".join(reversed(chain))

    all_names = list(viser_urdf.get_actuated_joint_names())
    lim = dict(zip(all_names, viser_urdf.get_actuated_joint_limits().values()))
    names = [n for n in all_names
             if SKIP not in n and any(t in n for t in POSABLE)]
    print(f"[bench] {len(names)} posable joints of {len(all_names)} actuated "
          f"({len(all_names) - len(names)} wheel/finger joints rendered but "
          "not posed)")

    def cfg_of(v: dict[str, float]) -> np.ndarray:
        """Slider dict -> the full actuated vector yourdfpy expects."""
        return np.array([v.get(n, 0.0) for n in all_names])

    def fk(v: dict[str, float]) -> dict[str, np.ndarray]:
        """4x4 poses of both EEs and the chest, in the URDF base frame."""
        urdf.update_cfg(cfg_of(v))
        out = {}
        for key, link in {**EE_LINKS, "ref": REF_LINK}.items():
            try:
                out[key] = np.array(urdf.get_transform(link, urdf.base_link))
            except Exception as e:                      # missing link in a variant
                print(f"[bench] FK for {link} failed: {e!r}")
        return out

    def load_poses() -> dict:
        if args.poses.exists():
            try:
                return json.loads(args.poses.read_text())
            except json.JSONDecodeError as e:
                print(f"[bench] {args.poses} unreadable ({e}) - starting fresh")
        return {}

    saved_poses = load_poses()
    sweep: dict = {"joint": None, "t0": 0.0}
    dirty = {"v": True}

    action_choices = (SESSION_PM_LABELS + STATIC_LABELS
                      + HEAD_WAIST_LABELS + [FREEFORM[0]])
    with server.gui.add_folder("动作 / action"):
        g_action = server.gui.add_dropdown("这是哪个动作", action_choices,
                                           initial_value=action_choices[0])
        b_save_pose = server.gui.add_button("保存为该动作的机器人姿态")
        b_load_pose = server.gui.add_button("载入该动作已存姿态")
        g_pose_state = server.gui.add_text("已存", "-")
    with server.gui.add_folder("preset 预设"):
        b_home = server.gui.add_button("home (tuck 起始姿态)")
        b_zero = server.gui.add_button("zero (关节全归零)")
        g_mirror = server.gui.add_checkbox("镜像:调右臂,左臂跟随", False)
    with server.gui.add_folder("sweep 单关节扫描"):
        g_joint = server.gui.add_dropdown("joint", names, initial_value=names[0])
        g_span = server.gui.add_slider("range fraction", 0.1, 1.0, 0.05, 0.6)
        g_period = server.gui.add_slider("period [s]", 2.0, 20.0, 0.5, 8.0)
        b_sweep = server.gui.add_button("start / stop")
        g_sweeping = server.gui.add_text("state", "idle")
    with server.gui.add_folder("readout 基座系 (x前 y左 z上)"):
        g_ee_r = server.gui.add_text("R_ee", "-")
        g_ee_l = server.gui.add_text("L_ee", "-")
        g_rel_r = server.gui.add_text("R_ee 相对胸(映射口径)", "-")
        g_rel_l = server.gui.add_text("L_ee 相对胸(映射口径)", "-")
    with server.gui.add_folder("头 / 腰 读数"):
        g_gaze = server.gui.add_text("头看哪 (方位/仰角)", "-")
        g_gaze_c = server.gui.add_text("头相对胸", "-")
        g_lean = server.gui.add_text("胸前倾角", "-")
        g_chestp = server.gui.add_text("胸位置 (高/前)", "-")
        server.gui.add_markdown(
            "**头**:三关节全 0 时是**低头看地 55°**(躯干工作位前倾),不是平视;"
            "`head_j1≈−55°` 才抬平。抬平后 `head_j2` 是干净的左右转头"
            "(方位 1:1、仰角零耦合)。`head_j1` 与 `head_j3` **同轴反号、冗余**,"
            "建议只用 j1。\n\n"
            "**腰**:`torso_j1/j2/j3` **全部绕同一根轴(base ±y)**——Vega 没有转腰、"
            "没有侧弯,整个躯干是矢状面内的三连杆。j1/j2 管胸的高度与前后,"
            "j3 只转胸不移动它。所以\"腰只有弯曲一个自由度\"是对的,但它由 3 个关节合成。")
    with server.gui.add_folder("链路标注"):
        g_chain = server.gui.add_checkbox("显示手臂链路 + EE", True)
        g_names = server.gui.add_checkbox("显示 link 名字", True)
        g_axes = server.gui.add_checkbox("显示坐标轴 (红=x 绿=y 蓝=z)", True)
        g_axlen = server.gui.add_slider("轴长 [m]", 0.02, 0.25, 0.01, 0.08)
        g_only_ee = server.gui.add_checkbox("只画 l8 和 EE(看那 90°)", True)
    with server.gui.add_folder("note 观察记录"):
        g_note = server.gui.add_text("这个关节做了什么?", "")
        b_rec = server.gui.add_button("record")
        g_count = server.gui.add_text("saved", "0")

    sliders: dict[str, viser.GuiInputHandle] = {}

    def stop_sweep() -> None:
        if sweep["joint"] is not None:
            sweep["joint"] = None
            g_sweeping.value = "idle"

    def mirror_to_left(rname: str) -> None:
        """Mirror a right-arm joint onto the left. j1/j2/j5 flip sign - the
        left arm's URDF limits are the right arm's negated - the rest copy."""
        lname = "L_arm_j" + rname.split("_j")[1]
        if lname not in sliders:
            return
        v = sliders[rname].value
        if rname.split("_j")[1] in MIRROR_FLIP:
            v = -v
        sliders[lname].value = float(np.clip(v, *lim[lname]))

    def add_slider(n: str) -> None:
        lo, hi = float(lim[n][0]), float(lim[n][1])
        s = server.gui.add_slider(n, lo, hi, 0.005,
                                  float(np.clip(HOME.get(n, 0.0), lo, hi)))
        sliders[n] = s

        @s.on_update
        def _(_, _n=n) -> None:
            if sweep["joint"] == _n:        # you took over - stop fighting you
                stop_sweep()
            if g_mirror.value and _n.startswith("R_arm"):
                mirror_to_left(_n)
            dirty["v"] = True

    for title, pred in GROUPS:
        members = [n for n in names if pred(n)]
        if members:
            with server.gui.add_folder(title):
                for n in members:
                    add_slider(n)
    for n in [n for n in names if n not in sliders]:
        add_slider(n)

    def vals() -> dict[str, float]:
        return {n: float(s.value) for n, s in sliders.items()}

    def _quat(T: np.ndarray) -> list[float]:
        return [float(v) for v in R.from_matrix(T[:3, :3]).as_quat()]   # xyzw

    def snapshot() -> dict:
        """Everything needed to use this pose as mapping ground truth.

        The vector that matters is chest->EE expressed in BASE AXES (x fwd,
        y left, z up) - that is the frame --map waist-abs commands in
        (waist_bind_base = chest position + a vertical offset, then the human's
        waist->wrist vector is added to it in base axes). The chest LINK frame
        is not that frame: arm_center is rotated 55 deg about y relative to the
        base (a dexmate URDF convention), so a chest-frame vector would be a
        different quantity and would silently corrupt any fit made from it.
        """
        v = vals()
        T = fk(v)
        rel_base, rel_chest_frame = {}, {}
        if "ref" in T:
            c = T["ref"][:3, 3]
            inv = np.linalg.inv(T["ref"])
            for h in EE_LINKS:
                if h in T:
                    rel_base[h] = (T[h][:3, 3] - c).tolist()
                    rel_chest_frame[h] = (inv @ T[h])[:3, 3].tolist()
        return {
            "joints": v,
            "ee_base": {h: T[h][:3, 3].tolist() for h in EE_LINKS if h in T},
            "ee_quat_base_xyzw": {h: _quat(T[h]) for h in EE_LINKS if h in T},
            "ee_minus_chest_base_axes": rel_base,      # <- the mapping's frame
            "ee_in_chest_link_frame": rel_chest_frame,  # kept for completeness
            "chest_base": T["ref"][:3, 3].tolist() if "ref" in T else None,
        }

    def refresh_pose_state() -> None:
        g_pose_state.value = (", ".join(sorted(saved_poses)) if saved_poses
                              else "(还没存过)")

    refresh_pose_state()

    @b_home.on_click
    def _(_) -> None:
        stop_sweep()
        for n, s in sliders.items():
            s.value = float(np.clip(HOME.get(n, 0.0), *lim[n]))
        dirty["v"] = True

    @b_zero.on_click
    def _(_) -> None:
        stop_sweep()
        for n, s in sliders.items():
            s.value = float(np.clip(0.0, *lim[n]))
        dirty["v"] = True

    @b_sweep.on_click
    def _(_) -> None:
        if sweep["joint"] is not None:
            stop_sweep()
        else:
            sweep["joint"], sweep["t0"] = g_joint.value, time.time()
            g_sweeping.value = f"sweeping {g_joint.value}"

    @b_save_pose.on_click
    def _(_) -> None:
        stop_sweep()
        slug = slug_of(g_action.value)
        saved_poses[slug] = {"label": g_action.value,
                             "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "note": g_note.value, **snapshot()}
        args.poses.write_text(json.dumps(saved_poses, ensure_ascii=False, indent=2))
        refresh_pose_state()
        print(f"[pose] saved '{slug}' ({g_action.value}) -> {args.poses}")

    @b_load_pose.on_click
    def _(_) -> None:
        stop_sweep()
        slug = slug_of(g_action.value)
        rec = saved_poses.get(slug)
        if rec is None:
            g_pose_state.value = f"'{slug}' 还没存过"
            return
        for n, v in rec["joints"].items():
            if n in sliders:
                sliders[n].value = float(np.clip(v, *lim[n]))
        dirty["v"] = True
        refresh_pose_state()
        print(f"[pose] loaded '{slug}'")

    n_saved = 0

    @b_rec.on_click
    def _(_) -> None:
        nonlocal n_saved
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": g_note.value,
               "action": slug_of(g_action.value), "action_label": g_action.value,
               "focus": g_joint.value, **snapshot()}
        with open(args.notes, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_saved += 1
        g_count.value = f"{n_saved}  ->  {args.notes}"
        print(f"[rec] {rec['time']}  action={rec['action']}  "
              f"focus={rec['focus']}  {rec['note']!r}")
        g_note.value = ""

    def fmt(p) -> str:
        return "-" if p is None else f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"

    print(f"[bench] open http://localhost:{args.port}")
    while True:
        if sweep["joint"] is not None:
            n = sweep["joint"]
            lo, hi = lim[n]
            mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo) * g_span.value
            u = (time.time() - sweep["t0"]) / max(g_period.value, 0.1)
            sliders[n].value = float(np.clip(mid + amp * math.sin(2 * math.pi * u),
                                             lo, hi))
            if g_mirror.value and n.startswith("R_arm"):
                mirror_to_left(n)
            dirty["v"] = True

        if dirty["v"]:
            dirty["v"] = False
            v = vals()
            viser_urdf.update_cfg(cfg_of(v))
            T = fk(v)
            if g_chain.value:
                for hand, links in CHAIN.items():
                    for ln in links:
                        if ln not in parent_of:
                            continue
                        base = node_path(ln)
                        ee = ln.endswith("_ee")
                        # child of the link node: position (0,0,0) IS the link
                        server.scene.add_icosphere(
                            f"{base}/mark", radius=0.028 if ee else 0.015,
                            color=(255, 40, 40) if ee else CHAIN_COL[hand],
                            position=(0.0, 0.0, 0.0))
                        if g_names.value:
                            server.scene.add_label(
                                f"{base}/tag",
                                f"{ln}  <<< 法兰/EE" if ee else ln,
                                position=(0.0, 0.0, 0.03))
                        if g_axes.value and (
                                not g_only_ee.value
                                or ln.endswith(("_l8", "_ee"))):
                            server.scene.add_frame(
                                f"{base}/ax", position=(0.0, 0.0, 0.0),
                                wxyz=(1.0, 0.0, 0.0, 0.0),
                                axes_length=g_axlen.value * (1.3 if ee else 1.0),
                                axes_radius=g_axlen.value * 0.05)

            g_ee_r.value = fmt(T["right"][:3, 3] if "right" in T else None)
            g_ee_l.value = fmt(T["left"][:3, 3] if "left" in T else None)
            if "ref" in T:
                # base axes, not the chest LINK frame - see snapshot()
                c = T["ref"][:3, 3]
                g_rel_r.value = fmt(T["right"][:3, 3] - c if "right" in T else None)
                g_rel_l.value = fmt(T["left"][:3, 3] - c if "left" in T else None)

            # head / waist, in words rather than joint angles: the point of
            # authoring a pose is what it LOOKS like, and a joint angle on a
            # 3-link sagittal column does not say that.
            try:
                Th = np.array(urdf.get_transform(HEAD_TIP, urdf.base_link))
                g = Th[:3, 0]                      # gaze, verified by FK sweep
                g_gaze.value = (f"方位 {np.degrees(np.arctan2(g[1], g[0])):+.0f}°  "
                                f"仰角 {np.degrees(np.arcsin(np.clip(g[2], -1, 1))):+.0f}°")
                if "ref" in T:
                    gc = T["ref"][:3, :3].T @ g
                    g_gaze_c.value = (
                        f"方位 {np.degrees(np.arctan2(gc[1], gc[0])):+.0f}°  "
                        f"仰角 {np.degrees(np.arcsin(np.clip(gc[2], -1, 1))):+.0f}°")
                    up = T["ref"][:3, 2]           # chest up axis, base frame
                    g_lean.value = (
                        f"{np.degrees(np.arccos(np.clip(up[2], -1, 1))):.0f}°  "
                        f"(0=直立)")
                    g_chestp.value = (f"高 {T['ref'][2, 3]:.3f} m  "
                                      f"前 {T['ref'][0, 3]:+.3f} m")
            except Exception:
                pass
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    raise SystemExit(main())
