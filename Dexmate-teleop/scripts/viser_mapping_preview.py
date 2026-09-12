#!/usr/bin/env python3
"""Offline mapping preview: recorded operator -> mapping -> IK -> robot, in Viser.

⚠ DO NOT USE THIS TO JUDGE A LIVE RUN. Use scripts/viser_isaac_mirror.py.

This bench re-derives the mapping and re-solves the IK, so it is a SECOND
implementation, and on 2026-08-02 it was measured landing the right arm on a
different IK branch than the consumer on the identical clip and parameters:

    R_arm_j1 median   consumer +1.7 deg      this bench +176.0 deg (on its limit)

Neither is wrong. A 7-DoF arm on a 6-DoF wrist target has a free dimension, and
which way the shoulder swings round is decided by whatever the target happened
to be when the solve first started moving - the consumer holds still until the
arm clutch closes, this bench starts on frame 0. Raising the posture cost to
pin the branch was measured and does not work: it still leaves 74 deg of
spread at posture_cost 5.0, by which point the wrist residual is 585 mm
(sim/dev_branch_determinism.py).

So this file is for OFFLINE exploration - laws, scales, reachability, palm
orientation - where a consistent branch within one session is enough. The
moment you want to know what the robot will actually do, mirror it.

Everything the live rig does to a tracker sample, minus the hardware and the
physics: replay a capture, apply a mapping law, solve the same Pink IK the
consumer solves, and pose the URDF in the browser next to the operator's whole
arm chain. One process, no Isaac, no GPU, no headset.

Two mapping laws, switchable live, because the first one has a structural flaw
the second fixes:

  waist-abs    EE = bind + s * (wrist - waist), bind = chest - 0.25 + h.
               What the consumer ships today. Scaling from the waist scales the
               waist->SHOULDER offset too, which is body geometry, not arm. The
               bind offset and the scaled vector then both account for the same
               drop: with h=-0.30 and s=1.5 the arms-down target lands 1.15 m
               below the chest, needing a 1.17 m arm on a 0.771 m robot.

  shoulder-abs EE = robot_shoulder + k * (wrist - human_shoulder),
               k = robot_arm / human_arm. Scaling applies to the arm vector
               alone, so torso proportions never enter. Arms-down maps to
               exactly 0.771 m below the robot's shoulder - reachable by
               construction, which is the point: the operator and the robot do
               not share body proportions (robot shoulder-offset/arm = 0.22,
               human ~0.35), so no single scale about one origin can fit both.

Human shoulders come from the body model (SMPL joints 16/17) when that channel
is live; on raw-channel captures it is not, so they are sliders. Their defaults
are not guesses: a straight arm puts the wrist on a sphere about the shoulder,
so fitting centre and radius over the three straight-arm poses (front / lateral
/ arms-down, both hands, mirrored) recovers both. That fit closes to +-1..6 mm
and gives shoulder = (-0.089, +-0.022, +0.211) from the stream's WAIST with a
0.671 m reach. Note the stream's "WAIST" is SMPL joint 9 = Spine3, an upper-
chest joint, not the anatomical waist - which is why the shoulder sits only
0.21 m above it. The fitted centre is an effective centre of rotation near the
spine rather than the anatomical shoulder; that is fine, it is what the data
actually pivots about.

Faithful: the mapping law, the Pink solver, its costs and joint limits - arm
posture and reachability shown here are real. Absent: actuator dynamics, the
consumer's one-euro target filter, its glitch gate and forearm bridge. Measured
against Isaac on the same take, EE-relative-to-chest agrees to ~25-35 mm median
(a standing offset, not misalignment). Good enough to judge posture and
workspace; do not quote its numbers as tracking error.

Usage:
    .venv/bin/python scripts/viser_mapping_preview.py logs/playlist_20260730.msgpack
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import msgpack
import numpy as np
import zmq
import viser
import viser.extras
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
from magicdexmate.pico.xr_pose import (  # noqa: E402
    OneEuroFilter, QuatOneEuroFilter, mat_to_pos_quat_wxyz, process_xr_pose)
from pink_vega_ik import ELBOW_FRAME, EE_FRAME, PinkVegaIK  # noqa: E402
from magicdexmate.palm_fix import (  # noqa: E402
    FINGERS_IN_EE, PALM_FIX, PALM_IN_EE, relax_roll)
from magicdexmate.bimanual_symmetry import (  # noqa: E402
    mirror_distance, symmetrise)
from magicdexmate.head_waist_map import (  # noqa: E402
    head_gaze, head_solve, waist_joints, NEUTRAL_ELEVATION)
from magicdexmate.pico.xr_pose import pitch_delta, pose7_to_world_rot  # noqa: E402

IK_URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
_SHARPA = pathlib.Path(__file__).resolve().parents[1] / "assets/vega_1_sharpa.urdf"
# Render the hands, solve on the bare arm: PinkVegaIK locks non-arm joints from
# a hardcoded list, so the hands' 44 unlisted joints would stay free and change
# the solve. Arm joint names are identical in both, so the answer transfers.
DEFAULT_URDF = _SHARPA if _SHARPA.exists() else IK_URDF
WRIST = {"right": "PC2310MLL4151712G", "left": "PC2310MLL4151662G"}
WRIST_FALLBACK = {"right": "RWRIST", "left": "LWRIST"}
ELBOW = {"right": "RELBOW", "left": "LELBOW"}
WAIST = "WAIST"
CHEST_LINK = "arm_center"
SMPL_SHOULDER = {"left": 16, "right": 17}
# The torso is LOCKED during teleop, but not at zero: vega_scene parks it here.
# Evaluating the URDF at zero instead moves the chest 417 mm and the elbow
# 154 mm, so every robot-side marker lands somewhere the robot never is - and
# the rendered robot stands in a posture it never adopts. (Pink's own model
# does lock the torso at zero, and that is correct: it solves chest-RELATIVE,
# so the absolute torso angle cancels. Only the drawing needs the real one.)
# Imported, not re-declared: this constant lived in five files and every
# copy said 55 deg of forward lean while all thirteen authored poses were
# at 10.3 deg. See magicdexmate/head_waist_map.py.
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402,F811
COL = {"right": (255, 170, 60), "left": (80, 200, 255),
       "waist": (170, 255, 120), "head": (240, 240, 240),
       "bone": (150, 150, 150), "robot": (200, 200, 200)}
HUMAN_OFFSET = np.array([0.0, 1.4, 0.0])     # draw the operator beside the robot


def _frame_of(m: dict, wrist_keys: dict[str, str]) -> dict:
    trk = m.get("trackers") or {}
    d = {r: np.asarray(trk[k], float) for r, k in wrist_keys.items()}
    d["waist"] = np.asarray(trk[WAIST], float)
    for r, k in ELBOW.items():
        if k in trk:
            d[f"elbow_{r}"] = np.asarray(trk[k], float)
    if m.get("head") is not None:
        d["head"] = np.asarray(m["head"], float)
    if m.get("body24"):
        d["body24"] = np.asarray(m["body24"], float)
    return d


def load(path: pathlib.Path, wrist_keys: dict[str, str]):
    out = []
    for m in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = m.get("trackers") or {}
        if WAIST not in trk or not all(k in trk for k in wrist_keys.values()):
            continue
        out.append((m.get("t_us", 0) / 1e6, _frame_of(m, wrist_keys)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=pathlib.Path, nargs="?",
                    help="recording to replay; omit with --live")
    ap.add_argument("--live", metavar="ADDR", nargs="?",
                    const="tcp://127.0.0.1:5581", default=None,
                    help="subscribe to a running producer instead of replaying. "
                         "Isaac renders the USD, which has no hands, so palm "
                         "orientation cannot be judged there - here it can, and "
                         "without a GPU. Kinematics only: no physics, no "
                         "actuator lag.")
    ap.add_argument("--law", choices=["chest-rel", "shoulder-rel",
                                      "shoulder-abs", "waist-abs"], default=None,
                    help="start on this law instead of the default. A law that "
                         "is only reachable by clicking is a law that never got "
                         "smoke-tested - this repo has shipped six such.")
    ap.add_argument("--posture-cost", type=float, default=None,
                    help="Pink posture-task weight. The default 1e-2 against a "
                         "position cost of 8.0 leaves the 7-DoF arm's redundant "
                         "dimension essentially unconstrained, so which elbow "
                         "branch the solve lands in is decided by history, not "
                         "by the target - which is why this bench and Isaac "
                         "disagree by 257 deg on R_arm_j1.")
    ap.add_argument("--joint-csv", default=None,
                    help="dump the solved joint angles [rad] per frame, in the "
                         "same column order as the consumer's --joint-csv, so "
                         "sim/dev_compare_isaac_viser.py can diff them. This "
                         "bench is where mapping choices get made by eye; if "
                         "it is not showing the motion the consumer sends, "
                         "every choice made on it is worthless.")
    ap.add_argument("--control-mode",
                    choices=["arms", "arms+head", "arms+head+waist"],
                    default=None, help="start in this mode. The bench defaults "
                                       "to arms+head+waist, which DRIVES the "
                                       "torso - an offline harness that locks "
                                       "the torso is then measuring a different "
                                       "thing, which is how the same law read "
                                       "17 mm in one place and 109 mm here.")
    ap.add_argument("--pos-scale", type=float, default=None,
                    help="start with this pos-scale instead of the slider default")
    ap.add_argument("--no-sym", action="store_true",
                    help="start with the bimanual symmetriser off")
    ap.add_argument("--headless-seconds", type=float, default=0.0,
                    help="run the update loop for N seconds and exit; smoke "
                         "test for the branches the GUI would have to be "
                         "clicked to reach")
    ap.add_argument("--urdf", type=pathlib.Path, default=DEFAULT_URDF,
                    help="model to DRAW (hands included by default)")
    ap.add_argument("--ik-urdf", type=pathlib.Path, default=IK_URDF,
                    help="model to SOLVE on - keep this the bare arm")
    ap.add_argument("--wrist-channel", choices=["auto", "model", "raw"],
                    default="auto",
                    help="auto (default) follows scripts/trackers.env, i.e. "
                         "whatever the consumer is mapping. PALM_FIX is only "
                         "valid on the model channel - see magicdexmate/palm_fix.py")
    ap.add_argument("--port", type=int, default=8083)
    args = ap.parse_args()

    live_sock = None
    if args.live:
        ctx = zmq.Context.instance()
        live_sock = ctx.socket(zmq.SUB)
        live_sock.setsockopt(zmq.CONFLATE, 1)
        live_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        live_sock.connect(args.live)
        # Follow scripts/trackers.env, the same file run_teleop.sh sources, so
        # the preview can never show a different wrist channel than the one the
        # consumer is actually mapping. Guessing here (raw-first) is what let
        # the constant and the channel drift apart in the first place.
        env_keys = None
        env_path = pathlib.Path(__file__).resolve().parents[1] / "scripts/trackers.env"
        if args.wrist_channel == "auto" and env_path.exists():
            env = dict(
                ln.split("#")[0].strip().split("=", 1)
                for ln in env_path.read_text().splitlines()
                if "=" in ln.split("#")[0])
            if "TRACKER_LEFT" in env and "TRACKER_RIGHT" in env:
                env_keys = {"right": env["TRACKER_RIGHT"].strip(),
                            "left": env["TRACKER_LEFT"].strip()}
                print(f"[preview] wrist channel from trackers.env: {env_keys}")
        elif args.wrist_channel == "raw":
            env_keys = WRIST
        elif args.wrist_channel == "model":
            env_keys = WRIST_FALLBACK
        order = [env_keys] if env_keys else [WRIST, WRIST_FALLBACK]
        print(f"[preview] LIVE from {args.live} - waiting for a frame ...")
        frames = []
        while not frames:
            try:
                m = msgpack.unpackb(live_sock.recv(zmq.NOBLOCK), raw=False)
            except zmq.Again:
                time.sleep(0.05)
                continue
            trk = m.get("trackers") or {}
            for k in order:
                if WAIST in trk and all(v in trk for v in k.values()):
                    keys = k
                    frames = [(0.0, _frame_of(m, k))]
                    break
        print(f"[preview] connected, wrist channel = "
              f"{'raw' if keys is WRIST else 'model'}")
    else:
        if args.capture is None:
            raise SystemExit("give a capture file, or use --live")
        keys = WRIST
        frames = load(args.capture, keys)
        if not frames:
            keys = WRIST_FALLBACK
            frames = load(args.capture, keys)
        if not frames:
            raise SystemExit(f"{args.capture}: no frames with wrists + {WAIST}")
    t0 = frames[0][0]
    frames = [(t - t0, d) for t, d in frames]
    times = np.array([t for t, _ in frames])
    chan = "raw" if keys is WRIST else "model"

    # are the body-model shoulders usable, or must they be parameters?
    def channel_live(key: str) -> bool:
        """A model-channel entry that never changes is a dead estimator, not a
        still operator. Drawing it looks like data and is not - the elbows in
        the 2026-07-29 raw batch move 4 times in 3723 frames."""
        vals = {tuple(d[key][:3]) for _, d in frames if key in d}
        return len(vals) > 10

    live_key = {k: channel_live(k) for k in
                ["right", "left", "waist", "elbow_right", "elbow_left", "head"]
                if any(k in d for _, d in frames)}
    dead = [k for k, v in live_key.items() if not v]
    if dead:
        print(f"[preview] frozen channels, not drawn: {dead}")
    b24 = [d["body24"] for _, d in frames if "body24" in d]
    live_b24 = len({tuple(map(tuple, b)) for b in b24[:400]}) > 10 if b24 else False
    print(f"[preview] {len(frames)} frames, {times[-1]:.1f}s, wrist channel = {chan}")
    print(f"[preview] body24 shoulders: {'live' if live_b24 else 'frozen -> sliders'}")

    print("[preview] building Pink IK ...")
    # Two solvers, built once: switching elbow_cost means rebuilding the task
    # set, and doing that per frame would reset the configuration the
    # differential solve depends on. Keeping both warm lets the slider be live.
    _pk = {} if args.posture_cost is None else {"posture_cost": args.posture_cost}
    ik = PinkVegaIK(urdf_path=str(args.ik_urdf), **_pk)
    ik_elbow = PinkVegaIK(urdf_path=str(args.ik_urdf), elbow_cost=0.25, **_pk)
    urdf = yourdfpy.URDF.load(str(args.urdf), load_meshes=True, build_scene_graph=True)
    all_joints = [j for j, jo in urdf.joint_map.items() if jo.type != "fixed"]
    base_cfg = np.array([TORSO_HOME.get(j, 0.0) for j in all_joints])
    urdf.update_cfg(base_cfg)

    def link(name):
        return np.array(urdf.get_transform(name, urdf.base_link))

    chest_T = link(CHEST_LINK)
    chest_pos = chest_T[:3, 3]
    cq = R.from_matrix(chest_T[:3, :3]).as_quat()
    chest_wxyz = np.array([cq[3], *cq[:3]])
    rob_sh = {h: link(f"vega_1_{h[0].upper()}_arm_l1")[:3, 3] for h in ("right", "left")}
    rob_arm = float(np.linalg.norm(link("R_ee")[:3, 3] - rob_sh["right"]))
    print(f"[preview] robot shoulders {np.round(rob_sh['right'] - chest_pos, 3).tolist()}"
          f" (rel chest), arm {rob_arm:.3f} m")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    vurdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")

    # Reference robot: the poses the operator authored by hand in the joint
    # bench. Drawn beside the mapped robot so "did the mapping reproduce my
    # pose" is a look, not an argument.
    #
    # It is ARM POSTURE ground truth only. Checked 2026-08-01: L/R_arm_j5, j6
    # and j7 are 0.000 in all six poses - the wrist three-axis was never moved,
    # so every pose carries the same default hand. j1-j4 span 1.2-3.1 rad and
    # the torso sits at a deliberate non-default working pose, so the arm and
    # the EE position are real. Do not read palm direction off this ghost.
    ref_poses, ref_urdf, ref_vurdf = {}, None, None
    ref_path = pathlib.Path(__file__).resolve().parents[1] / "logs/robot_action_poses.json"
    if ref_path.exists():
        import json
        ref_poses = json.load(open(ref_path))
        ref_urdf = yourdfpy.URDF.load(str(args.urdf), load_meshes=True,
                                      build_scene_graph=True)
        ref_vurdf = viser.extras.ViserUrdf(server, ref_urdf,
                                           root_node_name="/robot_ref")
        ref_root = server.scene.add_frame("/robot_ref", position=(0.0, -1.4, 0.0),
                                          show_axes=False, visible=False)
        print(f"[preview] reference poses: {len(ref_poses)} "
              f"({', '.join(ref_poses)}) - arm posture only, wrists are default")

    with server.gui.add_folder("playback"):
        g_play = server.gui.add_checkbox("play", True)
        g_speed = server.gui.add_slider("speed", 0.1, 3.0, 0.1, 1.0)
        g_t = server.gui.add_slider("t [s]", 0.0, float(times[-1]), 0.02, 0.0)
    with server.gui.add_folder("映射律"):
        g_law = server.gui.add_dropdown(
            "law", ["chest-rel 胸相对(解耦,2026-08-02)",
                    "shoulder-rel 肩到肩 + 左右相对锁定",
                    "shoulder-abs 肩到肩", "waist-abs 腰绑定"],
            # waist-abs wins on the operator's own priorities once measured:
            # relative wrist position is already 0 mm (both hands share one
            # bind point and one 1:1 scale) and orientation never breaches
            # the 10 deg limit, where both shoulder laws breach it on 35-43%
            # of frames. shoulder-rel was built for a problem waist-abs did
            # not have.
            # chest-rel added 2026-08-02 on the operator's proposal: read the
            # hand relative to the CHEST, in the chest frame, position AND
            # orientation. Measured on playlist_all13 at the authored upright
            # torso it takes the wrist residual from 277/218 mm to 17/20 mm and
            # the finger heading to 5 deg, and - the structural part - it gives
            # bit-identical numbers at the old and new TORSO_HOME, i.e. the arm
            # mapping stops depending on the torso at all. shoulder-abs /
            # shoulder-rel are NOT this: they anchored position at the shoulder
            # but left orientation absolute, which is the half that mattered.
            initial_value="waist-abs 腰绑定")
        if args.law:
            for _opt in g_law.options:
                if _opt.startswith(args.law):
                    g_law.value = _opt
                    break
        g_yaw = server.gui.add_slider("偏航修正 [deg]", -30.0, 30.0, 0.5, 0.0)
    with server.gui.add_folder("肩到肩 参数"):
        g_sh_up = server.gui.add_slider("人肩高于参考点 [m]", -0.10, 0.60, 0.005, 0.211)
        g_sh_out = server.gui.add_slider("人肩半宽 [m]", -0.05, 0.35, 0.005, 0.022)
        g_sh_back = server.gui.add_slider("人肩前后 [m]", -0.30, 0.20, 0.005, -0.089)
        g_arm = server.gui.add_slider("人臂长 [m]", 0.40, 0.90, 0.005, 0.671)
        g_k = server.gui.add_text("k = 机器人臂长/人臂长", "-")
    with server.gui.add_folder("腰绑定 参数"):
        g_bind = server.gui.add_slider("waist-bind-up [m]", -0.60, 0.60, 0.01, 0.20)
        # 1.25: 1:1 lands the hand 274 mm short on arms-down (it copies the
        # OPERATOR's 0.505 m reach onto a 0.771 m arm), but the 1.40 the
        # authored poses imply puts lateral-horizontal and raise-high OUT OF
        # REACH - and raise-high only on the right, which splits the two
        # arms' solutions. 1.25 is the largest scale that stays inside.
        g_scale = server.gui.add_slider("pos-scale", 0.6, 2.0, 0.01, 1.35)
    with server.gui.add_folder("显示"):
        g_human = server.gui.add_checkbox("操作者链", True)
        g_robot = server.gui.add_checkbox("机器人链标记", True)
        g_ax = server.gui.add_checkbox("朝向坐标轴", True)
        g_axlen = server.gui.add_slider("轴长 [m]", 0.05, 0.3, 0.01, 0.12)
    with server.gui.add_folder("肘(冗余自由度)"):
        g_elbow = server.gui.add_checkbox(
            "开启肘任务 (--elbow-weight 0.25) ⚠ 目前是坏的", False)
        g_asym = server.gui.add_text("机器人左右肘差(镜像)", "-")
        g_hasym = server.gui.add_text("你自己左右肘差(镜像)", "-")
        server.gui.add_markdown(
            "7 自由度手臂对 6 自由度腕目标有 **1 维冗余**。肘任务关掉时(默认)"
            "IK 只保证腕,肘由姿态任务自由落位——左右腕目标略有差别,两条臂就可能"
            "落进**不同的零空间解**,看起来就是「腕一样、肘差很多」。\n\n"
            "⚠ **肘任务当前不可用**,勾上只会看到解跑飞:实测腕残差从 30mm 崩到 "
            "**475mm**(0.25 和 1.0 权重都一样)。原因:肘目标是把**人的肘相对腰**"
            "直接加到机器人绑定点上,而人机比例不同(肩偏置/臂长 0.35 vs 0.22),"
            "那个点常常在机器人肘根本到不了的地方。\n\n"
            "**正确解法应该是约束『回转角』而不是肘的位置** —— 肘绕肩-腕连线转到"
            "哪个方位,这个量是无量纲的,人机比例不同也成立。待实现。")
    with server.gui.add_folder("控制模式(与 consumer 的 --control-mode 同名)"):
        g_cmode = server.gui.add_dropdown(
            "mode", ["arms", "arms+head", "arms+head+waist"],
            initial_value=args.control_mode or "arms+head+waist")
        g_gaze = server.gui.add_text("机器人视线 方位/仰角", "-")
        g_lean = server.gui.add_text("胸前倾 / 人弯腰", "-")
        server.gui.add_markdown(
            "⚠ **这个下拉必须手动设成与 consumer 的 `--control-mode` 一致** —— "
            "预览台连着 producer,看不到 consumer 用了哪个模式。默认曾是 "
            "`arms+head+waist`,而实跑命令是 `arms+head`,于是预览台里躯干在动、"
            "Isaac 里没动,看起来像『腰没锁住』。\n\n"
            "头/腰用的是 consumer 里同一个 `head_solve` / `waist_joints`。\n\n"
            "**arms** 下头和躯干保持初始位;**arms+head** 头跟随;"
            "**arms+head+waist** 躯干也沿你摆的路径走,手臂绑定点跟随实时胸位。\n\n"
            "纯运动学:没有 PD、没有摆动、没有啮合斜坡——所以这里看的是"
            "**映射对不对**,不是机器人抖不抖。")
    with server.gui.add_folder("对照:你亲手摆的标准姿势"):
        g_ref = server.gui.add_dropdown(
            "标准姿势", ["(不显示)"] + sorted(ref_poses),
            initial_value="(不显示)")
        server.gui.add_markdown(
            "画在 **y = −1.4 m** 处(映射结果在原点,操作者在 +1.4)。\n\n"
            "⚠ **只是手臂姿态的真值,不是朝向真值** —— 实测这 6 个姿势的 "
            "`arm_j5/j6/j7` **全是 0.000**,腕三轴从没摆过,所以每个姿势的手"
            "都是同一个默认朝向。看手臂形状和 EE 位置,别看掌心。")
    with server.gui.add_folder("朝向"):
        g_ori = server.gui.add_dropdown(
            "朝向模式", ["palm (乘 PALM_FIX,=实跑绝对式)",
                         "raw (直接用腕朝向,预览台的旧行为)"],
            initial_value="palm (乘 PALM_FIX,=实跑绝对式)")
        g_palmarrow = server.gui.add_checkbox("机器人掌心法向箭头", True)
        server.gui.add_markdown(
            f"腕通道 = **{'裸序列号' if keys is WRIST else '模型通道'}**"
            + ("\n\n⚠ PALM_FIX 是在模型通道上拟合的,喂裸通道会差约 90°"
               if keys is WRIST else ""))
    with server.gui.add_folder("readout"):
        g_palmdir = server.gui.add_text("掌心法向 R/L", "-")
        g_rel = server.gui.add_text("①左右腕相对位置误差", "-")
        g_sym = server.gui.add_checkbox("操作者左右对称时,强制机器人左右对称",
                                        not args.no_sym)
        g_sym_state = server.gui.add_text("对称化状态", "-")
        g_res = server.gui.add_text("IK 残差 R/L", "-")
        g_need = server.gui.add_text("需要臂长 R/L", "-")
        g_verdict = server.gui.add_text("可达", "-")
    # 2026-08-01: the residual above was being computed and shown all along,
    # and the parameters chosen on this bench (--pos-scale 1.35,
    # --waist-bind-up 0.08) turned out to leave the hand 272 mm from its own
    # target. A number in a list nobody reads is not a warning, so: say it in
    # words, draw the gap in 3D, and report WHY (a joint on its limit has no
    # DoF left, which is what "the IK locks up" looks like from outside).
    with server.gui.add_folder("⚠ 目标跟上了吗(2026-08-01 新增)"):
        g_track = server.gui.add_text("跟随判定", "-")
        g_ori_err = server.gui.add_text("掌法向 / 手指指向 误差 R,L", "-")
        g_pinned = server.gui.add_text("顶在限位的关节", "-")
        if args.pos_scale is not None:
            g_scale.value = args.pos_scale
        g_gap = server.gui.add_checkbox("显示手与目标之间的偏差连线", True)
        g_rollfree = server.gui.add_checkbox("放开手指指向,仅保持掌心朝向", False)
        server.gui.add_markdown(
            "**红球是目标,机器人的手是实际去处。**两者分开 = 映射在这一维已经"
            "饱和,操作者再动手,机器人不会跟。\n\n"
            "「放开手指指向」把绕掌法向的滚转让给解算器:实测把位置残差从 "
            "272/192mm 压到 21/4mm,代价是手指可能横着(丢 65–85°)。"
            "掌心朝哪不受影响。")

    JOINT_DUMP = ([f"{_s}_arm_j{_i}" for _s in ("R", "L") for _i in range(1, 8)]
                  + [f"head_j{_i}" for _i in (1, 2, 3)]
                  + [f"torso_j{_i}" for _i in (1, 2, 3)])
    jc_f = jc_w = None
    if args.joint_csv:
        import csv as _csv
        jc_f = open(args.joint_csv, "w", newline="")
        jc_w = _csv.writer(jc_f)
        jc_w.writerow(["t"] + [f"cmd_{n}" for n in JOINT_DUMP])
        print(f"[joints] writing {args.joint_csv}")

    # The consumer's engage ramp and one-euro filter, reproduced here. Without
    # them this bench was landing the RIGHT arm on a completely different IK
    # branch than Isaac: R_arm_j1 median +176.0 deg (parked on its limit) here
    # versus +1.7 deg there, measured 2026-08-02 with --joint-csv on both. Pink
    # is differential, so jumping to the absolute mapped target on frame 0 - as
    # this bench used to - throws the solve into the wound branch and it never
    # comes back. The consumer never sees that discontinuity because it walks
    # the target in from the settled EE pose over 1.5 s.
    #
    # This is the whole point of the bench: mapping parameters get chosen here
    # by eye, so if it is not on the same branch as the consumer, every choice
    # made on it describes a robot that will not exist.
    RAMP_S = 1.5
    ramp = {"t0": None, "home": {}}
    pos_filt = {h: OneEuroFilter(1.0, 2.0) for h in ("right", "left")}
    quat_filt = {h: QuatOneEuroFilter(1.0, 2.0) for h in ("right", "left")}

    state = {"t0": time.time()}

    def rel_waist(d, key):
        """A stream pose expressed relative to the waist, yaw-compensated -
        exactly the frame the consumer maps in."""
        return mat_to_pos_quat_wxyz(process_xr_pose(d[key], d["waist"]))

    # the ACTUAL bound port: viser auto-increments when the requested one is
    # taken, and printing the requested one sent the operator to a stale replay
    # process squatting on 8083 while their live session was on 8085
    _port = getattr(server, "get_port", lambda: args.port)()
    print(f"[preview] open http://localhost:{_port}"
          + ("" if _port == args.port else
             f"   (requested {args.port}, it was taken)"))
    while True:
        if live_sock is not None:
            try:
                m = msgpack.unpackb(live_sock.recv(zmq.NOBLOCK), raw=False)
                trk = m.get("trackers") or {}
                if WAIST in trk and all(v in trk for v in keys.values()):
                    frames = [(0.0, _frame_of(m, keys))]
            except zmq.Again:
                pass
            d = frames[0][1]
        else:
            span = float(times[-1]) or 1.0
            if args.headless_seconds:
                # Advance ONE RECORDED FRAME per iteration, not by wall clock.
                # Driven by the clock this loop reaches ~8 Hz on a 60 Hz take,
                # i.e. it feeds a DIFFERENTIAL solver every 8th frame - and a
                # subsampled Pink reports failure-to-converge as residual. That
                # is the error this repo made three times on 2026-07-31; here it
                # made the same law read 152 mm on this bench and 17 mm offline.
                state["fi"] = state.get("fi", 0) + 1
                if state["fi"] >= len(times):
                    state["fi"] = len(times) - 1
                g_t.value = float(times[state["fi"]])
            elif g_play.value:
                g_t.value = float(
                    ((time.time() - state["t0"]) * g_speed.value) % span)
            _, d = frames[int(np.clip(np.searchsorted(times, g_t.value), 0,
                                      len(frames) - 1))]

        th = np.radians(g_yaw.value)
        cy, sy = np.cos(th), np.sin(th)

        def yawfix(p):
            return np.array([cy * p[0] - sy * p[1], sy * p[0] + cy * p[1], p[2]])

        # ---- head / waist ---------------------------------------------------
        # Done BEFORE the arm mapping because in arms+head+waist the chest
        # moves, and every arm quantity below - bind point, shoulders, the
        # chest-relative IK transfer - is derived from it. Deriving the arm
        # targets from a stale chest is exactly the bug that made the hands
        # drift forward when the operator bent over.
        cmode = g_cmode.value
        torso_cfg = dict(TORSO_HOME)
        lean_deg = 0.0
        if cmode == "arms+head+waist":
            if state.get("lean0") is None:
                state["lean0"] = pose7_to_world_rot(d["waist"])
            # NOT negated. pitch_delta returns POSITIVE when the operator
            # bends forward; negating it drove the chest toward upright, and
            # since the authored table starts AT upright (10.3 deg) the whole
            # range clamped away - bending 57 deg moved the chest 0.0 deg.
            # This is the "bending and standing feel inverted" report.
            lean_deg = np.degrees(pitch_delta(pose7_to_world_rot(d["waist"]),
                                              state["lean0"]))
            lean_deg = float(np.clip(lean_deg, -25.0, 25.0))
            torso_cfg = {k: np.degrees(v) * np.pi / 180.0
                         for k, v in waist_joints(lean_deg).items()}
        urdf.update_cfg(np.array([torso_cfg.get(j, 0.0) for j in all_joints]))
        chest_T = link(CHEST_LINK)
        chest_pos = chest_T[:3, 3]
        _cq = R.from_matrix(chest_T[:3, :3]).as_quat()
        chest_wxyz = np.array([_cq[3], *_cq[:3]])
        rob_sh = {h: link(f"vega_1_{h[0].upper()}_arm_l1")[:3, 3]
                  for h in ("right", "left")}
        head_cfg = {}
        if cmode in ("arms+head", "arms+head+waist") and "head" in d:
            hg = process_xr_pose(d["head"], d["waist"])[:3, 0]
            haz = float(np.degrees(np.arctan2(hg[1], hg[0])))
            hel = float(np.degrees(np.arcsin(np.clip(hg[2], -1.0, 1.0))))
            head_cfg, state["hseed"] = head_solve(haz, hel, chest_T[:3, :3],
                                                  seed=state.get("hseed"))
            # Judge saturation BY THE RESULT, not against HEAD_PITCH_TABLE.
            # head_solve defaults to the "j1" pitch path, whose envelope is
            # head_j1's +-85 deg; the table's -50.5..+29.1 deg belongs to the
            # "authored" path only. Measuring against the wrong envelope
            # reported clamping that was not happening and printed a target
            # angle that had been clipped for display but not by the solver.
            # Same misleading readout as the one fixed in the consumer on
            # 2026-08-03 - the envelope is path-specific, the result is not.
            _g = head_gaze(head_cfg["head_j1"], head_cfg["head_j2"],
                           head_cfg["head_j3"], chest_T[:3, :3])
            _az = float(np.degrees(np.arctan2(_g[1], _g[0])))
            _el = float(np.degrees(np.arcsin(np.clip(_g[2], -1.0, 1.0))))
            _want_el = hel + NEUTRAL_ELEVATION
            _miss = max(abs((_az - haz + 180.0) % 360.0 - 180.0),
                        abs(_el - _want_el))
            g_gaze.value = (
                f"要求 {haz:+.0f}° / {_want_el:+.0f}°"
                f"   实得 {_az:+.0f}° / {_el:+.0f}°"
                + (f"   偏差 {_miss:.0f}°,已饱和" if _miss > 1.0 else ""))
        else:
            g_gaze.value = "-  当前模式未启用头部"
        g_lean.value = (f"胸 {np.degrees(np.arccos(np.clip(chest_T[2, 2], -1, 1))):.0f}°"
                        f"  /  人 {lean_deg:+.0f}°"
                        + ("" if cmode == "arms+head+waist" else "   当前模式未启用腰部"))

        # ---- operator side, all relative to the waist -----------------------
        hum = {"waist": np.zeros(3)}
        for h in ("right", "left"):
            hum[h] = yawfix(rel_waist(d, h)[0])
            if f"elbow_{h}" in d and live_key.get(f"elbow_{h}", False):
                hum[f"elbow_{h}"] = yawfix(rel_waist(d, f"elbow_{h}")[0])
        if "head" in d and live_key.get("head", False):
            hum["head"] = yawfix(rel_waist(d, "head")[0])
        if live_b24 and "body24" in d:
            for h, idx in SMPL_SHOULDER.items():
                p7 = d["body24"][idx]
                hum[f"sh_{h}"] = yawfix(mat_to_pos_quat_wxyz(
                    process_xr_pose(p7, d["waist"]))[0])
        else:
            for h, s in (("right", -1.0), ("left", 1.0)):
                hum[f"sh_{h}"] = np.array([g_sh_back.value, s * g_sh_out.value,
                                           g_sh_up.value])

        # ---- mapping --------------------------------------------------------
        shoulder_law = g_law.value.startswith("shoulder")
        chest_law = g_law.value.startswith("chest-rel")
        h_org = h_R = None
        if chest_law:
            _sl, _sr = hum["sh_left"], hum["sh_right"]
            h_org = 0.5 * (_sl + _sr)
            _y = _sl - _sr
            _y = _y / max(np.linalg.norm(_y), 1e-9)
            _z = h_org / max(np.linalg.norm(h_org), 1e-9)   # waist -> shoulders
            _x = np.cross(_y, _z)
            _x = _x / max(np.linalg.norm(_x), 1e-9)
            h_R = np.column_stack([_x, _y, np.cross(_x, _y)])
        k = rob_arm / max(g_arm.value, 1e-3)
        g_k.value = f"{k:.3f}"
        bind = chest_pos + np.array([0.0, 0.0, -0.25 + g_bind.value])
        targets, want_q, need = {}, {}, {}
        palm_mode = g_ori.value.startswith("palm")
        for h in ("right", "left"):
            T = process_xr_pose(d[h], d["waist"])
            if palm_mode:
                # same line the consumer runs: the operator's palm pose IS the
                # robot's palm pose. Only valid on the channel PALM_FIX was
                # fitted on - see magicdexmate/palm_fix.py.
                M4 = np.eye(4)
                M4[:3, :3] = T[:3, :3] @ PALM_FIX[h]
                if chest_law:
                    # the orientation goes through the SAME frame as the
                    # position. Leaving it absolute here is precisely what
                    # made the two shoulder laws breach the 10 deg line.
                    M4[:3, :3] = chest_T[:3, :3] @ h_R.T @ M4[:3, :3]
                q = mat_to_pos_quat_wxyz(M4)[1]
            else:
                q = mat_to_pos_quat_wxyz(T)[1]
                if chest_law:
                    M4 = np.eye(4)
                    M4[:3, :3] = chest_T[:3, :3] @ h_R.T @ T[:3, :3]
                    q = mat_to_pos_quat_wxyz(M4)[1]
            want_q[h] = q
            if chest_law:
                targets[h] = chest_pos + chest_T[:3, :3] @ (
                    h_R.T @ (hum[h] - h_org)) * g_scale.value
            elif shoulder_law:
                targets[h] = rob_sh[h] + k * (hum[h] - hum[f"sh_{h}"])
            else:
                targets[h] = bind + g_scale.value * hum[h]
            need[h] = float(np.linalg.norm(targets[h] - rob_sh[h]))
        if g_law.value.startswith("shoulder-rel") and "sh_right" in hum:
            # Pin the LEFT-RIGHT wrist vector to the operator's own, 1:1 and
            # unscaled - two hands on one object stay that far apart no matter
            # whose arms they are. The correction is DIFFERENTIAL, so it never
            # moves the pair as a whole; common-mode drift is the thing the
            # operator said may be given away. Measured over six takes this
            # takes the relative error from 62 mm (426 worst) to 0, for 31 mm
            # of absolute drift.
            e = (hum["left"] - hum["right"]) - (targets["left"] - targets["right"])
            targets["left"] = targets["left"] + 0.5 * e
            targets["right"] = targets["right"] - 0.5 * e
        # operator symmetric -> robot symmetric, enforced in the TARGETS where
        # it is exact, not hoped for from the solver
        if g_sym.value:
            Rw = {h: R.from_quat([want_q[h][1], want_q[h][2], want_q[h][3],
                                  want_q[h][0]]).as_matrix()
                  for h in ("right", "left")}
            d0 = mirror_distance(targets["right"], targets["left"])
            pr, pl, Rr, Rl, w = symmetrise(targets["right"], targets["left"],
                                           Rw["right"], Rw["left"])
            targets["right"], targets["left"] = pr, pl
            for h, Rm in (("right", Rr), ("left", Rl)):
                q4 = R.from_matrix(Rm).as_quat()
                want_q[h] = np.array([q4[3], q4[0], q4[1], q4[2]])
            g_sym_state.value = (f"镜像差 {d0 * 1000:.0f}mm -> 对称化权重 {w:.2f}"
                                 + ("  (完全对称化)" if w > 0.99 else
                                    "  (原样透传)" if w < 0.01 else "  (过渡)"))
        else:
            g_sym_state.value = "关闭"
        g_rel.value = (f"{np.linalg.norm((targets['left'] - targets['right']) - (hum['left'] - hum['right'])) * 1000:.0f} mm")
        # --- engage ramp + one-euro, byte-for-byte the consumer's ----------
        if ramp["t0"] is None:
            ramp["t0"] = g_t.value
            ik.refresh_fk()
            for _h in ("right", "left"):
                # BASE frame, not pink's world. set_target_chest re-plants
                # targets on the reduced model's chest, which sits at the ZERO
                # torso, so ik.ee_pos() is ~0.5 m away from the same point in
                # base coordinates. Ramping from the wrong frame is not a ramp:
                # frame 0 still presents a large step and the solve still walks
                # R_arm_j1 to its +176 deg limit, 3.8 deg per frame.
                _rel = ik._chest.rotation.T @ (ik.ee_pos(_h)
                                               - ik._chest.translation)
                ramp["home"][_h] = chest_pos + chest_T[:3, :3] @ _rel
        _a = min(1.0, max(0.0, (g_t.value - ramp["t0"]) / RAMP_S))
        _dt = 1.0 / 60.0
        for _h in ("right", "left"):
            targets[_h] = (1.0 - _a) * ramp["home"][_h] + _a * targets[_h]
            targets[_h] = pos_filt[_h](targets[_h], _dt)
            want_q[_h] = quat_filt[_h](want_q[_h], _dt)

        solver = ik_elbow if g_elbow.value else ik
        # keep the FULL target for scoring - grading a relaxation against its
        # own relaxed target is how you conclude that giving up worked
        want_full = {h: R.from_quat([want_q[h][1], want_q[h][2], want_q[h][3],
                                     want_q[h][0]]).as_matrix()
                     for h in ("right", "left")}
        for h in ("right", "left"):
            q_use = want_q[h]
            if g_rollfree.value:
                R_cur = solver.data.oMf[
                    solver.model.getFrameId(EE_FRAME[h])].rotation
                q4 = R.from_matrix(relax_roll(want_full[h], R_cur)).as_quat()
                q_use = np.array([q4[3], q4[0], q4[1], q4[2]])
            solver.set_target_chest(h, chest_pos, chest_wxyz, targets[h], q_use)
            if g_elbow.value and f"elbow_{h}" in hum:
                # the operator's own elbow, carried over by the SAME law as the
                # wrist - mixing laws here would move the elbow target into a
                # frame the wrist target does not live in
                el_t = (rob_sh[h] + k * (hum[f"elbow_{h}"] - hum[f"sh_{h}"])
                        if shoulder_law
                        else bind + g_scale.value * hum[f"elbow_{h}"])
                solver.set_elbow_target_chest(h, chest_pos, chest_wxyz, el_t)
        solver.solve()
        solver.refresh_fk()
        ik = solver

        # where the Sharpa palm ends up looking, straight off the solved EE
        # frame - the one number this preview exists to show
        palm_dir = {h: ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])].rotation
                    @ PALM_IN_EE for h in ("right", "left")}

        def _compass(n):
            ax = [("前", [1, 0, 0]), ("后", [-1, 0, 0]), ("左", [0, 1, 0]),
                  ("右", [0, -1, 0]), ("上", [0, 0, 1]), ("下", [0, 0, -1])]
            k = max(ax, key=lambda kv: float(np.dot(n, kv[1])))
            return f"{k[0]}({np.degrees(np.arccos(np.clip(np.dot(n, k[1]), -1, 1))):.0f}°)"
        g_palmdir.value = "  ".join(f"{h[0].upper()} {_compass(palm_dir[h])}"
                                    for h in ("right", "left"))

        g_need.value = (f"{need['right']:.3f} / {need['left']:.3f} m "
                        f"(臂长 {rob_arm:.3f})")
        worst = max(need.values())
        # elbow asymmetry, the number behind "wrists match but elbows do not"
        _el = {h: ik.data.oMf[ik.model.getFrameId(ELBOW_FRAME[h])].translation
               - chest_pos for h in ("right", "left")}
        _M = np.diag([1.0, -1.0, 1.0])
        g_asym.value = f"{np.linalg.norm(_el['right'] - _M @ _el['left']) * 1000:.0f} mm"
        if "elbow_right" in hum and "elbow_left" in hum:
            g_hasym.value = (f"{np.linalg.norm(hum['elbow_right'] - _M @ hum['elbow_left']) * 1000:.0f}"
                             " mm  <- 目标")
        g_verdict.value = ("✓ 可达" if worst <= rob_arm
                           else f"✗ 超出 {(worst - rob_arm) * 1000:.0f} mm")
        res_mm = {h: float(np.linalg.norm(
            ik.ee_pos(h)
            - ik.frame_tasks[h].transform_target_to_world.translation) * 1000)
            for h in ("right", "left")}
        g_res.value = "  ".join(f"{h[0].upper()} {res_mm[h]:.0f}mm"
                                for h in ("right", "left"))
        worst_res = max(res_mm.values())
        g_track.value = (
            f"✓ 跟随正常({worst_res:.0f}mm)" if worst_res < 30 else
            f"⚠ 偏差 {worst_res:.0f}mm,接近饱和" if worst_res < 80 else
            f"✗ 手离目标 {worst_res:.0f}mm —— 这一维已经不跟了")
        # palm normal vs finger heading, each against the FULL target
        eo = {}
        for h in ("right", "left"):
            Rg = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])].rotation
            Rw = want_full[h]
            eo[h] = tuple(float(np.degrees(np.arccos(np.clip(
                (Rg @ v) @ (Rw @ v), -1.0, 1.0))))
                for v in (PALM_IN_EE, FINGERS_IN_EE))
        g_ori_err.value = "  ".join(
            f"{h[0].upper()} {eo[h][0]:.0f}°/{eo[h][1]:.0f}°"
            for h in ("right", "left"))
        # a joint sitting on its limit has no DoF left - the direct readout
        # behind "sometimes the IK just locks up"
        pin_txt = []
        for h in ("right", "left"):
            hit = []
            for k in range(1, 8):
                i = ik.pin_names.index(f"{h[0].upper()}_arm_j{k}")
                qv = ik.config.q[i]
                if (qv - ik.model.lowerPositionLimit[i] < np.radians(1.0)
                        or ik.model.upperPositionLimit[i] - qv < np.radians(1.0)):
                    hit.append(f"j{k}")
            pin_txt.append(f"{h[0].upper()} " + (",".join(hit) if hit else "无"))
        g_pinned.value = "   ".join(pin_txt)
        qarm = dict(zip(ik.pin_names, ik.config.q))
        qarm.update(torso_cfg)
        qarm.update(head_cfg)
        if jc_w is not None:
            jc_w.writerow([f"{g_t.value:.4f}"]
                          + [f"{float(qarm.get(n, TORSO_HOME.get(n, 0.0))):.6f}"
                             for n in JOINT_DUMP])
        cfg = np.array([qarm.get(n, TORSO_HOME.get(n, 0.0)) for n in all_joints])
        vurdf.update_cfg(cfg)
        urdf.update_cfg(cfg)          # keep FK markers on the same posture

        if ref_vurdf is not None:
            sel = g_ref.value
            if sel in ref_poses:
                rj = ref_poses[sel]["joints"]
                ref_vurdf.update_cfg(np.array(
                    [rj.get(j, TORSO_HOME.get(j, 0.0)) for j in all_joints]))
            ref_root.visible = sel in ref_poses

        # ---- operator chain, drawn beside the robot -------------------------
        if g_human.value:
            base = (bind if not shoulder_law else chest_pos) + HUMAN_OFFSET
            pts = {n: base + v for n, v in hum.items()}
            for n, p in pts.items():
                c = COL.get(n.replace("sh_", "").replace("elbow_", ""), COL["bone"])
                server.scene.add_icosphere(f"/human/{n}", radius=0.03,
                                           color=c, position=p)
            bones = [("waist", "sh_right"), ("waist", "sh_left")]
            for h in ("right", "left"):
                mid = f"elbow_{h}" if f"elbow_{h}" in pts else None
                bones += ([(f"sh_{h}", mid), (mid, h)] if mid
                          else [(f"sh_{h}", h)])
            if "head" in pts:
                bones.append(("waist", "head"))
            for a, b in bones:
                server.scene.add_spline_catmull_rom(
                    f"/human/bone_{a}_{b}", np.stack([pts[a], pts[b]]),
                    color=COL["bone"], line_width=3.0)
            if g_ax.value:
                for h in ("right", "left"):
                    server.scene.add_frame(
                        f"/human/{h}_ax", position=pts[h], wxyz=want_q[h],
                        axes_length=g_axlen.value,
                        axes_radius=g_axlen.value * 0.05)
                server.scene.add_frame(
                    "/human/waist_ax", position=pts["waist"],
                    wxyz=(1.0, 0.0, 0.0, 0.0), axes_length=g_axlen.value,
                    axes_radius=g_axlen.value * 0.05)

        # ---- robot chain: the same points, on the robot ---------------------
        if g_robot.value:
            for h in ("right", "left"):
                sh = rob_sh[h]
                el = ik.data.oMf[ik.model.getFrameId(ELBOW_FRAME[h])].translation
                ee = ik.ee_pos(h)
                for tag, p, r in (("sh", sh, 0.03), ("el", el, 0.025),
                                  ("ee", ee, 0.03)):
                    server.scene.add_icosphere(f"/rob/{h}_{tag}", radius=r,
                                               color=COL[h], position=p)
                for tag, a, b in (("a", sh, el), ("b", el, ee)):
                    server.scene.add_spline_catmull_rom(
                        f"/rob/{h}_bone{tag}", np.stack([a, b]),
                        color=COL[h], line_width=3.0)
                server.scene.add_icosphere(f"/rob/{h}_tgt", radius=0.022,
                                           color=(255, 60, 60),
                                           position=targets[h])
                if g_gap.value and res_mm[h] > 5.0:
                    # the shortfall, drawn. A red ball floating away from the
                    # hand is the one thing you cannot miss while dragging a
                    # slider; the millimetre readout alone was missed.
                    server.scene.add_spline_catmull_rom(
                        f"/rob/{h}_gap", np.stack([ee, targets[h]]),
                        color=(255, 40, 40),
                        line_width=8.0 if res_mm[h] > 80 else 4.0)
                else:
                    server.scene.add_spline_catmull_rom(
                        f"/rob/{h}_gap", np.stack([ee, ee]),
                        color=(255, 40, 40), line_width=1.0)
                eeR = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])].rotation
                if g_palmarrow.value:
                    n = palm_dir[h]
                    server.scene.add_arrows(
                        f"/rob/{h}_palm", np.stack([ee, ee + 0.25 * n])[None],
                        np.array([[245, 245, 245]], dtype=np.uint8),
                        shaft_radius=0.008, head_radius=0.020, head_length=0.045)
                if g_ax.value:
                    q = R.from_matrix(eeR).as_quat()
                    server.scene.add_frame(
                        f"/rob/{h}_ax", position=ee,
                        wxyz=(q[3], q[0], q[1], q[2]),
                        axes_length=g_axlen.value * 0.8,
                        axes_radius=g_axlen.value * 0.04)
                    server.scene.add_frame(
                        f"/rob/{h}_tgt_ax", position=targets[h], wxyz=want_q[h],
                        axes_length=g_axlen.value,
                        axes_radius=g_axlen.value * 0.05)
            server.scene.add_icosphere("/rob/chest", radius=0.035,
                                       color=COL["robot"], position=chest_pos)
        state.setdefault("smoke", []).append(
            (res_mm["right"], res_mm["left"], eo["right"][1], eo["left"][1]))
        if args.headless_seconds and (
                state.get("fi", 0) >= min(len(times) - 1,
                                          int(args.headless_seconds * 60))):
            # median over the run, not the last frame: a single instant of a
            # 136 s take says nothing, and comparing one against another
            # harness's median is how two implementations of the same law get
            # to look different while both are right (or both wrong).
            a = np.asarray(state["smoke"][200:])
            print(f"[preview] headless smoke: law={g_law.value} "
                  f"scale={g_scale.value} sym={g_sym.value} n={len(a)}  "
                  f"pos med R {np.median(a[:, 0]):.1f} L {np.median(a[:, 1]):.1f} mm  "
                  f"finger med R {np.median(a[:, 2]):.1f} L {np.median(a[:, 3]):.1f} deg")
            return 0
        if not args.headless_seconds:
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    raise SystemExit(main())
