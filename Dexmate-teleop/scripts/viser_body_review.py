#!/usr/bin/env python3
"""Viser body-model review: judge a capture's SMPL-24 skeleton by eye.

`viser_capture_view.py` draws the tracker frames. This draws the thing the
mapping actually consumes - PICO's 24-joint body estimate - as a full skeleton,
so "is the PICO data good?" can be answered by looking at it instead of by
reading residuals.

Two failure modes it is meant to expose, both of which have already cost a
capture batch:

  FROZEN   a joint stops changing bit-for-bit. Drawn RED.
  脑补     the estimate keeps streaming a plausible pose that is not the one
           you held. Visible as a skeleton that does not match the take's
           label - arms short, wrists near the chest during "arms high".
           Bone-length and wrist-height readouts make it quantitative.

It also puts the open orientation question in front of your eyes. What
`--ori-mode palm` will do to the robot hand is drawn on the operator's wrist:

  WHITE arrow   palm normal. PALM_FIX carries the SMPL wrist frame to R_ee,
                and the Sharpa palm normal is +R_ee.x (verified against the
                hand's own kinematics and mesh, and against the official
                vega_1_f5d6.urdf), so palm_world = R_wrist @ PALM_FIX @ ex.
  YELLOW arrow  finger direction, +R_ee.z by the same construction.

Together they are the robot hand's pose, drawn where your hand was. For each
held pose, say whether that is where your palm faced and your fingers pointed.
That is the whole calibration question, with nothing to wear.

Note on frames: `record_of_relationship` calls the arm flange "EE", but its
axis table matches `*_arm_l8` exactly (0.00 deg) and `*_ee` only to 90 deg.
The two differ by Rz(+-90), mirrored L/R. Read it as l8 and everything agrees.

Usage:
    .venv/bin/python scripts/viser_body_review.py logs/pose_capture_20260731_ori
    .venv/bin/python scripts/viser_body_review.py logs/pose_capture_20260731_ori --port 8084
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import msgpack
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.palm_fix import (  # noqa: E402
    FINGERS_IN_EE, PALM_FIX, PALM_IN_EE)
from magicdexmate.pico.xr_pose import R_HEADSET_TO_WORLD  # noqa: E402

# SMPL kinematic tree (24 joints; -1 = root). Same table as view_skeleton.py.
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
          18, 19, 20, 21]
NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
         "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
         "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
         "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"]

# joints worth labelling in 3D - the ones the mapping reads or is argued about,
# plus the ankles/feet, which is where the two leg trackers land in the estimate
TAGGED = {0: "Pelvis", 9: "Spine3(waist)", 15: "Head",
          18: "L_Elbow", 19: "R_Elbow",
          20: "L_Wrist(20)", 21: "R_Wrist(21)", 22: "L_Hand(22)", 23: "R_Hand(23)",
          7: "L_Ankle(7)", 8: "R_Ankle(8)", 10: "L_Foot(10)", 11: "R_Foot(11)"}

WRIST_SETS = {
    "20/21  L_Wrist / R_Wrist (anatomical)": (20, 21),
    "22/23  L_Hand / R_Hand (what gear_sonic uses)": (22, 23),
}

# PALM_FIX / PALM_IN_EE / FINGERS_IN_EE come from magicdexmate.palm_fix - see
# that module for how the constant was fitted and which channel it is valid on.

BONE_PAIRS = [(16, 18), (18, 20), (17, 19), (19, 21)]   # upper arm / forearm


def load_take(path: pathlib.Path):
    """-> t[s], pos (N,24,3), rot (N,24,3,3), frozen (N,24) - all z-up world."""
    ts, rows = [], []
    for m in msgpack.Unpacker(open(path, "rb"), raw=False):
        b = m.get("body24")
        if not b:
            continue
        ts.append(m.get("t_us", len(ts) * 10_000) / 1e6)
        rows.append(np.asarray(b, dtype=float))
    if not rows:
        raise SystemExit(f"{path.name}: no body24 in this take "
                         "(producer was run without --body-full)")
    raw = np.stack(rows)                                     # (N,24,7)
    t = np.array(ts)
    t -= t[0]
    M = R_HEADSET_TO_WORLD
    pos = np.einsum("ij,nkj->nki", M, raw[:, :, :3])
    rot = np.zeros(raw.shape[:2] + (3, 3))
    for n in range(raw.shape[0]):
        q = raw[n, :, 3:7]
        bad = np.all(np.abs(q) < 1e-9, axis=1)
        q = np.where(bad[:, None], np.array([0.0, 0, 0, 1.0]), q)
        rot[n] = M @ R.from_quat(q).as_matrix() @ M.T
    frozen = np.zeros(raw.shape[:2], dtype=bool)
    frozen[1:] = np.all(raw[1:] == raw[:-1], axis=2)
    return t, pos, rot, frozen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=pathlib.Path,
                    help="capture directory, or a single .msgpack take")
    ap.add_argument("--port", type=int, default=8084,
                    help="8084: 8080 trace, 8081 joint bench, 8082 capture, 8083 mapping")
    args = ap.parse_args()

    if args.capture.is_dir():
        files = sorted(p for p in args.capture.glob("*.msgpack")
                       if "format_test" not in p.name)
    else:
        files = [args.capture]
    if not files:
        raise SystemExit(f"no takes found in {args.capture}")

    cache: dict[str, tuple] = {}

    def get(name: str):
        if name not in cache:
            cache[name] = load_take(next(p for p in files if p.stem == name))
            t, pos, _, fr = cache[name]
            span = np.linalg.norm(pos[:, 20] - pos[:, 21], axis=1)
            print(f"[body] {name}: {len(t)} frames {t[-1]:.1f}s  "
                  f"frozen {100 * fr.mean():.0f}%  "
                  f"wrist-to-wrist {span.mean():.2f}m")
        return cache[name]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/origin", axes_length=0.3, axes_radius=0.005)
    server.scene.add_grid("/grid", width=3.0, height=3.0, position=(0, 0, 0))

    g_take = server.gui.add_dropdown("动作 take", [p.stem for p in files],
                                     initial_value=files[0].stem)
    g_time = server.gui.add_slider("时间", 0, 1, 1, 0)
    g_play = server.gui.add_checkbox("播放", False)
    g_wrist = server.gui.add_dropdown("腕关节索引", list(WRIST_SETS),
                                      initial_value=list(WRIST_SETS)[0])
    g_palm = server.gui.add_checkbox("画掌心朝向箭头", True)
    g_axes = server.gui.add_checkbox("画关节坐标轴", True)
    g_axlen = server.gui.add_slider("坐标轴长度", 0.02, 0.25, 0.01, 0.10)
    g_tags = server.gui.add_checkbox("关节名标签", True)
    server.gui.add_markdown(
        "**白箭头** = 机器人手的**掌心法向**(现行 `--ori-mode palm` 会解出来的)\n\n"
        "**黄箭头** = 机器人手的**手指指向**\n\n"
        "对每个姿势判一句:这就是当时你手掌朝的方向、手指指的方向吗?\n\n"
        "腕关节 20/21 与 22/23 **朝向逐位相同**,只差 91/92mm 平移 —— "
        "换索引不会改变朝向。\n\n"
        "---\n\n"
        "**本查看器不做任何计算**:画的是文件里 body24 的 24 个关节位置+四元数原值,"
        "只做了一次 y-up→z-up 帧变换,无 FK / 拟合 / 插值。\n\n"
        "**但 body24 本身就是 PICO 算出来的人体模型**,不是追踪器读数。"
        "身上只有 5 个追踪器,24 个关节里绝大多数(肘 18/19、肩、脊柱、膝…)"
        "**是 PICO 推出来的,不是测出来的**。\n\n"
        "本批文件里 `trackers` 那 5 项与 body24[20/21/9/18/19] **逐位相同**,"
        "即这批**只有模型通道,没有裸追踪器数据**——"
        "所以看不到物理追踪器本身,只能看到它们影响后的模型输出。")
    g_info = server.gui.add_markdown("")

    def frame_index(t):
        return int(np.clip(round(g_time.value * (len(t) - 1)), 0, len(t) - 1))

    last = {"take": None}

    def redraw():
        name = g_take.value
        t, pos, rot, frozen = get(name)
        i = frame_index(t)
        p, Rj, fz = pos[i], rot[i], frozen[i]

        # bones
        seg, col = [], []
        for j, par in enumerate(PARENT):
            if par < 0:
                continue
            seg.append([p[par], p[j]])
            c = (255, 60, 60) if (fz[j] or fz[par]) else (90, 220, 140)
            col.append([c, c])
        server.scene.add_line_segments("/skel/bones", np.array(seg),
                                       np.array(col, dtype=np.uint8), line_width=4.0)
        server.scene.add_point_cloud("/skel/joints", p,
                                     np.where(fz[:, None], np.array([255, 60, 60]),
                                              np.array([230, 230, 230])).astype(np.uint8),
                                     point_size=0.018)

        for j, lbl in TAGGED.items():
            server.scene.add_label(f"/skel/tag_{j}", lbl, position=p[j],
                                   visible=g_tags.value)
            server.scene.add_frame(f"/skel/ax_{j}", axes_length=g_axlen.value,
                                   axes_radius=g_axlen.value * 0.06,
                                   position=p[j],
                                   wxyz=np.roll(R.from_matrix(Rj[j]).as_quat(), 1),
                                   visible=g_axes.value)

        li, ri = WRIST_SETS[g_wrist.value]
        if g_palm.value:
            starts, ends_p, ends_f = [], [], []
            for hand, j in (("left", li), ("right", ri)):
                R_ee = Rj[j] @ PALM_FIX[hand]
                starts.append(p[j])
                ends_p.append(p[j] + 0.22 * (R_ee @ PALM_IN_EE))
                ends_f.append(p[j] + 0.22 * (R_ee @ FINGERS_IN_EE))
            server.scene.add_arrows("/palm/normal",
                                    np.stack([np.array(starts), np.array(ends_p)], axis=1),
                                    np.array([[245, 245, 245]] * 2, dtype=np.uint8),
                                    shaft_radius=0.008, head_radius=0.020, head_length=0.045)
            server.scene.add_arrows("/palm/fingers",
                                    np.stack([np.array(starts), np.array(ends_f)], axis=1),
                                    np.array([[255, 210, 60]] * 2, dtype=np.uint8),
                                    shaft_radius=0.008, head_radius=0.020, head_length=0.045)
        else:
            for n in ("/palm/normal", "/palm/fingers"):
                try:
                    server.scene.remove_by_name(n)
                except Exception:
                    pass

        # readouts that make 脑补 quantitative
        ua_l, fa_l = (np.linalg.norm(p[18] - p[16]), np.linalg.norm(p[20] - p[18]))
        ua_r, fa_r = (np.linalg.norm(p[19] - p[17]), np.linalg.norm(p[21] - p[19]))
        head_z = p[15][2]
        # how much each part moved over the whole take - a static pose should
        # leave the legs near zero, and that is not evidence the leg trackers
        # are dead, only that they had nothing to report
        travel = {j: 1000 * np.ptp(pos[:, j], axis=0).max()
                  for j in (0, 7, 8, 18, 19, 20, 21)}
        g_info.content = (
            f"**{name}**  帧 {i + 1}/{len(t)}  t={t[i]:.2f}s  "
            f"冻结 {100 * fz.mean():.0f}%\n\n"
            f"| | 左 | 右 |\n|---|---|---|\n"
            f"| 上臂 | {ua_l * 1000:.0f} mm | {ua_r * 1000:.0f} mm |\n"
            f"| 小臂 | {fa_l * 1000:.0f} mm | {fa_r * 1000:.0f} mm |\n"
            f"| 腕高−头高 | {(p[li][2] - head_z) * 1000:+.0f} mm | "
            f"{(p[ri][2] - head_z) * 1000:+.0f} mm |\n"
            f"| 腕−腰 前后 | {(p[li][0] - p[9][0]) * 1000:+.0f} mm | "
            f"{(p[ri][0] - p[9][0]) * 1000:+.0f} mm |\n\n"
            f"双腕间距 {np.linalg.norm(p[li] - p[ri]):.2f} m\n\n"
            f"**全段位移**(mm):骨盆 {travel[0]:.0f} · 左踝 {travel[7]:.0f} · "
            f"右踝 {travel[8]:.0f} · 左肘 {travel[18]:.0f} · 右肘 {travel[19]:.0f} · "
            f"左腕 {travel[20]:.0f} · 右腕 {travel[21]:.0f}")

    for h in (g_take, g_time, g_wrist, g_palm, g_axes, g_axlen, g_tags):
        h.on_update(lambda _: redraw())

    redraw()
    print(f"[body] http://localhost:{args.port}  ({len(files)} takes)")
    while True:
        if g_play.value:
            t = get(g_take.value)[0]
            step = 1.0 / max(len(t) - 1, 1)
            g_time.value = (g_time.value + step * 3) % 1.0
        time.sleep(1 / 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
