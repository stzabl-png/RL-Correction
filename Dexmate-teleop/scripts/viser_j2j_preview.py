#!/usr/bin/env python3
"""J2J 形似直映离线预览:机器人跟着录制素材摆,操作者骨架并排对照。

    cd <仓库根目录>
    .venv/bin/python scripts/viser_j2j_preview.py                 # clip_headwaist
    .venv/bin/python scripts/viser_j2j_preview.py logs/playlist_all13.msgpack

浏览器开 http://localhost:8092。

**为什么要有这个台子**:J2J 的可行性报告(`sim/dev_j2j_feasibility.py`)给的
是五张表的数字;「臂形像不像人」「末端外推 20-30cm 是什么感觉」「肘伸直时
j3 保持是什么样」这三件事只有眼睛能判。这里把同一段素材离线走一遍 J2J:
左边机器人执行映射结果,右边操作者骨架原样回放;机器人肩上另画两段
**幽灵线 = 人的上臂/前臂方向按机器人自己的链长转录**——映射完美时机器人
手臂应与幽灵线重合,偏差就是肩心间隙 + 曲柄滞后的固有误差,当场可见。

**本预览不做任何映射计算**:关节角全部来自 `magicdexmate.j2j.J2JMapper`
(将来 `--map j2j` 也必须用它,一个量只有一份实现);限位读 URDF;
输入过 TrackerGlitchGate(与实机链路同款,通道与可行性脚本一致)。
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import viser
import viser.extras
import yourdfpy

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magicdexmate.j2j import (  # noqa: E402
    BODY, D2, HANDS, J2JMapper, TORSO_HOME)
from magicdexmate.sources.pico_source import (  # noqa: E402
    PicoLogSource, TrackerGlitchGate)

_SHARPA = ROOT / "assets/vega_1_sharpa.urdf"
BARE = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
URDF_PATH = _SHARPA if _SHARPA.exists() else BARE
NEAR_DEG = 1.0            # 距限位多近算顶住(与 viser_verify.py 同一判据)
FLAG_NAMES = ("肩死区", "肘死区(j3 保持)", "腕死区", "无解(保持上帧)")
# 骨架连线:腰-肩中点、肩-肩、左右 肩-肘-腕
SIDE = {"left": ("l_sh", "l_el", "l_wr"), "right": ("r_sh", "r_el", "r_wr")}


def precompute(path: pathlib.Path, mapper: J2JMapper):
    """素材整段过 J2J:关节角、旗标、人体点、量尺,全部一次算好。"""
    src = PicoLogSource(str(path))
    n = len(src.frames)
    t = np.array([(fr.t_us - src.frames[0].t_us) / 1e6 for fr in src.frames])
    gates = {k: TrackerGlitchGate() for k in (9, 18, 19, 20, 21)}  # 同可行性脚本
    q = {h: np.zeros((n, 7)) for h in HANDS}          # 执行角(无解 = 保持)
    flags = {h: np.zeros((n, 4), bool) for h in HANDS}
    beta = {h: np.zeros(n) for h in HANDS}
    P = {h: {k: np.full((n, 3), np.nan) for k in "SEW"} for h in HANDS}
    waist = np.zeros((n, 3))
    elbow_ang = {h: np.full(n, np.nan) for h in HANDS}   # 臂形尺:上臂方向夹角
    push = {h: np.full(n, np.nan) for h in HANDS}        # 腕点外推(肩对齐)[m]
    mapper.reset()
    hold = {h: mapper.HOME7[h].copy() for h in HANDS}
    n_miss = 0
    for i, fr in enumerate(src.frames):
        if fr.body24 is None:                 # 缺骨架帧:保持上帧,记无解
            n_miss += 1
            for h in HANDS:
                flags[h][i] = (False, False, False, True)
                q[h][i] = hold[h]
            continue
        b = np.asarray(fr.body24, float)
        use = {k: b[k] for k in (9, 16, 17, 18, 19, 20, 21)}
        for k, g in gates.items():
            use[k], _ = g.update(fr.t_us, b[k])
        out = mapper.map_body24(use)
        waist[i] = mapper._pt(use[9], use[9])            # 恒 0,占位保持结构
        for h in HANDS:
            r = out[h]
            flags[h][i] = r.flags
            for k, v in (("S", r.S), ("E", r.E), ("W", r.W)):
                if v is not None:
                    P[h][k][i] = v
            if r.q7 is not None:
                hold[h] = r.q7
                beta[h][i] = r.beta
            q[h][i] = hold[h]
        fk = mapper.fk_points_chest(q["right"][i], q["left"][i])
        for h in HANDS:
            if np.isnan(P[h]["S"][i]).any():
                continue
            u_h = P[h]["E"][i] - P[h]["S"][i]
            u_r = mapper.TILT @ fk[h]["u"]
            elbow_ang[h][i] = D2(np.arccos(np.clip(
                u_r @ (u_h / max(np.linalg.norm(u_h), 1e-9)), -1, 1)))
            w_r = mapper.TILT @ (fk[h]["WR"] - fk[h]["SH"])
            push[h][i] = float(np.linalg.norm(
                w_r - (P[h]["W"][i] - P[h]["S"][i])))
    if n_miss:
        print(f"[台子] {n_miss}/{n} 帧无 body24,已保持上帧(如实计入统计)")
    return {"n": n, "t": t, "q": q, "flags": flags, "beta": beta, "P": P,
            "elbow_ang": elbow_ang, "push": push, "path": path}


def limit_pins(mapper: J2JMapper, q, n):
    """(帧, 14) 顶限位布尔表 + 关节名(L1..7, R1..7)。"""
    names = [f"{s}_arm_j{i}" for s in "LR" for i in range(1, 8)]
    lo = np.array([mapper.LO[x] for x in names])
    hi = np.array([mapper.HI[x] for x in names])
    near = np.radians(NEAR_DEG)
    qq = np.concatenate([q["left"][:n], q["right"][:n]], axis=1)
    return (qq <= lo + near) | (qq >= hi - near), names, qq


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("material", nargs="?",
                    default=str(ROOT / "logs/clip_headwaist.msgpack"))
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--gap", type=float, default=1.6,
                    help="操作者骨架相对机器人的横向偏移,米")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="不开浏览器:先跑模块自检,再逐帧摆 N 帧核对读数后退出。")
    args = ap.parse_args()

    p = pathlib.Path(args.material)
    if not p.exists():
        raise SystemExit(f"没有这个素材:{p}")

    mapper = J2JMapper()
    if args.self_check:
        from magicdexmate.j2j import self_check
        print("== 模块自检 ==")
        bad = self_check(mapper)
        if bad:
            print("[自检] 模块尺子失败:", ", ".join(bad))
            return 1

    print(f"[台子] 素材 {p.name} 过 J2J …")
    d = precompute(p, mapper)
    n = d["n"]
    pins, jnames, q14 = limit_pins(mapper, d["q"], n)
    jump = np.concatenate([[0.0], D2(np.abs(np.diff(q14, axis=0)).max(axis=1))])

    # 整段统计(与可行性报告的表口径对齐,预览页与 stdout 各报一份)
    stat = {}
    for h in HANDS:
        ea = d["elbow_ang"][h][~np.isnan(d["elbow_ang"][h])]
        pu = d["push"][h][~np.isnan(d["push"][h])]
        fl = d["flags"][h]
        stat[h] = (f"臂形夹角 mean {ea.mean():.1f}°/p95 "
                   f"{np.percentile(ea, 95):.1f}°;腕外推 mean "
                   f"{1e3 * pu.mean():.0f}mm/p95 {1e3 * np.percentile(pu, 95):.0f}mm;"
                   f"肘死区 {100 * fl[:, 1].mean():.0f}% 帧,|β| p95 "
                   f"{np.percentile(np.abs(D2(d['beta'][h])), 95):.0f}°")
        print(f"[整段] {h:5s} {stat[h]}")
    print(f"[整段] 顶限位 {100 * pins.any(axis=1).mean():.2f}% 帧,"
          f">5° 跳变 {100 * (jump > 5.0).mean():.2f}%,p99 跳变 "
          f"{np.percentile(jump, 99):.1f}°/帧")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4.0, height=4.0)
    urdf = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=True,
                              build_scene_graph=True)
    movable = [j for j, jo in urdf.joint_map.items() if jo.type != "fixed"]
    robot = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")
    server.scene.add_label("/robot/tag", "J2J 执行", position=(0.0, 0.0, 1.9))

    # 骨架的竖直摆放:让人的肩中点与机器人肩中点同高(只为并排好看,
    # 不进任何计算 —— 量尺全部在肩对齐的相对坐标里,与摆放无关)
    sh_rob = np.mean([mapper.chest_in_base.act(mapper.GEO[h]["SH"])
                      for h in HANDS], axis=0)
    i0 = next(i for i in range(n)
              if not np.isnan(d["P"]["left"]["S"][i]).any())
    sh_hum = 0.5 * (d["P"]["left"]["S"][i0] + d["P"]["right"]["S"][i0])
    off = np.array([0.0, args.gap, sh_rob[2] - sh_hum[2]])
    server.scene.add_label("/op_tag", "操作者骨架(body24)",
                           position=off + np.array([0.0, 0.0, 1.0]))

    g = server.gui
    g.add_markdown(
        f"### J2J 形似直映预览 — {p.name}\n"
        "机器人执行 `magicdexmate.j2j` 的映射结果;白/黄幽灵线 = 人的上臂/"
        "前臂方向按机器人链长转录,机器人手臂与它的偏差就是映射的全部误差。"
        f"关节距限位 {NEAR_DEG:.0f}° 以内判顶住。")
    play = g.add_checkbox("播放", True)
    speed = g.add_slider("速度", 0.25, 4.0, 0.25, 1.0)
    frame = g.add_slider("帧", 0, n - 1, 1, 0)
    g_time = g.add_text("时间", "0.0 s")
    read = {}
    for h in HANDS:
        with g.add_folder("右臂" if h == "right" else "左臂"):
            read[h] = {"flag": g.add_text("此刻状态", "正常"),
                       "num": g.add_text("此刻读数", "-"),
                       "sum": g.add_text("整段", stat[h])}
    g_pin = g.add_text("此刻顶住的关节", "无")
    g_jump = g.add_text("此刻跳变", "0.0°")

    dt = float(np.median(np.diff(d["t"]))) if n > 1 else 0.02

    def draw(i: int):
        i = int(np.clip(i, 0, n - 1))
        g_time.value = f"{d['t'][i]:.1f} s"
        cfg = dict(TORSO_HOME)
        for h, s in (("right", "R"), ("left", "L")):
            for k in range(7):
                cfg[f"{s}_arm_j{k + 1}"] = float(d["q"][h][i][k])
        robot.update_cfg(np.array([cfg.get(j, 0.0) for j in movable]))
        # 操作者骨架(腰系点 + 摆放偏移)
        seg, ghost_u, ghost_f = [], [], []
        Pi = {h: {k: d["P"][h][k][i] for k in "SEW"} for h in HANDS}
        if not any(np.isnan(Pi[h]["S"]).any() for h in HANDS):
            seg.append([off + Pi["left"]["S"], off + Pi["right"]["S"]])
            for h in HANDS:
                seg.append([off + Pi[h]["S"], off + Pi[h]["E"]])
                seg.append([off + Pi[h]["E"], off + Pi[h]["W"]])
                # 幽灵线:人的方向 × 机器人链长,画在机器人肩上(基座系)
                G = mapper.GEO[h]
                sh_b = mapper.chest_in_base.act(G["SH"])
                u = Pi[h]["E"] - Pi[h]["S"]
                f = Pi[h]["W"] - Pi[h]["E"]
                u = u / max(np.linalg.norm(u), 1e-9)
                f = f / max(np.linalg.norm(f), 1e-9)
                Rb = mapper.chest_in_base.rotation
                el_b = sh_b + Rb @ (mapper.TILT_T @ u) * G["Lu"]
                ghost_u.append([sh_b, el_b])
                ghost_f.append([el_b, el_b + Rb @ (mapper.TILT_T @ f) * G["Lf"]])
        if seg:
            server.scene.add_line_segments(
                "/op/bones", np.array(seg),
                np.tile(np.array([90, 220, 140], np.uint8), (len(seg), 2, 1)),
                line_width=4.0)
            server.scene.add_line_segments(
                "/robot/ghost_u", np.array(ghost_u),
                np.tile(np.array([240, 240, 240], np.uint8), (2, 2, 1)),
                line_width=5.0)
            server.scene.add_line_segments(
                "/robot/ghost_f", np.array(ghost_f),
                np.tile(np.array([250, 210, 60], np.uint8), (2, 2, 1)),
                line_width=5.0)
        for h in HANDS:
            fl = d["flags"][h][i]
            read[h]["flag"].value = ("、".join(
                nm for nm, v in zip(FLAG_NAMES, fl) if v) or "正常")
            ea, pu = d["elbow_ang"][h][i], d["push"][h][i]
            read[h]["num"].value = (
                f"臂形夹角 {ea:.1f}°,腕外推 {1e3 * pu:.0f}mm,"
                f"β(丢弃的 j6 需求){D2(d['beta'][h][i]):+.0f}°"
                if np.isfinite(ea) else "骨架缺帧")
        hit = [jnames[k] for k in np.nonzero(pins[i])[0]]
        g_pin.value = "、".join(hit) if hit else "无"
        g_jump.value = f"{jump[i]:.1f}°" + ("  ⚠ 抖" if jump[i] > 5.0 else "")

    draw(0)
    frame.on_update(lambda _: draw(frame.value))

    if args.self_check:
        m = min(args.self_check, n)
        moved, pin_got, pin_want = [], 0, int(pins[:m].any(axis=1).sum())
        for k in range(m):
            draw(k)
            moved.append(q14[k][10])          # R_arm_j4:素材里必然在动
            if pins[k].any():
                pin_got += 1
        a = D2(np.ptp(np.array(moved))) > 3.0
        b = pin_got == pin_want
        c = all(np.isfinite(d["elbow_ang"][h][~np.isnan(d["elbow_ang"][h])]).all()
                for h in HANDS)
        print(f"[自检] 逐帧摆了 {m} 帧:R_arm_j4 行程 "
              f"{D2(np.ptp(np.array(moved))):.1f}° {'✓' if a else '✗ 没动'};"
              f"顶限位帧 {pin_got}/{pin_want} {'✓' if b else '✗'};"
              f"臂形尺全帧有限 {'✓' if c else '✗'}")
        return 0 if a and b and c else 1

    print(f"[台子] 浏览器打开 http://localhost:{args.port}")
    while True:
        if play.value:
            frame.value = (int(frame.value) + max(1, round(speed.value))) % n
            draw(frame.value)
        time.sleep(dt)


if __name__ == "__main__":
    sys.exit(main())
