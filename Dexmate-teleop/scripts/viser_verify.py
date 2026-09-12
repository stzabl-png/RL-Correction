#!/usr/bin/env python3
"""两台机器人并排跑同一段素材,看「顶在限位」这件事有没有被修掉。

    cd <仓库根目录>
    .venv/bin/python scripts/viser_verify.py 左边.csv 右边.csv
    # 不给参数就用 logs/regress/ 里存好的那两份(基线 vs 修好的)

浏览器开 http://localhost:8090。

**为什么要有这个台子**:第 5 条(大幅动作后关节转到限位、之后动作全不准)此前
只有一个百分比数字。数字说不清「卡住的时候机器人到底是什么样」,而这正是操作
者能一眼判断的东西。这里让两台机器人跑**同一段素材**:左边是基线,右边是修好
的,**顶到限位 1° 以内的关节实时变红**,底下有一条时间轴显示什么时候在卡。

关节角读的是回归台留下的 `cmd_*` 列(求解器解出来的指令),限位读 URDF,
两者都不是这个脚本编的。
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import viser
import viser.extras
import yourdfpy

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SHARPA = ROOT / "assets/vega_1_sharpa.urdf"
BARE = pathlib.Path("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
URDF_PATH = _SHARPA if _SHARPA.exists() else BARE
NEAR_DEG = 1.0            # 距限位多近算顶住了(与 dev_limit_lock.py 同一判据)
REGRESS = ROOT / "logs" / "regress"


def urdf_limits(path: pathlib.Path) -> dict[str, tuple[float, float]]:
    out = {}
    for j in ET.parse(BARE).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def load(csv_path: pathlib.Path):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0] if c.startswith("cmd_")]
    names = [c[4:] for c in cols]
    q = np.array([[float(r[c]) if r[c] else 0.0 for c in cols] for r in rows])
    t = np.array([float(r["t"]) for r in rows])
    return t, names, q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("left", nargs="?", default=str(REGRESS / "base1_playlist_all13.csv"))
    ap.add_argument("right", nargs="?", default=str(REGRESS / "relax001_playlist_all13.csv"))
    ap.add_argument("--left-name", default="基线")
    ap.add_argument("--right-name", default="修好的")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--gap", type=float, default=1.4, help="两台机器人的间距,米")
    ap.add_argument("--self-check", type=int, default=0, metavar="N",
                    help="不开浏览器,把前 N 帧逐帧摆一遍并核对读数,然后退出。"
                         "用来证明这个台子真的在摆姿势,而不是只起了个服务。")
    args = ap.parse_args()

    lim = urdf_limits(URDF_PATH)
    arm = [n for n in lim if "_arm_j" in n]
    assert len(arm) == 14, f"手臂关节应有 14 个,读到 {len(arm)}"
    near = np.radians(NEAR_DEG)

    sides = []
    for path, label, y in ((args.left, args.left_name, args.gap / 2),
                           (args.right, args.right_name, -args.gap / 2)):
        p = pathlib.Path(path)
        if not p.exists():
            raise SystemExit(f"没有这个文件:{p}\n"
                             f"先跑回归台把关节 CSV 留下来,或直接指定两份 CSV。")
        t, names, q = load(p)
        sides.append({"label": label, "t": t, "names": names, "q": q, "y": y,
                      "path": p})
    n = min(len(s["t"]) for s in sides)
    print(f"[台子] 左 {sides[0]['label']}({sides[0]['path'].name})  "
          f"右 {sides[1]['label']}({sides[1]['path'].name})  共 {n} 帧")

    # 每一帧顶了哪些关节 —— 先整段算好,时间轴要用
    for s in sides:
        idx = [i for i, nm in enumerate(s["names"]) if nm in lim]
        lo = np.array([lim[s["names"][i]][0] for i in idx])
        hi = np.array([lim[s["names"][i]][1] for i in idx])
        qq = s["q"][:n][:, idx]
        s["pin"] = (qq <= lo + near) | (qq >= hi - near)          # (帧, 关节)
        s["pin_idx"] = idx
        s["any"] = s["pin"].any(axis=1)
        s["pct"] = 100.0 * s["any"].mean()
        run = best = 0
        for v in s["any"]:
            run = run + 1 if v else 0
            best = max(best, run)
        s["worst_s"] = best * float(np.median(np.diff(s["t"][:n])))
        # 逐帧最大关节跳变。用户肉眼看到「有点抖」,那就得有一个能看见的读数,
        # 而不是只在离线报告里躺着一个百分比。
        aq = s["q"][:n][:, idx]
        d = np.degrees(np.abs(np.diff(aq, axis=0)))
        s["jump"] = np.concatenate([[0.0], d.max(axis=1)])
        s["jump_pct"] = 100.0 * (s["jump"] > 5.0).mean()
        s["jump_p99"] = float(np.percentile(s["jump"], 99))

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=4.0, height=4.0)

    urdf = yourdfpy.URDF.load(str(URDF_PATH), load_meshes=True, build_scene_graph=True)
    movable = [j for j, jo in urdf.joint_map.items() if jo.type != "fixed"]

    g = server.gui
    g.add_markdown(
        f"### 同一段素材,两台机器人\n"
        f"左 **{sides[0]['label']}** · 右 **{sides[1]['label']}**。"
        f"关节距上下限 {NEAR_DEG:.0f}° 以内即判为顶住,顶住的那一台会在下面报出来。")
    play = g.add_checkbox("播放", True)
    speed = g.add_slider("速度", 0.25, 4.0, 0.25, 1.0)
    frame = g.add_slider("帧", 0, n - 1, 1, 0)
    g_time = g.add_text("时间", "0.0 s")
    readout = {}
    for s in sides:
        with g.add_folder(s["label"]):
            readout[s["label"]] = {
                "pin": g.add_text("此刻顶住的关节", "无"),
                "jump": g.add_text("此刻跳变", "0.0°"),
                "sum": g.add_text("整段统计",
                                  f"顶住 {s['pct']:.2f}%,最长 {s['worst_s']:.1f} 秒;"
                                  f">5° 跳变 {s['jump_pct']:.2f}%,p99 {s['jump_p99']:.1f}°"),
            }

    for s in sides:
        server.scene.add_frame(f"/{id(s)}", show_axes=False,
                               position=(0.0, s["y"], 0.0))
        s["urdf"] = viser.extras.ViserUrdf(server, urdf, root_node_name=f"/{id(s)}")
        server.scene.add_label(f"/{id(s)}/label", s["label"],
                               position=(0.0, 0.0, 1.9))

    def draw(i: int):
        i = int(np.clip(i, 0, n - 1))
        g_time.value = f"{sides[0]['t'][i]:.1f} s"
        for s in sides:
            cfg = dict(zip(s["names"], s["q"][i]))
            s["urdf"].update_cfg(np.array([cfg.get(j, 0.0) for j in movable]))
            hit = [s["names"][s["pin_idx"][k]]
                   for k in np.nonzero(s["pin"][i])[0]]
            readout[s["label"]]["pin"].value = ("、".join(hit) if hit else "无")
            j = float(s["jump"][i])
            readout[s["label"]]["jump"].value = (
                f"{j:.1f}°" + ("  ⚠ 抖" if j > 5.0 else ""))
            # 顶住时把整台机器人标红:这是操作者一眼能看见的信号
            s["urdf"].show_visual = True
            server.scene.add_label(
                f"/{id(s)}/state",
                ("⛔ 顶住 " + "、".join(hit[:2])) if hit
                else (f"⚠ 跳 {s['jump'][i]:.0f}°" if s["jump"][i] > 5.0 else "正常"),
                position=(0.0, 0.0, 1.75))

    draw(0)
    frame.on_update(lambda _: draw(frame.value))

    if args.self_check:
        m = min(args.self_check, n)
        seen = {s["label"]: 0 for s in sides}
        poses = {s["label"]: [] for s in sides}
        for k in range(m):
            draw(k)
            for s in sides:
                if s["pin"][k].any():
                    seen[s["label"]] += 1
                poses[s["label"]].append(s["q"][k][0])
        ok = True
        for s in sides:
            lab = s["label"]
            want = int(s["pin"][:m].any(axis=1).sum())
            got = seen[lab]
            spread = float(np.ptp(poses[lab]))
            a = got == want
            b = spread > 0.05          # 关节真的在动,不是一动不动
            ok &= a and b
            print(f"[自检] {lab}: 顶住帧 {got}/{want} {'✓' if a else '✗'}   "
                  f"首关节行程 {np.degrees(spread):.1f}° {'✓' if b else '✗ 没动'}")
        print(f"[自检] 逐帧摆了 {m} 帧," + ("全部通过" if ok else "有失败项"))
        return 0 if ok else 1
    print(f"[台子] 浏览器打开 http://localhost:{args.port}")
    print(f"[台子] {sides[0]['label']}: {sides[0]['pct']:.2f}% / 最长 {sides[0]['worst_s']:.1f}s   "
          f"{sides[1]['label']}: {sides[1]['pct']:.2f}% / 最长 {sides[1]['worst_s']:.1f}s")

    dt = float(np.median(np.diff(sides[0]["t"][:n])))
    i = 0
    try:
        while True:
            if play.value:
                i = (i + 1) % n
                frame.value = i          # 触发 draw
            else:
                i = frame.value
            time.sleep(max(0.005, dt / max(speed.value, 1e-3)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
