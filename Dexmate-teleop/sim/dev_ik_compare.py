#!/usr/bin/env python3
"""Pink 与 cuRobo 两个 IK 后端的离线对比台。不依赖 Isaac。

    cd <仓库根目录> && .venv-isaac/bin/python sim/dev_ik_compare.py
    .venv-isaac/bin/python sim/dev_ik_compare.py --frames 800   # 截短真实素材

必须用 .venv-isaac(cuRobo 装在那里);.venv 没有 CUDA torch,起不来。

三份素材,两个后端同样喂,逐项打表。数字如实:cuRobo 更差也照实报。

  1. synthetic  可达轨迹:双手绕 HOME 画圆(半径 8cm)并同时摆动掌向
     (±25°,在腕量程内)。量的是正常跟随的精度与平滑。
  2. overrange  超量程(沿用 dev_relax_latch 的场景):右腕命令绕 EE 红色轴
     拧 150°(超出 j6 ±80° 量程)保持 6 秒,然后回到可达姿态 3 秒。量的是
     顶限位期间的行为(贴限位占比)与**需求回量程后能不能回来**(这正是
     用户抱怨的反面,两个后端都必须能回来,回不来直接判失败)。
  3. playlist   真实素材:logs/regress/base1_playlist_all13.csv 的 cmd 关节角
     (13 段拼接的签收回归素材)用 FK 反造 6D 腕目标流。不重写映射,拿到的
     就是真素材的目标分布(含贴近限位的姿态)。

指标(与回归台/dev_limit_lock 同一把尺):
  - 腕位置误差 mean/max [mm]、朝向误差 mean/max [deg](对解出的 q 做 FK,
    与命令目标比;不是求解器自报);
  - 贴限位帧占比 [%](14 关节任一距 URDF 限位 1° 以内,dev_limit_lock 同款
    算法)与最长连续 [s];
  - 逐帧关节跳变:>5° 帧占比 [%] 与 p99 [deg](回归台 jitter 同款);
  - 单步 solve() 耗时 p50/p95 [ms]。

素材 4(2026-08-11 增,「臂形要像操作者」硬需求的尺子):真实素材的目标流
不变,另从同一份回放 msgpack 提取操作者回转角流(magicdexmate.swivel.
operator_swivel_from_frame,与 teleop 的 _operator_swivel 同一约定,按
sample_at(i/50) 与目标流对齐;素材已验 100% 含 body24 与三个 tracker)。
量三个后端:pink(拉 home,不跟人)、curobo(默认,不跟人)、
curobo-follow(--null-bias 1 的零空间限速跟随)。
  - 臂形差:机器人回转角 vs 操作者回转角,逐帧 wrap 差 [deg] mean/max;
  - 臂形可复现性:同一腕目标重访(位置差<2cm、朝向差<10°、间隔>5s 的帧对,
    帧下采样 5 倍找对)时机器人回转角的差 [deg] mean/max —— 不跟人的后端
    臂形由历史决定,重访同一位姿臂形可以不同,这个数暴露的就是它。
对齐注:目标流前 1 秒是 auto-engage 延迟(cmd=home、人已在动),该段臂形差
无意义但只占 <1%,三个后端同样受累,对比成立。

每个会打印的数字先过「尺子自检」:拿构造出来的已知答案(贴限位序列、
含 10° 跳变的序列、恒定目标)验金属尺本身,尺子不对直接退出 —— 这个仓库
为「度量本身就错」付过学费(verify-the-metric-first)。

Pink 侧配置 = 当前控制台签收默认(ori_cost 0.5、substeps 3、
relax_at_limit 0.01、relax_margin 3.0);cuRobo 侧 = CuroboVegaIK 默认。
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_REAL = ROOT / "logs/regress/base1_playlist_all13.csv"
MSGPACK_REAL = ROOT / "logs/playlist_all13.msgpack"   # CSV_REAL 的同源回放
HZ = 50.0
NEAR_DEG = 1.0      # 贴限位判据,与 dev_limit_lock 一致
JUMP_DEG = 5.0      # 跳变判据,与回归台一致
WRIST_TRK = {"right": "RWRIST", "left": "LWRIST"}   # teleop 默认序列号
WAIST_TRK = "WAIST"


# ---------------------------------------------------------------- 尺子 ----
def pinned_stats(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    near = np.radians(NEAR_DEG)
    pinned = (q <= lo + near) | (q >= hi - near)
    any_p = pinned.any(axis=1)
    best = cur = 0
    for v in any_p:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return {"pct": float(100 * any_p.mean()), "longest_s": best / HZ,
            "joint_frames": int(pinned.sum())}


def jump_stats(q: np.ndarray) -> dict:
    d = np.degrees(np.abs(np.diff(q, axis=0)))
    worst = d.max(axis=1) if len(d) else np.zeros(1)
    return {"pct5": float(100 * (worst > JUMP_DEG).mean()),
            "p99": float(np.percentile(worst, 99))}


def ang_deg(Ra, Rb) -> float:
    return float(np.degrees(np.linalg.norm(pin.log3(Ra.T @ Rb))))


def ruler_selfcheck(lo: np.ndarray, hi: np.ndarray) -> int:
    """先拿已知答案验尺子。任何一条不符,退出码非零。"""
    bad = 0
    # 贴限位:一列恒在下限上的序列必须报 100%,离限位远的必须报 0%
    q = np.tile((lo + hi) / 2, (100, 1))
    r0 = pinned_stats(q, lo, hi)
    q2 = q.copy()
    q2[:, 5] = lo[5]                    # L_arm_j6 压在下限
    r1 = pinned_stats(q2, lo, hi)
    ok = r0["pct"] == 0.0 and r1["pct"] == 100.0 and r1["longest_s"] == 2.0
    print(f"[尺] 贴限位:量程中央 {r0['pct']:.0f}%,压死下限 {r1['pct']:.0f}%"
          f"(最长 {r1['longest_s']:.1f}s)  {'✅' if ok else '❌ 尺子坏了'}")
    bad += 0 if ok else 1
    # 跳变:静止序列 0%;每 25 帧插一个 10° 台阶(4% 的帧)后,占比要
    # 抓到约 4%,p99 也必须被顶到台阶量级(2% < 4%,p99 落在台阶上)。
    j0 = jump_stats(q)
    q3 = q.copy()
    for k in range(25, 100, 25):
        q3[k:, 3] += np.radians(10.0) * (1 if (k // 25) % 2 else -1)
    j1 = jump_stats(q3)
    ok = j0["pct5"] == 0.0 and 2.0 < j1["pct5"] < 6.0 and j1["p99"] > 5.0
    print(f"[尺] 跳变:静止 {j0['pct5']:.1f}%,4% 帧含 10° 台阶时报 "
          f"{j1['pct5']:.1f}% / p99 {j1['p99']:.1f}°"
          f"  {'✅' if ok else '❌ 尺子坏了'}")
    bad += 0 if ok else 1
    return bad


# ------------------------------------------------------------- 素材 ----
def fk_pose(model, data, q, frame):
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    M = data.oMf[model.getFrameId(frame)]
    return M.translation.copy(), M.rotation.copy()


def quat_wxyz(R) -> list:
    q = pin.Quaternion(R)
    return [q.w, q.x, q.y, q.z]


def material_synthetic(model, data, q_home) -> list:
    """双手画圆 + 掌向摆动(±25°,腕量程内),10 秒 500 帧。"""
    home = {h: fk_pose(model, data, q_home, EE_FRAME[h]) for h in EE_FRAME}
    frames = []
    for i in range(500):
        t = i / HZ
        ph = 2 * np.pi * t / 10.0
        tgt = {}
        for h in EE_FRAME:
            p0, R0 = home[h]
            p = p0 + np.array([0.04 * np.sin(ph),
                               0.08 * (np.cos(ph) - 1.0),
                               0.08 * np.sin(ph)])
            R = R0 @ pin.exp3(np.array([0.0, np.radians(25.0) * np.sin(ph),
                                        0.0]))
            tgt[h] = (p, quat_wxyz(R))
        frames.append(tgt)
    return frames


def material_overrange(model, data, q_home) -> list:
    """右腕绕 EE 红色轴拧 150°(超 j6 量程)保持 6 秒,回可达 3 秒。
    左手钉在 HOME。场景与 dev_relax_latch 第 4 条相同。"""
    home = {h: fk_pose(model, data, q_home, EE_FRAME[h]) for h in EE_FRAME}
    p0, R0 = home["right"]
    axis = R0[:, 0]
    R_bad = pin.exp3(np.radians(150.0) * axis) @ R0
    frames = []
    for i in range(300):
        frames.append({"right": (p0, quat_wxyz(R_bad)),
                       "left": (home["left"][0], quat_wxyz(home["left"][1]))})
    for i in range(150):
        frames.append({"right": (p0, quat_wxyz(R0)),
                       "left": (home["left"][0], quat_wxyz(home["left"][1]))})
    return frames


def material_from_joint_csv(model, data, pin_names, csv_path,
                            limit_frames=None) -> list:
    """把关节 CSV(回归台格式,cmd_<关节名> 列,50Hz)FK 成 6D 腕目标流。

    可复用:换一份素材 = 换 csv_path 重跑(S3 就是这么用)。返回与合成
    素材同构的帧列表 [{hand: (pos, quat_wxyz)}, ...]。"""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if limit_frames:
        rows = rows[:limit_frames]
    cols = {n: f"cmd_{n}" for n in pin_names}
    frames = []
    for r in rows:
        q = np.array([float(r[cols[n]]) for n in pin_names])
        tgt = {}
        for h in EE_FRAME:
            p, R = fk_pose(model, data, q, EE_FRAME[h])
            tgt[h] = (p.copy(), quat_wxyz(R))
        frames.append(tgt)
    return frames


# ------------------------------------------------ 素材 4:臂形的尺 ----
def operator_swivel_series(msgpack_path, n_frames) -> list:
    """回放素材的操作者回转角流 [{hand: rad|None}, ...],按 sample_at(i/HZ)
    与 50Hz 目标流对齐(与 teleop 回放消费同一采样法)。"""
    from magicdexmate.sources.pico_source import PicoLogSource
    from magicdexmate.swivel import operator_swivel_from_frame
    src = PicoLogSource(str(msgpack_path))
    out = []
    for i in range(n_frames):
        fr = src.sample_at(i / HZ)
        trk = fr.trackers or {}
        out.append({h: operator_swivel_from_frame(
            h, fr.body24, trk.get(WRIST_TRK[h]), trk.get(WAIST_TRK))
            for h in EE_FRAME})
    return out


def robot_swivel_series(model, data, qs) -> list:
    """解出的 q 流 -> 机器人回转角流(同一 swivel_angle 定义)。"""
    from magicdexmate.swivel import swivel_angle
    from pink_vega_ik import ELBOW_FRAME, SHOULDER_FRAME
    fid = {h: tuple(model.getFrameId(f) for f in
                    (SHOULDER_FRAME[h], ELBOW_FRAME[h], EE_FRAME[h]))
           for h in EE_FRAME}
    out = []
    for q in qs:
        pin.framesForwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        out.append({h: swivel_angle(data.oMf[a].translation,
                                    data.oMf[b].translation,
                                    data.oMf[c].translation)
                    for h, (a, b, c) in fid.items()})
    return out


def _wrap_deg(a: float) -> float:
    return float(np.degrees((a + np.pi) % (2 * np.pi) - np.pi))


def swivel_diff_stats(rob: list, op: list) -> dict:
    """逐帧 |机器人回转角 - 操作者回转角|(wrap 后,度)。任一侧取不到的
    帧不计,n 如实报。"""
    out = {}
    for h in EE_FRAME:
        d = [abs(_wrap_deg(r[h] - o[h]))
             for r, o in zip(rob, op) if r[h] is not None and o[h] is not None]
        out[h] = {"mean": float(np.mean(d)) if d else float("nan"),
                  "max": float(np.max(d)) if d else float("nan"),
                  "n": len(d)}
    return out


def revisit_stats(frames: list, rob: list, pos_tol_m=0.02, ori_tol_deg=10.0,
                  gap_s=5.0, stride=5) -> dict:
    """同一腕目标重访时臂形(回转角)的可复现性。

    帧对判据:同手目标位置差 < 2cm 且朝向差 < 10°,间隔 > 5s;帧按 stride
    下采样找对(全量两两配对是 O(n²) 的 47M 对,下采样 5 倍后 1.9M 对,
    对上的量级不变)。返回 {hand: {mean, max, pairs}},度。"""
    out = {}
    idx = np.arange(0, len(frames), stride)
    gap = int(gap_s * HZ)
    for h in EE_FRAME:
        P = np.array([frames[i][h][0] for i in idx])
        Rm = np.array([_rot_from_wxyz(frames[i][h][1]) for i in idx])
        sw = [rob[i][h] for i in idx]
        diffs = []
        for a in range(len(idx)):
            if sw[a] is None:
                continue
            close = np.where(np.linalg.norm(P[a + 1:] - P[a], axis=1)
                             < pos_tol_m)[0] + a + 1
            for b in close:
                if (idx[b] - idx[a]) <= gap or sw[b] is None:
                    continue
                if ang_deg(Rm[a], Rm[b]) < ori_tol_deg:
                    diffs.append(abs(_wrap_deg(sw[a] - sw[b])))
        out[h] = {"mean": float(np.mean(diffs)) if diffs else float("nan"),
                  "max": float(np.max(diffs)) if diffs else float("nan"),
                  "pairs": len(diffs)}
    return out


def shape_ruler_selfcheck() -> int:
    """臂形尺的已知答案自检:尺子不对,素材 4 的数字没有意义。"""
    bad = 0
    # 逐帧差:机器人流 = 操作者流 + 10°,mean 与 max 都必须是 10
    op = [{"right": 0.3 + 0.1 * np.sin(i / 7), "left": None}
          for i in range(200)]
    rob = [{"right": v["right"] + np.radians(10.0), "left": 0.0} for v in op]
    r = swivel_diff_stats(rob, op)
    ok = (abs(r["right"]["mean"] - 10.0) < 1e-6
          and abs(r["right"]["max"] - 10.0) < 1e-6
          and r["right"]["n"] == 200 and r["left"]["n"] == 0)
    print(f"[尺] 臂形逐帧差:+10° 偏移报 mean {r['right']['mean']:.2f}° "
          f"max {r['right']['max']:.2f}°(None 侧 n={r['left']['n']})"
          f"  {'✅' if ok else '❌ 尺子坏了'}")
    bad += 0 if ok else 1
    # 重访可复现性:同一目标位姿访问两次、第二次回转角差 30°,必须被抓到
    p = np.array([0.4, -0.2, 1.1])
    quat = [1.0, 0.0, 0.0, 0.0]
    far = (np.array([0.1, 0.5, 0.8]), [0.0, 1.0, 0.0, 0.0])
    n = 700     # 前 100 帧在目标 A,中间去别处,最后 100 帧回目标 A
    frames = []
    rob = []
    for i in range(n):
        at_a = i < 100 or i >= n - 100
        # 左臂目标每帧挪 1cm:间隔 >250 帧的两帧相距 >2.5m,凑不成对 ——
        # 否则恒定目标会造出 24 万个零差帧对,白算十几秒
        frames.append({"right": (p, quat) if at_a else far,
                       "left": (p + np.array([0.5 + 0.01 * i, 0.5, 0.5]), quat)})
        rob.append({"right": (np.radians(30.0) if i >= n - 100 else 0.0),
                    "left": 0.0})
    r = revisit_stats(frames, rob, stride=1)
    ok = r["right"]["pairs"] > 0 and abs(r["right"]["max"] - 30.0) < 1e-6
    print(f"[尺] 臂形重访:两次访问差 30° 报 max {r['right']['max']:.1f}°"
          f"({r['right']['pairs']} 对)  {'✅' if ok else '❌ 尺子坏了'}")
    bad += 0 if ok else 1
    return bad


# ------------------------------------------------------------- 跑一遍 ----
def run_backend(ik, frames, model, data, swivel: list | None = None) -> dict:
    """喂同一份目标流,返回全部指标。ik 需要 set_target/solve/reset_home。
    swivel 非空时逐帧喂 set_swivel_target(增益为零的后端存而不用,行为
    不变 —— pink 的 solve 只在 null_bias>0 时碰零空间(pink_vega_ik.py
    的门),curobo 同款门并有单测 test_set_swivel_target 逐位证明)。"""
    lo = model.lowerPositionLimit
    hi = model.upperPositionLimit
    ik.reset_home()
    qs, times = [], []
    perr, oerr = [], []
    for i, tgt in enumerate(frames):
        for h, (p, quat) in tgt.items():
            ik.set_target(h, p, quat)
            if swivel is not None:
                ik.set_swivel_target(h, swivel[i][h])
        t0 = time.perf_counter()
        q = ik.solve()
        times.append(time.perf_counter() - t0)
        qs.append(q.copy())
        for h, (p, quat) in tgt.items():
            pe, Re = fk_pose(model, data, q, EE_FRAME[h])
            perr.append(np.linalg.norm(pe - p) * 1000)
            R_tgt = _rot_from_wxyz(quat)
            oerr.append(ang_deg(Re, R_tgt))
    qs = np.array(qs)
    times = np.array(times) * 1000
    perr, oerr = np.array(perr), np.array(oerr)
    return {
        "q": qs,
        "位置误差mm": (float(perr.mean()), float(perr.max())),
        "朝向误差deg": (float(oerr.mean()), float(oerr.max())),
        "贴限位": pinned_stats(qs, lo, hi),
        "跳变": jump_stats(qs),
        "耗时ms": (float(np.percentile(times, 50)),
                 float(np.percentile(times, 95))),
    }


def _rot_from_wxyz(quat) -> np.ndarray:
    w, x, y, z = quat
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def print_table(name: str, res: dict):
    print(f"\n== {name} ==")
    hdr = (f"{'后端':8s} {'位置误差mm(均/最大)':>22s} {'朝向误差deg(均/最大)':>22s} "
           f"{'贴限位%':>8s} {'最长s':>6s} {'跳变>5°%':>9s} {'跳变p99°':>9s} "
           f"{'耗时ms(p50/p95)':>16s}")
    print(hdr)
    for backend, r in res.items():
        print(f"{backend:8s} "
              f"{r['位置误差mm'][0]:10.1f}/{r['位置误差mm'][1]:8.1f}   "
              f"{r['朝向误差deg'][0]:10.1f}/{r['朝向误差deg'][1]:8.1f}   "
              f"{r['贴限位']['pct']:8.2f} {r['贴限位']['longest_s']:6.1f} "
              f"{r['跳变']['pct5']:9.2f} {r['跳变']['p99']:9.2f} "
              f"{r['耗时ms'][0]:8.2f}/{r['耗时ms'][1]:6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=0,
                    help="截短真实素材到前 N 帧(0 = 全部 6889 帧)")
    ap.add_argument("--skip-real", action="store_true",
                    help="跳过真实素材(快速冒烟)")
    args = ap.parse_args()

    bad = 0

    # 账本模型(与两个后端共用同一 URDF 与关节序)
    ref = PinkVegaIK(dt=1.0 / HZ)
    model, data = ref.model, ref.data
    pin_names = ref.pin_names
    q_home = ref.q_home.copy()
    del ref

    bad += ruler_selfcheck(model.lowerPositionLimit.copy(),
                           model.upperPositionLimit.copy())
    bad += shape_ruler_selfcheck()
    if bad:
        print("尺子不过,后面的数字没有意义,退出。")
        return 1

    # 素材 3 的入口自检:CSV 第一行就是 HOME,FK 出来的腕位姿必须与
    # HOME FK 一致(<1mm),否则是列名/关节序读错了。
    if not args.skip_real:
        if not CSV_REAL.exists():
            print(f"❌ 缺素材 {CSV_REAL}")
            return 1
        with open(CSV_REAL) as fh:
            row0 = next(csv.DictReader(fh))
        q0 = np.array([float(row0[f"cmd_{n}"]) for n in pin_names])
        p_csv, _ = fk_pose(model, data, q0, EE_FRAME["right"])
        p_home, _ = fk_pose(model, data, q_home, EE_FRAME["right"])
        d0 = np.linalg.norm(p_csv - p_home) * 1000
        ok = d0 < 1.0
        print(f"[尺] 真实素材首帧 FK 距 HOME {d0:.2f}mm"
              f"  {'✅' if ok else '❌ CSV 解析错了'}")
        bad += 0 if ok else 1
        if bad:
            return 1

    # 两个后端。Pink 用控制台签收默认;cuRobo 用自身默认。
    print("\n构建后端 ...")
    t0 = time.time()
    pink = PinkVegaIK(dt=1.0 / HZ, orientation_cost=0.5, substeps=3,
                      relax_at_limit=0.01, relax_margin_deg=3.0)
    t_pink = time.time() - t0
    t0 = time.time()
    from curobo_vega_ik import CuroboVegaIK
    curobo = CuroboVegaIK(dt=1.0 / HZ)
    t_cur = time.time() - t0
    print(f"构建耗时:pink {t_pink:.1f}s  curobo {t_cur:.1f}s(含预热首解;"
          "冷 warp 缓存首次约 17s)")
    backends = {"pink": pink, "curobo": curobo}

    # ---- 稳态耗时基准 ---------------------------------------------------
    # 预热首解已吃在构建里,这里量的是 50Hz 控制环真正面对的稳态单步耗时
    # (cuRobo 侧含 ZMQ 往返,即部署路径)。这是 S6 架构选型的硬门槛数:
    # 环给 IK 的预算约 5ms。数字如实打,不达标不藏。
    print("\n== 稳态单步耗时(目标=HOME 位姿保持,300 步) ==")
    for n, ik in backends.items():
        ik.reset_home()
        home_tgt = {h: fk_pose(model, data, q_home, EE_FRAME[h])
                    for h in EE_FRAME}
        ts = []
        for _ in range(300):
            for h, (p, R) in home_tgt.items():
                ik.set_target(h, p, quat_wxyz(R))
            t0 = time.perf_counter()
            ik.solve()
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts) * 1000
        p50, p95 = np.percentile(ts, 50), np.percentile(ts, 95)
        # 自检:稳态 p95 必须进 50Hz 的控制周期(20ms),否则这个后端在
        # 该机器上根本追不上环,后面的精度对比失去意义。
        ok = p95 < 20.0
        print(f"  {n:8s} p50 {p50:6.2f}ms  p95 {p95:6.2f}ms  "
              f"p99 {np.percentile(ts, 99):6.2f}ms  max {ts.max():6.2f}ms"
              f"  {'✅ < 20ms 周期' if ok else '❌ 追不上 50Hz 环'}"
              + ("" if p50 < 5.0 else "  ⚠ 超 5ms 预算"))
        bad += 0 if ok else 1

    # ---- 素材 1:可达合成轨迹 -------------------------------------------
    frames = material_synthetic(model, data, q_home)
    res1 = {n: run_backend(ik, frames, model, data)
            for n, ik in backends.items()}
    print_table("素材 1:可达合成轨迹(圆弧 + 掌向摆动,10 秒)", res1)
    for n, r in res1.items():
        ok = r["位置误差mm"][0] < 5.0
        print(f"  [自检] {n} 可达素材位置误差均值 {r['位置误差mm'][0]:.1f}mm < 5mm"
              f"  {'✅' if ok else '❌ 连可达目标都跟不住'}")
        bad += 0 if ok else 1

    # ---- 素材 2:超量程 + 恢复 ------------------------------------------
    frames = material_overrange(model, data, q_home)
    res2 = {}
    for n, ik in backends.items():
        res2[n] = run_backend(ik, frames, model, data)
        # 恢复判据:需求回到可达 3 秒后,右腕朝向误差 < 5°、全臂距限位 > 1°
        q_end = res2[n]["q"][-1]
        _, R_end = fk_pose(model, data, q_end, EE_FRAME["right"])
        _, R_home = fk_pose(model, data, q_home, EE_FRAME["right"])
        e_end = ang_deg(R_end, R_home)
        slack_end = float(np.degrees(np.minimum(
            q_end - model.lowerPositionLimit,
            model.upperPositionLimit - q_end).min()))
        res2[n]["恢复"] = (e_end, slack_end)
    print_table("素材 2:超量程拧腕 6 秒 + 回可达 3 秒(dev_relax_latch 场景)", res2)
    for n, r in res2.items():
        e_end, slack_end = r["恢复"]
        ok = e_end < 5.0 and slack_end > 1.0
        print(f"  [自检] {n} 需求回量程 3 秒后:朝向误差 {e_end:.1f}°"
              f"  全臂距限位最近 {slack_end:.1f}°"
              f"  {'✅ 能回来' if ok else '❌ 回不来 —— 顶限位锁死'}")
        bad += 0 if ok else 1

    # ---- 素材 3:真实素材 -----------------------------------------------
    if not args.skip_real:
        frames = material_from_joint_csv(model, data, pin_names, CSV_REAL,
                                         args.frames or None)
        res3 = {n: run_backend(ik, frames, model, data)
                for n, ik in backends.items()}
        print_table(f"素材 3:真实素材 playlist_all13({len(frames)} 帧,"
                    "FK 反造目标)", res3)
        # 自检:素材本身由 pink 生成(cmd 已带防护),pink 重解不应偏出
        # 太多;这里只挡「完全失控」(均值 < 30mm),对比数字照表如实读。
        for n, r in res3.items():
            ok = r["位置误差mm"][0] < 30.0
            print(f"  [自检] {n} 真实素材位置误差均值 {r['位置误差mm'][0]:.1f}mm"
                  f" < 30mm  {'✅' if ok else '❌ 失控'}")
            bad += 0 if ok else 1

        # ---- 素材 4:臂形跟随(目标流同素材 3 + 操作者骨架) -------------
        if not MSGPACK_REAL.exists():
            print(f"❌ 缺回放素材 {MSGPACK_REAL}(臂形指标需要骨架)")
            return 1
        ops = operator_swivel_series(MSGPACK_REAL, len(frames))
        avail = float(100 * np.mean([
            all(o[h] is not None for h in EE_FRAME) for o in ops]))
        ok = avail > 90.0
        print(f"\n[尺] 操作者回转角可得率 {avail:.1f}%(双手同时可得的帧)"
              f"  {'✅' if ok else '❌ 骨架大面积缺失,素材 4 不可用'}")
        bad += 0 if ok else 1

        print("构建 curobo-follow(null_bias=1,零空间限速跟随)...")
        follow = CuroboVegaIK(dt=1.0 / HZ, null_bias=1.0)
        res_f = run_backend(follow, frames, model, data, swivel=ops)
        shape = {}
        for n, r in {"pink": res3["pink"], "curobo": res3["curobo"],
                     "cu-follow": res_f}.items():
            rob = robot_swivel_series(model, data, r["q"])
            shape[n] = {"d": swivel_diff_stats(rob, ops),
                        "rv": revisit_stats(frames, rob), "r": r}

        print(f"\n== 素材 4:臂形跟随(playlist_all13,{len(frames)} 帧)==")
        print(f"{'后端':10s} {'臂形差R°(均/最大)':>18s} {'臂形差L°(均/最大)':>18s} "
              f"{'重访差R°(均/最大/对)':>22s} {'重访差L°':>10s} "
              f"{'腕误差mm':>9s} {'跳变>5°%':>9s}")
        for n, s in shape.items():
            d, rv, r = s["d"], s["rv"], s["r"]
            print(f"{n:10s} "
                  f"{d['right']['mean']:8.1f}/{d['right']['max']:6.1f}   "
                  f"{d['left']['mean']:8.1f}/{d['left']['max']:6.1f}   "
                  f"{rv['right']['mean']:6.1f}/{rv['right']['max']:6.1f}"
                  f"/{rv['right']['pairs']:5d}   "
                  f"{rv['left']['mean']:6.1f}/{rv['left']['max']:6.1f} "
                  f"{r['位置误差mm'][0]:9.1f} {r['跳变']['pct5']:9.2f}")

        # 自检 1:跟随的生效证明(「接好了从没执行」是这个仓库的惯犯)
        st = follow.swivel_follow_stat
        ok = st["n"] > 0
        print(f"  [自检] cu-follow 跟随步执行 {st['n']} 帧、跳过 {st['skip']} 帧"
              f"  {'✅' if ok else '❌ NEVER FIRED'}")
        bad += 0 if ok else 1
        # 自检 2:跟随必须真把臂形差往下拉(否则机制无效),同时腕不失控
        m_f = np.mean([shape['cu-follow']['d'][h]['mean'] for h in EE_FRAME])
        m_0 = np.mean([shape['curobo']['d'][h]['mean'] for h in EE_FRAME])
        ok = m_f < m_0 and res_f["位置误差mm"][0] < 30.0
        print(f"  [自检] 臂形差均值 curobo {m_0:.1f}° -> cu-follow {m_f:.1f}°,"
              f"腕误差均值 {res_f['位置误差mm'][0]:.1f}mm"
              f"  {'✅ 跟随有效且腕未失控' if ok else '❌ 跟随无效或腕被拖走'}")
        bad += 0 if ok else 1

    print(f"\n{'✅ 全部通过' if bad == 0 else f'❌ {bad} 项失败'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
