#!/usr/bin/env python3
"""回归台:同一批素材跑一遍,把整套指标记下来,和另一次跑的结果逐项对比。

改动遥操作的任何一环之前先跑一次存基线,改完再跑一次对比。这样"有没有把别处
弄坏"由数据回答,而不是靠改完之后觉得没问题。

这个仓库为缺这一步付过代价:`--pink-substeps 2` 当初在一份素材上测着无害就
进了 --fast,后来量出它省 3% 帧率、赔掉 2 倍腕精度。所以默认跑多份素材,
并且一次输出全部指标,而不是只看正在改的那一项。

对比时的判据不是"数字有没有变",而是"变化有没有超过这套流程自身的重复性
噪声"。先用同一份配置连跑两次得到噪声地板,再拿它衡量改动。

用法
    # 存基线(改动之前)
    .venv/bin/python scripts/regress_teleop.py --save /tmp/base.json

    # 改完之后再跑一次,并对比
    .venv/bin/python scripts/regress_teleop.py --save /tmp/after.json \
        --extra "--null-bias 1.0"
    .venv/bin/python scripts/regress_teleop.py --compare /tmp/base.json /tmp/after.json

    # 噪声地板:同一配置连跑两次,两次之间的差就是噪声
    .venv/bin/python scripts/regress_teleop.py --save /tmp/n1.json
    .venv/bin/python scripts/regress_teleop.py --save /tmp/n2.json
    .venv/bin/python scripts/regress_teleop.py --compare /tmp/n1.json /tmp/n2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ISAAC = ROOT / ".venv-isaac/bin/python"

CLIPS = ["logs/clip_headwaist.msgpack", "logs/playlist_all13.msgpack"]
# 目前签收的那套配置。改动要在同一套配置下比,否则比的是两件事。
BASE_ARGS = ("--mode trackers --tracker-left LWRIST --tracker-right RWRIST "
             "--tracker-waist WAIST --map chest-anchor --ori-mode palm "
             "--control-mode arms+head+waist --waist-mode j3 --pos-scale 1.0 "
             "--physics-device cpu --physics-hz 50 --control_hz 50 --no-sym-lock "
             # 与 vega_console.py 的默认保持一致。基准必须跟着操作者实际在跑
             # 的那一套走,否则每次回归都在测一个没人跑的配置。
             # ⚠ --relax-at-limit 0.01 已随 2026-08-10 自锁修复回到控制台默认,
             #   但**不加进 BASE_ARGS**:加了旧基线就不可比。要测防护全开,用
             #   --extra "--relax-at-limit 0.01 --relax-margin 3.0";修后基线在
             #   logs/regress/relaxfix2_{idle,load4,load4b}.json(空载 7.00% /
             #   4 核负载 7.23% / 6.92%,负载敏感性已消失)。
             "--self-collision hold")
# 改动 BASE_ARGS 会让**旧基线不可比**。旧的那批(logs/regress/base1.json 等,
# 2026-08-08)是在完全没有防护的配置下跑的。
#
# ⚠ 跑回归时**机器要空载**,或者把负载记下来。2026-08-09 那次「同配置两跑差
# 30%」量到的其实是负载差异 —— 判据是「变化有没有超过噪声」,而负载会伪装成
# 噪声。(当时以为 --relax-at-limit 对负载极度敏感,2026-08-10 查清那是触发
# 条件读全部 7 关节造成的自锁、由扰动随机触发,已修;但「回归要控制负载条件」
# 这条教训本身仍然成立。)

PATS = {
    "右腕误差均值": r"right track err while engaged: mean\s+([\d.]+)mm",
    "右腕误差最大": r"right track err while engaged: mean\s+[\d.]+mm\s+max\s+([\d.]+)mm",
    "左腕误差均值": r"left track err while engaged: mean\s+([\d.]+)mm",
    "左腕误差最大": r"left track err while engaged: mean\s+[\d.]+mm\s+max\s+([\d.]+)mm",
    "头方位误差": r"head gaze fidelity.*?az ([\d.]+)deg",
    "头仰角误差": r"head gaze fidelity.*?el ([\d.]+)deg",
    "腰行程j3": r"torso travel:.*?torso_j3=([\d.]+)deg",
    "实时倍率": r"real-time factor: ([\d.]+)x",
}


def jitter(csv_path: pathlib.Path) -> dict:
    """逐帧最大关节跳变。抖动就量在这里。"""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    cols = [c for c in rows[0] if c.startswith("cmd_") and "_arm_j" in c]
    q = np.array([[float(r[c]) for c in cols] for r in rows])
    d = np.degrees(np.abs(np.diff(q, axis=0)))
    worst = d.max(axis=1)
    out = {"跳变中位": float(np.median(worst)),
           "跳变p99": float(np.percentile(worst, 99)),
           "跳变>5度占比%": float(100 * (worst > 5).mean())}
    # 大跳落在哪个关节,用来判断是不是零空间那三个
    big = np.where(worst > 5)[0]
    if len(big):
        names = np.array(cols)[d[big].argmax(axis=1)]
        j5 = sum(1 for n in names if n.endswith("j5"))
        out["大跳中j5占比%"] = float(100 * j5 / len(big))
    return out


def run_one(clip: str, extra: str, duration: float, keep: str = "") -> dict:
    # 关节 CSV 要留下:抖动到底是「关节动了末端也动」还是「关节白动」,
    # 只能事后对它做正向运动学才分得出,而这正是验收的关键一项。
    tmp = (pathlib.Path(keep) if keep
           else pathlib.Path(tempfile.mkdtemp()) / "j.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
               VEGA_WELD_WHEELS="1", VEGA_TORSO_HOME="legacy")
    cmd = (f"{ISAAC} -u sim/teleop_vega_pico.py --source replay --replay-file {clip} "
           f"{BASE_ARGS} {extra} --headless --duration {duration} --joint-csv {tmp}")
    p = subprocess.run(cmd, shell=True, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=1800)
    txt = p.stdout + p.stderr
    res = {}
    for name, pat in PATS.items():
        m = re.search(pat, txt, re.S)
        res[name] = float(m.group(1)) if m else None
    if tmp.exists():
        res.update(jitter(tmp))
    res["_missing"] = [k for k, v in res.items() if v is None]
    return res


def clip_seconds(clip: str) -> float:
    import msgpack
    t = [m.get("t_us", 0) / 1e6 for m in
         msgpack.Unpacker(open(ROOT / clip, "rb"), raw=False)]
    return (t[-1] - t[0]) if t else 0.0


def save(args) -> int:
    out = {"配置": BASE_ARGS + " " + args.extra, "素材": {}}
    for clip in args.clips:
        secs = clip_seconds(clip) + 2
        print(f"[回归] {clip}({secs:.0f} 秒)运行中 ...", flush=True)
        tag = pathlib.Path(args.save).stem + "_" + pathlib.Path(clip).stem
        keep = str(pathlib.Path(args.save).parent / f"{tag}.csv")
        out["素材"][clip] = run_one(clip, args.extra, secs, keep)
        out["素材"][clip]["_csv"] = keep
        r = out["素材"][clip]
        print(f"        腕误差 {r.get('右腕误差均值')}/{r.get('左腕误差均值')} mm  "
              f"跳变 p99 {r.get('跳变p99')}°  >5° {r.get('跳变>5度占比%')}%")
        if r["_missing"]:
            print(f"        未解析到:{r['_missing']}")
    pathlib.Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[回归] 已写入 {args.save}")
    return 0


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(pathlib.Path(a_path).read_text())
    b = json.loads(pathlib.Path(b_path).read_text())
    print(f"  基线配置:{a['配置']}")
    print(f"  对比配置:{b['配置']}\n")
    for clip in a["素材"]:
        if clip not in b["素材"]:
            continue
        print(f"  {clip}")
        ra, rb = a["素材"][clip], b["素材"][clip]
        for k in ra:
            if k.startswith("_") or ra[k] is None or rb.get(k) is None:
                continue
            d = rb[k] - ra[k]
            rel = (100 * d / ra[k]) if ra[k] else 0.0
            print(f"    {k:16s} {ra[k]:9.3f} -> {rb[k]:9.3f}   "
                  f"{d:+8.3f} ({rel:+6.1f}%)")
        print()
    print("  判据:先用同一配置连跑两次得到噪声地板,变化超过噪声才算真的变了。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", help="跑一遍并把指标写到这个文件")
    ap.add_argument("--compare", nargs=2, metavar=("基线", "对比"))
    ap.add_argument("--clips", nargs="+", default=CLIPS)
    ap.add_argument("--extra", default="", help="追加给仿真程序的参数")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.save:
        ap.error("需要 --save 或 --compare")
    return save(args)


if __name__ == "__main__":
    sys.exit(main())
