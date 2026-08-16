"""模板扫描汇总: 每个模板在**当前物体摆放**下能出什么样的抓取。

为什么需要它(2026-08-15): 选模板原来靠 VLM+几何打分(`select_grasp_template.py`), 但那套
判断的是"人是怎么抓的", 没有回答"这只手在这个摆放下**做不做得出来**"。实测 pour/17 的杯子:
`fingertip_large` 立着时 98% 的产出是虎口朝下(拇指接触点比小指低 5.4cm), 只有抓杯口那 8 个
是正常的 —— 这种事只有真跑一遍才知道。

判据(与 tools/filter_hand_orientation.py 一致):
    thumb_z = ±R[2,1]   右手取正, 左手取负
    >= -0.25 算"虎口朝上/水平"
★ 已验证 thumb_z 与"拇指接触点高度 − 小指接触点高度"相关系数 0.954, 是个有效代理量。

用法:
    python tools/scan_templates.py --glob 'output/scan_*_sharpa_wave_v2_left' \\
        [--table -0.056] [--top -0.076] [--csv out.csv]
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="实验目录通配, 如 'output/scan_*_sharpa_wave_v2_left'")
    ap.add_argument("--side", default="auto", choices=("left", "right", "auto"))
    ap.add_argument("--min-thumb-z", type=float, default=-0.25)
    ap.add_argument("--table", type=float, default=None, help="桌面 z(m), 用于换算相对高度")
    ap.add_argument("--top", type=float, default=None, help="物体顶 z(m)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for exp in sorted(glob.glob(a.glob)):
        m = re.search(r"scan_(.+?)_sharpa_wave", os.path.basename(exp))
        tmpl = m.group(1) if m else os.path.basename(exp)
        fs = sorted(glob.glob(f"{exp}/grasp_data/**/*_grasp.npy", recursive=True))
        if not fs:
            rows.append(dict(tmpl=tmpl, n=0))
            continue
        tz, ch, dth = [], [], []
        for f in fs:
            d = np.load(f, allow_pickle=True).item()
            q = np.asarray(d["grasp_qpos"], float).reshape(-1)
            side = a.side
            if side == "auto":
                side = "left" if str(d.get("hand_name", "")).endswith("_left") else "right"
            sgn = -1.0 if side == "left" else 1.0
            tz.append(sgn * float(R.from_quat(np.roll(q[3:7], -1)).as_matrix()[2, 1]))
            C = np.asarray(d["hand_cpn_w"], float)[:, :3]
            ch.append(float(np.median(C[:, 2])))
            b = [str(x).replace("left_", "").replace("right_", "") for x in d["hand_cbody"]]

            def h(nm):
                i = [k for k, x in enumerate(b) if x == nm]
                return float(C[i[0], 2]) if i else np.nan
            dth.append(h("thumb_DP") - h("pinky_DP"))
        tz, ch, dth = np.array(tz), np.array(ch), np.array(dth)
        ok = tz >= a.min_thumb_z
        rows.append(dict(tmpl=tmpl, n=len(fs), pass_frac=float(ok.mean()),
                         n_pass=int(ok.sum()), tz_med=float(np.median(tz)),
                         ch_med=float(np.median(ch)),
                         ch_pass=float(np.median(ch[ok])) if ok.any() else np.nan,
                         dth_med=float(np.nanmedian(dth))))

    rows.sort(key=lambda r: -(r.get("n_pass", 0)))
    hdr = (f"{'模板':<26}{'产量':>5}{'过虎口':>7}{'比例':>7}"
           f"{'thumb_z中位':>11}{'接触高度中位':>12}{'过闸者高度':>11}{'拇指-小指':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["n"] == 0:
            print(f"{r['tmpl']:<26}{0:>5}{'-':>7}{'-':>7}{'-':>11}{'-':>12}{'-':>11}{'-':>10}")
            continue
        def cm(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "   -   "
            if a.table is not None and a.top is not None:
                return f"{(x - a.table) / (a.top - a.table) * 100:6.0f}%"
            return f"{x*100:+6.1f}"
        print(f"{r['tmpl']:<26}{r['n']:>5}{r['n_pass']:>7}{r['pass_frac']:>6.0%}"
              f"{r['tz_med']:>+11.2f}{cm(r['ch_med']):>12}{cm(r['ch_pass']):>11}"
              f"{r['dth_med']*100:>+9.1f}cm")
    if a.table is not None:
        print("\n接触高度以 %物高 表示 (0%=桌面, 100%=物体顶)")
    print("拇指-小指 = 拇指接触点高度 − 小指接触点高度; 负 = 虎口朝下")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.csv}")


if __name__ == "__main__":
    main()
