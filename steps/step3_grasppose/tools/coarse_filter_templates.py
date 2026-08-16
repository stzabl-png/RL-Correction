"""粗筛模板 —— 只砍"几何上根本不可能"的, 不做语义判断。

为什么要有这一步(2026-08-15): 原来的 `select_grasp_template.py` 用 VLM 的
`contact_depth`/`palm_contact` 做**硬过滤**, 一刀砍掉 12 个模板, 其中就有实测最好用的
`1_Large_Diameter`(杯子上 60 个可用) —— VLM 说"指尖接触", 而所有环握模板 depth 都是
"full", 天然不兼容。而且尺寸闸已经判定"改按环握族出候选"、放开了 palm, 却忘了放开 depth,
逻辑自相矛盾。

所以这里只用两条**几何**判据, 语义留给后面的扫描和覆盖率排序去裁决:

  ① 接触手指数  |模板指数 − 目标指数| <= tol
  ② 隐含物体直径 模板接触点做圆拟合得到的直径, 与物体接触带直径之比在 [lo, hi] 内
     —— 太细的模板抓不住粗物体(手指闭不拢), 太粗的抓不住细物体(够不着)

隐含直径的算法: 接触法向都指向物体轴心, 所以柱轴 = 法向矩阵的最小奇异向量;
在垂直柱轴的平面里对接触点做 Kasa 圆拟合, 得到直径。

用法:
  python tools/coarse_filter_templates.py --hand assets/hand/sharpa_wave_v2_left \\
      --obj-diameter-mm 78 [--digits 4] [--tol 2] [--dia-lo 0.5] [--dia-hi 2.0]
"""

import argparse
import glob
import json
import os

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def implied_diameter(cpn):
    """接触点(n,6: 位置+法向) -> 隐含物体直径(m)。"""
    P, N = np.asarray(cpn, float)[:, :3], np.asarray(cpn, float)[:, 3:6]
    if len(P) < 3:
        return float(np.linalg.norm(P[0] - P[-1])) if len(P) == 2 else 0.0
    c = P.mean(0)
    u = np.linalg.svd(N)[2][-1]                    # 柱轴 = 与全部法向最正交的方向
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(u, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    XY = np.stack([(P - c) @ e1, (P - c) @ e2], 1)
    A = np.concatenate([2 * XY, np.ones((len(XY), 1))], 1)
    s = np.linalg.lstsq(A, (XY ** 2).sum(1), rcond=None)[0]
    return 2.0 * float(np.sqrt(max(s[2] + s[:2] @ s[:2], 1e-6)))


def n_digits(bodies):
    s = set()
    for x in bodies:
        x = str(x).replace("left_", "").replace("right_", "")
        for k in FINGERS:
            if x.startswith(k):
                s.add(k)
    return len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--obj-diameter-mm", type=float, required=True,
                    help="物体在接触带处的直径(mm)")
    ap.add_argument("--digits", type=int, default=None, help="目标接触手指数(VLM/几何给); 不给则不筛")
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--dia-lo", type=float, default=0.5)
    ap.add_argument("--dia-hi", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{a.hand}/init_tmpl/*.npy")):
        nm = os.path.basename(f)[:-4]
        t = np.load(f, allow_pickle=True).item()
        nd = n_digits(t["hand_cbody"])
        dia = implied_diameter(t["hand_cpn_w"]) * 1000
        ratio = dia / a.obj_diameter_mm
        ok_d = True if a.digits is None else abs(nd - a.digits) <= a.tol
        ok_s = a.dia_lo <= ratio <= a.dia_hi
        rows.append(dict(tmpl=nm, digits=nd, dia_mm=round(dia, 1),
                         ratio=round(ratio, 2), ok=bool(ok_d and ok_s),
                         why="" if ok_d and ok_s else ("指数" if not ok_d else "直径")))
    keep = [r for r in rows if r["ok"]]
    rows.sort(key=lambda r: (not r["ok"], abs(r["ratio"] - 1)))
    print(f"物体接触带直径 {a.obj_diameter_mm:.0f}mm"
          + (f", 目标指数 {a.digits}±{a.tol}" if a.digits else "")
          + f", 直径比容许 [{a.dia_lo}, {a.dia_hi}]\n")
    print(f"{'模板':<26}{'指数':>4}{'隐含直径':>9}{'比值':>7}  结果")
    for r in rows:
        print(f"{r['tmpl']:<26}{r['digits']:>4}{r['dia_mm']:>8.1f}mm{r['ratio']:>7.2f}  "
              f"{'✓' if r['ok'] else '✗ ' + r['why']}")
    print(f"\n保留 {len(keep)}/{len(rows)}")
    print(" ".join(r["tmpl"] for r in keep))
    if a.json:
        json.dump(rows, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
