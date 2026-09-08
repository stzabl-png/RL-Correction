"""汇总 eval10: 三视频 × 三臂, 每条 10 回合(开扰动). ★用户裁定(2026-09-07): 成功判据 = G3 精确倒水几何(倾角≥90°∧圆盘相交∧dz 连续25步), 不要求放回. 成功次数 = G3倒水 × 回合数; 按臂合计 /30."""
import re, glob, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "/home/lyh/Project/RL_Correction/logs/eval10"
RUNS = {  # 视频 -> {臂: run名}
    "pour17": {"Base": "PFX_ablP0", "NH": "PFX_ablNH", "NHNC": "PFX_ablNHNC"},
    "pour25": {"Base": "PFX_P25s51", "NH": "PFX_P25ablNH_s51", "NHNC": "PFX_P25ablNHNC_s51"},
    "pour31": {"Base": "PFX_P31s51", "NH": "PFX_P31ablNH_s51", "NHNC": "PFX_P31ablNHNC_s51"},
}
def parse(f):
    s = open(f, errors="ignore").read()
    m = re.search(r"★Success\(G3_pour ∧ placed, t0 口径\) = ([0-9.]+)\s+\(t0 分母 (\d+) 回合\)", s)
    d = re.search(r"分解: G3倒水 ([0-9.]+) · 放回 ([0-9.]+) · G3松 ([0-9.]+) · 走完时钟 ([0-9.]+)", s)
    g = re.search(r"sr/gate1=([0-9.]+) sr/gate2=([0-9.]+)", s)
    seed = re.search(r"seed=(\d+)", s)
    if not m: return None
    n = int(m.group(2)); rate = float(m.group(1))
    return dict(n=n, rate=rate, succ=round(float(d.group(1)) * n), succ_full=round(rate * n), g3=float(d.group(1)), placed=float(d.group(2)), loose=float(d.group(3)), clock=float(d.group(4)),
                g1=float(g.group(1)) if g else float("nan"), g2=float(g.group(2)) if g else float("nan"), seed=seed.group(1) if seed else "?")
tot = {a: [0, 0] for a in ("Base", "NH", "NHNC")}
print("判据 = G3 精确倒水 (不含放回); 括号内 = 倒水∧放回 的旧口径")
print(f"{'视频':7s} {'臂':5s} {'倒水成功/回合':>13s} {'G1':>5s} {'G2':>5s} {'放回':>5s} {'时钟':>5s} {'(倒∧放)':>7s}  run")
for v, arms in RUNS.items():
    for a, run in arms.items():
        fs = sorted(glob.glob(f"{D}/eval10_{run}_s*.log"))
        r = parse(fs[-1]) if fs else None
        if r is None: print(f"{v:7s} {a:5s} {'缺/未出':>9s}  run={run}"); continue
        tot[a][0] += r["succ"]; tot[a][1] += r["n"]
        print(f"{v:7s} {a:5s} {str(r['succ'])+'/'+str(r['n']):>13s} {r['g1']:5.2f} {r['g2']:5.2f} {r['placed']:5.2f} {r['clock']:5.2f} {str(r['succ_full'])+'/'+str(r['n']):>7s}  {run} (seed {r['seed']})")
print("\n按臂合计:")
for a, (s, n) in tot.items():
    print(f"  {a:5s} {s}/{n}  = {100*s/max(n,1):.1f}%")
