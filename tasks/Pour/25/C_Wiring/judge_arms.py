"""五线消融晨读: 从各线 TB 拉 sr/gate*, clock_frac, ep_rew/* 末值+趋势, 一表对比。
本机跑 A 的 TB; msc 四线先 rsync logs 回来或远程跑本脚本。
用法: python3 judge_arms.py <logdir1> <logdir2> ...  (每个= logs/E2E_Pour17_X)
"""
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["sr/gate1", "sr/gate2", "sr/gate3", "sr/gate4", "prog/clock_frac",
        "ep_rew/adv", "ep_rew/leash", "ep_rew/ms", "ep_rew/pen", "ep_rew/pen6",
        "ep_rew/bonus", "curr/p_t0", "curr/phase_b"]

rows = []
for d in sys.argv[1:]:
    tb = sorted(glob.glob(d + "/stage1_tb"))
    if not tb:
        rows.append((d, None)); continue
    ea = EventAccumulator(tb[-1]); ea.Reload()
    have = set(ea.Tags()["scalars"])
    vals = {}
    for t in TAGS:
        if t in have:
            s = ea.Scalars(t)
            mid = s[len(s) // 2].value if len(s) > 2 else float("nan")
            vals[t] = (s[-1].value, mid, s[-1].step)
    rows.append((d, vals))
for d, vals in rows:
    print(f"\n===== {d} =====")
    if not vals:
        print("  (无TB)"); continue
    step = next(iter(vals.values()))[2] if vals else 0
    print(f"  agent_steps={step/1e6:.2f}M")
    for t in TAGS:
        if t in vals:
            last, mid, _ = vals[t]
            print(f"  {t:18s} 末={last:8.3f}  中程={mid:8.3f}")
