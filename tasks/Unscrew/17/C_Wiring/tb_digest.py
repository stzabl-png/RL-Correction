"""训练看板摘要 (守夜用): 读 logs/<name>/ 下 TB 事件, 打印关键针最近值. 用法: $PY tb_digest.py logs/<name> [tags...]"""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d = sys.argv[1]
KEYS = sys.argv[2:] or ["sr/gate1", "sr/gate2", "sr/gate3", "sr/placed", "sr/gate4", "sr_t0/gate1", "sr_t0/gate3",
                        "ep_rew/pull", "ep_rew/leash", "ep_rew/ms", "ep_rew/pen", "diag/pull_N", "diag/pull_detach",
                        "diag/pull_phantom_N", "diag/cap_any", "diag/cl_on", "diag/carry_steps", "diag/escort_fail",
                        "diag/fall_peak", "term/M4_success", "curr/ema_gate1", "curr/p_t0", "Mean Rewards", "rewards/step", "episode_lengths/step"]
files = sorted(glob.glob(os.path.join(d, "**", "events.out.tfevents.*"), recursive=True), key=os.path.getmtime)
if not files:
    print("no events under", d); sys.exit(0)
ea = EventAccumulator(os.path.dirname(files[-1]), size_guidance={"scalars": 0}); ea.Reload()
tags = set(ea.Tags().get("scalars", []))
steps = 0
for k in KEYS:
    if k in tags:
        ev = ea.Scalars(k); last = ev[-1]; steps = max(steps, last.step)
        recent = [e.value for e in ev[-5:]]
        print(f"{k:24s} step {last.step:>10d}  last {last.value:9.4f}   recent5 {[round(v, 3) for v in recent]}")
print(f"[digest] {d}: {len(tags)} tags, latest step {steps}")
miss = [k for k in KEYS if k not in tags]
if miss: print("  (缺针:", ", ".join(miss[:12]), ")")
