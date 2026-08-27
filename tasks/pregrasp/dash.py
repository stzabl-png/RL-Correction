"""训练看板: 终端里持续刷新 5 个最重要的指标。

用法 (另开一个终端, 不需要 GPU, 不启 Isaac):
    python3 tasks/pregrasp/dash.py <run名>
    python3 tasks/pregrasp/dash.py Minimal_L --log ~/minimal.log

5 个指标 (为什么是这 5 个, 见每行的判读):
  ① 离目标 d_pos   —— 手离 GraspPose 还有多远。**这是任务本身**, 它不降就是没在学。
  ② 到位率 arrive  —— 训练期进 1cm 窗口的比例。带探索噪声, 会高估。
  ③ 确定性评测     —— 唯一可信的成功率口径 (每 100 epoch 一次)。
  ④ 臂离桌余量     —— 负值 = 穿桌 = 硬终止。掉到 0 附近说明在贴着桌子蹭。
  ⑤ 回合奖励       —— 总体趋势; 剧烈甩动 = 学习不稳。
"""
import argparse, glob, os, re, sys, time

p = argparse.ArgumentParser()
p.add_argument("run", help="run 名 (logs/<run>/)")
p.add_argument("--log", default=None, help="训练 stdout 日志 (取评测行与奖励)")
p.add_argument("--every", type=float, default=10.0, help="刷新间隔秒")
p.add_argument("--root", default="logs")
p.add_argument("--once", action="store_true", help="只打印一次 (自检用)")
a = p.parse_args()

from tensorboard.backend.event_processing import event_accumulator as ea

BAR = "─" * 66


def newest_event(run):
    g = glob.glob(os.path.join(a.root, run, "**", "events.out*"), recursive=True)
    return max(g, key=os.path.getmtime) if g else None


def series(acc, tag):
    if tag not in acc.Tags()["scalars"]:
        return []
    return [(x.step, x.value) for x in acc.Scalars(tag)]


def last(sr, n=1):
    return sr[-n][1] if len(sr) >= n else None


def trend(sr, k=20):
    """近 k 点 vs 前 k 点, 返回 (箭头, 变化量)。"""
    if len(sr) < 2 * k:
        return "·", 0.0
    new = sum(v for _, v in sr[-k:]) / k
    old = sum(v for _, v in sr[-2 * k:-k]) / k
    d = new - old
    return ("↓" if d < 0 else "↑" if d > 0 else "→"), d


def fmt(v, unit="", w=8, nd=2):
    return f"{'--':>{w}}{unit}" if v is None else f"{v:>{w}.{nd}f}{unit}"


while True:
    ev = newest_event(a.run)
    if sys.stdout.isatty():
        os.system("clear")
    print(f"  {a.run}    刷新 {time.strftime('%H:%M:%S')}    (Ctrl-C 退出)")
    print(BAR)
    if not ev:
        print(f"  等待 {a.root}/{a.run}/ 出现 TensorBoard 事件文件 ...")
    else:
        acc = ea.EventAccumulator(ev, size_guidance={"scalars": 0})
        acc.Reload()
        d = series(acc, "approach/d_pos_cm")
        ar = series(acc, "approach/arrive_rate")
        gap = series(acc, "diag/arm_table_gap_cm")
        rew = series(acc, "rewards/step") or series(acc, "rewards/iter")
        step = d[-1][0] if d else (ar[-1][0] if ar else 0)

        td, dd = trend(d)
        ta, da = trend(ar)

        print(f"  步数 {step/1e6:>6.2f}M")
        print(BAR)
        print(f"  ① 离目标 d_pos   {fmt(last(d), ' cm')}   {td} {dd:+.2f}      "
              f"{'← 在靠近' if dd < -0.05 else '← 没在降' if dd > -0.01 else ''}")
        print(f"  ② 到位率(训练)   {fmt(last(ar), '  ', nd=3)}   {ta} {da:+.3f}      "
              f"{'← 带噪声, 会高估' if last(ar) else ''}")

        sr, best, at = None, None, None
        if a.log and os.path.exists(os.path.expanduser(a.log)):
            txt = open(os.path.expanduser(a.log), "rb").read().decode("utf8", "ignore")
            m = re.findall(r"\[eval\] ep(\d+) ([\d.]+)M \| 确定性成功率\s+([\d.]+)%.*?最佳\s+([\d.]+)%", txt)
            if m:
                at, _, sr, best = m[-1][0], m[-1][1], float(m[-1][2]), float(m[-1][3])
            r = re.findall(r"Mean Rewards:\s*([-\d.]+)", txt)
            if r:
                rew = [(0, float(x)) for x in r[-60:]]
        print(f"  ③ 确定性评测     {fmt(sr, ' %')}   " +
              (f"最佳 {best:.2f}%  @ep{at}" if sr is not None else "(每100 epoch一次, 还没到)"))
        print(f"  ④ 臂离桌余量     {fmt(last(gap), ' cm')}   "
              f"{'⛔ 穿桌' if (last(gap) or 1) < 0 else '⚠ 贴着桌' if (last(gap) or 1) < 0.3 else ''}")
        tr_, dr = trend(rew, 10)
        print(f"  ⑤ 回合奖励       {fmt(last(rew))}   {tr_} {dr:+.1f}      "
              f"{'← 甩动剧烈' if abs(dr) > 20 else ''}")
    print(BAR)
    if a.once:
        break
    time.sleep(a.every)
