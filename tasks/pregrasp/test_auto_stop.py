"""auto_stop.StopDecider 的判据单测 —— **不需要 Isaac/GPU**, 秒级跑完.

  PYTHONPATH=. python -m tasks.pregrasp.test_auto_stop

改判据 (阈值语义 / 课程门 / 计数顺序) 之后必须重跑这个. 覆盖的都是真会踩的坑:
计数顺序反了、课程期误停、课程停摆时停不下来、蜜月尖峰误判 solved。
"""
from tasks.pregrasp.auto_stop import StopDecider


def run(name, seq, expect, **kw):
    """seq 元素: (sr, curr_ok[, stalled[, steps]]); expect = (触发在第几条, 原因前缀) 或 None"""
    s = StopDecider(**kw)
    fired = None
    for i, item in enumerate(seq):
        sr, ok = item[0], item[1]
        stalled = item[2] if len(item) > 2 else False
        steps = item[3] if len(item) > 3 else 10 ** 9
        r = s.update(sr, ok, stalled, steps)
        if r and fired is None:
            fired = (i, r.split(":")[0])
    status = "PASS" if fired == expect else "FAIL"
    print(f"[{status}] {name}\n        期望 {expect}  实际 {fired}  (best={s.best_sr:.3f})")
    return status == "PASS"


def main():
    ok = True

    # ---- 基本判据 ----
    ok &= run("课程期不触发 (sr=1.0 但课程未退火完)",
              [(1.0, False)] * 12, None)
    ok &= run("solved 需要连续 2 次",
              [(0.90, True), (0.995, True), (0.98, True), (0.991, True), (0.992, True)],
              (4, "solved"))
    ok &= run("plateau 在 5 次未提升后触发",
              [(0.60, True)] + [(0.605, True)] * 6, (5, "plateau"))
    ok &= run("持续爬升 (每次 +2pt) 不误停",
              [(0.30 + 0.02 * i, True) for i in range(12)], None)
    ok &= run("单次蜜月尖峰不判 solved",
              [(0.70, True), (0.995, True), (0.72, True), (0.73, True)], None)
    ok &= run("课程到位前的高 sr 不计入 hits",
              [(0.995, False)] * 5 + [(0.995, True)], None)
    # 计数顺序回归: best_sr 必须在比较**之后**更新, 否则微涨序列判不出平台期
    ok &= run("微涨 (< delta) 仍算平台期",
              [(0.50 + 0.002 * i, True) for i in range(8)], (5, "plateau"))

    # ---- 课程停摆分支 ----
    # 这条是 gate 的漏洞回归: 课程驱动量就是成功率, sr 卡低位 -> 课程冻住 ->
    # curriculum_done 永假 -> 最该早停的 run 反而永远停不下来
    ok &= run("绝望 run (sr 卡 2%, 课程停摆) 能停",
              [(0.02, False, True)] * 7, (5, "plateau"))
    ok &= run("课程推进期平坦不停 (任务正在变难)",
              [(0.40, False, False)] * 10, None)
    ok &= run("课程停摆时高 sr 也不判 solved",
              [(1.0, False, True)] * 3, None)
    ok &= run("推进与停摆交替不误停",
              [(0.40, False, i % 2 == 0) for i in range(14)], None)

    # ---- 步数下限 ----
    ok &= run("min_steps 内不停",
              [(0.02, False, True, 1_000_000)] * 10, None, min_steps=20_000_000)
    ok &= run("过了 min_steps 才停",
              [(0.02, False, True, 1_000_000)] * 6 + [(0.02, False, True, 25_000_000)],
              (6, "plateau"), min_steps=20_000_000)

    # ---- 只报第一次 ----
    s = StopDecider()
    n = sum(1 for _ in range(12) if s.update(0.5, True, False, 10 ** 9))
    ok &= (n == 1)
    print(f"[{'PASS' if n == 1 else 'FAIL'}] 触发只报一次 (实际 {n} 次)")

    print("\n全部通过" if ok else "\n有用例失败")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
