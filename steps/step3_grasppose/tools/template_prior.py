"""模板经验先验 —— 用过的物体越多, 选模板越准越快。

问题(2026-08-15): 给一个新物体选抓取模板, 原来两条路都不好使 ——
  * VLM 语义(`select_grasp_template.py`): 它的 `n_contact_fingers`/`palm_contact`/`contact_depth`
    做硬过滤会误杀。实测 pour/17 杯子, VLM 说"4 指、掌心不参与、指尖接触", 而实测冠亚军
    `18_Extensior_Type`(5 指+掌) 和 `1_Large_Diameter`(5 指+掌) 全被砍掉。
  * 全模板扫描: 准, 但 29 个模板要 6 分钟, 每来一个新物体都重跑一遍。

做法: 记住"哪类物体上哪个模板好用", 下次优先测它, 够用就停。不训练, 纯统计, 可解释。

档位键 = (形状, 物体接触带直径档, 是否空心) —— 三个维度都是实测有区分力的:
  * 直径档: 手能张多开是绝对量, 所以按物体接触带的绝对直径分档
    ★别用"模板隐含直径/物体直径"这个比值 —— 它每个模板各不相同, 同一个物体会落进
      多个档位, 查询时根本无从取舍。第一版这么做, 实测瓶子因此查到了杯子的档位
  * 是否空心: 腔内闸只对空心物体生效(杯子实测 65 个候选因此被拒), 实心物体恒不触发
  * 形状: 来自 VLM

★ 三个设计上的坑, 都是实测教训:
  1. 存**覆盖率**而不是胜场数 —— 赢在 66.8% 和赢在 35% 含金量差一倍
  2. 排序用**胜率/均值**而不是累计次数 —— 否则被测得多的天然占优, 会自我锁死
  3. **必须留探索位**(ε-greedy): 只测高分模板 + 早停 = 永远发现不了更好的。
     第一个物体的偶然结果会被无限放大。

用法:
    # 查询: 给档位, 返回按期望覆盖率排序的模板
    python tools/template_prior.py query --shape cylinder --obj-dia-mm 78 --hollow

    # 回写: 一次实测结果
    python tools/template_prior.py update --shape cylinder --obj-dia-mm 78 --hollow \\
        --tmpl 18_Extensior_Type --cov 0.668

    # 看全表
    python tools/template_prior.py show
"""

import argparse
import json
import os
import random

PRIOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "template_prior.json")

# ★档位必须只由**物体**决定, 不能掺模板的量。第一版用了"模板隐含直径/物体直径"这个比值,
#   但它每个模板各不相同, 同一个物体会落进多个档位, 查询时无从取舍(实测瓶子因此查到了
#   杯子档位的先验, 试了 5 个模板全军覆没)。改用物体接触带的绝对直径分档。
DIA_BINS = [(0, 35, "细"), (35, 60, "中"), (60, 90, "粗"), (90, 9999, "很粗")]


def dia_bin(dia_mm):
    for lo, hi, nm in DIA_BINS:
        if lo <= dia_mm < hi:
            return nm
    return "很粗"


def key_of(shape, obj_dia_mm, hollow):
    return f"{shape}|{dia_bin(obj_dia_mm)}|{'空心' if hollow else '实心'}"


def load():
    if os.path.isfile(PRIOR):
        return json.load(open(PRIOR))
    return {"schema": "template_prior_v1", "buckets": {}}


def save(d):
    os.makedirs(os.path.dirname(PRIOR), exist_ok=True)
    json.dump(d, open(PRIOR, "w"), ensure_ascii=False, indent=1)


def update(d, key, tmpl, cov, rej=None):
    b = d["buckets"].setdefault(key, {})
    e = b.setdefault(tmpl, {"n": 0, "cov_sum": 0.0, "cov_max": 0.0, "n_rej": 0})
    e["n"] += 1
    if rej:
        e["n_rej"] += 1
        e["last_rej"] = rej
    else:
        e["cov_sum"] += float(cov)
        e["cov_max"] = max(e["cov_max"], float(cov))
    return d


def rank(d, key, candidates=None, explore=0.15, seed=None):
    """→ [(tmpl, 期望覆盖率, 测过几次)], 已含探索位。candidates=粗筛保留的模板名列表。"""
    b = d["buckets"].get(key, {})
    known, unknown = [], []
    for t in (candidates if candidates else list(b)):
        e = b.get(t)
        if e and e["n"] > e["n_rej"]:
            known.append((t, e["cov_sum"] / max(e["n"] - e["n_rej"], 1), e["n"]))
        elif e and e["n_rej"] >= e["n"]:
            known.append((t, -1.0, e["n"]))          # 测过且总是被拒 -> 排最后
        else:
            unknown.append((t, None, 0))
    known.sort(key=lambda x: -x[1])
    out = known + unknown
    if explore > 0 and unknown:
        rnd = random.Random(seed)
        pick = rnd.choice(unknown)                    # ★探索位: 强制插一个没测过的
        out = [x for x in out if x[0] != pick[0]]
        ins = min(len(known), 2)                      # 插在前几名之后, 保证会被测到
        out = out[:ins] + [pick] + out[ins:]
    return out


def stop_threshold(d, key, frac=0.85, floor=0.30):
    """够用就停的阈值: 该档位历史最高 × frac, 但不低于 floor。"""
    b = d["buckets"].get(key, {})
    best = max([e["cov_max"] for e in b.values()] or [0.0])
    return max(best * frac, floor)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("query", "update"):
        p = sub.add_parser(c)
        p.add_argument("--shape", default="cylinder")
        p.add_argument("--obj-dia-mm", type=float, required=True,
                       help="物体接触带直径(mm)。★只能是物体的量, 不能用模板的隐含直径")
        p.add_argument("--hollow", action="store_true")
        if c == "update":
            p.add_argument("--tmpl", required=True)
            p.add_argument("--cov", type=float, default=0.0)
            p.add_argument("--rej", default=None, help="被拒原因(cavity/thumb_down/...)")
        else:
            p.add_argument("--candidates", nargs="*", default=None)
            p.add_argument("--explore", type=float, default=0.15)
    sub.add_parser("show")
    a = ap.parse_args()

    d = load()
    if a.cmd == "show":
        for k, b in d["buckets"].items():
            print(f"\n【{k}】  阈值 {stop_threshold(d, k):.1%}")
            for t, e in sorted(b.items(), key=lambda kv: -(kv[1]["cov_sum"] / max(kv[1]["n"] - kv[1]["n_rej"], 1))):
                ok = e["n"] - e["n_rej"]
                mean = e["cov_sum"] / ok if ok else 0.0
                print(f"   {t:<26} n={e['n']:<3} 均值 {mean:6.1%}  最高 {e['cov_max']:6.1%}"
                      + (f"  拒绝 {e['n_rej']}({e.get('last_rej','')})" if e["n_rej"] else ""))
        return
    key = key_of(a.shape, a.obj_dia_mm, a.hollow)
    if a.cmd == "update":
        save(update(d, key, a.tmpl, a.cov, a.rej))
        print(f"[prior] {key}  {a.tmpl}  cov={a.cov:.1%}" + (f"  拒绝={a.rej}" if a.rej else ""))
    else:
        print(f"档位 {key}   够用阈值 {stop_threshold(d, key):.1%}")
        for i, (t, c, n) in enumerate(rank(d, key, a.candidates, a.explore), 1):
            tag = "未测过(探索)" if c is None else (f"{c:.1%} (n={n})" if c >= 0 else "总被拒")
            print(f"  {i:>3}. {t:<26} {tag}")


if __name__ == "__main__":
    main()
