"""L5-26 自检第十二件: 判据摘要必须"该红时红、该静时静"。

它替代的是整文件哈希。整文件哈希的病: 分不出"判定变了"与"记账变了" ——
2026-08-29 一天内本目录被改三次全是纯记账, 整文件哈希会三次都把历史 ckpt 判红。
★一天误报三次的闸门, 人第二天就开始无视它 —— 比没有闸门更糟, 因为它还额外
 提供"我们有闸门"的错觉。

所以这件自检必须证明两个方向都成立, 缺一不可:
  该红: 改任意一个阈值 -> digest 必变
  该静: 加计数器 / 改奖励权重 -> digest 必不变
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import progress as P            # noqa: E402
import progress_batch as PB     # noqa: E402

ok = True
s0, d0 = P.criteria_digest()
print(f"  基线: schema={s0} digest={d0} 覆盖 {len(P.criteria_items())} 项")

# ① 确定性
same = all(P.criteria_digest()[1] == d0 for _ in range(5))
ok &= same
print(f"  {'✅' if same else '★✗'} 连算 5 次结果一致(确定性)")

# ② ★该红: 逐个阈值扰动, 每一个都必须让 digest 变
# ★L5-32: 从白名单**自动派生**, 不再手写。
#   手写清单在 2026-08-31 被抓到漏了 8 项(M3_ROT/M3_HOLD/M4_ARM/M4_HOLD/
#   CERT_RAMP/CERT_HOLD/CERT_RET/CERT_WAIT), 而当天新加的倒水几何三常量
#   也没进去 —— 也就是说"逐个阈值都该红"这句话过去只对**一半**阈值成立。
#   派生之后, 任何进白名单的新阈值自动被扫, 清单不可能再追不掉队。
CONSTS = [k for k, v in P.criteria_items().items()
          if not isinstance(v, dict) and v is not None]
bad = []
for c in CONSTS:
    old = getattr(P, c)
    setattr(P, c, (old * 1.01 + 1e-6) if isinstance(old, float) else old + 1)
    if P.criteria_digest()[1] == d0:
        bad.append(c)
    setattr(P, c, old)
ok &= not bad
print(f"  {'✅' if not bad else '★✗'} 该红: {len(CONSTS)} 个阈值逐个扰动都让 digest 变"
      + (f"  ★漏掉: {bad}" if bad else ""))
# 字典型阈值
for c in ("LEASH_POS", "LEASH_ROT"):
    old = dict(getattr(P, c))
    m = dict(old); m[2] = (m[2] * 1.01 if m[2] is not None else 0.1)
    setattr(P, c, m)
    good = P.criteria_digest()[1] != d0
    ok &= good
    setattr(P, c, old)
    print(f"  {'✅' if good else '★✗'} 该红: {c} 改档位值让 digest 变")
assert P.criteria_digest()[1] == d0, "扰动后没还原干净"

# ③ ★该静: 奖励权重不得进摘要 (它们是"给多少分", 不是"算不算成功")
quiet = []
for c in ("MS_REWARD", "WAGE", "WAGE_CAP", "W_OBJ", "W_HAND"):
    if not hasattr(P, c):
        continue
    old = getattr(P, c)
    m = ({k: v * 2 for k, v in old.items()} if isinstance(old, dict) else old * 2)
    setattr(P, c, m)
    if P.criteria_digest()[1] != d0:
        quiet.append(c)
    setattr(P, c, old)
ok &= not quiet
print(f"  {'✅' if not quiet else '★✗'} 该静: 奖励权重翻倍不影响 digest"
      + (f"  ★误报: {quiet}" if quiet else ""))

# ④ ★该静: 加计数器(今天真实发生过三次)不得影响 digest
B = PB.PourProgressBatch.__new__(PB.PourProgressBatch)
B._acc = {"ep": 0, "自造计数器": 0}
after = P.criteria_digest()[1]
ok &= (after == d0)
print(f"  {'✅' if after == d0 else '★✗'} 该静: 新增记账字段不影响 digest")

# ⑤ schema 与 digest 分开暴露(消费方要能区分"清单扩了" vs "阈值变了")
sep = isinstance(s0, int) and isinstance(d0, str) and len(d0) == 16
ok &= sep
print(f"  {'✅' if sep else '★✗'} schema(int) 与 digest(16位hex) 分开暴露")

# ⑥ items 表可用于精确 diff(不只给"变了/没变")
it = P.criteria_items()
has = all(k in it for k in ("D1_DROP", "CERT_RISE", "M4_DIST_ROT"))
none_rew = not any(k in it for k in ("MS_REWARD", "WAGE", "W_OBJ", "W_HAND"))
ok &= has and none_rew
print(f"  {'✅' if has and none_rew else '★✗'} items 含判据阈值且不含奖励权重")

print("\n✅ 第十二件自检通过: 判据摘要该红时红、该静时静"
      if ok else "\n★自检未通过 —— 见上")
sys.exit(0 if ok else 1)
