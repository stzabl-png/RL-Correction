"""L5-25 自检第十一件: 禁碰判据 collide_flags。

约定: **只有手垫与自己要操作的物体可以接触**, 其余一切接触都是禁碰。
判别句: 每一类禁碰都造一个只触发它的输入, 并配"不该红"的对照
(全零 / 阈值以下 / 手垫只碰自己物体), 证明它不是逢触必红。
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress_batch import collide_flags  # noqa: E402

THR = 1.0
KEYS = ("arm", "pad", "objobj", "d6")


def mk(arm=0.0, pad_net=0.0, pad_fil=0.0, oo=0.0, d6=0.0, N=2):
    v = lambda x, k: torch.full((N, k, 3), float(x) / (3 ** 0.5))  # noqa: E731
    return dict(net_arm=v(arm, 8), net_pad=torch.full((N, 10), float(pad_net)),
                fil_pad=torch.full((N, 10), float(pad_fil)),
                fm_objobj=v(oo, 1), fm_d6=v(d6, 4))


def run(tag, want, **kw):
    f = collide_flags(**mk(**kw), thr=THR)
    hit = sorted(k for k in KEYS if bool(f[k].any()))
    good = hit == sorted(want)
    print(f"  {'✅' if good else '★✗'} {tag:34s} 点名={hit}  期望={sorted(want)}")
    return good


ok = True
print("不该红的对照:")
ok &= run("全零(无任何接触)", [])
ok &= run("手垫只碰自己物体(净=滤=10N)", [], pad_net=10.0, pad_fil=10.0)
ok &= run("各处力都在阈值以下(0.4N)", [], arm=0.4, pad_net=0.4, oo=0.4, d6=0.4)
ok &= run("手垫净比滤只多 0.5N(阈下)", [], pad_net=10.5, pad_fil=10.0)

print("\n每类禁碰单独触发:")
ok &= run("臂节/掌根碰到东西 5N", ["arm"], arm=5.0)
ok &= run("手垫碰到自物体以外(净12 滤3)", ["pad"], pad_net=12.0, pad_fil=3.0)
ok &= run("瓶碰杯 5N", ["objobj"], oo=5.0)
ok &= run("左右两侧互撞 5N (修好的D6)", ["d6"], d6=5.0)

print("\n组合与边界:")
ok &= run("臂+物物同时", ["arm", "objobj"], arm=3.0, oo=3.0)
ok &= run("四类全中", list(KEYS), arm=3.0, pad_net=9.0, pad_fil=1.0, oo=3.0, d6=3.0)

# 空段(POUR_NO_D6 配方下这些传感器不存在)不得崩、不得误报
e3 = torch.zeros(2, 0, 3)
f = collide_flags(e3, torch.zeros(2, 10), torch.zeros(2, 10), e3, e3, thr=THR)
good = not any(bool(f[k].any()) for k in KEYS)
ok &= good
print(f"\n  {'✅' if good else '★✗'} 空传感器段不崩且不误报")

print("\n★覆盖边界: 臂只装了远端 4 节/侧(l5/l7/l8/掌根), 近端 l1~l4 无传感器 ——"
      "\n  本判据**不能声称检测到了全部碰撞**, 只覆盖远端可达段。")
print("✅ 第十一件自检通过: 禁碰判据可红可绿" if ok else "★自检未通过 —— 见上")
sys.exit(0 if ok else 1)
