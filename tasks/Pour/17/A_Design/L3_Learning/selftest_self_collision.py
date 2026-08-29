"""L5-24 自检第十件: world_fingerprint 的 self_collision 必须记**生效值**。

背景 (实测事故链):
  ① 采集器用 `getattr(..., False)` 默认值 -> 读不到时记 False。"读不到"和"关着"
     是两回事, 把默认值当事实交出去 = 假值。
  ② 改成 USD 优先后, 我把 `import omni.usd` 失败写成直接 return, **吃掉了 cfg 兜底**
     —— "记未知"矫枉过正成"有兜底也不用", 同样是丢信息。
  ③ 只查 env_0 顶层 prim、不查子 prim。实测在 msc 上报"无 authored 值", 而独立
     pxr 复核到的是 False —— 少查一层就把"已知"降级成"未知"。

判别句: 任何检查都必须能说出"它在什么情况下会失败"。下面每一条都造一个必然该红的
输入, 并且**成对造反向输入**(True 和 False 各一次), 以证明它不是恒返回某一个值。
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_WF = os.path.join(_HERE, "..", "..", "C_Wiring", "world_fingerprint.py")
_spec = importlib.util.spec_from_file_location("wf", _WF)
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)


class O:
    def __init__(s, **k):
        s.__dict__.update(k)


def fake_usd(authored, on_child=False):
    """注入假的 omni.usd / pxr。on_child=True: 值只 authored 在子 prim 上。"""
    class A:
        def __init__(s, v):
            s.v = v

        def IsValid(s):
            return True

        def HasAuthoredValue(s):
            return s.v is not None

        def Get(s):
            return s.v

    class Path:
        def __init__(s, t):
            s.t = t

        @property
        def pathString(s):
            return s.t

    class P:
        def __init__(s, v, t="/World/envs/env_0/Robot"):
            s.v, s.t = v, t

        def IsValid(s):
            return True

        def GetPath(s):
            return Path(s.t)

        def GetAttribute(s, n):
            return A(s.v)

    top = P(None if on_child else authored)
    kids = [top, P(authored, "/World/envs/env_0/Robot/base")] if on_child else [top]
    m = types.ModuleType("omni.usd")
    m.get_context = lambda: O(get_stage=lambda: O(GetPrimAtPath=lambda p: top))
    o = types.ModuleType("omni")
    o.usd = m
    px = types.ModuleType("pxr")
    px.Usd = O(PrimRange=lambda pr: kids)
    sys.modules["omni"] = o
    sys.modules["omni.usd"] = m
    sys.modules["pxr"] = px


def drop_usd():
    for k in ("omni.usd", "omni", "pxr"):
        sys.modules.pop(k, None)
    sys.modules["omni"] = None          # 触发 import 异常


def env(prim="/World/envs/env_.*/Robot"):
    return O(hand=O(cfg=O(prim_path=prim)))


def cfg(v):
    return O(articulation_props=O(enabled_self_collisions=v)) if v is not None else O()


# ★必须"注入即调用": 早先把 fake_usd 放在 append 前、调用放在循环里, 结果所有用例
# 都跑在最后一次注入的 stage 上, 全体误判。测试自己也会犯"环境在别处被改掉"的错。
CASES = []


def case(name, authored, on_child, cfg_v, want, wsrc):
    fake_usd(authored, on_child=on_child)
    CASES.append((name, wf._self_collision(env(), cfg(cfg_v)), want, wsrc))


case("USD=False 而 cfg=True", False, False, True, False, "USD")
case("USD=True 而 cfg=False", True, False, False, True, "USD")
case("值只在子prim(True cfg)", False, True, True, False, "USD")
case("值只在子prim(False cfg)", True, True, False, True, "USD")
case("USD子树无值, cfg=True", None, False, True, True, "cfg")
case("USD子树无值, cfg=False", None, False, False, False, "cfg")

ok = True
for name, (val, src), want, wsrc in CASES:
    good = (val is want) and (wsrc in src)
    ok &= good
    print(f"  {'✅' if good else '★✗'} {name:24s} -> {str(val):5s} 来源={src[:44]}")

drop_usd()
for name, c, want, wsrc in [("USD不可读, cfg=False", cfg(False), False, "cfg"),
                            ("USD不可读, cfg=True", cfg(True), True, "cfg"),
                            ("两边都读不到", O(), None, "unreadable")]:
    val, src = wf._self_collision(env(), c)
    good = (val is want) and (wsrc in src)
    ok &= good
    print(f"  {'✅' if good else '★✗'} {name:24s} -> {str(val):5s} 来源={src[:44]}")

print("\n✅ 第十件自检通过: USD优先/子prim可见/cfg兜底不被吃掉/读不到记None而非False"
      if ok else "\n★自检未通过 —— 见上")
sys.exit(0 if ok else 1)
