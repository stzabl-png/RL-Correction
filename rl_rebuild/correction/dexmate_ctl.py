"""DexMate 交互式控制台 — 在**第二个终端**里跑, 和 GUI 查看器并行.

原理: 它只读写 <仓库>/.env_viewer.json (纯 stdlib, 不加载 Isaac, 秒开).
env_viewer.py 每 0.5s 轮询这个文件, 所以你敲完回车约半秒就能在 GUI 里看到动作.

  # 终端 1 — GUI
  ./rl_rebuild/correction/viewer_loop.sh --clip Grasp2
  # 终端 2 — 控制器
  python3 -m rl_rebuild.correction.dexmate_ctl

命令:
  <关节名> <度数>      转关节, 例: L_arm_j1 90      (支持 Tab 补全)
  base <x> <y> <z>     整机位置 (米)
  yaw <度>             整机朝向
  list [过滤词]        列关节, 例: list arm / list torso
  show                 看当前设定
  zero <关节名>        单个归零
  reset                所有关节归零
  pause / play         暂停/恢复参考轨迹回放
  q                    退出 (不影响 GUI)
"""
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TWEAK = os.path.join(_REPO, ".env_viewer.json")
URDF = os.environ.get(
    "VEGA_URDF",
    "/home/lyh/luhr/MagicSim/Third_Party/curobo/curobo/content/assets/robot/"
    "vega_1p_sharpa/vega_1p_sharpa.urdf")


def load_joints():
    """从 URDF 读关节表: 名 -> (类型, 下限, 上限). 角度已转成度."""
    out = {}
    if not os.path.exists(URDF):
        print(f"⚠ 找不到 URDF ({URDF}), 无法校验关节名/限位")
        return out
    for j in ET.parse(URDF).getroot().findall("joint"):
        ty = j.get("type")
        if ty == "fixed":
            continue
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None:
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            if ty != "prismatic":
                lo, hi = math.degrees(lo), math.degrees(hi)
        else:
            lo = hi = None                      # continuous
        out[j.get("name")] = (ty, lo, hi)
    return out


def _filter_to_articulation(d):
    """以运行中 Articulation 的真实关节名为准 (轮子在 URDF 里有、USD 里没有)."""
    f = os.path.join(_REPO, "DEXMATE_ARTICULATION_JOINTS.txt")
    if not os.path.exists(f):
        return d
    real = {l.strip() for l in open(f) if l.strip()}
    dropped = sorted(set(d) - real)
    if dropped:
        print(f"注: {len(dropped)} 个 URDF 关节不在 Articulation 里(控制不了): {dropped}")
    return {k: v for k, v in d.items() if k in real}


JOINTS = _filter_to_articulation(load_joints())


def load_tw():
    try:
        with open(TWEAK) as f:
            return json.load(f)
    except Exception:
        return {}


def save_tw(tw):
    tmp = TWEAK + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tw, f, indent=2, ensure_ascii=False)
    os.replace(tmp, TWEAK)                      # 原子替换, 避免查看器读到半截


def dm(tw):
    return tw.setdefault("dexmate", {})


def show(tw):
    d = tw.get("dexmate", {})
    print(f"  底座 pos={d.get('base_pos')}  yaw={d.get('base_yaw_deg')}°")
    js = d.get("joints") or {}
    if not js:
        print("  关节: (全零位)")
    for k, v in sorted(js.items()):
        unit = "m" if JOINTS.get(k, ("revolute",))[0] == "prismatic" else "°"
        print(f"  {k:<32} {v:+8.2f}{unit}")
    print(f"  回放: {'⏸ 暂停' if tw.get('paused') else '▶ 播放'}")


def do_list(pat):
    hits = [(n, v) for n, v in JOINTS.items() if not pat or pat.lower() in n.lower()]
    if not hits:
        print(f"  没有匹配 '{pat}' 的关节")
        return
    print(f"  {'关节名':<34}{'类型':<11}{'范围'}")
    for n, (ty, lo, hi) in sorted(hits):
        rng = "连续(无限位)" if lo is None else (
            f"[{lo:+8.1f}, {hi:+8.1f}] {'m' if ty == 'prismatic' else '°'}")
        print(f"  {n:<34}{ty:<11}{rng}")
    print(f"  —— 共 {len(hits)} 个")


def set_joint(tw, name, val):
    if JOINTS and name not in JOINTS:
        cand = [n for n in JOINTS if name.lower() in n.lower()][:6]
        print(f"  ✗ 没有关节 '{name}'" + (f"  你是不是想要: {cand}" if cand else ""))
        return False
    ty, lo, hi = JOINTS.get(name, ("revolute", None, None))
    if lo is not None and not (lo - 1e-6 <= val <= hi + 1e-6):
        print(f"  ⚠ {val} 超出限位 [{lo:+.1f}, {hi:+.1f}] — 仍然写入, PhysX 会自己夹住")
    dm(tw).setdefault("joints", {})[name] = val
    unit = "m" if ty == "prismatic" else "°"
    print(f"  ✓ {name} = {val:+g}{unit}")
    return True


HELP = __doc__.split("命令:")[1]


def main():
    try:
        import readline
        names = list(JOINTS) + ["base", "yaw", "list", "show", "zero", "reset",
                                "pause", "play", "help", "q"]
        readline.set_completer(
            lambda t, s: ([n for n in names if n.startswith(t)] + [None])[s])
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    if not os.path.exists(TWEAK):
        print(f"⚠ {TWEAK} 不存在 — 先把 GUI 查看器跑起来")
    print(f"DexMate 控制台   目标文件: {TWEAK}")
    print(f"已加载 {len(JOINTS)} 个关节定义   输入 help 看命令, Tab 可补全关节名\n")
    show(load_tw())
    print()

    while True:
        try:
            line = input("dexmate> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出 (GUI 不受影响)")
            return
        if not line:
            continue
        p = line.split()
        c = p[0].lower()
        tw = load_tw()

        if c in ("q", "quit", "exit"):
            print("退出 (GUI 不受影响)")
            return
        if c in ("help", "h", "?"):
            print(HELP)
            continue
        if c == "show":
            show(tw)
            continue
        if c == "list":
            do_list(p[1] if len(p) > 1 else "")
            continue
        if c in ("pause", "play"):
            tw["paused"] = (c == "pause")
            save_tw(tw)
            print(f"  ✓ {'⏸ 已暂停' if tw['paused'] else '▶ 已恢复'}")
            continue
        if c == "reset":
            dm(tw)["joints"] = {}
            save_tw(tw)
            print("  ✓ 所有关节归零")
            continue
        if c == "zero" and len(p) == 2:
            dm(tw).get("joints", {}).pop(p[1], None)
            save_tw(tw)
            print(f"  ✓ {p[1]} 归零")
            continue
        if c == "base" and len(p) == 4:
            try:
                dm(tw)["base_pos"] = [float(x) for x in p[1:4]]
            except ValueError:
                print("  ✗ 用法: base <x> <y> <z>   (米)")
                continue
            save_tw(tw)
            print(f"  ✓ 底座位置 = {dm(tw)['base_pos']} m")
            continue
        if c == "yaw" and len(p) == 2:
            try:
                dm(tw)["base_yaw_deg"] = float(p[1])
            except ValueError:
                print("  ✗ 用法: yaw <度>")
                continue
            save_tw(tw)
            print(f"  ✓ 底座朝向 = {p[1]}°")
            continue
        # 默认: <关节名> <数值>
        if len(p) == 2:
            try:
                val = float(p[1])
            except ValueError:
                print(f"  ✗ '{p[1]}' 不是数字。输入 help 看命令")
                continue
            if set_joint(tw, p[0], val):
                save_tw(tw)
            continue
        print("  ✗ 看不懂。输入 help 看命令,list 看关节名")


if __name__ == "__main__":
    main()
