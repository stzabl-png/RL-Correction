#!/usr/bin/env python
"""世界指纹对账 —— 跑之前 1 秒告诉你"这个 ckpt 能不能在这台机器上跑"。

解决的问题(本项目已发生至少三次):
  ckpt 把物体位置/抓握目标/参考轨迹作为**绝对数值**烧进权重, 执行时不重新感知。
  世界的数值一变(站姿/桌高/物体出生位姿/母带), 记忆就对不上 —— 但**不报任何错**,
  因为 obs 形状没变(348 维还是 348 维)。这叫"同维异义", 维度闸对它是瞎的。

用法:
  # 1) 记录当前世界(训练/采集时, 与 ckpt 存在一起)
  python tools/check_world_fingerprint.py record --out <ckpt_dir>/world_fingerprint.json

  # 2) 加载 ckpt 前对账(不匹配非零退出, 并列出差异项)
  python tools/check_world_fingerprint.py check --expect <ckpt_dir>/world_fingerprint.json

  # 3) 只看当前世界
  python tools/check_world_fingerprint.py show

设计原则(见 docs/WORLD_FINGERPRINT.md):
  - **值对 ≠ 记录可信**: 每个量带 `_source`; 读不到记 null, 绝不用默认值冒充;
  - 不依赖 Isaac —— 只读文件(USD 属性用 pxr, 有就用, 没有就标 unavailable),
    所以任何机器上都能跑, 也不占 GPU。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 需要盯的世界依赖: (键, 相对仓库根的路径, 说明)
WATCHED_FILES = [
    ("robot_usd", "assets/vega_1p_sharpa_fixedtorso.usd",
     "机器人 USD —— 换站姿会让 ckpt 的绝对数值记忆全废(2026-08-26 实测 0/64)"),
    ("robot_usd_stance0803", "assets/vega_1p_sharpa_fixedtorso_stance0803.usd",
     "AAG-F 的出生世界存档(旧站姿)。不是病文件, 是版本错配, 永久禁删"),
    ("tape_pour17_v2", "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v2.npz",
     "Pour17 母带 —— evaluator 从中取瓶口/杯口方向与静置位姿, 换版=换判据"),
    ("scene_layout_pour17", "datasets/pour17/scene_layout.json",
     "物体静置位姿(重建侧摆放量; 注意其中的桌高不进物理)"),
    ("object_0_usd", "datasets/pour17/objects/object_0.usd", "杯 视觉/primary 版"),
    ("object_1_usd", "datasets/pour17/objects/object_1.usd", "瓶 视觉/primary 版"),
    ("object_0_usd_cache", "datasets/pour17/cache/object_0.usd", "杯 烘焙物理版(aux 实际加载)"),
    ("object_1_usd_cache", "datasets/pour17/cache/object_1.usd", "瓶 烘焙物理版"),
]

ENV_OVERRIDES = ["DEXMATE_FIXED_USD", "POUR_NO_D6", "POUR_HOLD_K", "POUR_VARIANT",
                 "POUR_KCAP", "POUR_LEASH_ROT_TILT", "AUTO_LABEL_VLM_GATE"]

KNOWN_USD = {
    "f77f235df53494508fefdf1d1518af32": "旧站姿 = AAG-F 的出生世界",
    "143385ad217f10f3fe730338d748246d": "新站姿 = Pour17 当前世界",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def usd_self_collision(usd: Path) -> dict:
    """从 USD 直接读 self-collision —— cfg 里写的可能没生效, 以 USD 为准。"""
    try:
        from pxr import Usd
    except Exception:
        return {"value": None, "_source": "unavailable: pxr 未在当前解释器 "
                "(见 baseline/pour17/world/flatten_usd.sh 的引导方式)"}
    try:
        st = Usd.Stage.Open(str(usd))
        for pr in st.Traverse():
            a = pr.GetAttribute("physxArticulation:enabledSelfCollisions")
            if a and a.HasAuthoredValue():
                return {"value": bool(a.Get()), "_source": f"USD:{pr.GetPath()}"}
        return {"value": None, "_source": "USD 中无 authored 值"}
    except Exception as e:
        return {"value": None, "_source": f"unreadable: {type(e).__name__}"}


def record(repo: Path) -> dict:
    files = {}
    for key, rel, why in WATCHED_FILES:
        p = repo / rel
        if p.is_file():
            files[key] = {"path": rel, "md5": md5(p), "bytes": p.stat().st_size,
                          "_source": f"file:{rel}", "why": why}
            if key == "robot_usd" and files[key]["md5"] in KNOWN_USD:
                files[key]["known_as"] = KNOWN_USD[files[key]["md5"]]
        else:
            files[key] = {"path": rel, "md5": None, "_source": "missing", "why": why}

    env = {k: os.environ.get(k) for k in ENV_OVERRIDES}
    eff_usd = env.get("DEXMATE_FIXED_USD") or str(repo / "assets/vega_1p_sharpa_fixedtorso.usd")
    eff = Path(eff_usd)
    fp = {
        "schema": "world_fingerprint_v1",
        "repo": str(repo),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "files": files,
        "env_overrides": env,
        "effective_robot_usd": {
            "path": str(eff), "md5": md5(eff) if eff.is_file() else None,
            "_source": ("env:DEXMATE_FIXED_USD" if env.get("DEXMATE_FIXED_USD")
                        else "default:assets/vega_1p_sharpa_fixedtorso.usd"),
            "note": "★ ckpt 能不能跑, 主要看这一项。DEXMATE_FIXED_USD 会覆写默认 USD。"},
        "self_collision": usd_self_collision(eff) if eff.is_file() else
                          {"value": None, "_source": "effective USD 不存在"},
    }
    if fp["effective_robot_usd"]["md5"] in KNOWN_USD:
        fp["effective_robot_usd"]["known_as"] = KNOWN_USD[fp["effective_robot_usd"]["md5"]]
    return fp


def _git(repo: Path, *args) -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args],
                                       text=True, timeout=20).strip()
    except Exception:
        return None


CRITICAL = ["effective_robot_usd", "tape_pour17_v2", "scene_layout_pour17",
            "object_0_usd_cache", "object_1_usd_cache"]


def compare(now: dict, want: dict) -> list[dict]:
    diffs = []
    a, b = now["effective_robot_usd"], want.get("effective_robot_usd", {})
    if a.get("md5") != b.get("md5"):
        diffs.append({"item": "effective_robot_usd", "critical": True,
                      "expected": b.get("md5"), "actual": a.get("md5"),
                      "expected_known_as": b.get("known_as"),
                      "actual_known_as": a.get("known_as"),
                      "fix": ("export DEXMATE_FIXED_USD=<期望的那份 USD 的绝对路径>  "
                              "—— 不要改 ckpt, 要把世界切回它的出生世界")})
    for key in want.get("files", {}):
        e, g = want["files"][key].get("md5"), now["files"].get(key, {}).get("md5")
        if e != g:
            diffs.append({"item": f"files.{key}", "critical": key in CRITICAL,
                          "expected": e, "actual": g,
                          "why": want["files"][key].get("why")})
    for k, e in (want.get("env_overrides") or {}).items():
        g = (now.get("env_overrides") or {}).get(k)
        if e != g:
            diffs.append({"item": f"env.{k}", "critical": k == "DEXMATE_FIXED_USD",
                          "expected": e, "actual": g})
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["record", "check", "show"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--expect", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--strict", action="store_true",
                    help="任何差异都非零退出(默认只有 critical 项才拦)")
    a = ap.parse_args()

    now = record(a.repo)

    if a.mode in ("record", "show"):
        u = now["effective_robot_usd"]
        print(f"生效机器人 USD : {u['path']}")
        print(f"  md5          : {u['md5']}  {u.get('known_as', '')}")
        print(f"  来源         : {u['_source']}")
        sc = now["self_collision"]
        print(f"self_collision : {sc['value']}   来源 {sc['_source']}")
        print(f"git            : {now['git_branch']} @ {str(now['git_commit'])[:8]}")
        act = {k: v for k, v in now["env_overrides"].items() if v}
        print(f"环境覆写       : {act or '无'}")
        if a.mode == "record":
            out = a.out or (a.repo / "world_fingerprint.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(now, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"-> {out}")
        return 0

    if a.expect is None or not a.expect.is_file():
        print(f"X 需要 --expect <world_fingerprint.json>: {a.expect}", file=sys.stderr)
        return 2
    want = json.loads(a.expect.read_text(encoding="utf-8"))
    diffs = compare(now, want)
    crit = [d for d in diffs if d.get("critical")]

    if not diffs:
        print("✅ 世界指纹完全一致 —— 可以跑")
        return 0
    print(f"{'❌' if crit else '⚠'} 世界指纹有 {len(diffs)} 处差异"
          f"({len(crit)} 处关键):\n")
    for d in diffs:
        mark = "❌关键" if d.get("critical") else "⚠ "
        print(f"{mark} {d['item']}")
        print(f"     期望 {d.get('expected')} {d.get('expected_known_as') or ''}")
        print(f"     实际 {d.get('actual')} {d.get('actual_known_as') or ''}")
        if d.get("why"):
            print(f"     影响 {d['why']}")
        if d.get("fix"):
            print(f"     ✚ 修法 {d['fix']}")
    if crit:
        print("\n★ 有关键差异 —— **不要跑**。ckpt 记的是绝对数值, 世界不对就是 0 分且不报错。")
    return 1 if (crit or a.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
