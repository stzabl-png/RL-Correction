#!/usr/bin/env python
"""闸门自证 —— 证明 check_world_fingerprint 不是恒返回"通过"的空断言。

规矩(来自 RL session 的判据自检家族): **造一个必然该红的输入, 看它红不红。**
闸门比采集器更需要反证 —— 采集器读错值只是记了个错数, **闸门读错值会放行一个
本该拦住的 ckpt**。

四例:
  1 同一世界            -> 绿 (退出 0)   证明不是恒红
  2 USD md5 被改        -> 红 (退出 1)   证明关键项真能拦
  3 files 段整段缺失    -> 无法核对 (退 1)  ★ 2026-08-29 实测漏放过: 曾报"完全一致"
  4 schema 不认识       -> 无法核对 (退 1)  旧格式 world.json 的形状

跑法: python tools/selftest_world_fingerprint.py
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from check_world_fingerprint import record  # noqa: E402


def run(expect: dict, tmp: Path, name: str):
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(expect, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, str(REPO / "tools" / "check_world_fingerprint.py"),
                        "check", "--expect", str(p)],
                       capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout


def main() -> int:
    now = record(REPO)
    tmp = Path(tempfile.mkdtemp())
    cases, ok = [], True

    # 1 同一世界 -> 必须绿
    rc, out = run(now, tmp, "same")
    cases.append(("① 同一世界 -> 绿", rc == 0, rc, "通过" in out or "一致" in out))

    # 2 USD md5 被改 -> 必须红
    bad = copy.deepcopy(now)
    bad["effective_robot_usd"]["md5"] = "0" * 32
    rc, out = run(bad, tmp, "badusd")
    cases.append(("② USD md5 不符 -> 红", rc == 1, rc, "effective_robot_usd" in out))

    # 3 ★ files 段整段缺失 -> 必须"无法核对"(曾在此漏放)
    hole = {"schema": "world_fingerprint_v1",
            "effective_robot_usd": {"md5": now["effective_robot_usd"]["md5"]}}
    rc, out = run(hole, tmp, "nofiles")
    cases.append(("③ files/env 缺失 -> 无法核对", rc == 1 and "无法核对" in out, rc,
                  "无法核对" in out))

    # 4 schema 不认识(旧格式 world.json 的形状) -> 必须"无法核对"
    oldfmt = {"variant": "OBJ", "usd": "default", "obs_dim": 503, "act_dim": 58}
    rc, out = run(oldfmt, tmp, "oldfmt")
    cases.append(("④ 旧格式 world.json -> 无法核对", rc == 1 and "无法核对" in out, rc,
                  "无法核对" in out))

    print("闸门自证 —— 造必然该红的输入, 看它红不红\n")
    for name, passed, rc, detail in cases:
        print(f"  {'✅' if passed else '❌'} {name:34s} 退出码 {rc}  判据命中 {detail}")
        ok &= passed
    print("\n" + ("✅ 四例全过 —— 闸门不是恒真断言, 缺数据不会被当成通过"
                  if ok else "❌ 有用例失败 —— 闸门可能在该拦的输入上放行"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
