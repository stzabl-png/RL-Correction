#!/usr/bin/env python
"""闸门自证 —— 证明 check_world_fingerprint 不是恒返回"通过"的空断言。

规矩(来自 RL session 的判据自检家族): **造一个必然该红的输入, 看它红不红。**
闸门比采集器更需要反证 —— 采集器读错值只是记了个错数, **闸门读错值会放行一个
本该拦住的 ckpt**。

七例:
  1 同一世界            -> 绿 (退出 0)   证明不是恒红
  2 USD md5 被改        -> 红 (退出 1)   证明关键项真能拦
  3 files 段整段缺失    -> 无法核对 (退 1)  ★ 2026-08-29 实测漏放过: 曾报"完全一致"
  4 schema 不认识       -> 无法核对 (退 1)  旧格式 world.json 的形状
  5 判据**阈值**变了    -> 红 (退 1)     且要指名是哪个键、从多少变成多少
  6 判据**清单扩了**但阈值没变 -> 不硬拦 (退 0)  ★ 只给哈希分不出这两者
  7 本机读不到判据摘要  -> 无法核对 (退 1)

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

    # 5 判据阈值变了 -> 必须红, 且指名道姓
    thr = copy.deepcopy(now)
    thr["criteria"]["digest"] = "deadbeefdeadbeef"
    thr["criteria"]["items"] = dict(thr["criteria"]["items"] or {})
    thr["criteria"]["items"]["D1_DROP"] = 0.08          # 真值 0.05
    rc, out = run(thr, tmp, "thr")
    cases.append(("⑤ 判据阈值变了 -> 红", rc == 1 and "D1_DROP" in out, rc,
                  "0.08" in out and "0.05" in out))

    # 6 清单扩了但阈值没变 -> 不该硬拦(这正是只给哈希时分不出的那种)
    ext = copy.deepcopy(now)
    ext["criteria"]["digest"] = "cafecafecafecafe"
    ext["criteria"]["items"] = {k: v for k, v in (ext["criteria"]["items"] or {}).items()}
    ext["criteria"]["items"].pop("D1_DROP", None)       # 期望比现状少一项 = 现状"新增"了它
    rc, out = run(ext, tmp, "ext")
    cases.append(("⑥ 清单扩了/阈值没变 -> 不硬拦", rc == 0 and "新增键" in out, rc,
                  "预期内变化" in out))

    # 7 本机读不到判据摘要 -> 无法核对(直接调 compare, 子进程里造不出这个场景)
    import check_world_fingerprint as CW
    blind_now = copy.deepcopy(now)
    blind_now["criteria"] = {"schema": None, "digest": None, "items": None,
                             "_source": "unavailable: 模拟"}
    d7 = CW.compare(blind_now, now)
    hit7 = any(x.get("item") == "criteria.digest" and x.get("verdict") == "UNVERIFIABLE"
               for x in d7)
    cases.append(("⑦ 读不到判据摘要 -> 无法核对", hit7, "-", hit7))

    print("闸门自证 —— 造必然该红的输入, 看它红不红\n")
    for name, passed, rc, detail in cases:
        print(f"  {'✅' if passed else '❌'} {name:34s} 退出码 {rc}  判据命中 {detail}")
        ok &= passed
    print("\n" + ("✅ 七例全过 —— 闸门不是恒真断言; 该红时红、该静时静、缺数据不当成通过"
                  if ok else "❌ 有用例失败 —— 闸门可能在该拦的输入上放行"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
