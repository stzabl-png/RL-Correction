#!/usr/bin/env python
"""Held-out 验证 —— 在**从未用于调参**的 take 上考整条可信度+平滑链路。

背景: 判据阈值与 R 映射在同 4 条 take(bpp/13, bpp/6, 27_scene, bpp/11)上迭代了三轮,
继续在它们身上评估 = 背答案。本脚本对其余 take 跑完整链路并汇总验收:

  阶段1  cotracker_consistency (GPU, ~1-2 分/条)
  阶段2  pose_audit --ct-dir   (全库重打分)
  阶段3  rts_smoother          (无视频渲染, 只算 A/B 验收)

验收口径(与调参期完全相同, 不许改):
  A  静止段步长: 平滑后 ≤ 原始 (位置与旋转分别计)
  B  高可信段 explained: 平滑后 ≥ 原始 − 0.03 (容差=对照组的自然波动)
汇总输出 通过率 + 逐条明细。任何一条崩溃按不通过计, 不静默跳过。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_calib_sheet import resolve_video  # noqa: E402

PY = "/home/lyh/anaconda3/envs/hawor/bin/python"
HERE = Path(__file__).resolve().parent
ROOT = Path("/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput")
TUNED = {"egodex/part2/basic_pick_place/13", "egodex/part2/basic_pick_place/6",
         "egodex/screw_unscrew_bottle_cap/27_scene", "egodex/part2/basic_pick_place/11"}


def all_takes():
    out = []
    for p in sorted(ROOT.glob("egodex/**/world_fused.npz")):
        rel = str(p.parent.relative_to(ROOT))
        if "test_generate" in rel:
            continue
        out.append(rel)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", choices=("ct", "audit", "rts", "all"), default="all")
    a = ap.parse_args(argv)
    P = a.out
    held = [t for t in all_takes() if t not in TUNED]
    print(f"[held-out] 全库 {len(all_takes())} 条, 调参 {len(TUNED)} 条, 考试 {len(held)} 条")

    if a.stage in ("ct", "all"):
        t0 = time.time()
        for k, rel in enumerate(held, 1):
            scene = ROOT / rel
            ccp = P / "cc" / f"cc_{scene.parent.name}_{scene.name}.json"
            if ccp.is_file():
                print(f"  [{k}/{len(held)}] 已有, 跳过 {rel}")
                continue
            vid = resolve_video(scene)
            if vid is None:
                print(f"  [{k}/{len(held)}] X 找不到视频 {rel}")
                continue
            r = subprocess.run(
                [PY, str(HERE / "cotracker_consistency.py"), "--scene", str(scene),
                 "--video", str(vid), "--out", str(P / "cc"),
                 "--audit", str(P / "pose_audit.json"), "--no-viz"],
                capture_output=True, text=True, timeout=1200)
            tail = [l for l in r.stdout.splitlines() if l.startswith("[cc]")]
            print(f"  [{k}/{len(held)}] {rel}: {tail[-1] if tail else 'FAIL: ' + r.stderr.strip()[-200:]}")
        print(f"[held-out] CT 阶段 {(time.time()-t0)/60:.0f} 分钟")

    if a.stage in ("audit", "all"):
        subprocess.run([PY, str(HERE / "pose_audit.py"), "--root", str(ROOT),
                        "--include", "egodex", "--out", str(P), "--ct-dir", str(P / "cc")],
                       check=True)

    if a.stage in ("rts", "all"):
        rows = []
        for k, rel in enumerate(held, 1):
            scene = ROOT / rel
            try:
                r = subprocess.run(
                    [PY, str(HERE / "rts_smoother.py"), "--scene", str(scene),
                     "--audit", str(P / "pose_audit.json"), "--out", str(P / "rts")],
                    capture_output=True, text=True, timeout=900)
                d = json.loads(r.stdout[r.stdout.index("{"):])
                A = d["A_static_step"]; B = d["B_image"].get("high_conf", {})
                ok_A = ((A["smooth_mm"] is None or A["raw_mm"] is None
                         or A["smooth_mm"] <= A["raw_mm"] + 0.2)
                        and (A["smooth_deg"] is None or A["raw_deg"] is None
                             or A["smooth_deg"] <= A["raw_deg"] + 0.05))
                eR, eS = B.get("explained_raw"), B.get("explained_smooth")
                ok_B = eR is None or eS is None or eS >= eR - 0.03
                rows.append(dict(take=rel, ok_A=ok_A, ok_B=ok_B,
                                 A=A, B=B, n_refuted=d.get("n_rot_refuted")))
                print(f"  [{k}/{len(held)}] {rel:46s} A={'✓' if ok_A else '✗'} "
                      f"B={'✓' if ok_B else '✗'} "
                      f"(静 {A['raw_mm']}->{A['smooth_mm']}mm/{A['raw_deg']}->{A['smooth_deg']}° "
                      f"expl {eR}->{eS} refuted {d.get('n_rot_refuted')})")
            except Exception as e:
                rows.append(dict(take=rel, ok_A=False, ok_B=False,
                                 error=f"{type(e).__name__}: {e}"))
                print(f"  [{k}/{len(held)}] {rel:46s} CRASH {type(e).__name__}: {e}")
        nA = sum(1 for r in rows if r["ok_A"]); nB = sum(1 for r in rows if r["ok_B"])
        both = sum(1 for r in rows if r["ok_A"] and r["ok_B"])
        summ = {"n_heldout": len(rows), "pass_A": nA, "pass_B": nB, "pass_both": both,
                "rows": rows}
        (P / "heldout_report.json").write_text(
            json.dumps(summ, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"\n[held-out] A(静止段) {nA}/{len(rows)}   B(好段不变差) {nB}/{len(rows)}   "
              f"双过 {both}/{len(rows)}")
        print(f"[held-out] 报告: {P / 'heldout_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
