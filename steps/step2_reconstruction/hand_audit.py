#!/usr/bin/env python
"""手轨迹体检 —— 用"物体在动的时候手跟不跟着动"判定腕轨迹可用性。

## 为什么需要它

框架文档把手轨迹列为"最重要且从未审计"的一项:残差策略每步都在跟踪它,
但没有任何判据说它准不准。已知缺陷有两类:
  * 绝对偏置(HaWoR 腕位误差 ~20cm) —— 用相对量可以绕开;
  * **轨迹冻结**(2026-08-13 实测: EgoDex 带 `confidences` 组的变体, 腕平移全程
    只有 1~8mm, 而同目录不带该组的变体是 20~45cm) —— 绕不开, 数据本身没信息。

## 判据(纯运动学, 不需要接触标注/不需要网格)

手抓着物体时两者刚体耦合 ⟹ **速度一致**。取物体真在动的帧(排除静置段),
比较手与物的逐帧位移:

    carry_err = median_t ||v_hand(t) - v_obj(t)|| / ||v_obj(t)||     (物体动的帧)

  ≈0    手完全跟着物体走(至少有一只手应当如此, 否则物体不会动)
  ≈1    手对物体的运动毫无响应 = 冻结/错帧/张冠李戴

每条 take 取两手中的较小者 `carry_err_best`: 任务里至少有一只手在搬运物体。
判定线(2026-08-13 在 screw_unscrew_bottle_cap 29 条上标定): <0.6 可用 / >0.9 冻结。

**只诊断不修正**, 与 pose_audit.py 同规矩。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MOVE_THR_M = 0.004      # 物体逐帧位移超过它才算"在动"(4mm/帧, 15fps 下 6cm/s)
MIN_MOVING_FRAMES = 8   # 动帧太少则判据无效


def audit_take(take: Path) -> dict:
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    if "hand_trans" not in z.files:
        return {"take": str(take), "error": "no hand_trans"}
    # ⚠ hand_trans **不是腕位**, 是 MANO 的 transl: 真腕位 = hand_trans + J0
    #   (MANO 模型常量, 右 ~87mm / 左 ~95mm, 且**左右符号相反**因为左手是镜像模型)。
    #   pour17 实测左 +95.3mm / 右 −95.6mm, 全在 X 轴 —— 拿它当腕位, 左右手会各偏
    #   9.5cm 且方向相反(重建侧 2026-08-15 指出)。腕位的权威来源是
    #   `replay_world.npz` 的 joints[:, 0]。本审计工具只看**相对运动**, 常量偏移
    #   不影响结论, 但仍标注在此以免被当成腕位引用。
    ht = np.asarray(z["hand_trans"], dtype=np.float64)               # (2, T, 3) = MANO transl
    ob = np.asarray(z["object_ob_in_world_all"], dtype=np.float64)   # (n_obj, T, 4, 4)
    valid = np.asarray(z["object_valid_all"]).astype(bool)           # (n_obj, T)
    T = min(ht.shape[1], ob.shape[1])
    po = ob[0, :T, :3, 3]
    ok = valid[0, :T]

    d_obj = np.linalg.norm(np.diff(po, axis=0), axis=1)
    both = ok[:-1] & ok[1:]
    moving = (d_obj > MOVE_THR_M) & both
    out = {
        "take": str(take),
        "n_frames": int(T),
        "moving_frames": int(moving.sum()),
        "wrist_span_cm": [float(np.linalg.norm(ht[i, :T].max(0) - ht[i, :T].min(0)) * 100)
                          for i in range(ht.shape[0])],
        "obj_span_cm": float(np.linalg.norm(po[ok].max(0) - po[ok].min(0)) * 100) if ok.any() else 0.0,
    }
    if moving.sum() < MIN_MOVING_FRAMES:
        out["verdict"] = "inconclusive"          # 物体几乎没动过, 判不了
        return out

    errs = []
    for i in range(ht.shape[0]):
        v_h = np.diff(ht[i, :T], axis=0)[moving]
        v_o = np.diff(po, axis=0)[moving]
        errs.append(float(np.median(np.linalg.norm(v_h - v_o, axis=1) / np.linalg.norm(v_o, axis=1))))
    out["carry_err"] = errs
    out["carry_err_best"] = float(min(errs))
    out["carrying_hand_index"] = int(np.argmin(errs))
    out["verdict"] = ("usable" if out["carry_err_best"] < 0.6 else
                      "frozen" if out["carry_err_best"] > 0.9 else "marginal")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="扫描根目录(递归找 world_fused.npz)或单条 take")
    ap.add_argument("--json", type=Path, default=None, help="结果写到这里")
    a = ap.parse_args(argv)

    takes = ([a.root] if (a.root / "world_fused.npz").exists() else
             sorted(Path(p).parent for p in glob.glob(str(a.root / "**" / "world_fused.npz"),
                                                      recursive=True)))
    rows = [audit_take(t) for t in takes]
    rows.sort(key=lambda r: r.get("carry_err_best", 9))
    print(f"{'take':<26} {'verdict':<12} {'carry_err':>9}  {'腕跨度L/R(cm)':>16} {'物跨度':>7} {'动帧':>5}")
    for r in rows:
        if "error" in r:
            print(f"{Path(r['take']).name:<26} ERROR: {r['error']}")
            continue
        ce = r.get("carry_err_best")
        sp = r["wrist_span_cm"]
        print(f"{Path(r['take']).name:<26} {r['verdict']:<12} "
              f"{('%.2f' % ce) if ce is not None else '   -':>9}  "
              f"{sp[0]:7.1f}/{sp[1]:<7.1f} {r['obj_span_cm']:7.1f} {r['moving_frames']:5d}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
