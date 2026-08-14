"""从 Video Prior 编译**关键帧链** —— 训练顺序、目标、容差的唯一来源。

============================ 这是什么 ============================

用户口径(2026-08-14 裁定):

    KeyFrame = 从 Video Prior 推理出的、**且它自己足够可信**的帧;
    关键帧之间可以再插高可信的过渡帧;**不可信的帧不参与**监督(不是降权)。

训练阶段划分**不许人工设定**,必须由本模块的输出按时间排出来(B 项裁定)。

============================ 三个来源(都通用,零任务知识) ============================

1. **抓握事件 = 物体开始运动**。"碰到" ≠ "抓稳" —— 2D 邻接在真接触前就触发
   (screw18 早 10 帧;pour17 的杯早 34 帧)。物体动了才是抓稳的运动学铁证。
   实测精度:screw18 给 f14 vs 人工真值 f13(差 1 帧)。
2. **驻留段 = 演示在某个相对构型上停留**。关键:很多任务的核心时刻**没有接触变化**
   (倒水时两手一直握着),纯接触驱动的抽取会**完全漏掉**它。
   窗口必须从"**所有物体都抓稳之后**"开始 —— 用接触起点会检出"还没被拿走时的静止"假阳性。
3. **收尾 = 各物体末次运动**。

============================ 可信度怎么用 ============================

- **通道级门控**用管线权威 `confidence_complete.json`(`rotation_usable` 等);
- **窗口内挑哪一帧**用本地 `pose_audit` 的逐帧 conf_pos。
  ⚠ 本地 audit 缺 CoTracker 输入时旋转分会退化为 0,**不得**用它覆盖权威的旋转裁决。
- 容差随可信度放宽:`tol = tol0 · (1 + k(1-c))` —— 低可信给**松容差**,而不是给错目标。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rl_rebuild.correction.scene_layout import (FINGERTIPS, _motion_onset,
                                                contact_onsets)

TOL0_M = 0.02          # 高可信时的位置容差
TOL_K = 1.5            # 容差随 (1-c) 放大的斜率
DWELL_MIN_LEN = 10     # 驻留段最短帧数
DWELL_PCT = 40         # 驻留判定的分位阈(窗口内)


def _obj_positions(Tw: np.ndarray) -> np.ndarray:
    return np.asarray(Tw[:, :, :3, 3], dtype=float)          # (n_obj, T, 3)


def _main_axis_world(Tw: np.ndarray, i: int) -> np.ndarray:
    """物体 i 的主轴(网格局部 y)在世界系的逐帧方向。"""
    a = np.asarray(Tw[i, :, :3, 1], dtype=float)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def last_motion(P: np.ndarray, win: int = 3) -> int:
    d = np.linalg.norm(P[win:] - P[:-win], axis=1)
    thr = max(0.008, float(np.median(d[:8])) * 3.0)
    idx = np.flatnonzero(d > thr)
    return int(idx[-1]) + win if len(idx) else len(P) - 1


def dwells(Tw: np.ndarray, lo: int, hi: int, *, win: int = 5, smooth: int = 7,
           pct: int = DWELL_PCT, min_len: int = DWELL_MIN_LEN) -> list[tuple[int, int]]:
    """相对构型的驻留段。**两物体**时用 obj1 相对 obj0;单物体时用它自身位姿。

    构型变化率 = 相对位移(cm) + 主轴转角(度)/10 —— 1cm 与 10° 同权(粗口径, 可调)。
    """
    n = Tw.shape[0]
    if n >= 2:
        rel = _obj_positions(Tw)[1] - _obj_positions(Tw)[0]
        ax = _main_axis_world(Tw, 1)
    else:
        rel = _obj_positions(Tw)[0]
        ax = _main_axis_world(Tw, 0)
    dv = np.linalg.norm(rel[win:] - rel[:-win], axis=1) * 100.0
    da = np.degrees(np.arccos(np.clip((ax[win:] * ax[:-win]).sum(1), -1, 1)))
    sc = dv + da / 10.0
    sm = np.convolve(sc, np.ones(smooth) / smooth, mode="same")
    hi = min(hi, len(sm))
    if hi - lo < min_len:
        return []
    m = np.zeros(len(sm), bool)
    m[lo:hi] = sm[lo:hi] < np.percentile(sm[lo:hi], pct)
    out, s = [], None
    for t, x in enumerate(m):
        if x and s is None:
            s = t
        if not x and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    if s is not None and len(m) - s >= min_len:
        out.append((s, len(m)))
    return out


def per_frame_conf(take_id: str, poseqa: Path) -> dict[str, np.ndarray]:
    """本地 pose_audit 的逐帧 conf_pos(只用于窗口内排序, 不做通道裁决)。"""
    f = poseqa / "pose_audit.json"
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    out = {}
    for e in d.get("takes", []):
        if take_id not in str(e.get("take", "")) or "per_frame" not in e:
            continue
        oid = e.get("object") or "object_0"
        out[oid] = np.array([r.get("conf_pos", 0) for r in e["per_frame"]], float)
    return out


def _pick_frame(window: tuple[int, int], conf: np.ndarray | None) -> tuple[int, float]:
    """窗口内取 conf 最高的一帧; 无逐帧 conf 时取中点。→ (帧, c∈[0,1])"""
    a, b = int(window[0]), int(window[1])
    if conf is None or len(conf) == 0:
        return (a + b) // 2, 0.5
    a2, b2 = max(0, a), min(len(conf), max(a + 1, b))
    seg = conf[a2:b2]
    if len(seg) == 0:
        return (a + b) // 2, 0.5
    k = int(np.argmax(seg))
    return a2 + k, float(seg[k]) / 100.0


def _tol(c: float) -> float:
    return float(TOL0_M * (1.0 + TOL_K * (1.0 - max(0.0, min(1.0, c)))))


def build(recon_dir: Path, replay_npz: Path, *, poseqa: Path | None = None,
          take_id: str | None = None) -> dict:
    """→ keyframes.json 的内容(见模块 docstring)。"""
    r = np.load(replay_npz, allow_pickle=True)
    w = np.load(recon_dir / "world_fused.npz", allow_pickle=True)
    Tw = np.asarray(w["object_ob_in_world_all"], float)
    oids = [str(x) for x in r["object_ids"]] if "object_ids" in r.files else ["object_0"]
    J = {"left": r["joints_left"], "right": r["joints_right"]}
    T = Tw.shape[1]

    conf_all = json.loads((recon_dir / "confidence_complete.json").read_text()) \
        if (recon_dir / "confidence_complete.json").is_file() else {}
    co = conf_all.get("objects") or {}
    pfc = per_frame_conf(take_id or recon_dir.name, poseqa) if poseqa else {}

    onsets = contact_onsets(recon_dir)
    if not onsets and len(oids) == 1:
        # 单物体 take 常常只有并集 contact_auto.json(screw18 实测)。
        # 并集只在**单物体**时等价于逐物体, 多物体时绝不能用(会把"摸另一个物体"算进来)。
        f = recon_dir / "contact_auto.json"
        if f.is_file():
            ann = (json.loads(f.read_text()).get("annotations") or {})
            best = None
            for hand in ("left", "right"):
                segs = ann.get(hand) or []
                if segs and (best is None or segs[0][0] < best[1]):
                    best = (hand, int(segs[0][0]))
            if best:
                onsets = {oids[0]: best}
    P = _obj_positions(Tw)

    # ---------- 1. 抓握事件(物体运动起始) ----------
    chain, roles = [], {}
    grasp_f = {}
    for i, oid in enumerate(oids):
        if oid not in onsets:
            continue
        hand, c0 = onsets[oid]
        roles[oid] = hand
        t, why = _motion_onset(P[i], int(c0))
        grasp_f[oid] = t
        cpos = (co.get(oid, {}).get("conf_pos_median") or 50.0) / 100.0
        anchor = J[hand][int(np.clip(t, 0, len(J[hand]) - 1))][FINGERTIPS].mean(0)
        # 目标为**相对量**: 指尖质心相对该物体质心(世界系差, 物体系由摆放给)
        rel = (anchor - P[i][int(np.clip(t, 0, T - 1))]).tolist()
        chain.append({"kind": "grasp", "t": int(t), "object": oid, "hand": hand,
                      "conf": round(cpos, 3), "tol_m": round(_tol(cpos), 4),
                      "target_rel_fingertip_centroid_m": [round(x, 4) for x in rel],
                      "source": f"物体运动起始({why}); 接触起点 f{c0}"})

    # ---------- 2. 驻留段(窗口 = 所有物体都抓稳之后) ----------
    lo = max(grasp_f.values()) if grasp_f else 0
    segs = dwells(Tw, lo, T - 5)
    rot_ok = {oid: bool(co.get(oid, {}).get("rotation_usable")) for oid in oids}
    for (a, b) in segs:
        c_use = min((co.get(o, {}).get("conf_pos_median") or 50.0) / 100.0 for o in oids)
        item = {"kind": "dwell", "t": [int(a), int(b)], "hold_steps": int(b - a),
                "conf": round(c_use, 3), "tol_m": round(_tol(c_use), 4),
                "source": f"相对构型驻留(窗口[{lo},{T-5}], 分位{DWELL_PCT}%, ≥{DWELL_MIN_LEN}帧)"}
        if len(oids) >= 2:
            rel = (P[1][a:b] - P[0][a:b]).mean(0)
            item["objects"] = [oids[0], oids[1]]
            item["target_rel_pos_m"] = [round(float(x), 4) for x in rel]
            # 朝向只有在**权威裁定可用**时才进目标
            if rot_ok.get(oids[1]):
                ax = _main_axis_world(Tw, 1)[a:b].mean(0)
                tilt = float(np.degrees(np.arccos(abs(np.clip(ax[2], -1, 1)))))
                item["target_tilt_deg"] = round(tilt, 1)
                item["tilt_object"] = oids[1]
            else:
                item["orientation"] = "弃用(rotation_usable=false)"
        chain.append(item)

    # ---------- 3. 收尾(各物体末次运动) ----------
    for i, oid in enumerate(oids):
        if oid not in roles:
            continue
        t = last_motion(P[i])
        cpos = (co.get(oid, {}).get("conf_pos_median") or 50.0) / 100.0
        chain.append({"kind": "release", "t": int(t), "object": oid, "hand": roles[oid],
                      "conf": round(cpos, 3), "tol_m": round(_tol(cpos), 4),
                      "target_pos_m": [round(float(x), 4) for x in P[i][min(t, T - 1)]],
                      "source": "物体末次运动"})

    # ---------- 4. 清理: release 落在某个 dwell 内 ⟹ 并入该 dwell ----------
    #   否则阶段链会出现"互相包含"的环(pour17 实测: 瓶释放 f122 落在收尾驻留 [109,137] 内)。
    #   语义上那本来就是同一件事: "放回并静置"。
    dw = [k for k in chain if k["kind"] == "dwell"]
    merged = []
    for k in chain:
        if k["kind"] == "release":
            host = next((d for d in dw if d["t"][0] <= k["t"] <= d["t"][1]), None)
            if host is not None:
                host.setdefault("absorbed_releases", []).append(
                    {"object": k["object"], "t": k["t"]})
                continue
        merged.append(k)
    chain = merged

    chain.sort(key=lambda k: (k["t"][0] if isinstance(k["t"], list) else k["t"]))
    for n, k in enumerate(chain):
        k["stage"] = n + 1                      # ★ 训练阶段 = 链上序号(不许人工设定)

    return {"schema_version": "keyframes_v1", "take": str(recon_dir), "n_frames": int(T),
            "fps": float(r["fps"]) if "fps" in r.files else None,
            "roles": roles, "rotation_usable": rot_ok,
            "interaction_window": [int(lo), int(T - 5)],
            "gate_rule": "gate_i = slowEMA(第 i 环达成率) ≥ 0.8 才放行第 i+1 环",
            "tolerance_rule": f"tol = {TOL0_M}·(1+{TOL_K}(1-c))",
            "keyframes": chain}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-dir", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--poseqa", type=Path, default=None)
    ap.add_argument("--take-id", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    d = build(a.recon_dir, a.replay, poseqa=a.poseqa, take_id=a.take_id)
    print(f"  角色 {d['roles']} | 交互窗口 {d['interaction_window']} | "
          f"旋转可用 {d['rotation_usable']}")
    for k in d["keyframes"]:
        t = f"[{k['t'][0]},{k['t'][1]}]" if isinstance(k["t"], list) else str(k["t"])
        extra = (f" 保持{k['hold_steps']}步" if k.get("hold_steps") else "") + \
                (f" 倾角{k['target_tilt_deg']}°" if k.get("target_tilt_deg") else "")
        print(f"  S{k['stage']}  {k['kind']:8s} t={t:>10s}  "
              f"{k.get('object', k.get('objects', ''))}  c={k['conf']:.2f} "
              f"tol={k['tol_m']*100:.1f}cm{extra}")
    out = a.out or (a.recon_dir / "keyframes.json")
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
