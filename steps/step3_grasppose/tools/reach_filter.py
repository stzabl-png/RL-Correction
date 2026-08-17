"""可达性闸 —— 机械臂在**视频 yaw** 下够不够得到这个 GraspPose。

为什么必须有这一关(2026-08-17):
  Step3 的排序此前**完全没有可达性这一维**, 纯几何。实测代价: 瓶子那个在几何评分里
  双料第一的候选(总分 0.968、对视频接触点覆盖率 93%, 两项都是最高), 在 Isaac 里
  零动作合拢只有 **3.25 个接触垫**真接上(判据 ≥4)。
  ⇒ **几何最优 ≠ 物理可用**, 中间隔着可达性和接触力学。可达性这一半是纯运动学、
    秒级、不要 GPU, 没有理由不在 Step3 就算掉。

★ 本文件**不实现任何判据**, 全部调 RL 侧 `tasks/pregrasp/screen_prior.py` 的实现:

    video_yaw_deg(npz, canon_rot)        视频里物体静置段的 yaw
    object_placement(...)                物体在 env 世界系的位置(**物体听手**)
    gate1(prior_npz, obj_pos, video_yaw) yaw 扫描, 出可达带与 best_yaw

  这么做是刻意的: 这些量在训练时也要用同一套定义, 两边各维护一份迟早漂移 ——
  RL 侧 `object_placement` 的 docstring 就记着自己另算一份导致 pour17 差 5.2cm 的教训。
  合仓之后 Step3 能直接 import 它们(实测 dexonomy 环境即可, 不需要 Isaac/MagicSim venv)。

★★ 四个必须传对的坑, 每个都实测过:

  1. **`hand` 要传 `robot_hand`, 不是 `hand`** —— 注册表里这两个字段可能不同
     (拧盖那条 clip: hand="left" 是轨迹名而实为物理右手)。取法照抄
     `cfg.get("robot_hand", cfg.get("hand", "right"))`。
     RL 侧 2026-08-16 刚因为左右手取错吃过一个 165° 的假警报。
  2. **`scene_layout_json` 不能漏** —— 漏了 `object_placement` 会给出差 **90mm** 的位置
     (本文件开发时实测)。补上之后与 RL 侧 `object_placement_from_clip` 逐位一致(0.000mm)。
  3. **`affordance` 不能漏** —— 它决定对齐的目标点, 漏了差 ~11cm。
  4. **prior 转换必须用系统 `python3`** —— Dexonomy 的 npy 是 numpy2 pickle,
     MagicSim venv 的 numpy1 读不了。这里照 `make_prior.py` 的 docstring 走 subprocess。

⚠ 结果随 `gs`(抓取锚帧)口径变: RL 侧 2026-08-16 把 gs 从"接触标注起点"改成"物体运动起始"后,
  pour17 瓶的摆放移动了 **9.1cm**。任何基于它的可达性结论都必须在**当前口径**下重算,
  不能沿用历史数字。

用法:
  # 有注册 clip 时(权威, 推荐)
  PYTHONPATH=<repo> python tools/reach_filter.py --clip Pour17_bottle \\
      --grasp-dir output/<exp>/top --oid pour17_object_1_right --json reach.json

  # 无注册 clip 时: 显式给四样(mesh/npz/affordance/scene_layout 缺哪个报哪个)
  ... --npz <replay_world.npz> --mesh <obj.obj> --hand right \\
      [--affordance <aff.npz>] [--scene-layout <scene_layout.json>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _rl():
    """把 RL 侧的三个函数拿进来。放在函数里 import, 免得没装 RL 依赖时本文件不可导入。"""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    pg = os.path.join(REPO, "tasks", "pregrasp")
    if pg not in sys.path:
        sys.path.insert(0, pg)
    from screen_prior import gate1, object_placement, video_yaw_deg
    return video_yaw_deg, object_placement, gate1


def make_prior(grasp_npy: str, info_json: str, out_npz: str) -> None:
    """Dexonomy npy -> RL prior npz。★必须用系统 python3(numpy2 pickle), 见文件头坑 4。"""
    r = subprocess.run(["python3", os.path.join(REPO, "tasks", "pregrasp", "make_prior.py"),
                        "--grasp_npy", grasp_npy, "--info_json", info_json, "--out", out_npz],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out_npz):
        raise RuntimeError(f"make_prior 失败: {r.stderr.strip()[-300:]}")


def resolve_cfg(a) -> dict:
    """把 clip 注册表 / 显式参数收敛成 object_placement 需要的一组值。"""
    cfg = {}
    if a.clip:
        sys.path.insert(0, REPO)
        from rl_rebuild.correction.clips import CLIPS
        if a.clip not in CLIPS:
            raise SystemExit(f"注册表里没有 clip={a.clip}; 已注册: {sorted(CLIPS)[:12]}…")
        cfg = dict(CLIPS[a.clip])
    for k, v in (("npz", a.npz), ("mesh", a.mesh), ("affordance", a.affordance),
                 ("scene_layout_json", a.scene_layout)):
        if v:
            cfg[k] = v
    if a.hand:
        cfg["robot_hand"] = a.hand
    # ★ robot_hand 优先, 见文件头坑 1
    cfg["_hand"] = cfg.get("robot_hand", cfg.get("hand", "right"))
    miss = [k for k in ("npz", "mesh") if not cfg.get(k)]
    if miss:
        raise SystemExit(f"缺必需项 {miss} —— 给 --clip 或显式传 --npz/--mesh")
    for k, why in (("affordance", "决定对齐目标点, 漏了差 ~11cm"),
                   ("scene_layout_json", "漏了物体位置差 ~90mm")):
        if not cfg.get(k):
            print(f"  ⚠ 没有 {k} ({why}) —— 结果会与 RL 侧不一致")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-dir", required=True, help="候选目录(递归找 *_grasp.npy)")
    ap.add_argument("--oid", required=True, help="Step3 的 oid, 用来定位 info/simplified.json")
    ap.add_argument("--clip", default=None, help="RL 注册表里的 clip 名(有就用, 权威)")
    ap.add_argument("--npz", default=None, help="replay_world.npz")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--affordance", default=None)
    ap.add_argument("--scene-layout", default=None)
    ap.add_argument("--hand", default=None, choices=("left", "right"))
    ap.add_argument("--table-top-z", type=float, default=0.85)
    ap.add_argument("--tol-deg", type=float, default=30.0, help="可达带要覆盖视频 yaw 的容差")
    ap.add_argument("--step", type=int, default=5, help="yaw 扫描步长(度)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    video_yaw_deg, object_placement, gate1 = _rl()
    cfg = resolve_cfg(a)
    info_json = os.path.join(HERE, "..", "assets", "object", "custom",
                             "processed_data", a.oid, "info", "simplified.json")
    info_json = os.path.abspath(info_json)
    if not os.path.isfile(info_json):
        raise SystemExit(f"找不到 {info_json} —— 先跑 import_object")
    canon = np.asarray(json.load(open(info_json))["canonical_from_input_rot_wxyz"], np.float64)

    vy, resid, stab = video_yaw_deg(cfg["npz"], canon)
    obj_pos = object_placement(cfg["npz"], cfg["mesh"], a.table_top_z, hand=cfg["_hand"],
                               affordance=cfg.get("affordance"),
                               semantics=cfg.get("semantics"),
                               scene_layout_json=cfg.get("scene_layout_json"))
    print(f"  视频 yaw {vy:.1f}° (残差 {resid:.1f}°, 静置稳定 {stab:.2f}°)"
          + ("   ⚠ 残差 >15° 说明'哪个面朝下'与视频都不一致, yaw 无意义" if resid > 15 else ""))
    print(f"  物体位置 {np.round(obj_pos, 4)}   手 {cfg['_hand']}")

    files = sorted(glob.glob(os.path.join(a.grasp_dir, "**", "*_grasp.npy"), recursive=True))
    if not files:
        raise SystemExit(f"{a.grasp_dir} 下没有 *_grasp.npy")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            p = os.path.join(td, "p.npz")
            try:
                make_prior(f, info_json, p)
                r = gate1(p, obj_pos, vy, tol_deg=a.tol_deg, step=a.step, hand=cfg["_hand"])
            except Exception as e:
                r = dict(ok=False, n_reach=0, dpsi=None, best_yaw=None,
                         best_err=float("nan"), band="", skipped=f"{type(e).__name__}: {e}")
            rows.append(dict(npy=os.path.abspath(f), **{k: r.get(k) for k in
                        ("ok", "n_reach", "dpsi", "best_yaw", "best_err", "band", "skipped")}))

    ok = [r for r in rows if r["ok"]]
    print(f"\n{'候选':<46}{'可达':>5}{'带宽':>6}{'Δψ':>8}{'best_yaw':>10}{'IK误差':>9}")
    for r in rows:
        dpsi = f"{r['dpsi']:.1f}°" if r["dpsi"] is not None else "-"
        byaw = f"{r['best_yaw']:.0f}°" if r["best_yaw"] is not None else "-"
        berr = f"{r['best_err']*100:.1f}cm" if r["best_err"] == r["best_err"] else "-"
        print(f"{os.path.basename(r['npy'])[:44]:<46}{'✓' if r['ok'] else '✗':>5}"
              f"{r['n_reach']:>6}{dpsi:>8}{byaw:>10}{berr:>9}")
    print(f"\n可达 {len(ok)}/{len(rows)}"
          + (f"   建议 --prior_yaw {ok[0]['best_yaw']:.0f}" if ok else "   ⚠ 全部够不到"))
    if a.json:
        json.dump({"video_yaw": vy, "video_yaw_resid": resid, "obj_pos": obj_pos.tolist(),
                   "hand": cfg["_hand"], "tol_deg": a.tol_deg, "rows": rows},
                  open(a.json, "w"), ensure_ascii=False, indent=1)
        print(f"-> {a.json}")


if __name__ == "__main__":
    main()
