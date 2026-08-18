#!/usr/bin/env python
"""Step2 → Step3 唯一入口: 给一条重建 take, 自动跑完它全部该抓的手和物体。

    python steps/step3_grasppose/run_take.py --take <重建take目录>
    python steps/step3_grasppose/run_take.py --dataset egodex_auto --task pour --take-id 17

在这之前 Step3 是断开的: 要人肉指定 object_id / 哪只手 / 接触带高度, 一次只能跑一只手,
**而且完全没查 Step2 已经做出的 confidence 裁决** —— `tools/upright_from_recon.py` 自己的
注释写着"朝向不可信时不要用它", 但没有任何代码在执行这句话。

═══ 读 Step2 的哪些产物 ═══

  contact/grasp_prompt.json
      grasps[]            该跑哪些 (object_id, hand) —— 工作清单的唯一来源
      rejected[]          跳过的组合与原因(如 no_contact_interval), 原样记进产物
      contact_region      height_pct_median → 评分的 --ref-height
                          ★这是 Step2 在**原始接触点云**上算的, 比 Step3 自己在重采样后的
                            region.npz 上现算更权威。pour/17 两者都是 61%, 是这条改动的对拍基准
  confidence_complete.json
      manifest_status     != active → 整条 take 跳过(被 take_manifest 择优淘汰或硬排除)
      rotation_usable     false → 不能拿重建朝向摆物体, 降级 stable_pose_only
      position_grade      poor → 照跑但标低置信
  vlm_grasp.json
      answer.<side>.object_shape        → 先验档位键的形状维(原来硬编码 cylinder)
      answer.<side>.n_contact_fingers   → 粗筛的指数目标(原来硬编码 4)
  world_fused.npz / objects/<oid>/object_mesh_scaled_final.obj
      摆放位姿与网格, 由 tools/take_grasp_pipeline.sh 自己读

═══ ⚠ 只读数值, 绝不用产物里烘死的绝对路径 ═══

grasp_prompt.json 的 `mesh` / `contact_cloud_npz` 和 confidence_complete.json 的
`poseqa_root` 写的都是**产出那台机器**的路径(pour/17 实测是 `/home/yanghong/...`)。
一律用传入的 take 目录重建路径。这条与 rl_rebuild/correction/paths.py 顶部同源。

═══ 输出 ═══

  <take>/grasp_pose_plan.json     Step4 从这一份取数, 不要自己扫目录
      (与 Step2 的 take_manifest.py 同一套哲学: 上游给裁决, 下游不重新判断)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: E402

SCHEMA = "grasp_pose_plan_v1"


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _stamp(p: Path) -> dict:
    """输入产物的身份戳, 写进 grasp_pose_plan.json 的 `inputs`。

    为什么每份输入都要记(2026-08-17, 两次实测教训):
      * Step2 在 UCB 上重生成了 `hand_without_target`, 本机没同步 —— 我读到的是旧快照,
        而**两边都不报错**。同样的事在 `confidence_complete.json` 上发生过一次。
      * 没有身份戳时, "这条 take 真不命中" 和 "我手上这份是旧的" **在产物里长得一模一样**。

    ★ 我自己算 `sha1`, 不只转述文件内嵌的 `provenance` —— 两个理由:
      1. 不是每份输入都有 provenance(`confidence_complete.json` 目前就没有)
      2. 内嵌的 `generator_commit` **会说谎**: Step2 实测发现, 用 scp 单独拷文件时它报的是
         "那台机器的仓停在哪", 不是"实际执行的是哪份代码"。内容哈希骗不了。
         (Step2 因此也加了 `generator_sha1` = 文件内容哈希, 并约定比对只看它。)
    """
    if not p.is_file():
        return {"path": str(p), "missing": True}
    import hashlib
    raw = p.read_bytes()
    out = {"path": str(p), "sha1": hashlib.sha1(raw).hexdigest()[:12],
           "mtime": p.stat().st_mtime}
    try:
        d = json.loads(raw)
        for k in ("schema_version", "provenance"):
            if k in d:
                out[k] = d[k]
    except Exception:
        pass
    return out


def confidence_of(take: Path) -> dict:
    """Step2 的 confidence 裁决。take 里的 confidence_complete.json 是 TAKE_MANIFEST 的
    take 级镜像 —— 用它而不是去够 poseqa_root, 因为那个路径是外机的。"""
    d = _load(take / "confidence_complete.json")
    if d is None:
        return {"present": False, "manifest_status": "unknown", "objects": {}}
    return {"present": True, "manifest_status": d.get("manifest_status", "unknown"),
            "objects": d.get("objects", {})}


def vlm_of(take: Path, side: str) -> dict:
    d = _load(take / "vlm_grasp.json") or {}
    return ((d.get("answer") or {}).get(side)) or {}


def run_one(take: Path, oid: str, side: str, tag: str, *, ref_height, shape, digits,
            placement_mode, dry_run: bool) -> dict:
    """跑一个 (object, hand)。返回该组合的产物小结。"""
    env = dict(os.environ)
    env.update(DEXO_ENV=paths.DEXO_ENV, SHAPE=shape, DIGITS=str(digits),
               PLACEMENT_MODE=placement_mode)
    if ref_height is not None:
        env["REF_H_OVERRIDE"] = str(ref_height)
    cmd = ["bash", str(HERE / "tools" / "take_grasp_pipeline.sh"),
           str(take), oid, side, tag]
    if dry_run:
        return {"dry_run": True, "cmd": " ".join(cmd),
                "env": {k: env[k] for k in ("SHAPE", "DIGITS", "PLACEMENT_MODE")
                        } | ({"REF_H_OVERRIDE": env["REF_H_OVERRIDE"]} if ref_height else {})}
    r = subprocess.run(cmd, env=env, cwd=str(HERE))
    hand_model = "sharpa_wave_v2" + ("_left" if side == "left" else "")
    exp = HERE / "output" / f"{tag}_{hand_model}"
    out = _load(exp / "summary.json") or {"error": "管线没产出 summary.json"}
    out["placement"] = _load(exp / "placement.json")
    out["returncode"] = r.returncode
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default=None, help="重建 take 目录(绝对路径)")
    ap.add_argument("--dataset", default="egodex_auto")
    ap.add_argument("--task", default=None)
    ap.add_argument("--take-id", default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="只跑指定组合, 形如 object_1:right")
    ap.add_argument("--force", action="store_true",
                    help="无视 manifest_status 不是 active 的裁决照跑(诊断用)")
    ap.add_argument("--dry-run", action="store_true", help="只解析并打印计划, 不真跑")
    a = ap.parse_args()

    take = Path(a.take) if a.take else Path(paths.take_dir(a.dataset, a.task, a.take_id))
    if not take.is_dir():
        sys.exit(f"take 目录不存在: {take}")

    prompt = _load(take / "contact" / "grasp_prompt.json")
    if prompt is None:
        sys.exit(f"缺 Step2 交接件 {take}/contact/grasp_prompt.json —— 先跑完 Step2 的接触提取")
    conf = confidence_of(take)

    print(f"══ take {take}")
    print(f"  Step2 裁决: manifest_status={conf['manifest_status']}"
          + ("" if conf["present"] else "  (无 confidence_complete.json)"))
    if conf["manifest_status"] not in ("active", "unknown") and not a.force:
        plan = {"schema": SCHEMA, "take": str(take), "status": "skipped",
                "why": f"Step2 裁决 manifest_status={conf['manifest_status']}"
                       "(被 take_manifest 择优淘汰或硬排除); --force 可强跑",
                "grasps": []}
        (take / "grasp_pose_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=1))
        print(f"  → 整条 take 跳过。{plan['why']}")
        return

    want = set(a.only or [])
    results, skipped = [], []
    for g in prompt.get("grasps", []):
        obj, side = g["object_id"], g["hand"]
        if want and f"{obj}:{side}" not in want:
            continue
        c = conf["objects"].get(obj, {})
        rot_ok = c.get("rotation_usable", True)
        mode = "recon" if rot_ok else "stable_only"
        region = g.get("contact_region") or {}
        v = vlm_of(take, side)
        shape = v.get("object_shape") or "cylinder"
        digits = int(v.get("n_contact_fingers") or 4)
        ref_h = region.get("height_pct_median")
        tag = f"{take.parent.name}{take.name}_{obj}_{side}"

        print(f"\n── {obj} × {side}手   形状={shape} 指数={digits} 接触带高度={ref_h}%")
        print(f"   confidence: pos={c.get('conf_pos_median')}({c.get('position_grade')}) "
              f"rot={c.get('conf_rot_median')} rotation_usable={rot_ok}"
              + ("" if rot_ok else "  → 摆放降级为 stable_pose_only(无视频依据)"))
        r = run_one(take, obj, side, tag, ref_height=ref_h, shape=shape, digits=digits,
                    placement_mode=mode, dry_run=a.dry_run)
        results.append({
            "object_id": obj, "hand": side, "tag": tag,
            "grasp_window_frames": g.get("grasp_window_frames"),
            "vlm": {"object_shape": shape, "n_contact_fingers": digits,
                    "palm_contact": v.get("palm_contact"),
                    "contact_depth": v.get("contact_depth")},
            "confidence": {"conf_pos_median": c.get("conf_pos_median"),
                           "conf_rot_median": c.get("conf_rot_median"),
                           "position_grade": c.get("position_grade"),
                           "rotation_usable": rot_ok,
                           "refuted_frames": c.get("refuted_frames")},
            "placement_source": ("stable_pose_only" if not rot_ok else "recon_head10f+snap"),
            "low_confidence": (c.get("position_grade") == "poor") or (not rot_ok),
            "result": r,
        })
    for r in prompt.get("rejected", []):
        skipped.append({"object_id": r.get("object_id"), "hand": r.get("hand"),
                        "why": r.get("why") or r.get("status"), "by": "step2_grasp_prompt"})

    # ── Step2 的"这只手没有可抓目标"信号(2026-08-17 加)
    # 名字刻意描述**观测**而非推断: 观测到的是"该手被判无接触 且 本 take 物体数少于手数";
    # "有个部件没分件出来"是**启发式推断**, 所以 confidence 标 heuristic。
    # ★不 gate 任何东西, 只记录+告警 —— 它分不开"真没抓东西"和"抓了但物体不存在"。
    # 全库命中率(Step2 实测): 拧瓶盖 25/29(86%), 倒水 10/33(30%), 合计 48 只手。
    # 那 86% 与独立数出的"单物体 take 占比"相等, 是这个信号有效的交叉验证。
    # ⚠ 字段缺失是正常的(旧产物 / 未同步到本机), 不当错误处理。
    hwt = prompt.get("hand_without_target") or []

    # ★ "一个抓取目标都没有" 必须是**显式状态**, 不能只表现为 grasps=[] ——
    #   否则它和"跑了但全失败"在产物里长得一模一样, 下游分不出该重跑还是该放弃。
    #   这不是罕见情况: Reconstruction 统计全库 62 条里 **13 条零手、29 条只有一只手**,
    #   典型成因是**物体没被分出来**(拧瓶盖 29 条里 25 条的盖从未被注册成实例,
    #   于是那只手在交接件里表现为"没有既稳又贴的帧段", 而不是"缺了个物体")。
    status = "ok" if results else ("no_grasp_targets" if not want else "filtered_out_by_--only")
    plan = {"schema": SCHEMA, "take": str(take), "status": status,
            "manifest_status": conf["manifest_status"],
            "n_objects_in_take": len({g["object_id"] for g in prompt.get("grasps", [])}
                                     | {r.get("object_id") for r in prompt.get("rejected", [])}),
            "hand_without_target": hwt,
            # 输入链的身份戳: mesh 的 source_sha1(在 info/simplified.json) → 这三份 → 本 plan
            "inputs": {"grasp_prompt": _stamp(take / "contact" / "grasp_prompt.json"),
                       "confidence_complete": _stamp(take / "confidence_complete.json"),
                       "vlm_grasp": _stamp(take / "vlm_grasp.json")},
            "grasps": results, "skipped": skipped}
    if status == "no_grasp_targets":
        plan["why"] = ("Step2 的 grasp_prompt.json 里 grasps[] 为空 —— 这条 take 没有任何"
                       "(物体×手)可抓。常见成因是物体没被重建/分件出来, 而不是抓取生成失败。")
    out = take / "grasp_pose_plan.json"
    if not a.dry_run:
        out.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    if hwt:
        print(f"\n⚠ Step2 判定有 {len(hwt)} 只手**没有可抓目标**(hand_without_target, 启发式):")
        for h in hwt:
            print(f"   {h.get('hand')}手: 本 take 只有 {h.get('n_objects_in_take')} 个物体 / "
                  f"{h.get('n_hands_with_target')} 只手有目标; 被拒原因 \"{h.get('rejected_reason')}\"")
        print("   ⇒ 大概率是**物体没被分件出来**(而不是抓取生成失败)。此信号不影响生成与排序。")
    if status == "no_grasp_targets":
        print(f"\n⚠ 本 take **没有任何抓取目标**(grasp_prompt 的 grasps[] 为空)。"
              f"\n  {len(skipped)} 只手被判无接触: "
              f"{[(s['hand'], s['why']) for s in skipped]}"
              f"\n  这通常不是抓取生成失败, 而是物体没被分件出来 —— 查 Step2 的实例注册。")
    print(f"\n══ {len(results)} 组已跑, {len(skipped)} 组被 Step2 判无接触"
          f"   状态 {status}"
          f"\n   → {out}" + ("  (dry-run 未写)" if a.dry_run else ""))
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
