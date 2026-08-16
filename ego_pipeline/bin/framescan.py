#!/usr/bin/env python3
"""闭环精修重建帧 / FP 注册帧 —— 正常链跑完之后的精修 pass(级联, 非网格)。

============================ 为什么要有这一步 ============================

选帧判据(遮挡少、清晰、mask 完整)**预测不了重建质量**。唯一可靠的办法是把候选真跑一遍
看结果。32 条 pour / 49 个物体实测:

    能用的物体   16/49 (33%)  ->  31/49 (63%)
    能用的 take   6/32 (19%)  ->  16/32 (50%)

============================ ★ 判据: 只认"被证伪帧数" ============================

同一条数据、frame_plan 一字不改、连跑 3 次(2026-08-15 实测, 见 memory):

    被证伪帧数   跨度 <=1 帧          ★ 三者里最稳, 但**不是绝对稳**
    conf_rot     跨度 ±25 分          单次不可信
    conf_pos     跨度 ±36 分          单次不可信, 禁止用于判优

⚠ 后续反例(2026-08-15 下午, take 23): 同一配置 f63/reg10 搜索时量到 证伪 0 / conf_rot 50.5,
  收尾重跑同一配置变成 证伪 5 / conf_rot 34.0。**被证伪数也会翻**, 只是概率低于 conf。
  ⇒ 所以收尾**绝不重跑赢家配置**(见下), 且报出去的数一律以磁盘产物为准。

根因: fp_pose(FoundationPose) 本身非确定性(ob_in_cam 每次都不同); SAM3D 与 confidence
都是确定的。被证伪数是"逐帧判 trust<0.20 再计数"的聚合量, 边界帧翻转互相抵消所以稳。

⇒ 判优顺序: status==active > 被证伪少 > conf_rot 高。**不比 conf_pos。**
⇒ 早停门槛 conf_rot>=ROT_SAFE(45) 而不是 30 —— 30 分线在噪声带里, 判定会掷骰子。
⇒ 落在 [ROT_MIN, ROT_SAFE) 的"压线"结果搜完统一重测 REPEAT 次取中位再定。

============================ 级联, 不打网格 ============================

    第一步  注册帧不动, 按 sam3d_candidates 顺序试重建帧   -> 解决 28/49
    第二步  只对仍不合格的, 在它已定的网格上换注册帧        -> 再捞 3
            ★ 换注册帧**不用重建网格**, 只跑 fp_pose+fuse+confidence

网格 3x3 要 9 次评估, 级联只要 3+3, 而第二步的边际收益只有 +3 个物体。
可行性依据: 换注册帧后, 原最优重建帧仍在前 3 的比例 85%。

⚠ **重建帧一变就必须重跑 sam3d + sam3d_scale**。只重跑 fp_pose 之后的步骤会用到磁盘上
  残留的别的候选的网格 —— 2026-08-15 在复测脚本里栽过一次, 9 项里 2 项量错。

============================ 注册帧候选 ============================

实测**没有任何图像特征能预测好的注册帧**(99 对配对样本, 7 个特征全部 45~56%,
滤掉噪声内的伪胜负后不变)。所以候选只是探针, 顺序如下:
    ① 现行值(fp_register_frame, 正常链已跑)
    ② 该物体自己的接触起始帧+10 (contact_auto 的 mask_adjacency_v1, 不依赖位姿)
    ③ 现行值 +10
frame_plan 里若已有 `fp_candidates` 则以它为准。

用法:
    python ego_pipeline/bin/framescan.py --dataset egodex_auto --video-id pour__17 --gpu 2
    python ego_pipeline/bin/framescan.py --dataset egodex_auto --video-list ids.txt --gpu 2
    ... --stage1-only     只精修重建帧(便宜, 拿九成收益)
    ... --dry-run         只打印计划
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RECON = REPO / "ego_pipeline" / "Reconstruction" / "recon_pipeline"

STEP_ENV = {"sam3d": "biv2ap", "sam3d_scale": "biv2ap", "fp_pose": "biv2ap",
            "fuse": "hawor", "confidence": "hawor"}
NO_GPU_ARG = {"fuse"}                 # fuse 的 argparse 没有 --gpu, 传了 rc=2
CUDA_VISIBLE = {"fp_pose"}            # 这一步靠 CUDA_VISIBLE_DEVICES 选卡
DEPTH_STEPS = {"sam3d_scale", "fp_pose"}   # 不传 depth-scale = 物体放错距离且全链不报警
MESH_STEPS = ("sam3d", "sam3d_scale")
POSE_STEPS = ("fp_pose", "fuse", "confidence")

ROT_MIN = 30       # take_manifest 的 rotation_usable 门槛
ROT_SAFE = 45      # 早停门槛: 高于噪声带(conf_rot 单次波动 ±25)
REPEAT = 3         # 压线结果的重测次数


def log(m: str) -> None:
    print(f"[framescan {time.strftime('%H:%M:%S')}] {m}", flush=True)


def conda_exe() -> str:
    for c in (os.getenv("CONDA_EXE"), shutil.which("conda"),
              os.path.expanduser("~/miniconda3/bin/conda")):
        if c and Path(c).is_file():
            return c
    raise SystemExit("framescan: 找不到 conda(设 CONDA_EXE 或把 conda 放进 PATH)")


# ------------------------------------------------------------------ 路径
def _paths(dataset: str, video_id: str):
    sys.path.insert(0, str(RECON))
    from _common.paths import interim_step_dir, FINAL_ROOT  # noqa: E402
    obj_dir = interim_step_dir(dataset, video_id, "sam2_object")
    take = Path(FINAL_ROOT) / dataset
    if os.environ.get("RECON_FINAL_NESTED") == "1" and "__" in video_id:
        for part in video_id.split("__"):
            take = take / part
    else:
        take = take / video_id
    return obj_dir, take


def depth_scale(dataset: str, video_id: str) -> float:
    sys.path.insert(0, str(RECON))
    from _common.paths import INTERIM_ROOT  # noqa: E402
    p = Path(INTERIM_ROOT) / dataset / video_id / "egodex_source.json"
    try:
        return float(json.loads(p.read_text()).get("depth_scale") or 1.0)
    except Exception:
        return 1.0


# ------------------------------------------------------------------ 选卡 / OOM 重试
#   sam3d_scale 实测峰值 31.86GB, sam3d/fp_pose ~26GB。给的卡放不下就换一张,
#   不然 nvdiffrast 那步会 torch.OutOfMemoryError 把整条 take 打断
#   (2026-08-15 实测: 同一张卡上并行两个任务 -> 只剩 4.3GB -> 要 5.73GB 时炸)。
NEED_MB = {"sam3d": 30000, "sam3d_scale": 34000, "fp_pose": 28000,
           "fuse": 2000, "confidence": 12000}
MAX_RETRY = 3
OOM_PAT = re.compile(r"out of memory|OutOfMemoryError", re.I)


def gpu_free_mb() -> dict[int, int]:
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for ln in o.strip().splitlines():
        p = [x.strip() for x in ln.split(",")]
        if len(p) == 3 and p[0].isdigit():
            out[int(p[0])] = int(p[2]) - int(p[1])
    return out


def pick_gpu(need_mb: int, prefer: int, exclude=()) -> int:
    """够用就留在 prefer 上(避免无谓迁移); 否则挑当前最空的一张。都不够就还给 prefer。"""
    free = gpu_free_mb()
    if not free:
        return prefer
    if prefer not in exclude and free.get(prefer, 0) >= need_mb:
        return prefer
    cand = [g for g, m in free.items() if g not in exclude and m >= need_mb]
    if not cand:
        return prefer
    return max(cand, key=lambda g: free[g])


# ------------------------------------------------------------------ 跑步骤
def run_steps(steps, dataset, video_id, video, gpu, logfh) -> tuple[bool, int]:
    """跑一串步骤。返回 (成功?, 现在用的卡)。OOM/失败自动换卡重试。"""
    ds = depth_scale(dataset, video_id)
    for st in steps:
        bad: set[int] = set()
        for attempt in range(MAX_RETRY):
            gpu = pick_gpu(NEED_MB.get(st, 8000), gpu, bad)
            cmd = [conda_exe(), "run", "--no-capture-output", "-n", STEP_ENV[st],
                   "python", str(RECON / st / "run_sequence.py"),
                   "--dataset", dataset, "--video-id", video_id,
                   "--video", str(video), "--force"]
            if st in DEPTH_STEPS:
                cmd += ["--depth-scale", f"{ds:.6f}"]
            env = dict(os.environ)
            if st in CUDA_VISIBLE:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd += ["--gpu", "0"]
            elif st not in NO_GPU_ARG:
                cmd += ["--gpu", str(gpu)]
            mark = logfh.tell() if logfh.seekable() else 0
            logfh.write(f"\n===== {st} {video_id} gpu={gpu} {time.strftime('%F %T')} =====\n")
            logfh.flush()
            rc = subprocess.run(cmd, cwd=str(REPO), stdout=logfh,
                                stderr=subprocess.STDOUT, env=env).returncode
            if rc == 0:
                break
            logfh.flush()
            tail = ""
            try:
                with open(logfh.name, "r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(mark)
                    tail = fh.read()
            except OSError:
                pass
            oom = rc in (137, -9) or bool(OOM_PAT.search(tail))
            log(f"★ {st} 失败 rc={rc} oom={oom} gpu={gpu} (第 {attempt + 1} 次)")
            if attempt == MAX_RETRY - 1:
                return False, gpu
            bad.add(gpu)
            if oom:
                time.sleep(20)          # 等别的进程把显存吐出来
    return True, gpu


def read_conf(take: Path) -> dict:
    d = json.loads((take / "confidence_complete.json").read_text())
    objs = d.get("objects") or {"object_0": d}
    return {k: {"conf_pos": v.get("conf_pos_median"), "conf_rot": v.get("conf_rot_median"),
                "refuted": v.get("refuted_frames"), "status": v.get("manifest_status")}
            for k, v in objs.items()}


# ------------------------------------------------------------------ 判优
def rank_key(r: dict):
    """判优顺序: active 优先 > 被证伪少 > conf_rot 高。★ 不比 conf_pos(噪声 ±36)。"""
    return (r.get("status") != "active",
            r["refuted"] if r.get("refuted") is not None else 10 ** 9,
            -(r.get("conf_rot") or 0))


def is_pass(r: dict) -> bool:
    return bool(r) and r.get("status") == "active" and r.get("refuted") == 0 \
        and (r.get("conf_rot") or 0) >= ROT_MIN


def is_safe(r: dict) -> bool:
    """早停: 远离 conf_rot 的噪声带, 判定可靠。"""
    return bool(r) and r.get("status") == "active" and r.get("refuted") == 0 \
        and (r.get("conf_rot") or 0) >= ROT_SAFE


def is_borderline(r: dict) -> bool:
    return bool(r) and r.get("status") == "active" and r.get("refuted") == 0 \
        and ROT_MIN <= (r.get("conf_rot") or 0) < ROT_SAFE


# ------------------------------------------------------------------ 候选
def fp_candidates(take: Path, oid: str, cur: int | None, plan_entry: dict) -> list[int]:
    if plan_entry.get("fp_candidates"):
        return [int(x) for x in plan_entry["fp_candidates"]]
    out = [cur] if cur is not None else []
    nf = 10 ** 9
    for name in (f"contact_auto_{oid}.json", "contact_auto.json"):
        p = take / name
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        nf = d.get("num_frames") or nf
        ann = d.get("annotations") or {}
        starts = [s[0] for side in ("left", "right") for s in (ann.get(side) or [])]
        if starts:
            out.append(min(starts) + 10)
        break
    if cur is not None:
        out.append(cur + 10)
    seen, res = set(), []
    for v in out:
        v = max(0, min(int(v), nf - 1 if nf < 10 ** 9 else int(v)))
        if v not in seen:
            seen.add(v); res.append(v)
    return res[:3]


# ------------------------------------------------------------------ 主流程
def scan_video(dataset, video_id, video, gpu, args) -> dict:
    sys.path.insert(0, str(RECON))
    from _common.frame_plan import load_frame_plan, write_frame_plan  # noqa: E402

    obj_dir, take = _paths(dataset, video_id)
    plan = load_frame_plan(obj_dir)
    objects = dict(plan.get("objects") or {})
    if not objects:
        log(f"{video_id}: 无 frame_plan, 跳过")
        return {}
    try:
        base = read_conf(take)
    except (OSError, json.JSONDecodeError, KeyError):
        log(f"{video_id}: 读不到 confidence_complete.json(正常链没跑完?), 跳过")
        return {}

    # 第 0 格 = 正常链已经跑过的那格, 直接收编, 不重复算
    state = {}
    for oid, e in objects.items():
        r = base.get(oid)
        cands = [int(c) for c in (e.get("sam3d_candidates") or
                                  ([e["sam3d_frame"]] if e.get("sam3d_frame") is not None else []))]
        cur = e.get("sam3d_frame")
        if cur is not None and (not cands or cands[0] != cur):
            cands = [int(cur)] + [c for c in cands if c != cur]
        state[oid] = {"cands": cands, "recon": cur, "fp": e.get("fp_register_frame"),
                      "best": dict(r) if r else None, "best_recon": cur,
                      "best_fp": e.get("fp_register_frame"),
                      "done": is_safe(r), "trials": ([{"stage": 0, "recon": cur,
                                                       "fp": e.get("fp_register_frame"), **(r or {})}]
                                                     if r else [])}
    need = [o for o, s in state.items() if not s["done"]]
    log(f"{video_id}: {len(objects)} 个物体, 需精修 {len(need)} 个 {need}")
    if args.dry_run:
        for o, s in state.items():
            log(f"  {o}: 重建帧候选={s['cands']} 注册帧候选="
                f"{fp_candidates(take, o, s['fp'], objects[o])} 现状={s['best']}")
        return {}
    if not need:
        return {o: s for o, s in state.items()}

    last_cfg = None          # 最后一次真正跑过的 (重建帧, 注册帧), 收尾据此决定跑不跑
    logfh = open(take / "framescan.log", "a", encoding="utf-8")
    try:
        # ── 第一步: 换重建帧(注册帧不动) ──
        maxc = max(len(s["cands"]) for s in state.values())
        for i in range(1, min(maxc, args.max_candidates)):
            active = [o for o in need if not state[o]["done"] and i < len(state[o]["cands"])]
            if not active:
                break
            for o in objects:
                objects[o]["sam3d_frame"] = (state[o]["cands"][i] if o in active
                                             else state[o]["best_recon"])
                objects[o]["fp_register_frame"] = state[o]["best_fp"]
            write_frame_plan(obj_dir, objects)
            log(f"  第一步 候选#{i} 重建帧=" +
                str({o: objects[o]['sam3d_frame'] for o in active}))
            ok_, gpu = run_steps(MESH_STEPS + POSE_STEPS, dataset, video_id, video, gpu, logfh)
            if not ok_:
                break
            last_cfg = ({o: objects[o]["sam3d_frame"] for o in objects},
                        {o: objects[o]["fp_register_frame"] for o in objects})
            res = read_conf(take)
            for o in active:
                r = res.get(o)
                if not r:
                    continue
                st = state[o]
                st["trials"].append({"stage": 1, "recon": objects[o]["sam3d_frame"],
                                     "fp": objects[o]["fp_register_frame"], **r})
                if rank_key(r) < rank_key(st["best"] or {"refuted": 10 ** 9}):
                    st.update(best=dict(r), best_recon=objects[o]["sam3d_frame"])
                if is_safe(r):
                    st["done"] = True
                    log(f"    ✓ {o} 定于 f{objects[o]['sam3d_frame']} "
                        f"(证伪 {r['refuted']}, conf_rot {r['conf_rot']})")

        # ── 第二步: 换注册帧(网格不动, 不重建) ──
        if not args.stage1_only:
            left = [o for o in need if not state[o]["done"]]
            if left:
                fpc = {o: fp_candidates(take, o, state[o]["fp"], objects[o]) for o in left}
                maxf = max((len(v) for v in fpc.values()), default=0)
                for j in range(1, maxf):
                    act = [o for o in left if not state[o]["done"] and j < len(fpc[o])]
                    if not act:
                        break
                    for o in objects:
                        objects[o]["sam3d_frame"] = state[o]["best_recon"]
                        objects[o]["fp_register_frame"] = (fpc[o][j] if o in act
                                                           else state[o]["best_fp"])
                    write_frame_plan(obj_dir, objects)
                    log(f"  第二步 注册帧#{j}=" +
                        str({o: objects[o]['fp_register_frame'] for o in act}))
                    # 网格没变 -> 只跑位姿三步
                    ok_, gpu = run_steps(POSE_STEPS, dataset, video_id, video, gpu, logfh)
                    if not ok_:
                        break
                    last_cfg = ({o: objects[o]["sam3d_frame"] for o in objects},
                                {o: objects[o]["fp_register_frame"] for o in objects})
                    res = read_conf(take)
                    for o in act:
                        r = res.get(o)
                        if not r:
                            continue
                        st = state[o]
                        st["trials"].append({"stage": 2, "recon": st["best_recon"],
                                             "fp": objects[o]["fp_register_frame"], **r})
                        if rank_key(r) < rank_key(st["best"] or {"refuted": 10 ** 9}):
                            st.update(best=dict(r), best_fp=objects[o]["fp_register_frame"])
                        if is_safe(r):
                            st["done"] = True
                            log(f"    ✓ {o} 注册帧定于 f{objects[o]['fp_register_frame']} "
                                f"(证伪 {r['refuted']}, conf_rot {r['conf_rot']})")

        # ── 收尾: 落在赢家配置上 ──
        # ★ 最后评估的那一格若**就是赢家**, 磁盘产物已经是对的, 绝不能再跑一遍:
        #   fp_pose 非确定性, 重跑=重新抽签。2026-08-15 实测 take 23 因此自毁 ——
        #   搜索时 f63/reg10 量到 conf_rot 50.5/证伪 0(达标), 收尾重跑同一配置
        #   变成 34.0/证伪 5(不合格), 白白把找到的好结果扔了。
        win_cfg = ({o: state[o]["best_recon"] for o in objects},
                   {o: state[o]["best_fp"] for o in objects})
        for o in objects:
            objects[o]["sam3d_frame"] = state[o]["best_recon"]
            objects[o]["sam3d_source"] = f"framescan 闭环最优(试了 {len(state[o]['trials'])} 格)"
            objects[o]["fp_register_frame"] = state[o]["best_fp"]
            objects[o]["fp_source"] = "framescan 闭环最优"
        write_frame_plan(obj_dir, objects)
        final_ok = True
        if last_cfg == win_cfg:
            log("  收尾: 最后一格就是赢家, 产物已就位(跳过重跑 —— 重跑=重新抽签)")
        else:
            final_ok, gpu = run_steps(MESH_STEPS + POSE_STEPS, dataset, video_id, video, gpu, logfh)
            if not final_ok:
                # 收尾没跑成 = 磁盘产物与 frame_plan 不一致, 必须显式记账, 不能静默留坑
                log(f"★ {video_id} 收尾未完成 —— 产物与 frame_plan 不一致, 需重跑本条")
            else:
                # ★ 以磁盘上实际产物为准改写 best: 报出去的数必须等于产物里的数
                real = read_conf(take)
                for o in objects:
                    if real.get(o):
                        if state[o]["best"] and rank_key(real[o]) > rank_key(state[o]["best"]):
                            log(f"  ! {o} 收尾重跑比搜索时差(证伪 {state[o]['best']['refuted']}"
                                f"->{real[o]['refuted']}, conf_rot {state[o]['best']['conf_rot']}"
                                f"->{real[o]['conf_rot']}) —— fp_pose 非确定性, 以产物为准")
                        state[o]["best"] = dict(real[o])

        # ── ★ 确认一轮: 单次结果偶尔会骗人, 合格的必须再跑一遍同配置验证 ──
        #   实测同一 take/物体/重建帧/注册帧, 两次跑出 证伪 0 vs 51、0 vs 9、0 vs 5。
        #   大多数时候稳, 但"偶尔崩"足以把坏帧当成好帧留下。
        #   ⚠ 这里**不做"重跑到过为止"** —— 那是在优化噪声(挑运气最好的一次抽签)。
        #     确认失败就如实标 unstable, 让下游知道这条不可靠, 而不是粉饰。
        confirmed = {}
        if not args.no_confirm:
            cand_pass = [o for o in objects if is_pass(state[o]["best"])]
            if cand_pass:
                log(f"  确认轮: 对合格的 {cand_pass} 用同一配置再跑一遍验证")
                ok2, gpu = run_steps(MESH_STEPS + POSE_STEPS, dataset, video_id, video, gpu, logfh)
                if ok2:
                    again = read_conf(take)
                    for o in cand_pass:
                        r2 = again.get(o)
                        if not r2:
                            continue
                        state[o]["trials"].append({"stage": "confirm", "recon": state[o]["best_recon"],
                                                   "fp": state[o]["best_fp"], **r2})
                        same = is_pass(r2)
                        confirmed[o] = same
                        if not same:
                            log(f"  ★ {o} 确认不通过: 搜索时 证伪 {state[o]['best']['refuted']}"
                                f"/conf_rot {state[o]['best']['conf_rot']}, 确认轮 证伪 {r2['refuted']}"
                                f"/conf_rot {r2['conf_rot']} —— 判为不稳定")
                        # 产物现在是确认轮的 -> 报出去的数必须跟产物一致
                        state[o]["best"] = dict(r2)
                else:
                    log("  ! 确认轮未跑成, 保留搜索结果(未经确认)")

        # ── 压线复测: conf_rot 落在噪声带里的, 重测取中位 ──
        bl = [o for o in objects if is_borderline(state[o]["best"])]
        if bl and not args.no_recheck:
            log(f"  压线复测 {bl} × {REPEAT} 次(conf_rot 在 [{ROT_MIN},{ROT_SAFE}) 的判定不可单次信)")
            acc = {o: [] for o in bl}
            for _ in range(REPEAT - 1):        # 收尾那次算第一次
                ok_, gpu = run_steps(MESH_STEPS + POSE_STEPS, dataset, video_id, video, gpu, logfh)
                if not ok_:
                    break
                res = read_conf(take)
                for o in bl:
                    if res.get(o):
                        acc[o].append(res[o])
            for o in bl:
                seq = [state[o]["best"]] + acc[o]
                crs = sorted((x.get("conf_rot") or 0) for x in seq)
                rfs = sorted((x.get("refuted") if x.get("refuted") is not None else 10 ** 9)
                             for x in seq)
                med = {"conf_rot": crs[len(crs) // 2], "refuted": rfs[len(rfs) // 2],
                       "status": state[o]["best"].get("status"),
                       "conf_pos": state[o]["best"].get("conf_pos")}
                log(f"    {o}: 单次 conf_rot {state[o]['best'].get('conf_rot')} -> "
                    f"{len(seq)} 次中位 {med['conf_rot']} (证伪 {rfs}) "
                    f"{'仍合格' if is_pass(med) else '★不合格'}")
                state[o]["best_median"] = med
    finally:
        logfh.close()

    finalized = final_ok
    confirmed = confirmed if "confirmed" in dir() else {}
    out = {o: {"recon": s["best_recon"], "fp": s["best_fp"], "best": s["best"],
               "best_median": s.get("best_median"), "trials": s["trials"],
               "pass": is_pass(s.get("best_median") or s["best"])} for o, s in state.items()}
    for o, v in out.items():
        if o in confirmed:
            v["confirmed"] = confirmed[o]
            if not confirmed[o]:
                v["unstable"] = True
    if not finalized:
        for v in out.values():
            v["finalize_failed"] = True
    (take / "framescan.json").write_text(
        json.dumps({"video_id": video_id, "rot_min": ROT_MIN, "rot_safe": ROT_SAFE,
                    "objects": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in out.values() if v["pass"])
    log(f"{video_id} 完成: {ok}/{len(out)} 个物体合格")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id")
    ap.add_argument("--video-list", type=Path, help="每行一个 video_id")
    ap.add_argument("--video", type=Path, help="视频路径(单条时必给)")
    ap.add_argument("--video-root", type=Path, help="批量时的视频根目录")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=3,
                    help="第一步最多试几个重建帧(实测第 4 名只多捞 7%%, 第 5 名 0%%)")
    ap.add_argument("--stage1-only", action="store_true", help="只精修重建帧, 不动注册帧")
    ap.add_argument("--no-recheck", action="store_true", help="跳过压线复测")
    ap.add_argument("--no-confirm", action="store_true",
                    help="跳过确认轮(默认开: 合格的用同配置再跑一遍验证, 挡住'运气好')")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    ids = []
    if a.video_id:
        ids.append((a.video_id, a.video))
    if a.video_list and a.video_list.is_file():
        for ln in a.video_list.read_text().splitlines():
            vid = ln.strip()
            if vid:
                v = (a.video_root / f"{vid.split('__')[-1]}.mp4") if a.video_root else None
                ids.append((vid, v))
    if not ids:
        ap.error("给 --video-id 或 --video-list")

    tot = ok = 0
    for vid, v in ids:
        if v is None:
            sys.path.insert(0, str(RECON))
            from _common.paths import INTERIM_ROOT  # noqa: E402
            parts = vid.split("__")
            v = Path(INTERIM_ROOT) / "_preflight" / a.dataset / "/".join(parts[:-1]) / f"{parts[-1]}.mp4"
        try:
            res = scan_video(a.dataset, vid, v, a.gpu, a)
        except Exception as exc:                      # 单条失败不拖垮整批
            log(f"★ {vid} 异常: {exc}")
            continue
        tot += len(res); ok += sum(1 for x in res.values() if x.get("pass"))
    if tot:
        log(f"全部完成: {ok}/{tot} 个物体合格")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
