#!/usr/bin/env python3
"""v17A 自动标注驱动 —— 视频没有物体标注时自动产出, 代替 --web 人工点选。

链路(全部幂等, 产物在就跳过):
  1. HOI-DETR 逐帧检测            (conda env codetr, GPU)   -> detections.json
  2. 实例发现 + SAM2 mask 传播    (conda env codetr, GPU)   -> episode mask_sequence.json
  3. 选择策略(本驱动的职责, adapter 刻意不做主):
       实例   默认 instance_0001 —— v17A 编号语义 = 第一个被接触的物体(任务主物体)
       重建帧 默认该实例最早的 accepted 帧 —— 通常手尚未接触、遮挡最小
       (2026-08-10 pour/11 人工选择的复刻: 杯子 f3 静置桌面)
  4. 只落 label_prompt.json(一帧+一个点), 重建管线自己的 SAM2 传播全片
     (2026-08-10 改: 原先走 import_v17a_masks.py 直接搬 v17A 的 mask, 等于把 v17A
      "挑重建帧"的严格质量门 min_largest_component_fraction=0.9 也套到了全片覆盖上。
      screw27 实测: 手臂横在瓶前使 135/140 帧被判 fragmented_mask, 搬过来只剩 5 帧,
      接触检测塌成 左[] 右[[152,154]]。改成只给点后同一实例同一帧拿到 188/188 帧,
      与人工标注 IoU 0.78、无一帧跟丢。质量门继续管选帧, 不再决定覆盖。)

⚠ 两份 manifest 各有各的用途, 别混:
  - **episode 级** mask_sequence.json (persistent_mask_sequence_v1): 单实例路径用这个;
    开朗的 mask 搬运 adapter 也只认它(他 README 的视频级示例有误)。
  - **视频级** video_mask_sequence.json (persistent_video_mask_sequence_v1): --instance all
    用这个 —— 注册出来的部件只出现在这里(clip4: object_0001+object_0002, 盖在 f59 注册;
    同一条 clip 的 episode 级只有 instance_0001)。喂错会让多物体静默塌成单物体。
⚠ VLM 透明门已内置(2026-08-10, 规则 v2): 逐实例判材质, 只过滤"空透明"(高置信);
  四案例校准全对(3号清水瓶滤/pour茶瓶留/金属瓶留/灰杯留)。AUTO_LABEL_VLM_GATE=0 关闭;
  服务不可达时 fail-open(全放行+醒目警告)。全部实例被滤时退出码 3(=本视频跳过, 非错误)。
⚠ 默认只注册主实例; --instance all 注册全部(走 tools/v17a_multi_object_prompt.py,
  一次重建 pass 内完成 -> 各物体共享同一世界系; 分开跑再合并会因 ViPE 焦距不确定而错位)。

用法(reconstruct.sh 自动调用; 也可手动):
  python3 auto_label_v17a.py --dataset egodex --dataset-root <root> \
      --video <绝对路径.mp4> [--instance instance_0002] [--recon-frame 40] \
      [--object-name cup] [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EGO = HERE.parent
RR_ROOT = EGO.parent
sys.path.insert(0, str(EGO))
from repo_paths import RECON_PIPELINE  # noqa: E402

RECON_ROOT = RECON_PIPELINE.parent            # ego_pipeline/Reconstruction
V17A_ROOT = RR_ROOT / "experimental" / "hoi_detr_v17a"
TP = RECON_ROOT / "third_party"
_HAWOR_PY = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
# 远程机没有本机 anaconda 路径 → 退回 conda run(hawor env 两台服务器都有)
HAWOR_CMD = ([_HAWOR_PY] if Path(_HAWOR_PY).is_file()
             else ["conda", "run", "--no-capture-output", "-n", "hawor", "python"])


def video_id_for(video: Path, root: Path) -> str:
    """与 _common.dataset 通用 discover 相同的 id 规则: 相对 root 去后缀, / -> __"""
    rel = video.resolve().relative_to(root.resolve())
    return str(rel.with_suffix("")).replace("/", "__")


def run(tag: str, cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print(f"[auto-label] {tag}", flush=True)
    e = dict(os.environ)
    e.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if env:
        e.update(env)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=e)
    if r.returncode:
        raise SystemExit(f"[auto-label] X {tag} 失败 (rc={r.returncode}); 兜底: ./reconstruct.sh --web 人工标注")


def run_soft(tag: str, cmd: list[str], cwd: Path) -> int:
    """跑一步但不因失败中止, 返回 returncode —— 交给调用方判断产物够不够用。"""
    print(f"[auto-label] {tag}", flush=True)
    e = dict(os.environ)
    e.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    e.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=e).returncode


def find_episode_manifest(out_dir: Path) -> Path | None:
    """找最新 ready 的 episode 级 mask_sequence.json (adapter 认的 schema)。"""
    best = None
    for p in sorted(out_dir.glob("episode_*/sequence_attempt_*/mask_sequence.json")):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("schema_version") == "persistent_mask_sequence_v1" and m.get("status") == "ready":
            best = p
    return best


def write_label_prompt(manifest: dict, inst: str, frame: int, sam2_dir: Path,
                       object_name: str) -> tuple[float, float]:
    """Hand the pipeline a click, not v17A's masks.

    v17A's per-frame gate (min_largest_component_fraction=0.9) exists to answer
    "which single frame is good enough to reconstruct from".  Importing its masks
    wholesale made that gate decide FULL-VIDEO coverage as well, which is a different
    question with a much looser answer.  Measured on screw_unscrew_bottle_cap/27: a
    forearm across the bottle splits the mask on 135 of 140 frames, so importing gave
    5 usable frames and contact detection collapsed to left[] right[[152,154]].  Feeding
    the same instance and frame in as a prompt point instead, and letting the pipeline's
    own SAM2 propagate, gives all 188 frames -- IoU 0.78 against a human label, with no
    frame lost or drifting.

    The click is the mask's pole of inaccessibility (deepest interior pixel), not its
    centroid: a centroid can fall outside a C-shaped or hand-split mask and would prompt
    SAM2 on background.
    """
    import cv2
    import numpy as np

    entry = None
    for f in manifest["frames"]:
        if int(f["frame_idx"]) == int(frame):
            entry = (f.get("objects") or {}).get(inst)
            break
    if entry is None:
        raise SystemExit(f"[auto-label] X 帧{frame} 没有 {inst} 的记录")
    src = entry.get("mask") or entry.get("raw_mask")
    if not src or not Path(src).is_file():
        raise SystemExit(f"[auto-label] X 帧{frame} 的 mask 文件缺失: {src}")
    m = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if m is None or not (m > 127).any():
        raise SystemExit(f"[auto-label] X 帧{frame} 的 mask 为空: {src}")
    dist = cv2.distanceTransform((m > 127).astype("uint8"), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)

    sam2_dir.mkdir(parents=True, exist_ok=True)
    prompt = {"schema_version": "sam2_object_prompt_v2",
              "objects": [{"object_id": "object_0", "frame_idx": int(frame),
                           "points": [[float(x), float(y)]], "labels": [1],
                           "locked": True, "name": object_name}]}
    (sam2_dir / "label_prompt.json").write_text(json.dumps(prompt, indent=1))
    return float(x), float(y)


EXIT_ALL_FILTERED = 3        # 与 reconstruct.sh 的约定: 本视频合法跳过, 不算失败
FP_ONSET_OFFSET = 10         # FP 配准帧 = 交互开始帧 + 10(同事实测, 2026-08-11 定)


def write_plan(sam2_dir: Path, entries: dict, n_frames: int | None) -> None:
    """frame_plan.json: fp=onset+10(有 onset 才写), sam3d 留空给杜邦的选帧器。
    entries: {final_object_id: v17A 实例 meta dict}"""
    sys.path.insert(0, str(RECON_PIPELINE))
    from _common.frame_plan import write_frame_plan  # noqa: E402
    objs = {}
    for oid, meta in entries.items():
        onset = meta.get("interaction_onset_frame", meta.get("activation_frame"))
        row = {"sam3d_frame": None,
               "sam3d_source": "reserved(dubang 选帧器待接; null=用 prompt 帧)"}
        if onset is not None:
            f = int(onset) + FP_ONSET_OFFSET
            if n_frames:
                f = min(f, n_frames - 1)
            row["fp_register_frame"] = f
            row["fp_source"] = f"interaction_onset({int(onset)})+{FP_ONSET_OFFSET}"
        else:
            row["fp_register_frame"] = None
            row["fp_source"] = "no_onset_metadata(用 prompt 帧)"
        objs[oid] = row
    write_frame_plan(sam2_dir, objs)
    print(f"[auto-label] frame_plan: " + ", ".join(
        f"{k}: fp={v['fp_register_frame']}" for k, v in objs.items()))


def gate_samples(manifest: dict, inst: str, n: int = 3) -> list[tuple[int, "Path"]]:
    """该实例跨时间取 n 个 (frame, mask) 样本给透明门。"""
    rows = [(int(f["frame_idx"]), (f.get("objects") or {}).get(inst))
            for f in manifest["frames"]]
    rows = [(fi, o) for fi, o in rows if o and o.get("raw_mask")]
    if not rows:
        return []
    idx = [0, len(rows) // 2, len(rows) - 1][:max(1, min(n, len(rows)))]
    return [(rows[i][0], Path(rows[i][1]["raw_mask"])) for i in sorted(set(idx))]


def vlm_gate(video: Path, manifest: dict, ids: list[str]) -> dict[str, dict]:
    """逐实例过 VLM 透明门。返回 {inst: verdict}; 服务不可达 → fail-open 返回 {}。"""
    if os.environ.get("AUTO_LABEL_VLM_GATE", "1") == "0":
        print("[auto-label] 透明门已禁用(AUTO_LABEL_VLM_GATE=0)")
        return {}
    import vlm_transparency_gate as G
    out: dict[str, dict] = {}
    for inst in ids:
        sm = gate_samples(manifest, inst)
        if not sm:
            continue
        try:
            v = G.judge_instance(video, sm)
        except G.VLMUnavailable as e:
            print(f"[auto-label] ⚠⚠ 透明门 fail-open(全放行): {e}", flush=True)
            return {}
        out[inst] = v
        print(f"[auto-label] 透明门 {inst}: {v['material']}({v['confidence']}) — {v['evidence'][:60]}")
    return out


def pick(manifest: dict, instance: str, recon_frame: str) -> tuple[str, int]:
    ids = manifest.get("object_ids") or []
    if not ids:
        raise SystemExit("[auto-label] X v17A 没发现任何物体实例")
    inst = instance if instance != "auto" else ("instance_0001" if "instance_0001" in ids else ids[0])
    if inst not in ids:
        raise SystemExit(f"[auto-label] X 实例 {inst} 不存在, 可选: {ids}")
    if recon_frame != "auto":
        return inst, int(recon_frame)
    for fr in manifest["frames"]:                      # 已按帧序; 最早 accepted 帧
        o = (fr.get("objects") or {}).get(inst)
        if o and o.get("status") == "accepted" and o.get("raw_mask"):
            return inst, int(fr["frame_idx"])
    raise SystemExit(f"[auto-label] X 实例 {inst} 没有任何 accepted 帧")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--video-id", default=None, help="缺省按通用规则从路径推导")
    # env default so multi-object can be switched on for a whole batch without touching
    # reconstruct.sh's argument forwarding, which cannot pass a flag that takes a value
    # (its `-*` branch shifts by 1, so the value is read as another video path).
    ap.add_argument("--instance", default=os.environ.get("AUTO_LABEL_INSTANCE", "auto"),
                    help="instance_000N / auto(=第一个被接触的物体) / all(=全部实例, 多物体); "
                         "默认可由环境变量 AUTO_LABEL_INSTANCE 设置")
    ap.add_argument("--recon-frame", default="auto", help="帧号或 auto(=最早 accepted 帧)")
    ap.add_argument("--object-name", default="object")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    video = a.video.resolve()
    vid = a.video_id or video_id_for(video, a.dataset_root)

    # 幂等: 重建侧标注已就位 -> 什么都不做
    sam2_dir = None
    try:
        sys.path.insert(0, str(RECON_PIPELINE))
        from _common.paths import interim_step_dir, is_step_complete  # noqa: E402
        sam2_dir = interim_step_dir(a.dataset, vid, "sam2_object")
        if is_step_complete(sam2_dir, "sam2_object") and not os.environ.get("AUTO_LABEL_FORCE"):
            print(f"[auto-label] 标注已就位, 跳过 {vid}")
            return 0
        # prompt 已写但 sam2_object 还没跑完(上次中断/人工改过 prompt) -> 别覆盖
        if (sam2_dir / "label_prompt.json").is_file() and not os.environ.get("AUTO_LABEL_FORCE"):
            print(f"[auto-label] label_prompt 已存在, 跳过 {vid} (AUTO_LABEL_FORCE=1 重写)")
            return 0
    except Exception as e:  # noqa: BLE001 - 查不了就当没有, 继续跑
        print(f"[auto-label] 完成态检查失败({e}), 继续", flush=True)

    ck_hoi = TP / "HOI-DETR" / "checkpoints" / "epoch_5.pth"
    ck_sam2 = TP / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    for ck in (ck_hoi, ck_sam2):
        if not ck.is_file():
            raise SystemExit(f"[auto-label] X 缺权重 {ck}")

    out = V17A_ROOT / "data" / "interim" / a.dataset / vid
    det = out / "hoi_detr_probe" / "detections.json"
    inst_out = out / "instance_pipeline_v17a"

    if not det.is_file():
        run("1/3 HOI-DETR 检测 (~2-3 分钟 GPU)", [
            "conda", "run", "--no-capture-output", "-n", "codetr", "python", "-m",
            "experiments.hoi_detr.run_sequence",
            "--dataset", a.dataset, "--video-id", vid, "--video", video,
            "--gpu", a.gpu, "--hoi-detr-root", TP / "HOI-DETR",
            "--checkpoint", ck_hoi, "--checkpoint-authorized",
            "--source-revision", "1b367292f3833afd64a204bd4d9d84519541d035",
            "--checkpoint-revision", "85719ac7bf20b8b67e26206faddf0d9582052046",
            "--frame-stride", "1"], cwd=V17A_ROOT)

    manifest_path = find_episode_manifest(inst_out)
    if manifest_path is None:
        # 走到这里说明目录里没有任何 ready 的 manifest。但上一次崩溃(CUDA 错误/被 kill)可能
        # 留下了半截 episode 目录, 而 run_instance_video_segmentation 拒绝写入非空目录
        # (FileExistsError), 于是每次重试都在同一处瞬间失败。既然已确认没有可用产物,
        # 残留就是纯垃圾, 清掉再跑 —— 否则一次崩溃会永久毒化这条视频。
        if inst_out.exists() and any(inst_out.iterdir()):
            import shutil as _sh
            print(f"[auto-label] 清理无可用 manifest 的残留目录: {inst_out}", flush=True)
            _sh.rmtree(inst_out, ignore_errors=True)
        # 这一步的最后阶段是"跨周期视觉身份链接"(产出视频级 manifest)。它可能失败而**逐 episode
        # 的 mask 完好** —— arctic/s05__laptop_grab_01 实测: status=failed_global_identity_linking
        # (第3周期匹配分 0.584 vs 次高 0.308 不够决定性), 但 4 个 episode 全是 ready,
        # episode_00 就覆盖 555 帧。整步判失败会把这些产物一起丢掉, 逼人去做本可避免的人工标注。
        rc = run_soft("2/3 实例发现 + SAM2 传播 (~1-2 分钟 GPU)", [
            "conda", "run", "--no-capture-output", "-n", "codetr", "python", "-m",
            "experiments.hoi_detr.run_instance_video_segmentation",
            "--video", video, "--detections", det, "--output-dir", inst_out,
            "--sam2-root", TP / "sam2", "--checkpoint", ck_sam2,
            "--model-cfg", "configs/sam2.1/sam2.1_hiera_l.yaml",
            "--gpu", a.gpu], cwd=V17A_ROOT)
        manifest_path = find_episode_manifest(inst_out)
        if rc and manifest_path is None:
            raise SystemExit(f"[auto-label] X 实例发现失败 (rc={rc}) 且无可用 episode manifest; "
                             f"兜底: ./reconstruct.sh --web 人工标注")
        if rc:
            print(f"[auto-label] ! 实例发现后段失败 (rc={rc}), 但 episode 级 mask 可用, 继续", flush=True)
    if manifest_path is None:
        raise SystemExit("[auto-label] X v17A 没产出 ready 的 episode manifest")

    if sam2_dir is None:                      # the completeness probe above failed
        sys.path.insert(0, str(RECON_PIPELINE))
        from _common.paths import interim_step_dir  # noqa: E402
        sam2_dir = interim_step_dir(a.dataset, vid, "sam2_object")
    if a.instance == "all":
        # Every interaction instance in ONE label_prompt, so the pipeline reconstructs them
        # in a single pass and they share a world frame.  Merging separate per-object runs
        # instead does NOT work: ViPE's focal estimate is not deterministic (58-98 px and
        # 3-13 cm apart across two runs of the same video), so each run lands in its own
        # world and the merge silently misplaces one object relative to the other.
        # hand masks let the tool prefer a frame where the hand is not on the object;
        # at label time the recon has usually not produced them yet, and the tool treats
        # "no hand masks" as "no objection", so this stays optional.
        take = None
        try:
            from _common.paths import final_video_dir  # noqa: E402
            t = final_video_dir(a.dataset, vid)
            take = t if (t / "masks/hands/frames").is_dir() else None
        except Exception:  # noqa: BLE001 - hand masks are an optional tie-breaker
            pass
        # VIDEO-level manifest here, not the episode-level one the single-instance path
        # uses.  The episode manifest carries one interaction instance (clip 4: just
        # instance_0001); the video manifest is where the registered components land
        # (clip 4: object_0001 + object_0002, the cap registered at frame 59).  Feeding
        # the episode file makes multi-object silently collapse to a single object.
        vman = inst_out / "video_mask_sequence" / "video_mask_sequence.json"
        if not vman.is_file():
            # 静默塌成单物体是这条链上最容易犯又最难发现的错, 所以退回必须刺眼且落到产物里。
            print(f"[auto-label] !! 要求 --instance all 但视频级 manifest 缺失 ({vman.name}); "
                  f"跨周期身份链接多半失败了。退回 episode 级 = **只注册主实例**。", flush=True)
            print(f"[auto-label] !! 该 take 是单物体标注, 若它本该是多部件, 下游会缺一个部件。",
                  flush=True)
            manifest = json.loads(manifest_path.read_text())
            inst, frame = pick(manifest, "auto", a.recon_frame)
            n = write_label_prompt(manifest, inst, frame, sam2_dir, a.object_name)
            (sam2_dir / "label_prompt_degraded.json").write_text(json.dumps(
                {"requested": "all", "delivered": "single_instance", "instance": inst,
                 "frame": int(frame), "reason": "video-level manifest missing "
                 "(global identity linking failed); episode-level masks were usable"},
                indent=1))
            print(f"[auto-label] ✓(降级) {vid}: {inst} 帧{frame} -> label_prompt.json (点 {n})")
            return 0
        vm = json.loads(vman.read_text())
        import vlm_transparency_gate as G
        verdicts = vlm_gate(video, vm, vm.get("object_ids") or [])
        excluded = [i for i, v in verdicts.items() if G.should_filter(v)]
        if excluded and len(excluded) == len(vm.get("object_ids") or []):
            print(f"[auto-label] 全部实例为空透明, 本视频跳过(透明规则 v2): {excluded}")
            return EXIT_ALL_FILTERED
        cmd = [sys.executable, RR_ROOT / "tools" / "v17a_multi_object_prompt.py",
               vman, "--step-dir", sam2_dir]
        for i in excluded:
            cmd += ["--exclude", i]
        if take is not None:
            cmd += ["--recon-take-dir", take]
        run("3/3 多物体 label_prompt", cmd, cwd=RR_ROOT)
        prompt = json.loads((sam2_dir / "label_prompt.json").read_text())
        n_obj = len(prompt["objects"])
        vmeta = vm.get("objects") or {}
        entries = {}
        for po in (prompt.get("provenance") or {}).get("objects") or []:
            src = po.get("instance") or po.get("source_object_id")
            fid = po.get("object_id")
            if fid and src and src in vmeta:
                entries[fid] = vmeta[src]
        if not entries:                     # provenance 对不上就逐个顺位对(保底)
            ids = [o["object_id"] for o in prompt["objects"]]
            entries = {fid: vmeta.get(src, {}) for fid, src in zip(ids, vm.get("object_ids") or [])}
        write_plan(sam2_dir, entries, vm.get("num_frames"))
        print(f"[auto-label] ✓ {vid}: {n_obj} 个物体 -> label_prompt.json, "
              f"sam2_object 将逐个传播全片")
        return 0

    manifest = json.loads(manifest_path.read_text())
    inst, frame = pick(manifest, a.instance, a.recon_frame)
    import vlm_transparency_gate as G
    verdicts = vlm_gate(video, manifest, [inst])
    if verdicts.get(inst) and G.should_filter(verdicts[inst]):
        others = [i for i in (manifest.get("object_ids") or []) if i != inst]
        fallback = None
        if a.instance == "auto":
            for cand in others:                      # 主实例被滤 → 顺位尝试其它实例
                v2 = vlm_gate(video, manifest, [cand]).get(cand)
                if v2 is None or not G.should_filter(v2):
                    fallback = cand
                    break
        if fallback is None:
            print(f"[auto-label] 实例 {inst} 为空透明且无可替补, 本视频跳过(透明规则 v2)")
            return EXIT_ALL_FILTERED
        inst, frame = pick(manifest, fallback, a.recon_frame)
        print(f"[auto-label] 透明门改选替补实例: {inst}")
    print(f"[auto-label] 选择: {inst} @ 帧{frame}  ({manifest_path})")
    n = write_label_prompt(manifest, inst, frame, sam2_dir, a.object_name)
    write_plan(sam2_dir, {"object_0": (manifest.get("objects") or {}).get(inst, {})},
               len(manifest.get("frames") or []) or None)
    print(f"[auto-label] ✓ {vid}: {inst} 帧{frame} -> label_prompt.json "
          f"(点 {n}), sam2_object 将自行传播全片")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
