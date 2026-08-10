#!/usr/bin/env python3
"""v17A 自动标注驱动 —— 视频没有物体标注时自动产出, 代替 --web 人工点选。

链路(全部幂等, 产物在就跳过):
  1. HOI-DETR 逐帧检测            (conda env codetr, GPU)   -> detections.json
  2. 实例发现 + SAM2 mask 传播    (conda env codetr, GPU)   -> episode mask_sequence.json
  3. 选择策略(本驱动的职责, adapter 刻意不做主):
       实例   默认 instance_0001 —— v17A 编号语义 = 第一个被接触的物体(任务主物体)
       重建帧 默认该实例最早的 accepted 帧 —— 通常手尚未接触、遮挡最小
       (2026-08-10 pour/11 人工选择的复刻: 杯子 f3 静置桌面)
  4. 开朗的 adapter 转 sam2_object 落盘格式(+label_prompt+完成标记), 重建管线自动跳过标注

⚠ adapter 只认 episode 级 mask_sequence.json (schema persistent_mask_sequence_v1),
  不是视频级 video_mask_sequence.json —— 开朗文档里的示例有误, 别改回去。
⚠ 透明物体过滤是上游 VLM 的职责(重建开始前), 本驱动不看材质。
⚠ 多实例视频 v1 只注册主实例; 多物体注册(--instance all)留待扩展。

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
    ap.add_argument("--instance", default="auto", help="instance_000N 或 auto(=第一个被接触的物体)")
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
        run("2/3 实例发现 + SAM2 传播 (~1-2 分钟 GPU)", [
            "conda", "run", "--no-capture-output", "-n", "codetr", "python", "-m",
            "experiments.hoi_detr.run_instance_video_segmentation",
            "--video", video, "--detections", det, "--output-dir", inst_out,
            "--sam2-root", TP / "sam2", "--checkpoint", ck_sam2,
            "--model-cfg", "configs/sam2.1/sam2.1_hiera_l.yaml",
            "--gpu", a.gpu], cwd=V17A_ROOT)
        manifest_path = find_episode_manifest(inst_out)
    if manifest_path is None:
        raise SystemExit("[auto-label] X v17A 没产出 ready 的 episode manifest")

    inst, frame = pick(json.loads(manifest_path.read_text()), a.instance, a.recon_frame)
    print(f"[auto-label] 选择: {inst} @ 帧{frame}  ({manifest_path})")
    run("3/3 adapter 转 sam2_object 格式", [
        *HAWOR_CMD, RECON_ROOT / "recon_kailang" / "v17_mask_adapter" / "import_v17a_masks.py",
        "--dataset", a.dataset, "--video-id", vid, "--video", video,
        "--manifest", manifest_path, "--source-object-id", inst,
        "--reconstruction-frame", frame, "--object-name", a.object_name],
        cwd=RECON_ROOT)
    print(f"[auto-label] ✓ {vid}: {inst} 帧{frame} -> sam2_object 就位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
