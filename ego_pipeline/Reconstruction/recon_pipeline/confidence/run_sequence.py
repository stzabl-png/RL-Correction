#!/usr/bin/env python3
"""Step: confidence —— fuse 之后给这条 take 打轨迹可信度分 + RTS 平滑 + 更新 take 清单。

薄包装: 真正的工具链在 RL_Correction/steps/step2_reconstruction/ (env: hawor, 与 fuse 同)。
评分标准与输出用法见那边的 CONFIDENCE_GUIDE.md; 本步骤的接入规划见 PIPELINE_INTEGRATION.md。

执行顺序 (cc 的种子帧质量依赖 audit 分数, audit 的旋转裁决又依赖 cc —— 所以 audit 跑两遍):
  1) pose_audit --scene            预打分(无 CT), 给 cc 挑种子帧用       ~6s  CPU
  2) cotracker_consistency         CoTracker 第二观察员                 ~1-2min GPU ~6.5GB
  3) pose_audit --scene --ct-dir   终打分(并入 CT 信号)                 ~6s  CPU
  4) rts_smoother                  平滑 + σ, 出 rts_<take>.npz          ~10s CPU
  5) take_manifest                 重新生成 take 级裁决清单              秒级 CPU
共享 json (pose_audit.json / TAKE_MANIFEST.json) 的并发安全由工具侧 poseqa_lock 保证。

路径 (均可用环境变量覆盖):
  RL_CONF_TOOLS ?= /home/lyh/Project/RL_Correction/steps/step2_reconstruction
  POSEQA_ROOT   ?= <FINAL_ROOT>/../../Data/VideoPrior/poseqa
                   (reconstruct.sh 设 RECON_FINAL_ROOT=$RR/Output/ReconstructOutput
                    => poseqa = $RR/Data/VideoPrior/poseqa, 与存量库同处)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent.parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.paths import FINAL_ROOT, final_video_dir, is_step_complete, write_step_completion  # noqa: E402

STEP = "confidence"
_DEV_TOOLS = Path("/home/lyh/Project/RL_Correction/steps/step2_reconstruction")   # 开发机
_REPO_TOOLS = RECON_ROOT.parent.parent / "confidence"                             # 仓内快照(远程机)
TOOLS = Path(os.environ.get(
    "RL_CONF_TOOLS", str(_DEV_TOOLS if _DEV_TOOLS.is_dir() else _REPO_TOOLS)))
POSEQA = Path(os.environ.get(
    "POSEQA_ROOT", str(FINAL_ROOT.parent.parent / "Data" / "VideoPrior" / "poseqa")))


def _run(tag: str, cmd: list[str], gpu: int | None = None) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[confidence] {tag}: {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run([str(c) for c in cmd], env=env).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset-root", type=Path, default=None)   # 协议兼容, 未用
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true",
                    help="额外输出 cc/conf 叠加视频 (批量默认关)")
    args, _extra = ap.parse_known_args(argv)

    scene = final_video_dir(args.dataset, args.video_id)
    if not (scene / "world_fused.npz").is_file():
        print(f"[confidence] X fuse 未完成 (无 world_fused.npz): {scene}", file=sys.stderr)
        return 2
    if is_step_complete(scene, STEP) and not args.force:
        print(f"[confidence] 已完成, 跳过 {scene} (--force 重跑)")
        return 0
    if not args.video.is_file():
        print(f"[confidence] X 源视频不存在: {args.video}", file=sys.stderr)
        return 2
    POSEQA.mkdir(parents=True, exist_ok=True)
    # 批量队列经 conda run -n hawor 进来 => sys.executable 已是 hawor;
    # run_pipeline.py 不切环境 => 用 repo_paths 的 HAWOR_PYTHON 兜底
    py = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
    if not Path(py).is_file():
        py = sys.executable
    audit_json = POSEQA / "pose_audit.json"

    rc = _run("audit(pre)", [py, TOOLS / "pose_audit.py", "--scene", scene, "--out", POSEQA])
    if rc:
        return rc
    cc_cmd = [py, TOOLS / "cotracker_consistency.py", "--scene", scene,
              "--video", args.video, "--out", POSEQA / "cc", "--audit", audit_json]
    if not args.visualize:
        cc_cmd.append("--no-viz")
    rc = _run("cotracker", cc_cmd, gpu=args.gpu)
    if rc:
        return rc
    rc = _run("audit(final)", [py, TOOLS / "pose_audit.py", "--scene", scene,
                               "--out", POSEQA, "--ct-dir", POSEQA / "cc"])
    if rc:
        return rc
    rc = _run("rts", [py, TOOLS / "rts_smoother.py", "--scene", scene,
                      "--audit", audit_json, "--out", POSEQA / "rts"])
    if rc:
        return rc
    rc = _run("manifest", [py, TOOLS / "take_manifest.py", "--audit", audit_json,
                           "--out", POSEQA / "TAKE_MANIFEST.json"])
    if rc:
        return rc
    if args.visualize:
        _run("conf_viz", [py, TOOLS / "conf_viz.py", "--audit", audit_json,
                          "--out", POSEQA / "confviz", "--ct-dir", POSEQA / "cc",
                          "--take", scene.relative_to(FINAL_ROOT)])

    # 把这条 take 的裁决摘要写进 completion marker, 批量日志里直接能看分数
    extra: dict = {"poseqa_root": str(POSEQA)}
    try:
        man = json.loads((POSEQA / "TAKE_MANIFEST.json").read_text())
        row = next((t for t in man["takes"] if str(scene).endswith(t["take"])), None)
        if row:
            extra["manifest_status"] = row["status"]   # 别叫 "status": 会覆盖 marker 的 complete 标记
            extra.update({k: row[k] for k in
                          ("conf_pos_median", "conf_rot_median",
                           "position_grade", "rotation_usable", "refuted_frames")})
    except Exception as e:                                   # 摘要失败不挡完成
        extra["summary_error"] = f"{type(e).__name__}: {e}"
    write_step_completion(scene, STEP, dataset=args.dataset,
                          video_id=args.video_id, extra=extra)
    print(f"[confidence] ✓ {args.video_id}: "
          f"{extra.get('manifest_status','?')} pos {extra.get('conf_pos_median','?')} "
          f"rot {extra.get('conf_rot_median','?')} "
          f"旋转{'可用' if extra.get('rotation_usable') else '弃用'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
