#!/usr/bin/env python3
"""资产检索(第 6 步) —— VLM 判 needs_retrieval 时, 用分件 CAD 顶替 SAM3D 重建。

============================ 为什么 ============================

重建产出的是**单个刚体网格**。若被操作物体在视频里会一分为多(瓶子→瓶身+瓶盖)或多合一,
单刚体在原理上表达不了这个动作 —— 拧盖的本质就是"盖相对瓶身转"。

实测(clip 0, screw_unscrew_bottle_cap):

    单件 SAM3D 网格(24.7cm)   conf_pos 50  conf_rot 23  -> 旋转**弃用**
    CAD 瓶身 + CAD 瓶盖        conf_pos 82  conf_rot 34  -> 旋转**可用**

============================ 命中时怎么跳过 sam3d ============================

`fp_pose` / `fuse` 只从 `sam3d_scale/objects/<oid>/object_mesh_scaled_final.obj` 取网格,
该目录的其它产物下游都不读。所以把 CAD 放进这个位置, `sam3d` 与 `sam3d_scale` 就没必要跑了
(CAD 本身米制, 也不需要尺度估计)。

跳过的做法: 本步**替这两步写完成标记**, 管线的 is_step_complete 自然就跳过它们。
标记里带 `skipped_by=retrieval` 与 `note`, 免得后来的人以为 sam3d 真跑过。
`--force` 会让它们重新跑, 这是有意的逃生阀。

★ 未命中(VLM 说不需要 / 资产库没有 / 部件分配不可靠)时**什么都不做**, 让 sam3d 正常跑。
  宁可回退重建, 也不要把错的部件装上去 —— 装错不会报错, 只会让下游拿着错的几何做接触和 RL。
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

from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402

REPO_ROOT = RECON_ROOT.parent.parent.parent
EGO = REPO_ROOT / "ego_pipeline"
STEP = "retrieval"
PROVIDED = ("sam3d", "sam3d_scale")


def _hawor_py() -> str:
    py = os.environ.get("HAWOR_PYTHON", "/home/lyh/anaconda3/envs/hawor/bin/python")
    return py if Path(py).is_file() else sys.executable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)          # 协议兼容; 本步纯 CPU
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--task", default=None, help="资产库任务名(默认从 video_id 切)")
    args, extra = ap.parse_known_args(argv)

    step_dir = interim_step_dir(args.dataset, args.video_id, STEP)
    if is_step_complete(step_dir, STEP) and not args.force:
        print(f"[retrieval] 已完成, 跳过 {args.video_id} (--force 重跑)")
        return 0

    cmd = [_hawor_py(), EGO / "bin" / "retrieve_assets.py",
           "--dataset", args.dataset, "--video-id", args.video_id, "--video", args.video]
    if args.task:
        cmd += ["--task", args.task]
    print(f"[retrieval] {' '.join(map(str, cmd))}", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT)).returncode

    # retrieve_assets.py 把结果写在 take 的 interim 根目录(不在某个 step 子目录下)
    itm_root = Path(os.environ.get("RECON_INTERIM_ROOT",
                                   REPO_ROOT / "Output/ReconstructOutput/interim"))
    rj = itm_root / args.dataset / args.video_id / "retrieval.json"
    doc = json.loads(rj.read_text()) if rj.is_file() else {"status": "missing"}

    if rc == 0 and doc.get("status") == "ok":
        for s in PROVIDED:
            write_step_completion(
                interim_step_dir(args.dataset, args.video_id, s), s,
                dataset=args.dataset, video_id=args.video_id,
                extra={"skipped_by": STEP,
                       "note": f"本步**没有真的运行 {s}**。网格由资产库 CAD 提供, "
                               f"已放进 sam3d_scale/objects/<oid>/object_mesh_scaled_final.obj "
                               f"(fp_pose/fuse 只读这个路径)。--force 可让它真跑。",
                       "assignment": doc.get("assignment")})
        print(f"[retrieval] ✓ 命中资产, 已替 {'/'.join(PROVIDED)} 写完成标记(跳过重建)")
    else:
        # rc==3 是"本条不用资产"的正常返回, 不是失败
        print(f"[retrieval] 不使用资产({doc.get('reason') or 'rc=%d' % rc}) -> sam3d 正常跑")

    write_step_completion(step_dir, STEP, dataset=args.dataset, video_id=args.video_id,
                          extra={"retrieval_json": str(rj), "hit": rc == 0 and
                                 doc.get("status") == "ok", "detail": doc})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
