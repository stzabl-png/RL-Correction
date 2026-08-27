"""把 `scene_layout.json` 的摆放做成可在 Isaac 里看的 replay npz。

用途: 肉眼核对"逐物体听手"摆得对不对 —— 物体**固定**在摆放位姿(不再跟随重建轨迹),
手仍然按重建轨迹动。于是要看的就一件事:

    每只手走到它那个物体的接触起始帧时, 手是不是正好到了该物体上。

⚠ 这不是训练数据, 只是核对用的预览。真正的 RL 初始状态只用摆放的**首帧**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    r = dict(np.load(a.replay, allow_pickle=True))
    lay = json.loads(a.layout.read_text())
    oids = [str(x) for x in r["object_ids"]] if "object_ids" in r else ["object_0"]
    T = len(r["joints_left"])

    poses = []
    for oid in oids:
        e = (lay.get("objects") or {}).get(oid)
        if e is None:                      # 没摆放的物体保留原轨迹, 不静默丢掉
            i = oids.index(oid)
            poses.append(np.asarray(r["obj_pose_all"][i], np.float32))
            print(f"  {oid}: 无摆放条目, 保留重建轨迹")
            continue
        p7 = np.concatenate([e["pos"], e["quat_wxyz"]]).astype(np.float32)
        poses.append(np.tile(p7, (T, 1)))
        print(f"  {oid}: 固定在 听{'左' if e['anchor_hand'] == 'left' else '右'}手@f"
              f"{e['onset_frame']} 的摆放位姿 {np.round(e['pos'], 3).tolist()}")
    r["obj_pose_all"] = np.stack(poses)
    r["obj_pose"] = poses[0]
    if "obj_valid_all" in r:
        r["obj_valid_all"] = np.ones_like(np.asarray(r["obj_valid_all"]), dtype=bool)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **r)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
