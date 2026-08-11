"""造一个只换参考帧的平行 take, 复用上游产物, 只重跑 sam3d->sam3d_scale->fp_pose->fuse。

⚠ vipe / sam3_hands 的产物**按 video_id 命名**(`vipe/depth/<video_id>.zip`、
`sam3_hands/<video_id>/`), 所以不能整目录软链 —— 变体的 video_id 不同, 下游会找不到文件
(实测 sam3d_scale 报 FileNotFoundError)。这里逐文件软链并改名。
"""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np

I = Path.home()/"Reconstruct_and_Retarget/Output/ReconstructOutput/interim/arctic"
src_id, frame = sys.argv[1], int(sys.argv[2])
dst_id = f"{src_id}_f{frame}"
src, dst = I/src_id, I/dst_id
if dst.exists(): shutil.rmtree(dst)
dst.mkdir(parents=True)

n_ren = 0
for step in ("vipe", "sam3_hands"):
    for p in (src/step).rglob("*"):
        rel = p.relative_to(src/step)
        out = dst/step/Path(str(rel).replace(src_id, dst_id))
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.symlink_to(p)
            n_ren += (src_id in str(rel))

(dst/"sam2_object").mkdir(exist_ok=True)
(dst/"sam2_object"/"video_segmentation").symlink_to(src/"sam2_object"/"video_segmentation")
for f in (src/"sam2_object").glob("*.json"):
    if f.name != "label_prompt.json":
        shutil.copy2(f, dst/"sam2_object"/f.name)

mp = src/"sam2_object"/"video_segmentation"/"masks"/f"frame_{frame:06d}_masks"/"object_0.png"
m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
assert m is not None and (m > 127).any(), f"该帧没有 object_0 mask: {mp}"
d = cv2.distanceTransform((m > 127).astype("uint8"), cv2.DIST_L2, 5)
y, x = np.unravel_index(int(np.argmax(d)), d.shape)
old = json.loads((src/"sam2_object"/"label_prompt.json").read_text())
(dst/"sam2_object"/"label_prompt.json").write_text(json.dumps(
    {"schema_version": old.get("schema_version", "sam2_object_prompt_v2"),
     "objects": [{"object_id": "object_0", "frame_idx": frame,
                  "points": [[float(x), float(y)]], "labels": [1], "locked": True}],
     "provenance": {"variant_of": src_id, "reason": "只换参考帧做真值对照",
                    "original_frame": old["objects"][0]["frame_idx"]}}, indent=1))
print(f"{dst_id}: 参考帧 {old['objects'][0]['frame_idx']} -> {frame}, 点 ({x},{y}), "
      f"mask {int((m>127).sum())}px, 改名软链 {n_ren} 个")
