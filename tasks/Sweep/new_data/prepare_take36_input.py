"""Reproduce take36's semantic input adapter; keep source data unchanged."""
from pathlib import Path
import argparse, json, os
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument("--config", default="tasks/Sweep/new_data/configs/take_36_v3.json")
a = p.parse_args()
cfg = json.loads((ROOT / a.config).read_text())
src = ROOT / cfg["source_data_dir"]
dst = ROOT / cfg["data_dir"]
assert src.name == "36" and dst.is_relative_to(ROOT / "tasks/Sweep/new_data/prepared")
count = cfg["source_trim"]["kept_frames"][1] + 1
assert cfg["source_trim"]["kept_frames"][0] == 0 and count == 294

def trim(v):
    if v.ndim and v.shape[0] == 300:
        return v[:count]
    if v.ndim > 1 and v.shape[1] == 300:
        return v[:, :count]
    return v

for name in ("poseqa", "objects", "retarget"):
    (dst / name).mkdir(parents=True, exist_ok=True)
for target, source in ((0, 1), (1, 0)):
    z = np.load(src / f"poseqa/rts_sweep_dustpan_36_object_{source}.npz",
                allow_pickle=True)
    np.savez_compressed(dst / f"poseqa/rts_sweep_dustpan_36_object_{target}.npz",
                        **{k: trim(z[k]) for k in z.files})
    for output, original in (
        (dst / f"objects/object_{target}", src / f"objects/object_{source}"),
        (dst / f"retarget/object_{target}_textured.usd",
         src / f"retarget/object_{source}_textured.usd"),
    ):
        if output.exists():
            assert output.resolve() == original.resolve()
        else:
            output.symlink_to(original)

original = np.load(src / "replay_world.npz", allow_pickle=True)
replay = {k: trim(original[k]) for k in original.files}
for key in ("obj_pose_all", "obj_verts_local", "obj_valid_all"):
    replay[key] = replay[key][[1, 0]]
# obj_pose is the original primary broom and must remain so.
for side in ("left", "right"):
    assert np.array_equal(replay["joints_" + side], original["joints_" + side][:count])
    replay["phase_" + side] = np.ones(count, dtype=np.int8)
replay["phase_obj"] = np.ones(count, dtype=np.int8)
replay_path = dst / "replay_world.npz"
np.savez_compressed(replay_path, **replay)

for side in ("left", "right"):
    z = np.load(src / f"ref_qpos_{side}.npz", allow_pickle=True)
    payload = {k: trim(z[k]) for k in z.files}
    payload["original_source_mtime"] = payload["source_mtime"]
    payload["source_mtime"] = np.array(os.path.getmtime(replay_path))
    payload["adapter_provenance"] = np.array(
        "Human joints unchanged; source prefix0..293; object axes reordered only")
    output = dst / f"ref_qpos_{side}.npz"
    assert not output.is_symlink()
    np.savez_compressed(output, **payload)

camera = np.load(src / "world_fused.npz", allow_pickle=True)["c2w"]
np.savez_compressed(dst / "world_fused.npz", c2w=trim(camera),
                    source=np.array(str(src / "world_fused.npz")),
                    scope=np.array("camera-only adapter"))
layout = dict(schema_version="sweep_held_scene_v1",
              scene_table_z=cfg["scene_table_z"], source_frame=0, objects={})
for i, role, hand in ((0, "dustpan", "left"), (1, "broom", "right")):
    pose = replay["obj_pose_all"][i, 0]
    layout["objects"][f"object_{i}"] = dict(
        identity=role, anchor_hand=hand, pos=pose[:3].tolist(),
        quat_wxyz=pose[3:7].tolist())
(ROOT / cfg["scene_layout"]).write_text(json.dumps(layout, indent=2) + "\n")
print(dst)
