"""把当前正式 v2 母带逐行录成一条纯参考诊断视频。

用途是人工核对 P0->P1 cuRobo、P1->task_end 连续数据轨迹、task_end->P0
cuRobo 的几何和双手时序。机器人与物体都逐行写入母带状态，因此它不宣称
零残差物理成功；物理可训练性由 smoke_zero/probe_acceptance 单独验收。

  ... record_reference.py --out /tmp/unscrew32_reference_current.mp4 \
      --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["POUR_NO_D6"] = "1"

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--fps", type=int, default=20)
p.add_argument("--stride", type=int, default=1,
               help="每隔多少母带行取一帧（默认逐行）")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.fps <= 0 or args.stride <= 0:
    p.error("--fps/--stride 必须为正数")

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_ref_record")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]
E.reset()

_REC_NPZ = os.environ.get("UNSCREW_RECORD_NPZ", TC.REF_V2)  # 可指定录 v1 (v2 未铸时抢先看)
with np.load(_REC_NPZ, allow_pickle=True) as _z:
    Z = {key: np.asarray(_z[key]) for key in _z.files}
T = len(Z["right_q"])
assert T == E.T_ROW and int(np.asarray(Z["seg_lens"]).sum()) == T
assert all(np.isfinite(Z[key]).all() for key in (
    "right_q", "left_q", "right_f", "left_f",
    "obj_pos_0", "obj_quat_0", "obj_pos_1", "obj_quat_1"))

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
cam = UsdGeom.Camera.Define(stage, "/World/ReferenceCam")
cam.CreateFocalLengthAttr().Set(16.0)
origin = E.scene.env_origins[0].detach().cpu().numpy()
look = Gf.Matrix4d()
look.SetLookAt(
    Gf.Vec3d(float(origin[0]) + 0.85, float(origin[1]) - 1.15, 1.60),
    Gf.Vec3d(float(origin[0]) - 0.15, float(origin[1]) + 0.10, 0.95),
    Gf.Vec3d(0, 0, 1),
)
UsdGeom.Xformable(cam).AddTransformOp().Set(look.GetInverse())
render_product = rep.create.render_product("/World/ReferenceCam", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach(render_product)

dev = E.device
zero_vel = torch.zeros(1, 6, device=dev)


def write_row(row: int) -> None:
    qfull = E.hand.data.default_joint_pos[:1].clone()
    qfull[:, E.map_ids_t] = E.ref58[row].unsqueeze(0)
    E.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
    E.hand.set_joint_position_target(qfull)
    for oi, art in ((0, E.object), (1, E.aux)):
        pose = torch.tensor(np.concatenate([
            Z[f"obj_pos_{oi}"][row] + origin,
            Z[f"obj_quat_{oi}"][row],
        ]), dtype=torch.float32, device=dev).unsqueeze(0)
        art.write_root_pose_to_sim(pose)
        art.write_root_velocity_to_sim(zero_vel)
    E.scene.write_data_to_sim()


frames: list[np.ndarray] = []


def grab() -> None:
    E.sim.render()
    data = annot.get_data()
    if data is not None and getattr(data, "size", 0):
        frames.append(np.asarray(data)[..., :3].astype(np.uint8).copy())


# 预热渲染器；预热帧不写进视频。
write_row(0)
for _ in range(3):
    E.sim.render()

rows = list(range(0, T, args.stride))
if rows[-1] != T - 1:
    rows.append(T - 1)
for row in rows:
    write_row(row)
    grab()

if not frames:
    raise RuntimeError("相机没有返回任何帧；启动时需带 --enable_cameras")
out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.mimsave(out, frames, fps=args.fps)
print(f"[reference-video] {out} | {len(frames)} 帧 @{args.fps}fps | "
      f"母带 {T} 行 seg={Z['seg_lens'].tolist()}", flush=True)

try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
os._exit(0)
