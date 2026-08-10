#!/usr/bin/env python
"""在 Isaac 里只放物体, 用逐帧位姿驱动它 —— 判断物体轨迹本身合不合理。

刻意**不**走 RL 那条链(clips/replay_grasp/DexmateCorrectionEnv): 那条链会做相机锚定、
手轨迹平移、resting_pose 覆盖, 看到的就不再是"这条物体轨迹"本身了。这里只有:
  一个支撑平面(画在 world 的 z_table 高度) + 一个物体 + 逐帧设位姿。
纯运动学摆放, 不 step 物理 —— 要看的是轨迹, 不是它站不站得住。

判读要点:
  - 物体底面应当**贴着**支撑面: 不悬空、不陷进去(HUD 打印 gap)
  - 无人接触的帧应当**完全静止**
  - 全程不应有突跳(HUD 打印逐帧位移/转角)

用法:
  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  $PY -m steps.step2_reconstruction.view_object_traj \\
      --npz .../27_scene_world_fused_corrected.npz \\
      --mesh .../27_scene_mesh_scaled_s0.825.obj --play
  # 与原轨迹并排对照 (原轨迹沿 +Y 挪开 0.5m)
  ... --compare .../27_scene/world_fused.npz --sep 0.5
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--npz", required=True, help="含 object_ob_in_world 的 npz")
p.add_argument("--mesh", required=True, help="与该轨迹配套的网格 obj")
p.add_argument("--compare", default=None, help="另一条轨迹的 npz (原始版), 并排显示")
p.add_argument("--sep", type=float, default=0.5, help="对照物体沿 +Y 的偏移 m")
p.add_argument("--play", action="store_true", help="循环播放")
p.add_argument("--fps", type=float, default=20.0)
p.add_argument("--t", type=int, default=0, help="不 --play 时停在哪一帧")
p.add_argument("--eye", default="0.9,-0.7,0.35")
p.add_argument("--lookat", default="0.52,-0.21,-0.20")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewobjtraj")
app = AppLauncher(args).app

import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402


def load_traj(path):
    z = np.load(path, allow_pickle=True)
    T = np.asarray(z["object_ob_in_world"], float)
    cor = (np.asarray(z["pose_corrected_by_frame"], bool)
           if "pose_corrected_by_frame" in z.files else np.zeros(len(T), bool))
    return T, cor


def mesh_to_usd(mesh_path: str) -> str:
    """网格 -> USD (纯视觉即可, 但沿用仓里 MeshConverter 的做法便于复用缓存)。"""
    import shutil
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    usd_dir = os.path.join(os.path.dirname(mesh_path), "_viewcache")
    stem = os.path.splitext(os.path.basename(mesh_path))[0].replace(".", "_")
    usd_name = stem + ".usd"
    usd = os.path.join(usd_dir, usd_name)
    if os.path.exists(usd) and os.path.getmtime(usd) >= os.path.getmtime(mesh_path):
        return usd
    os.makedirs(usd_dir, exist_ok=True)
    # MeshConverter 用 basename.split(".") 解析扩展名, 文件名里的小数点(如 s0.825)会让它炸
    src = mesh_path
    if os.path.basename(mesh_path).count(".") > 1:
        src = os.path.join(usd_dir, stem + ".obj")
        shutil.copyfile(mesh_path, src)
    MeshConverter(MeshConverterCfg(
        asset_path=src, usd_dir=usd_dir, usd_file_name=usd_name,
        force_usd_conversion=True,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    ))
    print(f"[view] {mesh_path} -> {usd}")
    return usd


T_fix, cor = load_traj(args.npz)
n = len(T_fix)
T_ref = None
if args.compare:
    T_ref, _ = load_traj(args.compare)
    assert len(T_ref) == n, "两条轨迹帧数不一致"

mesh = trimesh.load(args.mesh, force="mesh")
Vm = np.asarray(mesh.vertices, float)
long_axis = int(np.argmax(mesh.extents))

# 支撑面高度: 取"物体最低点"的中位数 —— 与修正脚本同一口径
zmins = np.array([((T[:3, :3] @ Vm.T).T + T[:3, 3])[:, 2].min() for T in T_fix])
z_table = float(np.median(zmins))
ctr = np.median(T_fix[:, :3, 3], axis=0)
print(f"[view] 帧数 {n}  修正帧 {int(cor.sum())}  支撑面 z={z_table:.4f}")

sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cpu"))
sim.set_camera_view(tuple(float(v) for v in args.eye.split(",")),
                    tuple(float(v) for v in args.lookat.split(",")))

# 支撑面 (视觉参考, 不参与物理)
sim_utils.MeshCuboidCfg(
    size=(0.8, 0.8, 0.01),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.34)),
).func("/World/support", sim_utils.MeshCuboidCfg(
    size=(0.8, 0.8, 0.01),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.34))),
    translation=(float(ctr[0]), float(ctr[1]), z_table - 0.005))
sim_utils.DomeLightCfg(intensity=2500.0).func(
    "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

usd = mesh_to_usd(args.mesh)
prims = {}
spec = [("fixed", "/World/obj_fixed", (0.35, 0.95, 0.40), 0.0)]
if T_ref is not None:
    spec.append(("orig", "/World/obj_orig", (0.95, 0.35, 0.35), args.sep))
for tag, path, color, dy in spec:
    cfg = sim_utils.UsdFileCfg(
        usd_path=usd,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
    )
    cfg.func(path, cfg)
    xf = UsdGeom.Xformable(sim.stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    prims[tag] = (xf.AddTransformOp(), dy)
    print(f"[view] {tag} -> {path}  颜色={color}  Y偏移={dy}")

if T_ref is not None:
    sim_utils.MeshCuboidCfg(
        size=(0.8, 0.8, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.28, 0.28)),
    ).func("/World/support_orig", sim_utils.MeshCuboidCfg(
        size=(0.8, 0.8, 0.01),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.28, 0.28))),
        translation=(float(ctr[0]), float(ctr[1]) + args.sep, z_table - 0.005))

sim.reset()


def put(op, T, dy):
    M = np.eye(4)
    M[:3, :3] = T[:3, :3]
    M[:3, 3] = T[:3, 3] + np.array([0.0, dy, 0.0])
    op.Set(Gf.Matrix4d(*M.T.flatten().tolist()))


def report(i):
    T = T_fix[i]
    Vw = (T[:3, :3] @ Vm.T).T + T[:3, 3]
    gap = Vw[:, 2].min() - z_table
    tilt = np.degrees(np.arccos(np.clip(abs(T[:3, :3][:, long_axis][2]), -1, 1)))
    d = np.linalg.norm(T[:3, 3] - T_fix[i - 1][:3, 3]) * 1000 if i > 0 else 0.0
    dR = T[:3, :3] @ T_fix[i - 1][:3, :3].T if i > 0 else np.eye(3)
    rr = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print(f"  帧{i:3d} {'[修正]' if cor[i] else '[原样]'} 倾角{tilt:5.1f}°  "
          f"离支撑面{gap*1000:+6.1f}mm  逐帧位移{d:5.1f}mm 转角{rr:5.2f}°")


print("\n判读: 倾角应≈0(静置)、离支撑面≈0(贴着)、无人碰时逐帧位移应≈0")
print("      绿=修正后" + ("   红=原始(沿+Y挪开)" if T_ref is not None else ""))
if args.play:
    i = 0
    dt = 1.0 / max(args.fps, 1e-3)
    while app.is_running():
        put(prims["fixed"][0], T_fix[i], prims["fixed"][1])
        if T_ref is not None:
            put(prims["orig"][0], T_ref[i], prims["orig"][1])
        if i % 10 == 0:
            report(i)
        sim.render()
        time.sleep(dt)
        i = (i + 1) % n
else:
    t = max(0, min(args.t, n - 1))
    put(prims["fixed"][0], T_fix[t], prims["fixed"][1])
    if T_ref is not None:
        put(prims["orig"][0], T_ref[t], prims["orig"][1])
    report(t)
    while app.is_running():
        sim.render()

app.close()
