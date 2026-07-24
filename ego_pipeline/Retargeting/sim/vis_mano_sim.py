#!/usr/bin/env python
"""Show the MANO (human) hand mesh in Isaac Sim - NO SharpaWave retargeting.

Loads mano_verts_right + mano_faces + obj_pose from a replay npz (e.g. from
scene_recon_to_replay.py), applies the same gravity/recenter/table-rest as
retarget_isaacsim, and animates the deformable MANO mesh frame by frame.

  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/vis_mano_sim.py \
      --traj replay_world.npz --object-usd object.usd --scene-rot up:gx,gy,gz --table
"""
import argparse
import os

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--traj", required=True)
parser.add_argument("--object-usd", default=None)
parser.add_argument("--no-object", action="store_true")
parser.add_argument("--scene-rot", default="obj0",
                    help="cv2zup | hoi4d | obj0 | up:gx,gy,gz | quat w,x,y,z")
parser.add_argument("--recenter", choices=["none", "table"], default="table")
parser.add_argument("--table", action="store_true")
parser.add_argument("--table-size", type=float, default=4.0)
parser.add_argument("--table-height", type=float, default=0.85)
parser.add_argument("--table-thickness", type=float, default=0.04)
parser.add_argument("--obj-lift", type=float, default=0.01)
parser.add_argument("--smooth", type=int, default=1)
parser.add_argument("--hand-color", default="0.3,0.3,0.3")
parser.add_argument("--floor-usd", default=os.path.expanduser(
    "~/Project/Reconstruct_and_Retarget/third_party/mano2gripper/Assets/Scene/"
    "Collected_default_environment/default_environment.usd"))
parser.add_argument("--cam-view", choices=["topdown", "iso", "manual"], default="iso")
parser.add_argument("--cam-dist", type=float, default=0.8)
parser.add_argument("--cam-eye", default="1.8,-1.8,1.8")
parser.add_argument("--cam-target", default="0,0,1")
parser.add_argument("--fps", type=float, default=15.0)
parser.add_argument("--auto", action="store_true")
parser.add_argument("--loop", action="store_true")
parser.add_argument("--snap-dir", default=None)
parser.add_argument("--snap-times", default="")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

import numpy as np  # noqa: E402


def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_to_rotmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def estimate_up_rotation(obj_xyz):
    a = np.asarray(obj_xyz, np.float64); n = len(a); tt = np.linspace(0, 1, n)[:, None]
    resid = a - (a[0] * (1 - tt) + a[-1] * tt)
    _, _, Vt = np.linalg.svd(resid - resid.mean(0), full_matrices=False)
    up = Vt[0]; proj = resid @ up
    if proj[np.argmax(np.abs(proj))] < 0:
        up = -up
    u = up / np.linalg.norm(up); fwd = np.array([0.0, 0, 1])
    wy = fwd - (fwd @ u) * u; wy /= np.linalg.norm(wy)
    return np.stack([np.cross(wy, u), wy, u])


def moving_average(a, w):
    if w <= 1:
        return a
    a = np.asarray(a, np.float64); pad = w // 2
    ap = np.pad(a, [(pad, pad)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    ker = np.ones(w) / w; flat = ap.reshape(ap.shape[0], -1)
    out = np.empty((a.shape[0], flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], ker, mode="valid")[: a.shape[0]]
    return out.reshape(a.shape)


d = np.load(args.traj)
verts = d["mano_verts_right"].astype(np.float64)            # (T,778,3)
faces = np.asarray(d["mano_faces"], np.int32)               # (F,3)
T = len(verts)
obj_pose = (np.asarray(d["obj_pose"], np.float64) if "obj_pose" in d.files
            else np.tile([0, 0, 0, 1, 0, 0, 0], (T, 1)).astype(np.float64))
fps = float(d["fps"]) if "fps" in d.files else args.fps

if args.smooth > 1:
    verts = moving_average(verts, args.smooth)
    obj_pose[:, :3] = moving_average(obj_pose[:, :3], args.smooth)

# ---- gravity / recenter (same conventions as retarget_isaacsim) -------------
if args.scene_rot == "cv2zup":
    Rs = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
elif args.scene_rot == "hoi4d":
    Rs = estimate_up_rotation(obj_pose[:, :3])
elif args.scene_rot == "obj0":
    Rs = quat_to_rotmat(obj_pose[0, 3:7]).T
elif args.scene_rot.startswith("up:"):
    u = np.array([float(x) for x in args.scene_rot[3:].split(",")], float); u /= np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0.0, 1, 0])
    wx = a - (a @ u) * u; wx /= np.linalg.norm(wx); Rs = np.stack([wx, np.cross(u, wx), u])
else:
    Rs = quat_to_rotmat(np.array([float(x) for x in args.scene_rot.split(",")]))
qs = rotmat_to_quat(Rs)
obj_pos_w = np.einsum("ij,tj->ti", Rs, obj_pose[:, :3])
obj_quat_w = np.stack([quat_mul(qs, obj_pose[i, 3:7]) for i in range(len(obj_pose))])
scene_shift = np.zeros(3)
if args.recenter == "table":
    # x,y: center the object's frame-0 on the table; z: rest the LOWEST point of the whole
    # object trajectory on the table, so the object sits on the table at its resting moment
    # and lifts ABOVE it (anchoring frame 0 sinks the rest below the table if frame 0 is high)
    scene_shift = np.array([-obj_pos_w[0, 0], -obj_pos_w[0, 1],
                            args.table_height + args.obj_lift - obj_pos_w[:, 2].min()])
verts_w = np.einsum("ij,tvj->tvi", Rs, verts) + scene_shift
obj_pos_w = obj_pos_w + scene_shift
obj_pose = np.concatenate([obj_pos_w, obj_quat_w], axis=1)
print(f"[mano] T={T} frames @ {fps:g}fps  scene_rot={args.scene_rot}")

# ---- boot Kit --------------------------------------------------------------
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, Vt  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane  # noqa: E402
from isaacsim.core.prims import SingleXFormPrim  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / max(fps, 1.0))
stage = omni.usd.get_context().get_stage()
if args.floor_usd and os.path.exists(args.floor_usd):
    add_reference_to_stage(os.path.abspath(args.floor_usd), "/World/ground")
else:
    GroundPlane("/World/ground", z_position=0.0)
UsdLux.DomeLight.Define(stage, "/World/light").CreateIntensityAttr(2500.0)
if args.table:
    th = args.table_thickness; white = np.array([1.0, 1.0, 1.0])
    FixedCuboid("/World/table_top", position=np.array([0.0, 0.0, args.table_height - th / 2.0]),
                scale=np.array([args.table_size, args.table_size, th]), color=white)

obj_present = (args.object_usd is not None) and not args.no_object
obj = None
if obj_present:
    add_reference_to_stage(os.path.abspath(args.object_usd), "/World/Object")
    if args.recenter == "table":
        try:
            rng = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Object")).ComputeLocalBound(
                Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
            rest = -float(rng.GetMin()[2])
            obj_pose[:, 2] += rest; verts_w[:, :, 2] += rest
            print(f"[mano] rest object bottom on table: +{rest * 100:.1f} cm")
        except Exception as e:
            print(f"[warn] obj bbox rest failed: {e!r}")
    obj = SingleXFormPrim("/World/Object", name="obj")

# ---- MANO mesh -------------------------------------------------------------
mano = UsdGeom.Mesh.Define(stage, "/World/ManoHand")
mano.CreateFaceVertexCountsAttr([3] * len(faces))
mano.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
mano.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts_w[0].astype(np.float32)))
mano.CreateDisplayColorAttr([Gf.Vec3f(*[float(x) for x in args.hand_color.split(",")])])
mano_pts = mano.GetPointsAttr()

world.reset()
world.get_physics_context().set_gravity(0.0)   # kinematic playback
if obj is not None:
    obj.set_world_pose(obj_pose[0, :3], obj_pose[0, 3:7])

# ---- camera ----------------------------------------------------------------
c = verts_w.reshape(-1, 3).mean(0)
cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
if args.cam_view == "topdown":
    eye = np.array([cx, cy - 0.13 * args.cam_dist, cz + args.cam_dist]); tgt = np.array([cx, cy, cz])
elif args.cam_view == "iso":
    dd = args.cam_dist; eye = np.array([cx + dd, cy - dd, cz + 0.7 * dd]); tgt = np.array([cx, cy, cz])
else:
    eye = np.array([float(x) for x in args.cam_eye.split(",")])
    tgt = np.array([float(x) for x in args.cam_target.split(",")])
try:
    set_camera_view(eye=eye, target=tgt)
except Exception as e:
    print(f"[warn] viewport camera: {e!r}")

cam = None
snap_times = [float(x) for x in args.snap_times.split(",") if x.strip()] if args.snap_times else []
if args.snap_dir:
    os.makedirs(args.snap_dir, exist_ok=True)
    try:
        from isaacsim.sensors.camera import Camera
        cam = Camera(prim_path="/World/snap_cam", resolution=(1280, 800), position=eye)
        cam.initialize(); cam.set_focal_length(2.4)
        set_camera_view(eye=eye, target=tgt, camera_prim_path="/World/snap_cam")
    except Exception as e:
        print(f"[warn] snap camera: {e!r}"); cam = None


def maybe_snap(t):
    if cam is None:
        return
    while snap_times and t >= snap_times[0]:
        st = snap_times.pop(0)
        try:
            import imageio
            imageio.imwrite(os.path.join(args.snap_dir, f"snap_{st:06.2f}.png"), cam.get_rgba()[:, :, :3])
            print(f"[snap] t={st:.2f}s")
        except Exception as e:
            print(f"[warn] snap {st}: {e!r}")


def place(frame):
    mano_pts.Set(Vt.Vec3fArray.FromNumpy(verts_w[frame].astype(np.float32)))
    if obj is not None:
        obj.set_world_pose(obj_pose[frame, :3], obj_pose[frame, 3:7])


import time as _time  # noqa: E402

_render = not args.headless
frame_dt = 1.0 / max(fps, 1.0)
print(f"[run] MANO mesh ({len(verts_w[0])} verts, {len(faces)} faces) + "
      f"object={os.path.basename(args.object_usd) if obj_present else '(none)'} @ {fps:g}fps")


def play_one_pass():
    # kinematic display: 1 render per data frame, paced to real-time fps in the GUI
    start = _time.perf_counter()
    for frame in range(T):
        place(frame)
        world.step(render=_render or cam is not None)
        maybe_snap((frame + 1) * frame_dt)
        if _render:
            target = start + (frame + 1) * frame_dt
            now = _time.perf_counter()
            if now < target:
                _time.sleep(target - now)
        if not app.is_running():
            break
    print(f"[done] played {T} frames")


try:
    if args.headless or args.auto:
        while app.is_running():
            play_one_pass()
            if not args.loop:
                break
            place(0)
    else:
        import queue
        import threading
        cmd_q = queue.Queue()

        def _reader():
            for line in __import__("sys").stdin:
                cmd_q.put(line.strip().lower())

        threading.Thread(target=_reader, daemon=True).start()
        print("\n[ready] ENTER to play, q+ENTER to quit.")
        while app.is_running():
            world.step(render=_render)
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                continue
            if cmd == "q":
                break
            play_one_pass(); place(0)
            print("\n[ready] ENTER to replay, q+ENTER to quit.")
finally:
    import threading
    import sys
    sys.stdout.flush()
    threading.Timer(20.0, lambda: os._exit(0)).start()
    app.close()
