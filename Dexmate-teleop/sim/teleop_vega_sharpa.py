#!/usr/bin/env python
"""Single-process teleop: Dexmate Vega-1P + SharpaWave, Wuji-glove driven.

What the glove provides (docs/references/01): 21-keypoint skeleton (fingers)
and wrist ORIENTATION from the dorsum IMU - NO wrist translation. So:

  fingers -> dex-retargeting -> 22 right-hand finger joint targets
  wrist   -> rpy/quat only   -> orientation-tracking differential IK on the
             7-DOF right arm: position target stays pinned at the R_ee spawn
             position, orientation target = spawn orientation composed with
             the glove's RELATIVE rotation (first valid frame = reference).
             When Pico/VR arrives (P4), translation plugs into the same IK
             by swapping the fixed position target for the tracked one.

Assets/gains come from MagicSim (sim/vega_sharpa_scene.py).

Run (in .venv-isaac, see scripts/setup_isaac_env.sh):
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_sharpa.py --source mock
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_sharpa.py --source wuji --hand right
  ... --headless --duration 3        # smoke (slow on powersave CPU governors)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vega-1P + SharpaWave glove teleop")
parser.add_argument("--source", choices=["mock", "wuji"], default="mock")
parser.add_argument("--motion", default="cycle",
                    choices=["open", "fist", "wave", "pinch", "cycle", "rpytest"])
parser.add_argument("--hand", choices=["right"], default="right",
                    help="left arm support comes with the dual-glove phase")
parser.add_argument("--mode", choices=["vector", "dexpilot"], default="vector")
parser.add_argument("--control_hz", type=float, default=60.0)
parser.add_argument("--wrist", choices=["ik", "hold", "off"], default="ik",
                    help="ik = orientation diff-IK from glove rpy; hold = IK pinned to the "
                         "captured spawn pose (diagnostic); off = arm holds spawn pose")
parser.add_argument("--wrist-align", default="0,0,0",
                    help="fixed rotation glove-wrist axes -> EE axes, euler xyz degrees. "
                         "Identity for the mock; calibrate on real-glove day (see references/01 §8.5)")
parser.add_argument("--wrist-max-deg", type=float, default=75.0,
                    help="clamp on the relative wrist rotation magnitude (safety)")
parser.add_argument("--jacobi-row", choices=["fixedbase", "body"], default="body",
                    help="jacobian row index: body = body_idx (correct for this USD, whose "
                         "ArticulationRootAPI sits on the /root_joint fixed joint - verified by "
                         "hold-test: 2mm vs 200-600mm with body_idx-1); fixedbase = body_idx-1")
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--stale_ms", type=float, default=150.0)
parser.add_argument("--conf_min", type=float, default=0.3)
parser.add_argument("--sn", default=None)
parser.add_argument("--address", default=None)
# self-inspection screenshots (works headless; forces --enable_cameras)
parser.add_argument("--snap-dir", default=None, help="save RGB snapshots into this directory")
parser.add_argument("--snap-times", default="0.5,3,5,8,11,15",
                    help="comma-separated sim times [s] to snapshot")
parser.add_argument("--cam-eye", default="2.2,-1.8,1.9")
parser.add_argument("--cam-target", default="0.1,-0.3,1.0")
parser.add_argument("--debug-joints", action="store_true",
                    help="print base/torso/arm joint states every 2 sim-seconds")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.snap_dir is not None:
    args_cli.enable_cameras = True  # offscreen render products in headless

# -- retarget + glove source BEFORE Kit boots (pinocchio/USD boost::python clash,
#    see sim/teleop_isaac_single.py) -------------------------------------------
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402
from magicdexmate.sources import make_source  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource  # noqa: E402

print(f"[setup] building retargeting (pre-Kit): sharpa_wave {args_cli.hand} {args_cli.mode}")
retargeting = build_sharpa_retargeting(args_cli.hand, args_cli.mode)
mapper = JointMapper(retargeting, args_cli.hand)
src_kwargs = {"motion": args_cli.motion} if args_cli.source == "mock" else \
    {"sn": args_cli.sn, "address": args_cli.address}
source = make_source(args_cli.source, args_cli.hand, **src_kwargs)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from sharpa_scene import sharpa_sdk_joint_names  # noqa: E402
from vega_sharpa_scene import EE_BODY, R_ARM_JOINTS, VegaSharpaSceneCfg, make_snap_camera_cfg  # noqa: E402


def _parse_vec3(s: str):
    x, y, z = (float(v) for v in s.split(","))
    return (x, y, z)


def main():
    sim_cfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4, bounce_threshold_velocity=0.2),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.2, -1.6, 1.8), target=(0.0, 0.0, 1.0))

    scene_cfg = VegaSharpaSceneCfg(num_envs=1, env_spacing=4.0)
    snap_times = []
    if args_cli.snap_dir is not None:
        scene_cfg.camera = make_snap_camera_cfg()
        snap_times = sorted(float(v) for v in args_cli.snap_times.split(",") if v.strip())
        os.makedirs(args_cli.snap_dir, exist_ok=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    snap_cam = None
    if args_cli.snap_dir is not None:
        snap_cam = scene["camera"]
        eye = torch.tensor([_parse_vec3(args_cli.cam_eye)], dtype=torch.float32)
        target = torch.tensor([_parse_vec3(args_cli.cam_target)], dtype=torch.float32)
        snap_cam.set_world_poses_from_view(eye.to(snap_cam.device), target.to(snap_cam.device))
        print(f"[snap] camera eye={args_cli.cam_eye} target={args_cli.cam_target} -> {args_cli.snap_dir}")

    def save_snap(sim_t: float):
        from PIL import Image

        rgb = snap_cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args_cli.snap_dir, f"snap_t{sim_t:05.1f}s.png")
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"\n[snap] saved {path}")

    robot: Articulation = scene["robot"]
    device = robot.device
    isaac_names = list(robot.joint_names)
    print(f"[env] articulation with {len(isaac_names)} joints")

    # -- finger mapping: SDK 22-order -> isaac indices of the right hand -------
    sdk_names = sharpa_sdk_joint_names(args_cli.hand)
    missing = [n for n in sdk_names if n not in isaac_names]
    if missing:
        raise SystemExit(f"hand joints missing in articulation: {missing}")
    finger_isaac_ids = torch.tensor([isaac_names.index(n) for n in sdk_names],
                                    dtype=torch.long, device=device)

    # -- wrist orientation IK on the right arm ---------------------------------
    arm_cfg = SceneEntityCfg("robot", joint_names=R_ARM_JOINTS, body_names=[EE_BODY[args_cli.hand]])
    arm_cfg.resolve(scene)
    ee_body_idx = arm_cfg.body_ids[0]
    arm_joint_ids = arm_cfg.joint_ids
    # fixed-base articulation: jacobian row of body i sits at i-1
    ee_jacobi_idx = ee_body_idx - 1 if args_cli.jacobi_row == "fixedbase" else ee_body_idx

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=device,
    )

    # fixed alignment: glove-wrist axes -> EE axes (applied by conjugation so the
    # ROTATION AXIS gets remapped; identity for the mock, calibrated for the glove)
    align_rpy = torch.tensor([[float(v) for v in args_cli.wrist_align.split(",")]],
                             device=device) * torch.pi / 180.0
    q_align = quat_from_euler_xyz(align_rpy[:, 0], align_rpy[:, 1], align_rpy[:, 2])
    q_align_inv = quat_inv(q_align)
    wrist_max_rad = args_cli.wrist_max_deg * torch.pi / 180.0
    cmd_rel_ee = None  # latest commanded relative EE rotation (for telemetry)

    def ee_pose_in_base():
        ee_pose_w = robot.data.body_state_w[:, ee_body_idx, 0:7]
        root_pose_w = robot.data.root_state_w[:, 0:7]
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return pos_b, quat_b

    targets = robot.data.default_joint_pos.clone()
    glove_quat0_inv = None          # reference glove orientation (first valid frame)

    sim_dt = sim.get_physics_dt()
    decimation = max(1, round(1.0 / sim_dt / args_cli.control_hz))
    stale_us = args_cli.stale_ms * 1000.0

    # Warm up physics so body/root state buffers and jacobians are valid AND the
    # arm has fully settled BEFORE capturing the EE reference - capturing at
    # tick 0 reads stale buffers, and capturing mid-settle leaves the IK a
    # ~0.15 m phantom error to fight (both measured).
    for _ in range(120):
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    ee_pos0, ee_quat0 = ee_pose_in_base()
    ee_pos0, ee_quat0 = ee_pos0.clone(), ee_quat0.clone()
    print(f"[ik] EE reference captured: pos={[round(v, 3) for v in ee_pos0[0].tolist()]} "
          f"quat={[round(v, 3) for v in ee_quat0[0].tolist()]}")
    if args_cli.debug_joints:
        lims = robot.data.joint_pos_limits[0]
        qw = robot.data.joint_pos[0]
        for j, name in zip(arm_joint_ids, [isaac_names[i] for i in arm_joint_ids]):
            print(f"[ik]   {name}: settled {qw[j]:+.2f} limits [{lims[j, 0]:+.2f}, {lims[j, 1]:+.2f}]")

    source.start()
    print(f"[run] source={args_cli.source} wrist={args_cli.wrist} "
          f"control={1.0 / (sim_dt * decimation):.0f}Hz  Ctrl-C to stop")

    rt_ms, n_retargets, last_t_us, last_print = [], 0, -1, time.time()
    t, step = 0.0, 0
    try:
        while simulation_app.is_running():
            if step % decimation == 0:
                # Mock runs phase-locked to SIM time: the sim is much slower than
                # wall clock (powersave CPU), so a wall-clock mock would sweep
                # through motion phases ~30x too fast and shake the wrist IK.
                # The real glove (wuji) stays on wall clock, as it must.
                if isinstance(source, MockGloveSource):
                    frame = source.sample_at(t)
                else:
                    frame = source.get_latest()
                now_us = time.time_ns() // 1000
                valid = (frame is not None and (now_us - frame.t_us) < stale_us
                         and frame.min_fingertip_conf() >= args_cli.conf_min)

                if valid and frame.t_us != last_t_us:
                    last_t_us = frame.t_us
                    # fingers
                    ref_value = compute_ref_value(retargeting, to_mano(frame.kp, args_cli.hand))
                    t0 = time.perf_counter()
                    q_sdk = mapper.to_sdk(retargeting.retarget(ref_value))
                    rt_ms.append((time.perf_counter() - t0) * 1e3)
                    targets[0, finger_isaac_ids] = torch.as_tensor(
                        q_sdk, dtype=torch.float32, device=device)
                    n_retargets += 1
                    # wrist orientation (glove gives rpy only - no translation)
                    if args_cli.wrist == "ik" and frame.wrist_quat is not None:
                        gq = torch.as_tensor(frame.wrist_quat, dtype=torch.float32,
                                             device=device).unsqueeze(0)
                        if glove_quat0_inv is None:
                            glove_quat0_inv = quat_inv(gq)
                        rel = quat_mul(glove_quat0_inv, gq)        # glove rotation since reference
                        rel = quat_mul(q_align, quat_mul(rel, q_align_inv))  # remap axes glove->EE
                        aa = axis_angle_from_quat(rel)             # safety clamp on magnitude
                        ang = aa.norm(dim=-1, keepdim=True)
                        if ang.item() > wrist_max_rad:
                            aa = aa / ang * wrist_max_rad
                            ang = torch.full_like(ang, wrist_max_rad)
                        rel = quat_from_angle_axis(ang.squeeze(-1), aa / ang.clamp(min=1e-9))
                        cmd_rel_ee = rel
                        quat_des = quat_mul(ee_quat0, rel)         # applied about the spawn EE pose
                        ik.set_command(torch.cat([ee_pos0, quat_des], dim=-1))

                if args_cli.wrist == "hold" and glove_quat0_inv is None:
                    glove_quat0_inv = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                    ik.set_command(torch.cat([ee_pos0, ee_quat0], dim=-1))  # pin to spawn pose

                if args_cli.wrist in ("ik", "hold") and glove_quat0_inv is not None:
                    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids]
                    ee_pos_b, ee_quat_b = ee_pose_in_base()
                    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
                    targets[0, torch.as_tensor(arm_joint_ids, device=device)] = ik.compute(
                        ee_pos_b, ee_quat_b, jacobian, joint_pos)[0]

                robot.set_joint_position_target(targets)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            t += sim_dt
            step += 1

            if snap_cam is not None and snap_times and t >= snap_times[0]:
                save_snap(snap_times.pop(0))

            if args_cli.debug_joints and step % round(2.0 / sim_dt) == 0:
                q = robot.data.joint_pos[0]
                names = isaac_names
                grp = {
                    "base": [n for n in names if n.startswith("dummy_base")],
                    "torso": [n for n in names if n.startswith("torso")],
                    "L_arm": [n for n in names if n.startswith("L_arm")],
                    "R_arm": [n for n in names if n.startswith("R_arm")],
                }
                msg = " | ".join(
                    f"{g}: " + " ".join(f"{q[names.index(n)].item():+.2f}" for n in joints)
                    for g, joints in grp.items()
                )
                ee_pos_b, ee_quat_b = ee_pose_in_base()
                err = "n/a" if ee_pos0 is None else f"{(ee_pos_b - ee_pos0).norm().item() * 1000:.0f}mm"
                flex_ids = [names.index(n) for n in names
                            if n.startswith("right_") and n.endswith(("_FE", "_PIP", "_DIP", "_IP"))]
                fcur = q[flex_ids].mean().item()
                ftgt = targets[0, flex_ids].mean().item()
                rot = ""
                if ee_quat0 is not None:
                    aa_cur = axis_angle_from_quat(quat_mul(quat_inv(ee_quat0), ee_quat_b))[0]
                    a_cur = aa_cur.norm().item()
                    ax_cur = (aa_cur / max(a_cur, 1e-9)).tolist()
                    rot = (f" | EE rot {a_cur * 57.3:5.1f}deg axis "
                           f"[{ax_cur[0]:+.2f} {ax_cur[1]:+.2f} {ax_cur[2]:+.2f}]")
                    if cmd_rel_ee is not None:
                        aa_c = axis_angle_from_quat(cmd_rel_ee)[0]
                        a_c = aa_c.norm().item()
                        ax_c = (aa_c / max(a_c, 1e-9)).tolist()
                        rot += (f" cmd {a_c * 57.3:5.1f}deg "
                                f"[{ax_c[0]:+.2f} {ax_c[1]:+.2f} {ax_c[2]:+.2f}]")
                print(f"\n[dbg] t={t:5.2f} {msg} | ee_pos err {err} | "
                      f"R-finger flex cur/tgt {fcur:+.2f}/{ftgt:+.2f}{rot}")

            if time.time() - last_print > 2.0 and rt_ms:
                a = np.array(rt_ms[-200:])
                print(f"\r[stats] sim t={t:6.1f}s retargets={n_retargets} "
                      f"retarget p50/p95 {np.percentile(a, 50):.2f}/{np.percentile(a, 95):.2f} ms",
                      end="", flush=True)
                last_print = time.time()
            if args_cli.duration > 0 and t >= args_cli.duration:
                print(f"\n[done] duration {args_cli.duration}s reached")
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        if rt_ms:
            a = np.array(rt_ms)
            print(f"\n[done] {n_retargets} retargets, retarget p50 {np.percentile(a, 50):.2f} ms", flush=True)
        # simulation_app.close() is known to hang on this stack; all results are
        # already flushed, so hard-exit if shutdown takes longer than 20 s.
        import threading

        killer = threading.Timer(20.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        simulation_app.close()


if __name__ == "__main__":
    main()
