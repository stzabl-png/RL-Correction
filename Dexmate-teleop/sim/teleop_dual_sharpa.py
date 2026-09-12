#!/usr/bin/env python
"""Dual-hand Isaac Lab consumer: two independent SharpaWave hands follow two ZMQ streams.

Same bridged Mode A as sim/test_env_sharpa.py, but spawns TWO floating hands (left
USD + right USD, side by side) and subscribes to BOTH producers, driving each hand
from its own qpos stream. Run two producers (one per glove/port):

  # terminal 1 (right glove -> :5556)
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right --pub tcp://*:5556
  # terminal 2 (left glove  -> :5557)
  PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand left  --pub tcp://*:5557
  # terminal 3 (this consumer)
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_dual_sharpa.py

Self-test without gloves (both hands wave): add --motion sine
Headless screenshots: --snap-dir DIR [--snap-times "1,3,5"] [--cam-eye x,y,z --cam-target x,y,z]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dual SharpaWave bridged-teleop consumer")
parser.add_argument("--motion", choices=["sine", "zmq"], default="zmq")
parser.add_argument("--sub-right", default="tcp://127.0.0.1:5556", help="right-hand qpos stream")
parser.add_argument("--sub-left", default="tcp://127.0.0.1:5557", help="left-hand qpos stream")
parser.add_argument("--control_hz", type=float, default=60.0)
parser.add_argument("--duration", type=float, default=0.0, help="exit after N sim-seconds (0 = forever)")
parser.add_argument("--snap-dir", default=None, help="save RGB snapshots into this dir (works headless)")
parser.add_argument("--snap-times", default="1,3,5", help="comma-separated sim times [s] to snapshot")
parser.add_argument("--cam-eye", default="0.0,-0.9,0.75")
parser.add_argument("--cam-target", default="0.0,0.0,0.42")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.snap_dir is not None:
    args_cli.enable_cameras = True  # offscreen render products in headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sharpa_scene

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from sharpa_scene import DualHandSceneCfg, sharpa_sdk_joint_names, sine_targets  # noqa: E402


def _parse_vec3(s):
    return [float(v) for v in s.split(",")]


class DualReceiver:
    """One SUB socket connected to both producers; keeps the newest qpos per hand."""

    def __init__(self, addresses):
        import zmq  # pip install pyzmq in the Isaac env

        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        for a in addresses:
            self.sock.connect(a)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.crc, self.qpos, self.valid = {}, {}, {}

    def poll(self):
        while True:
            try:
                m = json.loads(self.sock.recv_string(flags=self._zmq.NOBLOCK))
            except self._zmq.Again:
                return
            h = m.get("hand")
            if m["type"] == "hello":
                self.crc[h] = m["crc"]
            elif m["type"] == "qpos":
                if self.crc.get(h) is not None and m["crc"] != self.crc[h]:
                    continue  # stream restarted with a different joint order; wait for hello
                self.qpos[h] = m["qpos"]
                self.valid[h] = bool(m["valid"])


def main():
    sim_cfg = SimulationCfg(
        dt=1 / 240, render_interval=2, device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=0, bounce_threshold_velocity=0.2),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -0.9, 0.8), target=(0.0, 0.0, 0.42))

    scene_cfg = DualHandSceneCfg(num_envs=1, env_spacing=1.0)
    snap_times = []
    if args_cli.snap_dir is not None:
        scene_cfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/snap_cam", update_period=0.0, width=1280, height=800,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.05, 30.0)),
        )
        snap_times = sorted(float(v) for v in args_cli.snap_times.split(",") if v.strip())
        os.makedirs(args_cli.snap_dir, exist_ok=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robots = {"left": scene["robot_left"], "right": scene["robot_right"]}
    info = {}
    for h, rb in robots.items():
        inames = list(rb.joint_names)
        snames = sharpa_sdk_joint_names(h)
        miss = [n for n in inames if n not in snames]
        if miss:
            raise SystemExit(f"{h} USD joints not in SDK list: {miss}")
        info[h] = dict(
            names=inames,
            sdk2isaac=torch.tensor([snames.index(n) for n in inames], dtype=torch.long),
            limits=rb.data.joint_pos_limits[0].to("cpu"),
            targets=rb.data.default_joint_pos.clone(),
        )
        print(f"[dual] {h}: {len(inames)} joints")

    snap_cam = None
    if args_cli.snap_dir is not None:
        snap_cam = scene["camera"]
        eye = torch.tensor([_parse_vec3(args_cli.cam_eye)], dtype=torch.float32)
        target = torch.tensor([_parse_vec3(args_cli.cam_target)], dtype=torch.float32)
        snap_cam.set_world_poses_from_view(eye.to(snap_cam.device), target.to(snap_cam.device))
        print(f"[snap] camera eye={args_cli.cam_eye} target={args_cli.cam_target} -> {args_cli.snap_dir}")

    def save_snap(sim_t):
        from PIL import Image

        rgb = snap_cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args_cli.snap_dir, f"dual_t{sim_t:04.1f}s.png")
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"\n[snap] saved {path}")

    sim_dt = sim.get_physics_dt()
    decimation = max(1, round(1.0 / sim_dt / args_cli.control_hz))

    recv = None
    if args_cli.motion == "zmq":
        recv = DualReceiver([args_cli.sub_right, args_cli.sub_left])
        print(f"[dual] subscribing right={args_cli.sub_right}  left={args_cli.sub_left}")
        print("[dual] start the two teleop_retarget.py producers (one per glove/port)")

    t = 0.0
    step = 0
    while simulation_app.is_running():
        if step % decimation == 0:
            if recv is not None:
                recv.poll()
            for h, rb in robots.items():
                d = info[h]
                if args_cli.motion == "sine":
                    d["targets"][0] = sine_targets(t, d["names"], d["limits"]).to(d["targets"].device)
                elif recv.qpos.get(h) is not None and recv.valid.get(h):
                    q = torch.tensor(recv.qpos[h], dtype=torch.float32)
                    d["targets"][0] = q[d["sdk2isaac"]].to(d["targets"].device)
                rb.set_joint_position_target(d["targets"])
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        t += sim_dt
        step += 1
        if snap_cam is not None and snap_times and t >= snap_times[0]:
            save_snap(snap_times.pop(0))
        if args_cli.duration > 0 and t >= args_cli.duration:
            print(f"[dual] duration {args_cli.duration}s reached, exiting", flush=True)
            break

    import threading

    killer = threading.Timer(20.0, lambda: os._exit(0))
    killer.daemon = True
    killer.start()
    simulation_app.close()


if __name__ == "__main__":
    main()
