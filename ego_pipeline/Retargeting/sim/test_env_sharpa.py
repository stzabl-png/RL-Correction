#!/usr/bin/env python
"""Isaac Lab test environment for the SharpaWave hand (bridged-teleop consumer).

Scene/actuator config is shared with the single-process script in
sim/sharpa_scene.py and mirrors sharpa-rl-lab (pre-tuned USD gains, same PhysX).

Run with the Isaac Lab python. Modes:
  --motion sine   joints wave by themselves -> verifies env/asset/actuators alone
  --motion home   hold zero pose
  --motion zmq    follow the retarget stream from scripts/teleop_retarget.py
                  (bridged mode; needs `pip install pyzmq` in the Isaac env)

For the single-process alternative (glove + retarget inside Isaac), see
sim/teleop_isaac_single.py.

Examples:
  python sim/test_env_sharpa.py --motion sine
  python sim/test_env_sharpa.py --motion zmq --sub tcp://127.0.0.1:5556
  python sim/test_env_sharpa.py --motion sine --headless --duration 10   # CI smoke
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SharpaWave teleop test env")
parser.add_argument("--motion", choices=["sine", "home", "zmq"], default="sine")
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--usd", default=None, help="override hand USD path")
parser.add_argument("--sub", default="tcp://127.0.0.1:5556", help="ZMQ qpos stream (motion=zmq)")
parser.add_argument("--control_hz", type=float, default=60.0)
parser.add_argument("--duration", type=float, default=0.0, help="exit after N sim-seconds (0 = run forever)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sharpa_scene

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from sharpa_scene import SharpaSceneCfg, sharpa_sdk_joint_names, sine_targets  # noqa: E402


class ZmqQposReceiver:
    """Non-blocking reader of the scripts/teleop_retarget.py stream."""

    def __init__(self, address: str, hand: str):
        import zmq  # `pip install pyzmq` into the Isaac env for zmq mode

        self._zmq = zmq
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(address)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.hand = hand
        self.joint_names = None
        self.crc = None
        self.qpos = None
        self.valid = False

    def poll(self):
        """Drain everything pending; keep the newest qpos."""
        while True:
            try:
                msg = json.loads(self.sock.recv_string(flags=self._zmq.NOBLOCK))
            except self._zmq.Again:
                return
            if msg.get("hand") != self.hand:
                continue
            if msg["type"] == "hello":
                self.joint_names = msg["joint_names"]
                self.crc = msg["crc"]
            elif msg["type"] == "qpos":
                if self.crc is not None and msg["crc"] != self.crc:
                    continue  # stream restarted with different joint order; wait for hello
                self.qpos = msg["qpos"]
                self.valid = bool(msg["valid"])


def main():
    sim_cfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        device=args_cli.device,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.6, -0.4, 0.85), target=(0.0, 0.0, 0.45))

    scene_cfg = SharpaSceneCfg(num_envs=1, env_spacing=1.0)
    if args_cli.usd is not None:
        scene_cfg.robot.spawn.usd_path = os.path.abspath(args_cli.usd)
    elif args_cli.hand == "left":
        raise SystemExit("left hand: pass --usd pointing to a left_sharpa_wave USD")
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    isaac_names = list(robot.joint_names)
    print(f"[env] {len(isaac_names)} joints (Isaac order): {isaac_names}")

    sdk_names = sharpa_sdk_joint_names(args_cli.hand)
    missing = [n for n in isaac_names if n not in sdk_names]
    if missing:
        raise SystemExit(f"USD joints not in SDK list (check asset/hand side): {missing}")
    # targets_isaac[i] = qpos_sdk[sdk2isaac[i]]  (cf. rl-lab utils/misc.py idx mapping)
    sdk2isaac = torch.tensor([sdk_names.index(n) for n in isaac_names], dtype=torch.long)

    limits = robot.data.joint_pos_limits[0].to("cpu")  # (J, 2)
    receiver = ZmqQposReceiver(args_cli.sub, args_cli.hand) if args_cli.motion == "zmq" else None
    if receiver is not None:
        print(f"[env] subscribing qpos from {args_cli.sub} - start scripts/teleop_retarget.py in the teleop venv")

    sim_dt = sim.get_physics_dt()
    decimation = max(1, round(1.0 / sim_dt / args_cli.control_hz))
    targets = robot.data.default_joint_pos.clone()
    t = 0.0
    step = 0
    while simulation_app.is_running():
        if step % decimation == 0:
            if args_cli.motion == "sine":
                targets[0] = sine_targets(t, isaac_names, limits).to(targets.device)
            elif args_cli.motion == "zmq":
                receiver.poll()
                if receiver.qpos is not None and receiver.valid:
                    q_sdk = torch.tensor(receiver.qpos, dtype=torch.float32)
                    targets[0] = q_sdk[sdk2isaac].to(targets.device)
            # home: keep defaults (zeros)
            robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        t += sim_dt
        step += 1
        if args_cli.duration > 0 and t >= args_cli.duration:
            print(f"[env] duration {args_cli.duration}s reached, exiting", flush=True)
            break

    # simulation_app.close() is known to hang on this stack -> 20 s hard-exit guard
    import threading

    killer = threading.Timer(20.0, lambda: os._exit(0))
    killer.daemon = True
    killer.start()
    simulation_app.close()


if __name__ == "__main__":
    main()
