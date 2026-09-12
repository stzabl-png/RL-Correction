#!/usr/bin/env python
"""Single-process Isaac Lab teleop: glove -> retarget -> Sharpa hand, one process.

Unlike the bridged mode (scripts/teleop_retarget.py + sim/test_env_sharpa.py --motion zmq),
everything here runs inside the Isaac Lab python: the glove source (wuji or mock),
dex-retargeting, and the simulation. No ZMQ.

Prerequisite: install the retarget stack into the Isaac python first -
    bash scripts/install_into_isaaclab.sh ~/isaacsim/python.sh
(dex-retargeting's numpy>=2 pin is declarative only; it runs fine on Isaac's
numpy 1.26 - verified, see docs/references/03_dex_retargeting.md §6.)

Examples:
  python sim/teleop_isaac_single.py --source mock --motion cycle
  python sim/teleop_isaac_single.py --source wuji --hand right
  python sim/teleop_isaac_single.py --source mock --headless --duration 10   # smoke
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-process SharpaWave teleop in Isaac Lab")
parser.add_argument("--source", choices=["mock", "wuji"], default="mock")
parser.add_argument("--motion", default="cycle", choices=["open", "fist", "wave", "pinch", "cycle"],
                    help="mock source motion")
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--mode", choices=["vector", "dexpilot"], default="dexpilot")
parser.add_argument("--relax-distal", default=None, action=argparse.BooleanOptionalAction,
                    help="relax DexPilot's redundant thumb_IP/pinky_DIP over-curl on the open "
                         "hand (pinch opposition unchanged); default on for dexpilot")
parser.add_argument("--usd", default=None, help="override hand USD path")
parser.add_argument("--control_hz", type=float, default=60.0)
parser.add_argument("--duration", type=float, default=0.0, help="exit after N sim-seconds (0 = run forever)")
parser.add_argument("--stale_ms", type=float, default=150.0, help="glove staleness watchdog")
parser.add_argument("--conf_min", type=float, default=0.3, help="min fingertip confidence")
parser.add_argument("--sn", default=None, help="wuji glove serial number")
parser.add_argument("--address", default=None, help="wuji glove address, e.g. 192.168.1.101:50000")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# -- retargeting + glove source: MUST initialize BEFORE the Kit app boots ------
# pinocchio (boost::python) and Isaac's USD (also boost::python) clash if Kit
# loads first: "No Python class registered for C++ class std::vector<int>".
# Importing pinocchio and building the retargeting before AppLauncher avoids it.
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                       # sharpa_scene
sys.path.insert(0, os.path.dirname(_THIS_DIR))      # magicdexmate package (repo root)

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402
from magicdexmate.sources import make_source  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource  # noqa: E402

print(f"[setup] building retargeting (pre-Kit): sharpa_wave {args_cli.hand} {args_cli.mode}")
retargeting = build_sharpa_retargeting(args_cli.hand, args_cli.mode)
mapper = JointMapper(retargeting, args_cli.hand)
do_relax = args_cli.relax_distal if args_cli.relax_distal is not None else (args_cli.mode == "dexpilot")
src_kwargs = {"motion": args_cli.motion} if args_cli.source == "mock" else \
    {"sn": args_cli.sn, "address": args_cli.address}
source = make_source(args_cli.source, args_cli.hand, **src_kwargs)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from sharpa_scene import SharpaSceneCfg, sharpa_sdk_joint_names, sharpa_usd  # noqa: E402


def main():

    # -- sim + scene (same parameters as sim/test_env_sharpa.py) ---------------
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
        scene_cfg.robot.spawn.usd_path = sharpa_usd("left")
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    isaac_names = list(robot.joint_names)
    sdk_names = sharpa_sdk_joint_names(args_cli.hand)
    missing = [n for n in isaac_names if n not in sdk_names]
    if missing:
        raise SystemExit(f"USD joints not in SDK list (check asset/hand side): {missing}")
    if mapper.sdk_names != sdk_names:
        raise SystemExit("joint name list mismatch between mapping.py and sharpa_scene.py")
    sdk2isaac = torch.tensor([sdk_names.index(n) for n in isaac_names], dtype=torch.long)

    sim_dt = sim.get_physics_dt()
    decimation = max(1, round(1.0 / sim_dt / args_cli.control_hz))
    stale_us = args_cli.stale_ms * 1000.0
    targets = robot.data.default_joint_pos.clone()

    source.start()
    print(f"[run] source={args_cli.source} control={1.0 / (sim_dt * decimation):.0f}Hz  Ctrl-C to stop")

    rt_ms = []
    last_t_us = -1
    n_retargets = 0
    last_print = time.time()
    t = 0.0
    step = 0
    try:
        while simulation_app.is_running():
            if step % decimation == 0:
                # phase-lock the mock to sim time (sim can run far below realtime);
                # the real glove stays on wall clock
                if isinstance(source, MockGloveSource):
                    frame = source.sample_at(t)
                else:
                    frame = source.get_latest()
                now_us = time.time_ns() // 1000
                valid = (
                    frame is not None
                    and (now_us - frame.t_us) < stale_us
                    and frame.min_fingertip_conf() >= args_cli.conf_min
                )
                if valid and frame.t_us != last_t_us:
                    last_t_us = frame.t_us
                    ref_value = compute_ref_value(retargeting, to_mano(frame.kp, args_cli.hand))
                    t0 = time.perf_counter()
                    qpos = retargeting.retarget(ref_value)
                    rt_ms.append((time.perf_counter() - t0) * 1e3)
                    q_sdk = mapper.to_sdk(qpos)
                    if do_relax:
                        q_sdk = mapper.relax_distal(q_sdk, frame.kp)
                    targets[0] = torch.as_tensor(q_sdk, dtype=torch.float32)[sdk2isaac].to(targets.device)
                    n_retargets += 1
                robot.set_joint_position_target(targets)
                # else: hold last targets (watchdog freeze)

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            t += sim_dt
            step += 1

            if time.time() - last_print > 2.0 and rt_ms:
                a = np.array(rt_ms[-200:])
                print(
                    f"\r[stats] sim t={t:6.1f}s retargets={n_retargets} "
                    f"retarget p50/p95 {np.percentile(a, 50):.2f}/{np.percentile(a, 95):.2f} ms",
                    end="", flush=True,
                )
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
            print(f"\n[done] {n_retargets} retargets, retarget p50 {np.percentile(a, 50):.2f} ms "
                  f"p95 {np.percentile(a, 95):.2f} ms", flush=True)
        # simulation_app.close() is known to hang on this stack; results are
        # flushed already, so hard-exit if shutdown takes longer than 20 s.
        import threading

        killer = threading.Timer(20.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        simulation_app.close()


if __name__ == "__main__":
    main()
