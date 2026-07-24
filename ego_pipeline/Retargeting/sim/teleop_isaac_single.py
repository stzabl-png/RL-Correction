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
parser.add_argument("--source", choices=["mock", "wuji", "hawor"], default="mock")
parser.add_argument("--motion", default="cycle", choices=["open", "fist", "wave", "pinch", "cycle"],
                    help="mock source motion")
parser.add_argument("--hawor-npz", default=None, help="hawor joints .npz (scripts/hawor_to_joints.py)")
parser.add_argument("--hawor-fps", type=float, default=None, help="override hawor playback fps")
parser.add_argument("--loop", action="store_true", help="loop the hawor sequence instead of exiting at end")
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--mode", choices=["vector", "dexpilot"], default="dexpilot")
parser.add_argument("--usd", default=None, help="override hand USD path (default: repo's converted USD)")
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
from magicdexmate.sources.hawor_source import HaWoRSource  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource  # noqa: E402

# sources driven by playback time (not wall clock): sampled via sample_at(sim_t)
_REPLAY_SOURCES = (MockGloveSource, HaWoRSource)

print(f"[setup] building retargeting (pre-Kit): sharpa_wave {args_cli.hand} {args_cli.mode}")
retargeting = build_sharpa_retargeting(args_cli.hand, args_cli.mode)
mapper = JointMapper(retargeting, args_cli.hand)
if args_cli.source == "mock":
    src_kwargs = {"motion": args_cli.motion}
elif args_cli.source == "hawor":
    if args_cli.hawor_npz is None:
        raise SystemExit("--source hawor requires --hawor-npz PATH (see scripts/hawor_to_joints.py)")
    src_kwargs = {"npz": args_cli.hawor_npz, "fps": args_cli.hawor_fps, "loop": args_cli.loop}
else:
    src_kwargs = {"sn": args_cli.sn, "address": args_cli.address}
source = make_source(args_cli.source, args_cli.hand, **src_kwargs)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from sharpa_scene import SharpaSceneCfg, sharpa_sdk_joint_names  # noqa: E402


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
    else:
        # default to this repo's converted USD for the chosen hand
        # (assets/robots/hands/sharpa_wave/<hand>/<hand>_sharpa_wave.usd,
        # produced by scripts/convert_sharpa_urdf.py); no vendor USD needed.
        repo_root = os.path.dirname(_THIS_DIR)
        default_usd = os.path.join(
            repo_root, "assets", "robots", "hands", "sharpa_wave",
            args_cli.hand, f"{args_cli.hand}_sharpa_wave.usd")
        if not os.path.exists(default_usd):
            raise SystemExit(
                f"no USD for {args_cli.hand} hand at {default_usd}; "
                f"run: scripts/convert_sharpa_urdf.py --side {args_cli.hand}  (or pass --usd)")
        scene_cfg.robot.spawn.usd_path = default_usd
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
                if isinstance(source, _REPLAY_SOURCES):
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
            if getattr(source, "ended", False):
                print(f"\n[done] hawor sequence finished ({source.n_frames} frames @ {source.fps:g}fps)")
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
