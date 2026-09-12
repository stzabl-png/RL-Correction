"""Isaac half of the joint teach bench: drive Vega joint-by-joint from Viser.

No IK, no teleop, no mapping - raw joint position targets straight to the
articulation. The point is to learn what each joint DOES on this robot (sign,
axis, coupling, useful range) by moving one at a time and watching, so the
mapping work rests on observed behaviour instead of assumptions about the URDF.

Pair with `scripts/viser_joint_jog.py`, which runs in the py3.11 `.venv`
(viser cannot be installed next to isaacsim). They talk msgpack over ZMQ:

    viser  --PUB :5602-->  targets  --SUB-->  this
    this   --PUB :5601-->  state    --SUB-->  viser

State carries the joint names, limits and defaults every message, so the Viser
side can build correct sliders no matter which process started first, and never
has to hardcode a limit that the USD might disagree with.

Run (from the MagicDexMate root):
    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/joint_jog_sim.py
    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/joint_jog_sim.py --headless
"""
import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vega joint teach bench (Isaac side)")
parser.add_argument("--target-port", type=int, default=5602,
                    help="SUB: joint targets from the Viser bench")
parser.add_argument("--state-port", type=int, default=5601,
                    help="PUB: measured joint state + EE poses back to Viser")
parser.add_argument("--physics-hz", type=int, default=120)
parser.add_argument("--render-interval", type=int, default=4)
parser.add_argument("--max-speed", type=float, default=1.0,
                    help="rad/s ceiling on how fast a commanded target may "
                         "travel. Sliders jump; joints should not - this keeps "
                         "a dragged slider from becoming a step input")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import msgpack  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

sys.path.insert(0, "sim")
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from vega_scene import EE_BODY, VegaSceneCfg  # noqa: E402

# wheels are welded (or parked) and never jogged; everything else is fair game
SKIP = "_wheel_j"
STATE_HZ = 20.0


def main():
    sim_cfg = SimulationCfg(
        dt=1 / args_cli.physics_hz, render_interval=args_cli.render_interval,
        device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4,
                       bounce_threshold_velocity=0.2),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()

    names = list(robot.joint_names)
    jog_ids = [i for i, n in enumerate(names) if SKIP not in n]
    jog_names = [names[i] for i in jog_ids]
    # joint_pos_limits on current Isaac Lab; joint_limits is the deprecated
    # spelling still present on older builds
    _lim = getattr(robot.data, "joint_pos_limits", None)
    if _lim is None:
        _lim = robot.data.joint_limits
    limits = _lim[0].cpu().numpy()
    defaults = robot.data.default_joint_pos[0].cpu().numpy()
    meta = {
        "names": jog_names,
        "lo": [float(limits[i, 0]) for i in jog_ids],
        "hi": [float(limits[i, 1]) for i in jog_ids],
        "default": [float(defaults[i]) for i in jog_ids],
    }
    print(f"[jog] {len(jog_names)} jointed: {', '.join(jog_names)}")

    body_idx = {h: robot.body_names.index(b) for h, b in EE_BODY.items()
                if b in robot.body_names}
    if not body_idx:
        print(f"[jog] WARNING: no EE bodies {list(EE_BODY.values())} in this USD "
              "- EE readout disabled")

    # the wheels only exist on the un-welded USD; pin them so their numerical
    # blow-up cannot contaminate the joints being studied (2026-07-29 forensics)
    wheel_ids = [i for i, n in enumerate(names) if SKIP in n]
    if wheel_ids:
        wheel_ids_t = torch.tensor(wheel_ids, device=robot.device)
        wheel_straight = robot.data.default_joint_pos[:, wheel_ids].clone()
        wheel_zero = torch.zeros_like(wheel_straight)
        print(f"[jog] pinning {len(wheel_ids)} wheel joints "
              "(VEGA_WELD_WHEELS=0 in effect)")

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://127.0.0.1:{args_cli.target_port}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args_cli.state_port}")
    print(f"[jog] targets <- :{args_cli.target_port}   state -> :{args_cli.state_port}")
    print("[jog] start the bench:  .venv/bin/python scripts/viser_joint_jog.py")

    targets = robot.data.default_joint_pos.clone()
    wanted = {n: float(defaults[i]) for n, i in zip(jog_names, jog_ids)}
    name_to_id = {n: i for n, i in zip(jog_names, jog_ids)}
    step_cap = args_cli.max_speed * dt
    last_state = 0.0
    got_client = False

    try:
        while simulation_app.is_running():
            try:
                raw = sub.recv(zmq.NOBLOCK)
                msg = msgpack.unpackb(raw, raw=False)
                for n, v in (msg.get("targets") or {}).items():
                    if n in name_to_id:
                        wanted[n] = float(v)
                if not got_client:
                    got_client = True
                    print("\n[jog] bench connected")
            except zmq.Again:
                pass

            # slew-rate limit: a slider drag is a teleport, a joint is not
            for n, want in wanted.items():
                jid = name_to_id[n]
                cur = targets[0, jid].item()
                d = want - cur
                if d > step_cap:
                    d = step_cap
                elif d < -step_cap:
                    d = -step_cap
                targets[0, jid] = cur + d

            robot.set_joint_position_target(targets)
            if wheel_ids:
                robot.write_joint_state_to_sim(wheel_straight, wheel_zero,
                                               joint_ids=wheel_ids_t)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)

            now = time.time()
            if now - last_state > 1.0 / STATE_HZ:
                last_state = now
                q = robot.data.joint_pos[0].cpu().numpy()
                qd = robot.data.joint_vel[0].cpu().numpy()
                ee = {}
                rt = robot.data.root_state_w[:, 0:7]
                for h, bi in body_idx.items():
                    bw = robot.data.body_state_w[:, bi, 0:7]
                    p, qq = subtract_frame_transforms(rt[:, 0:3], rt[:, 3:7],
                                                      bw[:, 0:3], bw[:, 3:7])
                    ee[h] = [float(v) for v in p[0].tolist() + qq[0].tolist()]
                pub.send(msgpack.packb({
                    **meta,
                    "q": [float(q[i]) for i in jog_ids],
                    "qd": [float(qd[i]) for i in jog_ids],
                    "cmd": [float(targets[0, i].item()) for i in jog_ids],
                    "ee": ee,
                }, use_bin_type=True))
    except KeyboardInterrupt:
        print("\n[jog] stopping")
    finally:
        import os
        import threading
        killer = threading.Timer(20.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        simulation_app.close()


if __name__ == "__main__":
    main()
