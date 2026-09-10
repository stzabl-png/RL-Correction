"""Single-env zero-residual physical audit with visible cube and world-state trace."""
import argparse
import json
import os
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--out', required=True)
p.add_argument('--steps', type=int, default=180)
p.add_argument('--video-out', required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
slot = isaac_slot('task1_fixed_cube_probe')
app = AppLauncher(args).app
import numpy as np
import torch
import imageio
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
import omni.usd
from tasks.Sweep.new_data.powerdisk_env import PowerDiskEnv
from tasks.Sweep.new_data.trajectory_env import SE

root = Path('/home/msc-auto/RL_sweep')
out = (root/args.out).resolve()
assert out.is_relative_to(root/'logs')
out.mkdir(parents=True, exist_ok=True)
env = PowerDiskEnv(SE.build_cfg(1, task_config=args.config))
env.suppress_terminal_reset = True
env.force_replay = True  # Inspect the nominal path without waiting for cube progress.
stage = omni.usd.get_context().get_stage()
cam = UsdGeom.Camera.Define(stage, '/World/CubeProbeCamera')
cam.CreateFocalLengthAttr().Set(18)
cam.CreateClippingRangeAttr().Set(Gf.Vec2f(.01, 100))
m = Gf.Matrix4d()
m.SetLookAt(Gf.Vec3d(1.25,-1.65,1.55), Gf.Vec3d(.02,0,1.02), Gf.Vec3d(0,0,1))
UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
product = rep.create.render_product('/World/CubeProbeCamera', (1280,720))
annot = rep.AnnotatorRegistry.get_annotator('rgb')
annot.attach(product)
env.reset()
for _ in range(6):
    env.sim.render()
start = env.cube.data.root_pos_w[0].cpu().numpy().copy()
trace = {k: [] for k in ('cube_pos', 'cube_vel', 'pan_pose', 'broom_pose', 'row', 'gates', 'cum_res')}
video_path = (root/args.video_out).resolve()
assert video_path.is_relative_to(root/'outputs_video')
video_path.parent.mkdir(parents=True, exist_ok=True)
writer = imageio.get_writer(str(video_path), fps=20)
with torch.no_grad():
    for row in range(min(args.steps, env.T)):
        env.step(torch.zeros((1,14), device=env.device))
        env.sim.render()
        frame = np.asarray(annot.get_data())[...,:3].copy()
        writer.append_data(frame)
        if row in (0,9,19,49,99,args.steps-1):
            imageio.imwrite(out/f'frame_{row:04d}.png', frame)
        trace['cube_pos'].append(env.cube.data.root_pos_w[0].cpu().numpy().copy())
        trace['cube_vel'].append(env.cube.data.root_lin_vel_w[0].cpu().numpy().copy())
        for key, art in (('pan_pose',env.aux),('broom_pose',env.object)):
            trace[key].append(torch.cat((art.data.root_pos_w[0],art.data.root_quat_w[0])).cpu().numpy().copy())
        trace['row'].append(int(env.row[0]))
        trace['gates'].append(env._tick_out['gates'][0].cpu().numpy().copy())
        trace['cum_res'].append(env.cum_res[0].cpu().numpy().copy())
        if row%30 == 0:
            print('probe row',row,'cube',trace['cube_pos'][-1].tolist(), flush=True)
writer.close()
arr = {k:np.asarray(v) for k,v in trace.items()}
np.savez_compressed(out/'trace.npz', initial_cube_world=start, **arr)
xy = np.linalg.norm(arr['cube_pos'][:20,:2]-start[:2],axis=1)
speed = np.linalg.norm(arr['cube_vel'][:20],axis=1)
report = dict(config=args.config, frames=len(xy) if not len(arr['row']) else len(arr['row']),
    physical_cube_side_m=float(env.cube.cfg.spawn.size[0]),
    geometry_cube_half_m=float(env.geometry.cube_half),
    initial_cube_world_m=start.tolist(), first_second_max_xy_displacement_m=float(xy.max()),
    first_second_max_speed_mps=float(speed.max()),
    full_max_speed_mps=float(np.linalg.norm(arr['cube_vel'],axis=1).max()),
    first_second_stable=bool(xy.max()<.003 and speed.max()<.08),
    zero_residual=bool(np.max(np.abs(arr['cum_res']))==0),
    force_replay=True, training_ready=False,
    note='First-second stability audit only; inspect video and actual collision before training.')
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
slot.release()
sys.stdout.flush()
os._exit(0)
