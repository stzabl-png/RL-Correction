"""Record one deterministic Sweep2 episode under project-root outputs_video/."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="", help="empty means zero-residual reference replay")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=320)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep2_record")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_env as SE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
allowed = os.path.join(ROOT, "outputs_video")
out = os.path.abspath(args.out)
assert os.path.commonpath([allowed, out]) == allowed, f"video must be under {allowed}"

raw = SE.SweepEnv(SE.build_cfg(1))
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent = None
if args.checkpoint:
    with open(os.path.join(os.path.dirname(__file__), "ppo_sweep.yaml")) as f:
        acfg = yaml.safe_load(f)
    acfg["algorithm"]["num_actors"] = 1
    agent = PPO(env, output_dir=os.path.join(ROOT, "logs", "_record_tmp"),
                full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
    agent.restore_test(args.checkpoint); agent.set_eval()

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
cam = UsdGeom.Camera.Define(stage, "/World/SweepRecCam")
cam.CreateFocalLengthAttr().Set(18.0)
m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(0.95, -1.15, 1.45),
                               Gf.Vec3d(0.25, 0.0, 0.90), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(cam).AddTransformOp().Set(m.GetInverse())
rp = rep.create.render_product("/World/SweepRecCam", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb"); annot.attach(rp)

obs = env.reset(); frames = []; gmax = [0, 0, 0, 0]
with torch.no_grad():
    for t in range(args.steps):
        if agent is None:
            action = torch.zeros(1, SE.ACT_DIM, device=raw.device)
        else:
            action = agent.model.act_inference({
                "obs": agent.running_mean_std(obs["obs"]),
                "priv_info": obs["priv_info"]}).clamp(-1, 1)
        obs, reward, done, info = env.step(action)
        for i in range(4): gmax[i] = max(gmax[i], int(raw.progress.gates[0, i]))
        raw.sim.render(); data = annot.get_data()
        if data is not None and getattr(data, "size", 0):
            frames.append(np.asarray(data)[..., :3].astype(np.uint8))
        if bool(done[0]): break
os.makedirs(os.path.dirname(out), exist_ok=True)
imageio.mimsave(out, frames, fps=20)
print(f"[record_sweep] {out} | frames={len(frames)} step={t} gates={gmax}")
try: _slot.release()
except Exception: pass
sys.stdout.flush(); app.close(); os._exit(0)
