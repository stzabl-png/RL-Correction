"""Pour17 单集策略录像: 载 ckpt, 1 env t0 全程, mu 确定性, 相机 1280x720 mp4。
必须带与训练同套 POUR_* 环境变量 (B线的 squeeze 前馈是行为的一部分)。

  ... record_pour.py --checkpoint <last.pth> --out <mp4> --headless --enable_cameras
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["POUR_NO_D6"] = "1"   # 录像不装 D6 传感器 (1-env PhysX 断言规避)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=700)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pour17_rec")
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pour_env as PE  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
cfg = PE.build_cfg(num_envs=1)
raw = PE.PourEnv(cfg)
raw.force_entry = [0]
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
with open(os.path.join(_HERE, "ppo_pour.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = 1
agent = PPO(env, output_dir="/tmp/pour17_rec",
            full_config=ConfigWrapper(agent_cfg, {}, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

_st = omni.usd.get_context().get_stage()
_cam = UsdGeom.Camera.Define(_st, "/World/RecCam")
_cam.CreateFocalLengthAttr().Set(16.0)
_m = Gf.Matrix4d()
_m.SetLookAt(Gf.Vec3d(0.85, -1.15, 1.60), Gf.Vec3d(-0.15, 0.10, 0.95),
             Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(_cam).AddTransformOp().Set(_m.GetInverse())
_rp = rep.create.render_product("/World/RecCam", (1280, 720))
_annot = rep.AnnotatorRegistry.get_annotator("rgb")
_annot.attach(_rp)

obs = env.reset()
frames = []
with torch.no_grad():
    for t in range(args.steps):
        inp = {"obs": agent.running_mean_std(obs["obs"]),
               "priv_info": obs["priv_info"]}
        mu = agent.model.act_inference(inp)
        obs, rew, dones, infos = env.step(torch.clamp(mu, -1.0, 1.0))
        raw.sim.render()
        d = _annot.get_data()
        if d is not None and getattr(d, "size", 0):
            frames.append(np.asarray(d)[..., :3].astype(np.uint8))
        if bool(dones[0]):
            break
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
imageio.mimsave(args.out, frames, fps=15)
print(f"[record] {args.out} | {len(frames)} 帧 | 终步 {t} "
      f"M链={[int(x) for x in (raw.PB.ms1[0], raw.PB.ms2[0], raw.PB.ms3[0], raw.PB.ms4[0])]}")
try:
    _slot.release()
except Exception:
    pass
app.close()
