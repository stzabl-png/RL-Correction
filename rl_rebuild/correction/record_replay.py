"""无头录像: DexMate 臂+手 IK 跟踪重建轨迹, 存成 mp4.

  $PY -m rl_rebuild.correction.record_replay --clip Grasp2 --out videos/Grasp2.mp4

相机默认斜俯视全景 (0.9,0.9,1.35 -> 0,0,0.90). 要近距离平视对比图用
--eye 0.30,0,1.05 --lookat -0.70,0,1.05
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--out", default=None, help="默认 placement_videos/<clip>.mp4")
p.add_argument("--loops", type=float, default=1.0, help="录几遍参考轨迹")
p.add_argument("--fps", type=int, default=0,
               help="0=自动取源帧率(推荐, 保证时长与原视频一致)")
p.add_argument("--eye", default="0.9,0.9,1.35")
p.add_argument("--lookat", default="0.0,0.0,0.90")
p.add_argument("--res", default="1280,720")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.dexmate_follow import DexmateFollower  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402

out = args.out or os.path.join("placement_videos", f"{args.clip}.mp4")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
_eye = tuple(float(v) for v in args.eye.split(","))
_look = tuple(float(v) for v in args.lookat.split(","))
_res = tuple(int(v) for v in args.res.split(","))

cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.rsi_prob = 0.0
cfg.viewer = ViewerCfg(eye=_eye, lookat=_look, origin_type="world", resolution=_res)

# 帧率必须等于源帧率, 否则视频时长和原视频对不上.
# 实测源数据是 15fps, 之前硬编码写 20fps -> 快 1.33 倍.
_src_fps = float(np.load(clips.clip_entry(args.clip)["npz"], allow_pickle=True)["fps"])
_fps = args.fps if args.fps > 0 else int(round(_src_fps))
print(f"[rec] 源帧率 {_src_fps} -> 输出 {_fps} fps")

env = SharpaCorrectionEnv(cfg, render_mode="rgb_array")
env.reset()

# 飞手隐藏 (录的是 DexMate 在动)
try:
    import omni.usd
    from pxr import UsdGeom
    st = omni.usd.get_context().get_stage()
    for i in range(env.num_envs):
        pr = st.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        if pr.IsValid():
            UsdGeom.Imageable(pr).MakeInvisible()
    print("[rec] 飞手已隐藏")
except Exception as e:
    print(f"[rec] ⚠ 隐藏飞手失败: {e}")

fol = DexmateFollower(env, log=print)
zero = torch.zeros(1, 28, device=env.device)
n_steps = int((fol.T if fol.ok else 125) * args.loops)

# 渲染管线要先跑几帧才出图 (只 reset+render 会拿到纯黑)
for _ in range(8):
    env.step(zero)
for _ in range(5):
    env.render()

frames, errs = [], []
for k in range(n_steps):
    e = fol.step(k, log_every=40)
    if e is not None:
        errs.append(e)
    env.step(zero)
    img = env.render()
    if img is not None:
        frames.append(np.asarray(img))

if frames:
    import imageio.v2 as iio
    iio.mimwrite(out, frames, fps=_fps, macro_block_size=1)
    a = np.stack(frames[:5])
    print(f"[rec] 已保存 {out}  {len(frames)} 帧 @{_fps}fps = {len(frames)/_fps:.2f}秒  "
          f"画面标准差 {a.std():.1f} {'(⚠ 疑似纯色)' if a.std() < 5 else ''}")
else:
    print("[rec] ❌ 没抓到任何帧")
if errs:
    print(f"[rec] IK 跟踪误差: 均值 {np.mean(errs)*100:.1f}cm  "
          f"中位 {np.median(errs)*100:.1f}cm  最大 {np.max(errs)*100:.1f}cm")
env.close()
app.close()
