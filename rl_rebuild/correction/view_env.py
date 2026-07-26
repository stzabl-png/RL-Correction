"""打开 GUI 看训练环境到底在发生什么 (飞手 / DexMate 都能看).

不是回放器 —— 它跑的就是**训练用的那个 env**, 所以你看到的和 RL 看到的完全一致.
默认零残差 (只跟参考, 不加修正); 给 --checkpoint 就看策略在干什么.

  # 看 DexMate 现在的抓取 (当前默认设定)
  ./rl_rebuild/correction/view.sh

  # 看接近段 (目前是坏的: 手全程离物体 20-32cm)
  GRASP_APPROACH=40 ./rl_rebuild/correction/view.sh

  # 和飞手对照
  ./rl_rebuild/correction/view.sh --robot flying

窗口里左下角会实时打印: 手离物体多远 / 几根指尖在接触 / 物体抬了多高.
关窗口或 Ctrl-C 退出.
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--robot", default="dexmate", choices=("flying", "dexmate"))
p.add_argument("--clip", default="Grasp2")
p.add_argument("--checkpoint", default=None, help="不给就是零残差 (只跟参考)")
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--speed", type=float, default=1.0,
               help="播放速度倍率. 1.0=实时 20Hz; 0.3=慢放看细节; 0=不限速")
p.add_argument("--eye", default="0.75,0.75,1.35", help="相机位置")
p.add_argument("--lookat", default="0.0,0.0,0.92", help="相机看向")
p.add_argument("--every", type=int, default=10, help="每几步打印一行状态")
p.add_argument("--video", default=None,
               help="录像输出 mp4 (给了就转无头录制, 不开 GUI)")
p.add_argument("--loops", type=float, default=1.0, help="录几个完整回合")
p.add_argument("--res", default="1600,900")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
# 录像 = 无头 + 开相机; 否则开 GUI
args.headless = bool(args.video)
if args.video:
    args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewer")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.registry import make_env  # noqa: E402

EnvCls, CfgCls = make_env(args.robot)
cfg = CfgCls()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = args.num_envs
cfg.rsi_prob = 0.0                        # 从头抓, 别用 RSI 跳到中间
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world",
                       resolution=tuple(int(v) for v in args.res.split(",")))

env = EnvCls(cfg, render_mode="rgb_array" if args.video else None)
env.reset()
act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

policy = None
if args.checkpoint:
    print(f"[view] 载入策略 {args.checkpoint}")
    # 与 eval_policy 同一套载入方式; 失败就退回零残差, 不让窗口白开
    try:
        from rl_rebuild.correction.eval_policy import build_policy   # type: ignore
        policy = build_policy(env, args.checkpoint)
    except Exception as e:
        print(f"[view] ⚠ 载入失败 ({e}), 退回零残差")

print("\n" + "=" * 92)
print(f"[view] robot={args.robot}  clip={args.clip}  "
      f"{'策略' if policy else '零残差(只跟参考)'}  episode={env.ep_total} 步")
print(f"[view] 相位: 静置 0-{cfg.settle_steps-1} | 抓取 {cfg.settle_steps}-"
      f"{getattr(env,'lift_step0',0)-1} | 抬升 -{getattr(env,'hold_step0',0)-1} | hold 之后")
print("[view] 关窗口或 Ctrl-C 退出")
print("=" * 92)
print(f"{'步':>5}{'相位':>7}{'手离物体':>11}{'接触指数':>10}{'物体抬升':>11}"
      f"{'臂跟踪误差':>13}")

dt = (1.0 / cfg.target_hz) / args.speed if args.speed > 0 else 0.0
frames = []
n_max = int(env.ep_total * args.loops) if args.video else 10 ** 9
if args.video:
    dt = 0.0                                   # 录像不限速
    for _ in range(8):                         # 渲染管线要先跑几帧才出图 (否则纯黑)
        env.step(act)
    for _ in range(5):
        env.render()
k = 0
try:
    while app.is_running() and k < n_max:
        obs = env._get_observations() if policy else None
        if policy is not None:
            with torch.no_grad():
                act = policy(obs).clamp(-1, 1)
        t0 = time.time()
        env.step(act)
        k += 1
        if k % args.every == 0:
            ep = int(env.episode_length_buf[0])
            ph = ("静置" if ep < cfg.settle_steps else
                  "抓取" if ep < getattr(env, "lift_step0", 10**9) else
                  "抬升" if ep < getattr(env, "hold_step0", 10**9) else "hold")
            d = float(env._palm_obj_dist()[0]) * 100
            nc = int(env._tip_contacts()[0].sum())
            lift = float(env.object.data.root_pos_w[0, 2]
                         - env.scene.env_origins[0, 2] - env.obj_rest_z) * 100
            trk = float((env.wrist_pos_w[0] - env.scene.env_origins[0]
                         - env.ref_wrist_pos[int(env._ref_t()[0])]).norm()) * 100
            print(f"{ep:>5}{ph:>7}{d:>9.2f}cm{nc:>8}/5{lift:>9.2f}cm{trk:>11.2f}cm",
                  flush=True)
        if args.video:
            img = env.render()
            if img is not None:
                frames.append(np.asarray(img))
        if dt > 0:
            time.sleep(max(0.0, dt - (time.time() - t0)))
except KeyboardInterrupt:
    print("\n[view] 退出")

if args.video and frames:
    import imageio.v2 as iio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)) or ".", exist_ok=True)
    iio.mimwrite(args.video, frames, fps=int(cfg.target_hz), macro_block_size=1)
    a_ = np.stack(frames[:5])
    print(f"\n[view] 视频 -> {args.video}  {len(frames)} 帧 @{int(cfg.target_hz)}fps "
          f"= {len(frames)/cfg.target_hz:.1f}s  画面标准差 {a_.std():.1f}"
          f"{'  (⚠ 疑似纯色)' if a_.std() < 5 else ''}")
elif args.video:
    print("[view] ❌ 没抓到任何帧")
sys.stdout.flush()
_slot.release()
os._exit(0)
