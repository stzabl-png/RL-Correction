#!/usr/bin/env python3
"""把录好的关节指令喂进 Isaac 跑一遍物理,看机器人实际做得到做不到。

浏览器里的预览是纯运动学:把指令角度摆进 URDF,一定"做得到"。物理不是这样,
PD 跟不跟得上、重力下会不会塌、快段会不会超力矩,都要真跑一遍才知道。上真机
之前这一道是有意义的:真机也是靠自己的控制器去追同一批指令。

输入是 scripts/replay_check.py --record 存下来的那种文件,里面每帧带一个
cmd(指令关节角)。本程序按录制时的节奏把它设成关节目标,让物理跑,然后报告
指令与实际之间的差距。

判据:指令与实际的偏差要小且稳定。某个关节持续差很多,说明该关节在物理下追
不上,那么真机上大概率也追不上,而浏览器预览是看不出来的。

用法(先 cd 到 <仓库根目录>,脚本与运行环境都在那里)
    cd <仓库根目录>
    .venv-isaac/bin/python sim/replay_in_isaac.py --file logs/take_0001.msgpack
    .venv-isaac/bin/python sim/replay_in_isaac.py --file logs/take_0001.msgpack \\
        --headless --joint-csv /tmp/isaac_replay.csv

    # 让浏览器页面同时显示这次物理复演(用与实时遥操作相同的地址)
    .venv-isaac/bin/python sim/replay_in_isaac.py --file logs/take_0001.msgpack \\
        --pub-state tcp://*:5583
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import msgpack
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True, help="replay_check.py --record 存下的文件")
parser.add_argument("--physics-hz", type=float, default=50.0)
parser.add_argument("--joint-csv", default=None, help="逐帧写出指令与实际")
parser.add_argument("--pub-state", default=None,
                    help="把实际关节角发出去,浏览器页面可直接观看")
parser.add_argument("--loop", action="store_true",
                    help="循环播放。目视对照 Isaac 与浏览器时用得上")
parser.add_argument("--settle-s", type=float, default=2.0,
                    help="开跑前先让机器人在初始位稳定这么久")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # magicdexmate
from vega_scene import VegaSceneCfg  # noqa: E402


def load(path):
    with open(path, "rb") as fh:
        return [m for m in msgpack.Unpacker(fh, raw=False) if m.get("cmd")]


def main() -> int:
    frames = load(args_cli.file)
    if not frames:
        print("[复演] 文件里没有一帧带指令,无法复演")
        return 1
    t = [f.get("_t", 0.0) for f in frames]
    print(f"[复演] {args_cli.file}:{len(frames)} 帧,{t[-1] - t[0]:.1f} 秒")

    sim = SimulationContext(SimulationCfg(
        dt=1.0 / args_cli.physics_hz, render_interval=4, device="cpu",
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4,
                       bounce_threshold_velocity=0.2)))
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    names = list(robot.joint_names)
    targets = robot.data.default_joint_pos.clone()

    # 轮子每步钉住。这几个转向关节在物理里会数值发散,不钉住会污染整棵关节树,
    # 手臂精度会跟着崩 —— 这是 2026-07-29 查了很久才定位到的。
    wheel_ids = [i for i, n in enumerate(names) if "_wheel_j" in n]
    wheel_ids_t = torch.tensor(wheel_ids, device=robot.device, dtype=torch.long)
    wheel_home = robot.data.default_joint_pos[:, wheel_ids].clone()

    # 指令里出现过的关节,按名字对到 Isaac 的下标
    keys = [k for k in frames[0]["cmd"] if k in names]
    # dtype 要显式给 long:默认推断出来的类型不能当索引用
    ids = torch.tensor([names.index(k) for k in keys],
                       device=robot.device, dtype=torch.long)
    print(f"[复演] 指令覆盖 {len(keys)} 个关节:{', '.join(keys[:6])} ...")

    pub = None
    if args_cli.pub_state:
        import zmq
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(args_cli.pub_state)
        print(f"[复演] 实际关节角发布于 {args_cli.pub_state}")

    csv_w = None
    if args_cli.joint_csv:
        fh = open(args_cli.joint_csv, "w", newline="")
        csv_w = csv.writer(fh)
        csv_w.writerow(["t"] + [f"cmd_{k}" for k in keys]
                       + [f"got_{k}" for k in keys])

    def step_once():
        robot.set_joint_position_target(targets)
        robot.write_joint_state_to_sim(
            wheel_home, torch.zeros_like(wheel_home), joint_ids=wheel_ids_t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    for _ in range(int(args_cli.settle_s * args_cli.physics_hz)):
        step_once()

    dt_sim = sim.get_physics_dt()
    t0 = t[0]
    i = 0
    sim_t = 0.0
    err = []
    dur = t[-1] - t0
    # 用时间轴判结束,不要用下标。下标最多停在 len-1,拿 i < len(frames) 当
    # 条件永远为真,循环不会退出 —— 第一版就是这么写的,跑起来一直不停。
    n_loop = 0
    while (sim_t <= dur or args_cli.loop) and simulation_app.is_running():
        if sim_t > dur:            # 循环:回到片段开头,机器人姿势保持连续
            sim_t = 0.0
            i = 0
            n_loop += 1
            print(f"[复演] 第 {n_loop} 遍播完,重新开始")
        # 按录制时的时间轴推进,而不是每步取一帧:两者节奏不同会把动作放快或放慢
        while i + 1 < len(frames) and (t[i + 1] - t0) <= sim_t:
            i += 1
        cmd = frames[i]["cmd"]
        vals = torch.tensor([[float(cmd[k]) for k in keys]],
                            device=robot.device, dtype=targets.dtype)
        targets[:, ids] = vals
        step_once()
        sim_t += dt_sim

        got = robot.data.joint_pos[0, ids].cpu().numpy()
        want = np.array([float(cmd[k]) for k in keys])
        err.append(np.abs(want - got))
        if csv_w:
            csv_w.writerow([f"{sim_t:.4f}"] + [f"{v:.6f}" for v in want]
                           + [f"{v:.6f}" for v in got])
        if pub is not None:
            q = {n: float(v) for n, v in
                 zip(names, robot.data.joint_pos[0].cpu().numpy())}
            pub.send(msgpack.packb(
                {"t_us": int(sim_t * 1e6), "q": q, "cmd": dict(cmd),
                 "engaged": [], "mode": "replay", "arm": {}},
                use_bin_type=True))

    e = np.degrees(np.asarray(err))
    print(f"\n[复演] 共 {len(e)} 步,{sim_t:.1f} 秒")
    print(f"[复演] 指令与实际的偏差:所有关节 中位 {np.median(e):.3f}°  "
          f"p99 {np.percentile(e, 99):.2f}°  最大 {e.max():.2f}°")
    worst = e.mean(axis=0).argsort()[::-1][:5]
    print("[复演] 平均偏差最大的五个关节:")
    for j in worst:
        print(f"    {keys[j]:10s} 平均 {e[:, j].mean():6.3f}°  "
              f"最大 {e[:, j].max():6.2f}°")
    # 能对这个数失败的判据:偏差大到这个程度,浏览器预览是看不出来的
    bad = [keys[j] for j in range(len(keys)) if e[:, j].mean() > 5.0]
    print("[复演] " + ("全部关节平均偏差都在 5° 以内,物理下追得上"
                       if not bad else
                       f"⚠ 这些关节平均偏差超过 5°,物理下追不上:{bad}"))
    if csv_w:
        fh.close()
        print(f"[复演] 逐帧数据已写入 {args_cli.joint_csv}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
