"""逐行物理状态探针: 回答"五段流程在仿真里断在哪一环".

  ① 接近 ② 左手握瓶 ③ 左手转瓶到水平 ④ 瓶水平后右手拧 ⑤ 拧下后放桌上

逐步记录: row/k | 瓶倾角 盖倾角 | 左垫 右垫 | 瓶-左腕距 盖-右腕距 | 螺纹角 | G链 | fail_code
默认载 ckpt 用 mu 确定性回放; --zero 则零动作 (= 母带本身能不能走通)。
必须带与训练同套 POUR_* 环境变量。

  ... probe_phases.py --checkpoint <last.pth> --headless
  ... probe_phases.py --zero --headless
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["POUR_NO_D6"] = "1"

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="")
p.add_argument("--zero", action="store_true", help="零动作回放 (不载 ckpt)")
p.add_argument("--steps", type=int, default=700)
p.add_argument("--every", type=int, default=10, help="非交互段的打印间隔")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if not args.zero and not args.checkpoint:
    p.error("需要 --checkpoint 或 --zero")

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_phases")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
import task_env as PE  # noqa: E402
import task_config as TC  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

_CW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring")
cfg = PE.build_cfg(num_envs=1)
raw = PE.UnscrewEnv(cfg)
raw.force_entry = [0]
env = GymStyleEnvWrapper(raw, clip_actions=1.0)

agent = None
if not args.zero:
    from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
    from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
    with open(os.path.join(_CW, "ppo_task.yaml")) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["algorithm"]["num_actors"] = 1
    agent = PPO(env, output_dir="/tmp/unscrew_phases",
                full_config=ConfigWrapper(agent_cfg, {}, test=True),
                create_output_dir=False)
    agent.restore_test(args.checkpoint)
    agent.set_eval()

dev = raw.device
UPZ = torch.tensor([0.0, 0.0, 1.0], device=dev)


def tilt_deg(quat_w):
    """物体局部 +Z 旋到世界后与世界 +Z 的夹角 (deg)."""
    from isaaclab.utils.math import quat_apply
    up = quat_apply(quat_w, UPZ.unsqueeze(0).expand(len(quat_w), 3))
    c = (up[:, 2] / up.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


_prev = {"wlq": None, "bq": None}


def _qang(qa, qb):
    """两四元数间夹角 (deg)."""
    d = float(torch.abs((qa * qb).sum()).clamp(max=1.0))
    return float(np.degrees(2 * np.arccos(d)))


def snap():
    f = raw._pads_f().norm(dim=-1)[0]
    bot_p, cap_p = raw.object.data.root_pos_w[0], raw.aux.data.root_pos_w[0]
    tb = float(tilt_deg(raw.object.data.root_quat_w)[0])
    tc = float(tilt_deg(raw.aux.data.root_quat_w)[0])
    wl = raw.hand.data.body_pos_w[0, raw.wid["L"]]
    wr = raw.hand.data.body_pos_w[0, raw.wid["R"]]
    # 跟踪诊断: 实际左腕转速 / 瓶转速 (deg/步) / 左臂关节跟踪误差 (deg, max)
    wlq = raw.hand.data.body_quat_w[0, raw.wid["L"]].clone()
    bq = raw.object.data.root_quat_w[0].clone()
    w_rot = _qang(wlq, _prev["wlq"]) if _prev["wlq"] is not None else 0.0
    b_rot = _qang(bq, _prev["bq"]) if _prev["bq"] is not None else 0.0
    _prev["wlq"], _prev["bq"] = wlq, bq
    r0 = int(raw.row[0].clamp(max=raw.T_ROW - 1))
    q_cmd = raw.ref58[r0, 7:14] + raw.cum_res[0, 7:14]
    q_act = raw.hand.data.joint_pos[0, raw.map_ids_t[7:14]]
    trk = float(torch.rad2deg((q_cmd - q_act).abs().max()))
    # 铰链诊断: 左 5 垫沿瓶轴的高度 (相对瓶底, cm) 的散布; 散布小 = 五垫共线 = 可绕之转动
    from isaaclab.utils.math import quat_apply
    axis = quat_apply(raw.object.data.root_quat_w[:1], UPZ.unsqueeze(0))[0]
    pads_l = torch.stack([raw.hand.data.body_pos_w[0, raw._pad_bids[i]] for i in range(5)])
    h = ((pads_l - bot_p.unsqueeze(0)) * axis.unsqueeze(0)).sum(1) * 100
    hmask = f[:5] > 0.5
    hs = h[hmask] if hmask.any() else h
    return dict(
        wrot=w_rot, brot=b_rot, trk=trk,
        padh=f"{float(hs.min()):4.1f}~{float(hs.max()):4.1f}",
        row=int(raw.row[0]), k=int(raw.PB.k[0]),
        tb=tb, tc=tc,
        nl=int((f[:5] > 0.5).sum()), nr=int((f[5:] > 0.5).sum()),
        fr=float(f[5:].max()), fl=float(f[:5].sum()),
        dl=float((wl - bot_p).norm()) * 100, dr=float((wr - cap_p).norm()) * 100,
        botz=float(bot_p[2]) * 100, capz=float(cap_p[2]) * 100,
        screw=float(torch.rad2deg(raw.screw_angle[0])),
        g=[int(raw.PB.g1[0]), int(raw.PB.g2[0]), int(raw.PB.g3[0]), int(raw.PB.g4[0])],
        rel=int(raw.PB.released[0]), placed=int(raw.PB.placed[0]),
        fc=int(raw.fail_code[0]) if hasattr(raw, "fail_code") else -1,
    )


mode = "零动作" if args.zero else f"ckpt={os.path.basename(os.path.dirname(args.checkpoint))}"
_m = float(raw.obj_mass[0]) if hasattr(raw, "obj_mass") else float("nan")
_mu = float(raw.obj_fric[0]) if hasattr(raw, "obj_fric") else float("nan")
print(f"[phases] {mode} | 全链 {raw.T_ROW} 行 IA=[{raw.IA0},{raw.IA1}] RETREAT0={raw.RETREAT0} "
      f"| 瓶质量 {_m:.3f}kg 物体μ {_mu:.2f} "
      f"(指垫 SuperGrip {float(os.environ.get('POUR_PAD_FRIC', TC.PAD_FRICTION)):.1f}, "
      f"combine=multiply)",
      flush=True)
print(f"{'t':>4s} {'row':>4s} {'k':>3s} | {'瓶倾':>5s} {'盖倾':>5s} | {'垫L':>3s} {'垫R':>3s} {'R力':>5s} | "
      f"{'瓶-左腕':>7s} {'盖-右腕':>7s} | {'瓶z':>5s} {'盖z':>5s} | {'螺纹':>5s} | "
      f"{'腕转':>5s} {'瓶转':>5s} {'臂误':>5s} | G链     rel pl | fc",
      flush=True)

obs = env.reset()
gmax = [0, 0, 0, 0]
last_row = -1
with torch.no_grad():
    for t in range(args.steps):
        if args.zero:
            act = torch.zeros(1, raw.cfg.action_space if hasattr(raw.cfg, "action_space")
                              else PE.ACT_DIM, device=dev)
        else:
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
            act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        s = snap()
        for i, gv in enumerate(s["g"]):
            gmax[i] = max(gmax[i], gv)
        in_ia = raw.IA0 - 6 <= s["row"] <= raw.IA1 + 6
        if in_ia or t % args.every == 0 or bool(dones[0]) or s["row"] != last_row and s["row"] in (raw.IA0, raw.IA1, raw.RETREAT0):
            print(f"{t:4d} {s['row']:4d} {s['k']:3d} | {s['tb']:5.1f} {s['tc']:5.1f} | "
                  f"{s['nl']:3d} {s['nr']:3d} {s['fr']:5.2f} L{s['fl']:5.1f}N | {s['dl']:7.1f} {s['dr']:7.1f} | "
                  f"{s['botz']:5.1f} {s['capz']:5.1f} | {s['screw']:5.1f} | "
                  f"{s['wrot']:5.1f} {s['brot']:5.1f} {s['trk']:5.1f} | 垫高{s['padh']:>10s} | {s['g']} {s['rel']:>3d} {s['placed']:>2d} | {s['fc']}",
                  flush=True)
        last_row = s["row"]
        if bool(dones[0]):
            print(f"[phases] 终止 @t={t} row={s['row']} fail_code={s['fc']} "
                  f"(0=无 1x=D1掉 2x=D2倒 3x=D3偏 4=D4滑 5=D5插桌 6=D6互碰; 个位=物体 0瓶1盖)",
                  flush=True)
            break
print(f"[phases] G链最大={gmax} | 终步 t={t}", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
# 不调 app.close(): 本仓库 Isaac 关闭必卡死 (record_task 也是), 卡住的进程会
# 一直占显存; 产物已全部打印, 直接硬退出。
os._exit(0)
