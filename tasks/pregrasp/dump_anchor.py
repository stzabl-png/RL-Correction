"""把 env 里**实测的**臂基座锚点 `anchor_T` 落盘, 供离线可达性筛选(Step3)使用。

## 为什么需要它

`screen_prior.py::gate1` 的可达性用的是 `ArmIK(hand)` —— **URDF 名义基座**;
而 env 真正解 IK 用的是 `ArmIK(hand, anchor_link="arm_center", anchor_T=实测位姿)`。
`dexmate_env.py:290` 那条注释说得很直白:

    锚在**实测的 arm_center** 上, 不用"躯干在配置角度"这个假设 ——
    否则躯干沉降几度就让整条臂的基座偏掉, q_ref 会是个到不了的目标。

⟹ 两边解的不是同一个末端坐标系, 可达性结论必然打架。
2026-08-17 实测: 离线 Gate 1 说 yaw 175° 可达, env 里 `grasp ok=False` 够不着。

本脚本从活着的 env 里把 `anchor_T` 读出来存成 json, 让离线筛选能用**同一个基座**。

## 用法

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.dump_anchor --headless \\
        --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz \\
        --prior_yaw 19.5

产物: tasks/pregrasp/priors/anchor_T_<hand>.json
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--out", default=None, help="默认 tasks/pregrasp/priors/anchor_T_<hand>.json")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("dump_anchor")
app = AppLauncher(args).app

import datetime  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

T = np.asarray(E._anchor_T, dtype=float)
assert T.shape == (4, 4), f"anchor_T 形状不对: {T.shape}"

# 机器人 USD 路径 —— 换了机器人资产, 这份 anchor 就作废
_usd = ""
for _a in ("usd_path", "spawn"):
    _o = getattr(getattr(cfg.scene, "robot", None), _a, None)
    if isinstance(_o, str):
        _usd = _o
        break
    _usd = str(getattr(_o, "usd_path", "") or "")
    if _usd:
        break

out = args.out or os.path.join(os.path.dirname(__file__), "priors",
                               f"anchor_T_{cfg.hand_side}.json")
rec = dict(
    anchor_T=T.tolist(),
    anchor_link="arm_center",
    hand_side=cfg.hand_side,
    clip=args.clip,
    prior=os.path.abspath(args.grasp_prior),
    prior_yaw_deg=float(args.prior_yaw),
    robot_usd=_usd,
    dumped_at=datetime.datetime.now().isoformat(timespec="seconds"),
    note=("env 实测的臂基座位姿 (body_pos_w/body_quat_w of arm_center)。"
          "离线可达性筛选必须用它, 否则与 env 判据不一致 —— "
          "见 dexmate_env.py:290 与本文件 docstring。"
          "⚠ 换机器人 USD / 改躯干锁定角度后必须重新 dump。"),
)
with open(out, "w") as f:
    json.dump(rec, f, indent=2, ensure_ascii=False)

print(f"[dump_anchor] hand={cfg.hand_side} clip={args.clip}")
print(f"[dump_anchor] 臂基座位置 (env 局部系) = {np.round(T[:3, 3], 5).tolist()} m")
print(f"[dump_anchor] -> {out}")
app.close()
