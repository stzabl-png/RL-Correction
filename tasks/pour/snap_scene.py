"""场景快照: 从真实 env 里导出 Carry 构建器需要的权威口径 (scene_snap.npz)。

导出 (序 = [瓶(A/右), 杯(B/左)]):
  obj_pos/obj_quat    物体静置位姿 (env 系)
  grasp_pos/grasp_quat 真抓姿腕位姿 (_g2, env 系)
  q_grasp_arm         各臂抓姿关节 (phase2 斜坡末行)
  anchor_T            ArmIK 锚定变换
用法: SHARPA_WANDB=0 RL_HAND_JOINTS=1 PYTHONPATH=. $PY -m tasks.pour.snap_scene --headless
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("snap_scene")
app = AppLauncher(args).app

import os  # noqa: E402

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, "Pour17_bottle")
cfg.approach_only = True
cfg.pregrasp29 = True
cfg.pregrasp_phase2 = True          # 为了拿 _g2 (真抓姿) + _p2_arm (抓姿臂IK)
apply_grasp_prior(cfg, "tasks/pregrasp/priors/Pour17_bottle.npz", 19.5, approach=True)
cfg.direct_grasp_prob = 0.0
cfg.approach_t0_max = 0.0
cfg.stance_prob, cfg.retract_ratio = 1.0, 1.0
cfg.curobo_ref_npz = os.path.abspath("tasks/pregrasp/priors/curobo_pour17_direct1cm.npz")
cfg.curobo_ff_freeze_cm = 0.0
cfg.prior_b_npz = os.path.abspath("tasks/pregrasp/priors/Pour17_cup.npz")
cfg.prior_b_yaw_deg = 90.0
_a1, _o1 = cfg.action_space, cfg.observation_space
cfg.action_space, cfg.observation_space = 2 * _a1, 2 * _o1
cfg._obs_single = _o1
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
env = BimanualNativeEnv(cfg)
env.reset()

W = env.scene.env_origins[0].cpu().numpy().astype(np.float64)
out = {"anchor_T": np.asarray(env._anchor_T, np.float64)}
op, oq, gp, gq, qa = [], [], [], [], []
for ns in (env._A, env._B):                       # A=右/瓶, B=左/杯
    with BM.use_side(env, ns):
        op.append(env.object.data.root_pos_w[0].cpu().numpy() - W)
        oq.append(env.object.data.root_quat_w[0].cpu().numpy())
        gp.append(env._g2_pos_w.cpu().numpy())
        gq.append(env._g2_quat_w.cpu().numpy())
        qa.append(env._p2_arm[-1].cpu().numpy())
out.update(obj_pos=np.stack(op), obj_quat=np.stack(oq),
           grasp_pos=np.stack(gp), grasp_quat=np.stack(gq),
           q_grasp_arm=np.stack(qa))
np.savez("tasks/pour/scene_snap.npz", **out)
print(f"[snap] ✅ scene_snap.npz | 瓶位 {np.round(op[0], 3)} 杯位 {np.round(op[1], 3)} "
      f"| 抓腕距物 {np.linalg.norm(gp[0]-op[0])*100:.1f}/{np.linalg.norm(gp[1]-op[1])*100:.1f}cm")
try:
    _slot.release()
except Exception:
    pass
app.close()
