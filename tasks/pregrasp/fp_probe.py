"""摆放指纹独立探针 (无 Isaac): 走 clips.load_data_unit 的同一路径, 打印 builder 摆放。

  PYTHONPATH=. $PY tasks/pregrasp/fp_probe.py
两台机器各跑, diff 输出 —— 快速定位远端 15.6cm 摆放分歧的层。
"""
import numpy as np

from rl_rebuild.correction import clips
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior


cfg = GraspTaskCfg()
clips.configure_cfg(cfg, "Pour17_bottle")
cfg.approach_only = True
cfg.action_space = 7
apply_grasp_prior(cfg, "tasks/pregrasp/priors/Pour17_bottle.npz", 19.5, approach=True)
du = clips.load_data_unit(cfg)
print("[fp] mesh:", du.mesh_path)
print("[fp] obj_init_pos:", np.round(np.asarray(du.obj_init_pos, float), 4))
print("[fp] obj_init_quat:", np.round(np.asarray(du.obj_init_quat, float), 4))
w = np.asarray(du.ref_wrist_pos, float)
print("[fp] 腕参考 首帧:", np.round(w[0], 4), "| 末帧:", np.round(w[-1], 4),
      "| gs帧:", np.round(w[min(25, len(w) - 1)], 4))
print("[fp] 腕z范围:", round(float(w[:, 2].min()), 4), "~", round(float(w[:, 2].max()), 4))
