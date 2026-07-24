# REGRIP sweep R-D3 "GOALGRASP": per-episode goal grasp (joints + per-finger contact IDENTITY). ADDITIVE ONLY.
#
# Design doc: robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md (D3).
# Mechanism: an abstract n-count bonus is satisfiable by squeezing the SAME fingers (the G2 lesson); a
# concrete target specifying WHICH fingers touch at WHAT joint angles is not. Each episode is conditioned
# on one real >=3-finger grasp row from the validated enclosing cache; the reward is a dense, always-live
# gradient toward that row — no event has to fire first (sidesteps the G1-on-ball no-gradient trap).
# Mounted ON the perturbation backbone (shaping-only would repeat the v3a palm-rest failure).
#
# PREREQUISITE (one-time, GPU): rl_rebuild/scripts/annotate_enclosing_contacts.py replays the enclosing
# cache and saves per-row per-finger contact 5-vectors to <cache>_contacts.npy, printing calibration
# quantiles for gg_sigma_q / gg_eps_q (update the cfg defaults from that output if they differ a lot).
#
# Reward (settle- and held-gated, all bounded):
#   r_j = gg_w_j * exp(-||q - q_goal||_1 / gg_sigma_q)
#   r_c = gg_w_c * (1 - mean_f |eng_f - eng_goal_f|)          eng = per-finger IDENTITY {0,1} @ 0.1 N
#   one-time gg_bonus when (dist < gg_eps_q AND match > 0.9) sustained gg_streak_need steps, then a NEW
#   goal row is sampled for that env (implicit within-episode multi-goal curriculum).
#
# Intentionally NOT included (documented limitation): the goal is NOT observed (obs stays 195 so every
# eval script runs unmodified; the policy can only average over goals). If R-D3 shows partial migration,
# the v2 follow-up is goal-conditioned observations.
# Self-collision arm via SHARPA_SELF_COLLISION. Eval probes as in sharpa_wave_graspxl_regrip_gait.py.
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.utils import configclass

from .sharpa_wave_graspxl_adjusthold_perturb import (
    SharpaWaveGraspXLAdjustHoldPerturbCfg,
    SharpaWaveGraspXLAdjustHoldPerturbEnv,
)
from .sharpa_wave_graspxl_regrip_gait import _RegripSelfCollisionPostInit

_GX_OBJ = "002aa1853c974f3a9565e85f5e09a515"


@configclass
class SharpaWaveGraspXLRegripD3GoalGraspSphereCfg(_RegripSelfCollisionPostInit, SharpaWaveGraspXLAdjustHoldPerturbCfg):
    """R-D3 GOALGRASP cfg: perturb backbone + goal-grasp distance reward parameters."""
    gg_pool_path: str = f"cache/graspxl_enclosing_{_GX_OBJ}.npy"            # (M,29) grasp rows
    gg_contacts_path: str = f"cache/graspxl_enclosing_{_GX_OBJ}_contacts.npy"  # (M,5) from annotate script
    gg_min_fingers: float = 3.0     # goal pool filter: contact-sum in [gg_min_fingers, 5]
    gg_w_j: float = 1.0             # joint-distance term weight
    gg_w_c: float = 1.0             # contact-identity match weight
    gg_sigma_q: float = 7.0         # rad; L1 scale — calibrated 2026-07-01 (annotate: ref->goal p50=10.55 rad)
    gg_eps_q: float = 5.0           # rad; L1 success threshold — calibrated likewise (p10=10.02 -> halve it)
    gg_match_min: float = 0.9       # contact-identity match required for success
    gg_bonus: float = 2.0           # one-time success bonus, then the goal is resampled
    gg_streak_need: float = 10.0    # consecutive steps dist<eps AND match>min before the bonus
    gg_contact_thresh: float = 0.1  # N; per-finger engagement threshold for the identity vector


class SharpaWaveGraspXLRegripD3GoalGraspSphereEnv(SharpaWaveGraspXLAdjustHoldPerturbEnv):
    """Perturb regrip backbone + per-episode goal-grasp (joint + contact identity) reward."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        pool_path, con_path = cfg.gg_pool_path, cfg.gg_contacts_path
        if not os.path.isabs(pool_path):
            pool_path = os.path.join(os.getcwd(), pool_path)
        if not os.path.isabs(con_path):
            con_path = os.path.join(os.getcwd(), con_path)
        if not os.path.isfile(con_path):
            raise FileNotFoundError(
                f"[goalgrasp] missing contact annotations {con_path}; run "
                f"rl_rebuild/scripts/annotate_enclosing_contacts.py once before training this task.")
        pool = np.load(pool_path).astype(np.float32)                 # (M,29)
        cons = np.load(con_path).astype(np.float32)                  # (M,5)
        assert pool.ndim == 2 and pool.shape[1] == 29, f"goal pool must be (M,29), got {pool.shape}"
        assert cons.shape == (pool.shape[0], 5), f"contacts must be ({pool.shape[0]},5), got {cons.shape}"
        keep = (cons.sum(-1) >= float(cfg.gg_min_fingers)) & (cons.sum(-1) <= 5.0)
        assert keep.sum() > 0, "[goalgrasp] goal pool empty after the >=min_fingers filter"
        self._gg_pool_dof = torch.from_numpy(pool[keep, :22]).to(self.device)   # (P,22)
        self._gg_pool_eng = torch.from_numpy(cons[keep]).to(self.device)        # (P,5)
        print(f"[goalgrasp] goal pool: {int(keep.sum())}/{pool.shape[0]} rows with "
              f"{int(cfg.gg_min_fingers)}-5 annotated fingertip contacts")
        N = self.num_envs
        self._gg_goal_dof = torch.zeros(N, 22, device=self.device)
        self._gg_goal_eng = torch.zeros(N, 5, device=self.device)
        self._gg_streak = torch.zeros(N, device=self.device)
        self._gg_reached = torch.zeros(N, device=self.device)       # goals reached this episode
        self._gg_sample(torch.arange(N, device=self.device))

    def _gg_sample(self, ids: torch.Tensor):
        pick = torch.randint(0, self._gg_pool_dof.shape[0], (ids.numel(),), device=self.device)
        self._gg_goal_dof[ids] = self._gg_pool_dof[pick]
        self._gg_goal_eng[ids] = self._gg_pool_eng[pick]

    def _get_rewards(self) -> torch.Tensor:
        total = super()._get_rewards()
        if not hasattr(self, "_gg_goal_dof"):
            return total
        cfg = self.cfg
        # joint-space L1 distance to the goal row (articulation order restricted to actuated, both sides)
        q = self.hand_dof_pos
        if q.shape[1] != 22:
            q = q[:, self.actuated_dof_indices]
        dist = (q - self._gg_goal_dof).abs().sum(-1)                              # (N,)
        r_j = float(cfg.gg_w_j) * torch.exp(-dist / float(cfg.gg_sigma_q))
        # per-finger contact-IDENTITY match
        eng = (self.last_contacts > float(cfg.gg_contact_thresh)).float()         # (N,5)
        match = 1.0 - (eng - self._gg_goal_eng).abs().mean(-1)                    # (N,) in [0,1]
        r_c = float(cfg.gg_w_c) * match
        # gates: settle over + held (backbone convention)
        if hasattr(self, "_gx_settle"):
            active = (self._gx_settle == 0).float()
        else:
            active = torch.ones(self.num_envs, device=self.device)
        disp = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        held = (disp < float(getattr(cfg, "hold_disp_gate", 0.10))).float()
        gate = held * active
        r_goal = (r_j + r_c) * gate
        # one-time success bonus, then resample a NEW goal for that env
        ok = (dist < float(cfg.gg_eps_q)) & (match > float(cfg.gg_match_min)) & (gate > 0.5)
        self._gg_streak = torch.where(ok, self._gg_streak + 1.0, torch.zeros_like(self._gg_streak))
        success = self._gg_streak >= float(cfg.gg_streak_need)
        r_bonus = success.float() * float(cfg.gg_bonus)
        if bool(success.any()):
            ids = success.nonzero(as_tuple=False).squeeze(-1)
            self._gg_reached[ids] += 1.0
            self._gg_streak[ids] = 0.0
            self._gg_sample(ids)
        # diagnostics
        self.extras["gg_dist_mean"] = dist.mean()
        self.extras["gg_match_mean"] = match.mean()
        self.extras["gg_reached_mean"] = self._gg_reached.mean()
        self.extras["gg_rj_mean"] = (r_j * gate).mean()
        self.extras["gg_rc_mean"] = (r_c * gate).mean()
        return total + r_goal + r_bonus

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_gg_goal_dof"):
            return
        ids = self.hand._ALL_INDICES if env_ids is None else env_ids
        ids_t = ids if torch.is_tensor(ids) else torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._gg_sample(ids_t)
        self._gg_streak[ids_t] = 0.0
        self._gg_reached[ids_t] = 0.0
