# GraspXL in-hand rotation env for the rebuild: runs the SharpaWave rebuild core (torque control +
# tactile + adaptive gravity curriculum + HORA reward) on GraspXL object(s), with two grasp-init modes
# (cfg.graspxl_init): "cache" (reset directly to settled grasps) and "replay" (replay the GraspXL approach
# trajectory over a settle window, then RL). Supports a SINGLE object (graspxl_object_id) or MULTIPLE
# per-env objects (graspxl_object_ids, replay mode). The grasp is transformed into the rebuild's fixed
# hand-init frame so the finger<->object geometry is preserved.
from __future__ import annotations

import torch
from collections.abc import Sequence

import math
import os

from .sharpa_wave_env import SharpaWaveInhandRotateEnv
from ...graspxl.sharpa_dataset import resolve_sharpa_grasps, SHARPA_USD_JOINT_NAMES
from ...graspxl.grasp import build_grasp_tensors, quat_apply, quat_mul, quat_conjugate, axis_angle_to_quat


def _load_object_poses(env, cfg, object_id, art_perm, hand_pos, hand_quat):
    """Return per-pose lists (joint_art (L,22), obj_envrel (L,7)) for one object, in the hand-init frame."""
    dataset_root = getattr(cfg, "graspxl_dataset_root", None)
    n_pose = int(getattr(cfg, "graspxl_num_poses", 3))
    joint_seqs, obj_seqs = [], []
    for pi in range(n_pose):
        try:
            g = resolve_sharpa_grasps({"enable": True, "dataset_root": dataset_root,
                                       "object_id": object_id, "object_ids": [], "pose_index": pi}, 1)
            gt = build_grasp_tensors(g, len(SHARPA_USD_JOINT_NAMES), env.device)
        except Exception:
            if pi == 0:
                raise
            break
        L = int(gt.length[0].item())
        jt_usd = gt.joint[0, :L]; obj = gt.object_pose[0, :L]; wr = gt.wrist_pose[0, :L]
        jt_art = torch.zeros(L, 22, device=env.device); jt_art[:, art_perm] = jt_usd
        wq, wp = wr[:, 3:7], wr[:, :3]
        rel_pos = quat_apply(quat_conjugate(wq), obj[:, :3] - wp)
        rel_quat = quat_mul(quat_conjugate(wq), obj[:, 3:7])
        hq = hand_quat.unsqueeze(0).expand(L, 4)
        obj_envrel = torch.cat([quat_apply(hq, rel_pos) + hand_pos, quat_mul(hq, rel_quat)], dim=-1)
        # GRASP-SQUEEZE (graspxl_squeeze>0): the imported GraspXL grasp is a fingertip PINCH that holds but
        # can't sustain rotation (-> one-time-rotation-then-drop once stage-4 perception lets the policy
        # attempt rotation). Continue the finger-closing past the settled pose (extrapolate the net approach
        # closing on the floating hand's finger joints) into a tighter, more ENCLOSING grip — the proven
        # SHARPA_JOINT_EXTRAPOLATE idea. The object is held kinematically during the extra squeeze frames.
        sq = float(getattr(cfg, "graspxl_squeeze", 0.0))
        if sq > 0.0 and jt_art.shape[0] >= 2:
            delta = jt_art[-1] - jt_art[0]                                   # net finger-closing direction
            K = int(getattr(cfg, "graspxl_squeeze_frames", 15))
            ramp = torch.linspace(0.0, sq, K, device=env.device).unsqueeze(-1)
            extra_j = jt_art[-1].unsqueeze(0) + ramp * delta.unsqueeze(0)    # (K,22) close further
            jt_art = torch.cat([jt_art, extra_j], dim=0)
            obj_envrel = torch.cat([obj_envrel, obj_envrel[-1:].expand(K, -1)], dim=0)
        joint_seqs.append(jt_art); obj_seqs.append(obj_envrel)
    return joint_seqs, obj_seqs


class SharpaWaveGraspXLEnv(SharpaWaveInhandRotateEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.gx_mode = getattr(cfg, "graspxl_init", "cache")
        self._gx_drop_margin = float(getattr(cfg, "graspxl_drop_margin", 0.08))
        self._gx_up_margin = float(getattr(cfg, "graspxl_up_margin", 0.05))
        object_ids = list(getattr(cfg, "graspxl_object_ids", []) or [])
        if not object_ids:
            object_ids = [getattr(cfg, "graspxl_object_id")]
        self._gx_n_obj = len(object_ids)

        art_perm = torch.tensor([self.hand.joint_names.index(n) for n in SHARPA_USD_JOINT_NAMES],
                                device=self.device, dtype=torch.long)
        hp = cfg.hand_init_pose
        hand_pos = torch.tensor(hp[0], device=self.device, dtype=torch.float32)
        hand_quat = torch.tensor(hp[1], device=self.device, dtype=torch.float32)
        per_obj = [_load_object_poses(self, cfg, oid, art_perm, hand_pos, hand_quat) for oid in object_ids]

        # STAGE-2/3 (PC + world-model), per the standing stage-4 directive: the base env built a CYLINDER
        # point cloud (_canon_obj_pts). For GraspXL/mesh objects sample the object's ACTUAL surface so the
        # PointNet branch + world-model head see the real geometry. Single object -> shared (P,3) canonical
        # points; per-env object scale is 1 (the mesh is already real-size).
        if getattr(self, "enable_pointcloud", False):
            from ...graspxl.sharpa_dataset import (load_sharpa_grasp, sample_object_surface_points,
                                                   DEFAULT_SHARPA_DATASET_ROOT)
            droot = getattr(cfg, "graspxl_dataset_root", None) or DEFAULT_SHARPA_DATASET_ROOT
            g0 = load_sharpa_grasp(dataset_root=droot, object_id=object_ids[0], pose_index=0)
            pts = sample_object_surface_points(str(g0.object_obj_path), int(self.cfg.pc_num_object), seed=0)
            self._canon_obj_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            if self._gx_n_obj > 1:
                # MULTI-OBJECT PC (work #2, 2026-07-05): per-object canonical point sets, indexed by
                # each env's assigned object in _compute_pointcloud (guarded; single-object path
                # unchanged). This is what lets the PointNet branch SEE which geometry each env
                # holds — the mechanism for shape/size generalization.
                pts_all = [self._canon_obj_pts]
                for oid_ in object_ids[1:]:
                    gi = load_sharpa_grasp(dataset_root=droot, object_id=oid_, pose_index=0)
                    pi = sample_object_surface_points(str(gi.object_obj_path),
                                                      int(self.cfg.pc_num_object), seed=0)
                    pts_all.append(torch.tensor(pi, dtype=torch.float32, device=self.device))
                self._canon_obj_pts_multi = torch.stack(pts_all, dim=0)  # (No,P,3)
            print(f"[GraspXL] PC object points sampled from mesh: {self._canon_obj_pts.shape[0]} pts "
                  f"(obj {object_ids[0][:8]}); {self._gx_n_obj} object(s)")
        if getattr(self, "enable_pointcloud", False) or getattr(self, "enable_world_model", False):
            self._pc_obj_scale = torch.ones(self.num_envs, 1, 1, device=self.device)

        if self.gx_mode == "cache":
            assert self._gx_n_obj == 1, "cache mode supports a single object; use replay for multi-object"
            joint_seqs, obj_seqs = per_obj[0]
            K = int(getattr(cfg, "graspxl_cache_frames", 30))
            cache = torch.cat([torch.cat([jt[-min(K, jt.shape[0]):], ob[-min(K, ob.shape[0]):]], dim=-1)
                               for jt, ob in zip(joint_seqs, obj_seqs)], dim=0)
            self.saved_grasping_states = cache
            self.bucket_grasp = cache.shape[0] // self.cfg.scale_range[2]
            self.bucket_env = self.num_envs // self.cfg.scale_range[2]
            self.hand.data.default_joint_pos[:] = joint_seqs[0][-1].unsqueeze(0)
            print(f"[GraspXL] cache mode: {cache.shape[0]} settled grasps")
            return

        # ---- replay mode (single or multi object) ----
        No = self._gx_n_obj
        Pmax = max(len(js) for js, _ in per_obj)
        Lmax = max(jt.shape[0] for js, _ in per_obj for jt in js)
        self._gx_traj_joint = torch.zeros(No, Pmax, Lmax, 22, device=self.device)
        self._gx_traj_obj = torch.zeros(No, Pmax, Lmax, 7, device=self.device)
        self._gx_len = torch.ones(No, Pmax, device=self.device, dtype=torch.long)
        self._gx_zmax = torch.zeros(No, Pmax, device=self.device)
        self._gx_settled_z = torch.zeros(No, Pmax, device=self.device)
        self._gx_npose = torch.ones(No, device=self.device, dtype=torch.long)
        ref_joints = torch.zeros(No, 22, device=self.device)
        for oi, (js, osq) in enumerate(per_obj):
            self._gx_npose[oi] = len(js)
            ref_joints[oi] = js[0][-1]
            for pi, (jt, ob) in enumerate(zip(js, osq)):
                L = jt.shape[0]
                self._gx_traj_joint[oi, pi, :L] = jt; self._gx_traj_joint[oi, pi, L:] = jt[-1]
                self._gx_traj_obj[oi, pi, :L] = ob; self._gx_traj_obj[oi, pi, L:] = ob[-1]
                self._gx_len[oi, pi] = L
                self._gx_zmax[oi, pi] = ob[:L, 2].max()
                self._gx_settled_z[oi, pi] = ob[L - 1, 2]
        self._gx_obj_of_env = torch.arange(self.num_envs, device=self.device) % No  # spawn order: env i -> obj i%No
        self._gx_hold = int(getattr(cfg, "graspxl_hold_steps", 20))
        self._gx_settle_total = int(Lmax) + self._gx_hold
        self._gx_settle = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # DROP-IN RESETS (2026-07-07, user directive: NO engineered initial poses — "as long as it
        # can get the object, let it play"). gx_drop_frac of episodes spawn the object ABOVE the
        # palm with a random orientation; the hand holds its default open posture while the object
        # falls and rests (drop-settle window); seat/termination anchors re-bind to the ACTUAL rest
        # state at settle end. Fully procedural + object-agnostic (replaces dataset grasp poses).
        self._gx_drop_frac = float(getattr(cfg, "gx_drop_frac", 0.0))
        self._gx_is_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gx_drop_lift = float(getattr(cfg, "gx_drop_lift", 0.10))
        self._gx_drop_settle = int(getattr(cfg, "gx_drop_settle", 60))
        self._gx_drop_anchor = torch.tensor(getattr(cfg, "gx_drop_anchor", (-0.0954, 0.0017, 0.597)),
                                            device=self.device, dtype=torch.float32)
        self._gx_pose_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # per-env pos_diff reference = its object's settled grasp
        self.hand.data.default_joint_pos[:] = ref_joints[self._gx_obj_of_env]
        self.saved_grasping_states = None
        # STEP 3: SE(3) orientation randomization (rotate the whole hand+object rigidly so the palm faces
        # different ways; the grasp is preserved). Drop is then displacement-based (world-z is meaningless).
        self._gx_orient = bool(getattr(cfg, "graspxl_orient_rand", False))
        self._gx_orient_max = float(getattr(cfg, "graspxl_orient_max_angle", math.pi))
        # ANNEALED orientation-range curriculum: start near palm-up (small tilt cap) so the adaptive
        # gravity curriculum can reach FULL g, THEN expand the sampled-orientation cap toward
        # _gx_orient_max (drop-rate-gated, in _orient_dones). Fixes the baseline stall at ~52% g where
        # full SO(3) from step 1 kept the drop rate high and stalled the gravity curriculum.
        self._gx_orient_curr = bool(getattr(cfg, "graspxl_orient_curriculum", False))
        self._gx_orient_step = float(getattr(cfg, "graspxl_orient_step", 0.15))
        # expand the orientation cap only while rotation is MAINTAINED (mean rotate_reward above this),
        # so the cap can't race ahead of the gait into a holding/shaking collapse.
        self._gx_orient_rotate_gate = float(getattr(cfg, "graspxl_orient_rotate_gate", 0.0))
        self._gx_orient_cur = (float(getattr(cfg, "graspxl_orient_start", 0.3))
                               if self._gx_orient_curr else self._gx_orient_max)
        # EVAL/RENDER override: the curriculum state _gx_orient_cur is NOT in the checkpoint, so a fresh
        # eval/render env would reset it to orient_start (palm-up only). Set SHARPA_ORIENT_CAP (rad) to
        # force the sampled-orientation cap (e.g. =pi for full SO(3)).
        _cap = os.environ.get("SHARPA_ORIENT_CAP")
        if _cap is not None:
            self._gx_orient_cur = float(_cap)
            print(f"[GraspXL] SHARPA_ORIENT_CAP override -> orient_cur={self._gx_orient_cur:.4f} rad")
        self._gx_drop_disp = float(getattr(cfg, "graspxl_drop_disp", 0.10))
        # curriculum drop-rate gate: advance gravity/orient when the active drop rate is below this.
        # 5e-4 (default) demands near-perfect holding -> it STALLS when the policy tries to ROTATE (a few
        # rotation-induced drops keep the rate above it). A looser value lets rotation + the curriculum
        # coexist (still well within the held>=0.9 target).
        self._gx_curr_drop_thresh = float(getattr(cfg, "graspxl_curr_drop_thresh", 5e-4))
        self._gx_hand_pos = hand_pos.clone()
        self._gx_hand_quat = hand_quat.clone()
        self._gx_base_axis = torch.tensor(cfg.rot_axis, device=self.device, dtype=torch.float32)
        self._gx_qrand = torch.zeros(self.num_envs, 4, device=self.device); self._gx_qrand[:, 0] = 1.0
        print(f"[GraspXL] replay mode: {No} object(s), Pmax={Pmax}, Lmax={Lmax}, "
              f"settle_total={self._gx_settle_total}, orient_rand={self._gx_orient}")

    def _sample_qrand(self, n):
        axis = torch.randn(n, 3, device=self.device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        ang = torch.rand(n, device=self.device) * self._gx_orient_cur  # current (annealed) cap
        return axis_angle_to_quat(axis * ang.unsqueeze(-1))

    def _get_observations(self):
        obs = super()._get_observations()
        # SO(3) FIX: the proprio+tactile obs is egocentric -> identical at every palm orientation EXCEPT
        # gravity pulls in a different (unobserved) direction. Append the gravity vector in the HAND frame
        # (direction + magnitude relative to full g) so the policy can adapt its gait to any orientation.
        if getattr(self.cfg, "obs_hand_gravity", False):
            g = self.physics_sim_view.get_gravity()
            gvec = torch.tensor([g[0], g[1], g[2]], device=self.device, dtype=torch.float32) / 9.81
            g_hand = quat_apply(quat_conjugate(self.hand.data.root_quat_w),
                                gvec.unsqueeze(0).expand(self.num_envs, 3))
            obs["policy"] = torch.cat([obs["policy"], g_hand], dim=-1)
        return obs

    # ---------------- replay-mode (Path B) ----------------
    def _gx_frame(self):
        elapsed = (self._gx_settle_total - self._gx_settle).clamp(min=0)
        Lp = self._gx_len[self._gx_obj_of_env, self._gx_pose_id]
        return torch.minimum(elapsed, (Lp - 1).clamp(min=0))

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if self.gx_mode != "replay":
            return
        settling = self._gx_settle > 0
        if bool(settling.any()):
            frame = self._gx_frame()
            traj_j = self._gx_traj_joint[self._gx_obj_of_env, self._gx_pose_id, frame]  # (N,22)
            if self._gx_drop_frac > 0.0:
                traj_j = torch.where(self._gx_is_drop.unsqueeze(-1),
                                     self.hand.data.default_joint_pos, traj_j)
            self.cur_targets = torch.where(settling.unsqueeze(-1), traj_j, self.cur_targets)

    def _apply_action(self):
        super()._apply_action()
        if self.gx_mode != "replay":
            return
        settling = self._gx_settle > 0
        if self._gx_drop_frac > 0.0:
            settling = settling & (~self._gx_is_drop)
        if bool(settling.any()):
            ids = settling.nonzero(as_tuple=False).squeeze(-1)
            frame = self._gx_frame()[ids]
            obj = self._gx_traj_obj[self._gx_obj_of_env[ids], self._gx_pose_id[ids], frame]  # (K,7)
            pos, quat = self._gx_apply_qrand(obj[:, :3], obj[:, 3:7], self._gx_qrand[ids])
            state = torch.zeros(len(ids), 7, device=self.device)
            state[:, :3] = pos + self.scene.env_origins[ids]
            state[:, 3:7] = quat
            self.object.write_root_pose_to_sim(state, ids)
            self.object.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), ids)

    def _gx_apply_qrand(self, pos, quat, q):
        """Rigidly rotate an object pose (pos,quat) by q about the (fixed) hand-root position."""
        hp = self._gx_hand_pos.unsqueeze(0)
        return hp + quat_apply(q, pos - hp), quat_mul(q, quat)

    def _get_dones(self):
        if self.gx_mode == "replay":
            prev_settle = self._gx_settle
            self._gx_settle = torch.clamp(self._gx_settle - 1, min=0)
            if self._gx_drop_frac > 0.0:
                just_settled = (prev_settle == 1) & self._gx_is_drop
                if bool(just_settled.any()):
                    jid = just_settled.nonzero(as_tuple=False).squeeze(-1)
                    self._refresh_lab()
                    self.object_default_pose[jid, :3] = self.object_pos[jid]
                    self.object_default_pose[jid, 3:7] = self.object_rot[jid]
                    z = self.object_pos[jid, 2]
                    self.reset_height_upper[jid] = z + self._gx_up_margin + 0.10
                    self.reset_height_lower[jid] = z - self._gx_drop_margin
        if self.gx_mode == "replay" and self._gx_orient:
            return self._orient_dones()
        return super()._get_dones()

    def _orient_dones(self):
        """Orientation-invariant drop: object displaced from its (rotated) held pose by > drop_disp.
        Drives the same adaptive gravity curriculum on the displacement-drop rate."""
        import carb
        self._refresh_lab()
        disp = torch.norm(self.object_pos - self.object_default_pose[:, :3], dim=-1)
        settling = self._gx_settle > 0
        dropped = (disp > self._gx_drop_disp) & (~settling)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        active = (~settling).float()
        drop_rate = (dropped.float().sum() / active.sum().clamp_min(1.0))
        self.extras['height_reset_lower'] = drop_rate
        self.extras['height_reset_upper'] = torch.zeros((), device=self.device)
        self.extras['time_out'] = time_out.float().mean()
        self.extras['drop_disp_rate'] = drop_rate
        self.extras['orient_cur'] = torch.tensor(float(self._gx_orient_cur), device=self.device)
        # advance the curriculum (gravity magnitude, then orientation range) only while the policy both
        # HOLDS (low drop) AND keeps ROTATING (rotate_reward above the gate). Gating gravity on rotation
        # too is essential: otherwise gravity races to full while the policy quietly stops rotating, and
        # the warm-started gait then faces full-g + tilt all at once and collapses to holding.
        rotate_ok = float(self.extras.get("rotate_reward", 0.0)) > self._gx_orient_rotate_gate
        if drop_rate < self._gx_curr_drop_thresh and rotate_ok and self.cfg.gravity_curriculum and self.common_step_counter > 1000:
            g = self.physics_sim_view.get_gravity()
            amp = torch.sqrt(torch.tensor(g[0] ** 2 + g[1] ** 2 + g[2] ** 2))
            if amp < 10:
                self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -float(amp) - 0.05))
                print(f"update gravity: (0,0,{-float(amp) - 0.05:.4f}) rotate={float(self.extras.get('rotate_reward',0.0)):.3f}")
            elif (self._gx_orient_curr and self._gx_orient_cur < self._gx_orient_max - 1e-6
                  and float(self.extras.get("rotate_reward", 0.0)) > self._gx_orient_rotate_gate):
                # gravity is full -> expand the orientation cap, but ONLY while rotation is still
                # MAINTAINED (rotate_reward above the gate). Otherwise the cap raced ahead of the gait and
                # the policy quietly collapsed to holding/shaking. This keeps sustained rotation as SO(3) grows.
                self._gx_orient_cur = min(self._gx_orient_max, self._gx_orient_cur + self._gx_orient_step)
                print(f"update orient_cap: {self._gx_orient_cur:.4f} / {self._gx_orient_max:.4f} rad "
                      f"(rotate={float(self.extras.get('rotate_reward', 0.0)):.3f})")
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if self.gx_mode != "replay":
            super()._reset_idx(env_ids)
            ids = self.hand._ALL_INDICES if env_ids is None else env_ids
            self.object_default_pose[ids, :3] = (self.object.data.root_pos_w - self.scene.env_origins)[ids]
            return
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super(SharpaWaveInhandRotateEnv, self)._reset_idx(env_ids)  # DirectRLEnv framework reset
        n = len(env_ids)
        if self.cfg.randomize_pd_gains:
            rs = self._rand_pd_scales(self.cfg.randomize_p_gain_scale_lower, self.cfg.randomize_p_gain_scale_upper, n, self.num_hand_dofs)
            self.p_gain[env_ids] = self.p_gain_default[env_ids] * rs
            rs = self._rand_pd_scales(self.cfg.randomize_d_gain_scale_lower, self.cfg.randomize_d_gain_scale_upper, n, self.num_hand_dofs)
            self.d_gain[env_ids] = self.d_gain_default[env_ids] * rs
        oid = self._gx_obj_of_env[env_ids]
        npose = self._gx_npose[oid]
        pose = (torch.rand(n, device=self.device) * npose.float()).long().clamp_max_(npose - 1)
        self._gx_pose_id[env_ids] = pose
        self._gx_settle[env_ids] = self._gx_settle_total
        Lp = self._gx_len[oid, pose]
        obj0 = self._gx_traj_obj[oid, pose, 0].clone()
        jt0 = self._gx_traj_joint[oid, pose, 0]
        settled = self._gx_traj_obj[oid, pose, Lp - 1].clone()
        # SE(3) orientation randomization: rotate hand+object rigidly about the hand root (grasp preserved)
        q = self._sample_qrand(n) if self._gx_orient else None
        if q is not None:
            self._gx_qrand[env_ids] = q
            obj0[:, :3], obj0[:, 3:7] = self._gx_apply_qrand(obj0[:, :3], obj0[:, 3:7], q)
            settled[:, :3], settled[:, 3:7] = self._gx_apply_qrand(settled[:, :3], settled[:, 3:7], q)
            self.rot_axis[env_ids] = quat_apply(q, self._gx_base_axis.unsqueeze(0).expand(n, 3))
        else:
            self._gx_qrand[env_ids, 0] = 1.0; self._gx_qrand[env_ids, 1:] = 0.0
        ostate = torch.zeros(n, 13, device=self.device)
        ostate[:, :3] = obj0[:, :3] + self.scene.env_origins[env_ids]
        ostate[:, 3:7] = obj0[:, 3:7]
        self.object.write_root_pose_to_sim(ostate[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(ostate[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0
        if self._gx_orient:  # world-z bands unused (displacement drop) -> set generous
            self.reset_height_upper[env_ids] = settled[:, 2] + 0.30
            self.reset_height_lower[env_ids] = settled[:, 2] - 0.30
        else:
            self.reset_height_upper[env_ids] = self._gx_zmax[oid, pose] + self._gx_up_margin
            self.reset_height_lower[env_ids] = self._gx_settled_z[oid, pose] - self._gx_drop_margin
        self.object_default_pose[env_ids, :3] = settled[:, :3]
        self.object_default_pose[env_ids, 3:7] = settled[:, 3:7]
        if self._gx_drop_frac > 0.0:
            dm = torch.rand(n, device=self.device) < self._gx_drop_frac
            self._gx_is_drop[env_ids] = dm
            if bool(dm.any()):
                dids = env_ids[dm] if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=self.device)[dm]
                k = len(dids)
                dq = torch.randn(k, 4, device=self.device)
                dq = dq / dq.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                dstate = torch.zeros(k, 13, device=self.device)
                dstate[:, :3] = (self._gx_drop_anchor.unsqueeze(0)
                                 + torch.tensor([0.0, 0.0, self._gx_drop_lift], device=self.device)
                                 + torch.cat([0.02 * (torch.rand(k, 2, device=self.device) - 0.5),
                                              torch.zeros(k, 1, device=self.device)], dim=-1)
                                 + self.scene.env_origins[dids])
                dstate[:, 3:7] = dq
                self.object.write_root_pose_to_sim(dstate[:, :7], dids)
                self.object.write_root_velocity_to_sim(dstate[:, 7:], dids)
                djp = self.hand.data.default_joint_pos[dids]
                self.hand.write_joint_state_to_sim(djp, torch.zeros_like(djp), env_ids=dids)
                self.cur_targets[dids] = self.hand.data.default_joint_pos[dids]
                self.prev_targets[dids] = self.hand.data.default_joint_pos[dids]
                self._gx_settle[dids] = self._gx_drop_settle
                # provisional generous bands until settle-end re-anchor
                self.reset_height_upper[dids] = self._gx_drop_anchor[2] + 0.50
                self.reset_height_lower[dids] = self._gx_drop_anchor[2] - 0.12
        hand_default_state = self.hand.data.default_root_state.clone()[env_ids]
        hand_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        if q is not None:
            hand_default_state[:, 3:7] = quat_mul(q, hand_default_state[:, 3:7])
        self.hand.write_root_state_to_sim(hand_default_state, env_ids)
        self.prev_targets[env_ids] = jt0
        self.cur_targets[env_ids] = jt0
        self.hand.set_joint_position_target(jt0, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(jt0, torch.zeros_like(jt0), env_ids=env_ids)
        self._refresh_lab()
        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
