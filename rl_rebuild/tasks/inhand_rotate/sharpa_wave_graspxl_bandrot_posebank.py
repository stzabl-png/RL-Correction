# CLOSED-LOOP (2026-07-03): BandRot ball rotation initialized from an ARBITRARY pose cache selected
# at launch via the SHARPA_POSE_CACHE env var (path to a (N,29) cache). Reuses the full BandRot
# backbone (band rotation reward, Diverse 100%-cache reset). ADDITIVE ONLY.
from __future__ import annotations

import os

from isaaclab.utils import configclass

from .sharpa_wave_graspxl_bandrot import SharpaWaveGraspXLBandRotCfg, SharpaWaveGraspXLBandRotEnv


@configclass
class SharpaWaveGraspXLBandRotPoseBankCfg(SharpaWaveGraspXLBandRotCfg):
    """BandRot sphere cfg whose reset cache is chosen by SHARPA_POSE_CACHE at launch."""

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        cache = os.environ.get("SHARPA_POSE_CACHE", "").strip()
        if cache:
            self.enclosing_cache_path = cache
            print(f"[posebank] rotation resets from: {cache}")
        if os.environ.get("SHARPA_BAND_HELD", "0").strip() == "1":
            self.band_held_gate = True
            print("[posebank] band reward HELD-GATED (escape-spin loophole closed)")
        n4b = os.environ.get("SHARPA_BAND_N4", "").strip()
        if n4b:
            self.band_n4_boost = float(n4b)
            self.band_lown_scale = float(os.environ.get("SHARPA_BAND_LOWN", "0.3"))
            print(f"[posebank] band N4-BOOST: x{self.band_n4_boost} at n>=4, "
                  f"x{self.band_lown_scale} below (braking-curve-informed)")
        nsc = os.environ.get("SHARPA_BAND_NSCALE", "").strip()
        if nsc:
            self.band_n_scale_div = float(nsc)
            print(f"[posebank] band CONTINUOUS n-scaling: r_rot x n_eng/{self.band_n_scale_div} "
                  f"(no camping ledge)")
        hinc = os.environ.get("SHARPA_BAND_HOLD_INC", "").strip()
        if hinc:
            self.band_hold_income = float(hinc)
            print(f"[posebank] hold-persistence income: +{self.band_hold_income}/step while held")
        bfl = os.environ.get("SHARPA_BAND_FLOOR", "").strip()
        if bfl:
            # dead-zone fix (2026-07-03 eve): camped policies creep at 0.03-0.09 rad/s where the
            # default 0.15 floor pays zero — no gradient from creep to drive. 0.06 keeps the
            # measured 0.037 jitter attractor dead while restoring the slope.
            self.band_omega_floor = float(bfl)
            print(f"[posebank] band omega floor overridden: {self.band_omega_floor} rad/s")
        bsc = os.environ.get("SHARPA_BAND_SCALE", "").strip()
        if bsc:
            self.band_scale = float(bsc)
            print(f"[posebank] band scale overridden: {self.band_scale}")
        # LARGE-OBJECT ACCOMMODATION (2026-07-06, user directive: adjustment for big objects may
        # settle toward the palm / lower — do NOT structurally forbid it). The three ball-anchored
        # geometry gates become tunable; palm CONTACT itself was never penalized (fingertip-only
        # contact counting), so loosening these radii is sufficient to permit palm-supported grasps.
        hd = os.environ.get("SHARPA_HELD_DISP", "").strip()
        if hd:
            self.band_held_disp = float(hd)
            print(f"[posebank] held-gate displacement radius: {self.band_held_disp} m")
        cs = os.environ.get("SHARPA_CENTER_SIGMA", "").strip()
        if cs:
            self.center_reward_sigma = float(cs)
            print(f"[posebank] centering sigma: {self.center_reward_sigma} m")
        dm = os.environ.get("SHARPA_DROP_MARGIN", "").strip()
        if dm:
            self.graspxl_drop_margin = float(dm)
            print(f"[posebank] drop margin (termination below settled z): {self.graspxl_drop_margin} m")
        oids = os.environ.get("SHARPA_OBJ_IDS", "").strip()
        if oids:
            # MULTI-OBJECT TRAINING (work #2): comma-separated dataset ids; per-env object cycling
            # (env i -> obj i%No, base graspxl replay machinery). Pinch-replay resets only.
            from rl_rebuild.graspxl.object_cfg import (make_graspxl_multi_object_cfg, object_usd_for)
            ids = [s.strip() for s in oids.split(",") if s.strip()]
            self.graspxl_object_ids = ids
            self.object_cfg = make_graspxl_multi_object_cfg([object_usd_for(o) for o in ids])
            self.diverse_enclosing_frac = 0.0
            print(f"[posebank] MULTI-OBJECT: {len(ids)} objects, pinch-replay resets")
        pcen = os.environ.get("SHARPA_PC", "").strip()
        if pcen == "1":
            self.enable_pointcloud = True
            print("[posebank] POINTCLOUD obs ENABLED (PointNet branch must be on in the agent cfg)")
        oid = os.environ.get("SHARPA_OBJ_ID", "").strip()
        if oid:
            # OBJECT GENERALIZATION (work #2, 2026-07-05): swap the manipulated object by dataset
            # id. Resets fall back to the object's OWN pinch replay (the ball's enclosing/pose
            # caches are object-specific) unless SHARPA_POSE_CACHE explicitly provides one.
            # Mass stays OBJECT_MASS (0.10 kg) across objects to isolate geometry effects.
            from rl_rebuild.graspxl.object_cfg import make_graspxl_object_cfg, object_usd_for
            self.graspxl_object_id = oid
            self.object_cfg = make_graspxl_object_cfg(object_usd_for(oid))
            if not os.environ.get("SHARPA_POSE_CACHE", "").strip():
                self.diverse_enclosing_frac = 0.0
            print(f"[posebank] OBJECT override: {oid} "
                  f"(enclosing_frac={self.diverse_enclosing_frac})")
        enf = os.environ.get("SHARPA_ENC_FRAC", "").strip()
        if enf:
            # ADJUSTMENT+ROTATION integration (2026-07-05, the program's true track): fraction of
            # episodes resetting from the adjusted-pose cache; the REST start from the raw PINCH
            # replay and must adjust into support before rotation income flows (Q1 unified design,
            # now with the proven band-pass+burst-pay rotation income).
            self.diverse_enclosing_frac = float(enf)
            print(f"[posebank] MIXED resets: {self.diverse_enclosing_frac:.2f} adjusted-pose / "
                  f"{1-self.diverse_enclosing_frac:.2f} pinch (adjust->rotate in-episode)")
        gz = os.environ.get("SHARPA_GRAVITY_Z", "").strip()
        if gz:
            # low-gravity decisive test (2026-07-05): train/eval at FIXED low g. If rotation
            # sustains when holding is nearly free, the full-g gap is drop-prevention details;
            # if not, the bottleneck is structural (gait coordination / reward exhaustion).
            self.sim.gravity = (0.0, 0.0, float(gz))
            self.gravity_curriculum = False
            print(f"[posebank] FIXED gravity override: {self.sim.gravity}")
        eps = os.environ.get("SHARPA_EPISODE_S", "").strip()
        if eps:
            self.episode_length_s = float(eps)
            print(f"[posebank] episode length override: {self.episode_length_s}s")
        fsc = os.environ.get("SHARPA_FORCE_SCALE", "").strip()
        if fsc:
            # robust-margin arm (2026-07-04): the cum-axis failure is the ball ESCAPING during fast
            # rotation (height_reset_lower 1.4%/step). Train under the base env's HORA random object
            # forces (mass-scaled, decaying) so the policy learns grip margin — the same mechanism
            # that fixed regrip holding (perturbation curriculum). Probes stay clean (force off).
            self.force_scale = float(fsc)
            self.random_force_prob_scalar = float(os.environ.get("SHARPA_FORCE_PROB", "0.2"))
            print(f"[posebank] object-force perturbation ON: force_scale={self.force_scale}, "
                  f"prob={self.random_force_prob_scalar}")
        bem = os.environ.get("SHARPA_BAND_EMA", "").strip()
        if bem:
            # burst-pay fix (2026-07-04): alpha 0.9 delays band onset ~10 steps into each drive
            # burst — short recruit->drive->re-park bursts (the champion's cadence) go underpaid.
            self.ema_alpha = float(bem)
            print(f"[posebank] band EMA alpha overridden: {self.ema_alpha} (faster burst credit)")
        sth = os.environ.get("SHARPA_SPARSE_THETA", "").strip()
        if sth:
            self.sparse_theta = float(sth)
            sbn = os.environ.get("SHARPA_SPARSE_BONUS", "").strip()
            if sbn:
                self.sparse_bonus = float(sbn)
            print(f"[posebank] sparse ratchet overridden: theta={self.sparse_theta}, "
                  f"bonus={getattr(self, 'sparse_bonus', 5.0)} (finer per-burst credit)")
        dfr = os.environ.get("SHARPA_DROP_FRAC", "").strip()
        if dfr:
            # NO-ENGINEERED-POSES directive (user 2026-07-07): fraction of episodes spawn the
            # object ABOVE the palm (random orientation, free fall, settle, re-anchor) — fully
            # procedural acquisition; the policy must construct its own grasp then rotate.
            self.gx_drop_frac = float(dfr)
            lf = os.environ.get("SHARPA_DROP_LIFT", "").strip()
            if lf:
                self.gx_drop_lift = float(lf)
            self.diverse_enclosing_frac = 0.0 if self.gx_drop_frac >= 1.0 else self.diverse_enclosing_frac
            print(f"[posebank] DROP-IN resets: frac={self.gx_drop_frac} "
                  f"(procedural palm-drop, no engineered poses)")
        rax = os.environ.get("SHARPA_ROT_AXIS", "").strip()
        if rax:
            # PEN-LIKE OBJECTS (2026-07-07): a pen lies horizontally in a palm-up hand; rolling it
            # about its LONG axis needs the band projected onto a horizontal axis, not world-z.
            # Format "x,y,z" (normalized here). Per-env axis conditioning is the SO(3) plan Step 1;
            # this global knob is its minimal precursor.
            ax = [float(v) for v in rax.split(",")]
            n = (ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) ** 0.5
            self.rot_axis = (ax[0] / n, ax[1] / n, ax[2] / n)
            print(f"[posebank] rotation axis override: {self.rot_axis}")
        nps = os.environ.get("SHARPA_NUM_POSES", "").strip()
        if nps:
            self.graspxl_num_poses = int(nps)
            print(f"[posebank] DIVERSE START POSES: {self.graspxl_num_poses}/object (adjustment scaffold; "
                  f"default was only 3 — more varied starts = more adjustment practice)")
        dp = os.environ.get("SHARPA_DROP_PENALTY", "").strip()
        if dp:
            self.drop_penalty = float(dp)
            print(f"[posebank] DROP PENALTY: {self.drop_penalty} (terminal; minimal feasibility signal)")
        bse = os.environ.get("SHARPA_BAND_SCALE_END", "").strip()
        if bse:
            self.band_scale_end = float(bse)
            self.band_scaffold_anneal = float(os.environ.get("SHARPA_SCAFFOLD_ANNEAL", "40000"))
            print(f"[posebank] SCAFFOLD ANNEAL: band_scale -> {self.band_scale_end} & n-scaling -> flat "
                  f"over {self.band_scaffold_anneal:.0f} env-steps (converged obj = net-progress + drop)")
        npw = os.environ.get("SHARPA_NETPROG_W", "").strip()
        if npw:
            self.band_netprog_w = float(npw)
            self.band_netprog_margin = float(os.environ.get("SHARPA_NETPROG_MARGIN", "0.10"))
            print(f"[posebank] HIGH-WATER net-progress income: w={self.band_netprog_w} "
                  f"(treadmill-proof + recovery-safe; diagnostic-informed 2026-07-07)")
        bsw = os.environ.get("SHARPA_BACKSLIP_W", "").strip()
        if bsw:
            self.band_backslip_w = float(bsw)
            print(f"[posebank] ASYMMETRIC backslip cost: unheld backward rotation weighted "
                  f"{self.band_backslip_w} (forward still pays only while held)")
        rst = os.environ.get("SHARPA_BAND_ROTSTREAK", "").strip()
        if rst:
            self.band_rotstreak_scale = float(rst)
            print(f"[posebank] rotation-STREAK bonus: up to +{self.band_rotstreak_scale}/step after "
                  f"{float(os.environ.get('SHARPA_BAND_ROTSTREAK_CAP', '40'))} consecutive held-rotating steps")
            cap_env = os.environ.get("SHARPA_BAND_ROTSTREAK_CAP", "").strip()
            if cap_env:
                self.band_rotstreak_cap = float(cap_env)
        bcl = os.environ.get("SHARPA_BAND_CEIL", "").strip()
        if bcl:
            self.band_omega_ceil = float(bcl)
            print(f"[posebank] band-PASS ceiling: pay tapers to 0 above {self.band_omega_ceil} rad/s "
                  f"(anti-fling); ratchet integration clamped to it")
            cst = os.environ.get("SHARPA_BAND_CEIL_START", "").strip()
            can = os.environ.get("SHARPA_BAND_CEIL_ANNEAL", "").strip()
            if cst and can:
                self.band_omega_ceil_start = float(cst)
                self.band_omega_ceil_anneal = float(can)
                print(f"[posebank] ceiling ANNEALS {self.band_omega_ceil_start} -> "
                      f"{self.band_omega_ceil} over {self.band_omega_ceil_anneal:.0f} env-steps")
        csc = os.environ.get("SHARPA_CENTER_SCALE", "").strip()
        if csc:
            # reward-arithmetic fix (2026-07-03 night): bounded centering pays 1.0/step for a
            # centered STATIC hold — 3x the band at learnable omega. The camp was the correctly
            # priced optimum all along; rebalance so rotation out-pays holding at the margin.
            self.object_pos_reward_scale = float(csc)
            print(f"[posebank] centering (object_pos) scale overridden: {self.object_pos_reward_scale}")


class SharpaWaveGraspXLBandRotPoseBankEnv(SharpaWaveGraspXLBandRotEnv):
    pass


# Round-2 rotation variant: band reward + the G1 re-contact GAIT term (the cylinder-winning combo),
# resets from SHARPA_POSE_CACHE. Tests whether gait incentives + spread regrip inits crack the ball.
from .sharpa_wave_graspxl_bandgait import _BandGaitMixin, SharpaWaveGraspXLBandGaitSphereG1Cfg


@configclass
class SharpaWaveGraspXLBandGaitG1PoseBankCfg(SharpaWaveGraspXLBandGaitSphereG1Cfg):
    """BandGait-G1 sphere cfg whose reset cache is chosen by SHARPA_POSE_CACHE at launch."""

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        cache = os.environ.get("SHARPA_POSE_CACHE", "").strip()
        if cache:
            self.enclosing_cache_path = cache
            print(f"[posebank-g1] rotation resets from: {cache}")
        if os.environ.get("SHARPA_BAND_HELD", "0").strip() == "1":
            self.band_held_gate = True
            print("[posebank-g1] band reward HELD-GATED")


from .sharpa_wave_graspxl_bandgait import SharpaWaveGraspXLBandGaitSphereEnv


class SharpaWaveGraspXLBandGaitG1PoseBankEnv(SharpaWaveGraspXLBandGaitSphereEnv):
    pass


# 2026-07-03 AM (user): (1) held-gated band on the CYLINDER (validate the corrected reward on the
# known-rotatable object); (2) COMBINED regrip(R4RL)+rotation(held band) on the ball from the pinch —
# the first true joint adjust+rotate training (watch: does regrip income crowd out rotation?).
from .sharpa_wave_graspxl_bandrot import SharpaWaveBandRotCylinderCfg, SharpaWaveBandRotCylinderEnv


@configclass
class SharpaWaveBandRotCylinderHeldCfg(SharpaWaveBandRotCylinderCfg):
    """Cylinder band rotation with the escape-spin loophole closed (held-gated band + integration)."""
    band_held_gate = True
    band_held_min_fingers = 2
    band_held_disp = 0.05
    band_held_thresh = 0.15


class SharpaWaveBandRotCylinderHeldEnv(SharpaWaveBandRotCylinderEnv):
    pass
