import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-v0",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-PC-v0",   # STAGE 2: + point cloud + masks
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfgPC",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-WM-v0",   # STAGE 3: + point cloud + world-model loss
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfgWM",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Cache-v0",   # GraspXL object, reset from settled grasp
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLCacheCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-v0",   # GraspXL object, trajectory-replay init
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLReplayCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-PCWM-v0",  # STAGE-4 design (PC+world-model) on GraspXL, single obj
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLReplayPCWMCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-PCWM-Squeeze-v0",  # STAGE-4 PC+WM + grasp-squeeze (fix one-time rotation)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLReplayPCWMSqueezeCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Orient-PCWM-v0",  # STAGE-4 design + SE(3) orientation init, single obj
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientPCWMCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Orient-v0",   # STEP 3: SE(3) orientation randomization
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientCurr-v0",   # STEP 3b: orient rand + annealed orientation-range curriculum
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientCurrCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientCurrFT-v0",  # STEP 3c: reward-rebalanced fine-tune (sustained rotation)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientCurrFTCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientScratch-v0",  # STEP 3d: from-scratch, rotation-dominant reward
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientScratchCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-OrientGrav-v0",  # SO(3): official reward + hand-frame gravity obs
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLOrientGravCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Sustained-v0",  # SO(3) sustained: rotate + bounded-center anti-drop
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLSustainedCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustRotate-v0",  # adjust-then-rotate: readiness-potential reward, no pose penalty
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate:SharpaWaveGraspXLAdjustRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustrotate:SharpaWaveGraspXLAdjustRotateCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustRotate-Diverse-v0",  # 3B: adjust-rotate + friction max + diverse pinch/enclosing resets
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate_diverse:SharpaWaveGraspXLAdjustRotateDiverseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustrotate_diverse:SharpaWaveGraspXLAdjustRotateDiverseCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustRotate-Diverse-FixedG-v0",  # 3B FIXED-G: diverse adjust-rotate at fixed -4.9, gravity curriculum OFF
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate_diverse:SharpaWaveGraspXLAdjustRotateDiverseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustrotate_diverse:SharpaWaveGraspXLAdjustRotateDiverseFixedGCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustOnly-FixedG-v0",  # ADJUSTMENT-ONLY: readiness-Phi reward, NO rotation reward (rotate_scale=0), fixed g -4.9, pinch reset
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate:SharpaWaveGraspXLAdjustRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustonly:SharpaWaveGraspXLAdjustOnlyFixedGCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustOnly-Curr-v0",  # ADJUSTMENT-ONLY + drop-gated gravity CURRICULUM (rotation gate disabled), pinch reset
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate:SharpaWaveGraspXLAdjustRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustonly:SharpaWaveGraspXLAdjustOnlyCurrCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustOnly-Curr-v2",  # v2: multiplicative Phi (coverage gates spread) -> supportive multi-finger regrip
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjustrotate:SharpaWaveGraspXLAdjustRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjustonly:SharpaWaveGraspXLAdjustOnlyCurrV2Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustHold-Persistent-v3a",  # v3a: NON-potential hold+coverage reward (fixes cradle/palm-rest failure of PBRS)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjusthold:SharpaWaveGraspXLAdjustHoldPersistentEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjusthold:SharpaWaveGraspXLAdjustHoldPersistentCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-AdjustHold-Perturb-v1",  # regrip-HOLD via adversarial perturbation curriculum (forces a FINGERTIP grasp; palm-rest gets knocked off)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_adjusthold_perturb:SharpaWaveGraspXLAdjustHoldPerturbEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_adjusthold_perturb:SharpaWaveGraspXLAdjustHoldPerturbCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-Cylinder-v1",  # anti-jitter BAND reward on the CYLINDER
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot:SharpaWaveBandRotCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot:SharpaWaveBandRotCylinderCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-SphereEnc-v1",  # anti-jitter BAND reward on GraspXL ball, enclosing-cache init
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot:SharpaWaveGraspXLBandRotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot:SharpaWaveGraspXLBandRotCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

# --- finger-GAITING reward sweep on the CYLINDER (band+sparse backbone + ONE gait term each) -------------
# All 5 share ONE env (SharpaWaveBandGaitCylinderEnv); the gait term is chosen by the cfg's gait_mode.
# They reset from the harvested MULTI-KEYFRAME cache (cache/cylinder_gait_keyframes_0.5-0.5-1.npy).
gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Cylinder-G1-v1",  # G1: release->re-contact (sustained)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderG1Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Cylinder-G2-v1",  # G2: >=3 fingers load-share
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderG2Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Cylinder-G3-v1",  # G3: contact-age release penalty
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderG3Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Cylinder-G4-v1",  # G4: rotate at joint mid-range
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderG4Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Cylinder-G5-v1",  # G5: post-release re-grip recovery
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:SharpaWaveBandGaitCylinderG5Cfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

# --- Ball (GraspXL sphere) gait variants: G1-G3 (the cylinder winners) on the ball, enclosing-cache reset ---
for _G, _cfg, _tag in [
    ("G1", "SharpaWaveGraspXLBandGaitSphereG1Cfg", "release->re-contact"),
    ("G2", "SharpaWaveGraspXLBandGaitSphereG2Cfg", ">=3 finger load-share"),
    ("G3", "SharpaWaveGraspXLBandGaitSphereG3Cfg", "contact-age release"),
]:
    gym.register(
        id=f"Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGait-Sphere-{_G}-v1",
        entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandgait:SharpaWaveGraspXLBandGaitSphereEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandgait:{_cfg}",
            "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
        },
    )

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Multi-v0",   # STEP 2: multiple GraspXL objects, replay init
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_env:SharpaWaveGraspXLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveGraspXLMultiCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-WM-Multi-v0",   # STAGE 4: pc + world model + multi-scale objects
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfgWMMulti",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_grasp_env:SharpaWaveInhandRotateGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_grasp_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-GraspXL-v0",   # ENCLOSING grasp gen (drop-and-settle) for the GraspXL ball
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_grasp_env:SharpaWaveGraspXLGenGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_grasp_env:SharpaWaveGraspXLGenGraspCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_deploy_env:SharpaWaveInhandRotateDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_deploy_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------------------------------
# REGRIP sweep (2026-07-01): 8 rotation-free regrip variants on the smooth ball, each trained with
# self-collision ON and OFF (SHARPA_SELF_COLLISION env var read by the cfgs). Design doc:
# robotics-rl-expert/notes/experiment_playbooks/regrip_contact_switch_designs_2026-07-01.md
# ---------------------------------------------------------------------------------------------------
gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-G1-Sphere-v1",  # R-G1: re-contact EVENT reward, de-rotationed
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripGaitSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripG1SphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-G2-Sphere-v1",  # R-G2: load-share reward, regrip task
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripGaitSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripG2SphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-G3-Sphere-v1",  # R-G3: contact-age penalty, regrip task
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripGaitSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_regrip_gait:SharpaWaveGraspXLRegripG3SphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D1-LoadShed-Sphere-v1",  # R-D1: forced-release finger dropout
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_loadshed:SharpaWaveGraspXLRegripD1LoadShedSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_loadshed:SharpaWaveGraspXLRegripD1LoadShedSphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D2-Handoff-Sphere-v1",  # R-D2: reset from harvested >=3-finger states
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_handoff:SharpaWaveGraspXLRegripD2HandoffSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_handoff:SharpaWaveGraspXLRegripD2HandoffSphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D3-GoalGrasp-Sphere-v1",  # R-D3: goal grasp (joints + contact identity)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_goalgrasp:SharpaWaveGraspXLRegripD3GoalGraspSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_goalgrasp:SharpaWaveGraspXLRegripD3GoalGraspSphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D4-RotGrav-Sphere-v1",  # R-D4: rotating low-point lateral load
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_rotgrav:SharpaWaveGraspXLRegripD4RotGravSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_rotgrav:SharpaWaveGraspXLRegripD4RotGravSphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Regrip-D5-RGTB-Sphere-v1",  # R-D5: release-token recruitment ratchet
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_rgtb:SharpaWaveGraspXLRegripD5RGTBSphereEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_rgtb:SharpaWaveGraspXLRegripD5RGTBSphereCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------------------------------
# REGRIP v2 sweep (2026-07-02): 5 load-bearing designs x {ball, cylinder}, rotation-free, sink-terminated.
# Design doc: robotics-rl-expert/notes/reward_design/regrip_v2_load_bearing_designs_2026-07-02.md
# ---------------------------------------------------------------------------------------------------
gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R1-Ledger-Ball-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R1BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R1BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R1-Ledger-Cyl-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R1CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R1CylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R2-Chain-Ball-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R2BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R2BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R2-Chain-Cyl-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R2CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R2CylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R3-Margin-Ball-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R3BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R3BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R3-Margin-Cyl-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R3CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R3CylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4-Shed-Ball-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4-Shed-Cyl-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4CylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R5-Entropy-Ball-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R5BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R5BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R5-Entropy-Cyl-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R5CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R5CylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R1-Ledger-Ball-Net-v1",  # v3net A/B arm (separate critic, priv 23)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R1BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R1BallNetCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4-Shed-Ball-Net-v1",  # v3net A/B arm (separate critic, priv 23)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4BallNetCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

# R4-family 4x2 sweep registrations (2026-07-02 PM), all v3net agent cfg
gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4-Shed-Cyl-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4CylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4CylNetCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4L-ShedLedger-Ball-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4LBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4LBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4L-ShedLedger-Cyl-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4LCylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4LCylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4R-ShedRecruit-Ball-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4RBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4RBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4R-ShedRecruit-Cyl-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4RCylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4RCylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4S-ShedAllSwitch-Ball-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4SBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4SBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4S-ShedAllSwitch-Cyl-Net-v1",
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4SCylEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4SCylCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4-Shed-Ball-Net-H16-v1",  # horizon-16 control for the r4_ball stall diagnosis
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4BallNetCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net_h16.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-v1",  # closed-loop: rotation from SHARPA_POSE_CACHE
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-Net-v1",  # exploration arm: same env, v3net agent (sigma floor 0.1 + entropy 1e-3) — loop5 ladder showed the rotate-held optimum exists but entropy-0 collapses before the finger-walk is sampled
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net_p8.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-NetX-v1",  # EXPLOIT arm (2026-07-04): sigma floor 0.02 + entropy 0 — consolidation resumes; the 6d basin churns away under sigma 0.1
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net_p8_exploit.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-PoseBank-NetPC-v1",  # OBJECT-GEN arm (2026-07-05, work #2): p8 + PointNet branch; pair with SHARPA_PC=1 (+ SHARPA_OBJ_IDS)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandRotPoseBankCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg_v3net_p8_pc.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4RL-Ball-v1",  # closed-loop iter-2 regrip (old-net agent cfg)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4RLBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4RLBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandGaitG1-PoseBank-v1",  # closed-loop round-2: band+G1 gait from pose cache
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandGaitG1PoseBankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveGraspXLBandGaitG1PoseBankCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-BandRot-CylinderHeld-v1",  # held-gated band, cylinder validation
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveBandRotCylinderHeldEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_graspxl_bandrot_posebank:SharpaWaveBandRotCylinderHeldCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripRotate-Combined-Ball-v1",  # joint adjust+rotate from pinch
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripRotateCombinedBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripRotateCombinedBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripRotate-Mixed-Ball-v1",  # Track A: combined reward + mixed resets (Q1 architecture)
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripRotateMixedBallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripRotateMixedBallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-RegripV2-R4RLN4-Ball-v1",  # Track B escalation: >=4-finger regrip
    entry_point=f"rl_rebuild.tasks.inhand_rotate.sharpa_wave_regripv2_designs:RegripV2R4RLN4BallEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_regripv2_designs:RegripV2R4RLN4BallCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)
