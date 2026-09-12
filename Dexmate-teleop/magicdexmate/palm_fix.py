"""The wrist->EE orientation constant for absolute palm mapping.

Single source of truth. It used to be a literal in sim/teleop_vega_pico.py and
was copied into the viser tools; a constant that lives in two places is how the
two drift apart, which is exactly the failure this module exists to prevent.

--------------------------------------------------------------------------
WHICH CHANNEL THIS CONSTANT BELONGS TO  (read before changing anything)
--------------------------------------------------------------------------
PALM_FIX maps the *model channel* wrist frame - PICO's 24-joint body estimate,
joints 20/21 - to R_ee.  It was fitted on logs/pose_capture_20260731_ori, whose
`trackers` entries are body24[20/21] bit-for-bit (checked, 0.00 mm, all 10
takes).

It is NOT valid for the raw per-serial tracker channel, which reports the
physical device's own frame.  Feeding it raw wrists puts the palm 96 deg off
(median over 20 samples from the 2026-07-29 _raw batch; range 66-106; 20 of 20
wrong, both hands).  scripts/trackers.env selects the channel - keep it on the
model channel, or measure the raw<->model offset first and refit.

This is also why --ori-mode track always felt fine while palm did not: track is
incremental and anchors a constant frame offset away at engage; palm is
absolute and shows it the whole time.

--------------------------------------------------------------------------
HOW IT WAS FITTED, AND WHAT THE NUMBERS MEAN
--------------------------------------------------------------------------
R_ee = R_wrist @ PALM_FIX[hand].  Fitted over five held poses against the
operator's own description of where each palm pointed, after two failed
attempts:

  1. One pose, matched against an authored robot pose. Reproduced that pose to
     9.7 deg and nothing else - no redundancy, so it absorbed whatever was
     wrong with that single pair.
  2. Five poses, still against authored robot poses. Blew up to 25-71 deg
     apart. The authored poses were made before the hands were rendered, so
     five of six had the wrist at its default - the robot side was wrong, not
     the model.
  3. Five poses against the DESCRIPTIONS. The description defines the target
     frame outright, so no authored pose can corrupt it. Spread 5-21 deg.

Re-checked 2026-08-01 after the operator judged raise_high the only bad take:
dropping it moves the palm normal 0.4 deg, and a full 3-DoF refit (palm plus
finger direction, four good poses) lands 1.4 deg from these numbers. The
constant is right; the 8-10 deg residual is the data's own noise floor (the
same held pose, two takes 15 s apart, differs by 3.8-18.5 deg).

A caution for designing the next calibration: a palm direction is 2 DoF and a
rotation is 3.  F @ Rx(theta) @ ex == F @ ex, so the roll about the palm normal
- where the fingers point - is invisible to any description of the palm alone.
Describe the finger direction too, or that DoF stays unconstrained.

Left and right are mirrored, not equal: fitted independently they sit 145 deg
apart, and a sagittal mirror (y -> -y) brings them to 19. One matrix for both
hands flips a wrist.
"""
from __future__ import annotations

import numpy as np

MIRROR = np.diag([1.0, -1.0, 1.0])

PALM_FIX = {
    "right": np.array([
        [-0.07822, +0.99166, +0.10245],
        [-0.04372, +0.09925, -0.99410],
        [-0.99598, -0.08223, +0.03559],
    ]),
    "left": np.array([
        [-0.07822, -0.99166, +0.10245],
        [+0.04372, +0.09925, +0.99410],
        [-0.99598, +0.08223, +0.03559],
    ]),
}

# Sharpa hand geometry in the R_ee / L_ee frame. Measured three independent
# ways, all agreeing: the MCP flexion direction from the URDF kinematics, a PCA
# of the hand meshes (thinnest axis = palm normal, longest = fingers, 2.1 deg
# from the kinematic answer), and the same PCA on the official
# vega_1_f5d6.urdf, which is Dexmate's own shipped arm-plus-hand assembly.
# Both hands come out the same here - the mirroring lives in PALM_FIX, not in
# the hand geometry.
#
# Note on record_of_relationship: it calls the arm flange "EE", but its axis
# table matches *_arm_l8 exactly (0.00 deg) and *_ee only to 90 deg. Read it as
# l8 and the record, the URDF weld and these vectors all agree.
PALM_IN_EE = np.array([1.0, 0.0, 0.0])       # palm normal    = +ee.x
FINGERS_IN_EE = np.array([0.0, 0.0, 1.0])    # finger heading = +ee.z


def relax_roll(R_target, R_current, cap_deg=None):
    """Keep the palm NORMAL of R_target; take the roll about it from R_current.

    Two of the three rotational DoF say where the palm faces; the third is the
    roll about the palm normal, i.e. which way the fingers point. No "my palm
    faced X" description can observe that third one - F @ Rx(t) @ e_x == F @ e_x
    for every t - so it has always been the least-supported part of the target.

    It is also expensive. Measured 2026-08-01 on logs/playlist_all13.msgpack at
    the authored upright torso, --pos-scale 1.35 --waist-bind-up 0.08:

        roll pinned   wrist position residual   272 mm R / 192 mm L
        roll relaxed                             21 mm R /   4 mm L
                      cost: finger heading off by 65 deg R / 85 deg L

    The palm normal itself is unaffected either way. So this is a real trade,
    not a free win - the fingers can end up sideways, which matters for
    grasping - and the operator decides. cap_deg limits the give-away, though
    measurements say a small cap buys almost nothing (+-10..45 deg recovered
    only 272 -> 253 mm): the position only comes back with a large roll.

    R_target, R_current: 3x3, SAME frame. Returns 3x3.
    """
    M = np.asarray(R_target).T @ np.asarray(R_current)
    th = np.arctan2(M[2, 1] - M[1, 2], M[1, 1] + M[2, 2])
    if cap_deg is not None:
        th = float(np.clip(th, -np.radians(cap_deg), np.radians(cap_deg)))
    c, s = np.cos(th), np.sin(th)
    return np.asarray(R_target) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
