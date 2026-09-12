"""Operator head/waist -> Vega joint angles, fitted to poses the operator authored.

Single source of truth, same rule as magicdexmate/palm_fix.py: the constants
live here and everything imports them. Do not copy the tables.

WHERE THE NUMBERS COME FROM
---------------------------
Not from tracker data - there is nothing in a recording that decides what
"looking left" should look like on Dexmate. The operator authored eight poses
in scripts/viser_joint_jog.py and those are the ground truth. Re-author there
and re-run the fit to change them.

REFITTED 2026-08-01 to logs/robot_action_poses.json, the operator's second
batch, which supersedes the first (logs/robot_head_waist_poses.json). The two
disagree by a lot and the operator chose the newer one:

  chest lean, upright    10.3 deg now, 43.5 deg before
  chest lean, bent       46.1 deg (bow_30) now, 61.0 deg (waist_work) before
  neutral gaze          -10.4 deg now, -15.9 deg before

The first batch was authored in a session where the bench had been opened on a
different poses file, so the sliders sat at their defaults and several slugs
were saved unchanged - which is how "upright" came to mean 43.5 deg of forward
lean. Still true and carried over from that batch:

  neutral gaze is NOT level      teleop looks at the work in front of the
                                 robot, so a slight downward default is right.
  left/right forced symmetric    authored +68.8 / -50.5 deg; the operator took
                                 the smaller and mirrored it, so the yaw
                                 envelope is +-50 deg. Kept from batch one -
                                 head_left/head_right were not re-authored, and
                                 yaw is unaffected by the chest-lean change
                                 (the operator agreed to reuse them).

ROBOT KINEMATICS THAT SHAPE THIS (FK-verified, not read off URDF axes - those
are in each joint's own frame and mislead at the leaning working pose)

  head   Y-Z-Y chain. head_j2 is a clean yaw: it tracks azimuth ~1:1 (+-49.5
         deg of joint gives +-50 deg of gaze) and couples only 3.7 deg into
         elevation across the full sweep. At the authored upright the head at
         all-zeros looks 10.4 deg DOWN (it was 55 deg at the old torso home).
         head_j1 and head_j3 share an axis. The operator used both for
         up/down, so the pitch table carries the (j1, j3) pair they authored
         rather than locking j3.
  waist  torso_j1/j2/j3 ALL turn about the same axis - Vega has no waist yaw
         and no side bend, the torso is a three-link sagittal column, so one
         parameter (chest lean) indexes the whole path. Only TWO points are
         authored in this batch (upright 10.3 deg, bow 46.1 deg): waist_max was
         not saved, so the range stops at 46.1 and anything beyond clamps.

GAINS: 1:1, then clamp. Gaze direction should be veridical in teleop -
compressing the operator's range to fit the robot's makes small head motions
feel dead and breaks "the robot looks where I look". Saturating at the
envelope is the more legible failure. The operator's measured head range
(-114..+109 deg yaw, -71..+30 deg pitch over the 2026-07-30 sessions) is wider
than the robot's, so saturation will happen and is intended.
"""
from __future__ import annotations

import numpy as np

# --- authored envelope, logs/robot_head_waist_poses.json (2026-07-31) --------
# gaze elevation [deg] -> (head_j1, head_j3) [deg], the pair the operator used
HEAD_PITCH_TABLE = np.array([
    [-50.5, +37.9,  -2.2],     # head_down
    [-10.4,  +0.0,  +0.0],     # head_level   <- neutral
    [+29.1,  +5.0, +44.5],     # head_up (= look_up_45)
])
NEUTRAL_ELEVATION = -10.4      # deg, where the operator put "looking straight"
YAW_LIMIT = 50.0               # deg of gaze azimuth, symmetric by their choice
YAW_JOINT_PER_DEG = 49.53 / 50.0   # head_j2 per deg of azimuth, ~1:1

# chest lean [deg] -> (torso_j1, torso_j2, torso_j3) [deg]
WAIST_TABLE = np.array([
    [10.3, 66.7, 120.9, +43.8],    # stand_upright  <- neutral
    [46.1, 60.4, 113.7,  +7.2],    # bow_30
])
NEUTRAL_LEAN = 10.3            # deg, the operator's "upright"

# The chest pose everything else is referenced to. Was (60, 90, -25) deg = 55
# deg of forward lean, hardcoded in five separate files; every pose the
# operator ever authored - all thirteen - was instead posed at 10.3 deg, i.e.
# nearly upright. The robot therefore stood more bent than the operator's own
# "bow 30 degrees" pose, which is why bending and standing read as inverted.
# Single source of truth now; the five copies import it.
# float(), not np.radians() alone: numpy returns float64 and Isaac Lab's
# default_joint_pos is float32, so handing it these straight raises
# "Index put requires the source and destination dtypes match" at scene init.
# The literals this replaced were native Python floats, which is why the type
# never mattered before.
# VEGA_TORSO_HOME=legacy restores the 55 deg forward lean the operator signed
# off on live on 2026-08-01 ("the palm direction is right", "no problems"),
# which was an ARMS-ONLY run. It is the one code constant that separates that
# configuration from today's; everything else about that run - control-mode
# arms, pos_scale 1.0, waist_bind_up 0.20, no symmetriser - is a command-line
# flag. Kept switchable rather than edited so the two can be A/B'd without a
# diff, and because the head and waist TABLES below were refitted at the 10.3
# deg home: with 'legacy' they no longer apply, so use it with --control-mode
# arms only.
_LEGACY_TORSO = {"torso_j1": float(np.radians(60.0)),
                 "torso_j2": float(np.radians(90.0)),
                 "torso_j3": float(np.radians(-25.0))}
_AUTHORED_TORSO = {"torso_j1": float(np.radians(66.7)),
                   "torso_j2": float(np.radians(120.9)),
                   "torso_j3": float(np.radians(43.8))}
import os as _os
_HOME_SEL = _os.environ.get("VEGA_TORSO_HOME", "authored").lower()
try:
    # a NUMBER = the chest FORWARD LEAN in degrees, which is how the operator
    # talks about it ("it leans forward 55 deg now, what if I want 10"). Note
    # this is NOT Dexmate's torso "pitch angle", which is its complement:
    # pitch = 90 - lean (their zero position is upright at pitch 90, and their
    # own example (60,120,-30) -> pitch 0 is folded flat). Getting those two
    # the wrong way round makes the stance go the opposite direction, which is
    # what happened the first time this was written.
    #
    # Reached by moving j3 alone from the legacy pose: j3 is the cheapest of
    # the three in shoulder disturbance (7.8 mm per deg vs 10.3 and 17.1), so
    # sliding the home along it changes the stance without dragging the
    # shoulders further than necessary. Lets the hunch/upright trade be
    # measured as a curve instead of argued about as two options.
    _want_lean = float(_HOME_SEL)
    _want_pitch = 90.0 - _want_lean
    _base = _LEGACY_TORSO
    _p0 = (np.degrees(_base["torso_j1"] + _base["torso_j3"]
                      - _base["torso_j2"]) + 90.0)
    TORSO_HOME = dict(_base)
    TORSO_HOME["torso_j3"] = float(np.radians(np.clip(
        np.degrees(_base["torso_j3"]) + (_want_pitch - _p0), -90.0, 90.0)))
    print(f"[head_waist_map] VEGA_TORSO_HOME={_want_lean:g} -> chest leans "
          f"{_want_lean:g}deg forward (Dexmate pitch {_want_pitch:g}) via "
          f"torso_j3 = {np.degrees(TORSO_HOME['torso_j3']):.1f}deg "
          f"(j1/j2 held at legacy 60/90)")
except ValueError:
    TORSO_HOME = _LEGACY_TORSO if _HOME_SEL == "legacy" else _AUTHORED_TORSO
if _HOME_SEL == "legacy":
    print("[head_waist_map] VEGA_TORSO_HOME=legacy -> 55 deg forward lean "
          "(the 2026-08-01 arms-only configuration). The head/waist tables in "
          "this file were fitted at 10.3 deg and do NOT apply here; use "
          "--control-mode arms.")


# --- head chain, closed form -------------------------------------------------
# gaze = R_chest @ Ry(j1) @ Rz(j2) @ diag(1,-1,-1) @ Ry(j3), gaze = column 0.
# Constants read out of vega_1.urdf; agrees with a full URDF FK to 1e-6 deg.
# Solving in the BASE frame (not the chest frame) on purpose: the authored
# poses are base-frame, and it makes the waist mode fall out for free - lean
# the torso and the head compensates because R_chest is an input.
_FLIP = np.diag([1.0, -1.0, -1.0])          # head_j3's origin rpy = (pi, 0, 0)
HEAD_LIMITS = {"head_j1": (-1.4830, 1.4830),
               "head_j2": (-2.7920, 2.7920),
               "head_j3": (-1.3780, 1.4830)}


def _ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def head_gaze(j1: float, j2: float, j3: float, R_chest: np.ndarray) -> np.ndarray:
    """Head joints [rad] + chest rotation -> unit gaze direction, base frame."""
    return (R_chest @ _ry(j1) @ _rz(j2) @ _FLIP @ _ry(j3))[:, 0]


def _az_el(g):
    return (np.degrees(np.arctan2(g[1], g[0])),
            np.degrees(np.arcsin(np.clip(g[2], -1.0, 1.0))))


def head_solve(azimuth_deg: float, elevation_deg: float, R_chest: np.ndarray,
               seed: np.ndarray | None = None, pitch_path: str = "j1"):
    """head_joints, plus the raw solver state to warm-start the next call.

    The state is (path parameter, j2) in the SOLVER's coordinates. Warm-start
    with this and nothing else: an earlier version fed back the achieved gaze
    elevation, which is a different quantity as soon as the chest rotates, and
    on real waist-moving data that bad seed stalled the solve at a 17.8 deg
    median residual - worse than starting cold.
    """
    az = float(np.clip(azimuth_deg, -YAW_LIMIT, YAW_LIMIT))

    # WHICH PATH THE REDUNDANCY IS RESOLVED ALONG. Three joints for a two-axis
    # gaze, and head_j1/head_j3 share an axis, so one dimension is free.
    #
    #   "authored"  the (j1, j3) pair the operator posed. Preserves their choice
    #               of how to split up/down between the two pitch joints - but
    #               that pairing is TORSO-SPECIFIC and the table was fitted at
    #               the 10.3 deg home. Its j1 never goes negative, while at the
    #               legacy 55 deg home the head needs j1 ~ -55 just to look
    #               level: the target then sits off the end of the path, the
    #               solve pins there, and the cold-retry bounces around the
    #               endpoint. Reported live 2026-08-03 as "looking up does not
    #               work at all, looking down judders" - with left/right, which
    #               rides j2 and never touches this path, perfect at 0.03 deg.
    #   "j1"        j1 carries the pitch, j3 locked at 0. Torso-independent,
    #               because head_gaze() takes R_chest as an input and j1's full
    #               +-85 deg range is available at any torso home. This is what
    #               the 2026-07-31 kinematics note already recommended
    #               ("j1 and j3 are coaxial and redundant - use j1, lock j3").
    if pitch_path == "authored":
        el = float(np.clip(elevation_deg + NEUTRAL_ELEVATION,
                           HEAD_PITCH_TABLE[0, 0], HEAD_PITCH_TABLE[-1, 0]))
        s_lo, s_hi = HEAD_PITCH_TABLE[0, 0], HEAD_PITCH_TABLE[-1, 0]

        def on_path(s):
            return (np.radians(np.interp(s, HEAD_PITCH_TABLE[:, 0],
                                         HEAD_PITCH_TABLE[:, 1])),
                    np.radians(np.interp(s, HEAD_PITCH_TABLE[:, 0],
                                         HEAD_PITCH_TABLE[:, 2])))
    else:
        el = float(elevation_deg + NEUTRAL_ELEVATION)
        s_lo, s_hi = np.degrees(HEAD_LIMITS["head_j1"])

        def on_path(s):
            return np.radians(s), 0.0

    def resid(x):
        j1, j3 = on_path(x[0])
        a, e = _az_el(head_gaze(j1, x[1], j3, R_chest))
        return np.array([(a - az + 180.0) % 360.0 - 180.0, e - el])

    def run(x0):
        x = np.array(x0, dtype=float)
        for _ in range(24):
            r = resid(x)
            if np.max(np.abs(r)) < 1e-4:
                break
            J = np.column_stack([(resid(x + d) - r) / h for d, h in
                                 ((np.array([1e-3, 0.0]), 1e-3),
                                  (np.array([0.0, 1e-4]), 1e-4))])
            try:
                step = np.linalg.solve(J, r)
            except np.linalg.LinAlgError:
                break
            # per-component: x[0] is a path parameter in DEGREES (range ~130),
            # x[1] is a joint angle in RADIANS. One shared clip starves the
            # first - a 0.5 cap let the path parameter crawl 0.5 deg per
            # iteration and the solve stalled at a 17.8 deg median residual.
            x = x - np.clip(step, (-10.0, -0.5), (10.0, 0.5))
            x[0] = np.clip(x[0], s_lo, s_hi)
        return x, np.max(np.abs(resid(x)))

    cold = np.array([float(np.clip(el, s_lo, s_hi)),
                     np.radians(az * YAW_JOINT_PER_DEG)])
    x, e = run(cold if seed is None else seed)
    if e > 0.05:
        # ALWAYS retry cold on a bad residual, even though it costs ~1 ms on
        # saturated frames. Skipping the retry when the path parameter looked
        # pinned was tried and is unsafe: a warm start that lands pinned feeds
        # the next frame's warm start, which lands pinned too, and the solve
        # never escapes - median residual went from 0.0000 to 39.7 deg.
        x2, e2 = run(cold)
        if e2 < e:
            x, e = x2, e2
    j1, j3 = on_path(x[0])
    out = {"head_j1": j1, "head_j2": float(x[1]), "head_j3": j3}
    return {k: float(np.clip(v, *HEAD_LIMITS[k])) for k, v in out.items()}, x


def head_joints(azimuth_deg: float, elevation_deg: float,
                R_chest: np.ndarray, seed: np.ndarray | None = None) -> dict[str, float]:
    """Operator gaze relative to their body heading -> head joints [rad].

    azimuth: + is left. elevation: + is up, 0 = the operator looking level;
    NEUTRAL_ELEVATION shifts that onto the neutral the operator authored.

    Three joints for a two-axis target, so the redundancy is resolved along the
    (j1, j3) path the operator authored for pure pitch - their choice of how to
    split up/down between the two pitch joints is preserved, and the solve only
    slides along it.

    A separable table - j2 from azimuth, (j1, j3) from elevation - was tried
    first and is exact on either axis alone but drifts 6-16 deg on diagonal
    gazes, because the authored pitch path was posed at j2 = 0 and the Y-Z-Y
    chain couples. Hence the solve. Use head_solve() to keep the warm start.
    """
    return head_solve(azimuth_deg, elevation_deg, R_chest, seed)[0]


# Dexmate's own definition of the torso pitch, from dexcontrol/core/torso.py:
#     pitch = j1 + j3 + pi/2 - j2
# All three joints move it by the same amount, so "which one do I drive" is not
# about how much it bends but about how far it drags the SHOULDERS while doing
# it. Measured at the legacy home (60, 90, -25) for 25 deg of chest pitch:
#     torso_j1   427.7 mm of shoulder travel   17.1 mm/deg
#     torso_j2   256.5 mm                      10.3 mm/deg
#     torso_j3   194.4 mm                       7.8 mm/deg   <- least
# so j3 alone is the cheapest way to bend. Bending FORWARD decreases it.
J3_ONLY_RANGE = (-90.0, 90.0)      # deg, the joint's own limits


def waist_joints_j3(lean_deg: float, home: dict | None = None) -> dict[str, float]:
    """Operator lean -> chest pitch, driving ONLY torso_j3. j1/j2 hold at home.

    The operator's proposal, 2026-08-03: the real robot exposes the torso as a
    single conceptual pitch DoF, so bend with one joint and leave the rest
    alone. It is also the better engineering choice for two reasons the
    authored-path version cannot match:

      * it needs no authored table, so it is valid at ANY torso home. The
        WAIST_TABLE above was fitted at the 10.3 deg home and does NOT apply at
        the legacy 55 deg one - which is where the arms currently work best.
      * j3 is the cheapest of the three in shoulder disturbance (see above).

    lean_deg is a DELTA from the engage-time zero, same as waist_joints().
    """
    h = dict(home or TORSO_HOME)
    j3 = np.degrees(h["torso_j3"]) - float(lean_deg)     # forward bend = -j3
    j3 = float(np.clip(j3, *J3_ONLY_RANGE))
    return {"torso_j1": h["torso_j1"], "torso_j2": h["torso_j2"],
            "torso_j3": float(np.radians(j3))}


def waist_joints(lean_deg: float) -> dict[str, float]:
    """Operator torso lean -> torso joints [rad].

    lean_deg is a DELTA from the engage-time zero, not an absolute pitch: the
    waist tracker's mounting angle on the belt is arbitrary and changes every
    time it is put on, so its absolute tilt means nothing. 0 therefore lands on
    the authored "upright".

    1:1 into chest lean, clamped to the authored envelope. The robot's whole
    range is 30.1 deg, which is about what a person bends before it stops being
    a lean, so 1:1 fills it without needing a gain.
    """
    lean = float(np.clip(lean_deg + NEUTRAL_LEAN,
                         WAIST_TABLE[0, 0], WAIST_TABLE[-1, 0]))
    return {f"torso_j{i + 1}": float(np.radians(
        np.interp(lean, WAIST_TABLE[:, 0], WAIST_TABLE[:, i + 1])))
        for i in range(3)}


# The chest pose the arms are bound to when the waist is NOT being driven.
TORSO_HOLD = dict(TORSO_HOME)   # arms-only modes hold the authored upright
