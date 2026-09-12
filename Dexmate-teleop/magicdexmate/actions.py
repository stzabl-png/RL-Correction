"""The action catalogue - one definition, used by every tool that names a pose.

These are the ten poses the operator agreed to on 2026-07-29 and the ones the
labelled captures under `logs/pose_capture_*` are recorded against. Anything
that labels a pose (the capture recorder, the joint teach bench, the offline
analysers) imports from here, so a slug always means the same thing and notes
taken in one tool can be joined to captures made by another.

Each entry is (label, slug, hold_s):
  label   what the operator is asked to do, in the words they defined it in
  slug    the identifier used in filenames, manifests and notes
  hold_s  seconds to hold for a capture; None = the static default (5 s)
"""
from __future__ import annotations

# 2026-07-31 session: 5 trackers (2 wrists + 2 elbows + waist), one take each,
# operator keeping the trackers in the headset's view. Guard/punch were added
# because they are asymmetric - every earlier pose was mirror-symmetric, so a
# left/right mix-up in the mapping could not show up. These can.
SESSION_20260731: list[tuple[str, str, float | None]] = [
    ("双手前平举", "front_horizontal", None),
    ("双手举手姿势(上臂前伸、肘90°、小臂竖直向上)", "hand_raise", None),
    ("双手格挡(两臂抬起护在身前)", "block_both", None),
    ("左手出拳 + 右手格挡", "punch_left", None),
    ("右手出拳 + 左手格挡", "punch_right", None),
]

ACTIONS: list[tuple[str, str, float | None]] = [
    ("双手举高", "raise_high", None),
    ("双手前平举", "front_horizontal", None),
    ("双手侧平举", "lateral_horizontal", None),
    ("双手自然放身体两侧", "arms_down", None),
    ("双手叉腰", "hands_on_hips", None),
    # user definition 2026-07-29: upper arms extended horizontally FORWARD,
    # elbows ~90deg, forearms vertical pointing UP, both arms identical
    ("双手举手姿势(上臂前伸、肘90°、小臂竖直向上)", "hand_raise", None),
    # dynamic wrist-orientation takes (user 2026-07-29): arms held, wrists
    # rotated SLOWLY through the take - calibrates orientation axes/OFFSETS
    ("双手前平举+缓慢转腕(动态10s)", "front_wrist_roll", 10.0),
    ("双手侧平举+缓慢转腕(动态10s)", "lateral_wrist_roll", 10.0),
    # coverage/validation extras (assistant 2026-07-29): full-workspace sweep
    # and body-relative invariance (hands fixed to body, torso moves)
    ("双臂缓慢扫掠:身前→侧面→举高→放下(动态15s)", "slow_sweep", 15.0),
    ("双手轻贴胸前不动+躯干缓慢转/倾(动态10s)", "torso_disturb", 10.0),
    # 2026-07-31: the first ASYMMETRIC poses. Everything above is mirror-
    # symmetric, so a left/right swap anywhere in the mapping produces a pose
    # that still looks right - these are the ones that can catch it.
    ("双手格挡(两臂抬起护在身前)", "block_both", None),
    ("左手出拳 + 右手格挡", "punch_left", None),
    ("右手出拳 + 左手格挡", "punch_right", None),
]

# 2026-07-31 evening: the operator's second static batch, defined after seeing
# the first live run of --map shoulder-rel. Chosen to attack what that run got
# wrong rather than to re-cover ground:
#   1-2   arms FOLDED against the body. The failure was "when my arms come
#         close to my body, Dexmate's do not" - poses that hold the wrist near
#         the torso are the ones that expose a scale bug, and a fully extended
#         pose never would.
#   3-4   asymmetric-ish whole-body shapes with the palms facing each other or
#         gripping overhead, which pin the wrist ROLL that symmetric poses
#         leave free.
#   5-7   torso pitch: bent 30 deg / upright / looking up 45 deg, arms hanging.
#         The operator reported bending and standing felt inverted, and these
#         three differ ONLY in torso pitch, so they isolate its sign.
# Together with the six earlier poses this is a 13-pose static reference set.
SESSION_20260731_PM: list[tuple[str, str, float | None]] = [
    ("双臂贴近胸口、掌心朝向胸口、双臂呈直角", "arms_folded_chest", None),
    ("双臂放在腰侧且贴近腰、掌心朝向胸口", "arms_folded_waist", None),
    ("跑步预备姿势(上半身),两掌心相对", "run_ready", None),
    ("引体向上姿势", "pullup", None),
    ("弯腰 30 度,双手自然垂于两侧", "bow_30", None),
    ("直立站立,双手自然垂于两侧", "stand_upright", None),
    ("抬头上仰 45 度,双手自然垂于两侧", "look_up_45", None),
]

# 2026-07-31: head and waist, authored on the ROBOT side only.
#
# Deliberately not in ACTIONS: ACTIONS is what a human is asked to hold while
# being recorded (scripts/pose_capture.py iterates it), and these are not that.
# What "looking left" should look like on Dexmate is intent, not measurement -
# there is nothing in the tracker data that decides it - so the operator
# authors them in the joint bench and they become the ground truth the head and
# waist mapping is fitted to.
#
# Robot-side facts worth knowing while authoring these (all FK-verified, not
# read off the URDF axes - those are in each joint's own frame and mislead at
# the leaning working pose):
#   head   all three joints at 0 looks 55 deg DOWN, not level; head_j1 ~ -55
#          levels it, and only then is head_j2 a clean left/right yaw.
#          head_j1 and head_j3 share an axis (opposite signs) - use j1 only.
#   waist  torso_j1/j2/j3 ALL turn about the same axis. Vega has no waist yaw
#          and no side bend; the torso is a three-link sagittal column. j1/j2
#          set where the chest sits, j3 turns the chest without moving it.
HEAD_WAIST: list[tuple[str, str, float | None]] = [
    ("头:平视正前", "head_level", None),
    ("头:左看到位", "head_left", None),
    ("头:右看到位", "head_right", None),
    ("头:抬头看", "head_up", None),
    ("头:低头看", "head_down", None),
    ("腰:直立", "waist_upright", None),
    ("腰:前倾到合适的工作位", "waist_work", None),
    ("腰:弯到最大", "waist_max", None),
]
HEAD_WAIST_LABELS = [a[0] for a in HEAD_WAIST]
SESSION_PM_LABELS = [a[0] for a in SESSION_20260731_PM]

# picked when the pose being studied is not one of the ten
FREEFORM = ("(自定义 / 不在目录里)", "freeform", None)

LABELS = [a[0] for a in ACTIONS]
SLUGS = [a[1] for a in ACTIONS]
# slug_of must resolve every label any tool can offer, head/waist included
BY_LABEL = {a[0]: a for a in ACTIONS + HEAD_WAIST + SESSION_20260731_PM}
BY_SLUG = {a[1]: a for a in ACTIONS + HEAD_WAIST + SESSION_20260731_PM}

# hold_s is None == a pose you can stand still in. The four timed entries are
# motions (wrist rotation, sweep, torso disturbance) and have no single pose,
# so tools that author or label a STILL configuration - the joint teach bench -
# offer only these six. (User, 2026-07-30: "我只记录静态的那几个".)
# What a HUMAN can be asked to hold while recording. ACTIONS plus the second
# static batch; deliberately NOT the head/waist entries, which are authored on
# the robot and have no human counterpart. scripts/pose_capture.py filters its
# --poses against this - the 2026-07-31 PM batch was invisible to it until it
# was added here, and the recorder simply exited with 0 segments.
CAPTURABLE = ACTIONS + SESSION_20260731_PM

STATIC_ACTIONS = [a for a in ACTIONS if a[2] is None]
STATIC_LABELS = [a[0] for a in STATIC_ACTIONS]


def slug_of(label: str) -> str:
    """Catalogue slug for a label, or the freeform slug if it is not one."""
    entry = BY_LABEL.get(label)
    return entry[1] if entry else FREEFORM[1]
