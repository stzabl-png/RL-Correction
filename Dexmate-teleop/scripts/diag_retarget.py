#!/usr/bin/env python
"""Step-0 retarget diagnostic: quantify how well the robot hand tracks the human
target, per fingertip AND per finger-middle joint, offline on the mock glove.

Use it to (a) sanity-check the scaling_factor and (b) A/B the enriched vector
config vs dexpilot before touching the real glove. It does NOT publish anything.

Examples (teleop venv):
  # enriched vector (new default) on a few held poses
  .venv/bin/python scripts/diag_retarget.py --mode vector

  # side-by-side against the old dexpilot behaviour
  .venv/bin/python scripts/diag_retarget.py --mode dexpilot

What it prints per pose:
  - human target vector length (already * scaling_factor) vs robot achieved length,
    for every retarget target link (5 tips + 5 middle joints in vector mode);
  - the residual = ||robot_vec - scaling*human_vec|| in mm (what the optimiser minimises);
  - resulting SDK qpos highlights (thumb + a couple of finger DOFs).
And once at startup:
  - robot zero-pose wrist->middle_fingertip length, the number scaling_factor is
    relative to (robot_len / human_len). Compare against your own hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource  # noqa: E402


def _link_pos(robot, q_target, idx_pin2target, link_id):
    """World position of a link given a target-order qpos (fixed joints = 0)."""
    q_pin = np.zeros(robot.dof)
    q_pin[idx_pin2target] = q_target
    robot.compute_forward_kinematics(q_pin)
    return robot.get_link_pose(link_id)[:3, 3]


def robot_middle_fingertip_len(retargeting, hand: str) -> float:
    opt = retargeting.optimizer
    robot = opt.robot
    q0 = np.zeros(len(opt.target_joint_names))
    wrist = _link_pos(robot, q0, opt.idx_pin2target, robot.get_link_index(f"{hand}_hand_C_MC"))
    tip = _link_pos(robot, q0, opt.idx_pin2target, robot.get_link_index(f"{hand}_middle_fingertip"))
    return float(np.linalg.norm(tip - wrist))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--mode", choices=["vector", "dexpilot"], default="vector")
    p.add_argument("--poses", default="open,fist,pinch@1.5,pinch@2.5",
                   help="comma list; 'open'/'fist' or 'MOTION@T' sampled from the mock source")
    p.add_argument("--abduction-weight", type=float, default=1.0,
                   help="vector mode: abduction vector weight (Step 4.1); 0 == 5-vector baseline")
    p.add_argument("--pinch-weight", type=float, default=0.0,
                   help="vector mode: thumb->finger pinch vector weight (Step 5); 0 == baseline")
    args = p.parse_args()

    retargeting = build_sharpa_retargeting(
        args.hand, args.mode,
        abduction_weight=args.abduction_weight, pinch_weight=args.pinch_weight,
    )
    mapper = JointMapper(retargeting, args.hand)
    opt = retargeting.optimizer
    robot = opt.robot
    # builder.py may set a PER-VECTOR scaling array (fingers vs thumb); keep it as
    # an (n,) vector so the residual math below is correct for both scalar and array.
    scaling = np.asarray(opt.scaling, dtype=float).reshape(-1)  # (1,) or (n,)

    print(f"[cfg] hand={args.hand} mode={args.mode} type={opt.retargeting_type} "
          f"scaling_factor={np.array2string(np.unique(scaling), precision=3)} "
          f"#targets={opt.target_link_human_indices.shape[1]}")
    rlen = robot_middle_fingertip_len(retargeting, args.hand)
    print(f"[scaling] robot zero-pose wrist->middle_fingertip = {rlen*1000:.1f} mm; "
          f"scaling_factor should be ~ robot_len/human_len (measure your own hand)")

    # Human indices (task links) and their robot link ids, for per-vector reporting.
    task_h = opt.target_link_human_indices[1, :]
    origin_h = opt.target_link_human_indices[0, :]
    task_links = getattr(opt, "task_link_names", None)  # vector mode only

    for spec in args.poses.split(","):
        spec = spec.strip()
        if "@" in spec:
            motion, t = spec.split("@")
            t = float(t)
        else:
            motion, t = spec, 0.0
        src = MockGloveSource(hand=args.hand, motion=motion, noise=0.0)
        frame = src.sample_at(t)
        mano = to_mano(frame.kp, args.hand)
        ref = compute_ref_value(retargeting, mano)          # human target vectors (unscaled)
        q_target = retargeting.retarget(ref)                # target-order qpos
        q_sdk = mapper.to_sdk(q_target)

        print(f"\n=== pose '{spec}' ===")
        if task_links is not None:
            print(f"  {'target link':26s} {'humanIdx':>10s} {'human*s(mm)':>11s} "
                  f"{'robot(mm)':>10s} {'resid(mm)':>10s}")
            for k, link in enumerate(task_links):
                lid = robot.get_link_index(link)
                origin_lid = robot.get_link_index(opt.origin_link_names[k])
                rob_vec = _link_pos(robot, q_target, opt.idx_pin2target, lid) - \
                    _link_pos(robot, q_target, opt.idx_pin2target, origin_lid)
                s_k = scaling[k] if scaling.size > 1 else scaling[0]
                hum_vec = (mano[task_h[k]] - mano[origin_h[k]]) * s_k
                resid = np.linalg.norm(rob_vec - hum_vec) * 1000
                print(f"  {link:26s} {int(task_h[k]):>10d} {np.linalg.norm(hum_vec)*1000:>11.1f} "
                      f"{np.linalg.norm(rob_vec)*1000:>10.1f} {resid:>10.1f}")
        else:
            print("  (dexpilot: targets auto-generated; showing qpos only)")
        # qpos highlights
        hi = {n: q_sdk[i] for i, n in enumerate(mapper.sdk_names)
              if n.endswith(("_AA", "_IP")) or "thumb" in n}
        print("  qpos[thumb/AA/IP rad]: " +
              " ".join(f"{n.replace(args.hand+'_',''):s}={v:+.2f}" for n, v in hi.items()))


if __name__ == "__main__":
    main()
