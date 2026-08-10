#!/usr/bin/env python
"""replay_world.npz -> ref_qpos.npz, in the RECONSTRUCTION WORLD frame.

Why this exists rather than calling RL_Correction/rl_rebuild/correction/export_qpos.py:
  * that script writes the wrist into its table-recentred scene frame, which every
    consumer then has to undo;
  * as of 2026-08 it raises NameError at line 48 (`paths` is never imported), so it
    cannot currently produce anything.

Finger retargeting is invariant to rigid transforms of the input joints (DexPilot only
looks at fingertip-vs-wrist vectors), so retargeting the world-frame joints directly
gives the same 22 qpos while leaving the wrist where the reconstruction put it. A
downstream Umeyama fit then resolves to the identity, which doubles as a check.

Wrist orientation follows retarget_isaacsim.base_rot_from_joints (+z fingers, +y thumb
side, +x palm normal), INCLUDING the 180-degree palm flip that `--palm-flip left`
applies to the left hand by default -- without it a left hand comes out mirrored.

Must run under the Isaac venv (it owns dex_retargeting + pinocchio):
  third_party/MagicDexMate/.venv-isaac/bin/python -m contact.make_ref_qpos <take-dir> --hand right
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]          # ego_pipeline/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "Retargeting"))


def base_rot_from_joints(kp: np.ndarray, flip: bool) -> np.ndarray:
    """(21,3) MANO joints -> 3x3 SharpaWave base rotation (columns = base x/y/z)."""
    w = kp[0]
    z = kp[[5, 9, 13, 17]].mean(0) - w
    z = z / (np.linalg.norm(z) + 1e-9)
    radial = kp[5] - kp[17]
    y = radial - (radial @ z) * z
    y = y / (np.linalg.norm(y) + 1e-9)
    x = np.cross(y, z)
    R = np.stack([x, y, z], axis=1)
    if flip:                                        # matches --palm-flip left
        R[:, 0] *= -1
        R[:, 1] *= -1
    return R


def export(replay_npz, out_path, hand="right", mode="dexpilot") -> Path:
    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.retarget.mapping import JointMapper
    from contact.frames import matrix_to_quat_wxyz

    d = np.load(replay_npz, allow_pickle=True)
    joints = np.asarray(d[f"joints_{hand}"], dtype=np.float64)          # (T,21,3) world
    valid = np.asarray(d[f"valid_{hand}"], dtype=np.float64) > 0.5
    valid &= np.isfinite(joints).all((1, 2))
    T = len(joints)
    if valid.sum() < 3:
        raise SystemExit(f"{replay_npz}: only {valid.sum()} valid {hand}-hand frames")

    # hole-fill from the nearest valid frame so retargeting never sees a NaN; the
    # `valid` flag still marks these frames so consumers can skip them
    iv = np.flatnonzero(valid)
    src = iv[np.abs(np.arange(T)[:, None] - iv[None, :]).argmin(1)]
    joints = joints[src]

    retargeting = build_sharpa_retargeting(hand, mode)
    mapper = JointMapper(retargeting, hand)
    t0 = time.time()
    fq = np.zeros((T, 22))
    for i in range(T):
        fq[i] = mapper.to_sdk(retargeting.retarget(
            compute_ref_value(retargeting, to_mano(joints[i], hand))))

    flip = (hand == "left")                         # retarget_isaacsim default --palm-flip left
    quat = np.stack([matrix_to_quat_wxyz(base_rot_from_joints(joints[i], flip))
                     for i in range(T)])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             source_mtime=Path(replay_npz).stat().st_mtime,
             finger_qpos=fq.astype(np.float32),
             joint_names=np.array(mapper.sdk_names),
             wrist_pos=joints[:, 0].astype(np.float32),
             wrist_quat_wxyz=quat.astype(np.float32),
             valid=valid,
             fps=d["fps"] if "fps" in d.files else np.float32(30.0),
             retarget_mode=mode,
             hand=hand,
             frame_of_reference="recon_world",
             palm_flip=bool(flip))
    dq = np.abs(np.diff(fq, axis=0)).max(axis=1)
    print(f"[qpos]   {out_path}  {T} frames {hand}/{mode} in {time.time()-t0:.1f}s | "
          f"valid {int(valid.sum())}/{T} | qpos [{fq.min():.2f},{fq.max():.2f}] rad | "
          f"max frame-to-frame jump {dq.max():.2f} rad")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="RetargetOutput take dir (holds replay_world.npz)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--mode", default="dexpilot", choices=["vector", "dexpilot"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = a.out or a.take / "ref_qpos.npz"
    if out.exists() and not a.force:
        print(f"[qpos]   {out} exists, skipping (use --force)")
        return 0
    export(a.take / "replay_world.npz", out, a.hand, a.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
