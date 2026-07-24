# Copyright (c) 2026. SPDX-License-Identifier: Apache-2.0
"""Grasp trajectory tensors, SO(3) frame randomization, and gravity-ramp helpers.

Isaac-free, pure-torch (CPU-unit-testable). WXYZ quaternion convention throughout
(IsaacLab default; the SHARPA dataset PoseWxyz is already WXYZ).

Ported from gr00t/rl/envs/sharpa_flying_hand.py:
  - grasp trajectory replay  (~:2608-2657)
  - grasp_frame_dr           (~:1639-1721)
  - gravity ramp mix         (~:2550-2566)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------- #
# WXYZ quaternion math (self-contained)
# ---------------------------------------------------------------------------- #
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-9)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion(s) q (WXYZ)."""
    qw = q[..., :1]
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (qw * uv + uuv)


def axis_angle_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half = 0.5 * angle
    sin_over = torch.where(angle > 1e-8, torch.sin(half) / angle.clamp_min(1e-8), 0.5 * torch.ones_like(angle))
    xyz = axis_angle * sin_over
    w = torch.cos(half)
    return quat_normalize(torch.cat([w, xyz], dim=-1))


def random_uniform_quat(n: int, device) -> torch.Tensor:
    """Uniform random SO(3) quaternions (WXYZ), Shoemake's method."""
    u = torch.rand(n, 3, device=device)
    r1, r2 = torch.sqrt(1.0 - u[:, 0]), torch.sqrt(u[:, 0])
    t1 = 2.0 * torch.pi * u[:, 1]
    t2 = 2.0 * torch.pi * u[:, 2]
    return quat_normalize(
        torch.stack([torch.cos(t2) * r2, torch.sin(t1) * r1, torch.cos(t1) * r1, torch.sin(t2) * r2], dim=-1)
    )


def align_quat(v_from: torch.Tensor, v_to: torch.Tensor) -> torch.Tensor:
    """WXYZ quaternion(s) that rotate normalized v_from onto normalized v_to (half-vector method)."""
    a = v_from / torch.linalg.norm(v_from, dim=-1, keepdim=True).clamp_min(1e-9)
    b = (v_to / torch.linalg.norm(v_to, dim=-1, keepdim=True).clamp_min(1e-9)).expand_as(a)
    h = a + b
    hn = torch.linalg.norm(h, dim=-1, keepdim=True)
    near_anti = hn.squeeze(-1) < 1e-5
    h_norm = torch.where(hn < 1e-5, torch.ones_like(h), h)
    h_norm = h_norm / torch.linalg.norm(h_norm, dim=-1, keepdim=True).clamp_min(1e-9)
    w = torch.sum(a * h_norm, dim=-1, keepdim=True)
    xyz = torch.cross(a, h_norm, dim=-1)
    q = quat_normalize(torch.cat([w, xyz], dim=-1))
    if bool(near_anti.any()):  # antiparallel -> 180deg about any perpendicular axis
        ex = torch.tensor([1.0, 0.0, 0.0], device=a.device).expand_as(a)
        ey = torch.tensor([0.0, 1.0, 0.0], device=a.device).expand_as(a)
        perp = torch.cross(a, ex, dim=-1)
        perp = torch.where(torch.linalg.norm(perp, dim=-1, keepdim=True) < 1e-5, torch.cross(a, ey, dim=-1), perp)
        perp = perp / torch.linalg.norm(perp, dim=-1, keepdim=True).clamp_min(1e-9)
        q = torch.where(near_anti.unsqueeze(-1), torch.cat([torch.zeros_like(w), perp], dim=-1), q)
    return q


def quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    # Canonicalize to the w>=0 hemisphere: q and -q are the same rotation (double cover), but an
    # unflipped sign would flip the extracted axis-angle and could corrupt the per-step omega on a
    # near-pi single-step delta. Cheap hardening (no effect in the normal slow-rotation regime).
    q = torch.where(q[..., :1] < 0.0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt((1.0 - w * w).clamp_min(1e-12))
    axis = torch.where(s > 1e-6, q[..., 1:] / s, q[..., 1:])
    return axis * angle


# ---------------------------------------------------------------------------- #
# Grasp trajectory tensors
# ---------------------------------------------------------------------------- #
@dataclass
class GraspTensors:
    joint: torch.Tensor       # (N, T, 22) joint angles per frame (USD order)
    object_pose: torch.Tensor  # (N, T, 7) object pos(3)+quat_wxyz(4)
    wrist_pose: torch.Tensor   # (N, T, 7) wrist/hand-root pos(3)+quat_wxyz(4)
    length: torch.Tensor       # (N,) frames per env


def build_grasp_tensors(grasps: list, num_joints: int, device) -> GraspTensors:
    """Build padded per-env trajectory tensors from a list of SharpaGrasp records.

    Records may have trajectory_* (NPZ) or only a final pose (JSON) -> length-1 trajectory.
    """
    n = len(grasps)
    seqs_joint, seqs_obj, seqs_wrist, lengths = [], [], [], []
    for g in grasps:
        if g.trajectory_joint_positions_rad is not None:
            jt = torch.tensor(g.trajectory_joint_positions_rad, dtype=torch.float32)  # (T,22)
            op = torch.tensor(g.trajectory_object_positions, dtype=torch.float32)     # (T,3)
            oq = torch.tensor(g.trajectory_object_quats_wxyz, dtype=torch.float32)    # (T,4)
            wp = torch.tensor(g.trajectory_wrist_positions, dtype=torch.float32)      # (T,3)
            wq = torch.tensor(g.trajectory_wrist_quats_wxyz, dtype=torch.float32)     # (T,4)
        else:
            jt = torch.tensor([g.joint_positions_rad], dtype=torch.float32)
            op = torch.tensor([g.object_pose_world.position], dtype=torch.float32)
            oq = torch.tensor([g.object_pose_world.quat_wxyz], dtype=torch.float32)
            wp = torch.tensor([g.right_hand_world.position], dtype=torch.float32)
            wq = torch.tensor([g.right_hand_world.quat_wxyz], dtype=torch.float32)
        seqs_joint.append(jt)
        seqs_obj.append(torch.cat([op, oq], dim=-1))
        seqs_wrist.append(torch.cat([wp, wq], dim=-1))
        lengths.append(jt.shape[0])

    max_t = max(lengths)
    joint = torch.zeros(n, max_t, num_joints, device=device)
    obj = torch.zeros(n, max_t, 7, device=device)
    wrist = torch.zeros(n, max_t, 7, device=device)
    obj[..., 3] = 1.0  # default identity quat for padding
    wrist[..., 3] = 1.0
    for i, (jt, ot, wt, L) in enumerate(zip(seqs_joint, seqs_obj, seqs_wrist, lengths)):
        joint[i, :L] = jt.to(device)
        obj[i, :L] = ot.to(device)
        wrist[i, :L] = wt.to(device)
        # hold the last frame across the pad so settle phases after close_steps keep the final grasp
        joint[i, L:] = jt[-1].to(device)
        obj[i, L:] = ot[-1].to(device)
        wrist[i, L:] = wt[-1].to(device)
    return GraspTensors(joint=joint, object_pose=obj, wrist_pose=wrist,
                        length=torch.tensor(lengths, device=device, dtype=torch.long))


def frame_for_elapsed(elapsed: torch.Tensor, close_steps: int, length: torch.Tensor) -> torch.Tensor:
    """Map elapsed control steps within close_steps -> a trajectory frame id (per env)."""
    denom = max(1, close_steps - 1)
    progress = torch.clamp(elapsed / float(denom), 0.0, 1.0)
    frame = torch.round(progress * (length.float() - 1.0)).long()
    return torch.clamp(frame, torch.zeros_like(length), length - 1)


# ---------------------------------------------------------------------------- #
# Frame DR (SO(3) about the object pivot) applied to per-env initial states + trajectories
# ---------------------------------------------------------------------------- #
def apply_frame_dr(
    delta_quat: torch.Tensor,    # (N,4) WXYZ
    object_traj: torch.Tensor,   # (N,T,7) pos+quat
    wrist_traj: torch.Tensor,    # (N,T,7) pos+quat
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate object + wrist trajectories about each env's object-frame-0 pivot. Returns rotated copies."""
    pivot = object_traj[:, 0, :3]                       # (N,3) first-frame object position
    dq = delta_quat.unsqueeze(1)                        # (N,1,4)
    pv = pivot.unsqueeze(1)                             # (N,1,3)
    obj = object_traj.clone()
    wr = wrist_traj.clone()
    obj[..., :3] = pv + quat_apply(dq, object_traj[..., :3] - pv)
    wr[..., :3] = pv + quat_apply(dq, wrist_traj[..., :3] - pv)
    obj[..., 3:7] = quat_normalize(quat_mul(dq, object_traj[..., 3:7]))
    wr[..., 3:7] = quat_normalize(quat_mul(dq, wrist_traj[..., 3:7]))
    return obj, wr


# ---------------------------------------------------------------------------- #
# Gravity ramp: 0->1 mix over the gravity-ramp window of the settle schedule
# ---------------------------------------------------------------------------- #
def gravity_mix(
    elapsed: torch.Tensor, close_steps: int, zero_g_steps: int, ramp_steps: int
) -> torch.Tensor:
    """Per-env gravity mix in [0,1]: 0 during close+zero-g, ramps to 1 over ramp_steps, then 1."""
    ramp_start = float(close_steps + zero_g_steps)
    if ramp_steps <= 1:
        return (elapsed >= ramp_start).float()
    mix = (elapsed - ramp_start) / float(ramp_steps - 1)
    return torch.clamp(mix, 0.0, 1.0)
