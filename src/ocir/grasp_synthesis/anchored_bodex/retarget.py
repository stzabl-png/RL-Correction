"""MANO -> Sharpa Wave retargeting for anchored BODex seeding.

Two pieces:

- **Wrist calibration** (:func:`compute_calibration`, run once by
  ``scripts/grasp_synthesis/calibrate_mano_sharpa.py``): a scale-free Kabsch
  fit between flat-pose Sharpa fingertip/pad points (FK at q=0, base frame)
  and flat-pose MANO keypoints expressed in the keypoint-derived wrist frame
  (:func:`demo_analysis.wrist_frame_from_keypoints`). The result -- the fixed
  rigid transform ``T_calib`` mapping Sharpa ``base_link`` coordinates into
  the MANO wrist frame -- is stored in
  ``assets/.../grasp_synthesis/bodex/mano_transfer.yml`` together with the
  MANO-keypoint -> Sharpa-link correspondence table.

- **Fingertip-fit IK** (:class:`HandFitter`): given per-frame demo wrist
  poses and 21 keypoints (object frame), place the hand base at
  ``T_wrist @ T_calib`` and fit the finger joints (plus a small bounded wrist
  correction) so the mapped Sharpa keypoints match the MANO keypoints. Runs
  batched over frames with projected Adam; everything reuses the repo's
  existing cuRobo v2 ``Kinematics`` -- no external retargeting dependency.

Per-frame hand poses are the *anchors*: each optimization seed keeps its own
un-relaxed retargeted pose for the annealed pose prior and similarity metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ocir.grasp_synthesis.bodex_curobo_v2.backend import import_official_curobo

import_official_curobo()

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg

from ocir.grasp_synthesis.assets import SharpaWaveAsset
from ocir.grasp_synthesis.bodex_curobo_v2.seed_generator import matrix_to_quaternion
from ocir.grasp_synthesis.bodex_curobo_v2.solver import _load_yaml, _sphere_lookup

#: MANO keypoint -> Sharpa link correspondences. Keypoint indices follow
#: ``mano_model.pose_m_to_vertices_and_joints`` order: fingertips 4/8/12/16/20
#: (weight 1.0), PIP/IP-level keypoints (weight 0.3).
#:
#: Fingertips target the URDF's dedicated ``right_*_fingertip`` frames (fixed
#: elastomer-tip links, zero offset) -- the same frames the Sharpa teleop
#: retargeting configs use (MagicDexMate's dex-retargeting vector scheme) and
#: the true correspondent of a MANO tip keypoint. PIP-level keypoints target
#: the proximal-phalanx pad collision spheres ("link/sphere_index" entries,
#: resolved from the collision yaml).
#:
#: Deliberately NOT adopted from the teleop pipeline: its human->robot size
#: ``scaling_factor`` (~1.07). Teleop matches wrist-relative *vectors*, so
#: scaling is free; our targets are absolute keypoint positions in the object
#: frame, and scaling them would move the intended contacts off the object
#: surface. The size mismatch is absorbed by the bounded per-frame wrist
#: correction + joint fit instead.
DEFAULT_KEYPOINT_SPEC = (
    {"mano_joint": 4, "link": "right_thumb_fingertip", "weight": 1.0},
    {"mano_joint": 8, "link": "right_index_fingertip", "weight": 1.0},
    {"mano_joint": 12, "link": "right_middle_fingertip", "weight": 1.0},
    {"mano_joint": 16, "link": "right_ring_fingertip", "weight": 1.0},
    {"mano_joint": 20, "link": "right_pinky_fingertip", "weight": 1.0},
    {"mano_joint": 3, "sphere": "right_thumb_PP/0", "weight": 0.3},
    {"mano_joint": 6, "sphere": "right_index_PP/0", "weight": 0.3},
    {"mano_joint": 10, "sphere": "right_middle_PP/0", "weight": 0.3},
    {"mano_joint": 14, "sphere": "right_ring_PP/0", "weight": 0.3},
    {"mano_joint": 18, "sphere": "right_pinky_PP/0", "weight": 0.3},
)

DEFAULT_TRANSFER_RELPATH = "grasp_synthesis/bodex/mano_transfer.yml"


@dataclass(frozen=True)
class KeypointTarget:
    mano_joint: int
    link: str
    offset: np.ndarray  # (3,) position in the link frame
    weight: float


@dataclass(frozen=True)
class ManoTransferCalib:
    wrist_rot: np.ndarray  # (3,3): Sharpa base_link -> MANO wrist frame
    wrist_trans: np.ndarray  # (3,)
    keypoints: tuple[KeypointTarget, ...]

    @property
    def wrist_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.wrist_rot
        pose[:3, 3] = self.wrist_trans
        return pose


def transfer_path(asset: SharpaWaveAsset) -> Path:
    return asset.root / DEFAULT_TRANSFER_RELPATH


def load_mano_transfer(path: Path) -> ManoTransferCalib:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    keypoints = tuple(
        KeypointTarget(
            mano_joint=int(entry["mano_joint"]),
            link=str(entry["link"]),
            offset=np.asarray(entry["offset"], dtype=float),
            weight=float(entry["weight"]),
        )
        for entry in data["keypoint_map"]
    )
    return ManoTransferCalib(
        wrist_rot=np.asarray(data["wrist"]["r"], dtype=float).reshape(3, 3),
        wrist_trans=np.asarray(data["wrist"]["t"], dtype=float).reshape(3),
        keypoints=keypoints,
    )


def save_mano_transfer(path: Path, calib: ManoTransferCalib, report: dict) -> None:
    data = {
        "wrist": {"r": [[float(v) for v in row] for row in calib.wrist_rot], "t": [float(v) for v in calib.wrist_trans]},
        "keypoint_map": [
            {
                "mano_joint": int(kp.mano_joint),
                "link": kp.link,
                "offset": [float(v) for v in kp.offset],
                "weight": float(kp.weight),
            }
            for kp in calib.keypoints
        ],
        "calibration_report": report,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def build_keypoint_targets(asset: SharpaWaveAsset) -> tuple[KeypointTarget, ...]:
    """Instantiate the default correspondence table: dedicated URDF frames
    (zero offset) for fingertips, collision-sphere centers for pad keypoints."""

    collision_config = _load_yaml(asset.collision_spheres_path)
    targets = []
    for spec in DEFAULT_KEYPOINT_SPEC:
        if "link" in spec:
            link, center = spec["link"], np.zeros(3)
        else:
            link, sphere_idx = spec["sphere"].split("/")
            center, _ = _sphere_lookup(collision_config, link, int(sphere_idx))
        targets.append(
            KeypointTarget(mano_joint=int(spec["mano_joint"]), link=link, offset=center, weight=float(spec["weight"]))
        )
    return tuple(targets)


def kabsch(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted rigid transform (R, t) minimizing sum_i w_i ||R s_i + t - t_i||^2."""

    weights = np.asarray(weights, dtype=float)
    w = weights / weights.sum()
    src_c = (w[:, None] * source).sum(axis=0)
    tgt_c = (w[:, None] * target).sum(axis=0)
    src = source - src_c
    tgt = target - tgt_c
    h = (w[:, None] * src).T @ tgt
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    t = tgt_c - r @ src_c
    return r, t


def _axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Batched differentiable Rodrigues formula, (..., 3) -> (..., 3, 3)."""

    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=eps)
    k = axis_angle / theta
    kx, ky, kz = torch.unbind(k, dim=-1)
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky], dim=-1),
            torch.stack([kz, zero, -kx], dim=-1),
            torch.stack([-ky, kx, zero], dim=-1),
        ],
        dim=-2,
    )
    theta = theta.unsqueeze(-1)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(K.shape)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


@dataclass(frozen=True)
class RetargetResult:
    ref_actions: np.ndarray        # (F, 7 + J): [pos, quat_wxyz, joints]
    joint_names: tuple[str, ...]   # order of the J joint entries
    frame_indices: np.ndarray      # (F,) demo frame indices these anchors came from
    keypoint_residuals_m: np.ndarray  # (F, K) final per-keypoint distances
    report: dict


class HandFitter:
    """Batched fingertip-fit IK from MANO demo frames to Sharpa actions."""

    def __init__(
        self,
        asset: SharpaWaveAsset,
        calib: ManoTransferCalib,
        device_cfg: DeviceCfg,
        *,
        iters: int = 300,
        lr_start: float = 5e-2,
        lr_end: float = 5e-3,
        wrist_trans_bound_m: float = 0.01,
        wrist_rot_bound_rad: float = np.deg2rad(10.0),
        posture_reg: float = 1e-2,
    ):
        self.asset = asset
        self.calib = calib
        self.device_cfg = device_cfg
        self.iters = int(iters)
        self.lr_start = float(lr_start)
        self.lr_end = float(lr_end)
        self.wrist_trans_bound = float(wrist_trans_bound_m)
        self.wrist_rot_bound = float(wrist_rot_bound_rad)
        self.posture_reg = float(posture_reg)

        tool_frames = list(dict.fromkeys([kp.link for kp in calib.keypoints]))
        robot_cfg = RobotCfg.from_basic(
            urdf_path=str(asset.urdf_path),
            base_link=asset.config["base_link"],
            tool_frames=tool_frames,
            device_cfg=device_cfg,
        )
        self.kinematics = Kinematics(robot_cfg.kinematics)
        self.joint_names = tuple(self.kinematics.joint_names)
        limits = asset.config["joint_limits"]
        self.joint_lower = device_cfg.to_device([limits[name][0] for name in self.joint_names])
        self.joint_upper = device_cfg.to_device([limits[name][1] for name in self.joint_names])

        # Cupped-hand initial q from the BODex seeder config (same mapping as
        # bodex_curobo_v2's HeurGraspSeedGenerator).
        grasp_cfg = _load_yaml(asset.bodex_path("grasp_synthesis_config"))
        robot_config_yaml = _load_yaml(asset.bodex_path("robot_config"))
        cspace_names = list(robot_config_yaml["robot_cfg"]["kinematics"]["cspace"]["joint_names"])
        q_by_name = dict(zip(cspace_names, grasp_cfg["seeder_cfg"]["q"]))
        missing = [name for name in self.joint_names if name not in q_by_name]
        if missing:
            raise KeyError(f"seeder_cfg.q has no value for joints {missing}")
        self.init_q = device_cfg.to_device([q_by_name[name] for name in self.joint_names])

        self.kp_offsets = device_cfg.to_device(np.stack([kp.offset for kp in calib.keypoints]))  # (K,3)
        self.kp_weights = device_cfg.to_device(np.asarray([kp.weight for kp in calib.keypoints]))  # (K,)
        self.kp_mano_idx = [kp.mano_joint for kp in calib.keypoints]
        self.kp_links = [kp.link for kp in calib.keypoints]
        self.t_calib = device_cfg.to_device(calib.wrist_pose.astype(np.float32))  # (4,4)

    def _fk_keypoints_base(self, q: torch.Tensor) -> torch.Tensor:
        """FK keypoint positions in the base_link frame, (F, K, 3)."""

        from curobo._src.geom.transform import torch_quaternion_to_matrix

        kin_state = self.kinematics.compute_kinematics(
            JointState.from_position(q, joint_names=list(self.joint_names))
        )
        points = []
        for k, link in enumerate(self.kp_links):
            pose = kin_state.tool_poses.get_link_pose(link)
            rot = torch_quaternion_to_matrix(pose.quaternion)
            offset = self.kp_offsets[k].view(1, 3, 1)
            points.append((rot @ offset).squeeze(-1) + pose.position)
        return torch.stack(points, dim=1)

    def base_poses_from_wrist(self, wrist_poses_object: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Demo wrist poses (F,4,4) -> nominal Sharpa base (pos (F,3), rot (F,3,3))."""

        wrist = self.device_cfg.to_device(np.asarray(wrist_poses_object, dtype=np.float32))
        base = wrist @ self.t_calib.view(1, 4, 4)
        return base[:, :3, 3], base[:, :3, :3]

    def fit_frames(
        self,
        wrist_poses_object: np.ndarray,   # (F, 4, 4)
        keypoints_object: np.ndarray,     # (F, 21, 3)
        frame_indices: np.ndarray,
    ) -> RetargetResult:
        f = int(np.asarray(wrist_poses_object).shape[0])
        base_pos, base_rot = self.base_poses_from_wrist(wrist_poses_object)
        targets = self.device_cfg.to_device(
            np.asarray(keypoints_object, dtype=np.float32)[:, self.kp_mano_idx, :]
        )  # (F, K, 3)

        q = self.init_q.view(1, -1).repeat(f, 1).clone().requires_grad_(True)
        dt = torch.zeros((f, 3), device=q.device, dtype=q.dtype, requires_grad=True)
        dr = torch.zeros((f, 3), device=q.device, dtype=q.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([q, dt, dr], lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iters, eta_min=self.lr_end)

        def keypoints_world(q_, dt_, dr_):
            rot = base_rot @ _axis_angle_to_matrix(dr_)
            pos = base_pos + dt_
            kp_base = self._fk_keypoints_base(q_)
            return (rot.unsqueeze(1) @ kp_base.unsqueeze(-1)).squeeze(-1) + pos.unsqueeze(1), rot, pos

        for _ in range(self.iters):
            optimizer.zero_grad(set_to_none=True)
            kp_world, _, _ = keypoints_world(q, dt, dr)
            err = ((kp_world - targets) ** 2).sum(dim=-1)  # (F, K)
            loss = (self.kp_weights.view(1, -1) * err).sum(dim=-1).mean()
            loss = loss + self.posture_reg * ((q - self.init_q.view(1, -1)) ** 2).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                q.clamp_(self.joint_lower.view(1, -1), self.joint_upper.view(1, -1))
                dt.clamp_(-self.wrist_trans_bound, self.wrist_trans_bound)
                dr.clamp_(-self.wrist_rot_bound, self.wrist_rot_bound)

        with torch.no_grad():
            kp_world, rot, pos = keypoints_world(q, dt, dr)
            residuals = torch.linalg.norm(kp_world - targets, dim=-1)  # (F, K)
            quat = matrix_to_quaternion(rot)
            ref_actions = torch.cat([pos, quat, q], dim=-1).cpu().numpy()

        residuals_np = residuals.cpu().numpy()
        tip_res = residuals_np[:, :5]
        report = {
            "num_frames": f,
            "iters": self.iters,
            "tip_residual_mean_m": float(tip_res.mean()),
            "tip_residual_max_m": float(tip_res.max()),
            "all_residual_mean_m": float(residuals_np.mean()),
            "wrist_correction_max_trans_m": float(dt.detach().abs().max().item()),
            "wrist_correction_max_rot_rad": float(dr.detach().abs().max().item()),
        }
        return RetargetResult(
            ref_actions=ref_actions,
            joint_names=self.joint_names,
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            keypoint_residuals_m=residuals_np,
            report=report,
        )


def compute_calibration(
    asset: SharpaWaveAsset,
    mano_keypoints_flat: np.ndarray,  # (21, 3) flat-pose MANO keypoints, any frame
    device_cfg: DeviceCfg,
) -> tuple[ManoTransferCalib, dict]:
    """Kabsch-fit the fixed base_link -> MANO-wrist-frame transform from flat poses."""

    from ocir.grasp_synthesis.anchored_bodex.demo_analysis import wrist_frame_from_keypoints

    targets = build_keypoint_targets(asset)
    calib_identity = ManoTransferCalib(wrist_rot=np.eye(3), wrist_trans=np.zeros(3), keypoints=targets)
    fitter = HandFitter(asset, calib_identity, device_cfg)

    q_flat = torch.zeros((1, len(fitter.joint_names)), device=device_cfg.device, dtype=device_cfg.dtype)
    q_flat = torch.clamp(q_flat, fitter.joint_lower.view(1, -1), fitter.joint_upper.view(1, -1))
    with torch.no_grad():
        sharpa_points = fitter._fk_keypoints_base(q_flat)[0].cpu().numpy()  # (K,3) base frame

    wrist_pose = wrist_frame_from_keypoints(mano_keypoints_flat)
    inv = np.linalg.inv(wrist_pose)
    mano_wrist_frame = (
        np.asarray(mano_keypoints_flat, dtype=float)[[t.mano_joint for t in targets]] @ inv[:3, :3].T
        + inv[:3, 3]
    )

    weights = np.asarray([t.weight for t in targets])
    r, t = kabsch(sharpa_points, mano_wrist_frame, weights)
    aligned = sharpa_points @ r.T + t
    residuals = np.linalg.norm(aligned - mano_wrist_frame, axis=-1)
    report = {
        "per_point_residual_mm": {
            f"{targets[i].link}<-mano{targets[i].mano_joint}": float(residuals[i] * 1000.0) for i in range(len(targets))
        },
        "mean_residual_mm": float(residuals.mean() * 1000.0),
        "max_residual_mm": float(residuals.max() * 1000.0),
    }
    calib = ManoTransferCalib(wrist_rot=r, wrist_trans=t, keypoints=targets)
    return calib, report
