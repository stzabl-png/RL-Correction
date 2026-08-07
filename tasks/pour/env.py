"""IsaacLab environment for reference-guided bimanual pouring."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
)

from rl_rebuild.correction.env.dexmate_env_cfg import ARM_EFFORT
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
from rl_rebuild.correction.ref_builders.replay_grasp import (
    GENERIC_JOINT_ORDER,
    GENERIC_OPEN,
)
from tasks.pour.cfg import PourTaskCfg
from tasks.pour.core import (
    ACTION_DIM,
    AblationMode,
    CurriculumStage,
    FailureCode,
    LiquidState,
    PourPhase,
    combine_success,
    compute_reward,
    failure_codes,
    reward_phase_gates,
    split_bimanual_action,
    step_liquid_proxy,
    success_conditions,
)
from tasks.pour.reference import PourReference
from tasks.pour.scene import PourSceneManifest


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _np_qmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    value = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def _matrix_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    from tasks.pour.reference import rotation_matrix_to_wxyz

    return rotation_matrix_to_wxyz(rotation).astype(np.float64)


class PourTaskEnv(DirectRLEnv):
    cfg: PourTaskCfg

    def __init__(self, cfg: PourTaskCfg, render_mode: str | None = None, **kwargs):
        if not cfg.scene_manifest:
            raise ValueError("PourTaskCfg.scene_manifest is required")
        self.manifest = PourSceneManifest.load(cfg.scene_manifest)
        if cfg.screening_mode:
            if self.manifest.status != "pending_grasp_approval":
                raise RuntimeError(
                    "screening mode only accepts pending_grasp_approval scenes"
                )
            self.manifest.cup.validate(require_assets=True, require_geometry=True)
            self.manifest.bottle.validate(require_assets=True, require_geometry=True)
            self.manifest.left_grasp_prior = cfg.screening_left_prior
            self.manifest.right_grasp_prior = cfg.screening_right_prior
        else:
            self.manifest.validate(require_assets=True)
        self.reference = PourReference.load(self.manifest.reference_npz)
        if not self.reference.training_ready:
            raise RuntimeError(
                f"reference {self.reference.demo_id} is missing optimized object tracks "
                "or video contact points"
            )
        self.ablation = AblationMode(cfg.ablation)
        self.curriculum_stage = CurriculumStage(cfg.curriculum_stage)
        super().__init__(cfg, render_mode, **kwargs)
        self._resolve_robot_ids()
        self._settle_robot()
        self._build_reference_and_priors()
        self._allocate_task_state()

    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.bottle = RigidObject(self.cfg.object_cfg)
        self.cup = RigidObject(self.cfg.cup_cfg)

        sx, sy, sz = self.cfg.table_size
        table = sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.30, 0.25)),
        )
        table.func(
            "/World/envs/env_.*/Table",
            table,
            translation=(0.0, 0.0, self.cfg.table_top_z - sz / 2.0),
        )
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["bottle"] = self.bottle
        self.scene.rigid_objects["cup"] = self.cup

        low = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=3.0,
            dynamic_friction=3.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        low.func("/World/Materials/PourLowGrip", low)
        grip.func("/World/Materials/PourSuperGrip", grip)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(
                path, "/World/Materials/PourLowGrip", stronger_than_descendants=False
            )
        self._contact_sensors: dict[str, list[ContactSensor]] = {"left": [], "right": []}
        for side, object_name in (("left", "Cup"), ("right", "Bottle")):
            for index, finger in enumerate(FINGERS):
                body = f"{side}_{finger}_elastomer"
                for path in sim_utils.find_matching_prim_paths(
                    f"/World/envs/env_.*/Robot/{body}"
                ):
                    sim_utils.bind_physics_material(path, "/World/Materials/PourSuperGrip")
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Robot/{body}",
                        history_length=1,
                        filter_prim_paths_expr=[f"/World/envs/env_.*/{object_name}"],
                    )
                )
                self._contact_sensors[side].append(sensor)
                self.scene.sensors[f"{side}_tip_contact_{index}"] = sensor

        for object_name, spec in (("Cup", self.manifest.cup), ("Bottle", self.manifest.bottle)):
            material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(spec.friction),
                dynamic_friction=float(spec.friction),
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="multiply",
            )
            path = f"/World/Materials/Pour{object_name}"
            material.func(path, material)
            for prim in sim_utils.find_matching_prim_paths(
                f"/World/envs/env_.*/{object_name}"
            ):
                sim_utils.bind_physics_material(prim, path)
        dome_light = sim_utils.DomeLightCfg(intensity=2000.0)
        dome_light.func("/World/Light", dome_light)

    def _resolve_robot_ids(self) -> None:
        joint_names = list(self.robot.joint_names)
        body_names = list(self.robot.body_names)
        self.arm_ids: dict[str, list[int]] = {}
        self.finger_ids: dict[str, list[int]] = {}
        self.ee_ids: dict[str, int] = {}
        self.tip_body_ids: dict[str, list[int]] = {}
        self.finger_map: dict[str, torch.Tensor] = {}
        for side, prefix in (("left", "L"), ("right", "R")):
            arm_names = [f"{prefix}_arm_j{index}" for index in range(1, 8)]
            missing = [name for name in arm_names if name not in joint_names]
            if missing:
                raise RuntimeError(f"DexMate is missing {missing}")
            self.arm_ids[side] = [joint_names.index(name) for name in arm_names]
            side_fingers = [name for name in joint_names if name.startswith(f"{side}_")]
            if len(side_fingers) != 22:
                raise RuntimeError(f"{side} hand has {len(side_fingers)} joints, expected 22")
            self.finger_ids[side] = [joint_names.index(name) for name in side_fingers]
            ee_name = f"{side}_hand_C_MC"
            self.ee_ids[side] = body_names.index(ee_name)
            self.tip_body_ids[side] = [
                body_names.index(f"{side}_{finger}_elastomer") for finger in FINGERS
            ]
            mapping = []
            for name in side_fingers:
                match = [
                    index
                    for index, finger in enumerate(FINGERS)
                    if f"_{finger}_" in name or name.endswith(f"_{finger}")
                ]
                if len(match) != 1:
                    raise RuntimeError(f"cannot assign {name} to one finger")
                mapping.append(match[0])
            self.finger_map[side] = torch.tensor(
                mapping, dtype=torch.long, device=self.device
            )
        limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.arm_lower = {
            side: limits[..., 0][:, ids] for side, ids in self.arm_ids.items()
        }
        self.arm_upper = {
            side: limits[..., 1][:, ids] for side, ids in self.arm_ids.items()
        }
        self.finger_lower = {
            side: limits[..., 0][:, ids] for side, ids in self.finger_ids.items()
        }
        self.finger_upper = {
            side: limits[..., 1][:, ids] for side, ids in self.finger_ids.items()
        }
        self.arm_effort_limit = torch.tensor(
            [ARM_EFFORT[index] for index in range(1, 8)],
            dtype=torch.float32,
            device=self.device,
        )

    def _settle_robot(self) -> None:
        default = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(default, torch.zeros_like(default))
        self.robot.set_joint_position_target(default)
        self.robot.write_data_to_sim()
        for _ in range(int(self.cfg.settle_physics_steps)):
            self.sim.step(render=False)
        self.robot.update(self.sim.get_physics_dt())

    # ------------------------------------------------------------------
    def _load_prior(self, path: str, side: str, object_pose: np.ndarray) -> dict:
        with np.load(path, allow_pickle=False) as data:
            required = {"grasp", "pregrasp", "contact_centroid", "hand_side"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"{side} prior missing {sorted(missing)}")
            prior = {name: data[name].copy() for name in data.files}
        encoded_side = np.asarray(prior["hand_side"], dtype=np.uint8).tobytes().decode(
            "utf-8"
        )
        if encoded_side != side:
            raise ValueError(f"{side} prior declares hand_side={encoded_side!r}")
        generic_names = [
            name.replace("right_", f"{side}_") for name in GENERIC_JOINT_ORDER
        ]
        actual_names = [self.robot.joint_names[index] for index in self.finger_ids[side]]
        permutation = [generic_names.index(name) for name in actual_names]
        prior["permutation"] = np.asarray(permutation, dtype=np.int64)
        object_rotation = quat_to_R(object_pose[3:7])

        def world(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                object_rotation @ row[:3] + object_pose[:3],
                _np_qmul(object_pose[3:7], row[3:7]),
            )

        body_names = list(self.robot.body_names)
        anchor_id = body_names.index("arm_center")
        origin = self.scene.env_origins[0].cpu().numpy()
        anchor_pose = np.eye(4, dtype=np.float64)
        anchor_pose[:3, :3] = quat_to_R(
            self.robot.data.body_quat_w[0, anchor_id].cpu().numpy()
        )
        anchor_pose[:3, 3] = (
            self.robot.data.body_pos_w[0, anchor_id].cpu().numpy() - origin
        )
        ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_pose)
        grasp_pos, grasp_quat = world(prior["grasp"])
        grasp = ik.solve(grasp_pos, quat_to_R(grasp_quat), iters=240, pos_tol=2.0e-4)
        if not grasp["ok"]:
            raise RuntimeError(
                f"{side} approved prior is unreachable: {grasp['pos_err'] * 100:.2f}cm"
            )
        best = None
        for row in np.atleast_2d(prior["pregrasp"]):
            pos, quat = world(row)
            result = ik.solve(
                pos, quat_to_R(quat), q0=grasp["q"], iters=240, pos_tol=2.0e-4
            )
            if result["ok"]:
                distance = float(np.linalg.norm(result["q"] - grasp["q"]))
                if best is None or distance < best[0]:
                    best = (distance, result)
        if best is None:
            raise RuntimeError(f"{side} prior has no reachable pregrasp")
        prior.update(
            ik=ik,
            q_grasp=grasp["q"].astype(np.float32),
            q_pregrasp=best[1]["q"].astype(np.float32),
            grasp_world=np.r_[grasp_pos, grasp_quat].astype(np.float32),
            q_open=GENERIC_OPEN[permutation].astype(np.float32),
            q_close=prior["grasp"][7:29][permutation].astype(np.float32),
        )
        return prior

    def _build_reference_and_priors(self) -> None:
        cup_pose = np.asarray(self.manifest.cup.initial_pose_wxyz, dtype=np.float64)
        bottle_pose = np.asarray(self.manifest.bottle.initial_pose_wxyz, dtype=np.float64)
        self.prior = {
            "left": self._load_prior(self.manifest.left_grasp_prior, "left", cup_pose),
            "right": self._load_prior(
                self.manifest.right_grasp_prior, "right", bottle_pose
            ),
        }

        left_video = self.reference.left_wrist.astype(np.float64)
        right_video = self.reference.right_wrist.astype(np.float64)
        video_to_sim = np.asarray(
            self.manifest.video_to_sim_wxyz, dtype=np.float64
        )
        self.video_map_translation = video_to_sim[:3]
        self.video_map_rotation = quat_to_R(video_to_sim[3:7])

        self.q_video: dict[str, torch.Tensor] = {}
        self.video_wrist_world: dict[str, np.ndarray] = {}
        for side, video in (("left", left_video), ("right", right_video)):
            grasp_world = self.prior[side]["grasp_world"].astype(np.float64)
            if self.cfg.screening_mode:
                # Gate-1 for a candidate is reachability of its grasp and
                # pregrasp, already checked in _load_prior.  Full video-trajectory
                # IK is a training gate and must not reject a prior before Gate-2.
                self.q_video[side] = torch.tensor(
                    np.repeat(
                        self.prior[side]["q_grasp"][None], len(video), axis=0
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.video_wrist_world[side] = np.repeat(
                    grasp_world[None], len(video), axis=0
                )
                continue
            mapped_position = (
                video[:, :3] @ self.video_map_rotation.T
                + self.video_map_translation
            )
            # The common object mapping preserves scene geometry.  A constant
            # per-hand embodiment offset anchors the human wrist at the
            # approved robot grasp without changing its video displacement.
            position = mapped_position + (grasp_world[:3] - mapped_position[0])
            mapped_rotation = [
                self.video_map_rotation @ quat_to_R(row[3:7]) for row in video
            ]
            rotation0 = mapped_rotation[0]
            grasp_rotation = quat_to_R(grasp_world[3:7])
            quaternion = np.stack(
                [
                    _matrix_quat_wxyz(
                        grasp_rotation @ rotation0.T @ mapped_rotation[index]
                    )
                    for index in range(len(video))
                ]
            )
            solutions = self.prior[side]["ik"].solve_traj(
                position,
                quaternion,
                q_init=self.prior[side]["q_grasp"],
            )
            q = np.stack([value["q"] for value in solutions])
            valid = np.array(
                [value["ok"] and np.isfinite(value["q"]).all() for value in solutions]
            )
            if valid.mean() < 0.90:
                raise RuntimeError(
                    f"{side} video IK reachability {valid.mean() * 100:.1f}% < 90%"
                )
            bad = np.flatnonzero(~valid)
            good = np.flatnonzero(valid)
            if len(bad):
                q[bad] = q[good[np.abs(good[None] - bad[:, None]).argmin(axis=1)]]
            self.q_video[side] = torch.tensor(
                q, dtype=torch.float32, device=self.device
            )
            self.video_wrist_world[side] = np.c_[position, quaternion]

        self.object_reference: dict[str, torch.Tensor] = {}
        for name, raw, initial in (
            ("cup", self.reference.cup_pose, cup_pose),
            ("bottle", self.reference.bottle_pose, bottle_pose),
        ):
            raw = raw.astype(np.float64)
            position = (
                raw[:, :3] @ self.video_map_rotation.T
                + self.video_map_translation
            )
            quat = np.stack(
                [
                    _matrix_quat_wxyz(
                        self.video_map_rotation @ quat_to_R(raw[index, 3:7])
                    )
                    for index in range(len(raw))
                ]
            )
            self.object_reference[name] = torch.tensor(
                np.c_[position, quat], dtype=torch.float32, device=self.device
            )
            if np.linalg.norm(position[0] - initial[:3]) > 0.03:
                raise RuntimeError(
                    f"{name} ARKit-to-simulation mapping misses initial pose by "
                    f"{np.linalg.norm(position[0] - initial[:3]) * 100:.1f}cm"
                )

        self.video_phase = torch.tensor(
            self.reference.video_phase, dtype=torch.long, device=self.device
        )
        self.approach_ref: dict[str, torch.Tensor] = {}
        count = max(int(self.cfg.stance_prefix_frames), 2)
        u = torch.linspace(0.0, 1.0, count, device=self.device)
        u = u * u * (3.0 - 2.0 * u)
        default = self.robot.data.default_joint_pos[0]
        for side in ("left", "right"):
            start = default[self.arm_ids[side]]
            end = torch.tensor(
                self.prior[side]["q_pregrasp"], dtype=torch.float32, device=self.device
            )
            self.approach_ref[side] = start + u[:, None] * (end - start)

        self.open_q = {
            side: torch.tensor(
                self.prior[side]["q_open"], dtype=torch.float32, device=self.device
            )
            for side in ("left", "right")
        }
        self.close_q = {
            side: torch.tensor(
                self.prior[side]["q_close"], dtype=torch.float32, device=self.device
            )
            for side in ("left", "right")
        }
        self.contact_local = {
            "left": torch.tensor(
                self.reference.video_contact_left,
                dtype=torch.float32,
                device=self.device,
            ),
            "right": torch.tensor(
                self.reference.video_contact_right,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        self._build_arm_start_pools()

    def _build_arm_start_pools(self) -> None:
        """Precompute reachable ±3cm/±15deg pregrasp perturbations."""

        self.arm_start_pool: dict[str, torch.Tensor] = {}
        rng = np.random.default_rng(int(self.cfg.seed))
        for side in ("left", "right"):
            ik = self.prior[side]["ik"]
            q0 = self.prior[side]["q_pregrasp"].astype(np.float64)
            position0, rotation0 = ik.fk(q0)
            candidates = [q0.copy() for _ in range(32)]
            attempts = 0
            limit = max(int(self.cfg.start_pool_size) * 8, 256)
            while len(candidates) < int(self.cfg.start_pool_size) and attempts < limit:
                attempts += 1
                direction = rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1.0e-8)
                position = position0 + direction * rng.uniform(
                    0.0, self.cfg.start_jitter_position_m
                )
                axis = rng.normal(size=3)
                axis /= max(np.linalg.norm(axis), 1.0e-8)
                angle = rng.uniform(
                    -np.deg2rad(self.cfg.start_jitter_rotation_deg),
                    np.deg2rad(self.cfg.start_jitter_rotation_deg),
                )
                skew = np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                delta_rotation = (
                    np.eye(3)
                    + np.sin(angle) * skew
                    + (1.0 - np.cos(angle)) * (skew @ skew)
                )
                solved = ik.solve(
                    position,
                    delta_rotation @ rotation0,
                    q0=q0,
                    iters=100,
                    pos_tol=2.0e-3,
                )
                if solved["ok"] and np.isfinite(solved["q"]).all():
                    candidates.append(solved["q"])
            if len(candidates) < 64:
                raise RuntimeError(
                    f"{side} start jitter pool has only {len(candidates)} reachable poses"
                )
            self.arm_start_pool[side] = torch.tensor(
                np.stack(candidates), dtype=torch.float32, device=self.device
            )

    def _allocate_task_state(self) -> None:
        count, device = self.num_envs, self.device
        self.actions = torch.zeros(count, ACTION_DIM, device=device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.task_phase = torch.full(
            (count,), int(PourPhase.STANCE), dtype=torch.long, device=device
        )
        self.phase_step = torch.zeros(count, dtype=torch.long, device=device)
        self.reference_step = torch.zeros(count, dtype=torch.long, device=device)
        self.closure = torch.zeros(count, 2, device=device)
        self.finger_delta = torch.zeros(count, 2, 5, device=device)
        self.grasp_hold = torch.zeros(count, dtype=torch.long, device=device)
        self.contact_loss = torch.zeros(count, 2, dtype=torch.long, device=device)
        self.verify_hold = torch.zeros(count, dtype=torch.long, device=device)
        self.grasp_latched = torch.zeros(count, 2, dtype=torch.bool, device=device)
        # In SINGLE_GRASP, half the vectorized environments target each side;
        # the target swaps on every reset to avoid a permanent side bias.
        self.single_side = torch.zeros(count, dtype=torch.long, device=device)
        self.reset_count = torch.zeros(count, dtype=torch.long, device=device)
        self.success = torch.zeros(count, dtype=torch.bool, device=device)
        self.failure = torch.zeros(count, dtype=torch.long, device=device)
        self.liquid = LiquidState.full(
            count, device=device, mass=self.cfg.liquid.initial_mass
        )
        self.arm_target = {
            side: self.robot.data.default_joint_pos[:, self.arm_ids[side]].clone()
            for side in ("left", "right")
        }
        self.arm_target_previous = {
            side: value.clone() for side, value in self.arm_target.items()
        }
        self.finger_target = {
            side: self.open_q[side].expand(count, 22).clone()
            for side in ("left", "right")
        }
        self.previous_cup_mass = torch.zeros(count, device=device)
        self.previous_spill_mass = torch.zeros(count, device=device)
        self._signals: dict[str, torch.Tensor] = {}
        self._terminal_signals: dict[str, torch.Tensor] = {}
        self._reward_terms: dict[str, torch.Tensor] = {}
        self.proprio_hist = torch.zeros(
            count, self.cfg.prop_hist_len, 116, device=device
        )
        self.metric_episodes = torch.zeros((), dtype=torch.long, device=device)
        self.metric_successes = torch.zeros((), dtype=torch.long, device=device)
        self.metric_side_episodes = torch.zeros(2, dtype=torch.long, device=device)
        self.metric_side_successes = torch.zeros(2, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def _arm_reference(self, side: str) -> torch.Tensor:
        phase = self.task_phase
        count = self.num_envs
        default = self.robot.data.default_joint_pos[:, self.arm_ids[side]]
        result = default.clone()
        approach_index = self.phase_step.clamp(max=len(self.approach_ref[side]) - 1)
        approach = self.approach_ref[side][approach_index]
        result = torch.where((phase == int(PourPhase.APPROACH))[:, None], approach, result)

        pre = torch.tensor(
            self.prior[side]["q_pregrasp"], dtype=torch.float32, device=self.device
        )
        grasp = torch.tensor(
            self.prior[side]["q_grasp"], dtype=torch.float32, device=self.device
        )
        grasp_u = (self.phase_step.float() / 20.0).clamp(0.0, 1.0)[:, None]
        grasp_ref = pre + grasp_u * (grasp - pre)
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            side_index = 0 if side == "left" else 1
            active_side = self.single_side == side_index
            grasp_ref = torch.where(active_side[:, None], grasp_ref, pre)
        result = torch.where((phase == int(PourPhase.DUAL_GRASP))[:, None], grasp_ref, result)

        video_phase = phase >= int(PourPhase.ALIGN)
        video_index = self.reference_step.clamp(max=len(self.video_phase) - 1)
        video_ref = self.q_video[side][video_index]
        if self.ablation.use_video_trajectory:
            result = torch.where(video_phase[:, None], video_ref, result)
        else:
            result = torch.where(video_phase[:, None], self.arm_target[side], result)
        return result

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1.0, 1.0))
        left_action, right_action = split_bimanual_action(self.actions)
        action_by_side = {"left": left_action, "right": right_action}
        arm_scale = torch.tensor(
            self.cfg.arm_residual_max, dtype=torch.float32, device=self.device
        ) * float(self.cfg.arm_step_scale)
        for side_index, side in enumerate(("left", "right")):
            action = action_by_side[side]
            reference = self._arm_reference(side)
            feedforward = reference - self.arm_target_previous[side]
            active = self.task_phase != int(PourPhase.STANCE)
            if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
                active &= self.single_side == side_index
            proposal = self.arm_target[side] + feedforward + action[:, :7] * arm_scale
            if self.ablation.use_video_trajectory:
                proposal = torch.maximum(
                    torch.minimum(proposal, reference + self.cfg.arm_tube_rad),
                    reference - self.cfg.arm_tube_rad,
                )
            proposal = torch.maximum(
                torch.minimum(proposal, self.arm_upper[side]), self.arm_lower[side]
            )
            self.arm_target[side] = torch.where(
                active[:, None], proposal, reference
            )
            self.arm_target_previous[side] = reference

            in_grasp_or_later = self.task_phase >= int(PourPhase.DUAL_GRASP)
            if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
                in_grasp_or_later &= self.single_side == side_index
            increment = self.cfg.closure_ref_rate + action[:, 7] * self.cfg.closure_rate_max
            self.closure[:, side_index] = torch.where(
                in_grasp_or_later,
                (self.closure[:, side_index] + increment).clamp(0.0, self.cfg.closure_max),
                torch.zeros_like(self.closure[:, side_index]),
            )
            delta = self.finger_delta[:, side_index]
            delta = (
                delta + action[:, 8:13] * self.cfg.finger_delta_rate
            ).clamp(-self.cfg.finger_delta_max, self.cfg.finger_delta_max)
            self.finger_delta[:, side_index] = torch.where(
                in_grasp_or_later[:, None], delta, torch.zeros_like(delta)
            )
            depth = (
                self.closure[:, side_index, None]
                + self.finger_delta[:, side_index][:, self.finger_map[side]]
            ).clamp(0.0, self.cfg.closure_max)
            normalized = (depth / self.cfg.closure_max).clamp(0.0, 1.0)
            target = self.open_q[side] + normalized * (
                self.close_q[side] - self.open_q[side]
            )
            self.finger_target[side] = torch.maximum(
                torch.minimum(target, self.finger_upper[side]),
                self.finger_lower[side],
            )

    def _apply_action(self) -> None:
        for side in ("left", "right"):
            self.robot.set_joint_position_target(
                self.arm_target[side], joint_ids=self.arm_ids[side]
            )
            self.robot.set_joint_position_target(
                self.finger_target[side], joint_ids=self.finger_ids[side]
            )

    # ------------------------------------------------------------------
    def _contact_magnitudes(self, side: str) -> torch.Tensor:
        values = []
        for sensor in self._contact_sensors[side]:
            force = sensor.data.force_matrix_w
            if force is None:
                force = sensor.data.net_forces_w.unsqueeze(1)
            values.append(force.reshape(self.num_envs, -1, 3).norm(dim=-1).amax(dim=1))
        return torch.stack(values, dim=1)

    @staticmethod
    def _tilt(axis_w: torch.Tensor) -> torch.Tensor:
        axis = torch.nn.functional.normalize(axis_w, dim=1)
        return torch.acos(axis[:, 2].clamp(-1.0, 1.0))

    def _geometry(self) -> dict[str, torch.Tensor]:
        origins = self.scene.env_origins
        cup_pos = self.cup.data.root_pos_w - origins
        bottle_pos = self.bottle.data.root_pos_w - origins
        cup_quat = self.cup.data.root_quat_w
        bottle_quat = self.bottle.data.root_quat_w
        cup_center_local = torch.tensor(
            self.manifest.cup.opening_center_local,
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, 3)
        bottle_mouth_local = torch.tensor(
            self.manifest.bottle.opening_center_local,
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, 3)
        cup_axis_local = torch.tensor(
            self.manifest.cup.opening_axis_local,
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, 3)
        bottle_axis_local = torch.tensor(
            self.manifest.bottle.opening_axis_local,
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, 3)
        cup_center = cup_pos + quat_apply(cup_quat, cup_center_local)
        bottle_mouth = bottle_pos + quat_apply(bottle_quat, bottle_mouth_local)
        cup_axis = quat_apply(cup_quat, cup_axis_local)
        bottle_axis = quat_apply(bottle_quat, bottle_axis_local)
        rel = bottle_mouth - cup_center
        height = (rel * cup_axis).sum(dim=1)
        radial = rel - height[:, None] * cup_axis
        return {
            "cup_pos": cup_pos,
            "cup_quat": cup_quat,
            "bottle_pos": bottle_pos,
            "bottle_quat": bottle_quat,
            "cup_center": cup_center,
            "bottle_mouth": bottle_mouth,
            "cup_axis": cup_axis,
            "bottle_axis": bottle_axis,
            "mouth_rel": rel,
            "mouth_height": height,
            "radial_error": radial.norm(dim=1),
            "cup_tilt": self._tilt(cup_axis),
            "bottle_tilt": self._tilt(bottle_axis),
        }

    def _reference_errors(self, geometry: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.reference_step.clamp(max=len(self.video_phase) - 1)
        position_errors = []
        rotation_errors = []
        for name in ("cup", "bottle"):
            reference = self.object_reference[name][index]
            position_errors.append(reference[:, :3] - geometry[f"{name}_pos"])
            delta_q = quat_mul(
                quat_conjugate(reference[:, 3:7]), geometry[f"{name}_quat"]
            )
            rotation_errors.append(axis_angle_from_quat(delta_q))
        return torch.cat(position_errors, dim=1), torch.cat(rotation_errors, dim=1)

    def _contact_error(self, geometry: dict[str, torch.Tensor]) -> torch.Tensor:
        errors = []
        for side, object_name in (("left", "cup"), ("right", "bottle")):
            quat = geometry[f"{object_name}_quat"]
            pos = geometry[f"{object_name}_pos"]
            local = self.contact_local[side]
            count = len(local)
            target = pos[:, None] + quat_apply(
                quat[:, None].expand(-1, count, -1).reshape(-1, 4),
                local[None].expand(self.num_envs, -1, -1).reshape(-1, 3),
            ).reshape(self.num_envs, count, 3)
            tips = self.robot.data.body_pos_w[:, self.tip_body_ids[side]]
            target_w = target + self.scene.env_origins[:, None]
            errors.append(torch.cdist(tips, target_w).amin(dim=2).mean(dim=1))
        per_side = torch.stack(errors, dim=1)
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            return per_side.gather(1, self.single_side[:, None]).squeeze(1)
        return per_side.mean(dim=1)

    def _advance_phase(self, geometry: dict[str, torch.Tensor], left_grasp: torch.Tensor, right_grasp: torch.Tensor) -> None:
        phase = self.task_phase
        self.phase_step += 1
        next_phase = phase.clone()
        stance_done = (phase == int(PourPhase.STANCE)) & (
            self.phase_step >= self.cfg.phase_timeout[PourPhase.STANCE]
        )
        next_phase = torch.where(stance_done, int(PourPhase.APPROACH), next_phase)

        approach_done = (phase == int(PourPhase.APPROACH)) & (
            self.phase_step >= len(self.approach_ref["left"]) - 1
        )
        next_phase = torch.where(
            approach_done, int(PourPhase.DUAL_GRASP), next_phase
        )

        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            target_grasp = torch.where(
                self.single_side == 0, left_grasp, right_grasp
            )
        else:
            target_grasp = left_grasp & right_grasp
        both = target_grasp & (phase == int(PourPhase.DUAL_GRASP))
        self.grasp_hold = torch.where(
            both, self.grasp_hold + 1, torch.zeros_like(self.grasp_hold)
        )
        grasp_done = self.grasp_hold >= self.cfg.grasp_hold_steps
        self.grasp_latched[:, 0] |= left_grasp
        self.grasp_latched[:, 1] |= right_grasp
        if self.curriculum_stage >= CurriculumStage.ALIGN_POUR:
            next_phase = torch.where(grasp_done, int(PourPhase.ALIGN), next_phase)

        active_video = phase >= int(PourPhase.ALIGN)
        if self.ablation.use_video_trajectory:
            self.reference_step = torch.where(
                active_video,
                (self.reference_step + 1).clamp(max=len(self.video_phase) - 1),
                self.reference_step,
            )
            label = self.video_phase[self.reference_step]
            next_phase = torch.where(
                (phase == int(PourPhase.ALIGN)) & (label == int(PourPhase.POUR)),
                int(PourPhase.POUR),
                next_phase,
            )
            next_phase = torch.where(
                (phase == int(PourPhase.POUR)) & (label == int(PourPhase.RETURN)),
                int(PourPhase.RETURN),
                next_phase,
            )
            next_phase = torch.where(
                (phase == int(PourPhase.RETURN))
                & (self.reference_step >= len(self.video_phase) - 1),
                int(PourPhase.VERIFY),
                next_phase,
            )
        else:
            aligned = (
                geometry["radial_error"] <= self.cfg.liquid.cup_radius_m
            ) & (
                geometry["cup_tilt"]
                <= torch.deg2rad(torch.tensor(15.0, device=self.device))
            )
            next_phase = torch.where(
                (phase == int(PourPhase.ALIGN)) & aligned,
                int(PourPhase.POUR),
                next_phase,
            )
            next_phase = torch.where(
                (phase == int(PourPhase.POUR))
                & (self.liquid.bottle <= 0.20 * self.cfg.liquid.initial_mass),
                int(PourPhase.RETURN),
                next_phase,
            )
            returned = geometry["bottle_tilt"] <= torch.deg2rad(
                torch.tensor(self.cfg.success.max_return_tilt_deg, device=self.device)
            )
            next_phase = torch.where(
                (phase == int(PourPhase.RETURN)) & returned,
                int(PourPhase.VERIFY),
                next_phase,
            )

        changed = next_phase != phase
        self.phase_step = torch.where(changed, torch.zeros_like(self.phase_step), self.phase_step)
        just_grasped = changed & (next_phase == int(PourPhase.ALIGN))
        self.reference_step = torch.where(
            just_grasped, torch.zeros_like(self.reference_step), self.reference_step
        )
        self.task_phase = next_phase

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        geometry = self._geometry()
        left_force = self._contact_magnitudes("left")
        right_force = self._contact_magnitudes("right")
        left_grasp = (left_force > self.cfg.contact_force_thresh).sum(dim=1) >= self.cfg.grasp_min_pads
        right_grasp = (right_force > self.cfg.contact_force_thresh).sum(dim=1) >= self.cfg.grasp_min_pads

        liquid_step = step_liquid_proxy(
            self.liquid,
            bottle_up_axis_w=geometry["bottle_axis"],
            bottle_mouth_pos_w=geometry["bottle_mouth"],
            cup_center_pos_w=geometry["cup_center"],
            cup_up_axis_w=geometry["cup_axis"],
            in_pour_phase=self.task_phase == int(PourPhase.POUR),
            dt=float(self.cfg.sim.dt * self.cfg.decimation),
            cfg=self.cfg.liquid,
        )
        self.liquid = liquid_step.state
        self._advance_phase(geometry, left_grasp, right_grasp)

        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            grasp_stage_success = self.grasp_hold >= self.cfg.grasp_hold_steps
        elif self.curriculum_stage in (
            CurriculumStage.DUAL_GRASP,
            CurriculumStage.APPROACH_GRASP,
        ):
            grasp_stage_success = self.grasp_hold >= self.cfg.grasp_hold_steps
        else:
            grasp_stage_success = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        after_grasp = self.task_phase >= int(PourPhase.ALIGN)
        for index, grasp in enumerate((left_grasp, right_grasp)):
            lost = after_grasp & ~grasp
            self.contact_loss[:, index] = torch.where(
                lost,
                self.contact_loss[:, index] + 1,
                torch.zeros_like(self.contact_loss[:, index]),
            )
        left_drop = self.contact_loss[:, 0] >= self.cfg.contact_loss_steps
        right_drop = self.contact_loss[:, 1] >= self.cfg.contact_loss_steps
        cup_tip = geometry["cup_tilt"] > torch.deg2rad(
            torch.tensor(self.cfg.cup_tip_fail_deg, device=self.device)
        )
        spill = self.liquid.spill > (
            self.cfg.max_spill_fraction * self.cfg.liquid.initial_mass
        )
        timeout_limit = torch.tensor(
            self.cfg.phase_timeout, dtype=torch.long, device=self.device
        )[self.task_phase]
        timeout = self.phase_step >= timeout_limit
        align_or_pour = (self.task_phase == int(PourPhase.ALIGN)) | (
            self.task_phase == int(PourPhase.POUR)
        )
        align_miss = timeout & align_or_pour
        codes = failure_codes(
            left_drop=left_drop,
            right_drop=right_drop,
            cup_tip=cup_tip,
            align_miss=align_miss,
            spill=spill,
            timeout=timeout,
        )
        codes = torch.where(
            grasp_stage_success, torch.zeros_like(codes), codes
        )
        self.failure = torch.where(
            (self.failure == int(FailureCode.NONE)) & (codes != int(FailureCode.NONE)),
            codes,
            self.failure,
        )

        conditions = success_conditions(
            phase=self.task_phase,
            liquid=self.liquid,
            initial_mass=self.cfg.liquid.initial_mass,
            left_grasp=left_grasp,
            right_grasp=right_grasp,
            cup_tilt_rad=geometry["cup_tilt"],
            bottle_tilt_rad=geometry["bottle_tilt"],
            thresholds=self.cfg.success,
        )
        success_now = (
            combine_success(conditions)
            if self.curriculum_stage >= CurriculumStage.ALIGN_POUR
            else grasp_stage_success
        )
        self.verify_hold = torch.where(
            success_now,
            self.verify_hold + 1,
            torch.zeros_like(self.verify_hold),
        )
        self.success |= self.verify_hold >= self.cfg.success.verify_hold_steps
        terminated = self.success | (self.failure != int(FailureCode.NONE))
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        completed = terminated | truncated
        self.metric_episodes += completed.sum()
        self.metric_successes += (completed & self.success).sum()
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            for side_index in (0, 1):
                side_completed = completed & (self.single_side == side_index)
                self.metric_side_episodes[side_index] += side_completed.sum()
                self.metric_side_successes[side_index] += (
                    side_completed & self.success
                ).sum()
        self._signals = {
            **geometry,
            "left_force": left_force,
            "right_force": right_force,
            "left_grasp": left_grasp,
            "right_grasp": right_grasp,
            "transferred": liquid_step.transferred,
            "spilled": liquid_step.spilled,
            "success_now": success_now,
            "terminal": (terminated | truncated).clone(),
            "terminal_success": self.success.clone(),
            "terminal_failure": self.failure.clone(),
            "terminal_phase": self.task_phase.clone(),
            "terminal_steps": self.episode_length_buf.clone(),
            "terminal_bottle_mass": self.liquid.bottle.clone(),
            "terminal_cup_mass": self.liquid.cup.clone(),
            "terminal_spill_mass": self.liquid.spill.clone(),
            "terminal_cup_pose": torch.cat(
                [geometry["cup_pos"], geometry["cup_quat"]], dim=1
            ).clone(),
            "terminal_bottle_pose": torch.cat(
                [geometry["bottle_pos"], geometry["bottle_quat"]], dim=1
            ).clone(),
            **{
                f"success_condition/{name}": value.clone()
                for name, value in conditions.items()
            },
        }
        # DirectRLEnv resets completed environments before it builds the next
        # observation. Keep immutable terminal evidence separate from the live
        # signal cache so evaluation never reads post-reset object state.
        self._terminal_signals = {
            name: value.clone() for name, value in self._signals.items()
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        geometry = self._signals or self._geometry()
        position_error, rotation_error = self._reference_errors(geometry)
        arm_errors = []
        for side in ("left", "right"):
            arm_errors.append(
                (
                    self.robot.data.joint_pos[:, self.arm_ids[side]]
                    - self._arm_reference(side)
                ).norm(dim=1)
            )
        arm_error = torch.stack(arm_errors, dim=1)
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            arm_error = arm_error.gather(1, self.single_side[:, None]).squeeze(1)
        else:
            arm_error = arm_error.mean(dim=1)
        object_error = position_error.norm(dim=1) + 0.1 * rotation_error.norm(dim=1)
        trajectory_error = torch.where(
            self.task_phase >= int(PourPhase.ALIGN),
            object_error + 0.1 * arm_error,
            arm_error,
        )
        contact_error = self._contact_error(geometry)
        trajectory = torch.exp(-trajectory_error / 0.50)
        left_pad = (
            self._signals["left_force"] > self.cfg.contact_force_thresh
        ).float().mean(dim=1)
        right_pad = (
            self._signals["right_force"] > self.cfg.contact_force_thresh
        ).float().mean(dim=1)
        pad_score = torch.stack([left_pad, right_pad], dim=1)
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            pad_score = pad_score.gather(1, self.single_side[:, None]).squeeze(1)
            confidence = torch.tensor(
                self.reference.video_contact_confidence,
                dtype=torch.float32,
                device=self.device,
            )[self.single_side]
        else:
            pad_score = pad_score.mean(dim=1)
            confidence = torch.full_like(
                pad_score, float(self.reference.video_contact_confidence.mean())
            )
        contact = torch.exp(-contact_error / 0.05) * pad_score * confidence
        transfer = (self.liquid.cup - self.previous_cup_mass).clamp(min=0.0)
        spilled = (self.liquid.spill - self.previous_spill_mass).clamp(min=0.0)
        self.previous_cup_mass.copy_(self.liquid.cup)
        self.previous_spill_mass.copy_(self.liquid.spill)
        align = torch.exp(-geometry["radial_error"] / 0.04)
        gates = reward_phase_gates(self.task_phase)
        success_gate = gates["success"]
        if self.curriculum_stage < CurriculumStage.ALIGN_POUR:
            success_gate = success_gate | (
                self.task_phase == int(PourPhase.DUAL_GRASP)
            )
        terms = {
            "trajectory": trajectory * gates["trajectory"].float(),
            "contact": contact * gates["contact"].float(),
            "align": align * gates["align"].float(),
            "transfer": transfer * gates["transfer"].float(),
            "spill": spilled * gates["spill"].float(),
            "cup_upright": torch.cos(geometry["cup_tilt"]).clamp(min=0.0)
            * gates["cup_upright"].float(),
            "return_upright": torch.exp(-geometry["bottle_tilt"] / 0.35)
            * gates["return_upright"].float(),
            "action_rate": (self.actions - self.previous_actions).square().mean(dim=1),
            "success": self.success.float() * success_gate.float(),
        }
        reward, self._reward_terms = compute_reward(
            terms, ablation=self.ablation, weights=self.cfg.reward
        )
        success_rate = self.metric_successes.float() / self.metric_episodes.clamp(min=1)
        left_rate = self.metric_side_successes[0].float() / self.metric_side_episodes[0].clamp(min=1)
        right_rate = self.metric_side_successes[1].float() / self.metric_side_episodes[1].clamp(min=1)
        scalar_log = {
            "episodes": self.metric_episodes.float(),
            "success_rate": success_rate,
            "left_grasp_rate": left_rate,
            "right_grasp_rate": right_rate,
            "cup_fraction": (
                self.liquid.cup / self.cfg.liquid.initial_mass
            ).mean(),
            "spill_fraction": (
                self.liquid.spill / self.cfg.liquid.initial_mass
            ).mean(),
            **{f"reward/{name}": value.mean() for name, value in self._reward_terms.items()},
        }
        self.extras.update(scalar_log)
        self.extras["log"] = scalar_log
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        geometry = self._signals or self._geometry()
        origins = self.scene.env_origins
        velocity_scale = self.cfg.obs_vel_scale
        parts = []
        proprio = []
        for side in ("left", "right"):
            q = torch.cat(
                [
                    self.robot.data.joint_pos[:, self.arm_ids[side]],
                    self.robot.data.joint_pos[:, self.finger_ids[side]],
                ],
                dim=1,
            )
            qd = torch.cat(
                [
                    self.robot.data.joint_vel[:, self.arm_ids[side]],
                    self.robot.data.joint_vel[:, self.finger_ids[side]],
                ],
                dim=1,
            )
            # Actor proprioception includes both position and velocity for all
            # 7 arm + 22 hand joints on each side (116 values total).
            parts.extend([q, qd * velocity_scale])
            proprio.append(torch.cat([q, qd * velocity_scale], dim=1))
        for side in ("left", "right"):
            parts.append(
                torch.cat(
                    [
                        self.robot.data.body_pos_w[:, self.ee_ids[side]] - origins,
                        self.robot.data.body_quat_w[:, self.ee_ids[side]],
                        self.robot.data.body_lin_vel_w[:, self.ee_ids[side]],
                        self.robot.data.body_ang_vel_w[:, self.ee_ids[side]] * velocity_scale,
                    ],
                    dim=1,
                )
            )
        for name, asset in (("cup", self.cup), ("bottle", self.bottle)):
            parts.append(
                torch.cat(
                    [
                        geometry[f"{name}_pos"],
                        geometry[f"{name}_quat"],
                        asset.data.root_lin_vel_w,
                        asset.data.root_ang_vel_w * velocity_scale,
                    ],
                    dim=1,
                )
            )
        parts.extend(
            [
                geometry["mouth_rel"],
                geometry["cup_axis"],
                geometry["bottle_axis"],
                self._contact_magnitudes("left"),
                self._contact_magnitudes("right"),
            ]
        )
        phase_one_hot = torch.nn.functional.one_hot(
            self.task_phase, PourPhase.count()
        ).float()
        timeout = torch.tensor(
            self.cfg.phase_timeout, dtype=torch.float32, device=self.device
        )[self.task_phase].clamp(min=1.0)
        stage_one_hot = torch.nn.functional.one_hot(
            torch.full_like(self.task_phase, int(self.curriculum_stage)),
            len(CurriculumStage),
        ).float()
        target_side = torch.zeros(self.num_envs, 2, device=self.device)
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            target_side.scatter_(1, self.single_side[:, None], 1.0)
        parts.extend(
            [
                phase_one_hot,
                (self.phase_step.float() / timeout)[:, None],
                stage_one_hot,
                target_side,
            ]
        )

        index = self.reference_step.clamp(max=len(self.video_phase) - 1)
        next_index = (index + 1).clamp(max=len(self.video_phase) - 1)
        for side in ("left", "right"):
            arm_q = self.robot.data.joint_pos[:, self.arm_ids[side]]
            if self.ablation.use_video_trajectory:
                parts.append(self.q_video[side][index] - arm_q)
            else:
                parts.append(torch.zeros_like(arm_q))
        for side in ("left", "right"):
            if self.ablation.use_video_trajectory:
                parts.append(self.q_video[side][next_index] - self.q_video[side][index])
            else:
                parts.append(torch.zeros(self.num_envs, 7, device=self.device))
        parts.extend(
            [
                self.closure / self.cfg.closure_max,
                self.finger_delta.reshape(self.num_envs, 10) / self.cfg.finger_delta_max,
                self.actions,
            ]
        )
        position_error, rotation_error = self._reference_errors(geometry)
        if not self.ablation.use_video_trajectory:
            position_error = torch.zeros_like(position_error)
            rotation_error = torch.zeros_like(rotation_error)
        parts.extend([position_error, rotation_error])
        obs = torch.cat(parts, dim=1).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num()
        if obs.shape[1] != self.cfg.observation_space:
            raise RuntimeError(
                f"pour observation has {obs.shape[1]} values, expected {self.cfg.observation_space}"
            )

        cup_mass = self.cup.root_physx_view.get_masses().view(self.num_envs, 1)
        bottle_mass = self.bottle.root_physx_view.get_masses().view(self.num_envs, 1)
        priv = torch.cat(
            [
                cup_mass,
                bottle_mass,
                torch.full_like(cup_mass, float(self.manifest.cup.friction)),
                torch.full_like(bottle_mass, float(self.manifest.bottle.friction)),
                self.liquid.bottle[:, None],
                self.liquid.cup[:, None],
                self.liquid.spill[:, None],
                self._contact_magnitudes("left"),
                self._contact_magnitudes("right"),
                geometry["radial_error"][:, None],
                geometry["mouth_height"][:, None],
                geometry["bottle_tilt"][:, None],
                geometry["cup_tilt"][:, None],
                self.grasp_latched.float(),
            ],
            dim=1,
        ).nan_to_num()
        if priv.shape[1] != self.cfg.priv_info_dim:
            raise RuntimeError(
                f"pour privileged observation has {priv.shape[1]} values, expected {self.cfg.priv_info_dim}"
            )
        proprio_now = torch.cat(proprio, dim=1)
        self.proprio_hist[:, :-1].copy_(self.proprio_hist[:, 1:].clone())
        self.proprio_hist[:, -1] = proprio_now
        return {"policy": obs, "priv_info": priv, "proprio_hist": self.proprio_hist}

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        ids = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        super()._reset_idx(ids)
        count = len(ids)
        default = self.robot.data.default_joint_pos[ids].clone()
        start_phase = (
            PourPhase.STANCE
            if self.curriculum_stage in (
                CurriculumStage.APPROACH_GRASP,
                CurriculumStage.FULL,
            )
            else PourPhase.DUAL_GRASP
        )
        if start_phase == PourPhase.DUAL_GRASP:
            for side in ("left", "right"):
                pool = self.arm_start_pool[side]
                sample = torch.randint(
                    len(pool), (count,), dtype=torch.long, device=self.device
                )
                default[:, self.arm_ids[side]] = pool[sample]
        self.robot.write_joint_state_to_sim(default, torch.zeros_like(default), env_ids=ids)
        self.robot.set_joint_position_target(default, env_ids=ids)
        origins = self.scene.env_origins[ids]
        for asset, spec in ((self.cup, self.manifest.cup), (self.bottle, self.manifest.bottle)):
            pose = torch.tensor(
                spec.initial_pose_wxyz, dtype=torch.float32, device=self.device
            ).expand(count, 7).clone()
            pose[:, :3] += origins
            asset.write_root_pose_to_sim(pose, ids)
            asset.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.device), ids)

        self.actions[ids] = 0.0
        self.previous_actions[ids] = 0.0
        self.task_phase[ids] = int(start_phase)
        self.phase_step[ids] = 0
        self.reference_step[ids] = 0
        self.closure[ids] = 0.0
        self.finger_delta[ids] = 0.0
        self.grasp_hold[ids] = 0
        self.contact_loss[ids] = 0
        self.verify_hold[ids] = 0
        self.grasp_latched[ids] = False
        self.reset_count[ids] += 1
        id_tensor = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.single_side[ids] = (id_tensor + self.reset_count[ids]) % 2
        self.success[ids] = False
        self.failure[ids] = int(FailureCode.NONE)
        self.liquid.bottle[ids] = self.cfg.liquid.initial_mass
        self.liquid.cup[ids] = 0.0
        self.liquid.spill[ids] = 0.0
        self.previous_cup_mass[ids] = 0.0
        self.previous_spill_mass[ids] = 0.0
        self.proprio_hist[ids] = 0.0
        for side in ("left", "right"):
            q = default[:, self.arm_ids[side]]
            self.arm_target[side][ids] = q
            self.arm_target_previous[side][ids] = q
            self.finger_target[side][ids] = self.open_q[side]
        # A partial vector reset invalidates the cached geometry for the whole
        # batch. Terminal evidence remains available in _terminal_signals.
        self._signals = {}
