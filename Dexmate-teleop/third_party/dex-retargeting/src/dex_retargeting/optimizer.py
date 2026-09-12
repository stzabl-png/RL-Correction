from abc import abstractmethod
from typing import List, Optional

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        # reduction="none" so we can apply an optional per-vector weight below. With the
        # default weight (all ones) the weighted mean == reduction="mean", i.e. numerically
        # identical to upstream. Set `self.weight` (len == #vectors) to up-weight specific
        # vectors, e.g. proximal abduction vectors that would otherwise be averaged away.
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.weight = np.ones(len(target_task_link_names), dtype=np.float32)
        # Loss = (per_vec_huber * weight).sum() / loss_denominator. Default denominator =
        # #vectors makes it a plain mean == upstream reduction="mean". When AUXILIARY
        # vectors are appended (e.g. abduction vectors) the denominator should stay the
        # PRIMARY (fingertip) count so the primary vectors keep their original /N weight
        # and are NOT diluted (which would weaken flexion tracking relative to the fixed
        # norm_delta regularization). Set via builder alongside self.weight.
        self.loss_denominator = float(len(target_task_link_names))
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Step 5b (fj): CONDITIONAL pinch gating (mirrors DexPilotOptimizer.projected). A
        # "pinch vector" (thumb_tip -> finger_tip, marked in self.pinch_mask by the builder)
        # gets self.pinch_weight applied ONLY while the human thumb<->finger gap is small
        # (a real pinch), with project/escape hysteresis. Otherwise its weight is 0, so
        # non-pinch poses (open/fist/flexion) stay byte-identical to the no-pinch baseline
        # -> the good A1 flexion is untouched. Defaults below == no-op (no pinch vectors).
        self.pinch_mask = np.zeros(len(target_task_link_names), dtype=bool)
        self.pinch_weight = 0.0
        self.pinch_project_dist = 0.03  # engage when human gap < 3 cm
        self.pinch_escape_dist = 0.05   # release when human gap > 5 cm (hysteresis)
        self.pinch_projected = np.zeros(len(target_task_link_names), dtype=bool)

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        # Step 5b (fj): per-frame CONDITIONAL pinch weight. A pinch vector engages
        # self.pinch_weight only while the human thumb<->finger gap (the target vector's
        # length) is small, with project/escape hysteresis; otherwise its weight is 0.
        # Non-pinch vectors keep self.weight. With pinch_weight==0 (default) or no pinch
        # mask this is a no-op and eff_weight == self.weight -> byte-identical baseline.
        eff_weight = np.asarray(self.weight, dtype=np.float32).copy()
        if self.pinch_weight > 0.0 and self.pinch_mask.any():
            gap = np.linalg.norm(np.asarray(target_vector, dtype=np.float64), axis=1)
            self.pinch_projected[self.pinch_mask & (gap < self.pinch_project_dist)] = True
            self.pinch_projected[self.pinch_mask & (gap > self.pinch_escape_dist)] = False
            engaged = self.pinch_mask & self.pinch_projected
            eff_weight[self.pinch_mask] = 0.0
            eff_weight[engaged] = self.pinch_weight

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error.
            # Per-vector Huber, then weighted mean. weight all-ones == plain mean (upstream).
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            per_vec = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            weight = torch.as_tensor(eff_weight, dtype=per_vec.dtype)
            huber_distance = (per_vec * weight).sum() / self.loss_denominator
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = target_vector * self.scaling  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (6, 3)
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )  # (10, 3)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return result

        return objective
