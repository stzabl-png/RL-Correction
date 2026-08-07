from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tasks.pour.core import (
    ACTION_DIM,
    AblationMode,
    FailureCode,
    LiquidProxyConfig,
    LiquidState,
    PourPhase,
    RewardWeights,
    combine_success,
    compute_reward,
    failure_codes,
    reward_phase_gates,
    split_bimanual_action,
    step_liquid_proxy,
)
from tasks.pour.mirroring import mirror_pose29, mirror_prior_for_left, mirrored_joint_order
from tasks.pour.tracker import EpisodeRecord, SuccessTracker


class CoreTest(unittest.TestCase):
    def test_action_split_is_left_then_right(self):
        action = torch.arange(2 * ACTION_DIM, dtype=torch.float32).reshape(2, ACTION_DIM)
        left, right = split_bimanual_action(action)
        self.assertEqual(left.shape, (2, 13))
        self.assertEqual(right.shape, (2, 13))
        torch.testing.assert_close(left, action[:, :13])
        torch.testing.assert_close(right, action[:, 13:])
        with self.assertRaises(ValueError):
            split_bimanual_action(torch.zeros(2, 25))

    def test_liquid_proxy_conserves_mass_and_routes_spill(self):
        cfg = LiquidProxyConfig(flow_rate_per_second=1.0, cup_radius_m=0.05)
        state = LiquidState.full(2, device="cpu")
        bottle_axis = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        cup_up = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        cup = torch.zeros(2, 3)
        mouth = torch.tensor([[0.0, 0.0, 0.10], [0.20, 0.0, 0.10]])
        step = step_liquid_proxy(
            state,
            bottle_up_axis_w=bottle_axis,
            bottle_mouth_pos_w=mouth,
            cup_center_pos_w=cup,
            cup_up_axis_w=cup_up,
            in_pour_phase=torch.ones(2, dtype=torch.bool),
            dt=0.1,
            cfg=cfg,
        )
        torch.testing.assert_close(step.state.total(), torch.ones(2))
        self.assertGreater(float(step.transferred[0]), 0.0)
        self.assertEqual(float(step.transferred[1]), 0.0)
        self.assertGreater(float(step.spilled[1]), 0.0)

    def test_liquid_does_not_flow_outside_pour_phase(self):
        state = LiquidState.full(1, device="cpu")
        step = step_liquid_proxy(
            state,
            bottle_up_axis_w=torch.tensor([[1.0, 0.0, 0.0]]),
            bottle_mouth_pos_w=torch.tensor([[0.0, 0.0, 0.1]]),
            cup_center_pos_w=torch.zeros(1, 3),
            cup_up_axis_w=torch.tensor([[0.0, 0.0, 1.0]]),
            in_pour_phase=torch.zeros(1, dtype=torch.bool),
            dt=1.0,
            cfg=LiquidProxyConfig(),
        )
        torch.testing.assert_close(step.state.bottle, state.bottle)

    def test_ablation_only_removes_registered_video_terms(self):
        terms = {name: torch.ones(3) for name in (
            "trajectory", "contact", "align", "transfer", "spill",
            "cup_upright", "return_upright", "action_rate", "success"
        )}
        _, full = compute_reward(terms, ablation=AblationMode.FULL, weights=RewardWeights())
        _, pure = compute_reward(terms, ablation=AblationMode.PURE_RL, weights=RewardWeights())
        self.assertTrue(torch.all(full["trajectory"] > 0))
        self.assertTrue(torch.all(full["contact"] > 0))
        self.assertTrue(torch.all(pure["trajectory"] == 0))
        self.assertTrue(torch.all(pure["contact"] == 0))
        torch.testing.assert_close(full["transfer"], pure["transfer"])

    def test_failure_precedence_is_deterministic(self):
        yes = torch.tensor([True])
        no = torch.tensor([False])
        code = failure_codes(
            left_drop=yes,
            right_drop=yes,
            cup_tip=yes,
            align_miss=yes,
            spill=yes,
            timeout=yes,
        )
        self.assertEqual(int(code[0]), int(FailureCode.LEFT_DROP))
        code = failure_codes(
            left_drop=no,
            right_drop=no,
            cup_tip=no,
            align_miss=no,
            spill=yes,
            timeout=yes,
        )
        self.assertEqual(int(code[0]), int(FailureCode.SPILL))

    def test_reward_phase_gates_do_not_leak_late_rewards_into_grasp(self):
        phase = torch.tensor(
            [
                int(PourPhase.APPROACH),
                int(PourPhase.DUAL_GRASP),
                int(PourPhase.POUR),
                int(PourPhase.RETURN),
            ]
        )
        gates = reward_phase_gates(phase)
        self.assertEqual(gates["contact"].tolist(), [False, True, True, True])
        self.assertEqual(gates["transfer"].tolist(), [False, False, True, False])
        self.assertEqual(
            gates["return_upright"].tolist(), [False, False, False, True]
        )

    def test_success_combines_all_conditions(self):
        result = combine_success({
            "a": torch.tensor([True, True]),
            "b": torch.tensor([True, False]),
        })
        self.assertEqual(result.tolist(), [True, False])


class MirroringTest(unittest.TestCase):
    def test_pose_mirror_is_an_involution(self):
        pose = np.zeros(29, dtype=np.float64)
        pose[:3] = [0.1, 0.2, 0.3]
        pose[3:7] = [0.9238795, 0.0, 0.3826834, 0.0]
        pose[7:] = np.linspace(-0.2, 0.3, 22)
        np.testing.assert_allclose(mirror_pose29(mirror_pose29(pose)), pose, atol=1.0e-6)

    def test_joint_names_swap_without_reordering(self):
        self.assertEqual(
            mirrored_joint_order(["right_thumb_MCP", "right_index_PIP"]),
            ["left_thumb_MCP", "left_index_PIP"],
        )

    def test_mirrored_prior_is_marked_unapproved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "right.npz"
            target = Path(directory) / "left.npz"
            pose = np.zeros(29)
            pose[3] = 1.0
            np.savez(
                source,
                grasp=pose,
                squeeze=pose,
                pregrasp=pose[None],
                contact_pos=np.zeros((2, 3)),
                contact_normal=np.tile([0.0, 0.0, 1.0], (2, 1)),
                contact_centroid=np.zeros(3),
                canon_rot=np.array([1.0, 0.0, 0.0, 0.0]),
                hand_side=np.frombuffer(b"right", dtype=np.uint8),
            )
            mirror_prior_for_left(source, target)
            with np.load(target) as data:
                self.assertEqual(int(data["mirror_requires_physical_screen"]), 1)
                self.assertEqual(data["hand_side"].tobytes().decode("utf-8"), "left")


class TrackerTest(unittest.TestCase):
    def test_tracker_writes_stable_failure_name_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = SuccessTracker(Path(directory) / "episodes.jsonl")
            tracker.append([
                EpisodeRecord(
                    run_name="unit", demo_id="11", seed=42, episode=0,
                    success=False, terminal_phase=PourPhase.POUR.name.lower(),
                    failure_code=int(FailureCode.SPILL), steps=100,
                    left_grasp=True, right_grasp=True, cup_fraction=0.5,
                    spill_fraction=0.5, cup_tilt_deg=2.0, bottle_tilt_deg=80.0,
                    trajectory_return=1.0, contact_return=1.0,
                ),
                EpisodeRecord(
                    run_name="unit", demo_id="11", seed=42, episode=1,
                    success=True, terminal_phase=PourPhase.VERIFY.name.lower(),
                    failure_code=int(FailureCode.NONE), steps=120,
                    left_grasp=True, right_grasp=True, cup_fraction=0.9,
                    spill_fraction=0.1, cup_tilt_deg=2.0, bottle_tilt_deg=5.0,
                    trajectory_return=1.0, contact_return=1.0,
                ),
            ])
            summary = tracker.summary()
            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["failure_counts"], {"spill": 1})


if __name__ == "__main__":
    unittest.main()
