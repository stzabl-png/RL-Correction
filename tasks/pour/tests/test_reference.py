from __future__ import annotations

import tempfile
import unittest
import json
import hashlib
from pathlib import Path

import numpy as np

from tasks.pour.approve_grasps import approve_grasps
from tasks.pour.enums import PourPhase
from tasks.pour.reference import PourReference, infer_video_phases
from tasks.pour.scene import PourObjectSpec, PourSceneManifest
from tasks.pour.stage_reconstruction import merge_reconstruction


def _rot_y(degrees: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(degrees)
    value = np.zeros((len(angle), 4, 4), dtype=np.float64)
    value[:, 3, 3] = 1.0
    value[:, 1, 1] = 1.0
    value[:, 0, 0] = np.cos(angle)
    value[:, 0, 2] = np.sin(angle)
    value[:, 2, 0] = -np.sin(angle)
    value[:, 2, 2] = np.cos(angle)
    return value


class ReferenceTest(unittest.TestCase):
    @staticmethod
    def _pending_reference(length: int = 4) -> PourReference:
        return PourReference(
            demo_id="11",
            fps=20.0,
            camera_intrinsic=np.eye(3),
            camera_pose=np.tile([0, 0, 0, 1, 0, 0, 0], (length, 1)),
            left_wrist=np.tile([0, 0, 0, 1, 0, 0, 0], (length, 1)),
            right_wrist=np.tile([0, 0, 0, 1, 0, 0, 0], (length, 1)),
            left_joints=np.zeros((length, 25, 3)),
            right_joints=np.zeros((length, 25, 3)),
            video_phase=np.array([3, 3, 4, 5]),
            right_tilt_rad=np.zeros(length),
            hand_confidence=np.ones((length, 2)),
            source_frame=np.arange(length),
            bottle_pose=np.full((length, 7), np.nan),
            cup_pose=np.full((length, 7), np.nan),
            object_confidence=np.zeros((length, 2)),
            video_contact_left=np.empty((0, 3)),
            video_contact_right=np.empty((0, 3)),
            video_contact_confidence=np.zeros(2),
        )

    def test_phase_inference_selects_longest_sustained_tilt(self):
        degrees = np.r_[np.zeros(10), np.full(5, 65.0), np.zeros(10), np.full(20, 75.0), np.zeros(8)]
        phase, _, (start, end) = infer_video_phases(_rot_y(degrees))
        self.assertEqual((start, end), (25, 44))
        self.assertTrue(np.all(phase[start : end + 1] == int(PourPhase.POUR)))
        self.assertTrue(np.all(phase[end + 1 :] == int(PourPhase.RETURN)))

    def test_reference_roundtrip_preserves_missing_reconstruction_gate(self):
        length = 4
        artifact = self._pending_reference(length)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            artifact.save(path)
            loaded = PourReference.load(path)
            self.assertEqual(loaded.demo_id, "11")
            self.assertFalse(loaded.training_ready)

    def test_reconstruction_merge_enforces_frame_and_marks_manual_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.npz"
            scene_path = root / "scene.json"
            self._pending_reference().save(reference_path)
            obj = PourObjectSpec(
                label="pending", mesh="mesh.obj", usd="asset.usd", mass_kg=None,
                friction=None, initial_pose_wxyz=None, opening_center_local=None,
                opening_axis_local=None, opening_radius_m=None,
            )
            PourSceneManifest(
                demo_id="11", status="pending_reconstruction",
                reference_npz="reference.npz", left_grasp_prior="",
                right_grasp_prior="", cup=obj, bottle=obj,
            ).save(scene_path)
            frame = np.arange(4, dtype=np.float32)
            cup_pose = np.tile([0, 0, 0, 1, 0, 0, 0], (4, 1))
            bottle_pose = np.tile([0.2, 0, 0, 1, 0, 0, 0], (4, 1))
            for name, pose in (("cup", cup_pose), ("bottle", bottle_pose)):
                np.savez(
                    root / f"{name}.npz", pose=pose, source_frame=frame,
                    confidence=np.full(4, 0.9),
                    coordinate_frame=np.array("arkit_world"),
                    quaternion_order=np.array("wxyz"),
                )
            geometry = {
                "schema_version": 1,
                "demo_id": "11",
                "cup": {
                    "label": "black cup", "mesh": "cup.obj", "usd": "cup.usd",
                    "mass_kg": 0.2, "friction": 0.5,
                    "initial_pose_wxyz": [0, 0, 1, 1, 0, 0, 0],
                    "opening_center_local": [0, 0, 0.08],
                    "opening_axis_local": [0, 0, 1], "opening_radius_m": 0.04,
                },
                "bottle": {
                    "label": "brown bottle", "mesh": "bottle.obj", "usd": "bottle.usd",
                    "mass_kg": 0.4, "friction": 0.6,
                    "initial_pose_wxyz": [0.2, 0, 1, 1, 0, 0, 0],
                    "opening_center_local": [0, 0, 0.12],
                    "opening_axis_local": [0, 0, 1], "opening_radius_m": 0.02,
                },
                "contacts": {
                    "coordinate_frame": "object_local",
                    "left": {"object": "cup", "points": [[0.02, 0, 0.03]], "confidence": 0.8},
                    "right": {"object": "bottle", "points": [[0.01, 0, 0.04]], "confidence": 0.9},
                },
            }
            geometry_path = root / "geometry.json"
            geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
            merged, scene = merge_reconstruction(
                reference_path, scene_path, cup_track_path=root / "cup.npz",
                bottle_track_path=root / "bottle.npz", geometry_path=geometry_path,
            )
            self.assertTrue(merged.training_ready)
            self.assertEqual(scene.status, "pending_grasp_approval")
            np.testing.assert_allclose(
                scene.video_to_sim_wxyz[:3], [0, 0, 1], atol=1.0e-7
            )
            for asset in ("cup.obj", "cup.usd", "bottle.obj", "bottle.usd"):
                (root / asset).write_bytes(b"asset")
            paths = {}
            for side, object_name in (("left", "cup"), ("right", "bottle")):
                prior = root / f"{side}_prior.npz"
                grasp = np.zeros(29)
                grasp[3] = 1.0
                np.savez(
                    prior, grasp=grasp, pregrasp=grasp[None],
                    contact_centroid=np.zeros(3),
                    hand_side=np.frombuffer(side.encode("utf-8"), dtype=np.uint8),
                )
                digest = hashlib.sha256(prior.read_bytes()).hexdigest()
                report = root / f"{side}_report.json"
                report.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "hand_side": side,
                            "object": object_name,
                            "template": "3_Medium_Wrap",
                            "approved_by": "test-human",
                            "prior_sha256": digest,
                            "gate1_pass": True,
                            "gate2_pass": True,
                            "stable_lift_pass": True,
                            "pads_star": 4.0,
                            "q_star": 0.1,
                            "object_lift_m": 0.02,
                        }
                    ),
                    encoding="utf-8",
                )
                paths[f"{side}_prior"] = prior
                paths[f"{side}_report"] = report
            approved = approve_grasps(
                scene_path,
                left_prior_path=paths["left_prior"],
                right_prior_path=paths["right_prior"],
                left_report_path=paths["left_report"],
                right_report_path=paths["right_report"],
            )
            self.assertEqual(approved.status, "ready")
            self.assertEqual(approved.grasp_approval["left"]["approved_by"], "test-human")

    def test_scene_manifest_refuses_unapproved_training(self):
        obj = PourObjectSpec(
            label="object", mesh="", usd="", mass_kg=0.2, friction=0.5,
            initial_pose_wxyz=[0, 0, 0.9, 1, 0, 0, 0],
            opening_center_local=[0, 0, 0.1], opening_axis_local=[0, 0, 1],
            opening_radius_m=0.03,
        )
        manifest = PourSceneManifest(
            demo_id="11", status="pending_reconstruction", reference_npz="",
            left_grasp_prior="", right_grasp_prior="", cup=obj, bottle=obj,
        )
        manifest.validate(require_assets=False)
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            manifest.validate(require_assets=True)


if __name__ == "__main__":
    unittest.main()
