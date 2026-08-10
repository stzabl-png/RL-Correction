from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from rl_rebuild.correction import clips, frames as F
from rl_rebuild.correction.recon_kailang.static_reconstruction import (
    first_interaction,
    load_static_reconstruction,
)
from rl_rebuild.correction.recon_kailang import stage_static_reconstruction as staging


class StaticReconstructionPlacementTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.mesh = root / "object.obj"
        vertices = np.array([
            [0.1, -0.2, -0.05],
            [0.3, -0.2, -0.05],
            [0.1, 0.0, -0.05],
            [0.3, 0.0, -0.05],
            [0.1, -0.2, 0.05],
            [0.3, -0.2, 0.05],
            [0.1, 0.0, 0.05],
            [0.3, 0.0, 0.05],
        ])
        self.mesh.write_text(
            "".join(f"v {x} {y} {z}\n" for x, y, z in vertices), encoding="utf-8"
        )

        frames = np.arange(4, dtype=np.int32)
        obj_pos = np.tile(np.array([10.0, 5.0, 0.9]), (4, 1))
        q = np.sqrt(0.5)
        obj_quat = np.tile(np.array([q, 0.0, 0.0, q]), (4, 1))
        joints_right = np.tile(np.array([10.2, 4.9, 1.0]), (4, 21, 1))
        joints_left = np.tile(np.array([9.8, 5.1, 1.0]), (4, 21, 1))
        self.replay = root / "replay_world.npz"
        np.savez_compressed(
            self.replay,
            frames=frames,
            fps=np.float32(15.0),
            joints_left=joints_left.astype(np.float32),
            joints_right=joints_right.astype(np.float32),
            valid_left=np.ones(4, dtype=np.float32),
            valid_right=np.ones(4, dtype=np.float32),
            obj_pose=np.concatenate([obj_pos, obj_quat], axis=1).astype(np.float32),
            phase_left=np.zeros(4, dtype=np.int8),
            phase_right=np.array([0, 0, 1, 1], dtype=np.int8),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_fp_pose_center_xy_and_table_gap(self):
        data_unit, placement = load_static_reconstruction(
            str(self.replay),
            str(self.mesh),
            table_height=0.85,
            obj_gap=0.002,
            target_hz=None,
            return_placement=True,
        )

        self.assertEqual(placement.hand, "right")
        self.assertEqual(placement.source_frame, 2)
        self.assertEqual(data_unit.ref.interaction_seg[0], placement.aligned_frame)
        np.testing.assert_allclose(placement.center_xy, placement.wrist_xy, atol=1e-6)
        self.assertLess(placement.center_error_m, 1e-6)
        self.assertAlmostEqual(placement.bottom_gap_m, 0.002, places=6)

        expected_q = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
        np.testing.assert_allclose(data_unit.object_init_pose[3:], expected_q, atol=1e-6)

        vertices = F.load_obj_verts(str(self.mesh))
        rotated = F.rot_apply(
            np.broadcast_to(data_unit.object_init_pose[3:], (len(vertices), 4)), vertices
        )
        world = rotated + data_unit.object_init_pose[:3]
        self.assertAlmostEqual(float(world[:, 2].min()), 0.852, places=6)

    def test_no_interaction_is_rejected(self):
        with np.load(self.replay, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["phase_right"] = np.zeros(4, dtype=np.int8)
        no_contact = Path(self.tmp.name) / "no_contact.npz"
        np.savez_compressed(no_contact, **payload)
        with np.load(no_contact, allow_pickle=True) as data:
            with self.assertRaisesRegex(ValueError, "no phase_left/right"):
                first_interaction(data)

    def test_simultaneous_start_requires_hand(self):
        with np.load(self.replay, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["phase_left"] = np.array([0, 0, 1, 1], dtype=np.int8)
        tied = Path(self.tmp.name) / "tied.npz"
        np.savez_compressed(tied, **payload)
        with np.load(tied, allow_pickle=True) as data:
            with self.assertRaisesRegex(ValueError, "both start"):
                first_interaction(data)
            self.assertEqual(first_interaction(data, hand="left"), ("left", 2))

    def test_mesh_outside_table_is_rejected(self):
        with np.load(self.replay, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["joints_right"] = payload["joints_right"].copy()
        payload["joints_right"][:, :, 0] = 10.9
        outside = Path(self.tmp.name) / "outside.npz"
        np.savez_compressed(outside, **payload)
        with self.assertRaisesRegex(ValueError, "exceeds table"):
            load_static_reconstruction(
                str(outside), str(self.mesh), table_half=0.6, target_hz=None)

    def test_explicit_valid_placement_frame_preserves_interaction_label(self):
        with np.load(self.replay, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["valid_right"] = np.array([0, 0, 0, 1], dtype=np.float32)
        payload["joints_right"] = payload["joints_right"].copy()
        payload["joints_right"][3, :, :2] = np.array([10.35, 4.75])
        fallback = Path(self.tmp.name) / "fallback.npz"
        np.savez_compressed(fallback, **payload)

        _, placement = load_static_reconstruction(
            str(fallback), str(self.mesh), hand="right", placement_frame=3,
            target_hz=None, return_placement=True,
        )

        self.assertEqual(placement.source_frame, 2)
        self.assertEqual(placement.placement_frame, 3)
        self.assertEqual(placement.aligned_frame, 3)

    def test_invalid_explicit_placement_frame_is_rejected(self):
        with np.load(self.replay, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["valid_right"] = np.array([0, 0, 0, 1], dtype=np.float32)
        fallback = Path(self.tmp.name) / "invalid_fallback.npz"
        np.savez_compressed(fallback, **payload)
        with self.assertRaisesRegex(ValueError, "is not a valid hand frame"):
            load_static_reconstruction(
                str(fallback), str(self.mesh), hand="right", placement_frame=2,
                target_hz=None,
            )

    def test_clip_semantics_override_placeholder_cfg_mass(self):
        cfg = SimpleNamespace(
            clip_name="",
            place_mode="",
            object_cfg=SimpleNamespace(
                spawn=SimpleNamespace(
                    usd_path="placeholder.usd",
                    mass_props=SimpleNamespace(mass=0.2),
                )
            ),
        )
        clips.configure_cfg(cfg, "water_bottle_twist_static")
        self.assertEqual(cfg.clip_name, "water_bottle_twist_static")
        self.assertAlmostEqual(cfg.object_cfg.spawn.mass_props.mass, 0.53)
        self.assertEqual(cfg.place_mode, "object_only")

    def test_bottle_mass_override_does_not_change_existing_static_clip(self):
        cfg = SimpleNamespace(
            clip_name="",
            place_mode="",
            object_cfg=SimpleNamespace(
                spawn=SimpleNamespace(
                    usd_path="placeholder.usd",
                    mass_props=SimpleNamespace(mass=0.2),
                )
            ),
        )
        clips.configure_cfg(cfg, "task1_static_smoke")
        self.assertAlmostEqual(cfg.object_cfg.spawn.mass_props.mass, 0.2)

    def test_static_staging_is_complete_and_atomic(self):
        root = Path(self.tmp.name)
        recon = root / "source_reconstruction"
        retarget = root / "source_retarget"
        recon.mkdir()
        retarget.mkdir()
        (recon / "object_mesh_scaled_final.obj").write_bytes(self.mesh.read_bytes())
        np.savez_compressed(recon / "world_fused.npz", ok=np.array([1]))
        (retarget / "replay_world.npz").write_bytes(self.replay.read_bytes())
        np.savez_compressed(retarget / "ref_qpos.npz", ok=np.array([1]))

        training_root = root / "TrainingData"
        with mock.patch.object(staging, "_TD", training_root):
            destination = staging.stage_static(
                "egodex",
                "task1_static_smoke",
                str(recon),
                str(retarget),
                "egodex/test/basic_pick_place/23",
            )

        self.assertTrue(
            (destination / "reconstruction" / "object_mesh_scaled_final.obj").is_file())
        self.assertTrue((destination / "retarget" / self.replay.name).is_file())
        self.assertTrue((destination / "retarget" / "ref_qpos.npz").is_file())
        self.assertTrue((destination / "cache").is_dir())
        self.assertTrue((destination / "meta.json").is_file())
        provenance = destination / "provenance" / "step4_staging_run_manifest.json"
        self.assertTrue(provenance.is_file())
        run_manifest = json.loads(provenance.read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["status"], "ready")
        self.assertEqual(run_manifest["stage"], "step4_static_reconstruction_staging")
        self.assertTrue(run_manifest["outputs"]["mesh"]["exists"])
        self.assertEqual(
            run_manifest["outputs"]["mesh"]["path"],
            "reconstruction/object_mesh_scaled_final.obj",
        )
        meta = json.loads((destination / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(
            meta["manifest"]["retarget"], ["replay_world.npz", "ref_qpos.npz"]
        )
        self.assertFalse(any(training_root.glob(".*.staging-*")))


if __name__ == "__main__":
    unittest.main()
