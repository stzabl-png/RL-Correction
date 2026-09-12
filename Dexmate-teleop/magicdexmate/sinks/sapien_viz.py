"""SAPIEN realtime visualization: Sharpa hand follows retargeted qpos, with the
human keypoint skeleton drawn beside it. Debug sink, needs a display.

Modeled on dex-retargeting example/vector_retargeting/show_realtime_retargeting.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SapienViz:
    def __init__(
        self,
        urdf_path: str | Path,
        retargeting_joint_names: list[str],
        show_keypoints: bool = True,
        keypoint_offset=(0.0, -0.28, 0.05),
    ):
        import sapien
        from sapien.utils import Viewer

        self._sapien = sapien
        sapien.render.set_viewer_shader_dir("default")

        scene = sapien.Scene()
        scene.add_directional_light(np.array([1, 1, -1]), np.array([2.5, 2.5, 2.5]))
        scene.add_point_light(np.array([1, 0, 1]), np.array([2, 2, 2]), shadow=False)
        scene.add_point_light(np.array([-1, -1, 1]), np.array([2, 2, 2]), shadow=False)
        self.scene = scene

        loader = scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        self.robot = loader.load(str(urdf_path))
        self.robot.set_pose(sapien.Pose([0, 0, 0]))

        sapien_joint_names = [j.get_name() for j in self.robot.get_active_joints()]
        self._idx = np.array(
            [list(retargeting_joint_names).index(n) for n in sapien_joint_names], dtype=int
        )

        self._kp_actors = []
        self._kp_offset = np.asarray(keypoint_offset)
        if show_keypoints:
            for i in range(21):
                builder = scene.create_actor_builder()
                color = [0.9, 0.15, 0.15, 1] if i in (4, 8, 12, 16, 20) else [0.15, 0.5, 0.9, 1]
                builder.add_sphere_visual(
                    radius=0.004, material=sapien.render.RenderMaterial(base_color=color)
                )
                self._kp_actors.append(builder.build_kinematic(name=f"kp_{i}"))

        self.viewer = Viewer()
        self.viewer.set_scene(scene)
        if hasattr(self.viewer, "set_camera_xyz"):
            self.viewer.set_camera_xyz(x=0.45, y=-0.15, z=0.25)
            self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)
        self.viewer.control_window.show_origin_frame = False
        self.viewer.control_window.move_speed = 0.01

    def update(self, qpos_pin_order: np.ndarray, kp: np.ndarray | None = None) -> bool:
        """Render one frame. Returns False once the viewer window is closed."""
        self.robot.set_qpos(np.asarray(qpos_pin_order)[self._idx])
        if kp is not None and self._kp_actors:
            for actor, p in zip(self._kp_actors, kp + self._kp_offset):
                actor.set_pose(self._sapien.Pose(p))
        self.viewer.render()
        return not getattr(self.viewer, "closed", False)
