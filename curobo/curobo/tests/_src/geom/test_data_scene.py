# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aggregate scene obstacle data."""

from unittest.mock import Mock

from curobo._src.geom.data.data_scene import SceneData


def test_enable_obstacle_routes_to_storage_containing_name() -> None:
    """Test that obstacle enablement uses the storage containing its name."""
    cuboids = Mock()
    cuboids.has_name.return_value = False
    meshes = Mock()
    meshes.has_name.return_value = True
    scene_data = SceneData(cuboids=cuboids, meshes=meshes)

    scene_data.enable_obstacle("mesh_obstacle", enabled=False, env_idx=0)

    cuboids.has_name.assert_called_once_with("mesh_obstacle", 0)
    meshes.has_name.assert_called_once_with("mesh_obstacle", 0)
    cuboids.get_names.assert_not_called()
    meshes.get_names.assert_not_called()
    cuboids.set_enabled.assert_not_called()
    meshes.set_enabled.assert_called_once_with("mesh_obstacle", False, 0)


def test_update_obstacle_pose_routes_to_storage_containing_name() -> None:
    """Test that pose updates use the storage containing the obstacle name."""
    cuboids = Mock()
    cuboids.has_name.return_value = False
    meshes = Mock()
    meshes.has_name.return_value = True
    pose = Mock()
    scene_data = SceneData(cuboids=cuboids, meshes=meshes)

    scene_data.update_obstacle_pose("mesh_obstacle", pose=pose, env_idx=1)

    cuboids.has_name.assert_called_once_with("mesh_obstacle", 1)
    meshes.has_name.assert_called_once_with("mesh_obstacle", 1)
    cuboids.get_names.assert_not_called()
    meshes.get_names.assert_not_called()
    cuboids.update_pose.assert_not_called()
    meshes.update_pose.assert_called_once_with("mesh_obstacle", w_obj_pose=pose, env_idx=1)
