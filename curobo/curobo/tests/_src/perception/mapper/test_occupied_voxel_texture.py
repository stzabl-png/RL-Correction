# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Focused tests for occupied voxel subvoxel and texture output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import BlockSparseTSDFIntegrator
from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.projector_texture import (
    ProjectiveTextureProjector,
    ProjectiveTextureProjectorCfg,
)
from curobo._src.perception.mapper.storage import OccupiedVoxels
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


class _FakeBlockData:
    """Minimal block-data view for color accessor tests."""

    def __init__(self, rgbw_by_block: torch.Tensor) -> None:
        self.rgbw_by_block = rgbw_by_block

    def sample_rgbw_at_centers(
        self,
        centers: torch.Tensor,
        block_idx_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        del centers
        return self.rgbw_by_block[block_idx_per_voxel.long()].to(torch.float32)


class _FakeVoxelIntegrator:
    """Capture Mapper occupied-voxel extraction calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = object()

    def extract_occupied_voxels(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.result


def _make_pose(num_cameras: int = 1) -> Pose:
    position = torch.zeros((num_cameras, 3), dtype=torch.float32)
    quaternion = torch.zeros((num_cameras, 4), dtype=torch.float32)
    quaternion[:, 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def _make_intrinsics(num_cameras: int = 1) -> torch.Tensor:
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1)
    intrinsics[:, 0, 0] = 1.0
    intrinsics[:, 1, 1] = 1.0
    intrinsics[:, 0, 2] = 1.0
    intrinsics[:, 1, 2] = 1.0
    if num_cameras == 1:
        return intrinsics[0]
    return intrinsics


def _make_mapper(
    *,
    texture_num_cameras: int = 1,
    image_shape: tuple[int, int] = (3, 3),
) -> tuple[Mapper, _FakeVoxelIntegrator, list[dict[str, Any]]]:
    mapper = Mapper.__new__(Mapper)
    fake_integrator = _FakeVoxelIntegrator()
    render_calls: list[dict[str, Any]] = []
    mapper.config = SimpleNamespace(
        texture_num_cameras=texture_num_cameras,
        texture_camera_image_height=image_shape[0],
        texture_camera_image_width=image_shape[1],
    )
    mapper._device = torch.device("cpu")
    mapper._integrator = fake_integrator

    def render_depth(
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape_arg: tuple[int, int],
    ) -> torch.Tensor:
        render_calls.append(
            {
                "intrinsics": intrinsics,
                "pose": pose,
                "image_shape": image_shape_arg,
            }
        )
        return torch.ones((int(intrinsics.shape[0]), *image_shape_arg), dtype=torch.float32)

    mapper.render_depth = render_depth
    return mapper, fake_integrator, render_calls


def _make_integrator() -> BlockSparseTSDFIntegrator:
    integrator = BlockSparseTSDFIntegrator.__new__(BlockSparseTSDFIntegrator)
    integrator.config = SimpleNamespace(
        voxel_size=0.4,
        texture_num_cameras=1,
        texture_camera_image_height=3,
        texture_camera_image_width=3,
        depth_minimum_distance=0.1,
        depth_maximum_distance=5.0,
    )
    integrator._tsdf = SimpleNamespace(device="cpu")
    return integrator


def _make_texture_projector() -> ProjectiveTextureProjector:
    return ProjectiveTextureProjector(
        tsdf=SimpleNamespace(device="cpu"),
        renderer=SimpleNamespace(),
        config=ProjectiveTextureProjectorCfg(
            texture_num_cameras=1,
            image_height=3,
            image_width=3,
            depth_minimum_distance=0.1,
            depth_maximum_distance=5.0,
            voxel_size=0.4,
        ),
    )


def test_mapper_extract_occupied_voxels_defaults_to_surface_only() -> None:
    mapper, fake_integrator, _render_calls = _make_mapper()

    result = mapper.extract_occupied_voxels()

    assert result is fake_integrator.result
    assert fake_integrator.calls[0]["surface_only"] is True
    assert fake_integrator.calls[0]["sdf_threshold"] is None
    assert fake_integrator.calls[0]["subvoxel_factor"] == 1
    assert fake_integrator.calls[0]["texture_observations"] is None


def test_mapper_extract_occupied_voxels_preserves_explicit_full_occupied_path() -> None:
    mapper, fake_integrator, _render_calls = _make_mapper()

    mapper.extract_occupied_voxels(surface_only=False, sdf_threshold=0.2)

    assert fake_integrator.calls[0]["surface_only"] is False
    assert fake_integrator.calls[0]["sdf_threshold"] == 0.2


def test_subvoxel_factor_one_preserves_base_voxel_output() -> None:
    integrator = _make_integrator()
    centers = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    blocks = torch.tensor([7], dtype=torch.int32)
    voxels = OccupiedVoxels(centers, blocks, _FakeBlockData(torch.ones((8, 4))))

    expanded = integrator._expand_subvoxels(voxels, subvoxel_factor=1)

    torch.testing.assert_close(expanded.centers, centers)
    assert torch.equal(expanded.block_idx_per_voxel, blocks)
    assert expanded.texture_colors is None
    assert expanded.texture_valid is None
    assert expanded.subvoxel_factor == 1


def test_subvoxel_factor_two_count_geometry_and_repeated_block_indices() -> None:
    integrator = _make_integrator()
    centers = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    blocks = torch.tensor([7], dtype=torch.int32)
    voxels = OccupiedVoxels(centers, blocks, _FakeBlockData(torch.ones((8, 4))))

    expanded = integrator._expand_subvoxels(voxels, subvoxel_factor=2)

    assert expanded.centers.shape == (8, 3)
    assert torch.equal(expanded.block_idx_per_voxel, blocks.repeat_interleave(8))
    torch.testing.assert_close(expanded.centers.mean(dim=0), centers[0])
    expected_offsets = torch.tensor([-0.1, 0.1], dtype=torch.float32)
    for axis in range(3):
        actual_offsets = torch.unique(expanded.centers[:, axis] - centers[0, axis])
        torch.testing.assert_close(actual_offsets, expected_offsets)


def test_max_points_caps_after_subvoxel_expansion_by_downsampling_source_voxels() -> None:
    integrator = _make_integrator()
    centers = torch.arange(15, dtype=torch.float32).view(5, 3)
    blocks = torch.arange(5, dtype=torch.int32)
    voxels = OccupiedVoxels(centers, blocks, _FakeBlockData(torch.ones((8, 4))))

    limited = integrator._limit_voxels_for_point_cap(
        voxels,
        subvoxel_factor=2,
        max_points=16,
    )
    expanded = integrator._expand_subvoxels(limited, subvoxel_factor=2)

    assert expanded.centers.shape == (16, 3)
    assert torch.equal(expanded.block_idx_per_voxel, torch.tensor([0, 4]).repeat_interleave(8))


def test_invalid_subvoxel_factor_raises() -> None:
    with pytest.raises(ValueError, match="subvoxel_factor"):
        BlockSparseTSDFIntegrator._validate_subvoxel_factor(0)


def test_colors_uint8_prefers_texture_only_when_requested_and_valid() -> None:
    rgbw = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    voxels = OccupiedVoxels(
        centers=torch.zeros((3, 3), dtype=torch.float32),
        block_idx_per_voxel=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_data=_FakeBlockData(rgbw),
        texture_colors=torch.tensor(
            [[0, 255, 0], [255, 255, 0], [255, 0, 255]],
            dtype=torch.uint8,
        ),
        texture_valid=torch.tensor([True, False, False]),
    )

    torch.testing.assert_close(
        voxels.colors_uint8(),
        torch.tensor([[0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        voxels.colors_uint8(prefer_texture=False),
        torch.tensor([[255, 0, 0], [0, 0, 255], [128, 128, 128]], dtype=torch.uint8),
    )


def test_occupied_voxel_texture_uses_visibility_and_falls_back_to_persistent_color() -> None:
    projector = _make_texture_projector()
    rgbw = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    voxels = OccupiedVoxels(
        centers=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        block_idx_per_voxel=torch.tensor([0, 1], dtype=torch.int32),
        block_data=_FakeBlockData(rgbw),
    )
    rgb = torch.zeros((1, 3, 3, 3), dtype=torch.uint8)
    rgb[0, 1, 1] = torch.tensor([0, 255, 0], dtype=torch.uint8)
    depth = torch.ones((1, 3, 3), dtype=torch.float32)
    observation = CameraObservation(
        rgb_image=rgb,
        depth_image=depth,
        pose=_make_pose(),
        intrinsics=_make_intrinsics().unsqueeze(0),
    )

    textured = projector.texture_occupied_voxels(
        voxels,
        [observation],
        camera_min_distance=None,
        camera_max_distance=None,
        texture_depth_tolerance_m=None,
    )

    assert textured.texture_valid is not None
    assert textured.texture_valid.tolist() == [True, False]
    torch.testing.assert_close(
        textured.colors_uint8(),
        torch.tensor([[0, 255, 0], [0, 0, 255]], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        textured.colors_uint8(prefer_texture=False),
        torch.tensor([[255, 0, 0], [0, 0, 255]], dtype=torch.uint8),
    )


def test_mapper_delegates_rgbd_voxel_texture_without_rendering() -> None:
    mapper, fake_integrator, render_calls = _make_mapper()
    rgb = torch.zeros((3, 3, 3), dtype=torch.uint8)
    depth = torch.ones((3, 3), dtype=torch.float32)
    observation = CameraObservation(
        rgb_image=rgb,
        depth_image=depth,
        pose=_make_pose(),
        intrinsics=_make_intrinsics(),
    )

    mapper.extract_occupied_voxels(texture_observations=observation)

    assert render_calls == []
    assert fake_integrator.calls[0]["texture_observations"] is observation


def test_mapper_delegates_rgb_only_voxel_texture_without_rendering() -> None:
    mapper, fake_integrator, render_calls = _make_mapper(texture_num_cameras=2)
    observation = CameraObservation(
        rgb_image=torch.zeros((3, 3, 3), dtype=torch.uint8),
        pose=_make_pose(),
        intrinsics=_make_intrinsics(),
    )

    mapper.extract_occupied_voxels(texture_observations=observation)

    assert render_calls == []
    assert fake_integrator.calls[0]["texture_observations"] is observation
    assert observation.depth_image is None
