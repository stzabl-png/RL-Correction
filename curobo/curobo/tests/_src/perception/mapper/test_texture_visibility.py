# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Focused tests for texture visibility preparation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.projector_texture import (
    ProjectiveTextureProjector,
    ProjectiveTextureProjectorCfg,
    _texture_depth_visible,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


class _FakeTextureIntegrator:
    """Capture texture mesh extraction calls from :class:`Mapper`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def extract_textured_mesh(self, **kwargs: Any) -> "_FakeTextureIntegrator":
        self.calls.append(kwargs)
        return self


class _FakeTextureRenderer:
    """Capture visibility-depth renders from the TSDF integrator."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def render_depth(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        self.calls.append(
            {
                "intrinsics": intrinsics,
                "pose": pose,
                "image_shape": image_shape,
            }
        )
        return torch.full(
            (intrinsics.shape[0], *image_shape),
            1.25,
            dtype=torch.float32,
            device=intrinsics.device,
        )


def _make_pose(num_cameras: int = 1) -> Pose:
    position = torch.zeros((num_cameras, 3), dtype=torch.float32)
    quaternion = torch.zeros((num_cameras, 4), dtype=torch.float32)
    quaternion[:, 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def _make_intrinsics(num_cameras: int = 1) -> torch.Tensor:
    intrinsics = torch.eye(3, dtype=torch.float32).repeat(num_cameras, 1, 1)
    intrinsics[:, 0, 0] = 40.0
    intrinsics[:, 1, 1] = 40.0
    intrinsics[:, 0, 2] = 2.0
    intrinsics[:, 1, 2] = 1.5
    if num_cameras == 1:
        return intrinsics[0]
    return intrinsics


def _make_mapper(
    *,
    texture_num_cameras: int = 1,
    image_shape: tuple[int, int] = (3, 4),
) -> tuple[Mapper, _FakeTextureIntegrator, list[dict[str, Any]]]:
    mapper = Mapper.__new__(Mapper)
    fake_integrator = _FakeTextureIntegrator()
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
        camera_count = int(intrinsics.shape[0])
        return torch.full((camera_count, *image_shape_arg), 1.25, dtype=torch.float32)

    mapper.render_depth = render_depth
    return mapper, fake_integrator, render_calls


def _make_texture_projector(
    *,
    texture_num_cameras: int = 1,
    image_shape: tuple[int, int] = (3, 4),
) -> tuple[ProjectiveTextureProjector, _FakeTextureRenderer]:
    renderer = _FakeTextureRenderer()
    projector = ProjectiveTextureProjector(
        tsdf=SimpleNamespace(device="cpu"),
        renderer=renderer,
        config=ProjectiveTextureProjectorCfg(
            texture_num_cameras=texture_num_cameras,
            image_height=image_shape[0],
            image_width=image_shape[1],
            depth_minimum_distance=0.1,
            depth_maximum_distance=5.0,
            voxel_size=0.01,
        ),
    )
    return projector, renderer


def _make_observation(
    *,
    rgb: torch.Tensor,
    depth: torch.Tensor | None = None,
    num_cameras: int = 1,
) -> CameraObservation:
    return CameraObservation(
        rgb_image=rgb,
        depth_image=depth,
        pose=_make_pose(num_cameras),
        intrinsics=_make_intrinsics(num_cameras),
    )


def test_mapper_delegates_rgbd_texture_without_rendering() -> None:
    mapper, fake_integrator, render_calls = _make_mapper()
    rgb = torch.zeros((3, 4, 3), dtype=torch.uint8)
    depth = torch.full((3, 4), 1.5, dtype=torch.float32)
    observation = _make_observation(rgb=rgb, depth=depth)

    result = mapper.extract_textured_mesh(
        observation,
        refine_iterations=2,
        surface_only=False,
        camera_min_distance=0.2,
        camera_max_distance=3.0,
        texture_depth_tolerance_m=0.03,
    )

    assert result is fake_integrator
    assert render_calls == []
    assert len(fake_integrator.calls) == 1
    call = fake_integrator.calls[0]
    assert call["refine_iterations"] == 2
    assert not call["surface_only"]
    assert call["camera_min_distance"] == 0.2
    assert call["camera_max_distance"] == 3.0
    assert call["texture_depth_tolerance_m"] == 0.03
    assert call["texture_observations"] is observation


def test_mapper_delegates_rgb_only_texture_without_rendering() -> None:
    mapper, fake_integrator, render_calls = _make_mapper(image_shape=(5, 6))
    rgb = torch.zeros((5, 6, 3), dtype=torch.uint8)
    observation = _make_observation(rgb=rgb)

    mapper.extract_textured_mesh(observation)

    assert render_calls == []
    assert fake_integrator.calls[0]["texture_observations"] is observation
    assert observation.depth_image is None


def test_projector_renders_missing_depth_for_batched_texture() -> None:
    projector, renderer = _make_texture_projector(
        texture_num_cameras=2,
        image_shape=(5, 6),
    )
    rgb = torch.zeros((2, 5, 6, 3), dtype=torch.uint8)
    observation = _make_observation(rgb=rgb, num_cameras=2)

    batches = projector._normalize_projective_texture_observations(observation)

    assert len(renderer.calls) == 1
    assert renderer.calls[0]["intrinsics"].shape == (2, 3, 3)
    assert renderer.calls[0]["pose"].position.shape == (2, 3)
    assert renderer.calls[0]["image_shape"] == (5, 6)
    assert len(batches) == 1
    torch.testing.assert_close(
        batches[0][1],
        torch.full((2, 5, 6), 1.25, dtype=torch.float32),
    )


def test_projector_renders_one_batch_for_unbatched_texture_group() -> None:
    projector, renderer = _make_texture_projector(
        texture_num_cameras=2,
        image_shape=(5, 6),
    )
    observations = [
        _make_observation(rgb=torch.zeros((5, 6, 3), dtype=torch.uint8)),
        _make_observation(rgb=torch.zeros((5, 6, 3), dtype=torch.uint8)),
    ]

    batches = projector._normalize_projective_texture_observations(observations)

    assert len(renderer.calls) == 1
    assert renderer.calls[0]["intrinsics"].shape == (2, 3, 3)
    assert len(batches) == 1
    assert batches[0][1].shape == (2, 5, 6)


def test_projector_preserves_provided_depth_in_partially_missing_batch() -> None:
    projector, renderer = _make_texture_projector(texture_num_cameras=2)
    provided_depth = torch.full((3, 4), 2.0, dtype=torch.float32)
    observations = [
        _make_observation(
            rgb=torch.zeros((3, 4, 3), dtype=torch.uint8),
            depth=provided_depth,
        ),
        _make_observation(rgb=torch.zeros((3, 4, 3), dtype=torch.uint8)),
    ]

    batches = projector._normalize_projective_texture_observations(observations)

    assert len(renderer.calls) == 1
    torch.testing.assert_close(batches[0][1][0], provided_depth)
    torch.testing.assert_close(
        batches[0][1][1],
        torch.full((3, 4), 1.25, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "depth",
    [
        torch.ones((3, 4, 1), dtype=torch.float32),
        torch.ones((3, 4), dtype=torch.int32),
    ],
)
def test_invalid_texture_depth_shape_or_dtype_raises(depth: torch.Tensor) -> None:
    projector, _renderer = _make_texture_projector()
    rgb = torch.zeros((3, 4, 3), dtype=torch.uint8)
    observation = _make_observation(rgb=rgb, depth=depth)

    with pytest.raises(ValueError, match="depth_image"):
        projector._normalize_projective_texture_observations(observation)


def test_incomplete_unbatched_texture_camera_group_raises() -> None:
    projector, _renderer = _make_texture_projector(texture_num_cameras=2)
    rgb = torch.zeros((3, 4, 3), dtype=torch.uint8)
    depth = torch.ones((3, 4), dtype=torch.float32)
    observation = _make_observation(rgb=rgb, depth=depth)

    with pytest.raises(ValueError, match="complete camera batches"):
        projector._normalize_projective_texture_observations(observation)


def test_texture_visibility_depth_policy_rejects_occluded_and_invalid_depth() -> None:
    projected_z = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0], dtype=torch.float32)
    visibility_depth = torch.tensor([1.015, 1.03, 0.0, 2.03, float("nan")])

    default_visible = _texture_depth_visible(
        projected_z,
        visibility_depth,
        voxel_size=0.01,
        texture_depth_tolerance_m=None,
    )
    explicit_visible = _texture_depth_visible(
        projected_z,
        visibility_depth,
        voxel_size=0.01,
        texture_depth_tolerance_m=0.05,
    )

    assert default_visible.tolist() == [True, False, False, False, False]
    assert explicit_visible.tolist() == [True, True, False, True, False]
