# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Focused tests for block-sparse TSDF renderer batching."""

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.kernel.builder.builder_block_sparse_kernel import (
    make_block_sparse_kernels,
)
from curobo._src.perception.mapper.renderer import (
    BlockSparseTSDFRenderer,
    BlockSparseTSDFRendererCfg,
)
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation


def _make_pose(num_cameras: int = 1, device: str = "cpu") -> Pose:
    position = torch.zeros((num_cameras, 3), dtype=torch.float32, device=device)
    quaternion = torch.zeros((num_cameras, 4), dtype=torch.float32, device=device)
    quaternion[:, 0] = 1.0
    rotation = torch.eye(3, dtype=torch.float32, device=device).repeat(num_cameras, 1, 1)
    return Pose(position=position, quaternion=quaternion, rotation=rotation)


def _make_intrinsics(device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [[40.0, 0.0, 4.0], [0.0, 40.0, 3.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


class _ShapeRenderer(BlockSparseTSDFRenderer):
    """Renderer test double that reuses normalization and helper methods."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.config = BlockSparseTSDFRendererCfg()

    def render(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        render_inputs = self._normalize_render_inputs(intrinsics, pose)
        image_height, image_width = image_shape
        if render_inputs.single_camera:
            depth = torch.ones((image_height, image_width), dtype=torch.float32)
            normals = torch.zeros((image_height, image_width, 3), dtype=torch.float32)
            valid = torch.ones((image_height, image_width), dtype=torch.bool)
        else:
            depth = torch.ones(
                (render_inputs.camera_count, image_height, image_width), dtype=torch.float32
            )
            normals = torch.zeros(
                (render_inputs.camera_count, image_height, image_width, 3),
                dtype=torch.float32,
            )
            valid = torch.ones(
                (render_inputs.camera_count, image_height, image_width), dtype=torch.bool
            )
        normals[..., 2] = 1.0
        return depth, normals, valid

    def render_color(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        color = torch.full((*normals.shape[:-1], 3), 127, dtype=torch.uint8)
        return depth, normals, color, valid


class TestRendererInputNormalization:
    """Renderer inputs must be exact-batched without singleton broadcasting."""

    def test_accepts_unbatched_single_camera(self) -> None:
        renderer = _ShapeRenderer()
        render_inputs = renderer._normalize_render_inputs(_make_intrinsics(), _make_pose())

        assert render_inputs.camera_count == 1
        assert render_inputs.single_camera
        assert render_inputs.intrinsics.shape == (1, 3, 3)
        assert render_inputs.positions.shape == (1, 3)
        assert render_inputs.quaternions.shape == (1, 4)

    def test_accepts_exact_batched_intrinsics_and_pose(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = torch.stack((_make_intrinsics(), _make_intrinsics()))
        render_inputs = renderer._normalize_render_inputs(intrinsics, _make_pose(2))

        assert render_inputs.camera_count == 2
        assert not render_inputs.single_camera
        assert render_inputs.intrinsics.shape == (2, 3, 3)

    def test_accepts_exact_batched_vector_intrinsics(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = torch.tensor(
            [[40.0, 40.0, 4.0, 3.0], [35.0, 35.0, 4.0, 3.0]],
            dtype=torch.float32,
        )
        render_inputs = renderer._normalize_render_inputs(intrinsics, _make_pose(2))

        assert render_inputs.camera_count == 2
        assert render_inputs.intrinsics.shape == (2, 3, 3)
        torch.testing.assert_close(render_inputs.intrinsics[:, 2, 2], torch.ones(2))

    def test_rejects_batched_intrinsics_with_singleton_pose(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = torch.stack((_make_intrinsics(), _make_intrinsics()))

        with pytest.raises(ValueError, match="same camera count"):
            renderer._normalize_render_inputs(intrinsics, _make_pose())

    def test_rejects_unbatched_intrinsics_with_batched_pose(self) -> None:
        renderer = _ShapeRenderer()

        with pytest.raises(ValueError, match="same camera count"):
            renderer._normalize_render_inputs(_make_intrinsics(), _make_pose(2))


class TestRendererHelperShapes:
    """Convenience helpers should preserve the render shape convention."""

    def test_unbatched_single_camera_helper_shapes(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = _make_intrinsics()
        pose = _make_pose()
        image_shape = (3, 4)

        assert renderer.render_depth(intrinsics, pose, image_shape).shape == (3, 4)
        assert renderer.render_normals(intrinsics, pose, image_shape).shape == (3, 4, 3)
        assert renderer.render_color_only(intrinsics, pose, image_shape).shape == (3, 4, 3)
        assert renderer.render_shaded(intrinsics, pose, image_shape).shape == (3, 4, 3)
        assert renderer.render_depth_colormap(intrinsics, pose, image_shape).shape == (3, 4, 3)
        assert renderer.render_normal_colormap(intrinsics, pose, image_shape).shape == (
            3,
            4,
            3,
        )

    def test_batched_helper_shapes(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = torch.stack((_make_intrinsics(), _make_intrinsics()))
        pose = _make_pose(2)
        image_shape = (3, 4)

        assert renderer.render_depth(intrinsics, pose, image_shape).shape == (2, 3, 4)
        assert renderer.render_normals(intrinsics, pose, image_shape).shape == (2, 3, 4, 3)
        assert renderer.render_color_only(intrinsics, pose, image_shape).shape == (
            2,
            3,
            4,
            3,
        )
        assert renderer.render_shaded(intrinsics, pose, image_shape).shape == (2, 3, 4, 3)
        assert renderer.render_depth_colormap(intrinsics, pose, image_shape).shape == (
            2,
            3,
            4,
            3,
        )
        assert renderer.render_normal_colormap(intrinsics, pose, image_shape).shape == (
            2,
            3,
            4,
            3,
        )

    def test_explicit_batched_singleton_intrinsics_keep_leading_dimension(self) -> None:
        renderer = _ShapeRenderer()
        intrinsics = _make_intrinsics().unsqueeze(0)
        pose = _make_pose()

        assert renderer.render_depth(intrinsics, pose, (3, 4)).shape == (1, 3, 4)


def test_depth_only_buffer_allocation_does_not_create_color_buffer() -> None:
    renderer = BlockSparseTSDFRenderer.__new__(BlockSparseTSDFRenderer)
    renderer.device = "cpu"
    renderer._buffer_size = 0
    renderer._hit_points = None
    renderer._hit_normals = None
    renderer._hit_colors = None
    renderer._hit_depths = None
    renderer._hit_mask = None

    renderer._ensure_buffers(12)
    assert renderer._hit_colors is None

    renderer._ensure_buffers(12, include_color=True)
    assert renderer._hit_colors is not None
    assert renderer._hit_colors.shape == (12, 3)


def test_kernel_bundle_has_no_accelerated_raycast_exports() -> None:
    init_warp()
    kernels = make_block_sparse_kernels(block_size=8)

    assert hasattr(kernels, "raycast_block_sparse_kernel")
    assert hasattr(kernels, "raycast_block_sparse_color_kernel")
    assert not hasattr(kernels, "raycast_block_sparse_accelerated_kernel")
    assert not hasattr(kernels, "raycast_block_sparse_accelerated_color_kernel")
    assert "accelerated" not in kernels.raycast_block_sparse_kernel.key
    assert "accelerated" not in kernels.raycast_block_sparse_color_kernel.key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for renderer kernels")
def test_batched_render_matches_stacked_single_camera_renders() -> None:
    init_warp()
    device = "cuda:0"
    image_shape = (8, 8)
    intrinsics = _make_intrinsics(device)
    pose = _make_pose(device=device)

    integrator = BlockSparseTSDFIntegrator(
        BlockSparseTSDFIntegratorCfg(
            max_blocks=512,
            voxel_size=0.04,
            origin=torch.tensor([-0.8, -0.8, 0.0], dtype=torch.float32),
            grid_shape=(64, 64, 64),
            truncation_distance=0.12,
            device=device,
            image_height=image_shape[0],
            image_width=image_shape[1],
        )
    )
    depth = torch.full(image_shape, 0.8, dtype=torch.float32, device=device)
    rgb = torch.full((*image_shape, 3), 128, dtype=torch.uint8, device=device)
    integrator.integrate(
        make_observation(
            depth,
            rgb,
            pose.position[0],
            pose.quaternion[0],
            intrinsics,
        )
    )

    renderer = BlockSparseTSDFRenderer(integrator)
    batched_intrinsics = torch.stack((intrinsics, intrinsics))
    batched_pose = _make_pose(2, device=device)

    depth_b, normals_b, valid_b = renderer.render(
        batched_intrinsics, batched_pose, image_shape
    )
    depth_s, normals_s, valid_s = renderer.render(intrinsics, pose, image_shape)
    color_depth_b, color_normals_b, colors_b, color_valid_b = renderer.render_color(
        batched_intrinsics, batched_pose, image_shape
    )
    color_depth_s, color_normals_s, colors_s, color_valid_s = renderer.render_color(
        intrinsics, pose, image_shape
    )

    assert depth_b.shape == (2, *image_shape)
    assert normals_b.shape == (2, *image_shape, 3)
    assert valid_b.shape == (2, *image_shape)
    torch.testing.assert_close(depth_b, torch.stack((depth_s, depth_s)))
    torch.testing.assert_close(normals_b, torch.stack((normals_s, normals_s)))
    torch.testing.assert_close(valid_b, torch.stack((valid_s, valid_s)))
    torch.testing.assert_close(color_depth_b, torch.stack((color_depth_s, color_depth_s)))
    torch.testing.assert_close(color_normals_b, torch.stack((color_normals_s, color_normals_s)))
    torch.testing.assert_close(colors_b, torch.stack((colors_s, colors_s)))
    torch.testing.assert_close(color_valid_b, torch.stack((color_valid_s, color_valid_s)))
