# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for BlockSparseTSDFIntegrator."""

import json
import struct
from types import SimpleNamespace

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.perception.mapper.projector_texture import (
    ProjectiveTextureProjector,
    ProjectiveTextureProjectorCfg,
    _projective_texture_atlas_columns,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp
from curobo.tests._src.perception.mapper.conftest import make_observation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def warp_init():
    """Initialize Warp once per module."""
    init_warp()
    return True


@pytest.fixture
def device():
    return "cuda:0"


# =============================================================================
# Tests
# =============================================================================


class TestBlockSparseTSDFIntegrator:
    """Tests for the high-level integrator interface."""

    def test_projective_texture_atlas_wraps_wide_camera_sets(self):
        """Large texture frame sets pack into rows instead of one oversized strip."""
        atlas_columns = _projective_texture_atlas_columns(
            total_cameras=64,
            image_width=500,
        )
        assert atlas_columns == 32

        projector = ProjectiveTextureProjector(
            tsdf=SimpleNamespace(device="cpu"),
            renderer=SimpleNamespace(),
            config=ProjectiveTextureProjectorCfg(
                texture_num_cameras=1,
                image_height=2,
                image_width=3,
                depth_minimum_distance=0.1,
                depth_maximum_distance=5.0,
                voxel_size=0.01,
            ),
        )

        batches = []
        for camera_idx in range(5):
            rgb = torch.full((1, 2, 3, 3), camera_idx, dtype=torch.uint8)
            batches.append((rgb, None, None, None, None))

        atlas = projector._build_projective_texture_atlas(
            batches,
            atlas_columns=3,
        )

        assert atlas.shape == (4, 9, 3)
        assert atlas[0, 0, 0] == 0
        assert atlas[0, 3, 0] == 1
        assert atlas[0, 6, 0] == 2
        assert atlas[2, 0, 0] == 3
        assert atlas[2, 3, 0] == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_projective_texture_mixed_cpu_depth_moves_to_cuda(self, warp_init, device):
        """Combine supplied CPU depth with CUDA-rendered visibility depth."""
        image_height = 8
        image_width = 8
        integrator = BlockSparseTSDFIntegrator(
            BlockSparseTSDFIntegratorCfg(
                voxel_size=0.02,
                origin=torch.zeros(3, dtype=torch.float32),
                grid_shape=(32, 32, 32),
                max_blocks=32,
                device=device,
                image_height=image_height,
                image_width=image_width,
                texture_num_cameras=2,
            )
        )
        intrinsics = torch.tensor(
            [[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        pose = Pose(
            position=torch.zeros(3, dtype=torch.float32),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        )
        rgb = torch.zeros((image_height, image_width, 3), dtype=torch.uint8)
        supplied_depth = torch.ones((image_height, image_width), dtype=torch.float32)
        observations = [
            CameraObservation(
                rgb_image=rgb,
                depth_image=supplied_depth,
                pose=pose,
                intrinsics=intrinsics,
            ),
            CameraObservation(
                rgb_image=rgb,
                pose=pose,
                intrinsics=intrinsics,
            ),
        ]

        batches = integrator._texture_projector._normalize_projective_texture_observations(
            observations
        )

        assert len(batches) == 1
        visibility_depth = batches[0][1]
        assert visibility_depth.device.type == "cuda"
        torch.testing.assert_close(visibility_depth[0], supplied_depth.to(device=device))
        assert torch.count_nonzero(visibility_depth[1]) == 0

    def test_initialization(self, warp_init, device):
        """Test integrator initialization."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        assert integrator.tsdf is not None
        assert integrator.memory_usage_mb() > 0

    def test_visible_capacity_defaults_to_max_blocks(self, warp_init, device):
        """Omitted per-frame visible capacity preserves prior max_blocks behavior."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=123,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        assert config.max_visible_blocks_per_integration == 123
        assert integrator._camera_integrator.max_visible_blocks_per_integration == 123
        assert integrator._camera_integrator.pool_indices.shape == (123,)

    @pytest.mark.parametrize("capacity", [0, -1, 101])
    def test_visible_capacity_validation(self, warp_init, device, capacity):
        """Visible capacity must be positive and no larger than max_blocks."""
        with pytest.raises(ValueError, match="max_visible_blocks_per_integration"):
            BlockSparseTSDFIntegratorCfg(
                voxel_size=0.01,
                origin=torch.tensor([0.0, 0.0, 0.0]),
                grid_shape=(512, 512, 512),
                max_blocks=100,
                max_visible_blocks_per_integration=capacity,
                device=device,
                image_height=32,
                image_width=32,
            )

    def test_support_capacity_validation(self, warp_init, device):
        """Support capacity must be a positive construction-time value."""
        with pytest.raises(ValueError, match="max_support_pixels_per_block_camera"):
            BlockSparseTSDFIntegratorCfg(
                voxel_size=0.01,
                origin=torch.tensor([0.0, 0.0, 0.0]),
                grid_shape=(512, 512, 512),
                max_blocks=100,
                max_support_pixels_per_block_camera=0,
                device=device,
                image_height=32,
                image_width=32,
            )

    def test_visible_capacity_overflow_raises_loudly(self, warp_init, device):
        """Frames that discover more than C visible blocks must not truncate."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            max_visible_blocks_per_integration=1,
            feature_dim=3,
            feature_grid_height=1,
            feature_grid_width=1,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)
        integrator.tsdf.data.block_data.fill_(7.0)
        integrator.tsdf.data.block_grid_rgb.fill_(5.0)
        integrator.tsdf.data.block_features.fill_(3.0)
        integrator.tsdf.data.block_feature_weight.fill_(2.0)

        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        with pytest.raises(
            ValueError,
            match=(
                "num_visible_blocks=.* exceeds "
                "max_visible_blocks_per_integration=1"
            ),
        ):
            integrator.integrate(
                make_observation(depth, rgb, position, quaternion, intrinsics)
            )

        n_allocated = int(integrator.tsdf.data.num_allocated.item())
        assert n_allocated > 0
        assert torch.count_nonzero(integrator.tsdf.data.block_data[:n_allocated]).item() == 0
        assert torch.count_nonzero(integrator.tsdf.data.block_grid_rgb[:n_allocated]).item() == 0
        assert (
            torch.count_nonzero(integrator.tsdf.data.block_features[:n_allocated]).item()
            == 0
        )
        assert (
            torch.count_nonzero(
                integrator.tsdf.data.block_feature_weight[:n_allocated]
            ).item()
            == 0
        )

    def test_integrate(self, warp_init, device):
        """Test depth integration via integrate method."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Create test data
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate
        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        stats = integrator.get_stats()
        assert stats["num_allocated"] > 0
        assert stats["frame_count"] == 1

    def test_extract_mesh(self, warp_init, device):
        """Test mesh extraction."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        # Integrate multiple frames
        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 200, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        # Extract mesh
        mesh = integrator.extract_mesh()

        assert mesh.vertices is not None
        assert mesh.faces is not None

    def test_extract_mesh_tensors(self, warp_init, device):
        """Test raw tensor mesh extraction."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        vertices, triangles, normals, colors = integrator.extract_mesh_tensors()

        assert vertices.dtype == torch.float32
        assert triangles.dtype == torch.int32
        assert normals.dtype == torch.float32
        assert colors.dtype == torch.uint8
        if triangles.shape[0] > 0:
            assert vertices.shape[0] == triangles.shape[0] * 3
            assert torch.equal(
                triangles.reshape(-1),
                torch.arange(vertices.shape[0], dtype=torch.int32, device=device),
            )

    def test_extract_textured_mesh_rgb_only(self, warp_init, device, tmp_path):
        """Render visibility depth and project mesh UVs from RGB and camera data."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
            texture_camera_image_height=32,
            texture_camera_image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 64, 64
        tex_H, tex_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 10, dtype=torch.uint8, device=device)
        texture_rgb = torch.zeros((tex_H, tex_W, 3), dtype=torch.uint8, device=device)
        texture_rgb[..., 0] = torch.arange(tex_W, dtype=torch.uint8, device=device).view(
            1, tex_W
        )
        texture_rgb[..., 1] = torch.arange(tex_H, dtype=torch.uint8, device=device).view(
            tex_H, 1
        )
        texture_rgb[..., 2] = 200
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        texture_intrinsics = torch.tensor(
            [[250.0, 0.0, 16.0], [0.0, 250.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        texture_obs = make_observation(
            depth[:tex_H, :tex_W],
            texture_rgb,
            position,
            quaternion,
            texture_intrinsics,
        )
        texture_obs.depth_image = None
        textured_mesh = integrator.extract_textured_mesh(texture_obs)

        assert textured_mesh.vertices is not None
        assert textured_mesh.faces is not None
        assert textured_mesh.texture_uvs is not None
        assert textured_mesh.texture_image is not None
        assert textured_mesh.vertex_colors is not None
        assert textured_mesh.texture_uvs.shape == (textured_mesh.vertices.shape[0], 2)
        assert textured_mesh.texture_image.shape[0] >= tex_H
        assert textured_mesh.texture_image.shape[1:] == (tex_W, 3)
        assert torch.equal(textured_mesh.texture_image[:tex_H], texture_rgb)
        if textured_mesh.vertices.shape[0] > 0:
            assert torch.all(textured_mesh.texture_uvs >= 0.0)
            assert torch.all(textured_mesh.texture_uvs <= 1.0)
            assert torch.any(textured_mesh.texture_uvs > 0.0)
            assert torch.any(textured_mesh.vertex_colors[..., 2] > 0.5)
            projected = textured_mesh.vertex_colors[..., 2] > 0.5
            projected_v = textured_mesh.texture_uvs[projected, 1]
            if textured_mesh.texture_image.shape[0] > tex_H:
                projected_v = (
                    projected_v
                    * float(textured_mesh.texture_image.shape[0])
                    / float(tex_H)
                )
            expected_green = torch.round(projected_v * (tex_H - 1)) / 255.0
            assert torch.mean(
                torch.abs(textured_mesh.vertex_colors[projected, 1] - expected_green)
            ) < 0.02
            trimesh_mesh = textured_mesh.get_trimesh_mesh(process=False)
            assert type(trimesh_mesh.visual).__name__ == "TextureVisuals"
            mesh_path = tmp_path / "textured_mesh.glb"
            textured_mesh.save_as_mesh(str(mesh_path))
            glb = mesh_path.read_bytes()
            json_len, _json_type = struct.unpack_from("<II", glb, 12)
            gltf = json.loads(glb[20 : 20 + json_len].decode("utf-8"))
            primitive_attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
            assert "TEXCOORD_0" in primitive_attrs
            assert gltf.get("images") and "bufferView" in gltf["images"][0]

    def test_textured_mesh_falls_back_to_voxel_color_patches(self, warp_init, device):
        """Unprojected triangles get valid UVs into appended voxel-color texture patches."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=500,
            device=device,
            image_height=64,
            image_width=64,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 64, 64
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 10, dtype=torch.uint8, device=device)
        texture_rgb = torch.zeros((img_H, img_W, 3), dtype=torch.uint8, device=device)
        texture_rgb[..., 2] = 200
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        texture_obs = make_observation(depth, texture_rgb, position, quaternion, intrinsics)
        textured_mesh = integrator.extract_textured_mesh(
            texture_obs,
            camera_max_distance=0.2,
        )

        assert textured_mesh.vertices is not None
        assert textured_mesh.texture_uvs is not None
        assert textured_mesh.texture_image is not None
        assert textured_mesh.vertex_colors is not None
        assert textured_mesh.vertices.shape[0] > 0
        assert textured_mesh.texture_uvs.shape == (textured_mesh.vertices.shape[0], 2)
        assert textured_mesh.texture_image.shape[0] > img_H
        assert textured_mesh.texture_image.shape[1:] == (img_W, 3)
        assert torch.equal(textured_mesh.texture_image[:img_H], texture_rgb)
        assert torch.all(textured_mesh.texture_uvs >= 0.0)
        assert torch.all(textured_mesh.texture_uvs <= 1.0)
        fallback_v0 = float(img_H) / textured_mesh.texture_image.shape[0]
        assert torch.all(textured_mesh.texture_uvs[:, 1] >= fallback_v0)
        assert torch.all(textured_mesh.vertex_colors[..., 2] < 0.1)

        px = torch.floor(
            textured_mesh.texture_uvs[:, 0] * textured_mesh.texture_image.shape[1]
        ).to(torch.long)
        py = torch.floor(
            textured_mesh.texture_uvs[:, 1] * textured_mesh.texture_image.shape[0]
        ).to(torch.long)
        px = px.clamp(0, textured_mesh.texture_image.shape[1] - 1)
        py = py.clamp(0, textured_mesh.texture_image.shape[0] - 1)
        sampled_colors = textured_mesh.texture_image[py, px].float() / 255.0
        assert torch.max(torch.abs(sampled_colors - textured_mesh.vertex_colors)) < 1.0e-4

    def test_reset(self, warp_init, device):
        """Test integrator reset."""
        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 1.0, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 16.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        integrator.integrate(make_observation(depth, rgb, position, quaternion, intrinsics))

        assert integrator.get_stats()["num_allocated"] > 0

        integrator.reset()

        assert integrator.get_stats()["num_allocated"] == 0
        assert integrator.get_stats()["frame_count"] == 0

    def test_decay_recycle_functions(self, warp_init, device):
        """Test decay and recycle launch paths."""
        from curobo._src.perception.mapper.kernel.wp_decay import (
            decay_and_recycle,
            launch_recycle,
        )

        config = BlockSparseTSDFIntegratorCfg(
            voxel_size=0.01,
            origin=torch.tensor([0.0, 0.0, 0.0]),
            grid_shape=(512, 512, 512),
            max_blocks=100,
            device=device,
            image_height=32,
            image_width=32,
        )
        integrator = BlockSparseTSDFIntegrator(config)

        img_H, img_W = 32, 32
        depth = torch.full((img_H, img_W), 0.5, dtype=torch.float32, device=device)
        rgb = torch.full((img_H, img_W, 3), 128, dtype=torch.uint8, device=device)
        position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        intrinsics = torch.tensor(
            [[500.0, 0.0, 16.0], [0.0, 500.0, 16.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )

        # Integrate a few frames
        obs = make_observation(depth, rgb, position, quaternion, intrinsics)
        for _ in range(3):
            integrator.integrate(obs)

        initial_blocks = integrator.get_stats()["num_allocated"]
        assert initial_blocks > 0

        # Test decay_and_recycle
        recycled = decay_and_recycle(integrator._tsdf, 0.5)
        assert recycled >= 0

        # Test recycle-only launch path.
        launch_recycle(integrator._tsdf)

        # Heavily decay to trigger recycling
        for _ in range(20):
            decay_and_recycle(integrator._tsdf, 0.5)

        # Test decay_and_recycle (with aggressive factor)
        recycled = decay_and_recycle(integrator._tsdf, 0.1)
        assert recycled >= 0  # May or may not recycle depending on thresholds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
