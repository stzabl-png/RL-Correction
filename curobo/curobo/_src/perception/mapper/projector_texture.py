# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Visibility-tested projective texturing for block-sparse TSDF outputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
import warp as wp

from curobo._src.geom.transform import torch_quaternion_to_matrix
from curobo._src.geom.types import Mesh
from curobo._src.perception.mapper.renderer import BlockSparseTSDFRenderer
from curobo._src.perception.mapper.storage import BlockSparseTSDF, OccupiedVoxels
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream


_PROJECTIVE_TEXTURE_ATLAS_MAX_WIDTH_PX = 16_384

TextureBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _projective_texture_atlas_columns(total_cameras: int, image_width: int) -> int:
    """Return camera tiles per atlas row for common GL texture limits."""
    if total_cameras <= 0:
        log_and_raise(f"total_cameras must be positive, got {total_cameras}.")
    if image_width <= 0:
        log_and_raise(f"image_width must be positive, got {image_width}.")
    max_columns = max(1, _PROJECTIVE_TEXTURE_ATLAS_MAX_WIDTH_PX // image_width)
    return min(total_cameras, max_columns)


def _texture_depth_visible(
    projected_z: torch.Tensor,
    visibility_depth: torch.Tensor,
    *,
    voxel_size: float,
    texture_depth_tolerance_m: Optional[float],
) -> torch.Tensor:
    """Return samples whose projected depth agrees with visibility depth."""
    valid = (
        torch.isfinite(projected_z)
        & torch.isfinite(visibility_depth)
        & (projected_z > 0.0)
        & (visibility_depth > 0.0)
    )
    if texture_depth_tolerance_m is None:
        tolerance = torch.maximum(
            torch.full_like(projected_z, float(2.0 * voxel_size)),
            projected_z * 0.01,
        )
    else:
        tolerance = torch.full_like(projected_z, float(texture_depth_tolerance_m))
    return valid & (torch.abs(projected_z - visibility_depth) <= tolerance)


def _validate_texture_depth_tolerance(
    texture_depth_tolerance_m: Optional[float],
) -> float:
    """Return the kernel tolerance, using ``-1`` for the adaptive policy."""
    if texture_depth_tolerance_m is None:
        return -1.0
    depth_tolerance = float(texture_depth_tolerance_m)
    if depth_tolerance < 0.0 or not math.isfinite(depth_tolerance):
        log_and_raise(
            "texture_depth_tolerance_m must be non-negative and finite or None, got "
            f"{texture_depth_tolerance_m}."
        )
    return depth_tolerance


@dataclass(frozen=True)
class ProjectiveTextureProjectorCfg:
    """Construction-time projective texture dimensions and depth limits."""

    #: Cameras in each exact texture projection batch.
    texture_num_cameras: int
    #: Height of every RGB and visibility-depth image.
    image_height: int
    #: Width of every RGB and visibility-depth image.
    image_width: int
    #: Default minimum positive camera-frame depth used for projection.
    depth_minimum_distance: float
    #: Default maximum camera-frame depth used for projection.
    depth_maximum_distance: float
    #: TSDF voxel size used by the adaptive visibility tolerance.
    voxel_size: float


@dataclass(frozen=True)
class _PreparedMeshTextureProjection:
    """Validated texture batches and atlas for one mesh projection."""

    batches: list[TextureBatch]
    texture_atlas: torch.Tensor
    atlas_columns: int
    total_cameras: int
    depth_tolerance: float


class ProjectiveTextureProjector:
    """Project RGB observations onto TSDF meshes and occupied voxel samples."""

    def __init__(
        self,
        tsdf: BlockSparseTSDF,
        renderer: BlockSparseTSDFRenderer,
        config: ProjectiveTextureProjectorCfg,
    ) -> None:
        """Initialize the projector from sparse storage and a depth renderer."""
        self._tsdf = tsdf
        self._renderer = renderer
        self.config = config

    def prepare_mesh_projection(
        self,
        texture_observations: CameraObservation | Sequence[CameraObservation],
        *,
        texture_depth_tolerance_m: Optional[float],
    ) -> _PreparedMeshTextureProjection:
        """Validate mesh texture inputs and build their camera atlas."""
        depth_tolerance = _validate_texture_depth_tolerance(
            texture_depth_tolerance_m
        )
        batches = self._normalize_projective_texture_observations(
            texture_observations
        )
        total_cameras = len(batches) * self.config.texture_num_cameras
        atlas_columns = _projective_texture_atlas_columns(
            total_cameras,
            self.config.image_width,
        )
        texture_atlas = self._build_projective_texture_atlas(
            batches,
            atlas_columns=atlas_columns,
        )
        return _PreparedMeshTextureProjection(
            batches=batches,
            texture_atlas=texture_atlas,
            atlas_columns=atlas_columns,
            total_cameras=total_cameras,
            depth_tolerance=depth_tolerance,
        )

    def project_mesh(
        self,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
        normals: torch.Tensor,
        colors: torch.Tensor,
        projection: _PreparedMeshTextureProjection,
        *,
        camera_min_distance: Optional[float],
        camera_max_distance: Optional[float],
    ) -> Mesh:
        """Project texture observations onto triangle-soup mesh tensors.

        Args:
            vertices: ``(3 * M, 3)`` float32 triangle-soup vertices.
            triangles: ``(M, 3)`` int32 identity triangle indices.
            normals: ``(3 * M, 3)`` float32 vertex normals.
            colors: ``(3 * M, 3)`` uint8 fallback vertex colors.
            projection: Prepared texture batches and atlas.
            camera_min_distance: Optional minimum camera-frame projection depth.
            camera_max_distance: Optional maximum camera-frame projection depth.

        Returns:
            Textured mesh with fallback atlas patches for unprojected triangles.
        """
        batches = projection.batches
        total_cameras = projection.total_cameras
        atlas_columns = projection.atlas_columns
        texture_atlas = projection.texture_atlas

        vertices = vertices.contiguous()
        triangles = triangles.contiguous()
        normals = normals.contiguous()
        colors = colors.contiguous()
        uvs = torch.full(
            (vertices.shape[0], 2),
            -1.0,
            dtype=torch.float32,
            device=vertices.device,
        )
        projection_scores = torch.full(
            (triangles.shape[0],),
            -1.0e20,
            dtype=torch.float32,
            device=vertices.device,
        )

        if vertices.shape[0] > 0:
            min_distance = (
                self.config.depth_minimum_distance
                if camera_min_distance is None
                else camera_min_distance
            )
            max_distance = (
                self.config.depth_maximum_distance
                if camera_max_distance is None
                else camera_max_distance
            )
            device, stream = get_warp_device_stream(vertices)
            for batch_idx, (
                _rgb,
                visibility_depth,
                intrinsics,
                positions,
                quaternions,
            ) in enumerate(batches):
                wp.launch(
                    self._tsdf.kernels.project_mesh_uvs_kernel,
                    dim=triangles.shape[0],
                    inputs=[
                        wp.from_torch(vertices, dtype=wp.vec3),
                        vertices.shape[0],
                        wp.from_torch(visibility_depth, dtype=wp.float32),
                        wp.from_torch(intrinsics, dtype=wp.float32),
                        wp.from_torch(positions, dtype=wp.float32),
                        wp.from_torch(quaternions, dtype=wp.float32),
                        float(min_distance),
                        float(max_distance),
                        projection.depth_tolerance,
                        batch_idx * self.config.texture_num_cameras,
                        total_cameras,
                        atlas_columns,
                        wp.from_torch(uvs, dtype=wp.vec2),
                        wp.from_torch(projection_scores, dtype=wp.float32),
                        wp.from_torch(colors, dtype=wp.vec3ub),
                        1,
                        self._tsdf.get_warp_data(),
                    ],
                    device=device,
                    stream=stream,
                )

        vertex_valid_uv = (
            (uvs[:, 0] >= 0.0)
            & (uvs[:, 0] <= 1.0)
            & (uvs[:, 1] >= 0.0)
            & (uvs[:, 1] <= 1.0)
        )
        triangle_valid_uv = vertex_valid_uv.view(-1, 3).all(dim=1)
        texture_vertex_valid = triangle_valid_uv.repeat_interleave(3)
        texture_uvs = torch.where(
            texture_vertex_valid[:, None],
            uvs.clamp(0.0, 1.0),
            torch.zeros_like(uvs),
        )
        vertex_colors = colors.clone()
        if vertices.shape[0] > 0 and bool(texture_vertex_valid.any().item()):
            atlas_height = texture_atlas.shape[0]
            atlas_width = texture_atlas.shape[1]
            px = torch.round(texture_uvs[:, 0] * (atlas_width - 1)).to(torch.long)
            py = torch.round(texture_uvs[:, 1] * (atlas_height - 1)).to(torch.long)
            px = px.clamp(0, atlas_width - 1)
            py = py.clamp(0, atlas_height - 1)
            vertex_colors[texture_vertex_valid] = texture_atlas[
                py[texture_vertex_valid], px[texture_vertex_valid]
            ]

        fallback_triangles = torch.nonzero(~triangle_valid_uv, as_tuple=False).reshape(-1)
        if vertices.shape[0] > 0 and fallback_triangles.numel() > 0:
            texture_atlas, texture_uvs = self._append_fallback_texture_patches(
                texture_atlas,
                texture_uvs,
                vertex_colors,
                texture_vertex_valid,
                fallback_triangles,
            )

        return Mesh(
            name="block_sparse_tsdf_textured_mesh",
            vertices=vertices,
            faces=triangles,
            vertex_colors=(vertex_colors.float() / 255.0),
            vertex_normals=normals,
            texture_uvs=texture_uvs,
            texture_image=texture_atlas,
        )

    def _append_fallback_texture_patches(
        self,
        texture_atlas: torch.Tensor,
        texture_uvs: torch.Tensor,
        vertex_colors: torch.Tensor,
        texture_vertex_valid: torch.Tensor,
        fallback_triangles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append 2x2 fallback patches and assign their triangle UVs."""
        atlas_height = int(texture_atlas.shape[0])
        atlas_width = int(texture_atlas.shape[1])
        patch_size = 2
        patches_per_row = max(1, atlas_width // patch_size)
        fallback_count = int(fallback_triangles.numel())
        fallback_rows = math.ceil(fallback_count / patches_per_row) * patch_size
        fallback_atlas = torch.zeros(
            (fallback_rows, atlas_width, 3),
            dtype=texture_atlas.dtype,
            device=texture_atlas.device,
        )

        fallback_slots = torch.arange(
            fallback_count,
            dtype=torch.long,
            device=vertex_colors.device,
        )
        patch_x = (fallback_slots % patches_per_row) * patch_size
        patch_y = (fallback_slots // patches_per_row) * patch_size
        tri_base = fallback_triangles * 3
        c0 = vertex_colors[tri_base]
        c1 = vertex_colors[tri_base + 1]
        c2 = vertex_colors[tri_base + 2]
        avg_color = (
            (c0.to(torch.float32) + c1.to(torch.float32) + c2.to(torch.float32))
            / 3.0
        ).round().to(torch.uint8)
        fallback_atlas[patch_y, patch_x] = c0
        fallback_atlas[patch_y, patch_x + 1] = c1
        fallback_atlas[patch_y + 1, patch_x] = c2
        fallback_atlas[patch_y + 1, patch_x + 1] = avg_color

        texture_atlas = torch.cat((texture_atlas, fallback_atlas), dim=0)
        new_atlas_height = int(texture_atlas.shape[0])
        if bool(texture_vertex_valid.any().item()):
            texture_uvs[texture_vertex_valid, 1] *= float(atlas_height) / float(
                new_atlas_height
            )

        fallback_uvs = torch.empty(
            (fallback_count, 3, 2),
            dtype=torch.float32,
            device=vertex_colors.device,
        )
        x0 = patch_x.to(torch.float32)
        y0 = (patch_y + atlas_height).to(torch.float32)
        fallback_uvs[:, 0, 0] = (x0 + 0.5) / float(atlas_width)
        fallback_uvs[:, 0, 1] = (y0 + 0.5) / float(new_atlas_height)
        fallback_uvs[:, 1, 0] = (x0 + 1.5) / float(atlas_width)
        fallback_uvs[:, 1, 1] = (y0 + 0.5) / float(new_atlas_height)
        fallback_uvs[:, 2, 0] = (x0 + 0.5) / float(atlas_width)
        fallback_uvs[:, 2, 1] = (y0 + 1.5) / float(new_atlas_height)
        fallback_vertex_indices = torch.stack(
            (tri_base, tri_base + 1, tri_base + 2),
            dim=1,
        )
        texture_uvs[fallback_vertex_indices.reshape(-1)] = fallback_uvs.reshape(-1, 2)
        return texture_atlas, texture_uvs

    def _normalize_projective_texture_observations(
        self,
        texture_observations: CameraObservation | Sequence[CameraObservation],
    ) -> list[TextureBatch]:
        """Validate and batch RGB or RGB-D observations for texturing."""
        if isinstance(texture_observations, CameraObservation):
            observations = [texture_observations]
        else:
            observations = list(texture_observations)
        if len(observations) == 0:
            log_and_raise(
                "texture_observations must contain at least one CameraObservation."
            )

        batches: list[TextureBatch] = []
        pending: list[CameraObservation] = []
        for observation in observations:
            if observation.rgb_image is None:
                log_and_raise(
                    "CameraObservation.rgb_image is required for texture mapping."
                )
            if observation.pose is None:
                log_and_raise(
                    "CameraObservation.pose is required for texture mapping."
                )
            if observation.intrinsics is None:
                log_and_raise(
                    "CameraObservation.intrinsics is required for texture mapping."
                )

            if observation.rgb_image.ndim == 3:
                pending.append(observation)
                if len(pending) == self.config.texture_num_cameras:
                    batches.append(self._stack_projective_texture_batch(pending))
                    pending = []
            elif observation.rgb_image.ndim == 4:
                if pending:
                    log_and_raise(
                        "Cannot mix unbatched and batched texture observations in one call."
                    )
                batches.append(self._normalize_projective_texture_batch(observation))
            else:
                log_and_raise(
                    "rgb_image must be (H, W, 3) or (num_cameras, H, W, 3), got "
                    f"shape {tuple(observation.rgb_image.shape)}."
                )

        if pending:
            log_and_raise(
                f"Received {len(pending)} unbatched texture observations, but "
                f"texture_num_cameras={self.config.texture_num_cameras}; provide "
                "complete camera batches."
            )
        return batches

    def _stack_projective_texture_batch(
        self,
        observations: Sequence[CameraObservation],
    ) -> TextureBatch:
        """Stack one complete unbatched camera group and fill missing depth."""
        device = torch.device(self._tsdf.device)
        expected_depth_shape = (self.config.image_height, self.config.image_width)
        has_depth = [observation.depth_image is not None for observation in observations]
        for observation in observations:
            if observation.depth_image is None:
                continue
            if tuple(observation.depth_image.shape) != expected_depth_shape:
                log_and_raise(
                    "depth_image must align with unbatched rgb_image as "
                    f"{expected_depth_shape}, got {tuple(observation.depth_image.shape)}."
                )
            if observation.depth_image.dtype != torch.float32:
                log_and_raise(
                    "depth_image must be float32 camera-frame z depth in meters, got "
                    f"{observation.depth_image.dtype}."
                )

        depth_image = None
        if all(has_depth):
            depth_image = torch.stack(
                [observation.depth_image.to(device=device) for observation in observations]
            )
        batch = self._normalize_projective_texture_batch(
            CameraObservation(
                rgb_image=torch.stack(
                    [observation.rgb_image.to(device=device) for observation in observations]
                ),
                depth_image=depth_image,
                pose=Pose(
                    position=torch.cat(
                        [
                            observation.pose.position.view(1, 3).to(device=device)
                            for observation in observations
                        ]
                    ),
                    quaternion=torch.cat(
                        [
                            observation.pose.quaternion.view(1, 4).to(device=device)
                            for observation in observations
                        ]
                    ),
                ),
                intrinsics=torch.stack(
                    [observation.intrinsics.to(device=device) for observation in observations]
                ),
            )
        )
        if not any(has_depth) or all(has_depth):
            return batch

        rgb, visibility_depth, intrinsics, positions, quaternions = batch
        for camera_idx, observation in enumerate(observations):
            if observation.depth_image is not None:
                visibility_depth[camera_idx].copy_(
                    observation.depth_image.to(device=device)
                )
        return rgb, visibility_depth, intrinsics, positions, quaternions

    def _normalize_projective_texture_batch(
        self,
        observation: CameraObservation,
    ) -> TextureBatch:
        """Validate one texture batch and synthesize missing visibility depth."""
        rgb = observation.rgb_image
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        n_cameras = int(rgb.shape[0])
        expected_rgb_shape = (
            self.config.texture_num_cameras,
            self.config.image_height,
            self.config.image_width,
            3,
        )
        if tuple(rgb.shape) != expected_rgb_shape:
            log_and_raise(
                f"rgb_image shape mismatch: expected {expected_rgb_shape}, "
                f"got {tuple(rgb.shape)}."
            )
        if rgb.dtype != torch.uint8:
            log_and_raise(f"rgb_image dtype must be torch.uint8, got {rgb.dtype}.")

        intrinsics = observation.intrinsics
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if tuple(intrinsics.shape) != (self.config.texture_num_cameras, 3, 3):
            log_and_raise(
                "intrinsics must be (texture_num_cameras, 3, 3), got "
                f"shape {tuple(intrinsics.shape)}."
            )
        if intrinsics.dtype != torch.float32:
            log_and_raise(
                f"intrinsics dtype must be torch.float32, got {intrinsics.dtype}."
            )

        position = observation.pose.position
        quaternion = observation.pose.quaternion
        if position.numel() != n_cameras * 3:
            log_and_raise(
                f"pose.position must contain {n_cameras * 3} values, "
                f"got {position.numel()}."
            )
        if quaternion.numel() != n_cameras * 4:
            log_and_raise(
                "pose.quaternion must contain "
                f"{n_cameras * 4} values, got {quaternion.numel()}."
            )
        positions = position.view(n_cameras, 3)
        quaternions = quaternion.view(n_cameras, 4)
        if positions.dtype != torch.float32:
            log_and_raise(
                f"pose.position dtype must be torch.float32, got {positions.dtype}."
            )
        if quaternions.dtype != torch.float32:
            log_and_raise(
                f"pose.quaternion dtype must be torch.float32, got {quaternions.dtype}."
            )

        device = torch.device(self._tsdf.device)
        rgb = rgb.to(device=device).contiguous()
        intrinsics = intrinsics.to(device=device).contiguous()
        positions = positions.to(device=device).contiguous()
        quaternions = quaternions.to(device=device).contiguous()

        depth = observation.depth_image
        if depth is None:
            depth = self._renderer.render_depth(
                intrinsics,
                Pose(position=positions, quaternion=quaternions),
                (self.config.image_height, self.config.image_width),
            )
            depth = depth.clone()
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        expected_depth_shape = expected_rgb_shape[:3]
        if tuple(depth.shape) != expected_depth_shape:
            log_and_raise(
                "depth_image must align with rgb_image as "
                f"{expected_depth_shape}, got {tuple(depth.shape)}."
            )
        if depth.dtype != torch.float32:
            log_and_raise(
                "depth_image must be float32 camera-frame z depth in meters, got "
                f"{depth.dtype}."
            )

        return (
            rgb,
            depth.to(device=device).contiguous(),
            intrinsics,
            positions,
            quaternions,
        )

    def _build_projective_texture_atlas(
        self,
        batches: Sequence[TextureBatch],
        *,
        atlas_columns: int,
    ) -> torch.Tensor:
        """Pack batched RGB images into a row-major texture atlas."""
        total_cameras = len(batches) * self.config.texture_num_cameras
        atlas_rows = math.ceil(total_cameras / atlas_columns)
        atlas_width = atlas_columns * self.config.image_width
        atlas_height = atlas_rows * self.config.image_height
        texture_atlas = torch.zeros(
            (atlas_height, atlas_width, 3),
            dtype=torch.uint8,
            device=torch.device(self._tsdf.device),
        )
        for batch_idx, (rgb, _depth, _intrinsics, _positions, _quaternions) in enumerate(
            batches
        ):
            for camera_idx in range(self.config.texture_num_cameras):
                atlas_idx = batch_idx * self.config.texture_num_cameras + camera_idx
                atlas_row = atlas_idx // atlas_columns
                atlas_col = atlas_idx % atlas_columns
                y0 = atlas_row * self.config.image_height
                x0 = atlas_col * self.config.image_width
                texture_atlas[
                    y0 : y0 + self.config.image_height,
                    x0 : x0 + self.config.image_width,
                    :,
                ] = rgb[camera_idx]
        return texture_atlas

    def texture_occupied_voxels(
        self,
        voxels: OccupiedVoxels,
        texture_observations: CameraObservation | Sequence[CameraObservation],
        *,
        camera_min_distance: Optional[float],
        camera_max_distance: Optional[float],
        texture_depth_tolerance_m: Optional[float],
    ) -> OccupiedVoxels:
        """Color occupied voxel samples from RGB or RGB-D observations."""
        _validate_texture_depth_tolerance(texture_depth_tolerance_m)
        colors = torch.zeros(
            (len(voxels), 3),
            dtype=torch.uint8,
            device=voxels.centers.device,
        )
        valid = torch.zeros(
            (len(voxels),),
            dtype=torch.bool,
            device=voxels.centers.device,
        )
        if len(voxels) > 0:
            min_distance = (
                self.config.depth_minimum_distance
                if camera_min_distance is None
                else camera_min_distance
            )
            max_distance = (
                self.config.depth_maximum_distance
                if camera_max_distance is None
                else camera_max_distance
            )
            batches = self._normalize_projective_texture_observations(
                texture_observations
            )
            colors, valid = self._project_texture_points(
                voxels.centers,
                batches,
                camera_min_distance=float(min_distance),
                camera_max_distance=float(max_distance),
                texture_depth_tolerance_m=texture_depth_tolerance_m,
            )

        return OccupiedVoxels(
            centers=voxels.centers,
            block_idx_per_voxel=voxels.block_idx_per_voxel,
            block_data=voxels.block_data,
            texture_colors=colors,
            texture_valid=valid,
            subvoxel_factor=voxels.subvoxel_factor,
        )

    def _project_texture_points(
        self,
        points: torch.Tensor,
        batches: Sequence[TextureBatch],
        *,
        camera_min_distance: float,
        camera_max_distance: float,
        texture_depth_tolerance_m: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project world-frame points into texture views and sample visible RGB."""
        point_count = int(points.shape[0])
        texture_colors = torch.zeros(
            (point_count, 3),
            dtype=torch.uint8,
            device=points.device,
        )
        texture_valid = torch.zeros(
            (point_count,),
            dtype=torch.bool,
            device=points.device,
        )
        best_score = torch.full(
            (point_count,),
            -float("inf"),
            dtype=torch.float32,
            device=points.device,
        )

        for rgb, visibility_depth, intrinsics, positions, quaternions in batches:
            batch_colors, batch_valid, batch_score = self._project_texture_points_batch(
                points,
                rgb,
                visibility_depth,
                intrinsics,
                positions,
                quaternions,
                camera_min_distance=camera_min_distance,
                camera_max_distance=camera_max_distance,
                texture_depth_tolerance_m=texture_depth_tolerance_m,
            )
            update = batch_valid & (batch_score > best_score)
            if update.any():
                texture_colors[update] = batch_colors[update]
                texture_valid[update] = True
                best_score[update] = batch_score[update]

        return texture_colors, texture_valid

    def _project_texture_points_batch(
        self,
        points: torch.Tensor,
        rgb: torch.Tensor,
        visibility_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        positions: torch.Tensor,
        quaternions: torch.Tensor,
        *,
        camera_min_distance: float,
        camera_max_distance: float,
        texture_depth_tolerance_m: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project points into one exact texture camera batch."""
        camera_count = int(rgb.shape[0])
        point_count = int(points.shape[0])
        image_height = int(rgb.shape[1])
        image_width = int(rgb.shape[2])

        rotations = torch_quaternion_to_matrix(quaternions.to(dtype=torch.float32))
        points_rel = points.to(dtype=torch.float32).unsqueeze(0) - positions[:, None, :]
        points_cam = torch.matmul(points_rel, rotations)
        z = points_cam[..., 2]

        fx = intrinsics[:, 0, 0].view(camera_count, 1)
        fy = intrinsics[:, 1, 1].view(camera_count, 1)
        cx = intrinsics[:, 0, 2].view(camera_count, 1)
        cy = intrinsics[:, 1, 2].view(camera_count, 1)
        safe_z = torch.where(
            torch.isfinite(z) & (torch.abs(z) > 1.0e-12),
            z,
            torch.ones_like(z),
        )
        u = fx * points_cam[..., 0] / safe_z + cx
        v = fy * points_cam[..., 1] / safe_z + cy
        in_frame = (
            torch.isfinite(z)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (z > camera_min_distance)
            & (z <= camera_max_distance)
            & (u >= 0.0)
            & (u < float(image_width))
            & (v >= 0.0)
            & (v < float(image_height))
        )

        px = torch.floor(u).to(torch.long).clamp(0, image_width - 1)
        py = torch.floor(v).to(torch.long).clamp(0, image_height - 1)
        camera_indices = torch.arange(
            camera_count,
            device=points.device,
        ).view(camera_count, 1)
        sampled_depth = visibility_depth[camera_indices, py, px]
        visible = in_frame & _texture_depth_visible(
            z,
            sampled_depth,
            voxel_size=self.config.voxel_size,
            texture_depth_tolerance_m=texture_depth_tolerance_m,
        )

        candidate_score = torch.where(
            visible,
            -z,
            torch.full_like(z, -float("inf")),
        )
        best_score, best_camera = candidate_score.max(dim=0)
        valid = torch.isfinite(best_score)
        point_indices = torch.arange(point_count, device=points.device)
        best_px = px[best_camera, point_indices]
        best_py = py[best_camera, point_indices]
        colors = rgb[best_camera, best_py, best_px]
        return colors, valid, best_score
