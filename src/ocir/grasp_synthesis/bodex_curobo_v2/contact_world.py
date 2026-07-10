"""Single-object exact BODex contact queries for OCIR grasp synthesis.

This module ports the BODex contact-query semantics needed by Sharpa grasp
synthesis without importing anything from ``third_party/BODex``. It is
intentionally scoped to one manipulated object in the canonical object frame,
matching the rest of the OCIR Sharpa Wave grasp-synthesis pipeline.

Two contact-query modes are implemented, matching BODex's two-tier contact
model:

- Sphere-mesh (``get_sphere_contact_pdn``): distance from a robot collision
  sphere center to the object mesh surface. Uses official cuRobo v2's own
  ``WarpMeshQuery`` analytic signed-distance query instead of BODex's
  stochastic finite-difference Warp kernels -- same physical query (point to
  mesh signed distance), but with an exact analytic gradient rather than a
  Halton-perturbation finite-difference approximation.
- Mesh-mesh (``get_mesh_contact_pdn``): exact convex-hull GJK/EPA distance
  between a robot fingertip link and the object, via the standalone ``coal``
  package (the same underlying library BODex's own compiled extension wraps,
  but installed here as an independent dependency, not BODex code). Since
  plain ``coal.distance`` is not autograd-aware, this mirrors BODex's own
  strategy: a genuine finite-difference gradient over small translation-only
  pose perturbations, refreshed periodically and reattached to the current
  (FK-differentiable) link pose in between refreshes by the caller
  (``grasp_cost.py``).

The object mesh is convex-decomposed into multiple parts via ``coacd``
(matching original BODex's "one convex part per link" object treatment)
before being loaded into ``coal.ConvexBase`` hulls: see
``_coacd_convex_parts``. Per-part distances are combined with a min-over-parts
reduction in ``_narrow_phase``, approximating the concave object's true
surface as the union of its convex pieces. Robot links still use a single
whole-mesh convex hull (``_load_convex_hull``); see ``README.md``'s "Known
Limitation" note for that remaining gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import coal
import numpy as np
import torch
import trimesh

# `coacd` bundles its own native OpenMP/runtime libraries that segfault on
# import if a shared-library symbol they conflict with (e.g. torch's own
# OpenMP) hasn't been loaded first. Import order matters here: `torch` (and
# therefore curobo/warp, which pull it in) must come before `coacd`.
import coacd

# CoACD's own C++ logger otherwise prints its full per-run parameter dump and
# per-part progress straight to stdout; we log our own concise begin/end
# lines around each `run_coacd` call instead (see `_coacd_convex_parts`).
coacd.set_log_level("error")

from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
from curobo._src.geom.transform import pose_multiply, torch_quaternion_to_matrix
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp

from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import normalize_vector
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost import ContactBuffer

# `pose_multiply` (used by MeshContactPdnFunction) calls into Warp via
# `wp.from_torch`, which requires Warp to already be initialized regardless
# of whether a WarpMeshQuery has been constructed yet.
init_warp()


# ---------------------------------------------------------------------------
# URDF parsing helpers (no BODex dependency; ported verbatim)
# ---------------------------------------------------------------------------


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    tr = float(np.trace(matrix))
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        return np.asarray([0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s])
    i = int(np.argmax(np.diag(matrix)))
    if i == 0:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return np.asarray([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return np.asarray([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s])
    s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return np.asarray([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])


def _origin_pose(elem) -> tuple[np.ndarray, np.ndarray]:
    out = np.eye(4, dtype=float)
    origin = elem.find("origin")
    if origin is not None:
        if origin.attrib.get("xyz"):
            out[:3, 3] = np.fromstring(origin.attrib["xyz"], sep=" ", dtype=float)
        if origin.attrib.get("rpy"):
            out[:3, :3] = _rpy_to_matrix(np.fromstring(origin.attrib["rpy"], sep=" ", dtype=float))
    quat = _matrix_to_quat_wxyz(out[:3, :3])
    quat = quat / max(np.linalg.norm(quat), 1e-9)
    return out[:3, 3].copy(), quat.astype(float)


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    filename = filename.replace("package://", "")
    path = Path(filename)
    if path.is_absolute():
        return path
    candidates = [urdf_path.parent / path, urdf_path.parent.parent / path, urdf_path.parents[1] / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve mesh {filename!r} from {urdf_path}")


def _first_collision_mesh(urdf_path: Path, link_name: str) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    link = next((elem for elem in root.findall("link") if elem.attrib["name"] == link_name), None)
    if link is None:
        raise KeyError(f"link not found in URDF: {link_name}")
    for coll in list(link.findall("collision")) + list(link.findall("visual")):
        mesh = coll.find("geometry/mesh")
        if mesh is None or not mesh.attrib.get("filename"):
            continue
        scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ", dtype=float)
        if scale.size == 0:
            scale = np.ones(3, dtype=float)
        offset_t, offset_q = _origin_pose(coll)
        return _resolve_mesh_path(urdf_path, mesh.attrib["filename"]), scale.astype(float), offset_t, offset_q
    raise FileNotFoundError(f"no collision/visual mesh found for link {link_name}")


def _obb_param(mesh_path: Path, scale: np.ndarray) -> torch.Tensor:
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float) * scale.reshape(1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)
    obb = mesh.bounding_box_oriented.primitive
    transform = torch.tensor(obb.transform, dtype=torch.float32)
    extents = torch.tensor(obb.extents, dtype=torch.float32) * 0.5
    return torch.cat([transform.reshape(-1), extents], dim=0)


def _sphere_param(mesh_path: Path, scale: np.ndarray) -> torch.Tensor:
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float) * scale.reshape(1, 3)
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center[None, :], axis=1).max()
    return torch.tensor([center[0], center[1], center[2], radius], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sphere-mesh contact: cuRobo v2 WarpMeshQuery, analytic gradient
# ---------------------------------------------------------------------------


class SphereContactPdnFunction(torch.autograd.Function):
    """Differentiable sphere-to-mesh (position, distance, normal) query.

    Forward uses ``WarpMeshQuery.query_sdf`` (analytic, GPU): the closest
    surface point is recovered exactly as ``center - grad * sdf`` (since
    ``grad * sdf`` telescopes to ``point - closest_point`` for any +-1 sign
    convention), and the inward-pointing BODex-style normal is ``-grad``.
    Backward uses the exact SDF gradient (``grad``) for the distance term,
    and a first-order tangent-plane-projection Jacobian for the position
    term, matching the same locally-flat-surface assumption BODex's own
    finite-difference clamp makes -- but computed analytically instead of by
    stochastic perturbation.
    """

    @staticmethod
    def forward(ctx, centers: torch.Tensor, radii: torch.Tensor, mesh_query: WarpMeshQuery):
        flat_centers = centers.detach().reshape(-1, 3).contiguous()
        sdf, grad = mesh_query.query_sdf(flat_centers)
        sdf = sdf.view(centers.shape[:-1])
        grad = grad.view(centers.shape)
        position = centers.detach() - grad * sdf.unsqueeze(-1)
        distance = sdf - radii
        normal_inward = -grad
        ctx.save_for_backward(grad)
        return position, distance, normal_inward

    @staticmethod
    def backward(ctx, grad_position: torch.Tensor, grad_distance: torch.Tensor, grad_normal: torch.Tensor):
        (n,) = ctx.saved_tensors
        tangent_component = grad_position - (grad_position * n).sum(dim=-1, keepdim=True) * n
        grad_center = tangent_component + grad_distance.unsqueeze(-1) * n
        # d(normal)/d(center) is neglected to first order (curvature term),
        # matching BODex's own locally-flat-surface assumption.
        return grad_center, None, None


# ---------------------------------------------------------------------------
# Mesh-mesh contact: standalone `coal` convex hull GJK/EPA distance
# ---------------------------------------------------------------------------


def _load_convex_hull(mesh_path: Path, scale: np.ndarray) -> "coal.ConvexBase":
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale.reshape(1, 3)
    points = coal.StdVec_Vec3s()
    for vertex in vertices:
        points.append(vertex)
    return coal.ConvexBase.convexHull(points, False, None)


def _normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center and isotropically rescale ``vertices`` to a unit bounding radius.

    CoACD's concavity threshold is tuned for roughly unit-scale meshes, so
    object meshes (which may be authored in arbitrary absolute units) are
    normalized before decomposition. Scaling is isotropic (a single scalar,
    not per-axis) so the inverse transform exactly recovers the original
    shape -- only a translation and a uniform scale need to be undone.
    """

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_max + bbox_min) / 2.0
    norm_scale = float(np.linalg.norm(bbox_max - bbox_min) / 2.0)
    if norm_scale <= 0.0 or not np.isfinite(norm_scale):
        raise ValueError("cannot normalize a degenerate (zero-extent) mesh for convex decomposition")
    normalized = (vertices - center[None, :]) / norm_scale
    return normalized, center, norm_scale


def _coacd_convex_parts(mesh_path: Path, scale: np.ndarray, **coacd_kwargs) -> list[np.ndarray]:
    """Convex-decompose a mesh with CoACD, returning per-part vertex arrays.

    The mesh is normalized (see ``_normalize_vertices``) before being handed
    to CoACD, and each returned part is mapped back through the inverse
    normalization -- then through the caller-supplied URDF/asset ``scale`` --
    so callers receive parts in the same frame ``_load_convex_hull`` would
    have used for a single whole-mesh hull.
    """

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    normalized_vertices, center, norm_scale = _normalize_vertices(vertices)

    print(f"OCIR_BODEX_CUROBO_V2 coacd: decomposing {mesh_path.name} ...", flush=True)
    t0 = time.time()
    parts = coacd.run_coacd(coacd.Mesh(normalized_vertices, faces), **coacd_kwargs)
    print(
        f"OCIR_BODEX_CUROBO_V2 coacd: {mesh_path.name} -> {len(parts)} parts in {time.time() - t0:.1f}s",
        flush=True,
    )

    part_vertices = []
    for part_verts, _part_faces in parts:
        restored = np.asarray(part_verts, dtype=np.float64) * norm_scale + center[None, :]
        part_vertices.append(restored * scale.reshape(1, 3))
    return part_vertices


def _load_convex_parts(mesh_path: Path, scale: np.ndarray, **coacd_kwargs) -> list["coal.ConvexBase"]:
    hulls = []
    for vertices in _coacd_convex_parts(mesh_path, scale, **coacd_kwargs):
        points = coal.StdVec_Vec3s()
        for vertex in vertices:
            points.append(vertex)
        hulls.append(coal.ConvexBase.convexHull(points, False, None))
    return hulls


def _narrow_phase(
    obj_hulls: list,
    robot_hulls: list,
    robot_idx: torch.Tensor,
    obj_rot: torch.Tensor,
    obj_trans: torch.Tensor,
    robot_rot: torch.Tensor,
    robot_trans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched convex-convex distance via the standalone `coal` package.

    Runs one `coal.distance` call per (case, object part) pair in a Python
    loop (no BODex OpenMP-batched extension). This is only called every
    `count % 5` cost evaluations by `grasp_cost.py`, so the per-call Python
    overhead is not on the innermost hot path.

    The object is represented as a list of convex parts (from CoACD; see
    `_coacd_convex_parts`), so each case takes a min-over-parts reduction of
    the signed distance -- the standard way to approximate a concave shape's
    true surface with the union of its convex-decomposed pieces.
    """

    cases = int(robot_idx.shape[0])
    dist = np.empty((cases,), dtype=np.float64)
    normal = np.empty((cases, 3), dtype=np.float64)
    cp1 = np.empty((cases, 3), dtype=np.float64)
    cp2 = np.empty((cases, 3), dtype=np.float64)

    obj_rot_np = obj_rot.detach().cpu().numpy()
    obj_trans_np = obj_trans.detach().cpu().numpy()
    robot_rot_np = robot_rot.detach().cpu().numpy()
    robot_trans_np = robot_trans.detach().cpu().numpy()
    robot_idx_np = robot_idx.detach().cpu().numpy()

    req = coal.DistanceRequest(True)
    for i in range(cases):
        tf_obj = coal.Transform3s(obj_rot_np[i], obj_trans_np[i])
        tf_robot = coal.Transform3s(robot_rot_np[i], robot_trans_np[i])
        robot_hull = robot_hulls[int(robot_idx_np[i])]

        best_dist = np.inf
        best_normal = None
        best_cp1 = None
        best_cp2 = None
        for obj_hull in obj_hulls:
            res = coal.DistanceResult()
            coal.distance(obj_hull, tf_obj, robot_hull, tf_robot, req, res)
            if res.min_distance < best_dist:
                best_dist = res.min_distance
                # coal's DistanceResult.normal points from geometry-1 (object
                # part) toward geometry-2 (robot finger), i.e. outward from
                # the part's surface. BODex's grasp-matrix math expects an
                # inward-pointing normal (into the object, matching the
                # sphere-contact path's convention), so flip the sign here.
                best_normal = -np.asarray(res.normal)
                best_cp1 = np.asarray(res.getNearestPoint1())
                best_cp2 = np.asarray(res.getNearestPoint2())

        dist[i] = best_dist
        normal[i] = best_normal
        cp1[i] = best_cp1
        cp2[i] = best_cp2

    dist_t = torch.tensor(dist, device=robot_trans.device, dtype=robot_trans.dtype)
    normal_t = torch.tensor(normal, device=robot_trans.device, dtype=robot_trans.dtype)
    points_t = torch.tensor(np.concatenate([cp1, cp2], axis=-1), device=robot_trans.device, dtype=robot_trans.dtype)
    return dist_t, normal_t, points_t


class MeshContactPdnFunction(torch.autograd.Function):
    """Differentiable mesh-mesh (position, distance, normal) query via `coal`.

    Forward runs one non-differentiable `coal` GJK/EPA query per link, and
    (only if `robot_pose.requires_grad`) a set of finite-difference-perturbed
    queries. Perturbation is translation-only (matching BODex's own 3-dim
    perturbation template): rotation sensitivity is instead captured
    analytically by the caller's between-refresh FK reattachment
    (`grasp_cost.py`'s `raw_pos_robot_with_grad`). Backward returns a
    translation-only gradient, zero-padded into the quaternion slots of the
    7-dim pose gradient.
    """

    @staticmethod
    def forward(
        ctx,
        robot_pose: torch.Tensor,
        robot_offset_pose: torch.Tensor,
        obj_hulls: list,
        robot_hulls: list,
        perturb: torch.Tensor,
    ):
        b, h, n_links, _ = robot_pose.shape
        flat_pose = robot_pose.detach().reshape(-1, 7)
        robot_idx = torch.arange(n_links, device=robot_pose.device).repeat(b * h)
        offset = robot_offset_pose[robot_idx]
        r_trans, r_quat = pose_multiply(
            flat_pose[:, :3].contiguous(),
            flat_pose[:, 3:7].contiguous(),
            offset[:, :3].contiguous(),
            offset[:, 3:7].contiguous(),
        )
        r_rot = torch_quaternion_to_matrix(r_quat)
        obj_rot = torch.eye(3, device=robot_pose.device, dtype=robot_pose.dtype).view(1, 3, 3).expand(flat_pose.shape[0], 3, 3)
        obj_trans = torch.zeros((flat_pose.shape[0], 3), device=robot_pose.device, dtype=robot_pose.dtype)
        dist, normal, points = _narrow_phase(obj_hulls, robot_hulls, robot_idx, obj_rot, obj_trans, r_rot, r_trans)
        points = points.view(b, h, n_links, 6)
        dist = dist.view(b, h, n_links)
        normal = normal.view(b, h, n_links, 3)

        perturb_num = perturb.shape[-2]
        if robot_pose.requires_grad:
            # perturb: (b, h, n_links, perturb_num, 3) -- translation-only.
            perturb_permute = perturb.permute(3, 0, 1, 2, 4)  # (perturb_num, b, h, n_links, 3)
            perturb_batched = perturb_permute.reshape((perturb_num * b,) + perturb_permute.shape[2:])
            pert_pose = robot_pose.detach().repeat(perturb_num, 1, 1, 1).clone()
            pert_pose[..., :3] = pert_pose[..., :3] + perturb_batched
            p_b, p_h, p_n, _ = pert_pose.shape
            pert_flat = pert_pose.reshape(-1, 7)
            pert_idx = torch.arange(n_links, device=robot_pose.device).repeat(p_b * p_h)
            pert_offset = robot_offset_pose[pert_idx]
            pr_trans, pr_quat = pose_multiply(
                pert_flat[:, :3].contiguous(),
                pert_flat[:, 3:7].contiguous(),
                pert_offset[:, :3].contiguous(),
                pert_offset[:, 3:7].contiguous(),
            )
            pr_rot = torch_quaternion_to_matrix(pr_quat)
            po_rot = torch.eye(3, device=robot_pose.device, dtype=robot_pose.dtype).view(1, 3, 3).expand(pert_flat.shape[0], 3, 3)
            po_trans = torch.zeros((pert_flat.shape[0], 3), device=robot_pose.device, dtype=robot_pose.dtype)
            pnarrow_idx = torch.arange(n_links, device=robot_pose.device).repeat(p_b * p_h)
            pdist, pnormal, ppoints = _narrow_phase(obj_hulls, robot_hulls, pnarrow_idx, po_rot, po_trans, pr_rot, pr_trans)
            pert_points = ppoints.view((perturb_num,) + points.shape)
            pert_distance = pdist.view((perturb_num,) + dist.shape)
            pert_normal = pnormal.view((perturb_num,) + normal.shape)
            debug_pos = pert_points.permute(1, 2, 3, 0, 4)
            debug_normal = pert_normal.permute(1, 2, 3, 0, 4)
            grad_pos = (pert_points - points.unsqueeze(0)).unsqueeze(-1) / perturb_permute.unsqueeze(-2)
            grad_dist = (pert_distance - dist.unsqueeze(0)).unsqueeze(-1) / perturb_permute
            grad_normal = (pert_normal - normal.unsqueeze(0)).unsqueeze(-1) / perturb_permute.unsqueeze(-2)
            grad_pos = torch.nan_to_num(grad_pos, posinf=0.0, neginf=0.0).mean(dim=0)
            grad_dist = torch.nan_to_num(grad_dist, posinf=0.0, neginf=0.0).mean(dim=0)
            grad_normal = torch.nan_to_num(grad_normal, posinf=0.0, neginf=0.0).mean(dim=0)
            ctx.save_for_backward(grad_pos, grad_dist, grad_normal)
        else:
            debug_pos = points.unsqueeze(-2).expand(-1, -1, -1, perturb_num, -1)
            debug_normal = normal.unsqueeze(-2).expand(-1, -1, -1, perturb_num, -1)
        ctx.requi_grad = robot_pose.requires_grad
        ctx.pose_shape = robot_pose.shape
        return points, dist, normal, debug_pos, debug_normal

    @staticmethod
    def backward(ctx, p_grad_in, d_grad_in, n_grad_in, debug_pos, debug_normal):
        if not ctx.requi_grad:
            return None, None, None, None, None
        gp, gd, gn = ctx.saved_tensors
        translation_grad = (p_grad_in.unsqueeze(-2) @ gp).squeeze(-2)
        translation_grad = translation_grad + d_grad_in.unsqueeze(-1) * gd
        translation_grad = translation_grad + (n_grad_in.unsqueeze(-2) @ gn).squeeze(-2)
        pose_grad = translation_grad.new_zeros(ctx.pose_shape)
        pose_grad[..., :3] = translation_grad
        return pose_grad, None, None, None, None


# ---------------------------------------------------------------------------
# Combined single-object contact world
# ---------------------------------------------------------------------------


@dataclass
class SingleObjectContactWorld:
    object_mesh_path: Path
    robot_urdf_path: Path
    contact_link_names: tuple[str, ...]
    device_cfg: DeviceCfg
    coacd_kwargs: dict = None

    def __post_init__(self):
        mesh = trimesh.load(str(self.object_mesh_path), force="mesh", process=False)
        self._mesh_query = WarpMeshQuery(mesh, self.device_cfg.device)

        self._object_hulls = _load_convex_parts(
            self.object_mesh_path, np.ones(3, dtype=np.float64), **(self.coacd_kwargs or {})
        )
        self._robot_hulls = []
        offsets = []
        for link_name in self.contact_link_names:
            mesh_path, scale, offset_t, offset_q = _first_collision_mesh(self.robot_urdf_path, link_name)
            self._robot_hulls.append(_load_convex_hull(mesh_path, scale.astype(np.float64)))
            offsets.append(np.concatenate([offset_t, offset_q], axis=0))
        self._robot_offset_pose = self.device_cfg.to_device(np.asarray(offsets, dtype=np.float32))

    def get_sphere_contact_pdn(
        self,
        contact_robot_sphere: torch.Tensor,
        contact_query_buffer: ContactBuffer,
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor | None = None,
    ):
        centers = contact_robot_sphere[..., :3]
        radii = contact_robot_sphere[..., 3]
        position, distance, normal = SphereContactPdnFunction.apply(centers, radii, self._mesh_query)
        n_perturb = perturb.shape[-2]
        debug_pos = position.unsqueeze(-2).expand(*position.shape[:-1], n_perturb, 3)
        debug_normal = normal.unsqueeze(-2).expand(*normal.shape[:-1], n_perturb, 3)
        return position, distance, normal, debug_pos, debug_normal

    def get_mesh_contact_pdn(
        self,
        contact_robot_pose: torch.Tensor,
        perturb: torch.Tensor,
        env_query_idx: torch.Tensor | None,
        dist_upper_bound: torch.Tensor,
    ):
        return MeshContactPdnFunction.apply(
            contact_robot_pose,
            self._robot_offset_pose,
            self._object_hulls,
            self._robot_hulls,
            perturb,
        )
