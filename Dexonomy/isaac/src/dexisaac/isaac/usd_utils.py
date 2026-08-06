"""Minimal USD authoring helpers extracted from ocir's visualize_grasp.py.

Only the pieces simulate_grasp_traj.py needs (object mesh + material) live
here, so the vendored package does not drag in the full visualization stack.
"""
from __future__ import annotations

import numpy as np


def _vt_vec3f_array(values: np.ndarray):
    """(N,3) numpy -> Vt.Vec3fArray, zero-copy when the USD build supports it."""
    from pxr import Gf, Vt

    values = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1, 3))
    from_numpy = getattr(Vt.Vec3fArray, "FromNumpy", None)
    if from_numpy is not None:
        return from_numpy(values)
    return Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v]) for v in values])


def add_material(stage, path: str, color: tuple[float, float, float], roughness: float = 0.55):
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def define_mesh(stage, path: str, vertices: np.ndarray, faces: np.ndarray, material) -> dict:
    from pxr import UsdGeom, Vt

    vertices = np.asarray(vertices, dtype=float)
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int32).reshape(-1, 3))
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(_vt_vec3f_array(vertices))
    from_numpy = getattr(Vt.IntArray, "FromNumpy", None)
    if from_numpy is not None:
        mesh.CreateFaceVertexCountsAttr(from_numpy(np.full((faces.shape[0],), 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(from_numpy(faces.reshape(-1)))
    else:
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).astype(int).tolist()))
    mesh.CreateDoubleSidedAttr(True)
    bind_material(mesh.GetPrim(), material)
    return {"path": path, "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])}
