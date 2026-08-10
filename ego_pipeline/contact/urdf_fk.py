"""Minimal URDF forward kinematics for the SharpaWave hand (numpy + trimesh only).

Deliberately not pinocchio/yourdfpy: the SharpaWave URDF is a plain tree of `revolute`
and `fixed` joints with no mimic joints, so ~150 lines here keep the tool runnable in
any env that has numpy+trimesh (hawor / biv2ap / HV2RD) instead of adding a dependency
that none of them currently has.

Root link is `<side>_hand_C_MC`, which is exactly where export_qpos.py puts the wrist
(`wrist_pos` = MANO wrist joint, `wrist_quat_wxyz` = SharpaWave base orientation), so
`fk_world(q, wrist_pos, wrist_quat)` yields every link pose directly in world.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PAD_SUFFIXES = ("_elastomer",)          # tactile pads = the links that should touch
PALM_SUFFIXES = ("_hand_C_MC",)         # palm (root link)


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_matrix(axis, angle) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _origin(elem) -> np.ndarray:
    T = np.eye(4)
    o = elem.find("origin") if elem is not None else None
    if o is None:
        return T
    T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
    T[:3, :3] = rpy_to_matrix([float(v) for v in (o.get("rpy") or "0 0 0").split()])
    return T


class UrdfChain:
    """Parsed URDF: kinematic tree + per-link geometry."""

    def __init__(self, urdf_path):
        self.path = Path(urdf_path)
        self.dir = self.path.parent
        root = ET.parse(self.path).getroot()

        self.links: dict[str, dict] = {}
        for l in root.findall("link"):
            geoms = {"visual": [], "collision": []}
            for kind in geoms:
                for g in l.findall(kind):
                    mesh = g.find("geometry/mesh")
                    if mesh is None:                       # primitives unused by this URDF
                        continue
                    geoms[kind].append((self.dir / mesh.get("filename"), _origin(g)))
            self.links[l.get("name")] = geoms

        self.joints: dict[str, dict] = {}
        children = set()
        for j in root.findall("joint"):
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[j.get("name")] = dict(
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                origin=_origin(j),
                axis=np.array([float(v) for v in (ax.get("xyz") if ax is not None else "0 0 1").split()]),
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else -np.pi,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else np.pi,
            )
            children.add(j.find("child").get("link"))

        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise RuntimeError(f"expected exactly one root link, got {roots}")
        self.root = roots[0]
        self._mesh_cache: dict[Path, "object"] = {}

    # -- kinematics ---------------------------------------------------------
    def fk(self, q: dict) -> dict:
        """joint-name -> angle (rad); returns link-name -> 4x4 pose in the ROOT link frame.

        Unlisted revolute joints default to 0. Angles are clipped to the URDF limits
        (export_qpos.py already clips, this only guards hand-edited input).
        """
        out = {self.root: np.eye(4)}
        pending = list(self.joints.items())
        while pending:
            progressed = False
            still = []
            for name, j in pending:
                Tp = out.get(j["parent"])
                if Tp is None:
                    still.append((name, j))
                    continue
                T = Tp @ j["origin"]
                if j["type"] in ("revolute", "continuous"):
                    a = float(np.clip(q.get(name, 0.0), j["lower"], j["upper"]))
                    R = np.eye(4)
                    R[:3, :3] = axis_angle_to_matrix(j["axis"], a)
                    T = T @ R
                elif j["type"] != "fixed":
                    raise NotImplementedError(f"joint type {j['type']} ({name})")
                out[j["child"]] = T
                progressed = True
            if not progressed:
                raise RuntimeError(f"disconnected joints: {[n for n, _ in still]}")
            pending = still
        return out

    def fk_world(self, q: dict, root_pos, root_quat_wxyz) -> dict:
        """Same as fk(), with the root link placed at (root_pos, root_quat_wxyz)."""
        T0 = np.eye(4)
        T0[:3, :3] = quat_wxyz_to_matrix(root_quat_wxyz)
        T0[:3, 3] = np.asarray(root_pos, dtype=np.float64)
        return {k: T0 @ v for k, v in self.fk(q).items()}

    # -- geometry -----------------------------------------------------------
    def _mesh(self, path):
        if path not in self._mesh_cache:
            import trimesh
            self._mesh_cache[path] = trimesh.load(path, force="mesh", process=False)
        return self._mesh_cache[path]

    def link_names(self, suffixes=None) -> list:
        if suffixes is None:
            return list(self.links)
        return [n for n in self.links if any(n.endswith(s) for s in suffixes)]

    def surface(self, link_poses: dict, links=None, kind="collision"):
        """Concatenated world-space surface mesh of the requested links."""
        import trimesh
        parts = []
        for name in (links if links is not None else self.links):
            for path, T_geom in self.links[name][kind] or self.links[name]["visual"]:
                m = self._mesh(path).copy()
                m.apply_transform(link_poses[name] @ T_geom)
                parts.append(m)
        if not parts:
            raise RuntimeError("no geometry for the requested links")
        return trimesh.util.concatenate(parts)

    def sample_points(self, link_poses: dict, links=None, kind="collision",
                      n=4000, seed=0) -> tuple:
        """Uniform surface samples of the requested links -> (points (n,3), link_id (n,)).

        Sampling (not raw vertices) keeps the density uniform across links whose STLs
        have wildly different tessellation.
        """
        import trimesh
        names = list(links if links is not None else self.links)
        meshes, ids = [], []
        for i, name in enumerate(names):
            for path, T_geom in self.links[name][kind] or self.links[name]["visual"]:
                m = self._mesh(path).copy()
                m.apply_transform(link_poses[name] @ T_geom)
                meshes.append(m)
                ids.append(i)
        areas = np.array([max(m.area, 1e-12) for m in meshes])
        counts = np.maximum(1, np.round(n * areas / areas.sum()).astype(int))
        rng = np.random.default_rng(seed)
        pts, lid = [], []
        for m, i, c in zip(meshes, ids, counts):
            p, _ = trimesh.sample.sample_surface(m, int(c), seed=int(rng.integers(1 << 30)))
            pts.append(p)
            lid.append(np.full(len(p), i))
        return np.concatenate(pts), np.concatenate(lid), names


def load_hand(side: str, assets_root=None) -> UrdfChain:
    """side in {'right','left'} -> parsed SharpaWave URDF."""
    if assets_root is None:
        from repo_paths import RR_ROOT
        assets_root = RR_ROOT / "ego_pipeline/Retargeting/assets/robots/hands/sharpa_wave"
    return UrdfChain(Path(assets_root) / side / f"{side}_sharpa_wave.urdf")
