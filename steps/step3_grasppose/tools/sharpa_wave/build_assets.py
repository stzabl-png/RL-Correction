"""Build Dexonomy hand assets for the SharpaWave right hand from ocir-grasp-synthesis assets.

Generates:
  assets/hand/sharpa_wave/meshes/*.STL   (copied visual/collision meshes + CoACD parts)
  assets/hand/sharpa_wave/right.xml      (MJCF, Dexonomy conventions)
  assets/hand/sharpa_wave/skeleton.yaml  (from ocir collision spheres)
  assets/hand/sharpa_wave/keypoint.yaml  (contact candidates projected onto collision surface)
  assets/hand/sharpa_wave/body_group.yaml

Run with the dexonomy conda env python:
  /home/lyh/anaconda3/envs/dexonomy/bin/python tools/sharpa_wave/build_assets.py
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation as R

OCIR = Path("/home/lyh/Project/ocir-grasp-synthesis/assets/robots/hands/sharpa_wave")
# ---- 手别 (由 --side 设定). 左手 URDF 与右手是**精确镜像** (同一组关节角下 link
#      位置差 <0.002mm); 左手的碰撞球 yml 由 tools/sharpa_wave/mirror_left_spheres.py 生成 ----
SIDE = "right"
URDF = URDF_MESH_DIR = SPHERES_YML = OUT = OUT_MESH = XML_NAME = None


def _s(name: str) -> str:
    """把写死的 right_ 前缀换成当前手别。"""
    return name.replace("right_", f"{SIDE}_")


def set_side(side: str):
    global SIDE, URDF, URDF_MESH_DIR, SPHERES_YML, OUT, OUT_MESH, XML_NAME, EXTRA_EXCLUDES
    assert side in ("left", "right")
    SIDE = side
    URDF = OCIR / f"urdf/{side}_sharpa_wave/{side}_sharpa_wave.urdf"
    URDF_MESH_DIR = URDF.parent / "meshes"
    SPHERES_YML = OCIR / f"collision/curobo/sharpa_{side}.yml"
    OUT = Path(__file__).resolve().parents[2] / ("assets/hand/sharpa_wave"
               + ("" if side == "right" else "_left"))
    OUT_MESH = OUT / "meshes"
    XML_NAME = f"{side}.xml"
    EXTRA_EXCLUDES = [tuple(_s(x) for x in pair) for pair in EXTRA_EXCLUDES_RAW]

CONVEX_RATIO_OK = 0.95
COACD_THRESHOLD = 0.05

# Contact pairs to exclude, discovered by the validate step (design-clearance
# artifacts that penetrate at the neutral pose once meshes are convexified).
EXTRA_EXCLUDES_RAW = [
    ("right_hand_C_MC", "right_thumb_MC"),  # known 1.2-5.6mm clearance wedge (see ocir thumbfilter overlay)
]
EXTRA_EXCLUDES = list(EXTRA_EXCLUDES_RAW)

MIN_MASS = 2e-3
MIN_INERTIA = 1e-7


def rpy_to_quat_wxyz(rpy):
    q = R.from_euler("xyz", rpy).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


def fmt(arr, prec=6):
    return " ".join(f"{v:.{prec}g}" for v in np.asarray(arr).flatten())


# ---------------------------------------------------------------- URDF parsing

class Link:
    def __init__(self, el):
        self.name = el.get("name")
        self.visuals = []      # (mesh_file, pos, quat)
        self.collisions = []   # (mesh_file, pos, quat)
        for tag, store in (("visual", self.visuals), ("collision", self.collisions)):
            for g in el.findall(tag):
                mesh = g.find("geometry/mesh")
                if mesh is None:
                    continue
                o = g.find("origin")
                pos = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
                rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
                store.append((Path(mesh.get("filename")).name, pos, rpy_to_quat_wxyz(rpy)))
        inert = el.find("inertial")
        self.mass, self.com, self.inertia_link = None, None, None
        if inert is not None:
            self.mass = float(inert.find("mass").get("value"))
            o = inert.find("origin")
            self.com = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
            i = inert.find("inertia")
            I = np.array([
                [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
            ])
            Rm = R.from_euler("xyz", rpy).as_matrix()
            self.inertia_link = Rm @ I @ Rm.T  # tensor about COM, in link-frame axes


class Joint:
    def __init__(self, el):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        o = el.find("origin")
        self.pos = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
        self.quat = rpy_to_quat_wxyz(
            np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3))
        lim = el.find("limit")
        self.range = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None


def parse_urdf():
    root = ET.parse(URDF).getroot()
    links = {l.get("name"): Link(l) for l in root.findall("link")}
    joints = [Joint(j) for j in root.findall("joint")]
    children = {}  # parent link -> [joint] in URDF declaration order
    for j in joints:
        children.setdefault(j.parent, []).append(j)
    root_link = _s("right_hand_C_MC")
    return links, children, root_link


# ---------------------------------------------------------------- mesh assets

def prepare_meshes(links):
    """Copy meshes; CoACD-decompose concave collision meshes.

    Returns {collision_stl_name: [part_file_names]} (single-element list if convex enough).
    """
    OUT_MESH.mkdir(parents=True, exist_ok=True)
    vis_files = {f for l in links.values() for f, _, _ in l.visuals}
    col_files = {f for l in links.values() for f, _, _ in l.collisions}

    for f in sorted(vis_files):
        trimesh.load(URDF_MESH_DIR / f, force="mesh").export(OUT_MESH / f)  # -> binary STL

    col_parts = {}
    for f in sorted(col_files):
        m = trimesh.load(URDF_MESH_DIR / f, force="mesh")
        ratio = m.volume / m.convex_hull.volume
        if ratio >= CONVEX_RATIO_OK:
            m.export(OUT_MESH / f)
            col_parts[f] = [f]
            continue
        import coacd  # after trimesh/numpy to dodge OpenMP conflicts
        cm = coacd.Mesh(m.vertices, m.faces)
        parts = coacd.run_coacd(cm, threshold=COACD_THRESHOLD)
        names = []
        for k, (v, faces) in enumerate(parts):
            name = f"{Path(f).stem}_col{k}.STL"
            trimesh.Trimesh(v, faces).export(OUT_MESH / name)
            names.append(name)
        col_parts[f] = names
        print(f"[coacd] {f}: ratio={ratio:.3f} -> {len(names)} parts")
    return col_parts


# ---------------------------------------------------------------- MJCF build

def build_mjcf(links, children, root_link, col_parts):
    mj = ET.Element("mujoco", model=f"sharpa_wave_{SIDE}")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(mj, "option", cone="elliptic", impratio="10")

    default = ET.SubElement(mj, "default")
    d = ET.SubElement(default, "default", {"class": "sharpa_wave"})
    ET.SubElement(d, "joint", axis="0 0 1", damping="0.1")
    ET.SubElement(d, "position", kp="5", forcerange="-3 3")
    dv = ET.SubElement(d, "default", {"class": "visual"})
    ET.SubElement(dv, "geom", type="mesh", contype="0", conaffinity="0", group="2",
                  density="0", material="black")
    dc = ET.SubElement(d, "default", {"class": "collision"})
    ET.SubElement(dc, "geom", type="mesh", group="3")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "material", name="black", rgba="0.2 0.2 0.2 1")
    ET.SubElement(asset, "material", name="white", rgba="0.9 0.9 0.9 1")
    declared = set()

    def declare_mesh(fname):
        stem = Path(fname).stem
        if stem not in declared:
            ET.SubElement(asset, "mesh", file=fname)
            declared.add(stem)
        return stem

    world = ET.SubElement(mj, "worldbody")
    actuators = []

    def geom_els(body_el, link, extra_pos=None, extra_quat=None):
        """Add visual+collision geoms of `link` to body_el, optionally offset by a fixed-joint tf."""
        def compose(pos, quat):
            if extra_pos is None:
                return pos, quat
            Rq = R.from_quat(np.roll(extra_quat, -1))  # wxyz -> xyzw
            return extra_pos + Rq.apply(pos), np.roll((Rq * R.from_quat(np.roll(quat, -1))).as_quat(), 1)

        for f, pos, quat in link.visuals:
            p, q = compose(pos, quat)
            mat = "white" if "elastomer" in f else "black"
            ET.SubElement(body_el, "geom", {"class": "visual", "mesh": declare_mesh(f),
                          "pos": fmt(p), "quat": fmt(q), "material": mat})
        for f, pos, quat in link.collisions:
            p, q = compose(pos, quat)
            for part in col_parts[f]:
                ET.SubElement(body_el, "geom", {"class": "collision", "mesh": declare_mesh(part),
                              "pos": fmt(p), "quat": fmt(q)})

    def inertial_el(body_el, link):
        mass, com, I = link.mass, link.com, link.inertia_link
        if mass is None:
            return
        if mass < MIN_MASS:
            mass = MIN_MASS
            I = np.eye(3) * MIN_INERTIA
        else:
            I = I.copy()
            for k in range(3):
                I[k, k] = max(I[k, k], MIN_INERTIA)
        full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
        ET.SubElement(body_el, "inertial", pos=fmt(com), mass=f"{mass:.6g}",
                      fullinertia=fmt(full, prec=8))

    def add_body(parent_el, link_name, joint=None):
        link = links[link_name]
        attrs = {"name": link_name}
        if joint is not None:
            attrs["pos"] = fmt(joint.pos)
            attrs["quat"] = fmt(joint.quat)
        body_el = ET.SubElement(parent_el, "body", attrs)
        if joint is not None and joint.type == "revolute":
            ET.SubElement(body_el, "joint", name=joint.name, range=fmt(joint.range))
            actuators.append(joint)
        inertial_el(body_el, link)
        geom_els(body_el, link)
        # recurse; merge fixed-joint children (elastomer), drop empty fingertip frames
        for j in children.get(link_name, []):
            child = links[j.child]
            if j.type == "fixed":
                if child.visuals or child.collisions:
                    geom_els(body_el, child, extra_pos=j.pos, extra_quat=j.quat)
                # descend through fixed chains (fingertip below elastomer: empty, skipped)
                for jj in children.get(j.child, []):
                    if links[jj.child].visuals or links[jj.child].collisions:
                        raise NotImplementedError("nested fixed link with geometry")
            else:
                add_body(body_el, j.child, j)
        return body_el

    root_el = add_body(world, root_link)
    root_el.set("childclass", "sharpa_wave")

    contact = ET.SubElement(mj, "contact")
    for b1, b2 in EXTRA_EXCLUDES:
        ET.SubElement(contact, "exclude", body1=b1, body2=b2)

    act_el = ET.SubElement(mj, "actuator")
    for j in actuators:
        ET.SubElement(act_el, "position", name=j.name + "_act", joint=j.name,
                      ctrlrange=fmt(j.range))

    xml_str = minidom.parseString(ET.tostring(mj)).toprettyxml(indent="  ")
    xml_str = "\n".join(l for l in xml_str.split("\n") if l.strip())
    (OUT / XML_NAME).write_text(xml_str)
    print(f"[mjcf] wrote {OUT / XML_NAME} ({len(actuators)} actuated joints)")
    return [j.name for j in actuators]


# ------------------------------------------------------- annotation yaml files

def load_spheres():
    data = yaml.safe_load(SPHERES_YML.read_text())["collision_spheres"]
    return {k: [(np.array(s["center"]), float(s["radius"])) for s in v] for k, v in data.items()}


def build_skeleton(spheres):
    skel = {}
    for link, sph in spheres.items():
        if link == _s("right_hand_C_MC"):
            c = [s[0] for s in sph]
            segs = [np.concatenate([c[1], c[0]]), np.concatenate([c[3], c[2]])]
        else:
            # chain consecutive sphere centers, extended by the end-sphere radii
            c = [s[0] for s in sph]
            r = [s[1] for s in sph]
            segs = []
            for a, b in zip(range(len(c) - 1), range(1, len(c))):
                d = c[b] - c[a]
                n = d / (np.linalg.norm(d) + 1e-12)
                p0 = c[a] - n * r[a] if a == 0 else c[a]
                p1 = c[b] + n * r[b] if b == len(c) - 1 else c[b]
                segs.append(np.concatenate([p0, p1]))
        skel[link] = [[round(float(x), 6) for x in s] for s in segs]
    with (OUT / "skeleton.yaml").open("w") as f:
        yaml.safe_dump(skel, f, default_flow_style=None, sort_keys=False)
    print(f"[skeleton] {sum(len(v) for v in skel.values())} segments for {len(skel)} bodies")


def collect_body_meshes(links, children):
    """Per-body collision surface (original STLs, fixed-joint children merged), in body frame."""
    meshes = {}
    for name, link in links.items():
        parts = []
        for f, pos, quat in link.collisions:
            m = trimesh.load(URDF_MESH_DIR / f, force="mesh")
            parts.append(m)
        for j in children.get(name, []):
            if j.type != "fixed":
                continue
            child = links[j.child]
            for f, pos, quat in child.collisions:
                m = trimesh.load(URDF_MESH_DIR / f, force="mesh").copy()
                T = np.eye(4)
                T[:3, :3] = R.from_quat(np.roll(j.quat, -1)).as_matrix()
                T[:3, 3] = j.pos
                m.apply_transform(T)
                parts.append(m)
        if parts:
            meshes[name] = trimesh.util.concatenate(parts)
    return meshes


def build_keypoints(spheres, body_meshes, palmar_dirs):
    """keypoint.yaml: for every collision sphere, project center onto the body's collision
    surface along the palmar direction (fallback: nearest point). 6D = pos + outward normal."""
    kp = {}
    report = []
    for link, sph in spheres.items():
        mesh = body_meshes[link]
        # DP links: the elastomer pad faces local +Y (verified via elastomer face-area
        # stats, identical for thumb and fingers) — more reliable than the neutral-pose
        # palmar direction, which is wrong for the thumb.
        pdir = np.array([0.0, 1.0, 0.0]) if link.endswith("_DP") else palmar_dirs[link]
        entries = []
        for i, (c, r) in enumerate(sph):
            hit = None
            if pdir is not None:
                loc, _, tri = mesh.ray.intersects_location([c], [pdir], multiple_hits=False)
                if len(loc):
                    hit, normal = loc[0], mesh.face_normals[tri[0]]
            if hit is None:
                prox = trimesh.proximity.ProximityQuery(mesh)
                cl, dist, tri = prox.on_surface([c])
                hit, normal = cl[0], mesh.face_normals[tri[0]]
            # smooth the normal over a 3mm neighborhood (single-face normals can
            # catch seam bevels, e.g. the elastomer/DP notch)
            centroids = mesh.triangles_center
            near = np.linalg.norm(centroids - hit, axis=1) < 0.003
            if near.any():
                cand = mesh.face_normals[near]
                # only average faces facing the same way as the hit face
                cand = cand[cand @ normal > 0.3]
                if len(cand):
                    areas = mesh.area_faces[near][mesh.face_normals[near] @ normal > 0.3]
                    normal = (cand * areas[:, None]).sum(0)
                    normal /= np.linalg.norm(normal)
            if np.dot(normal, hit - c) < 0:
                normal = -normal
            entries.append([round(float(x), 6) for x in np.concatenate([hit, normal])])
            report.append((f"{link}/{i}", hit, normal))
        kp[link] = entries
    with (OUT / "keypoint.yaml").open("w") as f:
        yaml.safe_dump(kp, f, default_flow_style=None, sort_keys=False)
    print(f"[keypoint] {len(report)} keypoints")
    for name, hit, n in report:
        print(f"   {name:24s} pos=({fmt(hit)})  n=({fmt(n, 3)})")


def build_body_group():
    groups = [
        [_s("right_index_DP"), _s("right_index_MP"), _s("right_index_PP")],
        [_s("right_middle_DP"), _s("right_middle_MP"), _s("right_middle_PP")],
        [_s("right_ring_DP"), _s("right_ring_MP"), _s("right_ring_PP")],
        [_s("right_pinky_DP"), _s("right_pinky_MP"), _s("right_pinky_PP")],
        [_s("right_thumb_DP"), _s("right_thumb_PP"), _s("right_thumb_MC")],
        [_s("right_hand_C_MC"), _s("right_pinky_MC")],
    ]
    with (OUT / "body_group.yaml").open("w") as f:
        yaml.safe_dump({"body_group": groups}, f, default_flow_style=None, sort_keys=False)
    print("[body_group] written")


# ---------------------------------------------------------------- validation

def validate(joint_names, spheres):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(OUT / XML_NAME))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"\n[validate] nq={model.nq} nu={model.nu} nbody={model.nbody} mass={sum(model.body_mass):.3f}kg")
    mj_joints = [model.joint(i).name for i in range(model.njnt)]
    assert mj_joints == joint_names, "MJCF joint order != URDF order"
    print("[validate] MJCF joint order matches URDF/ocir joint_order: OK")

    # neutral-pose self-penetrations -> candidates for contact excludes
    bad = {}
    for i in range(data.ncon):
        con = data.contact[i]
        if con.dist < -1e-5:
            b1 = model.body(model.geom_bodyid[con.geom1]).name
            b2 = model.body(model.geom_bodyid[con.geom2]).name
            key = tuple(sorted([b1, b2]))
            bad[key] = min(bad.get(key, 0), con.dist)
    if bad:
        print("[validate] penetrating pairs at qpos0 (consider excludes):")
        for (b1, b2), d in sorted(bad.items(), key=lambda kv: kv[1]):
            print(f"   {b1} <-> {b2}  depth={-d*1000:.2f}mm")
    else:
        print("[validate] no self-penetration at qpos0")

    # frame conventions: fingertips should extend along +Z of the palm/root frame
    palmar_dirs = {}
    for link in spheres:
        bid = model.body(link).id
        Rw = data.xmat[bid].reshape(3, 3)
        palmar_dirs[link] = Rw.T @ np.array([1.0, 0, 0])  # world +X (= palm normal) in body frame
    tips = {l: data.xpos[model.body(l).id] for l in
            [_s(n) for n in ["right_index_DP", "right_middle_DP", "right_ring_DP",
                             "right_pinky_DP", "right_thumb_DP"]]}
    palm_z = data.xpos[model.body(_s("right_hand_C_MC")).id][2]
    print("[validate] fingertip DP world pos (palm at origin, +Z should dominate):")
    for l, p in tips.items():
        print(f"   {l:20s} {fmt(p)}")
    return palmar_dirs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=("left", "right"))
    args = ap.parse_args()
    set_side(args.side)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_MESH.mkdir(parents=True, exist_ok=True)
    print(f"[side] {args.side}   URDF={URDF.name}   OUT={OUT}")
    links, children, root_link = parse_urdf()
    col_parts = prepare_meshes(links)
    joint_names = build_mjcf(links, children, root_link, col_parts)
    spheres = load_spheres()
    palmar_dirs = validate(joint_names, spheres)
    build_skeleton(spheres)
    body_meshes = collect_body_meshes(links, children)
    build_keypoints(spheres, body_meshes, palmar_dirs)
    build_body_group()


if __name__ == "__main__":
    main()
