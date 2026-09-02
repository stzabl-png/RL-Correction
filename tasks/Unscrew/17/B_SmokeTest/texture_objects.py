"""运行时给 Sim 物体上图案纹理 (U14, 2026-09-02 用户裁定"所有物体都要有图案和纹理"):
重建网格无 UV/无 mtl/USD 无材质 -> 回转体用**圆柱投影**现场生成 primvars:st,
绑 UsdPreviewSurface + UsdUVTexture (程序化图案 PNG); 桌面平面投影。
用法: from texture_objects import apply_textures; apply_textures(E)  (场景 build 后调用)
已知限制: 环向 UV 有一条接缝色带 (vertex 插值 atan2 回绕), 只影响观感不影响判读。
"""
import os
import numpy as np

_D = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "..", "..", "datasets", "unscrew_bottle", "17",
                                  "cache", "textures"))


def _bind_tex(stage, mesh_prim, tex_png, name, st):
    from pxr import Sdf, UsdGeom, UsdShade
    gm = UsdGeom.Mesh(mesh_prim)
    pv = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set([(float(a), float(b)) for a, b in st])
    lp = f"/World/Looks/U14_{name}"
    mat = UsdShade.Material.Define(stage, lp)
    sh = UsdShade.Shader.Define(stage, lp + "/pbr")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    rd = UsdShade.Shader.Define(stage, lp + "/stReader")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(stage, lp + "/tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_png)
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        rd.ConnectableAPI(), "result")
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tx.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim()).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants)


def _meshes_under(stage, root_path):
    from pxr import Usd, UsdGeom
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return []
    return [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]


def _cyl_st(pts, wrap=1.0):
    p = np.asarray(pts, float)
    u = (np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi) + 0.5) * wrap
    z0, z1 = p[:, 2].min(), p[:, 2].max()
    v = (p[:, 2] - z0) / max(z1 - z0, 1e-6)
    return np.stack([u, v], 1)


def _planar_st(pts, scale=1.0):
    p = np.asarray(pts, float)
    return np.stack([(p[:, 0] - p[:, 0].min()) * scale,
                     (p[:, 1] - p[:, 1].min()) * scale], 1)


def apply_textures(E):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    n_ok = 0
    for ob, name, tag in ((getattr(E, "object", None), "bottle", "瓶"),
                          (getattr(E, "aux", None), "cap", "盖")):
        if ob is None:
            continue
        try:
            root = ob.cfg.prim_path.replace("env_.*", "env_0")
        except Exception:
            continue
        # 真实 SAM3D 纹理优先 (U15): egodex_auto 的 textured glb 抽出的 UV+贴图,
        # 最近邻迁移到场景 USD 顶点; 缺档退程序化图案。
        real_npz = os.path.join(_D, f"real_{name}_uv.npz")
        real_png = os.path.join(_D, f"real_{name}.png")
        use_real = os.path.isfile(real_npz) and os.path.isfile(real_png)
        if use_real:
            from scipy.spatial import cKDTree
            _z = np.load(real_npz)
            _tree = cKDTree(_z["verts"])
            _uv = np.asarray(_z["uv"], float)
        for i, mp in enumerate(_meshes_under(stage, root)):
            pts = np.asarray(UsdGeom.Mesh(mp).GetPointsAttr().Get(), float)
            if use_real:
                _d, _j = _tree.query(pts, k=1)
                st = _uv[_j]
                _bind_tex(stage, mp, real_png, f"{tag}_{i}", st)
            else:
                _bind_tex(stage, mp, os.path.join(_D, f"{name}.png"), f"{tag}_{i}", _cyl_st(pts))
            n_ok += 1
        print(f"[U14纹理] {tag}: {root} 绑{'真实 SAM3D 纹理' if use_real else '程序图案'}"
              + (f" (最近邻中位 {np.median(_d)*1000:.1f}mm)" if use_real else ""), flush=True)
    # 桌面: /World 下找名字含 Table 的 mesh
    from pxr import Usd
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if "table" in p.GetName().lower() and p.IsA(UsdGeom.Mesh):
            pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            _bind_tex(stage, p, os.path.join(_D, "table.png"), "table", _planar_st(pts, 4.0))
            print(f"[U14纹理] 桌: {p.GetPath()}", flush=True)
            n_ok += 1
            break
    print(f"[U14纹理] 共 {n_ok} 个 mesh 上妆", flush=True)
    return n_ok
