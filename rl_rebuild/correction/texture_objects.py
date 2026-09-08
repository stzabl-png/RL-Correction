"""运行时给 Sim 物体上图案纹理 (U14, 2026-09-02 用户裁定"所有物体都要有图案和纹理"):
重建网格无 UV/无 mtl/USD 无材质 -> 回转体用**圆柱投影**现场生成 primvars:st,
绑 UsdPreviewSurface + UsdUVTexture (程序化图案 PNG); 桌面平面投影。
用法: from texture_objects import apply_textures; apply_textures(E)  (场景 build 后调用)
已知限制: 环向 UV 有一条接缝色带 (vertex 插值 atan2 回绕), 只影响观感不影响判读。
"""
import os
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_D = os.path.join(_REPO, "datasets", "unscrew_bottle", "17", "cache", "textures")   # 默认 = unscrew17


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


def _attach_visual(stage, root, vis_json, tag):
    """U16: 视觉网格直接换成 SAM3D 带纹理 USD (挂 root 子节点, 跟随刚体), 原 CAD 视觉隐藏、碰撞保留。
    UV 迁移的碎花病根 (图集跨块插值) 由此绕开。"""
    import json as _json
    from pxr import Gf, Sdf, Usd, UsdGeom
    cfg = _json.load(open(vis_json))
    xp = root + "/TexturedVis"
    x = UsdGeom.Xform.Define(stage, xp)
    _u = cfg["usd"]
    if not os.path.isabs(_u) or not os.path.exists(_u):   # 相对路径/异机: 相对 json 所在目录解析 (2026-09-07)
        _u = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(vis_json)), _u))
    x.GetPrim().GetReferences().AddReference(_u)
    M = cfg["matrix"]
    # U16.1 定案 (2026-09-02, 离线合成探针): USD 行向量约定 p·M, 递 M_math^T。
    # (unscrew 瓶实测: 转置 6.1mm/z[0,0.197] ✓, 不转置 78mm 躺倒 ✗。此前"去转置"
    #  是被 pour 配对错误的坏矩阵污染的误判 —— 两个 bug 叠加互为烟雾弹。)
    m = Gf.Matrix4d(*[M[r][c] for r in range(4) for c in range(4)]).GetTranspose()
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(m)
    n_hide = 0
    for pr in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if str(pr.GetPath()).startswith(xp):
            continue
        if pr.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(pr).MakeInvisible()
            n_hide += 1
    print(f"[U16纹理] {tag}: 视觉=SAM3D USD (NN {cfg.get('nn_med_mm','?')}mm, physAPI {cfg.get('phys_apis')}), 隐藏原视觉 {n_hide} mesh", flush=True)


def apply_textures(E, tex_dir=None, names=(("object", "bottle", "瓶"), ("aux", "cap", "盖"))):
    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    n_ok = 0
    _D2 = tex_dir or _D
    for attr, name, tag in names:
        ob = getattr(E, attr, None)
        if ob is None:
            continue
        try:
            root = ob.cfg.prim_path.replace("env_.*", "env_0")
        except Exception:
            continue
        # 真实 SAM3D 纹理优先 (U15): egodex_auto 的 textured glb 抽出的 UV+贴图,
        # 最近邻迁移到场景 USD 顶点; 缺档退程序化图案。
        vis_json = os.path.join(_D2, f"real_{name}_visual.json")
        if os.path.isfile(vis_json):
            try:
                _attach_visual(stage, root, vis_json, tag)
                n_ok += 1
                continue
            except Exception as _ae:
                print(f"[U16纹理] {tag} 挂载失败 ({_ae}), 退 UV 迁移", flush=True)
        real_npz = os.path.join(_D2, f"real_{name}_uv.npz")
        real_png = os.path.join(_D2, f"real_{name}.png")
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
                _bind_tex(stage, mp, os.path.join(_D2, f"{name}.png"), f"{tag}_{i}", _cyl_st(pts))
            n_ok += 1
        print(f"[U14纹理] {tag}: {root} 绑{'真实 SAM3D 纹理' if use_real else '程序图案'}"
              + (f" (最近邻中位 {np.median(_d)*1000:.1f}mm)" if use_real else ""), flush=True)
    # 桌面: /World 下找名字含 Table 的 mesh
    from pxr import Usd
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if "table" in p.GetName().lower() and p.IsA(UsdGeom.Mesh):
            pts = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
            _bind_tex(stage, p, os.path.join(_D2, "table.png"), "table", _planar_st(pts, 4.0))
            print(f"[U14纹理] 桌: {p.GetPath()}", flush=True)
            n_ok += 1
            break
    print(f"[U14纹理] 共 {n_ok} 个 mesh 上妆", flush=True)
    return n_ok
