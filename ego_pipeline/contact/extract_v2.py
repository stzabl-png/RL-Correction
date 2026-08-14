#!/usr/bin/env python3
"""接触点提取 v2 —— 直接用**人手(MANO)**量, 不做对齐搜索。

============================ 为什么重写 ============================

旧链路(`tools/contact_align_heatmap.py`)的核心是一个**平移搜索**: 把 SharpaWave 的
5 个指垫平移到贴上物体表面。它存在的前提是"手位不可信、要靠 2D 证据反推"。

这个前提今天被两件事推翻:

1. **手位曾经确实错 16cm** —— `fuse` 把 MANO 的 `transl` 当世界点旋转(见
   `fuse/run_sequence.py` 的 J0 注释)。旧脚本的"准"其实是两个错抵消: 手偏 16cm,
   对齐器再推回 15.5cm, 咬合 IoU 看起来 1.000。修好手位后同一份代码 IoU 变 0.000。

2. **探针选错了。** 实测 clip0 接触区间内:

   | 探针 | 到瓶身表面 |
   |---|---|
   | **MANO 人手** | **0.3mm** (130 帧, p90 1.8mm) |
   | SharpaWave 指垫 | **61mm** |

   人手本来就贴着物体, 接触是现成的。那 6.1cm 缺口全部来自"人手→机器手"的重定向
   残差(指尖 FK vs MANO 实测差 1.3~2.0cm)。对齐搜索是在补一个**不该存在**的缺口 ——
   它为了消掉这 6.1cm, 把手整体挪了 21cm 到另一只手的位置(咬合 IoU 归零)。

★ 提取 Video Prior 要回答的是"**人**在视频里碰了物体哪里", 所以探针必须是人手。
  机器手的接触是下游 retarget 的事, 不该混进先验里。

============================ 做法 ============================

    对接触区间内的每一帧:
        d = 物体表面顶点 到 MANO 手网格(778 顶点) 的最近距离
        本帧接触 = d < tau
        ⊗ 2D 否证: 该顶点在本帧**画面上清晰可见**(没被手遮) -> 它没被碰, 强制置 0
    跨帧累计命中次数 -> 归一化成热度图

与旧法的三处不同:
  * **不搜索** —— 手位可信就直接量, 秒级(旧法 246~975 秒)
  * **多帧聚合** 而不是区间中点单帧 —— 单帧受该帧位姿抖动支配
  * **探针是人手全网格** 而不是机器手 5 个指垫 —— 覆盖手掌/侧面的接触

⚠ 仍然继承的限制: 物体位姿本身的误差(clip0 实测 `object projection vs SAM2 mask`
  IoU 只有 0.218)会直接进到接触点里。本模块不修它, 只如实记录 `obj_mask_iou`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _visible_mask(obj_pts_world, hand_pts_world, c2w, K, hw, hand_mask=None):
    """→ (可见且未被手遮 的顶点布尔, 投影像素)。用于 2D 否证。"""
    H, W = hw
    w2c = np.linalg.inv(c2w)
    pc = (w2c[:3, :3] @ obj_pts_world.T).T + w2c[:3, 3]
    ok = pc[:, 2] > 1e-3
    u = np.full(len(pc), -1.0)
    v = np.full(len(pc), -1.0)
    u[ok] = K[0, 0] * pc[ok, 0] / pc[ok, 2] + K[0, 2]
    v[ok] = K[1, 1] * pc[ok, 1] / pc[ok, 2] + K[1, 2]
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    vis = inb.copy()
    if hand_mask is not None:
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(v.astype(int), 0, H - 1)
        covered = hand_mask[vi, ui] > 0
        vis = inb & (~covered)          # 被手盖住的不算"清晰可见"
    return vis, np.stack([u, v], 1)


def hand_mask_agreement(D: Path, side: str, frames, c2w, K, hand, *, n=15):
    """重建的手落在**实测手 mask** 里的比例。→ (比例, 用到的帧数); 没有 mask 时返回 (None, 0)。

    ============================ 为什么要这道体检 ============================

    提取器分不清"手真没碰到"和"手被重建到了错的地方" —— 两者都表现为
    no_stable_contact。这一项直接问: 重建出来的手, 在图像里跟真实的手重合吗?

    实测标定(同一批 take, 好/坏两份 replay 对照):

      | 数据 | 左手 | 右手 |
      |---|---|---|
      | take0 (正常) | 0.555 | 0.589 |
      | take1/4 修好后 | 0.616~0.625 | 0.621~0.625 |
      | take1/4 **陈旧 replay** | **0.000** | **0.000~0.009** |

    分离干净(0.55+ vs 0.01-), 门槛 0.25 两边都留足余量。

    ⚠ EgoDex 的手 mask **含小臂**而 MANO 没有, 所以正确时也到不了 1.0, 0.55~0.63 就是
      正常水平 —— 别把"没到 0.9"当成有问题。
    """
    import cv2
    md = D / "masks/hands/frames"
    if not md.is_dir():
        return None, 0
    fr = list(frames)[::max(1, len(frames) // n)] if frames else []
    vals = []
    for t in fr:
        m = cv2.imread(str(md / f"frame_{t:06d}_masks" / f"{side}_hand_0.png"), 0)
        if m is None or (m > 127).sum() < 200 or t >= len(hand) or t >= len(c2w):
            continue
        Kc = np.linalg.inv(c2w[t])
        P = (Kc[:3, :3] @ hand[t].T).T + Kc[:3, 3]
        P = P[P[:, 2] > 1e-6]
        if len(P) < 10:
            vals.append(0.0)
            continue
        u = P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2]
        v = P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]
        H, W = m.shape
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        vals.append(0.0 if ok.sum() < 10 else
                    float((m[v[ok].astype(int), u[ok].astype(int)] > 127).mean() * ok.mean()))
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


def _azimuth_span(V: np.ndarray, w: np.ndarray, thr: float = 0.5) -> int:
    """热点绕物体主轴的方位跨度(度)。12 个 30° 扇区里被占用的数量 × 30。

    ★ 这是**下界**不是真值: 自遮挡门只保留相机可见的一面, 背面的真接触不可观测。
      下游拿它当"包裹范围"时只能当作"至少这么多"。
    """
    hot = w >= thr
    if not hot.any():
        return 0
    ax = int(np.argmax(np.ptp(V, axis=0)))
    rad = [i for i in range(3) if i != ax]
    c = V[:, rad].mean(0)
    ang = np.degrees(np.arctan2(*(V[hot][:, rad] - c).T[::-1])) % 360
    return int((np.histogram(ang, bins=12, range=(0, 360))[0] > 0).sum() * 30)


def _read_segs(D: Path, object_id: str, side: str) -> list:
    ca = D / f"contact_auto_{object_id}.json"
    if not ca.is_file():
        ca = D / "contact_auto.json"
    if not ca.is_file():
        return []
    return (json.loads(ca.read_text()).get("annotations") or {}).get(side) or []


def _gap_no_depth(obj_w, hand_w, c2w_t, K, *, k=16):
    """手到物体的距离, **扔掉沿视线(深度)那一维**。→ (每个物体点的面内距离 m, 沿视线距离 m)

    ============================ 为什么扔掉深度 ============================

    单目重建里深度是**近乎不可观测**的: 剪影类的能量项沿视线方向是平的, 手和物体各自
    都能沿射线滑动而不改变图像。于是 3D 距离里的深度分量基本是噪声, 面内分量才是信息。

    实测 clip0 左手×瓶身稳定窗(逐手指, 3D 最近距离 -> 分解):

      | 部位 | 3D | 沿视线 | 面内 | 图像里离瓶身 |
      |---|---|---|---|---|
      | 拇指 | 41.5mm | **40.6** | 9.9 | **0.1 px** |
      | 食/中/无名 | 0.3~0.4mm | 0.2 | 0.3 | 0.0~0.1 px |
      | 小指 | 4.3mm | 0.9 | 4.3 | 1.7 px |
      | 掌 | 62.2mm | 26.9 | 55.9 | **159.6 px** |

    拇指在**图像里就贴在瓶身上(0.1 像素)**, 3D 却差 41.5mm —— 其中 40.6mm 是深度。
    用 3D 距离判就漏掉拇指, 而握瓶子不可能没有拇指(与 ARCTIC 实测的"主病是漏指"一致)。
    ★ 同时手掌在图像里也是远的(159.6px, 而瓶身在图里才 168px 宽) —— 所以换成面内判据
      **不会把什么都判成接触**, 它恰好救回该救的、拒掉该拒的。这是本设计的可证伪点。

    ============================ 为什么不能只看 2D ============================

    只看像素会把"物体背面"和"手前方 30cm 的物体"一起算进来。所以:
      1. 先在**像素平面**找最近邻(而不是 3D 最近邻 —— 3D 最近邻会被深度误差带偏);
      2. 在这 k 个候选里挑**沿视线差最小**的那个 —— 手绕到物体背面时自动选背面, 不需要
         可见性假设(要求"朝向相机"会丢掉环握时绕到背后的那几根手指);
      3. 沿视线差仍然设一个**宽**上限, 只用来毙掉"隔着半个房间投影重合"这种粗错。
    """
    from scipy.spatial import cKDTree
    Kc = np.linalg.inv(c2w_t)
    Oc = (Kc[:3, :3] @ obj_w.T).T + Kc[:3, 3]
    Hc = (Kc[:3, :3] @ hand_w.T).T + Kc[:3, 3]
    zo = np.clip(Oc[:, 2], 1e-6, None)
    zh = np.clip(Hc[:, 2], 1e-6, None)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    uo = np.stack([Oc[:, 0] / zo * fx + K[0, 2], Oc[:, 1] / zo * fy + K[1, 2]], 1)
    uh = np.stack([Hc[:, 0] / zh * fx + K[0, 2], Hc[:, 1] / zh * fy + K[1, 2]], 1)
    kk = min(k, len(uh))
    dpx, idx = cKDTree(uh).query(uo, k=kk)
    if kk == 1:
        dpx, idx = dpx[:, None], idx[:, None]
    dz = np.abs(zh[idx] - zo[:, None])
    pick = np.argmin(dz, axis=1)                       # 深度最接近的那个手点
    r = np.arange(len(uo))
    # 像素距离换成米: 用**物体点自己的深度**, 免得手的深度误差再污染一次
    inplane = dpx[r, pick] * zo / ((fx + fy) * 0.5)
    return inplane, dz[r, pick]


def extract(recon_dir: Path, object_id: str, side: str, *, tau_mm: float = 8.0,
            max_frames: int = 60, use_2d_veto: bool = True, use_normal_gate: bool = True,
            stable_r_m: float = 0.015, touch_gap_m: float = 0.010,
            min_window: int = 5, n_probe: int = 60000, min_opposition: float = 0.40,
            object_follows_hand: bool = False, min_hand_agreement: float = 0.25,
            depth_blind: bool = True, ray_cap_m: float = 0.10) -> dict:
    import cv2
    import trimesh
    from scipy.spatial import cKDTree

    D = Path(recon_dir)
    w = np.load(D / "world_fused.npz", allow_pickle=True)
    # replay_world.npz 的正规位置是 RetargetOutput; 少数 take 的 recon 目录里也有一份副本
    rp = D / "replay_world.npz"
    if not rp.is_file():
        rp = Path(str(D).replace("ReconstructOutput", "RetargetOutput")) / "replay_world.npz"
    # ★ 时效检查: replay 比 world_fused 旧 -> 里面的手顶点来自上一次 fuse, 必须拒用。
    #   实测 take1/take4 的 replay(00:34) 比 world_fused(01:52) 早 78 分钟, 于是手顶点
    #   还是"修 fuse 手部变换 bug 之前"的版本 —— 投影到图上离真实手 379~432 px
    #   (take0 只有 42~69 px)。当时我误判成"物体位姿差", 还把正确报警的 2D 否证关掉了。
    #   "文件存在" != "结果有效", 这个坑今天已经踩过三次。
    wf = D / "world_fused.npz"
    if rp.is_file() and wf.is_file() and rp.stat().st_mtime < wf.stat().st_mtime:
        raise RuntimeError(
            f"{rp} 比 world_fused.npz 旧 —— 手顶点是上一次 fuse 的结果, 拒用。\n"
            f"  先重生成: python -m ego_pipeline.bridge.recon_to_replay {D}")
    if not rp.is_file():
        raise FileNotFoundError(
            f"缺 replay_world.npz(手的 MANO 网格来源)。找过 {D} 和 RetargetOutput。"
            f"先跑 `python -m ego_pipeline.bridge.recon_to_replay {D}`")
    r = np.load(rp, allow_pickle=True)
    oids = ([str(x) for x in w["object_ids"]] if "object_ids" in w.files else ["object_0"])
    oi = oids.index(object_id)
    allT = (w["object_ob_in_world_all"] if "object_ob_in_world_all" in w.files
            else w["object_ob_in_world"][None])
    c2w, K = w["c2w"], w["K"]
    hand = r[f"mano_verts_{side}"]
    tips = r[f"joints_{side}"][:, [4, 8, 12, 16, 20]]      # 五指尖(OpenPose-21), 判对生用

    mp = D / "objects" / object_id / "object_mesh_scaled_final.obj"
    if not mp.is_file():
        mp = D / "object_mesh_scaled_final.obj"
    mesh = trimesh.load(mp, process=False, force="mesh")
    # ★ 探针点必须在表面**均匀采样**, 不能直接用 mesh.vertices。
    #   CAD 网格的顶点分布可以极端不均: 实测 bottle_body.obj 67618 个顶点里 99.2% 挤在
    #   瓶口螺纹那 20% 高度, **中间 60% 的瓶身一个顶点都没有**(光滑圆柱只用几个大三角面)。
    #   用顶点当探针时, 手握在瓶身中部检测不到, 反而瓶底那 256 个顶点被判成接触 ——
    #   实测热点全落在 z=0~2% 的瓶底, 而视频里手握在中部。
    #   对 SAM3D 重建网格(顶点密集均匀)不会暴露, 对 CAD 必然出错。旧法用的就是采样。
    Vl, _fid = trimesh.sample.sample_surface(mesh, n_probe, seed=0)
    Vl = np.asarray(Vl, np.float64)
    Nl = np.asarray(mesh.face_normals, np.float64)[_fid]   # 每个探针点的外法向, 判对生用

    # ★ 遮挡门用**体素占据**判实体内外, 不用 mesh.contains。
    #   contains 是逐点射线求交, 代价随面数暴涨: 实测 screw/0(CAD 13.5万面) 单个"物体×手"
    #   从 2.9s 涨到 254.4s(87x), pour/17(SAM3D 68.8万面) 直接 OOM 被杀(退出码 137) ——
    #   把 extract_v2 "整条 take 2~5 秒"这个立身之本弄没了。
    #   体素只建一次, 之后是 O(1) 查表(align.ObjectGeometry 同款做法, 实测建表 ~1.7s)。
    #   pitch 跟物体尺寸走: 太粗会把薄壁填实(把真接触误判成穿透), 太细白费内存。

    # ★★ 先验模式: "物体跟着手走" —— 假设一定有抓握, 把物体整体挪到手上。
    #    ⚠ 这是**假设不是测量**, 产物走 contact_prior_*, 不进 GRASP 阶段标签, 不给 RL 当监督。
    #    为什么需要: SAM3D 单件网格的位姿误差(conf_pos 49~52)让手离物体 60~70mm,
    #    88% 是深度但面内也还差 26~28mm, 测不出接触。而证据修不了它 ——
    #    实测用物体自己的 mask 质心去校正**反而更差**(21.9->32.1mm), 因为 mask 被手
    #    挡掉一块, 残缺 mask 的质心与完整投影的质心本就不该对齐, 是有偏估计。
    #    与旧 align.py 同属"画像", 但代价从 1200 次网格搜索降到一次解析平移。
    prior_shift = None
    if object_follows_hand:
        # 先验模式必须关 2D 否证: 物体已经被挪走, 再拿它的投影去问"这块面在图里
        # 看不看得见"就没有意义了 —— 挪过的投影与 mask 必然对不上, 于是**全部**被毙
        # (实测 take1 否证 213307/热点 0, 关掉后热点 12024)。
        use_2d_veto = False
        segs0 = _read_segs(D, object_id, side)
        fr0 = [t for a, b in segs0 for t in range(a, min(b + 1, allT.shape[1], len(hand)))]
        need = []
        for t in fr0[::max(1, len(fr0) // 40)]:
            M = allT[oi, t]
            Vw = (M[:3, :3] @ Vl[::20].T).T + M[:3, 3]
            dd, jj = cKDTree(hand[t]).query(Vw)
            k = int(np.argmin(dd))
            need.append(hand[t][jj[k]] - Vw[k])        # 把物体挪这么多就贴上了
        if need:
            prior_shift = np.median(np.asarray(need), axis=0)   # 全窗一个常量, 不逐帧抖
            allT = allT.copy()
            allT[oi, :, :3, 3] += prior_shift          # 只平移, 朝向仍用 FoundationPose

    segs = _read_segs(D, object_id, side)
    frames = [t for a, b in segs for t in range(a, min(b + 1, allT.shape[1], len(hand)))]
    if not frames:
        return {"status": "no_contact_interval", "object_id": object_id, "side": side}

    # ★ 体检: 重建的手到底在不在真实的手上。不做这一项时, "手被重建错了"会伪装成
    #   "手没碰到物体"(都是 no_stable_contact) —— 实测 take1/4 用陈旧 replay 时手在图里
    #   偏了 379~432px, 我据此误判成"物体位姿差", 还把正确报警的 2D 否证关掉了。
    agree, n_agree = hand_mask_agreement(D, side, frames, c2w, K, hand)
    if agree is not None and agree < min_hand_agreement:
        return {"status": "hand_unreliable", "object_id": object_id, "side": side,
                "hand_mask_agreement": agree, "n_checked": n_agree,
                "threshold": min_hand_agreement,
                "why": ("重建的手只有 %.1f%% 落在实测手 mask 里(正常 55~63%%) —— "
                        "手被重建到了错的位置, 接触结果不可信。常见原因: "
                        "replay_world.npz 陈旧 / 手部重建失败" % (agree * 100))}

    # ★ 只在**稳定窗**内聚合, 不用整个接触区间。
    #   抓握的本质是"手物相对位姿稳定的那一段"; 触碰区间还包含接近、调整、松开。
    #   实测 clip0 左手×瓶身: 整个 130 帧区间里, 手在物体局部系的质心漂了 7~12cm
    #   (逐帧只 3.5mm, 是**慢漂移**不是抖动), 于是接触区一路移动, 130 帧摊平后
    #   **没有任何顶点在半数帧里被碰到**(热点=0)。
    #   佐证: 瓶盖那格的稳定窗 f20~f28, 与"瓶盖 0.1~0.2mm 完美接触"的独立测量(f18~f26)吻合。
    #   ⚠ 判据必须是"**稳定 且 贴合**"两条, 不能只看稳定 —— 手松开之后位置一样可以很"稳"。
    #     (曾据"f107~f128 手到面 37.3mm"举证, **该数已作废**: 那是用 mesh.vertices 当探针
    #      量的, 而 CAD 瓶身 99.2% 顶点堆在瓶口, 瓶身中段无顶点。改表面均匀采样后该段
    #      实测 0.2mm, 是真抓握。判据本身仍然需要两条, 只是这个反例不成立。)
    Vs = Vl[::max(1, len(Vl) // 4000)]                 # 贴合判据用稀疏采样点即可, 省时间
    loc, gap, oppo = [], [], []
    tree_l = cKDTree(Vl)
    for t in frames:
        M = allT[oi, t]
        inv = np.linalg.inv(M)
        loc.append(((inv[:3, :3] @ hand[t].T).T + inv[:3, 3]).mean(0))
        # ★ 逐帧对生度: 五个指尖各自最近的物体表面外法向, 取平均模长的补。
        #   指尖分布在物体两侧(真环握) -> 法向互相抵消 -> 接近 1; 全在一侧 -> 接近 0。
        tl = (inv[:3, :3] @ tips[t].T).T + inv[:3, 3]
        nn = Nl[tree_l.query(tl)[1]]
        nn = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        oppo.append(float(1.0 - np.linalg.norm(nn.mean(0))))
        Vw = (M[:3, :3] @ Vs.T).T + M[:3, 3]
        if depth_blind:
            ip, rz = _gap_no_depth(Vw, hand[t], c2w[t], K)
            ok = rz <= ray_cap_m
            # ★ 没有任何点满足视线上限 -> 本帧**判为远**, 不能退回去报面内距离。
            #   曾经写成 `ip.min() if not ok.any()`, 结果 egodex/pour/17 实测手离物体
            #   756~1137mm(一米开外), 而 2D 投影恰好重合, 于是被报成"间距 0.13mm 贴合",
            #   把一个粗错伪装成完美接触。深度盲是**忽略深度噪声**, 不是忽略深度本身 ——
            #   ray_cap 就是区分二者的那道线, 越过它必须如实说远。
            gap.append(float(ip[ok].min()) if ok.any() else float(rz.min()))
        else:
            gap.append(float(cKDTree(hand[t]).query(Vw)[0].min()))
    loc, gap, oppo = np.array(loc), np.array(gap), np.array(oppo)

    # ★ 判据是"稳定 且 贴合 且 **对生**"三条。只有前两条时实测挑中 clip0 左手×瓶身的
    #   f107~f128 —— 那是**松手阶段**(张开度 87->122mm, 对生度掉到 0.1x), 手松开之后
    #   位置一样可以又稳又贴。而 ARKit 真值显示该手在 f15~f100 才是真环抱(指尖方位角差
    #   中位 142°, f27~f54 达 167~180°)。
    #   ⚠ 对生度必须**进选窗判据**, 不能选完再报警 —— 事后算只能告诉你选错了, 挡不住。
    #   门槛标定(五指尖均匀铺开 θ): 180°->0.52, 150°->0.45, 120°->0.25, 90°->0.10。
    #   实测 clip0 左手×瓶身逐帧: f54 对生 0.50(方位角 180°), f110~126 只有 0.34~0.36
    #   (106~127°) —— 所以门槛必须 >0.36, 取 0.40(≈要求环抱 140°)。
    close = ((gap <= max(tau_mm / 1000.0, touch_gap_m))
             & (oppo >= min_opposition))                  # 本帧手确实贴着物体 且 对生
    # ★ 记录**全部**极大窗, 不只冠军。一条视频里可以有多个互不相同的稳定抓握 ——
    #   clip0 左手×瓶身实测有两个: f41~f60(拧盖时扶瓶, 20 帧) 与 f107~f128(拧完后拿起,
    #   22 帧)。冠军只赢 2 帧, 但两者是**不同抓法**。下游选 GraspPose 模板需要知道
    #   有几个候选, 只给一个会让"这条视频只有一种抓法"成为无根据的默认。
    i, cands = 0, []                                   # cands: (i, j) 每个 j 的极大窗
    for j in range(len(frames)):
        if not close[j]:                               # 不贴合的帧直接断开窗口
            i = j + 1
            cands.append(None)
            continue
        while i <= j and np.linalg.norm(loc[i:j + 1] - loc[i:j + 1].mean(0),
                                        axis=1).max() > stable_r_m:
            i += 1
        cands.append((i, j))
    # (i,j) 被 (i,j+1) 包含时丢弃, 只留真正的极大窗
    wins = [c for k, c in enumerate(cands)
            if c and (k + 1 >= len(cands) or cands[k + 1] is None or cands[k + 1][0] != c[0])]
    wins = sorted(((j - i + 1, i, j) for i, j in wins), reverse=True)
    # 极大窗之间还会大量重叠(同一段抓握左右滑一两帧就是一个新的"极大窗", 实测 51 个)。
    # 贪心去重: 长的优先, 与已收窗重叠超一半的丢掉 —— 留下的才是**不同的**抓握。
    kept = []
    for n, a, b in wins:
        if all(min(b, y) - max(a, x) + 1 <= 0.5 * min(n, m) for m, x, y in kept):
            kept.append((n, a, b))
    wins = kept
    best = wins[0] if wins else (0, 0, 0)
    win_n, wi, wj = best
    # ★ 最小帧数门槛: 太短的窗给出的"热点"没有统计意义 —— 窗只有 2 帧时, 任何被碰到的
    #   顶点都必然出现在 >=50% 的帧里, 于是"热点 588"纯属分母太小的产物(clip0 右手×瓶身
    #   实测就是这样: f74~f75 两帧, 588 个接触顶点全成了热点)。
    #   而那本来就该判"无稳定抓握" —— 右手在拧**盖**, 跨在盖与瓶口交界处, 对瓶身不是抓握。
    if 0 < win_n < min_window:
        win_n = 0
    if win_n == 0:                                     # 没有既稳又贴的段 -> 如实报告
        n_touch = int((gap <= max(tau_mm / 1000.0, touch_gap_m)).sum())
        n_opp = int((oppo >= min_opposition).sum())
        return {"status": "no_stable_contact", "object_id": object_id, "side": side,
                "n_interval_frames": len(frames),
                # ★ 分开报: 早先只给合并后的 close, 于是"贴着但不对生"和"根本够不着"
                #   都显示成"0 帧贴合", 病因指错。
                "n_touch_frames": n_touch, "n_opposed_frames_": n_opp,
                "blocker": ("够不着" if n_touch == 0 else
                            "贴合但不对生(不是抓握)" if n_opp == 0 else "窗口太短"),
                "n_close_frames": int(close.sum()),
                "gap_median_mm": float(np.median(gap) * 1000),
                "gap_min_mm": float(gap.min() * 1000),
                "opposition_median": float(np.median(oppo)),
                "n_opposed_frames": int((oppo >= min_opposition).sum()),
                "longest_window": int(best[0]), "min_window": min_window,
                "why": ("窗口 %d 帧 < 门槛 %d 帧, 判无稳定抓握" % (best[0], min_window)
                        if best[0] else "没有既稳又贴的帧段")}
    stable = {"n": int(win_n), "span": [int(frames[wi]), int(frames[wj])],
              "frac_of_interval": float(win_n / len(frames)),
              "radius_mm": stable_r_m * 1000,
              "close_frames": int(close.sum()), "close_frac": float(close.mean()),
              "gap_in_window_mm": float(np.median(gap[wi:wj + 1]) * 1000),
              "opposition_in_window": float(np.median(oppo[wi:wj + 1])),
              "opposition_over_interval": float(np.median(oppo)),
              "min_opposition": min_opposition,
              "drift_over_interval_mm": float(np.linalg.norm(loc - loc.mean(0), axis=1).max() * 1000),
              "n_candidates": len(wins),
              "candidates": [{"span": [int(frames[a]), int(frames[b])], "n": int(n),
                              "gap_mm": round(float(np.median(gap[a:b + 1]) * 1000), 2)}
                             for n, a, b in wins[:5] if n >= min_window]}
    frames = frames[wi:wj + 1]
    if len(frames) > max_frames:
        frames = list(np.array(frames)[np.linspace(0, len(frames) - 1, max_frames).astype(int)])

    hits = np.zeros(len(Vl), np.int32)
    vetoed = np.zeros(len(Vl), np.int32)
    wrong_side = np.zeros(len(Vl), np.int32)
    occ = None
    if use_normal_gate:
        # 遮挡门用**体素占据**判"实体内", 不用 mesh.contains。
        #   contains 在没有 embree 的机器上退化成纯 Python 射线求交(UCB 实测把 ssh 拖断);
        #   即使有 embree, 代价也随面数暴涨 —— 实测 screw/0(CAD 13.5万面) 单个"物体×手"
        #   2.9s -> 254.4s(87x), pour/17(SAM3D 68.8万面) 直接 OOM 被杀(退出码 137)。
        #   而 extract_v2 存在的全部理由就是"整条 take 2~5 秒"(旧提取器 8 分钟/手)。
        #   体素建一次 ~0.1s, 之后查询 O(1)。3mm 对"手在壁的哪一侧"足够。
        # ⚠ 这里曾经**建了两次**体素(两个 Agent 并行改同一文件, 一份成了死代码从未被调用),
        #   白付一次建表: screw/0 22.8s/4.2GB -> 7.0s/1.2GB, pour/17 10.1s/1.1GB -> 7.8s/0.6GB。
        try:
            occ = mesh.voxelized(pitch=0.003).fill()
        except Exception as e:
            print(f"  [warn] 体素化失败({e}), 遮挡门本次关闭", flush=True)
            occ = None
    seen = 0
    tau = tau_mm / 1000.0
    dmins = []
    for t in frames:
        M = allT[oi, t]
        Vw = (M[:3, :3] @ Vl.T).T + M[:3, 3]
        if depth_blind:
            d, rz = _gap_no_depth(Vw, hand[t], c2w[t], K)   # 面内距离(深度已扔掉)
            touch = (d < tau) & (rz <= ray_cap_m)
        else:
            d = cKDTree(hand[t]).query(Vw)[0]          # 每个物体顶点到人手的最近距离
            touch = d < tau
        dmins.append(float(d.min()))
        if use_normal_gate and occ is not None:
            # ★ 自遮挡门: 从接触点朝相机步进, 撞到自身实体 => 该点在物体背面/内壁。
            #   ego 视角下手在相机与物体之间, 背面的点手够不着 —— 薄壁物体上"最近点"
            #   分不清内外壁(pour/17 杯子 52% 热点在杯内), 下游会把手指伸进杯子里。
            #   ⚠ 前两版都错了, 记在这里避免重犯:
            #     ① 用面法向判 —— 重建网格 39~46% 的面法向指向体内, fix_normals 无效;
            #     ② 用全 3D 连线穿透判 —— 与 depth_blind 的面内口径冲突, 深度误差让连线
            #        穿过物体, 热点被误杀 92%(7195→273);
            #     ③ 用 z-buffer —— 6 万稀疏采样点建的深度图每像素摊不到 1 个样本, 门形同
            #        虚设(实测通过后仍有 61~80% 的点朝相机被自身遮挡)。
            #   本版用体素占据沿视线步进, 与口径无关, 也不依赖法向。
            idx = np.where(touch)[0]
            if len(idx):
                hw = Vw[idx]
                dv = c2w[t][:3, 3] - hw
                dv /= np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
                blocked = np.zeros(len(idx), bool)
                for step in (0.004, 0.007, 0.010, 0.014, 0.020, 0.028):
                    ql = ((hw + dv * step) - M[:3, 3]) @ M[:3, :3]
                    blocked |= occ.is_filled(ql)
                wrong_side[idx[blocked]] += 1
                touch[idx[blocked]] = False
        if use_2d_veto:
            hm = None
            for pat in (D / "masks/hands/frames" / f"frame_{t:06d}_masks" / f"{side}_hand_0.png",):
                if pat.is_file():
                    hm = cv2.imread(str(pat), 0)
            if hm is not None:
                vis, _ = _visible_mask(Vw, hand[t], c2w[t], K, hm.shape[:2], hm)
                veto = touch & vis                      # 又"接触"又"清晰可见" -> 否证
                vetoed += veto.astype(np.int32)
                touch = touch & (~vis)
        hits += touch.astype(np.int32)
        seen += 1

    weight = hits / max(seen, 1)
    # ★ 找到了窗口但一个点都没碰到 -> 判失败, **不能**落一个空产物。
    #   实测 clip0 右手×瓶身: 窗口 f65~f69 稳且面内贴合, 但接触点全被 2D 否证毙掉
    #   (右手当时在拧**盖**, 跨在盖与瓶口交界处, 对瓶身不是抓握) —— 空 npz 会让
    #   下游以为"这只手抓过瓶身, 只是没有热点"。
    # ★ 对生度(opposition): 抓握必须有**朝向相反**的接触, 否则物理上夹不住。
    #   算法与形状无关: 取热点外法向的平均模长。全朝同一边 -> |mean|≈1 -> 对生度≈0;
    #   真环握 -> 法向互相抵消 -> |mean|≈0 -> 对生度≈1。
    #   为什么必须查: clip0 左手实测五根指头全挤在 66° 一个扇区(拇指与中指差 46°,
    #   环握该是 180°), 拇指离轴 89.5mm 而瓶半径 32.7mm —— 重建的手**根本没环抱瓶子**。
    #   本提取器是**测量**不是拟合, 不会像旧的 align.py 那样把手搬到合理位置(它靠
    #   w_touch 这条"假设一定有抓握"的能量项把手拉上表面, 结果必然好看但那是优化器
    #   摆出来的)。所以这里如实报告"这不是抓握", 而不是悄悄输出一张假的接触图。
    hot_m = weight >= 0.5
    if hot_m.any():
        nh = Nl[hot_m] / np.maximum(np.linalg.norm(Nl[hot_m], axis=1, keepdims=True), 1e-9)
        opposition = float(1.0 - np.linalg.norm(nh.mean(0)))
    else:
        opposition = 0.0

    if int((weight > 0).sum()) == 0:
        return {"status": "no_contact_verts", "object_id": object_id, "side": side,
                "stable_window": stable, "vetoed_total": int(vetoed.sum()),
                "wrong_side_total": int(wrong_side.sum()),
                "why": "稳定窗内没有任何物体点被判接触"
                       + ("(全部被 2D 否证毙掉)" if vetoed.sum() else "")}
    return {"status": "ok", "object_id": object_id, "side": side,
            "frames_used": seen, "frame_span": [int(min(frames)), int(max(frames))],
            "stable_window": stable,
            "tau_mm": tau_mm, "depth_blind": depth_blind,
            "ray_cap_mm": ray_cap_m * 1000,
            # 两个对生度是**不同的量**, 别混:
            #  - opposition_tips(选窗用): 五指尖各自最近表面法向 -> 手是不是环抱着
            #  - opposition_area(诊断用): 热点法向, 受接触**面积**加权 -> 手指侧面积大时
            #    会被压低(真抓握也常是 0.2~0.3), 所以**不拿它做判定**, 否则选对了还报警
            "hand_mask_agreement": agree,
            "is_prior": bool(object_follows_hand),
            "prior_shift_mm": (None if prior_shift is None
                               else (np.asarray(prior_shift) * 1000).round(1).tolist()),
            # ★ 下游(GraspPose 侧)要的三件, 别让它自己从点云反推:
            #   ⚠ 语义准确表述是"**只报能看见的一面**", 不是"删掉了假接触" ——
            #     握瓶时手指确实绕到背面, 那些接触是真的、只是视频看不见。
            #     occluded_frac 高 + 物体旋转对称 -> 下游应做高度带展开。
            "occluded_points": int((wrong_side > 0).sum()),
            "occluded_frac": round(float((wrong_side > 0).sum()
                                         / max(int(((wrong_side > 0) | (hits > 0)).sum()), 1)), 3),
            "visibility_note": "本区域只覆盖相机可见的一面; 背面接触真实存在但不可观测",
            "azimuth_span_deg": _azimuth_span(Vl, weight),
            "azimuth_is_lower_bound": True,
            "opposition_tips": float(stable["opposition_in_window"]),
            "opposition_area": opposition,
            "is_graspable": bool(stable["opposition_in_window"] >= min_opposition),
            "weight": weight, "hits": hits, "vetoed": vetoed, "wrong_side": wrong_side,
            "n_contact_verts": int((weight > 0).sum()),
            "contact_frac": float((weight > 0).mean()),
            "hot_verts": int((weight >= 0.5).sum()),
            "min_dist_mm_median": float(np.median(dmins) * 1000),
            "min_dist_mm_p90": float(np.percentile(dmins, 90) * 1000),
            "probe_local": Vl}


def _write_ply(path: Path, V: np.ndarray, w: np.ndarray) -> None:
    """接触热度着色的点云(MeshLab / CloudCompare 直接打开)。turbo 配色, 灰=未接触。"""
    import matplotlib.cm as cm
    c = np.tile(np.array([[190, 196, 204]], np.uint8), (len(V), 1))
    hot = w > 0
    if hot.any():
        c[hot] = (np.asarray(cm.get_cmap("turbo")(w[hot]))[:, :3] * 255).astype(np.uint8)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(V))
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(V, c):
            fh.write("%.6f %.6f %.6f %d %d %d\n" % (x, y, z, r, g, b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir", type=Path)
    ap.add_argument("--object", default=None, help="默认全部物体")
    ap.add_argument("--side", default=None, help="默认左右都做")
    ap.add_argument("--tau-mm", type=float, default=8.0,
                    help="接触阈值; 人手贴合实测中位 0.3mm, 8mm 容忍位姿抖动")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--stable-r-mm", type=float, default=15.0,
                    help="稳定窗半径: 手在物体局部系的质心漂移不超过它。"
                         "实测 10mm 得 14~20%%区间, 30mm 得 17~33%% —— 15mm 是折中")
    ap.add_argument("--touch-gap-mm", type=float, default=10.0,
                    help="选窗时的贴合门槛: 该帧手到物体最近距离必须 <= 它。"
                         "只看'稳定'会挑到手已松开的段(实测 clip0 左手 f107~f128, 37mm)")
    ap.add_argument("--n-probe", type=int, default=60000,
                    help="物体表面均匀采样点数(探针)。不用 mesh.vertices —— CAD 顶点分布不均")
    ap.add_argument("--min-opposition", type=float, default=0.40,
                    help="逐帧对生度门槛。抓握必须有朝向相反的接触, 否则夹不住。"
                         "标定(五指尖均匀铺开 θ 角): θ=180°->0.52, 150°->0.45, "
                         "120°->0.25, 90°->0.10。默认 0.40 ≈ 要求至少环抱 140°。"
                         "设 0.30 时实测会放进松手阶段(clip0 左手 f107~f128, 对生 0.34)")
    ap.add_argument("--min-hand-agreement", type=float, default=0.25,
                    help="重建手落在实测手mask里的最低比例。实测正常 0.55~0.63, "
                         "手被重建错时 0.00~0.01 —— 不查的话它会伪装成'手没碰到'")
    ap.add_argument("--min-window", type=int, default=5,
                    help="稳定窗最少帧数; 低于它判'无稳定抓握'。2 帧窗的热点是分母太小的假象")
    ap.add_argument("--full-3d", dest="depth_blind", action="store_false",
                    help="用原始 3D 距离判接触。默认**扔掉沿视线那一维** —— 单目深度近乎"
                         "不可观测, 实测 clip0 拇指在图里贴着瓶身(0.1px)但 3D 差 41.5mm, "
                         "其中 40.6mm 是深度; 用 3D 判就漏掉拇指")
    ap.add_argument("--ray-cap-mm", type=float, default=100.0,
                    help="沿视线方向的宽上限, 只毙掉'隔很远投影恰好重合'的粗错")
    ap.add_argument("--object-follows-hand", action="store_true",
                    help="先验模式: 假设一定有抓握, 把物体整体平移到手上(朝向不动)。"
                         "产物走 contact_prior_*, **不进** GRASP 阶段标签、不给 RL 当监督 —— "
                         "它是假设不是测量。用于位姿太差测不出接触的 take")
    ap.add_argument("--no-2d-veto", dest="use_2d_veto", action="store_false")
    ap.add_argument("--no-occlusion-gate", dest="use_normal_gate", action="store_false",
                    help="关掉自遮挡门(只认朝相机那一面; 诊断用)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    w = np.load(a.recon_dir / "world_fused.npz", allow_pickle=True)
    oids = ([str(x) for x in w["object_ids"]] if "object_ids" in w.files else ["object_0"])
    out = a.out_dir or (a.recon_dir / "contact")
    out.mkdir(parents=True, exist_ok=True)
    import time
    phase_rows = []                       # 收集稳定窗, 最后写回 phase 模块的细粒度标签
    attempts = []                         # 所有尝试过的(物体×手), 含失败 —— 下游要靠它
                                          # 区分"试过但不可用"和"没试过"
    for oid in ([a.object] if a.object else oids):
        for side in ([a.side] if a.side else ("left", "right")):
            t0 = time.time()
            res = extract(a.recon_dir, oid, side, tau_mm=a.tau_mm,
                          max_frames=a.max_frames, use_2d_veto=a.use_2d_veto,
                          use_normal_gate=a.use_normal_gate,
                          stable_r_m=a.stable_r_mm / 1000.0,
                          touch_gap_m=a.touch_gap_mm / 1000.0, min_window=a.min_window,
                          n_probe=a.n_probe, depth_blind=a.depth_blind,
                          ray_cap_m=a.ray_cap_mm / 1000.0,
                          min_opposition=a.min_opposition,
                          object_follows_hand=a.object_follows_hand,
                          min_hand_agreement=a.min_hand_agreement)
            dt = time.time() - t0
            attempts.append({"object_id": oid, "side": side, "status": res["status"],
                             "why": res.get("why"),
                             "opposition": res.get("opposition_tips"),
                             "hand_mask_agreement": res.get("hand_mask_agreement")})
            pre = "contact_prior" if a.object_follows_hand else "contact_v2"
            f_out = out / f"{pre}_{oid}_{side}.npz"
            if res["status"] != "ok":
                # ★ 判定为"无稳定抓握"时必须**删掉旧产物**, 否则上一次跑的结果留在原地,
                #   下游/可视化会把它当成本次结果 —— 今天已经因为"把陈旧文件当新结果"
                #   误判过两次(对齐验证、fuse 修复后的 replay)。
                if f_out.is_file():
                    f_out.unlink()
                    print(f"  {oid} × {side}: 已删除上一次的产物(本次判定无稳定抓握)")
                if res["status"] == "hand_unreliable":
                    print(f"  {oid} × {side}: ⛔ hand_unreliable  {res['why']}")
                    continue
                if res["status"] == "no_contact_verts":
                    sw = res["stable_window"]
                    print(f"  {oid} × {side}: no_contact_verts  (窗 f{sw['span'][0]}~"
                          f"f{sw['span'][1]} 稳且贴, 但 {res['why']}; 否证 {res['vetoed_total']})")
                    continue
                extra = ("" if res["status"] != "no_stable_contact" else
                         f"  (病因: {res.get('blocker')}; 区间 {res['n_interval_frames']} 帧里 "
                         f"贴合 {res.get('n_touch_frames')} 帧 / 对生 {res.get('n_opposed_frames_')} 帧; "
                         f"间距中位 {res['gap_median_mm']:.0f}mm 最小 {res['gap_min_mm']:.1f}mm)")
                print(f"  {oid} × {side}: {res['status']}{extra}")
                continue
            f = f_out
            np.savez_compressed(f, **{k: v for k, v in res.items()
                                      if isinstance(v, np.ndarray)},
                                meta=json.dumps({k: v for k, v in res.items()
                                                 if not isinstance(v, np.ndarray)},
                                                ensure_ascii=False))
            # ★ 同时落一份彩色点云: grasp_prompt 把它当"精确几何"交给下游, 不写就是
            #   给了一条指向不存在文件的路径(实测 take1/4 就漏了 —— take0 那两个 .ply
            #   其实是手工可视化时留下的, 不是本步产出)。
            _write_ply(f_out.with_suffix(".ply"), res["probe_local"], res["weight"])
            sw = res["stable_window"]
            phase_rows.append({"object_id": oid, "side": side, "span": sw["span"],
                               "n": sw["n"], "gap_mm": round(sw["gap_in_window_mm"], 2),
                               "n_candidates": sw["n_candidates"],
                               "candidates": sw["candidates"]})
            print(f"  {oid} × {side}: 稳定窗 f{sw['span'][0]}~f{sw['span'][1]} "
                  f"({sw['n']}帧={sw['frac_of_interval']*100:.0f}%区间, "
                  f"全区间漂移 {sw['drift_over_interval_mm']:.0f}mm)")
            print(f"  {'':>{len(oid)+len(side)+5}}  用 {res['frames_used']} 帧  接触顶点 "
                  f"{res['n_contact_verts']} ({res['contact_frac']*100:.1f}%)  "
                  f"热点(≥半数帧) {res['hot_verts']}  对生度 {res['opposition_tips']:.2f}"
                  f"(面积口径 {res['opposition_area']:.2f})"
                  f"{'' if res['is_graspable'] else ' ⚠非抓握'}  手到面中位 "
                  f"{res['min_dist_mm_median']:.1f}mm  否证 {int(res['vetoed'].sum())}  "
                  f"[{dt:.1f}s] -> {f.name}")

    (out / ("contact_prior_summary.json" if a.object_follows_hand
            else "contact_v2_summary.json")).write_text(
        json.dumps({"schema_version": "contact_v2_summary_v1",
                    "is_prior": bool(a.object_follows_hand),
                    "n_ok": sum(1 for x in attempts if x["status"] == "ok"),
                    "attempts": attempts}, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.object_follows_hand:
        print("  [先验模式] 不写 GRASP 阶段标签 —— 先验不能当监督信号")
    else:
        write_grasp_phase(a.recon_dir, phase_rows, out)
    return 0


def write_grasp_phase(recon_dir: Path, rows: list, out_dir: Path) -> Path | None:
    """把稳定窗写回 phase 模块 —— **不要**把它留在 extract_v2 里自用。

    仓里早就有 `ego_pipeline/phase/`(逐帧语义标签 + provider 工厂), `phase/detect.py`
    出的 `contact_auto*.json` 正是本提取器的输入。但那套只有二分类 FREE/CONTACT;
    `types.py` 里 `2+` 一直空着写"预留给 APPROACH/MANIPULATE/RELEASE"。
    稳定窗恰好就是那个没实现的细粒度标签, 所以填进去(GRASP=2), 而不是另起炉灶 ——
    `phase/README.md` 承诺过: provider 换成细粒度后**下游零改动**(attach 工具 /
    retarget / RL correction 都按 int 处理)。

    产物 `contact_auto_grasp.json` 是 `contact_auto.json` 的超集, `phase/auto.py`
    优先读它。多物体时同一只手可能抓过多个物体, 逐帧标签取"哪个物体"记在 by_object 里。
    """
    if not rows:
        return None
    ca = recon_dir / "contact_auto.json"
    if not ca.is_file():
        cands = sorted(recon_dir.glob("contact_auto_object_*.json"))
        if not cands:
            return None
        ca = cands[0]
    doc = json.loads(ca.read_text())
    n = int(doc.get("num_frames") or 0)
    if n <= 0:
        return None

    from ..phase.types import CONTACT, FREE, GRASP, segments_to_dense
    dense = {}
    for side in ("left", "right"):
        segs = (doc.get("annotations") or {}).get(side) or []
        # 底子是**并集**的接触区间: 逐物体文件只覆盖自己那个物体, 直接用会漏掉别的
        d = segments_to_dense(segs, n, on=CONTACT, off=FREE)
        for f in sorted(recon_dir.glob("contact_auto_object_*.json")):
            sg = (json.loads(f.read_text()).get("annotations") or {}).get(side) or []
            d = np.maximum(d, segments_to_dense(sg, n, on=CONTACT, off=FREE))
        for r in rows:
            if r["side"] == side:
                s0, s1 = int(r["span"][0]), min(int(r["span"][1]), n - 1)
                d[s0:s1 + 1] = GRASP
        dense[side] = d.astype(int).tolist()

    p = out_dir / "contact_auto_grasp.json"
    p.write_text(json.dumps({
        "schema_version": "contact_auto_grasp_v1",
        "num_frames": n, "fps": doc.get("fps"),
        "method": "extract_v2 稳定窗(手物相对位姿稳定 且 贴合)",
        "labels": {"0": "free", "1": "contact", "2": "grasp"},
        "left": dense["left"], "right": dense["right"],
        "by_object": rows,
        "note": "GRASP 是 CONTACT 的子集; 下游用 phase.load_phase() 读, 无需认识本文件",
    }, ensure_ascii=False), encoding="utf-8")
    tot = {s: int(sum(1 for v in dense[s] if v == 2)) for s in dense}
    print(f"  阶段标签 -> {p.name}  (GRASP 帧数 左{tot['left']} 右{tot['right']} / 共{n}帧)")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
