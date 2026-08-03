"""replay_grasp — human-path-only grasp ref builder (无 cuRobo / 无 GraspPose).

用于 BODex 抓不出的小/扁物体 (物体2 起). 设计依据 memory:object2-affordance-plan.
数据可信度: 腕位姿(位置+朝向)✅ / 物体track ❌ / 手指pose ❌.

三步 (在 load_replay.load 之上):
  1. 接触窗: phase_<hand>==1 的首帧 = PreGrasp 帧 (= grasp_annotation 的接触起点).
  2. 摆放 (anchor_mode): 物体平放桌面 (最大支撑面朝下); 物体 track 弃用 (噪声太大).
     - "camera" (默认): 相机/手/物体 xy 刚体同步平移, 由"重建相机 -> 机器人 ZED"定死,
       三者相对关系保留重建原样; z 是两条独立约束 (物体贴桌 / 手全程最小抬升).
     - "palm" (旧): 把物体搬到 PreGrasp 合拢中心正下方 —— 会破坏上面那条相机对齐,
       实测让 Grasp0~9 全部变成跨中线的跨身抓取. 只为复现历史 run 保留.
  3. 手指参考: 丢掉不可信的重建手指, 换"傻瓜匀速合拢斜坡" (张开->通用 power grasp),
     PreGrasp 前张开, 窗内线性合拢, 之后保持. 残差 ±5.7° + affordance 奖励修正它.
"""
from __future__ import annotations

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction import place_camera as PC
from rl_rebuild.correction.load_replay import load
from rl_rebuild.correction.schema import GraspTarget, ObjectSemantics

# 通用 SharpaWave 抓握构型 (借自 pp0 cuRobo: 张开=帧0, 合拢=squeeze 末; 真实可行手型).
GENERIC_JOINT_ORDER = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA",
    "right_thumb_IP", "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP",
    "right_index_DIP", "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP",
    "right_middle_DIP", "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP",
    "right_ring_DIP", "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA",
    "right_pinky_PIP", "right_pinky_DIP"]
GENERIC_OPEN = np.array(
    [0., 0.08, 0., 0.08, 0., 0., 0.19, 0., 0., 0., 0.17, 0., 0., 0., 0.13, 0., 0., 0.1,
     0., 0.08, 0., 0.], dtype=np.float32)
GENERIC_CLOSED = np.array(
    [1.92, 0.08, 0.67, 0.08, 0.74, 0.38, 0.19, 0.71, 0.66, 0.35, 0.17, 0.77, 0.57, 0.18,
     0.13, 0.87, 0.55, 0.1, 0.03, 0.08, 0.92, 0.66], dtype=np.float32)


def _flat_rest_quat(verts: np.ndarray) -> np.ndarray:
    """把物体最大支撑面压到桌面 -> rest 四元数 (wxyz). 对薄/扁物体 = 平放."""
    faces = F.support_faces(verts, min_area=0.0005)
    if not faces:
        return np.array([1.0, 0, 0, 0])
    n_top = faces[0][1]                      # 最大支撑面的外法向
    down = np.array([0.0, 0, -1])            # 该面朝下 -> 外法向指向 -z
    axis = np.cross(n_top, down)
    s, c = np.linalg.norm(axis), float(n_top @ down)
    if s < 1e-8:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    ang = np.arctan2(s, c)
    axis = axis / s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis]).astype(np.float64)


def _close_ramp(L: int, gs: int, ge: int, close_scale: float = 1.0) -> np.ndarray:
    """(L,22) 傻瓜合拢斜坡: t<gs 张开, gs<=t<=ge 线性合拢, t>ge 保持合拢.
    close_scale>1 = 目标合拢构型更深 (薄物体需要更紧的握持)."""
    u = np.clip((np.arange(L) - gs) / max(ge - gs, 1), 0.0, 1.0)
    u = (u * u * (3.0 - 2.0 * u))[:, None]        # smoothstep: 合拢起停平滑, 不是匀速硬合
    closed = GENERIC_OPEN + (GENERIC_CLOSED - GENERIC_OPEN) * close_scale
    return (1 - u) * GENERIC_OPEN[None] + u * closed[None]


_URDF_CACHE = {}


def _urdf():
    """共享的只读 Urdf 实例.

    `Urdf()` 每次都重新 parse 一遍 XML, 而 `min_lift_for_clearance` 要对**每一帧**
    调 `hand_lowest_world` (Grasp2 = 166 次). 这两个函数只读不写, 可以共用一份.
    """
    from rl_rebuild.correction.kinematics import Urdf
    if "u" not in _URDF_CACHE:
        _URDF_CACHE["u"] = Urdf()
    return _URDF_CACHE["u"]


def grasp_center_local(hand: str = "right", closed=None) -> np.ndarray:
    """Sharpa 手**合拢终态**下五个胶垫的质心, 在手基座(hand_C_MC)坐标系.

    这是手指真正夹住东西的地方. 之前用的 `palm_offset=0.09`(腕+掌法向9cm)是给**飞手**
    调的几何代理, 与机器人手的实际合拢位置差 **6.88cm** —— 物体被系统性地摆在手指
    合拢位置的后方, 手一闭合就从物体上方掠过去抓空.
    实测后果: 19 条 clip 里只有锚点误差恰好小于容差的 3 条能抓起来;
    Grasp12 五个指尖离物体 8~10cm, 接触率 0.00/5.

    改成从 URDF 正解算出来, 不再有手调常数.
    """
    u = _urdf()
    q = {n.replace("right_", f"{hand}_"): float(v)
         for n, v in zip(GENERIC_JOINT_ORDER,
                         GENERIC_CLOSED if closed is None else closed)}
    tips = np.stack([u.link_pose(f"{hand}_{f}_elastomer", q, np.eye(4),
                                 f"{hand}_hand_C_MC")[:3, 3]
                     for f in ("thumb", "index", "middle", "ring", "pinky")])
    return tips.mean(0)


def finger_base_local(hand: str = "right", finger: str = "middle") -> np.ndarray:
    """指根 (MCP 关节原点) 在手基座系的位置.

    与 `grasp_center_local` 的区别 —— 它们代表两种不同的"把手对准物体"的含义:
      合拢中心: CLOSED 构型五胶垫的质心 [4.19,-0.86,14.40]cm, 离腕 15.02cm.
                = "让**夹持点**落在物体上". 但它是**开合相关**的 —— 手一张开,
                五指质心跑到 [0.39,1.01,15.65], 而且合拢时手在掌法向 x 上鼓到 7.14cm,
                合拢中心只在 x=4.19, 所以它下方还有 2.95cm 的手会戳桌.
      指根:     掌上的**固定点**, 不随开合变化 (MCP 关节的父连杆就是手基座,
                关节转动不移动它的原点). = "让**掌根**落在物体上方".
    """
    u = _urdf()
    link = u.joints[f"{hand}_{finger}_MCP_FE"]["child"]
    return u.link_pose(link, {}, np.eye(4), f"{hand}_hand_C_MC")[:3, 3]


def hand_lowest_world(wp, wq, hand: str = "right", qvec=None) -> float:
    """给定手基座的**世界位姿**, 返回手上所有 link 的最低世界 z.

    ⚠ 不能在手系里取 z 最小 —— 手系的 +z 是"手指指向", 不是世界的"下".
    手最低到哪, 取决于手当时的朝向, 必须转到世界系再取.
    用途: 算"张开的手不戳进桌面"需要多少余量, 替掉手调的 hover_gap.
    """
    from rl_rebuild.correction.kinematics import quat_to_R, _T
    u = _urdf()
    q = {n.replace("right_", f"{hand}_"): float(v)
         for n, v in zip(GENERIC_JOINT_ORDER,
                         GENERIC_OPEN if qvec is None else qvec)}
    T = _T(quat_to_R(np.asarray(wq, float)), np.asarray(wp, float))
    zs = [u.link_pose(l, q, T, f"{hand}_hand_C_MC")[2, 3]
          for l in u.parent_joint if l.startswith(f"{hand}_")]
    return float(min(zs)) if zs else float(wp[2])


def _affordance_target(affordance_npz):
    """affordance 加权重心 (物体系, m) — re-anchor 对准这里(=人手接触区/环), 而非质心."""
    a = np.load(affordance_npz, allow_pickle=True)
    pts = a["points_raw"].astype(np.float64)          # (P,3) 物体系
    hm = np.clip(a["heatmap"].astype(np.float64), 0, None)
    w = hm ** 2                                        # 强调高分区
    return (pts * w[:, None]).sum(0) / (w.sum() + 1e-9)


def load_replay_grasp(npz_path, mesh_path, usd_path="", clip_id="", hand="right",
                      target_hz=20.0, palm_offset=None, close_steps=None, hover_gap=None,
                      close_scale=None, table_height=0.85, obj_gap=0.002, affordance_npz=None,
                      clearance=None, freeze_wrist=True, anchor_mode=None, pregrasp_align=None,
                      semantics: ObjectSemantics | None = None, verbose=False):
    # 抓取几何旋钮 — 支持环境变量覆盖, 便于不改代码扫参 (GRASP_HOVER / GRASP_PALM_OFF / ...)
    import os as _o
    palm_offset = float(_o.environ.get("GRASP_PALM_OFF", 0.09)) if palm_offset is None else palm_offset
    # hover_gap=None -> 由几何推出 (推荐); 给了数就用给的 (兼容旧行为/扫参)
    _hg = _o.environ.get("GRASP_HOVER")
    if hover_gap is None and _hg is not None:
        hover_gap = float(_hg)
    clearance = (float(_o.environ.get("GRASP_CLEARANCE", 0.015))
                 if clearance is None else float(clearance))
    # 摆放锚: "camera"=相机锚定(默认) / "palm"=旧的掌心re-anchor(复现历史 run 用)
    anchor_mode = _o.environ.get("GRASP_ANCHOR", anchor_mode or "camera")
    assert anchor_mode in ("camera", "palm"), f"未知 anchor_mode={anchor_mode}"
    # PreGrasp 对齐 (只在 camera 锚下有意义; palm 锚按构造已经把物体摆到手下面了).
    # 关掉它 = 用重建原样的手物相对关系 —— 冒烟对照组用.
    if pregrasp_align is None:
        pregrasp_align = _o.environ.get("GRASP_PREGRASP_ALIGN", "1") == "1"
    close_steps = int(_o.environ.get("GRASP_CLOSE_STEPS", 30)) if close_steps is None else close_steps
    # close_scale: 合拢深度倍率 (>1 = 握更紧, 用于薄物体加大握持力)
    close_scale = float(_o.environ.get("GRASP_CLOSE_SCALE", 1.0)) if close_scale is None else close_scale
    # ---- 1. 接触窗 (源帧率) -> PreGrasp 起点 ----
    raw = np.load(npz_path, allow_pickle=True)
    src_fps = float(raw["fps"])
    ph = raw[f"phase_{hand}"].astype(int)
    contact = np.flatnonzero(ph == 1)
    gs_src = int(contact[0]) if len(contact) else int(0.3 * len(ph))

    # ---- 复用 load_replay: 得到重采样后的可信腕 path / mano / mesh 落桌 ----
    du = load(npz_path, mesh_path, usd_path=usd_path, clip_id=clip_id, hand=hand,
              table_height=table_height, obj_gap=obj_gap, target_hz=target_hz,
              semantics=semantics, verbose=verbose)
    r = du.ref
    L = r.L
    gs = int(round(gs_src * (target_hz / src_fps)))
    gs = int(np.clip(gs, 1, L - 2))
    ge = int(min(gs + close_steps, L - 1))

    # ---- 2. 摆放 ----
    wq = r.track_wrist[gs, 3:7].astype(np.float64)
    wp = r.track_wrist[gs, :3].astype(np.float64)
    # 抓取锚点 = Sharpa 合拢终态的五胶垫质心 (手指真正夹住东西的地方), 由 URDF 正解算出.
    # 旧做法是 wp + 掌法向 palm_offset(9cm), 与实际合拢位置差 6.88cm, 见 grasp_center_local.
    palm = wp + F.rot_apply(wq[None], grasp_center_local(hand)[None])[0]
    verts = F.load_obj_verts(mesh_path)
    rest_q = _flat_rest_quat(verts)
    v_rot = F.rot_apply(np.broadcast_to(rest_q, (len(verts), 4)), verts)
    obj_z = table_height + obj_gap - v_rot[:, 2].min()                        # 物体底贴桌
    # 抓取目标点 = affordance 加权重心(=环), 无 affordance 退化到质心. 变换到 rest 朝向:
    a_obj = _affordance_target(affordance_npz) if affordance_npz else verts.mean(0)
    a_off = F.rot_apply(rest_q[None], a_obj[None])[0]                         # 相对物体原点的世界偏移

    if anchor_mode == "camera":
        # 相机锚定: 相机/手/物体 **xy 刚体同步平移**, 三者相对关系保留重建原样.
        # 平移量由"重建相机(人头) xy -> 机器人 ZED 光心 xy"定死.
        # 物体留在 align_replay 给的位置 (首帧 xy 在原点), 只把姿态换成平放 rest.
        sxy = PC.camera_anchor_shift(mesh_path, PC.ZED_NOMINAL[:2], raw["obj_pose"][0, :2])
        if sxy is None:
            print("[replay_grasp][WARN] world_fused.npz 缺 c2w, 无法相机锚定 -> 退回 palm 锚定")
            anchor_mode = "palm"
        else:
            obj_pos = np.array([sxy[0], sxy[1], obj_z])
            r.track_wrist[:, :2] += sxy.astype(np.float32)
            r.mano_joints[..., :2] += sxy.astype(np.float32)
            # z: **全程**最小抬升, 让每一帧张开的手最低点都离桌面 clearance.
            # 只抬手不抬物体 (xy 是刚体, z 不是 —— 见 place_camera 的两条独立约束).
            # 旧实现只保证 PreGrasp 单帧, Grasp2 实测仍有 56/166 帧手低于桌面.
            dz = PC.min_lift_for_clearance(r.track_wrist[:, :3].astype(np.float64),
                                           r.track_wrist[:, 3:7].astype(np.float64),
                                           hand, table_height, clearance, hand_lowest_world)
            r.track_wrist[:, 2] += np.float32(dz)
            r.mano_joints[..., 2] += np.float32(dz)
            shift = np.array([sxy[0], sxy[1], dz])

            # ---- PreGrasp 对齐 ----
            # 相机锚定忠实保留了重建的手物相对关系, 而重建里**手根本没碰到物体**
            # (源数据最近的指尖离物体原点 9.9cm), 所以 gs 帧的手离物体还差十几厘米.
            # 这里把 gs 帧的腕挪到"手上的锚点悬停在 affordance 区正上方 hover"的位置.
            # 只改**位置不改朝向**: 朝向留重建的 —— 它有战绩 (零残差 100%, hold 段
            # 拇指 0.950 / 无名 0.905 / 小指 0.950 的真实对指, 见
            # docs/DEXMATE_TRAINING_FEASIBILITY.md:497). 对握轴朝向反而只有一次实测,
            # 且是失败的 (scripted_demo 合拢把物体撞飞 10.76cm).
            if pregrasp_align:
                afford_world = obj_pos + a_off
                anch = (finger_base_local(hand, "middle")
                        if PC.PREGRASP_ANCHOR == "middle_base" else grasp_center_local(hand))
                _hv = PC.PREGRASP_HOVER_GAP if hover_gap is None else hover_gap
                pg = PC.pregrasp_hover_pose(afford_world, wq, anch, _hv)
                delta = pg - r.track_wrist[gs, :3].astype(np.float64)
                # ⚠ 不能只改 gs 那一帧: 与 gs-1 之间会出现十几厘米的跳变, 位控臂跟不上,
                #   直接触发 term/stuck. 所以在**接触前**平滑地把修正量喂进去 (smoothstep),
                #   gs 之后整段保持同一个常量偏移 —— 于是 gs 之后的**相对**运动
                #   (搬运/放置的形状, 以及由它推出的物体参考) 完全不变.
                u = np.clip(np.arange(L, dtype=np.float64) / max(gs, 1), 0.0, 1.0)
                u = u * u * (3.0 - 2.0 * u)
                r.track_wrist[:, :3] += (u[:, None] * delta).astype(np.float32)
                r.mano_joints += (u[:, None, None] * delta).astype(np.float32)
                shift = shift + delta          # 记进总平移 (gs 之后的实际偏移量)
                # 挪完必须复查穿桌: delta 的 z 分量通常是**往下**的 (把手降到物体上方),
                # 会吃掉上面那次 clearance 抬升. 这里只报数不自动补抬 ——
                # 自动补抬会把 PreGrasp 又顶回够不着的高度 (旧 hover_gap 魔数就是这么来的).
                _nb = sum(hand_lowest_world(r.track_wrist[t, :3], r.track_wrist[t, 3:7],
                                            hand) < table_height for t in range(L))
                _lo_gs = hand_lowest_world(r.track_wrist[gs, :3], r.track_wrist[gs, 3:7], hand)
                print(f"[replay_grasp] PreGrasp 对齐({PC.PREGRASP_ANCHOR}, hover={_hv*100:.0f}cm): "
                      f"腕挪 {np.round(delta*100, 2).tolist()}cm ({np.linalg.norm(delta)*100:.2f}cm) "
                      f"| gs 帧张开手最低离桌 {(_lo_gs-table_height)*100:+.2f}cm "
                      f"| 全程穿桌 {_nb}/{L} 帧")
                if _nb:
                    print(f"[replay_grasp][WARN] 对齐后有 {_nb} 帧手低于桌面 —— "
                          f"接近段会撞桌, 要么减小 hover 前的 clearance 要么改锚点")

    if anchor_mode == "palm":
        # 旧摆放 (保留以复现历史 run): 物体原点 xy 使 affordance 区 xy == 掌心 xy.
        # ⚠ 这一步**破坏相机锚定** —— align_replay 刚把物体首帧放到原点、手物相对关系
        #   保留重建原样, 这里又把物体搬到掌心正下方. 实测后果 (Grasp0~9 全部):
        #   物体落到机器人左侧 y=+0.141 而交互手是右手, 每条 clip 都成了跨中线 25cm 的
        #   跨身抓取; 到交互手 37.0cm (相机锚定后 18.0cm).
        obj_pos = np.array([palm[0] - a_off[0], palm[1] - a_off[1], obj_z])
        afford_world = obj_pos + a_off                                        # affordance 区世界位
        # 掌心对准 affordance 区, 再整手上抬: 复位时张开的手不戳进物体(否则 PhysX
        # 退穿透把物体弹飞 50m/s -> NaN). 合拢时手指仍能下探到物体.
        lo_open = hand_lowest_world(wp, wq, hand)        # 张开的手最低点 (世界 z)
        need = (table_height + clearance) - lo_open      # 还差多少才够高
        lift = max(float(need), 0.0) if hover_gap is None else float(hover_gap)
        shift = afford_world - palm + np.array([0.0, 0.0, lift])
        r.track_wrist[:, :3] += shift.astype(np.float32)
        r.mano_joints += shift.astype(np.float32)
    # ref builder 施加的总平移. camera 模式下 env 还会补一次**活测残差** (见
    # dexmate_env._apply_camera_anchor): ZED_NOMINAL 是常数, 权威是训练 env 的实测值.
    r.builder_wrist_shift = shift.astype(np.float32).copy()
    # 抓取窗内冻结腕在 PreGrasp 位 (只合手指). 人手参考在抓取后会"拿起物体移走",
    # 但 grasp_only 把物体钉桌上 -> 腕跟着飘走会把物体留在原地(抓空气). pp0 的 cuRobo
    # close 段腕本就不动, 这里显式复刻: gs 之后腕保持 gs 帧位姿, 抬升由 grasp_only 硬编码接管.
    # 抓取窗内是否冻结腕:
    #   True  (grasp_only): 物体被钉在桌上, 腕跟着人手飘走就是抓空气, 所以冻结
    #   False (完整轨迹):   腕跟着人手走完 靠近->抓住->搬运->放置->归位
    if freeze_wrist:
        r.track_wrist[gs:] = r.track_wrist[gs]
    # 起点偏移测试旋钮 (仅测容忍度用): 把冻结的腕整体平移, 物体不动 -> 制造"手物错位"起点.
    _off = np.array([float(_o.environ.get(k, 0.0)) for k in
                     ("GRASP_TEST_DX", "GRASP_TEST_DY", "GRASP_TEST_DZ")], dtype=np.float32)
    if np.any(_off):
        r.track_wrist[:, :3] += _off
        r.mano_joints += _off

    # ---- 物体参考轨迹 ----
    # 重建的物体 track 噪声太大, 弃用. 但也不能一直是常量 —— 那样搬运段就没有目标.
    # 搬运时物体被**刚性握住**, 所以"物体去哪"完全由"手去哪"决定; 而腕在接触段有接触
    # 约束、是可信通道 (见 docs/RECON_TO_ENV_ALIGNMENT.md 的可信度原则).
    # 所以: 物体参考 = 抓取帧的物体位置 + 腕相对抓取帧的位移. 无噪声且自洽.
    # 交互段 = 靠近/抓住/搬运/放置 的分界, 由 phase_* 给出
    ph_full = F.resample(ph.astype(np.float64), L, kind="nearest") if L != len(ph) else ph
    _con = np.flatnonzero(np.asarray(ph_full).round().astype(int) == 1)
    re_src = int(_con[-1]) if len(_con) else min(ge, L - 1)
    r.interaction_seg = (gs, max(re_src, gs + 1))
    init_pose = np.concatenate([obj_pos, rest_q]).astype(np.float32)
    r.track_object = np.broadcast_to(init_pose, (L, 7)).copy()
    if not freeze_wrist:
        w = r.track_wrist[:, :3].astype(np.float64)
        rel = w - w[gs]                                  # 腕相对抓取帧的位移
        rel[:gs] = 0.0                                   # 抓住之前物体不动
        re_ = min(int(r.interaction_seg[1]) if r.interaction_seg else L - 1, L - 1)
        rel[re_:] = rel[re_]                             # 放下之后物体留在原地
        r.track_object[:, :3] = (obj_pos[None] + rel).astype(np.float32)

    # ---- 3. 傻瓜合拢斜坡 (丢弃不可信重建手指) ----
    r.human_finger = _close_ramp(L, gs, ge, close_scale).astype(np.float32)
    # 关节名跟着手侧走: GENERIC_JOINT_ORDER 里写的是 right_*, 左手 clip 直接用会让
    # env 的 perm 映射炸掉 ("left_index_MCP_FE is not in list"). 两只手关节名后缀一致.
    r.finger_names = [n.replace("right_", f"{hand}_") for n in GENERIC_JOINT_ORDER]
    r.anchor_wrist = r.anchor_finger = r.curobo_pregrasp = None      # 无 cuRobo
    r.valid[:] = True
    r.valid_seg = (0, L - 1)
    # grasp_phase_frame = 合拢完成帧 (grasp_only 用它当 grasp_end); finger_q 仅占位,
    # use_grasp_prior=False 时 env 不会拿它做模仿.
    r.grasp = GraspTarget(
        finger_q=GENERIC_CLOSED.copy(),
        wrist_pose=r.track_wrist[ge].copy(),
        contact_fingers=np.ones(5, dtype=bool),
        finger_names=[n.replace("right_", f"{hand}_") for n in GENERIC_JOINT_ORDER],
        grasp_phase_frame=ge)

    du.object_init_pose = init_pose
    du.goal_object_pose = np.concatenate(
        [[obj_pos[0], obj_pos[1], obj_z + 0.10], rest_q]).astype(np.float32)   # 抬 10cm
    if verbose:
        tgt = "affordance区" if affordance_npz else "质心"
        how = ("相机锚定 (手物相对关系保留重建原样)" if anchor_mode == "camera"
               else f"掌心对准{tgt} (旧锚, 会破坏相机对齐)")
        print(f"[replay_grasp] L={L} PreGrasp帧 gs={gs}(src {gs_src}) 合拢至 ge={ge} "
              f"| {how} 物体原点 xy=({obj_pos[0]:.3f},{obj_pos[1]:.3f}) z={obj_z:.3f} "
              f"| 腕 path 平移 xy={np.linalg.norm(shift[:2])*100:.1f}cm z={shift[2]*100:+.2f}cm")
    return du
