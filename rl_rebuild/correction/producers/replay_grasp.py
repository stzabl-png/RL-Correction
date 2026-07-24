"""replay_grasp — human-path-only grasp producer (无 cuRobo / 无 GraspPose).

用于 BODex 抓不出的小/扁物体 (物体2 起). 设计依据 memory:object2-affordance-plan.
数据可信度: 腕位姿(位置+朝向)✅ / 物体track ❌ / 手指pose ❌.

三步 (在 load_replay.load 之上):
  1. 接触窗: phase_<hand>==1 的首帧 = PreGrasp 帧 (= grasp_annotation 的接触起点).
  2. re-anchor: 用可信腕在 PreGrasp 帧的"掌心点"(腕 + palm_offset·手系+z, 与 env _palm_pos 同约定)
     定物体位置; 物体平放桌面 (最大支撑面朝下); 整条腕 path 平移使掌心与物体重合.
     物体 track 完全弃用 (track_object 覆盖成恒定 rest 位, 避免 freeze 把物体钉到噪声位).
  3. 手指参考: 丢掉不可信的重建手指, 换"傻瓜匀速合拢斜坡" (张开->通用 power grasp),
     PreGrasp 前张开, 窗内线性合拢, 之后保持. 残差 ±5.7° + affordance 奖励修正它.
"""
from __future__ import annotations

import numpy as np

from rl_rebuild.correction import frames as F
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


def _close_ramp(L: int, gs: int, ge: int) -> np.ndarray:
    """(L,22) 傻瓜合拢斜坡: t<gs 张开, gs<=t<=ge 线性合拢, t>ge 保持合拢."""
    u = np.clip((np.arange(L) - gs) / max(ge - gs, 1), 0.0, 1.0)[:, None]
    return (1 - u) * GENERIC_OPEN[None] + u * GENERIC_CLOSED[None]


def _affordance_target(affordance_npz):
    """affordance 加权重心 (物体系, m) — re-anchor 对准这里(=人手接触区/环), 而非质心."""
    a = np.load(affordance_npz, allow_pickle=True)
    pts = a["points_raw"].astype(np.float64)          # (P,3) 物体系
    hm = np.clip(a["heatmap"].astype(np.float64), 0, None)
    w = hm ** 2                                        # 强调高分区
    return (pts * w[:, None]).sum(0) / (w.sum() + 1e-9)


def load_replay_grasp(npz_path, mesh_path, usd_path="", clip_id="", hand="right",
                      target_hz=20.0, palm_offset=0.09, close_steps=30, hover_gap=0.02,
                      table_height=0.85, obj_gap=0.002, affordance_npz=None,
                      semantics: ObjectSemantics | None = None, verbose=False):
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

    # ---- 2. re-anchor: 掌心(可信腕) 定物体位置, 物体平放, 整条腕 path 平移 ----
    wq = r.track_wrist[gs, 3:7].astype(np.float64)
    wp = r.track_wrist[gs, :3].astype(np.float64)
    palm = wp + F.rot_apply(wq[None], np.array([[0.0, 0, palm_offset]]))[0]   # 手系+z=指向
    verts = F.load_obj_verts(mesh_path)
    rest_q = _flat_rest_quat(verts)
    v_rot = F.rot_apply(np.broadcast_to(rest_q, (len(verts), 4)), verts)
    obj_z = table_height + obj_gap - v_rot[:, 2].min()                        # 物体底贴桌
    # 抓取目标点 = affordance 加权重心(=环), 无 affordance 退化到质心. 变换到 rest 朝向:
    a_obj = _affordance_target(affordance_npz) if affordance_npz else verts.mean(0)
    a_off = F.rot_apply(rest_q[None], a_obj[None])[0]                         # 相对物体原点的世界偏移
    # 物体原点 xy 使 affordance 区 xy == 掌心 xy; 掌心对准 affordance 区(而非洞)
    obj_pos = np.array([palm[0] - a_off[0], palm[1] - a_off[1], obj_z])
    afford_world = obj_pos + a_off                                            # affordance 区世界位
    # 掌心对准 affordance 区, 再整手上抬 hover_gap: 复位时张开的手不戳进物体(否则 PhysX
    # 退穿透把物体弹飞 50m/s -> NaN). 合拢时手指仍能下探到物体.
    shift = afford_world - palm + np.array([0.0, 0.0, hover_gap])
    r.track_wrist[:, :3] += shift.astype(np.float32)
    r.mano_joints += shift.astype(np.float32)
    # 抓取窗内冻结腕在 PreGrasp 位 (只合手指). 人手参考在抓取后会"拿起物体移走",
    # 但 grasp_only 把物体钉桌上 -> 腕跟着飘走会把物体留在原地(抓空气). pp0 的 cuRobo
    # close 段腕本就不动, 这里显式复刻: gs 之后腕保持 gs 帧位姿, 抬升由 grasp_only 硬编码接管.
    r.track_wrist[gs:] = r.track_wrist[gs]

    # 物体: 弃用噪声 track, 用恒定 re-anchored rest 位 (freeze 会钉在这, 不能是噪声位)
    init_pose = np.concatenate([obj_pos, rest_q]).astype(np.float32)
    r.track_object = np.broadcast_to(init_pose, (L, 7)).copy()

    # ---- 3. 傻瓜合拢斜坡 (丢弃不可信重建手指) ----
    r.human_finger = _close_ramp(L, gs, ge).astype(np.float32)
    r.finger_names = list(GENERIC_JOINT_ORDER)
    r.anchor_wrist = r.anchor_finger = r.curobo_pregrasp = None      # 无 cuRobo
    r.interaction_seg = (gs, min(ge, L - 1))
    r.valid[:] = True
    r.valid_seg = (0, L - 1)
    # grasp_phase_frame = 合拢完成帧 (grasp_only 用它当 grasp_end); finger_q 仅占位,
    # use_grasp_prior=False 时 env 不会拿它做模仿.
    r.grasp = GraspTarget(
        finger_q=GENERIC_CLOSED.copy(),
        wrist_pose=r.track_wrist[ge].copy(),
        contact_fingers=np.ones(5, dtype=bool),
        finger_names=list(GENERIC_JOINT_ORDER),
        grasp_phase_frame=ge)

    du.object_init_pose = init_pose
    du.goal_object_pose = np.concatenate(
        [[obj_pos[0], obj_pos[1], obj_z + 0.10], rest_q]).astype(np.float32)   # 抬 10cm
    if verbose:
        tgt = "affordance区" if affordance_npz else "质心"
        print(f"[replay_grasp] L={L} PreGrasp帧 gs={gs}(src {gs_src}) 合拢至 ge={ge} "
              f"| 掌心对准{tgt} 物体原点 xy=({obj_pos[0]:.3f},{obj_pos[1]:.3f}) z={obj_z:.3f} "
              f"| 腕 path 平移 {np.linalg.norm(shift)*100:.1f}cm")
    return du
