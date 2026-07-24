"""
HaWoR MANO 手部逐帧运动可视化（世界坐标系）
=============================================
用法:
    python vis_hand_motion.py [--seq_dir <path>] [--fps 10] [--trail 5]

操作:
    空格键  暂停 / 继续
    Q       退出
    左箭头  上一帧（暂停时）
    右箭头  下一帧（暂停时）
"""

import argparse
import inspect
import time

import numpy as np
import torch

# ── 兼容性 patch ──────────────────────────────────────────────────────────────
inspect.getargspec = inspect.getfullargspec
np.bool    = np.bool_
np.int     = np.int_
np.float   = np.float64
np.complex = np.complex128
np.object  = object
np.str     = np.str_
np.unicode = np.str_
# ─────────────────────────────────────────────────────────────────────────────

import smplx
from smplx.lbs import batch_rodrigues
import joblib
import open3d as o3d

HAWOR_DATA = "/home/lyh/Project/Affordance2Grasp/third_party/hawor/_DATA"
MANO_RIGHT = f"{HAWOR_DATA}/data/mano/MANO_RIGHT.pkl"
MANO_LEFT  = f"{HAWOR_DATA}/data_left/mano_left/MANO_LEFT.pkl"
DEFAULT_SEQ = ("/home/lyh/Project/Affordance2Grasp/data_hub/RawData/"
               "EgoRawData/egodex/test/add_remove_lid/0")

# ── 坐标系 ────────────────────────────────────────────────────────────────────
# Pipeline outputs Z-up right-hand: +X=前, +Y=左, +Z=上
# 视觉约定: 红=前(+X), 绿=右(-Y), 蓝=上(+Z)
# Pipeline already outputs Z-up right-hand coordinates (X=fwd, Y=left, Z=up).
# No coordinate transform needed here.
WORLD_TO_VIS = np.eye(3, dtype=np.float64)


# MANO 16关节骨架连线（MANOLayer base 只输出16关节，索引0~15）
# 0=wrist, 1-4=食指, 5-8=中指, 9-12=无名指, 13-15=小指
JOINT_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # 食指
    (0, 5), (5, 6), (6, 7), (7, 8),      # 中指
    (0, 9), (9, 10), (10, 11), (11, 12), # 无名指
    (0, 13), (13, 14), (14, 15),         # 小指（MANO16只有3节）
]

HAND_COLORS = [
    [0.55, 0.75, 1.0],   # 左手：淡蓝
    [1.0, 0.75, 0.65],   # 右手：肤色
]

JOINT_COLORS = [
    [0.2, 0.4, 0.9],     # 左手关节：蓝
    [0.9, 0.3, 0.2],     # 右手关节：红
]


def load_mano():
    """
    加载左右手 MANO 模型。
    ws[0]=LEFT hand, ws[1]=RIGHT hand（按 HaWoR run_mano_twohands 定义）
    """
    mr = smplx.MANOLayer(model_path=MANO_RIGHT, is_rhand=True,  use_pca=False)
    ml = smplx.MANOLayer(model_path=MANO_LEFT,  is_rhand=False, use_pca=False)
    # 修正左手 shapedirs 符号（smplx 已知 bug，与 HaWoR process.py 一致）
    # 参考: mano.shapedirs[:, 0, :] *= -1  (N_verts, 3, N_betas) → 第0维=X轴
    ml.shapedirs[:, 0, :] *= -1
    return mr, ml


def mano_forward(model, trans, root_aa, pose_aa, betas):
    """
    axis-angle → rotation matrix → MANO LBS forward
    返回 verts (778,3), joints (16,3)，world frame，单位 meters
    """
    go = batch_rodrigues(root_aa).view(1, 1, 3, 3)              # (1,1,3,3)
    hp = batch_rodrigues(pose_aa.view(-1, 3)).view(1, 15, 3, 3) # (1,15,3,3)
    # MANOLayer.forward 内部 lbs 固定以 pose2rot=False 调用
    # 因此必须传入旋转矩阵，不能传 axis-angle
    out = model(betas=betas, global_orient=go, hand_pose=hp, transl=trans)
    verts  = out.vertices[0].detach().numpy()  # (778, 3)
    joints = out.joints[0].detach().numpy()    # (16,  3)
    # 重映射: HaWoR OpenCV → X=前, Y=右, Z=上
    verts  = (WORLD_TO_VIS @ verts.T).T
    joints = (WORLD_TO_VIS @ joints.T).T
    return verts, joints


def compute_all_frames(seq_dir, mano_r, mano_l):
    """
    预计算所有帧的 verts + joints。
    返回 all_data: list of frames，每帧是 list of (verts, joints, faces, color, jcolor, valid)
    """
    ws = joblib.load(f"{seq_dir}/world_space_res.pth")
    trans_all  = ws[0]   # (2, T, 3)
    root_all   = ws[1]   # (2, T, 3)
    pose_all   = ws[2]   # (2, T, 45)
    betas_all  = ws[3]   # (2, T, 10)
    valid_all  = ws[4]   # (2, T) bool

    T = trans_all.shape[1]
    # index 0: left, index 1: right
    models = [mano_l, mano_r]
    # Watertight palm faces
    faces  = [np.vstack([mano_l.faces, [[744, 745, 746]]]), 
              np.vstack([mano_r.faces, [[744, 745, 746]]])]

    all_data = []
    for t in range(T):
        frame_hands = []
        for h in range(2):
            if not valid_all[h, t]:
                frame_hands.append(None)
                continue
            verts, joints = mano_forward(
                models[h],
                trans_all[h, t:t+1].float(),
                root_all[h,  t:t+1].float(),
                pose_all[h,  t:t+1].float(),
                betas_all[h, t:t+1].float(),
            )
            frame_hands.append({
                "verts":  verts,
                "joints": joints,
                "faces":  faces[h],
                "color":  HAND_COLORS[h],
                "jcolor": JOINT_COLORS[h],
            })
        all_data.append(frame_hands)
        print(f"  预计算 frame {t:3d}/{T-1}  ", end="\r")

    print(f"  预计算完成，共 {T} 帧               ")
    return all_data, T


def make_hand_mesh(verts, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def make_skeleton_lineset(joints, color):
    """关节骨架 LineSet（只连接合法索引范围内的关节）"""
    n = len(joints)
    valid_conns = [(a, b) for a, b in JOINT_CONNECTIONS if a < n and b < n]
    if not valid_conns:
        return None
    ls = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(joints),
        o3d.utility.Vector2iVector(valid_conns)
    )
    ls.colors = o3d.utility.Vector3dVector([color] * len(valid_conns))
    return ls


def make_joint_spheres(joints, color, radius=0.006):
    """每个关节画一个小球（合并成一个 mesh 以节省 geometry 数量）"""
    combined = o3d.geometry.TriangleMesh()
    for j in joints:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.translate(j)
        combined += s
    combined.paint_uniform_color(color)
    combined.compute_vertex_normals()
    return combined


def make_wrist_trail(all_data, current_frame, trail_len, hand_idx):
    """历史轨迹线：过去 trail_len 帧的手腕位置连线"""
    pts = []
    for t in range(max(0, current_frame - trail_len), current_frame + 1):
        h = all_data[t][hand_idx]
        if h is not None:
            pts.append(h["joints"][0])  # joint 0 = wrist

    if len(pts) < 2:
        return None

    pts_v = o3d.utility.Vector3dVector(pts)
    lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(pts)-1)])
    ls = o3d.geometry.LineSet(pts_v, lines)
    trail_color = [c * 0.6 for c in HAND_COLORS[hand_idx]]
    ls.colors = o3d.utility.Vector3dVector(
        [trail_color for _ in range(len(pts)-1)]
    )
    return ls


def build_frame_geometries(all_data, frame_idx, trail_len=8):
    """构建该帧的所有 geometry 对象。"""
    geoms = []
    for h in range(2):
        hand = all_data[frame_idx][h]
        if hand is None:
            continue
        geoms.append(make_hand_mesh(hand["verts"], hand["faces"], hand["color"]))
        ls = make_skeleton_lineset(hand["joints"], hand["jcolor"])
        if ls is not None:
            geoms.append(ls)
        geoms.append(make_joint_spheres(hand["joints"], hand["jcolor"]))
        trail = make_wrist_trail(all_data, frame_idx, trail_len, h)
        if trail:
            geoms.append(trail)
    return geoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_dir", default=DEFAULT_SEQ)
    parser.add_argument("--fps",   type=float, default=10.0, help="播放帧率")
    parser.add_argument("--trail", type=int,   default=8,    help="轨迹拖尾帧数")
    args = parser.parse_args()

    print(f"序列: {args.seq_dir}")
    print("加载 MANO 模型...")
    mano_r, mano_l = load_mano()

    print("预计算所有帧的手部 mesh...")
    all_data, T = compute_all_frames(args.seq_dir, mano_r, mano_l)

    # ── Open3D Visualizer ────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("HaWoR MANO — World Frame Animation", width=1280, height=720)

    # 渲染选项：背景黑色
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.light_on = True

    # 坐标轴（自定义：红=+X(前), 绿=+Y(左), 蓝=+Z(上)）右手系
    ax_size = 0.05
    ax_pts = [[0,0,0], [ax_size,0,0], [0,ax_size,0], [0,0,ax_size]]
    ax_lines = [[0,1], [0,2], [0,3]]
    ax_colors = [[1,0.2,0.2], [0.2,1,0.2], [0.3,0.3,1]]  # R, G, B
    axes = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(ax_pts),
        o3d.utility.Vector2iVector(ax_lines)
    )
    axes.colors = o3d.utility.Vector3dVector(ax_colors)
    vis.add_geometry(axes)

    # 状态变量
    state = {"frame": 0, "paused": False, "last_t": time.time(), "quit": False}

    # 当前帧 geometries（用列表管理以便每帧替换）
    current_geoms = []

    def update_frame(new_frame):
        for g in current_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        current_geoms.clear()

        geoms = build_frame_geometries(all_data, new_frame, args.trail)
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=(new_frame == 0))
            current_geoms.append(g)

        vis.get_view_control()  # 刷新
        state["frame"] = new_frame

    # 初始化第 0 帧
    update_frame(0)

    # ── 初始视角：XY 水平面, Z 朝上 ──────────────────────────────────────────
    ctr = vis.get_view_control()
    ctr.set_up([0, 0, 1])           # +Z = 上
    ctr.set_front([-0.5, -0.5, 0.7]) # 从右后上方俯瞰（XY平面水平展开）
    ctr.set_lookat([0.35, 0.07, 0.27])  # 手部中心
    ctr.set_zoom(0.45)
    # ─────────────────────────────────────────────────────────────────────────

    # ── 键盘回调 ─────────────────────────────────────────────────────────────
    def on_space(vis_):
        state["paused"] = not state["paused"]
        print(f"{'暂停' if state['paused'] else '播放'}  frame={state['frame']}")
        return False

    def on_right(vis_):
        if state["paused"]:
            update_frame((state["frame"] + 1) % T)
            print(f"frame {state['frame']:3d}/{T-1}")
        return False

    def on_left(vis_):
        if state["paused"]:
            update_frame((state["frame"] - 1) % T)
            print(f"frame {state['frame']:3d}/{T-1}")
        return False

    def on_q(vis_):
        state["quit"] = True
        return False

    vis.register_key_callback(ord(" "), on_space)
    vis.register_key_callback(262, on_right)   # →
    vis.register_key_callback(263, on_left)    # ←
    vis.register_key_callback(ord("Q"),  on_q)
    vis.register_key_callback(ord("q"),  on_q)

    print(f"\n{'='*50}")
    print(f"  总帧数: {T}")
    print(f"  操作: 空格=暂停/播放  ←→=逐帧  Q=退出")
    print(f"{'='*50}\n")

    dt = 1.0 / args.fps

    # ── 主循环 ───────────────────────────────────────────────────────────────
    while vis.poll_events():
        if state["quit"]:
            break

        if not state["paused"]:
            now = time.time()
            if now - state["last_t"] >= dt:
                next_frame = (state["frame"] + 1) % T
                update_frame(next_frame)
                print(f"  frame {state['frame']:3d}/{T-1}  "
                      f"{'右手' if all_data[state['frame']][0] else '  '}"
                      f"{'左手' if all_data[state['frame']][1] else '  '}",
                      end="\r")
                state["last_t"] = now

        vis.update_renderer()

    vis.destroy_window()
    print("\n完成。")


if __name__ == "__main__":
    main()
