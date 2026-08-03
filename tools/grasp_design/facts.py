"""四组事实数据: 物体尺寸/位置 · 手指长度与合拢指尖相对腕的位置 · 交互段手物位移."""
import contextlib, io, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction import clips, frames as F, place_camera as PC
from rl_rebuild.correction.ref_builders.replay_grasp import (
    _flat_rest_quat, _affordance_target, grasp_center_local,
    GENERIC_OPEN, GENERIC_CLOSED)
from rl_rebuild.correction.kinematics import Urdf, quat_to_R, _T

CLIP, TZ, HAND = "Grasp2", 0.85, "right"
e = clips.clip_entry(CLIP)
raw = np.load(e["npz"], allow_pickle=True)
cfg = SimpleNamespace(clip_name=CLIP, target_hz=20.0, table_top_z=TZ, clearance=0.015,
                      freeze_wrist=False, hover_gap=None, anchor_mode="camera")
with contextlib.redirect_stdout(io.StringIO()):
    du = clips.load_data_unit(cfg)
r = du.ref

print("=" * 78)
print("1. 物体尺寸 / 是否平躺")
print("=" * 78)
verts = F.load_obj_verts(e["mesh"])
ext_raw = verts.max(0) - verts.min(0)
rest_q = _flat_rest_quat(verts)
vr = F.rot_apply(np.broadcast_to(rest_q, (len(verts), 4)), verts)
ext = vr.max(0) - vr.min(0)
print(f"mesh 顶点 {len(verts)}")
print(f"  mesh 自身坐标系 包围盒 XYZ = {np.round(ext_raw*100,2).tolist()} cm")
print(f"  平放(rest)后   包围盒 XYZ = {np.round(ext*100,2).tolist()} cm")
print(f"  -> 长 {ext[:2].max()*100:.2f} × 宽 {ext[:2].min()*100:.2f} × 高 {ext[2]*100:.2f} cm")
print(f"  水平最大半径 {np.linalg.norm(vr[:,:2],axis=1).max()*100:.2f} cm")
rq = raw["obj_pose"][:, 3:7].astype(np.float64)          # 重建姿态 (wxyz)
def tilt(q):
    R = quat_to_R(q / np.linalg.norm(q))
    return np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1)))
t0 = [tilt(q) for q in rq if np.isfinite(q).all() and np.linalg.norm(q) > 1e-6]
print(f"\n  env 里: **强制平放** (rest_q = 最大支撑面朝下), 底面贴桌 z={TZ}+0.2cm gap")
print(f"  重建里: 物体自身姿态相对'平放'倾斜 中位 {np.median(t0):.1f}° "
      f"(范围 {min(t0):.1f}~{max(t0):.1f}°) -> 重建姿态**不是**平躺, 被判为噪声弃用")
print(f"  平放后物体中心 z 比重建姿态口径低 ~1.36cm (view_placement 已注明, 有意设计)")

print()
print("=" * 78)
print("2. 物体坐标位置")
print("=" * 78)
obj = du.object_init_pose[:3].astype(np.float64)
oq = du.object_init_pose[3:7].astype(np.float64)
aff = PC.affordance_world(obj, oq, _affordance_target(e["affordance"]))
print(f"[env / 桌面局部系] (相机锚定, 训练用的就是这个)")
print(f"  物体原点   {np.round(obj,4).tolist()}   (离桌 {(obj[2]-TZ)*100:+.2f}cm)")
print(f"  姿态 wxyz  {np.round(oq,4).tolist()}")
print(f"  包围盒 z   [{TZ+0.002:.4f}, {TZ+0.002+ext[2]:.4f}]  离桌 +0.20 ~ +{(0.002+ext[2])*100:.2f}cm")
print(f"  affordance {np.round(aff,4).tolist()}   (离桌 {(aff[2]-TZ)*100:+.2f}cm)")
print(f"\n[对照] 旧 palm 锚的物体原点 = [0.063, 0.141, ...] (在机器人左侧, 右手跨身)")
op = raw["obj_pose"][:, :3].astype(np.float64)
print(f"\n[重建世界系 gravity_z_up_world, 原点在人头相机]")
print(f"  首帧 {np.round(op[0],4).tolist()}   末帧 {np.round(op[-1],4).tolist()}")
cam = PC.load_camera_xy(e["mesh"])
print(f"  相机(人头) xy 中位 {np.round(cam,4).tolist()}")
print(f"  相机锚定平移量 xy = {np.round(PC.camera_anchor_shift(e['mesh'], PC.ZED_NOMINAL[:2], op[0,:2]),4).tolist()}")

print()
print("=" * 78)
print("3. SharpaWave 手指长度 / 合拢后指尖相对腕的位置")
print("=" * 78)
u = Urdf()
JN = [n.replace("right_", f"{HAND}_") for n in
      ["right_thumb_CMC_FE","right_thumb_CMC_AA","right_thumb_MCP_FE","right_thumb_MCP_AA",
       "right_thumb_IP","right_index_MCP_FE","right_index_MCP_AA","right_index_PIP",
       "right_index_DIP","right_middle_MCP_FE","right_middle_MCP_AA","right_middle_PIP",
       "right_middle_DIP","right_ring_MCP_FE","right_ring_MCP_AA","right_ring_PIP",
       "right_ring_DIP","right_pinky_CMC","right_pinky_MCP_FE","right_pinky_MCP_AA",
       "right_pinky_PIP","right_pinky_DIP"]]
BASE = f"{HAND}_hand_C_MC"
FING = ("thumb", "index", "middle", "ring", "pinky")

def fk(qvec, link):
    q = {n: float(v) for n, v in zip(JN, qvec)}
    return u.link_pose(link, q, np.eye(4), BASE)[:3, 3]

print(f"基准坐标系 = 手基座 link `{BASE}` (即腕部法兰)\n")
print(f"{'指':<8}{'链节数':>7}{'骨长和(cm)':>12}{'张开时 基座→指尖(cm)':>22}{'合拢时(cm)':>13}")
seg_len = {}
for f in FING:
    tip = f"{HAND}_{f}_fingertip"
    ch = u.chain_between(BASE, tip)
    # 骨长: 沿链累加每个 joint origin 的平移模长 (与关节角无关)
    L = sum(np.linalg.norm(u.joints[n]["origin"][:3, 3]) for n in ch)
    seg_len[f] = L
    d_o = np.linalg.norm(fk(GENERIC_OPEN, tip))
    d_c = np.linalg.norm(fk(GENERIC_CLOSED, tip))
    print(f"{f:<8}{len(ch):>7}{L*100:>12.2f}{d_o*100:>22.2f}{d_c*100:>13.2f}")

print(f"\n合拢终态, 相对手基座 `{BASE}` 的位置向量 (cm):")
print(f"{'指':<8}{'fingertip (x,y,z)':>30}{'|d|':>8}   {'elastomer 胶垫 (x,y,z)':>30}{'|d|':>8}")
for f in FING:
    a = fk(GENERIC_CLOSED, f"{HAND}_{f}_fingertip") * 100
    b = fk(GENERIC_CLOSED, f"{HAND}_{f}_elastomer") * 100
    print(f"{f:<8}{str(np.round(a,2).tolist()):>30}{np.linalg.norm(a):>8.2f}   "
          f"{str(np.round(b,2).tolist()):>30}{np.linalg.norm(b):>8.2f}")
gc = grasp_center_local(HAND, GENERIC_CLOSED) * 100
print(f"\n五胶垫质心 (= grasp_center_local, 手真正夹住东西的地方):")
print(f"  {np.round(gc,2).tolist()} cm   离基座 {np.linalg.norm(gc):.2f} cm")
gco = grasp_center_local(HAND, GENERIC_OPEN) * 100
print(f"  张开时 {np.round(gco,2).tolist()} cm   离基座 {np.linalg.norm(gco):.2f} cm  "
      f"(合拢过程中前移 {np.linalg.norm(gc-gco):.2f} cm)")
tp = np.stack([fk(GENERIC_CLOSED, f"{HAND}_{f}_elastomer") for f in FING]) * 100
print(f"\n合拢时五胶垫两两最大间距 {max(np.linalg.norm(tp[i]-tp[j]) for i in range(5) for j in range(5)):.2f} cm"
      f"  (物体直径 {ext[:2].max()*100:.2f} cm)")

print()
print("=" * 78)
print("4. 重建轨迹: 交互段内手和物体各移动了多少")
print("=" * 78)
ph = raw[f"phase_{HAND}"].astype(int)
con = np.flatnonzero(ph == 1)
a, b = int(con[0]), int(con[-1])
fps = float(raw["fps"])
W = raw[f"joints_{HAND}"][:, 0, :3].astype(np.float64)     # 腕 (joint 0)
O = op
print(f"接触段 phase_right==1: 帧 [{a}, {b}]  共 {b-a+1} 帧 / {(b-a)/fps:.2f}s  (fps={fps:.1f}, 总 {len(ph)} 帧)")
print(f"物体 track 置信度 中位 {np.median(raw['obj_confidence'][a:b+1]):.3f}  "
      f"手 {np.median(raw['confidence_right'][a:b+1]):.3f}")

def rep(nm, X):
    s, t = X[a], X[b]
    net = np.linalg.norm(t - s)
    path = np.linalg.norm(np.diff(X[a:b+1], axis=0), axis=1).sum()
    print(f"\n  {nm}")
    print(f"    起点 {np.round(s,4).tolist()}")
    print(f"    终点 {np.round(t,4).tolist()}")
    print(f"    净位移向量 {np.round(t-s,4).tolist()}   |净位移| {net*100:.2f} cm")
    print(f"    路径长度 {path*100:.2f} cm   竖直净变化 {(t[2]-s[2])*100:+.2f} cm")
    return s, t, net, path

ws, wt, wnet, wpath = rep("人手腕 (joints_right[:,0])", W)
os_, ot, onet, opath = rep("物体 (obj_pose[:,:3])", O)

rel = W - O
print(f"\n  手物相对向量 (腕 − 物体):")
print(f"    接触起 {np.round(rel[a],4).tolist()}  |d| {np.linalg.norm(rel[a])*100:.2f} cm")
print(f"    接触末 {np.round(rel[b],4).tolist()}  |d| {np.linalg.norm(rel[b])*100:.2f} cm")
print(f"    相对漂移 {np.linalg.norm(rel[b]-rel[a])*100:.2f} cm  "
      f"(段内最大偏离起始相对位 {np.linalg.norm(rel[a:b+1]-rel[a],axis=1).max()*100:.2f} cm)")
print(f"\n  -> '一起移动'的口径: 手净移 {wnet*100:.2f}cm, 物体净移 {onet*100:.2f}cm, "
      f"两者之差 {abs(wnet-onet)*100:.2f}cm")
print(f"     刚性握持应当相对漂移≈0; 实测 {np.linalg.norm(rel[b]-rel[a])*100:.2f}cm "
      f"= 物体 track 噪声(它是被弃用的通道)")
zmax = O[a:b+1, 2].max()
print(f"\n  物体在接触段内最高抬到 z={zmax:.4f} (比接触起点高 {(zmax-O[a,2])*100:+.2f} cm)")
print(f"  手在接触段内最高 z={W[a:b+1,2].max():.4f} (比接触起点高 {(W[a:b+1,2].max()-W[a,2])*100:+.2f} cm)")
print(f"\n  ⚠ 以上是**重建世界系**(原点在人头相机)的数. env 里物体 track 被弃用:")
print(f"     replay_grasp 用 '物体参考 = 抓取帧位置 + 腕相对位移' 重造, 且 grasp_only 下")
print(f"     freeze_wrist=True 把腕冻结在 PreGrasp, 搬运段根本不执行 (目标只有抬 10cm).")
