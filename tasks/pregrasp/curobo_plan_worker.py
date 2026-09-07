"""cuRobo 规划子进程 —— **不启 Isaac**, 只做规划, 结果写 npz。

## 为什么必须是独立进程

cuRobo 的碰撞检查依赖 NVIDIA Warp, 而 **Isaac Sim 自带的 warp 版本与它不兼容**:
Isaac 起来之后再 import cuRobo 会崩在 wp_autograd.py
("TypeError: func() got an unexpected keyword argument 'module'")。
单独跑则完全正常。所以规划在**干净解释器**里做, 由 Isaac 侧 subprocess 调用。

## 结构: 照抄 V2AP 的 Franka 抓取模式 (DexImit gen_traj.py + curobo_util.py)

**从不直接规划到抓握位姿** —— 抓握位姿的手指贴着物体/离桌面近, IK 的碰撞余量会
直接把终点判死(2026-08-17 实测: 只有桌子当障碍也 1.2s 快速失败; act_dist 调小无效,
说明 IK 的余量是内置的, 不吃那个参数)。V2AP 的解法:

    每只手分两段:  ① 站姿 → **pregrasp**(抓握位姿沿径向后退 8cm) —— 满障碍规划
                   ② pregrasp → grasp —— 8cm 短腿, 空障碍(接触本来就要发生)

其余实测教训(每条都栽过):
  * 场景必须是**嵌套字典** {"cuboid":{名:{...}}, "mesh":{名:{...}}} ——
    传 {名:对象} 会**静默空场景**(pad 三档轨迹逐位相同为证)。
  * **每个世界各建一个规划器**; update_world 疑似静默无效, 不用。
  * 靶点 link = `{side}_hand_C_MC`(与 RL 的 ee_body 一致), 不是 R_ee/L_ee。
  * 躯干/头锁死(角度取 RL 的躯干锁死 USD), 否则规划器靠转躯干够目标。
  * `success` ≠ 到位; 必须自己验收: 终点误差 + 腕全程不低于**真实**桌面。
  * FK 输入必须 contiguous; 关节名用 js_solution 自带的(64 个), FK 用 planner 的 20 个。
  * 先右手后左手; 左手从右手终点构型出发(避开已就位的右臂);
    左手规划时**排除右手已占住的物体**(右手贴着瓶 ⟹ 起点会被判在碰撞里)。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

# 机器人配置优先显式 CUROBO_ROBOT_YML，其次兼容 MagicSim vendor 树，最后使用
# 仓库内由 NVlabs/curobo RobotBuilder 生成的自带配置。
def _robot_yml() -> str:
    """CUROBO_ROBOT_YML > MagicSim 树 > 仓库自带生成品 (跨机自给自足)。

    2026-08-30 起 cuRobo 不再依赖 MagicSim: `curobo.motion_planner` 这套 API 就是
    NVlabs/curobo 新版主线 (v0.8+, warp 内核), 机器人配置可由
    tools/make_vega1p_sharpa_curobo_yml.py 从仓库 URDF 自动生成。
    """
    env = os.environ.get("CUROBO_ROBOT_YML")
    if env:
        return env
    mag = os.path.join(
        os.environ.get("MAGICSIM_ROOT", "/home/lyh/luhr/MagicSim"),
        "Third_Party", "curobo", "curobo", "content", "configs", "robot",
        "magicsim_vega1p_sharpa.yml")
    if os.path.isfile(mag):
        return mag
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "vega_urdf",
        "vega_1p_sharpa_curobo.yml"))


ROBOT_YML = _robot_yml()


def load_robot_yaml(path: str) -> dict:
    """Load a robot YAML and repair bundled asset paths after relocation."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    kin = raw["robot_cfg"]["kinematics"]
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _abs(p):
        # 入库 yml 存的是仓库相对路径; 按**仓库根**解析 (worker 可能从任意
        # 工作目录被 spawn —— 靠 cwd 会时灵时不灵)。
        return p if not p or os.path.isabs(p) else os.path.join(repo, p)

    asset_root = _abs(kin.get("asset_root_path"))
    urdf_path = _abs(kin.get("urdf_path"))
    if (asset_root and os.path.isdir(asset_root)
            and urdf_path and os.path.isfile(urdf_path)):
        kin["asset_root_path"], kin["urdf_path"] = asset_root, urdf_path
        return raw

    bundled_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "datasets", "vega_urdf",
        "vega_1p_sharpa"))
    bundled_urdf = os.path.join(bundled_root, "vega_1p_sharpa.urdf")
    if not os.path.isfile(bundled_urdf):
        raise FileNotFoundError(
            f"cuRobo robot assets unavailable: yaml={path}, "
            f"asset_root={asset_root}, urdf={urdf_path}, "
            f"bundled={bundled_urdf}")
    kin["asset_root_path"] = bundled_root
    kin["urdf_path"] = bundled_urdf
    print(f"[worker] robot YAML 已重定位到仓库资产: {bundled_root}",
          flush=True)
    return raw


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def _savez_atomic(path, **payload):
    """Publish a planner result only after the NPZ is complete."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default=ROBOT_YML)
    ap.add_argument("--act_dist", type=float, default=0.015)
    ap.add_argument("--pregrasp_dist", type=float, default=0.08,
                    help="pregrasp 后退距离(m), V2AP 用 0.08")
    ap.add_argument("--attempts", type=int, default=20,
                    help="plan_pose 重试次数 (V2AP 用到 300; fork 默认 5 太少)")
    ap.add_argument("--table_pad", type=float, default=0.0,
                    help="规划用桌面垫高(m); 验收仍按真实桌面")
    ap.add_argument("--joint", type=int, default=0,
                    help="1=双手联合规划(同时移动, 两臂互查碰撞), 0=先右后左")
    ap.add_argument("--left_short_cm", type=float, default=5.0,
                    help="joint 模式: 左手终点从抓握位姿沿接近轴后退这么多 cm "
                         "(蹭杯段不进参考, 最后一段归 RL 冻结区)")
    ap.add_argument("--own_obstacle", type=int, default=0,
                    help="1=顺序模式 pregrasp 腿把**本手目标物体留在碰撞世界**里 "
                         "(直达小后退距离时必开: 否则转运安全无人背书, 2026-08-19 "
                         "1cm直达撞瓶实证; 旧契约'排除+距离兜底'只适用 >=5cm)")
    ap.add_argument("--lock_other_arm", type=int, default=0,
                    help="顺序模式规划某手时锁死另一侧臂7关节在起始值 (2026-08-20): "
                         "防规划器'顺手'甩另一臂 (充气障碍下实测左臂被右手规划蹭走 56°, "
                         "叠放前提崩坏; 历史上仅 3° 由 blend10 兜)。")
    ap.add_argument("--obj_inflate_map", type=str, default="",
                    help='JSON: 按物体名覆盖 obj_inflate, 如 {"obj_secondary":-0.02}。'
                         "用处 (T2-26): 撤退起点只被贴指的盖判碰, 盖深收缩解锁起点; "
                         "瓶保持浅收缩, 路径才会绕开瓶body (全局 -2cm 实测规划器"
                         "从瓶边穿过)。")
    ap.add_argument("--obj_inflate", type=float, default=0.0,
                    help="物体障碍充气 (m, 2026-08-20 用户裁定): 碰撞世界里把物体网格沿"
                         "顶点法线外推这么多再规划 (实际物体不变) —— 规划自动多留净空。"
                         "建议 0.01; 只作用于 objects, 桌面不充")
    ap.add_argument("--exclude_table", type=int, default=0,
                    help="1=把桌面排除出碰撞世界 (贴物短腿专用: 抓握位形离桌只有"
                         "几厘米, cuRobo 的手部碰撞球比实物保守会把目标判碰; "
                         "该腿只在站位高度附近平移, 撞桌风险为零)")
    ap.add_argument("--exclude_objects", type=int, default=0,
                    help="1=joint 模式把两个目标物体排除出碰撞世界 (接触短腿专用: "
                         "PreGrasp→GraspPose 本来就要贴物, 桌子仍是障碍)")
    a = ap.parse_args()

    with open(a.targets) as f:
        T = json.load(f)

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

    raw = load_robot_yaml(a.robot)

    _tool = list(T["tool_frames"])
    raw["robot_cfg"]["kinematics"]["tool_frames"] = _tool
    _lock = T.get("lock_joints") or []
    if _lock:
        _lj = dict(raw["robot_cfg"]["kinematics"].get("lock_joints") or {})
        for _n in _lock:
            _lj[_n] = float(T["start_joints"].get(_n, 0.0))
        raw["robot_cfg"]["kinematics"]["lock_joints"] = _lj
    print(f"[worker] tool_frames={_tool} | 锁死 {len(_lock)} 个躯干/头关节", flush=True)

    tp, td = list(T["table_pose"]), list(T["table_dims"])
    tzt_real = tp[2] + td[2] / 2.0                     # 真实桌面(验收用)
    if a.table_pad > 0:
        tp = [tp[0], tp[1], tp[2] + a.table_pad / 2.0]
        td = [td[0], td[1], td[2] + a.table_pad]
    objects = {o["name"]: o for o in T.get("objects", [])}

    _inflated = {}
    _inflate_map = json.loads(a.obj_inflate_map) if a.obj_inflate_map else {}
    def _mesh_path(o):
        """顶点沿法线推 inflate 米 (>0 充气 / <0 收缩), 存临时文件; 实物不变。

        收缩 (<0) 的用处: "贴着抓" 的目标位形若把原尺寸物体留在世界里必被判碰,
        而整个排除物体又会让这一腿完全失去避障 —— 收缩 1cm 量级两头兼顾:
        目标位形合法, 路径仍绕开物体本体。
        """
        _inf = float(_inflate_map.get(o.get("name", ""), a.obj_inflate))
        if _inf == 0.0:
            # 必须给**绝对路径**: 相对路径会被 cuRobo 当成它自己 content 目录下的
            # 资产 (报 "string is not a file: .../curobo/content/assets/<相对路径>")。
            # 充气>0 时走临时文件天然是绝对路径, 所以这条坑只在 inflate=0 时暴露。
            return os.path.abspath(o["mesh"])
        p = o["mesh"]
        _ck = (p, _inf)
        if _ck not in _inflated:
            import tempfile
            import trimesh
            m = trimesh.load(p, force="mesh")
            m.vertices = m.vertices + m.vertex_normals * _inf
            fp = tempfile.NamedTemporaryFile(
                suffix="_inf.obj", delete=False).name
            m.export(fp)
            print(f"[inflate] {o.get('name', '?')} {p} "
                  f"{'充气' if _inf > 0 else '收缩'} "
                  f"{abs(_inf)*1000:.0f}mm -> {fp}")
            _inflated[_ck] = fp
        return _inflated[_ck]

    def _scene_dict(exclude=(), no_world=False):
        if no_world:
            return {"cuboid": {}, "mesh": {}}
        d = {"cuboid": ({} if int(a.exclude_table)
                        else {"table": {"pose": [*tp, 1.0, 0.0, 0.0, 0.0],
                                        "dims": td}}),
             "mesh": {}}
        for nm, o in objects.items():
            if nm in exclude:
                continue
            d["mesh"][nm] = {"file_path": _mesh_path(o),
                             "pose": [*o["pos"], *o["quat"]]}
        return d

    _cache = {}

    def _get_planner(exclude=(), no_world=False, tag="", lock_extra=None):
        key = (("EMPTY",) if no_world else tuple(sorted(exclude))) \
            + (tuple(sorted((lock_extra or {}).items())))
        if key in _cache:
            return _cache[key]
        _rc = raw["robot_cfg"]
        if lock_extra:
            import copy
            _rc = copy.deepcopy(_rc)
            _lj2 = dict(_rc["kinematics"].get("lock_joints") or {})
            _lj2.update(lock_extra)
            _rc["kinematics"]["lock_joints"] = _lj2
            print(f"[worker] {tag}: 额外锁死 {sorted(lock_extra)} (另一侧臂)",
                  flush=True)
        _c = MotionPlannerCfg.create(
            robot=_rc,
            scene_model=[_scene_dict(exclude, no_world)],
            device_cfg=DeviceCfg(device="cuda:0", dtype=torch.float32),
            self_collision_check=True, max_batch_size=1, multi_env=True,
            max_goalset=1, collision_cache={"cuboid": 10, "mesh": 500},
            num_trajopt_seeds=4, num_ik_seeds=32, use_cuda_graph=False,
            optimizer_collision_activation_distance=float(a.act_dist),
        )
        _p = MotionPlanner(_c)
        # cspace 模式需要图规划器做长程种子 (直线关节种子必扫过桌面, 2 次
        # trajopt 尝试即放弃 —— 2026-08-27 终章三连败根因)
        _p.warmup(enable_graph=bool(T.get("cspace_goal")),
                  num_warmup_iterations=1)
        # ★ 躯干/底座/头对世界的碰撞必须关掉: 机器人站在 x=-0.5, 桌子横跨
        #   x∈[-0.6,+0.6] —— **躯干柱本来就站在桌子箱体里**(RL 环境里这对碰撞
        #   被过滤, 靠几何罚只管手臂)。不关的话**起点即碰撞**, 任何目标都秒失败
        #   (2026-08-17 实测: 连后退 8cm 的 pregrasp 都 5.5s 失败, 空世界则全通)。
        if not no_world:
            _links = ["vega_1p_base", "vega_1p_torso_l1", "vega_1p_torso_l2",
                      "vega_1p_torso_l3", "vega_1p_head_l1", "vega_1p_head_l2",
                      "vega_1p_head_l3"]
            try:
                _p.disable_link_collision(_links)
                print(f"[worker]   已关 躯干/底座/头 对世界的碰撞 ({len(_links)} links)",
                      flush=True)
            except Exception as _e:
                print(f"[worker]   ⚠ disable_link_collision 失败: "
                      f"{type(_e).__name__}: {_e}", flush=True)
        _w = "空" if no_world else f"桌 + {list(_scene_dict(exclude)['mesh'].keys())}"
        print(f"[worker] 规划器[{tag}] 就绪 | 世界: {_w}", flush=True)
        _cache[key] = _p
        return _p

    base = _get_planner(tag="满障碍")
    pj = list(base.joint_names)

    q0 = base.default_joint_state.position.clone()
    hit = 0
    for i, n in enumerate(pj):
        if n in T["start_joints"]:
            q0[i] = float(T["start_joints"][n]); hit += 1
    print(f"[worker] 起点 = env 站姿 ({hit}/{len(pj)} 关节)", flush=True)

    def _pose(pos, quat):
        return Pose(position=torch.tensor([list(pos)], device="cuda:0",
                                          dtype=torch.float32),
                    quaternion=torch.tensor([list(quat)], device="cuda:0",
                                            dtype=torch.float32))

    def _plan(planner, cur_q, goal_frame, goal_pos, goal_quat):
        """goal_frame 送目标, 其余 tool frame 钉在当前 FK。"""
        _pn = list(planner.joint_names)
        if _pn != pj:
            # 锁臂后该规划器的活动关节集缩小 —— 按名对齐, 否则 FK 名单不匹配报错
            _sel = [pj.index(n) for n in _pn]
            cur = JointState.from_position(cur_q[_sel].unsqueeze(0),
                                           joint_names=_pn)
        else:
            cur = JointState.from_position(cur_q.unsqueeze(0), joint_names=pj)
        kin = planner.compute_kinematics(cur)
        pd = {}
        for f in _tool:
            if f == goal_frame:
                pd[f] = _pose(goal_pos, goal_quat)
            else:
                p = kin.tool_poses.get_link_pose(f, make_contiguous=True)
                pd[f] = Pose(position=p.position.view(-1, 3).contiguous(),
                             quaternion=p.quaternion.view(-1, 4).contiguous())
        g = GoalToolPose.from_poses(pd, ordered_tool_frames=_tool, num_goalset=1)
        t0 = time.time()
        r = planner.plan_pose(g, cur, max_attempts=int(a.attempts),
                              enable_graph_attempt=2)
        dt = time.time() - t0
        ok = r is not None and bool(
            getattr(r, "success", torch.tensor([False])).flatten()[0])
        return r, ok, dt

    order = [f for f in _tool if f.startswith("right")] + \
            [f for f in _tool if not f.startswith("right")]
    tgt_obj = dict(T.get("hand_targets") or {})
    segs, cur_q, names, claimed = [], q0.clone(), None, set()
    _vgoal = {}          # 逐手验收靶点 (joint 模式左手 = 近点, 不是抓握位姿)
    _meta_pre = {}       # pregrasp 出身戳: 哪个障碍档 / 实际后退量

    def _plan_multi(planner, cq, goal_map):
        """goal_map 里的 frame 送目标, 其余钉在当前 FK (联合规划的核心)。"""
        cur = JointState.from_position(cq.unsqueeze(0), joint_names=pj)
        kin = planner.compute_kinematics(cur)
        pd = {}
        for f in _tool:
            if f in goal_map:
                pos, quat = goal_map[f]
                pd[f] = _pose(pos, quat)
            else:
                p = kin.tool_poses.get_link_pose(f, make_contiguous=True)
                pd[f] = Pose(position=p.position.view(-1, 3).contiguous(),
                             quaternion=p.quaternion.view(-1, 4).contiguous())
        g = GoalToolPose.from_poses(pd, ordered_tool_frames=_tool, num_goalset=1)
        t0 = time.time()
        r = planner.plan_pose(g, cur, max_attempts=int(a.attempts),
                              enable_graph_attempt=2)
        ok = r is not None and bool(
            getattr(r, "success", torch.tensor([False])).flatten()[0])
        return r, ok, time.time() - t0

    def _take(r):
        """规划结果 -> (numpy 轨迹, 关节名, 终点构型)。"""
        p_ = r.js_solution.position
        while p_.dim() > 2:
            p_ = p_.squeeze(0)
        nm = list(getattr(r.js_solution, "joint_names", None) or pj)
        idx_ = [nm.index(n) for n in pj]
        return p_.detach().cpu().numpy(), nm, p_[-1, idx_].clone()

    CS = T.get("cspace_goal") or {}
    if CS:
        # ======== cspace 模式 (2026-08-27): 关节空间直达目标 ========
        # 动机: 笛卡尔腕位目标 + 7自由度冗余 => 规划终帧≠指定关节位形, 末尾
        # 硬贴产生抖动。plan_cspace 全程避障且终帧就是目标关节, 无需贴合段。
        goal_q = q0.clone()
        for _i, _n in enumerate(pj):
            if _n in CS:
                goal_q[_i] = float(CS[_n])
        _gst = JointState.from_position(goal_q.unsqueeze(0), joint_names=pj)
        _cur = JointState.from_position(q0.clone().unsqueeze(0), joint_names=pj)
        # --exclude_objects 在 cspace 模式下同样生效 (2026-08-31): 机器段的
        # **最后一腿**是"净空点 -> 操作位置", 那一段本来就是要去贴物体, 把两个
        # 目标物体留在碰撞世界里 => 目标位形必被判碰 (goal in collision), 规划
        # 无解。桌子仍是障碍。这条口子原来只接在 joint/pose 模式上。
        _csp = (base if not int(a.exclude_objects)
                else _get_planner(exclude=set(objects.keys()),
                                  tag="cspace:排除目标物体"))
        print(f"[worker] === cspace: 关节空间直达目标"
              f"{' (目标物体已排除出碰撞世界)' if int(a.exclude_objects) else ''}"
              f" ===", flush=True)
        _t0 = time.time()
        _r = _csp.plan_cspace(_gst, _cur, max_attempts=int(a.attempts),
                              enable_graph_attempt=2)
        _dt = time.time() - _t0
        _ok = _r is not None and bool(
            getattr(_r, "success", torch.tensor([False])).flatten()[0])
        if not _ok:
            print("[worker] ❌ cspace 规划失败 —— 启动分诊", flush=True)
            try:
                _lo = _csp.kinematics.get_joint_limits() if hasattr(_csp, "kinematics") else None
            except Exception:
                _lo = None
            print(f"[cspace诊] success={getattr(_r,'success',None)} "
                  f"cspace_err={getattr(_r,'cspace_error',None)} "
                  f"pos_err={getattr(_r,'position_error',None)}", flush=True)
            _pnw = _get_planner(no_world=True, tag="cspace:无世界探针")
            _r2 = _pnw.plan_cspace(_gst, _cur, max_attempts=4,
                                   enable_graph_attempt=2)
            _ok2 = _r2 is not None and bool(
                getattr(_r2, "success", torch.tensor([False])).flatten()[0])
            print(f"[cspace诊] 无世界对照: {'✅通过 => 碰撞世界问题(起点/路径/目标被判碰)' if _ok2 else '❌也失败 => API/关节集问题'}",
                  flush=True)
            # 坐标系体检: 同一(底座)系里 腕位 vs 世界模型里的物体位置
            _cur0 = JointState.from_position(q0.clone().unsqueeze(0),
                                             joint_names=pj)
            _kin0 = base.compute_kinematics(_cur0)
            for _f0 in _tool:
                _wp0 = _kin0.tool_poses.get_link_pose(_f0, make_contiguous=True)\
                    .position.view(-1, 3)[0].detach().cpu().numpy()
                for _o0 in T.get("objects", []):
                    _op0 = np.asarray(_o0["pos"], float)
                    print(f"[cspace诊] {_f0} 腕@{np.round(_wp0*100,1)}cm ↔ "
                          f"{_o0['name']}@{np.round(_op0*100,1)}cm "
                          f"距 {np.linalg.norm(_wp0-_op0)*100:.1f}cm", flush=True)
            # 逐物体排除微动: 点名肇事物体
            for _ex1, _tag1 in (({"obj_secondary"}, "仅瓶(排杯)"),
                                ({"obj_primary"}, "仅杯(排瓶)")):
                _p1 = _get_planner(exclude=_ex1, tag=f"cspace:{_tag1}")
                _qb1 = q0.clone(); _qb1[0] += 0.035
                _r1 = _p1.plan_cspace(
                    JointState.from_position(_qb1.unsqueeze(0), joint_names=pj),
                    JointState.from_position(q0.clone().unsqueeze(0),
                                             joint_names=pj),
                    max_attempts=2, enable_graph_attempt=99)
                _ok1 = _r1 is not None and bool(
                    getattr(_r1, "success", torch.tensor([False])).flatten()[0])
                print(f"[cspace诊] 起点微动@{_tag1}: "
                      f"{'✅可行' if _ok1 else '❌判碰 ⟹ 该世界里的物体是肇事者'}",
                      flush=True)
            # 仅桌世界起点微动: 空世界(自碰开)已通过⟹自碰无辜; 此探针分清 桌vs物体
            _pto = _get_planner(exclude={"obj_primary", "obj_secondary"},
                                tag="cspace:仅桌探针")
            _qb0 = q0.clone(); _qb0[0] += 0.035
            _rt = _pto.plan_cspace(
                JointState.from_position(_qb0.unsqueeze(0), joint_names=pj),
                JointState.from_position(q0.clone().unsqueeze(0), joint_names=pj),
                max_attempts=2, enable_graph_attempt=99)
            _okt = _rt is not None and bool(
                getattr(_rt, "success", torch.tensor([False])).flatten()[0])
            print(f"[cspace诊] 起点微动@仅桌: "
                  f"{'✅可行 ⟹ 物体有责(指尖仍在物体激活区)' if _okt else '❌仍判碰 ⟹ 桌有责'}",
                  flush=True)
            # 端点微动探针: 满障碍下原地动 2° —— 失败=该端点本身在碰撞区
            for _tag3, _q3 in (("起点", q0.clone()), ("站姿目标", goal_q.clone())):
                _qa = _q3.clone(); _qb = _q3.clone(); _qb[0] += 0.035
                _ra = base.plan_cspace(
                    JointState.from_position(_qb.unsqueeze(0), joint_names=pj),
                    JointState.from_position(_qa.unsqueeze(0), joint_names=pj),
                    max_attempts=2, enable_graph_attempt=99)
                _oka = _ra is not None and bool(
                    getattr(_ra, "success", torch.tensor([False])).flatten()[0])
                print(f"[cspace诊] {_tag3}原地微动: {'✅可行' if _oka else '❌被判碰 => 病灶在此'}",
                      flush=True)
            _savez_atomic(a.out, ok=False, failed_frame="cspace", seconds=_dt,
                     noworld_ok=_ok2)
            return
        _arr, _names, _ = _take(_r)
        _idx = [_names.index(n) for n in pj]
        _jerr = float(np.abs(_arr[-1][_idx] - goal_q.cpu().numpy()).max())
        print(f"[worker] ✅ cspace | {_arr.shape[0]} 路点 | 终帧最大关节误差 "
              f"{np.degrees(_jerr):.2f}° | {_dt:.1f}s", flush=True)
        _qs = np.ascontiguousarray(_arr[:, _idx], dtype=np.float32)
        _kin = base.compute_kinematics(JointState.from_position(
            torch.as_tensor(_qs, device="cuda:0", dtype=torch.float32).contiguous(),
            joint_names=pj))
        _owp = {f: _kin.tool_poses.get_link_pose(f, make_contiguous=True)
                .position.view(-1, 3).detach().cpu().numpy() for f in _tool}
        _savez_atomic(a.out, ok=True, traj=_arr,
                 joint_names=np.array(_names, dtype=object),
                 tool_frames=np.array(_tool, dtype=object),
                 seg_frames=np.array(["cspace"], dtype=object),
                 seg_lens=np.array([_arr.shape[0]], np.int64),
                 **{f"wp_{f}": v for f, v in _owp.items()})
        return

    if int(a.joint):
        # ======== 联合模式 (裁定B, 2026-08-18) ========
        # 唯一目标 = Dexonomy 自带 pregrasp[0] (固定 ~5cm, 无候选/无档位/无降级),
        # 世界 = 桌 + 两个物体(旧版排除物体致参考穿杯, 零动作 100% L pushed 实锤)。
        # 后续进近合拢段(pregrasp[0]→grasp→squeeze)由 env 播数据斜坡, 不经 cuRobo。
        # 失败 = 直接报错给人看(GraspPose 可信, 不自动换候选)。
        rf = order[0]
        lf = order[1]
        PG = T.get("pregrasp_goals") or {}
        assert rf in PG and lf in PG, \
            "targets json 缺 pregrasp_goals —— 调用方需从 prior npz 的 pregrasp[0] 提供"
        gpr = np.asarray(PG[rf]["pos"], float)
        gqr = np.asarray(PG[rf]["quat"], float)
        gpl = np.asarray(PG[lf]["pos"], float)
        gql = np.asarray(PG[lf]["quat"], float)
        print(f"\n[worker] === joint/pregrasp0 (满障碍) -> "
              f"R{np.round(gpr, 3)} L{np.round(gpl, 3)} ===", flush=True)
        _pl1 = (base if not int(a.exclude_objects)
                else _get_planner(exclude=set(tgt_obj.values()),
                                  tag="joint:contact-leg(物体排除)"))
        r1, ok, dt = _plan_multi(_pl1, q0.clone(),
                                 {rf: (gpr, gqr), lf: (gpl, gql)})
        if not ok:
            print("[worker] ❌ joint/pregrasp0 规划失败 (不降级不换候选, 人工定夺)",
                  flush=True)
            _savez_atomic(a.out, ok=False, failed_frame="joint/pregrasp0", seconds=dt)
            return
        arr1, names, cur_q = _take(r1)
        segs.append(("both", "pregrasp", arr1))
        print(f"[worker] ✅ joint/pregrasp0 | {arr1.shape[0]} 路点 | {dt:.1f}s",
              flush=True)
        _meta_pre = {"pregrasp_world": "满障碍(pregrasp0)"}
        _vgoal[rf] = (gpr, gqr)
        _vgoal[lf] = (gpl, gql)
    else:
     for f in order:
        own = tgt_obj.get(f)
        gp = np.asarray(T["goals"][f]["pos"], float)
        gq = np.asarray(T["goals"][f]["quat"], float)
        oc = np.asarray(objects[own]["pos"], float) if own in objects else gp
        u = gp - oc
        u = u / max(np.linalg.norm(u), 1e-9)
        _R = _quat_to_R(gq / max(np.linalg.norm(gq), 1e-9))
        _tz = _R @ np.array([0.0, 0.0, 1.0])
        _up = np.array([0.0, 0.0, 1.0])
        _ru = u + _up
        _ru = _ru / max(np.linalg.norm(_ru), 1e-9)
        # ★ pregrasp 后退方向必须是**候选列表**, 不能只有径向:
        #   左手径向后退 8cm 实测**运动学不可达**(探针②: 空世界+站姿起点仍失败) ——
        #   同一朝向下手要缩到离躯干更近处, 锁死躯干的左臂摆不出来。
        #   候选: 径向(右手可行) / ∓工具z(plan_grasp 的接近轴) / 径向+抬高 / 纯抬高,
        #   各试 pregrasp_dist 和 5cm, 用**满障碍**规划器逐个试到第一个成功。
        cand = []
        for _d in (float(a.pregrasp_dist), 0.05):
            for _dir in (u, -_tz, _tz, _ru, _up):
                cand.append(gp + _dir * _d)
        # ★ pregrasp 段也要排除**本手的目标物体** (V2AP 语义: 目标物体从不进碰撞世界,
        #   转运安全靠后退距离保证)。实测: 左手运动学可行的候选都在抓握位姿上方 5cm,
        #   手指仍在杯口附近 —— 杯子留在世界里就把它们全判死(探针①: 空世界可行)。
        #   桌子 + **其他**物体仍是硬障碍。
        _excl = set(claimed) if int(a.own_obstacle) else (set(claimed) | {own})
        _lockx = None
        if int(a.lock_other_arm):
            _oth = "L" if f.startswith("right") else "R"
            _lockx = {f"{_oth}_arm_j{i}":
                      float(T["start_joints"][f"{_oth}_arm_j{i}"])
                      for i in range(1, 8)}
        stages = [
            ("pregrasp", _get_planner(exclude=_excl, lock_extra=_lockx,
                                      tag=f"{f}:pregrasp"), cand),
            # 空障碍走最后一小段 —— 接触本来就要发生, IK 内置余量才不会判死
            ("grasp", _get_planner(no_world=True, lock_extra=_lockx,
                                   tag=f"{f}:grasp"), [gp]),
        ]
        for stage, planner, cands in stages:
            _disposable = (stage == "pregrasp")     # 单次使用, 用完即毁省显存
            r, ok, dt = None, False, 0.0
            for _ci, pos in enumerate(cands):
                print(f"\n[worker] === {f} / {stage}[{_ci}/{len(cands)}] -> "
                      f"{np.round(pos, 3)} ===", flush=True)
                r, ok, dt = _plan(planner, cur_q, f, pos, gq)
                if ok:
                    break
            if not ok:
                print(f"[worker] ❌ {f}/{stage} 规划失败 ({dt:.1f}s)", flush=True)
                print("[worker] 自动隔离诊断:", flush=True)
                _pe = _get_planner(no_world=True, tag="诊断-空世界")
                _r2, ok2, _ = _plan(_pe, cur_q, f, pos, gq)
                _m2 = "是障碍挡的(桌或另一物体)" if ok2 else "不是障碍的问题"
                print(f"[worker]   探针① 同起点+空世界: "
                      f"{'可行' if ok2 else '仍失败'} -> {_m2}", flush=True)
                if not ok2:
                    _r3, ok3, _ = _plan(_pe, q0.clone(), f, pos, gq)
                    _m3 = ("是**起点构型**挡的(前一只手已就位)" if ok3
                           else "目标本身够不着(朝向/锁躯干/自碰)")
                    print(f"[worker]   探针② 站姿起点+空世界: "
                          f"{'可行' if ok3 else '仍失败'} -> {_m3}", flush=True)
                _savez_atomic(a.out, ok=False, failed_frame=f"{f}/{stage}", seconds=dt)
                return
            p_ = r.js_solution.position
            while p_.dim() > 2:
                p_ = p_.squeeze(0)
            names = list(getattr(r.js_solution, "joint_names", None) or pj)
            if any(n not in names for n in pj):
                # 锁臂规划: 解算缺锁死关节列 —— 补常值列(=起始值), 统一到 pj 宽度
                _full = torch.empty((p_.shape[0], len(pj)), dtype=p_.dtype,
                                    device=p_.device)
                for _j, _n in enumerate(pj):
                    _full[:, _j] = (p_[:, names.index(_n)]
                                    if _n in names else cur_q[_j])
                p_, names = _full, list(pj)
            arr = p_.detach().cpu().numpy()
            segs.append((f, stage, arr))
            idx = [names.index(n) for n in pj]
            cur_q = p_[-1, idx].clone()
            print(f"[worker] ✅ {f}/{stage} | {arr.shape[0]} 路点 | {dt:.1f}s", flush=True)
            if _disposable:
                # 显存节流: 每个规划器 ~1-1.5GB, 16GB 卡上还有 GUI Isaac(2.5GB+),
                # 四个全留会 OOM (2026-08-17 用户 GUI 实测)。pregrasp 的世界逐手不同、
                # 不会复用, 用完即毁; 只常驻 base(满障碍, 供 FK) 和 空世界(两手共用)。
                _k = tuple(sorted(set(claimed) | {own}))
                _cache.pop(_k, None)
                try:
                    planner.destroy()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        claimed.add(own)

    # ---------------- 合并 + FK(逐段, contiguous) ----------------
    traj = np.concatenate([s[2] for s in segs], axis=0)
    idx = [names.index(n) for n in pj]

    def _fk(seg):
        q = np.ascontiguousarray(seg[:, idx], dtype=np.float32)
        return base.compute_kinematics(JointState.from_position(
            torch.as_tensor(q, device="cuda:0", dtype=torch.float32).contiguous(),
            joint_names=pj))

    wp = {f: [] for f in _tool}
    for _f, _st, _seg in segs:
        _k = _fk(_seg)
        for f in _tool:
            wp[f].append(_k.tool_poses.get_link_pose(f, make_contiguous=True)
                         .position.view(-1, 3).detach().cpu().numpy())
    wp = {f: np.concatenate(v, axis=0) for f, v in wp.items()}

    # ---------------- 验收 (success ≠ 到位, 必须自己量) ----------------
    print(f"\n[worker] ===== 验收 (真实桌面 {tzt_real:.3f}m) =====", flush=True)
    bad = []
    for f in order:
        own_segs = [(st, sg) for (ff, st, sg) in segs if ff in (f, "both")]
        _k = _fk(own_segs[-1][1])                       # 最后一段 = grasp
        lp = _k.tool_poses.get_link_pose(f, make_contiguous=True)
        ep = lp.position.view(-1, 3)[-1].detach().cpu().numpy()
        eq = lp.quaternion.view(-1, 4)[-1].detach().cpu().numpy()
        # joint 模式左手靶点 = 近点; 顺序模式回落到抓握位姿
        _vg = _vgoal.get(f) or (np.asarray(T["goals"][f]["pos"], float),
                                np.asarray(T["goals"][f]["quat"], float))
        tp_ = np.asarray(_vg[0], float)
        tq_ = np.asarray(_vg[1], float)
        dp = np.linalg.norm(ep - tp_) * 100
        R1 = _quat_to_R(eq / np.linalg.norm(eq))
        R2 = _quat_to_R(tq_ / np.linalg.norm(tq_))
        dr = np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
        zmin = min(float(_fk(sg).tool_poses.get_link_pose(f, make_contiguous=True)
                         .position.view(-1, 3)[:, 2].min()) for _, sg in own_segs)
        cl = (zmin - tzt_real) * 100
        okp, okz = (dp < 2.0 and dr < 10.0), (cl > 0)
        print(f"[worker] {'✅' if okp else '❌'} {f} 终点误差 位置 {dp:.2f}cm "
              f"朝向 {dr:.1f}°", flush=True)
        print(f"[worker] {'✅' if okz else '❌'} {f} 腕全程最低 {zmin:.3f}m "
              f"(余量 {cl:+.1f}cm)", flush=True)
        if not okp:
            bad.append(f"{f} 终点差 {dp:.1f}cm/{dr:.0f}°")
        if not okz:
            bad.append(f"{f} 腕穿桌 {cl:.1f}cm")
    if bad:
        print(f"[worker] ❌❌ 验收不通过: {bad}", flush=True)
        _savez_atomic(a.out, ok=False, failed_frame="verify",
                 reasons=np.array(bad, dtype=object))
        return
    print("[worker] ✅✅ 验收通过", flush=True)

    for f in _tool:
        d = np.linalg.norm(np.diff(wp[f], axis=0), axis=1)
        st = np.linalg.norm(wp[f][-1] - wp[f][0])
        print(f"[worker] {f}: 行程 {d.sum()*100:6.1f}cm | 直线 {st*100:6.1f}cm "
              f"| 绕路系数 {d.sum()/max(st,1e-9):.2f}x", flush=True)
    _ee = {}
    for f in order:
        _vg = _vgoal.get(f) or (np.asarray(T["goals"][f]["pos"], float),
                                np.asarray(T["goals"][f]["quat"], float))
        _ee[f"end_expect_{f}"] = np.asarray(_vg[0], float)
    _savez_atomic(a.out, ok=True, traj=traj,
             joint_names=np.array(names, dtype=object),
             tool_frames=np.array(_tool, dtype=object),
             seg_lens=np.array([len(s[2]) for s in segs]),
             seg_frames=np.array([f"{s[0]}|{s[1]}" for s in segs], dtype=object),
             joint_mode=int(a.joint), left_short_cm=float(a.left_short_cm),
             **_meta_pre, **_ee, **{f"wp_{f}": wp[f] for f in _tool})


if __name__ == "__main__":
    main()
