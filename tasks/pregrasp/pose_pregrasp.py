"""静态摆位查看器: 双手直接摆在各自 pregrasp[0] (裁定B 的后退位), 不播任何轨迹。

用途: 单独评审 PreGrasp 摆位本身 (位置/朝向/指型/与物体的间隙), 把"终点对不对"
从"路径好不好看"里剥离出来。GUI 里可自由转视角, Ctrl+C 退出。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.pose_pregrasp \
        --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
        --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--prior_b", required=True)
p.add_argument("--prior_b_yaw", type=float, default=-1.0)
p.add_argument("--row", type=int, default=0,
               help="摆哪一档: 0=最远(~5cm) ... 5=最近; -1=抓握位姿 grasp")
p.add_argument("--tighten", type=float, default=0.0,
               help="row=-1 时手指向合拢锚点(close_anchor)收紧的比例, 0.2=收 20%%")
p.add_argument("--root_only", action="store_true",
               help="row=-1 静态摆位变体: 只保留每指**根部关节**的抓形角度 (拇指 CMC 两轴/"
                    "四指 MCP 两轴/小指含 CMC), 其余关节 (拇指 MCP·IP, 四指 PIP·DIP) 全部"
                    "伸直到 GENERIC_OPEN —— 看'只用根部弯曲'的手型与物体的关系")
p.add_argument("--export_ro", default="",
               help="把 ro_approach 六段编舞导出为训练参考 npz (right_q/left_q 臂7 + "
                    "right_q29/left_q29 全29 + 分段表), 导完即退。AAG 训练参考生产口")
p.add_argument("--ro_side_cm", type=float, default=0.0,
               help="侧移接近(2026-08-24): 整条接近路径沿 GraspPose 掌心向外平移该距离"
                    "(cm), 推进行程不变; advance 后新增一段侧向移回原终点")
p.add_argument("--ro_insert_frames", type=int, default=20,
               help="侧向移回段的帧数")
p.add_argument("--ro_sweep", default="",
               help="指尖包络扫描: 逗号分隔的 λ 列表 (root_bend 深度, 1=原编舞), "
                    "逐个回放并检测首次碰物, 数据选参数。例: --ro_sweep 1,0.75,0.5,0.25,0")
p.add_argument("--ro_lam", type=float, default=1.0,
               help="导出时用的 root_bend 深度 λ (配 --export_ro)")
p.add_argument("--patch_ro", default="",
               help="碰撞局部修补器 (2026-08-22 用户裁定): 零动作回放找首碰帧, 在 "
                    "advance 段做最小改动绕行 (分叉最晚/偏移最小/汇回原轨), 导出修补"
                    "版参考到该 npz + 同名 .patch.json; 终点与其余全程逐帧不变")
p.add_argument("--ro_patchfile", default="",
               help="加载 .patch.json 重建 ramp (GUI 看修补后效果用)")
p.add_argument("--ro_approach", action="store_true",
               help="--root_only 编舞反放 = 任务开始前 Approach: 站姿→物外5cm点(全直)"
                    "→弯根部→沿射线进5cm→合拢成完整 GraspPose (一次 Enter 连播)")
p.add_argument("--shake", type=float, default=0.0,
               help="物理抓稳测试: 合拢→静置→抬升此高度(m, 建议 0.10)→腕±20°晃动→悬停。"
                    "只给 PD 目标不钉关节, 物体掉不掉见真章。需配 --row -1")
p.add_argument("--close", type=int, default=0,
               help="row=-1 时演示手指从伸直(GENERIC_OPEN)物理合拢到抓姿指型, 值=合拢帧数"
                    "(建议 120)。物理 PD 驱动不钉关节, 物体在场 —— 指扫到物体会被顶住/"
                    "推移, Δz/Δxy 逐段打印。与 --shake 互斥(shake 优先)")
p.add_argument("--close_to", choices=["grasp", "pose1"], default="grasp",
               help="--close 的合拢终点: grasp=一路合到抓形(旧行为) / pose1=只合到"
                    " pregrasp[0] 指型(拇指对掌)就停 —— 即 Pose0→'Pose0.5' 演示")
p.add_argument("--ladder", type=int, default=0,
               help="row=-1 时演示 Dexonomy pregrasp 指型阶梯: 手指按 row0(张)→row5→grasp"
                    " 7 个关键帧逐段物理合拢, 值=每段帧数(建议 40)。腕保持在真抓姿"
                    "(注意: 真实阶梯腕位也逐档后退, 此演示只播指型走廊)")
p.add_argument("--full", type=int, default=0,
               help="全程参考演示 (2026-08-19 定稿编舞): 站姿 60 帧摆 Pose1(拇指对掌)→"
                    "Pose1 保持沿 --traj 轨迹接近→(--insert_frames N: 插入腿上指型阶梯"
                    " row0→row5→到位合拢 | --full_to pose1: 停在 Pose0.5 不合拢)。"
                    "值=每行播几帧(建议 2)。物理 PD, 物体在场。需配 --traj")
p.add_argument("--full_to", choices=["grasp", "pose1"], default="grasp",
               help="--full 的指型终点: grasp=三段全走(旧行为) / pose1=构型完成后"
                    "全程保持 Pose1 指型, 不走阶梯不合拢 —— 即 Pose0(站姿)→Pose0.5 演示")
p.add_argument("--insert_frames", type=int, default=0,
               help="--full: 轨迹末尾这么多**行**是插入腿(1cm→GraspPose)。>0 时编舞改为: "
                    "站姿摆 Pose1 → 全程保持 → 指型阶梯只在插入腿展开 → 到位后合拢")
p.add_argument("--post_squeeze", type=int, default=0,
               help="--full 尾段追加: 合拢后 grasp→squeeze 再收紧这么多帧 (建议 40)")
p.add_argument("--post_lift", type=float, default=0.0,
               help="--full 尾段追加: squeeze 后双手腕竖直抬升 (m, 建议 0.02); "
                    "IK 逐帧链式, 从当前腕位插值到 GraspPose 腕位+抬升高")
p.add_argument("--export_bundle", default="",
               help="把 --full(+尾段) 编舞打包导出到该目录 (给 DP 训练): "
                    "traj_joint.npz(逐物理帧 64 关节目标+关节名+分段表) + "
                    "scene_isaac.json(物体网格/初始位姿/桌面/控制频率/摩擦)。"
                    "配 --headless 跑一遍即导出(仍会播放)")
p.add_argument("--post_carry", default="",
               help="--full 尾段再追加: 抬升后 20 帧 blend 接上该 carry npz 的 220 行"
                    "人手轨迹 (right_q/left_q), 手指保持 squeeze —— 全链演示: 接近→"
                    "合拢→squeeze→抬升→倒水轨迹")
p.add_argument("--post_carry_sub", type=int, default=3,
               help="--post_carry 每行播几帧 (3≈33s 全程)")
p.add_argument("--carry_probe", default="",
               help="零策略滑移探针 (2026-08-20): 双手生成在 GraspPose → 40 帧收紧到 "
                    "squeeze → 30 帧安顿拍滑移基线 → 双臂直写该 carry npz 的 220 行"
                    "人手轨迹, 全程打印双手腕系相对位姿漂移(=slip 口径), 终局判抓稳。"
                    "需配 --row -1")
p.add_argument("--probe_sub", type=int, default=12,
               help="--carry_probe 每行几个物理步。★12=与训练/视频同速(0.05s/行); "
                    "2 曾致 8 倍速播放, 加速度爆表把物体甩飞(2026-08-20 假阴性事故)")
p.add_argument("--traj29", default="",
               help="播放 29 维全参考 npz (right_q29/left_q29, 臂7+指22) —— FC-D 训练参考"
                    "文件本体, GUI 所见 = RL 所学。物理 PD, 物体在场, Δ 监控照旧")
p.add_argument("--traj29_sub", type=int, default=2, help="--traj29 每行播几帧")
p.add_argument("--traj", default="",
               help="非空=回放一条 npz 双臂关节参考 (right_q/left_q) 循环播放, "
                    "直写关节无 cuRobo (轻量 GUI, 看 leg2 短腿等)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pose_pregrasp")
app = AppLauncher(args).app

import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_env import BimanualApproachEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GENERIC_CLOSED, GENERIC_JOINT_ORDER, GENERIC_OPEN  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.direct_grasp_prob = 0.0
cfg.approach_t0_max = 0.0
cfg.retract_start = True   # 静态摆位不跑动力学; 只为过双臂防雷断言 (B 侧 q_ref 未重建)
cfg.prior_b_npz = os.path.abspath(args.prior_b)
cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = cfg.action_space, cfg.observation_space
cfg.action_space, cfg.observation_space = 2 * _a1, 2 * _o1
cfg._obs_single = _o1
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0

# 镜像摩擦 (2026-08-20, 与训练同款): 非交互侧指垫 SuperGrip + 副物体材质对齐 ——
# 否则查看器里左垫↔杯仍是 μ=0.1 冰面, 物理判读全部失真
cfg.extra_supergrip_bodies = [
    (n.replace("right_", "left_") if n.startswith("right_")
     else n.replace("left_", "right_")) for n in cfg.fingertip_bodies]
cfg.aux_grip_parity = True

# 左手 5 指垫传感器 (2026-08-21): v1 查看器原只有右侧。⚠不能追加进
# cfg.contact_sensors —— 那是 _pad_signals 的数据源, 定长 5 (追加即观测炸,
# 本日实测)。改为查看器局部子类, 在 _setup_scene 里**另起一组**注册。
from isaaclab.sensors import ContactSensor as _CS  # noqa: E402
from isaaclab.sensors import ContactSensorCfg as _CSC  # noqa: E402


class _ViewerEnv(BimanualApproachEnv):
    def _setup_scene(self):
        super()._setup_scene()
        self._left_pad_sensors = []
        for _i, _n in enumerate(self.cfg.fingertip_bodies):
            _sc = _CSC(prim_path=("/World/envs/env_.*/Robot/"
                                  + _n.replace("right_", "left_")),
                       history_length=1,
                       filter_prim_paths_expr=["/World/envs/env_.*/Aux"])
            _s = _CS(_sc)
            self._left_pad_sensors.append(_s)
            self.scene.sensors[f"tipL_{_i}"] = _s


env = _ViewerEnv(cfg)
env.reset()

hand = env.hand
jn = list(hand.joint_names)
q = hand.data.default_joint_pos[0].clone()

_ro_anim, _ro_play, _ro_stage, _ro_retreat, _ro_squeeze, _ro_export = [], None, 0, {}, [], {}
for side_name, side, prior in ((env._A_name, env._A, args.grasp_prior),
                               (env._B_name, env._B, args.prior_b)):
    z = np.load(prior)
    row29 = np.asarray(z["grasp"], np.float64)
    # 裁定B3: row>=0 一律 = 掌心锚点 PreGrasp (env._pregrasp_w 只有这一档)
    with BM.use_side(env, side):
        pw = (env._grasp_pos_w.cpu().numpy(), env._grasp_quat_w.cpu().numpy()) \
            if args.row < 0 else env._pregrasp_w[0]
        anchor_T = env._anchor_T
    pos = np.asarray(pw[0], np.float64)
    quat = np.asarray(pw[1], np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    ik = ArmIK(side_name, anchor_link="arm_center", anchor_T=anchor_T)
    pfx = "R" if side_name == "right" else "L"
    arm_ids = [jn.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
    r = ik.solve(pos, quat_to_R(quat), q0=q[arm_ids].cpu().numpy().astype(np.float64),
                 iters=300)
    print(f"[pose] {side_name}: IK {'✅' if r['ok'] else '❌'} "
          f"pos_err={r.get('pos_err', float('nan'))*100:.2f}cm | 档 row={args.row} "
          f"| 腕位 {np.round(pos, 3)}")
    for i, jid in enumerate(arm_ids):
        q[jid] = float(r["q"][i])
    # 手指 (裁定B2, 2026-08-18): 到位闩前手指**全程伸直** —— pregrasp 档显示张开
    # 指型(GENERIC_OPEN), 只有 row=-1 (GraspPose) 才显示数据指型。npz/GENERIC 序
    # 逐名配对写入, 不依赖 USD 序。
    fin = row29[7:29] if args.row < 0 else GENERIC_OPEN
    if args.row < 0 and args.tighten > 0:
        # 收紧: 抓姿指型 → 合拢锚点, 按比例插值 —— 0.2 = 往里收 20%。
        # 锚点優先 close_anchor (Pour17 两个先验都没有), 缺省用 GENERIC_CLOSED
        # (env 里 q_close 的同款模板)。GENERIC 序逐位对应, 与写入循环同序。
        _cl = (np.asarray(z["close_anchor"], np.float64)
               if "close_anchor" in z.files else np.asarray(GENERIC_CLOSED, np.float64))
        fin = row29[7:29] + args.tighten * (_cl - row29[7:29])
        print(f"[pose] {side_name}: 手指收紧 {args.tighten*100:.0f}% "
              f"(锚点={'close_anchor' if 'close_anchor' in z.files else 'GENERIC_CLOSED'})")
    if args.row < 0 and args.root_only:
        # 只留根部弯曲: 拇指 CMC_FE/AA + 四指 MCP_FE/AA + 小指 CMC 保持抓形值,
        # 其余 (拇指 MCP_FE/AA/IP, 四指 PIP/DIP) 伸直到 GENERIC_OPEN
        _ROOT = ("thumb_CMC_FE", "thumb_CMC_AA", "index_MCP_FE", "index_MCP_AA",
                 "middle_MCP_FE", "middle_MCP_AA", "ring_MCP_FE", "ring_MCP_AA",
                 "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA")
        _full_fin = fin.copy()
        # 拇指根部例外 (2026-08-21 thumbfix 撞瓶案): 接近型的拇指根取 Pose1
        # (pregrasp[0]) 值 —— 深弯拇指只在最后合拢段展开, 接近段保持已验证的净空
        _p1_22 = np.asarray(z["pregrasp"][0], np.float64)[7:29]
        _ro_fin = np.array([
            (float(p1) if "thumb_CMC" in nm else v)
            if any(r in nm for r in _ROOT) else float(o)
            for nm, v, o, p1 in zip(GENERIC_JOINT_ORDER, fin,
                                    GENERIC_OPEN, _p1_22)])
        # 编舞 (2026-08-21 用户再更正): 起手=完整 GraspPose; Enter① 伸直(留根部)
        # → root_only; Enter② **全部伸直(根部也伸)** —— 验证"贴物全张开"是否零碰物
        # (动画期间逐帧打印物体 Δ)。三元组: (关节id, root_only值, 模板值, 全开值)
        for nm, rv, fv, ov in zip(GENERIC_JOINT_ORDER, _ro_fin, _full_fin,
                                  np.asarray(GENERIC_OPEN, np.float64)):
            _ro_anim.append((jn.index(nm.replace("right_", f"{side_name}_")),
                             float(rv), float(fv), float(ov)))
        # 阶段② 前置后退腿 (2026-08-21 用户点单): 沿"物心→腕"射线退 5cm, 逐帧 IK
        with BM.use_side(env, side):
            _op0 = env.obj_init_pos.cpu().numpy().astype(np.float64)
        _dir = pos - _op0
        _dir = _dir / max(np.linalg.norm(_dir), 1e-9)
        def _build_ramp(_ikh, _pos, _R, _ray, _seed0, _patch=None):
            """advance 段 40 行 IK ramp; _patch=(k_lo,k_hi,dvec,delta) 时在窗内叠加
            平滑凸包偏移 (两端为零 ⟹ 从原轨分出又汇回, 终点/其余全程不变)。"""
            _sd_, _rp_, _err = _seed0.copy(), [], 0.0
            for _k in range(40):
                _t3 = _pos + _ray * 0.05 * ((_k + 1) / 40.0)
                if _patch is not None:
                    _kl, _kh, _dv, _de = _patch
                    if _kl <= _k <= _kh:
                        _u3 = (_k - _kl + 1) / (_kh - _kl + 2)
                        _t3 = _t3 + np.asarray(_dv) * _de * float(
                            np.sin(np.pi * _u3) ** 2)
                # ⚠ pos_tol 默认 5e-3(5mm) —— 与我们要修的碰撞同数量级, 等于给
                #   advance 每帧埋 5mm 腕位误差。收到 0.2mm。
                _r3 = _ikh.solve(_t3, _R, q0=_sd_, iters=200,
                                 pos_tol=2e-4, rot_tol=5e-3)
                _sd_ = _r3["q"]
                _err = max(_err, float(_r3.get("pos_err", 0.0)))
                _rp_.append(_sd_.copy())
            return np.stack(_rp_), _err

        _patch0 = None
        if args.ro_patchfile:
            import json as _json
            _pj = _json.loads(open(args.ro_patchfile).read())
            if side_name in _pj:
                _e = _pj[side_name]
                _patch0 = (int(_e["k_lo"]), int(_e["k_hi"]),
                           np.asarray(_e["dvec"], np.float64), float(_e["delta"]))
                print(f"[ro_patch] {side_name}: 载入补丁 行[{_e['k_lo']},{_e['k_hi']}] "
                      f"偏移 {np.round(_e['dvec'],2)}×{_e['delta']*1000:.1f}mm")
        _ramp, _ikerr0 = _build_ramp(ik, pos, quat_to_R(quat), _dir,
                                     r["q"].copy(), _patch0)
        _ramp = list(_ramp)
        # ===== 侧移接近 (2026-08-24 用户裁定) =====
        # 现象: advance 前推时**中指根部区域**蹭到物体(GUI 实证 + 几何审计
        #   主贴 middle_MP/ring_MP)。λ 扫描已证伪"弯浅一点"(λ=0 手指全张开照样碰),
        #   ⟹ 不是指型问题, 是**路径贴着物体表面走**。
        # 方案: 整条接近路径**侧向平移** ro_side_cm 沿"GraspPose 掌心向外",
        #   推进行程**不变**(仍 5cm); advance 结束后新增一段**侧向移回**,
        #   落到原 advance 终点。示意: (1,1)→(1,6) 改成 (1.5,1)→(1.5,6)→(1,6)。
        # 掌心向外 = −normalize(R(腕四元数)·anchor_local), anchor_local =
        #   五指垫合拢质心在腕系的位置(URDF 正解, 非手调常数), 指向掌心向内。
        _side_cm = float(getattr(args, "ro_side_cm", 0.0))
        _ins = []
        if _side_cm > 0.0:
            with BM.use_side(env, side):
                _al = env.anchor_local.cpu().numpy().astype(np.float64)
            _pout = -(quat_to_R(quat) @ _al)
            _pout = _pout / max(np.linalg.norm(_pout), 1e-9)
            _off = _pout * (_side_cm / 100.0)
            print(f"[ro_side] {side_name}: 掌心向外 {np.round(_pout,3)} "
                  f"| 侧移 {_side_cm}cm | anchor_local {np.round(_al*100,2)}cm")
            # 侧移版 advance: 每帧腕位 += _off, 逐帧 IK (行程不变)
            _sd9, _rp9, _e9 = r["q"].copy(), [], 0.0
            for _k in range(40):
                _t9 = pos + _dir * 0.05 * ((_k + 1) / 40.0) + _off
                _r9 = ik.solve(_t9, quat_to_R(quat), q0=_sd9, iters=200,
                               pos_tol=2e-4, rot_tol=5e-3)
                _sd9 = _r9["q"]; _e9 = max(_e9, float(_r9.get("pos_err", 0.0)))
                _rp9.append(_sd9.copy())
            # 插入段: 从侧移后的 advance 终点 → 原 advance 终点 (纯侧向 0.5cm)
            # ⚠ 必须**从原终点反向生成再反转** —— 7 自由度臂同一腕位有多组零空间解,
            #   若从侧移解正向链过去, 末帧会落在另一分支(实测差 0.40°/0.18°),
            #   与后面 hold2 的 grasp_arm 不同源, 白白引入一个跳变。
            _N9 = int(getattr(args, "ro_insert_frames", 20))
            _p_end = pos + _dir * 0.05 * (1.0 / 40.0)      # 原 advance 终点腕位
            _sd10, _back10 = _ramp[0].copy(), []           # 种子 = 原 ramp[0] 解
            for _k in range(_N9):
                _u10 = _k / max(_N9 - 1, 1)                # 0 → 1
                _s10 = _u10 * _u10 * (3 - 2 * _u10)
                _t10 = _p_end + _off * _s10                # 原终点 → 侧移终点
                _r10 = ik.solve(_t10, quat_to_R(quat), q0=_sd10, iters=200,
                                pos_tol=2e-4, rot_tol=5e-3)
                _sd10 = _r10["q"]; _back10.append(_sd10.copy())
            _rp10 = _back10[::-1]                          # 反转 ⟹ 末帧 = 原解
            _ramp = list(np.stack(_rp9))
            _ins = list(np.stack(_rp10))
            print(f"[ro_side] {side_name}: 侧移 ramp IK 误差 {_e9*100:.2f}cm | "
                  f"插入段 {_N9} 帧")
            _dbg=np.degrees(np.abs(np.stack(_rp10)[-1]-np.stack(_rp10)[0])).max()
            print(f"[ro_side] {side_name}: 插入段行程 {_dbg:.4f}° | "
                  f"首帧靶 {np.round((_p_end+_off*1.0)*100,3)} 末帧靶 {np.round(_p_end*100,3)}cm")
        _s22 = (np.asarray(z["squeeze"], np.float64)[7:29]
                if "squeeze" in z.files else _full_fin)
        for nm, fv, sv in zip(GENERIC_JOINT_ORDER, _full_fin, _s22):
            _ro_squeeze.append((jn.index(nm.replace("right_", f"{side_name}_")),
                                float(fv), float(sv)))
        _ro_retreat[side_name] = (arm_ids, np.stack(_ramp),
                                  hand.data.default_joint_pos[0].cpu().numpy()[arm_ids])
        _ro_export[side_name] = dict(
            arm_ids=arm_ids, stance=_ro_retreat[side_name][2].copy(),
            ramp=np.stack(_ramp), grasp_arm=np.asarray(r["q"], np.float64).copy(),
            fin_open=np.asarray(GENERIC_OPEN, np.float64).copy(),
            fin_ro=_ro_fin.copy(), fin_grasp=_full_fin.copy(),
            fin_squeeze=_s22.copy(), insert=(np.stack(_ins) if _ins else None),
            # 修补器素材 (2026-08-22): IK 句柄/抓姿位姿/射线, 供 --patch_ro 重建 ramp
            _ik=ik, _pos=pos.copy(), _R=quat_to_R(quat), _ray=_dir.copy(),
            _seed=r["q"].copy(), _build=_build_ramp)
        print(f"[pose] {side_name}: 起手=GraspPose | Enter①伸直(留根部) "
              f"| Enter②退5cm(射线 {np.round(_dir,2)})→全伸直(验证零碰物)")
    for name, val in zip(GENERIC_JOINT_ORDER, fin):
        q[jn.index(name.replace("right_", f"{side_name}_"))] = float(val)

if (args.patch_ro or args.ro_sweep) and _ro_export:
    # ---- 碰撞局部修补器 (2026-08-22 用户裁定) --------------------------------
    # 语义: 完全照参考走; 首碰帧 tc 处回退, 在 advance 段窗口内加最小平滑偏移绕行,
    # 绕过后汇回原轨 (终点/其余全程逐帧不变); 候选按 (窗口最小, 偏移最小) 序取首个
    # 通过者。合拢段(≥240帧)的接触是任务本身, 不算碰撞。
    _pdt = env.sim.get_physics_dt()
    _q0h = hand.data.default_joint_pos[0].clone()
    _sides = (("right", env._A), ("left", env._B))
    # ⚠ 物体快照必须在 sim.reset + 静置**之后**拍 —— reset 前的 data 缓冲是脏的
    # (2026-08-22 实测: 早拍 ⟹ reset 后物体回 USD 默认位, 帧60 被臂扫飞 10cm)
    _o0 = {}
    for _tg, _sd in _sides:
        with BM.use_side(env, _sd):
            _o0[_tg] = (env.obj_init_pos.clone()
                        + env.scene.env_origins[0],
                        env.obj_init_quat.clone().to(torch.float32))
    try:
        env.sim.reset()                     # 主循环之前, 物理需先初始化一次
    except Exception:
        pass

    def _reset_pr():
        for _tg2, _sd2 in _sides:
            with BM.use_side(env, _sd2):
                _pz = torch.cat([_o0[_tg2][0], _o0[_tg2][1]]).unsqueeze(0)
                env.object.write_root_pose_to_sim(_pz)
                env.object.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=env.device))
        hand.write_joint_state_to_sim(_q0h.unsqueeze(0),
                                      torch.zeros_like(_q0h).unsqueeze(0))
        hand.set_joint_position_target(_q0h.unsqueeze(0))
        hand.write_data_to_sim()
        for _ in range(12):
            env.sim.step(render=True); env.scene.update(_pdt)
        _ref = {}                       # 检测基准 = 静置后实测 (数学位姿差着
        for _tg2, _sd2 in _sides:       # 毫米级物理沉降, 会误报 —— r2 实测 2.4mm)
            with BM.use_side(env, _sd2):
                _ref[_tg2] = (env.object.data.root_pos_w[0].clone(),
                              env.object.data.root_quat_w[0].clone())
        return _ref

    _RO_LAM = [1.0]          # 指尖包络扫描用的可变容器 (λ: root_bend 深度)

    def _fq(_t, _ramps):
        """ro_approach 帧→全关节目标 (与交互连播 827-860 行同数学)。"""
        _qq = _q0h.clone()
        for _sn3, _dd in _ro_export.items():
            _aid = _dd["arm_ids"]
            _rmp = _ramps[_sn3]; _st3 = _dd["stance"]
            if _t < 100:
                _al = _t / 99.0; _s3 = _al * _al * (3 - 2 * _al)
                for _i4, _j4 in enumerate(_aid):
                    _qq[_j4] = float(_st3[_i4]) + _s3 * (
                        float(_rmp[-1, _i4]) - float(_st3[_i4]))
            elif _t < 180:
                for _i4, _j4 in enumerate(_aid):
                    _qq[_j4] = float(_rmp[-1, _i4])
            elif _t < 220:
                _k4 = 39 - (_t - 180)
                for _i4, _j4 in enumerate(_aid):
                    _qq[_j4] = float(_rmp[max(_k4, 0), _i4])
            else:
                for _i4, _j4 in enumerate(_aid):
                    _qq[_j4] = float(_dd["grasp_arm"][_i4])
        for _j4, _rv4o, _fv4, _ov4 in _ro_anim:
            # 指尖包络: root_bend 的目标位形按 λ 向"全张开"回退。
            # λ=1 = 原编舞(根弯尖直 = 前伸最长); λ<1 = 弯浅 ⟹ 包络收回
            _rv4 = _ov4 + _RO_LAM[0] * (_rv4o - _ov4)
            if _t < 120:
                _qq[_j4] = _ov4
            elif _t < 180:
                _al = (_t - 120) / 59.0; _s3 = _al * _al * (3 - 2 * _al)
                _qq[_j4] = _ov4 + _s3 * (_rv4 - _ov4)
            elif _t < 240:
                _qq[_j4] = _rv4
            elif _t < 300:
                _al = (_t - 240) / 59.0; _s3 = _al * _al * (3 - 2 * _al)
                _qq[_j4] = _rv4 + _s3 * (_fv4 - _rv4)
        if _t >= 300:
            for _j4, _fv4, _sv4 in _ro_squeeze:
                _al = min((_t - 300) / 39.0, 1.0)
                _s3 = _al * _al * (3 - 2 * _al)
                _qq[_j4] = _fv4 + _s3 * (_sv4 - _fv4)
        return _qq

    def _qang(_qa, _qb):
        _d5 = float(torch.abs(torch.sum(_qa * _qb)).clamp(max=1.0))
        return float(np.degrees(2 * np.arccos(_d5)))

    def _play_pr(_ramps, _t_hi=240, _det_lo=60, _only=None):
        """回放 [0,_t_hi); 在 _det_lo 起检测碰撞 (物体位移>3mm 或转动>1.0°,
        基准=本次回放静置后实测位姿)。_only=侧名 ⟹ 只检测该侧 (候选验收用:
        被修侧必须自身全净, 另一侧的蹭留给下一轮 —— r4 教训: 交叉掩护误收)。"""
        _ref = _reset_pr()
        _first = None
        for _t in range(_t_hi):
            _qq = _fq(_t, _ramps)
            hand.set_joint_position_target(_qq.unsqueeze(0))
            hand.write_data_to_sim()
            env.sim.step(render=True); env.scene.update(_pdt)
            if _t >= _det_lo and _first is None:
                for _tg2, _sd2 in _sides:
                    if _only is not None and _tg2 != _only:
                        continue
                    with BM.use_side(env, _sd2):
                        _op = env.object.data.root_pos_w[0]
                        _oq = env.object.data.root_quat_w[0]
                    _dp5 = float(torch.norm(_op - _ref[_tg2][0])) * 1000
                    _dr5 = _qang(_oq, _ref[_tg2][1])
                    if _dp5 > 3.0 or _dr5 > 1.0:
                        _first = (_t, _tg2, _dp5, _dr5)
                        break
            if _first is not None:
                break
        return _first

    _RC = {_sn3: _dd["ramp"].copy() for _sn3, _dd in _ro_export.items()}

    # ===== 指尖包络扫描 (2026-08-24 用户点单) =====
    # 几何审计实测: advance 段主贴部位是 middle_MP / ring_MP / index_DP —— 是**指节**,
    # 不是掌背。机理: root_bend 把手指摆成"根弯尖直"(前伸最长的构型), 然后 advance
    # **按腕**沿射线前送 5cm, 没算上弯出去的指节多占的那一截 ⟹ 指节先于腕到达。
    # 这里对 root_bend 深度 λ 做扫描, 用**回放实测首次碰物**选参数, 不靠猜。
    if args.ro_sweep:
        _lams = [float(x) for x in args.ro_sweep.split(",")]
        print(f"\n[ro_sweep] 指尖包络扫描: λ = {_lams}", flush=True)
        print(f"[ro_sweep] λ=1 是当前编舞(根弯尖直); λ 越小 root_bend 弯得越浅\n")
        _best = None
        for _lam in _lams:
            _RO_LAM[0] = _lam
            _h = _play_pr(_RC, _t_hi=240, _det_lo=60)
            if _h is None:
                print(f"  λ={_lam:<5} ✅ 0-240 全程**零碰物**", flush=True)
                if _best is None:
                    _best = _lam        # 取最大的零碰 λ (弯得越深越接近原设计)
            else:
                _t9, _sd9, _dp9, _dr9 = _h
                _seg9 = ("stance" if _t9 < 100 else "hold1" if _t9 < 120
                         else "root_bend" if _t9 < 180 else "advance" if _t9 < 220
                         else "hold2")
                print(f"  λ={_lam:<5} ❌ 步{_t9:3d}({_seg9}) {_sd9} 手 "
                      f"位移{_dp9:.2f}mm 转{_dr9:.2f}°", flush=True)
        print(f"\n[ro_sweep] 最大的零碰 λ = {_best}"
              if _best is not None else
              "\n[ro_sweep] ⚠ 所有 λ 都碰 —— 问题不在 root_bend 深度, 需换思路",
              flush=True)
        _RO_LAM[0] = 1.0
        try:
            _slot.release()
        except Exception:
            pass
        os._exit(0)

    _patches = {}
    for _round in range(3):                     # 逐侧迭代: 修完一侧再修下一侧
        print(f"[patch_ro] 轮{_round+1}: 回放检测 (0-240) ...", flush=True)
        _hit0 = _play_pr(_RC)
        if _hit0 is None:
            break
        _tc, _cs, _dp0, _dr0 = _hit0
        print(f"[patch_ro] ★ 首碰: 帧{_tc} 侧={_cs} 位移{_dp0:.1f}mm "
              f"转动{_dr0:.2f}°", flush=True)
        assert 180 <= _tc < 240, \
            f"首碰帧 {_tc} 不在 advance 窗 [180,240) —— 臂补丁治不了, 需人工判"
        # 同侧允许重修 (整窗重搜, 新补丁整体替换旧补丁)
        _kc = max(39 - min(_tc - 180, 39), 0)   # 撞击时的 ramp 行
        _ray = _ro_export[_cs]["_ray"]
        _up = np.array([0.0, 0.0, 1.0])
        _e1 = np.cross(_ray, _up); _e1 /= max(np.linalg.norm(_e1), 1e-9)
        _e2 = np.cross(_ray, _e1); _e2 /= max(np.linalg.norm(_e2), 1e-9)
        _found = None
        _diag = [(_e1 + _up) / np.sqrt(2), (-_e1 + _up) / np.sqrt(2),
                 (_e2 + _up) / np.sqrt(2), (-_e2 + _up) / np.sqrt(2)]
        for _hw in (4, 7, 11, 16, 25):          # 窗口半宽: 最小优先
            _kl, _kh = max(_kc - _hw, 0), min(_kc + _hw, 39)
            print(f"[patch_ro]   搜窗 [{_kl},{_kh}] ...", flush=True)
            for _de in (0.002, 0.004, 0.008, 0.012, 0.016):  # 偏移: 最小优先
                for _dv in [_e1, -_e1, _e2, -_e2, _up] + _diag:
                    _dd2 = _ro_export[_cs]
                    _rp2, _ike = _dd2["_build"](
                        _dd2["_ik"], _dd2["_pos"], _dd2["_R"], _ray,
                        _dd2["_seed"].copy(), (_kl, _kh, _dv, _de))
                    if _ike > 0.005:
                        continue
                    _RT = dict(_RC); _RT[_cs] = _rp2
                    # 接受判据: **被修侧自身**全程干净 (另一侧留下一轮)
                    if _play_pr(_RT, _only=_cs) is None:
                        _found = (_kl, _kh, _dv, _de, _rp2)
                        break
                if _found: break
            if _found: break
        assert _found, f"[patch_ro] {_cs} 全候选失败 —— 扩大搜索或人工修"
        _kl, _kh, _dv, _de, _rp2 = _found
        print(f"[patch_ro] ✅ 补丁: {_cs} ramp行[{_kl},{_kh}] "
              f"方向{np.round(_dv, 2)} 峰值{_de*1000:.1f}mm", flush=True)
        _RC[_cs] = _rp2
        _patches[_cs] = dict(k_lo=int(_kl), k_hi=int(_kh),
                             dvec=[float(x) for x in _dv], delta=float(_de),
                             src_hit_frame=int(_tc))
    _hitf = _play_pr(_RC)
    assert _hitf is None, f"[patch_ro] 三轮后仍碰: {_hitf}"
    print("[patch_ro] ② 全程验证 (0-340, 含合拢+squeeze) ...", flush=True)
    _hitv = _play_pr(_RC, _t_hi=340, _det_lo=60)
    if _hitv is not None and _hitv[0] < 240:
        raise AssertionError(f"[patch_ro] 验证期合拢前又碰: {_hitv}")
    for _cs2, _rpx in _RC.items():
        _ro_export[_cs2]["ramp"] = _rpx
    if _patches:
        import json as _json2
        _pjf = args.patch_ro.replace(".npz", "") + ".patch.json"
        open(_pjf, "w").write(_json2.dumps(_patches, indent=1))
        print(f"[patch_ro] 补丁参数已存 {_pjf} ({len(_patches)} 侧)", flush=True)
    else:
        print("[patch_ro] ✅ 基线零碰 —— 按原样导出", flush=True)
    args.export_ro = args.patch_ro           # 借用下方 export 块写修补版 npz

if args.export_ro and _ro_export:
    # AAG 参考导出 (2026-08-21 用户裁定"标准成功轨迹当参考"): 360 行 @20Hz
    # 段表: 站姿→5cm(100) 稳(20) 弯根部(60) 进5cm(40) 稳(20) 合拢(60) squeeze(40) 稳(20)
    _has_ins = any(_d.get("insert") is not None for _d in _ro_export.values())
    _NI = int(len(next(iter(_ro_export.values()))["insert"])) if _has_ins else 0
    _SEG = ([("stance_to_5cm", 100), ("hold1", 20), ("root_bend", 60),
             ("advance", 40), ("insert", _NI), ("hold2", 20), ("close", 60),
             ("squeeze", 40), ("hold3", 20)] if _has_ins else
            [("stance_to_5cm", 100), ("hold1", 20), ("root_bend", 60),
             ("advance", 40), ("hold2", 20), ("close", 60),
             ("squeeze", 40), ("hold3", 20)])
    _T = sum(n for _, n in _SEG)
    _out = {}
    def _ss(n):
        _a = np.linspace(0, 1, n)
        return _a * _a * (3 - 2 * _a)
    for _sn2, _d in _ro_export.items():
        arm = np.zeros((_T, 7)); fin = np.zeros((_T, 22))
        t = 0
        w = _ss(100)[:, None]
        arm[t:t+100] = (1-w)*_d["stance"][None] + w*_d["ramp"][-1][None];         fin[t:t+100] = _d["fin_open"][None]; t += 100
        arm[t:t+20] = _d["ramp"][-1]; fin[t:t+20] = _d["fin_open"]; t += 20
        w = _ss(60)[:, None]
        arm[t:t+60] = _d["ramp"][-1];         fin[t:t+60] = (1-w)*_d["fin_open"][None] + w*_d["fin_ro"][None]; t += 60
        arm[t:t+40] = _d["ramp"][::-1]; fin[t:t+40] = _d["fin_ro"]; t += 40
        if _has_ins:                       # 侧向移回段: 手指保持 fin_ro 不动
            arm[t:t+_NI] = _d["insert"]; fin[t:t+_NI] = _d["fin_ro"]; t += _NI
        arm[t:t+20] = _d["grasp_arm"]; fin[t:t+20] = _d["fin_ro"]; t += 20
        w = _ss(60)[:, None]
        arm[t:t+60] = _d["grasp_arm"];         fin[t:t+60] = (1-w)*_d["fin_ro"][None] + w*_d["fin_grasp"][None]; t += 60
        w = _ss(40)[:, None]
        arm[t:t+40] = _d["grasp_arm"];         fin[t:t+40] = (1-w)*_d["fin_grasp"][None] + w*_d["fin_squeeze"][None]; t += 40
        arm[t:t+20] = _d["grasp_arm"]; fin[t:t+20] = _d["fin_squeeze"]; t += 20
        _out[f"{_sn2}_q"] = arm.astype(np.float32)
        _out[f"{_sn2}_q29"] = np.concatenate([arm, fin], 1).astype(np.float32)
    _out["seg_names"] = np.array([x for x, _ in _SEG])
    _out["seg_lens"] = np.array([n for _, n in _SEG])
    _out["grasp_row"] = np.array(220 + _NI)   # 腕到抓姿的行号 (insert 段末)
    _out["source"] = np.array("ro_approach 六段编舞 thumbfix v2 (2026-08-21 AAG 参考)")
    np.savez(args.export_ro, **_out)
    _sd = max(float(np.degrees(np.abs(np.diff(_out[k], axis=0)).max()))
              for k in _out if k.endswith("_q29"))
    print(f"[export_ro] ✅ {args.export_ro} | {_T} 行 | 全局最大步差 {_sd:.2f}°",
          flush=True)
    try:
        _slot.release()
    except Exception:
        pass
    os._exit(0)

if args.row < 0 and args.root_only and args.ro_approach:
    q = hand.data.default_joint_pos[0].clone()   # Approach 模式: 起手=对称站姿
    print("[pose] ro_approach: 起手=站姿, Enter 连播 Approach 编舞")

# ---- 对账: 场景里物体的**实际**位姿 vs 靶点数学用的位姿 ----
W = env.scene.env_origins[0].cpu().numpy()
for tag, side in (("A/主物", env._A), ("B/副物", env._B)):
    with BM.use_side(env, side):
        actual_p = env.object.data.root_pos_w[0].cpu().numpy() - W
        actual_q = env.object.data.root_quat_w[0].cpu().numpy()
        math_p = env.obj_init_pos.cpu().numpy()
        gp = env._grasp_pos_w.cpu().numpy()
    print(f"[对账] {tag}: 实际位置 {np.round(actual_p, 3)} | 数学 obj_init "
          f"{np.round(math_p, 3)} | Δ={np.linalg.norm(actual_p - math_p)*100:.2f}cm")
    print(f"[对账] {tag}: 实际朝向 {np.round(actual_q, 3)} | 抓姿靶点(世界) "
          f"{np.round(gp, 3)} | 腕-物距 {np.linalg.norm(gp - actual_p)*100:.2f}cm")

if args.headless and not (args.carry_probe or args.full or args.close
                           or args.ladder or args.traj29):
    # 一步 FK 后量指尖-物体距离 (裁定B2 伸直指型下的真实间隙)
    _qb = q.unsqueeze(0)
    hand.write_joint_state_to_sim(_qb, torch.zeros_like(_qb))
    hand.set_joint_position_target(_qb)
    hand.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.sim.get_physics_dt())
    bn = list(hand.body_names)
    for side_name, side in ((env._A_name, env._A), (env._B_name, env._B)):
        with BM.use_side(env, side):
            op = env.object.data.root_pos_w[0]
        tips = [i for i, n in enumerate(bn)
                if n.startswith(f"{side_name}_") and n.endswith("_DP")]
        tp = hand.data.body_pos_w[0, tips]
        d3 = (tp - op).norm(dim=1) * 100
        dxy = (tp[:, :2] - op[:2]).norm(dim=1) * 100
        print(f"[指距] {side_name}: 指尖到物体中心 3D min={float(d3.min()):.1f}cm / "
              f"水平 min={float(dxy.min()):.1f}cm (指尖体={len(tips)}个)")
    print("[pose] headless 自检模式, 不渲染, 退出")
    try:
        _slot.release()
    except Exception:
        pass
    os._exit(0)      # 硬退 (2026-08-20: app.close() 挂死占槽 30 分钟案, 三入口同修)

_traj_R = _traj_L = None
if args.traj:
    _z2 = np.load(args.traj)
    _traj_R = np.asarray(_z2["right_q"], np.float32)
    _traj_L = np.asarray(_z2["left_q"], np.float32)
    _aidR = [jn.index(f"R_arm_j{i}") for i in range(1, 8)]
    _aidL = [jn.index(f"L_arm_j{i}") for i in range(1, 8)]
    # 手指动画: 帧0=伸直(GENERIC_OPEN) -> 末帧=抓姿指型, 随插入进度线性合拢
    _fin_anim = []      # [(jid, open_val, grasp_val)]
    for _sn, _prior in (("right", args.grasp_prior), ("left", args.prior_b)):
        _zf = np.load(_prior)
        for _nm, _ov, _gv in zip(GENERIC_JOINT_ORDER, GENERIC_OPEN,
                                 np.asarray(_zf["grasp"], np.float64)[7:29]):
            _fin_anim.append((jn.index(_nm.replace("right_", f"{_sn}_")),
                              float(_ov), float(_gv)))
    print(f"[pose] 轨迹回放模式: {len(_traj_R)} 帧循环 (臂=规划轨迹, "
          f"指=伸直→抓姿随进度合拢)")

# ---------------- 物理抓稳测试 (--shake): 预算整条关节目标带 ----------------
_shake_q = None
if args.shake > 0:
    assert args.row < 0, "--shake 需配 --row -1 (从真抓姿开始)"
    from rl_rebuild.correction.kinematics import quat_to_R as _q2R
    _T_close, _T_settle, _T_lift, _T_wig, _T_hold = 40, 60, 60, 120, 90
    _T = _T_close + _T_settle + _T_lift + _T_wig + _T_hold
    _rows = np.tile(q.cpu().numpy()[None], (_T, 1)).astype(np.float64)
    for side_name, prior in ((env._A_name, args.grasp_prior),
                             (env._B_name, args.prior_b)):
        zc = np.load(prior)
        r29 = np.asarray(zc["grasp"], np.float64)
        _cl2 = (np.asarray(zc["close_anchor"], np.float64)
                if "close_anchor" in zc.files else np.asarray(GENERIC_CLOSED, np.float64))
        pfx = "R" if side_name == "right" else "L"
        aid = [jn.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
        fid = [jn.index(nm.replace("right_", f"{side_name}_"))
               for nm in GENERIC_JOINT_ORDER]
        with BM.use_side(env, env._A if side_name == env._A_name else env._B):
            gp0 = env._grasp_pos_w.cpu().numpy().astype(np.float64)
            gq0 = env._grasp_quat_w.cpu().numpy().astype(np.float64)
            aT = env._anchor_T
        gq0 = gq0 / max(np.linalg.norm(gq0), 1e-12)
        ikS = ArmIK(side_name, anchor_link="arm_center", anchor_T=aT)
        seed = _rows[0, aid].copy()
        # ① 合拢段: 手指 抓形→收紧(tighten, 至少 0.15), 臂原地
        _tt = max(args.tighten, 0.15)
        for k in range(_T_close):
            al = (k + 1) / _T_close
            _rows[k, fid] = r29[7:29] + al * _tt * (_cl2 - r29[7:29])
        _rows[_T_close:, fid] = _rows[_T_close - 1, fid]      # 之后指型保持
        # ② 静置段: 全保持 (已由 tile 覆盖)
        # ③ 抬升段: 逐帧 IK, 种子链式
        t0 = _T_close + _T_settle
        for k in range(_T_lift):
            al = (k + 1) / _T_lift
            tgt = gp0 + np.array([0.0, 0.0, args.shake * al])
            rr = ikS.solve(tgt, _q2R(gq0), q0=seed, iters=120)
            seed = rr["q"]
            _rows[t0 + k, aid] = seed
        # ④ 晃动段: 顶端绕世界 z 轴 ±20° 两个正弦周期
        t1 = t0 + _T_lift
        top = gp0 + np.array([0.0, 0.0, args.shake])
        for k in range(_T_wig):
            th = np.radians(20.0) * np.sin(2 * np.pi * 2 * k / _T_wig)
            zq = np.array([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)])
            def _qm(a, b):
                w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
                return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                                 w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
            rr = ikS.solve(top, _q2R(_qm(zq, gq0)), q0=seed, iters=120)
            seed = rr["q"]
            _rows[t1 + k, aid] = seed
        _rows[t1 + _T_wig:, aid] = seed                      # ⑤ 悬停
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    print(f"[shake] 抓稳测试: 合拢{_T_close}(收紧{max(args.tighten,0.15)*100:.0f}%)→静置"
          f"{_T_settle}→抬{args.shake*100:.0f}cm/{_T_lift}帧→±20°晃{_T_wig}帧→悬停 | "
          f"通过判据: 物体抬升 ≥{args.shake*80:.0f}%cm 且不掉")

# ---------------- 合拢演示 (--close): 指伸直 -> 抓姿指型, 臂原地 ----------------
if args.close > 0 and _shake_q is None:
    assert args.row < 0, "--close 需配 --row -1 (在真抓姿位置合拢)"
    _T_c, _T_h = int(args.close), 90
    _rows = np.tile(q.cpu().numpy()[None], (_T_c + _T_h, 1)).astype(np.float64)
    for side_name, prior in ((env._A_name, args.grasp_prior),
                             (env._B_name, args.prior_b)):
        zc = np.load(prior)
        _tgt22 = (np.asarray(zc["pregrasp"][0], np.float64)[7:29]
                  if args.close_to == "pose1"
                  else np.asarray(zc["grasp"], np.float64)[7:29])
        fid = [jn.index(nm.replace("right_", f"{side_name}_"))
               for nm in GENERIC_JOINT_ORDER]
        _op = np.asarray(GENERIC_OPEN, np.float64)
        for k in range(_T_c):
            al = (k + 1) / _T_c
            _rows[k, fid] = _op + al * (_tgt22 - _op)
        _rows[_T_c:, fid] = _tgt22
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    print(f"[close] 合拢演示: 指伸直→{'Pose1指型(拇指对掌)' if args.close_to == 'pose1' else '抓姿'}"
          f" {_T_c} 帧 (物理 PD, 物体在场) → 保持 {_T_h} 帧")

# ------- Dexonomy 指型阶梯演示 (--ladder): row0(张)→...→row5→grasp 逐段合拢 -------
if args.ladder > 0 and _shake_q is None:
    assert args.row < 0, "--ladder 需配 --row -1 (在真抓姿位置播指型走廊)"
    _Ts, _T_h = int(args.ladder), 90
    _T = 6 * _Ts + _T_h
    _rows = np.tile(q.cpu().numpy()[None], (_T, 1)).astype(np.float64)
    for side_name, prior in ((env._A_name, args.grasp_prior),
                             (env._B_name, args.prior_b)):
        zc = np.load(prior)
        _keys = [np.asarray(zc["pregrasp"][i], np.float64)[7:29] for i in range(6)]
        _keys.append(np.asarray(zc["grasp"], np.float64)[7:29])
        fid = [jn.index(nm.replace("right_", f"{side_name}_"))
               for nm in GENERIC_JOINT_ORDER]
        for s in range(6):
            for k in range(_Ts):
                al = (k + 1) / _Ts
                _rows[s * _Ts + k, fid] = _keys[s] + al * (_keys[s + 1] - _keys[s])
        _rows[6 * _Ts:, fid] = _keys[6]
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    print(f"[ladder] Dexonomy 指型阶梯: 6 段 × {_Ts} 帧 (row0 张开→row5→grasp, "
          f"物理 PD) → 保持 {_T_h} 帧 | ⚠ 腕固定在抓姿, 真实阶梯腕位逐档后退")

# ---- 全程参考演示 (--full): cuRobo 臂轨迹 + 三段指型 Pose0→Pose1→阶梯→抓形 ----
if args.full > 0 and _shake_q is None:
    assert _traj_R is not None, "--full 需配 --traj (cuRobo 双臂关节轨迹 npz)"
    _S = int(args.full)
    _Ka = len(_traj_R)
    # pose1 模式: 臂动之前先给 60 帧"站姿摆构型"段 (Pose0→Pose1, 臂原地)
    _pre = 60 if (args.full_to == "pose1" or args.insert_frames > 0) else 0
    _T1, _Tc, _Th = _Ka * _S, 90, 90
    _T = _pre + _T1 + _Tc + _Th
    _rows = np.tile(q.cpu().numpy()[None], (_T, 1)).astype(np.float64)
    for t in range(_pre):                                 # 臂: 构型段原地(轨迹首行)
        for i, j in enumerate(_aidR):
            _rows[t, j] = float(_traj_R[0, i])
        for i, j in enumerate(_aidL):
            _rows[t, j] = float(_traj_L[0, i])
    for t in range(_T1):                                  # 臂: 逐行跟 cuRobo
        k = min(t // _S, _Ka - 1)
        for i, j in enumerate(_aidR):
            _rows[_pre + t, j] = float(_traj_R[k, i])
        for i, j in enumerate(_aidL):
            _rows[_pre + t, j] = float(_traj_L[k, i])
    for i, j in enumerate(_aidR):
        _rows[_pre + _T1:, j] = float(_traj_R[-1, i])
    for i, j in enumerate(_aidL):
        _rows[_pre + _T1:, j] = float(_traj_L[-1, i])
    assert args.full_to == "pose1" or args.insert_frames > 0, \
        "--full 定稿编舞只有两种: --full_to pose1 (停在 Pose0.5) 或 --insert_frames N " \
        "(全程到抓形)。中途成形路线已证伪删除 (2026-08-19: 60%/85%/插入位成形均撞物)"
    for side_name, prior in ((env._A_name, args.grasp_prior),
                             (env._B_name, args.prior_b)):
        zc = np.load(prior)
        keys = [np.asarray(zc["pregrasp"][i], np.float64)[7:29] for i in range(6)]
        keys.append(np.asarray(zc["grasp"], np.float64)[7:29])
        fid = [jn.index(nm.replace("right_", f"{side_name}_"))
               for nm in GENERIC_JOINT_ORDER]
        op = np.asarray(GENERIC_OPEN, np.float64)
        if args.full_to == "pose1":                       # 臂动前摆好 Pose1, 全程保持
            for t in range(_pre):
                al = (t + 1) / _pre
                _rows[t, fid] = op + al * (keys[0] - op)
            _rows[_pre:, fid] = keys[0]
            continue
        _ins = int(args.insert_frames) * _S
        if _ins > 0:      # 完整编舞: 站姿Pose1 → 保持 → 阶梯只铺插入腿 → 到位合拢
            for t in range(_pre):
                al = (t + 1) / _pre
                _rows[t, fid] = op + al * (keys[0] - op)
            _rows[_pre:_pre + _T1 - _ins, fid] = keys[0]
            seg = max(_ins // 5, 1)
            base = _pre + _T1 - _ins
            for s in range(5):
                for k2 in range(seg):
                    t = base + s * seg + k2
                    if t >= _pre + _T1:
                        break
                    al = (k2 + 1) / seg
                    _rows[t, fid] = keys[s] + al * (keys[s + 1] - keys[s])
            _rows[base + 5 * seg:_pre + _T1, fid] = keys[5]
            for k2 in range(_Tc):
                al = (k2 + 1) / _Tc
                _rows[_pre + _T1 + k2, fid] = keys[5] + al * (keys[6] - keys[5])
            _rows[_pre + _T1 + _Tc:, fid] = keys[6]
            continue
    if (int(args.post_squeeze) > 0 or float(args.post_lift) > 0.0) \
            and args.full_to != "pose1":
        # ---- 尾段追加 (2026-08-20 用户点单): grasp→squeeze 收紧 → 双手抬升 ----
        from rl_rebuild.correction.kinematics import quat_to_R as _q2Rx
        _Tsq = int(args.post_squeeze)
        _Tlift = 40 if float(args.post_lift) > 0 else 0
        _Th2 = 60
        _ext = np.tile(_rows[-1][None], (_Tsq + _Tlift + _Th2, 1))
        for side_name, prior in ((env._A_name, args.grasp_prior),
                                 (env._B_name, args.prior_b)):
            zc3 = np.load(prior)
            g22 = np.asarray(zc3["grasp"], np.float64)[7:29]
            s22 = (np.asarray(zc3["squeeze"], np.float64)[7:29]
                   if "squeeze" in zc3.files else g22)
            pfx = "R" if side_name == "right" else "L"
            aid = [jn.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
            fid = [jn.index(nm.replace("right_", f"{side_name}_"))
                   for nm in GENERIC_JOINT_ORDER]
            for k in range(_Tsq):
                al = (k + 1) / max(_Tsq, 1)
                _ext[k, fid] = g22 + al * (s22 - g22)
            if _Tsq > 0:
                _ext[_Tsq:, fid] = s22
            if _Tlift > 0:
                with BM.use_side(env, env._A if side_name == env._A_name
                                 else env._B):
                    gp0 = env._grasp_pos_w.cpu().numpy().astype(np.float64)
                    gq0 = env._grasp_quat_w.cpu().numpy().astype(np.float64)
                    aT3 = env._anchor_T
                gq0 = gq0 / max(np.linalg.norm(gq0), 1e-12)
                ik3 = ArmIK(side_name, anchor_link="arm_center", anchor_T=aT3)
                seed = _ext[0, aid].copy()
                w0 = ik3.fk(seed)[0]              # 当前腕位 (1cm 停点) 起插
                top = gp0 + np.array([0.0, 0.0, float(args.post_lift)])
                for k in range(_Tlift):
                    al = (k + 1) / _Tlift
                    tgt = w0 + al * (top - w0)
                    rr = ik3.solve(tgt, _q2Rx(gq0), q0=seed, iters=120)
                    seed = rr["q"]
                    _ext[_Tsq + k, aid] = seed
                _ext[_Tsq + _Tlift:, aid] = seed
        _rows = np.concatenate([_rows, _ext], 0)
        print(f"[full+] 尾段: squeeze {_Tsq} 帧 → 抬升 "
              f"{float(args.post_lift)*1000:.0f}mm ({_Tlift} 帧) → 悬停 {_Th2} 帧")
        if args.post_carry:
            # ---- 再接人手轨迹 (2026-08-20 用户点单): blend20 → carry 220 行 ----
            zcar = np.load(args.post_carry)
            _cs = max(int(args.post_carry_sub), 1)
            _cT = int(np.asarray(zcar["right_q"]).shape[0])
            _B2, _Th3 = 20, 60
            _car = np.tile(_rows[-1][None], (_B2 + _cT * _cs + _Th3, 1))
            for side_name in (env._A_name, env._B_name):
                pfx = "R" if side_name == "right" else "L"
                aid = [jn.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
                cq = np.asarray(zcar["right_q" if side_name == "right"
                                     else "left_q"], np.float64)
                a0 = _rows[-1, aid]
                for k in range(_B2):                     # 降回 carry 首行 (抓姿)
                    al = (k + 1) / _B2
                    al = al * al * (3 - 2 * al)
                    _car[k, aid] = a0 + al * (cq[0] - a0)
                for t in range(_cT):
                    for k in range(_cs):
                        _car[_B2 + t * _cs + k, aid] = cq[t]
                _car[_B2 + _cT * _cs:, aid] = cq[-1]
            _rows = np.concatenate([_rows, _car], 0)
            print(f"[full+] 再接人手轨迹: blend {_B2} 帧 → {_cT} 行 ×{_cs} 帧 "
                  f"→ 收尾 {_Th3} 帧 (手指保持 squeeze)")
    if args.export_bundle:
        # ---- DP 训练打包 (2026-08-20 用户点单): 逐帧关节目标 + Isaac 场景参数 ----
        import json as _json
        _bd = os.path.abspath(args.export_bundle)
        os.makedirs(_bd, exist_ok=True)
        _segs = {"stance_pose1": _pre, "approach": _T1, "close": _Tc, "hold": _Th}
        if int(args.post_squeeze) > 0 or float(args.post_lift) > 0:
            _segs.update({"squeeze": int(args.post_squeeze),
                          "lift": 40 if float(args.post_lift) > 0 else 0,
                          "hover": 60})
        np.savez(os.path.join(_bd, "traj_joint.npz"),
                 q_target=_rows.astype(np.float32),        # (T,64) 逐物理帧 PD 目标
                 joint_names=np.array(jn),
                 ctrl_hz=np.array(20.0), physics_frames_per_row=np.array(1),
                 seg_names=np.array(list(_segs.keys())),
                 seg_lens=np.array(list(_segs.values())),
                 arm_traj_src=np.array(os.path.abspath(args.traj)),
                 source=np.array("pose_pregrasp --full 定稿编舞 + squeeze/lift 尾段"))
        _scene = {"clip": args.clip, "ctrl_hz": 20.0,
                  "table": {"top_z": float(cfg.table_top_z),
                            "size": [float(x) for x in cfg.table_size]},
                  "objects": {}, "priors": {
                      "right": os.path.abspath(args.grasp_prior),
                      "left": os.path.abspath(args.prior_b),
                      "prior_yaw_right": float(args.prior_yaw),
                      "prior_yaw_left": float(args.prior_b_yaw)},
                  "friction": {"fingertip_pads": "SuperGrip 3.0/multiply",
                               "object": "3.0 (PhysX multiply 合成 9.0 有效)"},
                  "robot": "DexMate vega + SharpaWave 两侧 (RL_HAND_JOINTS=1, 29自由度/侧)"}
        for _tag, _side in (("right", env._A), ("left", env._B)):
            with BM.use_side(env, _side):
                _scene["objects"][_tag] = {
                    "init_pos": env.obj_init_pos.cpu().numpy().tolist(),
                    "init_quat": env.obj_init_quat.cpu().numpy().tolist()}
        _e2 = clips.clip_entry(args.clip)
        _scene["objects"]["right"]["mesh"] = os.path.abspath(_e2["mesh"])
        _sec2 = _e2.get("secondary") or {}
        if _sec2.get("mesh"):
            _scene["objects"]["left"]["mesh"] = os.path.abspath(_sec2["mesh"])
        with open(os.path.join(_bd, "scene_isaac.json"), "w") as _f:
            _json.dump(_scene, _f, indent=2, ensure_ascii=False)
        print(f"[export] ✅ DP 打包: {_bd}/traj_joint.npz + scene_isaac.json "
              f"(共 {_rows.shape[0]} 帧)")
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    print(f"[full] 全程参考 (定稿编舞): 站姿摆Pose1 {_pre} 帧 → 臂 {_Ka} 行 ×{_S} 帧"
          f" Pose1保持 | 终点={'Pose0.5(不合拢)' if args.full_to == 'pose1' else f'插入腿阶梯+合拢 {_Tc} 帧'}"
          f" → 保持 {_Th} 帧")

# ---- 零策略滑移探针 (--carry_probe): 抓姿出生→squeeze→跟人手轨迹, 量腕系滑移 ----
_probe = None
if args.carry_probe and _shake_q is None:
    assert args.row < 0, "--carry_probe 需配 --row -1 (从真抓姿出生)"
    _zc = np.load(args.carry_probe, allow_pickle=True)
    _cR = np.asarray(_zc["right_q"], np.float64)      # (220,7) 携带臂轨迹 (首行=抓姿)
    _cL = np.asarray(_zc["left_q"], np.float64)
    _PS = int(args.probe_sub)
    _SQ, _SET, _HLD = 40, 30, 60
    _T = _SQ + _SET + len(_cR) * _PS + _HLD
    _rows = np.tile(q.cpu().numpy()[None], (_T, 1)).astype(np.float64)
    _aidRp = [jn.index(f"R_arm_j{i}") for i in range(1, 8)]
    _aidLp = [jn.index(f"L_arm_j{i}") for i in range(1, 8)]
    for t in range(_T):                                # 臂: 前段停首行, 后段逐行
        k = 0 if t < _SQ + _SET else min((t - _SQ - _SET) // _PS, len(_cR) - 1)
        _rows[t, _aidRp] = _cR[k]
        _rows[t, _aidLp] = _cL[k]
    for side_name, prior in ((env._A_name, args.grasp_prior),
                             (env._B_name, args.prior_b)):
        zc2 = np.load(prior)
        g22 = np.asarray(zc2["grasp"], np.float64)[7:29]
        s22 = np.asarray(zc2["squeeze"], np.float64)[7:29]
        fid = [jn.index(nm.replace("right_", f"{side_name}_"))
               for nm in GENERIC_JOINT_ORDER]
        for t in range(_SQ):                           # 指: grasp→squeeze 微收紧
            al = (t + 1) / _SQ
            _rows[t, fid] = g22 + al * (s22 - g22)
        _rows[_SQ:, fid] = s22                         # 之后保持 squeeze
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    _bn_all = list(hand.body_names)
    _probe = {"base_idx": _SQ + _SET - 1, "rel0": {}, "smax": {"right": [0., 0.], "left": [0., 0.]},
              "ee": {"right": _bn_all.index("right_hand_C_MC"),
                     "left": _bn_all.index("left_hand_C_MC")}}
    print(f"[probe] 零策略滑移探针: squeeze {_SQ} 帧 → 安顿 {_SET} 帧(拍基线) → "
          f"人手轨迹 {len(_cR)} 行 ×{_PS} 帧 → 保持 {_HLD} 帧 | 判稳线 4cm/30°")

# ---- 29 维全参考播放 (--traj29): 臂+指全由文件驱动, 所见即所学 ----
if args.traj29 and _shake_q is None:
    _z29 = np.load(args.traj29, allow_pickle=True)
    _S29 = int(args.traj29_sub)
    _R29 = np.asarray(_z29["right_q29"], np.float64)
    _L29 = np.asarray(_z29["left_q29"], np.float64)
    assert len(_R29) == len(_L29), "两手行数必须一致"
    _T29, _Th29 = len(_R29) * _S29, 90
    _rows = np.tile(q.cpu().numpy()[None], (_T29 + _Th29, 1)).astype(np.float64)
    _aidR29 = [jn.index(f"R_arm_j{i}") for i in range(1, 8)]
    _aidL29 = [jn.index(f"L_arm_j{i}") for i in range(1, 8)]
    _fidR29 = [jn.index(nm) for nm in GENERIC_JOINT_ORDER]
    _fidL29 = [jn.index(nm.replace("right_", "left_")) for nm in GENERIC_JOINT_ORDER]
    for t in range(_T29):
        k = min(t // _S29, len(_R29) - 1)
        _rows[t, _aidR29] = _R29[k, :7]; _rows[t, _fidR29] = _R29[k, 7:29]
        _rows[t, _aidL29] = _L29[k, :7]; _rows[t, _fidL29] = _L29[k, 7:29]
    _rows[_T29:, _aidR29] = _R29[-1, :7]; _rows[_T29:, _fidR29] = _R29[-1, 7:29]
    _rows[_T29:, _aidL29] = _L29[-1, :7]; _rows[_T29:, _fidL29] = _L29[-1, 7:29]
    _shake_q = torch.tensor(_rows, dtype=torch.float32, device=q.device)
    _obj_z0 = {}
    for tag, side in (("right", env._A), ("left", env._B)):
        with BM.use_side(env, side):
            _obj_z0[tag] = env.object.data.root_pos_w[0].cpu().numpy().copy()
    _segs = list(_z29["seg_lens"]) if "seg_lens" in _z29.files else []
    print(f"[traj29] 全参考播放: {len(_R29)} 行 ×{_S29} 帧 | 段长 {_segs} → 保持 {_Th29} 帧")

qb = q.unsqueeze(0)
vb = torch.zeros_like(qb)
print("[pose] GUI 渲染中 (Ctrl+C 退出) ...")
dt = env.sim.get_physics_dt()
_fi = 0
while app.is_running():
    if _shake_q is not None:
        _k = min(_fi, len(_shake_q) - 1)
        qb = _shake_q[_k].unsqueeze(0)
        if _fi == 0:
            hand.write_joint_state_to_sim(qb, vb)            # 只在第 0 帧摆初始状态
        hand.set_joint_position_target(qb)
        hand.write_data_to_sim()
        env.sim.step(render=True)
        env.scene.update(dt)
        if _fi % 60 == 0 or _fi == len(_shake_q) - 1:
            _msg = []
            for tag, side in (("right", env._A), ("left", env._B)):
                with BM.use_side(env, side):
                    _d = env.object.data.root_pos_w[0].cpu().numpy() - _obj_z0[tag]
                _msg.append(f"{tag} 物体Δz {+_d[2]*100:5.1f}cm Δxy {np.linalg.norm(_d[:2])*100:4.1f}cm")
            _tag = ("shake" if args.shake > 0 else
                    ("ladder" if args.ladder > 0 else
                     ("full" if args.full > 0 else "close")))
            print(f"[{_tag}] 帧{_k:4d} | " + " | ".join(_msg), flush=True)
        if _fi == len(_shake_q) - 1 and args.shake > 0:
            for tag, side in (("right", env._A), ("left", env._B)):
                with BM.use_side(env, side):
                    _dz = float(env.object.data.root_pos_w[0, 2]) - _obj_z0[tag][2]
                print(f"[shake] ★ {tag}: {'✅ 抓稳' if _dz > 0.8*args.shake else '❌ 掉了/没抬起'}"
                      f" (Δz={_dz*100:.1f}cm / 目标 {args.shake*100:.0f}cm)")
        if _probe is not None and _fi >= _probe["base_idx"]:
            _msgp = []
            for tag, side in (("right", env._A), ("left", env._B)):
                _e = _probe["ee"][tag]
                _pw = hand.data.body_pos_w[0, _e].cpu().numpy()
                _qw = hand.data.body_quat_w[0, _e].cpu().numpy()
                with BM.use_side(env, side):
                    _po = env.object.data.root_pos_w[0].cpu().numpy()
                    _qo = env.object.data.root_quat_w[0].cpu().numpy()
                _Rw = quat_to_R(_qw)
                _prel = _Rw.T @ (_po - _pw)
                _Rrel = _Rw.T @ quat_to_R(_qo)
                if _fi == _probe["base_idx"]:
                    _probe["rel0"][tag] = (_prel, _Rrel)
                    continue
                _p0, _R0 = _probe["rel0"][tag]
                _dp = float(np.linalg.norm(_prel - _p0)) * 100
                _c = (np.trace(_R0.T @ _Rrel) - 1) / 2
                _dr = float(np.degrees(np.arccos(np.clip(_c, -1, 1))))
                _probe["smax"][tag][0] = max(_probe["smax"][tag][0], _dp)
                _probe["smax"][tag][1] = max(_probe["smax"][tag][1], _dr)
                _msgp.append(f"{tag} 滑移 {_dp:4.1f}cm/{_dr:4.1f}°")
            try:      # 右垫接触读数 (v1 查看器只有交互侧传感器, 左侧无——如实标注)
                _Fs = torch.cat([s_.data.force_matrix_w.view(1, 1, 3)
                                 for s_ in env._contact_sensors], dim=1)[0].nan_to_num(0.0)
                _mg = _Fs.norm(dim=-1)
                _msgp.append(f"右垫 {int((_mg > 0.5).sum())}/5 触 {float(_mg.sum()):.1f}N")
            except Exception:
                pass
            if _msgp and (_fi % 40 == 0 or _fi == len(_shake_q) - 1):
                print(f"[probe] 帧{_fi:4d} | " + " | ".join(_msgp), flush=True)
            if _fi == len(_shake_q) - 1:
                for tag in ("right", "left"):
                    _mp, _mr = _probe["smax"][tag]
                    _ok = _mp < 4.0 and _mr < 30.0
                    print(f"[probe] ★ {tag}: {'✅ 抓稳' if _ok else '❌ 滑了'} | "
                          f"全程最大滑移 {_mp:.1f}cm / {_mr:.1f}° (判稳线 4cm/30°)",
                          flush=True)
        if args.headless and _fi >= len(_shake_q) - 1:
            break                      # 无头跑带段: 播完即退 (探针/演示的批量口径)
        _fi += 1
        continue
    if _traj_R is not None:
        # 裁定 (2026-08-18晚): 参考只管臂 —— 手指全程伸直(GENERIC_OPEN),
        # 合拢是 RL 的领地, 回放不做手指动画。末帧停 20 帧再循环。
        _Ka = len(_traj_R)
        _k = min(_fi % (_Ka + 20), _Ka - 1)
        for _i, _j in enumerate(_aidR):
            q[_j] = float(_traj_R[_k, _i])
        for _i, _j in enumerate(_aidL):
            q[_j] = float(_traj_L[_k, _i])
        for _j, _ov, _gv in _fin_anim:
            q[_j] = _ov
        qb = q.unsqueeze(0)
        _fi += 1
    elif _ro_anim:
        # 单次 Enter 连播 (2026-08-21 定稿): GraspPose →60帧 直指(留根部) →20保持
        # →40帧 沿物心→腕射线退5cm →60帧 全伸直 →20保持 →100帧 平滑回站姿
        import select as _sel
        import sys as _sys
        if _ro_play is None:
            if _ro_stage == 2:
                # 播完定格: 纯 PD 保持最后目标, 不回落瞬移 (会再穿模)
                hand.set_joint_position_target(qb)
                hand.write_data_to_sim()
                env.sim.step(render=True)
                env.scene.update(dt)
                continue
            _r, _, _ = _sel.select([_sys.stdin], [], [], 0)
            if _r and _ro_stage == 0:
                _sys.stdin.readline()
                _ro_play, _ro_stage = 0, 1
                _ro_obj0 = {}
                for _tg, _sd in (("right", env._A), ("left", env._B)):
                    with BM.use_side(env, _sd):
                        _ro_obj0[_tg] = env.object.data.root_pos_w[0].cpu().numpy().copy()
                print("[root_only] ▶ 连播: " + (
                    "站姿→5cm点→弯根部→进5cm→合拢GraspPose"
                    if getattr(args, "ro_approach", False)
                    else "直指(留根部)→退5cm→全伸直→回站姿") + " (300帧)",
                    flush=True)
        elif getattr(args, "ro_approach", False):
            # Approach 连播 (回撤的逆序): 站姿→5cm点(全直)→弯根部→进5cm→合拢
            _t = _ro_play
            if _t < 100:                    # ①' 站姿 → 物外5cm点 (全直)
                _al = _t / 99.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _sn, (_aids, _rmp, _st) in _ro_retreat.items():
                    for _i2, _j2 in enumerate(_aids):
                        _a1 = float(_st[_i2])
                        q[_j2] = _a1 + _s2 * (float(_rmp[-1, _i2]) - _a1)
            elif _t < 120:
                pass                        # 保持
            elif _t < 180:                  # ②' 全直 → 弯根部 (root_only)
                _al = (_t - 120) / 59.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _j, _rv, _fv, _ov in _ro_anim:
                    q[_j] = _ov + _s2 * (_rv - _ov)
            elif _t < 220:                  # ③' 沿射线进 5cm (退坡倒放)
                _k3 = 39 - (_t - 180)
                for _sn, (_aids, _rmp, _st) in _ro_retreat.items():
                    for _i2, _j2 in enumerate(_aids):
                        q[_j2] = float(_rmp[max(_k3, 0), _i2])
            elif _t < 240:
                pass                        # 保持 (腕已在抓姿位, 指=根弯型)
            elif _t < 300:                  # ④' 合拢中/远端 → 完整 GraspPose
                _al = (_t - 240) / 59.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _j, _rv, _fv, _ov in _ro_anim:
                    q[_j] = _rv + _s2 * (_fv - _rv)
            elif _t < 340:                  # ⑤' 用力握紧: grasp → squeeze
                _al = (_t - 300) / 39.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _j, _fv, _sv in _ro_squeeze:
                    q[_j] = _fv + _s2 * (_sv - _fv)
            # 340-460: 保持 squeeze, 读垫压
            qb = q.unsqueeze(0)
            _ro_play += 1
            if _ro_play % 30 == 0 or (340 <= _ro_play and _ro_play % 20 == 0):
                _m = []
                for _tg, _sd in (("right", env._A), ("left", env._B)):
                    with BM.use_side(env, _sd):
                        _d = env.object.data.root_pos_w[0].cpu().numpy() - _ro_obj0[_tg]
                    _m.append(f"{_tg} Δ{np.linalg.norm(_d)*1000:.1f}mm")
                if _ro_play >= 320:
                    try:
                        _Fa = torch.cat(
                            [s_.data.force_matrix_w.view(1, 1, 3)
                             for s_ in (list(env._contact_sensors)
                                        + list(env._left_pad_sensors))],
                            dim=1)[0].nan_to_num(0.0)
                        _mg2 = _Fa.norm(dim=-1).cpu().numpy()
                        _fingers = ["拇", "食", "中", "无", "小"]
                        _rtxt = " ".join(f"{f}{v:.1f}" for f, v in
                                         zip(_fingers, _mg2[:5]))
                        _ltxt = " ".join(f"{f}{v:.1f}" for f, v in
                                         zip(_fingers, _mg2[5:10]))
                        _m.append(f"右垫N[{_rtxt}] 左垫N[{_ltxt}]")
                    except Exception as _e:
                        _m.append(f"垫压读取失败:{_e}")
                print(f"[ro_approach] 帧{_ro_play:3d} " + " | ".join(_m), flush=True)
            if _ro_play >= 460:
                _ro_play, _ro_stage = None, 2
                print("[ro_approach] ✅ 连播完成: Approach→握紧(squeeze)→垫压已打印, "
                      "定格", flush=True)
            # ★ 播放期纯 PD (shake 同款纪律): 不瞬移关节, 手指被物体顶住是真物理;
            #   每帧 write_joint_state 会把手指硬传送穿过物体 (2026-08-21 左手穿模实锤)
            hand.set_joint_position_target(qb)
            hand.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(dt)
            continue
        else:
            _t = _ro_play
            if _t < 60:                     # ① full → root_only
                _al = _t / 59.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _j, _rv, _fv, _ov in _ro_anim:
                    q[_j] = _fv + _s2 * (_rv - _fv)
            elif _t < 80:
                pass                        # 保持
            elif _t < 120:                  # ② 退 5cm
                _k3 = _t - 80
                for _sn, (_aids, _rmp, _st) in _ro_retreat.items():
                    for _i2, _j2 in enumerate(_aids):
                        q[_j2] = float(_rmp[_k3, _i2])
            elif _t < 180:                  # ③ 全伸直
                _al = (_t - 120) / 59.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _j, _rv, _fv, _ov in _ro_anim:
                    q[_j] = _rv + _s2 * (_ov - _rv)
            elif _t < 200:
                pass                        # 保持
            elif _t < 300:                  # ④ 平滑回站姿 (关节空间 smoothstep)
                _al = (_t - 200) / 99.0
                _s2 = _al * _al * (3 - 2 * _al)
                for _sn, (_aids, _rmp, _st) in _ro_retreat.items():
                    for _i2, _j2 in enumerate(_aids):
                        _a0 = float(_rmp[-1, _i2])
                        q[_j2] = _a0 + _s2 * (float(_st[_i2]) - _a0)
            qb = q.unsqueeze(0)
            _ro_play += 1
            if _ro_play % 30 == 0 or _ro_play == 300:
                _m = []
                for _tg, _sd in (("right", env._A), ("left", env._B)):
                    with BM.use_side(env, _sd):
                        _d = env.object.data.root_pos_w[0].cpu().numpy() - _ro_obj0[_tg]
                    _m.append(f"{_tg} Δ{np.linalg.norm(_d)*1000:.1f}mm")
                print(f"[root_only] 帧{_ro_play:3d} 物体扰动: " + " | ".join(_m),
                      flush=True)
            if _ro_play >= 300:
                _ro_play = None
                _ro_stage = 2               # 播完定格站姿, 不再响应
                print("[root_only] ✅ 连播完成, 定格站姿", flush=True)
            hand.set_joint_position_target(qb)
            hand.write_data_to_sim()
            env.sim.step(render=True)
            env.scene.update(dt)
            continue
    hand.write_joint_state_to_sim(qb, vb)
    hand.set_joint_position_target(qb)
    hand.write_data_to_sim()
    env.sim.step(render=True)
    env.scene.update(dt)
try:
    _slot.release()
except Exception:
    pass
os._exit(0)          # 硬退 (2026-08-20 三入口同修, 跳过 Isaac 关闭流程)
