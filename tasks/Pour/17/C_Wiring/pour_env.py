"""Pour17 训练环境 (接线工程 2026-08-28 开工)。

体制总纲 (判据单一来源 = A_Design/L3_Learning/progress*.py, 本文件只做接线):
  前馈 = 母带行 (时钟行), 残差 = 绝对锚有界累积 (#13 拍板1), 零动作 = 放音。
  动作 58 = [R臂7, L臂7, R指22, L指22]。
  全链行指针 self.row (0..514): 机器行开环 1/步; 交互行 = IA0 + 进度机时钟(棘轮);
  时钟耗尽后继续开环走 seam2+retreat。
  换基锚 (#13 修正二): row 到达 EXIT0(seam2 首行) 时捕获 offset=实测-母带,
  沿剩余行线性衰减到零 (末行精确落站姿)。
  奖励 = adv + leash + ms (进度机) - 死线罚; env 侧死线 D2(机器段)/D4/D5/D7 + D6(TODO)。
  观测 495 维, 布局见 _get_observations 注释 (#12 四块+继承块, 变维攒一次)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior
from tasks.pregrasp.env import GraspTaskEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
_L3 = os.path.abspath(os.path.join(_HERE, "..", "A_Design", "L3_Learning"))
sys.path.insert(0, _L3)
from progress import (PourProgress, _axis_tilt, CERT_RAMP, CERT_RET,  # noqa: E402
                      G1_HOLD)
from progress_batch import (  # noqa: E402
    M2_HOLD, M3_HOLD, M4_HOLD, PourProgressBatch, TABLE_Z)
import progress_batch as PBM  # noqa: E402

MASTER = os.environ.get("POUR_REF_NPZ") or os.path.abspath(os.path.join(
    _HERE, "..", "A_Design", "L2_Reference", "pour17_reference_v2.npz"))
OBS_DIM = 503   # v4: +滑移块8(每侧: 滑移量/滑速/垫压和/垫数) 2026-08-28 拍板
ACT_DIM = 58
LOOK_KS = (1, 2, 4, 8, 16)          # #12 拍板2: 几何梯前瞻
DEV_ARM_MACHINE = 0.05              # 机器段(照谱)累积界
# L5-10 用户拍板: 交互段残差界按置信档 —— 越不信参考, 给策略越大的绕开权限
# (原全段 0.08 与"红档容差8cm"不自洽: 保证能力仅 3.2cm, 名义自由度是空转)
DEV_ARM_TIER = {2: 0.05, 1: 0.08, 0: 0.10}   # 绿/黄/红
D4_SLIP = 0.05                      # #10: 滑移 5cm; 超时动态见 __init__ (L5-3)
FAIL_PEN, D6_PEN, D6_CAP = -10.0, -0.5, -10.0
COLLIDE_FTH = 1.0                    # 禁碰力阈值 N
# 禁碰处置: off=只记账不改行为(默认) / pen=逐步小罚 / kill=死线终止
COLLIDE_MODE = os.environ.get("POUR_COLLIDE", "off")
COLLIDE_PEN = float(os.environ.get("POUR_COLLIDE_PEN", "0.5"))
PAD_FTH = 0.5                       # N, 垫接触力阈 (M1 冒烟同款)
PADS_MIN = 3                        # G1 垫数阈 (L5-1: 探针标定, 左手位形上限3)
_LPADS = ["left_thumb_elastomer", "left_index_elastomer", "left_middle_elastomer",
          "left_ring_elastomer", "left_pinky_elastomer"]


def _mouth_local(z, rows, oi, half):
    q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
    w, x, y, zz = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                  [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                  [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up


def build_cfg(num_envs=1):
    """M1 冒烟同款配方 + 双手化字段."""
    from isaaclab.sensors import ContactSensorCfg
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, "Pour17_bottle")
    cfg.contact_sensors = list(cfg.contact_sensors) + [
        ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}",
                         history_length=1,
                         filter_prim_paths_expr=["/World/envs/env_.*/Aux"])
        for n in _LPADS]
    # D6 跨侧互撞传感器 (拍板: 全外壳跨侧, FK可达剪枝=远端臂节+腕+手垫):
    # 右侧远端外壳 × 左侧远端外壳, 判力不判距
    # POUR_NO_D6=1 跳过 (录像 1-env 下 PhysX 过滤展开数断言, 训练 512env 无此病)
    if os.environ.get("POUR_NO_D6") == "1":
        cfg.sensor_slices = {"pad_r": (0, 5), "pad_l": (5, 10),
                             "d6": (10, 10), "noc_arm": (10, 10),
                             "palm": (10, 10), "obj_obj": (10, 10)}   # 全空段
        cfg.approach_only = True
        apply_grasp_prior(cfg, "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz",
                          19.5, approach=True)
        cfg.scene.num_envs = num_envs
        cfg.obj_jitter_xy = 0.0
        cfg.action_space = ACT_DIM
        cfg.observation_space = OBS_DIM
        return cfg
    # ★L5-25 修 D6 绑定 (实测 bug): 原写法源 prim 用正则匹配 9 个体, 而每个
    # filter 表达式只展开 512 个 —— PhysX 要求"每个 filter 展开数 == 源体数",
    # 于是启动即报 `expected 4608, found 512` 且力矩阵恒零。**D6 从未工作过**
    # (5 条线 7900 万步 pen6 恒为 0)。正确形态见垫传感器: **1 个源体 : N 个 filter**。
    _RD = ["R_arm_l5", "R_arm_l7", "R_arm_l8", "right_hand_C_MC"]
    _LD = ["L_arm_l5", "L_arm_l7", "L_arm_l8", "left_hand_C_MC"]
    _lf = [f"/World/envs/env_.*/Robot/{n}" for n in
           ("L_arm_l5", "L_arm_l7", "L_arm_l8", "left_hand_C_MC")]
    _new = [ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}",
                             history_length=1, filter_prim_paths_expr=_lf)
            for n in _RD]                                   # D6: 4 个 (右体 × 左组)
    # ★禁碰传感器 (拍板: 只有手和自己要操作的物体可以碰): 臂节与掌根**不该碰任何
    # 东西**, 所以用不带 filter 的净接触力, 判据精确且便宜(无力矩阵)。
    # 掌根归组更正 (实测): 零动作参考回放里 813 次"禁碰"全部来自掌根
    # (right_hand_C_MC 810 次 23.97N / left_hand_C_MC 417 次 4.97N), 臂连杆
    # l5/l7/l8 一次都没碰过。而掌根碰的正是它自己要操作的物体 -- 掌根是手的一部分,
    # 按"手可以碰自己的物体"本就该允许。若按原归组上 pen, 等于罚策略跟随参考。
    _new += [ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}",
                              history_length=1)
             for n in (_RD[:3] + _LD[:3])]                  # 禁碰臂连杆: 6 个
    # 掌根: 与手垫同规矩(净力 - 对自物体的力); 碰自己物体合法, 碰别的算禁碰
    _new += [ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/right_hand_C_MC",
                              history_length=1,
                              filter_prim_paths_expr=["/World/envs/env_.*/Object"]),
             ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/left_hand_C_MC",
                              history_length=1,
                              filter_prim_paths_expr=["/World/envs/env_.*/Aux"])]
    # 物体互撞(瓶 vs 杯): 需要物体 spawn 开 activate_contact_sensors, 否则
    # ContactSensor 初始化直接 RuntimeError(实测)。开它=改世界, 故挂开关默认关。
    _oc = os.environ.get("POUR_OBJ_CONTACT") == "1"
    if _oc:
        # ★不能在 spawn 阶段设 activate_contact_sensors: 主物体的 usd 是**纯视觉
        # 网格**, 刚体是 correction_env._setup_scene 运行时才贴的, spawn 时
        # prim 下没有 rigid body ⟹ IsaacLab 直接 ValueError(实测)。
        # 正确位置 = 贴完刚体之后, 见 PourEnv._setup_scene 里的 ContactReportAPI。
        pass
        _new += [ContactSensorCfg(prim_path="/World/envs/env_.*/Object",
                                  history_length=1,
                                  filter_prim_paths_expr=["/World/envs/env_.*/Aux"])]
    cfg.contact_sensors = list(cfg.contact_sensors) + _new
    # ★索引区段显式记账: 原来靠 [:5]/[5:10]/[10:] 魔法切片, 一加传感器就串位。
    cfg.sensor_slices = {"pad_r": (0, 5), "pad_l": (5, 10), "d6": (10, 14),
                         "noc_arm": (14, 20), "palm": (20, 22),
                         "obj_obj": (22, 23) if _oc else (22, 22)}
    cfg.approach_only = True
    apply_grasp_prior(cfg, "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz",
                      19.5, approach=True)
    cfg.scene.num_envs = num_envs
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    return cfg


class PourEnv(GraspTaskEnv):
    """接线 env: 父类只借场景/资产/传感器, RL 面 (动作/观测/奖励/终止/重置) 全覆写."""

    def _setup_scene(self):
        super()._setup_scene()
        if os.environ.get("POUR_OBJ_CONTACT") == "1":
            # 物体接触上报: 必须在刚体贴好之后、传感器初始化(sim.reset)之前。
            # 只对 env_0 施加, 其余 env 由场景克隆复制(与刚体化同一套路)。
            import omni.usd
            from pxr import PhysxSchema
            _stg = omni.usd.get_context().get_stage()
            _done = []
            for _pth in ("/World/envs/env_0/Object", "/World/envs/env_0/Aux"):
                _pr = _stg.GetPrimAtPath(_pth)
                if _pr and _pr.IsValid():
                    PhysxSchema.PhysxContactReportAPI.Apply(_pr)
                    _done.append(_pth.rsplit("/", 1)[-1])
            assert len(_done) == 2, \
                f"POUR_OBJ_CONTACT=1 但只给 {_done} 加上了接触上报 —— " \
                f"缺的那个物体的碰撞检测不到, 不能静默继续"
            print(f"[setup] 物体接触上报已开: {_done}", flush=True)
        self._all_sensors = list(self._contact_sensors)
        self._slices = dict(getattr(self.cfg, "sensor_slices", {}))
        if not self._slices:                       # 兜底: 老配方(无禁碰传感器)
            self._slices = {"pad_r": (0, 5), "pad_l": (5, 10),
                            "d6": (10, len(self._all_sensors)),
                            "noc_arm": (0, 0), "palm": (0, 0),
                            "obj_obj": (0, 0)}
        self._contact_sensors = self._all_sensors[:5]   # 父类内部形状 5 (冒烟同款)

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        dev, N = self.device, self.num_envs
        self._master_path = MASTER          # 世界指纹用
        z = np.load(MASTER, allow_pickle=True)
        rows_h = np.where(np.asarray(z["source"]) == 1)[0]
        # ---- 母带 -> 58 维布局 + 关节映射 ----
        jn = list(self.hand.joint_names)
        fin = [str(n) for n in z["fin_names"]]
        ids = [jn.index(f"{P}_arm_j{i}") for P in ("R", "L") for i in range(1, 8)]
        for s in ("right", "left"):
            ids += [jn.index(n.replace("right_", f"{s}_")) for n in fin]
        assert len(set(ids)) == ACT_DIM
        self.map_ids = ids                               # python list (isaaclab 口径)
        self.map_ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
        T = len(np.asarray(z["source"]))
        ref = np.zeros((T, ACT_DIM))
        ref[:, 0:7] = np.asarray(z["right_q"], np.float64)
        ref[:, 7:14] = np.asarray(z["left_q"], np.float64)
        ref[:, 14:36] = np.asarray(z["right_f"], np.float64)
        ref[:, 36:58] = np.asarray(z["left_f"], np.float64)
        self.ref58 = torch.tensor(ref, dtype=torch.float32, device=dev)
        self.SRC = torch.tensor(np.asarray(z["source"], np.int64), device=dev)
        self.T_ROW = T
        self.IA0, self.IA1 = int(rows_h[0]), int(rows_h[-1])
        segl = [int(v) for v in z["seg_lens"]]
        self.EXIT0 = self.IA1 + 1                        # seam2 首行 = 换基捕获点
        self.D7 = int(self.T_ROW * 1.2) + 120            # 超时 (放回窗扩张+认证机余量)
        self.RETREAT0 = segl[0] + segl[1] + segl[2] + segl[3]
        # 物体静置位 (母带交互首行), env 系 (母带=世界系, 建env后与场景静置位对账)
        self.rest_pose = {oi: torch.tensor(np.concatenate([
            np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows_h][0],
            np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows_h][0]]),
            dtype=torch.float32, device=dev) for oi in (0, 1)}
        # ---- 认证行 (5mm测试臂参考) + 人手形状先验 (P-HYB) ----
        if "cert_arm7_right" in z:
            _ca = np.concatenate([np.asarray(z["cert_arm7_right"], np.float64),
                                  np.asarray(z["cert_arm7_left"], np.float64)])
            self.cert_arm14 = torch.tensor(_ca, dtype=torch.float32, device=dev)
        else:
            self.cert_arm14 = None
        _hp = os.environ.get("POUR_VARIANT", "HYB").upper() == "HYB"
        if _hp and "human_right_q" in z:
            _hr = np.asarray(z["human_right_q"], np.float64)[rows_h]
            _hl = np.asarray(z["human_left_q"], np.float64)[rows_h]
            _dh = np.concatenate([np.diff(_hr, axis=0), np.diff(_hl, axis=0)],
                                 axis=1)
            self.hand_dh = torch.tensor(np.vstack([_dh, np.zeros((1, 14))]),
                                        dtype=torch.float32, device=dev)
        else:
            self.hand_dh = None
        self._prev_armq = torch.zeros(N, 14, device=dev)
        # ---- 进度机 (判据单一来源) ----
        mb = _mouth_local(z, rows_h, 1, 0.087)
        mc = _mouth_local(z, rows_h, 0, 0.066)
        self.KCAP = int(os.environ.get("POUR_KCAP", "0"))   # LIFT 单科考: >0 生效
        self._variant = os.environ.get("POUR_VARIANT", "HYB").upper()
        assert self._variant in ("HYB", "OBJ"), self._variant
        self.PB = PourProgressBatch(
            MASTER, num_envs=N, device=str(dev),
            mouth_local_bot=mb, mouth_local_cup=mc,
            leash_rot_tilt=os.environ.get("POUR_LEASH_ROT_TILT") == "1",
            kcap=self.KCAP if self.KCAP > 0 else None,
            no_hand_ref=(self._variant == "OBJ"))
        _ps = PourProgress(MASTER, mouth_local_bot=mb, mouth_local_cup=mc,
                           no_hand_ref=(self._variant == "OBJ"))
        print(f"[PourEnv] 变体={self._variant} "
              f"({'纯物轨消融' if self._variant == 'OBJ' else '置信门控双参考+形状指引'})")
        # RSI 进入点表 (#11 自动推导), env 级: (全链行, 交互行, ms预置, 物体源)
        self._ps = _ps
        # 渐进RSI (L5-1 拍板3): 初始仅t0; 训练循环按 EMA(gate_k)>=0.2 调 unlock()
        self.unlocked = set(int(x) for x in
                            os.environ.get("POUR_UNLOCK", "").split(",") if x)
        self._rebuild_entries()
        self.p_t0 = 0.2                                  # 训练循环按 EMA(sr/gate4) 更新
        # C线消融: POUR_BONUS_NOW=1 → 贴实奖金开局即发 (拍板点3的实验分支)
        self.phase_b = os.environ.get("POUR_BONUS_NOW") == "1"
        self.pad_pot_max = torch.zeros(N, 2, device=dev)  # 相B垫贴实势 earn-only 棘轮
        self.lift_hold = torch.zeros(N, dtype=torch.long, device=dev)
        self.lift_done = torch.zeros(N, dtype=torch.bool, device=dev)
        self._lift_acc = {"ep": 0, "succ": 0}
        self.force_entry = None                          # 冒烟用: 指定各env进入点序号
        # ---- 残差机械 (#13) ----
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        # ★L5-31 消融旗 POUR_ARM_FREE: `straight` 臂 —— **不把参考翻译到关节空间**。
        #   由来(用户裁定, 读法B): "物体轨迹直接当目标, 不解算 IK 变成臂参考"。
        #   但动作是 q_tgt = ff + cum_res, 而 ff **必须**是 58 维关节角 —— 物体的
        #   7 维位姿和关节角不是同一个空间, 中间必须有一次转换, 那次转换就是 IK。
        #   所以"去掉 IK"落地为: **臂列的 ff 冻结在交互段首行(抓握姿势)**,
        #   策略只能靠残差自己走出整个倒水动作。
        #   ★手指列不冻 —— 冻它会混进第二个变量(手指前馈是另一件事)。
        #   ★物体侧完全不动: adv/leash/时钟/G3 只看物体, 不看臂 ⟹ **稠密奖励照常**。
        #     这正是 straight 与 goal 的分界: straight 保留物体逐行跟踪, goal 才去掉。
        #   ★残差界必须同时放开, 否则结构上做不到: 参考自身相对冻结姿势的最大偏离
        #     是 R_j5 的 1.954 rad(112°), 而现行累计界只有 0.05~0.10 rad(3~6°)。
        #     两者不可分割 —— 判读时必须写明 straight 与 base 相差**两项**。
        #   ★按比例放大而非拍平: 现行 0.05/0.08/0.10 同乘 20 → 1.0/1.6/2.0 rad,
        #     **只放大不改形状**, 让 dev_arm 随 conf 变化的结构保留下来 ⟹
        #     straight 与 base_oh 之间仍然只差"有没有臂前馈"这一件事。
        self.arm_free = os.environ.get("POUR_ARM_FREE") == "1"
        _afs = float(os.environ.get("POUR_ARM_FREE_SCALE", "20.0"))
        arm_step = to(self.cfg.arm_residual_max) * float(self.cfg.arm_step_scale)
        if self.arm_free:
            # 每步界定在 0.05 rad(2.9°): 参考逐行增量 P95=0.029 rad 的约 1.7 倍;
            # 累积到 2.0 rad 需 40 步, 而交互段有 273 行 —— 走得到位又不会一步跨过。
            arm_step = torch.full_like(arm_step, 0.05)
        fin_step = to(self.cfg.finger_residual_max) * float(self.cfg.finger_step_scale)
        self.step_bound = torch.cat([arm_step, arm_step, fin_step, fin_step])
        fin_dev = to(self.cfg.finger_residual_max) * float(self.cfg.finger_dev_scale)
        self.dev_fin = torch.cat([fin_dev, fin_dev])     # (44,)
        self.cum_res = torch.zeros(N, ACT_DIM, device=dev)
        self.rebase_off = torch.zeros(N, ACT_DIM, device=dev)
        self.rebase_armed = torch.zeros(N, dtype=torch.bool, device=dev)
        # ---- 逐env状态 ----
        self.row = torch.zeros(N, dtype=torch.long, device=dev)
        self.grasp_d0 = torch.full((N, 2), float("nan"), device=dev)
        self.d6_acc = torch.zeros(N, device=dev)
        self.last_act = torch.zeros(N, ACT_DIM, device=dev)
        self._prev_d = torch.zeros(N, 2, device=dev)       # 上步手物距 (滑速用)
        self._prev_pf = torch.zeros(N, 2, device=dev)      # 上步垫压和 (反射用)
        self._slip_obs = torch.zeros(N, 8, device=dev)     # 滑移块缓存
        self._tick_out = None
        # 身体索引
        bn = list(self.hand.body_names)
        self.wid = {"R": bn.index("right_hand_C_MC"), "L": bn.index("left_hand_C_MC")}
        _pads = ["thumb", "index", "middle", "ring", "pinky"]
        self._pad_bids = [bn.index(f"right_{n}_elastomer") for n in _pads] + \
            [bn.index(f"left_{n}_elastomer") for n in _pads]
        self.hand_bids = [i for i, n in enumerate(bn)
                          if ("elastomer" in n or "hand" in n)]
        self.arm_jids_t = self.map_ids_t[:14]
        # ---- 难度消融旋钮 (L5-13, 2026-08-29 用户拍板): 物体质量/摩擦运行时覆写 ----
        # POUR_OBJ_MASS: 两物体质量(kg) | POUR_OBJ_FRIC: 两物体摩擦
        # 配合 POUR_PAD_FRIC(指垫摩擦, 在 correction_env)。不设则保持原值。
        _m = os.environ.get("POUR_OBJ_MASS")
        _fr = os.environ.get("POUR_OBJ_FRIC")
        if _m or _fr:
            for _nm, _art in (("瓶", self.object), ("杯", self.aux)):
                try:
                    _v = _art.root_physx_view
                    if _m:
                        _ms = _v.get_masses().clone()
                        _ms[:] = float(_m) / max(_ms.shape[1], 1)
                        _v.set_masses(_ms, torch.arange(_ms.shape[0]))
                    if _fr:
                        _mp = _v.get_material_properties().clone()
                        _mp[..., 0] = float(_fr)      # static
                        _mp[..., 1] = float(_fr)      # dynamic
                        _v.set_material_properties(_mp, torch.arange(_mp.shape[0]))
                    _ms2 = _v.get_masses()[0].sum()
                    _mp2 = _v.get_material_properties()[0][0]
                    print(f"[PourEnv] 难度覆写 {_nm}: 质量={float(_ms2):.3f}kg "
                          f"摩擦={float(_mp2[0]):.2f}/{float(_mp2[1]):.2f}")
                except Exception as _e:
                    print(f"[PourEnv] ★难度覆写 {_nm} 失败: {type(_e).__name__}: {_e}")
        if os.environ.get("POUR_PAD_FRIC"):
            print(f"[PourEnv] 指垫摩擦覆写 = {os.environ['POUR_PAD_FRIC']}")

        # ---- 参考系 z 对齐 (接线口径#3): 瓶母带静置 z 穿桌 1.6cm 被物理弹飞 ----
        # 病根: recon网格顶点(母带净空钳制用) vs 烘焙碰撞USD 的原点/尺度系统差。
        # 修法: 瓶全轨 z 平移换基到父类物理推导的 obj_rest_z (烘焙USD立姿合法z);
        # xy/朝向不动 (杯已实证母带位姿物理稳定 0.2cm; 瓶立姿朝向已验算)。
        dz1 = float(self.obj_rest_z) - float(self.rest_pose[1][2])
        print(f"[PourEnv] 瓶 z 换基 Δz={dz1*100:+.2f}cm (obj_rest_z="
              f"{float(self.obj_rest_z):.4f} vs 母带 {float(self.rest_pose[1][2]):.4f})")
        assert abs(dz1) < 0.05, f"瓶 z 差 {dz1*100:.1f}cm 异常大, 先查资产"
        self.PB.ref_obj[1][:, 2] += dz1
        self.PB.rest[1] = self.PB.ref_obj[1][0].clone()
        self.rest_pose[1][2] += dz1
        # ---- B线消融: squeeze 掺前馈 (山丘剖面: 合拢渐入→交互全量→缝2渐出) ----
        self.sq_add = None
        if os.environ.get("POUR_SQUEEZE_FF") == "1":
            _sqr = np.asarray(np.load(
                "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")["squeeze"],
                np.float64).reshape(-1)[7:29]
            _sql = np.asarray(np.load(
                "tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")["squeeze"],
                np.float64).reshape(-1)[7:29]
            _br = float(os.environ.get("POUR_BETA_R", "2.0"))
            _bl = float(os.environ.get("POUR_BETA_L", "1.0"))
            _dsq = np.concatenate([
                _br * (_sqr - ref[self.IA0, 14:36]),
                _bl * (_sql - ref[self.IA0, 36:58])])
            print(f"[PourEnv] squeeze 剂量: βR={_br} βL={_bl} (L5-1 探针标定)")
            # 剖面在手指门推导(CLOSE0)之后回填
            self._sq_delta = torch.tensor(_dsq, dtype=torch.float32, device=dev)
        # ---- 相A手指门 (#13 拍板3 + 修正一): 合拢段起全开, 转运/撤退冻结 ----
        # 合拢起点自动推导: approach 段内手指列首次显著离开首行 (>0.05 rad)
        fin_mv = (self.ref58[:self.IA0, 14:] - self.ref58[0, 14:]).abs().max(dim=1)
        _mv_rows = (fin_mv.values > 0.05).nonzero()
        self.CLOSE0 = int(_mv_rows[0]) if len(_mv_rows) else self.IA0
        # 2026-08-28 用户GUI裁定: Approach 全程(含合拢梯)手指冻结, 只照规划走完
        # 不改原轨迹; 缝1(焊接窗)起手指探索全开 = 贴实抓握的正业, 归交互段。
        self.APP_END = segl[0]
        gate_rows = torch.zeros(self.T_ROW, device=dev)
        gate_rows[self.APP_END:self.RETREAT0] = 1.0    # 缝1→交互→缝2 开
        self.fin_gate_rows = gate_rows
        # 逐行臂残差界 (机器行=0.05; 交互行按 conf 档查表)
        dev_rows = torch.full((self.T_ROW,), DEV_ARM_MACHINE, device=dev)
        _tm = self.PB.tmix.detach().cpu().numpy()
        for _k in range(self.PB.N_ROW):
            dev_rows[self.IA0 + _k] = DEV_ARM_TIER[int(_tm[_k])] \
                * (_afs if self.arm_free else 1.0)
            # ★只放大**交互段**: 机器行(Approach/Retreat)保持 DEV_ARM_MACHINE。
            #   整体乘会把接近段的臂权限也放大 20 倍 —— 那段是 cuRobo 带碰撞检查
            #   的可行规划, 放大权限曾致撞杯(2026-08-28 用户裁定"接近段臂残差同冻")。
        if self.arm_free:
            # ★★L5-31 认证行豁免 (2026-08-30 实测后补): **认证那一行不放大**。
            #   实测 straight 两条跑到 2M 步 cert_pass 恒 0, 死因分项给出铁证:
            #     rise_bot 0.999 / rise_cup 0.63~0.75   ← 99.9% 是"瓶子没升到 5mm"
            #     slip_r/l 0.19~0.24                    ← 与 base(0.17~0.20)相当, **抓得住**
            #     cert_att 3.1~3.4 次/回合(顶到 3 次上限), base 仅 0.97~1.11
            #   机制: G2 认证靠把臂参考插值向"+5mm 抬升行", 而 5mm 换算到关节只有
            #   零点几度; straight 把权限从 ±3° 放大到 ±115°, **认证信号被自己的
            #   动作幅度淹没** —— 像量身高时被测的人在原地蹦跳: 尺子没问题、刻度
            #   没问题, 是被晃动盖住了。
            #   ★不是"抓不稳"(滑移正常), 是"抬"这个微动作测不出来。
            #   解: 认证恒发生在 `r == IA0`(见 _ff_row 的 `a * (r == IA0)`), 把那一行
            #   的界钉回基线值 ⟹ **straight 与 base 在认证行用完全相同的限额**,
            #   反而更可比。物理理由: 认证是"静止测稳定", 本就不需要大权限;
            #   需要大权限的是倒水那一段。
            dev_rows[self.IA0] = DEV_ARM_TIER[int(_tm[0])]
            print(f"[PourEnv]   ★认证行(交互首行)界豁免放大, 保持 "
                  f"{DEV_ARM_TIER[int(_tm[0])]} —— 否则 5mm 抬升信号被动作幅度淹没",
                  flush=True)
        self.dev_arm_rows = dev_rows
        _cnt = {t: int((_tm == t).sum()) for t in (2, 1, 0)}
        _sc = _afs if self.arm_free else 1.0
        print(f"[PourEnv] 残差界按档: 绿{DEV_ARM_TIER[2]*_sc:.3g}({_cnt[2]}行) "
              f"黄{DEV_ARM_TIER[1]*_sc:.3g}({_cnt[1]}行) 红{DEV_ARM_TIER[0]*_sc:.3g}({_cnt[0]}行)")
        if self.arm_free:
            print(f"[PourEnv] ★POUR_ARM_FREE=1 (straight 臂): 臂前馈**冻结在交互段首行**, "
                  f"手指前馈不动; 累计界×{_afs:g}, 每步界 0.05rad(2.9°)", flush=True)
            print(f"[PourEnv]   物体侧不变 —— adv/leash/时钟/G3 只看物体, 稠密奖励照常。",
                  flush=True)
        # ★L5-31 轴对称假设 —— 把一个此前完全沉默的物体前提喊出来。
        #   `_axis_only_R`(参考反解IK) 与 `_tilt`(皮筋rot/G3/placed/G4/死线) 全链
        #   都丢弃"绕长轴的自转"。对瓶/杯正确(两者轴对称); 换非轴对称物体
        #   (带把手的杯、勺、盒、壶嘴瓶) 则**判据静默失效**: 物体绕长轴转任意角度,
        #   所有 Gate 照样全绿, 没有任何检查会红。
        _ub = [round(float(x), 3) for x in self.PB.up.tolist()]
        _uc = [round(float(x), 3) for x in self.PB.upc.tolist()]
        print(f"[PourEnv] ★轴对称假设: 长轴 瓶={_ub} 杯={_uc}; "
              f"绕长轴自转在 参考IK/皮筋/G3/placed/G4/死线 全链**不判**")
        # ★ 必须两个键都读: 生成器把新字段写进 **meta_v5**, 而 `meta` 是 v1 遗留的
        #   描述串。第一版只读 `meta` ⟹ 这条断言永远不会触发 —— 正是"写了但从不执行"
        #   那一族(今天刚提交的教训, 转头自己又埋了一次)。
        _mt = " ; ".join(str(z[k]) for k in ("meta", "meta_v5") if k in z)
        if "up_local=" in _mt:
            _dec = _mt.split("up_local=")[1].split(";")[0]
            _cur = ",".join(str(x) for x in _ub)
            if _dec.replace(" ", "") != _cur:
                raise SystemExit(
                    f"[PourEnv] ★母带声明的长轴 up_local={_dec} 与判据在用的 {_cur} "
                    f"不一致 —— 参考按一个轴反解IK、判据按另一个轴打分, 必然自相矛盾")
            print(f"[PourEnv]   母带声明 up_local={_dec} —— 与判据一致 ✓")
        else:
            print("[PourEnv]   ⚠ 母带未声明 up_local —— 记为**未知**, "
                  "不是'已核对'(2026-08-27 前的母带都没有这一项)")
        # 2026-08-28 用户裁定补全: "不改动原轨迹"含臂 —— Approach 段臂残差同冻
        # (规划已是带碰撞检查的可行解, ±2cm臂权限曾致撞杯; 撤退臂保留=D8躲避正业)
        arm_gate = torch.ones(self.T_ROW, device=dev)
        arm_gate[:self.APP_END] = 0.0
        self.arm_gate_rows = arm_gate
        print(f"[PourEnv] 手指门: 冻结 [0,{self.APP_END}) 与 [{self.RETREAT0},末] "
              f"(Approach全程冻结, 缝1起开)")
        if getattr(self, "_sq_delta", None) is not None:
            prof = torch.zeros(self.T_ROW, device=dev)
            r1 = torch.arange(self.APP_END, self.IA0, device=dev)
            prof[r1] = (r1 - self.APP_END).float() / max(self.IA0 - self.APP_END, 1)
            prof[self.IA0:self.IA1 + 1] = 1.0
            r2 = torch.arange(self.IA1 + 1, self.RETREAT0, device=dev)
            prof[r2] = 1.0 - (r2 - self.IA1).float() / max(
                self.RETREAT0 - self.IA1, 1)
            self.sq_add = prof.unsqueeze(1) * self._sq_delta.unsqueeze(0)
            print(f"[PourEnv] B线: squeeze 掺前馈已开 (|Δ|max="
                  f"{float(self._sq_delta.abs().max()):.3f} rad, 山丘剖面)")
        # ---- RSI 焊接热身: 进入后物体钳位 K 步, 手指压实成形后放开 ----
        self.HOLD_K = int(os.environ.get("POUR_HOLD_K", "15"))   # D线消融旋钮
        self.hold_left = torch.zeros(N, dtype=torch.long, device=dev)
        self.hold_pose = {oi: torch.zeros(N, 7, device=dev) for oi in (0, 1)}
        # TB 计数
        self.racc = {"adv": 0.0, "leash": 0.0, "ms": 0.0, "pen": 0.0,
                     "pen6": 0.0, "bonus": 0.0, "regrip": 0.0, "slope": 0.0,
                     "wage": 0.0, "shape": 0.0, "n": 0}
        # ★L5-23: 旧版这里是个"死骨架" —— D1~D8 键建了但从未被写过, 也没有任何
        # 出口。若当初有人接上去, 会读到一串 0 并得出"没有死于 D1~D8"的假结论。
        self.tb = {k: 0 for k in
                   ("term/D1_pre", "term/D2_pre", "term/D3_pre", "term/D4_slip",
                    "term/D5_table", "term/D9_collide", "term/D7_timeout",
                    "term/judge_fail", "term/M4_success",
                    "term/collide_arm", "term/collide_pad",
                    "term/collide_objobj", "term/collide_d6",
                    "term/collide_any")}
        self.tb["d6_pen_sum"] = 0.0
        self.tb["ep"] = 0

    def _rebuild_entries(self):
        """按已解锁 Gate 重建出生表 (渐进RSI)."""
        self.entries = []
        for row_i, ms, label in self._ps.entry_table(self.unlocked):
            if label == "t0":
                self.entries.append((0, -1, frozenset(), "rest", "t0"))
            elif label == "ret":
                self.entries.append((self.RETREAT0, row_i, frozenset(ms),
                                     "rest", label))
            else:
                self.entries.append((self.IA0 + row_i, row_i, frozenset(ms),
                                     "ref", label))
        if self.KCAP > 0:                              # LIFT: 撤退点无意义
            self.entries = [e for e in self.entries
                            if e[4] in ("t0", "g1", "g2")]
        print(f"[PourEnv] RSI出生表: {[e[4] for e in self.entries]}")

    # ================= 动作 =================
    def _pre_physics_step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self.last_act = a.clone()
        r0 = self.row.clamp(max=self.T_ROW - 1)
        delta = a * self.step_bound
        delta[:, :14] *= self.arm_gate_rows[r0].unsqueeze(1)   # Approach臂冻(照谱)
        delta[:, 14:] *= self.fin_gate_rows[r0].unsqueeze(1)   # 相A手指门(硬冻)
        self.cum_res = self.cum_res + delta
        r = self.row.clamp(max=self.T_ROW - 1)
        dev_arm = self.dev_arm_rows[r].unsqueeze(1)
        self.cum_res[:, :14] = torch.maximum(
            torch.minimum(self.cum_res[:, :14], dev_arm), -dev_arm)
        self.cum_res[:, 14:] = torch.maximum(
            torch.minimum(self.cum_res[:, 14:], self.dev_fin), -self.dev_fin)
        # 换基捕获: 首次到达 EXIT0
        cap = self.rebase_armed & (r >= self.EXIT0)
        if cap.any():
            qnow = self.hand.data.joint_pos[:, self.map_ids_t]
            self.rebase_off[cap] = (qnow - self.ref58[self.EXIT0].unsqueeze(0))[cap]
            self.rebase_armed[cap] = False
        self._ff = self._ff_row(r)
        self.q_tgt = self._ff + self.cum_res

    def _ff_row(self, r):
        ff = self.ref58[r]
        if getattr(self, "arm_free", False):
            # 臂列冻结在交互段首行; 手指列保持逐行(它不是本消融的对象)。
            # ★只冻**交互段** [IA0, IA1]: Approach 段的臂参考是 cuRobo 带碰撞检查的
            #   可行规划, 冻掉它机器人会从第 0 步就往抓握姿势走, 整个接近段被毁;
            #   Retreat 段同理。本消融问的是"交互段要不要臂参考", 不是"全程"。
            #   (第一版写成无条件冻结, 会静默毁掉 Approach —— 这里改掉。)
            _in_ia = ((r >= self.IA0) & (r <= self.IA1)).unsqueeze(1).float()
            ff = ff.clone()
            ff[:, :14] = (ff[:, :14] * (1 - _in_ia)
                          + self.ref58[self.IA0][:14].unsqueeze(0) * _in_ia)
        if self.sq_add is not None:
            ff = ff.clone()
            ff[:, 14:] += self.sq_add[r]
        # 认证窗 (L5-1): 站位行上臂参考向 +5mm 认证行插值 (斜坡5/保持5/放回5)
        if self.cert_arm14 is not None and hasattr(self, "PB"):
            ph, ct = self.PB.cert_phase, self.PB.cert_t.float()
            a = torch.where(
                ph == 1, (ct + 1) / CERT_RAMP,
                torch.where(ph == 2, torch.ones_like(ct),
                            torch.where(ph == 3, 1.0 - (ct + 1) / CERT_RET,
                                        torch.zeros_like(ct)))).clamp(0, 1)
            a = a * (r == self.IA0).float()
            if bool((a > 0).any()):
                ff = ff.clone()
                ff[:, :14] = (ff[:, :14] * (1 - a).unsqueeze(1)
                              + self.cert_arm14.unsqueeze(0) * a.unsqueeze(1))
        span = max(self.T_ROW - 1 - self.EXIT0, 1)
        fade = ((self.T_ROW - 1 - r).float() / span).clamp(0.0, 1.0)
        post = (r >= self.EXIT0).float()
        return ff + self.rebase_off * (fade * post).unsqueeze(1)

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_tgt, joint_ids=self.map_ids)

    # ================= 每步记账 (dones 先于 rewards 被调) =================
    def _get_dones(self):
        self._tick()
        o = self._tick_out
        return o["terminated"], o["timeout"]

    def _read_objs(self):
        org = self.scene.env_origins
        bot = torch.cat([self.object.data.root_pos_w - org,
                         self.object.data.root_quat_w], dim=1)
        cup = torch.cat([self.aux.data.root_pos_w - org,
                         self.aux.data.root_quat_w], dim=1)
        return cup, bot                                  # obj0=杯, obj1=瓶

    def _pads_f(self):
        F = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in (self._seg("pad_r") + self._seg("pad_l"))],
                      dim=1).nan_to_num(0.0)
        return F                                          # (N,10,3) 前5右vs瓶 后5左vs杯

    def _seg(self, key):
        a, b = self._slices.get(key, (0, 0))
        return self._all_sensors[a:b]

    def _d6_hit(self):
        ss = self._seg("d6")
        if not ss:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fm = torch.cat([s.data.force_matrix_w.reshape(self.num_envs, -1, 3)
                        for s in ss], dim=1).nan_to_num(0.0)
        return (fm.norm(dim=-1) > COLLIDE_FTH).any(dim=1)

    def _collide(self):
        """禁碰检测。返回 dict of (N,) bool。
        约定: **只有手垫与自己要操作的物体可以接触**; 其余一切接触都是禁碰。
        - arm : 臂节/掌根碰到任何东西 (无 filter 的净接触力, 精确)
        - pad : 手垫碰到了自己物体以外的东西 (净力 - 对自己物体的力)
        - objobj : 瓶碰杯
        - d6 : 左右两侧互撞 (修好后的 D6)
        ★覆盖边界: 臂只装了远端 4 节/侧 (l5/l7/l8/掌根, 沿用 FK 可达剪枝),
        近端 l1~l4 未覆盖 —— 不能声称"检测到了全部碰撞"。"""
        N, dev = self.num_envs, self.device
        E = torch.zeros(N, 0, 3, device=dev)
        cat = (lambda ss, a: torch.cat([getattr(s.data, a).reshape(N, -1, 3)
                                        for s in ss], dim=1) if ss else E)
        pads = self._seg("pad_r") + self._seg("pad_l") + self._seg("palm")
        if pads:
            netp = torch.stack([s.data.net_forces_w.reshape(N, -1, 3)
                                .norm(dim=-1).amax(dim=1) for s in pads], dim=1)
            filp = torch.stack([s.data.force_matrix_w.reshape(N, -1, 3)
                                .norm(dim=-1).amax(dim=1) for s in pads], dim=1)
        else:
            netp = filp = torch.zeros(N, 0, device=dev)
        return PBM.collide_flags(cat(self._seg("noc_arm"), "net_forces_w"),
                                 netp, filp,
                                 cat(self._seg("obj_obj"), "force_matrix_w"),
                                 cat(self._seg("d6"), "force_matrix_w"),
                                 thr=COLLIDE_FTH)

    def _tick(self):
        N, dev = self.num_envs, self.device
        holding = self.hold_left > 0
        if holding.any():
            hids = holding.nonzero().squeeze(1)
            org = self.scene.env_origins[hids]
            for oi, art in ((0, self.aux), (1, self.object)):
                pose = self.hold_pose[oi][hids].clone()
                pose[:, :3] += org
                art.write_root_pose_to_sim(pose, env_ids=hids)
                art.write_root_velocity_to_sim(
                    torch.zeros(len(hids), 6, device=dev), env_ids=hids)
            self.hold_left[holding] -= 1
        cup, bot = self._read_objs()
        armq_r = self.hand.data.joint_pos[:, self.map_ids_t[:7]]
        armq_l = self.hand.data.joint_pos[:, self.map_ids_t[7:14]]
        # ---- G1 垫数 (L5-1: 双手各>=3/5垫; 判据在进度机, 这里只出原料) ----
        f = self._pads_f().norm(dim=-1)                  # (N,10)
        pads3 = ((f[:, :5] > PAD_FTH).sum(dim=1) >= PADS_MIN) \
            & ((f[:, 5:] > PAD_FTH).sum(dim=1) >= PADS_MIN)
        cand = pads3                                     # 兼容旧消费方
        org_w = self.scene.env_origins
        wr_pos = self.hand.data.body_pos_w[:, self.wid["R"]] - org_w
        wl_pos = self.hand.data.body_pos_w[:, self.wid["L"]] - org_w
        # ---- 进度机 (仅交互行起管辖) ----
        run_mask = (self.row >= self.IA0) & ~holding
        out = self.PB.step(cup, bot, armq_r, armq_l, pads3, wr_pos, wl_pos,
                           run_mask=run_mask)
        # ---- 形状指引 (P-HYB, L5-6 档位化): 0.2×W_HAND(绿0/黄0.5/红0.8)×cos+
        #      人手行只给"怎么动"的形状糖, 不打鞭、不做绝对位姿参考 ----
        r_shape = torch.zeros(N, device=dev)
        if self.hand_dh is not None:
            ki_s = self.PB.k.clamp(max=self.PB.N_ROW - 1)
            dq_act = torch.cat([armq_r, armq_l], dim=1) - self._prev_armq
            cs = torch.nn.functional.cosine_similarity(
                dq_act, self.hand_dh[ki_s], dim=1)
            r_shape = 0.2 * out["w_hand"] * cs.clamp(min=0.0) \
                * (self.PB.g2 & run_mask).float()
        self._prev_armq = torch.cat([armq_r, armq_l], dim=1).detach()
        # ---- env 侧死线 ----
        # ★L5-25 禁碰: 只有手垫与自己要操作的物体可以接触, 其余接触一律记为禁碰。
        _col = self._collide()
        for _k, _v in _col.items():
            self.tb["term/collide_" + _k] += int((_v & ~holding).sum())
        col_any = (_col["arm"] | _col["pad"] | _col["objobj"] | _col["d6"]) \
            & ~holding
        self.tb["term/collide_any"] += int(col_any.sum())
        # ★L5-23 死因分项(env 侧): 原来五种死因 OR 成一个布尔, 死了说不出为什么。
        _ez = lambda: torch.zeros(N, dtype=torch.bool, device=dev)   # noqa: E731
        _ec = {"D1_pre": _ez(), "D2_pre": _ez(), "D3_pre": _ez(),
               "D4_slip": _ez(), "D5_table": _ez(), "D9_collide": _ez()}
        pre = self.row < self.IA0
        for _o, code in ((cup, "cup"), (bot, "bot")):
            _ec["D1_pre"] |= pre & (_o[:, 2] < TABLE_Z - 0.05)         # D1 机器段
        for oi, _o in ((0, cup), (1, bot)):                            # D2 机器段
            up = self.PB.up if oi == 1 else torch.tensor(
                [0.0, 1.0, 0.0], device=dev)
            qn = _o[:, 3:7] / _o[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
            upw = quat_apply(qn, up.unsqueeze(0).expand(len(qn), 3))
            tilt = torch.acos((upw[:, 2] / upw.norm(dim=1).clamp(min=1e-9))
                              .clamp(-1, 1))
            _ec["D2_pre"] |= pre & (tilt > np.radians(30))
            rest = self.rest_pose[oi]
            _ec["D3_pre"] |= pre & ((_o[:, :3] - rest[:3]).norm(dim=1) > 0.35)
        # D4 滑移 (交互行, M1 后, 相对 M1 时刻基线)
        d_r = (self.hand.data.body_pos_w[:, self.wid["R"]]
               - self.object.data.root_pos_w).norm(dim=1)
        d_l = (self.hand.data.body_pos_w[:, self.wid["L"]]
               - self.aux.data.root_pos_w).norm(dim=1)
        need0 = self.PB.g2 & torch.isnan(self.grasp_d0[:, 0])
        self.grasp_d0[:, 0] = torch.where(need0, d_r, self.grasp_d0[:, 0])
        self.grasp_d0[:, 1] = torch.where(need0, d_l, self.grasp_d0[:, 1])
        in_ia = (self.row >= self.IA0) & (self.row <= self.IA1)
        slip = in_ia & (~torch.isnan(self.grasp_d0[:, 0])) & (
            ((d_r - self.grasp_d0[:, 0]).abs() > D4_SLIP)
            | ((d_l - self.grasp_d0[:, 1]).abs() > D4_SLIP))
        _ec["D4_slip"] |= slip
        # ---- v5移植 (2026-08-28 拍板): 滑移量/滑速/垫压 → 反射奖+斜坡罚+观测块 ----
        d_now = torch.stack([d_r, d_l], dim=1)
        pf_now = torch.stack([f[:, :5].sum(dim=1), f[:, 5:].sum(dim=1)], dim=1)
        d0 = torch.nan_to_num(self.grasp_d0, nan=0.0)
        has0 = ~torch.isnan(self.grasp_d0[:, 0])
        slip_amt = (d_now - d0) * has0.float().unsqueeze(1)          # (N,2) m
        slip_v = (d_now - self._prev_d).clamp(min=0.0)               # 正向滑速 m/步
        gate_g = (self.PB.g2 & has0).float()
        # ③ 握紧反射: 滑速起 → 垫压正差分给奖 ×min(滑速/1cm,1), regrip_w=1.0
        dv = (slip_v / 0.01).clamp(max=1.0)
        dfp = (pf_now - self._prev_pf).clamp(0.0, 3.0)
        r_reflex = 1.0 * ((dfp / 3.0) * dv).sum(dim=1) * gate_g
        # ④ 滑移斜坡: 超线性, 满值=0.5×死线罚/侧, D4 悬崖前的梯度
        pen_slope = -0.5 * ((slip_amt.clamp(min=0.0) / D4_SLIP) ** 2) \
            .clamp(max=1.0).sum(dim=1) * gate_g * in_ia.float()
        # ② 滑移观测块 (每侧: 滑移量/5cm, 滑速/1cm, 垫压/f0, 垫数/5)
        SQF = float(self.cfg.squeeze_f0)
        npads = torch.stack([(f[:, :5] > PAD_FTH).sum(dim=1),
                             (f[:, 5:] > PAD_FTH).sum(dim=1)], dim=1).float()
        self._slip_obs = torch.cat([
            (slip_amt / 0.05).clamp(-3, 3), (slip_v / 0.01).clamp(0, 3),
            (pf_now / SQF).clamp(0, 3), npads / 5.0], dim=1)
        self._prev_d = d_now.detach().clone()
        self._prev_pf = pf_now.detach().clone()
        # D5 撞桌: 手部体中心低于桌面
        hz = self.hand.data.body_pos_w[:, self.hand_bids, 2]
        _ec["D5_table"] |= (hz < TABLE_Z - 0.005).any(dim=1)
        if COLLIDE_MODE == "kill":
            _ec["D9_collide"] = col_any
        # D7 超时 (纯终止零罚)
        timeout = self.episode_length_buf >= self.D7
        fail_env = _ez()
        for _v in _ec.values():
            fail_env |= _v
        fail_env &= ~holding
        # 一次死亡可同时命中多因, 各自计数(占比之和可 >1)
        for _k, _v in _ec.items():
            self.tb["term/" + _k] += int((_v & ~holding).sum())
        fail = out["fail"] | fail_env
        lift_new = torch.zeros(N, dtype=torch.bool, device=dev)
        if self.KCAP > 0:
            at_top = self.PB.g2 & (self.PB.k >= self.KCAP) & ~fail & ~holding
            self.lift_hold = torch.where(at_top, self.lift_hold + 1,
                                         torch.zeros_like(self.lift_hold))
            lift_new = (self.lift_hold >= 20) & ~self.lift_done
            self.lift_done |= lift_new
        terminated = fail | out["done"] | lift_new        # done 含成功终局
        self.tb["term/D7_timeout"] += int((timeout & ~terminated).sum())
        self.tb["term/judge_fail"] += int(out["fail"].sum())
        # ---- 行指针推进 (热身期冻结) ----
        self.row = torch.where(pre & ~holding, self.row + 1, self.row)
        ia = run_mask & (self.PB.k < self.PB.N_ROW - 1)
        self.row = torch.where(ia, self.IA0 + self.PB.k, self.row)
        # L5-1修: k走满时 row 停在 IA0+k_max=IA1-1, 旧条件 row>=IA1 永假 → 死锁
        post = run_mask & (self.PB.k >= self.PB.N_ROW - 1) \
            & (self.row >= self.IA1 - 1) & ~holding
        self.row = torch.where(post, (self.row + 1).clamp(max=self.T_ROW - 1),
                               self.row)
        # ---- 奖励合成 ----
        pen = torch.where(fail, torch.full((N,), FAIL_PEN, device=dev),
                          torch.zeros(N, device=dev))
        d6 = _col["d6"] & ~holding
        pen6 = torch.where(d6 & (self.d6_acc > D6_CAP),
                           torch.full((N,), D6_PEN, device=dev),
                           torch.zeros(N, device=dev))
        if COLLIDE_MODE == "pen":
            pen6 = pen6 + torch.where(
                col_any & (self.d6_acc > D6_CAP),
                torch.full((N,), -COLLIDE_PEN, device=dev),
                torch.zeros(N, device=dev))
        self.d6_acc += pen6
        self.tb["d6_pen_sum"] += float(-pen6.sum())
        rew = (out["adv"] + out["leash"] + out["ms"] + out["wage"] + pen
               + r_reflex + pen_slope + r_shape + 15.0 * lift_new.float()) \
            * (~holding).float() + pen6
        bonus = torch.zeros(N, device=dev)
        # 相B赏钱 (#13 拍板3, 轻量同族实现): 垫贴实势 earn-only 棘轮, 合拢→缝2 窗内
        if self.phase_b:
            r_now = self.row.clamp(max=self.T_ROW - 1)
            in_win = (r_now >= self.APP_END) & (r_now < self.RETREAT0) \
                & ~holding & self.PB.g1        # L5-1: 仅G1后发放 (BCE4农耕闸)
            pot = torch.stack([(f[:, :5].clamp(0, 3) / 3).mean(dim=1),
                               (f[:, 5:].clamp(0, 3) / 3).mean(dim=1)], dim=1)
            if os.environ.get("POUR_BONUS_DIST") == "1":
                # C线修正 (2026-08-28): 力势没接触就无梯度(实测哑火 0.0003/步),
                # 改 pad_near 语义 = 距离势(圆柱代理: 垫到物轴向距-半径, 5cm 内线性
                # 升温) 与力势各半 —— 靠近有钱, 贴实更有钱
                bn_pos = self.hand.data.body_pos_w
                pads_r = torch.stack([bn_pos[:, self._pad_bids[i]]
                                      for i in range(5)], dim=1)
                pads_l = torch.stack([bn_pos[:, self._pad_bids[i + 5]]
                                      for i in range(5)], dim=1)
                obj_xy = {"R": self.object.data.root_pos_w[:, :2],
                          "L": self.aux.data.root_pos_w[:, :2]}
                dpot = []
                for pads, side, rr in ((pads_r, "R", 0.035), (pads_l, "L", 0.035)):
                    d = ((pads[:, :, :2] - obj_xy[side].unsqueeze(1))
                         .norm(dim=-1) - rr).clamp(min=0)
                    dpot.append((1 - d / 0.05).clamp(0, 1).mean(dim=1))
                pot = 0.5 * torch.stack(dpot, dim=1) + 0.5 * pot
            inc = (pot - self.pad_pot_max).clamp(min=0) * in_win.float().unsqueeze(1)
            self.pad_pot_max = torch.maximum(self.pad_pot_max, pot)
            bonus = 0.5 * inc.sum(dim=1)
            rew = rew + bonus
        nh = (~holding).float()
        self.racc["adv"] += float((out["adv"] * nh).sum())
        self.racc["leash"] += float((out["leash"] * nh).sum())
        self.racc["ms"] += float((out["ms"] * nh).sum())
        self.racc["pen"] += float((pen * nh).sum())
        self.racc["pen6"] += float(pen6.sum())
        self.racc["regrip"] += float((r_reflex * (~holding).float()).sum())
        self.racc["slope"] += float((pen_slope * (~holding).float()).sum())
        self.racc["bonus"] += float(bonus.sum())
        self.racc["wage"] += float((out["wage"] * nh).sum())
        self.racc["shape"] += float((r_shape * nh).sum())
        self.racc["n"] += N
        # TB
        succ = self.PB.g4
        self._tick_out = {"terminated": terminated, "timeout": timeout & ~terminated,
                          "rew": rew, "cand": cand, "out": out, "cup": cup,
                          "bot": bot, "armq_r": armq_r, "armq_l": armq_l,
                          "fail_env": fail_env, "succ": succ}

    def _get_rewards(self):
        return self._tick_out["rew"]

    def pop_lift(self):
        ep = max(self._lift_acc["ep"], 1)
        out = {"sr/lift": self._lift_acc["succ"] / ep}
        self._lift_acc = {"ep": 0, "succ": 0}
        return out

    def pop_term(self):
        """死因分项台账 (每 epoch 倾倒)。分母 = 本窗结算回合数;
        空分母发 NaN 而非 0.0 —— 与 L5-17 同规矩, "没结算"不得冒充"没死"。"""
        ep = self.tb["ep"]
        nan = float("nan")
        out = {k: ((v / ep) if ep > 0 else nan)
               for k, v in self.tb.items() if k.startswith("term/")}
        out["n/term_ep"] = float(ep)
        out["term/d6_collide_sum"] = (self.tb["d6_pen_sum"] / ep) if ep > 0 else nan
        for k in self.tb:
            self.tb[k] = 0.0 if isinstance(self.tb[k], float) else 0
        return out

    def pop_racc(self):
        """逐项奖惩台账 (每 epoch 倾倒, 均值/步/env)."""
        n = max(self.racc["n"], 1)
        out = {f"ep_rew/{k}": v / n for k, v in self.racc.items() if k != "n"}
        self.racc = {"adv": 0.0, "leash": 0.0, "ms": 0.0, "pen": 0.0,
                     "pen6": 0.0, "bonus": 0.0, "regrip": 0.0, "slope": 0.0,
                     "wage": 0.0, "shape": 0.0, "n": 0}
        return out

    # ================= 观测 (#12 定稿, 495 维, 变维攒一次) =================
    def _get_observations(self):
        N, dev = self.num_envs, self.device
        vs = float(self.cfg.obs_vel_scale)
        org = self.scene.env_origins
        q = self.hand.data.joint_pos[:, self.map_ids_t]
        qd = self.hand.data.joint_vel[:, self.map_ids_t]
        r = self.row.clamp(max=self.T_ROW - 1)
        cup, bot = self._read_objs()
        # 腕
        wp = {s: self.hand.data.body_pos_w[:, self.wid[s]] - org for s in ("R", "L")}
        wq = {s: self._qsign(self.hand.data.body_quat_w[:, self.wid[s]])
              for s in ("R", "L")}
        wlv = {s: self.hand.data.body_lin_vel_w[:, self.wid[s]] for s in ("R", "L")}
        wav = {s: self.hand.data.body_ang_vel_w[:, self.wid[s]] for s in ("R", "L")}
        ff = self._ff_row(r)          # 始终现算 (重置后缓存行会陈旧)
        # 残差归一
        src = self.SRC[r]
        dev_arm = self.dev_arm_rows[r].unsqueeze(1)
        res_n = torch.cat([self.cum_res[:, :14] / dev_arm,
                           self.cum_res[:, 14:] / self.dev_fin], dim=1)
        # 前瞻 (双臂q差14 + w_obj1) x5
        look = []
        for kk in LOOK_KS:
            rk = (r + kk).clamp(max=self.T_ROW - 1)
            look.append(self.ref58[rk][:, :14] - self.ref58[r][:, :14])
            in_ia_k = (rk >= self.IA0) & (rk <= self.IA1)
            wk = torch.where(in_ia_k,
                             self.PB.WO[self.PB.tmix[(rk - self.IA0).clamp(
                                 0, self.PB.N_ROW - 1)]],
                             torch.ones(N, device=dev))
            look.append(wk.unsqueeze(1))
        # 触觉
        F = self._pads_f()                                # (N,10,3)
        qin_r = quat_conjugate(wq["R"]).unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        qin_l = quat_conjugate(wq["L"]).unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        Gw = torch.cat([
            quat_apply(qin_r, F[:, :5].reshape(-1, 3)).reshape(N, 15),
            quat_apply(qin_l, F[:, 5:].reshape(-1, 3)).reshape(N, 15)],
            dim=1) / float(self.cfg.squeeze_f0)
        tipc = (F.norm(dim=-1) > PAD_FTH).float()
        tq = (self.hand.data.applied_torque[:, self.arm_jids_t] * 0.1).clamp(-3, 3)
        # 制度量纲 5 (机器行: [1,0,0,1,0])
        ki = (r - self.IA0).clamp(0, self.PB.N_ROW - 1)
        in_ia = ((r >= self.IA0) & (r <= self.IA1)).float().unsqueeze(1)
        w_obj = self.PB.WO[self.PB.tmix[ki]].unsqueeze(1)
        lp = torch.minimum(self.PB.LP[self.PB.tp[0][ki]],
                           self.PB.LP[self.PB.tp[1][ki]]).unsqueeze(1)
        rot_ban = self.PB.rot_ban[torch.minimum(
            self.PB.tr[0][ki], self.PB.tr[1][ki])].float().unsqueeze(1)
        lr = torch.minimum(self.PB.LR[self.PB.tr[0][ki]],
                           self.PB.LR[self.PB.tr[1][ki]]).clamp(max=3.15).unsqueeze(1)
        gate_hand = (self.PB.tmix[ki] == 0).float().unsqueeze(1)
        regime = torch.cat([
            w_obj * in_ia + (1 - in_ia) * 1.0,
            lp * in_ia, lr * in_ia * (1 - rot_ban),
            rot_ban * in_ia + (1 - in_ia) * 1.0,
            gate_hand * in_ia], dim=1)
        # 物体块 38
        blocks = []
        for _o, s in ((cup, "L"), (bot, "R")):           # 杯-左腕 / 瓶-右腕
            qi = quat_conjugate(wq[s])
            blocks += [quat_apply(qi, _o[:, :3] - wp[s]),
                       self._qsign(quat_mul(qi, _o[:, 3:7])),
                       (self.aux if s == "L" else self.object)
                       .data.root_lin_vel_w,
                       (self.aux if s == "L" else self.object)
                       .data.root_ang_vel_w * vs]
        qb = bot[:, 3:7] / bot[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
        mb_w = bot[:, :3] + quat_apply(qb, self.PB.mb.unsqueeze(0).expand(N, 3))
        qc = cup[:, 3:7] / cup[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
        mc_w = cup[:, :3] + quat_apply(qc, self.PB.mc.unsqueeze(0).expand(N, 3))
        upv = quat_apply(qb, self.PB.up.unsqueeze(0).expand(N, 3))
        tiltcos = (upv[:, 2] / upv.norm(dim=1).clamp(min=1e-9)).unsqueeze(1)
        devs = []
        for oi, _o in ((0, cup), (1, bot)):
            refp = torch.where(in_ia.bool(),
                               self.PB.ref_obj[oi][ki][:, :3],
                               self.rest_pose[oi][:3].unsqueeze(0).expand(N, 3))
            dq = self.PB.ref_obj[oi][ki][:, 3:7]
            qo = _o[:, 3:7] / _o[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
            dang = 2 * torch.acos((qo * dq).sum(1).abs().clamp(max=1.0))
            devs += [_o[:, :3] - refp, dang.unsqueeze(1)]
        # 进度状态 7
        hold = torch.where(
            ~self.PB.g1, self.PB.g1_run.float() / G1_HOLD,
            torch.where(
                ~self.PB.g2,
                (self.PB.cert_phase.float() * 5 + self.PB.cert_t.float()) / 15.0,
                torch.where(~self.PB.g3, self.PB.m2_run.float() / M2_HOLD,
                            torch.where(~self.PB.placed,
                                        self.PB.m3_run.float() / M3_HOLD,
                                        self.PB.m4_run.float() / M4_HOLD)))
        ).clamp(0, 1)
        prog = torch.cat([
            (self.row.float() / (self.T_ROW - 1)).unsqueeze(1),
            self.PB.g1.float().unsqueeze(1), self.PB.g2.float().unsqueeze(1),
            self.PB.g3.float().unsqueeze(1), self.PB.g4.float().unsqueeze(1),
            hold.unsqueeze(1), src.float().unsqueeze(1)], dim=1)
        obs = torch.cat([
            q[:, :14], q[:, 14:],                              # 58 本体位置
            qd[:, :14] * vs, qd[:, 14:] * vs,                  # 58 本体速度
            wp["R"], wq["R"], wlv["R"], wav["R"] * vs,         # 13
            wp["L"], wq["L"], wlv["L"], wav["L"] * vs,         # 13
            ff - q,                                            # 58 跟踪误差
            res_n,                                             # 58 残差状态
            *look,                                             # 75 前瞻
            Gw.clamp(-3, 3), tipc,                             # 40 触觉
            tq,                                                # 14 力矩
            self.last_act,                                     # 58 上步动作
            regime,                                            # 5  制度量纲
            *blocks, mb_w - mc_w, tiltcos, *devs,              # 38 物体块
            prog,                                              # 7  进度状态
            self._slip_obs,                                    # 8  滑移块(v4)
        ], dim=1).float().clamp(-float(self.cfg.clip_obs),
                                float(self.cfg.clip_obs)).nan_to_num(0.0)
        assert obs.shape[1] == OBS_DIM, f"obs {obs.shape[1]} != {OBS_DIM}"
        # priv (critic 侧 12 维): 质量/摩擦 + 双手 tip 力 —— pregrasp 前7维同宗
        tip_f = F.norm(dim=-1)
        priv = torch.cat([self.obj_mass.float(), self.obj_fric.float(),
                          tip_f.float()], dim=1) \
            .clamp(-float(self.cfg.clip_obs), float(self.cfg.clip_obs)) \
            .nan_to_num(0.0)
        return {"policy": obs, "priv_info": priv}

    # ================= 重置 (RSI #11) =================
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        # TB 结账
        if self._tick_out is not None:
            o = self._tick_out
            self.tb["ep"] += len(env_ids)
            self.tb["term/M4_success"] += int(o["succ"][env_ids].sum())
            if self.KCAP > 0:
                self._lift_acc["ep"] += len(env_ids)
                self._lift_acc["succ"] += int(self.lift_done[env_ids].sum())
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        # ---- 采样进入点 ----
        if self.force_entry is not None:
            pick = [self.force_entry[int(i) % len(self.force_entry)]
                    for i in env_ids.cpu().tolist()]
        else:
            pick = []
            u = torch.rand(n)
            for i in range(n):
                if float(u[i]) < self.p_t0 or len(self.entries) <= 1:
                    pick.append(0)
                else:
                    pick.append(1 + int(torch.randint(len(self.entries) - 1,
                                                      (1,))))
        rows_env, rows_ia, g1, g2, g3, plc, objsrc = [], [], [], [], [], [], []
        for pi in pick:
            er, ir, ms, osrc, _ = self.entries[pi]
            rows_env.append(er); rows_ia.append(max(ir, 0))
            g1.append(1 in ms); g2.append(2 in ms); g3.append(3 in ms)
            plc.append("placed" in ms)
            objsrc.append(osrc)
        dev = self.device
        rows_env_t = torch.tensor(rows_env, dtype=torch.long, device=dev)
        self.row[env_ids] = rows_env_t
        # 进度机
        t0_mask = torch.tensor([p == 0 for p in pick], device=dev)
        ent_ids = env_ids[~t0_mask]
        if len(ent_ids):
            sel = (~t0_mask).nonzero().squeeze(1).cpu().tolist()
            self.PB.enter(ent_ids,
                          torch.tensor([rows_ia[i] for i in sel], device=dev),
                          torch.tensor([g1[i] for i in sel], device=dev),
                          torch.tensor([g2[i] for i in sel], device=dev),
                          torch.tensor([g3[i] for i in sel], device=dev),
                          torch.tensor([plc[i] for i in sel], device=dev))
        if t0_mask.any():
            self.PB.reset_idx(env_ids[t0_mask])
        # ---- 写机器人/物体状态 ----
        qfull = self.hand.data.default_joint_pos[env_ids].clone()
        qfull[:, self.map_ids_t] = self.ref58[rows_env_t]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull),
                                           env_ids=env_ids)
        self.hand.set_joint_position_target(qfull, env_ids=env_ids)
        org = self.scene.env_origins[env_ids]
        for oi, art in ((0, self.aux), (1, self.object)):
            pose = torch.zeros(n, 7, device=dev)
            for i, (pi, osrc) in enumerate(zip(pick, objsrc)):
                if osrc == "rest":
                    pose[i] = self.rest_pose[oi]
                else:
                    pose[i] = self.PB.ref_obj[oi][rows_ia[i]]
            pose[:, :3] += org
            art.write_root_pose_to_sim(pose, env_ids=env_ids)
            art.write_root_velocity_to_sim(
                torch.zeros(n, 6, device=dev), env_ids=env_ids)
        # ---- 残差/记账状态 ----
        for oi in (0, 1):
            pose_l = torch.zeros(n, 7, device=dev)
            for i, osrc in enumerate(objsrc):
                pose_l[i] = self.rest_pose[oi] if osrc == "rest" \
                    else self.PB.ref_obj[oi][rows_ia[i]]
            self.hold_pose[oi][env_ids] = pose_l
        self.hold_left[env_ids] = self.HOLD_K
        self.cum_res[env_ids] = 0.0
        self.rebase_off[env_ids] = 0.0
        # seam2_ret 进入=精确落在母带行, 无需换基; 其余武装待捕获
        armed = torch.tensor([self.entries[p][0] < self.EXIT0 for p in pick],
                             device=dev)
        self.rebase_armed[env_ids] = armed
        self.grasp_d0[env_ids] = float("nan")
        self._prev_armq[env_ids] = self.ref58[rows_env_t][:, :14]
        self.d6_acc[env_ids] = 0.0
        self.pad_pot_max[env_ids] = 0.0
        self.lift_hold[env_ids] = 0
        self.lift_done[env_ids] = False
        self._prev_d[env_ids] = 0.0
        self._prev_pf[env_ids] = 0.0
        self._slip_obs[env_ids] = 0.0
        self.last_act[env_ids] = 0.0
