"""双臂原生底座 v2 —— 侧命名空间 + 属性路由(2026-08-18 立项, 用户裁定"底层按双臂写")。

## 与 v1 (bimanual_env.py) 的本质区别

v1 是"快照换入换出": 每次切侧把 ~200 个属性 setattr 进 env、用完再抄回来。
四类结构性事故全部源于这套机制:
  ① 快照时机错(构造时还不存在的字段被 hasattr 跳过 → 两侧静默共用)
  ② 引用别名(SideState 存引用, 原地写互相覆盖 → 必须事后深拷贝补救)
  ③ 忘了重建(B 侧限位/碰撞体/起点池要人肉列清单, 漏一个 = 静默串台)
  ④ 写回丢失(离开上下文忘写回 = 该侧更新蒸发)

v2 把侧状态**永久放在各自命名空间**里, `__getattribute__`/`__setattr__` 按当前
激活侧路由 SIDE_ATTRS 里的名字。切侧 = 指针翻转(O(1), 无拷贝):
  ①④ 机制上消失 —— 没有快照、没有写回, 读写永远落在激活侧的命名空间;
  ②   B 侧构造时对 A 命名空间做一次性克隆, 之后两侧张量物理隔离,
      `_audit_sides()` 逐字段查 data_ptr 别名兜底;
  ③   关节 id/限位/碰撞体/table_bids/cross 对 = `_bind_side()` **一个函数原子重建**,
      两侧都调它 —— "忘了重建"从人肉清单变成代码结构。

单边逻辑仍然复用 env.py 的同一份代码(奖励/终止/观测原样跑两遍) —— 这一点与 v1
相同, 是有意的: 两套逻辑各自维护 = 台账里反复出问题的模式。

## 兼容面

* `env._A` / `env._B`: `_SideNS`, 带 `.name` / `.data`(活字典) —— check_ff_bi 等
  工具的 `side.data[...]` 读法不变。
* `BM.use_side(env, side)`: 对 v2 实例自动分派到 `env._use(side)`(见 bimanual.py)。
* SIDE_ATTRS 仍是唯一真理来源(路由集合直接由它生成), `assert_covered`/
  `list_uncovered` 照常跑: v2 里侧字段不进 vars(env), 扫出来的可疑字段就是
  "该分侧却没进清单"的新增字段, 语义比 v1 更干净。

## 已知不变的雷 (与 v1 相同)

B 侧 q_ref(人手参考)没有换侧重建 —— 人手重建只有右手。任何"人手首帧起步"的双臂
配置都会把左臂摆成右臂的姿势 (2026-08-18 PGA 实锤: 腕在 1.9m 高空)。构造尾部
硬拦截: 必须 retract_start 或纯直接抓取起步。
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence

import numpy as np
import torch

from tasks.pregrasp import bimanual as BM
from tasks.pregrasp.env import GraspTaskEnv

# 路由集合 = SIDE_ATTRS + 动作缓冲。
# actions_buf/prev_actions 在 BM.SHARED_OK 里(v1 靠手工塞进 SideState.data 换),
# v2 里必须进路由: 父类 `_pre_physics_step` 会 copy_ 单侧 7/29 维切片, 共用全宽
# 缓冲直接维度崩 (v1 注释里 2026-08-17 冒烟实测)。
_SIDE = frozenset(BM.SIDE_ATTRS) | {"actions_buf", "prev_actions"}


def _qmul_np(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


class _SideNS:
    """一只手的常驻状态容器。`.data` 是活字典 (工具脚本兼容 v1 的 side.data 读法)。"""

    __slots__ = ("name", "data")

    def __init__(self, name: str):
        self.name = name
        self.data = {}

    def keys(self):
        return self.data.keys()


class BimanualNativeEnv(GraspTaskEnv):
    """两只手各自接近/抓取自己的物体。动作 = [A 侧, B 侧] 拼接。"""

    _bi_native = True          # BM.use_side 靠这个分派

    # ---------------------------------------------------------------- 属性路由
    def __getattribute__(self, name):
        if name in _SIDE:
            # `_sig` 一名两义: 侧内(奖励/终止代码) = 本侧信号; 侧外(评测/工具) =
            # 双侧合并版。v1 靠"换出后 env 上留的是合并版"实现, v2 用深度区分:
            # 所有内部访问都发生在 _use 上下文里 (depth>0), 深度 0 的只有外部读者。
            # 不区分的后果: A 侧奖励读到 B 的 newly_arrive (串奖励), 或评测读到
            # A 单侧的 newly_success (成功率口径错)。
            if name == "_sig":
                try:
                    if object.__getattribute__(self, "_use_depth") == 0:
                        m = object.__getattribute__(self, "_sig_merged")
                        if m is not None:
                            return m
                except AttributeError:
                    pass
            cur = object.__getattribute__(self, "_cur")
            try:
                return cur.data[name]
            except KeyError:
                pass
            # 类层面默认值仍可见 (如 env.py 的 `_fgate_dg = None` 类声明 —— 那条
            # 就是为了"实例值别被 __init__ 抹掉"设计的, 这里保持同一语义)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in _SIDE:
            object.__getattribute__(self, "_cur").data[name] = value
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in _SIDE:
            object.__getattribute__(self, "_cur").data.pop(name, None)
        else:
            object.__delattr__(self, name)

    @contextlib.contextmanager
    def _use(self, side):
        """激活某一侧。side 可以是 _SideNS 或侧名。无拷贝、无写回 —— 指针翻转。"""
        ns = side if isinstance(side, _SideNS) else self._ns[side]
        prev = object.__getattribute__(self, "_cur")
        object.__setattr__(self, "_cur", ns)
        object.__setattr__(self, "_use_depth",
                           object.__getattribute__(self, "_use_depth") + 1)
        try:
            yield
        finally:
            object.__setattr__(self, "_cur", prev)
            object.__setattr__(self, "_use_depth",
                               object.__getattribute__(self, "_use_depth") - 1)

    # ---------------------------------------------------------------- 构造
    def __init__(self, cfg, render_mode=None, **kw):
        A_name = cfg.hand_side
        B_name = "left" if A_name == "right" else "right"
        _A, _B = _SideNS(A_name), _SideNS(B_name)
        # 路由基础设施必须先于 super().__init__ 就位 (父类构造期的侧字段写入都要路由)
        object.__setattr__(self, "_ns", {A_name: _A, B_name: _B})
        object.__setattr__(self, "_cur", _A)
        object.__setattr__(self, "_A", _A)
        object.__setattr__(self, "_B", _B)
        object.__setattr__(self, "_A_name", A_name)
        object.__setattr__(self, "_B_name", B_name)
        object.__setattr__(self, "_use_depth", 0)
        object.__setattr__(self, "_sig_merged", None)

        # ---- 摩擦/触觉镜像 (2026-08-20 用户裁定: 左右数值一致, 数据各自独立) ----
        # ① B 侧 5 指垫传感器: prim 各建各的, filter 指向 Aux(副物体), 与 A 侧零共享;
        # ② B 侧指垫 SuperGrip 名单 (材质绑定在基类 setup 执行, extra_supergrip_bodies);
        # ③ 副物体材质与主物同策略 (aux_grip_parity) —— 否则 SuperGrip(multiply,3.0)
        #    对 ScrewAux(average,0.5) 按 PhysX 优先级 multiply 胜出 = 左垫×杯 1.5,
        #    仍不等于右垫×瓶 9.0。
        _fnA = list(cfg.fingertip_bodies)
        _fnB = [n.replace(f"{A_name}_", f"{B_name}_") for n in _fnA]
        if len(getattr(cfg, "contact_sensors", []) or []) == 5:
            from isaaclab.sensors import ContactSensorCfg as _CSC
            cfg.contact_sensors = list(cfg.contact_sensors) + [
                _CSC(prim_path=f"/World/envs/env_.*/Robot/{n}", history_length=1,
                     filter_prim_paths_expr=["/World/envs/env_.*/Aux"])
                for n in _fnB]
            print(f"[bi-native] 触觉镜像: B 侧 5 垫传感器已建 (filter=Aux) "
                  f"| B 侧 SuperGrip 名单 {len(_fnB)} 项 | aux_grip_parity=ON")
        cfg.extra_supergrip_bodies = _fnB
        cfg.aux_grip_parity = True

        # ---- A 侧: 完全按单臂路径构造 (场景/物体/先验都在这里建好) ----
        super().__init__(cfg, render_mode, **kw)
        # 传感器全表快照 (前5=A侧, 后5=B侧) —— _bind_side 按侧切分, 互不可见
        object.__setattr__(self, "_all_contact_sensors",
                           list(self._contact_sensors))

        prior_b = getattr(cfg, "prior_b_npz", "")
        yaw_b = float(getattr(cfg, "prior_b_yaw_deg", -1.0))
        assert prior_b, "双臂 env 需要 cfg.prior_b_npz (另一只手的 GraspPose)"
        aux_off = getattr(self, "aux_rel_offset_np", None)
        assert aux_off is not None, \
            "这条 clip 没有第二个物体 (aux) —— 双臂各抓各的无从谈起"

        # 构造期可能还不存在的运行时字典 —— 显式建空 (v1 的 hasattr 跳过教训)
        for _k in ("_sig", "_ep_sums"):
            if _k not in _A.data:
                _A.data[_k] = {}

        A_obj = self.obj_init_pos.cpu().numpy().astype(np.float64)
        B_obj = A_obj + np.asarray(aux_off, np.float64)

        # ---- B 命名空间 = A 的一次性隔离拷贝 (没重建的字段继承 A 值, 与 v1 语义一致;
        #      张量全克隆 ⟹ 两侧从此物理隔离, 原地写不可能互相覆盖) ----
        _cloned = 0
        for _k, _v in _A.data.items():
            if torch.is_tensor(_v):
                _B.data[_k] = _v.clone()
                _cloned += 1
            elif isinstance(_v, dict):
                _B.data[_k] = dict(_v)
            elif isinstance(_v, list):
                _B.data[_k] = list(_v)
            else:
                _B.data[_k] = _v
        print(f"[bi-native] B 侧命名空间就绪: 继承 {len(_B.data)} 项 (张量克隆 {_cloned})")

        # ---- B 侧原生构造 (与 v1 同一套已验证的数学, 状态机制换成路由) ----
        with self._use(_B):
            self._build_side_b(cfg, prior_b, yaw_b, A_obj, B_obj, aux_off)
        # ---- A 侧对称重绑 (两侧的关节/限位/碰撞体走同一个函数 —— 对称性由结构保证) ----
        with self._use(_A):
            self._bind_side(A_name)

        # ---- 动作缓冲逐侧独立, 宽度 = 单侧动作维 (共用会让父类 copy_ 维度崩) ----
        _n = int(cfg.action_space) // 2
        for _sd in (_A, _B):
            _sd.data["actions_buf"] = torch.zeros(self.num_envs, _n, device=self.device)
            _sd.data["prev_actions"] = torch.zeros(self.num_envs, _n, device=self.device)
        print(f"[bi-native] 动作缓冲逐侧独立: 每侧 {_n} 维")

        # ---- cuRobo 前馈参考: 逐侧接入 (与 v1 同款) ----
        _ref = getattr(cfg, "curobo_ref_npz", "")
        if _ref:
            _z = np.load(_ref, allow_pickle=True)
            _st = max(1, int(getattr(cfg, "curobo_ref_stride", 2)))
            for _nm in (A_name, B_name):
                _q = np.asarray(_z[f"{_nm}_q"], np.float32)[::_st].copy()
                _p = torch.tensor(_q, dtype=torch.float32, device=self.device)
                _ns = self._ns[_nm]
                _q_st = self.hand.data.default_joint_pos[0, _ns.data["arm_jids"]]
                _d0 = float((_p[0] - _q_st).abs().max())
                assert (not getattr(cfg, "ref_start_is_stance", True)) or _d0 < 0.02, (
                    f"{_nm} 侧参考首帧与站姿差 {np.degrees(_d0):.2f}° "
                    f"—— 起点会跳变, 参考文件与场景不配套 "
                    f"(carry 等首行≠站姿的参考需 cfg.ref_start_is_stance=False)")
                _ns.data["retract_path"] = _p
            # ---- 段表: 按"数据有没有"派生, 不按旗 ----
            # ★ 2026-08-26 修 (RL_Pour 会诊发现): 原来段表只在 fin_ref_npz 分支里派生,
            #   ⟹ 不开指参考的旗集 (e2e) 即使 npz 带了 seg_names 也没人读, 于是
            #   ⑤e/⑥c/⑥d 三条预检**拿不到段就打空表, 看起来像"检查通过"**。
            #   这与"数据源关掉 ⟹ 守门员静默跳过"同族: 派生器的触发条件必须是
            #   数据可得性, 不是某面无关的功能旗。seg_gate 的长度断言仍只在 seg_gate 时活。
            if "seg_names" in _z.files and "seg_lens" in _z.files:
                _acc0, _sg0 = 0, []
                for _sn0, _sl0 in zip(_z["seg_names"], _z["seg_lens"]):
                    _sg0.append((str(_sn0), _acc0 // _st,
                                 (_acc0 + int(_sl0) - 1) // _st))
                    _acc0 += int(_sl0)
                for _nm0 in (A_name, B_name):
                    self._ns[_nm0].data["_seg_rows"] = list(_sg0)
                print(f"[seg] 段表已从 curobo_ref 派生: {len(_sg0)} 段 "
                      f"(stride {_st}, 播放行空间) —— 供 ⑤e/⑥c/⑥d 切段")
            if getattr(cfg, "fin_ref_npz", ""):
                # AAG: 指参考逐侧装载 ({side}_q29[:,7:29]) —— ref_t 同步索引
                _zf = np.load(cfg.fin_ref_npz, allow_pickle=True)
                for _nm2 in (A_name, B_name):
                    _ns2 = self._ns[_nm2]
                    with self._use(_ns2):
                        # ★ 2026-08-23 修 (影响 AAG 全部八代 v3->v3.7):
                        # 原来直接 [7:29] 装载, **漏了 _generic_perm 换序**。
                        # npz 的 22 维是 GENERIC_JOINT_ORDER 序 —— 实测
                        # close末(行299) ≡ prior['grasp'] 0.00°、
                        # squeeze末(行339) ≡ prior['squeeze'] 0.00°, 与 prior 的
                        # 裸 [7:29] 同序, 而 prior 加载器一律要过 _generic_perm。
                        # 漏换序 = 手型全串位 (env.py 2026-07-30 同款坑复发):
                        # 参考里指关节行程峰值 97°/均 34°, 串位后 GUI 里看着就是
                        # "手指根本不会弯", 而 fin_track/form_pot 全程在拿串位
                        # 目标当真值罚/奖。
                        _f22 = np.asarray(_zf[f"{_nm2}_q29"],
                                          np.float32)[::_st, 7:29]
                        _f22 = _f22[:, self._generic_perm]
                        _f22 = np.clip(_f22, self.dof_lower[0].cpu().numpy(),
                                       self.dof_upper[0].cpu().numpy())
                        _ns2.data["_fin_ref_path"] = torch.tensor(
                            _f22, dtype=torch.float32, device=self.device)
                        # 段表(播放行空间), 供分段探索门控用
                        if "seg_names" in _zf.files and "seg_lens" in _zf.files:
                            _acc, _sg = 0, []
                            for _sn, _sl in zip(_zf["seg_names"], _zf["seg_lens"]):
                                _sg.append((str(_sn), _acc // _st,
                                            (_acc + int(_sl) - 1) // _st))
                                _acc += int(_sl)
                            _ns2.data["_seg_rows"] = _sg
                        _trv = np.degrees(np.abs(_f22[-1] - _f22[0]))
                        print(f"[aag] {_nm2} 指参考换序后行程: 峰 {_trv.max():.1f}° "
                              f"均 {_trv.mean():.1f}° (串位时这两个数会明显变小)")
                print(f"[aag] 指参考逐行 ON: {cfg.fin_ref_npz} "
                      f"({_f22.shape[0]} 行/侧, stride {_st}, GENERIC->USD 换序已应用)")
            print(f"[curobo_ref] ✅ 前馈已接: A({A_name}) "
                  f"{tuple(_A.data['retract_path'].shape)} | B({B_name}) "
                  f"{tuple(_B.data['retract_path'].shape)} | stride {_st} | 首帧=站姿 已验证")

        # ---- 防雷 (与 v1 同): B 侧 q_ref 从未重建, 人手首帧起步 = 左臂乱摆 ----
        assert getattr(cfg, "retract_start", False) or \
            float(getattr(cfg, "direct_grasp_prob", 0.0)) >= 1.0, (
            "双臂环境必须开退避起点族 (approach_only 自动开; 完整任务加 --retract) "
            "—— B 侧 q_ref 未重建, 人手首帧起步 = 左臂乱摆 (2026-08-18 PGA 事故)")

        if getattr(cfg, "fin_start_pose1", False):
            # FC-C (2026-08-19 定稿, DESIGN_LOOP 同日晚): 每侧 q_open ← 该侧先验
            # pregrasp[0] (Pose1, 拇指对掌) ⟹ reset 从 Pose1 出发, 合拢轴变成
            # Pose1→Pose2 纯四指卷握走廊; 拇指横扫(65~79°)不再发生在物体旁。
            # q_close 不动(仍=合拢锚), fin_quiet 语义自动变为"保持 Pose1"。
            for _ns, _pth in ((self._A, cfg.grasp_prior_npz),
                              (self._B, cfg.prior_b_npz)):
                _zp = np.load(_pth, allow_pickle=True)
                with self._use(_ns):
                    _p1 = np.asarray(_zp["pregrasp"][0],
                                     np.float64)[7:29][self._generic_perm]
                    _p1 = np.clip(_p1, self.dof_lower[0].cpu().numpy(),
                                  self.dof_upper[0].cpu().numpy())
                    self.q_open = torch.tensor(_p1, dtype=torch.float32,
                                               device=self.device)
                    _dg = float(torch.rad2deg(
                        (self.q_open - self._p2_fin).abs().mean()))
                    print(f"[fin_pose1] {_ns.name} 侧 q_open ← pregrasp[0] "
                          f"(Pose1) | 距抓形均值 {_dg:.1f}°")

        if getattr(cfg, "fin_open_from_ref", False):
            # 编舞/E2E (2026-08-26, RL_Training 方案 fin_pose1 同机制换源):
            # 每侧 q_open ← 指参考首行 ⟹ tmpl0 = q_open + 0·(...) = ref[0],
            # 出生即参考起始手型(绷直 2.16°); closure=0 语义仍=完全张开,
            # obs 不说谎(耦合A), 自动对表 _q_init 自动跟对(耦合B)。
            # ref[0] 比原 q_open 更张开 ⟹ 合拢行程只增不减, 不挤走廊。
            for _ns in (self._A, self._B):
                with self._use(_ns):
                    _fr0 = getattr(self, "_fin_ref_path", None)
                    if _fr0 is not None:
                        self.q_open = _fr0[0].clone()
                        print(f"[fin_open_from_ref] {_ns.name} 侧 q_open ← "
                              f"指参考首行 (|均值| {float(torch.rad2deg(self.q_open.abs().mean())):.2f}°)")
                    else:
                        print(f"[fin_open_from_ref] ⚠ {_ns.name} 侧无指参考"
                              f"(fin_ref_npz 未装载), q_open 保持原样")

        if getattr(cfg, "fin_ref_track", False):
            # FC-D (2026-08-20): 逐侧构建指参考关键帧表 K(8,22)=[open,r0..r5,grasp]
            # 与四段状态缓冲 (全部 SIDE_ATTRS 路由, 左右零共享)。
            for _ns, _pth in ((self._A, cfg.grasp_prior_npz),
                              (self._B, cfg.prior_b_npz)):
                _zk = np.load(_pth, allow_pickle=True)
                with self._use(_ns):
                    _lo = self.dof_lower[0].cpu().numpy()
                    _hi = self.dof_upper[0].cpu().numpy()
                    _rows = [self.q_open.cpu().numpy()]
                    for _i in range(6):
                        _r = np.asarray(_zk["pregrasp"][_i],
                                        np.float64)[7:29][self._generic_perm]
                        _rows.append(np.clip(_r, _lo, _hi))
                    _rows.append(self._p2_fin.cpu().numpy())
                    self._fcd_keys = torch.tensor(np.stack(_rows), dtype=torch.float32,
                                                  device=self.device)
                    self._fcd_close_t = torch.zeros(self.num_envs, dtype=torch.long,
                                                    device=self.device)
                    self._fcd_shape_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                                       device=self.device)
                    self._fcd_pad_done = torch.zeros(self.num_envs, 5, dtype=torch.bool,
                                                     device=self.device)
                    self._fcd_ref_last = self.q_open.unsqueeze(0).expand(
                        self.num_envs, 22).clone()
                    self._fcd_cand_ema = 0.0     # 参考退火驱动 (candidate 率慢 EMA)
                    self._fcd_g = 0.0            # 退火进度 (棘轮, 0=全拉力 1=归零)
                    print(f"[fcd] {_ns.name} 侧指参考 K(8,22) 就位 | open→Pose1 均差 "
                          f"{float(torch.rad2deg((self._fcd_keys[1] - self._fcd_keys[0]).abs().mean())):.1f}°"
                          f" | r5→grasp 均差 "
                          f"{float(torch.rad2deg((self._fcd_keys[7] - self._fcd_keys[6]).abs().mean())):.1f}°")

        # ★ 2026-08-25: 两侧构造全部完成后**统一重装相位表** —— 详见
        #   GraspTaskEnv._reinstall_phase_table 的说明(构造期两次写可能落到不同侧)。
        for _nsP in (self._A, self._B):
            with self._use(_nsP):
                self._reinstall_phase_table()
        _pt = {}
        for _nsP in (self._A, self._B):
            with self._use(_nsP):
                _pt[_nsP.name] = [int(x) for x in self.phase_timeout_t.tolist()]
        print(f"[bi-native] 相位表已统一重装: "
              + " | ".join(f"{k} PREGRASP={v[0]}" for k, v in _pt.items())
              + f" | ep_total={self.ep_total}")
        assert len(set(map(tuple, _pt.values()))) == 1, (
            f"两侧相位表仍不一致(重装后不该发生): {_pt}")

        self._audit_sides()
        BM.assert_covered(self)
        _unc = BM.list_uncovered(self)
        if _unc:
            print(f"[bi-native] ⚠ 未入清单的张量类字段 {len(_unc)} 个 (v2 里它们是共享的; "
                  f"该分侧的补进 SIDE_ATTRS):")
            for _k, _t in _unc:
                print(f"           {_k:28s} {_t}")

    # ------------------------------------------------------------ 原子换侧
    def _bind_side(self, side_name: str):
        """一个函数重建**全部**关节/body 派生状态。v1 的三次事故(限位/碰撞体/
        table_bids 各漏一次)都是因为这坨状态散在多处、靠人记 —— 收拢成原子操作。"""
        self._resolve_joint_ids(force_side=side_name)
        # 关节限位: 左右臂**镜像非对称** (L_j2 [-26,+89] vs R_j2 [-89,+26]),
        # _resolve_joint_ids 只换 jids 不换表 (2026-08-18 左臂恒差 19° 实锤)
        _lim = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.arm_lower = _lim[..., 0][:, self.arm_jids]
        self.arm_upper = _lim[..., 1][:, self.arm_jids]
        _bn = list(self.hand.body_names)
        P = "R" if side_name == "right" else "L"
        O = "L" if P == "R" else "R"
        _to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                     device=self.device)
        self._build_collide_ids(_bn, P, O, _to)
        # 与桌/物做碰撞检查的**本侧**手部连杆 (手 + l7/l8/ee)
        self.table_bids = [i for i, n in enumerate(_bn)
                           if n.startswith(f"{side_name}_")
                           or n in (f"{P}_arm_l7", f"{P}_arm_l8", f"{P}_ee")]
        # ⚠ 手指列表与 env.py:202 逐字一致
        self.cross_a = [_bn.index(f"{side_name}_{f}_DP")
                        for f in ("index", "middle", "ring")]
        self.cross_b = [_bn.index(f"{side_name}_{f}_DP")
                        for f in ("middle", "ring", "pinky")]
        # 指垫连杆逐侧 (2026-08-19 铁案): cfg.fingertip_bodies 写死 right_*,
        # tip_ids 只在基类构造时解析一次 ⟹ B 侧一直在量**右手**指垫 —— 历史双臂
        # 训练的左手 pad 类信号(指垫距离/接触门控/FC 判据)全部串台。按侧重解析。
        self.tip_ids = [_bn.index(n.replace("right_", f"{side_name}_"))
                        for n in self.cfg.fingertip_bodies]
        # 触觉传感器逐侧 (2026-08-20 镜像案): 全表前5=A(交互手, filter=Object),
        # 后5=B(镜像手, filter=Aux)。各侧只见自己的 5 个, 杜绝左右串线。
        _all_cs = getattr(self, "_all_contact_sensors", None)
        if _all_cs is not None and len(_all_cs) == 10:
            self._contact_sensors = (list(_all_cs[:5])
                                     if side_name == self._A_name
                                     else list(_all_cs[5:]))

    # ------------------------------------------------------------ B 侧构造
    def _build_side_b(self, cfg, prior_b, yaw_b, A_obj, B_obj, aux_off):
        """在 B 命名空间激活的前提下跑。数学与 v1 (bimanual_env.py) 逐行同源。"""
        from rl_rebuild.correction.kinematics import quat_to_R
        _to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                     device=self.device)
        self._bind_side(self._B_name)

        zb = np.load(prior_b)
        a = np.radians(yaw_b)
        yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
        oq_b = _qmul_np(yq, np.asarray(zb["canon_rot"], np.float64))
        B_gp = quat_to_R(oq_b) @ np.asarray(zb["grasp"][:3], np.float64) + B_obj
        B_gq = _qmul_np(oq_b, np.asarray(zb["grasp"][3:7], np.float64))
        self._grasp_pos_w = _to(B_gp)
        self._grasp_quat_w = _to(B_gq)
        self.obj_init_pos = _to(B_obj)
        # 裁定B3: B 侧 PreGrasp = 抓姿沿"接触质心→腕"方向平移 pregrasp_palm_cm (掌心锚点)
        _zbg = np.asarray(zb["grasp"], np.float64)
        _zbc = np.asarray(zb["contact_centroid"], np.float64)
        _ub = _zbg[:3] - _zbc
        _ub = _ub / max(np.linalg.norm(_ub), 1e-9)
        _pcm = float(getattr(cfg, "pregrasp_palm_cm", 5.0)) / 100.0
        self._pregrasp_w = [
            (quat_to_R(oq_b) @ (_zbg[:3] + _pcm * _ub) + B_obj,
             _qmul_np(oq_b, _zbg[3:7]))]
        if getattr(cfg, "pregrasp29", False):
            # PreGrasp29: B 侧靶点同样整体换成掌心 PreGrasp (与 A 侧口径一致)
            self._grasp_pos_w = _to(self._pregrasp_w[0][0])
            self._grasp_quat_w = _to(self._pregrasp_w[0][1])
        if getattr(cfg, "pregrasp_phase2", False):
            # Phase2 B 侧: 腕 PreGrasp→B_gp 直线逐帧 IK (cuRobo 左手末帧当种子);
            # 指型终点 = zb 抓姿 22 关节。
            from rl_rebuild.correction.kinematics import ArmIK as _ArmIK
            from rl_rebuild.correction.kinematics import quat_to_R as _q2R
            _gqn2 = np.asarray(B_gq, np.float64)
            _gqn2 = _gqn2 / max(np.linalg.norm(_gqn2), 1e-12)
            _Rb2 = _q2R(_gqn2)
            _ikb2 = _ArmIK(self._B_name, anchor_link="arm_center",
                           anchor_T=self._anchor_T)
            _zr2 = np.load(cfg.curobo_ref_npz)
            _seed2 = np.asarray(_zr2["left_q"], np.float64)[-1]
            _pp0b = np.asarray(self._pregrasp_w[0][0], np.float64)
            _K2b = int(getattr(cfg, "phase2_steps", 50))
            _rows2, _errs2 = [], []
            for _k in range(_K2b):
                _al = _k / max(_K2b - 1, 1)
                _pt = (1 - _al) * _pp0b + _al * np.asarray(B_gp, np.float64)
                _rk = _ikb2.solve(_pt, _Rb2, q0=_seed2, iters=200)
                _seed2 = _rk["q"]
                _rows2.append(_rk["q"])
                _errs2.append(_rk["pos_err"])
            self._p2_arm = _to(np.stack(_rows2).astype(np.float32))
            _fg2 = np.clip(np.asarray(zb["grasp"], np.float64)[7:29][self._generic_perm],
                           self.dof_lower[0].cpu().numpy(),
                           self.dof_upper[0].cpu().numpy())
            self._p2_fin = _to(_fg2.astype(np.float32))
            # m2 派生阶梯 (裁定C, 与 A 侧 env.py 同款): 按 B 侧自己的起始偏差铺
            _d0b = float((self.q_open - self._p2_fin).abs().mean())
            _stepb = np.radians(float(getattr(cfg, "phase2_ladder_step_deg", 5.0)))
            _ladb, _rb = [], _d0b - _stepb
            while _rb > 0.35 + 1e-6:
                _ladb.append(_rb)
                _rb -= _stepb
            _ladb.append(0.35)
            self._m2_ladder = _to(np.asarray(_ladb, np.float32))
            self._m2_lvl = torch.zeros(self.num_envs, dtype=torch.long,
                                       device=self.device)
            print(f"[phase2] m2 阶梯(B): 起始偏差 {np.degrees(_d0b):.1f}° -> "
                  f"{[f'{np.degrees(x):.1f}°' for x in _ladb]}")
            if getattr(cfg, "fin_cart", False):
                self._fc_local = None                            # 待测 (物理测量, 见 _fc_measure)
                self._fc_done = torch.zeros(self.num_envs, 5, dtype=torch.bool,
                                            device=self.device)
                self._fc_prev = torch.full((self.num_envs, 5), float("nan"),
                                           device=self.device)
            self._p2_t = torch.zeros(self.num_envs, dtype=torch.long,
                                     device=self.device)
            self._g2_pos_w = _to(np.asarray(B_gp, np.float64))
            self._g2_quat_w = _to(_gqn2)
            self._g2_run = torch.zeros(self.num_envs, dtype=torch.long,
                                       device=self.device)
            self._g2_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            self._m1_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            self._m2_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                        device=self.device)
            print(f"[phase2] B 侧就绪: 真抓姿靶点+抓姿指型 | (斜坡 {_K2b} 帧留档, "
                  f"IK 误差 max {max(_errs2)*100:.2f}cm)")
        # ★ B 侧的**物体** = 第二个刚体 (奖励的 _target_w 读 object.data, 不是靶点)
        assert getattr(self, "aux", None) is not None, "aux 刚体不存在, B 侧没有目标物体"
        self.object = self.aux
        zb_c = zb["contact_centroid"] if "contact_centroid" in zb.files else None
        if zb_c is not None:
            self.aff_local = _to(np.asarray(zb_c, np.float64))
        # ★ B 侧物体表面点云按**杯网格**重建 (2026-08-19 铁案: 此前继承 A 的瓶点云,
        #   左手表面距离参照的是"摆在杯位置的幻影瓶")
        from rl_rebuild.correction import clips as _clips
        _sec = (_clips.clip_entry(cfg.clip_name).get("secondary") or {}).get("mesh")
        if _sec:
            import trimesh as _tm
            _m = _tm.load(_sec, force="mesh")
            _pts, _fidx = _tm.sample.sample_surface(_m, int(cfg.n_obj_points))
            self.obj_points = _to(np.asarray(_pts, np.float64))
            print(f"[bi-native] B 侧 obj_points 已按杯网格重建 "
                  f"({int(cfg.n_obj_points)} 点, {os.path.basename(_sec)})")
        # ★ 起点池按 B 侧重建 (A 池存的是右臂关节角, 套到左臂 = 非法位形)
        if getattr(cfg, "start_pool", ""):
            self._sp_init()
            print(f"[bi-native] B 侧起点池已按 {self._B_name} 臂重建")
        # ---- 五件套 (完整抓取链才需要): q_close / q_pregrasp / 左臂抓姿 IK /
        #      抬升斜坡 / _fgate_dg —— 与 v1 逐行同源 ----
        if getattr(cfg, "l5_couple", False) or (
                getattr(cfg, "pregrasp29", False)
                and not getattr(cfg, "approach_only", False)):
            from rl_rebuild.correction.kinematics import ArmIK
            self.obj_init_quat = _to(oq_b)
            _gqn = np.asarray(B_gq, np.float64)
            _gqn = _gqn / max(np.linalg.norm(_gqn), 1e-12)
            _Rb = quat_to_R(_gqn)
            _ikb = ArmIK(self._B_name, anchor_link="arm_center",
                         anchor_T=self._anchor_T)
            _zr = np.load(cfg.curobo_ref_npz)
            _q_near = np.asarray(_zr["left_q"], np.float64)[-1]
            # ⚠ 必须用 cuRobo 左手末帧当 IK 种子 (冷启动掉错误盆地, 实测 4.53cm)
            _rg = _ikb.solve(np.asarray(B_gp, np.float64), _Rb,
                             q0=_q_near, iters=300, pos_tol=2e-4)
            _fk_near = _ikb.fk(_q_near)[0]
            assert np.isfinite(_rg["q"]).all() and _rg["pos_err"] < 0.01, (
                f"B 侧抓姿 IK 失败 (err {_rg.get('pos_err', 1.0)*100:.2f}cm)\n"
                f"  诊断: FK(近点种子)={np.round(_fk_near, 3)} | "
                f"B_gp={np.round(np.asarray(B_gp, np.float64), 3)} | "
                f"|FK(近)−B_gp|={np.linalg.norm(_fk_near - np.asarray(B_gp)) * 100:.2f}cm "
                f"(应≈几cm; 大偏差=anchor_T 或坐标系错)\n"
                f"  A_obj={np.round(A_obj, 3)} aux_off={np.round(np.asarray(aux_off, np.float64), 3)} "
                f"B_obj={np.round(B_obj, 3)} oq_b={np.round(oq_b, 3)} yaw_b={yaw_b}")
            self._prior_q_grasp = _to(_rg["q"])
            self.q_pregrasp = _to(_q_near.astype(np.float32))
            _fsrc = (np.asarray(zb["close_anchor"], np.float64)
                     if "close_anchor" in zb.files
                     else np.asarray(zb["grasp"], np.float64)[7:29])
            _fb = np.clip(_fsrc[self._generic_perm],
                          self.dof_lower[0].cpu().numpy(),
                          self.dof_upper[0].cpu().numpy())
            self.q_close = _to(_fb)
            self._fgate_dg = float(np.linalg.norm(
                np.asarray(B_gp, np.float64) - np.asarray(B_obj, np.float64)))
            # 抬升斜坡 (收紧 pos_tol + 雅可比补步 + 实抬核验, 与 env.py:936 同款)
            lo_np = self.arm_lower[0].cpu().numpy().astype(np.float64)
            hi_np = self.arm_upper[0].cpu().numpy().astype(np.float64)
            lift, qw, _nf = [], _rg["q"].copy(), 0
            for _i in range(cfg.lift_steps + 1):
                _tgt = np.asarray(B_gp, np.float64) + np.array(
                    [0.0, 0.0, cfg.lift_height * _i / max(cfg.lift_steps, 1)])
                _r = _ikb.solve(_tgt, _Rb, q0=qw, iters=300, pos_tol=2e-4)
                if _r["ok"]:
                    qw = _r["q"].copy()
                elif _i > 0:
                    _nf += 1
                    _J = _ikb.jacobian(qw)
                    _dp = _tgt - _ikb.fk(qw)[0]
                    _dq = np.linalg.lstsq(_J[:3], _dp, rcond=None)[0]
                    qw = np.clip(qw + _dq, lo_np, hi_np)
                else:
                    qw = _r["q"].copy()
                lift.append(qw.copy())
            self.q_lift = _to(np.stack(lift))
            self.q_lift_delta = self.q_lift - self.q_lift[0:1]
            _lv = int(round(min(cfg.verify_lift_m
                                / (cfg.lift_height / max(cfg.lift_steps, 1)),
                                cfg.lift_steps)))
            _rise = float(_ikb.fk(lift[_lv])[0][2] - _ikb.fk(lift[0])[0][2])
            assert _rise >= 0.9 * cfg.verify_lift_m, (
                f"B 侧抬升斜坡只抬得动 {_rise*1000:.2f}mm —— 左臂竖直 IK 病态")
            print(f"[bi-native] B 侧五件套: 抓姿IK {_rg['pos_err']*100:.2f}cm | "
                  f"近点=cuRobo 左手末帧 | 指模板 {os.path.basename(prior_b)} | "
                  f"抬升第{_lv}级实抬 {_rise*1000:.1f}mm (IK补步 {_nf}) | "
                  f"d_g={self._fgate_dg*100:.1f}cm | ⚠ 参与指分类沿用 A 侧")

    # ------------------------------------------------------------ 构造终审计
    def _audit_sides(self):
        """构造对称性的机器审计 —— v1 时代靠人眼盯打印, 现在全部 assert。"""
        A, B = self._A, self._B
        # ① 内存别名: 两侧任何同名张量不得共享存储 (v1 的"原地写互相覆盖"事故类)
        _alias = [k for k in A.data.keys() & B.data.keys()
                  if torch.is_tensor(A.data[k]) and torch.is_tensor(B.data[k])
                  and A.data[k].numel() > 0
                  and A.data[k].data_ptr() == B.data[k].data_ptr()]
        assert not _alias, f"两侧张量共享同一块内存: {_alias} —— 原地写会互相覆盖"
        # ② 必须相异的字段
        _ov = set(map(int, A.data["arm_jids"])) & set(map(int, B.data["arm_jids"]))
        assert not _ov, f"两侧臂关节重叠 {sorted(_ov)} —— 会互相覆盖"
        assert A.data["ee_id"] != B.data["ee_id"], (
            f"两侧末端 body 相同 (ee_id={A.data['ee_id']}) —— 测量会串台")
        assert A.data["object"] is not B.data["object"], \
            "两侧目标物体是同一个 —— 奖励会串台"
        _csA = A.data.get("_contact_sensors"); _csB = B.data.get("_contact_sensors")
        if _csA is not None and _csB is not None:
            assert len(_csA) == 5 and len(_csB) == 5, (
                f"逐侧触觉传感器应各 5 个, 实际 A={len(_csA)} B={len(_csB)}")
            assert not (set(map(id, _csA)) & set(map(id, _csB))), (
                "两侧触觉传感器有共享对象 —— 左右串线 (2026-08-20 镜像案红线)")
        assert not torch.equal(A.data["arm_lower"], B.data["arm_lower"]), \
            "两侧臂限位表完全相同 —— DexMate 左右臂限位是镜像非对称的, 相同=没换侧"
        assert set(A.data["table_bids"]) != set(B.data["table_bids"]), \
            "两侧 table_bids 相同 —— 碰撞检查在用同一只手"
        assert list(A.data["tip_ids"]) != list(B.data["tip_ids"]), \
            "两侧 tip_ids 相同 —— 左手在量右手的指垫 (2026-08-19 铁案, 不许回归)"
        if torch.is_tensor(A.data.get("obj_points")) and \
                torch.is_tensor(B.data.get("obj_points")):
            assert not torch.equal(A.data["obj_points"], B.data["obj_points"]), \
                "两侧 obj_points 逐位相同 —— B 在用幻影瓶点云"
        _da = float(torch.norm(A.data["_grasp_pos_w"] - B.data["_grasp_pos_w"]))
        assert _da > 0.05, f"两侧靶点只差 {_da*100:.1f}cm —— 八成串台了"
        print(f"[bi-native] 审计通过: 别名 0 | A={A.name} 臂 {list(A.data['arm_jids'])} | "
              f"B={B.name} 臂 {list(B.data['arm_jids'])} | 靶距 {_da*100:.1f}cm | "
              f"末端 A={A.data['ee_body']} B={B.data['ee_body']}")

    # ---------------------------------------------------------------- 双跑包装
    # (语义与 v1 逐行同源 —— 差别只在切侧从快照换入换出变成指针翻转)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        n = actions.shape[1] // 2
        assert actions.shape[1] == 2 * n, f"动作维度 {actions.shape[1]} 不是偶数"
        _k = getattr(self, "_bi_dbg", 0)
        _dbg = _k < 400 and _k % 40 == 0
        for _tag, _side, _act in ((self._A_name[0].upper(), self._A, actions[:, :n]),
                                  (self._B_name[0].upper(), self._B, actions[:, n:])):
            with self._use(_side):
                _t0 = self.arm_tgt.clone() if hasattr(self, "arm_tgt") else None
                super()._pre_physics_step(_act)
                if _dbg:
                    _dt = (float((self.arm_tgt - _t0).abs().max())
                           if _t0 is not None else float("nan"))
                    _d = float((self._anchor_w() - self._target_w()).norm(dim=1).mean())
                    print(f"[bi-dbg] 步{_k:4d} {_tag}: 动作 {float(_act.norm(dim=1).mean()):.4f} "
                          f"| 臂目标Δ {_dt*57.3:7.4f}° | 离目标 {_d*100:6.2f}cm "
                          f"| 冻结 {int((self.freeze_ctr > 0).sum()):3d} "
                          f"| 相位 {int(self.task_phase[0])}", flush=True)
        object.__setattr__(self, "_bi_dbg", _k + 1)

    def _apply_action(self) -> None:
        with self._use(self._A):
            super()._apply_action()
        with self._use(self._B):
            super()._apply_action()

    def _get_observations(self) -> dict:
        # 父类 `_check_obs_dim` 拿 cfg.observation_space 比 —— 逐侧调用时换成单侧宽度
        _full = self.cfg.observation_space
        self.cfg.observation_space = int(getattr(self.cfg, "_obs_single", _full // 2))
        try:
            with self._use(self._A):
                oa = super()._get_observations()
            with self._use(self._B):
                ob = super()._get_observations()
        finally:
            self.cfg.observation_space = _full
        out = dict(oa)
        for k in ("policy", "priv_info", "proprio_hist"):
            if k in oa and k in ob and torch.is_tensor(oa[k]):
                out[k] = torch.cat([oa[k], ob[k]], dim=-1)
        return out

    def _get_rewards(self, return_terms: bool = False):
        with self._use(self._A):
            ra = super()._get_rewards(return_terms)
        with self._use(self._B):
            rb = super()._get_rewards(return_terms)
        if return_terms:
            return {"A": ra, "B": rb}      # 逐侧 terms 字典, 调用方各自处理
        return ra + rb          # 无耦合项: 两手各自往自己目标去

    def _get_dones(self):
        with self._use(self._A):
            ta, ua = super()._get_dones()
        with self._use(self._B):
            tb, ub = super()._get_dones()
        _k = getattr(self, "_bi_dbg", 0)
        if _k < 400 and _k % 40 == 0:
            _br = []
            for _tag, _sd in ((self._A_name[0].upper(), self._A),
                              (self._B_name[0].upper(), self._B)):
                _sg = _sd.data.get("_sig") or {}
                _hit = {k: int(v.sum()) for k, v in _sg.items()
                        if hasattr(v, "dtype") and v.dtype == torch.bool and int(v.sum())}
                _br.append(f"{_tag}:{_hit}")
            print(f"[bi-dbg] 步{_k:4d} 终止: A {int(ta.sum()):3d} B {int(tb.sum()):3d} "
                  f"| 回合步 {int(self.episode_length_buf[0])}\n"
                  f"          触发项 {' || '.join(_br)}", flush=True)
        # ---- 合并 _sig + 双臂成功语义 (与 v1 逐行同源) ----
        _sa = dict(self._A.data.get("_sig") or {})
        _sb = self._B.data.get("_sig") or {}
        for _k2, _v in _sb.items():
            if _k2 in _sa and torch.is_tensor(_v) and _v.dtype == torch.bool:
                _sa[_k2] = _sa[_k2] | _v
            elif _k2 not in _sa:
                _sa[_k2] = _v
        # ★★ 2026-08-24 裁定 B (与 env.py:2007 同一处修正, 两边必须同步改):
        #   fin_ref_track + phase2 时也走 arrived & g2_done —— 否则 else 分支要
        #   A/B 两侧的 succeeded, 而那要 candidate→微抬升验证, 在 approach_only
        #   下相位到不了 GRASP, 双臂成功恒 False。
        if getattr(self.cfg, "approach_only", False) and \
                (not getattr(self.cfg, "fin_ref_track", False)
                 or getattr(self.cfg, "pregrasp_phase2", False)):
            # ★ FC-D 走 else 分支: 各侧胜利 = 微抬升验证通过 (真抓稳), 不是
            #   arrived&g2 模板口径 (2026-08-20 用户裁定, 与 env.py 同一处修正)
            _aa = self._A.data.get("arrived")
            _ab = self._B.data.get("arrived")
            if getattr(self.cfg, "pregrasp_phase2", False):
                _ga = self._A.data.get("_g2_done")
                _gb = self._B.data.get("_g2_done")
                # ★ 逐侧 g2 事件计数必须在**这里**做: 活动侧(A)的 _g2_done 挂在
                #   env_raw 上, 成功即终止会当步复位清掉, 训练侧 step 之后再采样
                #   必然读到 0 (实测 right 0 次 vs 成功 123 次自相矛盾, 是探针的错
                #   不是代码的错)。这里是两侧都还没复位的唯一时刻。
                if not hasattr(self, "_g2_evt"):
                    object.__setattr__(self, "_g2_evt", {"A": 0, "B": 0})
                    object.__setattr__(self, "_g2_pv", {})
                for _tg, _gx in (("A", _ga), ("B", _gb)):
                    if _gx is None:
                        continue
                    _p = self._g2_pv.get(_tg)
                    if _p is not None:
                        self._g2_evt[_tg] += int((_gx & ~_p).sum())
                    self._g2_pv[_tg] = _gx.clone()
                if False:
                    print(f"[g2读取] A(_A={self._A.name}) _g2_done="
                          f"{'None!!' if _ga is None else f'ok 均值{float(_ga.float().mean()):.3f}'}"
                          f" | B(_B={self._B.name}) _g2_done="
                          f"{'None!!' if _gb is None else f'ok 均值{float(_gb.float().mean()):.3f}'}"
                          f" | arrived A={'None' if _aa is None else 'ok'}"
                          f" B={'None' if _ab is None else 'ok'}", flush=True)
                if _ga is not None:
                    _aa = _aa & _ga
                if _gb is not None:
                    _ab = _ab & _gb
        else:
            # 各侧"胜利" = 本回合已通过微抬升验证; 双臂成功 = 两侧都通过
            _aa = self._A.data.get("succeeded")
            _ab = self._B.data.get("succeeded")
        if _aa is not None and _ab is not None:
            _both = _aa & _ab
            if not hasattr(self, "_bi_done_once"):
                object.__setattr__(self, "_bi_done_once", torch.zeros_like(_both))
            _sa["newly_success"] = _both & ~self._bi_done_once
            object.__setattr__(self, "_bi_done_once", self._bi_done_once | _both)
        # 合并版存独立槽位 (深度 0 的外部读者拿它); 各侧命名空间里的 _sig 保持本侧原味
        object.__setattr__(self, "_sig_merged", _sa)
        # 失败(任一手)立即终止; 成功要两手都到; 到位后作恶(fell/thrown/...)仍判失败
        _t_fail = torch.zeros_like(ta)
        if _aa is not None and _ab is not None:
            _t_fail = (ta & ~_aa) | (tb & ~_ab)
            for _sg in (self._A.data.get("_sig") or {}, _sb):
                for _k2 in ("fell", "thrown", "toppled", "pushed", "table_crash", "stuck"):
                    _v2 = _sg.get(_k2)
                    if torch.is_tensor(_v2) and _v2.dtype == torch.bool:
                        _t_fail = _t_fail | _v2
            if getattr(self.cfg, "success_nonterminal", False):
                # e2e: 双侧真抓稳=carry 启动门, 不终止 (成功终止由 carry 末帧口径管)
                _term = _t_fail
            else:
                _term = _t_fail | _both
        else:
            _term = ta | tb
        return _term, (ua | ub)

    def _fc_do_play(self) -> bool:
        """目标点是否用"逐行播参考"来拍 (仅在 fc_target_ref_end 口径下有意义)。"""
        return bool(int(getattr(self.cfg, "fc_play_ref", 0))
                    and getattr(self.cfg, "fc_target_ref_end", False))

    def _fc_measure(self):
        """FC 逐指目标的物理测量 (首次 reset 一次性): 双手摆到真抓姿(臂=phase2 斜坡
        末行, 指=模板), 物体摆到静置位, 步进一帧, 直接读指垫 link 原点 → 表达在各自
        物体局部系。与运行时测量同口径, 不依赖 URDF/USD 命名一致性。"""
        q = self.hand.data.default_joint_pos.clone()
        # ★ 2026-08-24 用户裁定: 目标姿态从 grasp 改成 **squeeze**。
        # 原来摆 _p2_fin(=grasp 模板)测目标点, 但**参考的终态是 squeeze**(更深):
        #   右手 grasp→squeeze 最大关节差 24.4°, 左手 **37.5°**(middle_PIP 55.8→93.3)。
        # 于是策略按参考做到 squeeze, 判据却拿它跟 grasp 的指尖位置比 ⟹ 判"差 6cm",
        # g2 永远 0。判据语义("手指到达它该到的接触点")不变, 变的是那个点取在哪 ——
        # 参考本身奔着 squeeze 去, 取 squeeze 才自洽。
        # ★★ 2026-08-24 判死并修复: 目标点必须在**参考自己的终态**下测。
        # 原来摆的是 env 自解的 `_p2_arm[-1]` + `_p2_fin`(grasp 模板), 与参考真正
        # 走到的终态**不是同一个姿态** —— 实测 右手臂差 **33.5°**(同一腕靶点的两个
        # 独立 IK 解, 7 自由度零空间 + 5mm/2.9° 容差), 左手指差 **37.6°**(grasp vs
        # squeeze)。于是"目标点"整体摆在另一个姿态上, 五指同步偏 2.6~4.8cm, 而腕判据
        # 却达标 55% —— g2 因此**结构性恒 0**, 八代皆亡, 连零动作基线也差 3.7/6.6cm。
        # 修: 臂用 retract_path[-1], 指用 _fin_ref_path[-1] —— 都是参考的最后一行。
        # 硬验收: 零动作下 _fc_d 应 ≈ 0。
        _use_ref = getattr(self.cfg, "fc_target_ref_end", False)
        for ns in (self._A, self._B):
            with self._use(ns):
                _qa = self._p2_arm[-1]
                _fq2 = self._p2_fin
                if _use_ref:
                    _rp = ns.data.get("retract_path")
                    _fp = ns.data.get("_fin_ref_path")
                    if _rp is not None:
                        _da = float(torch.rad2deg((_rp[-1] - _qa).abs().max()))
                        _qa = _rp[-1]
                    else:
                        _da = float("nan")
                    if _fp is not None:
                        _df = float(torch.rad2deg((_fp[-1] - _fq2).abs().max()))
                        _fq2 = _fp[-1]
                    else:
                        _df = float("nan")
                    print(f"[fin_cart] {ns.name} 目标姿态 = **参考终态** "
                          f"(相对原口径: 臂差 {_da:.1f}° 指差 {_df:.1f}°)")
                q[:, self.arm_jids] = _qa.unsqueeze(0)
                q[:, self.hand_jids] = _fq2.unsqueeze(0)
                # ★★ 2026-08-24 判死并修复 (第二版): 物体是**自由刚体**, 复位到
                # obj_init_quat 后会在重力下**沉降到另一个朝向**。原来拍照在沉降前,
                # 而运行时检查在沉降后 —— 实测姿态差 右 9.10° / 左 65.97°。
                # 目标点是**相对物体**存的: 物体朝向错多少, 五个目标点就绕物心整体
                # 转多少 ⟹ 零动作完美复现参考也差 3~5cm, g2 结构性恒 0, 成功奖金
                # 九代一分没发, 课程永远停摆。
                # 修: 摆回物体后**先静置沉降 fc_settle 帧**再拍照, 让拍照条件与
                # 运行条件一致。硬验收: "拍照 vs 运行 物体位姿一致" + "零动作@参考终段
                # 指误差 <1cm"。
                # (第一版试过"完全不摆物体"—— 更糟, 因为 _fc_measure 在首次 reset
                #  **之前**跑, 那时物体还没被摆到静置位, 左手误差飙到 59cm。)
                # 原注释存档: 曾以为是 obj_init_quat 没跟 prior_yaw 更新, 已证伪
                # 但那不是物体在场景里的真实静置朝向 —— 实测拍照 vs 运行的姿态差
                # **右 9.10° / 左 65.97°**(杯 yaw 90° 是在别处施加的, obj_init_quat
                # 没跟着更新)。目标点是**相对物体**存的, 物体朝向错多少, 五个目标点
                # 就绕物心整体转多少 ⟹ 零动作完美复现参考也差 3~5cm, g2 结构性恒 0,
                # 成功奖金九代一分没发过, 课程永远停摆。
                # 修: 拍照**不再摆物体**, 直接用它此刻的真实位姿 —— 从根上消除
                # "obj_init_quat / root_quat_w 两个朝向来源"这个隐患。
                og = self.scene.env_origins
                # ★★★ 2026-08-24 真根因: **副物体(B侧)复位时不走 obj_init_quat**。
                # `_reset_idx` 对主体用 obj_init_quat, 但 Aux 走 `reset_aux_free`,
                # 姿态取 **layout 的 aux_quat_lay_np**(注释明写"不做任何偏航旋转")。
                # 而这里拍照却用 obj_init_quat 摆它 ⟹ 两个来源相差 **66.78°**,
                # 目标点整体绕物心转掉 ⟹ 左手指误差恒 5cm, g2 结构性恒 0。
                # (右手是主体, 只差 1.21°, 所以右手一直"接近但差一点"。)
                # 修: B 侧改用 layout 姿态, 与复位口径一致。
                _oq_use = self.obj_init_quat
                if ns is self._B and getattr(self, "aux_quat_lay_np", None) is not None:
                    _oq_use = torch.tensor(self.aux_quat_lay_np,
                                           dtype=torch.float32, device=self.device)
                    _dlt = float(torch.rad2deg(2 * torch.acos(
                        (self.obj_init_quat * _oq_use).sum().abs().clamp(max=1.0))))
                    print(f"[fin_cart] {ns.name} 摆位改用 layout 姿态 "
                          f"(与 obj_init_quat 差 {_dlt:.2f}°) —— 与复位口径一致")
                pose = torch.cat([self.obj_init_pos.unsqueeze(0).expand(
                    self.num_envs, 3) + og,
                    _oq_use.unsqueeze(0).expand(self.num_envs, 4)], dim=1)
                ids = torch.arange(self.num_envs, device=self.device)
                self.object.write_root_pose_to_sim(pose, ids)
                self.object.write_root_velocity_to_sim(
                    torch.zeros(self.num_envs, 6, device=self.device), ids)
        # ① 先让物体在重力下沉降到稳定朝向 (手仍在默认位, 离物体远)
        _st = int(getattr(self.cfg, "fc_settle_steps", 0))   # 0: 复位后有冻结窗钉住物体, 不该沉降
        for _ in range(_st):
            self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())
        # ② 再把手传送到参考终态, 步进一帧读指尖
        # ★ fc_play_ref 时**不能**先瞬移到终态: 那一步的穿透冲量当场把瓶子
        #   撞飞 (实测第0行就 2.07cm/17.2°, 第10行 13.2cm/96° —— 全是翻滚)。
        if not self._fc_do_play():
            self.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            self.hand.set_joint_position_target(q)
            self.hand.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.sim.get_physics_dt())
        # ★★ 2026-08-24 判死: 拍照是把**已经捏紧的手**瞬移进物体, 一步之内产生
        #   巨大穿透冲量, 物体被弹开 —— 而运行时手指是逐步合拢的, 物体是被慢慢
        #   推到平衡位。两者相对几何天生对不上: 实测右手物体位置差 1.44cm + 姿态
        #   9.48°, 折算到指尖就是整只手刚性偏 1.91cm (五指离散度仅 0.86cm ⟹ 确系
        #   刚性平移而非指型问题)。臂/腕在两个时刻几乎完全重合(0.31cm/0.68°),
        #   指关节也跟得上(均 1.79°) —— 所以错的既不是臂也不是指, 是**目标点**。
        # 修: 手摆进去后再让物体在手里**稳定 fc_post_settle 步**才拍, 让 _fc_local
        #   记录的是"物体在这个抓姿里的平衡相对位姿", 与运行时可达的状态同口径。
        # ★★★ 2026-08-24 定论: "瞬移拍照"两条路都死。不静置 ⟹ 已捏紧的手瞬移进
        #   物体, 一步之内穿透冲量把物体弹开 1.44cm(右); 静置 ⟹ 物体在掌心里
        #   **转 33~37°**。两者的相对几何都不是运行时可达的状态, 零动作恒差 3cm。
        #   根因: 目标点是**物体相对**的, 而抓取过程必然推动物体 —— 唯一同口径的
        #   参照只能来自"把参考真的播一遍"。
        # 修: fc_play_ref=1 时逐行播 retract_path/_fin_ref_path (位置目标+物理步进,
        #   与运行同口径), 播到末行再拍。硬验收: 零动作@参考终段 应 ≈0。
        if self._fc_do_play():
            _dec = max(1, int(getattr(self.cfg, "decimation", 1)))
            _pk = []
            for ns in (self._A, self._B):
                with self._use(ns):
                    _pk.append((ns.data.get("retract_path"),
                                ns.data.get("_fin_ref_path"),
                                self.arm_jids, self.hand_jids))
            _n = max([p[0].shape[0] for p in _pk if p[0] is not None]
                     + [p[1].shape[0] for p in _pk if p[1] is not None] + [1])
            _qp = self.hand.data.joint_pos.clone()
            for _rp, _fp, _aj, _hj in _pk:            # 先落到第 0 行, 免得起手就跳
                if _rp is not None: _qp[:, _aj] = _rp[0].unsqueeze(0)
                if _fp is not None: _qp[:, _hj] = _fp[0].unsqueeze(0)
            self.hand.write_joint_state_to_sim(_qp, torch.zeros_like(_qp))
            for _r in range(_n):
                for _rp, _fp, _aj, _hj in _pk:
                    if _rp is not None:
                        _qp[:, _aj] = _rp[min(_r, _rp.shape[0] - 1)].unsqueeze(0)
                    if _fp is not None:
                        _qp[:, _hj] = _fp[min(_r, _fp.shape[0] - 1)].unsqueeze(0)
                self.hand.set_joint_position_target(_qp)
                self.hand.write_data_to_sim()
                for _ in range(_dec):
                    self.sim.step(render=False)
                if _r == _n - 1:
                    self.scene.update(self.sim.get_physics_dt())
                    _msg = [f"[play] 行{_r:3d}"]
                    for _tg, _ns9 in (("A", self._A), ("B", self._B)):
                        with self._use(_ns9):
                            _dp9 = float((self.object.data.root_pos_w[0]
                                          - self.scene.env_origins[0]
                                          - self.obj_init_pos).norm()) * 100
                            _qn9 = self.object.data.root_quat_w[0]
                            _q09 = (self.obj_init_quat if _ns9 is self._A
                                    else torch.tensor(self.aux_quat_lay_np,
                                                      dtype=torch.float32,
                                                      device=self.device))
                            _dr9 = float(torch.rad2deg(2 * torch.acos(
                                (_qn9 * _q09).sum().abs().clamp(max=1.0))))
                            _ae9 = float(torch.rad2deg(
                                (self.hand.data.joint_pos[0, self.arm_jids]
                                 - _qp[0, self.arm_jids]).abs().max()))
                        _msg.append(f"{_tg}: 物移{_dp9:5.2f}cm 转{_dr9:6.2f}° 臂滞{_ae9:5.2f}°")
                    print(" | ".join(_msg))
            self.scene.update(self.sim.get_physics_dt())
            print(f"[fin_cart] 逐行播参考 {_n} 行 ×{_dec} 子步后拍照 (与运行同口径)")
        _ps = int(getattr(self.cfg, "fc_post_settle", 0))
        for _ in range(_ps):
            self.hand.set_joint_position_target(q)
            self.hand.write_data_to_sim()
            self.sim.step(render=False)
        if _ps:
            self.scene.update(self.sim.get_physics_dt())
            print(f"[fin_cart] 摆手后静置 {_ps} 步再拍照 (让物体沉到抓姿平衡位)")
        for tag, ns in (("A", self._A), ("B", self._B)):
            with self._use(ns):
                og = self.scene.env_origins[0]
                tp = self.hand.data.body_pos_w[0, self.tip_ids] - og
                op = self.object.data.root_pos_w[0] - og
                oq = self.object.data.root_quat_w[0]
                w, x, y, z_ = oq
                R = torch.tensor([[1-2*(y*y+z_*z_), 2*(x*y-w*z_), 2*(x*z_+w*y)],
                                  [2*(x*y+w*z_), 1-2*(x*x+z_*z_), 2*(y*z_-w*x)],
                                  [2*(x*z_-w*y), 2*(y*z_+w*x), 1-2*(x*x+y*y)]],
                                 device=self.device)
                self._fc_local = (R.T @ (tp - op.unsqueeze(0)).T).T.contiguous()
                dd = self._fc_local.norm(dim=1)
                # ★ 存下"拍照那一刻"的物体位姿, 供预检与运行时逐数对比
                ns.data["_fc_meas_op"] = op.detach().clone()
                ns.data["_fc_meas_oq"] = oq.detach().clone()
                # ★ 存拍照时的腕位姿 + 臂关节, 供"整只手是否被平移"对比
                ns.data["_fc_meas_wp"] = (self.wrist_pos_w[0] - og).detach().clone()
                ns.data["_fc_meas_wq"] = self.wrist_quat_w[0].detach().clone()
                ns.data["_fc_meas_aq"] = self.hand.data.joint_pos[
                    0, self.arm_jids].detach().clone()
                print(f"[fin_cart] {tag} 拍照时物体: 位置 "
                      f"{[f'{float(v)*100:.2f}' for v in op]}cm 姿态 "
                      f"{[f'{float(v):.4f}' for v in oq]}")
            # ★ 自洽性自检: 用**运行时那套公式**在拍照这一帧反算 _fc_d, 按定义必须 ≈0。
            #   不为 0 ⟹ 测量式与运行式不自洽(判据 bug); 为 0 ⟹ 5cm 是回合中攒出来的。
            with self._use(ns):
                _ogc = self.scene.env_origins[0]
                _tpc = self.hand.data.body_pos_w[0, self.tip_ids] - _ogc
                _opc = self.object.data.root_pos_w[0] - _ogc
                _oqc = self.object.data.root_quat_w[0]
                from isaaclab.utils.math import quat_apply as _qac
                _twc = _qac(_oqc.unsqueeze(0).expand(5, 4), self._fc_local) \
                    + _opc.unsqueeze(0)
                _sc = (_tpc - _twc).norm(dim=1) * 100
            print(f"[fin_cart] {tag} ★自洽性: 拍照帧反算 _fc_d = "
                  f"{[f'{float(v):.4f}' for v in _sc]}cm  (定义上必须 ≈0)")
            print(f"[fin_cart] {tag} 逐指目标已物理测量 (离物心 "
                  f"{[f'{float(x)*100:.1f}' for x in dd]}cm)")
        object.__setattr__(self, "_fc_ready", True)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """两侧各复位一次; 日志按侧加前缀合并 (原 key 保留两侧均值给课程钩子)。"""
        if getattr(self.cfg, "fin_cart", False) and not getattr(self, "_fc_ready", False):
            self._fc_measure()
        per, flat = {}, {}
        # 铁案4 修复: 真实回合长度在 A 侧清零前快照, 两侧 ep_rew 共用同一除数
        if env_ids is not None:
            object.__setattr__(self, "_bi_ep_len",
                               self.episode_length_buf[env_ids].clone())
        if getattr(self, "_bi_done_once", None) is not None and env_ids is not None:
            self._bi_done_once[env_ids] = False
        for tag, side in ((self._A_name[0].upper(), self._A),
                          (self._B_name[0].upper(), self._B)):
            # B 侧复位不许重摆场景: aux 的相对偏移会被二次应用 (2026-08-18 实锤)
            object.__setattr__(self, "_bi_skip_scene", side is self._B)
            with self._use(side):
                super()._reset_idx(env_ids)
                _lg = self.extras.get("log", {}) or {}
                for _k in _lg:
                    self.extras.pop(_k, None)
                per[tag] = dict(_lg)
                flat.update({f"{tag}/{_k}": _v for _k, _v in _lg.items()})
        object.__setattr__(self, "_bi_skip_scene", False)
        _keys = set().union(*(d.keys() for d in per.values())) if per else set()
        for _k in _keys:
            _vs = [d[_k] for d in per.values() if _k in d and isinstance(d[_k], (int, float))]
            if _vs:
                flat[_k] = sum(_vs) / len(_vs)
        self.extras.update(flat)
        self.extras["log"] = flat
