"""Pink (differential IK via QP) driver for the Vega-1P dual arm.

Why this exists: the DifferentialIKController (DLS) path has no null-space
term, so redundant arm DOFs drift and wind up (L_arm_j1 wrapping its full
range). This is exactly what the reference GR00T stack avoids by using Pink -
a QP IK with a posture (null-space) task and hard joint limits. With no grip
clutch on the final hardware (head + 2 wrist + 1 waist tracker), there's no
"release to un-wind" escape, so limit-aware IK is required, not optional.

Design (mirrors decoupled_wbc/.../body_ik_solver.py):
  - pinocchio model reduced to the 14 arm joints (everything else locked at
    its reference pose; the base is fixed, torso driven separately).
  - one FrameTask per end-effector (L_ee / R_ee), 6-DoF pose, position +
    orientation cost.
  - a WeightedPostureTask biasing the redundant DOFs back toward the home
    pose - this is the anti-winding term DLS lacked.
  - hard joint limits from the URDF, enforced by Pink + a clip.

Targets are EE poses in the robot BASE frame (== the pinocchio model world,
since the model is fixed-base at `vega_1`), matching Isaac's ee_pose_in_base.
"""
from __future__ import annotations

import os

import numpy as np
import pink
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation as R
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from magicdexmate.home_pose import ARM_HOME as HOME_ARM  # noqa: E402
from magicdexmate.swivel import swivel_angle as _swivel_angle  # noqa: E402

# NOTE: build PinkVegaIK BEFORE importing isaaclab. isaacsim clobbers
# pinocchio's eigenpy converters (std::vector<string>/<bool>), so pink's
# Configuration can't be constructed once Kit is loaded; constructing it first
# registers the bindings and the runtime solve then works fine.

DEFAULT_URDF = os.environ.get(
    "VEGA_URDF",
    os.path.expanduser("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf"))
# 默认姿势只有一份,在 magicdexmate/home_pose.py(import 在文件上方,与其它
# magicdexmate 模块一起)。此处曾是它的副本,靠一行「MUST stay in sync」的注释
# 维持一致 —— 而真机那边现在也要用同一个姿势(回合开始时把机器人送回去),
# 再留副本迟早会分叉。

ARM_PIN_ORDER = [f"{s}_arm_j{i}" for s in "LR" for i in range(1, 8)]
# every non-arm actuated joint, to lock in the reduced model. Hardcoded so we
# never index model.names: isaacsim clobbers pinocchio's std::vector<string>
# Python binding, so reading names by index throws (getJointId(str) is fine).
_NON_ARM_JOINTS = ([f"{p}_wheel_j{i}" for p in "BLR" for i in (1, 2)]
                   + [f"torso_j{i}" for i in (1, 2, 3)]
                   + [f"head_j{i}" for i in (1, 2, 3)])
EE_FRAME = {"right": "R_ee", "left": "L_ee"}
ELBOW_FRAME = {"right": "R_arm_l4", "left": "L_arm_l4"}
# shoulder end of the arm, for the swivel angle. j1's frame is the first joint
# of the chain; the fitted kinematic centre sits near it and the swivel angle is
# invariant to a shift ALONG the shoulder->wrist axis, so the frame is fine.
SHOULDER_FRAME = {"right": "R_arm_j1", "left": "L_arm_j1"}
# arm root: the link the arms hang from, AFTER the torso joints. Targeting EE
# poses relative to this frame makes the arm solve independent of the torso
# angle (torso is above arm_center in the chain), which kills the arm<->torso
# coupling when the waist tracker leans the torso. It's also base-frame-
# convention independent (a global USD<->URDF rotation cancels in a
# link-relative pose), so no Isaac<->pinocchio calibration is needed.
CHEST_FRAME = "arm_center"


class WeightedPostureTask(PostureTask):
    """PostureTask with per-joint weights (copied from decoupled_wbc)."""

    def __init__(self, cost, weights, lm_damping=0.0, gain=1.0):
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = np.asarray(weights, dtype=np.float64)

    def compute_error(self, configuration):
        return self.weights * super().compute_error(configuration)

    def compute_jacobian(self, configuration):
        return self.weights[:, np.newaxis] * super().compute_jacobian(configuration)


def _se3(pos, quat_wxyz):
    rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    return pin.SE3(rot, np.asarray(pos, dtype=np.float64))


class PinkVegaIK:
    def __init__(self, urdf_path: str = DEFAULT_URDF, home_arm: dict | None = None,
                 position_cost=8.0, orientation_cost=2.0, lm_damping=10.0,
                 posture_cost=1e-2, gain=1.0, substeps=3, dt=1.0 / 60.0,
                 elbow_cost=0.0, elbow_lm_damping=None, null_bias=0.0,
                 smooth_cost=0.0, relax_at_limit=1.0, relax_margin_deg=2.0,
                 relax_pos_at_limit=False):
        if home_arm is None:
            home_arm = HOME_ARM
        full = pin.buildModelFromUrdf(urdf_path)
        lock = [full.getJointId(n) for n in _NON_ARM_JOINTS if full.existJointName(n)]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        # reduced-model joint order is the full order minus the locked joints;
        # for this URDF that leaves exactly ARM_PIN_ORDER (verified offline).
        self.pin_names = list(ARM_PIN_ORDER)

        # home configuration (redundancy resolves toward this)
        q_home = np.zeros(self.model.nq)
        for i, n in enumerate(self.pin_names):
            q_home[i] = home_arm.get(n, 0.0)
        self.q_home = q_home
        self.substeps = substeps
        self.dt = dt
        # NULL-SPACE redundancy resolution. A 7-DoF arm on a 6-DoF wrist target
        # has exactly one free dimension - the elbow swinging about the
        # shoulder-wrist axis - and NOTHING in the target pins it, so a
        # differential solver keeps whichever branch its history happened to
        # put it in. Measured 2026-08-02: the same clip, same law, same
        # parameters, solved from two different start frames, put R_arm_j1 at
        # +176 deg in one run and -130 deg in the other, and the two arms pick
        # branches independently. That is the "the IK locks up", "the two arms
        # diverge" and "the upper arm points 30 deg wrong" reports, all three.
        #
        # Every earlier attempt added the secondary objective as a WEIGHTED
        # TASK (posture cost, elbow position, swivel angle) and so made it
        # compete with the wrist: raising posture_cost to 5.0 still left 74 deg
        # of branch spread and had already blown the wrist residual out to
        # 585 mm. Projecting instead costs the wrist NOTHING to first order,
        # because the correction lies in the exact null space of the wrist
        # task's own Jacobian. Set null_bias=0 to disable.
        self.null_bias = float(null_bias)
        # 顶到限位时把朝向权重乘以这个数;1.0 = 关闭
        self._relax_scale = float(relax_at_limit)
        self._relax_margin = float(np.radians(relax_margin_deg))
        # 放松曲线的**地板**:slack 低于它就全额放松,margin 是它上面的过渡带。
        # 没有地板时平衡点数学上就卡在墙上:margin 3° 的 smoothstep 在 slack
        # 1° 处朝向权重还剩 26%,足以继续把腕往墙上推,实测平衡 slack 约
        # 0.5°,正好落在「贴限位」判据(1°)里面 —— 于是放松开着、指标却
        # 纹丝不动。地板取判据的两倍:2° 以内朝向权重(0.005)已低于
        # posture(0.01),posture 把腕拉离墙面,平衡点稳定在地板附近、
        # 判据之外。需求回到量程内时 slack 一过渡带,权重自动复原。
        self._relax_floor = float(np.radians(2.0))
        # 位置饱和的对称机制(2026-08-10 晚):clip_headwaist 上腕部残余贴限位
        # 的 94% 与同臂肩肘同时发生 —— 位置够不着时,权重 8 的位置任务会借
        # 腕关节多抓几毫米,把整条手臂压在限位上磨。这里在肩肘(j1-j4)贴
        # 限位时把**位置目标**向当前末端插值:u=1 时位置任务零推力,但仍然
        # 刚性钉在可达点上 —— 不是降权重,手臂没有游荡的自由度,所以没有
        # 朝向那版第一稿的自锁问题;触发关节正是被位置任务压住的,放松位置
        # 它就离墙,结构上自释放。
        self._relax_pos = bool(relax_pos_at_limit)
        self._slack_ema_pos: dict = {}
        self.relax_stat_pos = {"n": 0}
        self._ori_cost0 = float(orientation_cost)
        self._pos_cost0 = float(position_cost)
        self.relax_stat = {"n": 0, "steps": 0}
        # 触发条件的时间平滑。0.25 ≈ 4 个控制步的时间常数(50Hz 下约 80ms):
        # 足以滤掉单帧抖动,又不至于让手臂真的贴上限位时反应不过来。
        self._slack_ema: dict = {}
        self._slack_alpha = 0.25
        # 放松强度 u 的**不对称迟滞**(2026-08-11 凌晨,腕抖 A/B 定位后加):
        # 在放松地板的平衡点附近,u 随 slack 逐帧波动,权重与目标插值跟着
        # 抖,整条手臂的解都会起涟漪 —— 实测开 relax 时腕部微振荡(0.1-5°
        # 的方向反转)3.2-7.0 次每秒,关掉只有 0.5-0.7,差 5-12 倍,这正是
        # 操作者肉眼看到的手腕抖。修法:u 增大(要放松,安全方向)跟得快,
        # u 减小(恢复朝向)只能慢——50Hz 下 0.06 ≈ 0.3 秒时间常数,平衡点
        # 附近的扑动被吸掉,真正离开限位区后约半秒内权重平滑复原。
        self._u_state: dict = {}
        self._u_pos_state: dict = {}
        self._u_attack = 1.0
        self._u_release = 0.06
        # Small span, many iterations. The null direction is a LINEARISATION;
        # scanning +-1.2 rad of it leaves the manifold (measured: the EE drifts
        # 58 mm / 14.6 deg when the line search is run without the wrist task
        # to pull it back). Inside solve() the wrist task runs every substep
        # and corrects between steps, so short hops converge along the true
        # manifold instead of chording across it.
        self.null_span = 0.35
        self._arm_cols = {}
        for hand in EE_FRAME:
            pre = "R" if hand == "right" else "L"
            self._arm_cols[hand] = [self.pin_names.index(f"{pre}_arm_j{i}")
                                    for i in range(1, 8)]

        self.config = pink.Configuration(self.model, self.data, q_home.copy())
        self.config.model.lowerPositionLimit = self.model.lowerPositionLimit
        self.config.model.upperPositionLimit = self.model.upperPositionLimit

        # arm_center is fixed in the reduced model (torso is locked, and it sits
        # above the arm joints), so its world pose is constant. Cache it: it's
        # the frame targets are expressed relative to.
        pin.forwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)
        cm = self.data.oMf[self.model.getFrameId(CHEST_FRAME)]
        self._chest = pin.SE3(cm.rotation.copy(), cm.translation.copy())

        self.frame_tasks = {}
        for hand, frame in EE_FRAME.items():
            t = FrameTask(frame, position_cost=position_cost,
                          orientation_cost=orientation_cost,
                          lm_damping=lm_damping, gain=gain)
            self.frame_tasks[hand] = t
        # optional low-weight elbow position tasks (elbow_cost > 0): pull the
        # elbow point (L/R_arm_l4 = j4 origin) toward the operator's elbow so
        # the redundant DOF resolves human-like instead of toward q_home
        self.elbow_tasks = {}
        if elbow_cost > 0.0:
            # lm_damping is NOT inherited from the wrist task by default.
            # It was, and that is a bug: the wrist runs position_cost 8.0 with
            # damping 10.0, and handing the same damping to a 0.25-cost elbow
            # task makes the damping 40x the cost. Measured: the solve goes
            # target-INDEPENDENT unstable - position-based and swivel-based
            # elbow targets both produced an identical 475.1 mm wrist residual
            # at cost 0.25, while cost 1.0 (a 10:1 ratio, same as the wrist's)
            # was stable at 44 mm. Default to keeping the wrist's ratio.
            eld = (elbow_lm_damping if elbow_lm_damping is not None
                   else lm_damping * elbow_cost / max(position_cost, 1e-9))
            for hand, frame in ELBOW_FRAME.items():
                t = FrameTask(frame, position_cost=elbow_cost,
                              orientation_cost=0.0, lm_damping=eld,
                              gain=gain)
                self.elbow_tasks[hand] = t
        self.posture = WeightedPostureTask(
            cost=posture_cost, weights=np.ones(self.model.nv))
        # V2AP's smoothness term (ik_utils.py): a posture task whose target is
        # re-set to the CURRENT configuration before every solve. It penalises
        # the step itself, so a fast/awkward target produces a slower approach
        # instead of a joint jump - damping at the demand side, where V2AP put
        # it (their cost 0.2 against position 50; scaled to our position 8.0
        # the equivalent neighbourhood is ~0.03, but the flag is free).
        self.smooth = None
        if smooth_cost > 0.0:
            self.smooth = WeightedPostureTask(
                cost=smooth_cost, weights=np.ones(self.model.nv))
        # seed every task's target from the home configuration
        for t in self.frame_tasks.values():
            t.set_target_from_configuration(self.config)
        for t in self.elbow_tasks.values():
            t.set_target_from_configuration(self.config)
        self.posture.set_target(q_home.copy())

        try:
            import qpsolvers
            self.solver = "quadprog" if "quadprog" in qpsolvers.available_solvers \
                else qpsolvers.available_solvers[0]
        except Exception:
            self.solver = "quadprog"

    # -- chest-relative targeting (no base-frame calibration needed) ----------
    def set_target_chest(self, hand: str, chest_pos, chest_quat_wxyz,
                         tgt_pos, tgt_quat_wxyz):
        """Set the EE target from Isaac-frame poses of arm_center (chest) and the
        desired EE. The target is taken relative to the chest (torso-independent)
        and re-expressed in pinocchio's world via the model's own fixed chest
        pose - so torso lean never disturbs the arm solve, and no Isaac<->pin
        base calibration is needed (the chest-relative pose is frame-agnostic)."""
        chest_isaac = _se3(chest_pos, chest_quat_wxyz)
        tgt_isaac = _se3(tgt_pos, tgt_quat_wxyz)
        tgt_rel_chest = chest_isaac.inverse() * tgt_isaac
        # USD<->URDF EE-link local frames can differ in ORIENTATION even when
        # positions coincide (validate_chest only checks positions). Without
        # this correction the ori task chases a constantly-rotated target and
        # skews the position optimum. Calibrated once by calibrate_ee_rot().
        fix = getattr(self, "ee_rot_fix", {}).get(hand)
        if fix is not None:
            tgt_rel_chest = pin.SE3(fix @ tgt_rel_chest.rotation,
                                    tgt_rel_chest.translation)
        tgt_world = self._chest * tgt_rel_chest
        self.last_target = getattr(self, "last_target", {})
        self.last_target[hand] = tgt_world
        self.frame_tasks[hand].set_target(tgt_world)

    def set_elbow_target_chest(self, hand: str, chest_pos, chest_quat_wxyz,
                               tgt_pos):
        """Position-only elbow target via the same chest-relative transfer
        (orientation part is ignored - the elbow task has ori cost 0)."""
        if hand not in self.elbow_tasks:
            return
        chest_isaac = _se3(chest_pos, chest_quat_wxyz)
        tgt_isaac = _se3(tgt_pos, [1.0, 0.0, 0.0, 0.0])
        rel = chest_isaac.inverse() * tgt_isaac
        self.elbow_tasks[hand].set_target(self._chest * rel)

    def calibrate_ee_rot(self, chest_pos, chest_quat_wxyz, ee_pos,
                         ee_quat_wxyz, isaac_q: dict) -> dict:
        """One-time USD->URDF EE-frame orientation offset, per hand, from the
        same settled-pose data validate_chest uses. Returns {hand: deg}."""
        self.set_config_from_isaac(isaac_q)
        pin.forwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)
        self.ee_rot_fix = {}
        angles = {}
        for h in EE_FRAME:
            T = self.data.oMf[self.model.getFrameId(EE_FRAME[h])]
            rel_pin = self._chest.inverse() * pin.SE3(T.rotation.copy(),
                                                      T.translation.copy())
            rel_isaac = (_se3(chest_pos[h], chest_quat_wxyz[h]).inverse()
                         * _se3(ee_pos[h], ee_quat_wxyz[h]))
            R_fix = rel_pin.rotation @ rel_isaac.rotation.T
            self.ee_rot_fix[h] = R_fix
            angles[h] = float(np.rad2deg(np.linalg.norm(pin.log3(R_fix))))
        # restore the runtime start configuration
        self.config.q = self.q_home.copy()
        self.config.update()
        return angles

    def validate_chest(self, chest_pos, chest_quat_wxyz, ee_pos, ee_quat_wxyz,
                       isaac_q: dict):
        """Sanity check at startup: Isaac's EE-relative-to-chest at its settled
        config must match pinocchio's. Returns the position residual [mm]."""
        self.set_config_from_isaac(isaac_q)
        pin.forwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)
        return {h: float(np.linalg.norm(
            (_se3(chest_pos[h], chest_quat_wxyz[h]).inverse()
             * _se3(ee_pos[h], ee_quat_wxyz[h])).translation
            - (self._chest.inverse()
               * pin.SE3(self.data.oMf[self.model.getFrameId(EE_FRAME[h])].rotation,
                         self.data.oMf[self.model.getFrameId(EE_FRAME[h])].translation)
               ).translation) * 1000)
                for h in EE_FRAME}

    def reset_relax(self):
        """清掉触发条件的平滑状态。回合归位时调 —— 上一段的「离限位多近」
        跟这一段没有关系。"""
        self._slack_ema.clear()
        self._slack_ema_pos.clear()

    def reset_home(self) -> float:
        """把内部构型拉回默认姿势,返回最大关节改动(度)。

        一段遥操作开始时用。求解器的构型是整场会话的持久开环状态:一旦绕进
        坏支或贴上限位,重新啮合只换锚点不换构型,坏姿态会被继承下去。参考
        实现(V2AP / T-Rex)在每个回合开始把机器人**物理地**送回默认关节位,
        再从那个姿势重建 IK 种子和映射锚点 —— 坏构型活不过一个回合。这是那
        一套的求解器侧。
        """
        self._slack_ema.clear()
        self._slack_ema_pos.clear()
        self._u_state.clear()
        self._u_pos_state.clear()
        d = float(np.degrees(np.abs(self.config.q - self.q_home)).max())
        self.config.q = self.q_home.copy()
        self.config.update()
        return d

    def set_config_from_isaac(self, isaac_q: dict):
        q = np.array([isaac_q.get(n, self.config.q[i])
                      for i, n in enumerate(self.pin_names)])
        # Isaac's arm can sit exactly at (or a float-eps past) a joint limit -
        # e.g. after DLS winding L_arm_j6 parks at +-1.396. Pink's check_limits
        # rejects a seed even 1e-6 outside, so clip just inside.
        m = 1e-4
        q = np.clip(q, self.model.lowerPositionLimit + m,
                    self.model.upperPositionLimit - m)
        self.config.q = q
        self.config.update()

    def set_target(self, hand: str, pos_base, quat_base_wxyz):
        self.frame_tasks[hand].set_target(_se3(pos_base, quat_base_wxyz))

    def hold_target(self, hand: str):
        """Freeze this arm's target at its current FK pose (disengaged arm)."""
        self.frame_tasks[hand].set_target_from_configuration(self.config)
        if hand in self.elbow_tasks:
            self.elbow_tasks[hand].set_target_from_configuration(self.config)

    def set_swivel_target(self, hand: str, angle_rad: float | None):
        """Where the elbow should sit on its circle about the shoulder->wrist
        axis. This is the ONE dimension the wrist target leaves free."""
        if not hasattr(self, "_swivel_tgt"):
            self._swivel_tgt = {}
        self._swivel_tgt[hand] = angle_rad

    def select_branch(self, hand: str, angle_rad: float | None,
                      iters: int = 20) -> float | None:
        """一次性把这条臂挪到操作者所在的那一支上,只在离合闭合时调用。

        为什么是一次性的:每个子步都跑线搜索(null_bias>0)实测不可用,它把
        手臂拖离腕目标的速度超过腕任务把它拉回的速度 —— 2026-08-05 在
        playlist_all13 上量到腕误差 12.7mm 涨到 69.6mm、逐帧跳变 p99 由 7.3°
        涨到 31.2°、实时倍率掉到 0.97x。

        而分支本来就只在啮合那一刻被定死,之后跟踪是连续的,不会自己跳到另一
        支上。所以在啮合时选一次、之后完全不碰,既拿到了确定的臂形,也不会在
        跟踪期间和腕任务打架。

        返回选完之后机器人实际的回转角,便于打印核对;拿不到操作者回转角时返回
        None 并且什么都不做。
        """
        if angle_rad is None:
            return None
        saved_bias = self.null_bias
        saved_tgt = dict(getattr(self, "_swivel_tgt", {}))
        self.null_bias = 1.0
        for h in EE_FRAME:                     # 只动这一条臂
            self.set_swivel_target(h, angle_rad if h == hand else None)
        try:
            for _ in range(iters):
                self.solve()
        finally:
            # 无论如何都要还原,否则一次异常就把逐步线搜索永久打开了
            self.null_bias = saved_bias
            self._swivel_tgt = saved_tgt
        self.refresh_fk()
        return self._robot_swivel(hand)

    def _robot_swivel(self, hand: str) -> float | None:
        sh = self.data.oMf[self.model.getFrameId(
            SHOULDER_FRAME[hand])].translation
        el = self.data.oMf[self.model.getFrameId(ELBOW_FRAME[hand])].translation
        wr = self.data.oMf[self.model.getFrameId(EE_FRAME[hand])].translation
        return _swivel_angle(sh, el, wr)

    def _null_line_search(self, q: np.ndarray) -> np.ndarray:
        """Pick the point on each arm's null-space LINE whose swivel is closest
        to the operator's. A line search, not a gradient step, because the
        redundant dimension is a CIRCLE: local descent - the posture cost, a
        null-space gradient, a narrowed joint range - cannot cross to the other
        side of it, which is why all three were measured to leave 74-250 deg of
        branch spread. Sampling the whole line can.
        """
        tgt = getattr(self, "_swivel_tgt", {})
        if not tgt:
            return q
        for hand in EE_FRAME:
            want = tgt.get(hand)
            if want is None:
                continue
            cols = self._arm_cols[hand]
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            J = pin.getFrameJacobian(self.model, self.data,
                                     self.model.getFrameId(EE_FRAME[hand]),
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, cols]
            _, sv, Vt = np.linalg.svd(J)
            if sv[-1] < 1e-3 * max(sv[0], 1e-9):
                continue
            n = Vt[-1]
            lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
            best, best_e = 0.0, None
            for a in np.linspace(-self.null_span, self.null_span, 41):
                qq = q.copy()
                qq[cols] = np.clip(q[cols] + a * n, lo[cols], hi[cols])
                pin.framesForwardKinematics(self.model, self.data, qq)
                got = self._robot_swivel(hand)
                if got is None:
                    continue
                e = abs((got - want + np.pi) % (2 * np.pi) - np.pi)
                if best_e is None or e < best_e:
                    best, best_e = a, e
            if best_e is not None:
                q[cols] = np.clip(q[cols] + self.null_bias * best * n,
                                  lo[cols], hi[cols])
        pin.framesForwardKinematics(self.model, self.data, q)
        return q

    def _null_step(self, q: np.ndarray) -> np.ndarray:
        """One step of each arm's null-space toward q_home. EE pose unchanged."""
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for hand, frame in EE_FRAME.items():
            cols = self._arm_cols[hand]
            J = pin.getFrameJacobian(self.model, self.data,
                                     self.model.getFrameId(frame),
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, cols]
            # right singular vector of the smallest singular value = the one
            # direction in this arm's joint space that the wrist cannot see
            # J is 6x7, so Vt is 7x7 and Vt[6] spans the null space. sv holds
            # the SIX non-zero singular values; skip only when the smallest of
            # those has collapsed, i.e. near a singularity, where the null
            # space is wider than one dimension and this projection is not
            # trustworthy. (Written the other way round first, which skipped
            # every well-conditioned pose - so the step never ran at all and
            # the joints did not move by 0.00 deg. The self-check caught it
            # only after being made to push the arm AWAY from q_home first;
            # at q_home the correction is zero and any bug looks like success.)
            _, sv, Vt = np.linalg.svd(J)
            if sv[-1] < 1e-3 * max(sv[0], 1e-9):
                continue
            n = Vt[-1]
            err = self.q_home[cols] - q[cols]
            q[cols] = q[cols] + self.null_bias * float(n @ err) * n
        return np.clip(q, self.model.lowerPositionLimit,
                       self.model.upperPositionLimit)

    def _relax_pinned(self) -> int:
        """顶到限位的那条手臂,这一步临时把朝向权重压下去。

        实测的失效是:腕部 j6(±80°)/ j7(−79 至 64°)被指令的掌心朝向推到
        限位上,顶死之后朝向一直是错的(14.15% 的帧至少一个手臂关节贴在限位
        1° 以内,最长连续 5.2 秒)。

        此前试过的六个办法里,改**朝向需求**的(降增益、朝向缩放、增量朝向)
        全都更糟,因为它们**一直**在改;改**求解器历史**的(重播种、平滑项、
        回合归位)全都无效,因为顶限位与历史无关。所以这里只在**顶到限位的
        那一帧、那一条手臂**上放松朝向 —— 正常跟随时一点不动。

        返回这一步放松了几条手臂,给生效证明用。
        """
        n = 0
        fk_stale = True     # 目标插值需要当前 EE 朝向;一次 solve 至多算一次
        for hand, cols in self._arm_cols.items():
            # **只看 j6/j7。** 这个机制要治的失效只有一种:掌心朝向把腕部
            # j6(±80°)/ j7(143° 总量程)推到限位 —— 决定朝向可达性的只有
            # 这两个关节。第一版取的是全部 7 个关节的最小 slack,结果是一个
            # 自锁(2026-08-10 查清,此前记作「对负载极度敏感,根因未知」):
            # 任何扰动让**肩肘**短暂贴限位 -> 触发,朝向权重压到 1% -> 失去
            # 朝向约束的手臂被 3 自由度位置任务放任游荡,肩关节持续贴死 ->
            # slack 恒 0 -> 放松永不解除。三次变糟的跑主力顶限位关节全是
            # j2(肩),单段连续 35.4 秒,锁在哪条臂哪条腕误差就翻倍;负载
            # 只是把双稳态踢进吸收态的扰动源(有一跑实时倍率 2.37 照样锁)。
            # 肩肘顶限位是位置/冗余的事,放松朝向不对症,还把唯一能把手臂
            # 拉回来的约束拿掉了。腕关节没有这个问题:放松期间 posture 把
            # j6/j7 拉回量程中央,放松自解除(dev_relax_latch.py 第 3 条)。
            wrist = cols[5:]
            q = self.config.q[wrist]
            lo = self.model.lowerPositionLimit[wrist]
            hi = self.model.upperPositionLimit[wrist]
            # 离限位还有多远(取最近的那个关节),归一化到 [0,1]:
            # 1 = 已经贴上,0 = 还在 margin 以外。
            slack = np.minimum(q - lo, hi - q).min()
            # 对 slack 做时间平滑:单帧尖刺穿不过去,触发看的是「最近一段
            # 时间离限位多近」。(这层 EMA 当初是冲着负载敏感性加的,没治住
            # —— 因为那不是单帧噪声,是上面说的自锁;修好自锁后这层留着
            # 只为它本来的用途。)
            prev = self._slack_ema.get(hand)
            slack = (slack if prev is None
                     else prev + self._slack_alpha * (slack - prev))
            self._slack_ema[hand] = slack
            # 放松强度 u 的状态机(2026-08-11 凌晨第三版,前两版的失败都量过):
            #   连续调制版 —— 在地板平衡点附近 u 逐帧扑动,腕部微振荡 3.2-7.0
            #   次每秒(关掉只有 0.5-0.7),就是操作者看到的手腕抖;
            #   进快出慢的迟滞版 —— 释放看的是 slack,可需求仍不可达时权重一
            #   恢复就被推回墙,形成慢速极限环,最近距墙 0.2°,更糟。
            # 正确语义:**保持/释放看「需求还在量程外吗」,不看离墙多远。**
            #   进入:腕已贴地板,或已进过渡带且需求明显达不到;
            #   保持:只要需求与当前朝向仍差得远(还不可达)就保持全额放松,
            #        腕由 posture 安静地停在地板附近,什么都不扑动;
            #   释放:操作者把手转回来(需求差角变小)才放,且平滑放完。
            e_raw = None
            t = self.frame_tasks[hand]
            if t.transform_target_to_world is not None:
                if fk_stale:
                    pin.framesForwardKinematics(self.model, self.data,
                                                self.config.q)
                    pin.updateFramePlacements(self.model, self.data)
                    fk_stale = False
                _R_cur = self.data.oMf[
                    self.model.getFrameId(EE_FRAME[hand])].rotation
                e_raw = float(np.degrees(np.linalg.norm(pin.log3(
                    t.transform_target_to_world.rotation.T @ _R_cur))))
            _pu = self._u_state.get(hand, 0.0)
            engage = (slack < self._relax_floor
                      or (slack < self._relax_floor + self._relax_margin
                          and e_raw is not None and e_raw > 15.0))
            hold = _pu > 0.05 and e_raw is not None and e_raw > 10.0
            u_tgt = 1.0 if (engage or hold) else 0.0
            _a = self._u_attack if u_tgt > _pu else self._u_release
            u = _pu + _a * (u_tgt - _pu)
            self._u_state[hand] = u
            if self._relax_scale >= 1.0:
                u = 0.0     # 朝向放松整体关闭(--relax-at-limit 1.0)
            want = self._ori_cost0 * (1.0 - u * (1.0 - self._relax_scale))
            t = self.frame_tasks[hand]
            # 位置饱和的触发:**只看肩 j1-j3,不看肘 j4**。肘贴下限是收拢
            # 动作的正常姿态(基线里 L/R_j4 各 115/126 帧、连续段短),把
            # 插值用在收肘时刻等于白白放弃跟踪 —— 第一版含 j4 实测左腕误差
            # 12.8→18.8mm(+47%)。肩的量程饱和才是「够不着」。地板也比
            # 腕部的收一半(1°):肩不需要 2° 的停靠距离,只需要不磨秒级。
            u_pos = 0.0
            if self._relax_pos:
                sh = cols[:3]
                s2 = float(np.minimum(self.config.q[sh]
                                      - self.model.lowerPositionLimit[sh],
                                      self.model.upperPositionLimit[sh]
                                      - self.config.q[sh]).min())
                prev2 = self._slack_ema_pos.get(hand)
                s2 = (s2 if prev2 is None
                      else prev2 + self._slack_alpha * (s2 - prev2))
                self._slack_ema_pos[hand] = s2
                # 与朝向侧同款状态机:保持/释放看「位置需求还够不够得着」
                # (需求点与当前末端的距离),不看离限位多远 —— 连续调制会
                # 扑动,按 slack 释放会形成「权重回来→被推回墙」的慢速循环,
                # 两版都实测淘汰过(机理见朝向侧注释)。
                e_pos = None
                if t.transform_target_to_world is not None:
                    if fk_stale:
                        pin.framesForwardKinematics(self.model, self.data,
                                                    self.config.q)
                        pin.updateFramePlacements(self.model, self.data)
                        fk_stale = False
                    e_pos = float(np.linalg.norm(
                        t.transform_target_to_world.translation
                        - self.data.oMf[
                            self.model.getFrameId(EE_FRAME[hand])].translation
                    ) * 1000.0)
                _pp = self._u_pos_state.get(hand, 0.0)
                _floor2 = self._relax_floor * 0.5
                engage2 = (s2 < _floor2
                           or (s2 < _floor2 + self._relax_margin
                               and e_pos is not None and e_pos > 50.0))
                hold2 = _pp > 0.05 and e_pos is not None and e_pos > 30.0
                u_tgt2 = 1.0 if (engage2 or hold2) else 0.0
                _a = self._u_attack if u_tgt2 > _pp else self._u_release
                u_pos = _pp + _a * (u_tgt2 - _pp)
                self._u_pos_state[hand] = u_pos
                # 权重也要一起降(对称于朝向那版的教训,方向相反):只插值
                # 不降权时,目标=当前 + 权重 8 是一个强阻尼器 —— 罚的是每步
                # 位移 —— 把手臂**冻在**墙上(实测 slack 0.5°),posture
                # (0.01)拉不动。降到 0.08 阻尼变弱,posture 把肩肘从墙上
                # 请下来;目标=当前保证手臂没有游荡的自由度,不会重演自锁。
                want_pos = self._pos_cost0 * (1.0 - u_pos * 0.99)
                if abs(float(np.mean(t.cost[:3])) - want_pos) > 1e-12:
                    t.cost[:3] = want_pos
            if (u > 0.0 or u_pos > 0.0) \
                    and t.transform_target_to_world is not None:
                # 饱和时把朝向**目标**也向「当前朝向」插值,u=1 时朝向任务
                # 不再产生任何推力。只压权重不够:权重是乘在误差上的,超量程
                # 150° 时 0.005 × 2.6 rad 的梯度仍与 posture(0.01)相当,
                # 腕照样被压在墙上 0.1° 处(dev_relax_latch 第 4 条抓的就是
                # 这个)。目标插值让推力随 u 归零,与超量程多大无关;需求
                # 回到量程内时 u 归零,目标自动回到原始需求 —— 目标每个
                # 控制步都会被 set_target 重写,这里的改写只活一步。
                if fk_stale:
                    pin.framesForwardKinematics(self.model, self.data,
                                                self.config.q)
                    pin.updateFramePlacements(self.model, self.data)
                    fk_stale = False
                M_cur = self.data.oMf[self.model.getFrameId(EE_FRAME[hand])]
                T = t.transform_target_to_world
                R_new = (T.rotation @ pin.exp3(
                    u * pin.log3(T.rotation.T @ M_cur.rotation))
                    if u > 0.0 else T.rotation)
                p_new = (T.translation
                         + u_pos * (M_cur.translation - T.translation)
                         if u_pos > 0.0 else T.translation)
                t.set_target(pin.SE3(R_new, p_new))
            if abs(float(np.mean(t.cost[3:])) - want) > 1e-12:
                t.cost[3:] = want
            n += int(u > 0.0)
            self.relax_stat_pos["n"] += int(u_pos > 0.0)
        return n

    def solve(self):
        if self._relax_scale < 1.0 or self._relax_pos:
            self.relax_stat["n"] += self._relax_pinned()
            self.relax_stat["steps"] += 1
        if self.smooth is not None:
            # once per control step, not per substep - matching V2AP, and so
            # the substeps can still converge within the step
            self.smooth.set_target(self.config.q.copy())
        for _ in range(self.substeps):
            tasks = [*self.frame_tasks.values(),
                     *self.elbow_tasks.values(), self.posture]
            if self.smooth is not None:
                tasks.append(self.smooth)
            v = solve_ik(self.config, tasks, dt=self.dt, solver=self.solver)
            q = self.config.q + v * self.dt
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
            if self.null_bias > 0.0:
                q = (self._null_line_search(q)
                     if getattr(self, "_swivel_tgt", None) else self._null_step(q))
            self.config.q = q
            self.config.update()
        return self.config.q.copy()

    def q_map_to(self, isaac_names: list[str]) -> dict[int, float]:
        """Return {isaac_joint_index: angle} for the 14 arm joints."""
        q = self.config.q
        name_to_q = {n: q[i] for i, n in enumerate(self.pin_names)}
        return {isaac_names.index(n): float(name_to_q[n])
                for n in self.pin_names if n in isaac_names}

    def refresh_fk(self):
        """Recompute frame placements for the current q. config.update() does
        not refresh oMf, so ee_pos() reads stale FK without this (the round-3
        pink_task_err artifact)."""
        pin.framesForwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)

    def ee_pos(self, hand: str) -> np.ndarray:
        fid = self.model.getFrameId(EE_FRAME[hand])
        return self.data.oMf[fid].translation.copy()
