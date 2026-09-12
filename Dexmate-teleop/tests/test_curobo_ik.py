"""CuroboVegaIK 接口单测:teleop 用到的每个方法一个用例。

    .venv-isaac/bin/python -m pytest tests/test_curobo_ik.py -q

必须用 .venv-isaac 跑:cuRobo 只装在那里,且客户端用 sys.executable 起求解
子进程 —— 换个解释器,子进程 import curobo 就失败。不依赖 Isaac(客户端只有
pinocchio + ZMQ),但依赖 GPU:没有可用的 nvidia-smi 时整个模块 skip 并写明
原因,而不是假绿。

接口契约 = sim/teleop_vega_pico.py 里 `pink_ik.` 的全部实际用点(2026-08-11
grep):pin_names / config.q+update / solve / set_target_chest /
set_elbow_target_chest / hold_target / set_swivel_target / refresh_fk /
_robot_swivel / ee_pos / q_map_to / reset_home / set_config_from_isaac /
calibrate_ee_rot / validate_chest / null_bias / relax_stat / relax_stat_pos /
last_target。语义基准 = PinkVegaIK 的同名实现。

耗时:求解子进程整个模块只建两次(session fixture + close 专测),warp 缓存
热时各约 3-8 秒;全模块热缓存约 60-90 秒,冷缓存(首次)可到 2 分钟。
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
import pinocchio as pin
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))

if shutil.which("nvidia-smi") is None or subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True).returncode != 0:
    pytest.skip("GPU 不可用(nvidia-smi 缺失或失败)—— cuRobo 求解子进程"
                "需要 CUDA,这组接口测试无法运行", allow_module_level=True)

from curobo_vega_ik import CuroboVegaIK, _pid_alive  # noqa: E402
from pink_vega_ik import ARM_PIN_ORDER, EE_FRAME  # noqa: E402

DT = 1.0 / 50.0


@pytest.fixture(scope="module")
def ik():
    solver = CuroboVegaIK(dt=DT, local_iters=40)   # teleop 默认档
    yield solver
    solver.close()


@pytest.fixture()
def home_tgt(ik):
    """每个用例从干净状态出发:构型回 HOME、清标定、目标全部钉回 HOME FK。
    session 级共享一个子进程(省两次构建),状态隔离靠这里。"""
    ik.reset_home()
    ik.ee_rot_fix = {}
    ik.null_bias = 0.0
    ik._swivel_tgt = {}
    ik.refresh_fk()
    tgt = {}
    for hand, frame in EE_FRAME.items():
        M = ik.data.oMf[ik.model.getFrameId(frame)]
        q = pin.Quaternion(M.rotation)
        tgt[hand] = (M.translation.copy(), np.array([q.w, q.x, q.y, q.z]))
        ik.set_target(hand, *tgt[hand])
    return tgt


def _chest_pose(ik):
    q = pin.Quaternion(ik._chest.rotation)
    return ik._chest.translation.copy(), np.array([q.w, q.x, q.y, q.z])


# ---------------------------------------------------------------- 基本账务 --
def test_pin_names_and_config(ik, home_tgt):
    assert ik.pin_names == list(ARM_PIN_ORDER)
    assert ik.config.q.shape == (14,)
    # dev_relax_latch 测试范式:直接写 config.q 后调 config.update()
    ik.config.q = ik.q_home.copy()
    ik.config.update()          # 必须存在且无害


def test_solve_tracks_reachable_target(ik, home_tgt):
    """HOME 上方 2cm 的可达目标,60 步(1.2s)内收敛到 10mm 以内。
    速度钳制下每帧至多约 3°,静态目标 1 秒余量充足。"""
    p, quat = home_tgt["right"]
    ik.set_target("right", p + np.array([0.0, 0.0, 0.02]), quat)
    n0 = len(ik.solve_ms)
    for _ in range(60):
        q = ik.solve()
    assert q.shape == (14,) and np.all(np.isfinite(q))
    ik.refresh_fk()
    err = np.linalg.norm(ik.ee_pos("right")
                         - (p + np.array([0.0, 0.0, 0.02]))) * 1000
    assert err < 10.0, f"可达目标 60 步后残差 {err:.1f}mm"
    # solve_ms 与真实调用次数一致 —— close() 打的分位数就靠它,不许缺记
    assert len(ik.solve_ms) - n0 == 60


def test_solve_respects_limits_and_rate(ik, home_tgt):
    """超臂展目标:不抛异常、恒在位置限位内、逐帧跳变有界。

    速度限位不是逐帧硬钳(2026-08-11 实测:1m 目标跳变后的追赶步最大
    7.8°/帧 ≈ 2.5 倍 vel×dt;离线台架在真实素材上也量到 0.03% 的 >5° 跳变
    残留)—— 判据取 3 倍 vel×dt(≈9.3°):再涨就是结构变了,而 Pink 在同类
    场景下会一步跳几十度。位置限位则必须硬:一帧越界就是优化约束坏了。"""
    p, quat = home_tgt["right"]
    ik.set_target("right", p + np.array([1.0, -0.5, 0.5]), quat)
    step_cap = np.degrees(ik.model.velocityLimit * DT) * 3.0
    prev = ik.config.q.copy()
    for _ in range(30):
        q = ik.solve()
        assert np.all(q >= ik.model.lowerPositionLimit - 1e-9)
        assert np.all(q <= ik.model.upperPositionLimit + 1e-9)
        jump = np.degrees(np.abs(q - prev))
        assert np.all(jump <= step_cap), \
            f"逐帧跳变 {jump.max():.1f}° 超过 3 倍速度钳制 {step_cap.max():.1f}°"
        prev = q


def test_hold_target_freezes_arm(ik, home_tgt):
    """hold_target = 目标冻结在当前 FK 位姿,解一步不许动(服务端预热
    自检用的同一判据 0.5°)。"""
    for hand in EE_FRAME:
        ik.hold_target(hand)
    q0 = ik.config.q.copy()
    ik.solve()
    d = np.degrees(np.abs(ik.config.q - q0)).max()
    assert d < 0.5, f"hold 中的手臂动了 {d:.2f}°"


# ------------------------------------------------------------ chest 系换算 --
def test_set_target_chest_identity(ik, home_tgt):
    """把「Isaac 的 chest 位姿」喂成 pinocchio 自己的 chest 位姿时,
    chest 相对换算必须是恒等:世界目标原样进 _tgt 与 last_target。"""
    from curobo_vega_ik import _se3
    cp, cq = _chest_pose(ik)
    p, quat = home_tgt["right"]
    want_p = p + np.array([0.03, -0.02, 0.05])
    ik.set_target_chest("right", cp, cq, want_p, quat)
    got = ik._tgt["right"]
    assert np.allclose(got.translation, want_p, atol=1e-9)
    assert np.allclose(got.rotation, _se3(want_p, quat).rotation, atol=1e-9)
    assert ik.last_target["right"] is got   # teleop 调试读的是 last_target


def test_set_elbow_target_chest_noop(ik, home_tgt):
    """肘部任务未实现(Pink 默认 elbow_cost=0 同样无操作):调用必须
    静默、且不动腕目标。"""
    cp, cq = _chest_pose(ik)
    before = {h: (ik._tgt[h].translation.copy(), ik._tgt[h].rotation.copy())
              for h in EE_FRAME}
    ik.set_elbow_target_chest("right", cp, cq, [0.3, -0.2, 1.0])
    for h in EE_FRAME:
        assert np.allclose(ik._tgt[h].translation, before[h][0])
        assert np.allclose(ik._tgt[h].rotation, before[h][1])


def test_set_swivel_target(ik, home_tgt):
    """None(清除)必须静默;非 None 在 null_bias=0(teleop 默认)下只存
    不动 —— 解一步构型必须与不设目标时逐位相同(默认行为不变的证明)。"""
    ik.set_swivel_target("right", None)
    ik.set_swivel_target("left", None)

    def settle():
        # 两个分支走完全相同的历史(归位 -> hold -> 解两步),服务端的
        # 热启动/速度状态因此逐位一致,才能做逐位对比。
        ik.reset_home()
        for hand in EE_FRAME:
            ik.hold_target(hand)
        ik.solve()
        return ik.solve()

    q_ref = settle()
    ik.set_swivel_target("right", 0.5)     # 有目标但增益为零
    q_with = settle()
    assert np.allclose(q_with, q_ref, atol=1e-12)
    assert ik.swivel_follow_stat["n"] == 0


def test_swivel_follow(ik, home_tgt):
    """臂形跟随生效证明(2026-08-11 硬需求):null_bias=1 时喂一个偏离
    当前 20° 的回转角目标,60 步(1.2s)内回转角必须显著挪过去(>10°),
    同时腕位置不许被拖走(零空间步一阶无损,残差 < 15mm)。"""
    ik.refresh_fk()
    s0 = ik._robot_swivel("right")
    assert s0 is not None
    want = s0 + np.radians(20.0)
    ik.null_bias = 1.0
    p, quat = home_tgt["right"]
    for _ in range(60):
        ik.set_target("right", p, quat)
        ik.set_target("left", *home_tgt["left"])
        ik.set_swivel_target("right", want)
        ik.solve()
    assert ik.swivel_follow_stat["n"] > 0, "跟随步从没执行"
    ik.refresh_fk()
    s1 = ik._robot_swivel("right")
    moved = np.degrees((s1 - s0 + np.pi) % (2 * np.pi) - np.pi)
    assert moved > 10.0, f"回转角只挪了 {moved:.1f}°"
    err = np.linalg.norm(ik.ee_pos("right") - p) * 1000
    assert err < 15.0, f"跟随把腕拖走了 {err:.1f}mm"


def test_robot_swivel(ik, home_tgt):
    """teleop 在选支窗口结束时读 _robot_swivel(先 refresh_fk)。"""
    ik.refresh_fk()
    for hand in EE_FRAME:
        a = ik._robot_swivel(hand)
        assert a is None or np.isfinite(a)


# ------------------------------------------------------------ FK 与关节映射 --
def test_refresh_fk_ee_pos(ik, home_tgt):
    """ee_pos 必须与独立构建的 pinocchio 缩减模型逐位一致 —— 挡的是
    关节序/锁关节配错(配错不报错,只安静地错)。"""
    from pink_vega_ik import DEFAULT_URDF, _NON_ARM_JOINTS
    full = pin.buildModelFromUrdf(DEFAULT_URDF)
    lock = [full.getJointId(n) for n in _NON_ARM_JOINTS
            if full.existJointName(n)]
    ref_model = pin.buildReducedModel(full, lock, pin.neutral(full))
    ref_data = ref_model.createData()
    pin.framesForwardKinematics(ref_model, ref_data, ik.config.q)
    pin.updateFramePlacements(ref_model, ref_data)
    ik.refresh_fk()
    for hand, frame in EE_FRAME.items():
        want = ref_data.oMf[ref_model.getFrameId(frame)].translation
        assert np.allclose(ik.ee_pos(hand), want, atol=1e-9)


def test_q_map_to(ik, home_tgt):
    """打乱顺序、掺入非手臂名,映射必须仍覆盖 14 关节且逐位对应。"""
    names = list(reversed(ARM_PIN_ORDER)) + ["torso_j1", "head_j2"]
    m = ik.q_map_to(names)
    assert len(m) == 14
    for i, v in m.items():
        assert v == pytest.approx(
            float(ik.config.q[ARM_PIN_ORDER.index(names[i])]))


# ------------------------------------------------------------ 回合与重播种 --
def test_reset_home(ik, home_tgt):
    """返回值 = 最大关节改动(度),与手工计算一致;构型必须回 HOME。"""
    q = ik.q_home.copy()
    q[3] += 0.3
    ik.config.q = q
    d = ik.reset_home()
    assert d == pytest.approx(np.degrees(0.3), abs=1e-6)
    assert np.allclose(ik.config.q, ik.q_home)
    assert ik.reset_home() == pytest.approx(0.0)


def test_set_config_from_isaac(ik, home_tgt):
    """与 Pink 版同语义:超限位的值夹到限位内 1e-4,缺失的关节保持现值。"""
    i6 = ARM_PIN_ORDER.index("L_arm_j6")
    keep = ARM_PIN_ORDER.index("R_arm_j7")
    before = float(ik.config.q[keep])
    isaac_q = {n: 0.1 for n in ARM_PIN_ORDER if n != "R_arm_j7"}
    isaac_q["L_arm_j6"] = 99.0          # 远超 ±1.396
    ik.set_config_from_isaac(isaac_q)
    assert ik.config.q[i6] == pytest.approx(
        ik.model.upperPositionLimit[i6] - 1e-4)
    assert ik.config.q[keep] == pytest.approx(before)
    assert ik.config.q[0] == pytest.approx(0.1)


# ------------------------------------------------------- 启动期标定/自检 --
def _fake_isaac_settled(ik, R_off=None):
    """用模型自身 FK 伪造一份「Isaac 落定读数」:chest 与 EE 位姿取 pin 世界
    系(chest 相对换算与基座系约定无关,所以这样喂是自洽的);R_off 是叠在
    EE 朝向上的已知 USD↔URDF 偏差。"""
    ik.refresh_fk()
    cp, cq = _chest_pose(ik)
    chest_pos = {h: cp for h in EE_FRAME}
    chest_quat = {h: cq for h in EE_FRAME}
    ee_pos, ee_quat = {}, {}
    for h, frame in EE_FRAME.items():
        M = ik.data.oMf[ik.model.getFrameId(frame)]
        Rm = M.rotation if R_off is None else M.rotation @ R_off
        q = pin.Quaternion(Rm)
        ee_pos[h] = M.translation.copy()
        ee_quat[h] = np.array([q.w, q.x, q.y, q.z])
    isaac_q = {n: float(ik.q_home[i]) for i, n in enumerate(ARM_PIN_ORDER)}
    return chest_pos, chest_quat, ee_pos, ee_quat, isaac_q


def test_validate_chest_self_consistent(ik, home_tgt):
    """喂模型自身 FK,残差必须≈0(teleop 启动打的就是这个数,阈值 20mm;
    这里收紧到 0.5mm —— 自洽数据差出毫米级就是换算写错了)。"""
    resid = ik.validate_chest(*_fake_isaac_settled(ik))
    assert set(resid) == {"right", "left"}
    for h, mm in resid.items():
        assert mm < 0.5, f"{h} 自洽残差 {mm:.2f}mm"


def test_calibrate_ee_rot(ik, home_tgt):
    """已知答案验证度量本身:叠一个 17° 的 EE 朝向偏差进「Isaac 读数」,
    标定必须量出 17°;之后 set_target_chest 必须用 ee_rot_fix 把它修正掉
    (修正后的目标 = 未偏差目标)。标定完构型须回 HOME。"""
    ang0 = ik.calibrate_ee_rot(*_fake_isaac_settled(ik))
    for h in EE_FRAME:
        assert ang0[h] < 0.1, f"零偏差标出 {ang0[h]:.2f}°"

    R_off = pin.exp3(np.radians(17.0) * np.array([0.0, 0.0, 1.0]))
    ang = ik.calibrate_ee_rot(*_fake_isaac_settled(ik, R_off))
    for h in EE_FRAME:
        assert ang[h] == pytest.approx(17.0, abs=0.1)
    assert np.allclose(ik.config.q, ik.q_home)

    cp, cq = _chest_pose(ik)
    p, quat = home_tgt["right"]
    M_true = ik.data.oMf[ik.model.getFrameId(EE_FRAME["right"])]
    q_biased = pin.Quaternion(M_true.rotation @ R_off)
    ik.set_target_chest("right", cp, cq, p,
                        [q_biased.w, q_biased.x, q_biased.y, q_biased.z])
    d = np.degrees(np.linalg.norm(
        pin.log3(ik._tgt["right"].rotation.T @ M_true.rotation)))
    assert d < 0.1, f"标定后目标仍偏 {d:.2f}°"


# ------------------------------------------------------------ 统计与开关位 --
def test_relax_stats_stay_zero(ik, home_tgt):
    """限位放松未实现:统计必须恒为零(teleop summary 据此如实打
    NEVER FIRED),解一步也不许变。"""
    assert ik.relax_stat == {"n": 0, "steps": 0}
    assert ik.relax_stat_pos == {"n": 0}
    ik.solve()
    assert ik.relax_stat["n"] == 0 and ik.relax_stat_pos["n"] == 0


def test_null_bias_assignable(ik, home_tgt):
    """teleop 在 --branch-at-engage 分支会整段赋值这个属性;值被收下但
    不参与求解(选支本身在 set_swivel_target 处抛错)。"""
    assert ik.null_bias == 0.0
    ik.null_bias = 1.0
    ik.solve()
    ik.null_bias = 0.0


def test_iters_validation():
    """--curobo-iters 非 20 的正倍数必须当场报错(不起子进程),错误里带
    实测前沿数据,不许安静地跑一个没人测过的配置。"""
    with pytest.raises(ValueError, match="20 的正倍数"):
        CuroboVegaIK(dt=DT, local_iters=30)
    with pytest.raises(ValueError, match="20 的正倍数"):
        CuroboVegaIK(dt=DT, local_iters=0)


# ---------------------------------------------------------------- 生命周期 --
def test_close_kills_subprocess():
    """close 后子进程必须退出、不留孤儿;重复 close 无害。用独立实例测,
    不动 session 那个(它还要给别的用例用)。"""
    solver = CuroboVegaIK(dt=DT, local_iters=40)
    pid = solver._proc.pid
    assert _pid_alive(pid)
    solver.solve()                      # 起码解过一步,close 才有统计可打
    solver.close()
    t0 = time.time()
    while _pid_alive(pid) and time.time() - t0 < 10.0:
        time.sleep(0.1)
    assert not _pid_alive(pid), f"子进程 {pid} close 后 10s 仍在 —— 孤儿"
    solver.close()                      # 第二次调用必须无害
