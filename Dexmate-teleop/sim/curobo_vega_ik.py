"""cuRobo 后端:Vega-1 双臂 14 关节逆运动学,与 PinkVegaIK 鸭子类型兼容。

这是 README「后续计划」第 1 条的求解器部分。已接进 teleop 主程序
(2026-08-11,`--ik curobo`,teleop 侧传 `--curobo-iters` 默认 40;默认
`--ik pink` 行为逐位不变)。接口契约以 `sim/teleop_vega_pico.py` 里
`pink_ik.` 的全部实际用点为准,单测在 tests/test_curobo_ik.py(挂在
scripts/check_all.py);没有用到、也没有对应机制的成员(冗余回转角控制、
限位放松)不做假实现 —— 要么如实返回零统计,要么直接抛
NotImplementedError,绝不出现「接好了从没执行」。

为什么是独立进程 + ZMQ(2026-08-10 实测,三条路都走过):

  1. cuRobo 0.8 装进 .venv-isaac 本身没问题(只追加了 25 个包,现有包一个
     没动),单独跑通过全部对比台;
  2. 但与 Isaac 同进程**两个方向都死**:cuRobo 先 import 会初始化
     site-packages 的 warp 1.14,Kit 随后加载自带的 omni.warp.core-1.8.2
     扩展时把 warp 模块搅成半初始化态(isaacsim.core.* 全部
     "module 'warp.types' has no attribute 'array'");反过来 Kit 先起,
     cuRobo 拿到的是 Kit 捆绑的 warp 1.8.2,缺 0.8 需要的新 API
     (TypeError: func() got an unexpected keyword argument 'module')。
     一山不容二 warp,版本三角无解;
  3. 于是求解器放在独立子进程里(同一个 .venv-isaac,只是不 import Isaac),
     主进程留 pinocchio 账本 + ZMQ 客户端。端口按仓库约定用 17xxx 段
     (17830-17849 自选空位),绝不碰生产端口。IPC 往返实测 0.2-0.4ms,
     对 7.5ms 的单步解算是零头。

结构:两件套。
  - **主进程(本类)只管账**:pinocchio 模型、限位、FK、chest 系换算、
    validate/calibrate,与 PinkVegaIK 共用同一批常量(从 pink_vega_ik
    import,不复制),坐标系约定因此逐位相同 —— 实测同一 HOME 构型下
    cuRobo 与 pinocchio 的 EE 位姿差 0.000mm/0.000°(服务端建成时自检,
    差超过 0.5mm/0.1° 直接拒绝启动)。
  - **子进程只管解**:cuRobo MotionRetargeter 的流式接口(solve_frame),
    每帧从当前构型热启动的速度受限局部 IK(纯 L-BFGS),关节位置限位是
    优化器的硬约束。URDF 手臂速度限位 2.4-2.7 rad/s,在 50Hz 下即每帧至多
    约 2.8-3.1°—— 逐帧跳变被结构性钳制,这是与 Pink(微分 IK,无步长
    钳制)的本质差别之一。

与 Pink 版的已知行为差异(离线对比台 sim/dev_ik_compare.py 量化,
playlist_all13 全 6889 帧):贴限位帧 5.85%→0.01%、>5° 跳变 7.90%→0.03%,
代价是腕位置误差均值 0.0→3.9mm(速度钳制造成的跟随滞后)与单步耗时
0.33→约 8ms。朝向超出腕量程时没有 Pink 的「饱和放松」机制,但实测超量程
拧腕 150° 场景下它经由整只腕(j5-j7 联动)绕行达成了需求朝向、全程贴限位
0%,需求回量程后 1.5 秒指数收敛回 0°—— 不锁死。

肘部低权位置任务不支持(按 Pink 默认 elbow_cost=0 同样是无操作;且搬肘部
位置本来就是被判死的路 —— magicdexmate/swivel.py 记载五次实测全败,腕残差
30→475mm)。操作者回转角改为**客户端零空间限速跟随**(2026-08-11,响应
「臂形要像操作者」硬需求):每帧解算后,在腕任务雅可比的零空间方向上朝
操作者回转角挪一小步,步长受 swivel_rate_deg 限速 —— 零空间修正对腕位姿
一阶无损,限速防止与腕任务打架(Pink 的 --null-bias 每子步取满线搜索最优
被实测判不可用,教训在此)。null_bias=0(默认)时整个机制不执行,行为与
接入时逐位相同。

耗时(RTX 3070 Ti Laptop,预热后稳态与预热分开报;50Hz 环给 IK 的预算
约 5ms,这组数是进程内/异步架构选型的硬门槛):

    构建+预热首解:warp 缓存热约 3s,冷缓存首次约 17s(等待上限 120s)
    稳态单步(iters=局部 L-BFGS 次数,须为 20 的倍数;默认 100):
      iters=200  核心 p50 11.3ms          —— 上游默认,无收益
      iters=100  核心 p50  7.3ms / p95 8.4ms,经 IPC p50 8.4ms(默认)
      iters= 40  核心 p50  5.2ms,经 IPC p50 5.9ms —— 质量近乎不减
                 (playlist 位置均 4.0→4.9mm、贴限位 0.01→0.09%),预算
                 吃紧时的候选
      iters= 20  核心 p50  4.5ms,但超量程后**恢复失败**(卡 95.7°),
                 不可用 —— 迭代数的硬下限在 20 与 40 之间
    ZMQ 往返占 0.7-1.1ms;2026-08-10 曾把本类临时接进 teleop 冒烟:
    同步 IPC + iters=100 下 50Hz 回放实时倍率 1.05x,没有掉帧。

本类仍须在 AppLauncher 之前构建(pinocchio 账本怕 isaacsim 覆写 eigenpy;
子进程也要赶在 Kit 抢占 GPU 之前把 CUDA 图捕获做完)。

服务端入口(通常不用手动起,客户端自己拉):
    .venv-isaac/bin/python sim/curobo_vega_ik.py --serve --ready-file /tmp/x
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pinocchio as pin

import pathlib as _pl
_sys_path_root = str(_pl.Path(__file__).resolve().parents[1])
if _sys_path_root not in sys.path:
    sys.path.insert(0, _sys_path_root)
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from magicdexmate.home_pose import ARM_HOME as HOME_ARM  # noqa: E402
from magicdexmate.swivel import swivel_angle as _swivel_angle  # noqa: E402
from pink_vega_ik import (  # noqa: E402
    ARM_PIN_ORDER,
    CHEST_FRAME,
    DEFAULT_URDF,
    EE_FRAME,
    ELBOW_FRAME,
    SHOULDER_FRAME,
    _NON_ARM_JOINTS,
    _se3,
)

PORT_RANGE = range(17830, 17850)    # 仓库约定:测试/内部通信走 17xxx
SERVER_BUILD_TIMEOUT_S = 120.0      # 冷 warp 缓存实测 17s,给足余量
SOLVE_TIMEOUT_MS = 2000             # 单步 p95 约 8ms;超时即认定服务端死亡


def _check_iters(local_iters: int):
    """iters 必须是 20 的正倍数(上游 L-BFGS 以 20 次为一个外层块;非倍数
    值没有对应的实测数据点,配错了宁可当场报错也不要安静地跑一个没人测过
    的配置)。客户端在起子进程之前查,服务端在构造核心时再查一遍。"""
    if local_iters < 20 or local_iters % 20 != 0:
        raise ValueError(
            f"local_iters={local_iters} 非法:须为 20 的正倍数(实测前沿:"
            "100→IPC p50 8.3ms 质量最优,40→5.9ms 几乎不减质,20 超量程"
            "恢复失败不可用)")


# ============================================================ 服务端 ====
class _CuroboArmCore:
    """子进程里的求解核心。只在服务端 import cuRobo/torch/warp。"""

    def __init__(self, urdf_path: str, home_arm: dict, dt: float,
                 position_weight: float, orientation_weight: float,
                 local_iters: int):
        _check_iters(local_iters)
        import torch
        from curobo.motion_retargeter import (
            MotionRetargeter,
            MotionRetargeterCfg,
        )
        from curobo.types import GoalToolPose, JointState, Pose, ToolPoseCriteria
        self._torch = torch
        self._GoalToolPose = GoalToolPose
        self._Pose = Pose

        robot_dict = {
            "robot_cfg": {
                "kinematics": {
                    "urdf_path": urdf_path,
                    "asset_root_path": str(_pl.Path(urdf_path).parent),
                    "base_link": "vega_1",
                    "tool_frames": [EE_FRAME["left"], EE_FRAME["right"]],
                    # base->EE 链之外的分支(轮、头)不进 cuRobo 的运动学树,
                    # 只有躯干在链上,须显式锁死(锁 0,与 pinocchio 的
                    # neutral 归约一致;chest 相对目标使锁值不影响手臂解)。
                    "lock_joints": {f"torso_j{i}": 0.0 for i in (1, 2, 3)},
                    "cspace": {
                        "joint_names": list(ARM_PIN_ORDER),
                        "default_joint_position":
                            [home_arm.get(n, 0.0) for n in ARM_PIN_ORDER],
                        "null_space_weight": [1.0] * 14,
                        "cspace_distance_weight": [1.0] * 14,
                        "max_jerk": 500.0,
                        "max_acceleration": 15.0,
                    },
                }
            }
        }
        # 朝向权重默认 0.05:位置以米计、朝向以弧度计,0.05 使「10mm 位置
        # 误差」与「11° 朝向误差」代价相当,与 Pink 默认(position 8.0 :
        # orientation 0.5)的比例一致。对比台扫过 0.02/0.05/0.15:0.15 在
        # 超量程时拿位置换朝向(位置误差均值 96mm),0.02 朝向恢复偏慢,
        # 0.05 两头都稳。
        crit = ToolPoseCriteria.track_position_and_orientation(
            xyz=[position_weight] * 3, rpy=[orientation_weight] * 3)
        cfg = MotionRetargeterCfg.create(
            robot=robot_dict,
            tool_pose_criteria={EE_FRAME["left"]: crit,
                                EE_FRAME["right"]: crit.clone()},
            num_envs=1,
            self_collision_check=False,     # 自碰撞由指令层 JointGuard 负责
            load_collision_spheres=False,
            optimization_dt=dt,
            local_ik_num_iters=local_iters,
        )
        self._rt = MotionRetargeter(cfg)
        # 上游缺陷的修正:MotionRetargeter._build_local_ik_solver 写的是
        # `solver.use_lm_seed = False`,但 solve_pose 读的是
        # `solver.config.use_lm_seed` —— 影子属性,LM 种子级实际仍开着,
        # 且双 EE 时被构造函数强制成 128 个随机种子。后果实测(超量程拧腕
        # 场景):随机种子逐帧换最优,位置误差均值 176mm、恢复期在 7-11°
        # 打摆 10 秒不收敛。真正关掉后:保持段位置误差均值 5mm,需求回
        # 量程后 1.5 秒指数收敛到 0°。exit_early 也要关:热启动种子常已在
        # 5mm/0.05rad 容差内,开着会不做优化直接返回旧解,跟踪永远滞后。
        _ls = self._rt._local_ik_solver
        _ls.config.use_lm_seed = False
        _ls.config.exit_early = False

        cj = list(self._rt.joint_names)
        if sorted(cj) != sorted(ARM_PIN_ORDER):
            raise RuntimeError(f"cuRobo 关节集与手臂 14 关节不符: {cj}")
        self._pin2cu = np.array([cj.index(n) for n in ARM_PIN_ORDER])
        self._cu2pin = np.array([ARM_PIN_ORDER.index(n) for n in cj])
        self._tool_frames = list(self._rt.tool_frames)
        self._last_q = None     # 上一次返回的解(cuRobo 序),判热启动连续性

        # FK 一致性自检:同一 HOME 构型下 cuRobo 与 pinocchio 的 EE 位姿
        # 必须一致(阈值 0.5mm / 0.1°)。挡住的是 URDF/锁关节/关节序配错
        # —— 配错时跟踪不会报错,只会安静地错。实测差 0.000mm/0.000°。
        full = pin.buildModelFromUrdf(urdf_path)
        lock = [full.getJointId(n) for n in _NON_ARM_JOINTS
                if full.existJointName(n)]
        model = pin.buildReducedModel(full, lock, pin.neutral(full))
        data = model.createData()
        q_home = np.array([home_arm.get(n, 0.0) for n in ARM_PIN_ORDER])
        pin.framesForwardKinematics(model, data, q_home)
        pin.updateFramePlacements(model, data)
        js = JointState.from_position(
            torch.as_tensor(q_home[self._pin2cu][None],
                            device="cuda", dtype=torch.float32),
            joint_names=cj)
        ks = self._rt.kinematics.compute_kinematics(js)
        for hand, frame in EE_FRAME.items():
            p_cu = ks.tool_poses[frame].position.view(-1).cpu().numpy()
            q_cu = ks.tool_poses[frame].quaternion.view(-1).cpu().numpy()
            M = data.oMf[model.getFrameId(frame)]
            dp = float(np.linalg.norm(p_cu - M.translation)) * 1000
            R_cu = _se3(p_cu, q_cu).rotation
            da = float(np.degrees(np.linalg.norm(
                pin.log3(R_cu.T @ M.rotation))))
            if dp > 0.5 or da > 0.1:
                raise RuntimeError(
                    f"cuRobo 与 pinocchio 的 {frame} FK 不一致:"
                    f"{dp:.2f}mm / {da:.2f}° —— 模型配置有错,拒绝启动")

        # 预热:原地解一帧,把 CUDA 图捕获、warp 内核加载都吃在构造期。
        # 目标 = HOME 自身位姿,解完必须仍在 HOME 附近,动了就是配置错了。
        tgt = {}
        for hand, frame in EE_FRAME.items():
            M = data.oMf[model.getFrameId(frame)]
            qq = pin.Quaternion(M.rotation)
            tgt[hand] = (list(map(float, M.translation)),
                         [qq.w, qq.x, qq.y, qq.z])
        q1 = self.solve(q_home, tgt)
        d = float(np.degrees(np.abs(q1 - q_home)).max())
        if d > 0.5:
            raise RuntimeError(
                f"预热求解应停在 HOME(目标=当前位姿),却动了 {d:.2f}°")

    def solve(self, q_pin: np.ndarray, targets: dict) -> np.ndarray:
        """q_pin: 当前 14 关节(ARM_PIN_ORDER 序);targets: {hand: (pos,
        quat_wxyz)},世界(基座)系。返回解出的 14 关节(同序)。"""
        t = self._torch
        want = q_pin[self._pin2cu]
        # 外部改过构型(reseed/reset_home/测试直接赋值)时,上一帧速度已无
        # 意义,清掉;连续跟踪时保留,速度/加速度正则才有对象。
        if self._last_q is None or np.abs(self._last_q - want).max() > 1e-9:
            self._rt._prev_velocity = None
        self._rt._prev_solution = t.as_tensor(
            want[None], device="cuda", dtype=t.float32)
        poses = {}
        for hand, frame in EE_FRAME.items():
            p, quat = targets[hand]
            poses[frame] = self._Pose(
                position=t.as_tensor(
                    np.asarray(p, dtype=np.float32)[None], device="cuda"),
                quaternion=t.as_tensor(
                    np.asarray(quat, dtype=np.float32)[None], device="cuda"))
        goal = self._GoalToolPose.from_poses(
            poses, ordered_tool_frames=self._tool_frames, num_goalset=1)
        r = self._rt.solve_frame(goal)
        q_cu = r.joint_state.position.view(-1).cpu().numpy().astype(np.float64)
        self._last_q = q_cu.copy()
        return q_cu[self._cu2pin]


def _serve(args) -> int:
    """REP 循环:一次一问一答。协议 json,数组用 list。"""
    import zmq
    core = _CuroboArmCore(args.urdf, HOME_ARM, args.dt,
                          args.position_weight, args.orientation_weight,
                          args.local_iters)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    port = None
    for p in PORT_RANGE:
        try:
            sock.bind(f"tcp://127.0.0.1:{p}")
            port = p
            break
        except zmq.ZMQError:
            continue
    if port is None:
        print(f"[curobo-server] {PORT_RANGE} 全被占", file=sys.stderr)
        return 1
    # 就绪文件先写临时名再改名:客户端读到的文件一定是完整的
    tmp = args.ready_file + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(str(port))
    os.replace(tmp, args.ready_file)
    print(f"[curobo-server] ready on tcp://127.0.0.1:{port}", flush=True)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    while True:
        # 5 秒醒一次查父进程:teleop 被 kill -9 时不能留孤儿占着 GPU
        if not poller.poll(5000):
            if args.parent_pid and not _pid_alive(args.parent_pid):
                print("[curobo-server] 父进程没了,退出", flush=True)
                return 0
            continue
        req = json.loads(sock.recv())
        op = req.get("op")
        if op == "ping":
            sock.send_string(json.dumps({"ok": True}))
        elif op == "solve":
            try:
                q = core.solve(
                    np.asarray(req["q"], dtype=np.float64),
                    {h: (req[h][:3], req[h][3:]) for h in ("left", "right")})
                sock.send_string(json.dumps({"ok": True,
                                             "q": [float(v) for v in q]}))
            except Exception as e:   # 解算异常要回给客户端,不能哑死
                sock.send_string(json.dumps({"ok": False, "err": repr(e)}))
        elif op == "quit":
            sock.send_string(json.dumps({"ok": True}))
            return 0
        else:
            sock.send_string(json.dumps({"ok": False,
                                         "err": f"未知 op {op!r}"}))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ============================================================ 客户端 ====
class _ArmConfig:
    """PinkVegaIK.config 的最小替身:teleop 读 `config.q`、测试写
    `config.q` 后调 `config.update()`。这里 q 就是唯一状态,update 无事可做。"""

    def __init__(self, q: np.ndarray):
        self.q = q

    def update(self):
        pass


class CuroboVegaIK:
    def __init__(self, urdf_path: str = DEFAULT_URDF,
                 home_arm: dict | None = None, dt: float = 1.0 / 50.0,
                 position_weight: float = 1.0, orientation_weight: float = 0.05,
                 local_iters: int = 100, null_bias: float = 0.0,
                 swivel_rate_deg: float = 60.0):
        # local_iters 默认 100:200 次(上游默认)与 100 次在全部三份素材
        # 上指标一致,耗时 11.3ms → 7.5ms。teleop 侧(--curobo-iters)默认 40。
        _check_iters(local_iters)
        if home_arm is None:
            home_arm = HOME_ARM

        # ---- pinocchio 账本(与 PinkVegaIK 同一套构造) ----
        full = pin.buildModelFromUrdf(urdf_path)
        lock = [full.getJointId(n) for n in _NON_ARM_JOINTS
                if full.existJointName(n)]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        self.pin_names = list(ARM_PIN_ORDER)
        self.q_home = np.array([home_arm.get(n, 0.0) for n in self.pin_names])
        self.dt = float(dt)
        self.config = _ArmConfig(self.q_home.copy())

        # 回转角跟随的增益(teleop 的 --null-bias;branch_at_engage 分支也会
        # 直接赋值这个属性)。0 = 机制整体关闭,solve 里那段代码不执行。
        # 与 Pink 的线搜索不同,这里是限速的零空间小步(见 _swivel_follow)。
        self.null_bias = float(null_bias)
        self._swivel_tgt: dict = {}
        # 回转角跟随速度上限 [rad/s]。默认 60°/s:操作者甩臂实测能到
        # 100°/s+,但跟随步是叠在解算之后的,快过腕任务的纠错速度就会打架
        # (Pink 满步线搜索腕误差 12.7→69.6mm 的教训);台架扫参定稿前先取
        # 保守值,量化数据出来后在这里更新。
        self._swivel_rate = float(np.radians(swivel_rate_deg))
        # 生效证明:n = 实际执行了跟随步的帧数,skip = 有目标但因奇异/
        # 敏感度过低跳过的帧数。teleop 的 summary 打不打由它说了算。
        self.swivel_follow_stat = {"n": 0, "skip": 0}
        self._arm_cols = {}
        for hand in EE_FRAME:
            pre = "R" if hand == "right" else "L"
            self._arm_cols[hand] = [self.pin_names.index(f"{pre}_arm_j{i}")
                                    for i in range(1, 8)]
        # 限位放松机制未实现:统计恒为零。teleop 的 summary 会如实打出
        # 「NEVER FIRED」,而不是假装放松过。
        self.relax_stat = {"n": 0, "steps": 0}
        self.relax_stat_pos = {"n": 0}

        pin.forwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)
        cm = self.data.oMf[self.model.getFrameId(CHEST_FRAME)]
        self._chest = pin.SE3(cm.rotation.copy(), cm.translation.copy())

        # 目标:每手一个世界系 SE3,起始为 HOME 位姿(等价 Pink 的
        # set_target_from_configuration 播种)
        self._tgt = {}
        for hand, frame in EE_FRAME.items():
            M = self.data.oMf[self.model.getFrameId(frame)]
            self._tgt[hand] = pin.SE3(M.rotation.copy(), M.translation.copy())
        self.last_target = dict(self._tgt)

        # 每次 solve 的经-IPC 墙钟毫秒数。在管线里跑时 Kit/渲染与求解子进程
        # 争同一块 GPU,离线台架的 5.9ms(iters=40)不代表管线内真实耗时;
        # close() 时打一行分位数,与台架数并排看就是 GPU 争用的量化。
        self.solve_ms: list = []

        # ---- 求解子进程 ----
        self._proc = None
        self._sock = None
        self._spawn_server(urdf_path, dt, position_weight,
                           orientation_weight, local_iters)
        atexit.register(self.close)

    def _spawn_server(self, urdf_path, dt, pw, ow, iters):
        import zmq
        fd, ready = tempfile.mkstemp(prefix="curobo_ik_ready_")
        os.close(fd)
        os.unlink(ready)                  # 服务端就绪时才重新出现
        log = os.path.join(tempfile.gettempdir(),
                           f"curobo_ik_server_{os.getpid()}.log")
        self._log_path = log
        cmd = [sys.executable, os.path.abspath(__file__), "--serve",
               "--urdf", urdf_path, "--dt", str(dt),
               "--position-weight", str(pw), "--orientation-weight", str(ow),
               "--local-iters", str(iters),
               "--ready-file", ready, "--parent-pid", str(os.getpid())]
        # 子进程绝不能继承 Isaac 相关注入;干净环境 + 日志落盘(不能用
        # PIPE:没人读会写满缓冲把子进程卡死)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        self._proc = subprocess.Popen(
            cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
        t0 = time.time()
        port = None
        while time.time() - t0 < SERVER_BUILD_TIMEOUT_S:
            if self._proc.poll() is not None:
                tail = ""
                try:
                    tail = "\n".join(
                        open(log).read().strip().splitlines()[-8:])
                except OSError:
                    pass
                raise RuntimeError(
                    f"cuRobo 求解子进程启动失败(退出码 "
                    f"{self._proc.returncode})。日志尾部:\n{tail}")
            if os.path.exists(ready):
                port = int(open(ready).read().strip())
                os.unlink(ready)
                break
            time.sleep(0.2)
        if port is None:
            self.close()
            raise RuntimeError(
                f"cuRobo 求解子进程 {SERVER_BUILD_TIMEOUT_S:.0f} 秒内未就绪"
                f"(冷 warp 缓存正常约 17 秒);日志:{log}")
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://127.0.0.1:{port}")
        self._port = port
        r = self._request({"op": "ping"})
        if not r.get("ok"):
            raise RuntimeError(f"cuRobo 服务端 ping 失败:{r}")
        print(f"[curobo] solver subprocess ready on tcp://127.0.0.1:{port} "
              f"({time.time()-t0:.1f}s)")

    def _request(self, obj: dict, timeout_ms: int = SOLVE_TIMEOUT_MS) -> dict:
        import zmq
        self._sock.send_string(json.dumps(obj))
        if not self._sock.poll(timeout_ms, zmq.POLLIN):
            self.close()
            raise RuntimeError(
                f"cuRobo 服务端 {timeout_ms}ms 无应答(单步 p95 约 8ms),"
                f"判定死亡;日志:{getattr(self, '_log_path', '?')}")
        return json.loads(self._sock.recv())

    def close(self):
        """结束求解子进程。atexit 注册过;重复调用无害。"""
        if self.solve_ms and not getattr(self, "_stats_printed", False):
            # 只在真解过至少一步时打(没解过就没有数,不打空行);打过一次
            # 就不再打 —— close 会被 atexit 重复调用。
            self._stats_printed = True
            a = np.asarray(self.solve_ms)
            print(f"[curobo] solve wall-time over IPC: n={len(a)} "
                  f"p50 {np.percentile(a, 50):.2f}ms "
                  f"p95 {np.percentile(a, 95):.2f}ms "
                  f"p99 {np.percentile(a, 99):.2f}ms max {a.max():.2f}ms "
                  f"(离线台架 iters=40 p50 5.9ms,差值即 GPU 争用)")
        if self._sock is not None:
            try:
                self._sock.send_string(json.dumps({"op": "quit"}))
                self._sock.poll(500)
            except Exception:
                pass
            self._sock.close()
            self._sock = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc_done = True

    # -- chest 相对目标(与 PinkVegaIK 同一套数学) ---------------------------
    def set_target_chest(self, hand: str, chest_pos, chest_quat_wxyz,
                         tgt_pos, tgt_quat_wxyz):
        chest_isaac = _se3(chest_pos, chest_quat_wxyz)
        tgt_isaac = _se3(tgt_pos, tgt_quat_wxyz)
        tgt_rel_chest = chest_isaac.inverse() * tgt_isaac
        fix = getattr(self, "ee_rot_fix", {}).get(hand)
        if fix is not None:
            tgt_rel_chest = pin.SE3(fix @ tgt_rel_chest.rotation,
                                    tgt_rel_chest.translation)
        tgt_world = self._chest * tgt_rel_chest
        self.last_target[hand] = tgt_world
        self._tgt[hand] = tgt_world

    def set_elbow_target_chest(self, hand: str, chest_pos, chest_quat_wxyz,
                               tgt_pos):
        """肘部低权任务未实现。Pink 默认(elbow_cost=0)下这个调用同样是
        无操作,行为一致;若要肘部跟随,用 Pink 后端。"""
        return

    def set_target(self, hand: str, pos_base, quat_base_wxyz):
        self._tgt[hand] = _se3(pos_base, quat_base_wxyz)
        self.last_target[hand] = self._tgt[hand]

    def hold_target(self, hand: str):
        """脱开的手臂:目标冻结在当前构型的 FK 位姿。"""
        pin.framesForwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[self.model.getFrameId(EE_FRAME[hand])]
        self._tgt[hand] = pin.SE3(M.rotation.copy(), M.translation.copy())

    def set_swivel_target(self, hand: str, angle_rad: float | None):
        """这条臂的回转角目标(操作者肘绕自己肩-腕轴的角度),None = 清除。
        只在 null_bias > 0 时参与求解(见 _swivel_follow);teleop 的
        --null-bias 0 默认下这里只是存值,行为与不支持时逐位相同。"""
        self._swivel_tgt[hand] = angle_rad

    def _swivel_follow(self):
        """解算后的零空间限速步:把每条臂的回转角朝操作者目标挪一点。

        为什么放在客户端而不是 cuRobo 内部:仓库里两条已付学费的路都不走 ——
        ① 肘部位置任务(人机比例不同,目标常不可达,腕残差 30→475mm,
        magicdexmate/swivel.py 记载);② Pink 的每子步满步线搜索(与腕任务
        争,腕误差 12.7→69.6mm)。这里取零空间方向(腕任务雅可比 SVD 的
        末奇异向量,对腕位姿一阶无损),而且步长限速(_swivel_rate × dt ×
        null_bias),错不动腕、也追不疯。敏感度用数值差分现算,不依赖
        cuRobo 内部结构 —— 协议与服务端零改动。

        代价(如实):客户端改写 q 后,服务端下一帧检测到构型变动会清掉
        速度状态,加速度正则失去历史;速度钳制不受影响(它相对客户端送去
        的 q)。抖动是否可接受由台架素材 4 出数。
        """
        q = self.config.q
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for hand, want in self._swivel_tgt.items():
            if want is None:
                continue
            got = self._robot_swivel(hand)
            if got is None:
                self.swivel_follow_stat["skip"] += 1
                continue
            err = (want - got + np.pi) % (2 * np.pi) - np.pi
            cols = self._arm_cols[hand]
            pin.computeJointJacobians(self.model, self.data, q)
            J = pin.getFrameJacobian(
                self.model, self.data, self.model.getFrameId(EE_FRAME[hand]),
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, cols]
            _, sv, Vt = np.linalg.svd(J)
            if sv[-1] < 1e-3 * max(sv[0], 1e-9):
                self.swivel_follow_stat["skip"] += 1   # 奇异态,零空间不可信
                continue
            n = Vt[-1]
            # 数值敏感度:沿零空间方向挪 1 毫弧度,回转角变多少
            eps = 1e-3
            qq = q.copy()
            qq[cols] = qq[cols] + eps * n
            pin.framesForwardKinematics(self.model, self.data, qq)
            got_eps = self._robot_swivel(hand)
            pin.framesForwardKinematics(self.model, self.data, q)  # 还原 FK
            if got_eps is None:
                self.swivel_follow_stat["skip"] += 1
                continue
            g = ((got_eps - got + np.pi) % (2 * np.pi) - np.pi) / eps
            if abs(g) < 1e-2:      # 该方向几乎不动回转角,除出来是天文步长
                self.swivel_follow_stat["skip"] += 1
                continue
            # 双重限速,量纲各自成立:回转角侧每帧至多 rate×dt×bias,
            # 关节侧每帧至多 0.05 rad(≈2.9°,与手臂速度限位 vel×dt 同量级
            # —— 敏感度 g 小时 ds/g 会放大,必须再钳一次关节步长)。
            ds = float(np.clip(err, -self._swivel_rate * self.dt * self.null_bias,
                               +self._swivel_rate * self.dt * self.null_bias))
            step = float(np.clip(ds / g, -0.05, 0.05))
            q[cols] = np.clip(q[cols] + step * n,
                              self.model.lowerPositionLimit[cols],
                              self.model.upperPositionLimit[cols])
            self.swivel_follow_stat["n"] += 1
            pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        self.config.q = q

    # -- 求解 -----------------------------------------------------------------
    def solve(self):
        req = {"op": "solve", "q": [float(v) for v in self.config.q]}
        for hand in ("left", "right"):
            M = self._tgt[hand]
            q = pin.Quaternion(M.rotation)
            req[hand] = [float(M.translation[0]), float(M.translation[1]),
                         float(M.translation[2]),
                         float(q.w), float(q.x), float(q.y), float(q.z)]
        t0 = time.perf_counter()
        r = self._request(req)
        self.solve_ms.append((time.perf_counter() - t0) * 1000.0)
        if not r.get("ok"):
            raise RuntimeError(f"cuRobo 求解失败:{r.get('err')}")
        self.config.q = np.asarray(r["q"], dtype=np.float64)
        if self.null_bias > 0.0 and any(
                v is not None for v in self._swivel_tgt.values()):
            self._swivel_follow()
        return self.config.q.copy()

    # -- 与 Pink 同名同义的账务方法 -------------------------------------------
    def q_map_to(self, isaac_names: list[str]) -> dict[int, float]:
        q = self.config.q
        name_to_q = {n: q[i] for i, n in enumerate(self.pin_names)}
        return {isaac_names.index(n): float(name_to_q[n])
                for n in self.pin_names if n in isaac_names}

    def refresh_fk(self):
        pin.framesForwardKinematics(self.model, self.data, self.config.q)
        pin.updateFramePlacements(self.model, self.data)

    def ee_pos(self, hand: str) -> np.ndarray:
        fid = self.model.getFrameId(EE_FRAME[hand])
        return self.data.oMf[fid].translation.copy()

    def reset_home(self) -> float:
        """回合开始:构型拉回默认姿势,返回最大关节改动(度)。服务端下一次
        solve 看到 q 跳变会自行清掉速度状态,无需另发消息。"""
        d = float(np.degrees(np.abs(self.config.q - self.q_home)).max())
        self.config.q = self.q_home.copy()
        return d

    def set_config_from_isaac(self, isaac_q: dict):
        q = np.array([isaac_q.get(n, self.config.q[i])
                      for i, n in enumerate(self.pin_names)])
        m = 1e-4
        q = np.clip(q, self.model.lowerPositionLimit + m,
                    self.model.upperPositionLimit - m)
        self.config.q = q

    def _robot_swivel(self, hand: str) -> float | None:
        sh = self.data.oMf[self.model.getFrameId(
            SHOULDER_FRAME[hand])].translation
        el = self.data.oMf[self.model.getFrameId(ELBOW_FRAME[hand])].translation
        wr = self.data.oMf[self.model.getFrameId(EE_FRAME[hand])].translation
        return _swivel_angle(sh, el, wr)

    def calibrate_ee_rot(self, chest_pos, chest_quat_wxyz, ee_pos,
                         ee_quat_wxyz, isaac_q: dict) -> dict:
        """USD->URDF EE 系朝向偏差的一次性标定,与 Pink 版同一套数学。"""
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
        self.config.q = self.q_home.copy()
        return angles

    def validate_chest(self, chest_pos, chest_quat_wxyz, ee_pos, ee_quat_wxyz,
                       isaac_q: dict):
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


def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="以求解服务端运行")
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--dt", type=float, default=1.0 / 50.0)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--orientation-weight", type=float, default=0.05)
    ap.add_argument("--local-iters", type=int, default=100)
    ap.add_argument("--ready-file", default="")
    ap.add_argument("--parent-pid", type=int, default=0)
    args = ap.parse_args()
    if not args.serve:
        ap.error("本文件作为脚本只支持 --serve(客户端请 import CuroboVegaIK)")
    if not args.ready_file:
        ap.error("--serve 需要 --ready-file")
    return _serve(args)


if __name__ == "__main__":
    raise SystemExit(_main())
