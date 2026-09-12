#!/usr/bin/env python
"""PICO -> Vega-1P dual-arm teleop in Isaac Lab (upper body only).

FINAL RIG HAS NO CONTROLLERS: headset + wrist/waist body trackers only.
Nothing on the operating path may require a grip/trigger/button press.

Pipeline (P4 of plans/00_roadmap.md, sim phase):
  producer (py3.10): scripts/teleop_pico_producer.py -> ZMQ raw poses :5581
  this script (.venv-isaac): raw poses -> yaw-compensated wrist frames
  (magicdexmate.pico.xr_pose) -> clutched incremental 6DoF targets ->
  DifferentialIKController per arm -> joint position targets.

Two input mappings (--mode):
  trackers (default, the final rig): wrist target = a wrist tracker,
    referenced to the WAIST tracker (body-relative - the natural G1-style
    mapping, per gear_sonic/scripts/pico_manager_thread_server.py). Waist
    also drives the torso. Engagement is automatic once the trackers are
    present (no grip involved); re-zero/pause = keyboard R/SPACE.
    Tracker names default to the producer's fixed LWRIST/RWRIST/WAIST.
  controllers (legacy, kept for bring-up comparison only): wrist target =
    held controller pose, referenced to the HEADSET. Grip-driven clutch.

Clutch model (per hand):
  ENGAGE  = automatic in trackers mode (trackers present + settle delay);
            grip >= 0.6 in legacy controllers mode. Captures the wrist +
            reference + current EE poses; target = EE_ref + scaled delta
            of the wrist relative to the (frozen) reference.
  release = arm freezes at the last commanded pose.
  re-engage = re-zeroes (keyboard R forces one in trackers mode).

No hands/fingers, base parked, head fixed (phase-1 scope). EE targets are
one-euro filtered by default (--filter-mincutoff 0 disables); the reference
is frozen at engage (--head-ref live restores per-frame reference tracking).

Run (mock, headless smoke with screenshots):
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_pico.py \
      --source mock --headless --duration 12 --snap-dir /tmp/vega_pico_snaps
Run (live, after starting the producer in a py3.10 env):
  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_pico.py --source zmq
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vega-1P PICO dual-arm teleop")
parser.add_argument("--source", choices=["mock", "zmq", "replay"], default="mock")
parser.add_argument("--sub", default="tcp://127.0.0.1:5581", help="producer PUB address")
parser.add_argument("--replay-file", default=None,
                    help="producer --record file (required with --source replay)")
parser.add_argument("--replay-loop", action="store_true",
                    help="loop the replay file instead of holding the last frame")
parser.add_argument("--mock-stress", action="store_true",
                    help="aggressive mock trajectory (probe IK joint winding)")
parser.add_argument("--mock-demo", action="store_true",
                    help="pose-hold demo trajectory (forward/lateral/overhead, "
                         "3s holds) + automatic mapping scorecard: at each "
                         "hold midpoint the EE displacement from its anchor "
                         "is compared against the skeleton's ground-truth "
                         "offset mapped into the robot frame. The headless "
                         "regression for skeleton->robot mapping consistency")
parser.add_argument("--pink-smooth-cost", type=float, default=0.0,
                    help="IK 内部的平滑项权重,0 = 关闭(默认)。目标 = 当前"
                         "关节,即惩罚这一步本身的移动量,快而别扭的目标会变成"
                         "慢一点的逼近而不是关节跳变。V2AP 的做法(其 0.2 对"
                         "position 50;按我们的 position 8 等比约 0.03)。")
parser.add_argument("--wrist-map", choices=("absolute", "incremental"), default="",
                    help="腕部朝向映射,**一个选择顶两个开关**。两者的取舍是:\n"
                         "  absolute  = 机器人掌心朝你掌心朝的方向(2026-08-01 "
                         "实机签收的那套)。代价:任何手腕姿势都直接换算成绝对"
                         "角度,而 j6 只有 ±80 度、j7 只有 143 度总量程,而人的"
                         "前臂滚转有 150 至 180 度 —— 可能从第一帧就超量程,顶死"
                         "之后朝向一直是错的(实测 14.15%% 的帧顶在限位上)。\n"
                         "  incremental = 只映射「相对回合起点转了多少」,起点是"
                         "默认姿势(j6/j7 都在量程正中央,两边各有 80/64 度余量),"
                         "所以只要单段回合内的腕部转动不超过余量就撞不到限位;"
                         "常量安装角也被自动约掉。代价:机器人掌心与你的掌心"
                         "差一个固定角,要靠看画面适应。\n"
                         "  展开成:absolute = --ori-mode palm --anchor current;"
                         "incremental = --ori-mode track --anchor home。"
                         "**不要手工把这两个开关配成别的组合** —— 增量朝向配"
                         "漂移锚点实测是 96.98%%,比什么都不做糟得多。\n"
                         "  ⚠ 2026-08-09 离线实测:**增量式在我们这条链路上并没有"
                         "更不容易超量程,反而差得多**。四种条件下 incremental 都是"
                         "80 至 97%% 的帧顶在限位上,absolute 都是 14 至 60%%"
                         "(锚点两种、短素材、锁躯干,逐个排除过)。腕误差也从"
                         "18/13mm 涨到 50/47mm。最像根因的是**追踪器数据本身**:"
                         "PICO 的腕朝向同一姿势两次穿戴能差 3.8 至 18.5 度,"
                         "增量式把每帧误差都累进去,绝对式每帧独立不累积。"
                         "要用它必须自己戴上头显判手感,别拿这些离线数字背书。")
parser.add_argument("--ori-ref", choices=("home", "anchor"), default="home",
                    help="--ori-scale 缩放时以什么为参考。anchor = 啮合时的实测"
                         "末端(**已实测证明是坏的**:它不是中立参考,把目标往"
                         "它那儿拉等于破坏绝对掌心映射,顶限位从 14%% 涨到 71%%);"
                         "home(默认)= 启动时定标的默认姿势末端,量程中央。")
parser.add_argument("--relax-at-limit", type=float, default=1.0,
                    help="腕部 j6/j7 顶到限位时,把那一条手臂这一步的朝向权重乘以"
                         "这个数,并把朝向目标向当前朝向插值(饱和时不再往墙上推),"
                         "1.0(默认)= 关闭,控制台默认传 0.01。曾因「负载下 "
                         "81.25%%」撤回过:那是旧版触发条件读全部 7 关节造成的"
                         "自锁,2026-08-10 已修(只看 j6/j7 + 2° 地板 + 目标插值),"
                         "修后空载 7.00%% / 4 核负载 7.23%%,负载敏感性消失。"
                         "机理回归:sim/dev_relax_latch.py。")
parser.add_argument("--relax-margin", type=float, default=2.0,
                    help="距限位多少度算顶住了,配合 --relax-at-limit")
parser.add_argument("--relax-pos-at-limit", action="store_true",
                    help="肩肘(j1-j4)贴限位(= 位置够不着)时,把位置目标向"
                         "当前末端插值,不再把手臂压在限位上磨。动机:clip_headwaist"
                         " 上腕部残余贴限位的 94%% 与同臂肩肘同时发生 —— 位置任务"
                         "在够不着时借腕关节抓毫米。与 --relax-at-limit 独立。")
parser.add_argument("--ori-scale", type=float, default=1.0,
                    help="朝向缩放,1.0(默认)= 1:1。相对啮合锚点的转动按此比例"
                         "缩小。顶限位的是腕部 j6(±80°)与 j7(−79 至 64°),"
                         "而人的前臂滚转有 150 至 180°,1:1 必然把腕推出量程。"
                         "**但实测反而更糟,不要再试**:playlist_all13 上顶限位"
                         "的帧从 14.15%% 涨到 71.24%%(0.7)与 83.86%%(0.5),"
                         "最长连续卡死 5.2 秒涨到 90 秒以上。原因是锚点 ee_q0 "
                         "取自啮合时的实测末端,把目标往它那儿拉等于破坏了已经"
                         "签收的绝对掌心映射。开关留着只是为了记住这个结果。")
parser.add_argument("--pink-gain", type=float, default=1.0,
                    help="IK 每次迭代收敛掉多少比例的位姿误差。1.0(默认)= 全速"
                         "收敛,子步内只受求解器速度盒约束,一个幅度大或别扭的"
                         "目标会让关节直冲限位。V2AP 用 0.2,即把一阶低通做进"
                         "求解器本身 —— 他们防怪姿态靠的不是加约束,是降增益。")
parser.add_argument("--reseed-at-engage", action="store_true",
                    help="离合闭合时把求解器的内部构型重置为仿真实测的关节角。"
                         "不加这个开关时,该构型是整场会话的持久开环状态:一旦"
                         "绕进坏支或贴上限位,重新啮合也只换锚点不换构型,坏姿态"
                         "会被继承下去,后续动作全部不准。")
parser.add_argument("--joint-vel-scale", type=float, default=0.0,
                    help="手臂关节速度限幅,单位为真机速度上限的倍数,0 = 关闭"
                         "(默认)。1.0 = 按硬件真实上限([2.4,2.4,2.7...] rad/s)"
                         "限幅 —— 超过它的指令在真机上本来就做不到,限幅只是让"
                         "仿真不再假装做得到。V2AP 用 0.4。出自其动作环的移植,"
                         "见 magicdexmate/joint_guard.py。")
parser.add_argument("--self-collision", choices=["off", "hold"], default="off",
                    help="自碰撞防护。hold = 指令姿势(含手爪余量 margin)一旦"
                         "互撞,手臂保持上一个安全指令不动,直到目标离开碰撞区。"
                         "检测 0.14ms/步。手的体积不在模型里(URDF 到 l8 为止),"
                         "margin 补一部分;桥接的 80mm 腕距门限保留为最后一道。")
parser.add_argument("--collision-margin", type=float, default=0.015,
                    help="自碰撞判定的安全距离,单位为米")
parser.add_argument("--branch-strength", type=float, default=1.0,
                    help="选支窗口内线搜索的强度,配合 --branch-at-engage。"
                         "1.0 = 每步都取满线搜索最优解,挪得快但与腕任务争得凶;"
                         "小于 1 则每步只挪一部分,需要配更长的窗口。")
parser.add_argument("--branch-window", type=float, default=0.5,
                    help="选支窗口长度,单位为秒,配合 --branch-at-engage。"
                         "窗口内每个控制步都跑一次线搜索,让分支逐步挪过去,"
                         "期间腕任务照常跟踪;窗口结束后完全不再碰零空间。"
                         "设为 0 等于一步跳过去,实测会在每次啮合后留下瞬态。")
parser.add_argument("--branch-at-engage", action="store_true",
                    help="离合闭合时用操作者自己的回转角选一次手臂分支,之后"
                         "完全不碰零空间。默认关闭。冗余那一维没有任何东西钉住,"
                         "解算器沿用历史落到的那一支,同素材同参数实测最坏差 238°,"
                         "表现为臂形不可复现、左臂偶发关节白动。与 --null-bias "
                         "的区别是只做一次,不在跟踪期间和腕任务争。")
parser.add_argument("--null-bias", type=float, default=0.0,
                    help="用操作者自己的回转角钉住手臂那一维冗余。"
                         "0 = 关闭,是默认值,也是目前唯一可用的值。"
                         "⚠ 打开会让效果全面变差,不要因为静态测试的数字而打开:"
                         "2026-08-05 在 playlist_all13 上实测,腕误差由 17.8/12.7mm "
                         "涨到 44.3/69.6mm,逐帧跳变 p99 由 7.3° 涨到 31.2°,"
                         "实时倍率由 2.18x 掉到 0.97x。原因见 workflow_fj.md 续十九:"
                         "线搜索每个子步都取满最优解,把手臂拖离腕目标的速度"
                         "超过腕任务把它拉回来的速度。静态目标上收敛得很好"
                         "(两臂 0.2° 到位、腕残差亚毫米),但那不是实际用法。"
                         "留着这个开关是为了继续试:候选做法是只在离合闭合那一刻"
                         "做一次全局选支,之后纯局部跟踪,不要每步都拉。")
parser.add_argument("--null-k", type=float, default=0.0,
                    help="EXPERIMENTAL null-space home-bias gain per control "
                         "step (0 = off, the verified default). 2026-07-20 "
                         "sweep: no benefit on feasible trajectories, harmful "
                         "when the EE task is infeasible (clamped joints break "
                         "the projection); prefer clutch re-zeroing, or Pink "
                         "IK if real-hw winding demands a solver fix")
parser.add_argument("--null-cap", type=float, default=0.01,
                    help="max null-space step per control tick [rad]; the pull "
                         "is uniformly scaled down to this (scaling keeps it "
                         "inside null(J)). Uncapped k*(q_home-q) demands "
                         "~rad/s-scale jumps when wound up and destabilizes "
                         "the IK (max EE err >1m in the sweep); 0.01 at 60Hz "
                         "= 0.6rad/s. inf restores the uncapped behavior")
parser.add_argument("--mode", choices=["controllers", "trackers"],
                    default="trackers",
                    help="trackers (default, the final rig - no controllers): "
                         "wrist target from a wrist tracker, referenced to the "
                         "WAIST tracker (body-relative, the natural G1-style "
                         "mapping); engages automatically, keyboard R/SPACE "
                         "to re-zero/pause. controllers = LEGACY grip-clutch "
                         "path (wrist target from a held controller, headset "
                         "reference), kept for comparison only")
parser.add_argument("--tracker-left", default="LWRIST",
                    help="[--mode trackers] left wrist tracker name/serial "
                         "(default = the producer's fixed body-joint name, "
                         "also what the mock source publishes)")
parser.add_argument("--tracker-right", default="RWRIST",
                    help="[--mode trackers] right wrist tracker name/serial")
parser.add_argument("--tracker-waist", default="WAIST",
                    help="[--mode trackers] waist tracker serial = the body "
                         "root the wrist targets are relative to, and the "
                         "torso-lean source")
parser.add_argument("--ik", choices=["pink", "dls", "curobo"], default="pink",
                    help="pink = Pink QP IK with a null-space posture task + "
                         "hard joint limits (prevents the redundant-DOF winding "
                         "DLS suffers; the reference G1 solver, required now "
                         "that there's no grip clutch to un-wind with). dls = "
                         "the old DifferentialIKController, kept for comparison. "
                         "curobo = cuRobo 速度受限局部 IK,GPU 求解子进程"
                         "(端口 17830-17849,见 curobo_vega_ik.py):逐帧跳变"
                         "被 URDF 速度限位结构性钳制,离线台架 >5° 跳变 "
                         "7.90%%→0.03%%、贴限位 5.85%%→0.01%%,代价腕位置误差"
                         "均值 +4mm(速度钳制的跟随滞后)。不支持选支与回转角"
                         "(--branch-at-engage / --null-bias>0 会在用到的那一刻"
                         "抛 NotImplementedError),没有限位放松(--relax-* 统计"
                         "恒为零,summary 如实打 NEVER FIRED)")
parser.add_argument("--curobo-iters", type=int, default=40,
                    help="[--ik curobo] 每帧局部 L-BFGS 迭代数,须为 20 的倍数。"
                         "离线台架实测(RTX 3070 Ti,50Hz 素材,经 IPC 单步):"
                         "100 → p50 8.3ms,质量最优;40(默认)→ 5.9ms,几乎"
                         "不减质(playlist 位置均值 4.0→4.9mm、贴限位 "
                         "0.01→0.09%%);20 → 4.5ms 但超量程后恢复失败"
                         "(卡 95.7° 不收敛),不可用 —— 硬下限在 20 与 40 "
                         "之间")
parser.add_argument("--hands", choices=["both", "right", "left"], default="both")
parser.add_argument("--control_hz", type=float, default=60.0)
parser.add_argument("--physics-hz", type=float, default=240.0,
                    help="physics step rate. KEEP 240: 2026-07-21 sweep found "
                         "120 nearly doubles the real-time factor (0.31x->"
                         "0.58x) but the stiff arm PD (KP up to 2500) goes "
                         "unstable at dt=1/120 - tracking error blows up "
                         "5x (115mm->610mm). Lower it only if you also soften "
                         "the arm gains in vega_scene.py")
parser.add_argument("--render-interval", type=int, default=4,
                    help="physics steps per render (GUI/camera only; a no-op "
                         "in headless-without-cameras). The old default 2 = "
                         "120fps rendering, pure waste; 4 = 60fps")
parser.add_argument("--head-ref", choices=["engage", "live"], default="engage",
                    help="[--mode controllers only] headset reference for the "
                         "yaw-comp frame. engage = freeze the headset pose at "
                         "clutch close, so head motion during a hold no longer "
                         "moves the arms (hardware day #1 issue); live = old "
                         "per-frame behavior. Trackers mode ALWAYS uses the "
                         "live waist reference (body-relative needs a current "
                         "root; a frozen one turns leaning/turning into arm "
                         "drift - 2026-07-26 live session)")
parser.add_argument("--filter-mincutoff", type=float, default=1.0,
                    help="one-euro min cutoff [Hz] on the EE target (pos + "
                         "quat). Lower = smoother at rest; 0 disables the "
                         "filter entirely")
parser.add_argument("--filter-beta", type=float, default=2.0,
                    help="one-euro speed coefficient: opens the cutoff with "
                         "target speed so fast moves keep low lag (2.0 = "
                         "26mm lag at a fast 1 m/s sweep, jitter at rest "
                         "still cut to ~24%%)")
parser.add_argument("--auto-engage", action="store_true",
                    help="[deprecated no-op] trackers mode ALWAYS engages "
                         "automatically now (the final rig has no controllers "
                         "so a grip gate would be a dead path); flag kept so "
                         "old launch commands don't break")
parser.add_argument("--auto-engage-delay", type=float, default=1.0,
                    help="[--mode trackers] wait this long after start before "
                         "capturing neutral, so the robot has settled first")
parser.add_argument("--no-torso", action="store_true",
                    help="[--mode trackers] don't drive the torso from the "
                         "waist tracker (arms only)")
parser.add_argument("--torso-tracker", default="off",
                    help="[--mode controllers] chest Swift tracker -> torso "
                         "lean. off | auto (first tracker seen) | <serial>. "
                         "Engages/freezes with the arm clutch. In --mode "
                         "trackers the waist tracker drives the torso "
                         "automatically (see --no-torso)")
parser.add_argument("--torso-scale", type=float, default=0.7,
                    help="operator chest-lean -> robot chest-lean gain. The "
                         "belt sits on the pelvis and the operator hinges "
                         "ABOVE it, so the tracker under-reads the bend: "
                         "measured over the 2026-08-03 session its forward "
                         "swing was 18.7 deg while the torso axis (waist->head) "
                         "swung 40.4. The operator then reported bending 40 deg "
                         "for under 10 of robot bend, which back-solves to ~27 "
                         "deg of tracker reading, i.e. a 0.67 under-read. 0.7 "
                         "puts their full bend near the 30 deg allowance. This "
                         "is a back-solve from one report, not a fit - a clean "
                         "upright->max-bend recording would pin it properly.")
parser.add_argument("--torso-fwd-deg", type=float, default=30.0,
                    help="how much further FORWARD than the home the chest may "
                         "lean, deg. Operator's number, 2026-08-03: home is 55 "
                         "deg of lean, so the chest works over 45..85. The "
                         "joint would allow 65 more (a 120 deg fold), which is "
                         "not what anyone wants.")
parser.add_argument("--torso-back-deg", type=float, default=10.0,
                    help="how much further BACK than the home. Same source: 10.")
parser.add_argument("--torso-max-deg", type=float, default=60.0,
                    help="[--waist-mode authored only] symmetric clamp; j3 mode "
                         "uses --torso-fwd-deg / --torso-back-deg.")
parser.add_argument("--torso-sign", type=int, choices=[1, -1], default=-1,
                    help="lean-forward direction flip; default verified in "
                         "sim (torso_j1 axis is -y in base -> forward lean "
                         "= negative j1 delta)")
parser.add_argument("--pub-state", default="tcp://127.0.0.1:5583",
                    help="publish Isaac's ACTUAL joint positions here so a "
                         "viewer can mirror them instead of re-deriving the "
                         "mapping. Two implementations of one mapping drift - "
                         "the preview twice showed something the robot was not "
                         "doing (a different --control-mode, a different law). "
                         "Mirroring what Isaac actually has cannot drift. "
                         "Empty string disables.")
parser.add_argument("--no-sym-lock", dest="sym_lock", action="store_false",
                    help="disable the bimanual symmetriser. On by default: "
                         "the operator asked that arms which are symmetric on "
                         "them come out symmetric on the robot, and that can "
                         "only be guaranteed in the targets")
parser.add_argument("--control-mode",
                    choices=["arms", "arms+head", "arms+head+waist"],
                    default="arms",
                    help="which joint groups are commanded. Chosen HERE, before "
                         "anything is sent to the robot - the robot does not "
                         "decide. arms (default) = today's verified behaviour, "
                         "torso and head held. arms+head = the headset drives "
                         "head_j1..j3. arms+head+waist = the waist tracker's "
                         "lean also drives torso_j1..j3 along the authored "
                         "path, and the arm bind point follows the live chest")
parser.add_argument("--pos-scale", type=float, default=1.35,
                    help="human-wrist-delta -> EE-delta scale. Operator picked "
                         "1.35 with --waist-bind-up 0.08 from the preview "
                         "(2026-08-01). NOTE: at these values four authored "
                         "poses need more than the 0.771 m arm - lateral "
                         "(both arms, 0.79) and arms-down / bow-30 (RIGHT "
                         "only, 0.79-0.81). One-arm-only saturation is what "
                         "splits the left/right solution; the bimanual "
                         "symmetriser covers arms-down (34 mm mirror gap) but "
                         "only weakly covers bow-30 (105 mm). Watch for it.")
parser.add_argument("--pos-max", type=float, default=0.80,
                    help="clamp on the commanded EE translation delta [m]")
parser.add_argument("--wrist-max-deg", type=float, default=120.0,
                    help="clamp on the commanded EE rotation delta")
parser.add_argument("--hold", action="store_true",
                    help="diagnostic: pin both arms to the spawn EE pose (IK sanity)")
parser.add_argument("--jacobi-row", choices=["fixedbase", "body"], default="fixedbase",
                    help="fix_root_link=True makes this a standard fixed-base "
                         "articulation -> row body_idx-1 expected; verify via --hold")
parser.add_argument("--stale_ms", type=float, default=200.0,
                    help="producer-link staleness: no new frame for this long "
                         "(wall clock) = link dead")
parser.add_argument("--device-stale-ms", type=float, default=300.0,
                    help="device staleness: SDK device_ts frozen for this long "
                         "= headset asleep/streaming stopped, even though the "
                         "producer keeps sending. Both trigger a clutch release")
parser.add_argument("--tracker-freeze-ms", type=float, default=500.0,
                    help="body-estimate freeze detection: all tracker values "
                         "bit-identical for this long = frozen estimator. The "
                         "device clock keeps ticking through this failure "
                         "(2026-07-26 live: 15s of waving, zero data motion) "
                         "so --device-stale-ms can't see it. Forces a clutch "
                         "release like any staleness; 0 disables")
parser.add_argument("--no-glitch-gate", action="store_true",
                    help="disable the per-tracker glitch gate. The gate "
                         "rejects physically impossible pose steps (2026-07-26 "
                         "audit: hundreds of wrist spikes up to 170deg/step "
                         "while waist/head were clean) and turns sustained "
                         "relocations into a clean re-anchor instead of "
                         "swinging the arm through the jump")
parser.add_argument("--ori-mode", choices=["palm", "track", "hold"], default="track",
                    help="wrist orientation: track = follow the tracker's "
                         "orientation delta (default). hold = freeze the EE "
                         "orientation at its engage pose and track position "
                         "only - use while the source's orientation channel "
                         "is noisy (live wrists spiked 170deg/step; a bad "
                         "orientation target also drags position through the "
                         "8:2 IK cost trade)")
parser.add_argument("--elbow-weight", type=float, default=0.0,
                    help="Pink elbow position-task cost (0 = off). Pulls the "
                         "robot elbow (L/R_arm_l4) toward the operator's "
                         "elbow (LELBOW/RELBOW, body joints 18/19) so the "
                         "redundant DOF resolves human-like instead of "
                         "toward q_home. Try 1.0 (EE position cost is 8)")
parser.add_argument("--pink-substeps", type=int, default=3,
                    help="Pink QP substeps per control tick (convergence "
                         "speed vs CPU); 6 halves the transient lag")
parser.add_argument("--pink-ori-cost", type=float, default=0.5,
                    help="Pink FrameTask orientation cost (position cost is 8). "
                         "0.5 (2026-07-27 sweep): keeps wrist orientation "
                         "tracking while no longer trading away position "
                         "accuracy; the old 2.0 cost position dearly whenever "
                         "the ori demand got hard. ~0 = position-only")
parser.add_argument("--anchor", choices=["current", "home"], default="current",
                    help="EE-side clutch anchor. current = wherever the EE is "
                         "at engage (ratchet-style large-workspace roaming, "
                         "default). home = always the robot's settled home EE "
                         "pose: engage while holding the matching neutral "
                         "posture (hands in front, elbows down) and the "
                         "mapping becomes repeatable imitation-style - your "
                         "hands relative to your body = its hands relative "
                         "to its body, identical after every re-anchor")
parser.add_argument("--map", choices=["anchored", "waist-abs", "shoulder-rel",
                                     "chest-rel", "chest-anchor"],
                    default="anchored",
                    help="Wrist mapping law. anchored = incremental from the "
                         "engage-time anchor pair (original behavior). "
                         "waist-abs = user-specified absolute mapping "
                         "(2026-07-28): torso locked, human waist bound to the "
                         "robot's anatomical waist (chest - 0.25 m), and the "
                         "human waist->wrist vector commands the robot "
                         "waist->EE vector at --pos-scale (1:1 recommended: "
                         "shrinking pulls targets into the low dead zone). "
                         "No anchor state, no drift; requires Pink.")
parser.add_argument("--waist-bind-up", type=float, default=0.20,
                    help="[--map waist-abs] extra vertical offset [m] added to "
                         "the robot-side waist bind point (chest - 0.25 m + "
                         "this). 0.20 calibrated on the 2026-07-29 labeled "
                         "pose capture: 100%% of real human poses (incl. "
                         "hands-on-hips / arms-down) reach with 0 mm Pink "
                         "error; 0.0 loses the low workspace. It was moved to "
                         "0.08 on 2026-08-01 from the preview bench, where the "
                         "hands looked better placed - but the bench showed no "
                         "tracking readout, and 0.08 at the authored upright "
                         "torso puts arm_j2 on its limit for 91-99%% of frames: "
                         "median wrist residual 272/192 mm vs 33/36 mm at 0.20, "
                         "and 0 mm on the bending clip. It costs 9 mm on the "
                         "score against the operator's own thirteen authored "
                         "poses (243 -> 252 mm). Back to 0.20 on that evidence; "
                         "the bench now prints the residual so the visual "
                         "choice can be made knowing what it costs.")
parser.add_argument("--waist-yaw-fix-deg", type=float, default=0.0,
                    help="[--map waist-abs] yaw correction [deg] applied to "
                         "the body-relative wrist vectors - compensates the "
                         "waist tracker's mounting yaw. 2026-07-29 session "
                         "fitted +4.3 deg from mirror-pose symmetry; per-"
                         "session auto-calibration (hand-line) is the "
                         "long-term default")
parser.add_argument("--duration", type=float, default=0.0)
parser.add_argument("--head-pitch-path", choices=["j1", "authored"],
                    default="j1",
                    help="how up/down is split between the two coaxial pitch "
                         "joints. 'j1' puts it all on head_j1 and locks j3 at "
                         "0: torso-independent, because head_gaze() takes the "
                         "chest rotation as an input and j1 has +-85 deg. "
                         "'authored' follows the (j1,j3) pair posed in the "
                         "joint bench - it is torso-SPECIFIC, and at the legacy "
                         "55 deg home its j1 (never negative) cannot even reach "
                         "level, so looking up fails 1:1 with how far you look "
                         "up: measured -5/-15/-30/-45 deg of shortfall at +0/"
                         "+10/+25/+40. Reported live as 'looking up does not "
                         "work at all'.")
parser.add_argument("--control-addr", default="tcp://127.0.0.1:5584",
                    help="switch channel published by the mirror viewer. The "
                         "smoothing strength and the control mode can be moved "
                         "from there while wearing the headset; losing the "
                         "channel just leaves the last values in place - it is "
                         "the HARDWARE bridge that treats silence as off.")
parser.add_argument("--joint-smooth", type=float, default=6.0,
                    help="anti-jitter on EVERY commanded joint, as a one-euro "
                         "min-cutoff in Hz. LOWER = smoother and laggier; 0 "
                         "disables. This sits on the final joint targets, after "
                         "the IK, so it catches jitter from any source - the "
                         "wrist channel's own noise, the IK, the head solve - "
                         "which the existing one-euro on the EE target cannot, "
                         "because that one runs before the solver. Adjustable "
                         "live from the mirror viewer.")
parser.add_argument("--waist-mode", choices=["j3", "authored"], default="j3",
                    help="how --control-mode ...+waist bends the torso. 'j3' "
                         "drives torso_j3 alone and holds j1/j2 at the scene's "
                         "home: no authored table, so it is valid at ANY torso "
                         "home, and it is the cheapest of the three joints in "
                         "shoulder disturbance (7.8 mm per deg of pitch, vs "
                         "10.3 for j2 and 17.1 for j1, measured at the legacy "
                         "home). 'authored' walks the (j1,j2,j3) path the "
                         "operator posed in the joint bench - only valid at the "
                         "10.3 deg torso home it was fitted at.")
parser.add_argument("--physics-device", default=None,
                    help="'cpu' or 'cuda'. Overrides AppLauncher's --device, "
                         "which is unreachable here: argparse resolves "
                         "'--device' as an abbreviation of this file's older "
                         "--device-stale-ms and rejects 'cpu' as a float. "
                         "Worth having anyway - one small articulation pays "
                         "GPU physics' fixed per-step cost with nothing to "
                         "amortise it over, and physics is 50 ms of the 80 ms "
                         "control step measured 2026-08-02 (rtf 0.20x).")
parser.add_argument("--forearm-bridge", type=float, default=1.5,
                    metavar="SECONDS",
                    help="[--mode trackers] bridge wrist POSITION dropouts by "
                         "sweeping the forearm arc from the still-live wrist "
                         "ORIENTATION, giving up after this many seconds (0 "
                         "disables, leaving the position frozen as before). "
                         "The PICO trackers are tracked by the headset "
                         "cameras, so position dies whenever the wrist leaves "
                         "their view while the IMU keeps reporting rotation; "
                         "measured on the 2026-07-29 takes this cuts the p90 "
                         "error of a 1 s dropout from 222mm to 118mm")
parser.add_argument("--trace-csv", default=None,
                    help="dump the whole chain per control step to CSV: human "
                         "wrist-rel-waist, commanded EE target, actual EE "
                         "(base frame), per hand. This is the raw material for "
                         "judging mapping quality the way the operator does - "
                         "over the WHOLE run, not at hold endpoints")
parser.add_argument("--joint-csv", default=None,
                    help="dump the COMMANDED and ACHIEVED joint angles [rad] "
                         "per control step. Pair with the same flag on "
                         "scripts/viser_mapping_preview.py and diff them with "
                         "sim/dev_compare_isaac_viser.py: the bench is where "
                         "mapping choices get made visually, so it has to be "
                         "showing the motion this consumer will send. Joints, "
                         "not pictures - joints are what reaches the robot.")
# The joints that carry the motion: 14 arm + 3 head + 3 torso. Wheels are
# parked and the hands are not in this USD, so nothing else moves.
JOINT_DUMP = ([f"{s_}_arm_j{i}" for s_ in ("R", "L") for i in range(1, 8)]
              + [f"head_j{i}" for i in (1, 2, 3)]
              + [f"torso_j{i}" for i in (1, 2, 3)])
# self-inspection screenshots (works headless; forces --enable_cameras)
parser.add_argument("--snap-dir", default=None)
parser.add_argument("--snap-times", default="0.5,2,4,6,8,11")
parser.add_argument("--cam-eye", default="2.4,-2.0,2.0")
parser.add_argument("--cam-target", default="0.0,0.0,1.1")
parser.add_argument("--debug-joints", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.snap_dir is not None:
    args_cli.enable_cameras = True

# --wrist-map 展开成它底下的两个开关。**成对设置,不允许拆开** —— 增量朝向
# 必须配默认姿势锚点(从量程正中央出发才有余量可用),配上漂移锚点实测是
# 96.98% 的帧顶在限位上,比什么都不做糟得多。一个选择顶两个开关,就是为了
# 让那种组合根本没机会出现。
if args_cli.wrist_map:
    _pair = {"absolute": ("palm", "current"),
             "incremental": ("track", "home")}[args_cli.wrist_map]
    args_cli.ori_mode, args_cli.anchor = _pair
    print(f"[wrist] 腕部朝向映射 = {args_cli.wrist_map}"
          f"(--ori-mode {_pair[0]} --anchor {_pair[1]})"
          + ("  掌心与你一致,但可能超腕关节量程"
             if args_cli.wrist_map == "absolute"
             else "  从量程中央出发只走增量,掌心与你差一个固定角"))

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))

from scipy.spatial.transform import Rotation as _Rot  # noqa: E402

from magicdexmate.pico.arm_fusion import ForearmFuser  # noqa: E402

# --ori-mode palm: absolute wrist orientation, no anchor. R_ee = R_wrist @
# PALM_FIX[hand]. The constant, how it was fitted, and - critically - which
# tracker channel it is valid on all live in magicdexmate/palm_fix.py. It is
# valid for the MODEL channel (body24 joints 20/21) only; scripts/trackers.env
# picks the channel.
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.bimanual_symmetry import (  # noqa: E402
    mirror_distance, symmetrise)
from magicdexmate.swivel import swivel_angle as _swivel_angle  # noqa: E402

# 操作者的肩/肘/腕在 body24 里的下标(SMPL 顺序)
_SMPL_SHOULDER = {"left": 16, "right": 17}
_SMPL_ELBOW = {"left": 18, "right": 19}


def _operator_swivel(hand, frame, trk_use, wrist_key, waist_pose):
    """操作者的肘绕自己肩-腕轴的角度,弧度;取不到时返回 None。

    7 自由度手臂对 6 自由度腕目标多出一维,而腕目标不约束它,解算器只能沿用
    历史落到的那一支。这个角度就是把那一维钉住用的:它无量纲,人和机器人比例
    不同也成立,而且放到机器人自己的圆上一定可达,所以不会和腕任务打架。

    只搬角度,不搬肘的位置。搬位置试过五次都失败:人机肩偏置与臂长不同,那个
    点常在机器人肘够不到的地方,解算器只能在腕和肘之间二选一,实测腕残差从
    30mm 崩到 475mm。

    左右两臂互为镜像,同一姿势解出的角度符号相反,这是这套定义本身的性质
    (镜像下 u 同向、v 反向),两侧各自送各自的值即可,不要取反。
    """
    if waist_pose is None or frame.body24 is None:
        return None
    w = trk_use.get(wrist_key)
    if w is None:
        return None
    ref = np.asarray(waist_pose, dtype=np.float64)
    try:
        p_w = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(w, dtype=np.float64), ref))[0]
        p_s = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(frame.body24[_SMPL_SHOULDER[hand]], dtype=np.float64),
            ref))[0]
        p_e = mat_to_pos_quat_wxyz(process_xr_pose(
            np.asarray(frame.body24[_SMPL_ELBOW[hand]], dtype=np.float64),
            ref))[0]
    except (IndexError, TypeError, ValueError):
        return None
    return _swivel_angle(p_s, p_e, p_w)
# Robot shoulder offsets from the chest and the arm reach, measured on the
# URDF: shoulder centre -> elbow 308.2 mm, elbow -> wrist centre 307.6 mm.
ROB_SH = {"right": np.array([-0.070, -0.231, 0.0]),
          "left": np.array([-0.070, +0.231, 0.0])}
ROB_ARM = 0.6158    # 308.2 + 307.6 mm, shoulder centre -> wrist centre
# Head / waist mapping, fitted to poses the operator authored in the joint
# bench (logs/robot_head_waist_poses.json). Same rule as palm_fix: one home for
# the constants, no copies.
from magicdexmate.head_waist_map import (  # noqa: E402
    HEAD_PITCH_TABLE, NEUTRAL_ELEVATION, YAW_LIMIT, head_gaze, head_solve,
    waist_joints, waist_joints_j3)
from magicdexmate.pico.xr_pose import (  # noqa: E402
    OneEuroFilter,
    QuatOneEuroFilter,
    R_HEADSET_TO_WORLD,
    mat_to_pos_quat_wxyz,
    pitch_delta,
    pose7_to_world_rot,
    process_xr_pose,
)
from magicdexmate.sources.pico_source import (  # noqa: E402
    DeviceStallDetector,
    GripLatch,
    MockPicoSource,
    PicoLogSource,
    PicoZmqSource,
    TrackerFreezeDetector,
    TrackerGlitchGate,
)

# Build Pink IK BEFORE AppLauncher: isaacsim clobbers pinocchio's eigenpy
# converters, so pink.Configuration can't be constructed once Kit is up. Built
# first, the runtime solve works fine (see pink_vega_ik.py).
pink_ik = None
joint_guard = None      # defined HERE, not only inside the pink try-block:
                        # --ik dls would otherwise leave it undefined and the
                        # exit layer would NameError - the "clean exit 0" class.
if args_cli.ik == "pink":
    try:
        from pink_vega_ik import PinkVegaIK  # noqa: E402

        if args_cli.joint_vel_scale > 0.0 or args_cli.self_collision != "off":
            from magicdexmate.joint_guard import JointGuard
            joint_guard = JointGuard(
                control_hz=args_cli.control_hz,
                vel_scale=args_cli.joint_vel_scale,
                margin_m=args_cli.collision_margin,
                collision=(args_cli.self_collision != "off"))
            print(f"[guard] built before Kit load: vel_scale="
                  f"{args_cli.joint_vel_scale} collision={args_cli.self_collision}"
                  f" pairs={len(joint_guard.pairs)}")
            # 环境障碍物的尺寸是场地相关的,填错了最可能的表现是「判据恒为真、
            # 手臂从第一帧就被冻住」,而且不会说为什么。默认姿势下就相撞即报错。
            from magicdexmate.home_pose import ARM_HOME as _HOME
            joint_guard.env_self_check(_HOME)
        pink_ik = PinkVegaIK(dt=1.0 / args_cli.control_hz,
                             orientation_cost=args_cli.pink_ori_cost,
                             substeps=args_cli.pink_substeps,
                             elbow_cost=args_cli.elbow_weight,
                             null_bias=args_cli.null_bias,
                             gain=args_cli.pink_gain,
                             relax_at_limit=args_cli.relax_at_limit,
                             relax_margin_deg=args_cli.relax_margin,
                             relax_pos_at_limit=args_cli.relax_pos_at_limit,
                             smooth_cost=args_cli.pink_smooth_cost)
        print(f"[pink] IK built ({len(pink_ik.pin_names)} arm joints) before Kit load")
    except Exception as e:  # pink/pinocchio missing or URDF unreadable
        joint_guard = None      # a half-built guard must not survive the fallback
        print(f"[pink] build failed ({e!r}); falling back to --ik dls")
        args_cli.ik = "dls"
elif args_cli.ik == "curobo":
    # cuRobo 后端与 Pink 鸭子类型兼容(接口契约 = 本文件 pink_ik. 的全部
    # 用点),沿用同一个变量名。同样必须在 AppLauncher 之前构建:pinocchio
    # 账本怕 isaacsim 覆写 eigenpy;求解子进程也要赶在 Kit 抢占 GPU 之前把
    # CUDA 图捕获做完(见 curobo_vega_ik.py 的 docstring)。
    try:
        from curobo_vega_ik import CuroboVegaIK  # noqa: E402

        if args_cli.joint_vel_scale > 0.0 or args_cli.self_collision != "off":
            from magicdexmate.joint_guard import JointGuard
            joint_guard = JointGuard(
                control_hz=args_cli.control_hz,
                vel_scale=args_cli.joint_vel_scale,
                margin_m=args_cli.collision_margin,
                collision=(args_cli.self_collision != "off"))
            print(f"[guard] built before Kit load: vel_scale="
                  f"{args_cli.joint_vel_scale} collision={args_cli.self_collision}"
                  f" pairs={len(joint_guard.pairs)}")
            from magicdexmate.home_pose import ARM_HOME as _HOME
            joint_guard.env_self_check(_HOME)
        pink_ik = CuroboVegaIK(dt=1.0 / args_cli.control_hz,
                               local_iters=args_cli.curobo_iters)
        print(f"[curobo] IK built ({len(pink_ik.pin_names)} arm joints, "
              f"iters={args_cli.curobo_iters}) before Kit load")
    except Exception as e:  # curobo/GPU 不可用或求解子进程没起来
        joint_guard = None      # a half-built guard must not survive the fallback
        print(f"[curobo] build failed ({e!r}); falling back to --ik dls")
        args_cli.ik = "dls"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from vega_scene import ARM_JOINTS, EE_BODY, VegaSceneCfg, make_snap_camera_cfg  # noqa: E402


def _parse_vec3(s: str):
    x, y, z = (float(v) for v in s.split(","))
    return (x, y, z)


class ArmChannel:
    """One controller -> one arm: clutch state + IK bookkeeping."""

    def __init__(self, hand: str, scene, robot, device, jacobi_row: str,
                 null_k: float = 0.0, null_cap: float = float("inf")):
        self.hand = hand
        cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS[hand], body_names=[EE_BODY[hand]])
        cfg.resolve(scene)
        self.joint_ids = cfg.joint_ids
        self.joint_ids_t = torch.as_tensor(cfg.joint_ids, device=device)
        self.body_idx = cfg.body_ids[0]
        self.jacobi_idx = self.body_idx - 1 if jacobi_row == "fixedbase" else self.body_idx
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=device,
        )
        self.robot = robot
        self.null_k = null_k
        self.null_cap = null_cap
        self.q_home = robot.data.default_joint_pos[0, self.joint_ids_t].clone()
        self.engaged = False
        self.has_command = False
        self.latch = GripLatch()                    # hysteresis on the analog grip
        self.ref0: np.ndarray | None = None         # reference pose frozen at engage
                                                    # (headset in controllers mode,
                                                    #  waist tracker in trackers mode)
        self.pos_filter: OneEuroFilter | None = None
        self.quat_filter: QuatOneEuroFilter | None = None
        self.human_p: np.ndarray | None = None      # wrist rel waist (--trace-csv)
        self.pico_p0: np.ndarray | None = None      # controller ref (yaw-comp frame)
        self.pico_q0_inv: torch.Tensor | None = None
        self.ee_p0: torch.Tensor | None = None      # EE ref (base frame)
        self.ee_q0: torch.Tensor | None = None
        self.home_ee: tuple | None = None           # settled home EE pose
                                                    # (--anchor home target)
        self.pico_el0: np.ndarray | None = None     # elbow anchor (mapped frame)
        self.el_p0: np.ndarray | None = None        # robot elbow anchor (base)
        self.el_tgt: np.ndarray | None = None       # live elbow target (base)
        self.cmd_pos: torch.Tensor | None = None    # last commanded EE position
        self.cmd_quat: torch.Tensor | None = None
        self.solved_pos: torch.Tensor | None = None  # target at the last IK solve
        self.solved_quat: torch.Tensor | None = None
        self.hold_db_pos = 1e-3                      # 1 mm / ~0.3 deg deadband:
        self.hold_db_ang = 5e-3                      # below this, don't re-solve
        self._dbg_n = 0

    def ee_pose_in_base(self):
        ee_pose_w = self.robot.data.body_state_w[:, self.body_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

    def set_pose_command(self, pos: torch.Tensor, quat: torch.Tensor):
        self.ik.set_command(torch.cat([pos, quat], dim=-1))
        self.has_command = True
        self.cmd_pos = pos
        self.cmd_quat = quat

    def compute(self, targets: torch.Tensor):
        if not self.has_command:
            return
        # Deadband: a DLS solve has no null-space term, so re-solving a target
        # that hasn't materially moved lets the redundant DOFs drift/wind
        # (seen as L_arm_j1 wrapping its full +-pi range while holding still).
        # If the target barely changed since the last solve, keep the last
        # joints - the robot is already there.
        if self.solved_pos is not None:
            dp = (self.cmd_pos - self.solved_pos).norm().item()
            dq = axis_angle_from_quat(
                quat_mul(self.cmd_quat, quat_inv(self.solved_quat))).norm().item()
            if dp < self.hold_db_pos and dq < self.hold_db_ang:
                return
        self.solved_pos = self.cmd_pos.clone()
        self.solved_quat = self.cmd_quat.clone()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, self.joint_ids]
        ee_pos_b, ee_quat_b = self.ee_pose_in_base()
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        q_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        if self.null_k > 0.0:
            # DLS has no null-space term -> redundant DOFs drift/wind (mock run
            # saw j3 at its -176 deg limit). Project a home-pose pull through
            # I - J+J so the EE task is untouched.
            J = jacobian[0]
            N = torch.eye(J.shape[1], device=J.device, dtype=J.dtype) \
                - torch.linalg.pinv(J) @ J
            dq_null = self.null_k * (N @ (self.q_home - joint_pos[0]))
            step = dq_null.abs().max()
            if step > self.null_cap:
                dq_null = dq_null * (self.null_cap / step)
            q_des = q_des + dq_null.unsqueeze(0)
        if os.environ.get("VEGA_IK_DEBUG") and self._dbg_n < 16:
            self._dbg_n += 1
            ang = axis_angle_from_quat(
                quat_mul(self.cmd_quat, quat_inv(ee_quat_b))).norm().item() * 57.3
            print(f"[ikdbg] {self.hand} #{self._dbg_n:02d} "
                  f"pos_err {(ee_pos_b - self.cmd_pos).norm().item() * 1000:7.2f}mm "
                  f"ang_err {ang:7.2f}deg "
                  f"dq_max {(q_des - joint_pos).abs().max().item():8.5f}rad")
        targets[0, self.joint_ids_t] = q_des[0]


class TorsoChannel:
    """Waist/chest Swift tracker -> torso_j1 lean, clutched with the arms.

    Engages when any arm is engaged: captures the tracker's ABSOLUTE world
    orientation as zero, then maps its lean delta (world-y pitch, forward
    positive) onto torso_j1 around its default position. Using the absolute
    world pitch (not a reference-relative frame) is correct here: the waist
    IS the body, so its tilt vs gravity is the lean. Disengage = freeze; the
    next engage re-zeroes, same ratchet as the arms.
    """

    def __init__(self, robot, serial_sel: str, scale: float, max_rad: float,
                 sign: int):
        names = list(robot.joint_names)
        self.joint_id = names.index("torso_j1")
        self.q0 = float(robot.data.default_joint_pos[0, self.joint_id].item())
        limits = getattr(robot.data, "joint_pos_limits",
                         getattr(robot.data, "joint_limits", None))
        margin = 0.05
        self.lo = float(limits[0, self.joint_id, 0].item()) + margin
        self.hi = float(limits[0, self.joint_id, 1].item()) - margin
        self.serial_sel = serial_sel
        self.scale = scale
        self.max_rad = max_rad
        self.sign = float(sign)
        self.engaged = False
        self.rot0: np.ndarray | None = None
        self.filter = OneEuroFilter(min_cutoff=1.0, beta=1.0)  # beta in rad units
        self.serial_seen: str | None = None

    def select(self, trackers: dict) -> np.ndarray | None:
        if not trackers:
            return None
        if self.serial_sel == "auto":
            sn = sorted(trackers)[0]
        elif self.serial_sel in trackers:
            sn = self.serial_sel
        else:
            return None
        self.serial_seen = sn
        return trackers[sn]

    def engage(self, tracker_pose7: np.ndarray):
        self.rot0 = pose7_to_world_rot(tracker_pose7)
        self.filter.reset()
        self.engaged = True

    def update(self, tracker_pose7: np.ndarray, targets: torch.Tensor, dt: float):
        rot_now = pose7_to_world_rot(tracker_pose7)
        lean = pitch_delta(rot_now, self.rot0) * self.scale
        lean = float(np.clip(lean, -self.max_rad, self.max_rad))
        lean = float(self.filter(np.array([lean]), dt)[0])
        q = float(np.clip(self.q0 + self.sign * lean, self.lo, self.hi))
        targets[0, self.joint_id] = q


class WaistChannel(TorsoChannel):
    """--control-mode arms+head+waist: lean -> torso_j1..j3 along the path the
    operator authored, instead of TorsoChannel's single-joint torso_j1 delta.

    Vega's three torso joints all turn about the same axis - there is no waist
    yaw and no side bend - so the torso is a one-parameter sagittal column and
    "how much lean" indexes the whole path. The operator authored its ends
    (upright / working / max) in the joint bench; magicdexmate.head_waist_map
    owns the table.

    Still a DELTA from the engage-time zero, like TorsoChannel: the waist
    tracker's mounting angle on the belt is arbitrary and changes per wearing,
    so its absolute pitch means nothing. Zeroing at engage cancels it.
    """

    RAMP_S = 1.5        # same ramp the arms use, for the same reason

    def __init__(self, robot, serial_sel, scale, max_rad, sign, mode="j3",
                 fwd_rad=None, back_rad=None):
        super().__init__(robot, serial_sel, scale, max_rad, sign)
        self.mode = mode
        self.fwd_rad = max_rad if fwd_rad is None else fwd_rad
        self.back_rad = max_rad if back_rad is None else back_rad
        names = list(robot.joint_names)
        self.robot = robot
        # the pose j1/j2 hold at in j3-only mode: whatever the scene was built
        # with, so this works at any torso home without an authored table
        self.home = {n: float(robot.data.default_joint_pos[0, names.index(n)])
                     for n in ("torso_j1", "torso_j2", "torso_j3")}
        self.ids = {n: names.index(n) for n in ("torso_j1", "torso_j2", "torso_j3")}
        self.q_start: dict[str, float] = {}
        self.t_since = 0.0

    def engage(self, tracker_pose7):
        # Zero ONCE per run, not on every re-engage. The reason the zero exists
        # at all - stated two docstrings up - is that the belt's mounting angle
        # is arbitrary; that angle changes per WEARING, not per clutch cycle.
        # Re-zeroing on each re-engage made the lean a ratchet: on the
        # 2026-07-31 head/waist clip the arm clutch re-engaged five times and
        # every one of them redefined however-far-you-were-bent as upright, so
        # the torso kept sliding back to the start and waist following never
        # actually ran. The ramp below still re-arms each time.
        first = self.rot0 is None
        if first:
            super().engage(tracker_pose7)
            print("[waist] lean zero captured (held for the whole run - "
                  "re-engaging the arm clutch no longer re-zeroes it)")
        else:
            self.filter.reset()
            self.engaged = True
        # Ramp in from wherever the torso actually IS. The authored "upright"
        # is not the scene's start pose - they differ by up to 15.5 deg on
        # torso_j3 - so commanding the mapped value directly steps the whole
        # upper body on the frame the clutch closes. The arms already ramp for
        # exactly this reason; the torso carries the arms, so it matters more.
        self.q_start = {n: float(self.robot.data.joint_pos[0, i].item())
                        for n, i in self.ids.items()}
        self.t_since = 0.0

    def update(self, tracker_pose7, targets, dt: float):
        lean_raw = pitch_delta(pose7_to_world_rot(tracker_pose7),
                               self.rot0) * self.scale
        # ASYMMETRIC: bending forward and leaning back are not the same move.
        # A person bends ~80 deg forward and barely 15 back; the robot is
        # allowed 30 forward and 10 back from its home. One symmetric clamp
        # cannot express that, and the one that was here (25 both ways) cut the
        # operator's 71.8 deg of bend to a third while the summary reported its
        # own clamp value back as healthy 1:1 tracking.
        lean = float(np.clip(lean_raw, -self.back_rad, self.fwd_rad))
        self.sat_n = getattr(self, "sat_n", 0) + (
            lean_raw > self.fwd_rad or lean_raw < -self.back_rad)
        self.tot_n = getattr(self, "tot_n", 0) + 1
        self.raw_lo = min(getattr(self, "raw_lo", lean_raw), lean_raw)
        self.raw_hi = max(getattr(self, "raw_hi", lean_raw), lean_raw)
        lean = float(self.filter(np.array([lean]), dt)[0])
        self.t_since += dt
        a = min(1.0, self.t_since / self.RAMP_S)
        # self.sign is NOT applied here. It exists for TorsoChannel, whose
        # single-joint torso_j1 delta needed -1 because of that joint's axis
        # direction; this channel commands a chest LEAN, which is unambiguous -
        # bending forward increases it. Applying the legacy -1 drove the chest
        # toward upright, and the authored table starts at upright, so the
        # whole range clamped: 57 deg of operator bend moved the chest 0.0 deg.
        # "did the operator's bend actually reach the command" - separate from
        # "did the torso move", which the [summary] torso travel line answers.
        # Both are needed: the lean can be dead while the torso moves (ramp) and
        # the torso can be dead while the lean is live (clamped table).
        d = np.degrees(lean)
        self.lean_lo = min(getattr(self, "lean_lo", d), d)
        self.lean_hi = max(getattr(self, "lean_hi", d), d)
        # ...and the CHEST PITCH that comes out, by Dexmate's own definition
        # (dexcontrol/core/torso.py): pitch = j1 + j3 + pi/2 - j2, decreasing
        # as the torso folds forward. Reporting the operator's lean alone is
        # not enough: the previous waist version had the sign wrong, so 57 deg
        # of operator bend moved the chest 0.0 deg, and nothing in the output
        # said so. This makes "did it bend, and which way" a reading.
        _wj = (waist_joints_j3(d, self.home) if self.mode == "j3"
               else waist_joints(d))
        _p = np.degrees(_wj["torso_j1"] + _wj["torso_j3"] - _wj["torso_j2"]) + 90.0
        self.pitch_lo = min(getattr(self, "pitch_lo", _p), _p)
        self.pitch_hi = max(getattr(self, "pitch_hi", _p), _p)
        wj = (waist_joints_j3(np.degrees(lean), self.home)
              if self.mode == "j3" else waist_joints(np.degrees(lean)))
        for n, q in wj.items():
            targets[0, self.ids[n]] = (1.0 - a) * self.q_start.get(n, q) + a * q


class HeadChannel:
    """--control-mode arms+head: headset gaze -> head_j1..j3.

    Input is the headset orientation with the waist tracker's yaw removed -
    the same `process_xr_pose` frame the arms map in - so turning the whole
    body does not sweep the robot's head, only turning the head relative to the
    body does.

    Solved in the BASE frame against the LIVE chest rotation, which is what
    makes the waist mode free: lean the torso and the head compensates to hold
    the commanded gaze. Verified offline - chest lean 43.6 -> 73.5 deg while
    the achieved gaze stayed at -15.9 deg to within 0.01.

    Not clutched with the arms. Freezing the operator's view when they let go
    of the arm clutch is disorienting, and the head commands nothing that can
    collide or wind up.
    """

    def __init__(self, robot, pitch_path="j1"):
        self.pitch_path = pitch_path
        names = list(robot.joint_names)
        self.ids = {n: names.index(n) for n in ("head_j1", "head_j2", "head_j3")}
        self.seed: tuple | None = None
        self.filter = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        self.gaze_stat = {"n": 0, "az": 0.0, "el": 0.0, "az_hi": 0.0,
                          "el_hi": 0.0, "yaw_clamp": 0, "pitch_clamp": 0}

    def update(self, head_pose7, waist_pose7, R_chest, targets, dt: float):
        T = process_xr_pose(np.asarray(head_pose7, dtype=np.float64),
                            np.asarray(waist_pose7, dtype=np.float64))
        g = T[:3, 0]                       # gaze; raw -z fwd -> world +x
        az = float(np.degrees(np.arctan2(g[1], g[0])))
        el = float(np.degrees(np.arcsin(np.clip(g[2], -1.0, 1.0))))
        az, el = self.filter(np.array([az, el]), dt)
        q, self.seed = head_solve(float(az), float(el), R_chest, seed=self.seed,
                                  pitch_path=self.pitch_path)
        # Does the head actually LOOK where it was asked to? Never measured
        # until 2026-08-03: everything so far only checked that the joints
        # moved. The solve is done in the BASE frame with R_chest as an input,
        # so it should be torso-independent by construction - but the authored
        # pitch table and the +-50 deg yaw envelope were fitted at one torso
        # home, and a clamp is silent. Report the achieved gaze against the
        # requested one, and how often the envelope is what stopped it.
        R = head_gaze(q["head_j1"], q["head_j2"], q["head_j3"], R_chest)
        got_az = float(np.degrees(np.arctan2(R[1], R[0])))
        got_el = float(np.degrees(np.arcsin(np.clip(R[2], -1.0, 1.0))))
        st = self.gaze_stat
        st["n"] += 1
        st["az"] += abs((got_az - az + 180.0) % 360.0 - 180.0)
        st["el"] += abs(got_el - (el + NEUTRAL_ELEVATION))
        st["az_hi"] = max(st["az_hi"], abs((got_az - az + 180.0) % 360.0 - 180.0))
        st["el_hi"] = max(st["el_hi"], abs(got_el - (el + NEUTRAL_ELEVATION)))
        # "was it clamped" measured from the OUTCOME, not from a table's
        # bounds: the bounds are path-specific, and after the pitch path moved
        # to j1 the old test kept reporting 86% of frames clamped while the
        # elevation error was 0.00 deg. A readout that cannot be wrong about
        # the thing it names is worth more than one that names a mechanism.
        if abs((got_az - az + 180.0) % 360.0 - 180.0) > 1.0:
            st["yaw_clamp"] += 1
        if abs(got_el - (el + NEUTRAL_ELEVATION)) > 1.0:
            st["pitch_clamp"] += 1
        for n, v in q.items():
            targets[0, self.ids[n]] = v


def main():
    device_arg = args_cli.physics_device or args_cli.device
    sim_cfg = SimulationCfg(
        dt=1 / args_cli.physics_hz,
        render_interval=args_cli.render_interval,
        device=device_arg,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4, bounce_threshold_velocity=0.2),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.4, -1.8, 1.9), target=(0.0, 0.0, 1.1))

    scene_cfg = VegaSceneCfg(num_envs=1, env_spacing=4.0)
    snap_times = []
    if args_cli.snap_dir is not None:
        scene_cfg.camera = make_snap_camera_cfg()
        snap_times = sorted(float(v) for v in args_cli.snap_times.split(",") if v.strip())
        os.makedirs(args_cli.snap_dir, exist_ok=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    snap_cam = None
    if args_cli.snap_dir is not None:
        snap_cam = scene["camera"]
        eye = torch.tensor([_parse_vec3(args_cli.cam_eye)], dtype=torch.float32)
        target = torch.tensor([_parse_vec3(args_cli.cam_target)], dtype=torch.float32)
        snap_cam.set_world_poses_from_view(eye.to(snap_cam.device), target.to(snap_cam.device))

    def save_snap(sim_t: float):
        from PIL import Image

        rgb = snap_cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        path = os.path.join(args_cli.snap_dir, f"snap_t{sim_t:05.1f}s.png")
        Image.fromarray(rgb[..., :3]).save(path)
        print(f"\n[snap] saved {path}")

    robot: Articulation = scene["robot"]
    device = robot.device
    isaac_names = list(robot.joint_names)
    state_pub = None
    if args_cli.pub_state:
        try:
            import msgpack as _mp
            import zmq as _zmq
            state_pub = _zmq.Context.instance().socket(_zmq.PUB)
            state_pub.setsockopt(_zmq.SNDHWM, 2)
            state_pub.bind(args_cli.pub_state)
            print(f"[state] publishing Isaac joint positions on {args_cli.pub_state}")
        except Exception as e:
            print(f"[state] publisher off ({e})")
            state_pub = None
    print(f"[env] articulation with {len(isaac_names)} joints")
    if os.environ.get("VEGA_DUMP_JOINTS"):
        print("[wheels]", [n for n in isaac_names if "wheel" in n.lower()])
        for an, act in robot.actuators.items():
            js = [isaac_names[i] for i in act.joint_indices]
            print(f"[act] {an}: k={float(act.stiffness.flatten()[0]):.1f} "
                  f"d={float(act.damping.flatten()[0]):.1f} joints={js}")
        body_names = list(robot.body_names)
        masses = robot.data.default_mass[0]
        for i, bn in enumerate(body_names):
            if "wheel" in bn.lower():
                print(f"[mass] {bn}: {float(masses[i]):.5f} kg")
        # per-DOF physics params (joint friction was the prime suspect for the
        # move-crawl / hold-perfect symptom, 2026-07-27)
        pv = robot.root_physx_view
        try:
            fric = pv.get_dof_friction_coefficients()[0]
            damp = pv.get_dof_dampings()[0]
            stif = pv.get_dof_stiffnesses()[0]
            arma = pv.get_dof_armatures()[0]
            maxf = pv.get_dof_max_forces()[0]
            maxv = pv.get_dof_max_velocities()[0]
            for i, n in enumerate(isaac_names):
                if "R_arm_j" in n or n == "torso_j1":
                    print(f"[dof] {n}: kp={float(stif[i]):.0f} "
                          f"kd={float(damp[i]):.0f} fric={float(fric[i]):.3f} "
                          f"arma={float(arma[i]):.4f} maxF={float(maxf[i]):.0f} "
                          f"maxV={float(maxv[i]):.3f}")
        except Exception as e:
            print(f"[dof] dump failed: {e!r}")

    hands = ["right", "left"] if args_cli.hands == "both" else [args_cli.hands]
    arms = {h: ArmChannel(h, scene, robot, device, args_cli.jacobi_row,
                          null_k=args_cli.null_k, null_cap=args_cli.null_cap)
            for h in hands}
    if args_cli.filter_mincutoff > 0.0:
        for arm in arms.values():
            arm.pos_filter = OneEuroFilter(args_cli.filter_mincutoff,
                                           args_cli.filter_beta)
            arm.quat_filter = QuatOneEuroFilter(args_cli.filter_mincutoff,
                                                args_cli.filter_beta)

    # torso source: in trackers mode the waist tracker drives it by default;
    # in controllers mode it's the optional --torso-tracker.
    if args_cli.mode == "trackers":
        torso_sel = None if args_cli.no_torso else args_cli.tracker_waist
    else:
        torso_sel = None if args_cli.torso_tracker == "off" else args_cli.torso_tracker
    mode_head = args_cli.control_mode in ("arms+head", "arms+head+waist")
    mode_waist = args_cli.control_mode == "arms+head+waist"
    torso = None
    if torso_sel is not None:
        cls = WaistChannel if mode_waist else TorsoChannel
        torso = (cls(robot, torso_sel, args_cli.torso_scale,
                     np.deg2rad(args_cli.torso_max_deg), args_cli.torso_sign,
                     mode=args_cli.waist_mode,
                     fwd_rad=np.deg2rad(args_cli.torso_fwd_deg),
                     back_rad=np.deg2rad(args_cli.torso_back_deg))
                 if mode_waist else
                 cls(robot, torso_sel, args_cli.torso_scale,
                     np.deg2rad(args_cli.torso_max_deg), args_cli.torso_sign))
        print(f"[torso] tracker='{torso_sel}' -> "
              + (("torso_j3 alone, j1/j2 held at "
                  f"{np.degrees(torso.home['torso_j1']):.1f}/"
                  f"{np.degrees(torso.home['torso_j2']):.1f}deg"
                  if args_cli.waist_mode == "j3"
                  else "torso_j1..j3 along the authored path")
                 if mode_waist else "torso_j1")
              + f" (default {torso.q0:+.3f}, limits [{torso.lo:+.2f}, {torso.hi:+.2f}])")
    head_ch = HeadChannel(robot, args_cli.head_pitch_path) if mode_head else None
    if head_ch is not None:
        print(f"[head] pitch path = {args_cli.head_pitch_path}"
              + ("  (j1 carries pitch, j3 locked at 0 - torso-independent)"
                 if args_cli.head_pitch_path == "j1" else
                 "  (the authored (j1,j3) pair - only valid at the 10.3deg "
                 "torso home it was fitted at)"))
    print(f"[mode] control-mode = {args_cli.control_mode}"
          + ("" if mode_head else "  (head held)")
          + ("" if mode_waist else "  (torso not on the authored path)"))

    # Pink IK (built pre-Kit above): resolve EE + chest body indices. Targets
    # are fed relative to arm_center (chest) so torso lean doesn't disturb the
    # arm solve, so we need arm_center's live pose each step.
    if pink_ik is not None:
        from pink_vega_ik import CHEST_FRAME, ELBOW_FRAME  # noqa: E402
        ee_body_idx = {}
        elbow_body_idx = {}
        if args_cli.elbow_weight > 0.0:
            for h, bname in ELBOW_FRAME.items():
                c = SceneEntityCfg("robot", body_names=[bname])
                c.resolve(scene)
                elbow_body_idx[h] = c.body_ids[0]
        for h, bname in EE_BODY.items():
            c = SceneEntityCfg("robot", body_names=[bname])
            c.resolve(scene)
            ee_body_idx[h] = c.body_ids[0]
        _chest_cfg = SceneEntityCfg("robot", body_names=[CHEST_FRAME])
        _chest_cfg.resolve(scene)
        chest_body_idx = _chest_cfg.body_ids[0]

    def ee_pose_in_base_of(body_idx):
        ee = robot.data.body_state_w[:, body_idx, 0:7]
        rt = robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(rt[:, 0:3], rt[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    # per-run diagnostics: joint excursions vs limits + tracking error
    _wheel_diag = [i for i, n in enumerate(isaac_names) if "_wheel_j" in n]
    all_arm_ids = sorted({j for a in arms.values() for j in a.joint_ids}
                         | ({torso.joint_id} if torso is not None else set())
                         | set(_wheel_diag))
    all_arm_ids_t = torch.as_tensor(all_arm_ids, device=device)
    q_seen = {"lo": None, "hi": None}
    q_all = {"lo": None, "hi": None}   # every joint, for the head/torso report
    # proof that --map shoulder-rel actually fired: a law that is wired up
    # but never reached looks identical to one that is not wired at all,
    # which has already happened twice on this file.
    rel_stat = {"n": 0, "err_mm": []}
    sym_stat = {"n": 0, "w": 0.0}
    swivel_stat = {"n": 0, "miss": 0}
    branch_stat = {"n": 0, "miss": 0}
    reseed_stat = {"n": 0, "max": 0.0}
    epoch = {"v": 0, "n": 0, "max": 0.0}
    ori_scale_stat = {"n": 0}     # 遥操作分段;0 = 还没开过新的一段
    # 主循环外先给个值:它在解算那一段里赋值,而关节 CSV 写在别处,
    # 某一帧没走到那段就会 NameError,而 Isaac 会把它吞成 exit 0。
    branch_on = False
    chest_stat = {"n": 0}
    state_n = {"i": 0}
    # base should stay perfectly parked: track wheel-joint motion to catch it
    # twitching (a zero-stiffness actuator won't hold position under solver noise)
    wheel_ids = [i for i, n in enumerate(isaac_names) if "_wheel_j" in n]
    wheel_ids_t = torch.as_tensor(wheel_ids, device=device) if wheel_ids else None
    wheel_zero_vel = torch.zeros((1, len(wheel_ids)), device=device) if wheel_ids else None
    wheel_straight = torch.zeros((1, len(wheel_ids)), device=device) if wheel_ids else None

    def pin_wheels():
        # The heavy, offset swerve wheels (L/R rolling links 3.7 kg) are
        # numerically unstable in this USD and spin up regardless of drive
        # tuning. Hard-pin their joint state (straight, zero velocity) before
        # each physics step so the parked base can't spin. Cosmetic backstop;
        # doesn't touch the arm task.
        if wheel_ids:
            robot.write_joint_state_to_sim(
                wheel_straight, wheel_zero_vel, joint_ids=wheel_ids_t)

    wheel_step = 0.0                                # max per-step wheel motion
    wheel_prev = None                              # set on first tick (skip warm-up)
    wheel_warmup = 30                              # control ticks to ignore
    err_stats = {h: [0.0, 0.0, 0] for h in hands}  # [max, sum, count]
    lat_stats = [0.0, 0.0, 0]                      # wire age [sum_ms, max_ms, n]
    run_clock = {"sim": 0.0, "wall": None}         # for the real-time factor
    # Where does the frame budget actually go? arms+head+waist runs at 0.12x
    # real time and the operator reports it as lag; the candidates (head_solve,
    # the per-arm live-chest lookup, the symmetriser's scipy rotation mean, the
    # Pink solve, physics, render) span CPU and GPU and guessing between them
    # has no chance. Costs one time.time() per stage per control step.
    prof = {k: 0.0 for k in ("source", "sym", "arms", "head", "pink",
                             "physics", "other")}
    prof_n = {"steps": 0}

    def _tick(key, t0):
        now = time.perf_counter()
        prof[key] += now - t0
        return now

    # --mock-demo scorecard: ground truth per hold midpoint, in the robot
    # frame. Raw offsets (dx outward, dy up, dz fwd[-z]) map to robot
    # (x fwd, y left, z up): fwd -dz -> +x ; up +dy -> +z ; outward +dx ->
    # right hand -y / left hand +y. Sample times = MOVE_T(1.5) + hold mids.
    demo_checks, demo_results = [], []
    if args_cli.mock_demo:
        _s = args_cli.pos_scale   # expectations scale with --pos-scale

        def _sc(v):
            return tuple(_s * x for x in v)

        # (t, label, expected pos delta per hand, expected ori delta per hand
        #  as (axis, deg) in the robot frame - None = don't check).
        # Ori ground truth: mock roll = raw-z rotation -> robot R(x, -45) for
        # the right wrist, mirrored +45 for the left; mock pitch = raw-x
        # rotation (mirror-invariant) -> robot R(y, -45) both hands.
        demo_checks = [
            (8.0, "forward 25cm", {"right": _sc((0.25, 0.0, 0.0)),
                                   "left": _sc((0.25, 0.0, 0.0))}, None),
            (18.0, "lateral 30cm", {"right": _sc((0.0, -0.30, 0.0)),
                                    "left": _sc((0.0, 0.30, 0.0))}, None),
            (28.0, "overhead 30cm", {"right": _sc((0.0, 0.0, 0.30)),
                                     "left": _sc((0.0, 0.0, 0.30))}, None),
            (38.0, "wrist roll 45", {"right": (0.0, 0.0, 0.0),
                                     "left": (0.0, 0.0, 0.0)},
             {"right": ((1.0, 0.0, 0.0), -45.0),
              "left": ((1.0, 0.0, 0.0), 45.0)}),
            (45.0, "wrist pitch 45", {"right": (0.0, 0.0, 0.0),
                                      "left": (0.0, 0.0, 0.0)},
             {"right": ((0.0, 1.0, 0.0), -45.0),
              "left": ((0.0, 1.0, 0.0), -45.0)}),
        ]

    def print_summary():
        if q_seen["lo"] is None:
            return
        limits = getattr(robot.data, "joint_pos_limits",
                         getattr(robot.data, "joint_limits", None))
        # head / torso travel: --control-mode is easy to wire up and have do
        # nothing (the waist channel was gated out by the waist-abs branch and
        # every mode produced identical numbers), so report the excursion and
        # let the run answer "did it actually move" instead of assuming.
        if joint_guard is not None:
            print("[summary] " + joint_guard.summary()
                  + ("" if joint_guard.stat["checked"] or joint_guard.step_max
                     else "   <-- NEVER FIRED"))
        if args_cli.branch_at_engage:
            print(f"[summary] branch selected at clutch close on "
                  f"{branch_stat['n']} engagements"
                  f" (no operator elbow on {branch_stat['miss']})"
                  + ("" if branch_stat["n"] else "   <-- NEVER FIRED"))
        if args_cli.relax_at_limit < 1.0:
            _rs = pink_ik.relax_stat if pink_ik is not None else {"n": 0, "steps": 0}
            print(f"[summary] orientation relaxed on {_rs['n']} arm-steps of "
                  f"{_rs['steps']} control steps"
                  + ("" if _rs["n"] else "   <-- NEVER FIRED"))
        if args_cli.relax_pos_at_limit:
            _rp = (pink_ik.relax_stat_pos if pink_ik is not None else {"n": 0})
            print(f"[summary] position target blended on {_rp['n']} arm-steps"
                  + ("" if _rp["n"] else "   <-- NEVER FIRED"))
        if args_cli.ori_scale < 1.0:
            print(f"[summary] orientation scaled by {args_cli.ori_scale} on "
                  f"{ori_scale_stat['n']} arm-frames"
                  + ("" if ori_scale_stat["n"] else "   <-- NEVER FIRED"))
        print(f"[summary] teleop epochs started: {epoch['n']}, "
              f"largest solver reset {epoch['max']:.1f}deg"
              + ("" if epoch["n"] else "   <-- NEVER FIRED (页面没有开过新的一段)"))
        if args_cli.reseed_at_engage:
            print(f"[summary] solver reseeded at {reseed_stat['n']} engagements, "
                  f"largest joint change {reseed_stat['max']:.1f}deg"
                  + ("" if reseed_stat["n"] else "   <-- NEVER FIRED"))
        if args_cli.null_bias > 0.0:
            # 生效证明。这个仓库出现过 9 次"接好了从没执行",退出码还都是 0,
            # 所以开了就必须能从读数上看出它到底跑了多少帧。
            print(f"[summary] swivel target fed on {swivel_stat['n']} arm-frames"
                  f" (no operator elbow on {swivel_stat['miss']})"
                  + ("" if swivel_stat["n"] else
                     "   <-- NEVER FIRED (no body24 elbow in the stream?)"))
        if args_cli.sym_lock:
            print(f"[summary] bimanual symmetriser ran on {sym_stat['n']} frames, "
                  f"last weight {sym_stat['w']:.2f}"
                  + ("" if sym_stat["n"] else "   <-- NEVER FIRED"))
        if args_cli.map in ("chest-rel", "chest-anchor"):
            print(f"[summary] {args_cli.map} built the chest frames on "
                  f"{chest_stat['n']} frames"
                  + ("" if chest_stat["n"] else
                     "   <-- NEVER FIRED (no shoulders in the stream?)"))
        if args_cli.map == "shoulder-rel":
            print(f"[summary] shoulder-rel applied on {rel_stat['n']} arm-frames"
                  + ("" if rel_stat["n"] else "   <-- NEVER FIRED"))
        if head_ch is not None and head_ch.gaze_stat["n"]:
            g = head_ch.gaze_stat
            n = g["n"]
            print(f"[summary] head gaze fidelity: |asked - achieved| mean "
                  f"az {g['az'] / n:.2f}deg el {g['el'] / n:.2f}deg  "
                  f"(max {g['az_hi']:.1f}/{g['el_hi']:.1f});  off by >1deg on "
                  f"yaw {100 * g['yaw_clamp'] / n:.0f}% pitch "
                  f"{100 * g['pitch_clamp'] / n:.0f}% of frames")
        if torso is not None and hasattr(torso, "lean_lo"):
            span = torso.lean_hi - torso.lean_lo
            print(f"[summary] operator lean commanded: {torso.lean_lo:+.1f}..."
                  f"{torso.lean_hi:+.1f}deg (span {span:.1f})"
                  + ("" if span > 1.0 else "   <-- LEAN NEVER REACHED THE MAP"))
            if hasattr(torso, "raw_lo"):
                print(f"[summary] operator lean RAW (before the clamp): "
                      f"{np.degrees(torso.raw_lo):+.1f}..."
                      f"{np.degrees(torso.raw_hi):+.1f}deg; clamped on "
                      f"{100 * torso.sat_n / max(torso.tot_n, 1):.0f}% of frames"
                      + ("   <-- RAISE --torso-max-deg"
                         if torso.sat_n > 0.05 * torso.tot_n else ""))
            if hasattr(torso, "pitch_lo"):
                pspan = torso.pitch_hi - torso.pitch_lo
                # forward bend must DECREASE the Dexmate pitch angle
                sense = ("forward bend -> chest folds forward  OK"
                         if pspan > 1.0 else "   <-- CHEST NEVER MOVED")
                print(f"[summary] chest pitch (Dexmate definition) commanded: "
                      f"{torso.pitch_hi:+.1f}...{torso.pitch_lo:+.1f}deg "
                      f"(span {pspan:.1f}, gain {pspan / max(span, 1e-6):.2f}x "
                      f"of operator lean); {sense}")
        for grp in ("head_j", "torso_j"):
            if q_all["lo"] is None:
                break
            ids = [i for i, n in enumerate(isaac_names) if n.startswith(grp)]
            if not ids:
                continue
            span = [(isaac_names[i],
                     np.degrees(q_all["hi"][i].item() - q_all["lo"][i].item()))
                    for i in ids]
            moved = max(s for _, s in span)
            print(f"[summary] {grp[:-2]} travel: "
                  + "  ".join(f"{n}={s:.1f}deg" for n, s in span)
                  + ("" if moved > 0.5 else "   <-- DID NOT MOVE"))
        print("\n[summary] arm joint excursions vs limits (rad):")
        for k, j in enumerate(all_arm_ids):
            lo_s, hi_s = q_seen["lo"][k].item(), q_seen["hi"][k].item()
            if limits is not None:
                lo_l, hi_l = limits[0, j, 0].item(), limits[0, j, 1].item()
                margin = min(lo_s - lo_l, hi_l - hi_s)
                flag = "  <-- NEAR LIMIT" if margin < 0.10 else ""
                print(f"  {isaac_names[j]:>9s}: seen [{lo_s:+6.2f}, {hi_s:+6.2f}]"
                      f"  limits [{lo_l:+6.2f}, {hi_l:+6.2f}]  margin {margin:+5.2f}{flag}")
            else:
                print(f"  {isaac_names[j]:>9s}: seen [{lo_s:+6.2f}, {hi_s:+6.2f}]")
        for h, (mx, sm, n) in err_stats.items():
            if n:
                print(f"[summary] {h} track err while engaged: "
                      f"mean {sm / n * 1000:6.1f}mm  max {mx * 1000:6.1f}mm  (n={n})")
        if wheel_ids:
            print(f"[summary] base parked: max wheel-joint motion per control "
                  f"step {wheel_step * 1000:.3f} mrad "
                  f"({'STATIC' if wheel_step < 1e-3 else 'TWITCHING'})")
        if lat_stats[2]:
            print(f"[summary] wire latency (producer stamp -> consume): "
                  f"mean {lat_stats[0] / lat_stats[2]:5.1f}ms  "
                  f"max {lat_stats[1]:5.1f}ms  (n={lat_stats[2]})")
        if demo_results:
            worst = max(r[4] for r in demo_results)
            print(f"[summary] demo mapping scorecard (worst dev {worst:.1f}mm):")
            for label, h, got, want, dev in demo_results:
                print(f"  {label:>13s} {h:>5s}: dev {dev:6.1f}mm "
                      f"got ({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f}) "
                      f"want ({want[0]:+.2f},{want[1]:+.2f},{want[2]:+.2f})")
        if fuse_stats[0]:
            print(f"[summary] forearm bridge filled {fuse_stats[0]} wrist "
                  "samples where position was dead but orientation was live")
        if run_clock["wall"] is not None:
            wall = time.time() - run_clock["wall"]
            if wall > 0:
                print(f"[summary] real-time factor: {run_clock['sim'] / wall:.2f}x "
                      f"({run_clock['sim']:.1f} sim-s / {wall:.1f} wall-s)")
            if dt_stat["n"]:
                mean_dt = dt_stat["sum"] / dt_stat["n"]
                print(f"[summary] filter sample dt: mean {mean_dt * 1000:.1f}ms "
                      f"max {dt_stat['max'] * 1000:.1f}ms  (sim control period "
                      f"{control_dt * 1000:.1f}ms, ratio {mean_dt / control_dt:.2f}x)"
                      + ("   <-- WALL-CLOCK dt IS IN USE"
                         if mean_dt > control_dt * 1.2 else
                         "   (real time; the two agree, as they should)"))
            if run_clock["wall"] is not None and wall > 0:
                tot = sum(prof.values()) or 1.0
                n = max(prof_n["steps"], 1)
                print(f"[summary] where the wall time went ({n} control steps, "
                      f"{tot:.1f}s accounted of {wall:.1f}s):")
                for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
                    print(f"  {k:>8s}: {v:7.1f}s  {100 * v / tot:5.1f}%  "
                          f"{1000 * v / n:6.2f} ms/control-step")
        # stdout to a pipe is block-buffered and the shutdown path may os._exit
        sys.stdout.flush()

    wrist_max_rad = args_cli.wrist_max_deg * np.pi / 180.0
    targets = robot.data.default_joint_pos.clone()
    # Per-joint anti-jitter, on the FINAL targets. One-euro rather than a plain
    # low-pass: it barely filters while the operator is moving (so it does not
    # add lag where lag is felt) and clamps down hard at rest, which is exactly
    # where jitter is visible. Strength is live-adjustable from the viewer.
    joint_filt = {"f": None, "cut": float(args_cli.joint_smooth)}
    from magicdexmate.control_link import ControlSubscriber  # noqa: E402
    ctl_sub = ControlSubscriber(args_cli.control_addr)
    live_mode = {"v": args_cli.control_mode}
    smooth_ids = [i for i, n in enumerate(isaac_names)
                  if n.startswith(("L_arm_j", "R_arm_j", "head_j", "torso_j"))]

    sim_dt = sim.get_physics_dt()
    decimation = max(1, round(1.0 / sim_dt / args_cli.control_hz))
    stale_us = args_cli.stale_ms * 1000.0

    # Settle physics before touching state buffers / capturing references
    # (see teleop_vega_sharpa.py: tick-0 buffers are stale, mid-settle capture
    # leaves the IK a phantom error to fight).
    #
    # Settle to CONVERGENCE, not to a step count. This was 120 steps = 0.6 s,
    # which was silently enough only because the old torso home happened to sit
    # where the articulation starts. The torso joints are velocity-limited to
    # 0.9 rad/s, so the authored home (torso_j2 = 120.9 deg, ~31 deg away)
    # needs ~2.3 s. At 0.6 s the chest was still 14.2 deg short - and the waist
    # bind point, which is captured right here and then held constant, got
    # snapshotted mid-flight. The torso kept creeping the remaining 20 cm
    # during the run while the bind point did not, which is the whole of the
    # 8 mm -> 195 mm wrist tracking regression. Exit code was 0 throughout.
    # Criterion is "stopped moving", not "reached the target": the head drive
    # saturates at its 6 Nm URDF effort limit and droops to an equilibrium it
    # never leaves, so requiring target-match would spin here forever. The
    # per-joint command-vs-achieved report below says what actually landed.
    settle_steps, settle_moved = 0, float("inf")
    # ...and the head is excluded from the criterion entirely: its drive
    # saturates at 6 Nm and it never stops swinging, so waiting on it just
    # burns 8 s of startup on every launch.
    settle_ids = [i for i, n in enumerate(isaac_names) if not n.startswith("head_j")]
    q_prev = robot.data.joint_pos[0, settle_ids].clone()
    for settle_steps in range(1, 2001):
        robot.set_joint_position_target(targets)
        pin_wheels()          # hold the parked base straight through settle
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if settle_steps % 20:
            continue
        q_now = robot.data.joint_pos[0, settle_ids]
        settle_moved = (q_now - q_prev).abs().max().item()
        q_prev = q_now.clone()
        if settle_moved < np.radians(0.02) and settle_steps >= 120:
            break
    settle_err = (targets[0, settle_ids]
                  - robot.data.joint_pos[0, settle_ids]).abs().max().item()
    print(f"[settle] {settle_steps} steps ({settle_steps * sim_dt:.2f}s), last 20 steps "
          f"moved {np.degrees(settle_moved):.3f}deg; worst joint {np.degrees(settle_err):.2f}deg "
          f"off target" + ("" if settle_moved < np.radians(0.02) else "   <-- STILL MOVING"))
    # Did the torso actually GET to the authored home? 2026-08-01: it did not -
    # torso_j2 was commanded to 120.9 deg and settled at 106.7, because the USD
    # caps it there while the URDF says 3.14 rad. Everything downstream (the
    # waist bind point, the arm workspace, "is the chest leaning forward") is
    # computed from the chest, so a silently clamped torso poisons the whole
    # mapping and shows up only as a large wrist tracking error. Print the
    # command-vs-achieved and let the run say so.
    _t_lim = getattr(robot.data, "joint_pos_limits",
                     getattr(robot.data, "joint_limits", None))
    for n in ("torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"):
        if n not in isaac_names:
            continue
        j = isaac_names.index(n)
        cmd, got = targets[0, j].item(), robot.data.joint_pos[0, j].item()
        lim = ("" if _t_lim is None else
               f"  limits [{np.degrees(_t_lim[0, j, 0].item()):+.1f}, "
               f"{np.degrees(_t_lim[0, j, 1].item()):+.1f}]")
        flag = "   <-- DID NOT REACH IT" if abs(cmd - got) > np.radians(1.0) else ""
        print(f"[home] {n}: commanded {np.degrees(cmd):+7.1f}deg  "
              f"settled {np.degrees(got):+7.1f}deg{lim}{flag}")
    for h, arm in arms.items():
        p, q = arm.ee_pose_in_base()
        arm.home_ee = (p.clone(), q.clone())        # --anchor home reference
        print(f"[ik] {h} EE settled: pos={[round(v, 3) for v in p[0].tolist()]} "
              f"quat={[round(v, 3) for v in q[0].tolist()]}")
        if args_cli.hold:
            arm.set_pose_command(p.clone(), q.clone())
    if os.environ.get("VEGA_DUMP_JOINTS"):
        jp = robot.data.joint_pos[0]
        for h, arm in arms.items():
            names = [isaac_names[j] for j in arm.joint_ids]
            vals = [round(jp[j].item(), 4) for j in arm.joint_ids]
            print(f"[qdump] {h}: " + " ".join(f"{n}={v}" for n, v in zip(names, vals)))

    if pink_ik is not None:
        from pink_vega_ik import ARM_PIN_ORDER  # noqa: E402
        cp, cq = ee_pose_in_base_of(chest_body_idx)
        chest_pos = {h: cp[0].tolist() for h in ee_body_idx}
        chest_quat = {h: cq[0].tolist() for h in ee_body_idx}
        ee_pos, ee_quat = {}, {}
        for h, bidx in ee_body_idx.items():
            p, q = ee_pose_in_base_of(bidx)
            ee_pos[h], ee_quat[h] = p[0].tolist(), q[0].tolist()
        isaac_q = {n: float(robot.data.joint_pos[0, isaac_names.index(n)].item())
                   for n in ARM_PIN_ORDER if n in isaac_names}
        resid = pink_ik.validate_chest(chest_pos, chest_quat, ee_pos, ee_quat, isaac_q)
        rmax = max(resid.values())
        print(f"[pink] chest-relative EE match Isaac<->pinocchio: "
              f"R {resid['right']:.2f}mm L {resid['left']:.2f}mm "
              f"({'OK' if rmax < 20 else 'HIGH - check URDF/frame match'})")
        ang = pink_ik.calibrate_ee_rot(chest_pos, chest_quat, ee_pos, ee_quat,
                                       isaac_q)
        print(f"[pink] EE-frame orientation offset (USD vs URDF, corrected): "
              f"R {ang['right']:.1f}deg L {ang['left']:.1f}deg")

    # waist-abs mapping: robot-side waist bind point, from the LIVE settled
    # chest pose (anatomical: human waist sits 0.25 m below the chest/shoulder
    # line - same constant the puppet skeleton uses). Torso is locked in this
    # mode, so the point is constant in the base frame.
    waist_bind_base = None
    yaw_fix_cs = None
    if args_cli.map in ("waist-abs", "shoulder-rel", "chest-rel",
                        "chest-anchor"):
        if pink_ik is None:
            raise SystemExit(f"--map {args_cli.map} requires Pink IK (--ik pink)")
        waist_bind_base = np.array(cp[0].tolist()) + np.array(
            [0.0, 0.0, -0.25 + args_cli.waist_bind_up])
        if abs(args_cli.waist_yaw_fix_deg) > 1e-6:
            th = np.radians(args_cli.waist_yaw_fix_deg)
            yaw_fix_cs = (float(np.cos(-th)), float(np.sin(-th)))
        print(f"[map] {args_cli.map}: bind point (base frame) "
              f"{np.round(waist_bind_base, 3).tolist()}, scale "
              f"{args_cli.pos_scale}, yaw-fix {args_cli.waist_yaw_fix_deg}deg, "
              + ("torso ON the authored path (bind point follows the live chest)"
                 if mode_waist else "torso locked"))

    if args_cli.source == "mock":
        if args_cli.mock_demo:
            source = MockPicoSource.demo()
        elif args_cli.mock_stress:
            source = MockPicoSource.stress()
        else:
            source = MockPicoSource()
    elif args_cli.source == "replay":
        if not args_cli.replay_file:
            raise SystemExit("--source replay requires --replay-file")
        source = PicoLogSource(args_cli.replay_file, loop=args_cli.replay_loop)
    else:
        source = PicoZmqSource(sub=args_cli.sub)
    source.start()
    stall_detector = DeviceStallDetector(stall_ms=args_cli.device_stale_ms)
    freeze_detector = TrackerFreezeDetector(freeze_ms=args_cli.tracker_freeze_ms) \
        if args_cli.tracker_freeze_ms > 0 else None
    # Per-channel freeze watch. The aggregate detector above only fires when
    # EVERY tracker dies together - blind on the hybrid rig (raw wrists + model
    # waist), where live wrists mask a dead body model. The waist matters most:
    # in --map waist-abs it is the mapping origin, so a frozen one turns every
    # target into garbage while the arms still look alive (2026-07-29 replay of
    # the _raw batch: 280mm left-arm error, zero warnings).
    channel_freeze: list[TrackerFreezeDetector] = []
    if args_cli.mode == "trackers" and args_cli.tracker_freeze_ms > 0:
        channel_freeze = [
            TrackerFreezeDetector(freeze_ms=args_cli.tracker_freeze_ms,
                                  keys={sn}, label=lbl)
            for lbl, sn in (("left wrist", args_cli.tracker_left),
                            ("right wrist", args_cli.tracker_right),
                            ("waist", args_cli.tracker_waist))]
    # position bridge, one per consumed wrist (the waist is not bridged: its
    # tracker sits where the cameras never see it, so there is no live-position
    # window to fit an arc from - it must come from the body model)
    fusers: dict[str, ForearmFuser] = {}
    if args_cli.mode == "trackers" and args_cli.forearm_bridge:
        fusers = {sn: ForearmFuser(max_bridge_s=args_cli.forearm_bridge)
                  for sn in (args_cli.tracker_left, args_cli.tracker_right)}
    fuse_prev: dict[str, np.ndarray] = {}
    fuse_dead: dict[str, bool] = {}
    fuse_stats = [0]
    fuse_warned = [False]
    glitch_gates: dict[str, TrackerGlitchGate] = {}
    if args_cli.mode == "trackers" and not args_cli.no_glitch_gate:
        glitch_gates = {sn: TrackerGlitchGate() for sn in
                        {args_cli.tracker_left, args_cli.tracker_right,
                         args_cli.tracker_waist, "LELBOW", "RELBOW"}}
    if args_cli.mode == "trackers":
        print(f"[run] mode=trackers wrists=L:{args_cli.tracker_left}/"
              f"R:{args_cli.tracker_right} waist={args_cli.tracker_waist} "
              "engage=auto (no controllers)")
    print(f"[run] source={args_cli.source} mode={args_cli.mode} hands={hands} "
          f"hold={args_cli.hold} control={1.0 / (sim_dt * decimation):.0f}Hz  "
          "Ctrl-C to stop")

    # keyboard controls (GUI only; no controllers on the final rig): R re-zeroes
    # (re-anchors the clutch so the operator can recenter), SPACE pauses.
    kb_state = {"rezero": False, "paused": False}
    try:
        import carb  # noqa: E402
        import omni.appwindow  # noqa: E402

        _kbd = omni.appwindow.get_default_app_window().get_keyboard()

        def _on_key(e, *a):
            if e.type == carb.input.KeyboardEventType.KEY_PRESS:
                if e.input == carb.input.KeyboardInput.R:
                    kb_state["rezero"] = True
                elif e.input == carb.input.KeyboardInput.SPACE:
                    kb_state["paused"] = not kb_state["paused"]
                    print(f"\n[kb] {'PAUSED' if kb_state['paused'] else 'resumed'}")
            return True

        _kb_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
            _kbd, _on_key)
        print("[kb] R = re-zero (recenter), SPACE = pause  (focus the sim window)")
    except Exception as e:
        print(f"[kb] keyboard unavailable ({type(e).__name__}) - headless/no window")

    wrist_serial = {"left": args_cli.tracker_left, "right": args_cli.tracker_right}
    elbow_serial = {"left": "LELBOW", "right": "RELBOW"}

    def target_ref(frame, h, trackers):
        """(wrist target pose7, reference pose7) for hand h, or (None, None)
        if a required tracker is missing this frame.
        - controllers mode: target = held controller, reference = headset.
        - trackers mode: target = wrist tracker, reference = waist tracker
          (body-relative, the natural G1-style mapping), taken from the
          glitch-gated `trackers` dict."""
        if args_cli.mode == "trackers":
            tgt = trackers.get(wrist_serial[h])
            ref = trackers.get(args_cli.tracker_waist)
            return (tgt, ref) if (tgt is not None and ref is not None) else (None, None)
        return frame.controller(h), frame.head

    jc_f = jc_w = None
    q_full_prev = None
    if args_cli.joint_csv:
        import csv as _csv
        jc_f = open(args_cli.joint_csv, "w", newline="")
        jc_w = _csv.writer(jc_f)
        # 多一列 branch:这一帧是不是处在选支窗口里。窗口内手臂在有意迁移,
        # 那段的关节运动是受控的,不能和平时的抖动混在一起统计。
        jc_w.writerow(["t"] + [f"cmd_{n}" for n in JOINT_DUMP]
                      + [f"got_{n}" for n in JOINT_DUMP] + ["branch"])
        print(f"[joints] writing {args_cli.joint_csv} "
              f"({len(JOINT_DUMP)} joints, commanded + achieved)")

    trace_f = trace_w = None
    if args_cli.trace_csv:
        import csv
        trace_f = open(args_cli.trace_csv, "w", newline="")
        trace_w = csv.writer(trace_f)
        trace_w.writerow(["t", "hand",
                          "human_x", "human_y", "human_z",     # wrist rel waist
                          "cmd_x", "cmd_y", "cmd_z",           # EE target, base
                          "ee_x", "ee_y", "ee_z",              # EE actual, base
                          "err_mm"])
        print(f"[trace] writing {args_cli.trace_csv}")

    run_clock["wall"] = time.time()
    t, step, last_print = 0.0, 0, time.time()
    n_frames, last_t_us, was_fresh = 0, -1, True
    is_zmq = isinstance(source, PicoZmqSource)
    warned_waiting, got_first = False, False
    warned_no_trk = False
    control_dt = sim_dt * decimation
    # ...but that is SIM time, and the tracker samples arrive in WALL time. The
    # one-euro filters and the head/waist channels smooth a real-world signal,
    # so they need the real interval between samples. When the sim runs slower
    # than real time the two diverge and the filter silently stops filtering:
    # its cutoff adapts on |dx/dt|, and feeding a dt that is N times too small
    # makes the estimated hand speed N times too large, which opens the cutoff
    # wide. At the 0.12x this rig actually achieves in arms+head+waist that is
    # an 8x error - reported live, 2026-08-02, as "the palm jitters badly", and
    # it gets worse exactly as the sim gets slower, which is why it reads as a
    # lag problem rather than a filter problem.
    wall_prev = {"t": None}

    dt_stat = {"n": 0, "sum": 0.0, "max": 0.0}

    def sample_dt() -> float:
        now = time.time()
        prev, wall_prev["t"] = wall_prev["t"], now
        if prev is None:
            return control_dt
        # clamp: a scheduler hiccup must not hand the filter a 2 s dt
        dt = float(min(max(now - prev, control_dt * 0.25), 0.25))
        dt_stat["n"] += 1
        dt_stat["sum"] += dt
        dt_stat["max"] = max(dt_stat["max"], dt)
        return dt
    try:
        while simulation_app.is_running():
            _t0 = time.perf_counter()
            if step % decimation == 0:
                prof_n["steps"] += 1
                # real sample interval for everything that smooths a real-world
                # signal - see sample_dt()'s note. Taken once per control step
                # so every consumer of it sees the same number.
                samp_dt = sample_dt()
                _cs = ctl_sub.latest(time.time())
                if _cs is not None:
                    _c = float(_cs.get("smooth", joint_filt["cut"]))
                    if abs(_c - joint_filt["cut"]) > 1e-6:
                        joint_filt["cut"], joint_filt["f"] = _c, None
                        print(f"[smooth] 防抖动强度 -> {_c:.1f} Hz"
                              + ("  (关闭)" if _c <= 0 else ""))
                    if _cs.get("shutdown"):
                        # 控制台要求正常退出。走正常收尾(simulation_app.close)
                        # 而不是被信号杀:Ctrl-C / kill 长跑的 Isaac 会把
                        # nvidia_uvm 搞坏(nvidia-smi 正常、CUDA unknown error),
                        # 2026-08-06 又复现一次,这一条就是那次加的。
                        print("[ctl] 收到停止指令,正常退出")
                        break
                    # 一段新的遥操作。参考实现(V2AP / T-Rex)每个回合开始都把
                    # 机器人物理送回默认关节位,再从那个姿势重建 IK 种子与映射
                    # 锚点 —— 坏构型活不过一个回合。我们此前是连续漫游、从不
                    # 复位,所以求解器一旦绕进坏支或贴上限位就出不来(实测顶
                    # 限位的帧占 14.15%,最长连续 5.2 秒)。这里是求解器侧:
                    # 构型拉回默认姿势,并让两条手臂在下一帧重新啮合、重取锚点。
                    _e = int(_cs.get("epoch") or 0)
                    if _e != epoch["v"]:
                        epoch["v"] = _e
                        if pink_ik is not None:
                            _d = pink_ik.reset_home()
                            epoch["n"] += 1
                            epoch["max"] = max(epoch["max"], _d)
                            print(f"[epoch] 第 {_e} 段开始:求解器构型回默认位,"
                                  f"最大关节改动 {_d:.1f}°")
                        for _a in arms.values():
                            _a.engaged = False      # 下一帧走啮合分支,重取锚点
                    _m = _cs.get("mode")
                    if _m in ("arms", "arms+head", "arms+head+waist") \
                            and _m != live_mode["v"]:
                        live_mode["v"] = _m
                        print(f"[mode] 切换到 {_m}"
                              + ("  (需重启才能改变已建立的通道;当前只影响下发门控)"
                                 if _m != args_cli.control_mode else ""))
                # mock/replay are phase-locked to sim time (see source docstrings)
                if isinstance(source, (MockPicoSource, PicoLogSource)):
                    frame = source.sample_at(t)
                else:
                    frame = source.get_latest()
                _t0 = _tick("source", _t0)
                now_us = time.time_ns() // 1000
                link_fresh = frame is not None and (
                    isinstance(source, (MockPicoSource, PicoLogSource))
                    or (now_us - frame.t_us) < stale_us
                )
                # device clock stall = frozen headset (poses stop, but the
                # producer keeps wall-clock-stamping, so link_fresh can't see
                # it); tracker value freeze = frozen BODY ESTIMATE (device
                # clock keeps ticking, only the joint values die)
                device_stalled = stall_detector.update(frame)
                tracker_frozen = freeze_detector.update(frame) \
                    if freeze_detector is not None else False
                frozen_channels = [d.label for d in channel_freeze
                                   if d.update(frame)]
                fresh = (link_fresh and not device_stalled
                         and not tracker_frozen and not frozen_channels)

                if frame is None:
                    # nothing received yet: expected briefly at startup, so no
                    # scary STALE line - one friendly notice if it drags on
                    if not warned_waiting and time.time() - run_clock["wall"] > 3.0:
                        print(f"\n[wait] no producer frames yet on {args_cli.sub}"
                              " - producer running? address right?")
                        warned_waiting = True
                else:
                    if not got_first:
                        got_first = True
                        if is_zmq:
                            print(f"\n[ok] producer stream connected ({args_cli.sub})")
                    # edge-triggered on a freshness transition, so the operator
                    # gets one clear line instead of a silent freeze / recovery
                    if fresh != was_fresh:
                        trackers_mode = args_cli.mode == "trackers"
                        if not fresh:
                            if device_stalled:
                                why = "device poses FROZEN (headset asleep?)"
                            elif tracker_frozen:
                                why = ("tracker values FROZEN (body tracking "
                                       "stopped - shake the trackers, look at "
                                       "your wrists)")
                            elif frozen_channels:
                                what = " + ".join(frozen_channels)
                                why = (f"{what} FROZEN (that channel is dead "
                                       "while the others stream) - out of the "
                                       "headset FOV, or the body model never "
                                       "started: redo the full-body calibration")
                            else:
                                why = "producer link STALE (down?)"
                            hint = "re-anchors automatically on recovery" \
                                if trackers_mode else "re-grip to resume"
                            print(f"\n[warn] {why} - arms disengaged, {hint}")
                        else:
                            hint = "re-anchoring at the current pose" \
                                if trackers_mode else "squeeze grip to re-engage"
                            print(f"\n[ok] stream live again - {hint}")
                        was_fresh = fresh

                # keyboard R re-zeroes: drop engagement so the next frame
                # re-anchors the clutch at the current pose (recenter)
                if kb_state["rezero"]:
                    for arm in arms.values():
                        arm.engaged = False
                    if torso is not None:
                        torso.engaged = False
                    kb_state["rezero"] = False
                    print("\n[kb] re-zeroed - re-anchoring on next frame")

                # any staleness (or a keyboard pause) force-releases the clutch:
                # a resumed stream then re-zeroes on the next grip instead of
                # snapping the arm from a stale reference to wherever the hand
                # drifted during the freeze
                if not fresh or kb_state["paused"]:
                    for arm in arms.values():
                        arm.engaged = False
                    if torso is not None:
                        torso.engaged = False

                if (fresh and not kb_state["paused"]
                        and frame.t_us != last_t_us and not args_cli.hold):
                    last_t_us = frame.t_us
                    n_frames += 1
                    if is_zmq:
                        age_ms = (now_us - frame.t_us) / 1000.0
                        lat_stats[0] += age_ms
                        lat_stats[1] = max(lat_stats[1], age_ms)
                        lat_stats[2] += 1
                    if (args_cli.mode == "trackers" and not warned_no_trk
                            and not frame.trackers):
                        print("\n[warn] --mode trackers but no trackers in the "
                              "stream - headset Full Body tracking not running? "
                              "(is_body_data_available stuck False; see "
                              "workflow_fj.md 2026-07-23)")
                        warned_no_trk = True
                    # forearm bridge: these trackers lose POSITION whenever the
                    # headset cameras cannot see them while ORIENTATION keeps
                    # streaming, so a dropout leaves a stale translation on a
                    # live rotation. Rebuild the translation from the arc the
                    # wrist sweeps about the elbow (see pico/arm_fusion.py:
                    # 1 s dropout p90 222mm -> 118mm on the 2026-07-29 takes).
                    # Runs BEFORE the glitch gate so an estimate still has to
                    # pass the same human-plausibility check as a measurement.
                    trk_use = frame.trackers
                    if fusers and frame.trackers:
                        trk_use = dict(frame.trackers)
                        t_s = frame.t_us / 1e6
                        for sn, fu in fusers.items():
                            p7 = trk_use.get(sn)
                            if p7 is None:
                                continue
                            p7 = np.asarray(p7, dtype=float)
                            prev = fuse_prev.get(sn)
                            live = prev is None or not np.array_equal(p7[:3], prev)
                            fuse_prev[sn] = p7[:3].copy()
                            rot = pose7_to_world_rot(p7)
                            pos_w = R_HEADSET_TO_WORLD @ p7[:3]
                            est, ok = fu.update(t_s, pos_w, rot, live)
                            fuse_dead[sn] = not ok
                            if not live and ok:
                                # back to the raw y-up frame the rest of the
                                # pipeline (process_xr_pose) expects
                                p7 = p7.copy()
                                p7[:3] = R_HEADSET_TO_WORLD.T @ est
                                trk_use[sn] = p7
                                fuse_stats[0] += 1
                        if any(fuse_dead.values()) and not fuse_warned[0]:
                            dead = [k for k, v in fuse_dead.items() if v]
                            print(f"\n[fuse] position lost too long on {dead} "
                                  "- bridge gave up, falling back to the freeze "
                                  "detector")
                            fuse_warned[0] = True
                    if glitch_gates and frame.trackers:
                        # keep whatever the bridge produced - re-copying
                        # frame.trackers here would silently discard it
                        trk_use = dict(trk_use)
                        gate_reanchor = False
                        for sn, gate in glitch_gates.items():
                            p = trk_use.get(sn)
                            if p is None:
                                continue
                            use, reanchor = gate.update(frame.t_us, p)
                            trk_use[sn] = use
                            gate_reanchor = gate_reanchor or reanchor
                        if gate_reanchor:
                            for arm in arms.values():
                                arm.engaged = False
                            if torso is not None:
                                torso.engaged = False
                            print("\n[gate] tracker relocated (sustained jump) "
                                  "- re-anchoring all channels")
                    # --map shoulder-rel: anchor each arm at its own SHOULDER (torso
                    # proportions never enter), then pin the LEFT-RIGHT wrist vector to the
                    # operator's own, 1:1 and unscaled - two hands on one object stay that
                    # far apart whoever's arms they are. The correction is DIFFERENTIAL, so
                    # it never moves the pair as a whole; common-mode drift is what the
                    # operator agreed to give away. Offline over six takes: relative error
                    # 62 mm (426 worst) -> 0, for 31 mm of absolute drift, orientation
                    # unchanged (>10 deg on 1% of frames).
                    # Operator symmetric -> robot symmetric, enforced on the
                    # TARGETS. Pink returns mirrored joints for exactly mirrored
                    # targets (0.1 deg), so making the pair an exact mirror is
                    # the only place this can be guaranteed; weighting the
                    # solver was tried and does nothing. Real inputs are never
                    # exactly mirrored - the operator's own wrists sit 19-182 mm
                    # apart across their thirteen poses - and the 7-DoF null
                    # space turns tens of mm into 90-150 deg of joint spread.
                    _t0 = _tick("other", _t0)
                    # --- chest-rel: the two chest frames, once per step ------
                    chest_rel_R = None
                    if args_cli.map in ("chest-rel", "chest-anchor"):
                        _r = trk_use.get(args_cli.tracker_waist)
                        _sl = trk_use.get("L_SHOULDER")
                        _sr = trk_use.get("R_SHOULDER")
                        if _sl is None and frame.body24 is not None:
                            _sl, _sr = frame.body24[16], frame.body24[17]
                        if _r is not None and _sl is not None and _sr is not None:
                            _pl = process_xr_pose(
                                np.asarray(_sl, dtype=np.float64),
                                np.asarray(_r, dtype=np.float64))[:3, 3]
                            _pr = process_xr_pose(
                                np.asarray(_sr, dtype=np.float64),
                                np.asarray(_r, dtype=np.float64))[:3, 3]
                            _org = 0.5 * (_pl + _pr)
                            _y = _pl - _pr
                            _ny = np.linalg.norm(_y)
                            _nz = np.linalg.norm(_org)
                            if _ny > 1e-6 and _nz > 1e-6:
                                # operator chest frame from POSITIONS: PICO's
                                # skeleton is a template FK fit, so a joint's
                                # own rotation is inferred while the shoulder
                                # positions are the part observation pins
                                # (upper-arm bone length CV 0.1%).
                                _y = _y / _ny
                                _z = _org / _nz
                                _x = np.cross(_y, _z)
                                _x = _x / max(np.linalg.norm(_x), 1e-9)
                                _hR = np.column_stack([_x, _y, np.cross(_x, _y)])
                                _cp, _cq = ee_pose_in_base_of(chest_body_idx)
                                _cp = _cp[0].cpu().numpy()
                                _cqn = _cq[0].cpu().numpy()
                                _cR = _Rot.from_quat([_cqn[1], _cqn[2], _cqn[3],
                                                      _cqn[0]]).as_matrix()
                                chest_rel_R = (_cR, _cp, _org, _hR)
                                chest_stat["n"] += 1
                    sym_lock = None
                    if args_cli.sym_lock and args_cli.map == "waist-abs" \
                            and waist_bind_base is not None:
                        _r = trk_use.get(args_cli.tracker_waist)
                        _pq = {}
                        for _h in arms:
                            _w = trk_use.get(wrist_serial[_h])
                            if _w is None or _r is None:
                                _pq = {}
                                break
                            _T = process_xr_pose(np.asarray(_w, dtype=np.float64),
                                                 np.asarray(_r, dtype=np.float64))
                            _p = _T[:3, 3]
                            if yaw_fix_cs is not None:
                                _cy, _sy = yaw_fix_cs
                                _p = np.array([_cy * _p[0] - _sy * _p[1],
                                               _sy * _p[0] + _cy * _p[1], _p[2]])
                            _pq[_h] = (waist_bind_base + _p * args_cli.pos_scale,
                                       _T[:3, :3] @ PALM_FIX[_h])
                        if len(_pq) == 2:
                            _pr, _pl, _Rr, _Rl, _w = symmetrise(
                                _pq["right"][0], _pq["left"][0],
                                _pq["right"][1], _pq["left"][1])
                            sym_lock = {"right": (_pr, _Rr), "left": (_pl, _Rl)}
                            sym_stat["w"] = _w
                            sym_stat["n"] += 1
                    _t0 = _tick("sym", _t0)
                    rel_lock_p = None
                    if args_cli.map == "shoulder-rel":
                        # Take the waist reference HERE. `ref` is assigned
                        # inside the per-arm loop below, so reading it from a
                        # pre-pass is a NameError on the first control step -
                        # and Isaac swallows it into a clean exit 0, which is
                        # why this looked like "the loop never ran".
                        _ref = trk_use.get(args_cli.tracker_waist)
                        raw = {}
                        for _h, _a in arms.items():
                            _w = trk_use.get(wrist_serial[_h])
                            _s = trk_use.get("L_SHOULDER" if _h == "left" else "R_SHOULDER")
                            if _s is None and frame.body24 is not None:
                                # pre-2026-07-31 recordings have no named
                                # shoulder; body24 always did
                                _s = frame.body24[16 if _h == "left" else 17]
                            if _w is None or _s is None or _ref is None:
                                raw = {}
                                break
                            _e_raw = trk_use.get(
                                "L_ELBOW" if _h == "left" else "R_ELBOW")
                            if _e_raw is None and frame.body24 is not None:
                                _e_raw = frame.body24[18 if _h == "left" else 19]
                            if _e_raw is None:
                                raw = {}
                                break
                            _pw = mat_to_pos_quat_wxyz(process_xr_pose(
                                np.asarray(_w, dtype=np.float64), _ref))[0]
                            _ps = mat_to_pos_quat_wxyz(process_xr_pose(
                                np.asarray(_s, dtype=np.float64), _ref))[0]
                            _pe = mat_to_pos_quat_wxyz(process_xr_pose(
                                np.asarray(_e_raw, dtype=np.float64), _ref))[0]
                            raw[_h] = (_pw, _ps, _pe)
                        if len(raw) == 2:
                            _cp = ee_pose_in_base_of(chest_body_idx)[0][0].cpu().numpy()
                            _t = {}
                            for _h in raw:
                                _pw, _ps, _pe = raw[_h]
                                # scale by the operator's ARM LENGTH (a
                                # constant), never by the current shoulder-wrist
                                # distance. Dividing by the live distance
                                # normalises the vector, which pins the robot's
                                # hand at FULL EXTENSION no matter how folded
                                # the operator's arm is - the operator reported
                                # exactly that: "when my arms come close to my
                                # body, Dexmate's do not".
                                _hl = (np.linalg.norm(_pe - _ps)
                                       + np.linalg.norm(_pw - _pe))
                                _k = ROB_ARM / max(_hl, 1e-3)
                                _t[_h] = _cp + ROB_SH[_h] + _k * (_pw - _ps)
                            _e = ((raw["left"][0] - raw["right"][0])
                                  - (_t["left"] - _t["right"]))
                            rel_lock_p = {"left": _t["left"] + 0.5 * _e,
                                          "right": _t["right"] - 0.5 * _e}
                    for h, arm in arms.items():
                        tgt, ref = target_ref(frame, h, trk_use)
                        # engage: automatic in trackers mode (no controllers on
                        # the final rig - a grip gate would never open); the
                        # grip clutch only drives the legacy controllers mode
                        if args_cli.mode == "trackers":
                            engaged_now = tgt is not None and t >= args_cli.auto_engage_delay
                        else:
                            engaged_now = arm.latch.update(frame.grip(h))
                        if engaged_now and tgt is None:
                            engaged_now = False   # can't engage without a target
                        if engaged_now and not arm.engaged:
                            # engage: capture references. The reference pose
                            # (headset / waist tracker) is frozen here with
                            # --head-ref engage, so reference drift during a hold
                            # no longer moves the arm.
                            if args_cli.reseed_at_engage and pink_ik is not None:
                                # 求解器的内部构型是整场会话的持久开环状态,而
                                # 下面的锚点取自 Isaac 实测的末端 —— 两者本来
                                # 就可能对不上,更要紧的是一旦构型绕进坏支或
                                # 贴上限位,重新啮合只换锚点不换构型,坏姿态被
                                # 继承下去。这里把它拉回实测关节角,让锚点与
                                # 求解起点重新落在同一个构型上。
                                _iq = {n: float(
                                    robot.data.joint_pos[0, isaac_names.index(n)].item())
                                    for n in ARM_PIN_ORDER if n in isaac_names}
                                _before = pink_ik.config.q.copy()
                                pink_ik.set_config_from_isaac(_iq)
                                _d = np.degrees(np.abs(pink_ik.config.q - _before)).max()
                                reseed_stat["n"] += 1
                                reseed_stat["max"] = max(reseed_stat["max"], _d)
                                print(f"[reseed] {h} 啮合时重播种,最大关节改动 {_d:.1f}°")
                            arm.ref0 = np.asarray(ref, dtype=np.float64).copy()
                            T0 = process_xr_pose(tgt, arm.ref0)
                            arm.pico_p0, q0 = mat_to_pos_quat_wxyz(T0)
                            arm.pico_q0_inv = quat_inv(torch.as_tensor(
                                q0, dtype=torch.float32, device=device).unsqueeze(0))
                            if args_cli.anchor == "home" and arm.home_ee is not None:
                                # imitation-style: EE anchor is always the
                                # settled home pose - engage in the matching
                                # neutral posture and the mapping is identical
                                # after every re-anchor (no ratchet drift)
                                arm.ee_p0 = arm.home_ee[0].clone()
                                arm.ee_q0 = arm.home_ee[1].clone()
                            else:
                                p, q = arm.ee_pose_in_base()
                                arm.ee_p0, arm.ee_q0 = p.clone(), q.clone()
                            if arm.pos_filter is not None:
                                arm.pos_filter.reset()
                                arm.quat_filter.reset()
                            arm.solved_pos = None   # force a solve on re-engage
                            arm.pico_el0 = arm.el_p0 = arm.el_tgt = None
                            if (pink_ik is not None and elbow_body_idx
                                    and args_cli.mode == "trackers"):
                                el_raw = trk_use.get(elbow_serial[h])
                                if el_raw is not None:
                                    T_el = process_xr_pose(
                                        np.asarray(el_raw, dtype=np.float64),
                                        arm.ref0)
                                    arm.pico_el0, _ = mat_to_pos_quat_wxyz(T_el)
                                    p_el, _ = ee_pose_in_base_of(elbow_body_idx[h])
                                    arm.el_p0 = p_el[0].cpu().numpy().copy()
                            arm.engage_t = t
                            # 选支要等腕目标设好之后再做,这里只留个待办标记。
                            # 在这一刻做会用到上一帧的目标。
                            arm.want_branch = True
                            print(f"\n[clutch] {h} engaged")
                        arm.engaged = engaged_now
                        if not arm.engaged:
                            continue
                        if tgt is None:
                            continue   # tracker dropout mid-hold: hold last cmd
                        # trackers mode: ALWAYS live waist reference. Body-
                        # relative mapping only cancels leaning/turning/stepping
                        # if the root is current; freezing it at engage (the
                        # controllers-mode --head-ref default) turns body motion
                        # into arm-target drift that feels like accumulating
                        # error (2026-07-26 live session).
                        if args_cli.mode == "trackers":
                            ref_use = ref
                        else:
                            ref_use = arm.ref0 if args_cli.head_ref == "engage" else ref
                        T = process_xr_pose(tgt, ref_use)
                        p_now, q_now_np = mat_to_pos_quat_wxyz(T)
                        arm.human_p = p_now.copy()   # wrist rel waist, for --trace-csv
                        if args_cli.map in ("waist-abs", "shoulder-rel",
                                            "chest-rel", "chest-anchor"):
                            # absolute law: robot waist->EE vector = human
                            # waist->wrist vector (p_now IS wrist-rel-waist in
                            # base axes). No anchor state; ramp in from the
                            # engage pose so engage never jerks the arm.
                            if mode_waist:
                                # The bind point is derived from the chest, and
                                # in this mode the chest MOVES. Recomputing it
                                # from the live chest is what keeps the arms
                                # bound to the body instead of to a stale point
                                # in the base frame - leave it at the startup
                                # value and the hands drift forward as you bend.
                                waist_bind_base = np.array(
                                    ee_pose_in_base_of(chest_body_idx)[0][0]
                                    .cpu().numpy()) + np.array(
                                        [0.0, 0.0, -0.25 + args_cli.waist_bind_up])
                            if yaw_fix_cs is not None:
                                cy, sy = yaw_fix_cs
                                p_now = np.array([cy * p_now[0] - sy * p_now[1],
                                                  sy * p_now[0] + cy * p_now[1],
                                                  p_now[2]])
                            p_abs = waist_bind_base + p_now * args_cli.pos_scale
                            if chest_rel_R is not None \
                                    and args_cli.map == "chest-anchor":
                                # CHEST-ANCHOR: origin from the chest, axes
                                # from GRAVITY. Takes the correct half of each
                                # of the two laws that came before it.
                                #
                                #   waist-abs   axes gravity-aligned, so level
                                #               is level - but the bind point
                                #               ALSO follows the chest while
                                #               the operator's own bend is
                                #               already inside
                                #               (wrist - waist), so bending
                                #               counts twice.
                                #   chest-rel   bend counts once - but it
                                #               rotates everything by
                                #               R_chest_robot, and the robot's
                                #               chest leans 55 deg, so the
                                #               operator's level front-raise
                                #               came out angled 55 deg DOWN.
                                #               Reported live 2026-08-03:
                                #               "the arms look pressed down -
                                #               level should be level with
                                #               respect to the GROUND, not to
                                #               the robot's waist".
                                #
                                # Anchoring the origin at the chest removes the
                                # double count (the hand follows the body);
                                # leaving the axes alone keeps level level.
                                _cP = chest_rel_R[1]
                                _hO = chest_rel_R[2]
                                p_abs = _cP + (p_now - _hO) * args_cli.pos_scale
                            elif chest_rel_R is not None:
                                # CHEST-RELATIVE. The operator's hand is read
                                # relative to their CHEST and re-expressed in
                                # the robot's chest frame, so:
                                #   * their own torso bend leaves the arm
                                #     target alone and only drives the waist.
                                #     waist-abs counts it TWICE once the torso
                                #     is driven - once inside wrist-rel-waist,
                                #     once through the bind point following the
                                #     live chest, which moves 194 mm at 25 deg
                                #     of bend and 307 mm at 40.
                                #   * the orientation demand rotates WITH the
                                #     chest, so the torso home stops mattering
                                #     to the arms. Under waist-abs the demand
                                #     is fixed in the base frame, which is why
                                #     going upright breaks the arms in a way no
                                #     amount of --pos-scale / --waist-bind-up
                                #     can reach: those translate the target,
                                #     and the failure is on the orientation
                                #     side (position alone 1.6 mm, orientation
                                #     alone 0.0 deg, together 272 mm).
                                _cR = chest_rel_R[0]          # robot chest R
                                _cP = chest_rel_R[1]          # robot chest pos
                                _hO, _hR = chest_rel_R[2], chest_rel_R[3]
                                p_abs = _cP + _cR @ (_hR.T @ (p_now - _hO)) \
                                    * args_cli.pos_scale
                            if sym_lock is not None:
                                p_abs = sym_lock[h][0]
                            if rel_lock_p is not None and h in rel_lock_p:
                                p_abs = rel_lock_p[h]
                                rel_stat["n"] += 1
                            a = min(1.0, (t - arm.engage_t) / 1.5)
                            blend = (1.0 - a) * arm.ee_p0[0].cpu().numpy() + a * p_abs
                            pos_des = torch.as_tensor(
                                blend, dtype=torch.float32,
                                device=device).unsqueeze(0)
                        else:
                            # translation: scaled, clamped delta in the yaw-comp
                            # frame (x fwd / y left / z up == base convention)
                            dp = (p_now - arm.pico_p0) * args_cli.pos_scale
                            n = np.linalg.norm(dp)
                            if n > args_cli.pos_max:
                                dp *= args_cli.pos_max / n
                            pos_des = arm.ee_p0 + torch.as_tensor(
                                dp, dtype=torch.float32, device=device).unsqueeze(0)
                        if args_cli.ori_mode == "palm":
                            # absolute: the operator's palm pose IS the robot's
                            # palm pose. No engage anchor, so nothing to drift
                            # and re-anchoring never rotates the hand.
                            M4 = np.eye(4)
                            M4[:3, :3] = (sym_lock[h][1] if sym_lock is not None
                                          else T[:3, :3] @ PALM_FIX[h])
                            if (chest_rel_R is not None and sym_lock is None
                                    and args_cli.map == "chest-rel"):
                                # chest-anchor deliberately does NOT do this:
                                # rotating the palm demand by the robot's chest
                                # is what tilted it down 55 deg.
                                M4[:3, :3] = (chest_rel_R[0] @ chest_rel_R[3].T
                                              @ M4[:3, :3])
                            quat_des = torch.as_tensor(
                                mat_to_pos_quat_wxyz(M4)[1], dtype=torch.float32,
                                device=device).unsqueeze(0)
                        elif args_cli.ori_mode == "hold":
                            # orientation channel not trusted (live wrists
                            # spiked 170deg/step): keep the engage orientation,
                            # track position only
                            quat_des = arm.ee_q0
                        else:
                            # rotation: world-frame delta, angle-clamped,
                            # pre-multiplied
                            q_now = torch.as_tensor(q_now_np, dtype=torch.float32,
                                                    device=device).unsqueeze(0)
                            rel = quat_mul(q_now, arm.pico_q0_inv)
                            aa = axis_angle_from_quat(rel)
                            ang = aa.norm(dim=-1, keepdim=True)
                            if ang.item() > wrist_max_rad:
                                aa = aa / ang * wrist_max_rad
                                ang = torch.full_like(ang, wrist_max_rad)
                            rel = quat_from_angle_axis(ang.squeeze(-1),
                                                       aa / ang.clamp(min=1e-9))
                            quat_des = quat_mul(rel, arm.ee_q0)
                        if arm.pos_filter is not None:
                            # one-euro on the final target: kills controller
                            # jitter at rest, opens up with speed (low lag)
                            pos_des = torch.as_tensor(
                                arm.pos_filter(pos_des[0].cpu().numpy(), samp_dt),
                                dtype=torch.float32, device=device).unsqueeze(0)
                            if args_cli.ori_mode != "hold":
                                quat_des = torch.as_tensor(
                                    arm.quat_filter(quat_des[0].cpu().numpy(),
                                                    samp_dt),
                                    dtype=torch.float32, device=device).unsqueeze(0)
                        _oref = (arm.home_ee[1] if args_cli.ori_ref == "home"
                                 and arm.home_ee is not None else arm.ee_q0)
                        if args_cli.ori_scale < 1.0 and _oref is not None:
                            # 朝向缩放:把「相对啮合时锚点的转动」按比例缩小。
                            # 顶限位的是腕部 j6(±80°)与 j7(−79 至 64°),而
                            # 人的前臂滚转有 150 至 180° —— 1:1 映射必然把腕
                            # 推出量程,顶死之后朝向就一直是错的(实测 14.15%
                            # 的帧至少一个手臂关节贴在限位 1° 以内,最长连续
                            # 5.2 秒)。位置早就有 --pos-scale,朝向一直没有。
                            _rel = quat_mul(quat_des, quat_inv(_oref))
                            _aa = axis_angle_from_quat(_rel)
                            quat_des = quat_mul(
                                quat_from_angle_axis(
                                    _aa.norm(dim=-1) * args_cli.ori_scale,
                                    _aa / _aa.norm(dim=-1, keepdim=True).clamp(min=1e-9)),
                                _oref)
                            ori_scale_stat["n"] += 1
                        arm.set_pose_command(pos_des, quat_des)
                        if arm.pico_el0 is not None or (
                                args_cli.map == "waist-abs" and elbow_body_idx):
                            el_raw = trk_use.get(elbow_serial[h])
                            if el_raw is not None:
                                T_el = process_xr_pose(
                                    np.asarray(el_raw, dtype=np.float64), ref_use)
                                p_el_now, _ = mat_to_pos_quat_wxyz(T_el)
                                if args_cli.map == "waist-abs":
                                    if yaw_fix_cs is not None:
                                        cy, sy = yaw_fix_cs
                                        p_el_now = np.array(
                                            [cy * p_el_now[0] - sy * p_el_now[1],
                                             sy * p_el_now[0] + cy * p_el_now[1],
                                             p_el_now[2]])
                                    arm.el_tgt = (waist_bind_base
                                                  + p_el_now * args_cli.pos_scale)
                                else:
                                    arm.el_tgt = arm.el_p0 + (
                                        p_el_now - arm.pico_el0) * args_cli.pos_scale

                    _t0 = _tick("arms", _t0)
                    if torso is not None and (mode_waist
                                              or args_cli.map == "anchored"):
                        # torso rides the arm clutch: any arm engaged = follow.
                        # waist-abs normally locks the torso - the waist IS the
                        # fixed reference the mapping is built on - but
                        # --control-mode arms+head+waist is the explicit
                        # request to move it anyway, and the bind point is
                        # recomputed from the live chest to match.
                        any_eng = any(a.engaged for a in arms.values())
                        pose7 = torso.select(trk_use) if any_eng else None
                        if any_eng and pose7 is not None:
                            if not torso.engaged:
                                torso.engage(pose7)
                                print(f"\n[clutch] torso engaged "
                                      f"(tracker {torso.serial_seen})")
                            torso.update(pose7, targets, control_dt)
                        if not any_eng:
                            torso.engaged = False

                    # head: not clutched - see HeadChannel. Needs the waist
                    # pose for the heading reference and the live chest.
                    if head_ch is not None and frame.head is not None:
                        w7 = torso.select(trk_use)
                        if w7 is not None:
                            hcp, hcq = ee_pose_in_base_of(chest_body_idx)
                            hq = hcq[0].cpu().numpy()
                            R_chest = _Rot.from_quat(
                                [hq[1], hq[2], hq[3], hq[0]]).as_matrix()
                            head_ch.update(frame.head, w7, R_chest,
                                           targets, samp_dt)

                _t0 = _tick("head", _t0)
                if pink_ik is not None:
                    # open-loop: Pink's own configuration is the source of truth
                    # and evolves continuously (Isaac's stiff PD follows it).
                    # Reseeding from Isaac's actual - possibly wound - state each
                    # step defeats the posture task's anti-winding, so we don't.
                    # Targets are fed relative to arm_center (live chest pose) so
                    # torso lean doesn't perturb the arm solve.
                    cp, cq = ee_pose_in_base_of(chest_body_idx)
                    cp, cq = cp[0].cpu().numpy(), cq[0].cpu().numpy()
                    branch_on = False
                    for h in ("right", "left"):
                        arm = arms.get(h)
                        if arm is not None and arm.has_command:
                            pink_ik.set_target_chest(
                                h, cp, cq, arm.cmd_pos[0].cpu().numpy(),
                                arm.cmd_quat[0].cpu().numpy())
                            if arm.el_tgt is not None:
                                pink_ik.set_elbow_target_chest(h, cp, cq,
                                                               arm.el_tgt)
                            # 一次性选支:只在离合闭合后的第一帧做,之后完全
                            # 不碰零空间。分支本来就只在啮合那一刻被定死,而
                            # 逐步线搜索实测会和腕任务打架(见 --null-bias)。
                            # 选支窗口:啮合之后的一小段时间里每步都跑线搜索,
                            # 让分支逐步挪过去,期间腕任务照常跟踪。一步跳过去
                            # 会在每次啮合后留一个瞬态,实测腕误差峰值由 120mm
                            # 涨到 456mm、均值涨 15mm —— 抖动是压下去了,代价
                            # 换到了跟踪上。窗口结束后完全不再碰零空间。
                            if args_cli.branch_at_engage:
                                if getattr(arm, "want_branch", False):
                                    arm.want_branch = False
                                    arm.branch_left = int(
                                        args_cli.branch_window
                                        * args_cli.control_hz)
                                    arm.branch_ask = _operator_swivel(
                                        h, frame, trk_use, wrist_serial[h],
                                        trk_use.get(args_cli.tracker_waist))
                                    if arm.branch_ask is None:
                                        arm.branch_left = 0
                                        branch_stat["miss"] += 1
                                        print(f"[branch] {h}: 取不到操作者"
                                              "回转角,本次不选支")
                                    else:
                                        branch_stat["n"] += 1
                                if getattr(arm, "branch_left", 0) > 0:
                                    arm.branch_left -= 1
                                    pink_ik.set_swivel_target(h, arm.branch_ask)
                                    branch_on = True
                                    if arm.branch_left == 0:
                                        pink_ik.refresh_fk()
                                        _g = pink_ik._robot_swivel(h)
                                        print(f"[branch] {h}: 选支窗口结束,"
                                              f"要求 "
                                              f"{np.degrees(arm.branch_ask):+.1f}°"
                                              f" 实得 "
                                              f"{np.degrees(_g):+.1f}°"
                                              if _g is not None else
                                              f"[branch] {h}: 选支窗口结束")
                                else:
                                    pink_ik.set_swivel_target(h, None)
                            # 把操作者自己的回转角交给解算器,钉住那一维冗余。
                            # 不开时(--null-bias 0,默认)这里整段不执行,
                            # 解算器行为与之前逐位相同。
                            if args_cli.null_bias > 0.0:
                                _a = _operator_swivel(
                                    h, frame, trk_use, wrist_serial[h],
                                    trk_use.get(args_cli.tracker_waist))
                                pink_ik.set_swivel_target(h, _a)
                                swivel_stat["n" if _a is not None else "miss"] += 1
                        else:
                            pink_ik.hold_target(h)
                            if args_cli.null_bias > 0.0:
                                pink_ik.set_swivel_target(h, None)
                    if args_cli.branch_at_engage:
                        # 线搜索是解算器的全局开关,只有当至少一条臂还在选支
                        # 窗口里时才打开;另一条臂的目标已置 None,不受影响。
                        pink_ik.null_bias = (args_cli.branch_strength
                                             if branch_on else args_cli.null_bias)
                    pink_ik.solve()
                    if os.environ.get("VEGA_PINK_DEBUG") and step % 240 == 0:
                        # split the loss: pink task err (target vs pink's own
                        # FK) vs PD err (pink q vs Isaac q, worst arm joint)
                        pink_ik.refresh_fk()
                        for h in ("right", "left"):
                            tgt_w = getattr(pink_ik, "last_target", {}).get(h)
                            if tgt_w is None:
                                continue
                            perr = np.linalg.norm(
                                pink_ik.ee_pos(h) - tgt_w.translation) * 1000
                            qmap = pink_ik.q_map_to(isaac_names)
                            worst = max(qmap, key=lambda j: abs(
                                targets[0, j].item()
                                - robot.data.joint_pos[0, j].item()))
                            qerr = abs(targets[0, worst].item()
                                       - robot.data.joint_pos[0, worst].item())
                            tw = tgt_w.translation
                            pe = pink_ik.ee_pos(h)
                            qd = robot.data.joint_vel[0, worst].item()
                            tau = robot.data.applied_torque[0, worst].item()
                            print(f"\n[pinkdbg] t={t:5.1f} {h}: "
                                  f"pink_task_err={perr:6.1f}mm "
                                  f"tgtW=({tw[0]:+.3f},{tw[1]:+.3f},{tw[2]:+.3f}) "
                                  f"eeW=({pe[0]:+.3f},{pe[1]:+.3f},{pe[2]:+.3f}) | "
                                  f"worst {isaac_names[worst]} tgt="
                                  f"{targets[0, worst].item():+.2f} isaac="
                                  f"{robot.data.joint_pos[0, worst].item():+.2f}"
                                  f" err={qerr:.3f}rad"
                                  f" qd={qd:+.2f} tau={tau:+.1f}")
                    for j, val in pink_ik.q_map_to(isaac_names).items():
                        targets[0, j] = val
                else:
                    for arm in arms.values():
                        arm.compute(targets)
                if joint_filt["cut"] > 0.0:
                    if joint_filt["f"] is None:
                        joint_filt["f"] = OneEuroFilter(joint_filt["cut"], 0.5)
                    v = targets[0, smooth_ids].cpu().numpy()
                    v = joint_filt["f"](v, samp_dt)
                    targets[0, smooth_ids] = torch.as_tensor(
                        v, dtype=torch.float32, device=device)
                if joint_guard is not None:
                    # The exit layer (V2AP port): velocity clamp + collision
                    # hold on the FINAL command, so Isaac, the recorder, the
                    # publisher and the hardware bridge all see the same
                    # guarded numbers.
                    _cmd = {n: float(targets[0, i].item())
                            for i, n in enumerate(isaac_names)
                            if n in JOINT_DUMP}
                    _out = joint_guard.apply(_cmd)
                    for _n, _v in _out.items():
                        if _v != _cmd[_n]:
                            targets[0, isaac_names.index(_n)] = _v
                robot.set_joint_position_target(targets)

                if jc_w is not None:
                    # Joint angles, because joints are what gets sent to the
                    # robot - matching pictures is not the same claim. Both the
                    # COMMAND (what the mapping decided) and the ACHIEVED (what
                    # the physics did), so a mismatch against the Viser bench
                    # can be attributed to the mapping or to the PD, not just
                    # noticed. The bench has no physics, so only `cmd` is
                    # comparable to it.
                    jc_w.writerow([f"{t:.4f}"]
                                  + [f"{targets[0, isaac_names.index(n)].item():.6f}"
                                     for n in JOINT_DUMP if n in isaac_names]
                                  + [f"{q_full_prev[isaac_names.index(n)].item():.6f}"
                                     if q_full_prev is not None else ""
                                     for n in JOINT_DUMP if n in isaac_names]
                                  + [1 if branch_on else 0])

                q_full = robot.data.joint_pos[0]
                q_full_prev = q_full
                state_n["i"] += 1
                if state_pub is not None and state_n["i"] % 2 == 0:
                    # what Isaac ACTUALLY has, not what we commanded: a viewer
                    # showing the command would still hide a PD that is not
                    # keeping up.
                    # ...and the TARGET alongside it, so the mirror can show
                    # "did the hand get there" without recomputing the mapping.
                    # That readout is the whole reason parameters were once
                    # chosen on a bench that was 272 mm from its own target and
                    # never said so - but a mirror that derived the target
                    # itself would be a second implementation again, which is
                    # the thing this channel exists to avoid.
                    _tgt = {}
                    for _h, _arm in arms.items():
                        if _arm.engaged and _arm.cmd_pos is not None:
                            _ee, _ = _arm.ee_pose_in_base()
                            _tgt[_h] = {
                                "cmd": [float(v) for v in
                                        _arm.cmd_pos[0].tolist()],
                                "ee": [float(v) for v in _ee[0].tolist()],
                                "quat": [float(v) for v in
                                         _arm.cmd_quat[0].tolist()]}
                    try:
                        state_pub.send(_mp.packb(
                            {"t_us": int(t * 1e6),
                             # 主机墙上时钟。t_us 是**仿真时间**,而仿真时间与
                             # 墙上时间的比值是实时倍率 —— 实测在 0.12 到 3.26
                             # 倍之间,逐档逐次都在变。拿 t_us 去和手套、触觉、
                             # 相机那几路(它们打的都是墙钟)对齐,必然是错的。
                             # 这一路要进统一录制,就得有一个同口径的戳。
                             "t_wall_us": int(time.time() * 1e6),
                             # what Isaac ACTUALLY has
                             "q": {n: float(v) for n, v in
                                   zip(isaac_names, q_full.cpu().numpy())},
                             # what was COMMANDED - this is what a real robot
                             # should be driven with. Isaac's achieved state is
                             # a simulated PD's opinion of it; the hardware has
                             # its own controller and wants the target.
                             "cmd": {n: float(targets[0, i].item())
                                     for i, n in enumerate(isaac_names)
                                     if n in JOINT_DUMP},
                             "engaged": [h for h, a in arms.items() if a.engaged],
                             # the bridge needs this: in arms / arms+head the
                             # torso is not part of the mapping at all and the
                             # operator sets it by hand, so the bridge must not
                             # read it, initialise it, or command it. Only
                             # arms+head+waist owns the torso.
                             "mode": args_cli.control_mode,
                             "arm": _tgt},
                            use_bin_type=True), _zmq.NOBLOCK)
                    except Exception:
                        pass
                q_all["lo"] = q_full.clone() if q_all["lo"] is None \
                    else torch.minimum(q_all["lo"], q_full)
                q_all["hi"] = q_full.clone() if q_all["hi"] is None \
                    else torch.maximum(q_all["hi"], q_full)
                q_now = robot.data.joint_pos[0, all_arm_ids_t]
                q_seen["lo"] = q_now.clone() if q_seen["lo"] is None \
                    else torch.minimum(q_seen["lo"], q_now)
                q_seen["hi"] = q_now.clone() if q_seen["hi"] is None \
                    else torch.maximum(q_seen["hi"], q_now)
                if wheel_ids:
                    q_wh = robot.data.joint_pos[0, wheel_ids_t]
                    if wheel_prev is not None and step > wheel_warmup:
                        wheel_step = max(wheel_step, (q_wh - wheel_prev).abs().max().item())
                    wheel_prev = q_wh.clone()
                for h, arm in arms.items():
                    if arm.engaged and arm.cmd_pos is not None:
                        ee_p, _ = arm.ee_pose_in_base()
                        e = (ee_p - arm.cmd_pos).norm().item()
                        st = err_stats[h]
                        st[0] = max(st[0], e)
                        st[1] += e
                        st[2] += 1
                        if trace_w is not None:
                            hp = arm.human_p if arm.human_p is not None \
                                else (float("nan"),) * 3
                            cp_ = arm.cmd_pos[0].tolist()
                            ep_ = ee_p[0].tolist()
                            trace_w.writerow(
                                [f"{t:.4f}", h] + [f"{v:.5f}" for v in
                                                   (*hp, *cp_, *ep_)]
                                + [f"{e * 1000:.3f}"])
                if demo_checks and t >= demo_checks[0][0]:
                    _, label, exp, exp_ori = demo_checks.pop(0)
                    for h, arm in arms.items():
                        if arm.ee_p0 is None:
                            continue
                        ee_p, ee_q = arm.ee_pose_in_base()
                        got = (ee_p - arm.ee_p0)[0].tolist()
                        want = exp[h]
                        dev = float(np.linalg.norm(
                            np.asarray(got) - np.asarray(want))) * 1000
                        # stage attribution: what the mapping COMMANDED vs
                        # what the robot did (isolates mapping vs IK/PD)
                        cmd_d = (arm.cmd_pos - arm.ee_p0)[0].tolist() \
                            if arm.cmd_pos is not None else [0.0] * 3
                        ori_txt = ""
                        if exp_ori is not None:
                            axis, deg = exp_ori[h]
                            axis_t = torch.tensor([axis], dtype=torch.float32,
                                                  device=device)
                            rel_exp = quat_from_angle_axis(
                                torch.tensor([np.deg2rad(deg)],
                                             dtype=torch.float32,
                                             device=device), axis_t)
                            q_exp = quat_mul(rel_exp, arm.ee_q0)
                            ang = axis_angle_from_quat(
                                quat_mul(ee_q, quat_inv(q_exp))).norm().item()
                            ori_txt = f" ori_dev {np.rad2deg(ang):5.1f}deg"
                            dev = max(dev, np.rad2deg(ang) * 2.0)  # deg->score
                        demo_results.append((label, h, got, want, dev))
                        print(f"\n[demo] {label:>14s} {h:>5s}: "
                              f"got Δ=({got[0]:+.3f},{got[1]:+.3f},{got[2]:+.3f}) "
                              f"cmd Δ=({cmd_d[0]:+.3f},{cmd_d[1]:+.3f},{cmd_d[2]:+.3f}) "
                              f"want Δ=({want[0]:+.2f},{want[1]:+.2f},{want[2]:+.2f}) "
                              f"dev {dev:5.1f}mm{ori_txt}")

            _t0 = _tick("pink", _t0)
            pin_wheels()          # keep the parked base straight every step
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            _t0 = _tick("physics", _t0)
            t += sim_dt
            run_clock["sim"] = t
            step += 1

            if snap_cam is not None and snap_times and t >= snap_times[0]:
                save_snap(snap_times.pop(0))

            if args_cli.debug_joints and step % round(2.0 / sim_dt) == 0:
                q = robot.data.joint_pos[0]
                parts = []
                for h, arm in arms.items():
                    ee_p, _ = arm.ee_pose_in_base()
                    err = "-" if arm.cmd_pos is None else \
                        f"{(ee_p - arm.cmd_pos).norm().item() * 1000:.0f}mm"
                    js = " ".join(f"{q[j].item():+.2f}" for j in arm.joint_ids)
                    parts.append(f"{h}[{'E' if arm.engaged else '.'}] q=({js}) track_err={err}")
                print(f"\n[dbg] t={t:5.2f} " + " | ".join(parts))

            if time.time() - last_print > 2.0:
                wall = time.time() - run_clock["wall"]
                rtf = t / wall if wall > 0 else 0.0
                age = f" age={lat_stats[0] / lat_stats[2]:3.0f}ms" if lat_stats[2] else ""
                eng = [h for h, a in arms.items() if a.engaged]
                if torso is not None and torso.engaged:
                    eng.append("torso")
                print(f"\r[stats] sim t={t:6.1f}s rtf={rtf:4.2f}x{age} "
                      f"frames={n_frames} engaged={eng}",
                      end="", flush=True)
                last_print = time.time()
            if args_cli.duration > 0 and t >= args_cli.duration:
                print(f"\n[done] duration {args_cli.duration}s reached")
                break
    except KeyboardInterrupt:
        pass
    finally:
        print_summary()
        if trace_f is not None:
            trace_f.close()
        if jc_f is not None:
            jc_f.close()
            print(f"[trace] wrote {args_cli.trace_csv}")
        source.stop()
        # simulation_app.close() is known to hang on this stack (see
        # teleop_vega_sharpa.py); hard-exit if shutdown takes > 20 s.
        import threading

        killer = threading.Timer(20.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        simulation_app.close()


if __name__ == "__main__":
    main()
