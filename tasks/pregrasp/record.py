"""checkpoint -> PreGrasp 抓取回放录像 (mp4). 验收用: 亲眼看抓姿与验证抬升.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.record \
      --checkpoint logs/PreGrasp_0/<run>/stage1_nn/last.pth --headless

默认录 ep_total(153) 步 ≈ 连续 3 个回合 (成功即重置, 会看到跳变, 属正常);
--fps 10 半速慢放看细节. c0=0 正式口径, gentle=1.0.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="Grasp2")
parser.add_argument("--grasp_prior", type=str, default="", help="prior npz 路径 (B 组)")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="钉死的物体 yaw (度) = screen_prior Gate1 的 best_yaw")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--c0_max", type=float, default=0.0,
                    help="复位合拢随机上限: 0=正式口径 / 0.9=训练分布 (看失败模式用)")
parser.add_argument("--out_dir", type=str, default=None, help="默认 <run_dir>/videos")
parser.add_argument("--eye", type=str, default="0.55,-0.85,1.45")
parser.add_argument("--lookat", type=str, default="-0.10,-0.08,0.92")
parser.add_argument("--fps", type=int, default=0, help="0=实时(20fps); 10=半速慢放")
parser.add_argument("--steps", type=int, default=0, help="0=ep_total (≈3 回合)")
parser.add_argument("--approach", action="store_true",
                    help="端到端口径: 从人手轨迹起点 q_ref[0] 出发, 接近+抓取全程. "
                         "配 --stance_prefix K 时起点是默认站姿 (PLAN D2)")
parser.add_argument("--stance_prefix", type=int, default=0,
                    help="必须与训练时相同 (0=无前缀)")
parser.add_argument("--orient_blend", action="store_true",
                    help="必须与训练时相同 (改 obs 的参考通道)")
parser.add_argument("--approach_only", action="store_true",
                    help="只学接近的 run (7 维动作). 必须与训练时同值, 否则场景/动作维度对不上")
parser.add_argument("--minimal", action="store_true",
                    help="最简接近 run (--minimal 训的). 起点固定站姿 / 无前馈 / 公差 1cm")
parser.add_argument("--place", action="store_true",
                    help="必须与训练时相同 (成功判据=放置达标)")
parser.add_argument("--bimanual", action="store_true",
                    help="双臂 run (--bimanual 训的). 需配 --prior_b")
parser.add_argument("--prior_b", type=str, default="", help="B 手(左) GraspPose npz")
parser.add_argument("--prior_b_yaw", type=float, default=-1.0)
parser.add_argument("--curobo_ref", type=str, default="",
                    help="cuRobo 前馈参考 npz, 必须与训练相同")
parser.add_argument("--ff_freeze_cm", type=float, default=5.0,
                    help="必须与训练相同 (近端冻结前馈阈值)")
parser.add_argument("--ff_pull", type=float, default=0.08,
                    help="必须与训练相同 (弱回拉锚定强度)")
parser.add_argument("--l5", action="store_true",
                    help="L5 run (冻结区合指耦合)。录像全程从站姿跑, 不用近点直起")
parser.add_argument("--dyn_far", type=float, default=-1.0,
                    help="远场残差上限倍数覆盖, 必须与训练一致否则回放动作幅度失真 (-1=cfg 默认)")
parser.add_argument("--pregrasp29", action="store_true",
                    help="PG 系任务口径: 靶点=掌心PreGrasp, 29维/侧 (配 RL_HAND_JOINTS=1)")
parser.add_argument("--pregrasp_grasp", action="store_true",
                    help="PGA 回放口径: pregrasp29 靶点但走完整抓取链 (无 approach_only)")
parser.add_argument("--pour_carry", action="store_true",
                    help="CARRY 回放口径 (与 train --pour_carry 一致)")
parser.add_argument("--carry_npz", type=str, default="tasks/pour/carry_pour17.npz")
parser.add_argument("--phase2", action="store_true",
                    help="PG2: 闩后 RL 自走到 GraspPose 的回放口径")
parser.add_argument("--bi_native", action="store_true",
                    help="v2 原生双臂底座回放 (与 train --bi_native 一致; FC 系必须, "
                         "否则落回 v1 退役底座)")
parser.add_argument("--fin_cart", action="store_true",
                    help="FC 回放口径: 逐指笛卡尔目标 (与 train --fin_cart 一致; "
                         "只建奖励侧缓冲, 不动观测宽度)")
parser.add_argument("--fin_pose1", action="store_true",
                    help="FC-C 回放口径: reset 指型 = Pose1 (与 train --fin_pose1 一致)")
parser.add_argument("--carry_squeeze", action="store_true",
                    help="CARRY4 回放口径: q_close ← squeeze (与 train --carry_squeeze 一致)")
parser.add_argument("--carry_progress", action="store_true",
                    help="CARRY3 回放口径: 进度时钟 (与 train --carry_progress 一致; "
                         "缺了它参考时钟外生走表, C3 策略观测分布错位)")
parser.add_argument("--fcd", action="store_true",
                    help="FC-D 回放口径: 四段指参考 (fin_ref_track)。⚠ 行为承重: "
                         "训练时指前馈=四段参考, 缺了它回放骑在合拢模板上, 残差全错位")
parser.add_argument("--pour_succ", action="store_true",
                    help="Pour 回放口径: 倒水里程碑判据 (B6 改判后为里程碑非终点)")
parser.add_argument("--table_fail", action="store_true",
                    help="撞桌即Fail 口径 (与 train --table_fail/--fcd 一致)")
parser.add_argument("--carry_stable", action="store_true",
                    help="C5S/PourS 回放口径: 稳抓链 (向心塑形+启动前 candidate)")
parser.add_argument("--pour_e2e", action="store_true",
                    help="E2E 回放口径: 站姿起步+四段指参考+缝行时钟 (与 train --pour_e2e 一致)")
parser.add_argument("--pour_free", action="store_true",
                    help="PourS-v3 回放口径 (行为承重): 自由探索段 [93,140) 时钟冻结/"
                         "跟丢判挂起, 缺了它回放在自由段被跟丢判杀掉")
parser.add_argument("--aag", action="store_true",
                    help="AAG 回放口径 (行为承重): 指参考=标准成功轨迹逐行 (同 --curobo_ref 文件)")
parser.add_argument("--grasp_cent_fix", action="store_true",
                    help="cent_fix 回放口径 (行为承重: candidate 门控影响相位切换)")
parser.add_argument("--arm_ff_gate", action="store_true",
                    help="AAG-v3 回放口径 (行为承重): 到位前臂残差归零=纯前馈")
parser.add_argument("--c0_min", type=float, default=-1.0,
                    help="起步合拢度**固定**为该值 (0.9=每回合都握着物体起步) —— record 默认 c0 随机, 抽到张开手瓶子秒脱手, 录不到成功回合")
parser.add_argument("--ff_play", action="store_true",
                    help="v8 回放口径 (★行为承重): 自由段续播臂前馈 —— 缺了它回放"
                         "时手臂在倒水段被冻结, 策略根本倒不了水")
parser.add_argument("--pour_trend", type=str, default="",
                    help="v7 回放口径 (行为承重: 趋势钟影响自由段收入结构)")
parser.add_argument("--eps_pos_cm", type=float, default=-1.0,
                    help="v3.2+ 回放口径: 到位门 eps 覆盖 (cm), minimal 落定后生效")
parser.add_argument("--fin_gate", action="store_true",
                    help="回放口径 (行为承重): 手指行进门, 到位前指参考钳弯根行")
parser.add_argument("--obj_obs", action="store_true",
                    help="回放口径: 物体特权进 actor (+10/侧, obs 变维)")
parser.add_argument("--step_rew_log", type=str, default="",
                    help="回放逐步奖惩账本目录 (与训练 --step_rew_log 同规格)")
parser.add_argument("--fin_open_from_ref", action="store_true",
                    help="每侧 q_open ← 指参考首行 (出生即绷直; 需 --aag 装载指参考)")
parser.add_argument("--pour_choreo", action="store_true",
                    help="编舞播放器: 完整参考纯脚本走完(站姿→cuRobo→抓→交互→归位→"
                         "倒放cuRobo松手撤退), 门警全免; 配 --zero_res + e2e_full npz")
parser.add_argument("--zero_res", action="store_true",
                    help="0残差播放: 不加载策略, 动作恒零 (纯前馈+脚本段; "
                         "--checkpoint 传 none 即可)")
parser.add_argument("--pour_retreat", action="store_true",
                    help="E2E第五段撤退 (与 train --pour_retreat 一致)")
parser.add_argument("--pour_place", action="store_true",
                    help="v10 回放口径 (行为承重: 松手斜坡+★不终止, 与 train --pour_place 一致)")
parser.add_argument("--pour_diag", type=str, default="",
                    help="倒水诊断: 逐步逐env记 e_pos(双侧)/c_t/pf_on/★/done 存npz (无视频)")
parser.add_argument("--dump", type=str, default="",
                    help="非空=不渲染视频, 逐步全量状态转储到该 npz (双侧臂/指/腕/"
                         "物体/垫力/相位, 前4 env) —— 比录像快一个量级")
parser.add_argument("--fc_ref_end", action="store_true",
                    help="★行为承重(判据): FC 目标点用参考终态, 必须与训练同值")
parser.add_argument("--fc_squeeze", action="store_true",
                    help="★行为承重(判据): FC 目标点摆 squeeze, 必须与训练同值")
parser.add_argument("--seg_gate", action="store_true",
                    help="★行为承重: 分段探索门控(按段×关节组缩放残差步长), 必须与训练同值")
parser.add_argument("--fin_gate_dpos_cm", type=float, default=0.0,
                    help="★行为承重: 指参考软闸(腕距<该值即放行), 必须与训练同值")
parser.add_argument("--fin_dev_scale", type=float, default=-1.0,
                    help="★行为承重: 指残差累计上限倍数, 必须与训练同值 (默认40=无界)")
parser.add_argument("--arm_abs", action="store_true",
                    help="回放口径: 方案C 臂绝对参考+有界残差 (必须与训练同值)")
parser.add_argument("--arm_abs_dev_deg", type=float, default=2.86)
parser.add_argument("--cand_diag", action="store_true",
                    help="candidate 逐门统计: 不渲染, 全 env 逐步累计 6 个与门的通过率, "
                         "退出时打表。回答'candidate 恒零到底卡在哪一道门'")
parser.add_argument("--pours_v6", action="store_true",
                    help="PourS-v6 回放口径 (行为承重: 阶段纯净化判据全套+obs+10/侧)")
parser.add_argument("--pours_v5", action="store_true",
                    help="PourS-v5 回放口径 (行为承重: obs+8/侧, slip 死线 8cm/60°, "
                         "thrown 28cm —— 缺了回放在自由段被旧死线杀掉且 obs 维度崩)")
parser.add_argument("--grip_mu", type=float, default=-1.0,
                    help=">0 时把指垫 SuperGrip μ 钉在该值 (摩擦课程退火态不在 ckpt 里; "
                         "回放中期 ckpt 先 grep 训练日志 '摩擦退火: μ→' 取当时值)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True                     # 离屏渲染必需

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("record")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

ckpt = os.path.abspath(args.checkpoint) if args.checkpoint not in ("", "none") else ""
run_dir = os.path.dirname(os.path.dirname(ckpt))
out_dir = args.out_dir or os.path.join(run_dir, "videos")
tag = (os.path.splitext(os.path.basename(ckpt))[0]
       + (f"_c0max{args.c0_max:.1f}" if args.c0_max > 0 else ""))
os.makedirs(out_dir, exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.grasp_prior:
    if args.approach_only:
        # ⚠ 必须在 apply_grasp_prior **之前** —— 与 train.py 同因: 观测宽度与动作维度
        #   都在 apply_grasp_prior 里按 _obs_base(cfg) 算死。放后面会得到
        #   "观测宽度 139 != observation_space 151" (2026-08-16 实测踩到)。
        env_cfg.approach_only = True
        env_cfg.action_space = 7
    if getattr(args, "pregrasp29", False):
        # PG1/PG2 回放 (2026-08-18晚): approach_only 语义 + 29 维(RL_HAND_JOINTS=1),
        # 必须在 apply_grasp_prior 之前 (观测宽度按 pregrasp29 分支算)
        env_cfg.approach_only = True
        env_cfg.pregrasp29 = True
        if getattr(args, "phase2", False):
            env_cfg.pregrasp_phase2 = True
        if getattr(args, "fin_cart", False):
            env_cfg.fin_cart = True
        if getattr(args, "fin_pose1", False):
            env_cfg.fin_start_pose1 = True
        if getattr(args, "fcd", False):
            # FC-D: 指前馈=四段参考 (行为承重), 撞桌口径与 train --fcd 同捆
            env_cfg.fin_ref_track = True
            env_cfg.table_touch_fail = True
        if getattr(args, "aag", False):
            env_cfg.fin_ref_npz = os.path.abspath(args.curobo_ref)
            if getattr(args, "fin_open_from_ref", False):
                env_cfg.fin_open_from_ref = True
            env_cfg.approach_extra0 = 400
            env_cfg.near_exempt_m = 0.07   # 与 train --aag 同捆 (行为承重: 圈外碰物判死)
        if getattr(args, "arm_ff_gate", False):
            env_cfg.arm_ff_gate = True     # AAG-v3 回放口径 (行为承重: 到位前臂纯前馈)
    # ⚠ v5/v6/trend 接线必须在**顶层** —— PourS 回放不带 --pregrasp29, 嵌在它
    #   块里会静默不生效而 obs+10 又生效 ⟹ 宽度错配 (2026-08-22 录像三连败根因)
    if getattr(args, "pours_v5", False):
        env_cfg.pours_v5 = True        # v5 回放口径 (与 train --pours_v5 同捆)
        env_cfg.carry_slip_pos = 0.08
        env_cfg.carry_slip_rot = 1.0472        # 60° (rad)
        env_cfg.max_obj_height = 0.28
    if getattr(args, "ff_play", False):
        env_cfg.pour_ff_play = True
    if getattr(args, "pours_v6", False):
        env_cfg.pours_v6 = True        # v6 回放口径 (行为承重全套, 与 train 同捆)
        env_cfg.pour_succ_tilt_deg = 93.0
    if getattr(args, "pour_trend", ""):
        env_cfg.pour_trend_npz = os.path.abspath(args.pour_trend)
    if getattr(args, "fin_gate", False):
        env_cfg.fin_prog_gate = True
    if getattr(args, "obj_obs", False):
        env_cfg.obj_in_actor = True
    if getattr(args, "pour_carry", False):
        # CARRY 回放 (2026-08-19): 与 train --pour_carry 同配置
        # 2026-08-25 v9S终评破案: 基础 episode_length_s=12s(360步) 在 approach
        # 前奏(~145步)后剩余时间塞不下 carry4a 103行窗 ⟹ 历代 record 评测对
        # 长窗参考结构性 0★ (v8D 47行窗恰好塞得下, 87.3%可测纯属窗短)。抬钟:
        env_cfg.episode_length_s = max(
            float(getattr(env_cfg, "episode_length_s", 12.0)), 40.0)
        env_cfg.approach_only = True
        env_cfg.pregrasp29 = True
        env_cfg.pregrasp_phase2 = True
        env_cfg.pour_carry = True
        env_cfg.carry_npz = os.path.abspath(args.carry_npz)
        if getattr(args, "carry_progress", False):
            env_cfg.carry_progress = True
            # ★ 2026-08-23: 训练侧 --carry_progress 会把回合预算设成 600 (进度制),
            # 这里漏了 ⟹ 回放回合被 230 掐断, 摄像机永远等不到倒水那一刻
            # (实测 env0 27 步一个回合, 0/51 成功, 而同批其他 env ★12 次)
            env_cfg.approach_only_steps = 600
        if getattr(args, "carry_squeeze", False):
            env_cfg.carry_squeeze = True
        if getattr(args, "pour_succ", False):
            env_cfg.pour_succ = True
        if getattr(args, "pour_free", False):
            env_cfg.pour_free = True
        if getattr(args, "pour_place", False):
            env_cfg.pour_place = True
            if getattr(args, "pour_retreat", False):
                env_cfg.pour_retreat = True
        if getattr(args, "pour_choreo", False):
            env_cfg.pour_choreo = True
            env_cfg.ref_start_is_stance = True
            import numpy as _np_c
            _zc = _np_c.load(os.path.abspath(args.carry_npz), allow_pickle=True)
            env_cfg.carry_ref_off = int(_zc["seam"]) if "seam" in _zc.files else 0
            # 预算全放大: 编舞要走完 763 行手参考 (approach 402 分支 gs+extra)
            env_cfg.approach_extra0 = 900
            env_cfg.approach_only_steps = 2200
            env_cfg.carry_start_deadline = 1200
            print(f"[choreo] record 口径: seam={env_cfg.carry_ref_off}")
        if getattr(args, "carry_stable", False):
            env_cfg.carry_pad_reward = True
            env_cfg.carry_stable = True
        if getattr(args, "grasp_cent_fix", False):
            env_cfg.grasp_centrip_thresh = -1.0
            env_cfg.cent_signed = True
        if getattr(args, "pour_e2e", False):
            # E2E 回放口径 (行为承重: 指前馈=四段参考, 时钟缝行, 站姿起步)
            env_cfg.pour_e2e = True
            # e2e ckpt 的 obs 基数按 approach_only=False 算 (train e2e_squeeze
            # 同款, 199 vs 154 差 45/侧 —— 2026-08-26 首录 state_dict 差 90 破案),
            # 必须在 apply_grasp_prior 之前落定
            env_cfg.approach_only = False
            # train e2e 块三件套同款 (双臂断言要 retract_start; 回放=正式口径全站姿)
            env_cfg.retract_start = True
            env_cfg.stance_prob = 1.0
            env_cfg.retract_ratio = 1.0
            env_cfg.fin_cart = True
            env_cfg.fin_ref_track = True
            env_cfg.table_touch_fail = True
            env_cfg.pregrasp_phase2 = True
            env_cfg.ref_start_is_stance = True
            env_cfg.success_nonterminal = True
            env_cfg.fail_tilt_deg = 60.0
            import numpy as _np_e
            _ze = _np_e.load(args.curobo_ref, allow_pickle=True)
            env_cfg.carry_ref_off = int(_ze["seam"]) if "seam" in _ze.files else 0
            env_cfg.carry_start_deadline = 300
            env_cfg.approach_only_steps = 760
        if getattr(args, "table_fail", False):
            env_cfg.table_touch_fail = True
        if float(getattr(args, "grip_mu", -1.0)) > 0.0:
            # 摩擦课程回放: μ 钉在训练日志当时值 (hi=lo ⟹ 常值, 不退火)
            env_cfg.friction_curriculum = True
            env_cfg.friction_hi = env_cfg.friction_lo = float(args.grip_mu)
        env_cfg.ref_start_is_stance = False
        env_cfg.fail_tilt_deg = 999.0
        env_cfg.approach_hit_obj_m = 0.0
        env_cfg.push_fail_dist = 10.0
        env_cfg.push_fail_dist_loose = 10.0
        # 进度制(carry_progress)预算 600, 与训练一致; 否则 230 会把回合掐断
        env_cfg.approach_only_steps = 600 if getattr(args, "carry_progress", False) else 230
        env_cfg.curobo_ref_stride = 1
        env_cfg.eps_pos = env_cfg.eps_pos0 = 0.001
        env_cfg.eps_rot = env_cfg.eps_rot0 = 0.02
    if getattr(args, "pregrasp_grasp", False):
        env_cfg.pregrasp29 = True     # PGA: 完整抓取链, 无 approach_only (obs=199)
        env_cfg.retract_start = True  # 与训练 --retract 一致 (双臂防雷断言也要求:
        #                               B 侧 q_ref 未重建, 人手首帧起步=左臂乱摆)
    apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    raise SystemExit("--approach 必须配 --grasp_prior (对齐势的终点来自 GraspPose)")
env_cfg.orient_blend = args.orient_blend   # 与 approach 无关: 参考通道常开, obs 必须对齐
if args.place:
    env_cfg.place_task = True
    env_cfg.episode_length_s = 20.0
    env_cfg.freeze_wrist = False   # 与训练一致: gs 后腕参考解冻 (搬运段)
if args.approach:
    env_cfg.direct_grasp_prob = 0.0   # 全部回合从接近段起步
    env_cfg.approach_t0_max = 0.0     # 从人手轨迹 t=0 (q_ref[0]) 出发
    env_cfg.stance_prefix_frames = args.stance_prefix
# ---- 接近段 run 的场景必须与训练**逐项一致**, 否则录的不是同一个任务 ----
#      (台账 §2.24: 启动横幅不一致的对照全部作废)
if args.minimal:
    env_cfg.minimal_no_ff = True
    # ⚠ 2026-08-17: minimal 默认已改成**保留远松近紧**(减速靠它)。这里若还写死
    #   fixed_res=True / dyn_arm_near=1.0, 录的策略会在 10cm/s 下跑 —— 而它是在
    #   3~30cm/s 下训的, 场景不一致 ⟹ 录出来必然失败(实测 0/1, 而训练评测是 100%)。
    env_cfg.minimal_fixed_res = False
    import numpy as _np
    env_cfg.eps_pos, env_cfg.eps_rot = 0.01, _np.radians(15.0)
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
    env_cfg.w_imit0_approach = 0.0
    env_cfg.w_imit_ramp = 0.0
if args.minimal:      # 与 train.py 同因: 上面的块会被覆盖, 这里最终落定
    env_cfg.stance_prob, env_cfg.retract_ratio = 1.0, 1.0
if getattr(args, "pour_carry", False):
    import numpy as _np_c4
    _zc4 = _np_c4.load(os.path.abspath(args.carry_npz), allow_pickle=True)
    if "free_lo" in _zc4.files:   # carry4: 自由段行号以 npz 为准 (时间轴不同)
        env_cfg.pour_free_lo = int(_zc4["free_lo"])
        env_cfg.pour_free_hi = int(_zc4["free_hi"])
        print(f"[carry4] 自由段行号从 npz 接管: [{env_cfg.pour_free_lo},"
              f"{env_cfg.pour_free_hi})")
        # 窗行数 103 (carry3 是 47): 时钟≤1行/步 ⟹ 预算同比放大, 否则全员 pf_trunc
        env_cfg.pour_free_budget = max(int(env_cfg.pour_free_budget),
                                       int((env_cfg.pour_free_hi
                                            - env_cfg.pour_free_lo) * 3.2))
        env_cfg.approach_only_steps = max(int(getattr(env_cfg, "approach_only_steps", 600)), 800) \
            + (450 if getattr(args, "pour_place", False) else 0)   # 尾段回正+松手 (train 同款)
        print(f"[carry4] 预算随窗放大: pf_budget={env_cfg.pour_free_budget} "
              f"approach_only_steps={env_cfg.approach_only_steps}")
if getattr(args, "pour_carry", False):
    # ★ 2026-08-23 修 (回放批量早夭破案): "arrive 永不触发"封印必须在 minimal 块
    # **之后**重新落定 —— train.py 把它放在最末尾(1255), record 原来放在 pour_carry
    # 分支里, 被 minimal 的 eps=1cm/15° 盖掉 ⟹ 基座到位成功复活, 单侧 succ_t 在
    # ~14 步终止 (kill dump 实名), 64env 里 45 个循环早夭, "5.4% 成功率"由此稀释
    env_cfg.eps_pos = env_cfg.eps_pos0 = 0.001
    env_cfg.eps_rot = env_cfg.eps_rot0 = 0.02
if float(getattr(args, "c0_min", -1.0)) >= 0.0:
    env_cfg.closure_init_min = float(args.c0_min)
    env_cfg.closure_init_max = max(float(args.c0_min), env_cfg.closure_init_max)
if float(getattr(args, "eps_pos_cm", -1.0)) > 0.0:
    # v3.2+ 回放口径 (行为承重: eps 决定相位切换) —— 须在 minimal 落定后覆盖
    env_cfg.eps_pos = env_cfg.eps_pos0 = float(args.eps_pos_cm) / 100.0
    env_cfg.direct_grasp_prob = 0.0
    print(f"[minimal] 录像场景: stance_prob={env_cfg.stance_prob} "
          f"eps={env_cfg.eps_pos*100:.2f}cm no_ff={env_cfg.minimal_no_ff}")
_EnvCls = GraspTaskEnv
if getattr(args, "pour_carry", False):
    from tasks.pour.carry_env import PourCarryEnv as _EnvCls  # noqa: N813
if args.l5:
    env_cfg.l5_couple = True          # 与训练一致: 冻结区解锁手指+c(d) 追踪
    env_cfg.direct_grasp_prob = 0.0   # 录像看全程, 不用近点直起
if args.dyn_far > 0:
    env_cfg.dyn_arm_far = float(args.dyn_far)
    print(f"[record] 远场残差上限覆盖: {env_cfg.dyn_arm_far}x (与训练对齐)")
if args.curobo_ref:
    # ⚠ 必须在 minimal 块把 no_ff 写成 True **之后**再改回 False (与训练一致)
    assert args.bimanual, "--curobo_ref 目前只接了双臂 env"
    env_cfg.curobo_ref_npz = os.path.abspath(args.curobo_ref)
    env_cfg.minimal_no_ff = False
    env_cfg.curobo_ff_freeze_cm = args.ff_freeze_cm
    env_cfg.curobo_ff_pull = args.ff_pull
    print(f"[record] cuRobo 前馈: {os.path.basename(args.curobo_ref)} "
          f"近端 {args.ff_freeze_cm:.0f}cm 冻结 | 回拉 {args.ff_pull:.2f}/步")
if args.bimanual:
    # 与 train.py:785 逐项一致: 构造前翻倍观测/动作 + 特权维度
    assert args.prior_b and os.path.exists(args.prior_b), f"缺 --prior_b {args.prior_b}"
    if getattr(args, "pour_carry", False):
        from tasks.pour.carry_env import PourCarryEnv as _EnvCls  # noqa: F811
    elif getattr(args, "bi_native", False):
        from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv as _EnvCls  # noqa: F811
    else:
        from tasks.pregrasp.bimanual_env import BimanualApproachEnv as _EnvCls  # noqa: F811
    env_cfg.prior_b_npz = os.path.abspath(args.prior_b)
    env_cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
    if getattr(args, "pours_v5", False):
        # v5 特权滑移块/侧 (apply_grasp_prior 之后); v6 +10 (多阶段两维)
        env_cfg.observation_space += 10 if getattr(args, "pours_v6", False) else 8
    if getattr(args, "obj_obs", False):
        env_cfg.observation_space += 13   # 物体特权块(13=位置3+姿态4+线速3+角速3)/侧
    _a1, _o1 = env_cfg.action_space, env_cfg.observation_space
    env_cfg.action_space = 2 * _a1
    env_cfg.observation_space = 2 * _o1
    env_cfg._obs_single = _o1
    _bumped = []
    for _sec, _key in (("algorithm", "priv_info_dim"), ("network", "actor_priv_dim")):
        _d = agent_cfg.get(_sec, {})
        if _key in _d:
            _d[_key] = 2 * int(_d[_key])
            _bumped.append(f"{_sec}.{_key}={_d[_key]}")
    assert len(_bumped) == 2, f"特权维度没找全: {_bumped}"
    print(f"[record] 双臂: 动作 {_a1}->{env_cfg.action_space} | 观测 {_o1}->{env_cfg.observation_space}")
# ★ 2026-08-24: 这两个是**行为承重**旗, 之前只加在 train.py ⟹ 回放跑的是硬闸+无界指残差,
#   与训练不是同一个环境(实测 n_pads 均 0.20 vs 训练 1.37, 差 6~7 倍)。录像同样受害。
if getattr(args, "fc_ref_end", False):
    env_cfg.fc_target_ref_end = True
    print("[record] FC 目标点 = 参考终态")
if getattr(args, "fc_squeeze", False):
    env_cfg.fc_target_squeeze = True
    print("[record] FC 目标点 = squeeze")
if getattr(args, "seg_gate", False):
    env_cfg.seg_gate = True
    print("[record] 分段探索门控 ON (与训练同口径)")
if float(getattr(args, "fin_gate_dpos_cm", 0.0)) > 0:
    env_cfg.fin_gate_dpos_cm = float(args.fin_gate_dpos_cm)
    print(f"[record] 指参考软闸: 腕距 < {env_cfg.fin_gate_dpos_cm}cm 放行")
if float(getattr(args, "fin_dev_scale", -1.0)) > 0:
    env_cfg.phase2_fin_dev_scale = float(args.fin_dev_scale)
    env_cfg.finger_dev_scale = float(args.fin_dev_scale)
    print(f"[record] 指残差上限 x{args.fin_dev_scale}")
env_cfg.scene.num_envs = args.num_envs
if getattr(args, "step_rew_log", ""):
    env_cfg.step_reward_log = os.path.abspath(args.step_rew_log)
env_cfg.closure_init_max = args.c0_max
env_cfg.viewer = ViewerCfg(
    eye=tuple(float(x) for x in args.eye.split(",")),
    lookat=tuple(float(x) for x in args.lookat.split(",")),
    origin_type="env", env_index=0, resolution=(720, 540))

_norender = bool(getattr(args, "dump", "")) or bool(getattr(args, "cand_diag", False)) \
    or bool(getattr(args, "pour_diag", ""))
if getattr(args, "pour_choreo", False):
    # 编舞预算终装 (构造前最后一刻, 防被后行覆盖 —— "钉子落点晚于消费点"第四犯):
    # 763 手行 + 启动余量; 402/481 两分支都盖到
    env_cfg.approach_extra0 = 1100
    env_cfg.approach_only_steps = 2400
    env_cfg.carry_start_deadline = 1400
    env_cfg.pour_free_budget = 5000   # fv8 破案: 0残差 ★口距门永不满足 ⟹
                                      # pf 永不退场, 278 步例行处决(=830案真凶)
    # 指行号从 npz 段表**现算** (train.py:986 同法; RL_Training 复核:
    # 钉死常数是"改了参考忘了改常数"第四犯的温床 —— 240 落在 hold2 = 提前
    # 20 行握拳, 350 少走 9 行 squeeze)。stride 换了/段长改了自动跟上。
    import numpy as _np_seg
    _zsg = _np_seg.load(os.path.abspath(args.carry_npz), allow_pickle=True)
    if "seg_names" in _zsg.files:
        _acc, _seg = 0, {}
        for _n, _l in zip(_zsg["seg_names"], _zsg["seg_lens"]):
            _seg[str(_n)] = (_acc, _acc + int(_l)); _acc += int(_l)
        _stc = int(getattr(env_cfg, "curobo_ref_stride", 1))
        env_cfg.aag_grasp_row = _seg["close"][0] // _stc
        env_cfg.fin_hold_row = (_seg["root_bend"][1] - 1) // _stc
        env_cfg.fin_end_row = (_seg["squeeze"][1] - 1) // _stc
        print(f"[choreo] 预算终装: extra0=1100 only_steps=2400 | 指行号段表现算"
              f"(stride{_stc}): grasp={env_cfg.aag_grasp_row} "
              f"hold={env_cfg.fin_hold_row} end={env_cfg.fin_end_row}")
    else:
        print("[choreo] ⚠ npz 无段表, 指行号维持默认 (可能错位)")
base = _EnvCls(env_cfg, render_mode=None if _norender else "rgb_array")
base.gentle = 1.0
n_steps = args.steps or base.ep_total
if args.fps > 0:
    base.metadata["render_fps"] = args.fps
if not _norender:
    base = gym.wrappers.RecordVideo(
        base, video_folder=out_dir, name_prefix=tag,
        step_trigger=lambda s: s == 0, video_length=n_steps, disable_logger=True)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)

agent = PPO(env, output_dir=os.path.join(out_dir, ".tmp"),
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
if getattr(args, "zero_res", False):
    print("[record] 0残差模式: 不加载策略, 动作恒零")
else:
    print(f"[record] loading {ckpt}")
    agent.restore_test(ckpt)
agent.set_eval()

obs_dict = env.reset()
succ = ep = 0
_dmp = [] if getattr(args, "dump", "") else None
_pd = [] if getattr(args, "pour_diag", "") else None
_cd = {} if getattr(args, "cand_diag", False) else None


def _cand_gates(raw):
    """逐侧累计 candidate 六门的通过率。

    candidate = GRASP段 & n_pads≥4 & cent≥阈 & rel_spd<阈 & obj_rot<阈 & disp<阈,
    再连续保持 candidate_hold_steps 步。恒零时必须知道是哪一道墙, 否则只能瞎调。
    分母口径: 每道门都在 **active** 内算, 另给"前门全过后该门再筛掉多少"的条件通过率。
    """
    import numpy as _np
    import tasks.pregrasp.bimanual as _BM
    from tasks.pregrasp.env import Phase
    cfg = raw.cfg
    for _tag, _sd in (("R", raw._A), ("L", raw._B)):
        sg = _sd.data.get("_sig") or {}
        if not sg:
            continue
        with _BM.use_side(raw, _sd):
            ph = raw.task_phase
            act = sg["active"]
            n = act.sum().item()
            if n == 0:
                continue
            # 到位链 (to_grasp 的三门+保持): 右手位置门过 17% 却只有 4% 进 GRASP,
            # 说明卡在朝向/腕速/保持之一 —— 与 candidate 链同表打出来才能对账
            _ag = {
                "[到位]位置<%.2gcm" % (cfg.eps_pos * 100): sg["d_pos"] < cfg.eps_pos,
                "[到位]朝向<%.3g°" % _np.degrees(cfg.eps_rot): sg["d_rot"] < cfg.eps_rot,
                "[到位]腕速<%.2g" % cfg.switch_vel_max:
                    raw.wrist_linvel_w.norm(dim=1) < cfg.switch_vel_max,
                "[到位]保持%d步" % cfg.switch_hold: raw.switch_run >= cfg.switch_hold,
            }
            g = {
                "GRASP段": (ph == Phase.GRASP),
                "n_pads≥%d" % cfg.success_min_pads: sg["n_pads"] >= cfg.success_min_pads,
                "向心≥%.2g" % cfg.grasp_centrip_thresh: sg["cent"] >= cfg.grasp_centrip_thresh,
                "物体未转<%.3g" % cfg.grip_rot_max: sg["obj_rot"] < cfg.grip_rot_max,
                "未推移<%.3g" % cfg.push_fail_dist: sg["disp"] < cfg.push_fail_dist,
            }
            d = _cd.setdefault(_tag, {"N": 0, "gate": {}, "cum": {},
                                      "pads": _np.zeros(8), "cand_run_max": 0})
            d["N"] += n
            run = act.clone()
            for k, v in _ag.items():
                d["gate"][k] = d["gate"].get(k, 0) + int((v & act).sum())
                run = run & v
                d["cum"][k] = d["cum"].get(k, 0) + int(run.sum())
            run = act.clone()
            for k, v in g.items():
                d["gate"][k] = d["gate"].get(k, 0) + int((v & act).sum())
                run = run & v
                d["cum"][k] = d["cum"].get(k, 0) + int(run.sum())
            for p in range(min(8, int(sg["n_pads"].max().item()) + 1)):
                d["pads"][p] += int(((sg["n_pads"] == p) & act).sum())
            d["cand_run_max"] = max(d["cand_run_max"],
                                    int(raw.cand_run[act].max().item()) if n else 0)
            # g2 (本任务真成功判据) 逐门: 腕→真抓姿 <eps_pos & <eps_rot & 指笛卡尔 <fin_cart_tol
            _g2d = getattr(raw, "_g2_d", None)
            _g2f = getattr(raw, "_g2_fe", None)
            if _g2d is not None and _g2f is not None:
                d.setdefault("g2", {"n": 0, "wok": 0, "fok": 0,
                                    "fmin": 9e9, "fsum": 0.0, "both": 0})
                gg = d["g2"]
                _wok = (_g2d < cfg.eps_pos) & act
                _fok = (_g2f < cfg.fin_cart_tol) & act
                gg["n"] += n
                gg["wok"] += int(_wok.sum()); gg["fok"] += int(_fok.sum())
                gg["both"] += int((_wok & _fok & raw.arrived).sum())
                gg["fmin"] = min(gg["fmin"], float(_g2f[act].min()) * 100)
                gg["fsum"] += float(_g2f[act].sum()) * 100
                _fcd = getattr(raw, "_fc_d", None)
                _sgr = getattr(raw, "_seg_rows", None)
                if _fcd is not None and _sgr:
                    _sgd = gg.setdefault("seg", {})
                    for _nm8, _lo8, _hi8 in _sgr:
                        _m8 = act & (raw.ref_t >= _lo8) & (raw.ref_t <= _hi8)
                        if not bool(_m8.any()):
                            continue
                        _e8 = _sgd.setdefault(_nm8, [0, None])
                        _e8[0] += int(_m8.sum())
                        _v8 = (_fcd[_m8].sum(dim=0) * 100).detach().cpu().numpy()
                        _e8[1] = _v8 if _e8[1] is None else _e8[1] + _v8
                    gg["rtmax"] = max(gg.get("rtmax", 0), int(raw.ref_t[act].max()))
_K = 4                                     # 全量状态只记前 K 个 env


def _side_state(raw, side):
    """逐侧全量状态快照 (前 _K env): 臂/指/腕/物体/垫力/相位。"""
    import tasks.pregrasp.bimanual as _BM
    with _BM.use_side(raw, side):
        F = torch.cat([s_.data.force_matrix_w.view(raw.num_envs, 1, 3)
                       for s_ in raw._contact_sensors], dim=1)[:_K]
        dpw, drw, _ = raw._align_err()
        return dict(
            arm_q=raw.arm_q[:_K].cpu().numpy(),
            fin_q=raw.finger_q[:_K].cpu().numpy(),
            wrist_p=(raw.wrist_pos_w[:_K]
                     - raw.scene.env_origins[:_K]).cpu().numpy(),
            wrist_q=raw.wrist_quat_w[:_K].cpu().numpy(),
            obj_p=(raw.object.data.root_pos_w[:_K]
                   - raw.scene.env_origins[:_K]).cpu().numpy(),
            obj_q=raw.object.data.root_quat_w[:_K].cpu().numpy(),
            obj_v=raw.object.data.root_lin_vel_w[:_K].cpu().numpy(),
            pad_F=F.cpu().numpy(),                         # (K,5,3)
            d_pos=dpw[:_K].cpu().numpy(), d_rot=drw[:_K].cpu().numpy(),
            ref_t=raw.ref_t[:_K].cpu().numpy(),
            phase=raw.task_phase[:_K].cpu().numpy(),
            arrived=raw.arrived[:_K].cpu().numpy(),
            cand=raw.got_candidate[:_K].cpu().numpy())


with torch.no_grad():
    for t in range(n_steps + 2):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        if getattr(args, "zero_res", False):
            mu = torch.zeros(env.unwrapped.num_envs,
                             env.unwrapped.cfg.action_space,
                             device=env.unwrapped.device)
        else:
            mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        raw = env.unwrapped
        s = raw._sig
        if _cd is not None:
            _cand_gates(raw)
        if _pd is not None and getattr(raw, "_c_sa", None) is not None:
            # 偏轨闸标定采样: e_pos=物体离自身参考 (dev_reset_m 同口径)
            _st = getattr(raw, "_pour_ms_new", None)
            _pd.append(dict(
                eA=raw._c_sa["e_pos"].cpu().numpy().astype("f4"),
                eB=raw._c_sb["e_pos"].cpu().numpy().astype("f4"),
                ct=raw._c_t.cpu().numpy().astype("i2"),
                pf=raw._pf_on.cpu().numpy(),
                star=(_st.cpu().numpy() if _st is not None
                      else raw._pf_on.cpu().numpy() & False),
                done=done.cpu().numpy().astype(bool)))
        if _pd is not None and bool(done.any()) \
                and getattr(raw, "_kill_dbg", None) is not None:
            # 死因实名: 逐终止组件计数; base_term 再下钻 _sig_merged 布尔旗
            _dd2 = done.bool()
            _kd = raw._kill_dbg
            _parts = {k: int((v & _dd2).sum()) for k, v in _kd.items()}
            _bt = _kd["base_term"] & _dd2
            if bool(_bt.any()):
                for k, v in (getattr(raw, "_sig_merged", None) or {}).items():
                    if torch.is_tensor(v) and v.dtype == torch.bool \
                            and v.ndim == 1 and k != "active" \
                            and bool((v & _bt).any()):
                        _parts[f"sig.{k}"] = int((v & _bt).sum())
                _ah = getattr(raw, "_approach_hit", None)
                if _ah is not None:
                    _parts["approach_hit*"] = int((_ah & _bt).sum())
            print(f"[kill] 步 {t}: {int(_dd2.sum())} env 终止 → "
                  + " ".join(f"{k}:{v}" for k, v in sorted(_parts.items())
                             if v), flush=True)
        if _dmp is not None:
            _dmp.append({"R": _side_state(raw, raw._A),
                         "L": _side_state(raw, raw._B),
                         "act": mu[:_K].cpu().numpy(),
                         "rew": r[:_K].cpu().numpy(),
                         "done": done[:_K].cpu().numpy()})
        if bool(done[0]):
            ep += 1
            ok = bool(s["newly_success"][0])
            succ += ok
            why = "✅ 成功" if ok else next(
                (k for k in ("fell", "thrown", "pushed", "stuck",
                             "table_crash", "timeout")
                 if k in s and bool(s[k][0])), "?")
            print(f"[record] env0 回合 {ep} @步 {t}: {why}")

if _pd is not None:
    import numpy as _np3
    _np3.savez_compressed(args.pour_diag,
                          **{k: _np3.stack([f[k] for f in _pd]) for k in _pd[0]})
    print(f"[pour_diag] {len(_pd)} 步 × {_pd[0]['eA'].shape[0]} env "
          f"→ {args.pour_diag} | ★总数 {int(sum(f['star'].sum() for f in _pd))}")
if _dmp is not None:
    import numpy as _np2
    _out = {}
    for _sd in ("R", "L"):
        for _k in _dmp[0][_sd]:
            _out[f"{_sd}_{_k}"] = _np2.stack([f[_sd][_k] for f in _dmp])
    for _k in ("act", "rew", "done"):
        _out[_k] = _np2.stack([f[_k] for f in _dmp])
    _np2.savez_compressed(args.dump, **_out)
    print(f"[dump] 全量状态已存 {args.dump}: {len(_dmp)} 步 × {_K} env "
          f"(臂/指/腕/物体/垫力/相位 双侧)")
if _cd is not None:
    print("\n" + "=" * 74)
    print("[cand_diag] candidate 六门通过率 (分母 = active 步数; 累计 = 前门全过后仍在)")
    for _tag, d in _cd.items():
        N = max(d["N"], 1)
        print(f"\n--- {_tag} 手  (active 步数 {N})  cand_run 峰值 {d['cand_run_max']} ---")
        print(f"{'门':<16}{'单门通过':>12}{'累计通过':>12}   ← 累计掉到 0 的那一行就是墙")
        for k in d["gate"]:
            print(f"{k:<16}{d['gate'][k]/N:>11.2%}{d['cum'][k]/N:>12.2%}")
        if "g2" in d:
            gg = d["g2"]; _n = max(gg["n"], 1)
            _tolcm = float(getattr(env_cfg, "fin_cart_tol", 0.01)) * 100
            print(f"★ g2(真成功判据): 腕达标 {gg['wok']/_n:.2%} | "
                  f"指达标(<{_tolcm:.1f}cm) {gg['fok']/_n:.2%} | "
                  f"两者+arrived {gg['both']/_n:.2%}")
            print(f"   指误差: 均 {gg['fsum']/_n:.2f}cm  最小 {gg['fmin']:.2f}cm  "
                  f"(阈 {_tolcm:.1f}cm)")
            print(f"      ref_t 峰值 {gg.get(chr(39)+chr(39), 0) if False else gg.get('rtmax','?')}")
            if gg.get("seg"):
                print(f"      {'段':<14}{'样本':>8}{'拇':>7}{'食':>7}{'中':>7}{'无':>7}{'小':>7}")
                for _nm8, _e8 in gg["seg"].items():
                    if _e8[0] == 0:
                        continue
                    _a8 = _e8[1] / _e8[0]
                    print(f"      {_nm8:<14}{_e8[0]:>8}"
                          + "".join(f"{_a8[jj]:>7.2f}" for jj in range(5)))
        _p = d["pads"] / N
        print("n_pads 分布: " + "  ".join(
            f"{i}:{_p[i]:.1%}" for i in range(6) if _p[i] > 0.0005))
    print("=" * 74)
print(f"\n[record] env0: {succ}/{ep} 成功  |  视频: {out_dir}/{tag}-episode-0.mp4")
try:
    _slot.release()
    print("[record] GPU 槽位已释放, 硬退出 (跳过 Isaac 关闭流程)")
except Exception:
    pass
import os as _os
import sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
