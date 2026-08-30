"""PreGrasp 任务训练入口.

  PY=/home/lyh/luhr/MagicSim/.venv/bin/python
  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --num_envs 1024 --headless
  快速冒烟:  ... --num_envs 128 --max_agent_steps 100000 --headless

与 rl_rebuild/correction/train.py 的关系: 只保留通用机制 (GPU 独占槽位 / num_envs
功耗软上限 / 里程碑记录), **不带** RSI 退火 / imitation 退火 / cuRobo 引导 —— 那些
钩子直接摸 rw.lam_* , 对本任务全是空集, 塞进旧入口只会污染已验证的文件.
录像: record.py 还不认识本任务 (动作 30 维 + 无参考回放), 暂不接 autorecord.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="Grasp2", help="数据源 (clips.py 注册表)")
parser.add_argument("--name", type=str, default="PreGrasp_0",
                    help="run 名 (= logs/<name>/ 与 TensorBoard 实验名)")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--grasp_prior", action="store_true",
                    help="启用 Dexonomy GraspPose prior (tasks/pregrasp/priors/<clip>.npz)")
parser.add_argument("--prior_npz", type=str, default=None,
                    help="显式指定 prior npz (覆盖 --grasp_prior 的默认路径); 多候选筛选后用它")
parser.add_argument("--prior_yaw", type=float, default=-1.0,
                    help="**钉死**的物体 yaw (度) = screen_prior Gate1 的 best_yaw. "
                         "不给 = 退回旧的自搜 yaw (会破坏与视频一致性, 接近段不可用)")
parser.add_argument("--direct_grasp_prob", type=float, default=None,
                    help="接近段课程: 直接从抓取相位起步的回合比例初值 (默认取 cfg 的 0.5). "
                         "调高 = 更多数据保抓取, 少给接近段探索")
parser.add_argument("--ppo3", action="store_true",
                    help="组3 PPO: entropy .005 / e_clip .1 / mini_epochs 8 / horizon 64 "
                         "(多试但步子小)")
parser.add_argument("--no_eps_curr", action="store_true",
                    help="关掉切换阈值课程 (直接用最终的 1.04cm/2.18°) —— E 组消融")
parser.add_argument("--no_imit", action="store_true",
                    help="模仿罚恒为 0 (完全不要求像人) —— F 组消融")
parser.add_argument("--ref_look", type=int, default=None,
                    help="参考前瞻帧数 (1=原行为 7 维; 5=借鉴 ConTrack, 35 维, 几何间隔 1/2/4/8/16)")
parser.add_argument("--approach", action="store_true",
                    help="开接近段 (pick_lift 完整任务). 不给 = 只训抓取 (旧任务)")
parser.add_argument("--retract", action="store_true",
                    help="**完整任务(接近+抓握)**用退避式起点族取代沿人手轨迹采 t0。"
                         "与 --approach_only 的区别: 这个仍然要抓要抬, 判据是抓稳+微抬升。")
parser.add_argument("--approach_only", action="store_true",
                    help="**只学接近**: 站姿->GraspPose, 不抓不抬不合拢 "
                         "(docs/APPROACH_DESIGN.md). 自动开退避起点族 + 外壳口径臂罚。")
parser.add_argument("--pregrasp_only", action="store_true",
                    help="裁定2026-08-18晚: 双手到掌心PreGrasp稳定=成功终止; 22指全开"
                         "(需RL_HAND_JOINTS=1)但到位前fin_quiet软抑制; 靶点=PreGrasp")
parser.add_argument("--fin_cart", action="store_true",
                    help="FC 实验 (2026-08-19): m2/成功的指型判据换成逐指笛卡尔到位; "
                         "逐指一次性面包屑 + 扰动罚全程在")
parser.add_argument("--fin_pot", type=float, default=0.0,
                    help="逐指势差分引导系数 (A/B 变量: A=0 纯离散, B>0)")
parser.add_argument("--fcd", action="store_true",
                    help="FC-D (2026-08-20 用户三裁定): 指参考四段跟踪(Pose0起手学构型/"
                         "阶梯腕距查表/合拢限速) + 慢合拢残差闸(近0.3x/合拢0.2x) + "
                         "Pad首触/持续/向心/稳抓奖励 + 撞桌即Fail。配套摩擦触觉镜像已内建")
parser.add_argument("--fin_pose1", action="store_true",
                    help="FC-C (2026-08-19 定稿): 每侧 q_open ← 先验 pregrasp[0] "
                         "(Pose1 拇指对掌) —— reset 从 Pose1 出发, 合拢轴=四指卷握走廊; "
                         "配套 Pose1 手型规划的 curobo_ref (pose1_1cm.npz)")
parser.add_argument("--pour_carry", action="store_true",
                    help="Carry 实验 (Pour P0+P1 MVP, 2026-08-19): GraspPose 起步, "
                         "学稳抓+带物体追踪重建轨迹。需配 --pregrasp_only --phase2 "
                         "--curobo_ref <carry npz>")
parser.add_argument("--carry_progress", action="store_true",
                    help="CARRY3: 进度时钟 (握稳才启动/跟上才前进/按里程发钱/高置信里程碑)")
parser.add_argument("--carry_tilt_ms", default="",
                    help="CARRY4 扩展A: 瓶体倾角几何里程碑(度, 逗号分隔, 如 '30,60'); "
                         "置信度无关, 参考峰值 71° ⟹ 档位必须 <71")
parser.add_argument("--carry_ms_low", default="",
                    help="CARRY4 扩展B: 沙漠低置信帧小糖宽门里程碑 (行号, 如 '110,145')")
parser.add_argument("--carry_squeeze", action="store_true",
                    help="CARRY4 通用公约: q_close ← squeeze 模板 (承载段收紧轴交给策略)")
parser.add_argument("--carry_pad_reward", action="store_true",
                    help="C5 (A2 裁定): 垫接触奖励进 carry (首触+持续, 甜甜圈判据)")
parser.add_argument("--friction_curr", action="store_true",
                    help="C5 (B4 裁定): 指垫摩擦开局 friction_hi, 随稳抓存活退火回 3.0")
parser.add_argument("--table_fail", action="store_true",
                    help="真机红线 (2026-08-20 用户裁定: 所有任务开启): 手部 body 原点"
                         "低于桌面+4mm ⟹ 立即终止; 罚带保留=允许贴近不允许碰撞")
parser.add_argument("--pour_succ", action="store_true",
                    help="Pour (B6 裁定): 纯几何成功 = 倾角≥71°&瓶口对杯口6cm&保持10步; "
                         "含 B5 瓶口低于质心一次性糖")
parser.add_argument("--pour_free", action="store_true",
                    help="自由探索倒水段 (2026-08-21 用户裁定): carry行[93,140)参考停用/"
                         "行进门·跟丢判挂起, Success Tracker(倾角糖+口对口势差分+pour成功"
                         "判据)引导; 成功→重拍rest锚跳行140续追。需配 --pour_succ")
parser.add_argument("--step_rew_log", type=str, default="",
                    help="逐步奖惩记录目录 (2026-08-21): 每步各奖励项均值+前8探针env"
                         "全明细分片落盘, 事后逐步回放状态↔奖惩")
parser.add_argument("--fc_earn", action="store_true",
                    help="fc_pot 只奖不罚 (基线归零: 严格复现参考时该项实测 -0.046/步)")
parser.add_argument("--qd_soft_arm", type=float, default=-1.0,
                    help=">0 覆盖臂关节软速度阈 (标定判据: 零动作下 qvel_hard≈0)")
parser.add_argument("--pre_close", type=float, default=0.0,
                    help="合拢前禁触罚权重 (窗口 ref_t<aag_grasp_row=close段起始). "
                         "罚 = w×(窗口内指垫接触数 + 物体位移超死区的厘米数)")
parser.add_argument("--geom_audit", action="store_true",
                    help="几何审计: 逐 body 外壳-物体距离分布 + **用物理标定等效接触阈** "
                         "(物体开始动那一刻的 shell_obj_d), 不靠猜原点偏移")
parser.add_argument("--fc_ref_end", action="store_true",
                    help="★FC 目标点在**参考终态**下测(臂=参考末行, 指=参考末行)。原口径用 env 自解位形, 与参考差 臂33.5°/指37.6° ⟹ g2 结构性恒 0")
parser.add_argument("--fc_squeeze", action="store_true",
                    help="FC 逐指目标点摆 squeeze 而非 grasp (参考终态是 squeeze, 取 grasp 会把做对的事判成差 6cm)")
parser.add_argument("--squeeze_grip", type=float, default=0.0,
                    help="⑤ squeeze 段抓力奖励权重: pad_near(垫→表面势差)+"
                         "grip_pot(quality=压入−不平衡−力矩 的势差), 均 earn-only")
parser.add_argument("--seg_gate", action="store_true",
                    help="分段探索门控 (2026-08-24 用户编排): 按参考段号×关节组分配"
                         "残差步长 —— ①臂动指冻 ②只拇指 ③臂动指冻 ④⑤全开")
parser.add_argument("--pre_close_shell_cm", type=float, default=0.0,
                    help="合拢前外壳-物体最小间隙(cm), 低于它就罚; 0=关。"
                         "★余量必须由预检的参考实测标定, 否则又在罚参考")
parser.add_argument("--pre_close_tilt_deg", type=float, default=5.0,
                    help="禁触窗口的倾角死区(度); 手指能把物体转歪而位移不大, 必须单列")
parser.add_argument("--pre_close_dead_cm", type=float, default=1.0,
                    help="位移死区(cm); 参考自身窗口内上限实测 右0.65/左0.00, 默认1.0 保证不罚参考")
parser.add_argument("--fin_gate_dpos_cm", type=float, default=0.0,
                    help=">0: 指参考行进门改**软闸** —— 腕距 < 该值(cm)就放行, 不必等 "
                         "arrived 闩死。治'腕漂过 eps ⟹ 下游收入断崖归零'的局部最优陷阱")
parser.add_argument("--cent_earn", action="store_true",
                    help="向心塑形只奖不罚 (方向基准硬编码指向物体原点, 对杯这类"
                         "原点在底/抓上沿的物体结构性为负; 且它是唯一碰到就扣的稠密罚)")
parser.add_argument("--fin_dev_scale", type=float, default=-1.0,
                    help="手指残差累计上限 = 逐关节标定界 × 该值 (pregrasp29 默认 40 "
                         "= 中位 316° = 无界, 已判死). 2.0 = 中位 ±15.8°, 够压紧不够漂")
parser.add_argument("--qvel_settle", action="store_true",
                    help="复位冻结窗口内不罚关节速度 (该窗口动作被归零, 策略无影响力; "
                         "且复位传送产生伪速度. 预检实测右手 -0.048/步 全是这个)")
parser.add_argument("--qd_soft_fin", type=float, default=-1.0,
                    help=">0 覆盖指关节软速度阈 (标定判据同上: 零动作 qvel_hard≈0)")
parser.add_argument("--preflight", type=int, default=0,
                    help="起飞前自检: 不训练, 零动作跑 N 步后打全套行为侧体检表并退出 "
                         "(与训练同一段配置代码 ⟹ 预检说的就是训练会发生的)")
parser.add_argument("--arm_abs", action="store_true",
                    help="方案C (2026-08-23 用户裁定): 臂改绝对参考+有界累加残差 "
                         "(q_cmd = q_ref[t] + arm_res, |arm_res| <= arm_abs_dev), "
                         "与手指路径同构; 差分前馈积分器退役, ff_pull 自动失去意义")
parser.add_argument("--arm_abs_dev_deg", type=float, default=2.86,
                    help="方案C 臂残差逐关节上限 (度); 2.86deg = 0.05rad = arm_dev_max")
parser.add_argument("--aag", action="store_true",
                    help="AAG (2026-08-21 用户裁定): 指参考=标准成功轨迹逐行 "
                         "(aag_pour17_ref.npz, 六段编舞含 thumbfix squeeze 深拇段); "
                         "臂前馈同文件; 1cm 契约退役 (参考直达抓姿腕位); 需配 --fcd")
parser.add_argument("--pours_v6", action="store_true",
                    help="PourS-v6 (2026-08-21 深夜用户裁定, 阶段纯净化): 物体位姿约束"
                         "只在非交互段; 交互段只要求手物零相对移动。toppled 交互豁免/"
                         "thrown→偏轨30cm硬闸/cent 只启动前/杯直立删/cross 纯碰撞/"
                         "倒水成功=回合终止(不撤离)/成功线93°(参考瓶平台×0.9)。需配 --pours_v5")
parser.add_argument("--pours_v5", action="store_true",
                    help="PourS-v5 (2026-08-21 用户裁定'抓稳优先'): 滑移逐步增量罚+"
                         "握紧反射+持续抓稳项+杯直立+左右互碰罚+特权滑移进观测(+8/侧); "
                         "slip 死线 8cm/60°, thrown 28cm。obs 变维 ⟹ 必须从头训; "
                         "需配 --pour_free --carry_stable --carry_pad_reward")
parser.add_argument("--prog_pot", type=float, default=0.0,
                    help="v6P (2026-08-22 用户裁定): 全局进度势权重 —— 奖励'重建轨迹"
                         "复现了百分之几', 0→100% 共发这么多 (40=与挂机收入同量级)")
parser.add_argument("--mouth_w", type=float, default=0.0,
                    help="自由段口对口势权重覆盖 (默认 5; 15=强对齐)")
parser.add_argument("--ff_tol", type=float, default=0.0,
                    help="自由段倾角行进门容差覆盖 (默认 20°; 10=更贴参考)")
parser.add_argument("--fin_slow", action="store_true",
                    help="指参考半速播放 (stride 2→1): 编舞本是 1 帧/步 授权的, "
                         "2x 快播超出关节速度限 (诊断 qvel_hard −464)")
parser.add_argument("--c0_min", type=float, default=-1.0,
                    help="诊断: 起步合拢度固定值 (1.0=直接从抓姿闭合起步, 验证 "
                         "candidate 判据可达性)")
parser.add_argument("--e2e_squeeze", action="store_true",
                    help="E2E-FULL: 站姿接近→抓取→倒→放→松 全链, squeeze 手指体制 "
                         "(pour_e2e 的 env 机制, 不绑 fcd; 需 --pregrasp_grasp --carry_progress)")
parser.add_argument("--pour_place", action="store_true",
                    help="v10: 放稳(原carry成功)→RELEASE 松手斜坡; ★降级回里程碑出段续追")
parser.add_argument("--asym_pen", type=float, default=0.0,
                    help="双侧基座收入不对称罚权重 (平坦脊→斜坡; 温和版, 判据不动)")
parser.add_argument("--pour_retreat", action="store_true",
                    help="E2E第五段: 松手成功后臂脚本撤回出生站姿, 撤完=全链终点")
parser.add_argument("--prog_joint", action="store_true",
                    help="v9J: 进度棘轮改联合乘积 (每步倾角分×对齐分的历史最大), 治分时刷分")
parser.add_argument("--prog_align", action="store_true",
                    help="v9: 窗内进度=倾角×对齐乘积 (倾角奖励不再方向无关)")
parser.add_argument("--ff_play", action="store_true",
                    help="v8: 自由段续播臂前馈, 时钟按倾角剖面推进 (破 50° 天花板)")
parser.add_argument("--tilt_prog", action="store_true",
                    help="v6T: 自由段抓稳粮改由**倾角创新高**解锁 (原整窗豁免=窝在"
                         "窗里挂机领钱, 比倾到底划算 16 倍)")
parser.add_argument("--dgp_rsi_hi", type=float, default=0.0,
                    help="RSI 课程: 起始直抓起步比例 (0.8=先学抓), 随稳抓能力棘轮"
                         "退火到 --direct_grasp_prob; 0=关")
parser.add_argument("--obj_disturb_pre", type=float, default=1.0,
                    help="AAG-Local: 到位前物体扰动罚倍率 (8=掌蹭主罚)")
parser.add_argument("--fin_fade", action="store_true",
                    help="接触后淡出指参考罚 (按已触垫数比例) —— 防'罚它抓东西'")
parser.add_argument("--fin_gate", action="store_true",
                    help="手指行进门: 到位前指参考钳在弯根部行 ('跟上才前进')")
parser.add_argument("--rew_sched", action="store_true",
                    help="相位奖励日程表 (2026-08-26): PREGRASP 段把抓取期罚组置零, "
                         "进 GRASP 后**线性渐入** rew_sched_ramp 步 (渐入而非硬切 —— "
                         "硬切的边界断崖会让策略学'躲门'), GRASP 段接近组降权。")
parser.add_argument("--rew_sched_ramp", type=int, default=-1,
                    help="进 GRASP 后罚组渐入的步数 (默认用 cfg 的 60; 0=硬切)")
parser.add_argument("--grip_g", action="store_true",
                    help="握力信任标量 g (2026-08-25): ④段慢档累积 + 驱动前段密集项 "
                         "×(1−g) 衰减。**不发任何握力奖励** —— 捏紧由参考自身 squeeze "
                         "+ g2 目标点完成 (用户裁定)。交互段快档在 tasks/pour 侧。")
parser.add_argument("--obj_obs", action="store_true",
                    help="物体特权进 actor (+13/侧: 腕系位置3+相对姿态4+线速度3+**角速度3**)")
parser.add_argument("--pour_trend", type=str, default="",
                    help="v7 趋势钟: 自由段倾角趋势参考 npz (bottle_deg/cup_deg)")
parser.add_argument("--pad_gate", action="store_true",
                    help="v3.3: pad_first 碰垫糖只在到位后发 (门前蹭垫工资拐走左手)")
parser.add_argument("--form_pot", type=float, default=0.0,
                    help="v3.4: 形态正向势差分权重 (2.0=温和; 向参考形态推进给小钱)")
parser.add_argument("--arrive_bonus", type=float, default=-1.0,
                    help="v3.4: 到位大奖覆盖 (r_arrive, 8.0=加码强调'进门给大钱')")
parser.add_argument("--tilt_pot", type=float, default=0.0,
                    help="v6c: 自由段倾角势差分权重 (5.0=满倾角累计+5, 补稀疏糖空档)")
parser.add_argument("--eps_pos_cm", type=float, default=-1.0,
                    help="AAG-v3.2 (2026-08-22): 到位门 eps_pos 覆盖 (cm), 在 minimal "
                         "落定后生效; 1.35=盖住纯前馈停靠点(L1.09/R1.00)+噪声")
parser.add_argument("--grip_prog_gate", type=int, default=0,
                    help="v6 挂机补丁 (2026-08-22): grip_hold 进度门 —— 近 N 步 c_t 无"
                         "前进则停发 (pf 窗豁免); 0=关。推荐 20")
parser.add_argument("--arm_pre_left", type=float, default=-1.0,
                    help="左手消融1 (2026-08-22): 左臂到位前带宽单独覆盖 (0.35=给宽杯"
                         "绕障余地; <0=同 arm_ff_pre)")
parser.add_argument("--fin_near_relax", type=float, default=0.0,
                    help="左手消融2 (2026-08-22): 腕距<3cm 时 fin_track/fin_quiet 乘此"
                         "系数 (0.3=近场松绑手指变形; 0=关)")
parser.add_argument("--arm_ff_gate", action="store_true",
                    help="AAG-v3 (2026-08-21 用户裁定): 到位前臂残差归零=纯前馈走参考 "
                         "(回放已验证 0.3-0.6cm 落点+零碰物), 到位后 ramp 到 ±2° 小带; "
                         "臂上探索只在抓握段花, 学习预算全给手指 (JSRL h=到位行特例)")
parser.add_argument("--grasp_cent_fix", action="store_true",
                    help="cent_fix (2026-08-20 用户裁定): candidate 去向心门(阈-1) + "
                         "向心塑形带负梯度 clamp(-0.5,1) —— 托架构型下旧 max(cent,0) "
                         "恒零=梯度死区, 五线 candidate 全零的根因修复")
parser.add_argument("--pour_e2e", action="store_true",
                    help="Pour 端到端 (2026-08-20 用户裁定): 站姿→接近→真抓稳(双侧微抬升"
                         "验证=启动门)→进度时钟追踪→倒水→还原走完; 参考=curobo_pour17_e2e.npz"
                         "(缝行 seam 存 npz); 需配 --fcd --carry_progress, 禁 --carry_squeeze")
parser.add_argument("--carry_stable", action="store_true",
                    help="甜甜圈稳抓链进 carry (2026-08-20 用户裁定): 全程向心塑形 + "
                         "启动前 candidate(≥4垫&向心≥0.3&静&保持4步)+5; 依赖 --carry_pad_reward")
parser.add_argument("--w_toppled", type=float, default=0.0,
                    help="拍倒显式小罚 (FC-D RSI-B 线 2026-08-20): >0 时独立罚 toppled, "
                         "不解封 APPROACH_OFF 里的整个 fail 项; 量级应 < candidate +5")
parser.add_argument("--carry_npz", type=str, default="tasks/pour/carry_pour17.npz",
                    help="build_carry_ref 产物 (物体目标序列)")
parser.add_argument("--m1_cm", type=float, default=-1.0,
                    help="覆盖 cfg.phase2_m1_cm (m1 里程碑腕距门, cm)。"
                         "0=删除 m1 里程碑(简化二: 链条剩 arrive→m2→成功); -1=用 cfg 默认")
parser.add_argument("--phase2", action="store_true",
                    help="Phase2: 到位后播 PreGrasp→GraspPose 名义斜坡(腕直线插入+22指"
                         "缓慢合到抓姿, 不要求抓稳); 成功推迟到斜坡走完。需配 --pregrasp_only")
parser.add_argument("--pregrasp_grasp", action="store_true",
                    help="PG-A (2026-08-18晚): 1cm掌心PreGrasp口径+22指全开, 但走完整"
                         "抓取链 —— 到位切GRASP相位, 成功=抓取候选+微抬升验证(物理判决)")
parser.add_argument("--obj_jitter", type=float, default=None,
                    help="覆盖物体 xy 抖动 (m); 课程 3mm->6->10->15, 每档一条 run")
# ---- 2026-08-16 五条对照实验的单变量旋钮 (docs/DESIGN_LOOP §2.26) ----
parser.add_argument("--bimanual", action="store_true",
                    help="双臂接近: 一张网同时输出左右手动作 (动作/观测都翻倍). "
                         "需配 --prior_b 指定另一只手的 GraspPose")
parser.add_argument("--bi_native", action="store_true",
                    help="用 v2 原生双臂底座 (bimanual_native_env: 侧命名空间+属性路由, "
                         "构造对称性由结构保证). 须配 --bimanual")
parser.add_argument("--prior_b", type=str, default="",
                    help="另一只手的 GraspPose npz (双臂用)")
parser.add_argument("--prior_b_yaw", type=float, default=-1.0,
                    help="另一只手那个物体的钉死 yaw (度)")
parser.add_argument("--curobo_ref", type=str, default="",
                    help="cuRobo 逐侧臂参考 npz (view_curobo_plan --save_plan 导出). "
                         "接成前馈: 零动作=沿规划走, RL 只学修正。需配 --bimanual")
parser.add_argument("--ff_freeze_cm", type=float, default=5.0,
                    help="腕距目标小于此值(cm)时冻结 cuRobo 前馈, 收尾交给纯残差; 0=不冻结")
parser.add_argument("--ff_pull", type=float, default=0.08,
                    help="弱回拉锚定: 每步把 q_cmd 往计划位形拉这个比例(冻结区外); 0=关")
parser.add_argument("--l5", action="store_true",
                    help="L5 边靠近边合指+微抬升: 冻结区解锁手指, 合拢参考追踪 c(d)。"
                         "须配 --bimanual --curobo_ref, 不配 --approach_only")
parser.add_argument("--dyn_far", type=float, default=-1.0,
                    help="远场残差上限倍数覆盖 (-1=用 cfg 默认 3.0)。带 cuRobo 前馈时建议 1.0: "
                         "3.0 档是无前馈时代为'自己横跨16cm'设的, 有前馈后只剩甩手的本钱 "
                         "(2026-08-17 L5 ep200 录像实测手甩过头顶)")
parser.add_argument("--l5_direct", type=float, default=0.0,
                    help="近点直起比例。⚠ 默认 0: 直起与 cuRobo 前馈互斥 (2026-08-17 ep100 "
                         "实测: 直起回合 ff 从近点再播全程增量, 左臂顶死 stuck~100%)。"
                         "待实现'直起回合 ref_t 初始化到末帧'后再开 0.3")
parser.add_argument("--start_pool", choices=("", "uniform", "mastery"), default="",
                    help="自生成起点池: uniform=各档等权 | mastery=权重∝(1-掌握度)+地板. "
                         "起点取自策略真到过的状态, 取代手工构造的退避课程")
parser.add_argument("--warm_start", type=str, default="",
                    help="从**动作维度不同**的 ckpt 热启动(如 7 维接近段 -> 13/29 维抓取): "
                         "形状对得上的层照搬, 动作头只搬前 N 维, 其余保持新初始化。"
                         "普通同维度续训请用 --load_path")
parser.add_argument("--time_shape", action="store_true",
                    help="用时塑形: 到位奖励按用时衰减(给绕路加价). "
                         "**按成功率放行** —— 慢EMA<gate 前不打折, 免得早期把信号削没")
parser.add_argument("--time_shape_gate", type=float, default=0.5,
                    help="成功率慢EMA 到这个值才开始打折")
parser.add_argument("--time_shape_ref", type=float, default=94.0,
                    help="不打折的基准用时(步); 94 = 冠军实测")
parser.add_argument("--minimal_fixed_res", action="store_true",
                    help="最简模式下**关掉**残差的远松近紧(退回恒定步长). "
                         "只为复现 2026-08-17 之前的 Minimal_L 行为做对照")
parser.add_argument("--minimal", action="store_true",
                    help="最简接近训练: 起点固定站姿 / 无前馈 / "
                         "无课程无退火 / 残差**远松近紧**(减速靠它, 不靠参考) / 奖励=密集接近+到位")
parser.add_argument("--eps_final_cm", type=float, default=None,
                    help="覆盖到位判据的**终值**位置公差 (cm). 判据实验: 放到 2cm 若能训通 "
                         "=> 卡的是精度; 仍 0% => 问题不在精度")
parser.add_argument("--eps_final_deg", type=float, default=None,
                    help="覆盖到位判据终值朝向公差 (度)")
parser.add_argument("--r_reach", type=float, default=None,
                    help="覆盖到位终端奖励 (默认 100). 调小 = 让 shaping 主导, 测'终端悬崖'假设")
parser.add_argument("--dyn_arm_near", type=float, default=None,
                    help="覆盖近端残差缩放 (默认 0.3). 1.0 = 关掉近端收紧")
parser.add_argument("--curr_start_target", type=float, default=0.3,
                    help="起点课程退火到位所需的成功率慢EMA (默认 0.3). 调高 = 课程推进更保守")
parser.add_argument("--ar_ema", type=float, default=0.995,
                    help="arrive_rate 慢 EMA 系数 (课程回退速度; N3 试 0.98, 半衰期 2M→0.5M 步)")
parser.add_argument("--stance_prefix", type=int, default=0,
                    help=">0: 参考轨迹前拼 K 帧 默认站姿->人手轨迹起点 (PLAN D2 端到端起点); 评测须同值")
parser.add_argument("--orient_blend", action="store_true",
                    help="方案一 (§2.15): 参考姿态混合到 GraspPose, 边前进边转向; 评测须同 flag")
parser.add_argument("--cone", action="store_true",
                    help="方案二 (§2.16): 锥形信任管接替增量模仿罚; 训练时配 --no_imit")
parser.add_argument("--grasp_first", action="store_true",
                    help="先抓后飞门控 (§2.17): 阶段A 全回合从抖动 GraspPose 起步纯练抓取, "
                         "sr/from_grasp 慢 EMA ≥ gf_target 后放行接近分支")
parser.add_argument("--gf_target", type=float, default=0.8,
                    help="门控毕业线 (抖动起点分布上的抓取成功率 EMA; 0.9 会与技能平台期打平, 永不触发)")
parser.add_argument("--place", action="store_true",
                    help="PickAndPlace (§2.19): 抬升验证后接 搬运+放置, 成功=放到人手示范位 ±3cm; 评测须同 flag")
parser.add_argument("--start_jitter", action="store_true",
                    help="直接抓取回合起点用抖动池 (±eps0 包络), 不依赖 --grasp_first. "
                         "用途: 给已会飞的 ckpt 补'从到达偏差状态起抓'这门课 (O1 交接断裂的解药)")
# 大batch(3072)调参覆盖 (不填=用 ppo.yaml 默认)
parser.add_argument("--kl_threshold", type=float, default=None, help="覆盖 kl_threshold (放开策略步长)")
parser.add_argument("--mini_epochs", type=int, default=None, help="覆盖 mini_epochs (每batch多更新)")
parser.add_argument("--minibatch", type=int, default=None, help="覆盖 minibatch_size (不填=min(num_envs*8,32768))")
parser.add_argument("--lr", type=float, default=None, help="覆盖 learning_rate (√k 缩放)")
parser.add_argument("--entropy_coef", type=float, default=None, help="覆盖 entropy_coef (大batch保探索)")
parser.add_argument("--adam_betas", type=str, default=None, help="覆盖 Adam betas, 逗号分隔 如 0.5,0.9")
# ---- 自动停止 (2026-08-09): 周期性进程内确定性评测 + 收敛/平台期判停 ----
parser.add_argument("--auto_stop", choices=("off", "dry", "on"), default="off",
                    help="off=完全不跑(默认, 行为不变) | dry=评测+记录但**不停** (先用它验证判据) "
                         "| on=达标或平台期就结束训练")
parser.add_argument("--eval_every", type=int, default=100,
                    help="每多少 epoch 插一次确定性评测 (100 epoch≈3.3M 步≈27min, "
                         "单次评测 ~82s = 5%% overhead)")
parser.add_argument("--eval_steps", type=int, default=160,
                    help="评测控制步数; 必须 > 回合上限 (153) 才能保证每个 env 走完一回合")
parser.add_argument("--stop_target_sr", type=float, default=0.99,
                    help="早退线: 确定性成功率达到它 (且连续 --stop_target_hits 次) 就停")
parser.add_argument("--stop_target_hits", type=int, default=2,
                    help="达标要连续几次评测 (2 次 = 稳住 ~55min, 滤掉蜜月尖峰)")
parser.add_argument("--stop_patience", type=int, default=5,
                    help="主判据: 最佳成功率连续这么多次评测没涨够 --stop_delta 就停")
parser.add_argument("--stop_delta", type=float, default=0.01,
                    help="算'涨了'的最小增量 (1pt ≈ 3×SE@n=1024, 小于它当噪声)")
parser.add_argument("--stop_min_steps", type=float, default=20e6,
                    help="这么多步之前一律不停 (只记账). 难物体的爬坡期可以很长, "
                         "别让平台期判据在前期把它掐死")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---- 功耗防护 (同 rl_rebuild/correction/train.py; 事故记录见 CLAUDE.md ⚡条) ----
_MAX_ENVS = int(os.environ.get("RL_MAX_ENVS", "1024"))
if args.num_envs > _MAX_ENVS:
    print(f"[train] ⚠ num_envs {args.num_envs} 超过软上限 {_MAX_ENVS}, 已压回 "
          f"(需要更大规模用 RL_MAX_ENVS={args.num_envs} 显式覆盖)")
    args.num_envs = _MAX_ENVS

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("train")

app = AppLauncher(args).app

import contextlib  # noqa: E402
import json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from datetime import datetime  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.auto_stop import StopDecider  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402


def _scalar_snapshot(cfg):
    """cfg 上所有标量字段的浅快照 (给 eval_distribution 的还原自检用)."""
    out = {}
    for k in dir(cfg):
        if k.startswith("_"):
            continue
        try:
            v = getattr(cfg, k)
        except Exception:
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


@contextlib.contextmanager
def eval_distribution(raw):
    """把 env 临时切到 `tasks/pregrasp/eval.py` 的**口径A** (正式对外口径), 退出原样还原.

    ⚠ 用 contextmanager 而不是手写成对赋值: 漏还原任何一项都是**静默失效** ——
    不报错, 但之后的训练跑在被偷偷改过的分布上, 几小时后看曲线才发现 (§6.5 同款)。

    与训练分布的差别 (这就是 CLAUDE.md 说"训练期 TB 会低估"的全部来源):
      closure_init_max 0.9→0   手从完全张开起步, 不给合拢脚手架
      direct_grasp_prob →0     全部回合从接近段起步 (课程只退火到 0.1, 永远差这一截)
      approach_t0_max   →0     从 q_ref[0] 出发, 无起点随机化
      eps_pos/eps_rot   →final 切换阈值用最终的紧公差
      arm_start_pool    →None  eval.py 不建抖动池, 起点是精确 pregrasp
      gentle            →1.0   全价惩罚 (只影响奖励读数, 不影响成功判据)
    """
    cfg = raw.cfg
    _KEYS = ("closure_init_max", "direct_grasp_prob", "approach_t0_max",
             "eps_pos", "eps_rot", "retract_ratio", "stance_prob")
    saved = {k: getattr(cfg, k) for k in _KEYS if hasattr(cfg, k)}
    saved_gentle, saved_pool = float(raw.gentle), raw.arm_start_pool
    saved_to0 = int(raw.phase_timeout_t[0])
    # 全量标量快照: 上面那份 _KEYS 只还原"我知道自己改了的", 这份负责抓"改了但忘了列"
    # —— 后者是静默失效, 不 assert 的话要等几小时后看曲线才发现
    audit = _scalar_snapshot(cfg)
    try:
        cfg.closure_init_max = 0.0
        raw.gentle = 1.0
        raw.arm_start_pool = None
        if getattr(cfg, "approach", False):
            cfg.direct_grasp_prob = 0.0
            cfg.approach_t0_max = 0.0
            if hasattr(raw, "_eps_final"):
                cfg.eps_pos, cfg.eps_rot = raw._eps_final
            raw.phase_timeout_t[0] = raw.gs + int(cfg.approach_extra_steps)
        if getattr(cfg, "retract_start", False):
            # ★ 退避式接近的正式口径 = **全部从对称站姿起步**。
            #   注意方向: 这里设成 **1.0**(最难), 而不是像 approach_t0_max 那样设成 0。
            #   两套课程方向相反, 混了就是把评测变成"空降到终点"(DESIGN_LOOP §2.24)。
            cfg.stance_prob = 1.0
            cfg.retract_ratio = 1.0
        if getattr(cfg, "approach_only", False):
            raw.phase_timeout_t[0] = int(cfg.approach_only_steps)
        yield
    finally:
        for k, v in saved.items():
            setattr(cfg, k, v)
        raw.gentle, raw.arm_start_pool = saved_gentle, saved_pool
        raw.phase_timeout_t[0] = saved_to0
        drift = {k: (v, audit[k]) for k, v in _scalar_snapshot(cfg).items()
                 if k in audit and v != audit[k]}
        assert not drift, (
            f"评测后 cfg 没还原干净 (评测值, 原值): {drift} —— "
            f"eval_distribution 改了这些字段但没列进 _KEYS, 训练分布已被污染")


class MilestonePPO(PPO):
    """PPO + 首达各成功率档位的步数记录 (sample-efficiency 指标)
    + 周期性确定性评测/自动停止 (--auto_stop).
    与旧 CorrectionPPO 的区别: 没有任何 reward/RSI 退火钩子."""

    MILESTONES = (0.01, 0.10, 0.25, 0.50, 0.80, 0.90)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ms_path = os.path.join(self.output_dir, "milestones.json")
        self._ms_hit: dict = {}
        self._sr_ema = 0.0
        self._sr_slow = 0.0     # 慢速 EMA (4 倍慢): 只有**巩固**的成功率才推高价格
        # ---- 自动停止 (train.py 尾部按 CLI 覆盖; 判据在 auto_stop.StopDecider) ----
        self._auto_stop = "off"
        self._eval_every, self._eval_steps = 100, 160
        self._stopper = StopDecider()
        self._last_curr_sig = None
        self._eval_hist: list = []
        self._stop_path = os.path.join(self.output_dir, "auto_stop.json")

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        sr = self.extra_info.get("success_rate")
        if sr is None:
            return
        self._sr_ema = 0.98 * self._sr_ema + 0.02 * float(sr)
        self.writer.add_scalar("success_rate_ema", self._sr_ema, self.agent_steps)
        # 笨拙课程: 轻柔类惩罚按成功率涨价 (0.2 -> 1.0), 见 cfg.gentle_init 注释
        # ⚠ 三重保护 (每条都是一次失败 run 换来的):
        #   棘轮只涨不跌 —— 无棘轮出极限环 (sr 19%→7%, 靠退步换降价);
        #   慢速 EMA 定价 —— 用瞬时 EMA 定价会在蜜月尖峰上把价顶到 0.78, 超出已巩固
        #     技能, 尝试期望值转负, 技能被慢性剪除 (续跑实测 14.5%→5.8%);
        #   每 epoch 限速 +0.005 —— 全程涨价至少铺开 5M 步, 价格永远不冲到技能前头.
        self._sr_slow = 0.995 * self._sr_slow + 0.005 * float(sr)
        raw = getattr(self, "_raw_env", None)
        if raw is not None:
            tgt = 0.2 + 0.8 * min(self._sr_slow / 0.2, 1.0)
            cur = getattr(raw, "gentle", 0.2)
            raw.gentle = max(cur, min(tgt, cur + 0.005))
            self.writer.add_scalar("gentle", raw.gentle, self.agent_steps)
        # ---- 先抓后飞门控 (§2.17): 阶段A 纯抓取(dgp 钉 1.0), 毕业才放行接近课程 ----
        gated = getattr(self, "_grasp_first", False) and \
            not getattr(self, "_gate_open", False)
        if gated and raw is not None:
            fg = self.extra_info.get("sr/from_grasp")
            if fg is not None:
                # 0.95 而非课程惯用的 0.995: 毕业是一次性事件, 0.995 的半衰期 (~2M 步)
                # 会让进度条比技能本身晚数百万步; 0.95 仍要求 ~1M 步的持续高成功率.
                self._fg_slow = 0.95 * getattr(self, "_fg_slow", 0.0) + 0.05 * float(fg)
                self.writer.add_scalar("curr/gate_fg_slow", self._fg_slow, self.agent_steps)
                if self._fg_slow >= getattr(self, "_gf_target", 0.9):
                    self._gate_open = True
                    gated = False
                    raw.cfg.direct_grasp_prob = 0.5   # 放行: 回基线初值
                    self._dgp0 = 0.5                  # 退火锚点重置
                    self._sr_slow = 0.0               # 接近课程从头自然启动
                    print(f"[grasp_first] 毕业 @ {self.agent_steps/1e6:.2f}M 步 "
                          f"(抖动起点分布上的抓取慢EMA≥{self._gf_target}), 接近分支已放行")
        # ---- 组1 可行性课程: 由 **arrive_rate** 驱动 ----------------------
        # 相关能力是"到得了", 不是"抓得住", 所以驱动量用 arrive_rate 而非 success_rate.
        # 三条一起从"宽松"收到"目标", 纪律同 gentle: 慢速 EMA + 每 epoch 限速 + 棘轮.
        # ---- 用时塑形的放行闸 (2026-08-17) ----
        # 先学会到位(成功率慢EMA 站上 gate), 再逐步把用时折扣拉满。
        # 从零就打折 ⟹ 早期到不了位 -> 折扣把唯一强信号削没 -> 可能永远学不会。
        if raw is not None and getattr(self, "_time_shape", False):
            _g = float(getattr(self, "_ts_gate", 0.5))
            if self._sr_slow >= _g:
                _lam = min(1.0, float(raw.cfg.time_shape_lambda) + 0.02)
            else:
                _lam = 0.0
            if abs(_lam - float(raw.cfg.time_shape_lambda)) > 1e-9:
                raw.cfg.time_shape_lambda = _lam
                if abs(_lam - round(_lam, 1)) < 1e-9:
                    print(f"[time_shape] λ -> {_lam:.2f} @ {self.agent_steps/1e6:.1f}M "
                          f"(成功率慢EMA {self._sr_slow:.3f} ≥ {_g})")
            self.writer.add_scalar("curr/time_shape_lambda", _lam, self.agent_steps)

        # ★ 最简模式: 从这里 return, 后面三段课程(可行性/接近/退避)全部不跑。
        #   此前的 sr_ema/sr_slow 更新与 TB 写入照常, 只是没人再去改 cfg。
        #   (后面只剩接触分数图, 默认关, 跳过无副作用)
        if getattr(self, "_minimal", False):
            return
        ar = self.extra_info.get("approach/arrive_rate")
        if raw is not None and getattr(raw.cfg, "approach", False) and ar is not None \
                and not gated:
            a = getattr(self, "_ar_ema", 0.995)
            self._ar_slow = a * getattr(self, "_ar_slow", 0.0) + (1.0 - a) * float(ar)
            g = min(self._ar_slow / max(raw.cfg.curr_arrive_target, 1e-6), 1.0)
            c = raw.cfg.curr_rate
            ep_f, er_f = raw._eps_final
            # ---- 回退闸 (2026-08-16 用户裁定) ----
            # 原实现: 收紧受每 epoch 限速, **放松却是瞬时的** —— `max(目标, 当前−限速)`
            #   里 g 一掉目标就变松, max 直接取松的那个。而 g 由 arrive_rate 驱动,
            #   arrive_rate 本身在剧烈抖动(实测 0.07~0.58), 等于让噪声在开车:
            #   成功率掉 -> 公差放松 -> 策略学会"贴着桌子勉强够角度"的糙解 -> 精度上不去
            #   -> 成功率再掉。实测 eps_rot 在 5.6~10.5° 之间来回甩, 从不收敛。
            # 用户裁定: **允许退, 但要真学坏了才退** —— 不是一有波动就退。
            #   判据: 慢 EMA 连续 curr_retreat_epochs 次低于历史最好的 curr_retreat_frac。
            _best = max(getattr(self, "_ar_best", 0.0), self._ar_slow)
            self._ar_best = _best
            _bad = self._ar_slow < _best * raw.cfg.curr_retreat_frac
            self._ar_bad = (getattr(self, "_ar_bad", 0) + 1) if _bad else 0
            self._allow_loosen = self._ar_bad >= raw.cfg.curr_retreat_epochs
            if self._allow_loosen:
                self._ar_bad = 0          # 退一档就重新计数, 免得连着退
            # ★ 一次只退一个旋钮 (CLAUDE.md: 一次只改一个参数, 否则判读表失效)。
            #   优先退**起点课程**(粗旋钮, 且是实测塌掉的直接原因: far 0.66→1.0 时
            #   成功率 0.87→0); 只有起点课程已经退到底了, 才轮到放松公差。
            _start_floor = (float(getattr(raw.cfg, "stance_prob", 0.0)) <= 0.0 and
                            float(getattr(raw.cfg, "retract_ratio", 0.0)) <= 0.0)
            _allow_loosen = self._allow_loosen and _start_floor
            _tp = max(raw.cfg.eps_pos0 + (ep_f - raw.cfg.eps_pos0) * g,
                      raw.cfg.eps_pos - c * raw.cfg.eps_pos0)
            _tr = max(raw.cfg.eps_rot0 + (er_f - raw.cfg.eps_rot0) * g,
                      raw.cfg.eps_rot - c * raw.cfg.eps_rot0)
            if not _allow_loosen:         # 平时是棘轮: 只许收紧, 不许比现在更松
                _tp = min(_tp, raw.cfg.eps_pos)
                _tr = min(_tr, raw.cfg.eps_rot)
            elif _tp > raw.cfg.eps_pos or _tr > raw.cfg.eps_rot:
                print(f"[curr] 公差回退 @ {self.agent_steps/1e6:.2f}M: "
                      f"arrive 慢EMA {self._ar_slow:.3f} < 历史最好 {_best:.3f} × "
                      f"{raw.cfg.curr_retreat_frac} 连续 {raw.cfg.curr_retreat_epochs} 次 | "
                      f"eps {raw.cfg.eps_pos*100:.2f}->{_tp*100:.2f}cm "
                      f"{np.degrees(raw.cfg.eps_rot):.1f}->{np.degrees(_tr):.1f}°")
            raw.cfg.eps_pos, raw.cfg.eps_rot = _tp, _tr
            bud = raw.cfg.approach_extra0 + \
                (raw.cfg.approach_extra_steps - raw.cfg.approach_extra0) * g
            _pt0 = raw.gs + int(max(bud, raw.cfg.approach_extra_steps))
            if getattr(raw.cfg, "retract_start", False):
                # 第五犯拦截 (2026-08-26 1M-TB 抓获): 课程更新器每 epoch 按
                # gs+extra 公式重写相位0预算, 把 init 统一重装的 1500 打回 145
                # (R侧 TB 首1500→末145, arrive 因此恒0)。retract_start 体制的
                # 预算语义 = approach_only_steps, 课程只许在其上收紧, 不许击穿
                _pt0 = max(_pt0, int(raw.cfg.approach_only_steps))
            raw.phase_timeout_t[0] = _pt0
            if not getattr(self, "_no_imit", False):
                raw.cfg.w_imit_ramp = min(g, raw.cfg.w_imit_ramp + c)
            self.writer.add_scalar("curr/ar_slow", self._ar_slow, self.agent_steps)
        # 接近段课程: 直接抓取起步比例 0.5 -> 0.1, t0 上限 0.8 -> 0 (逼它最终从头做).
        # 与 gentle 同一套纪律: 慢速 EMA 定价 + 每 epoch 限速 + 棘轮只降不升.
        if raw is not None and getattr(raw.cfg, "approach", False) and not gated:
            g = min(self._sr_slow / 0.3, 1.0)          # 巩固成功率 30% 时退火到位
            # ⚠ 退火终点固定 0.1, **起点取 run 的初值** —— 原来把 0.5 写死在公式里,
            #   初值调成 0.8 时第一次更新就会被 min() 直接压回 0.5, 实验等于没做.
            if not hasattr(self, "_dgp0"):
                self._dgp0 = float(raw.cfg.direct_grasp_prob)
            raw.cfg.direct_grasp_prob = min(raw.cfg.direct_grasp_prob,
                                            max(0.1, self._dgp0 - (self._dgp0 - 0.1) * g))
            raw.cfg.approach_t0_max = min(getattr(raw.cfg, "approach_t0_max", 0.8),
                                          max(0.0, 0.8 - 0.8 * g))
            self.writer.add_scalar("curr/direct_grasp_prob", raw.cfg.direct_grasp_prob,
                                   self.agent_steps)
            self.writer.add_scalar("curr/approach_t0_max", raw.cfg.approach_t0_max,
                                   self.agent_steps)
        # ---- 退避式接近的课程 (2026-08-16, docs/APPROACH_DESIGN.md §2) ----
        # ★ 方向与上面那两条**相反**, 别混:
        #     approach_t0_max: 1 -> 0  (0 = 从头做 = 最难)
        #     retract_ratio  : 0 -> 1  (1 = d 铺满 [0,D] = 最难)
        #     stance_prob    : 0 -> 1  (1 = 全从站姿 = 正式任务口径)
        #   2026-08-16 就是把这两套方向搞混, 让评测变成"空降到抓握帧"(DESIGN_LOOP §2.24)。
        # 两段式: 先把 retract_ratio 拉满(学会各种距离的接近), 再把 stance_prob 拉满
        # (学会从站姿这个更远、姿态也不同的起点出发)。
        if raw is not None and getattr(raw.cfg, "retract_start", False):
            g = min(self._sr_slow / max(self._curr_start_target, 1e-6), 1.0)
            # 两段式 (2026-08-16 用户裁定, 恢复我误删的第②段):
            #   ① g∈[0,0.5]: retract_ratio 0→1  放宽下界, 从"只在终点"扩到"铺满整条路"
            #   ② g∈[0.5,1]: stance_prob  0→1  压低上界, 从"铺满"收到"全部最远端"
            #   bias=1 时训练分布 **完全等于** 评测分布(全部从站姿出发)。
            # ⚠ 只有第②段能让"最远那一档"从 1/84 变成 100% —— 少了它, 拉满后仍是均匀,
            #   而评测 100% 考最远档, 于是 TB 高、评测 0(2026-08-16 实测 0.37~0.55 vs 0%)。
            # ★ 起点课程也接回退闸 (2026-08-16 用户裁定)。原来是**棘轮**(只增不减):
            #   实测 far=0.66 时成功率 0.87, 一拉到 1.0 全塌成 0, 而 far 退不回去
            #   ⟹ 永远卡在"最难档 + 0 成功率", 再也学不出来。
            #   现在: 平时只增; 连续 curr_retreat_epochs 个 epoch 真退化才准退一档。
            #   ⚠ 回退必须同时压低**棘轮上限** `_cap_r/_cap_f`, 否则下一个 epoch 的
            #     `max(当前, 2g-1)` 会立刻把它弹回 1.0 —— 等于没退。
            _step = float(raw.cfg.curr_start_retreat_step)
            if not hasattr(self, "_cap_r"):
                self._cap_r, self._cap_f = 1.0, 1.0
            if getattr(self, "_allow_loosen", False) and not (
                    raw.cfg.stance_prob <= 0.0 and raw.cfg.retract_ratio <= 0.0):
                if raw.cfg.stance_prob > 0.0:      # 先退 far(第二段), 退完再退 ratio
                    self._cap_f = max(0.0, self._cap_f - _step)
                    raw.cfg.stance_prob = min(raw.cfg.stance_prob, self._cap_f)
                    _what = f"far -> {raw.cfg.stance_prob:.2f}"
                else:
                    self._cap_r = max(0.0, self._cap_r - _step)
                    raw.cfg.retract_ratio = min(raw.cfg.retract_ratio, self._cap_r)
                    _what = f"ratio -> {raw.cfg.retract_ratio:.2f}"
                print(f"[curr] 起点课程回退 @ {self.agent_steps/1e6:.2f}M: {_what} "
                      f"(arrive 慢EMA {getattr(self,'_ar_slow',0.0):.3f} 连续 "
                      f"{raw.cfg.curr_retreat_epochs} 次低于历史最好的 "
                      f"{raw.cfg.curr_retreat_frac})")
            else:
                # ★ 上限也要能**涨回来**, 否则"退一次就永远退" —— 那是把原来的
                #   只增棘轮换成只减棘轮, 同样学不出来。用户要的是"退回去学扎实,
                #   再往上冲"。恢复判据与回退对称: 慢EMA 重新站上历史最好的
                #   curr_retreat_frac 之上, 连续同样多个 epoch, 就把上限抬回一档。
                _ok = getattr(self, "_ar_slow", 0.0) >= \
                    getattr(self, "_ar_best", 0.0) * raw.cfg.curr_retreat_frac
                self._ar_good = (getattr(self, "_ar_good", 0) + 1) if _ok else 0
                if self._ar_good >= raw.cfg.curr_retreat_epochs and \
                        min(self._cap_f, self._cap_r) < 1.0:
                    self._ar_good = 0
                    if self._cap_r < 1.0:          # 恢复顺序与回退相反: 先补 ratio
                        self._cap_r = min(1.0, self._cap_r + _step)
                        _w = f"ratio上限 -> {self._cap_r:.2f}"
                    else:
                        self._cap_f = min(1.0, self._cap_f + _step)
                        _w = f"far上限 -> {self._cap_f:.2f}"
                    print(f"[curr] 起点课程恢复 @ {self.agent_steps/1e6:.2f}M: {_w} "
                          f"(arrive 慢EMA {getattr(self,'_ar_slow',0.0):.3f} 已站稳)")
            _tr_r = min(self._cap_r, 2.0 * g)
            _tr_f = min(self._cap_f, max(0.0, 2.0 * g - 1.0))
            raw.cfg.retract_ratio = max(raw.cfg.retract_ratio, _tr_r)
            raw.cfg.stance_prob = max(raw.cfg.stance_prob, _tr_f)
            self.writer.add_scalar("curr/far_bias", raw.cfg.stance_prob, self.agent_steps)
            # 上限曲线: 判读"是不是在回退/恢复"只能看这两条(far_bias 本身看不出上限)
            self.writer.add_scalar("curr/cap_far", self._cap_f, self.agent_steps)
            self.writer.add_scalar("curr/cap_ratio", self._cap_r, self.agent_steps)
            self.writer.add_scalar("curr/retract_ratio", raw.cfg.retract_ratio,
                                   self.agent_steps)

        # 接触分数图: 每 epoch 折扣一次 + 定期落盘 (常驻, 见 cfg.score_map)
        if raw is not None and getattr(raw, "score_on", False):
            raw.decay_score()
            if self.epoch_num % 100 == 0:
                raw.save_score_map(os.path.join(self.output_dir, "score_map.npz"),
                                   extra_meta=dict(epoch=int(self.epoch_num),
                                                   agent_steps=int(self.agent_steps)))
        for m in self.MILESTONES:
            key = f"{int(m * 100)}%"
            if key not in self._ms_hit and self._sr_ema >= m:
                self._ms_hit[key] = {
                    "agent_steps": int(self.agent_steps),
                    "epoch": int(self.epoch_num),
                    "wall_min": round((self.data_collect_time + self.rl_train_time) / 60, 1),
                }
                with open(self._ms_path, "w") as f:
                    json.dump(self._ms_hit, f, indent=1)
                print(f"[milestone] success_rate_ema >= {key} @ {self.agent_steps} steps")

    # ================= 自动停止 =================
    def _curriculum_done(self, raw) -> bool:
        """课程是否已全部退火到位.

        判停前**必须**过这一关: 三条课程都在把任务变难 (gentle 涨价 / 阈值收紧 /
        起点脚手架撤除), 退火期成功率平甚至跌是设计内现象, 此时判平台期必误杀。
        """
        c = raw.cfg
        if getattr(self, "_grasp_first", False) and not getattr(self, "_gate_open", False):
            return False                      # 阶段A 还没毕业
        if raw.gentle < 0.995:                # 笨拙课程 (两种任务都有)
            return False
        if not getattr(c, "approach", False):
            return True                       # grasp-only: 只有笨拙课程
        if c.direct_grasp_prob > 0.1 + 1e-6 or getattr(c, "approach_t0_max", 0.0) > 1e-6:
            return False
        if hasattr(raw, "_eps_final"):        # 单调收紧, 落点是精确的 final 值
            if c.eps_pos > raw._eps_final[0] * 1.01 or c.eps_rot > raw._eps_final[1] * 1.01:
                return False
        if not getattr(self, "_no_imit", False) and getattr(c, "w_imit_ramp", 1.0) < 0.999:
            return False
        return True

    def _curriculum_signature(self, raw):
        """所有课程变量的快照. 两次评测之间**完全没变** = 课程停摆.

        为什么需要它: 每条课程的驱动量都是成功率 (gentle←sr_slow, eps←ar_slow,
        dgp←sr_slow), 成功率卡在低位时课程**也一起冻住**, `_curriculum_done`
        永远为假 —— 于是最该早停的绝望 run (sr 卡 2%) 反而永远停不下来。
        课程没在推进时"任务变难"这个理由不成立, 平坦就是真的没学到, 允许判平台期。
        (solved 分支不放行: 课程没退火完的高成功率是在更容易的任务上刷的.)
        """
        c = raw.cfg
        return (round(float(raw.gentle), 4),
                round(float(getattr(c, "direct_grasp_prob", 0.0)), 4),
                round(float(getattr(c, "approach_t0_max", 0.0)), 4),
                round(float(getattr(c, "eps_pos", 0.0)), 6),
                round(float(getattr(c, "eps_rot", 0.0)), 6),
                round(float(getattr(c, "w_imit_ramp", 0.0)), 4),
                bool(getattr(self, "_gate_open", False)))

    @torch.no_grad()
    def deterministic_eval(self, steps: int):
        """进程内确定性评测 (只跑 mu, 无 sigma 采样), 口径 = eval.py 口径A.

        **每个 env 只记第一个完成回合** -> n = num_envs 个独立样本。不这么做的话
        (像 eval.py 现在那样在固定步窗里数所有完成回合) 会系统性**高估**: 成功回合
        短 (提前 terminated), 同样时间里能多跑几个, 窗口边界偏向成功。
        """
        raw = self._raw_env
        n, dev = raw.num_envs, raw.device
        self.set_eval()          # 同时把 running_mean_std 切 eval -> 评测数据不污染归一化统计量
        with eval_distribution(raw):
            obs = self.env.reset()
            done_once = torch.zeros(n, dtype=torch.bool, device=dev)
            succ = torch.zeros(n, dtype=torch.bool, device=dev)
            used = steps
            for t in range(steps):
                inp = {"obs": self.running_mean_std(obs["obs"]),
                       "priv_info": obs["priv_info"]}
                if self.use_pc:
                    inp["pointcloud"] = obs["pointcloud"]
                act = torch.clamp(self.model.act_inference(inp), -1.0, 1.0)
                obs, _r, done, _info = self.env.step(act)
                d = done.bool()
                succ |= d & ~done_once & raw._sig["newly_success"]
                done_once |= d
                if bool(done_once.all()):
                    used = t + 1
                    break
        # ⚠ 训练侧 obs 必须重取: 上面 reset 打断了 1024 个在跑的回合, self.obs 已失效
        self.obs = self.env.reset()
        self.set_train()
        return float(succ.float().mean()), int(done_once.sum()), used

    def phase_table_sentinel(self):
        """★ 运行时哨兵 (2026-08-26): 每 epoch 断言两侧相位表相等且容得下参考。

        为什么构造期的 `_reinstall_phase_table` + init 断言不够 ——
        `phase_timeout_t` 有**两族写手**:
          ① 构造期: env.py 的 `if cfg.approach:` / `if cfg.retract_start:` 两次写,
             中途换侧会让两侧拿到不同值(2026-08-25 e2e R=145/L=1050)。
             已由 `_reinstall_phase_table()` 后置统一 + init 断言根治。
          ② **课程期**: 本文件的 approach 课程更新器每 epoch 重写 `phase_timeout_t[0]`,
             同样是分侧写 ⟹ 构造期修好了, 跑起来又被打回去
             (2026-08-26 实测: R 侧 TB 首 1500 → 末 145, arrive 因此恒 0)。
        **init 断言只在出生时跑,拦不住运行时写手** —— 只有 epoch 边界的哨兵能全拦。

        判死条件(任一): 两侧不相等 / 预算 < 参考行数。发现即抛,不让它白跑。
        """
        raw = self._raw_env
        _ns = getattr(raw, "_ns", None)
        if _ns is None:                     # 单臂环境: 不存在归属问题
            return
        # ★ 采集段包 try: **本哨兵自己的实现 bug 不该杀掉一条 12 小时的训练**。
        #   采集失败 ⟹ 只告警并停用; 而下面"真违规"的抛错照抛(那才是它的职责)。
        try:
            import tasks.pregrasp.bimanual as _BMS
            tab, need = {}, {}
            for _nm, _sd in _ns.items():
                with _BMS.use_side(raw, _sd):
                    tab[_nm] = [int(x) for x in raw.phase_timeout_t.tolist()]
                _rp = _sd.data.get("retract_path")
                if _rp is not None:
                    need[_nm] = int(_rp.shape[0])
        except Exception as _eS:
            if not getattr(self, "_pts_broken", False):
                self._pts_broken = True
                print(f"[哨兵] ⚠ 采集失败, 哨兵停用 (不影响训练): {_eS!r}")
            return
        _vals = list(tab.values())
        if len(_vals) >= 2 and _vals[0] != _vals[1]:
            raise RuntimeError(
                f"[哨兵] 两侧 phase_timeout_t 在运行期变得不一致: {tab}\n"
                f"  ⟹ 短的那侧钟先响, 纯前馈走到半路被重置, 双侧 arrive 门永不齐。\n"
                f"  查谁在 epoch 边界重写它 (构造期已由 _reinstall_phase_table 兜底, "
                f"这里报警说明是**课程期**写手)。")
        for _nm, _n in need.items():
            if tab[_nm][0] < _n:
                raise RuntimeError(
                    f"[哨兵] {_nm} 接近预算 {tab[_nm][0]} < 参考行数 {_n} "
                    f"⟹ 纯前馈永远走不完, arrive 恒 0。")
        if not getattr(self, "_pts_logged", False):
            self._pts_logged = True
            print(f"[哨兵] 相位表运行时哨兵已上岗: "
                  + " | ".join(f"{k} PREGRASP={v[0]}" for k, v in tab.items())
                  + (f" | 参考 {need}" if need else ""))
        for _nm, _v in tab.items():
            self.writer.add_scalar(f"{_nm[0].upper()}/sentinel/pregrasp_budget",
                                   float(_v[0]), self.agent_steps)

    def maybe_eval_and_stop(self):
        """epoch 边界钩子 (ckpt 刚落盘的干净点). 由 train.py 与 gpu_guard 让出点串联."""
        if self._auto_stop == "off" or self.epoch_num % self._eval_every:
            return
        raw = self._raw_env
        curr_ok = self._curriculum_done(raw)
        sig = self._curriculum_signature(raw)
        stalled = (sig == self._last_curr_sig)      # 首次评测 _last=None -> False
        self._last_curr_sig = sig
        sr, n_done, used = self.deterministic_eval(self._eval_steps)
        self.writer.add_scalar("eval/success_rate", sr, self.agent_steps)
        self.writer.add_scalar("eval/curriculum_done", float(curr_ok), self.agent_steps)
        self.writer.add_scalar("eval/curriculum_stalled", float(stalled), self.agent_steps)
        self._eval_hist.append(dict(agent_steps=int(self.agent_steps), epoch=int(self.epoch_num),
                                    sr=round(sr, 4), curriculum_done=bool(curr_ok),
                                    curriculum_stalled=bool(stalled)))
        if n_done < raw.num_envs:
            print(f"[eval] ⚠ 只有 {n_done}/{raw.num_envs} 个 env 在 {used} 步内跑完一回合 —— "
                  f"--eval_steps 需要 > 回合上限", flush=True)

        st = self._stopper
        was_best = sr > st.best_sr
        reason = st.update(sr, curr_ok, stalled, self.agent_steps)
        if was_best:
            # ⚠ 与基类的 best.pth 不同: 那个按 mean_rewards 存, 多项奖励里回报高 ≠ 成功率高
            self.save(os.path.join(self.nn_dir, "eval_best"))

        # flush: Isaac 的 C 层和 Python 共用 fd 但缓冲策略不同, 不 flush 的话这几行会被
        # 吞掉 (冒烟实测 49 条进度只落盘 5 条). 判停结论也落 auto_stop.json + TB, 双保险.
        print(f"[eval] ep{self.epoch_num} {self.agent_steps/1e6:.1f}M | 确定性成功率 "
              f"{sr*100:6.2f}% (n={n_done}) | 最佳 {st.best_sr*100:.2f}% | "
              f"课程{'已到位' if curr_ok else ('停摆' if stalled else '推进中')} | "
              f"达标{st.hits} 平台{st.stale}", flush=True)
        with open(self._stop_path, "w") as f:
            json.dump(dict(mode=self._auto_stop, best_sr=st.best_sr,
                           fired=st.fired, history=self._eval_hist), f, indent=1)
        if reason:
            print(f"[auto_stop] 触发 @ {self.agent_steps/1e6:.2f}M 步 ({self.epoch_num} epoch): {reason}",
                  flush=True)
            if self._auto_stop == "on":
                # 不动基类循环结构: 把上限拉到当前步数, while 条件下一轮自然为假
                self.max_agent_steps = self.agent_steps
                print("[auto_stop] 结束训练 (最佳权重在 stage1_nn/eval_best.pth)", flush=True)
            else:
                print("[auto_stop] dry-run: **只报告不停止**, 继续训练以便对照判据是否过早/过晚",
                      flush=True)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
if args.obj_jitter is not None:
    env_cfg.obj_jitter_xy = args.obj_jitter
if args.direct_grasp_prob is not None:
    env_cfg.direct_grasp_prob = args.direct_grasp_prob
if args.ref_look is not None:
    env_cfg.ref_look_frames = args.ref_look
if args.time_shape:
    env_cfg.time_shape_ref_steps = float(args.time_shape_ref)
    print(f"[实验] 用时塑形: 基准 {args.time_shape_ref:.0f} 步, "
          f"decay {env_cfg.time_shape_decay}, 成功率≥{args.time_shape_gate} 才放行 "
          f"| 94步→{100*env_cfg.time_shape_decay**0:.0f}分 "
          f"150步→{100*env_cfg.time_shape_decay**(150-args.time_shape_ref):.0f}分 "
          f"211步→{100*env_cfg.time_shape_decay**(211-args.time_shape_ref):.0f}分")
if args.start_pool:
    env_cfg.start_pool = args.start_pool
    print(f"[实验] 起点池 = {args.start_pool}")
if args.minimal:
    # ---- 最简接近训练 (2026-08-16 用户裁定) --------------------------------
    # 目的: 把课程/退火/参考前馈这些机器全部拿掉, 只留"越近越有奖 + 撞了就死",
    #       看纯密集奖励能不能自己把手从站姿开到 GraspPose 1cm 内。
    env_cfg.minimal_no_ff = True        # 无前馈: 手全靠策略自己走 (不播参考路径)
    # ★ 残差步长: 默认**保留动态缩放**(远松近紧)。
    #   2026-08-17 教训: 第一版 minimal 把它也关了(minimal_fixed_res=True, 恒定 1x =
    #   全程 10cm/s), 而到位判据要求腕速 <5cm/s ⟹ **结构上永远过不了**。
    #   Minimal_L 跑满 40M 步卡在 3.4cm, 测的不是"纯密集奖励行不行", 而是
    #   "一个停不下来的策略行不行"。用 --minimal_fixed_res 可以显式退回旧行为做对照。
    env_cfg.minimal_fixed_res = bool(args.minimal_fixed_res)
    if args.minimal_fixed_res:
        env_cfg.dyn_arm_near = 1.0
        print("[minimal] ⚠ 残差步长**恒定** 1x —— 近目标处仍是 10cm/s, 判据要 <5cm/s")
    # ⚠ stance_prob/retract_ratio 不在这里设 —— 下面 approach_only/retract 块会
    #   把它们覆盖回 0.0。统一挪到 env 构造前的"最终落定"块 (见下)。
    env_cfg.eps_pos = 0.01              # 到位 = 腕位置差 < 1cm
    env_cfg.eps_rot = np.radians(15.0)  # 朝向放宽到 15° (实测轻松达标, 不做约束)
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot  # 无退火
    env_cfg.w_imit0_approach = 0.0      # 不要求像人
    env_cfg.w_imit_ramp = 0.0
    args.no_imit = True
    # ⚠ 横幅必须说**实际行为**: 上一版写死"残差恒定 {near}x", 而默认已改成动态缩放,
    #   near 只是**近端**倍率 —— 又一次"打印的和生效的不一致"。
    _rd = (f"恒定 {env_cfg.dyn_arm_near:.1f}x = {env_cfg.dyn_arm_near*10:.0f}cm/s **全程**"
           if env_cfg.minimal_fixed_res else
           f"动态 {env_cfg.dyn_arm_near:.1f}x(≤2cm) -> {env_cfg.dyn_arm_far:.1f}x(≥15cm) "
           f"= {env_cfg.dyn_arm_near*10:.0f}~{env_cfg.dyn_arm_far*10:.0f}cm/s")
    print("[minimal] 最简接近训练: 起点=站姿(固定) | 无前馈 | 无课程无退火 | "
          f"残差{_rd} | 到位=1cm/15° | imit=0")
if args.no_eps_curr:                 # E 组: 阈值不放松, 从一开始就是最终值
    env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
# ---- 五条对照实验的单变量覆盖 (必须在 GraspTaskEnv 构造前, 因为 env 在
#      __init__ 里把 (eps_pos, eps_rot) 快照成 _eps_final) ----
if args.eps_final_cm is not None:
    env_cfg.eps_pos = args.eps_final_cm / 100.0
    print(f"[实验] 到位位置公差终值覆盖为 {args.eps_final_cm:.2f}cm")
if args.eps_final_deg is not None:
    env_cfg.eps_rot = np.radians(args.eps_final_deg)
    print(f"[实验] 到位朝向公差终值覆盖为 {args.eps_final_deg:.1f}°")
if args.r_reach is not None:
    env_cfg.r_reach = args.r_reach
    print(f"[实验] r_reach 覆盖为 {args.r_reach}")
if args.dyn_arm_near is not None:
    env_cfg.dyn_arm_near = args.dyn_arm_near
    print(f"[实验] dyn_arm_near 覆盖为 {args.dyn_arm_near}")
env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.orient_blend = args.orient_blend
env_cfg.cone_trust = args.cone
if args.grasp_first:      # 阶段 A: 100% 直接抓取回合 (毕业后训练钩子放行回 0.5)
    assert args.approach, "--grasp_first 是接近任务的课程, 必须配 --approach"
    env_cfg.direct_grasp_prob = 1.0
if args.place:
    assert args.approach, "--place 基于端到端任务, 必须配 --approach"
    env_cfg.place_task = True
    env_cfg.episode_length_s = 20.0   # 回合变长 (搬运+放置 ~100 步), 抬高兜底上限
    # grasp_only 时代把 gs 后的腕参考冻结 (物体钉桌上, 腕飘走=抓空气); place 的物体
    # 是真实物理且就该被搬走 —— 解冻, 参考播完整的 靠近→抓住→搬运→放置 (ref builder 注释)
    env_cfg.freeze_wrist = False
def _fatal_cfg(msg):
    """旗参数致命错误 —— 必须用 os._exit。

    ★ 2026-08-29 (RL 会话实测教训): AppLauncher(第 386 行) 之后 `raise SystemExit`
    **会被 Isaac 吞掉** —— 退出码变 0 且训练照常跑下去, 等于拿着错配置白跑一整轮。
    "报了警却照样跑" 比不设闸更危险 (看日志的人会以为闸放行了)。
    """
    import sys as _s
    print("=" * 78, flush=True)
    print(f"[FATAL] {msg}", flush=True)
    print("=" * 78, flush=True)
    _s.stdout.flush(); _s.stderr.flush()
    os._exit(2)

_prior = args.prior_npz or (os.path.join(_HERE, "priors", f"{args.clip}.npz")
                            if args.grasp_prior else None)
if _prior:
    assert os.path.exists(_prior), f"缺 prior 文件 {_prior} (先跑 make_prior.py / screen_prior.py)"
    # 一次做齐三件事: 挂 prior / 关 pregrasp_align(D7) / 钉 yaw(D1). 见 cfg.apply_grasp_prior
    if args.approach_only:
        # ⚠ 必须在 apply_grasp_prior **之前** —— 动作空间(7)与观测维度都依赖它,
        #   apply_grasp_prior 里就会按 _obs_base(cfg) 把 observation_space 算死。
        env_cfg.approach_only = True
        env_cfg.action_space = 7
    if getattr(args, "pregrasp_only", False) or getattr(args, "e2e_squeeze", False):
        # e2e_squeeze (2026-08-25): 借道本分支拿全套 pour 旗接线, 但走完整抓取链
        #   (approach_only=False) —— pregrasp_only 旗集行为逐字节不变
        # 裁定 (2026-08-18 晚, 用户): PreGrasp29 接近任务 ——
        #   ① 靶点=掌心 PreGrasp; 双手同动同学, 双侧稳定到位 = 成功终止 (复用
        #     approach_only 语义, 避免"学好了干等");
        #   ② 22 指全开 (需 RL_HAND_JOINTS=1, 29 维/侧), 但到位前 fin_quiet 软抑制;
        #   ③ 到位后由 switch_hold 持续判据天然鼓励定住。
        assert os.environ.get("RL_HAND_JOINTS") == "1", \
            "--pregrasp_only 需要 RL_HAND_JOINTS=1 (22 指全开是任务定义的一部分)"
        env_cfg.approach_only = not getattr(args, "e2e_squeeze", False)
        # ↑ e2e: False=到位切 GRASP 走完整抓取链; 其余模式 True=到位即成功终止
        env_cfg.pregrasp29 = True         # 靶点换 PreGrasp + 指残差解锁 + fin_quiet
        if getattr(args, "phase2", False):
            env_cfg.pregrasp_phase2 = True   # 到位后名义斜坡驱动到 GraspPose
        if getattr(args, "fin_cart", False):
            env_cfg.fin_cart = True
            env_cfg.fin_pot = float(args.fin_pot)
            print(f"[fin_cart] 逐指笛卡尔目标 ON | 势差分 fin_pot={args.fin_pot} | "
                  f"扰动罚 w={env_cfg.w_obj_disturb} 全程在")
        if getattr(args, "fin_pose1", False):
            env_cfg.fin_start_pose1 = True
            print("[fin_pose1] reset 指型 = Pose1 (每侧先验 pregrasp[0], 拇指对掌)")
        if getattr(args, "fcd", False):
            env_cfg.fin_ref_track = True
            env_cfg.table_touch_fail = True
            assert not getattr(args, "fin_pose1", False), \
                "FC-D 与 --fin_pose1 互斥: 裁定① reset=Pose0, 构型由参考段教"
            print("[fcd] 四段指参考 ON | 慢合拢闸 近0.3x/合拢0.2x | "
                  "Pad/向心/稳抓奖励 ON | 撞桌即Fail ON (tol "
                  f"{env_cfg.table_touch_tol*1000:.0f}mm)")
        if float(getattr(args, "w_toppled", 0.0)) > 0.0:
            env_cfg.w_toppled = float(args.w_toppled)
            print(f"[fcd] 拍倒显式罚 ON: -{env_cfg.w_toppled}/次 (独立项, fail 仍关)")
        if getattr(args, "step_rew_log", ""):
            env_cfg.step_reward_log = os.path.abspath(args.step_rew_log)
            print(f"[steprew] 逐步奖惩记录 ON -> {env_cfg.step_reward_log}")
        if getattr(args, "fc_earn", False):
            env_cfg.fc_pot_earn_only = True
            print("[基线归零] fc_pot earn-only ON (只奖进步, 不罚参考自身的 squeeze 压深)")
        if float(getattr(args, "fin_gate_dpos_cm", 0.0)) > 0:
            env_cfg.fin_gate_dpos_cm = float(args.fin_gate_dpos_cm)
            print(f"[软闸] 指参考行进门: 腕距 < {env_cfg.fin_gate_dpos_cm}cm 即放行 "
                  f"(原=只认 arrived 闩死, 断崖)")
        if getattr(args, "cent_earn", False):
            env_cfg.cent_earn_only = True
            print("[基线归零] 向心塑形 earn-only ON (稠密负罚会教出'别碰物体')")
        if float(getattr(args, "fin_dev_scale", -1.0)) > 0:
            env_cfg.phase2_fin_dev_scale = float(args.fin_dev_scale)
            env_cfg.finger_dev_scale = float(args.fin_dev_scale)
            print(f"[方案C-指] 手指残差累计上限 x{args.fin_dev_scale} "
                  f"(原 40 = 中位 316° = 无界积分器, 已判死)")
        if getattr(args, "qvel_settle", False):
            env_cfg.qvel_settle_exempt = True
            print("[基线归零] 冻结窗口内免罚关节速度 ON")
        if float(getattr(args, "qd_soft_fin", -1.0)) > 0:
            env_cfg.qd_soft_fin = float(args.qd_soft_fin)
            print(f"[基线归零] 指软速度阈 -> {env_cfg.qd_soft_fin} rad/s")
        if float(getattr(args, "qd_soft_arm", -1.0)) > 0:
            env_cfg.qd_soft_arm = float(args.qd_soft_arm)
            print(f"[基线归零] 臂软速度阈 -> {env_cfg.qd_soft_arm} rad/s")
        if getattr(args, "arm_abs", False):
            env_cfg.arm_abs_res = True
            env_cfg.arm_abs_dev = float(args.arm_abs_dev_deg) * 3.141592653589793 / 180.0
            print(f"[方案C] 臂绝对参考+有界残差 ON: |arm_res| <= "
                  f"{args.arm_abs_dev_deg:.2f}deg ({env_cfg.arm_abs_dev:.4f}rad) "
                  f"| 差分前馈积分器退役 (ff_pull 在此模式下不参与计算)")
        if getattr(args, "aag", False):
            assert getattr(args, "fcd", False), "--aag 需配 --fcd (fin_ref_track 机制)"
            env_cfg.fin_ref_npz = os.path.abspath(args.curobo_ref)
            env_cfg.approach_extra0 = 400      # 参考 360 行, PREGRASP 预算 gs+400
            # ★ 2026-08-24: 三个行号常数改成**从参考自身的段表推导**。
            # 原来写死 120/90/175 是按"360行/stride2=180行制"标的; 侧移版参考多了
            # insert 段变成 380 行 ⟹ 全部错位。这类"改了参考忘了改常数"是本项目
            # 反复出问题的模式, 用推导彻底根治。
            _zs = np.load(os.path.abspath(args.curobo_ref), allow_pickle=True)
            _stz = max(1, int(getattr(env_cfg, "curobo_ref_stride", 2)))
            if "seg_names" in _zs.files and "seg_lens" in _zs.files:
                _acc, _segz = 0, {}
                for _sn, _sl in zip(_zs["seg_names"], _zs["seg_lens"]):
                    _segz[str(_sn)] = (_acc // _stz, (_acc + int(_sl) - 1) // _stz)
                    _acc += int(_sl)
                env_cfg.aag_grasp_row = _segz["close"][0]        # 合拢段起始
                env_cfg.fin_hold_row = _segz["root_bend"][1]     # 弯根部末行
                env_cfg.fin_end_row = _segz["squeeze"][1]        # 指参考推进上限
                env_cfg.squeeze_row0 = _segz["squeeze"][0]       # ⑤ 抓力奖励窗口起点
                env_cfg._seg_count = len(_segz)
                print(f"[aag] 行号由段表推导 (共 {len(_segz)} 段, stride {_stz}): "
                      f"aag_grasp_row={env_cfg.aag_grasp_row} "
                      f"fin_hold_row={env_cfg.fin_hold_row} "
                      f"fin_end_row={env_cfg.fin_end_row}")
            else:
                env_cfg.aag_grasp_row = 120    # 老参考无段表, 回退写死值
                print("[aag] ⚠ 参考无段表, 行号回退写死值 120/90/175")
            if getattr(args, "fin_slow", False):
                env_cfg.curobo_ref_stride = 1  # 指参考 360 行 (原速)
                env_cfg.aag_grasp_row = 240    # 行号随之翻倍
                env_cfg.fin_end_row = 350
                env_cfg.fin_hold_row = 180
                print("[AAG-L] 指参考半速 ON: stride 1 (360行), 合拢起始行 240, "
                      "上限 350 —— 治 2x 快播的速度罚")
            env_cfg.near_exempt_m = 0.07       # 弯指前进段指尖超前腕, 豁免圈随编舞放宽
            print(f"[aag] 标准成功轨迹参考 ON: {os.path.basename(args.curobo_ref)} "
                  f"(指参考逐行含 squeeze 深拇段 | 1cm 契约退役 | 预算 gs+400)")
        if getattr(args, "fc_ref_end", False):
            env_cfg.fc_target_ref_end = True
            print("[fin_cart] 逐指目标点 = **参考终态** (原=env自解位形, 差 臂33.5°/指37.6°)")
        if getattr(args, "fc_squeeze", False):
            env_cfg.fc_target_squeeze = True
            print("[fin_cart] 逐指目标点 = squeeze (原 grasp; 参考终态是 squeeze)")
        if float(getattr(args, "squeeze_grip", 0.0)) > 0:
            env_cfg.squeeze_grip_w = float(args.squeeze_grip)
            print(f"[⑤抓力] squeeze 段 ON: w={env_cfg.squeeze_grip_w} | "
                  f"pad_near(垫→表面) + grip_pot(quality) 双项 earn-only")
        if getattr(args, "seg_gate", False):
            env_cfg.seg_gate = True
            print(f"[分段门控] ON | 臂 {env_cfg.seg_arm_scale} | "
                  f"拇指 {env_cfg.seg_thumb_scale} | 其余指 {env_cfg.seg_other_scale}")
        if float(getattr(args, "pre_close_shell_cm", 0.0)) > 0:
            env_cfg.pre_close_shell_cm = float(args.pre_close_shell_cm)
            print(f"[合拢前禁触] 外壳-物体最小间隙 {env_cfg.pre_close_shell_cm}cm")
        if float(getattr(args, "pre_close", 0.0)) > 0:
            env_cfg.pre_close_w = float(args.pre_close)
            env_cfg.pre_close_dead_cm = float(args.pre_close_dead_cm)
            env_cfg.pre_close_tilt_deg = float(args.pre_close_tilt_deg)
            print(f"[合拢前禁触] w={env_cfg.pre_close_w} 死区={env_cfg.pre_close_dead_cm}cm "
                  f"倾角死区={env_cfg.pre_close_tilt_deg}° "
                  f"| 窗口 ref_t < {getattr(env_cfg,'aag_grasp_row','?')} (close 段起始)")
        if getattr(args, "pours_v5", False):
            assert getattr(args, "pour_free", False) \
                and getattr(args, "carry_stable", False) \
                and getattr(args, "carry_pad_reward", False), \
                "--pours_v5 需配 --pour_free --carry_stable --carry_pad_reward"
            if args.load_path:
                # v5/v6 同代 ckpt 续训合法 (obs 同维); 误载 v4 及更早的旧 ckpt 会在
                # 加载时因形状不符炸出来, 无需在此一刀切 (2026-08-22: 原 assert 把
                # v6→v6 加进度门的合法续训也拦了)
                print("[pours-v5] ⚠ 续训模式: 请确认 ckpt 来自同代 v5/v6 run "
                      "(旧代 ckpt 会在加载时报形状不符)")
            env_cfg.pours_v5 = True
            env_cfg.carry_slip_pos = 0.08          # 死线让位给连续梯度 (4cm→8cm)
            env_cfg.carry_slip_rot = np.radians(60.0)          # 30°→60°
            env_cfg.max_obj_height = 0.28          # 参考峰值 18cm + 用户规矩 10cm
            # (observation_space 的 +8 在 bimanual 翻倍前做 —— apply_grasp_prior
            #  会在本块之后按 _obs_base 重算, 在这里加会被抹掉, 冒烟实锤)
            print(f"[pours-v5] 抓稳优先 ON: 滑移逐步增量罚 w={env_cfg.slip_step_w} + "
                  f"握紧反射 w={env_cfg.regrip_w} + 持续抓稳 w={env_cfg.grip_hold_w}"
                  f"(F_ref {env_cfg.grip_f_ref}N cap {env_cfg.grip_f_max}N) + "
                  f"杯直立 w={env_cfg.cup_upright_w}(免罚 {env_cfg.cup_tilt_free_deg}°) + "
                  f"互碰罚 w={env_cfg.cross_pen_w}(边距 {env_cfg.cross_margin*100:.0f}cm) | "
                  f"死线 8cm/60° | thrown 28cm | obs+8/侧")
        if getattr(args, "pours_v6", False):
            assert getattr(args, "pours_v5", False), "--pours_v6 需配 --pours_v5"
            env_cfg.pours_v6 = True
            env_cfg.pour_succ_tilt_deg = 93.0   # 参考瓶倾角平台103°×0.9 (滤波+平台法)
            print(f"[pours-v6] 阶段纯净化 ON: toppled 交互豁免 | thrown 退役→偏轨"
                  f"{env_cfg.dev_reset_m*100:.0f}cm 硬闸 | cent 仅启动前 | 杯直立删 | "
                  f"cross 纯碰撞(5mm/96点) | 倒水成功=终止(不撤离) | "
                  f"成功线 {env_cfg.pour_succ_tilt_deg:.0f}° | obs+10/侧(含阶段)")
        if getattr(args, "arm_ff_gate", False):
            assert getattr(args, "aag", False), "--arm_ff_gate 需配 --aag (纯前馈依赖标准成功轨迹参考)"
            env_cfg.arm_ff_gate = True
            print(f"[aag-v3.1] 臂残差门控 ON: 到位前 {env_cfg.arm_ff_pre:.2f}x 小带"
                  f"(微调避蹭+补两把尺缺口), 到位后 ramp {env_cfg.arm_ff_ramp} 步升到 "
                  f"{env_cfg.arm_ff_post:.2f}x (≈±2°)")
        if getattr(args, "pour_trend", ""):
            assert os.path.exists(args.pour_trend), f"缺趋势文件 {args.pour_trend}"
            env_cfg.pour_trend_npz = os.path.abspath(args.pour_trend)
            print(f"[v7] 趋势钟接线: {os.path.basename(args.pour_trend)} "
                  f"(容差 {env_cfg.pour_trend_tol_deg}° 微糖 {env_cfg.pour_trend_w}/步)")
        if getattr(args, "pad_gate", False):
            env_cfg.pad_first_arrived_only = True
            print("[v3.3] pad_first 碰垫糖: 到位后才发 (剥夺'不进门'的工资)")
        if float(getattr(args, "dgp_rsi_hi", 0.0)) > 0.0:
            env_cfg.dgp_rsi_hi = float(args.dgp_rsi_hi)
            print(f"[AAG-L] RSI 课程 ON: 直抓起步 {env_cfg.dgp_rsi_hi:.2f} → "
                  f"{float(args.direct_grasp_prob or 0.0):.2f} (随稳抓能力棘轮退火;"
                  f" 终值以 minimal 落定为准)")
        if float(getattr(args, "obj_disturb_pre", 1.0)) != 1.0:
            env_cfg.obj_disturb_pre_x = float(args.obj_disturb_pre)
            print(f"[AAG-L] 到位前扰动罚 ×{env_cfg.obj_disturb_pre_x:.0f} "
                  f"(={env_cfg.w_obj_disturb * env_cfg.obj_disturb_pre_x:.1f}/单位速度)")
        if getattr(args, "fin_gate", False):
            env_cfg.form_pot_earn_only = True   # 与行进门同捆 (语义: 只赚不罚)
        if getattr(args, "fin_fade", False):
            env_cfg.fin_track_contact_fade = True
            print("[AAG-L] 指参考接触淡出 ON: 每多一个触垫, fin_track 拉力降 1/n_active")
        if getattr(args, "fin_gate", False):
            env_cfg.fin_prog_gate = True
            print(f"[AAG-L] 手指行进门 ON: 到位前指参考钳在行 {env_cfg.fin_hold_row} "
                  f"(弯根部末) —— 腕没到, 合拢时钟不走")
        if getattr(args, "rew_sched", False):
            env_cfg.rew_sched = True
            if int(getattr(args, "rew_sched_ramp", -1)) >= 0:
                env_cfg.rew_sched_ramp = int(args.rew_sched_ramp)
            print(f"[rew_sched] 相位奖励日程表 ON: PREGRASP 置零 "
                  f"{list(env_cfg.rew_sched_pre)} | 进 GRASP 渐入 "
                  f"{env_cfg.rew_sched_ramp} 步 | GRASP 降权 "
                  f"{dict(env_cfg.rew_sched_grasp_scale)}")
        if getattr(args, "grip_g", False):
            env_cfg.grip_g = True
            print(f"[grip_g] 握力信任标量 ON: ④段慢档 {env_cfg.grip_g_up*env_cfg.grip_g_slow_frac:.5f}/步 "
                  f"| 滑移>{env_cfg.grip_g_slip_cm}cm 掉 {env_cfg.grip_g_down_mult}× "
                  f"| 衰减项 {list(env_cfg.grip_g_decay_terms)} (×(1−g))")
        if getattr(args, "obj_obs", False):
            env_cfg.obj_in_actor = True
            print("[AAG-L] 物体特权进 actor ON (+13/侧: 腕系位置3+相对姿态4+线速3+角速3)")
        if float(getattr(args, "form_pot", 0.0)) > 0.0:
            env_cfg.fin_form_pot_w = float(args.form_pot)
            print(f"[v3.4] 形态正向势差分 ON: w={env_cfg.fin_form_pot_w} "
                  f"(candidate 前, 向参考形态推进给小钱)")
        if float(getattr(args, "arrive_bonus", -1.0)) > 0.0:
            env_cfg.r_arrive = float(args.arrive_bonus)
            print(f"[v3.4] 到位大奖加码: r_arrive={env_cfg.r_arrive}")
        if float(getattr(args, "prog_pot", 0.0)) > 0.0:
            env_cfg.pour_prog_w = float(args.prog_pot)
            print(f"[v6P] 全局进度势 ON: w={env_cfg.pour_prog_w} "
                  f"(φ=复现行数/{env_cfg.pour_free_hi}, 自由段用倾角棘轮折算; "
                  f"0→100% 共发 {env_cfg.pour_prog_w})")
        if float(getattr(args, "mouth_w", 0.0)) > 0.0:
            env_cfg.pour_free_w_mouth = float(args.mouth_w)
            print(f"[v8] 口对口势权重覆盖: {env_cfg.pour_free_w_mouth}")
        if float(getattr(args, "ff_tol", 0.0)) > 0.0:
            env_cfg.pour_ff_tol_deg = float(args.ff_tol)
            print(f"[v8] 倾角行进门容差覆盖: {env_cfg.pour_ff_tol_deg}°")
        if getattr(args, "prog_align", False):
            env_cfg.pour_prog_align = True
        if getattr(args, "prog_joint", False):
            assert getattr(args, "prog_align", False), "--prog_joint 需配 --prog_align"
            env_cfg.pour_prog_joint = True
        if getattr(args, "e2e_squeeze", False):
            assert getattr(args, "carry_progress", False), "--e2e_squeeze 需 --carry_progress"
            assert not getattr(args, "pregrasp_only", False), \
                "--e2e_squeeze 走完整抓取链, 不配 --pregrasp_only (用 --pregrasp_grasp)"
            env_cfg.pour_e2e = True
            env_cfg.ref_start_is_stance = True
            env_cfg.success_nonterminal = True     # 真抓稳=carry启动门, 不终止
            env_cfg.fail_tilt_deg = 30.0           # 接近段拍倒判负; 启动后豁免
            # 60→30 (2026-08-26 r8e 定案): tilt 罚在死区~60° 间每步 −2 无界流血
            # (撞歪不倒的物体整回合吃满罚, 比标定量级大165倍) —— 30° 即终止,
            # 失血变一次性 −10, 练"别撞倒"而非"撞倒后忍受"。倒水段 93-127° 由
            # started 豁免伞覆盖 (env.py:2190 路径已核通)
            _ze2 = np.load(os.path.abspath(args.carry_npz), allow_pickle=True)
            env_cfg.carry_ref_off = int(_ze2["seam"]) if "seam" in _ze2.files else 0
            env_cfg.retract_start = True       # 双臂防雷断言 (B侧q_ref重建)
            env_cfg.stance_prob = 1.0          # 出生=站姿桶 t0=0 沿接近参考
            env_cfg.retract_ratio = 1.0
            env_cfg.carry_start_deadline = 500     # 接近~380行 + 抓稳链 + 余量
            env_cfg.approach_only_steps = 1100
            print(f"[e2e-sq] 全链 squeeze 体制 ON: seam={env_cfg.carry_ref_off} "
                  f"启动限期{env_cfg.carry_start_deadline} 预算{env_cfg.approach_only_steps}")
        if getattr(args, "pour_place", False):
            env_cfg.pour_place = True
            if getattr(args, "pour_retreat", False):
                env_cfg.pour_retreat = True
            if float(getattr(args, "asym_pen", 0.0)) > 0.0:
                env_cfg.asym_pen_w = float(args.asym_pen)
                print(f"[e2e] 双侧收入不对称罚 ON: λ={env_cfg.asym_pen_w}")
                print(f"[e2e] RETREAT ON: 松手后臂 lerp 回站姿 "
                      f"{env_cfg.pour_ret_steps}步, 撤完=全链终点(+10)")
            print(f"[v10] PLACE+RELEASE ON: ★降级里程碑→出段续追 | 放稳(+20)→"
                  f"松手斜坡{env_cfg.pour_rel_steps}步(乘积棘轮 w={env_cfg.pour_rel_w})"
                  f"→静置{env_cfg.pour_rel_hold}步=成功(+10)")
            print("[v9J] 联合乘积棘轮 ON: 进度=每步(倾角分×对齐分)历史最大 (同框写进进度)")
            print("[v9] 进度势改判: 窗内进度 = 倾角 × 对齐 (乘积, 治'朝任意方向倾')")
        if getattr(args, "ff_play", False):
            env_cfg.pour_ff_play = True
            print(f"[v8] 自由段臂前馈续播 ON (行进门=倾角剖面 ±{env_cfg.pour_ff_tol_deg}°)")
        if getattr(args, "tilt_prog", False):
            env_cfg.pour_tilt_prog = True
            print("[v6T] 自由段进度改判 ON: 抓稳粮只在**倾角创新高**时发 "
                  "(窗内挂机模式关闭)")
        if float(getattr(args, "tilt_pot", 0.0)) > 0.0:
            env_cfg.pour_tilt_pot_w = float(args.tilt_pot)
            print(f"[v6c] 自由段倾角势差分 ON: w={env_cfg.pour_tilt_pot_w} "
                  f"(telescoping, 满倾角累计+{env_cfg.pour_tilt_pot_w:.0f})")
        if int(getattr(args, "grip_prog_gate", 0)) > 0:
            env_cfg.grip_prog_gate = int(args.grip_prog_gate)
            print(f"[v6-补丁] grip_hold 进度门 ON: 近 {env_cfg.grip_prog_gate} 步"
                  f"时钟未前进停发 (pf 窗豁免) —— 行军粮不是养老金")
        if float(getattr(args, "arm_pre_left", -1.0)) >= 0.0:
            env_cfg.arm_ff_pre_left = float(args.arm_pre_left)
            print(f"[消融1] 左臂 pre 带宽覆盖: {env_cfg.arm_ff_pre_left:.2f}x")
        if float(getattr(args, "fin_near_relax", 0.0)) > 0.0:
            env_cfg.fin_near_relax = float(args.fin_near_relax)
            print(f"[消融2] 指拘束近场松绑: 腕距<{env_cfg.fin_near_m*100:.0f}cm 时 "
                  f"fin_track/fin_quiet ×{env_cfg.fin_near_relax:.2f}")
        if float(getattr(args, "m1_cm", -1.0)) >= 0.0:
            env_cfg.phase2_m1_cm = float(args.m1_cm)   # 0 ⟹ 门恒假 = m1 不存在
            print(f"[phase2] m1 门覆盖: {env_cfg.phase2_m1_cm}cm"
                  + (" (=删除 m1, 简化二)" if env_cfg.phase2_m1_cm == 0 else ""))
    if getattr(args, "pour_carry", False):
        # Carry (2026-08-19 用户批准): 复用 pregrasp29+phase2 全机制, 判据/奖励在
        # PourCarryEnv 覆盖。基座的碰物/推移/倾倒判据按任务豁免 (携带必然碰物、
        # 瓶必然翻 113°), 失败改由 slip/跟丢/fell 承担。
        env_cfg.pour_carry = True
        env_cfg.carry_npz = os.path.abspath(args.carry_npz)
        env_cfg.ref_start_is_stance = False   # 参考首行 = 抓姿 IK (不是站姿)
        env_cfg.fail_tilt_deg = 999.0
        env_cfg.approach_hit_obj_m = 0.0
        env_cfg.push_fail_dist = 10.0
        env_cfg.push_fail_dist_loose = 10.0
        env_cfg.approach_only_steps = 230     # 热身40+路径142+余量
        if getattr(args, "carry_tilt_ms", ""):
            env_cfg.carry_tilt_ms_deg = tuple(
                float(x) for x in args.carry_tilt_ms.split(","))
            print(f"[carry4] 倾角里程碑: {env_cfg.carry_tilt_ms_deg}°")
        if getattr(args, "carry_ms_low", ""):
            env_cfg.carry_ms_low_idx = tuple(
                int(x) for x in args.carry_ms_low.split(","))
            print(f"[carry4] 低置信小糖宽门: 行 {env_cfg.carry_ms_low_idx}")
        if getattr(args, "carry_squeeze", False):
            env_cfg.carry_squeeze = True
            print("[carry4] q_close ← squeeze 模板 (收紧轴交给策略)")
        if getattr(args, "carry_pad_reward", False):
            env_cfg.carry_pad_reward = True
            print("[c5] 垫接触奖励 ON")
        if getattr(args, "carry_stable", False):
            assert getattr(args, "carry_pad_reward", False), \
                "--carry_stable 依赖 --carry_pad_reward (逐侧垫接触传感器)"
            env_cfg.carry_stable = True
            print("[c5] 稳抓链 ON: 向心塑形 + 启动前 candidate (甜甜圈判据)")
        if getattr(args, "friction_curr", False):
            env_cfg.friction_curriculum = True
            print(f"[c5] 摩擦课程 ON: {env_cfg.friction_hi} → {env_cfg.friction_lo}")
        if getattr(args, "table_fail", False):
            env_cfg.table_touch_fail = True
            print(f"[table_fail] 撞桌即Fail ON (tol {env_cfg.table_touch_tol*1000:.0f}mm)")
        if getattr(args, "pour_free", False):
            assert getattr(args, "pour_succ", False) and                 getattr(args, "carry_progress", False),                 "--pour_free 需配 --pour_succ 与 --carry_progress"
            env_cfg.pour_free = True
            print(f"[pour_free] 自由探索段 ON: 行[{env_cfg.pour_free_lo},"
                  f"{env_cfg.pour_free_hi}) | 预算 {env_cfg.pour_free_budget} 步 | "
                  f"口对口势差分 w={env_cfg.pour_free_w_mouth}")
        if getattr(args, "pour_succ", False):
            env_cfg.pour_succ = True
            print(f"[pour] 几何成功判据 ON: ≥{env_cfg.pour_succ_tilt_deg}° & "
                  f"{env_cfg.pour_succ_mouth_r*100:.0f}cm & {env_cfg.pour_succ_hold}步")
        if getattr(args, "carry_progress", False):
            env_cfg.carry_progress = True
            env_cfg.approach_only_steps = 600  # 进度制预算: 40+2.5步/帧×~210帧 (数据推算)
            print("[carry3] 进度时钟模式: 预算 600 步")
        env_cfg.curobo_ref_stride = 1
        if getattr(args, "pour_e2e", False):
            # E2E (2026-08-20 用户裁定): 站姿→接近→真抓稳→追踪→倒→还原 一枪全程。
            # 必须在 carry_progress(600) 之后落定, 否则预算被盖回
            assert getattr(args, "fcd", False) and getattr(args, "carry_progress", False), \
                "--pour_e2e 需配 --fcd (四段指参考+真抓稳链) 与 --carry_progress"
            assert not getattr(args, "carry_squeeze", False), \
                "--pour_e2e 与 --carry_squeeze 互斥: 手指全程由 fcd 四段参考+残差驱动"
            env_cfg.pour_e2e = True
            env_cfg.ref_start_is_stance = True     # 合并参考首行=站姿 (覆盖 carry 的 False)
            env_cfg.success_nonterminal = True     # 真抓稳=启动门, 不终止
            env_cfg.fail_tilt_deg = 60.0           # 接近段拍倒判负恢复 (启动后 env 内豁免)
            _ze = np.load(args.curobo_ref, allow_pickle=True)
            env_cfg.carry_ref_off = int(_ze["seam"]) if "seam" in _ze.files else 0
            env_cfg.carry_start_deadline = 300     # 接近145+candidate/verify+余量
            env_cfg.approach_only_steps = 760      # 热身40+接近145+2.5步/帧×220+余量
            print(f"[e2e] Pour 端到端 ON: 参考缝行 {env_cfg.carry_ref_off} | "
                  f"启动限期 300 | 回合预算 760 | 真抓稳(双侧verify)=启动门(非终止) | "
                  f"摩擦锚=过倾斜峰t≥140 | 接近段拍倒 60° 判负(启动后豁免)")
        print(f"[carry] MVP 配置: npz={args.carry_npz} | 预算 230 步 | 豁免 tilt/hit/push")
    if getattr(args, "pregrasp_grasp", False):
        # PG-A: pregrasp29 口径(1cm靶点/近场豁免/指解锁/落位糖) + 完整抓取链
        # (不设 approach_only ⟹ 到位切 GRASP, cand→微抬升验证=成功)
        assert os.environ.get("RL_HAND_JOINTS") == "1", \
            "--pregrasp_grasp 需要 RL_HAND_JOINTS=1"
        env_cfg.pregrasp29 = True
    apply_grasp_prior(env_cfg, _prior, args.prior_yaw, approach=args.approach)
elif args.approach:
    _fatal_cfg("--approach 必须配 prior (对齐势的终点来自 GraspPose)")
if args.retract and not args.approach_only:
    # 完整任务 + 退避起点族: 仍然要抓要抬, 只是起点换成"从 GraspPose 沿 radial 退 d"。
    # 动作空间保持 13 (抓握需要手指通道), 判据保持抓稳+微抬升。
    if not args.approach:
        _fatal_cfg("--retract 必须同时给 --approach")
    env_cfg.retract_start = True
    env_cfg.retract_ratio = 0.0        # 训练从最简单端起 (与 approach_only 同一套语义)
    env_cfg.stance_prob = 0.0
    print("[retract] 完整任务(接近+抓握)改用退避式起点族; 动作空间保持 13 维")
if args.approach_only:
    if not args.approach:
        _fatal_cfg("--approach_only 必须同时给 --approach (要接近段的相位机)")
    env_cfg.retract_start = True
    # ★ **训练**的课程初值 = 最简单端。cfg 里的默认 1.0/1.0 是**评测口径**
    #   (全部从站姿起步), 训练必须从 0 起, 由 sr_slow 逐步拉满。
    #   两个旋钮都是"越大越难" —— 与 approach_t0_max 方向相反, 别混
    #   (2026-08-16 就是混了这个, 评测被改成"空降到终点", DESIGN_LOOP §2.24)。
    env_cfg.retract_ratio = 0.0
    env_cfg.stance_prob = 0.0
    # (--eval_steps 的自动跟随统一放到 env 构造之后, 按真正的 ep_total 算)
    print(f"[approach_only] 只学接近: 站姿->GraspPose | 预算 "
          f"{env_cfg.approach_only_steps} 步 | 到位判据 "
          f"{env_cfg.eps_pos0*100:.0f}cm/{env_cfg.eps_rot0*57.3:.0f}° 保持 "
          f"{env_cfg.switch_hold} 步 | 硬终止: 臂外壳穿桌 / 手碰物体")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
agent_cfg["seed"] = args.seed
agent_cfg["algorithm"]["experiment_name"] = args.name
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent_cfg["algorithm"]["minibatch_size"] = min(args.num_envs * 8, 32768)
if args.minibatch is not None:
    agent_cfg["algorithm"]["minibatch_size"] = args.minibatch
if args.kl_threshold is not None:
    agent_cfg["algorithm"]["kl_threshold"] = args.kl_threshold
if args.mini_epochs is not None:
    agent_cfg["algorithm"]["mini_epochs"] = args.mini_epochs
if args.lr is not None:
    agent_cfg["algorithm"]["learning_rate"] = args.lr
if args.entropy_coef is not None:
    agent_cfg["algorithm"]["entropy_coef"] = args.entropy_coef
if args.adam_betas is not None:
    agent_cfg["algorithm"]["adam_betas"] = [float(x) for x in args.adam_betas.split(",")]
if args.ppo3:
    agent_cfg["algorithm"]["entropy_coef"] = 0.005
    agent_cfg["algorithm"]["e_clip"] = 0.1
    agent_cfg["algorithm"]["mini_epochs"] = 8
    agent_cfg["algorithm"]["horizon_length"] = 64
if args.max_agent_steps is not None:
    agent_cfg["algorithm"]["max_agent_steps"] = args.max_agent_steps
if args.load_path is not None:
    agent_cfg["load_path"] = args.load_path

log_dir = os.path.join("logs", args.name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# ---- 世界指纹 (2026-08-28, AAG-F 0/64 回放事故后加) ----
# 站姿 USD = **世界版本**。旧站姿权重在新站姿世界里回放会全员超时, 而 obs 维度
# 一个字节都不变 ⟹ 维度闸拦不住。实现统一在 world_id.py (回放侧 check 同源)。
from tasks.pregrasp import world_id as _wid   # noqa: E402
_wid.write(log_dir)

print(f"[train] log_dir={log_dir}  clip={args.clip}  envs={args.num_envs}  "
      f"obj_jitter={env_cfg.obj_jitter_xy*1000:.0f}mm  "
      f"minibatch={agent_cfg['algorithm']['minibatch_size']}")

if getattr(args, "grasp_cent_fix", False):
    # cent_fix (2026-08-20 用户裁定, 五线通用): ①candidate 去向心门 (阈=-1 恒过,
    # 及格线="≥4垫+物静+保持", 微抬升验证仍是真考官); ②向心塑形带负梯度
    # clamp(-0.5,1) —— 压错扣分给方向。背景见 cfg.cent_signed 注释。
    env_cfg.grasp_centrip_thresh = -1.0
    env_cfg.cent_signed = True
    print("[cent_fix] candidate 去向心门 (阈-1) + 向心塑形带负梯度 clamp(-0.5,1)")

if args.curobo_ref:
    assert args.bimanual, "--curobo_ref 目前只接了双臂 env"
    assert os.path.exists(args.curobo_ref), f"缺参考文件 {args.curobo_ref}"
    env_cfg.curobo_ref_npz = os.path.abspath(args.curobo_ref)
    # ⚠ 必须在 minimal 最终落定**之前**改 no_ff, 让最终打印说的是真话
    env_cfg.minimal_no_ff = False
    env_cfg.curobo_ff_freeze_cm = args.ff_freeze_cm
    env_cfg.curobo_ff_pull = args.ff_pull
    # ⚠ 打印必须反映**实际执行的**路径 (2026-08-23 教训: 旧版无条件打印"回拉 0.08/步",
    #   而回拉当时嵌在 freeze 块里从不执行 —— 日志说谎烧掉了 AAG 八代)
    if getattr(env_cfg, "arm_abs_res", False):
        print(f"[实验] cuRobo 参考: {os.path.basename(args.curobo_ref)} "
              f"| 方案C 绝对模式 ⟹ 差分前馈与回拉锚**均不参与计算** "
              f"(ff_freeze_cm/ff_pull 本轮无效)")
    else:
        print(f"[实验] cuRobo 前馈参考: {os.path.basename(args.curobo_ref)} "
              f"(no_ff 改回 False, 零动作=沿规划走, 近端 {args.ff_freeze_cm:.0f}cm 内冻结 ff 交给残差, "
              f"回拉 {args.ff_pull:.2f}/步"
              + ("" if args.ff_pull > 0 else " = 关") + ")")

if args.minimal or getattr(args, "e2e_squeeze", False):
    # ---- 最简/e2e 模式的**最终落定** (必须在所有 approach/retract 块之后) -------
    # e2e_squeeze (2026-08-25): 同病同药 —— 不钉的话 stance_prob/retract_ratio 被
    # approach/retract 块盖回 0 ⟹ 出生落到退避末端(相位1), 实测 2M Mean=0.05 静默
    # 教训: 第一版把这些写在前面, 被 approach_only 块的 `retract_ratio=0 / stance_prob=0`
    #   覆盖回去 ⟹ 起点落到路径**末端**(= GraspPose 本身), 冒烟表现为
    #   arrive_rate 从第一个 epoch 就是 1.000、steps_to_arrive=0 —— 看着像大成功,
    #   其实是"手一开始就在终点"。与今天 `_fgate_dg` 被 __init__ 默认值抹掉同一个形状。
    env_cfg.stance_prob = 1.0      # >=1.0 触发 reset 里的"钉死 t0=0"分支 = 起点固定站姿
    env_cfg.retract_ratio = 1.0
    # FC-D.1 RSI (2026-08-20 用户裁定): 显式 --direct_grasp_prob 优先于 minimal 的钉零
    #   —— 不写这一条, RSI 旗会被这里静默盖回 0 (发射前核查抓到的雷)
    env_cfg.direct_grasp_prob = 0.0 if args.direct_grasp_prob is None \
        else float(args.direct_grasp_prob)
    if float(getattr(args, "c0_min", -1.0)) >= 0.0:
        # 诊断口径: closure 起步固定 (1.0 = 直接从合拢模板起步)
        env_cfg.closure_init_max = float(args.c0_min)
        env_cfg.closure_init_min = float(args.c0_min)
        print(f"[诊断] 起步合拢度固定 = {env_cfg.closure_init_max:.2f}")
    if float(getattr(args, "eps_pos_cm", -1.0)) > 0.0:
        # AAG-v3.2 (2026-08-22 检验裁定): 纯前馈停靠点腕口径 L=1.09/R=1.00cm 而门=1.00
        # —— 左手差 0.9mm 永远差门票, 从没体验过门后收入流, 被碰垫糖拐走 (账本实锤)。
        # 放到 1.35cm 盖住停靠点+噪声; 抓握质量由 g2/m1 更严判据兜底 (GUI 验证过
        # 从停靠点合拢能成抓)。必须在 minimal 落定之后覆盖, 否则被钉回 1cm。
        env_cfg.eps_pos = env_cfg.eps_pos0 = float(args.eps_pos_cm) / 100.0
        print(f"[v3.2] 到位门覆盖: eps_pos={env_cfg.eps_pos*100:.2f}cm "
              f"(纯前馈停靠 L 1.09/R 1.00cm + 噪声余量)")
    print(f"[minimal] 最终生效: stance_prob={env_cfg.stance_prob} "
          f"retract_ratio={env_cfg.retract_ratio} "
          f"direct_grasp_prob={env_cfg.direct_grasp_prob} "
          f"eps={env_cfg.eps_pos*100:.2f}cm/{np.degrees(env_cfg.eps_rot):.1f}° "
          f"no_ff={env_cfg.minimal_no_ff} res={env_cfg.dyn_arm_near}x")

_EnvCls = GraspTaskEnv
if args.l5:
    # ---- L5 最终落定 (必须在 minimal 最终落定之后, bimanual 翻倍之前) ----
    assert args.bimanual and args.curobo_ref, "--l5 须配 --bimanual --curobo_ref"
    assert not args.approach_only, "--l5 是抓取段训练, 不配 --approach_only"
    env_cfg.l5_couple = True
    env_cfg.direct_grasp_prob = float(args.l5_direct)
    if args.dyn_far > 0:
        env_cfg.dyn_arm_far = float(args.dyn_far)
        print(f"[L5] 远场残差上限覆盖: {env_cfg.dyn_arm_far}x "
              f"(运输归前馈, 残差全程只做修正)")
    print(f"[L5] 最终生效: 冻结区合指耦合 ON (w={env_cfg.l5_couple_w}) | "
          f"direct_grasp_prob={env_cfg.direct_grasp_prob} | "
          f"成功=双手微抬升 | 动作 13/侧")

if getattr(args, "pour_carry", False):
    if getattr(args, "e2e_squeeze", False):
        # E2E (2026-08-26 3M验尸): arrive 是相位发动机, 不许封死 —— 封印是
        # 抓稳出生时代的设计 (v8D 回放封印同族)。门取 F 线同口径无退火
        # (1.35cm/15°): 接近参考=side05 同源, 门也同源; epoch1 实测 0.8cm/5.9°
        # 在此门内, 旧封印 0.1cm/1.1° 连黄金起手都不认 ⟹ 3M 假健康游荡
        env_cfg.eps_pos = env_cfg.eps_pos0 = 0.0135
        env_cfg.eps_rot = env_cfg.eps_rot0 = float(np.radians(15.0))
        print("[e2e] 到位门 F线同口径: 1.35cm/15° 无退火 (封印豁免)")
    else:
        # arrive 永不触发 (基座成功通道封死, 成功语义完全归 PourCarryEnv)
        env_cfg.eps_pos = env_cfg.eps_pos0 = 0.001
        env_cfg.eps_rot = env_cfg.eps_rot0 = 0.02
    # carry4 (2026-08-24): 手锚融合参考自带自由段行号 (时间轴与 carry3 不同,
    # 硬编码 93/140 会指错段) —— npz 有 free_lo/hi 就以它为准
    _zc4 = np.load(env_cfg.carry_npz, allow_pickle=True)
    if "free_lo" in _zc4.files:
        env_cfg.pour_free_lo = int(_zc4["free_lo"])
        env_cfg.pour_free_hi = int(_zc4["free_hi"])
        print(f"[carry4] 自由段行号从 npz 接管: [{env_cfg.pour_free_lo},"
              f"{env_cfg.pour_free_hi})")
        # 窗行数 103 (carry3 是 47): 时钟≤1行/步 ⟹ 预算同比放大, 否则全员 pf_trunc
        env_cfg.pour_free_budget = max(int(env_cfg.pour_free_budget),
                                       int((env_cfg.pour_free_hi
                                            - env_cfg.pour_free_lo) * 3.2))
        env_cfg.approach_only_steps = max(int(getattr(env_cfg, "approach_only_steps", 600)), 800) \
            + (700 if getattr(args, "pour_place", False) else 0)
        print(f"[carry4] 预算随窗放大: pf_budget={env_cfg.pour_free_budget} "
              f"approach_only_steps={env_cfg.approach_only_steps}")
if getattr(args, "pours_v5", False):
    # v5 特权滑移块 +8/侧 (v6: +10, 多 started/pf_on 两维阶段信号) —— 必须在
    # apply_grasp_prior 之后(它按 _obs_base 算死)、bimanual ×2 之前
    env_cfg.observation_space += 10 if getattr(args, "pours_v6", False) else 8
if getattr(args, "obj_obs", False):
    env_cfg.observation_space += 13   # 物体特权块(13=位置3+姿态4+线速3+角速3)/侧 (同上时序纪律)
if args.bimanual:
    # ★ 必须在构造前翻倍 —— observation_space 在 apply_grasp_prior 里已按单臂算死,
    #   这里把它和 action_space 一起 ×2。顺序错了会得到"观测宽度 X != cfg.observation_space"
    #   (2026-08-17 在 record.py 上踩过同一个坑)。
    assert args.prior_b, "--bimanual 必须配 --prior_b (另一只手的 GraspPose)"
    assert os.path.exists(args.prior_b), f"缺 {args.prior_b}"
    if getattr(args, "pour_carry", False):
        from tasks.pour.carry_env import PourCarryEnv as _EnvCls  # noqa: N813
        print("[bimanual] ★ PourCarryEnv (Carry MVP, v2 底座)")
    elif args.bi_native:
        from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv as _EnvCls  # noqa: N813
        print("[bimanual] ★ v2 原生底座 (bimanual_native_env)")
    else:
        from tasks.pregrasp.bimanual_env import BimanualApproachEnv as _EnvCls  # noqa: N813
    env_cfg.prior_b_npz = os.path.abspath(args.prior_b)
    env_cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
    _a1, _o1 = env_cfg.action_space, env_cfg.observation_space
    env_cfg.action_space = 2 * _a1
    #   (动作缓冲逐侧独立各 _a1 维 —— 见 bimanual_env 里的说明 —— 所以每侧观测
    #    宽度就是单臂的 _o1, 直接 ×2 即可)
    env_cfg.observation_space = 2 * _o1
    env_cfg._obs_single = _o1            # 每侧宽度, 供双臂 env 逐侧核验时用
    # ★ 特权信息 (critic 专用) 也是逐侧拼接的 -> 网络那边的维度必须同步翻倍。
    #   不翻会得到 "mat1 and mat2 shapes cannot be multiplied (64x356 and 324x256)"
    #   —— 356 = 292 + 64(翻倍后的 priv), 而网络按 324 = 292 + 32 建的。
    #   (两个键不在同一段: priv_info_dim 在 algorithm, actor_priv_dim 在 network)
    _bumped = []
    for _sec, _key in (("algorithm", "priv_info_dim"), ("network", "actor_priv_dim")):
        _d = agent_cfg.get(_sec, {})
        if _key in _d:
            _d[_key] = 2 * int(_d[_key])
            _bumped.append(f"{_sec}.{_key}={_d[_key]}")
    assert len(_bumped) == 2, f"特权维度没找全, 只翻了 {_bumped} —— 网络维度会对不上"
    print(f"[bimanual] 特权信息翻倍: {' | '.join(_bumped)}")
    print(f"[bimanual] 动作 {_a1}->{env_cfg.action_space} | "
          f"观测 {_o1}->{env_cfg.observation_space} | B 手先验 {os.path.basename(args.prior_b)} "
          f"yaw={args.prior_b_yaw}")
env_raw = _EnvCls(env_cfg)
env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
agent = MilestonePPO(env, output_dir=log_dir, full_config=ConfigWrapper(agent_cfg, env_cfg))
agent._raw_env = env_raw
if args.warm_start:
    # ---- 跨动作维度的热启动 (2026-08-17) ----
    # `load_state_dict` 是严格匹配的: 7 维接近段的 ckpt 塞不进 13/29 维的抓取网络。
    # 这里逐张量比形状: 一样的照搬(躯干/critic/观测归一化), 不一样的**按前缀对齐**
    # 搬过去(动作头的前 7 行 = 臂), 剩下的保持新初始化。
    # ⚠ 必须打印**逐项结果** —— 静默跳过整个 backbone 会让"热启动"变成"从零训",
    #   而曲线看起来一模一样, 查不出来。
    import torch as _t
    _ck = _t.load(args.warm_start, map_location="cpu", weights_only=False)
    _src = _ck.get("model", _ck)
    _dst = agent.model.state_dict()
    _same = _part = _skip = 0
    for _k, _v in _dst.items():
        if _k not in _src:
            _skip += 1
            continue
        _sv = _src[_k]
        if _sv.shape == _v.shape:
            _dst[_k] = _sv.clone(); _same += 1
        elif (getattr(args, "l5", False) and _sv.shape[0] == 14
              and _v.shape[0] == 26 and _sv.shape[1:] == _v.shape[1:]):
            # L5 双臂 14->26: 逐侧搬动作头的**臂行** —— 布局是 [A(7臂+6指), B(7臂+6指)],
            # 前缀切片会把 L4 的 B 臂行灌进 A 指槽位。指行保持新初始化。
            _v[0:7] = _sv[0:7].clone()
            _v[13:20] = _sv[7:14].clone()
            _dst[_k] = _v; _part += 1
            print(f"[warm] 逐侧搬臂行 {_k}: 14->26 (A[0:7], B[13:20]; 指行新init)")
        elif (getattr(args, "l5", False) and _sv.dim() >= 2
              and _sv.shape[0] == _v.shape[0] and _sv.shape[1] != _v.shape[1]):
            # L5: 每侧观测宽度/排布变了 (approach_only 关掉后手指通道插在中段),
            # 输入列前缀拷贝会错位 —— 宁可新 init 也不移植错特征。
            _skip += 1
            print(f"[warm] ⚠ L5 跳过输入层 {_k}: {tuple(_sv.shape)} vs "
                  f"{tuple(_v.shape)} (观测排布变化)")
        elif _sv.dim() == _v.dim() and all(
                a <= b for a, b in zip(_sv.shape, _v.shape)):
            _sl = tuple(slice(0, n) for n in _sv.shape)
            _v[_sl] = _sv.clone(); _dst[_k] = _v; _part += 1
            print(f"[warm] 部分搬运 {_k}: {tuple(_sv.shape)} -> {tuple(_v.shape)}")
        else:
            _skip += 1
            print(f"[warm] ⚠ 跳过 {_k}: {tuple(_sv.shape)} vs {tuple(_v.shape)}")
    agent.model.load_state_dict(_dst)
    if "running_mean_std" in _ck:
        try:
            agent.running_mean_std.load_state_dict(_ck["running_mean_std"])
            print("[warm] 观测归一化已搬运")
        except Exception as _e:
            print(f"[warm] ⚠ 观测归一化没搬成 ({type(_e).__name__}) —— "
                  f"观测维度变了, 新的要重新统计")
    print(f"[warm] 热启动完成: 整层照搬 {_same} | 部分搬运 {_part} | 跳过 {_skip} "
          f"<- {os.path.basename(args.warm_start)}")
    assert _same + _part > 0, "一个张量都没搬成 —— 等于从零训, 检查 ckpt 是否匹配"

agent._time_shape = args.time_shape
agent._ts_gate = args.time_shape_gate
agent._minimal = args.minimal    # 最简模式: 训练钩子里所有课程整段跳过
agent._no_imit = args.no_imit    # F 组: 模仿罚恒 0        # 笨拙课程钩子用: 按 sr_ema 更新 env.gentle
agent._ar_ema = args.ar_ema      # N3 组: 可行性课程 EMA 速度
agent._curr_start_target = args.curr_start_target   # 起点课程推进的保守度
if abs(args.curr_start_target - 0.3) > 1e-9:      # 静默 flag 必须自证已生效
    print(f"[实验] curr_start_target 覆盖为 {args.curr_start_target}")
agent._grasp_first = args.grasp_first
agent._gf_target = args.gf_target
if args.grasp_first or args.start_jitter:
    # ---- §2.17 阶段A 起点抖动池 (用户设计, 2026-08-03) ----
    # 覆盖课程全程合法到达误差的包络 U(0, eps_pos0)×U(0, eps_rot0): 接近段到达
    # 永远 ≤ 当时的 eps ≤ 这个包络, 毕业 = 在这个分布上抓取 ≥ gf_target ——
    # 交接自带容错. 复用 tol_curve.py 验证过的 IK 抖动法 + arm_start_pool 钩子.
    import numpy as _np
    from rl_rebuild.correction.kinematics import ArmIK as _ArmIK
    _ik = _ArmIK(env_cfg.hand_side, anchor_link="arm_center", anchor_T=env_raw._anchor_T)
    _sq = env_raw.q_pregrasp.cpu().numpy().astype(_np.float64)
    _p0, _R0 = _ik.fk(_sq)
    _rng = _np.random.default_rng(args.seed)

    def _axang(v, th):
        v = v / _np.linalg.norm(v)
        K = _np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return _np.eye(3) + _np.sin(th) * K + (1 - _np.cos(th)) * K @ K

    _qs = [_sq] * 64                       # ~14% 精确起点
    _tries = 0
    while len(_qs) < 448 and _tries < 1600:
        _tries += 1
        _u = _rng.normal(size=3)
        _r = _ik.solve(_p0 + _u / _np.linalg.norm(_u) * _rng.uniform(0, env_cfg.eps_pos0),
                       _axang(_rng.normal(size=3), _rng.uniform(0, env_cfg.eps_rot0)) @ _R0,
                       q0=_sq, iters=80)
        if _r["ok"] and _r["pos_err"] < 0.002:
            _qs.append(_r["q"])
    env_raw.arm_start_pool = torch.tensor(_np.stack(_qs), dtype=torch.float32,
                                          device=env_raw.device)
    print(f"[grasp_first] 起点抖动池 {len(_qs)} 位形 (≤{env_cfg.eps_pos0*100:.0f}cm/"
          f"≤{_np.degrees(env_cfg.eps_rot0):.0f}°, 含 64 精确起点; IK 命中 "
          f"{len(_qs)-64}/{_tries}) | 毕业线 {args.gf_target}")
agent._auto_stop = args.auto_stop
# ★ eval_steps 必须 > 实际回合上限, 否则会有 env 跑不完一回合 ⟹ 成功率分母不是
#   num_envs(下面有断言拦)。**按 env 真正算出来的 ep_total 跟随**, 不要按 cfg 常量猜 ——
#   2026-08-16 我先按 approach_only_steps(250) 猜, 提到 300; 完整任务的 ep_total 是
#   403, 于是同一道断言又崩一次。ep_total 只有 env 构造完才知道, 所以修正放在这里。
if getattr(env_cfg, "retract_start", False) and args.eval_steps <= env_raw.ep_total:
    args.eval_steps = int(env_raw.ep_total * 1.2)
    print(f"[retract] --eval_steps 自动提到 {args.eval_steps} "
          f"(env 实际回合上限 ep_total={env_raw.ep_total})")
agent._eval_every, agent._eval_steps = args.eval_every, args.eval_steps
agent._stopper = StopDecider(target_sr=args.stop_target_sr, target_hits=args.stop_target_hits,
                             patience=args.stop_patience, delta=args.stop_delta,
                             min_steps=int(args.stop_min_steps))
if args.auto_stop != "off":
    assert args.eval_steps > env_raw.ep_total, \
        (f"--eval_steps {args.eval_steps} ≤ 回合上限 {env_raw.ep_total} —— "
         f"会有 env 跑不完一回合, 成功率分母不是 num_envs")
    print(f"[auto_stop] {args.auto_stop} | 每 {args.eval_every} epoch 评测 "
          f"({args.num_envs} env × ≤{args.eval_steps} 步) | 达标线 {args.stop_target_sr:.0%}"
          f"×{args.stop_target_hits} | 平台期 {args.stop_patience} 次未涨 {args.stop_delta:.0%}"
          f" | {args.stop_min_steps/1e6:.0f}M 步前不停")

# 录像让出点: epoch 边界检查暂停请求, 避免两个 Isaac 同时满载触发电源 OCP
# ⚠ 串联而非覆盖 —— 直接赋值会把 gpu_guard 的让出点顶掉 (录像再也拿不到卡)
_yield_slot = _slot.yield_if_paused
# ★ 哨兵排在最前: 相位表被课程期写手打回去时, 要在这一 epoch 就抛,
#   别等到评测才发现 arrive 恒 0 (2026-08-26 同族第五犯: 构造期修好, 课程期又改回)
agent.epoch_hook = lambda: (agent.phase_table_sentinel(),
                            _yield_slot(), agent.maybe_eval_and_stop())

if args.resume and agent_cfg["load_path"] not in (None, "None"):
    print(f"[train] resume from {agent_cfg['load_path']}")
    agent.restore_train(agent_cfg["load_path"])


# ============================ 起飞前自检 ============================
# 2026-08-23 用户点单。**纪律: 验行为, 不验配置。**
# 今天一天四个静默失效 (回拉锚嵌在死分支 / --aag 没设 fin_ref_track /
# 指参考漏换关节序 / 录像多传 --obj_obs), 共性都是"配置写了、日志也打印了、
# 代码路径根本没执行"。所以下面每一项都必须从**运行时数值**取证,
# 打印 cfg 的值一律不算数。
# 位置: 在 agent 构建之后、agent.train() 之前 ⟹ 与训练**同一段配置代码**。
if int(getattr(args, "preflight", 0)) > 0:
    import hashlib as _hl
    import numpy as _np3
    import torch as _t3

    from tasks.pregrasp import bimanual as _BM3
    _N = int(args.preflight)
    _fail, _warn = [], []
    # ★ 2026-08-25: 相位预算安装 —— 训练路径由钩子装 (line ~589), preflight 不经过
    #   ⟹ 零动作 env 跑在默认预算(~145)上, ~85步被截断, ⑩ 门恒 0 (五发实测破案)
    if hasattr(env_raw, "phase_timeout_t"):
        _b10 = int(getattr(env_cfg, "approach_only_steps", 800))
        env_raw.phase_timeout_t[0] = max(int(env_raw.phase_timeout_t[0]), _b10)
        print(f"[preflight] 相位0预算安装: {int(env_raw.phase_timeout_t[0])} 步")
    if getattr(args, "pour_place", False):
        # 零动作口径: 纯放音无反馈, 物体跟踪漂移随行数累积, 5cm 跟丢闸在行~68 必杀
        # (九发显微镜实名定罪 carry_lost)。训练态 5cm 不动 (64env 评测背书零误杀);
        # 预检探针放宽至 20cm 只为让参考放完全程, ⑩ 门才有被测机会。
        env_raw.cfg.carry_lost_m = 0.20
        print("[preflight] 零动作口径: carry_lost_m 临时放宽至 20cm (训练态仍 5cm)")

    # ★ 2026-08-29 (与 RL 会话会诊): _chk 曾被同时当"断言"和"回显"用
    #   (`_chk(True, "observation_space", ...)` 只是把 cfg 复读一遍)。
    #   混用之后**读日志的人无法区分哪些行是断言、哪些只是回显**。
    #   现在分成两个函数, 并对断言计数 —— 计数让"检查静默消失"变得可见:
    #   一个 section 打了标题却零断言, 正是 ⑥c 空表那种病的长相。
    _nassert = [0, 0]      # [硬, 软]

    def _chk(ok, name, detail, hard=True):
        """断言 —— 必须能说出"什么情况下会红"。说不出的, 用 _show。"""
        _nassert[0 if hard else 1] += 1
        print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
        if not ok:
            (_fail if hard else _warn).append(name)

    def _show(name, detail):
        """回显 —— 只是打印一个值/事实, **不做任何判断**, 不计入断言数。"""
        print(f"  [ii] {name}: {detail}", flush=True)

    print("\n" + "=" * 78)
    print(f"[preflight] 起飞前自检 — 零动作 {_N} 步 | 验行为不验配置")
    print("=" * 78, flush=True)

    # ---- ① 文件身份 (md5: 防拿错文件 / 文件被悄悄改过) ----
    print("\n① 参考与先验文件身份")
    for _tag, _p in (("臂参考 curobo_ref", getattr(env_cfg, "curobo_ref_npz", "")),
                     ("指参考 fin_ref  ", getattr(env_cfg, "fin_ref_npz", "")),
                     ("右先验 grasp_prior", getattr(env_cfg, "grasp_prior_npz", "")),
                     ("左先验 prior_b   ", getattr(env_cfg, "prior_b_npz", ""))):
        if not _p:
            print(f"  [--] {_tag}: (未设置)")
            continue
        _m = _hl.md5(open(_p, "rb").read()).hexdigest()[:12]
        _z3 = _np3.load(_p, allow_pickle=True)
        _src = str(_z3["source"]) if "source" in _z3.files else "(无 source 字段)"
        print(f"  [OK] {_tag}: {os.path.basename(_p)}  md5={_m}")
        print(f"       source = {_src}")

    # ---- ② 指参考换序自检 (串位时行程会明显变小 —— 廉价但决定性) ----
    print("\n② 指参考装载 (GENERIC->USD 换序是否真的应用)")
    _sides3 = [(env_raw._A_name, env_raw._A), (env_raw._B_name, env_raw._B)] \
        if hasattr(env_raw, "_A") else []
    for _nm3, _sd3 in _sides3:
        _fp = _sd3.data.get("_fin_ref_path")
        if _fp is None:
            _chk(False, f"{_nm3} 指参考",
                 "_fin_ref_path 为 None —— 指参考根本没装载! (squeeze 体制无此物, pour 模式降软)",
                 hard=not getattr(args, "pour_place", False))
            continue
        _tr3 = _t3.rad2deg((_fp[-1] - _fp[0]).abs())
        _chk(float(_tr3.max()) > 60.0, f"{_nm3} 指参考",
             f"{tuple(_fp.shape)} 行程 峰 {float(_tr3.max()):.1f}° 均 "
             f"{float(_tr3.mean()):.1f}°  (串位时峰值会掉到 60° 以下)")

    # ---- ③ 观测/动作维度 ----
    print("\n③ 维度")
    # ★ 2026-08-29: 原来这两行是 `_chk(True, ...)` —— 字面上的恒真断言。
    #   它印 [OK] 让人以为验过了, 其实只是把 cfg 的值复读一遍 (自己验自己)。
    #   判别法: 说不出"什么情况下会红"的检查 = 恒真断言。
    #   真正的外部对账对象是 ckpt 里的归一化器形状 —— 它记录了训练时的真实 obs 维。
    _lp = getattr(args, "load_path", None)
    if _lp and os.path.exists(_lp):
        try:
            import torch as _t9
            _ck = _t9.load(_lp, map_location="cpu", weights_only=False)
            _od = int(_ck["running_mean_std"]["running_mean"].shape[0])
            _chk(_od == int(env_cfg.observation_space),
                 "observation_space vs ckpt",
                 f"cfg {env_cfg.observation_space} vs ckpt {_od}"
                 + ("" if _od == int(env_cfg.observation_space) else
                    "  ⟹ 旗集与该 ckpt 不配套, restore 会形状不匹配"))
        except Exception as _e9:
            _chk(False, "observation_space vs ckpt",
                 f"读 ckpt 失败: {type(_e9).__name__}: {_e9} —— 未验, 不是通过", hard=False)
    else:
        _show("维度", f"action={env_cfg.action_space} obs={env_cfg.observation_space}"
              f"  —— 无 --load_path, **未与任何外部基准对账**")
    print(f"  [ii] 录像/回放必须用同样的旗产出同样的维度, 否则 state_dict 尺寸不匹配")

    # ---- ④ 零动作 rollout ----
    print(f"\n④ 零动作 {_N} 步 (残差恒为 0 = 严格复现参考)")
    _obs3 = env.reset()
    _zero = _t3.zeros((env_cfg.scene.num_envs, env_cfg.action_space),
                      device=env_raw.device)
    _absdev, _qerr, _nctrl = {}, {}, {}
    import collections as _co3
    _term = _co3.Counter(); _eplen = []; _padpk = {}; _qdpk = [0.0, 0.0]; _shellmin = {}; _endfe = {}; _endsnap = {}; _g2cnt = {}; _candg = {}; _succg = [0, 0]; _arrived_ever = {}; _carry_ev = {}; _fgv = {}; _latch = {}; _g2prev = {}; _endbest = {}; _g2rst = {}; _g2gate = {}; _objrot = {}; _envpk = {}
    _BM3.use_side and [setattr(_s.env if hasattr(_s,"env") else env_raw, "_dbg_adv", [0.,0.,0.,0.,0.]) for _s in ()]
    setattr(env_raw, "_dbg_adv", [0.,0.,0.,0.,0.])
    _geom = {} if getattr(args, "geom_audit", False) else None
    _gr5x = int(getattr(env_cfg, "aag_grasp_row", 0))
    _rsum7 = 0.0
    with _t3.no_grad():
        for _i3 in range(_N):
            _st7 = env.step(_zero)
            try:
                _rsum7 += float(_st7[1].mean())
            except Exception:
                pass
            # ⑩ 死因显微镜 (2026-08-25): env0 回合长度回卷 = 刚死过, 打全量状态
            #   旗守卫: 只在 pour_place/e2e_squeeze 旗集运行 (RL_Training 对账要求,
            #   其余旗集逐字节等价)
            if (getattr(args, "pour_place", False)
                    or getattr(args, "e2e_squeeze", False)):
                if _i3 % 150 == 0:
                    # ② 五状态活性探针 (2026-08-26 收入面接线前置审计)
                    for _pn3 in ("_A", "_B"):
                        _pns3 = getattr(env_raw, _pn3, None)
                        if _pns3 is None: continue
                        _pd3 = _pns3.data
                        def _pv3(k):
                            v = _pd3.get(k)
                            if v is None: return "无"
                            try: return f"{float(v.float().mean()):.4f}"
                            except Exception: return "?"
                        print(f"  [活性] 步{_i3} {_pn3}: arrived={_pv3('arrived')} "
                              f"_fc_d={_pv3('_fc_d')} _fcd_ref_last={_pv3('_fcd_ref_last')} "
                              f"_g2_d={_pv3('_g2_d')} pad_touched={_pv3('pad_touched')}",
                              flush=True)
                _ct10 = getattr(env_raw, "_c_t", None)
                if _ct10 is not None:
                    object.__setattr__(env_raw, "_pf_ct_max",
                                       max(int(getattr(env_raw, "_pf_ct_max",
                                                       0)),
                                           int(_ct10.max())))
                _len_now = int(env_raw.episode_length_buf[0])
                if _len_now < int(getattr(env_raw, "_dbg_prev_len", 0)):
                    print(f"  [死因] 步{_i3}: 上回合长 "
                          f"{int(getattr(env_raw, '_dbg_prev_len', -1))} "
                          f"| ep_total={int(getattr(env_raw, 'ep_total', -1))} "
                          f"| phase_to0={int(env_raw.phase_timeout_t[0])}",
                          flush=True)
                object.__setattr__(env_raw, "_dbg_prev_len", _len_now)
                # 死亡当步: env0 的终止组件旗实名 (carry级 + 基座sig级)
                _kd10 = getattr(env_raw, "_kill_dbg", None)
                if _kd10 is not None:
                    _hit10 = [k for k, v in _kd10.items()
                              if hasattr(v, "__getitem__") and bool(v[0])]
                    if _hit10:
                        print(f"  [死因旗] 步{_i3}: env0 carry级 {_hit10}",
                              flush=True)
                _sgx = getattr(env_raw, "_sig_merged", None) or {}
                _hitx = [k for k, v in _sgx.items()
                         if hasattr(v, "dtype") and str(v.dtype) == "torch.bool"
                         and v.ndim == 1 and k != "active" and bool(v[0])]
                if _hitx:
                    print(f"  [死因旗] 步{_i3}: env0 sig级 {_hitx}", flush=True)
            _sg3 = getattr(env_raw, "_sig", None) or {}
            for _kk in ("fell", "thrown", "pushed", "stuck", "table_crash",
                        "timeout", "toppled", "newly_success", "verify_fail"):
                if _kk in _sg3 and hasattr(_sg3[_kk], "sum"):
                    _c3 = int(_sg3[_kk].sum())
                    if _c3:
                        _term[_kk] += _c3
            _eplen.append(int(env_raw.episode_length_buf[0]))
            if _geom is not None and _gr5x > 0:
                for _nm7, _sd7 in _sides3:
                    with _BM3.use_side(env_raw, _sd7):
                        _per = getattr(env_raw, "_shell_obj_per", None)
                        if _per is None: continue
                        _w7 = (env_raw.ref_t > 0) & (env_raw.ref_t < _gr5x)
                        if not bool(_w7.any()): continue
                        _dsp = (env_raw.object.data.root_pos_w
                                - env_raw.scene.env_origins
                                - env_raw.obj_start_pos)[:, :2].norm(dim=1)
                        _g7 = _geom.setdefault(_nm7, {"per": [], "dsp": []})
                        _g7["per"].append(_per[_w7].detach().cpu().numpy())
                        _g7["dsp"].append(_dsp[_w7].detach().cpu().numpy())
                        _g7.setdefault("rt", []).append(env_raw.ref_t[_w7].detach().cpu().numpy())
            _gr5 = int(getattr(env_cfg, "aag_grasp_row", 0))
            for _nm6, _sd6 in _sides3:
                _sg6 = _sd6.data.get("_sig") or {}
                if "shell_obj_d" in _sg6 and _gr5 > 0:
                    with _BM3.use_side(env_raw, _sd6):
                        _w6 = (env_raw.ref_t > 0) & (env_raw.ref_t < _gr5)
                    if bool(_w6.any()):
                        _v6 = float(_sg6["shell_obj_d"][_w6].min())
                        _shellmin[_nm6] = min(_shellmin.get(_nm6, 9.0), _v6)
            for _nmA, _sdA in _sides3:
                with _BM3.use_side(env_raw, _sdA):
                    _av = env_raw.arrived
                _arrived_ever[_nmA] = max(_arrived_ever.get(_nmA, 0),
                                          int(_av.sum()))
            # carry/pour 契约钟 (RL_Pour 给的读法): _c_t 与 free_lo 同在契约行空间,
            # seam(carry_ref_off) 只作用于手行索引, 这里不用换算。
            # 闩语义: 记录每个 env **首次** arrived 时的 ref_t (RL_Pour 2026-08-25 提出)
            for _nmL, _sdL in _sides3:
                with _BM3.use_side(env_raw, _sdL):
                    _arL = env_raw.arrived
                    _rtL = env_raw.ref_t
                _pv = _latch.setdefault(_nmL, {})
                _prev = _pv.get("mask")
                if _prev is None:
                    _pv["mask"] = _arL.clone()
                    _pv["row"] = _t3.where(_arL, _rtL,
                                           _t3.full_like(_rtL, -1))
                else:
                    _new = _arL & ~_prev
                    if bool(_new.any()):
                        _pv["row"] = _t3.where(_new, _rtL, _pv["row"])
                    _pv["mask"] = _prev | _arL
            _fi2c = getattr(env_raw, "_dbg_fi2", None)
            if _fi2c is not None:
                _fh5 = int(getattr(env_raw.cfg, "fin_hold_row", 0))
                _fgv["past"] = max(_fgv.get("past", 0),
                                   int((_fi2c > _fh5).sum()))
                _fgv["n"] = int(_fi2c.numel())
                _fgv["hold"] = _fh5
            _gg5 = getattr(env_raw, "_grip_g", None)
            if _gg5 is not None:
                _fgv["g_max"] = max(_fgv.get("g_max", 0.0), float(_gg5.max()))
            _ct4 = getattr(env_raw, "_c_t", None)
            if _ct4 is not None:
                _st4 = getattr(env_raw, "_c_started", None)
                if _st4 is not None:
                    _carry_ev["started"] = max(_carry_ev.get("started", 0),
                                               int(_st4.sum()))
                _lo4 = int(getattr(env_raw.cfg, "pour_free_lo", -1))
                if _lo4 >= 0:
                    _carry_ev["reach"] = max(_carry_ev.get("reach", 0),
                                             int((_ct4 >= _lo4).sum()))
                _carry_ev["ct_max"] = max(_carry_ev.get("ct_max", 0),
                                          int(_ct4.max()))
            _bd = getattr(env_raw, "_bi_done_once", None)
            if _bd is not None:
                _succg[0] = max(_succg[0], int(_bd.sum()))
                _succg[1] = int(_bd.numel())
            for _nmC, _sdC in _sides3:
                with _BM3.use_side(env_raw, _sdC):
                    _cd = getattr(env_raw, "_cand_dbg", None)
                    if _cd:
                        _acc = _candg.setdefault(_nmC, {})
                        for _k, _v in _cd.items():
                            _a = _acc.setdefault(_k, [0, 0])
                            _a[0] += int(_v.numel()); _a[1] += int(_v.sum())
                        _a = _acc.setdefault("cand_run>0", [0, 0])
                        _a[0] += int(env_raw.cand_run.numel())
                        _a[1] += int((env_raw.cand_run > 0).sum())
                        _a = _acc.setdefault("got_candidate", [0, 0])
                        _a[0] += int(env_raw.got_candidate.numel())
                        _a[1] += int(env_raw.got_candidate.sum())
            for _nm9, _sd9 in _sides3:
                _sg9 = _sd9.data.get("_seg_rows")
                if not _sg9: continue
                with _BM3.use_side(env_raw, _sd9):
                    _fe9 = getattr(env_raw, "_g2_fe", None)
                    if _fe9 is None: continue
                    # ★ 2026-08-24: 成功接到 g2 后回合会在 squeeze 末就终止,
                    #   原来的"hold3 起始行"窗口采不到样 ⟹ 放宽到 squeeze 起始行。
                    #   仍取窗口内**最小**误差, 语义不变(复现参考时应落在目标点上)。
                    _m9 = env_raw.ref_t >= int(_sg9[-2][1])   # squeeze 起始行
                    # ★ 成功接 g2 后, g2_done 置位当步回合就终止并复位 ⟹ 采闩标志
                    #   必漏(实测 right 0/2212 而 newly_success=123 自相矛盾)。
                    #   改数**新达成事件**: g2_done 由 False 翻 True 的次数。
                    _gd9 = getattr(env_raw, "_g2_done", None)
                    if _gd9 is not None:
                        _c9 = _g2cnt.setdefault(_nm9, [0, 0])
                        _c9[0] += int(_m9.sum())
                        _pv = _g2prev.get(_nm9)
                        if _pv is not None:
                            _c9[1] += int((_gd9 & ~_pv).sum())
                            _r9 = _g2rst.setdefault(_nm9, [0, 0.0, 0])
                            _r9[0] += int((_pv & ~_gd9).sum())
                            _r9[1] += float(_gd9.float().mean())
                            _r9[2] += 1
                        _g2prev[_nm9] = _gd9.clone()   # ★ 必须无条件更新
                    _gg = getattr(env_raw, "_g2_gates", None)
                    if _gg and bool(_m9.any()):
                        _ga9 = _g2gate.setdefault(_nm9, {})
                        for _k, _v in _gg.items():
                            _a = _ga9.setdefault(_k, [0, 0])
                            _a[0] += int(_m9.sum()); _a[1] += int((_m9 & _v).sum())
                        _a = _ga9.setdefault("腕转(度)", [0, 0.0])
                        _a[0] += int(_m9.sum())
                        _a[1] += float(_t3.rad2deg(env_raw._g2_a[_m9]).sum())
                    if bool(_m9.any()):
                        _endfe.setdefault(_nm9, []).append(
                            float(_fe9[_m9].min()) * 100)
                    _cur9 = float(_fe9[_m9].min()) * 100 if bool(_m9.any()) else 9e9
                    if bool(_m9.any()) and _cur9 <= _endbest.get(_nm9, 9e9):
                        _endbest[_nm9] = _cur9
                        _e0 = int(_np3.argmin(_t3.where(_m9, _fe9, _t3.full_like(_fe9, 9e9)).detach().cpu().numpy()))
                        _og9 = env_raw.scene.env_origins[_e0]
                        _tp9 = (env_raw.hand.data.body_pos_w[_e0, env_raw.tip_ids]
                                - _og9)
                        _op9 = env_raw.object.data.root_pos_w[_e0] - _og9
                        _oq9 = env_raw.object.data.root_quat_w[_e0]
                        from isaaclab.utils.math import quat_apply as _qa9
                        _tw9 = _qa9(_oq9.unsqueeze(0).expand(5, 4),
                                    env_raw._fc_local) + _op9.unsqueeze(0)
                        _fp9 = _sd9.data.get("_fin_ref_path")
                        _wp9 = (env_raw.wrist_pos_w[_e0] - _og9).detach().cpu().numpy()
                        _wq9 = env_raw.wrist_quat_w[_e0].detach().cpu().numpy()
                        _aq9 = env_raw.hand.data.joint_pos[
                            _e0, env_raw.arm_jids].detach().cpu().numpy()
                        _qa9j = env_raw.hand.data.joint_pos[_e0, env_raw.hand_jids]
                        _endsnap[_nm9] = (_tp9.detach().cpu().numpy(),
                                          _tw9.detach().cpu().numpy(),
                                          _op9.detach().cpu().numpy(),
                                          _oq9.detach().cpu().numpy(),
                                          int(env_raw.ref_t[_e0]),
                                          _qa9j.detach().cpu().numpy(),
                                          (_fp9[-1].detach().cpu().numpy()
                                           if _fp9 is not None else None),
                                          env_raw.finger_tgt[_e0].detach().cpu().numpy(),
                                          int(getattr(env_raw, "_dbg_fi2",
                                              env_raw.ref_t)[_e0]),
                                          _wp9, _wq9, _aq9,
                                          _sd9.data.get("_fc_meas_wp"),
                                          _sd9.data.get("_fc_meas_wq"),
                                          _sd9.data.get("_fc_meas_aq"),
                                          (_sd9.data.get("retract_path")[-1]
                                           if _sd9.data.get("retract_path") is not None
                                           else None))
            for _nmR, _sdR in _sides3:
                _sgR = _sdR.data.get("_seg_rows")
                if not _sgR: continue
                with _BM3.use_side(env_raw, _sdR):
                    _q0R = env_raw.obj_start_quat            # 本回合复位时的姿态
                    _qnR = env_raw.object.data.root_quat_w
                    _dot = (_q0R * _qnR).sum(dim=1).abs().clamp(max=1.0)
                    _angR = 2.0 * _t3.rad2deg(_t3.acos(_dot))   # (N,) 相对复位的转角
                    _dspR = ((env_raw.object.data.root_pos_w
                              - env_raw.scene.env_origins
                              - env_raw.obj_start_pos).norm(dim=1) * 100)
                    _rtR = env_raw.ref_t.clone()
                for _snR, _loR, _hiR in _sgR:
                    _mR = (_rtR >= _loR) & (_rtR <= _hiR)
                    if not bool(_mR.any()): continue
                    _e = _objrot.setdefault(_nmR, {}).setdefault(_snR, [0, 0.0, 0.0, 0.0])
                    _e[0] += int(_mR.sum())
                    _e[1] += float(_angR[_mR].sum())
                    _e[2] = max(_e[2], float(_angR[_mR].max()))
                    _e[3] += float(_dspR[_mR].sum())
                    # ★ 逐 env 走廊翻倒台账 (2026-08-25, RL_Pour 联查): 段内每个 env
                    #   的**峰值**转角, 用来算"多少比例的 env 在走廊里被顶歪"。
                    #   均值会把抽签式的少数翻倒稀释掉 —— 必须看逐 env 峰。
                    _pe = _envpk.setdefault(_nmR, {}).setdefault(
                        _snR, _t3.zeros(env_raw.num_envs, device=_angR.device))
                    _t3.maximum(_pe, _t3.where(_mR, _angR,
                                               _t3.zeros_like(_angR)), out=_pe)
            _qdpk[0] = max(_qdpk[0], float(env_raw.arm_qd.abs().max()))
            _qdpk[1] = max(_qdpk[1], float(env_raw.finger_qd.abs().max()))
            for _nm4, _sd4 in _sides3:
                with _BM3.use_side(env_raw, _sd4):
                    _Fq = _t3.cat([_s.data.force_matrix_w.view(
                        env_raw.num_envs, 1, 3)
                        for _s in env_raw._contact_sensors], dim=1).norm(dim=-1)
                _on4 = (_Fq > 0.5).sum(dim=1)
                _b4 = int(_on4.max())
                if _b4 > _padpk.get(_nm4, (0, None))[0]:
                    _e4 = int(_on4.argmax())
                    _nm5b = ("拇", "食", "中", "无", "小")
                    _padpk[_nm4] = (_b4, " ".join(
                        f"{_nm5b[_j]}{float(_Fq[_e4, _j]):.1f}N"
                        if float(_Fq[_e4, _j]) > 0.5 else f"{_nm5b[_j]}·"
                        for _j in range(5)))
            if _i3 == _N - 1:
                for _nm3, _sd3 in _sides3:
                    with _BM3.use_side(env_raw, _sd3):
                        _rp = _sd3.data.get("retract_path")
                        # ⚠ 只能对 in_ctrl(PREGRASP) 的 env 比对: GRASP 相位下臂
                        #   走的是 arm_center±band 分支, 本来就不跟参考 (不是 bug)
                        _msk = (env_raw.task_phase == 0)
                        if _rp is not None and bool(_msk.any()):
                            # ⚠ off-by-one: q_cmd 是用**推进前**的 ref_t 算的,
                            #   _get_dones 之后 ref_t 才 +1 ⟹ 这里必须回退一行
                            _t4 = (env_raw.ref_t - 1).clamp(min=0).clamp(
                                max=_rp.shape[0] - 1)
                            _qerr[_nm3] = float(_t3.rad2deg(
                                (env_raw.q_cmd - _rp[_t4]).abs()[_msk]).max())
                        _absdev[_nm3] = float(
                            _t3.rad2deg(env_raw.arm_res.abs()).max())
                        _nctrl[_nm3] = int(_msk.sum())

    # ---- ⑤ 方案C 行为探针: 零动作下 q_cmd 必须逐位等于参考 ----
    print("\n⑤ 方案C (绝对参考+有界残差) 行为探针")
    if getattr(env_cfg, "arm_abs_res", False):
        for _nm3 in _qerr:
            _chk(_qerr[_nm3] < 0.05, f"{_nm3} q_cmd≡q_ref[t]",
                 f"最大偏差 {_qerr[_nm3]:.4f}°  (仅 PREGRASP 相位 {_nctrl[_nm3]} env; 零动作下必须 ~0)")
            _chk(_absdev[_nm3] < 1e-6, f"{_nm3} arm_res",
                 f"最大 {_absdev[_nm3]:.6f}°  (零动作下必须恒 0)")
        _show("ff/ff_pull", "绝对模式下差分前馈与回拉锚均不参与计算 (结构事实, 非检查)")
    else:
        _chk(False, "arm_abs_res", "未开启 —— 走的是差分积分器老路", hard=False)

    # ---- ⑤a approach_only 下的死相位闸 ----
    # approach_only 把 to_grasp 强制清零 ⟹ 进了 GRASP 就**永远出不来**(candidate 结构
    # 不可达); 而 arrive 判据是 `ok = in_app & ...`, GRASP 相位的 env 永远不可能 arrive,
    # 也就永远不可能贡献 g2 成功。RSI(direct_grasp_prob/dgp_rsi_hi) 让 env 出生即 GRASP,
    # 于是那一批全是死的。实测 AAGE v2: 左手 40~58% env 卡死相位, arrive/pad_* 精确归零。
    if getattr(env_cfg, "approach_only", False):
        # ---- ⑤i 相位表一致性 + 预算足够 + 纯前馈可达 (2026-08-25 新增三条) ----
        print("\n⑤i 相位表/预算/可达性 (e2e R=145 L=1050 事故的守门员)")
        _ptab = {}
        for _nm3, _sd3 in _sides3:
            with _BM3.use_side(env_raw, _sd3):
                _ptab[_nm3] = [int(x) for x in env_raw.phase_timeout_t.tolist()]
        _vals = list(_ptab.values())
        # ★ 原条件是 `len(_vals) < 2 or ...` —— 采不到两侧时**短路成通过**,
        #   正是"没数据长得像通过"。缺侧本身就是故障, 必须红。
        _chk(len(_vals) == 2 and _vals[0] == _vals[1],
             "两侧 phase_timeout_t 逐项相等",
             " | ".join(f"{k}={v}" for k, v in _ptab.items()) +
             "  (不等 ⟹ 构造期两次写落到了不同侧, 短的那侧钟先响, "
             "纯前馈走到半路被重置, 双侧 arrive 门永不齐)")
        # 预算必须容得下参考行数 (e2e: 参考 380 行 vs 预算 145 步 ⟹ 永远走不完)
        for _nm3, _sd3 in _sides3:
            _rp3 = _sd3.data.get("retract_path")
            if _rp3 is None:
                # 传了 --curobo_ref 却没装上 ⟹ 前馈掉回人手轨迹+gs 上限, 必须判死;
                # 没传则本条不适用, 静默跳过是对的。
                if getattr(args, "curobo_ref", ""):
                    _chk(False, f"{_nm3} 接近预算容得下参考",
                         "传了 --curobo_ref 但 retract_path 未装载 ⟹ 前馈掉回 "
                         "q_ref[t] 并被 _ff_end(≈gs) 截断")
                continue
            _need = int(_rp3.shape[0])
            _bud = _ptab[_nm3][0]
            _chk(_bud >= _need + 30, f"{_nm3} 接近预算容得下参考",
                 f"预算 {_bud} 步 vs 参考 {_need} 行 (+30 裕量)  "
                 f"(不够 ⟹ 纯前馈走到半路钟就响, arrive 恒 0)")
        # 纯前馈(零动作)必须真能走到 arrive —— 参考自己走不到, 策略再学也没用
        for _nm3, _sd3 in _sides3:
            # ★ 2026-08-26 (RL_Pour ④): 守门员在**数据缺失时必须 FAIL, 不能静默跳过** ——
            #   静默跳过 = 检查看起来"过了", 和没写这条检查没区别。
            #   arrived 每种配置都有, 取不到只可能是采集器坏了。
            _ar3 = _arrived_ever.get(_nm3)
            if _ar3 is None:
                _chk(False, f"{_nm3} 零动作纯前馈可达 arrive",
                     "采集不到 arrived —— 守门员本身坏了, 不是被检查项坏了")
                continue
            # ★ 2026-08-26 收紧: 原来写的是 `_ar3 > 0`(只要 1 个 env 曾到位就算过),
            #   而三个兄弟检查都是 ≥90%。后果:RL_Pour 的 e2e 把 arrive 门封印成
            #   0.1cm/1.1°(pour_carry 块的历史设计), 1400 步 × 512 env 里只要有**一个**
            #   env 瞬时穿过就判过 —— 预检绿灯, 训练却 189 个 epoch arrive 恒 0, 白跑 3M。
            #   **"至少一个"从来不是可达性判据, 它只证明"不是完全不可能"。**
            _N3 = env_raw.num_envs
            _chk(_ar3 >= 0.9 * _N3, f"{_nm3} 零动作纯前馈可达 arrive",
                 f"{_ar3}/{_N3} 曾到位 ({_ar3/max(1,_N3):.0%})  "
                 f"(<90% ⟹ 参考走不到 或 到位门过紧; 先修参考/门, 别调策略)")

        # 第 4 条 (RL_Pour 提供口径): 纯前馈必须能把契约钟推到 free_lo
        _lo4 = int(getattr(env_cfg, "pour_free_lo", -1))
        if _lo4 >= 0 and _carry_ev:
            _N4 = env_raw.num_envs
            _stR = _carry_ev.get("started", 0)
            # ★ 先断言启动率 —— 启动门没过时 _c_t 恒 0, 直接断言行数会误报成"走不到"
            _chk(_stR >= 0.9 * _N4, "carry 启动门 (零动作)",
                 f"{_stR}/{_N4} env 曾启动 ({_stR/max(1,_N4):.0%})  "
                 f"(启动门没过 ⟹ _c_t 恒 0, 下一条会误报)")
            if _stR > 0:
                _rc = _carry_ev.get("reach", 0)
                _chk(_rc >= 0.9 * _N4, "零动作纯前馈可达 free_lo",
                     f"{_rc}/{_N4} env 的 _c_t 走到 ≥{_lo4} "
                     f"(峰 {_carry_ev.get('ct_max', 0)})  "
                     f"(走不到 ⟹ 参考/预算不够, 先修它别调策略)")

        # ★ 闩语义硬项 (2026-08-25, RL_Pour 提出): arrive 到位闩必须在 close 段之后
        #   才触发。若在 advance/insert 段内就闩上 —— 在**会切相位**的配置里
        #   (approach_only=False) 相位切 GRASP ⟹ _adv 变 False ⟹ **编舞时钟冻在
        #   半路**, 而 fin_prog_gate 因 arrived 放行 ⟹ 手指在腕还没走完走廊时就合拢
        #   = 握拳横在瓶位。本 session 的 approach_only 栈相位恒 PREGRASP、时钟照走,
        #   所以同样闩上也不冻 —— 但这条检查对两种栈都该有(闩在中段本身就是错位)。
        _gr0 = int(getattr(env_cfg, "aag_grasp_row", 0))
        if _gr0 > 0 and _latch:
            for _nmL, _pv in _latch.items():
                _rw = _pv.get("row")
                if _rw is None:
                    continue
                _ok_rows = _rw[_rw >= 0]
                if _ok_rows.numel() == 0:
                    continue
                _mn = int(_ok_rows.min())
                # ★ 只在**会切相位**的配置里判死 —— 本机 F 线(approach_only, 确定性
                #   100%)实测就闩在 ref_t=105(advance 段内)却毫无问题, 因为相位恒
                #   PREGRASP、_adv 恒 True、编舞时钟照走。致命的不是"闩在中段", 而是
                #   闩之后**相位切走导致 _adv=False、时钟冻结**(approach_only=False 才有)。
                #   一刀切判死会在已知良品上误报 —— 这条差点被我写成误报源。
                _hard = not getattr(env_cfg, "approach_only", False)
                _chk(_mn >= _gr0, f"{_nmL} 到位闩时刻 ≥ close 段起始行",
                     f"最早闩在 ref_t={_mn} vs close 起始 {_gr0}  "
                     + ("(会切相位的配置: 闩后 _adv=False ⟹ 编舞时钟冻在半路, "
                        "而 fin_gate 因 arrived 放行 ⟹ 手指提前合拢=握拳横在物体位)"
                        if _hard else
                        "(approach_only: 相位恒 PREGRASP, 时钟照走 ⟹ 仅提示不判死; "
                        "F 线 100% 配方实测也闩在 105)"),
                     hard=_hard)

        # ★ fin_gate 钳死闸 (2026-08-25, RL_Pour 揭出): --fin_gate 若没配到位判据
        #   (eps_pos_cm), 腕永远进不了放行距离 ⟹ 指参考被永久钳在 fin_hold_row
        #   ⟹ 永不合拢。**验行为不验配置**: 直接看指参考行有没有越过钳位。
        if getattr(env_cfg, "fin_prog_gate", False) and _fgv.get("n"):
            _pz = _fgv.get("past", 0)
            _chk(_pz >= 0.9 * _fgv["n"], "指参考越过钳位行 (零动作)",
                 f"{_pz}/{_fgv['n']} env 的指参考行曾 > fin_hold_row({_fgv['hold']})  "
                 f"(钳死 ⟹ 永不合拢; 查 --fin_gate 是否配了 --eps_pos_cm/--fin_gate_dpos_cm)")
        if getattr(env_cfg, "grip_g", False):
            _chk(_fgv.get("g_max", 0.0) > 0.0, "grip_g 慢档有推进 (零动作)",
                 f"峰值 {_fgv.get('g_max', 0.0):.4f}  "
                 f"(恒 0 ⟹ 没进 squeeze 段或快照没拍上, 交棒给 pour 侧时会是 0)")

        print("\n⑤a 死相位闸 (approach_only: 进 GRASP 就出不来)")
        for _nm3, _sd3 in _sides3:
            with _BM3.use_side(env_raw, _sd3):
                _fg = float((env_raw.task_phase == 1).float().mean())
            _chk(_fg < 0.02, f"{_nm3} GRASP相位占比",
                 f"{_fg:.1%}  (>2% = 有 env 出生即卡死相位, 查 direct_grasp_prob/dgp_rsi_hi)")

    if getattr(env_cfg, "seg_gate", False):
        print("\n⑤e 分段探索门控")
        for _nm3, _sd3 in _sides3:
            _sg = _sd3.data.get("_seg_rows"); _tm = _sd3.data.get("_thumb_mask")
            # 判据不写死段数 —— 与门控表长一致即可(段数由参考决定)
            _chk(_sg is not None and len(_sg) == len(env_cfg.seg_arm_scale),
                 f"{_nm3} 段表({len(_sg or [])}段, 门控表{len(env_cfg.seg_arm_scale)}项)",
                 str([(n, l, h) for n, l, h in (_sg or [])]) if _sg else "缺失!")
            _chk(_tm is not None and int(_tm.sum()) > 0, f"{_nm3} 拇指掩码",
                 f"{int(_tm.sum()) if _tm is not None else 0}/22 关节识别为拇指")
        print(f"     臂 {env_cfg.seg_arm_scale}")
        print(f"     拇 {env_cfg.seg_thumb_scale}")
        print(f"     余 {env_cfg.seg_other_scale}")
    if _geom:
        print("\n⑤g ★ 几何审计 (零动作, 合拢前窗口)")
        _bn = list(env_raw.hand.body_names)
        for _nm7, _g7 in _geom.items():
            with _BM3.use_side(env_raw, env_raw._ns[_nm7]):
                _bid = list(env_raw.table_bids)
                _npts = int(env_raw.obj_points.shape[0])
                _ext = (env_raw.obj_points.max(0).values
                        - env_raw.obj_points.min(0).values).cpu().numpy() * 100
            P = _np3.concatenate(_g7["per"], 0)      # (T*, K)
            D = _np3.concatenate(_g7["dsp"], 0)      # (T*,)
            mn = P.min(axis=1)
            print(f"\n  ── {_nm7} ──  {len(_bid)} 个 body | 物体 {_npts} 采样点 "
                  f"包围盒 {_ext[0]:.1f}x{_ext[1]:.1f}x{_ext[2]:.1f}cm | 样本 {len(mn)}")
            _q = _np3.percentile(mn*100, [0,1,5,25,50])
            print(f"     外壳-物体最近距离(body原点口径) cm: "
                  f"min {_q[0]:.2f} p1 {_q[1]:.2f} p5 {_q[2]:.2f} "
                  f"p25 {_q[3]:.2f} 中位 {_q[4]:.2f}")
            _am = P.argmin(axis=1)
            _u, _c = _np3.unique(_am, return_counts=True)
            _o = _np3.argsort(-_c)[:4]
            print("     最常成为最近点的 body: " + "  ".join(
                  f"{_bn[_bid[_u[z]]]}={_c[z]/len(_am):.0%}" for z in _o))
            print("     逐 body 最小距离(cm): " + "  ".join(
                  f"{_bn[_bid[z]].split(chr(95),1)[-1]}={P[:,z].min()*100:.2f}"
                  for z in _np3.argsort(P.min(axis=0))[:5]))
            _dd = _np3.diff(D, prepend=D[0])
            # ⚠ 只取**移动起始**样本: 物体一旦被推走会持续漂, 把后续步也算进来会
            #    污染分布(右手实测 p90=18.43cm, 明显是已经推开之后的读数)。
            #    判据: 这一步在动, 且累计位移还很小(<1mm) = 刚接触那一刻。
            _mv = (_np3.abs(_dd) > 2e-5) & (D < 1e-3)
            if _mv.sum() > 10:
                _t = _np3.percentile(mn[_mv]*100, [10,50,90])
                print(f"     ★物理标定 —— 物体开始动(|Δ|>0.02mm)时的距离 cm: "
                      f"p10 {_t[0]:.2f} 中位 {_t[1]:.2f} p90 {_t[2]:.2f}")
                print(f"       ⟹ **等效接触阈 ≈ {_t[1]:.2f}cm**(原点口径)。余量设在它**之上**才是真正的\"不许碰\"; 设在它之下等于允许接触。")
            else:
                print("     ★物理标定: 窗口内物体几乎没动, 无法标定(样本不足)")
            # —— 按参考段号拆解: 到底哪一段、哪个部位先碰 ——
            R = _np3.concatenate(_g7["rt"], 0)
            _segs7 = _sd3.data.get("_seg_rows") or []
            print("     ── 逐段拆解 (段内最近距离 / 最常最近的部位) ──")
            for _sn7, _lo7, _hi7 in _segs7:
                _m7 = (R >= _lo7) & (R <= _hi7)
                if _m7.sum() < 20: continue
                _p7 = P[_m7]; _mn7 = _p7.min(axis=1)
                _am7 = _p7.argmin(axis=1)
                _u7, _c7 = _np3.unique(_am7, return_counts=True)
                _top = _u7[_np3.argmax(_c7)]
                _dm7 = D[_m7]
                print(f"       {_sn7:<14}行{_lo7:3d}-{_hi7:3d} "
                      f"最近 {_mn7.min()*100:5.2f}cm 中位 {_np3.median(_mn7)*100:5.2f}cm"
                      f" | 主贴 {_bn[_bid[_top]].split(chr(95),1)[-1]:<18}"
                      f" | 物体位移 中位 {_np3.median(_dm7)*100:.2f} "
                      f"p90 {_np3.percentile(_dm7,90)*100:.2f} "
                      f"p99 {_np3.percentile(_dm7,99)*100:.2f}cm")
                # ⚠ 不报 max: 16629 样本里的单个离群(回合复位瞬间 obj_start_pos
                #   基线未更新)会给出 8.7cm 这种假数, 2026-08-24 我据此误判过一次
            # —— 首次移动那一刻 ——
            _first = _np3.where((_np3.abs(_np3.diff(D, prepend=D[0])) > 2e-5) & (D < 5e-4))[0]
            if len(_first):
                _f0 = _first[0]
                _bi0 = int(P[_f0].argmin())
                print(f"     ★首次移动 @ref_t={int(R[_f0])} : 最近部位 "
                      f"**{_bn[_bid[_bi0]]}** 距 {P[_f0, _bi0]*100:.2f}cm"
                      f" | 该刻前五近: " + "  ".join(
                      f"{_bn[_bid[z]].split(chr(95),1)[-1]}={P[_f0,z]*100:.2f}"
                      for z in _np3.argsort(P[_f0])[:5]))

    print("\n⑤f 参考自身外壳-物体间隙 (零动作, 合拢前窗口)")
    for _nm3, _sd3 in _sides3:
        _v = _shellmin.get(_nm3)
        print(f"  [ii] {_nm3}: 窗口内最小间隙 "
              + (f"**{_v*100:.2f}cm** ⟹ 余量须设在其下才不罚参考"
                 if _v is not None else "(无记录)"))

    # ---- ⑤h FC 目标姿态 vs 参考终态 (定位"五指同步偏"的根因) ----
    print("\n⑤h FC 目标姿态 vs 参考实际终态")
    for _nm3, _sd3 in _sides3:
        _rp9 = _sd3.data.get("retract_path")
        _p2a = _sd3.data.get("_p2_arm")
        _p2f = _sd3.data.get("_p2_fin")
        _frp = _sd3.data.get("_fin_ref_path")
        if _rp9 is None or _p2a is None:
            print(f"  [--] {_nm3}: 缺数据"); continue
        _da = float(_t3.rad2deg((_p2a[-1] - _rp9[-1]).abs().max()))
        _df = (float(_t3.rad2deg((_p2f - _frp[-1]).abs().max()))
               if (_p2f is not None and _frp is not None) else float("nan"))
        _chk(getattr(env_cfg, 'fc_target_ref_end', False) or _da < 1.0, f"{_nm3} 臂: 测量位形 vs 参考终态",
             hard=not getattr(args, "pour_place", False),
             detail=f"最大关节差 {_da:.2f}°  (>1° ⟹ 两次独立IK落在不同零空间分支, 指尖会整体偏移)")
        print(f"       指: 测量位形(_p2_fin) vs 参考终态 最大差 {_df:.2f}°")
        # ★ 目标点是否落在物体**内部** —— 传送式测量会把指尖压进物体, 真实 PD 手
        #   顶着表面永远够不到 ⟹ g2 结构性不可达
        _fcl = _sd3.data.get("_fc_local")
        _opt = _sd3.data.get("obj_points")
        if _fcl is not None and _opt is not None:
            _dd9 = _t3.cdist(_fcl.unsqueeze(0), _opt.unsqueeze(0))[0]   # (5,M)
            _near = (_dd9.min(dim=1).values * 100).detach().cpu().numpy()
            _rt = _fcl.norm(dim=1) * 100
            _rs = _opt.norm(dim=1).max() * 100
            print(f"       目标点→物体表面 最近距离(cm): "
                  + "  ".join(f"{v:.2f}" for v in _near))
            print(f"       目标点离物心 {[f'{float(v):.1f}' for v in _rt]}cm | 物体最大半径 {float(_rs):.1f}cm")

    # ---- ⑤d 合拢前禁触窗口 ----
    if float(getattr(env_cfg, "pre_close_w", 0.0)) > 0:
        print("\n⑤d 合拢前禁触")
        _gr = int(getattr(env_cfg, "aag_grasp_row", 0))
        _chk(_gr > 0, "禁触窗口",
             f"ref_t < {_gr} (close 段起始) | 死区 "
             f"{getattr(env_cfg,'pre_close_dead_cm','?')}cm | w={env_cfg.pre_close_w}")

    # ---- ⑤c 手指残差界 (臂有 ±arm_abs_dev, 手指也必须有 —— 否则同款漂移) ----
    print("\n⑤c 手指残差累计上限 (无界=漂移温床, 臂那个 bug 的同款)")
    for _nm3, _sd3 in _sides3:
        with _BM3.use_side(env_raw, _sd3):
            _fdm = _t3.rad2deg(env_raw.finger_dev_max)
        _chk(float(_fdm.median()) < 60.0, f"{_nm3} finger_dev_max",
             hard=not getattr(args, "pour_place", False),
             detail=f"逐关节 中位 {float(_fdm.median()):.1f}° 峰 {float(_fdm.max()):.1f}° "
             f"(>60° 视为无界; 臂的对照是 {getattr(env_cfg,'arm_abs_dev',0)*57.3:.1f}°)")

    # ---- ⑤b 回合寿命 (参考要 ~250 步才播完; 活不到就永远见不到 close/squeeze) ----
    print("\n⑤b 回合寿命与终止原因 (零动作)")
    _mx = max(_eplen) if _eplen else 0
    _chk(int(_term.get("newly_success", 0)) > 0 or (_mx >= 230), "回合寿命",
         hard=not getattr(args, "pour_place", False),
         detail=f"零动作下最长回合 {_mx} 步 "
         f"(参考需 ~250 步播完 close+squeeze; 活不到就永远学不到合拢)")
    print(f"     终止原因累计(全env×{_N}步): "
          + (", ".join(f"{k}={v}" for k, v in _term.most_common()) or "(无)"))

    # ---- ⑥ 基线行为 (黄金回归基准: 任何一项漂了都说明配置/代码变了) ----
    print("\n⑥ 零动作基线行为 (黄金基准, 漂了就是有东西被改坏)")
    for _nm3, _sd3 in _sides3:
        _bp = _padpk.get(_nm3, (0, None))
        print(f"  {_nm3:6s} 全程峰值垫数={_bp[0]}"
              + (f"  逐垫力[{_bp[1]}]" if _bp[1] else ""))
        with _BM3.use_side(env_raw, _sd3):
            _F3 = _t3.cat([_s.data.force_matrix_w.view(env_raw.num_envs, 1, 3)
                           for _s in env_raw._contact_sensors], dim=1).norm(dim=-1)
            _mg = _F3[0].cpu().numpy()
            _np3d = int((_mg > 0.5).sum())
            _od3 = float((env_raw.object.data.root_pos_w[0]
                          - env_raw.scene.env_origins[0]
                          - env_raw.obj_start_pos[0]).norm()) * 100
            _g2d = float(getattr(env_raw, "_g2_d", _t3.zeros(1))[0]) * 100
            _g2f = float(getattr(env_raw, "_g2_fe", _t3.zeros(1))[0])
            _g2k = bool(getattr(env_raw, "_g2_done",
                                _t3.zeros(1, dtype=_t3.bool))[0])
            _arr3 = bool(env_raw.arrived[0])
        _nm5 = ("拇", "食", "中", "无", "小")
        _pads = " ".join(f"{_nm5[_j]}{_mg[_j]:.1f}N" if _mg[_j] > 0.5
                         else f"{_nm5[_j]}·" for _j in range(5))
        _u = "cm" if getattr(env_cfg, "fin_cart", False) else "rad"
        _fv = _g2f * 100 if _u == "cm" else _g2f
        print(f"  {_nm3:6s} 垫数={_np3d} [{_pads}] 物体位移={_od3:.2f}cm")
        print(f"  {'':6s} arrived={_arr3} 腕→真抓姿={_g2d:.2f}cm "
              f"指最差={_fv:.2f}{_u} g2_done={_g2k}")
        if _u == "cm" and getattr(env_cfg, "fc_target_ref_end", False):
            _ev = _endfe.get(_nm3)
            _emin = min(_ev) if _ev else float("nan")
            _chk(_emin < 1.0, f"{_nm3} 零动作@参考终段 指误差",
                 f"最小 {_emin:.2f}cm ({len(_ev or [])} 次采样)  "
                 f"(目标点取参考终态 ⟹ 完美复现参考时必须 ≈0)")
            _cg3 = _candg.get(_nm3)
            if _cg3:
                print(f"  {'':6s} ★candidate 逐闸通过率(零动作): " + "  ".join(
                    f"{_k}={100*_v[1]/max(1,_v[0]):.1f}%" for _k, _v in _cg3.items()))
                _cf = _cg3.get("★合闸", [1, 0])
                # ★ approach_only 下 candidate 链是**设计上**跑不起来的遗留指标
                #   (相位永远到不了 GRASP), 成功走 g2 —— 所以只提示不判死。
                print(f"  {'':6s}    candidate 合闸 {_cf[1]}/{_cf[0]} "
                      f"= {100*_cf[1]/max(1,_cf[0]):.2f}% "
                      f"(approach_only 下为遗留死指标, 成功看 g2, 不判死)")
            if _nm3 == _sides3[-1][0]:
                _ns5 = int(_term.get("newly_success", 0))
                # ★ 2026-08-26 同族收紧(见 arrive 那条): ">0" 只证明"不是完全
                #   不可能", 一次侥幸就能发证。改用 env 数的 1/4 作底 —— F 线零动作
                #   实测 123 次 / 64 env(≈2 次每 env), 16 是安全下限。
                _N5 = env_raw.num_envs
                _chk(_ns5 >= 0.25 * _N5, "双臂成功 (零动作)",
                     f"{_ns5} 次 (须 ≥{0.25 * _N5:.0f} 才算可复现)  "
                     f"(参考本身就是成功抓握 ⟹ 零动作必须有 env 成功; "
                     f"恒 0 = 成功判据没接上, 终局奖金一分不发)")
            _c3 = _g2cnt.get(_nm3)
            if _c3 and _c3[0]:
                _gg3 = _g2gate.get(_nm3)
                if _gg3:
                    print(f"  {'':6s}    ★g2 逐闸(终段): " + "  ".join(
                        f"{_k}={100*_v[1]/max(1,_v[0]):.1f}%" if _k != "腕转(度)"
                        else f"腕转均{_v[1]/max(1,_v[0]):.2f}°"
                        for _k, _v in _gg3.items()))
                # ★ 权威计数在 bimanual 内部(两侧都还没复位的唯一时刻)。step 之后
                #   再采样, 活动侧(A)的 _g2_done 已被"成功即终止"的复位清掉 ⟹ 必读 0
                #   (实测 right 0 次 vs 双臂成功 123 次自相矛盾, 错在探针不在代码)。
                _evt = getattr(env_raw, "_g2_evt", None)
                _tg3 = "A" if _nm3 == getattr(getattr(env_raw, "_A", None), "name", "") \
                    else "B"
                _n3 = int(_evt.get(_tg3, 0)) if _evt else _c3[1]
                _chk(_n3 >= 0.25 * env_raw.num_envs,
                     f"{_nm3} g2 达成次数 (零动作)",
                     f"{_n3} 次 (step 后采样口径 {_c3[1]} 次, 活动侧必偏低, 仅供参考)  "
                     f"(参考本身就能抓成 ⟹ 零动作必须命中, 否则 RL 永远拿不到成功信号)")
            _sn = _endsnap.get(_nm3)
        if locals().get("_sn") is not None and len(_sn) >= 9 \
                and _sn[6] is not None:   # 防御: 非 seg_gate 旗集不进上面的采样循环
            import numpy as _np8
            _dr8 = _np8.degrees(_np8.abs(_sn[5] - _sn[6]))
            _dt8 = _np8.degrees(_np8.abs(_sn[5] - _sn[7]))
            _pf8 = _np8.linalg.norm(_sn[0] - _sn[1], axis=1) * 100
            print(f"  {'':6s} ★取样@ref_t={_sn[4]} (指参考行={_sn[8]}): "
                  f"指关节 vs 参考末行 max {_dr8.max():.2f}° 均 {_dr8.mean():.2f}° | "
                  f"vs 下发目标 max {_dt8.max():.2f}° 均 {_dt8.mean():.2f}°")
            print(f"  {'':6s} ★逐指距目标(cm): " +
                  " ".join(f"{_n}{_v:.2f}" for _n, _v in zip("拇食中无小", _pf8)))
            _dv8 = (_sn[0] - _sn[1]) * 100
            print(f"  {'':6s} ★指尖偏移向量(cm) 均 [{_dv8.mean(0)[0]:+.2f} "
                  f"{_dv8.mean(0)[1]:+.2f} {_dv8.mean(0)[2]:+.2f}] "
                  f"离散度(各指绕均值) {_np8.linalg.norm(_dv8-_dv8.mean(0),axis=1).mean():.2f}cm "
                  f"(离散小 ⟹ 整只手刚性平移)")
        if locals().get("_sn") is not None and len(_sn) >= 16 \
                and _sn[12] is not None:   # 防御: 非 seg_gate 旗集 (同 2050 处)
            import numpy as _np9
            _wpr, _wqr, _aqr = _sn[9], _sn[10], _sn[11]
            _wpm = _sn[12].detach().cpu().numpy(); _wqm = _sn[13].detach().cpu().numpy()
            _aqm = _sn[14].detach().cpu().numpy()
            _dw = (_wpr - _wpm) * 100
            _dq = _np9.degrees(2*_np9.arccos(min(1.0, abs(float((_wqr*_wqm).sum())))))
            _da9 = _np9.degrees(_np9.abs(_aqr - _aqm))
            print(f"  {'':6s} ★腕位 取样 vs 拍照: 位置差 [{_dw[0]:+.2f} {_dw[1]:+.2f} "
                  f"{_dw[2]:+.2f}]cm 共 {_np9.linalg.norm(_dw):.2f}cm | 姿态差 {_dq:.2f}° "
                  f"| 臂关节差 max {_da9.max():.2f}° 均 {_da9.mean():.2f}°")
            if _sn[15] is not None:
                _rpe = _sn[15].detach().cpu().numpy()
                print(f"  {'':6s} ★臂关节 取样 vs 参考末行 max "
                      f"{_np9.degrees(_np9.abs(_aqr-_rpe)).max():.2f}° | "
                      f"拍照 vs 参考末行 max "
                      f"{_np9.degrees(_np9.abs(_aqm-_rpe)).max():.2f}°")
            if _sn:
                _tp9, _tw9, _op9, _oq9, _rt9 = _sn[:5]
                _nm5b = ("拇", "食", "中", "无", "小")
                print(f"       ── 逐指世界位置 @ref_t={_rt9} (env系, cm) ──")
                print(f"       {'指':<4}{'实际XYZ':>26}{'目标XYZ':>26}{'差XYZ':>26}{'距':>7}")
                for _z9 in range(5):
                    _a9 = _tp9[_z9]*100; _b9 = _tw9[_z9]*100; _c9 = _a9-_b9
                    print(f"       {_nm5b[_z9]:<4}"
                          f"{_a9[0]:>8.2f}{_a9[1]:>9.2f}{_a9[2]:>9.2f}"
                          f"{_b9[0]:>8.2f}{_b9[1]:>9.2f}{_b9[2]:>9.2f}"
                          f"{_c9[0]:>8.2f}{_c9[1]:>9.2f}{_c9[2]:>9.2f}"
                          f"{_np3.linalg.norm(_c9):>7.2f}")
                _mean = (_tp9-_tw9).mean(axis=0)*100
                _spread = ((_tp9-_tw9)*100 - _mean).__abs__().max()
                print(f"       ★ 五指平均偏移 [{_mean[0]:+.2f} {_mean[1]:+.2f} {_mean[2]:+.2f}]cm | 逐指离散度 {_spread:.2f}cm")
                print(f"       ★ 判读: 离散度<<平均 ⟹ **整体平移**(物体/基座错位); 离散度≈平均 ⟹ 逐指各自顶不到")
                print(f"       物体位置 {_np3.round(_op9*100,2)}cm 姿态 {_np3.round(_oq9,3)}")
                _mo = _sd3.data.get("_fc_meas_op"); _mq = _sd3.data.get("_fc_meas_oq")
                if _mo is not None:
                    _mo9 = _mo.detach().cpu().numpy(); _mq9 = _mq.detach().cpu().numpy()
                    _dp0 = (_op9 - _mo9) * 100
                    _dq0 = float(2 * _np3.degrees(_np3.arccos(
                        min(1.0, abs(float(_np3.dot(_mq9, _oq9)))))))
                    print(f"       拍照时物体 位置 {_np3.round(_mo9*100,2)}cm "
                          f"姿态 {_np3.round(_mq9,4)}")
                    _chk(abs(_dp0).max() < 0.5 and _dq0 < 2.0,
                         f"{_nm3} 拍照 vs 运行 物体位姿一致",
                         f"位置差 [{_dp0[0]:+.2f} {_dp0[1]:+.2f} {_dp0[2]:+.2f}]cm | "
                         f"姿态差 {_dq0:.2f}°  ← 差多少, 目标点就整体错多少")
                    # 把差异拆成两段: 拍照→回合起点 / 回合起点→终段
                    with _BM3.use_side(env_raw, _sd3):
                        _q0b = env_raw.obj_start_quat[0].detach().cpu().numpy()
                        _p0b = env_raw.obj_start_pos[0].detach().cpu().numpy()
                    _d1 = float(2*_np3.degrees(_np3.arccos(
                        min(1.0, abs(float(_np3.dot(_mq9, _q0b)))))))
                    _d2 = float(2*_np3.degrees(_np3.arccos(
                        min(1.0, abs(float(_np3.dot(_q0b, _oq9)))))))
                    print(f"       ★拆解: 拍照→回合起点 {_d1:.2f}°  |  "
                          f"回合起点→终段 {_d2:.2f}°  "
                          f"| 回合起点位置 {_np3.round(_p0b*100,2)}cm")
                    # 手指实际关节 vs 参考指令 —— 验"PD 顶着物体到不了位"
                    with _BM3.use_side(env_raw, _sd3):
                        _fa = env_raw.hand.data.joint_pos[0, env_raw.hand_jids]
                        _fr = _sd3.data.get("_fin_ref_path")
                        _ft = env_raw.finger_tgt[0]
                    if _fr is not None:
                        _dcmd = float(_t3.rad2deg((_fa - _fr[-1]).abs().max()))
                        _dtgt = float(_t3.rad2deg((_fa - _ft).abs().max()))
                        _dmean = float(_t3.rad2deg((_fa - _fr[-1]).abs().mean()))
                        print(f"       ★指关节 实际 vs 参考末行: 最大 {_dcmd:.2f}° "
                              f"均 {_dmean:.2f}°  |  实际 vs 当前下发目标: 最大 {_dtgt:.2f}°"
                              f"   (大 ⟹ PD 顶着物体到不了位)")
                    with _BM3.use_side(env_raw, _sd3):
                        _fi2d = getattr(env_raw, "_dbg_fi2", None)
                        _rt   = env_raw.ref_t[0]
                        _arr  = env_raw.arrived[0]
                        _dpn  = env_raw._align_err()[0][0]
                    _nrow = _fr.shape[0] if _fr is not None else -1
                    print(f"       ★指参考行: 实际用 {int(_fi2d[0]) if _fi2d is not None else -1}"
                          f" / 应为 {_nrow-1} (ref_t={int(_rt)})"
                          f"  | arrived={bool(_arr)} 腕距={float(_dpn)*100:.2f}cm"
                          f"  | fin_hold_row={int(getattr(env_raw.cfg,chr(39)+chr(39).join([]) or 'fin_hold_row',-1))}")
                    _da = getattr(env_raw, "_dbg_adv", None)
                    if _da and _da[0] > 0:
                        print(f"       ★GRASP相位时钟: 采样 {int(_da[0])} env·步, 放行 {int(_da[1])} "
                              f"({100*_da[1]/_da[0]:.1f}%) | 指跟踪误差 均 {_da[2]/_da[0]:.2f}° "
                              f"峰 {_da[3]:.2f}° (容差 {float(env_raw.cfg.fin_adv_tol_deg):.0f}°) "
                              f"| 已到上限行 {int(_da[4])} 次")
                        env_raw._dbg_adv = [0.,0.,0.,0.,0.]

    print("\n⑥c ★ 物体被碰动/碰转 —— 逐段 (零动作, 相对本回合复位姿态)")
    # ★ 2026-08-26: "没数据"绝不能长得像"通过"。⑥c/⑥d 逐段统计依赖 _seg_rows,
    #   缺表时内层是 `if not _sg: continue` (静默), 表头照打、表体全空 ——
    #   读的人会当成"扰动为零 = 很干净"。这里一次性把未验状态喊出来。
    if not any(_sd.data.get("_seg_rows") for _, _sd in _sides3):
        print("     ⚠ 未验 —— 参考 npz 不带 seg_names/seg_lens, 无段可切。")
        print("       ⑥c/⑥d 本次**没有检查任何东西**, 空表 ≠ 扰动为零。")
        print("       补法: 给参考 npz 加 seg_names/seg_lens (段表随数据走, 不随旗走)。")
    for _nmR, _dR in _objrot.items():
        print(f"  ── {_nmR} ──")
        print(f"     {'段':<14}{'样本':>8}{'转角均':>9}{'转角峰':>9}{'位移均':>9}")
        for _snR, _e in _dR.items():
            if _e[0] == 0: continue
            print(f"     {_snR:<14}{_e[0]:>8}{_e[1]/_e[0]:>9.2f}°"
                  f"{_e[2]:>8.2f}°{_e[3]/_e[0]:>8.2f}cm")

    # ---- ⑥b 参考自身的关节速度 vs 软阈 (qvel_hard 是否在罚参考) ----
    if _envpk:
        print("\n⑥d ★ 走廊逐 env 翻倒台账 (零动作; 段内每 env 转角峰的分布)")
        print(f"     {'段':<14}{'>10°':>7}{'>30°':>7}{'>60°':>7}{'峰':>9}")
        for _nmP, _dP in _envpk.items():
            print(f"  ── {_nmP} ──")
            for _snP, _v in _dP.items():
                _n = _v.numel()
                print(f"     {_snP:<14}"
                      f"{float((_v > 10).float().mean()) * 100:>6.1f}%"
                      f"{float((_v > 30).float().mean()) * 100:>6.1f}%"
                      f"{float((_v > 60).float().mean()) * 100:>6.1f}%"
                      f"{float(_v.max()):>8.1f}°")

    print("\n⑥b 参考自身速度 vs 软阈 (零动作)")
    # ⚠ 峰值被复位瞬间的关节传送污染(实测指 1112 rad/s), 只作参考;
    #    真判据在 ⑦ —— 直接看 qvel_hard 这一项的**奖励值**
    print(f"  [ii] 参考峰值 臂 {_qdpk[0]:.2f} / 指 {_qdpk[1]:.1f} rad/s "
          f"vs 软阈 {getattr(env_cfg,'qd_soft_arm','?')} / "
          f"{getattr(env_cfg,'qd_soft_fin','?')} (峰值含复位传送伪值, 仅参考)")

    # ---- ⑦ 逐项奖励表 (基线归零审计: 零动作 = 复现参考, 各项各给了多少?) ----
    print("\n⑦ 零动作逐项奖励 (= 光复现参考能拿多少分 / 被罚多少)")
    _srl = getattr(env_raw, "_srl", None)
    if not _srl or not _srl["rows"]:
        _chk(False, "step_rew_log", "没有 _srl 行 —— --step_rew_log 没生效?", hard=False)
    else:
        _ks3 = sorted(set(k for r in _srl["rows"] for k in [None]) ) # 占位
        import collections as _co
        _acc = _co.defaultdict(list)
        for _r in _srl["rows"]:
            _acc[_r["side"]].append(_r["terms_mean"])
        _kn = getattr(env_raw, "_srl_keys", None)
        for _sname, _vs in _acc.items():
            _M = _np3.stack(_vs).mean(0)
            print(f"  ── {_sname} ({len(_vs)} 步) 合计 {_M.sum():+.4f}/步 ──")
            _idx = _np3.argsort(_M)
            _neg = [i for i in _idx if _M[i] < -1e-5][:6]
            _pos = [i for i in _idx[::-1] if _M[i] > 1e-5][:6]
            _lab = (_kn if _kn else [f"#{i}" for i in range(len(_M))])
            print("     最负: " + "  ".join(f"{_lab[i]}={_M[i]:+.4f}" for i in _neg))
            print("     最正: " + "  ".join(f"{_lab[i]}={_M[i]:+.4f}" for i in _pos))
        _BAD = 0.005    # 收紧 (AAGE v3: cent_shape -0.0034 在 0.01 下漏网,
                        #  它是稠密罚 ⟹ 梯度里赢过稀疏正奖, 教出"别碰物体")
        for _sname, _vs in _acc.items():
            _M = _np3.stack(_vs).mean(0)
            _lab = (_kn if _kn else [f"#{i}" for i in range(len(_M))])
            _pun = [(_lab[i], _M[i]) for i in range(len(_M)) if _M[i] < -_BAD]
            # ★★ 2026-08-26 新增(RL_Pour r5 自杀通道验尸): 合计必须**为正**。
            #   合计 ≤ 0 ⟹ "活着每步都在亏钱" ⟹ **任何零成本的提前终止都是最优解**
            #   (r5 实测: 策略学会开局拍桌自杀, term/table_crash 0→0.160,
            #    而 Mean 反而"改善" —— 那是回合变短的会计幻觉, 不是学会了)。
            #   这条比"无项在罚参考"更根本: 单项都不罚, 合计仍可能被 time/imit 拖成负。
            # ★ 2026-08-26 (二次修正, RL_Pour 的论证形式更准):
            #   真判据不是"活着为正", 是**"死比活贵"** —— 自杀成为最优解的充要条件是
            #       |终局罚| < |每步基线| × 剩余回合步数
            #   实例: r5 每步 −0.014 × 1650 步 = −23, 而 r_drop 只有 −10
            #   ⟹ 死比活便宜 13 分, 即使 fail 全额也堵不住。
            #   "合计>0" 是充分不必要条件; 这里两条都报, 硬判用平衡式。
            _tot = float(_M.sum())
            _ep = int(getattr(env_raw, "ep_total", 0)) or 1
            _drop = abs(float(getattr(env_cfg, "r_drop", 0.0)))
            _cost = abs(min(_tot, 0.0)) * _ep         # 活满回合的累计代价
            if _tot > 0.0:
                _verdict = "活着为正, 无自杀动机"
            elif _drop > _cost:
                _verdict = "OK: 死更贵"
            else:
                _verdict = "★死更便宜 ⟹ 自杀是最优解"
            _chk(_tot > 0.0 or _drop > _cost,
                 f"{_sname} 活着不比死贵",
                 f"每步 {_tot:+.4f} × {_ep} 步 = 活满代价 {_cost:.1f} vs 终局罚 "
                 f"{_drop:.1f} ({_verdict}; F 线对照 每步 +0.6167)")
            _chk(not _pun, f"{_sname} 无项在罚参考",
                 (f"仍在罚: " + ", ".join(f"{k}={v:+.4f}" for k, v in _pun))
                 if _pun else f"所有负项 |值| < {_BAD}")
        print("  [ii] 判读: 零动作 = 严格复现参考。这里的**正分**是白拿的工资"
              "(策略可以躺着领),\n       **负分**是在罚参考本身 —— 两者都要压到接近 0,"
              "否则奖励和'学参考'冲突。")

    # ---- ⑧ 逐步奖惩落盘 ----
    print("\n⑧ 逐步奖惩记录")
    _lp = getattr(env_cfg, "step_reward_log", "")
    if _lp:
        _nrow = len(_srl["rows"]) if _srl else 0
        _chk(_nrow > 0, "step_rew_log",
             hard=not getattr(args, "pour_place", False),
             detail=f"{_lp} 已累计 {_nrow} 行 (每 1000 行落一片 npz)")
    else:
        _chk(False, "step_rew_log", "未开启 —— 训练将没有逐步奖惩账本", hard=False)

    # ---- ⑨ 录像口径 ----
    print("\n⑨ 录像 (autorec) 口径")
    print(f"  [ii] 录像必须逐旗对齐本次训练, 且 obs 必须 = {env_cfg.observation_space}")
    print(f"  [ii] 本线 --obj_obs={'ON' if getattr(env_cfg,'obj_in_actor',False) else 'OFF'}"
          f"  --arm_abs={'ON' if getattr(env_cfg,'arm_abs_res',False) else 'OFF'}"
          f"  --fin_cart={'ON' if getattr(env_cfg,'fin_cart',False) else 'OFF'}")

    _rps7 = _rsum7 / max(_N, 1)
    for _sd7 in (getattr(env_raw, "_A", None), getattr(env_raw, "_B", None)):
        if _sd7 is None: continue
        _es7 = _sd7.data.get("_ep_sums") or {}
        _it7 = []
        for _k7, _v7 in _es7.items():
            try: _it7.append((_k7, float(_v7.mean())))
            except Exception: pass
        _it7.sort(key=lambda x: x[1])
        _nm7 = getattr(_sd7, "name", "?")
        print(f"  [ii] {_nm7} 零动作逐项(均/回合): 失血 "
              + " ".join(f"{k}={v:+.3f}" for k, v in _it7[:5])
              + " | 进账 "
              + " ".join(f"{k}={v:+.3f}" for k, v in _it7[-5:]))
    _chk(_rps7 > 0.0, "⑦+ 零动作基线合计为正(硬)",
         f"{_rps7:+.4f}/步 (≤0 ⟹ 活着是负期望, 自杀/提前终止成为最优解 —— "
         f"F线对照 +0.6167/步)")
    if getattr(args, "pour_place", False):
        print("\n⑩ v10 PLACE/RELEASE 零动作门")
        _sp10 = int(getattr(env_raw, "_stat_place", 0))
        _sr10 = int(getattr(env_raw, "_stat_rel", 0))
        _hard10 = not getattr(args, "e2e_squeeze", False)
        # 可达性/判据拆分 (2026-08-26 RL_Training 复核): 可达性零动作可测→硬;
        # 判据达标有零残差彩票→软。保住"结构断 vs 运气差"的分辨力
        _ctmax10 = int(getattr(env_raw, "_pf_ct_max", -1))
        _cT10 = int(getattr(env_raw, "_c_T", 0))
        if _ctmax10 >= 0 and _cT10 > 0 and \
                not getattr(args, "e2e_squeeze", False):
            # e2e 豁免: 训练口径的启动焊轴要求真实抓稳验证 (闭环承重), 零动作
            # 钟必不走 —— e2e 的全程可达性由编舞口径 (--pour_choreo) 验证
            # (2026-08-26 fv10: 契约钟零残差走到底)
            _chk(_ctmax10 >= _cT10 - 2, "⑩0 契约钟全程可达(硬)",
                 f"零动作 max _c_t = {_ctmax10}/{_cT10 - 1} "
                 f"(须走到末行-1: 链路结构完整性)")
        # e2e 降软 (2026-08-26): 从站姿零动作要 ~1100 步才走到放稳段, 且零残差
        # 倒水段有先天彩票 (编舞十验) —— 硬门=结构性必挂; 可达性已由四条守门员
        # + 编舞验证覆盖
        _chk(_sp10 > 0, "⑩a PLACE 零动作门",
             f"放稳事件 {_sp10} 次 (参考含回正落定 ⟹ 零动作必须放稳成功)",
             hard=_hard10)
        _chk(_sr10 > 0, "⑩b RELEASE 零动作门",
             f"松手成功 {_sr10} 次 (放稳后零动作跑斜坡 ⟹ 物体必须保持静置)",
             hard=_hard10)
    print("\n" + "=" * 78)
    print(f"[preflight] 本次共执行 {_nassert[0]} 项硬断言 + {_nassert[1]} 项软断言"
          f"  (断言数骤降 = 有检查因缺数据被静默跳过, 值得追)", flush=True)
    if _fail:
        print(f"[preflight] ❌ 硬失败 {len(_fail)} 项: {_fail}")
    else:
        print("[preflight] ✅ 全部硬项通过" + (f" (软告警: {_warn})" if _warn else ""))
    print("=" * 78, flush=True)
    try:
        _slot.release()
    except Exception:
        pass
    import sys as _sys3
    _sys3.stdout.flush()
    os._exit(1 if _fail else 0)
# ========================== 起飞前自检 结束 ==========================

agent.train()
print("[train] 训练结束")
if getattr(env_raw, "score_on", False):
    env_raw.save_score_map(os.path.join(log_dir, "score_map.npz"),
                           extra_meta=dict(final=True))
# ---- 根治 Isaac 退出挂死 (2026-08-19, 用户点名): carb 关闭流程会死锁, 进程
# "活着但退不掉", flock 槽位+显存被永久攥住 (一晚拦路两次)。正事已毕 (ckpt/
# score_map 都落盘), 主动放槽 + 硬退出, 完全绕开会死锁的清理段。
try:
    _slot.release()
    print("[train] GPU 槽位已释放, 硬退出 (跳过 Isaac 关闭流程)")
except Exception:
    pass
import sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
os._exit(0)
