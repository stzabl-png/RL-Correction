"""扭盖任务 v2 — 轨迹跟随口径 (init_videos/1.mp4 的完整任务).

用户裁定 (2026-08-25, 台账 v2 节): 要优化的是**人手示范轨迹** ——
瓶立桌上、左手拿起顺时针转 ~90°、右手拧开盖、盖放回桌面. 结构:

  瓶身   重建轨迹**运动学回放** (拿起/转平/放回由轨迹承载; 真实左手抓握 = v3)
  左臂   人手参考**视觉重放** (IK 左腕轨迹 + ref_qpos_left; 不在动作空间)
  右臂   残差策略, 参考 = retarget 右手轨迹的 IK (tube 结构性保证像人)
  时钟   门控: 人只示范了一小段拧动; 播到拧盖窗末端(hold帧)后冻结,
         拧满 720° 释放才继续播"收手+放盖"段
  成功   释放 **且** 盖落到参考终点 ±place_tol 的桌面上 (释放不再终止回合)

规范系: align_replay 是纯物体驱动 (identity 旋转 + 瓶首帧落桌平移), 左/右手
加载得到同一规范 —— BottleReconstructionEnv.cap_ref 就是对齐好的右手轨迹;
盖参考终点 = 原始 npz 的 obj_pose_all[1] + (对齐瓶轨[0] − 原始瓶轨[0]) 平移.

继承 UnscrewTaskEnv 复用: 帽接触传感器 / drive_mask 螺纹静摩擦 / 观测追加 10 维.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul

from rl_rebuild.correction import clips, frames as F
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
from rl_rebuild.correction.load_replay import load as load_replay
from tasks.recon_kailang.bottle_reconstruction.screw_joint import assembled_cap_pose
from tasks.recon_kailang.bottle_reconstruction.unscrew_env import (
    UnscrewTaskCfg,
    UnscrewTaskEnv,
)


@configclass
class UnscrewRefTaskCfg(UnscrewTaskCfg):
    episode_length_s = 30.0           # ep_total ≈ 402 步 @20Hz, 上限要盖住
    clamp_body = False                # v2: 瓶身回放轨迹, 不钉死
    # ---- v4: 双臂动作空间 (用户裁定: 这是双臂任务, 左手左臂必须受控) ----
    # 动作 58 = 基类右侧 29 (臂7+指22, 语义不变) + 左侧 29 (同构镜像).
    # 左臂: 在左参考轨迹上累积残差, 关节空间偏差钳制 (tube 的关节版);
    # 左指: rate/total 累积, 真实合拢抓握瓶身 (接触可测量).
    # Stage A: 瓶仍运动学回放 (握着"按示范运动的把手", 先学协调与真实握形);
    # Stage B (A 的左握指标达标后): 瓶动态化, 左手真扛重量+抗拧盖反扭矩.
    action_space = 58
    # 基类 415 (含动作58) + 任务 10 + 左侧 70 (q7+qd7+指22+参考差7+残差22+瓶接触5)
    observation_space = 495
    left_arm_dev_max = 0.35           # rad, 左臂累积残差离参考的钳制
    w_left_grip = 0.03                # ≥3 指握瓶的每步小额 (总量 ~12 < 任务核)
    # U33 (2026-08-27): 左手指垫补 SuperGrip. 基类绑定是"整手 LowGrip(0.2) ->
    # fingertip_bodies 覆盖回 SuperGrip(3.0)", 而 fingertip_bodies 只有右侧 5 根,
    # **左手指垫从建仓起一直停在 0.2** —— 整条左手抓握阶梯 (U19~U32a) 都是在
    # 15× 摩擦劣势下做的. 名单 = fingertip_bodies 的左侧镜像 (与 left_tip_ids 同源).
    extra_supergrip_bodies = ["left_thumb_elastomer", "left_index_elastomer",
                              "left_middle_elastomer", "left_ring_elastomer",
                              "left_pinky_elastomer"]
    # ---- U19: 接触密集化 (2026-08-26 Dyn1 取证: "躺平"局部最优 —— 放倒瓶后
    # 双手悬停, 400 步零接触. 稀疏接触奖励从未被采样 = 零梯度; 碰翻即终止的
    # 风险把两只手都推离了物体, 再多步数也爬不出来) ----
    # 年金形式而非势差分: 贴近本身每步给分, "靠近"严格优于"远悬停", 且在接触
    # 距离处饱和 (clamp 到 r0) —— 从回避区到接触是连续上坡, 摸到不吃亏.
    # 权重 0 = 关闭 (Stage A 语义不变); Dyn cfg 里打开.
    w_left_app = 0.0                  # 左指尖->瓶 贴近年金
    w_cap_app = 0.0                   # 右指尖->盖 贴近年金
    sigma_app = 0.04                  # m, 年金衰减尺度
    r0_left = 0.05                    # m, 左握触时指尖-瓶root几何底距 (FK: 径向 4~6.3cm)
    r0_cap = 0.02                     # m, 右触盖时指尖-盖root底距 (盖半径)
    # ---- U20: 左手"别着急" (2026-08-26 用户判读: 接触太急拿不稳, 甚至把瓶
    # 碰飞, 碰飞后手还在乱动. 曲线佐证: grip_any 冲 7% 跌回 1% —— 触到即
    # 扰动被罚退, poking 不是 holding; μ 策略无平滑代价, dither 免费) ----
    # ② 接触窗内罚指尖-瓶**相对**速度: 拿稳 = 指尖随瓶动 (rel≈0 不罚),
    #    拍击/戳 = 大相对速度重罚; 不与"跟随移动参考"冲突.
    # ③ 持握年金: ≥2 指连续接触计数爬坡, hold_horizon 步满额 —— 戳一下不值钱.
    # ④ 左半动作变化率平滑罚: 累积残差管线里恒定运动 Δa≈0, 高频抖动 Δa 大,
    #    专杀 dither 不罚有目的的平滑运动.
    # ⑤ 碰飞即判负: 瓶线速度超限终止 (被动参考拖动峰值 ~0.3 m/s, 1.0 安全).
    # 权重 0 = 全关 (Stage A 语义不变); Dyn cfg 里打开. ①=r0_left 在 Dyn 里收 3cm.
    w_left_slow = 0.0                 # ② 接触窗相对速度罚
    gentle_zone = 0.08                # m, ② 的作用半径 (指尖-瓶root)
    w_left_hold = 0.0                 # ③ 持握年金满额
    hold_horizon = 10                 # 步 (0.5s), ③ 连续接触到满额
    w_left_act_rate = 0.0             # ④ 左半 Δa² 平滑罚
    term_body_speed = 0.0             # m/s, ⑤ >0 时瓶速超限即终止
    # ---- U21: 触碰台阶 (2026-08-26 Dyn3 @10M 判读: 轻/稳/贴齐活 —— 指尖悬在
    # 瓶面外 ~4mm, 接触率仍 1.6%. 悬停处贴近年金已近饱和, "摸上去"边际收益≈0
    # 而扰动风险 >0; 梯子从"贴近"直接跳"≥2 指持续", 缺"碰到 1 指"一级.
    # 传感器真值计分, 免疫距离指标以瓶 root 为参考的几何偏差) ----
    w_left_touch = 0.0                # ≥1 指接触年金 (梯子的缺失一级)
    # ---- U23: 右手"物态解锁" (2026-08-26 用户裁定, 取代 U22 的固定帧冻结) ----
    # U22 尸检 (Dyn5 @3M): 时钟冻在源 35 连左手自己的参考进程 (闭合/收腕斜坡
    # 25→40) 一起冻死, 左接触率归零、全场停摆, gate 一次未开. 教训: 门控不能
    # 拿固定时间/帧当轴, 要拿**物体状态**当轴.
    # 机制: 时钟不再为门冻结 (左手参考全程正常播放); 解锁前右手 29 维残差
    # 清零 —— 纯跟参考走 ("右手其实都是不用动的"), 任务奖励同步门控;
    # 当**右指尖-真实盖距离** < gate_cap_dist 时解锁 (латch). 因果隐含:
    # 参考只把右手带到空中预抓位, 真盖只有被左手把瓶转运到位才会出现在那
    # —— 躺桌/直立的盖离预抓位 20cm+, 永远不解锁, 无法钻营.
    grip_gate = False                 # 开关 (Dyn 生效)
    gate_cap_dist = 0.08              # m, 右指尖-真实盖距离解锁阈
    # U23b: 纯距离门被"顺路经过"骗开 (Dyn6 首窗 51%: 回合开头参考瓶还在桌上,
    # 右手参考扫向参考盖位时从未被搬走的真盖旁经过). 加两个**物态**副条件
    # (仍无时间轴): 瓶已横转 (倾角>60°, 直立未动=0° 挡住) 且已离桌
    # (z > 桌面+gate_body_lift; 数据实测持瓶拧盖 z≈+5.6~6.2cm, 躺倒≈+3.5cm).
    gate_tilt_cos = 0.5               # cos60°, R33 < 此值 = 倾角超 60°
    gate_body_lift = 0.05             # m, 瓶 root 离桌高度阈 (躺倒 0.035 不过)
    # ---- U24b: 落桌判负 (用户裁定: 瓶落桌就是不对, 握紧就要握紧) ----
    # 参考瓶在空中 (ref z > 桌+lift) 而真瓶滞留桌面 (z < 桌+lift) 连续 N 步
    # = 掉瓶失败, 终止. 释放后参考瓶回桌 → 规则自动解除 ("桌上"只在示范说
    # 可以的时候合法). 复检 (pk19 修复后): 被动地板 ~步44-49 本就因跟丢终止,
    # 此规则同刻触发, 地板无回归; 但"躺平 400 步存活"生态位被封死.
    term_body_drop_steps = 0          # >0 启用; 连续步数
    # ---- U25: 失败终止罚 (Dyn8 尸检: 失败回合每步净回报为负 + 碰飞即终止
    # = 自杀按钮, 策略学会开局抽飞瓶止损, body_fly 1.0/tip_speed 2.1 m/s.
    # 一次性大额罚让"以失败结束"严格劣于"挣扎着活", 终止规则保留) ----
    fail_penalty = 0.0                # 失败终止 (lost/fly/drop/cap_fell/wrist) 一次性罚
    # ---- U26: 拇指对握 (2026-08-27 用户发现"有根手指位置不对劲" -> 放大确认
    # 左拇指直挺竖立不参与抓握. 归因: 参考拇指闭合正常 (CMC_FE 0.73→1.92),
    # 是策略残差主动掰直 —— 因为 grip3/hold 计酬"任意 N 指"即可, 四指爪抓
    # 白拿全额, 而无拇指对压的爪抓抗不住旋转扭矩 (举到中途脱手的根源) ----
    grip_need_thumb = False           # True: grip3/hold 计数要求拇指在接触集内
    # ---- U27: 右手持续触盖爬坡年金 (Dyn10 @6M: 左链全面点火后右手卡在
    # 单指拨盘, cap_contact2 精确为 0 —— 左手同款稀疏零梯度病, 同款药) ----
    w_cap_hold = 0.0                  # ≥2 指连续触盖计数爬坡, hold_horizon 步满额
    # U28: 盖版触碰台阶 (U27 判据提前证伪: contact2/hold 双 0, 事件从未发生
    # 梯度无处附着; 且 r0_cap=2cm 贴近年金悬停即满额, 真碰零边际 —— U21 同病)
    w_cap_touch = 0.0                 # ≥1 指触盖年金 (梯子缺失一级)
    # ---- U32: 手型 (2026-08-27 用户看 Dyn15 视频: "两只手的手指都有点不对";
    # 裁定 左手握瓶=五指全到位 (如握杯), 右手拧盖=拇/食/中三指精捏).
    # 根因是 U26 搭便车病的未清剩余: 计酬口径只数**根数**不看**身份** ——
    # 左手 grip3 在 3 指封顶 (第 4/5 指零边际), 右手 n_cap>=2 完全不分身份
    # (小指+无名指扒盖 = 拇食中精捏, 同价). 药: 两处都加**按身份分级**的
    # 正边际, 且**纯增量不撤旧档** (右手 contact2 当前≈0, 撤档会断重学路径).
    w_left_full = 0.0                 # 左: grip3 之上, 第4/5指各给 1/2 额 (需拇指)
    w_cap_triad = 0.0                 # 右: 拇/食/中 每碰一根给 1/3 额
    # ---- U35: 拧盖窗掌轴对准 (2026-08-28 定量取证: 提取腕姿态流在拧盖窗
    # 与携带盖轴夹角均 112.9°, 理想对握 180° —— 上游腕跟踪丢失, 姿态流
    # 与位置流一样不可信; live 61.8° vs 参考 67.1° = 策略忠实跟踪了错参考,
    # 侧蹭式握法拧到 ~200° 就得重抓, 拇指永远落不到对侧) ----
    wr_align_cap = False              # True: 拧盖窗内 hand_z 最短弧旋到 -螺轴
    hold_yaw_fix_deg = 0.0            # U35c: 携带段持瓶朝向 yaw 重定向 (度)
    gate_axis_deg = 0.0               # U37: >0 时解锁门加"瓶轴-参考轴夹角<此值"
    # ---- U38 (2026-08-29 用户裁定): "右手动作参考置信度不低, 位置可能不准,
    # 位置肯定可以用 RL 学" —— 动作用参考, 位置归 RL. 实测: 拧盖窗人手手指
    # 流是拇指主导 (拇 37°/中 24°/食 7.5°), 但冻结时钟把手指参考钉死在一帧
    # 静姿 —— 可信的手势从未被播放, 拧转动作全靠残差瞎摸 (摸出食中拨).
    fin_gesture_loop = False          # True: 冻结窗内右指参考循环播放拧转手势
                                      # ⚠ U38a 已证伪 (Dyn22 尸检: 循环窗含
                                      # 松手段, 反复命令张指与接触收入冲突);
                                      # 保留旋钮仅为存档, 勿再开.
    w_fin_imit = 0.0                  # 右手拇/食/中 关节贴手势参考的年金
    sigma_fin_imit = 0.25             # rad, 上项的 exp 尺度
    # U39: 右手拇指对握门 (U26 左手同款药: 拇指不触盖, contact/chold 不计酬)
    cap_need_thumb = False
    # U42: 单指拨盘不计酬 —— 拧转进度奖 r_screw 须 ≥N 根 (拇/食/中) 触盖.
    # 0=关 (旧行为). 用户裁定 ≥2: "扭了之后我要拿下来, 一手指扭下来了手也拿不住"
    screw_rew_min_triad = 0
    wr_reach_m = 0.0                  # U35b 腕-盖锚距; 0=公式(盖高+净空+伸指垂距
                                      # =23.5cm, 伸指顶抓口径). 深倾斜段该距离
                                      # 沿水平轴伸出会超可达域 (IK 顶替 @57-84
                                      # 取证) —— 卷指对握口径 ~0.19 更合适.
    # U29: 拧转窗年金衰减 (Dyn12 @18M 温床取证: screw 694° 停在自动释放前
    # 不拧完, release 42.6%→16%, cap_hold 2→14 —— 冻结时钟窗内盖侧年金
    # 0.21/步恒流吃到超时, 严格优于"释放 10 一次 + 后段 -6 风险".
    # 药: 解锁后未释放每步计 dwell, 盖侧四项年金线性衰减到 0;
    # 释放瞬间冻结衰减因子 —— 早释放 = 携带段保留更多年金, 激励同向) ----
    # U29b (Dyn13 尸检): dwell 从解锁起计是缺陷 —— 热启动混乱期右手技能先被
    # 打掉, 重学期解锁后 400 步年金即死, 只剩右手乱动的 -6 风险, 解锁净负 →
    # 策略学会把瓶端离右手可及区保持锁定 (gate 64.5%→0.4%, contact2 恒 0,
    # 均奖 60=纯左侧年金). 改: dwell 只在"触盖或贴近<cap_dwell_near"步累计 ——
    # 重学期(远处)因子恒 1, 熟练挤奶期(贴盖)快速累满归零.
    cap_annuity_decay = 0             # >0 启用; engaged 该步数内年金线性归零
    cap_dwell_near = 0.04             # m, 贴近判定半径 (悬停挤奶也计 dwell)
    grip_gate_src_frame = 35          # (U22 遗留, U23 起不再冻结时钟, 仅存档)
    gate_hold_steps = 5               # (U22 遗留, 不再使用)
    gate_body_err = 0.10              # (U22 遗留, 不再使用)
    # 左握几何校准 (2026-08-26 离线 FK 扫描): 提取的左腕站位离瓶偏远 ~6cm
    # (SharpaWave 比例 vs 人手 + 腕位重建粗), retarget 左指卷曲低估 ~46°.
    # 收腕 6cm + FE 屈曲 +0.8 rad 后五指尖落在抓取证据区 (轴向 0.5~6.7cm,
    # 径向 4.0~6.3cm, 离瓶面 0.7~3cm) —— 残差 ±69° 可触达, 握持梯度激活.
    left_standoff_fix = 0.06          # m, 左腕参考沿径向向瓶轴收进
    left_close_bias = 0.8             # rad, 左指屈曲类关节参考偏置
    # U30a (2026-08-27 用户观察 Dyn9 视频"左手抓瓶偏上" + pk20 实测): 携带段
    # 参考腕轴向 ~10cm/拇指 22-23cm 越过盖底 18cm, 挤占右手拧盖区; 而演示
    # 原始数据左腕轴向 ~0cm (抓瓶下段). 机制: U24a 投影系下测的冻结偏移
    # × 携带段原始系四元数回放, 倾角修正被吃两次, 整体抬 ~8cm.
    left_axial_fix = 0.0              # m, 左腕参考沿瓶轴向瓶根下移 (随合拢斜坡渐进)
    # ---- Stage B: 瓶动态化 (A/B 判读结论: 运动学瓶下握持无因果作用,
    # left_grip 年金打不过任务核 advantage —— 必须让"不握就掉") ----
    dynamic_body = False              # False = Stage A (回放); True = Stage B
    k_body_track = 10.0               # 瓶-参考距离势差分系数
    w_body_near = 0.02                # 贴参考的小额年金 (exp 核)
    sigma_body = 0.05
    # 0.28 校准 (2026-08-26): 被动回放的转动滞后峰值 ≈0.21 (地板能力须存活),
    # "不抬不转"漏洞的稳态误差 ≈0.31 (仍被拦截).
    term_body_lost = 0.28             # m, 瓶离参考综合误差超此终止 (掉瓶=失败)
    # 姿态项必须计入 (2026-08-26 Dyn0 判读): 只算位置时策略学会"瓶留桌上不抬不转"
    # (位置差 7cm 不触发终止), 在直立瓶上拧 —— 示范的转 90° 语义丢失.
    # 0.15 m/rad: 直立 vs 横平 (≈1.6 rad) 贡献 0.24 > 阈值, 不转即失败.
    lam_body_rot = 0.15               # m/rad, 瓶姿态测地角折算
    # 关节空间卡死判定对本任务**停用** (pk9/pk10 实测): 人手拧盖窗的肘姿这台
    # 机器人撑不住, j4 稳态偏差冲到 1.99 rad, 但腕端全程 ≤5.5cm —— 臂在用另一种
    # 肘构型执行同一条末端轨迹, 不是卡死. 有意义的信号是**末端跟不上**:
    # 换成 term_wrist_err (2× 零动作包络) 的滞留判定, 在 env._get_dones 实现.
    # 代价 (台账 U12): 拧盖窗内 j4 力矩饱和, 该关节残差权限失效; 拧转由 j6/j7 承担.
    term_arm_err = 9.0                # 实际关闭
    term_wrist_err = 0.12             # m; 腕离参考持续超此值 = 真跟丢
    term_wrist_steps = 20
    # ---- 参考时钟 ----
    # 源帧 48: 瓶正横平 (倾角 ~100°, 峰值在源 40) 且右手在拧 (phase_right [35,72]).
    # ⚠ 不能选拧盖窗末端 68 —— 那时瓶已回正 (人手快, 盖"视觉上"已开), 在横平段
    # 冻结才符合"人在瓶横着时拧"的示范语义; 释放后播放瓶回正+放盖段.
    hold_src_frame = 48
    twist_budget_steps = 220          # 冻结窗预算 (11s, ~7 个 j7 换把循环)
    # ---- 放置 ----
    place_src_frame = 90              # 源帧: 盖已稳定躺桌 (读参考终点用)
    place_tol = 0.08                  # m
    # "真躺桌"判定 (2026-08-25 修): 拧盖时瓶横平, 盖在瓶口 z≈0.88 距目标仅 7.5cm,
    # 旧上限 0.89 让"释放即成功" (record 实测 57 步完成, 盖从未落桌).
    # 盖 root 各种躺姿离桌面 ∈ [+0.2cm, +1.9cm] -> 容差 2.5cm; 加静置速度排除空中/手中.
    # (原始实测值是 0.85 桌面下的 z ∈ [0.852,0.869]/上限 0.875; 判据是
    #  table_top_z + place_z_tol 的**相对**量, 换 0.87 桌面后自动变 0.895, 行为不变.
    #  ⚠ 换场景后仍建议用 record 复测一次盖的躺姿高度, 确认 2.5cm 容差没被吃掉.)
    place_z_tol = 0.025               # m, 盖 root 离**桌面**上限 (相对量)
    place_speed = 0.05                # m/s, 盖静置速度上限
    # U41 (2026-08-30 用户裁定): 成功 = "右手拿着盖放到桌上", 不是盖自己落桌.
    # 旧判据只看"盖在目标点附近静止", 拧脱后自由落体也算 —— 判定错误.
    place_carry_steps = 10            # 释放后右手 ≥2 指尖持盖须累计 ≥ 这么多控制步
    place_max_fall = 0.8              # m/s, 释放后盖下落峰速上限; 自由落体 (~2m/s) 否决
    # ---- v3: 腕参考 = 数据姿势锚定 (用户裁定: 必须从提取的初始姿势出发) ----
    # 数据事实: 腕**平移**流是静态填充, 但腕**姿态**流是活的 (左 64°/右 27°,
    # recon_world 系, 与规范系同向), 静态腕点本身即提取的人手姿势 (局部正确).
    # 右手证据帧 (contact_heatmap_frame0044): 腕距盖 17.3cm, 手指向与腕->盖
    # 方向仅差 29.5° —— 就是伸向盖的抓姿. 位置=物体轨迹锚定, 姿态=数据流.
    grasp_anchor_src = 44             # 右手-盖抓取证据帧 (contact/ 目录)
    contact_start_src = 35            # 右手接触起始帧 (contact_auto_grasp)
    # ---- v3: 螺纹真实化 (用户裁定: 检查旋转建模) ----
    # 每物理子步相对角速度阻尼 —— 螺纹粘滞摩擦抽象: 轻弹一下会立刻衰减,
    # 必须持续接触+持续切向驱动才能拧动 (0.9^12 ≈ 0.28/控制步).
    screw_omega_damping = 0.9
    release_bonus = 10.0              # 释放不再终止, 奖减半 (总回报大头给放置)
    place_bonus = 20.0
    k_place = 10.0                    # 放置势差分 (释放后开)
    # ---- 跟随 ----
    lam_imit = 0.02                   # 腕跟随小额 (总量 ~8 < 任务核 ~68)
    sigma_imit = 0.05
    left_replay = True                # 加载左手参考数据 (v4 起左臂=参考前馈+策略残差, 非回放; 旧名保留)


@configclass
class UnscrewDynTaskCfg(UnscrewRefTaskCfg):
    """Stage B: 动态瓶 —— 左手真扛重量, 瓶轨迹从回放变为跟踪目标."""
    dynamic_body = True
    # U19 (Dyn2 起): 贴近年金开启. 满额 ~0.06+0.06=0.12/步 × ~350 步 ≈ 42,
    # 低于任务核 (~68) 且是其先决路径 (不贴近就没有握/拧), 同向不竞争.
    w_left_app = 0.06
    w_cap_app = 0.06
    # U20 (Dyn3 起): 左手别着急 —— 最后一厘米梯度 + 轻贴 + 持握 + 去抖 + 碰飞判负.
    # ① U19 年金在 4.3cm 饱和 (r0=5cm), 最后一厘米零梯度, 策略停在悬停 (7M 取证).
    r0_left = 0.03
    w_left_slow = 0.05
    w_left_hold = 0.06
    w_left_act_rate = 0.02
    term_body_speed = 1.0
    # U21 (Dyn4 起): 触碰台阶. 梯子: 贴近 0.06 → +碰1指 0.04 → +持2指 0.06
    # → +握3指 0.03 → 瓶跟踪因果收益. 每级都有正边际.
    w_left_touch = 0.04
    # U22 (Dyn5, 已尸检) -> U23 (Dyn6 起): 右手物态解锁 —— 指尖离真盖 <8cm
    # 才放开右手残差与奖励; 瓶参考加强为"较强参考" (用户裁定: 瓶轨迹可信).
    grip_gate = True
    w_body_near = 0.05
    # U24 (Dyn8 起): a=参考静置段直立投影已在 env 内生效; b=落桌判负.
    term_body_drop_steps = 10
    # U25 (Dyn9 起): 失败终止罚 —— 封死"抽飞瓶自杀止损"按钮.
    fail_penalty = 6.0
    # U26 (Dyn10 起): 拇指对握 —— 四指爪抓不再计酬, 对握才算握.
    grip_need_thumb = True
    # U27 (Dyn11 起): 右手持续触盖爬坡 —— 单指拨盘不值钱, 稳捏才值钱.
    w_cap_hold = 0.06
    # U28 (Dyn12 起): 盖版触碰台阶.
    w_cap_touch = 0.04
    # U29 (Dyn13 起)/U29b (Dyn14 起): 拧转窗年金衰减 —— 封死挤奶温床.
    # U31 (Dyn16 起): 回滚年金衰减 —— 两代时钟 (U29 解锁计/U29b 贴近计) 都在
    # 计数边界上诱发回避 (Dyn13 避门; Dyn15 tip_cap 恰停 4.5cm≈边界外悬停,
    # contact2 @9.9M 精确 0 vs Dyn12 同期 51%). 反挤奶改走胡萝卜路线:
    # 释放奖 10→20 + U30b 拧转阈值 270° 本身已大幅缩短温床窗口.
    cap_annuity_decay = 0
    release_bonus = 20.0
    # U32 (Dyn17 起): 手型 —— 左五指包握 / 右拇食中精捏.
    w_left_full = 0.03
    w_cap_triad = 0.05
    # U34 (Dyn18 起): 拧转速率按 triad 接触指数分级 (配合 clips 里
    # max_angular_velocity 20→2 rad/s: 慢拧 + 找准位置 + 三指).
    screw_triad_drive = True
    # U35 (Dyn19 起): 拧盖窗掌轴对准 —— 见 __init__ 里的修正块.
    wr_align_cap = True
    wr_reach_m = 0.19                 # U35b: 卷指对握口径 (23.5 超可达域)
    hold_yaw_fix_deg = -35.0          # U35c: 瓶口从正右(-y) 转向右前方
    gate_axis_deg = 20.0              # U37: 左手须按新朝向携带才解锁右手
    # U38 (Dyn22, 已尸检): a=手势循环**回滚** (含松手段, 有害);
    # b=静姿模仿年金保留 (基线=hold 帧贴合姿, 作姿态先验).
    fin_gesture_loop = False
    w_fin_imit = 0.05
    # U39 (Dyn23 起): 右手拇指对握门.
    cap_need_thumb = True
    # U42 (Dyn24 13M 起): 拧转进度奖按三指根数线性计酬.
    # Dyn24@13M: 真螺纹下单指拨盘照样破锁 (螺旋约束吸掉径向推力, 像拨旋钮),
    # cap_triad_frac 精确 0. 而 U41 placed 要求 ≥2 指持盖运输 —— 拧转段
    # 手型必须和携带段对齐, 否则脱扣瞬间就掉盖.
    # U42b (run6 20M): =2 硬门证伪 (梯度断崖, 单指点火路断掉, 触盖趴平 2%
    # / rew_screw 恒 0 / release 0). 课程: 先 1 (单指 1/3 额, 留点火路 +
    # 每根手指真实边际) 后 2 (contact2>30% 或 release>20% 时收口).
    screw_rew_min_triad = 1
    # U36 (Dyn19 起): 完成奖须超过"泡满全程年金"的机会成本.
    # Dyn18@25M 定量: 年金 0.36/步 × 402 步 ≈ 145 ≈ Mean Rewards 143 (精确吻合),
    # 而 placed 终止回合 = 放弃剩余年金, 第250步完成只值 130 —— 泡着赢 15 分.
    # 100 分使完成@250 = 210, 翻转 65 分. 不引入任何策略可回避的计数边界.
    place_bonus = 100.0
    # U30a (Dyn15 起): 左腕参考轴向下移 —— 抓瓶下段, 让开右手拧盖区.
    left_axial_fix = 0.06


class UnscrewRefTaskEnv(UnscrewTaskEnv):
    """右手沿人手参考拧盖并放盖; 瓶身/左臂回放示范."""

    cfg: UnscrewRefTaskCfg

    def __init__(self, cfg: UnscrewRefTaskCfg, render_mode=None, **kwargs):
        # Stage A: 瓶身 = kinematic 刚体 (回放物体的正统做法): 手指接触它像接触
        # 地面, 不产生互侵去穿透冲量 (v3 实测动态瓶+写回 body_v_max 0.96 m/s).
        # Stage B: 瓶真动态 —— 左手必须真握住跟着参考走, 掉了就终止.
        if not cfg.dynamic_body:
            cfg.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True)
        super().__init__(cfg, render_mode, **kwargs)
        # ---- 双臂: 残差缩放镜像 (左侧沿用右侧的逐关节标定界) + 左侧缓冲 ----
        assert cfg.action_space == 58
        assert self.left_arm_jids is not None, "双臂任务需要左侧参考 (left_replay)"
        self.res_scale = torch.cat([self.res_scale, self.res_scale])
        N, dev = self.num_envs, self.device
        self.q_cmd_left = torch.zeros(N, 7, device=dev)
        self.q_left_prev = torch.zeros(N, 7, device=dev)
        self.left_finger_res = torch.zeros(N, 22, device=dev)
        self.left_arm_tgt = torch.zeros(N, 7, device=dev)
        self.left_arm_tgt_prev = torch.zeros(N, 7, device=dev)
        self.prev_body_d = torch.zeros(N, device=dev)     # Stage B 瓶跟踪势状态
        self.left_fin_tgt = torch.zeros(N, 22, device=dev)
        self.left_fin_tgt_prev = torch.zeros(N, 22, device=dev)
        limits = self.hand.root_physx_view.get_dof_limits().to(dev)
        self.left_arm_lower = limits[..., 0][:, self.left_arm_jids]
        self.left_arm_upper = limits[..., 1][:, self.left_arm_jids]
        self.left_dof_lower = limits[..., 0][:, self.left_hand_jids]
        self.left_dof_upper = limits[..., 1][:, self.left_hand_jids]

    # ---- 参考构建 (覆盖 v1 的常量预抓姿) -----------------------------
    def _build_cap_pregrasp(self):
        cfg, dev = self.cfg, self.device
        self.left_tip_ids = [
            self.hand.body_names.index(n.replace("right_", "left_"))
            for n in cfg.fingertip_bodies]
        self._lhold = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._body_fly = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._drop_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._body_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._fail_step = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._chold = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._cap_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        entry = clips.clip_entry(cfg.clip_name)
        rr = self.cap_ref.ref                      # 右手轨迹 (对齐规范系)
        assert rr.L == self.L, f"右手轨迹 L={rr.L} != 主 du L={self.L}"

        # 锚在实测 arm_center (v1 同款)
        bn = list(self.hand.body_names)
        org = self.scene.env_origins[0].cpu().numpy()
        ac = bn.index("arm_center")
        anchor_T = np.eye(4)
        anchor_T[:3, :3] = quat_to_R(self.hand.data.body_quat_w[0, ac].cpu().numpy())
        anchor_T[:3, 3] = self.hand.data.body_pos_w[0, ac].cpu().numpy() - org
        self._anchor_T = anchor_T

        def _fmt_ranges(idx):
            idx = sorted(int(i) for i in idx)
            out, s, p = [], idx[0], idx[0]
            for i in idx[1:]:
                if i != p + 1:
                    out.append(f"{s}-{p}" if p > s else f"{s}")
                    s = i
                p = i
            out.append(f"{s}-{p}" if p > s else f"{s}")
            return ",".join(out)

        def _ik_track(side, P, Q):
            ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
            # 推导参考的手朝向副轴是构造出来的软目标 —— 压低姿态权重专注位置.
            sols = ik.solve_traj(np.asarray(P, float), np.asarray(Q, float), w_rot=0.25)
            q = np.stack([s["q"] for s in sols])
            pe = np.array([s["pos_err"] for s in sols])
            # ⚠ 顶替判据用**位置**达标, 不用 sol["ok"] —— ok 含 rot_tol=0.05(2.9°),
            # 构造姿态几乎必然超差, 会把 136/138 帧误判不可达并全部顶替成同一帧,
            # 参考塌成常数点 (smoke5 实测: "可达 1.4%" 而位置误差中位才 0.24cm).
            good_m = np.isfinite(q).all(1) & np.isfinite(pe) & (pe < 0.02)
            bad = np.flatnonzero(~good_m)
            if len(bad):
                good = np.flatnonzero(good_m)
                assert len(good), f"{side} 整条轨迹 IK 位置全不达标"
                q[bad] = q[good[np.abs(good[None] - bad[:, None]).argmin(1)]]
            print(f"[unscrew-ref] {side} IK: 位置达标(<2cm) {good_m.mean()*100:.1f}% | "
                  f"误差中位 {np.median(pe[good_m])*100:.2f}cm "
                  f"95分位 {np.percentile(pe[good_m], 95)*100:.2f}cm | 顶替 {len(bad)} 帧"
                  + (f" @{_fmt_ranges(bad)}" if len(bad) else ""))
            return q

        # ================ v2.1: 物体锚定的手腕参考 =====================
        # ⚠ 数据事实 (2026-08-25 实测): 本 bundle 的 joints_left/right **腕部
        # 全程静止** (位移 0.1~0.27cm, 上游腕跟踪丢失用静态点填充), 手指 qpos
        # 流是活的 (变化最大 50°). 真实运动全在物体轨迹里 (瓶转 106°, 盖被
        # 送到桌面 14~19cm, meta 质量评级 good). 因此腕参考从**物体轨迹推导**
        # (ref_builder 的既有哲学), 手指参考照用活的 qpos 流.
        L = self.L
        obj_p = np.asarray(self.du.ref.track_object[:, :3], float)
        obj_q = np.asarray(self.du.ref.track_object[:, 3:7], float)
        if cfg.dynamic_body:
            # U24a: 参考瓶姿态"静置段"直立投影 (总根因修复). 原始 FoundationPose
            # 静置帧带 ~19.9° 倾角噪声 > 18.3° 翻倒极限 —— 动态瓶被 settle 冻结
            # 钉回该姿态后必在 0.7s 内自倒躺平 (pk18/19: ω=0、双手无接触自倒;
            # 之前 dyn_smoke 的"被动带转 91°存活"实为倒下的 90°, 假地板).
            # 视频里瓶静置时就是立着的 —— 投影是数据纠错不是篡改.
            # 按原始倾角混合: <22° 全投影 (静置段), >40° 全原始 (拿起后真实
            # 旋转), 中间线性过渡 —— 无帧轴, 随真实运动自然切换.
            r33 = 1 - 2 * (obj_q[:, 1] ** 2 + obj_q[:, 2] ** 2)
            tilt_deg = np.degrees(np.arccos(np.clip(r33, -1.0, 1.0)))
            yaw = np.arctan2(2 * (obj_q[:, 0] * obj_q[:, 3] + obj_q[:, 1] * obj_q[:, 2]),
                             1 - 2 * (obj_q[:, 2] ** 2 + obj_q[:, 3] ** 2))
            q_up = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                             np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)
            wmix = np.clip((tilt_deg - 22.0) / 18.0, 0.0, 1.0)[:, None]
            sgn = np.where((q_up * obj_q).sum(1, keepdims=True) < 0.0, -1.0, 1.0)
            q_mix = (1.0 - wmix) * q_up * sgn + wmix * obj_q
            obj_q = q_mix / np.linalg.norm(q_mix, axis=1, keepdims=True)
            # U35c: 携带段持瓶朝向重定向 —— 人手举瓶口朝正右 (-y 0.96) 顺手,
            # 机器人右臂做轴向对握时肘部须伸到腕外侧, 超臂长 (IK 顶替 @57-84
            # 两轮取证, reach 23.5→19 无效 = 刚性几何约束, 非距离问题).
            # 深倾斜段绕世界 z 转 yaw 把瓶口指向机器人右前方; 幅度随 wmix
            # (静置 0 不动, 携带满额), 位置不动只转朝向. 属 retarget 级修正:
            # 人的持瓶方向是相对人身体的, 换机器人身体该重定向.
            if cfg.hold_yaw_fix_deg:
                hy = np.deg2rad(cfg.hold_yaw_fix_deg) * wmix[:, 0]
                cy, sy = np.cos(hy / 2), np.sin(hy / 2)
                zw = np.zeros_like(hy)
                qy = np.stack([cy, zw, zw, sy], axis=1)
                aw2, av2 = qy[:, :1], qy[:, 1:]
                bw2, bv2 = obj_q[:, :1], obj_q[:, 1:]
                obj_q = np.concatenate(
                    [aw2 * bw2 - (av2 * bv2).sum(1, keepdims=True),
                     aw2 * bv2 + bw2 * av2 + np.cross(av2, bv2)], axis=1)
                obj_q /= np.linalg.norm(obj_q, axis=1, keepdims=True)
            n_proj = int((wmix < 1.0).sum())
            print(f"[unscrew-ref] U24a 静置段直立投影: {int((wmix == 0).sum())} 帧全投影 "
                  f"/ {n_proj} 帧受影响 (原始首帧倾角 {tilt_deg[0]:.1f}°)")
        axis = F.rot_apply(obj_q, np.tile([0.0, 0.0, 1.0], (L, 1)))   # 瓶轴(=螺轴)

        src_len = 104
        s2e = (L - 1) / (src_len - 1)
        with np.load(entry["npz"], allow_pickle=True) as d:
            body_raw = d["obj_pose"][:, :7].astype(float)
            cap_raw = d["obj_pose_all"][1, :, :3].astype(float)
        raw_body0 = body_raw[0, :3]
        shift = obj_p[0] - raw_body0
        # 盖脱离帧从数据测: 盖轨迹与"瓶推导装配位"的偏离首超 4cm 处.
        # ⚠ 不能用 phase_right 窗末 (源72): 人在横平段就拧开了, 之后盖在右手里
        # 而瓶已回正 —— 用相位窗会在拼接处产生 24cm 跳变 (实测).
        axis_raw = F.rot_apply(body_raw[:, 3:7], np.tile([0.0, 0.0, 1.0], (src_len, 1)))
        off = float(self.screw_spec.closed_offset_m)
        sep_d = np.linalg.norm(cap_raw - (body_raw[:, :3] + off * axis_raw), axis=1)
        sep_src = int(np.argmax(sep_d > 0.04)) if (sep_d > 0.04).any() else src_len - 1
        engage_end = max(1, int(round(sep_src * s2e)))
        print(f"[unscrew-ref] 盖脱离帧: 源 {sep_src} (对齐 {engage_end}) | "
              f"脱离前偏差中位 {np.median(sep_d[:max(sep_src,1)])*100:.1f}cm")
        # U38: 拧转手势循环窗 [接触起, 盖脱离) —— 拇指主导的真实拧转动作段
        self._fin_loop_lo = int(round(cfg.contact_start_src * s2e))
        self._fin_loop_hi = max(engage_end, self._fin_loop_lo + 2)
        self._hold_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._t_fin = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._triad_fj = [k for k, n in enumerate(self.hand_joint_names)
                          if ("thumb" in n or "index" in n or "middle" in n)]
        ti = np.linspace(0.0, src_len - 1, L)
        cap_free = np.stack([np.interp(ti, np.arange(src_len), cap_raw[:, k] + shift[k])
                             for k in range(3)], axis=1)
        # 盖中心序列: 合拢期由瓶推导 (obj_pose_all 的盖轨迹在合拢期与瓶冗余且更噪),
        # 释放期用盖自己的重建轨迹 (被右手送到桌面的那段真实运动).
        cap_c = obj_p + off * axis
        cap_c[engage_end:] = cap_free[engage_end:]
        cap_c = F.smooth_channels(cap_c, sigma=1.5)   # 拼接处轻度平滑

        # ---- 腕四元数流: 数据的活姿态 (recon_world = 规范系方向), 重采样到 L ----
        def _resample_quats(qraw):
            q = np.asarray(qraw, float).copy()
            for i in range(1, len(q)):                 # 符号连续化
                if np.dot(q[i], q[i - 1]) < 0:
                    q[i] = -q[i]
            i0 = np.clip(np.floor(ti).astype(int), 0, src_len - 1)
            i1 = np.clip(i0 + 1, 0, src_len - 1)
            w = (ti - i0)[:, None]
            out = (1 - w) * q[i0] + w * q[i1]
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
            # 重建姿态逐帧噪声直接进 IK 会让重放臂抽动 (用户验收指出"左手乱动").
            # 位置流在 load_replay 里有高斯平滑, 姿态流这里补上 (同 sigma 量级).
            return F.smooth_quats(out, sigma=2.0)

        qr_raw = np.load(os.path.join(os.path.dirname(entry["npz"]),
                                      "ref_qpos_right.npz"), allow_pickle=True)
        wr_Q = _resample_quats(qr_raw["wrist_quat_wxyz"])

        # ---- 右腕位置: 世界系恒定偏移锚定在证据帧 src44 ----
        # (人手不随瓶共转 —— 数据 quat 只转 27° 而瓶转 106°; 瓶是转进手里的)
        a_grasp = int(round(cfg.grasp_anchor_src * s2e))
        a_touch = int(round(cfg.contact_start_src * s2e))
        wr_static = np.asarray(rr.track_wrist[0, :3], float)   # 静态腕点(=提取姿势)
        off_w = wr_static - cap_c[a_grasp]
        wr_P = cap_c + off_w[None]
        wr_P[:a_touch] = wr_static                             # 接触前: 人手等待位
        for t in range(a_touch, a_grasp):                      # 线性混入抓取锚定
            u = (t - a_touch) / max(a_grasp - a_touch, 1)
            wr_P[t] = (1 - u) * wr_static + u * (cap_c[t] + off_w)
        wr_P[:, 2] = np.maximum(wr_P[:, 2], cfg.table_top_z + 0.02)

        # ---- U35: 拧盖窗掌轴对准 ----
        # 把 hand_z(腕→指尖) 按最短弧旋到 -螺轴 (指尖指向盖 = 对握姿).
        # 接近段 (a_touch→a_grasp) 权重渐进; 窗内全额; 盖脱离后冻结修正量
        # (送放段整体常量旋转, 流形连续, 不与放置动作打架).
        if cfg.wr_align_cap:
            ez = np.tile([0.0, 0.0, 1.0], (L, 1))
            hz = F.rot_apply(wr_Q, ez)
            tgt = -axis / np.linalg.norm(axis, axis=1, keepdims=True)
            uu = np.zeros(L)
            uu[a_touch:a_grasp] = np.linspace(
                0.0, 1.0, max(a_grasp - a_touch, 1), endpoint=False)
            uu[a_grasp:] = 1.0
            cr = np.cross(hz, tgt)
            dt_ = (hz * tgt).sum(1)
            ang_f = np.arctan2(np.linalg.norm(cr, axis=1), dt_)
            ax_n = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-9)
            ang_f[engage_end:] = ang_f[engage_end]
            ax_n[engage_end:] = ax_n[engage_end]
            half = 0.5 * ang_f * uu
            aw = np.cos(half)[:, None]
            av = np.sin(half)[:, None] * ax_n
            bw, bv = wr_Q[:, :1], wr_Q[:, 1:]
            wr_Q = np.concatenate(
                [aw * bw - (av * bv).sum(1, keepdims=True),
                 aw * bv + bw * av + np.cross(av, bv)], axis=1)
            wr_Q /= np.linalg.norm(wr_Q, axis=1, keepdims=True)
            chk = F.rot_apply(wr_Q, ez)
            res_d = np.degrees(np.arccos(np.clip((chk * tgt).sum(1), -1, 1)))
            # U35b (Dyn19@5M 尸检): 姿态转正但腕位仍锚在提取静态腕点上 ——
            # 该点离盖 17.3cm (上游腕跟踪丢失, 与姿态流同罪). 旧姿态下指尖
            # 歪打正着探到 9-11cm, 转正后被甩到 14cm, 解锁门(8cm)结构死锁
            # (gate 12.9%→5.5% 递减取证). 对握构型必须整体成立:
            # 腕 = 盖心 + reach·螺轴, 指尖曲回落在盖顶 (基类顶抓的同款几何,
            # 只是轴从"世界竖直"换成"携带后的螺轴").
            cap_h = float(F.load_obj_verts(self.cap_entry["mesh"])[:, 2].max())
            reach = cfg.wr_reach_m or (cap_h + cfg.cap_clearance_m
                                       + cfg.hand_drop_m)
            anchor = cap_c + reach * (axis / np.linalg.norm(
                axis, axis=1, keepdims=True))
            for t in range(a_touch, a_grasp):
                wr_P[t] = (1 - uu[t]) * wr_static + uu[t] * anchor[t]
            wr_P[a_grasp:engage_end] = anchor[a_grasp:engage_end]
            # 脱离后: 沿脱离帧的螺轴方向冻结偏移, 跟随盖的送放轨迹 (连续)
            wr_P[engage_end:] = (cap_c[engage_end:]
                                 + reach * (axis[engage_end]
                                            / np.linalg.norm(axis[engage_end])))
            wr_P[:, 2] = np.maximum(wr_P[:, 2], cfg.table_top_z + 0.02)
            print(f"[unscrew-ref] U35 掌轴对准: 窗内残差 "
                  f"{res_d[a_grasp:engage_end].mean():.1f}° "
                  f"(修正前该窗均 {np.degrees(ang_f[a_grasp:engage_end].mean()):.1f}°) "
                  f"| U35b 腕锚定 reach={reach*100:.1f}cm "
                  f"(旧 off_w 模长 {np.linalg.norm(off_w)*100:.1f}cm 弃用)")

        q_r = _ik_track("right", wr_P, wr_Q)
        self.q_ref = to(q_r)
        self.ref_wrist_pos = to(wr_P)
        self.ref_wrist_quat = to(wr_Q)
        perm = [rr.finger_names.index(n) for n in self.hand_joint_names]
        self.ref_finger = to(rr.human_finger[:, perm])   # 手指流是活的, 照用

        # ---- 瓶身: 回放主 du 的重建轨迹 ----
        # ⚠ 用本函数顶部的 obj_p/obj_q (dynamic 下含 U24a 静置段直立投影),
        # 不要再从 track_object 原始数据重建 —— 曾把投影冲掉 (pk19 归因).
        self.ref_obj_pos = to(obj_p)
        self.ref_obj_quat = to(obj_q)
        self.obj_init_pos = self.ref_obj_pos[0].clone()
        self.obj_init_quat = self.ref_obj_quat[0].clone()
        # 帽初始: 合拢在轨迹首帧的瓶口上 (覆盖静态摆放算出的旧值)
        body0 = np.concatenate([self.obj_init_pos.cpu().numpy(),
                                self.obj_init_quat.cpu().numpy()])
        self.cap_init_pose_np = assembled_cap_pose(body0, self.screw_spec)

        # ---- 左臂: 参考数据加载 (执行层 = 参考前馈 + 策略残差, 见 _pre_physics_step) ----
        self.left_arm_jids = self.left_hand_jids = None
        if cfg.left_replay:
            du_l = load_replay(
                entry["npz"], entry["mesh"], usd_path=entry["usd"],
                clip_id=f"{cfg.clip_name}:left", hand="left",
                table_height=cfg.table_top_z, obj_gap=0.002,
                target_hz=cfg.target_hz, semantics=entry["semantics"],
                initial_pose_mode="preserve")
            rl_ = du_l.ref
            assert rl_.L == self.L
            # 左腕位置 = 瓶位姿 ∘ 固定抓握偏移 (静态腕点=首帧真实握位, 握着瓶
            # 就随瓶动; grasp_prompt 证据: 接触区在瓶下段 高度7~26%, 与该偏移一致).
            # 左腕姿态 = 数据的活四元数流 (64° 真实旋转), 不再用瓶合成.
            wl0 = np.asarray(rl_.track_wrist[0], float)           # (7,) 冻结握位
            q0c = obj_q[0] * np.array([1.0, -1.0, -1.0, -1.0])    # 瓶首帧四元数共轭
            r_rel = F.rot_apply(q0c[None], (wl0[:3] - obj_p[0])[None])[0]
            # 站位校准: 径向向瓶轴收进 left_standoff_fix. ⚠ 收进量随抓取相位
            # **渐进** (与合拢斜坡同窗): 瓶直立时用原始站位 (历史无碰撞),
            # 拿起过程中才收 —— 一步到位的收进会让掌部在 reset 瞬移时插进
            # 直立瓶, 去穿透把瓶拍翻 (dyn_smoke5 判读).
            r_ax = np.array([0.0, 0.0, r_rel[2]])
            r_rad = r_rel - r_ax
            rn = float(np.linalg.norm(r_rad))
            b0s = int(round(25 * s2e)); b1s = int(round(40 * s2e))
            blend_s = np.clip((np.arange(L) - b0s) / max(b1s - b0s, 1), 0.0, 1.0)
            shrink_seq = 1.0 - (blend_s * cfg.left_standoff_fix) / max(rn, 1e-6)
            # U30a: 轴向随同一斜坡下移 (直立段保持原站位, 拿起过程中降下去)
            ax_seq = r_rel[2] - blend_s * cfg.left_axial_fix
            rr_seq = np.zeros((L, 3))
            rr_seq[:, 2] = ax_seq
            rr_seq += r_rad[None] * shrink_seq[:, None]
            wl_P = obj_p + F.rot_apply(obj_q, rr_seq)
            ql_raw = np.load(os.path.join(os.path.dirname(entry["npz"]),
                                          "ref_qpos_left.npz"), allow_pickle=True)
            wl_Q = _resample_quats(ql_raw["wrist_quat_wxyz"])
            q_l = _ik_track("left", wl_P, wl_Q)
            # 重放臂的关节轨迹整体再平滑一遍: 抹掉 IK 顶替段的平台-跳变
            # (左臂 37 帧顶替) 和残余高频 —— 左臂不进观测, 只影响视觉.
            q_l = F.smooth_channels(q_l, sigma=2.0)
            self.q_left = to(q_l)
            jn = list(self.hand.joint_names)
            self.left_arm_jids = [jn.index(f"L_arm_j{i}") for i in range(1, 8)]
            left_names = [n for n in jn if n.startswith("left_")]
            self.left_hand_jids = [jn.index(n) for n in left_names]
            perm_l = [rl_.finger_names.index(n) for n in left_names]
            # 手指流平滑: 人手微调 + retarget 噪声原样回放会看起来"手指乱动"
            lf_np = F.smooth_channels(
                np.asarray(rl_.human_finger[:, perm_l], float), sigma=2.0)
            # 卷曲校准: retarget 低估屈曲 ~46°, 屈曲类关节加偏置并按限位钳制
            fe_mask = np.array([1.0 if ("_FE" in n or n.endswith(("_PIP", "_DIP", "_IP")))
                                else 0.0 for n in left_names])
            lim = self.hand.root_physx_view.get_dof_limits()[0].cpu().numpy()
            # 未偏置版 = reset 时的**状态** (张开, 不与瓶穿插);
            # 偏置版 = 目标参考 (PD 在前几步温柔合拢到瓶上).
            # ⚠ 动态瓶模式下不能用偏置姿态直接写关节状态: 指节与瓶重叠的
            #   去穿透会把 0.53kg 的瓶瞬间抛起 55cm (pk15 实测 z=1.4m).
            lf_open = np.clip(lf_np, lim[self.left_hand_jids, 0],
                              lim[self.left_hand_jids, 1])
            lf_closed = np.clip(lf_np + cfg.left_close_bias * fe_mask,
                                lim[self.left_hand_jids, 0],
                                lim[self.left_hand_jids, 1])
            # 合拢斜坡 (pk16 判读): 一次性合到偏置会在回合开头把**直立**的瓶
            # 拍翻/打飞 (偏置几何为横平瓶校准). 参考逐帧从张开混到偏置合拢,
            # 混合窗对齐"人拿起瓶"的相位 (源 25→40), 慢合拢不翻瓶且合示范语义.
            b0 = int(round(25 * s2e)); b1 = int(round(40 * s2e))
            blend = np.clip((np.arange(L) - b0) / max(b1 - b0, 1), 0.0, 1.0)[:, None]
            self.left_finger_open = to(lf_open)
            self.left_finger = to((1 - blend) * lf_open + blend * lf_closed)

        # ---- 门控时钟与时间轴 ----
        self.hold_frame = int(round(cfg.hold_src_frame * s2e))
        self.gate_frame = int(round(cfg.grip_gate_src_frame * s2e))   # U22
        self.gate_unlocked = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._gate_step = torch.zeros_like(self.gate_unlocked)
        self.ref_clock = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self._clock_prev = torch.zeros_like(self.ref_clock)
        self.ep_total = (cfg.settle_steps + (self.L - 1)
                         + cfg.twist_budget_steps + cfg.hold_steps)

        # ---- 盖参考终点 (原始盖轨迹 + 规范系平移, 复用上面的 shift) ----
        self.cap_goal = to(cap_raw[cfg.place_src_frame] + shift)
        # ---- 放置/释放状态 ----
        self.placed_latch = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self._placed_step = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        self.prev_place_d = torch.zeros(self.num_envs, device=dev)
        self._cap_carry_steps = torch.zeros(self.num_envs, device=dev)   # U41
        self._cap_fall_peak = torch.zeros(self.num_envs, device=dev)     # U41
        self.wrist_err_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=dev)

        tilt = torch.rad2deg(torch.arccos(
            (1 - 2 * (self.ref_obj_quat[:, 1] ** 2 + self.ref_obj_quat[:, 2] ** 2)
             ).clamp(-1, 1))).max()
        print(f"[unscrew-ref] 瓶轨迹回放 L={self.L} 起点 "
              f"{[round(float(v), 3) for v in self.obj_init_pos]} 最大倾角 {tilt:.0f}° | "
              f"hold帧 {self.hold_frame} (源 {cfg.hold_src_frame}) | "
              f"盖目标 {[round(float(v), 3) for v in self.cap_goal]} | "
              f"ep_total {self.ep_total}")

    # ---- 场景: 左指尖-瓶身接触传感器 (握持质量的数据源) ----------------
    def _setup_scene(self):
        super()._setup_scene()
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        self._left_bottle_sensors = []
        for i, name in enumerate(self.cfg.fingertip_bodies):
            lname = name.replace("right_", "left_")
            scfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{lname}",
                history_length=1,
                filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            )
            s = ContactSensor(scfg)
            self._left_bottle_sensors.append(s)
            self.scene.sensors[f"left_bottle_contact_{i}"] = s

    def _left_bottle_contacts(self) -> torch.Tensor:
        """(N,5) 左指尖-瓶身接触力是否超阈."""
        f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in self._left_bottle_sensors], dim=1)
        return (f.norm(dim=-1) > self.cfg.contact_force_thresh).float()

    # ---- 门控参考时钟 -------------------------------------------------
    def _ref_t(self) -> torch.Tensor:
        return self.ref_clock

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._clock_prev.copy_(self.ref_clock)
        adv = self.episode_length_buf >= self.cfg.settle_steps
        stay = (self.ref_clock == self.hold_frame) & ~self.released_latch
        self.ref_clock = (self.ref_clock + (adv & ~stay).long()).clamp(max=self.L - 1)
        if self.cfg.grip_gate:
            # U23: 解锁前右手 29 维残差清零 —— 纯跟参考走, 不给乱动的自由度
            actions = actions.clone()
            actions[~self.gate_unlocked, :29] = 0.0
        super()._pre_physics_step(actions)      # 右侧 29 维走基类原管线
        # ---- U38: 冻结窗内右指参考基线换成循环播放的拧转手势 ----
        # 基类 finger_tgt = ref_finger[冻结帧] + 累积残差; 这里只换基线不动
        # 残差: tgt += ref_finger[循环帧] - ref_finger[冻结帧]. 手势 (拇指主导
        # 捻动) 来自数据, 位置校正归 RL —— 用户裁定的分工.
        if self.cfg.fin_gesture_loop:
            stay2 = (self.ref_clock == self.hold_frame) & ~self.released_latch
            self._hold_ctr = torch.where(stay2, self._hold_ctr + 1,
                                         torch.zeros_like(self._hold_ctr))
            span = self._fin_loop_hi - self._fin_loop_lo
            t_now = self.ref_clock
            self._t_fin = torch.where(
                stay2, self._fin_loop_lo + (self._hold_ctr % span), t_now)
            swap = self.ref_finger[self._t_fin] - self.ref_finger[t_now]
            self.finger_tgt = (self.finger_tgt + swap).clamp(
                self.dof_lower, self.dof_upper)
        # ---- 左侧 29 维: 与右侧同构的残差管线 (臂前馈+累积+偏差钳, 指 rate/total) ----
        cfg = self.cfg
        t = self.ref_clock
        a_left = self.actions_buf[:, 29:] * self.res_scale[29:]
        la, lf = a_left[:, :7], a_left[:, 7:]
        q_base = self.q_left[t]
        self.q_cmd_left = self.q_cmd_left + (q_base - self.q_left_prev) + la
        dev = (self.q_cmd_left - q_base).clamp(-cfg.left_arm_dev_max,
                                               cfg.left_arm_dev_max)
        self.q_cmd_left = (q_base + dev).clamp(self.left_arm_lower,
                                               self.left_arm_upper)
        self.q_left_prev = q_base.clone()
        self.left_finger_res = (self.left_finger_res + lf).clamp(
            -cfg.finger_total_max, cfg.finger_total_max)
        self.left_arm_tgt_prev.copy_(self.left_arm_tgt)
        self.left_fin_tgt_prev.copy_(self.left_fin_tgt)
        self.left_arm_tgt.copy_(self.q_cmd_left)
        self.left_fin_tgt = (self.left_finger[t] + self.left_finger_res).clamp(
            self.left_dof_lower, self.left_dof_upper)

    def _apply_action(self) -> None:
        if not self.cfg.dynamic_body:
            # Stage A: 瓶身运动学回放按**物理子步**插值: 20Hz 整步跳变会让接触中
            # 的手指/帽每 50ms 吃一次去穿透冲量 (全场抖动主源). 位置线性、四元数
            # nlerp, 写入轨迹一致速度, 接触看到的是连续运动的表面.
            w = min((self._substep + 1) / self.cfg.decimation, 1.0)
            t0, t1 = self._clock_prev, self.ref_clock
            p0, p1 = self.ref_obj_pos[t0], self.ref_obj_pos[t1]
            q0, q1 = self.ref_obj_quat[t0], self.ref_obj_quat[t1]
            sign = torch.where((q0 * q1).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
            q0 = q0 * sign
            qi = q0 + (q1 - q0) * w
            qi = qi / qi.norm(dim=1, keepdim=True).clamp(min=1e-8)
            origins = self.scene.env_origins
            self.object.write_root_pose_to_sim(
                torch.cat([p0 + (p1 - p0) * w + origins, qi], dim=1))
            dt = float(self.cfg.sim.dt) * self.cfg.decimation
            lin = (p1 - p0) / dt
            ang = 2.0 * quat_mul(q1, quat_conjugate(q0))[:, 1:4] / dt  # 小角近似
            self.object.write_root_velocity_to_sim(torch.cat([lin, ang], dim=1))
        else:
            w = min((self._substep + 1) / self.cfg.decimation, 1.0)  # 左目标插值仍要
        # 左侧目标子步插值 (与基类右侧 substep_interp 同构)
        la = self.left_arm_tgt_prev + (self.left_arm_tgt - self.left_arm_tgt_prev) * w
        lf = self.left_fin_tgt_prev + (self.left_fin_tgt - self.left_fin_tgt_prev) * w
        self.hand.set_joint_position_target(la, joint_ids=self.left_arm_jids)
        self.hand.set_joint_position_target(lf, joint_ids=self.left_hand_jids)
        super()._apply_action()                 # 右侧关节目标 + 螺旋约束 (读新鲜瓶位)

    def apply_screw_constraint(self, extra_cap_torque_local=None, *,
                               integrate_angle: bool = True, drive_mask=None,
                               omega_damping=None):
        if drive_mask is None:
            drive_mask = self._cap_contacts().sum(dim=1) >= 1
        from tasks.recon_kailang.bottle_reconstruction.env import (
            BottleReconstructionEnv,
        )
        BottleReconstructionEnv.apply_screw_constraint(
            self, extra_cap_torque_local, integrate_angle=integrate_angle,
            drive_mask=drive_mask, omega_damping=self.cfg.screw_omega_damping)

    # ---- 终止: 放置成功 / 盖掉落 / 基础发散 ---------------------------
    # ---- 观测: 基类 415 + 任务 10 + 左侧 70 --------------------------
    def _get_observations(self) -> dict:
        out = super()._get_observations()        # 基类 415 + v1 任务追加 10 (defer 校验)
        t = self.ref_clock
        larm_q = self.hand.data.joint_pos[:, self.left_arm_jids]
        larm_qd = self.hand.data.joint_vel[:, self.left_arm_jids]
        lfin_q = self.hand.data.joint_pos[:, self.left_hand_jids]
        vs = self.cfg.obs_vel_scale
        extra = torch.cat([
            larm_q,                                            # 7
            larm_qd * vs,                                      # 7
            lfin_q,                                            # 22
            self.q_left[t] - larm_q,                           # 7 左臂参考误差
            self.left_finger_res / self.cfg.finger_total_max,  # 22 左指积分器状态
            self._left_bottle_contacts(),                      # 5 左指尖-瓶接触
        ], dim=1).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        obs = torch.cat([out["policy"], extra], dim=1)
        self._check_obs_dim(obs)
        out["policy"] = obs
        return out

    def _check_obs_dim(self, obs: torch.Tensor) -> None:
        # 三段拼接 (基类 415 -> +10 -> +70): 只在最终宽度处校验
        if obs.shape[1] != self.cfg.observation_space:
            if getattr(self, "_defer_obs_check", False) or obs.shape[1] in (415, 425):
                return
        super()._check_obs_dim(obs)

    def _body_track_err(self, obj_pos: torch.Tensor) -> torch.Tensor:
        """瓶-参考综合误差 = 位置 + λ·姿态测地角 (位置-only 会被'不抬不转'钻空)."""
        t = self.ref_clock
        d = (obj_pos - self.ref_obj_pos[t]).norm(dim=1)
        qr = self.ref_obj_quat[t]
        qo = self.object.data.root_quat_w
        ang = 2.0 * torch.arccos((qo * qr).sum(dim=1).abs().clamp(max=1.0))
        return d + self.cfg.lam_body_rot * ang

    def _get_dones(self):
        # 跳过 v1 的 (knock | release) 终止 —— 直接走场景层 dones
        from tasks.recon_kailang.bottle_reconstruction.env import (
            BottleReconstructionEnv,
        )
        terminated, truncated = BottleReconstructionEnv._get_dones(self)
        # 任务级释放规则: 拧满 = 螺纹完全脱开 = 盖自由. 场景层的 detach 要求
        # "顶满时仍有正角速度"(最后一推), 而 drive_mask 下松手后 ω=0 —— 确定性
        # 策略拧到 720° 停手就永远卡在 engaged (last.pth 回放实测: 720° 保持
        # 160 步不释放直到超时). 物理上满 2 圈螺纹行程已走完, 直接判脱开.
        full = (self.screw_engaged & self.screw_has_depth
                & (self.screw_angle >= self._max_angle - 1e-4))
        self.screw_engaged[full] = False
        released = self.screw_has_depth & ~self.screw_engaged
        self._released_step = released & ~self.released_latch
        self.released_latch |= released

        origins = self.scene.env_origins
        cap_pos = self.cap.data.root_pos_w - origins
        place_d = (cap_pos - self.cap_goal).norm(dim=1)
        on_table = cap_pos[:, 2] < (self.cfg.table_top_z + self.cfg.place_z_tol)
        at_rest = self.cap.data.root_lin_vel_w.norm(dim=1) < self.cfg.place_speed
        # U41 记账: 释放后右手持盖 (≥2 指尖触盖) 的累计步数 + 盖下落峰速.
        # 成功必须"拿着放", 自由落体/甩落一票否决.
        held = self._cap_contacts().sum(dim=1) >= 2
        self._cap_carry_steps += (self.released_latch & held).float()
        fall = (-self.cap.data.root_lin_vel_w[:, 2]).clamp(min=0.0)
        self._cap_fall_peak = torch.maximum(
            self._cap_fall_peak,
            torch.where(self.released_latch, fall, torch.zeros_like(fall)))
        placed = (self.released_latch & (place_d < self.cfg.place_tol)
                  & on_table & at_rest
                  & (self._cap_carry_steps >= self.cfg.place_carry_steps)
                  & (self._cap_fall_peak < self.cfg.place_max_fall))
        self._placed_step = placed & ~self.placed_latch
        self.placed_latch |= placed
        cap_fell = cap_pos[:, 2] < (self.cfg.table_top_z - 0.05)
        self._knock = cap_fell                     # 复用 v1 的诊断槽位
        if self.cfg.dynamic_body:
            # Stage B: 掉瓶/没跟上参考 = 失败终止 (握持的因果压力来源)
            obj_pos = self.object.data.root_pos_w - origins
            body_d = self._body_track_err(obj_pos)
            settled = self.episode_length_buf >= self.cfg.settle_steps
            self._body_lost = (body_d > self.cfg.term_body_lost) & settled
            terminated = terminated | self._body_lost
            if self.cfg.grip_gate:
                # U23 解锁: 右指尖离**真实盖** < 阈值. 参考只把手带到空中预抓
                # 位, 真盖须被左手转运到位才够得着 —— 物态门, 无帧轴.
                tips_d = ((self.hand.data.body_pos_w[:, self.tip_ids]
                           - origins[:, None])
                          - cap_pos[:, None]).norm(dim=-1).min(dim=1).values
                # U23b: 瓶已横转且离桌 (物态副条件, 挡"顺路经过"与躺倒)
                oq = self.object.data.root_quat_w
                r33 = 1 - 2 * (oq[:, 1] ** 2 + oq[:, 2] ** 2)
                carried = ((r33 < self.cfg.gate_tilt_cos)
                           & (obj_pos[:, 2] > self.cfg.table_top_z
                              + self.cfg.gate_body_lift))
                # U37: 瓶轴须对齐参考轴 (pk24: U35c 重定向后左手 yaw 差 31.4°
                # 不迁移 —— r_body 的 yaw 梯度在 16cm 误差处近乎平坦, 重转
                # 持握的瓶有掉瓶风险, 不值得. 门条件是教会 U23b 倾斜+抬升的
                # 同一机制: 不对齐 = 右侧收入全锁, 值函数替梯度说话)
                if self.cfg.gate_axis_deg > 0:
                    _ox, _oy2, _oz2, _ow2 = oq[:, 1], oq[:, 2], oq[:, 3], oq[:, 0]
                    _axl = torch.stack([2 * (_ox * _oz2 + _ow2 * _oy2),
                                        2 * (_oy2 * _oz2 - _ow2 * _ox),
                                        r33], dim=1)
                    _t = self.ref_clock.clamp(max=self.L - 1)
                    _rq = self.ref_obj_quat[_t]
                    _rx, _ry, _rz, _rw = _rq[:, 1], _rq[:, 2], _rq[:, 3], _rq[:, 0]
                    _axr = torch.stack([2 * (_rx * _rz + _rw * _ry),
                                        2 * (_ry * _rz - _rw * _rx),
                                        1 - 2 * (_rx ** 2 + _ry ** 2)], dim=1)
                    _gap = torch.rad2deg(torch.arccos(
                        (_axl * _axr).sum(1).clamp(-1.0, 1.0)))
                    carried = carried & (_gap < self.cfg.gate_axis_deg)
                    self.diag.add("body_axis_gap_deg", _gap, mask=settled)
                self._gate_step = (~self.gate_unlocked & settled & carried
                                   & (tips_d < self.cfg.gate_cap_dist))
                self.gate_unlocked |= self._gate_step
            if self.cfg.term_body_speed > 0:
                # U20⑤: 碰飞即判负 —— 不留"瓶已丢但误差未超限"的乱动窗口
                self._body_fly = (
                    (self.object.data.root_lin_vel_w.norm(dim=1)
                     > self.cfg.term_body_speed) & settled)
                terminated = terminated | self._body_fly
            if self.cfg.term_body_drop_steps > 0:
                # U24b: 参考瓶在空中而真瓶滞留桌面 = 掉瓶失败 (握紧的因果必需)
                ref_z = self.ref_obj_pos[self.ref_clock][:, 2]
                zthr = self.cfg.table_top_z + self.cfg.gate_body_lift
                drop_now = settled & (ref_z > zthr) & (obj_pos[:, 2] < zthr)
                self._drop_ctr = torch.where(
                    drop_now, self._drop_ctr + 1, torch.zeros_like(self._drop_ctr))
                self._body_dropped = (self._drop_ctr
                                      >= self.cfg.term_body_drop_steps)
                terminated = terminated | self._body_dropped
        # 腕端滞留判定 (替代已停用的关节空间卡死判定, 见 cfg 注释)
        werr = (self.wrist_pos_w - origins
                - self.ref_wrist_pos[self.ref_clock]).norm(dim=1)
        off = werr > self.cfg.term_wrist_err
        self.wrist_err_ctr = torch.where(
            off, self.wrist_err_ctr + 1, torch.zeros_like(self.wrist_err_ctr))
        wrist_lost = self.wrist_err_ctr >= self.cfg.term_wrist_steps
        # U25: 本步以失败方式终止的 env (成功 placed 除外), 供奖励一次性重罚
        self._fail_step = (terminated | cap_fell | wrist_lost) & ~self.placed_latch
        return terminated | self.placed_latch | cap_fell | wrist_lost, truncated

    # ---- 奖励 ---------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        origins = self.scene.env_origins
        active = self.episode_length_buf >= cfg.settle_steps
        t = self.ref_clock

        cap_pos = self.cap.data.root_pos_w - origins
        tips = self.tip_pos_w - origins[:, None]
        d_cap = (tips - cap_pos[:, None]).norm(dim=-1).min(dim=1).values
        capc = self._cap_contacts()
        n_cap = capc.sum(dim=1)
        wrist_err = (self.wrist_pos_w - origins - self.ref_wrist_pos[t]).norm(dim=1)
        place_d = (cap_pos - self.cap_goal).norm(dim=1)

        obj_pos = self.object.data.root_pos_w - origins
        body_d = self._body_track_err(obj_pos)
        newly = active & (self.episode_length_buf == cfg.settle_steps)
        if newly.any():
            self.prev_tip_cap[newly] = d_cap[newly]
            self.prev_screw[newly] = self.screw_angle[newly]
            self.prev_place_d[newly] = place_d[newly]
            self.prev_body_d[newly] = body_d[newly]

        if cfg.grip_gate and self._gate_step.any():
            self.prev_tip_cap[self._gate_step] = d_cap[self._gate_step]  # U22 防跳变
        r_app = cfg.k_approach * (self.prev_tip_cap - d_cap).clamp(-0.02, 0.02)
        self.prev_tip_cap.copy_(d_cap)
        dtheta = self.screw_angle - self.prev_screw
        self.prev_screw.copy_(self.screw_angle)
        r_screw = cfg.k_screw * dtheta
        # U39: 拇指对握门 (U26 左手同款) —— 拇指不触盖, contact/chold 不计酬
        thumb_cap = (capc[:, 0] > 0) if cfg.cap_need_thumb else torch.ones_like(
            capc[:, 0], dtype=torch.bool)
        r_contact = cfg.w_cap_contact * ((n_cap >= 2) & thumb_cap).float()
        r_ctouch = cfg.w_cap_touch * (n_cap >= 1).float()   # U28: 碰到盖的台阶
        # U32b: 精捏手型 —— 拇/食/中 (capc 列序 0/1/2) 每碰一根 1/3 额;
        # 环/小扒盖不计 => 正确三指严格多挣, 且每根都有独立边际 (可爬)
        n_triad = capc[:, :3].sum(dim=1)
        r_ctriad = cfg.w_cap_triad * (n_triad / 3.0)
        # U42: 单指拨盘不计酬 —— 拧转进度奖须 ≥screw_rew_min_triad 根
        # (拇/食/中) 触盖: 2 指 2/3 额, 3 指全额, 单/零指为 0.
        if cfg.screw_rew_min_triad > 0:
            _tri = n_triad.float()
            r_screw = r_screw * torch.where(
                _tri >= cfg.screw_rew_min_triad,
                (_tri / 3.0).clamp(max=1.0), torch.zeros_like(_tri))
        # U27: 右手持续触盖爬坡 (镜像 U20③; U39 起须含拇指)
        self._chold = torch.where((n_cap >= 2) & thumb_cap, self._chold + 1,
                                  torch.zeros_like(self._chold))
        r_chold = cfg.w_cap_hold * (self._chold.float()
                                    / cfg.hold_horizon).clamp(max=1.0)
        lcon = self._left_bottle_contacts()
        n_left = lcon.sum(dim=1)
        # U26: 拇指对握门 —— fingertip_bodies[0] = 拇指; 开启后 grip3/hold
        # 要求拇指在接触集内 (对握抗扭, 四指爪抓不算握)
        thumb_c = (lcon[:, 0] > 0) if cfg.grip_need_thumb else torch.ones_like(
            lcon[:, 0], dtype=torch.bool)
        grip3 = (n_left >= 3) & thumb_c
        r_lgrip = cfg.w_left_grip * grip3.float()           # 左手真实握瓶
        # U32a: 全掌包握 —— grip3 之上第 4/5 指各 1/2 额 (握杯是五指都到位;
        # 三指爪抓抗不住携带+拧盖反扭). 纯增量: 3 指原收入不动
        r_lfull = cfg.w_left_full * grip3.float() * (
            (n_left.clamp(max=5) - 3.0).clamp(min=0.0) / 2.0)
        r_ltouch = cfg.w_left_touch * (n_left >= 1).float()  # U21: 碰到 1 指的台阶
        # U19: 贴近年金 (密集接触梯度; 权重 0 时恒 0, Stage A 不受影响)
        ltips = self.hand.data.body_pos_w[:, self.left_tip_ids] - origins[:, None]
        d_ltip = (ltips - obj_pos[:, None]).norm(dim=-1).min(dim=1).values
        r_lapp = cfg.w_left_app * torch.exp(
            -(d_ltip - cfg.r0_left).clamp(min=0) / cfg.sigma_app)
        r_capp = cfg.w_cap_app * torch.exp(
            -(d_cap - cfg.r0_cap).clamp(min=0) / cfg.sigma_app)
        if cfg.grip_gate:
            # U22: 解锁前右手全部任务奖励清零 (拧躺瓶的盖零收益);
            # 右腕由 r_imit 守在停靠参考位, 不另设罚.
            rgate = self.gate_unlocked.float()
            r_app = r_app * rgate
            r_screw = r_screw * rgate
            r_contact = r_contact * rgate
            r_capp = r_capp * rgate
            r_chold = r_chold * rgate
            r_ctouch = r_ctouch * rgate
            r_ctriad = r_ctriad * rgate
        # U29/U29b: 拧转窗年金衰减 —— 触盖或贴近 (<cap_dwell_near) 的步数计
        # dwell, 盖侧年金线性归零; 释放后 dwell 停走, 衰减因子冻结
        # (早释放 = 携带段年金更厚). r_app/r_screw 是势差分/进度型, 不衰减.
        if cfg.cap_annuity_decay > 0:
            engaged = (n_cap >= 1) | (d_cap < cfg.cap_dwell_near)
            self._cap_dwell += (self.gate_unlocked & ~self.released_latch
                                & engaged).long()
            fdecay = (1.0 - self._cap_dwell.float()
                      / cfg.cap_annuity_decay).clamp(min=0.0)
            r_contact = r_contact * fdecay
            r_ctouch = r_ctouch * fdecay
            r_chold = r_chold * fdecay
            r_capp = r_capp * fdecay
        # U20②: 接触窗内罚指尖-瓶相对速度 (拿稳=指尖随瓶动, 拍击=大相对速度)
        near = d_ltip < cfg.gentle_zone
        lrel = (self.hand.data.body_lin_vel_w[:, self.left_tip_ids]
                - self.object.data.root_lin_vel_w[:, None])
        lspeed = lrel.norm(dim=-1).mean(dim=1)
        r_lslow = -cfg.w_left_slow * near.float() * lspeed.clamp(max=1.0)
        # U20③: 持握年金 —— ≥2 指连续接触计数爬坡, 戳一下不值钱
        self._lhold = torch.where((n_left >= 2) & thumb_c, self._lhold + 1,
                                  torch.zeros_like(self._lhold))
        r_lhold = cfg.w_left_hold * (self._lhold.float()
                                     / cfg.hold_horizon).clamp(max=1.0)
        # U20④: 左半动作变化率平滑罚 (累积管线里恒定运动 Δa≈0, 抖动 Δa 大)
        da = self.actions_buf[:, 29:] - self.prev_actions[:, 29:]
        r_lact = -cfg.w_left_act_rate * (da * da).mean(dim=1)
        r_imit = cfg.lam_imit * torch.exp(-wrist_err / cfg.sigma_imit)
        # U38b: 右拇/食/中贴手势参考的年金 (动作从数据学, 位置由 RL 修)
        r_fimit = torch.zeros_like(r_imit)
        if cfg.w_fin_imit > 0:
            _qf = self.hand.data.joint_pos[:, self.hand_jids]
            _fe = (_qf - self.ref_finger[self._t_fin])[:, self._triad_fj]
            _fe = _fe.abs().mean(dim=1)
            r_fimit = cfg.w_fin_imit * torch.exp(-_fe / cfg.sigma_fin_imit)
            if cfg.grip_gate:
                r_fimit = r_fimit * self.gate_unlocked.float()
            d0 = self.diag
            d0.add("fin_gesture_err", _fe, mask=self.gate_unlocked)
            d0.add("rew_fimit", r_fimit, mask=active)
        # 放置势: 释放后才开 (释放瞬间初始化, 防一步假分)
        if self._released_step.any():
            self.prev_place_d[self._released_step] = place_d[self._released_step]
        gate = self.released_latch.float()
        r_place = cfg.k_place * (self.prev_place_d - place_d).clamp(-0.02, 0.02) * gate
        self.prev_place_d.copy_(place_d)
        r_release = cfg.release_bonus * self._released_step.float()
        r_placed = cfg.place_bonus * self._placed_step.float()
        # Stage B: 瓶跟随参考 —— 势差分 (别让偏差涨) + 贴近小额年金
        r_body = torch.zeros_like(r_screw)
        if cfg.dynamic_body:
            r_body = (cfg.k_body_track * (self.prev_body_d - body_d).clamp(-0.02, 0.02)
                      + cfg.w_body_near * torch.exp(-body_d / cfg.sigma_body))
        self.prev_body_d.copy_(body_d)

        total = ((r_app + r_screw + r_contact + r_chold + r_ctouch + r_ctriad
                  + r_lgrip + r_lfull
                  + r_ltouch + r_lapp + r_capp + r_lslow + r_lhold + r_lact
                  + r_imit + r_fimit + r_place + r_body)
                 * active.float() + r_release + r_placed
                 - cfg.fail_penalty * self._fail_step.float())

        d = self.diag
        d.tick(active)
        d.add("cap_contact2_frac", (n_cap >= 2).float(), mask=active)
        d.add("cap_contact_any_frac", (n_cap >= 1).float(), mask=active)
        d.add("screw_deg", torch.rad2deg(self.screw_angle), mode="max")
        d.add("release", self.released_latch.float(), mode="max")
        d.add("placed", self.placed_latch.float(), mode="max")
        # U41 哨兵: 持盖运输步数 / 释放后盖下落峰速 (判"拿着放"还是"自己落")
        d.add("cap_carry_steps", self._cap_carry_steps, mode="max")
        d.add("cap_fall_peak", self._cap_fall_peak, mode="max")
        d.add("place_dist_cm", (place_d * 100).clamp(max=100.0),
              mask=self.released_latch)
        d.add("wrist_err_cm", wrist_err * 100.0, mask=active)
        d.add("tip_cap_cm", d_cap * 100.0, mask=active)
        d.add("cap_fell_frac", self._knock.float(), mode="max")
        d.add("left_grip3_frac", grip3.float(), mask=active)
        d.add("left_grip_any_frac", (n_left >= 1).float(), mask=active)
        d.add("left_thumb_frac", (lcon[:, 0] > 0).float(), mask=active)
        # U30a 哨兵: 接触中的左指尖沿瓶轴平均高度 (瓶根=0, 盖底≈18cm)
        _oq = self.object.data.root_quat_w
        _qx, _qy, _qz, _qw = _oq[:, 1], _oq[:, 2], _oq[:, 3], _oq[:, 0]
        _bax = torch.stack([2 * (_qx * _qz + _qw * _qy),
                            2 * (_qy * _qz - _qw * _qx),
                            1 - 2 * (_qx ** 2 + _qy ** 2)], dim=1)
        _tip_ax = ((ltips - obj_pos[:, None]) * _bax[:, None]).sum(-1)
        _lc = (lcon > 0).float()
        _grip_ax = (_tip_ax * _lc).sum(1) / _lc.sum(1).clamp(min=1)
        d.add("left_grip_axial_cm", _grip_ax * 100.0, mask=(n_left >= 1))
        d.add("body_track_cm", (body_d * 100.0).clamp(max=100.0), mask=active)
        if cfg.dynamic_body:
            d.add("body_lost", getattr(self, "_body_lost",
                                       torch.zeros_like(active)).float(), mode="max")
            d.add("rew_body", r_body, mask=active)
        d.add("rew_screw", r_screw, mask=active)
        d.add("rew_approach", r_app, mask=active)
        d.add("rew_contact", r_contact, mask=active)
        d.add("rew_lgrip", r_lgrip, mask=active)
        d.add("left_tip_bottle_cm", d_ltip * 100.0, mask=active)
        d.add("rew_lapp", r_lapp, mask=active)
        d.add("rew_capp", r_capp, mask=active)
        d.add("left_tip_speed", lspeed, mask=active & near)
        d.add("left_hold_steps", self._lhold.float(), mask=active)
        d.add("cap_hold_steps", self._chold.float(), mask=active)
        # U32 哨兵: 手型 —— 用几根指 / 是不是对的那几根
        d.add("cap_n_triad", n_triad, mask=(n_cap >= 1))         # 触盖时拇食中几根
        d.add("cap_thumb_frac", (capc[:, 0] > 0).float(), mask=(n_cap >= 1))
        d.add("cap_triad_frac", (n_triad >= 3).float(), mask=active)
        d.add("cap_wrong_frac", ((capc[:, 3:].sum(dim=1) > 0)
                                 & (n_triad < 2)).float(), mask=(n_cap >= 1))
        # 观测缺口守卫: 环/小也裹上盖 (五指抓 = 与精捏同价, 无罚). 若此项高
        # 而演示是三指精捏 -> 下一杆加环小触盖小额罚 (登记 U33 候选)
        d.add("cap_extra_frac", (capc[:, 3:].sum(dim=1) > 0).float(),
              mask=(n_cap >= 1))
        # U35 哨兵: 掌轴 vs 盖轴夹角 (0=对握姿, 90=侧蹭; 解锁后计)
        _cq = self.cap.data.root_quat_w
        _cx, _cy, _cz, _cw = _cq[:, 1], _cq[:, 2], _cq[:, 3], _cq[:, 0]
        _cax = torch.stack([2 * (_cx * _cz + _cw * _cy),
                            2 * (_cy * _cz - _cw * _cx),
                            1 - 2 * (_cx ** 2 + _cy ** 2)], dim=1)
        _hq = self.wrist_quat_w
        _hx, _hy, _hz, _hw = _hq[:, 1], _hq[:, 2], _hq[:, 3], _hq[:, 0]
        _hax = torch.stack([2 * (_hx * _hz + _hw * _hy),
                            2 * (_hy * _hz - _hw * _hx),
                            1 - 2 * (_hx ** 2 + _hy ** 2)], dim=1)
        _cosa = (_cax * _hax).sum(1).abs().clamp(max=1.0)
        d.add("palm_cap_deg", torch.rad2deg(torch.arccos(_cosa)),
              mask=self.gate_unlocked)
        # U34 哨兵: 实际生效的拧转速率增益 (1/3=单指慢拧, 1.0=三指全速)
        if cfg.screw_triad_drive:
            _gain = torch.where(n_triad > 0,
                                (n_triad / 3.0).clamp(min=cfg.screw_drive_floor,
                                                      max=1.0),
                                torch.zeros_like(n_triad))
            d.add("screw_gain", _gain, mask=(n_cap >= 1))
        # U40 哨兵: 真实螺纹副 —— 指尖经摩擦锥传入的轴向力矩 / 解锁占比
        if self.screw_spec.breakaway_torque_nm is not None:
            d.add("screw_tau_mNm", 1000.0 * self.screw_tau_ema.abs(),
                  mask=self.screw_engaged)
            d.add("screw_unlocked", (~self.screw_locked).float(),
                  mask=self.screw_engaged, mode="max")
        d.add("left_n_fingers", n_left.float(), mask=(n_left >= 1))
        d.add("left_grip5_frac", ((n_left >= 5) & thumb_c).float(), mask=active)
        d.add("rew_ctriad", r_ctriad, mask=active)
        d.add("rew_lfull", r_lfull, mask=active)
        d.add("rew_ctouch", r_ctouch, mask=active)
        d.add("rew_chold", r_chold, mask=active)
        d.add("cap_dwell_steps", self._cap_dwell.float(), mask=active)
        d.add("rew_ltouch", r_ltouch, mask=active)
        d.add("rew_lslow", r_lslow, mask=active)
        d.add("rew_lhold", r_lhold, mask=active)
        d.add("rew_lact", r_lact, mask=active)
        if cfg.dynamic_body and cfg.term_body_speed > 0:
            d.add("body_fly", self._body_fly.float(), mode="max")
        if cfg.grip_gate:
            d.add("gate_unlocked", self.gate_unlocked.float(), mode="max")
        if cfg.dynamic_body and cfg.term_body_drop_steps > 0:
            d.add("body_drop", self._body_dropped.float(), mode="max")
        d.add("rew_imit", r_imit, mask=active)
        d.add("rew_place", r_place, mask=active)
        self.extras.update(d.publish())
        return total

    # ---- 复位 ---------------------------------------------------------
    def _reset_robot(self, env_ids, starts, origins):
        super()._reset_robot(env_ids, starts, origins)
        if self.left_arm_jids is not None:
            # 左臂/左手硬写到轨迹首帧, 不从站姿甩过去.
            # 按基类同款公式重建全量 q (不能读 data buffer —— 刚写完可能未刷新);
            # 右侧取值与基类写入一致, 不破坏其 arm_tgt/q_cmd 记账.
            q = self.hand.data.default_joint_pos[env_ids].clone()
            q[:, self.arm_jids] = self.q_ref[starts]
            q[:, self.hand_jids] = self.ref_finger[starts]
            q[:, self.left_arm_jids] = self.q_left[starts]
            q[:, self.left_hand_jids] = self.left_finger_open[starts]
            self.hand.write_joint_state_to_sim(
                q, torch.zeros_like(q), env_ids=env_ids)
            self.hand.set_joint_position_target(q, env_ids=env_ids)
            if hasattr(self, "q_cmd_left"):      # 双臂缓冲 (init 尾段才建立)
                self.q_cmd_left[env_ids] = self.q_left[starts]
                self.q_left_prev[env_ids] = self.q_left[starts]
                self.left_finger_res[env_ids] = 0.0
                self.left_arm_tgt[env_ids] = self.q_left[starts]
                self.left_arm_tgt_prev[env_ids] = self.q_left[starts]
                self.left_fin_tgt[env_ids] = self.left_finger_open[starts]
                self.left_fin_tgt_prev[env_ids] = self.left_finger_open[starts]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.cap._ALL_INDICES if env_ids is None else env_ids
        super()._reset_idx(ids)
        self.ref_clock[ids] = 0
        self._clock_prev[ids] = 0
        self.placed_latch[ids] = False
        self._placed_step[ids] = False
        self.prev_place_d[ids] = 0.0
        self._cap_carry_steps[ids] = 0.0
        self._cap_fall_peak[ids] = 0.0
        self.wrist_err_ctr[ids] = 0
        self.gate_unlocked[ids] = False
        self._gate_step[ids] = False
        self._lhold[ids] = 0
        self._drop_ctr[ids] = 0
        self._body_dropped[ids] = False
        self._chold[ids] = 0
        self._cap_dwell[ids] = 0
        self._hold_ctr[ids] = 0
