> 分析时点为human修复前。旧take9 Cube15已停止；HumanV2已恢复双侧独立人手引导并从头训练。下文3M指标属于旧run，不属于HumanV2。take32的human修复仍待完成。

# Sweep2迁移诊断：基于现有代码与产物

本轮只读分析现有训练与回放，不启动新实验，不修改baseline或运行中训练。不能将当前状态概括成四条通用资产训练全部失败：take32最新版本未训练，36/80当前版本未准备，take9 cube15仅有3M早期诊断。

## 已核实

1. take9 cube15 3M checkpoint：actual_agent_steps=3014656；统计窗口26 episodes，Gate1/2=1、Gate3/4=0；push=0，shape约0.01585、task约0.00157（代码pop_rates为每env-step均值，非每回合总和）。源：logs/checkpoints/Task3Take9FixedCube15__20260910_policy_0003M/metrics.json。不是独立评估或最终失败证明。
2. take32 level reference：566帧，最大左腕IK位置19.9766mm，旋转4.9983deg，最大相邻关节变化5.4009deg。源：logs/task1_take32_level_20260910/level_report.json。
3. take32实际回放：由trajectory_trace.npz中的pan_pose算上轴倾角，min/median/p95/max=8.09/17.50/25.06/25.32deg；局部入口端点[-/+0.06,-0.01435,0.108]实际最低高度距桌min/median/p95/max=2.87/4.49/5.75/18.66mm。局部法向参考为0deg并不等于实际水平。
4. take32左臂第5关节实际q-q_target：20帧后p5/median/p95=19.01/21.81/36.25deg，最大36.35deg。左右FixedJoint误差均小于0.00014mm、0.000021deg，实际关节FK与仿真手腕最大误差均小于0.00062mm。当前证据指向控制跟踪/负载/约束执行层，而非FixedJoint滑移或该链坐标错误；尚无力矩/接触力轨迹，不能确定是增益、力矩上限、接触阻挡还是其他具体原因。来源：outputs_video/Task3_take32_fixed_level_v1/full_zero/trajectory_trace.npz、对应prior、ArmIK。实际工具参考位置误差pan中位约92mm、broom约24mm，仅当前零残差回放，不能当作PPO表现。
5. take9 15mm零回放：实际cube位移首次>=5mm为控制row47，配置contact_row151；180帧全在盆外，TrainingEnv._signals的above_floor过滤180帧全为false。未过滤的progress最大0.01988，过滤后0。这支持盆外进度被过滤，但不能单凭零回放证明整个训练的唯一原因。
6. training_env.py同时用above_floor过滤entered/fully_inside/deep_inside以及progress/full_progress/deep_progress；盆底profile在盆外通过clamp外推，不只作用于盆内。防盆下假成功有必要，盆外approach shaping被同样过滤是迁移适配的可疑副作用。SweepEnv还以height_ok乘push奖励。
7. take32选点1125个候选：948个初始净空合格，186个存在后续向内近接触行，但所有都被接触前完整凸包安全条件淘汰；因此不能称几何无可行解。凸包分离是保守充分证明，未证分离不等于mesh实际碰撞；检查的是假设cube停在初始位置的完整历史，不是首次接触后的动力学。
8. Sweep2 reference的human_right_q/human_left_q均不同于名义q；take9 human_left_q==left_q；take32两侧human_q==名义q。因此当前human shaping部分/全部变成名义关节运动方向奖励，不能说完整算法语义未变。
9. residual包络：右0.10–0.25rad、左0.05–0.12rad，受confidence控制。take9 pan confidence中位94%；take32 pan80%。不能直接用单关节跟踪误差证明全14D无解，但局部修正范围与执行误差明显不匹配，需局部可达性证据。
10. 固定初始化把工具直接写到参考，因此初始化0偏差只是写入瞬间的状态；物理步后误差仍可能出现，不能将0偏差作为动态问题已解决。

## 判断与下一步讨论

主要瓶颈是尚未建立从通用资产/抓姿到任务接触几何、物理可执行轨迹、可达残差范围和有效reward的统一迁移契约。手腕IK/初帧对齐/视频好看不足以证明同一个可学习任务。原Sweep2成功是一个共同调通的实例，不是自动证明完整跨轨迹迁移。

推荐次序：先用现有回放定位实际关节跟踪原因；随后按实际扫把工作面与pan入口定义共同接触阶段/物块位置，保留小幅残差可修正的误差；单独核对盆外推进与盆内成功的奖励语义；恢复/明确定义human辅助流。再以一条新轨迹建立最小受控对照，避免四条长跑混合变量。零残差不必扫入成功，IK也不需重新收紧至毫米级。15mm减少初始重叠但也缩小有效接触尺度，不保证训练更容易。
