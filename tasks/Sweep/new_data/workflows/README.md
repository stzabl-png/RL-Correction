# 当前固定轨迹工作流

代码发布入口；当前执行根目录 `/home/msc-auto/RL_sweep`。服务器既有数据/网格/USD/抓姿/参考未随本次代码提交上传，依赖校验见server_dependencies.json。不能声称只克隆代码即可训练。

- restore_take9_human.py：从take9原始人手运动恢复双侧human输入，保持其他轨迹数组不变；输出已存在时拒绝覆盖。复用Sweep2 builder的函数但不执行其主程序，使用用户允许的实用IK容差。
- launch_take9_human_v2.sh：当前24M/1024环境/seed42启动命令，已在运行。文件中的run名是现有结果，禁止直接重复启动；新实验须使用独立名字和已授权预算。
- build_take32_level.py：已完成调平的构建配方，保留历史实现；其中human_left_q来自名义q，不能作为正式训练输入。必须按独立人手来源修复后再训练。
- record_take32_level.py：使用fixed初始化的无物块诊断回放。初始化写入零误差不等于动态跟踪零误差。
- prepare_take32_cube.py：当前未成功的选点器，全部候选被保守接触前条件淘汰。保留用于诊断，不是已完成的训练准备方案。
- TRANSFER_DIAGNOSIS.md：已核实证据、推断范围和建议顺序。

所有视频写入outputs_video，数值与日志写入logs。原Sweep2成功基准不变。旧diagnostic_human_from_arm只能用于诊断，不能用于full训练。当前新轨迹范围9、32、36、80；non-fixed后置。
