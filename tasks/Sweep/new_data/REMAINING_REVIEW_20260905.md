> 历史审计记录，不是当前状态。2026-09-06用户已验收32 v3、80 v6；最新步骤、失败原因和清理后路径统一见 /home/msc-auto/RL_sweep/Codex_new_data.md。旧视频已按用户要求清理，非视频证据见 logs/task3_docs_cleanup_20260906/old_video_evidence/。

# Task3 take80 / 128 / 180 处理记录（2026-09-05）
## 当前结论
take32 v3 已由用户明确验收。take80 v3 完整无方块物理录像已完成，待用户验收。take128、take180 困难段未过检查，按用户允许的范围保留证据暂时跳过，不发布可训练 reference。未启动训练/resume；单轨迹训练算法未改。Codex_tasks.md 与 arm_shell_points.npz 未触碰，未操作其他用户进程。

## take80：完整录像待验收
配置 tasks/Sweep/new_data/configs/take_80_v3.json；reference tasks/Sweep/new_data/references/take_80_reference_v3.npz。
ego 双手柄部抓法。pan rank2 fingertip_small__25_35_grasp.npy，broom rank2 fingertip_middle__19_18_grasp.npy。canonical 转 input 时计入 COM；pan 独立 dustpan_v3 翻盆体、保持柄部接触区，并根据原 mesh 尺寸降低入口；broom_v2 保留源网格。共享初始场景注册，保留 source0..299 的完整双工具6D轨迹，20Hz449行。仅重新计算换抓姿的一侧，未改右侧原有有效参考。
完整录像 outputs_video/Task3_take80_v3/full_zero/trajectory.mp4，22.45s，1280x720@20fps；相机与成功 Sweep2 相同。已连续播放至 ended=true。ego80、128、180 原视频另已连续播放完整10s。
审计：同目录 numeric_audit.json、trajectory_trace.npz、runtime_coordinate_chain.json；logs/task3_remaining_20260905/fk80.json。
左右臂 FK 位置最大约2mm；右/左关节步长5.133°/1.691°。物理扫帚位置误差 median4.706mm/max34.725mm，簸箕 median3.461mm/max9.812mm；局部运动跟踪滞后保留。簸箕最低网格约低于桌面2.227mm。扫帚源运动较高，最低离桌46.074mm，未人为压低。闭环手—工具约束误差<0.000131mm。cube 隐藏/无碰撞，奖励为零，done无cube影响，zero residual 全行回放。

## take128：连续性失败暂存
ego 左手抓pan侧边，保留侧边抓法；原pan +Y向上，不翻面，仅修中央入口，保护abs(x)>=65mm抓区。broom 原头部朝向错误，独立 broom_v3 做柄部过渡的头部修复。shared scene_table_z=.7492041086193705，修复mesh后整体注册升高7.9814mm，未单独平移工具。
右手柄部候选43（8_Prismatic_2_Finger__7_20_grasp.npy）完整IK出现153.213°跳变；共享高度校正后仍未通过。候选62（8_Prismatic_2_Finger__14_11_grasp.npy）重新计算右臂和human-right，位置误差<2mm，但机器人关节跳173.603°，不能当连续轨迹发布。
最终诊断配置 take_128_v5.json；失败candidate references/take_128_reference_v5.npz.candidate.npz。日志 right128_v5.log；此前 build128/reanchor128日志也保留在 logs/task3_remaining_20260905。没有完整可验收录像，不能以首帧可达替代全程连续性。若后续继续，优先检查源姿态与机器人可达域；不得放宽8°门槛或改训练算法。

## take180：已替换正确扫帚，完整轨迹仍受阻
用户授权明显错误资产可直接用其他轨迹正确资产替代。原180扫帚近似勺状、刷毛细节缺失，独立生成 broom_donor32_v3，复用用户已验收take32扫帚及其对应prior。
配置 take_180_v3.json；prior take_180_broom_donor32_v3.npz。脚本 substitute_broom_asset.py，严格限定当前测量的32→180变换。无需安装环境。
坐标链：donor候选canonical→带COM的donor input→recipient input→identity USD rigid root。刚体R=diag(-1,-1,1)，t=(-.01171733235,.01726142322,-.01937352774)m，scale=1。长轴+Z不变，donor -Y刷毛映射recipient +Y（原始世界up≈-.974Y），接触中心对齐；手指角与手—工具相对几何完全保留。take180工具root轨迹未改，不复制take32运动。
审计 assets/take_180/broom_donor32_v3/substitution_audit.json：175036v/350072faces、watertight、面索引保持、USD网格坐标误差4.21e-9m、相对接触坐标误差4.17e-17m；donor canonical与prior一致已验证。已观看CPU双视角 grasp 图 logs/task3_remaining_20260905/donor180_grasp.png。此图是静态几何检查，不是物理通过证据。
首帧CPU IK可达，但旧资产完整构建row310无解（9.28cm/19.76°）。新donor在同一row310（source163.917、15.5s）32初值审计0/32成功，最佳15.7772cm/29.0068°；未继续浪费全程构建。日志 audit180_donor_row310.log，命令 launch_audit180_donor.sh（仅CPU IK审计，不是训练）。
新donor首帧Isaac也未通过：共享初始化试图单独平移簸箕[4.0458,-.5302,1.7790]mm，Task3断言将其拦截。见 static180_v3.log、launch_static180_v3.sh。不得删除断言后宣称通过。簸箕ramp代理最低点约低于桌2.12mm，有接触影响可能，尚未确认全部根因。
因此只交付正确资产和诊断证据，未生成take180完整可用reference/物理录像。暂时跳过，不训练。

## 复现与下一步
所有路径相对 /home/msc-auto/RL_sweep。运行环境 /home/msc-auto/rlcorr-venv/bin/python，PYTHONPATH=.，VEGA_URDF=/home/msc-auto/data/vega_urdf/vega_1p_sharpa.urdf，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。完整运行命令均保存在 logs/task3_remaining_20260905/launch_*.sh；每个失败日志与配置版本对应，不能把失败candidate当通过reference。
donor配置禁止经过 source-only prior_frame.prepare / prepare_broom_usd 重写，需使用 substitute_broom_asset.py 保留附加坐标变换。
下一步由用户查看take80录像。32已验收；80未得到用户验收；128/180保留阻碍，四条未全部通过。不进行cube start、gate、reward标定、random policy或训练。

# 后续更新：修复take80资产并继续128/180（本节优先）
用户认可take80运动走向，但拒绝其扫帚资产；要求先修80，再继续128/180，确实未过则记录原因，并清理无用途杂产物。先前take80 v3不再作为可验收资产版本。

## take80 v4：正确扫帚替换和完整重录完成，待用户视觉确认
- donor为用户已验收的take32 v3扫帚与对应抓姿。资产在assets/take_80/broom_donor32_v4；配置configs/take_80_v4.json；prior priors/take_80_broom_donor32_v4.npz。
- donor input→take80 input：Rz(+90°)，长轴+Z保持，刷毛donor -Y→take80 +X（原source world up≈[-.974,-.225,-.023]）。按contact centroid作刚体平移，scale=1。同步映射整个prior，手指角和手—工具接触几何不变。
- canonical/COM、donor input、recipient OBJ、identity USD rigid root均已验证，见prepared/take_80_v4_frames.json及资产substitution_audit.json；运行时world链和物理手—工具闭环在录像trace中核验。
- 只重算右臂及human-right；左侧和双工具完整运动保留。新右臂有14.285°单步，沿用take32已验证的共享局部时间加密，不丢任何原始行，共增加13行，449→462行/23.1秒。保留原始双工具pos/quat、source_frame、left_q、human_left_q、left_f和confidence各行逐字节一致，证据logs/task3_assets_20260905/preserve80_audit.json。
- retime_grasp_candidate.py验证新增帧与四组完整FK：robot-right2.188mm/1.138°/step5.492°，left1.993mm/.861°/step1.691°；human-right1.962mm/1.140°/step3.429°，human-left1.999mm/1.099°/step2.423°。初始求解算法未变，局部时间加密只在FK通过后采用；>30°分支跳变直接拒绝。
- 静态物理right1.426mm/0.137°、left2.529mm/0.668°，未触发单工具registration。
- 完整视频outputs_video/Task3_take80_v4/full_zero/trajectory.mp4，成功Sweep2相机，1280x720@20fps。已实际连续播放至ended=true/23.1s并看起始/中段/终态。
- 全462行zero residual、cube隐藏且无碰撞/终止影响。新扫帚最低距桌+3.935mm；原簸箕最低-2.227mm。扫帚跟踪median5.252mm/max34.697mm，簸箕median3.461mm/max9.812mm，局部跟踪滞后如实保留。FixedJoint闭环最大0.000168mm。numeric_audit.json和trajectory_trace.npz在录像目录。
- 当前是代理数值/视觉检查完成，不替代用户对新资产版本的验收。没有训练、resume或cube参数校准。

## take128：正确资产后仍未通过，暂存
- configs/take_128_v6.json改用donor32扫帚，Rz180°映射刷毛到input+Y向下；保持原ego左手pan侧边抓法，独立pan mesh不变。canonical/COM/OBJ/USD审计通过prepared/take_128_v6_frames.json。
- 当前摆放下完整右臂虽然各点<2mm，但发生350.664°跳变，右第3关节从-175.955°转到+174.709°，恰在有限位关节两端（范围±175.955°），不能用角度unwrap越过限位伪装连续。source217.5→218.0，证据discontinuity128.json。human-right本身连续（max2.913°）。
- 原始source79→80和80→81工具旋转20.850°/19.099°，同期human wrist仅0.121°/.094°，工具rotation confidence45/52。source128_orientation.json记录这种工具—人手重建不一致；不把它单独当成所有IK失败的已证实根因。
- 额外检查7组共享XY注册、4组共享yaw/XY注册；全部保持双工具相对关系和原运动，仅修改整体初始摆放。共享yaw+45°（总yaw31°）抽查双手18/18可达，因此对其重新做完整右臂同算法求解，仍有58.779°最大步长，未通过。right128_yaw31.log及reference*.robot_failure.json保存失败；超30°后省去无意义human重建。该候选没有完整reference，更没有物理全程录像。
- v6静态right附着误差3.759mm超过3mm阈值（left.373mm），也未放行。未绕开断言。
- 当前结论是本轮有限抓姿/整体注册尝试未找到满足完整轨迹、关节限位和连续性的方案，不声称数学上不存在解。按用户允许暂时跳过，未放宽限制或修改源轨迹。

## take180：已换正确资产，定位完整运动问题，暂存
- 保留上一轮donor32_v3正确扫帚与抓姿，不再使用原勺状粗糙资产。
- 使用build_reference.py --geometry_only --geometry_output导出完整原运动注册几何，明确不生成任何可训练关节reference。文件prepared/take180_registered_geometry_v3.npz。
- 原失败row310/source163.9167目标wrist世界z=.81953m，桌面z=.87m，低约50.47mm；全程wrist最低.81517m。对应ego164画面中手腕在桌面上方。正确替换扫帚的目标mesh全程最低-21.306mm（source138），这些是几何目标检查，不是物理实测。
- source159→160工具旋转28.936°/human1.655°；source163→164工具41.429°/human4.306°。详见source180_geometry.json与ego180_frame164.jpg。
- 7组共享XY平移，每组双手8个关键source共16pose均只通过11/16，关键不可达未消除。registration180.json及日志完整保留。平面内平移/yaw不改变上述目标高度；不通过抬高全部工具使簸箕离桌、删段或抑制工具旋转掩盖问题。
- 上一轮32初值row310也0/32成功；静态物理单独pan平移约4.45mm被Task3拦截，未证明解决。本轮因上游完整运动检查仍失败，不重复启动无意义全程物理录像。
- 当前完整重建轨迹、抓姿和资产约束下未放行，需后续回查上游tool pose与human wrist一致性；保留证据后暂时跳过。

## 清理与运行边界
- logs/task3_assets_20260905/cleanup_manifest.json保存逐文件路径、尺寸、SHA256和删除原因。
- 删除2个与final完全相同且无脚本引用的take80 candidate副本、3个可再生Task3 pyc，以及一份生成失败、无配置/launcher引用的take80旧dustpan_v2网格（35,287,881 bytes）；原始输入、修复规格、拒绝图片均保留。移除已失败的空录像目录。
- 删除上一轮本地浏览器播放暂存目录/tmp/task3_remaining_20260905（13,177,914 bytes）；远端证据和已交付的本地最终视频/报告保留。本地清单随本轮用户报告交付。
- 本轮已核实清理约48.84MB（十进制）。保留有效资产、完整录像、失败原因、必要失败candidate和可复现命令；未清理datasets、成功Sweep2或其他用户目录。
- 启动命令均在logs/task3_assets_20260905/launch_*.sh，运行仍是既有rlcorr-venv、Task3单环境诊断。protected_before.sha256核对Codex_tasks.md、arm_shell_points.npz及共享训练相关文件全部未变。
- 本轮新代码/改动均在tasks/Sweep/new_data；无训练算法改动、无训练/resume、无commit/push。

最终收尾：本轮当前本地播放暂存也已清理（2,202,246 bytes），最终本地视频和报告保留；累计清理约51.04MB。全部Task3诊断进程已结束，无训练/resume。

## 16. 最新：take80 v6 保留旧柄，只替换刷头（待用户确认）
用户指出 v4 刷毛反向，并明确允许保留旧柄只改毛刷。v4 已被拒绝，不得当作验收录像。根因是用首帧 world-down 选择 donor Rz90°，未检查源运动先朝内、再朝下的转腕过程。v5 整体 donor 无旋转虽修正方向，但首帧 IK 失败，已放弃。
当前 configs/take_80_v6.json 使用 broom_head_v6：旧 v3 柄部 z<=.012m 原顶点不动，保留全部原 contact 区；take32 已验收刷头仅平移 [0,-.008,.028]m、无旋转，刷毛 input -Y。柄/头分别封闭并重叠连接，不宣称布尔焊接。graft_take80_broom_head.py 可复现。只有一个 USD rigid root，canonical→input OBJ→USD 闭环通过。
完全复用 take_80_reference_v3.npz 与旧双手 prior，449行22.45秒；新完整录像 outputs_video/Task3_take80_v6/full_zero/trajectory.mp4，沿用成功 Sweep2 摄像机。cube 隐藏、碰撞/终止关闭，zero residual。已将旧v3与新v6近景连续并排播放至末尾，起始朝内、随后朝下，未观察到柄头脱节。物理刷毛方向在row50起 world-Z 最大-.975725（偏离向下约12.65度），不是只检查首帧。
numeric_audit.json：扫帚 tracking 最大34.727mm、中位4.717mm，簸箕最大9.812mm；固定抓取位置误差<.000121mm。刷头全程最低离桌25.322mm，簸箕最低-2.227mm。这些原轨迹跟踪/接触局限仍保留，不宣称已证明扫块成功。当前只交付资产/抓姿/完整轨迹复核，待用户视觉确认。
logs/task3_take80_brush_direction_20260905 保存构建、失败原因、坐标与方向审计、命令。128/180仍保持上一节的暂存及原因，无新放行。无训练/resume；Codex_tasks.md、arm_shell_points.npz及既有共享训练文件hash全部未变，自己的诊断进程已结束。
经验：替换资产必须在整个功能运动段核对工作面方向，不能以首帧朝下来决定旋转。
