# magicdexmate_Yanghong —— 我们这条线的全部覆盖,唯一真源。
#
# 这个文件**只放环境变量**,不放逻辑。逻辑在 bin/ 里,原版代码在 MDM_UPSTREAM。
#
# 为什么是 env var 而不是改原版:原版 scripts/lib/zed_common.sh 本来就把 ZED
# 这条线的旋钮做成了 MDM_* 环境变量,9 个 bringup 脚本全 source 它。所以下面
# 这些覆盖是**原版设计好的入口**,不是我们硬撬的 —— 原版一个字节都不用动。
#
# 用法:
#   source ~/luhr/magicdexmate_Yanghong/env.sh
# 或者别管它 —— bin/ 下的脚本每个都自己 source 一遍,直接用 bin/ 就行。

# ── 原版在哪 ────────────────────────────────────────────────────────────────
# 所有转发都以它为准。搬家改这一行。
export MDM_UPSTREAM="${MDM_UPSTREAM:-$HOME/luhr/magicdexmate_Yanghong/MagicDexMate}"
export MDM_YH_ROOT="${MDM_YH_ROOT:-$HOME/luhr/magicdexmate_Yanghong}"

# ── nano 上的 ZED streamer ──────────────────────────────────────────────────
# 2026-09-09 起**不在这里覆盖**:`setsid nohup … </dev/null` 已经是副本
# scripts/lib/zed_common.sh 的默认起流命令(理由写在那一行上面)。这里再设一遍
# 只会制造第二个真源 —— 哪天改了代码默认值,env.sh 会不声不响地把它顶回去。
# 临时要换分辨率/帧率:用原版旋钮 MDM_ZED_RES / MDM_ZED_FPS,它们现在插得进来了。

# ── 相机名 = 端口 ───────────────────────────────────────────────────────────
# ✅ **已现场确认 2026-09-09**:用手挡住左腕相机镜头,:30001 那格变黑 → 30001=左腕。
#    与原版占位值恰好一致,所以值没变,但从今天起它是实测,不是猜的。
#    验法(换相机/换支架后重验):view-wrist 开两格,挡左腕镜头,黑的那格是 left_wrist。
#
# 序列号 ↔ 端口(设备表顺序决定端口,base 必须是 30000,见 bin/view-all 的注释):
#   59595115   ZED X Mini    立体  :30000  头部
#   303351487  ZED X One GS  单目  :30001  左腕  ✅ 2026-09-09
#   307208395  ZED X One GS  单目  :30002  右腕  ✅ 2026-09-09
export MDM_NANO_ZED_STREAMS='head=30000,left_wrist=30001,right_wrist=30002'

# view_all.py 用的是**序列号=名字**,和上面那条(名字=端口)不是一个格式,
# 所以得单独写一份。bin/view-all 会把它变成 --zed-names。
# 与上面那条是同一事实的两种写法(名字=端口 vs 序列号=名字),改一条必须改另一条。
export MDM_YH_ZED_NAMES='59595115=head,303351487=left_wrist,307208395=right_wrist'

# nano 的地址/账号/跳板 —— 与原版默认相同,写在这里是为了出问题时一眼看到。
export MDM_NANO_HOST="${MDM_NANO_HOST:-192.168.50.22}"
export MDM_NANO_USER="${MDM_NANO_USER:-dexmate-nano}"
export MDM_NANO_JUMP="${MDM_NANO_JUMP:-dexmate}"

# ── Yanghong home 位 ────────────────────────────────────────────────────────
# 蹲角 28.2 与 Yanghong 站姿都已落进副本代码:
#   magicdexmate/head_waist_map.py  _SQUAT_DEG 默认 28.2
#   magicdexmate/home_pose.py       ARM_HOME_YANGHONG(拖拽线专用,tests 钉着)
# 所以这里不再设 VEGA_TORSO_SQUAT_DEG —— 同上,别造第二个真源。
