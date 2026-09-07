#!/usr/bin/env bash
# Unscrew clip17 当前正式配方 (2026-09-07, v61 = 拧盖 / PULL_v1 = 拔盖), 只导出环境变量。
# 用法:  source tasks/Unscrew/part4/C_Wiring/recipe_v61.sh            # 拧盖 (默认)
#        MODE=pull source tasks/Unscrew/part4/C_Wiring/recipe_v61.sh  # 拔盖
# 然后:  tasks/Unscrew/part4/C_Wiring/launch_remote.sh <NAME> <GPU> 42 HYB 17
# launch_remote.sh 自己会导出 POUR_SQUEEZE_FF/BONUS_NOW/BONUS_DIST/PAD_FRIC/VARIANT/CLIP,
# 这里只放 UNSCREW_* 这一层 (逐字对应 logs/launch_v61.sh / launch_pull_v1.sh)。
export UNSCREW_CLIP=17
export UNSCREW_STAGE_C_FRAC=0.25                 # 25% 回合从"对准态" c 出生, 双臂冻结
export UNSCREW_CAP_GRASP=user_thumb_index        # 用户手调的拇食双指盖抓姿 (priors/Screw27_cap_candidates)
export UNSCREW_CAP_PINCH=8
export UNSCREW_MAX_TILT=80
export UNSCREW_CERT_WAGE=3.0                     # T2-32 举升认证连续工资
export UNSCREW_CAP_WAGE=3.0                      # T2-33 触盖计件
export UNSCREW_CAP_APPROACH=3.0                  # T2-34 右垫→盖心 12→2cm 接近坡
export UNSCREW_PINCH_WAGE=0.05                   # T2-36 捏且转/捏且拔 每步工资 (>=2 垫 + 当步爬新高)
export POUR_UNLOCK=1,2,3                         # 渐进 RSI 解锁 (三个 gate 都开)
if [[ "${MODE:-screw}" == "pull" ]]; then
  export UNSCREW_CAP_MODE=pull                   # T2-40 拔盖: 螺纹副 → 轴向塞盖副
  export UNSCREW_PULL_FULL_MM=15                 # 拔出 15mm 即脱扣 (= 满角 30° 语义)
  export UNSCREW_PULL_BREAKAWAY_N=2.0            # 静锁解锁力
  export UNSCREW_PULL_KINETIC_N=0.8              # 滑动库仑阻力
  export UNSCREW_PULL_VISCOUS=40                 # N·s/m 过阻尼
  export UNSCREW_PULL_VMAX=0.15                  # m/s
else
  export UNSCREW_CAP_MODE=screw
fi
echo "[recipe] clip17 MODE=${UNSCREW_CAP_MODE} 已导出 (验收凭据: $([[ $UNSCREW_CAP_MODE == pull ]] && echo acceptance_v2_pull.json || echo acceptance_v2.json))"
