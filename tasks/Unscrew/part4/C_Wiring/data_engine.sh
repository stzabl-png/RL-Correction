#!/usr/bin/env bash
# Unscrew 数据引擎: 每次用一条重建数据 -> RL 恢复好轨迹 -> 批量导出演示。
#
#   PY=~/miniforge3/envs/isaac/bin/python bash tasks/Unscrew/part4/C_Wiring/data_engine.sh 32
#   bash .../data_engine.sh all          # 按 README 榜单顺序跑双good双可用 9 条
#
# 每条 clip 的完整流水 (框架 CHECKLIST 顺序, 不可乱):
#   1 probe_rest        env 实测静置位/anchor_T/站姿 -> env_rest.json
#   2 make_reference    v1 母带 (实测锚重跑离线 IK)
#   3 plan_machine_segs cuRobo Approach+Retreat (缺 NVlabs cuRobo 时跳过,
#                       用 smoothstep 占位 —— 只够冒烟, 无碰撞背书)
#   4 make_reference    重跑, 剪进规划行
#   5 selftest 家族     判据离线全绿 (不开 Isaac)
#   6 smoke_zero        零动作冒烟 (A 静置对账 <5mm 铁则)
#   7 probe_beta        βL 标定 (人工判读后回填 task_config)
#   8 build_reference   v2 重铸 (站位捕获 + 物体轨迹反解 IK + 认证行)
#   9 probe_ikcheck / probe_acceptance   运动学核查 + 零动作硬闸
#  10 train_task        发射 (SHARPA_WANDB=0 POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1
#                       POUR_BONUS_DIST=1 POUR_VARIANT=HYB --headless)
#  11 eval_task         确定性评测 (唯一成功率口径)
#  12 record/export     演示导出 (多种子/多回合 = 一条数据的多组生成)
#
# 本脚本自动跑 1-6; 7-12 涉及算力/人工判读, 打印命令不代跑。
set -e
cd "$(dirname "$0")/../../../.."
PY=${PY:-$HOME/miniforge3/envs/isaac/bin/python}
TMPDIR=${TMPDIR:-$HOME/tmp}
mkdir -p "$TMPDIR"
export TMPDIR OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
D=tasks/Unscrew/part4
CLIPS=${1:-32}
[ "$CLIPS" = all ] && CLIPS="32 89 60 17 67 53 68 36 8"

for C in $CLIPS; do
  echo "================ clip $C ================"
  export UNSCREW_CLIP=$C SHARPA_WANDB=0 PYTHONPATH=.
  $PY $D/C_Wiring/probe_rest.py --headless
  $PY $D/A_Design/L2_Reference/make_reference.py
  if $PY -c "import curobo" 2>/dev/null; then
    $PY $D/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py --headless
    $PY $D/A_Design/L1_Data/Motion_Planning/plan_machine_segs.py --retreat --headless
    $PY $D/A_Design/L2_Reference/make_reference.py
  else
    echo "[engine] ⚠ 本机无 NVlabs cuRobo, 机器段=占位 —— 只够冒烟"
  fi
  for t in selftest_regime selftest_progress selftest_rsi selftest_variants \
           selftest_progress_batch; do
    $PY $D/A_Design/L3_Learning/$t.py
  done
  $PY $D/C_Wiring/smoke_zero.py --headless
  cat <<EOF
[engine] clip $C 冒烟链完毕. 后续 (人工判读节点, 命令自取):
  UNSCREW_CLIP=$C ... $D/B_SmokeTest/probe_beta.py --headless        # βL 标定
  UNSCREW_CLIP=$C ... $D/B_SmokeTest/build_reference.py --headless   # v2
  UNSCREW_CLIP=$C ... $D/B_SmokeTest/probe_ikcheck.py --headless
  UNSCREW_CLIP=$C ... $D/B_SmokeTest/probe_acceptance.py --headless
  UNSCREW_CLIP=$C SHARPA_WANDB=0 POUR_SQUEEZE_FF=1 POUR_BONUS_NOW=1 \\
    POUR_BONUS_DIST=1 POUR_VARIANT=HYB PYTHONPATH=. \\
    $PY $D/C_Wiring/train_task.py --name Unscrew${C}_0 --num_envs 512 --headless
  UNSCREW_CLIP=$C ... $D/C_Wiring/eval_task.py --checkpoint logs/Unscrew${C}_0/stage1_nn/last.pth --headless
EOF
done
