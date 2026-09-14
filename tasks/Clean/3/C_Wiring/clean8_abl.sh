#!/usr/bin/env bash
# take8 倒扣盘·运动变体 · 三臂消融发车 (台账 §5.22)
#   与 take18 **同资产同抓法, 只差擦拭运动** —— 盘=t18 倒扣盘(Clean18_plate_left),
#   布=take3 资产与先验; 母带 clean8_reference_v1.npz (源帧 0:293, sponge_yaw 165)。
#   ⚠ take8 参考行程只有 59cm, 沿用 70cm 门槛 success 会结构性为 0 ⟹ CLEAN_TRAVEL_MIN_CM=47
#     (按 take3 定线的同口径 = 参考自身 80%; 奖励斜率走 travel_norm 恒 0.70 不受影响)。
#   ⚠ 覆盖率在倒扣盘上饱和 (参考自身 0.88~1.00), 判读时不要看覆盖, 看行程/接触行。
#   用法: bash clean8_abl.sh <base|a1|a2> SEED [GPU] [额外 KEY=VAL ...]
#   例:   bash clean8_abl.sh a1 42            # 本机默认 GPU0
#         bash clean8_abl.sh base 42 1        # 指定 GPU1 (Denso/msc 这类多卡机)
#
# 三臂 (对位 take3 的 §5.15, 共用 SOFT_REL=1 W=0.5 这个冠军承重项):
#   base = 人手指姿指引 (CLEAN_S2_HAND_REF=1)
#   a1   = 撤人手指引 (什么都不加)          <- take3 那边 A1 可以复用冠军存档, take18 **必须真跑**
#   a2   = a1 + 拍平置信度 (CONF_FLAT=1)
#
# take18 专有四件套 (缺一件就会静默跑成 take3 的场景, 见 §5.16):
#   CLEAN_CLIP / CLEAN_PRIOR_PLATE / CLEAN_REF_NPZ / CLEAN_S2_DIE_ROT_DEG
#   ⚠ 转角死线 180 = 关掉 (用户 2026-09-10 裁定"只要不是抓不住掉落"), 位移3cm/掉落/落桌/盘倾角30° 仍在;
#     软罚跨度恒 20° 不受影响 (整形留着, 铡刀去掉)。三臂必须同值, 否则消融被混淆。
set -euo pipefail
ARM=${1:?用法: clean8_abl.sh <base|a1|a2> SEED [GPU] [KEY=VAL ...]}; SEED=${2:?缺 SEED}; shift 2
GPU=0; if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then GPU=$1; shift; fi
REPO=/home/lyh/Project/RL_Correction
[ -d "$REPO" ] || REPO=$HOME/RL_Correction                      # Denso/msc 上的路径
# 解释器按候选逐个探 (各机环境名不同: Denso/UCBY/msc 叫 isaac, taitan-vision 叫 env_isaaclab)。
# 2026-09-11 教训: 写死两条路径 -> taitan 上 nohup 直接 "No such file or directory", 白发一次车。
PY=""
for _c in /home/lyh/luhr/MagicSim/.venv/bin/python \
          "$HOME/miniconda3/envs/isaac/bin/python" \
          "$HOME/miniconda3/envs/env_isaaclab/bin/python" \
          "$HOME/miniforge3/envs/isaac/bin/python"; do
  [ -x "$_c" ] || continue
  "$_c" -c "import isaaclab" >/dev/null 2>&1 || continue      # 能 import 才算数
  PY="$_c"; break
done
[ -n "$PY" ] || { echo "[$(basename $0)] 找不到能 import isaaclab 的解释器" >&2; exit 2; }
echo "[$(basename $0)] 解释器: $PY"
cd "$REPO"

case "$ARM" in
  base) ARMFLAGS=(CLEAN_S2_HAND_REF=1);   NAME=Clean8_ablBase_s$SEED ;;
  a1)   ARMFLAGS=();                      NAME=Clean8_ablA1_s$SEED ;;
  a2)   ARMFLAGS=(CLEAN_S2_CONF_FLAT=1);  NAME=Clean8_ablA2_s$SEED ;;
  *) echo "未知臂 '$ARM' (只认 base|a1|a2)" >&2; exit 2 ;;
esac

T18=(
  CLEAN_CLIP=Clean8_plate
  CLEAN_PRIOR_PLATE=$REPO/tasks/pregrasp/priors/Clean18_plate_left.npz
  CLEAN_REF_NPZ=tasks/Clean/3/A_Design/L2_Reference/clean8_reference_v1.npz
  CLEAN_TRAVEL_MIN_CM=47
  CLEAN_S2_DIE_ROT_DEG=180
  CLEAN_S2_SOFT_REL=1
)
export OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH="$REPO"
export POUR_OBJ_MASS=0.3 POUR_OBJ_FRIC=1.0 POUR_PAD_FRIC=1.0    # G-A 物理规矩 (Clean: 盘 0.3kg)
export CUDA_VISIBLE_DEVICES=$GPU
# 多卡机必须关 gpu_guard: flock 是**按机器**抢的不按卡, 同机第二条会静默排队等到天荒地老 (台账/记忆)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NGPU:-1}" -gt 1 ]; then export RL_ISAAC_NO_GUARD=1; echo "[clean8_abl] 多卡机 (${NGPU}卡) -> RL_ISAAC_NO_GUARD=1"; fi
# 共享机上 /tmp/isaaclab/logs 可能属于别的用户 (UCBY 实测 PermissionError, IsaacLab 的
# configure_logging 默认写 tempfile.gettempdir()/isaaclab/logs)。只在写不进去时才改 TMPDIR,
# 免得连带把别的缓存也搬走。
if ! mkdir -p /tmp/isaaclab/logs 2>/dev/null || [ ! -w /tmp/isaaclab/logs ]; then
  export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR/isaaclab/logs"
  echo "[clean8_abl] /tmp/isaaclab/logs 不可写 -> TMPDIR=$TMPDIR"
fi
mkdir -p launch_logs
echo "[clean8_abl] 臂=$ARM 名=$NAME seed=$SEED GPU=$GPU"
echo "[clean8_abl] take18 四件套: ${T18[*]}"
echo "[clean8_abl] 臂旗: ${ARMFLAGS[*]:-(无)} | 额外: $*"
setsid env "${T18[@]}" "${ARMFLAGS[@]}" "$@" nohup "$PY" -u tasks/Clean/3/C_Wiring/train_task.py \
  --headless --num_envs "${NUM_ENVS:-512}" --seed "$SEED" --name "$NAME" \
  --max_agent_steps "${MAX_STEPS:-30000000}" > "launch_logs/$NAME.log" 2>&1 < /dev/null &
echo "[clean8_abl] pid=$! log=launch_logs/$NAME.log"
echo "[clean8_abl] 发车后核验: grep -E '难度覆写|指垫摩擦|CleanTaskEnv\] N=|判据几何' launch_logs/$NAME.log"
