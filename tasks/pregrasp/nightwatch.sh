#!/usr/bin/env bash
# 夜间守护(2026-08-17 版): 本机 L5_pour17 掉了自动重起; 远端只汇报进度不重起
# (A2/B2 已停, 双卡机现在跑 G1/A_timeshape, 各有自己的起法, 别在这里瞎重起抢卡)。
#
# 用法(本机跑):
#   setsid nohup bash tasks/pregrasp/nightwatch.sh > ~/nightwatch.log 2>&1 < /dev/null &
# 停:
#   touch ~/.cache/rl_correction/nightwatch.stop
set -uo pipefail
LOCK="$HOME/.cache/rl_correction"; mkdir -p "$LOCK"
STOP="$LOCK/nightwatch.stop"; rm -f "$STOP"
DUAL=msc-auto@128.32.164.89
UCB=yanghong@169.229.192.185
POLL="${NIGHTWATCH_POLL:-300}"

log(){ echo "[$(date +%m-%d\ %H:%M)] $*"; }

restart_local(){
  log "  -> 重起 L5_pour17"
  cd /home/lyh/Project/RL_Correction
  SHARPA_WANDB=0 PYTHONPATH=. setsid nohup /home/lyh/luhr/MagicSim/.venv/bin/python -u \
    -m tasks.pregrasp.train --headless --name L5_pour17 --clip Pour17_bottle --num_envs 1024 \
    --prior_npz tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
    --bimanual --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180 \
    --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz --ff_freeze_cm 5 --ff_pull 0.08 --l5 --dyn_far 2.0 \
    --approach --minimal --kl_threshold 0.02 --auto_stop dry \
    --max_agent_steps 40000000 --seed 42 >> ~/l5_pour17.log 2>&1 < /dev/null &
}

log "夜间守护启动 | 巡检间隔 ${POLL}s | 停止: touch $STOP"
while :; do
  [ -f "$STOP" ] && { log "收到停止标记, 退出"; exit 0; }

  # ---- 本机 L5_pour17 ----
  n=$(ps -eo args | grep -c '[B]iRef_pour17_L' || true)
  if [ "${n:-0}" -lt 1 ]; then
    if grep -qa "max steps achieved" ~/l5_pour17.log 2>/dev/null; then
      log "L5_pour17 已正常跑完(max steps), 不重起"
    else
      log "⚠ L5_pour17 掉了 (末尾: $(tail -2 ~/l5_pour17.log 2>/dev/null | tr -d '\0' | tail -1 | cut -c1-120))"
      # 先清可能占显存的僵尸 Isaac, 等 10 秒再起(carb mutex)
      for p in $(ps -eo pid,args | awk '$0 ~ /[p]regrasp.train/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
      sleep 12; restart_local; sleep 90
    fi
  fi

  # ---- 进度快照 (只汇报, 不动手) ----
  x=$(grep -a '确定性成功率' ~/l5_pour17.log 2>/dev/null | tail -1)
  [ -n "$x" ] && log "  本机 BiRef: $x"
  timeout 50 ssh -o BatchMode=yes $DUAL 'for f in ~/l4_bi.log; do [ -f "$f" ] || continue; e=$(grep -a "确定性成功率" "$f"|tail -1); [ -n "$e" ] && echo "  $(basename $f): $e"; done' 2>/dev/null
  timeout 50 ssh -o BatchMode=yes $UCB 'for f in ~/c_bi.log ~/d_bi.log; do [ -f "$f" ] || continue; e=$(grep -a "确定性成功率" "$f"|tail -1); [ -n "$e" ] && echo "  $(basename $f): $e"; done' 2>/dev/null

  sleep "$POLL"
done
