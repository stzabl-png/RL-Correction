#!/usr/bin/env bash
# L3 -> L4 交接: 等 L3 (auto_stop) 正常结束 -> 起 L4 (2.0 联合参考) -> 守护/录像切到 L4。
# 独立后台跑, 不依赖交互会话。日志: ~/l4_handoff.log
set -uo pipefail
cd /home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
log(){ echo "[$(date +%m-%d\ %H:%M)] $*"; }

log "等待 L3 结束 (auto_stop 达标或 40M)..."
while pgrep -f "name BiRef_pour17_L3" >/dev/null 2>&1; do sleep 60; done
log "L3 进程已退出. 末两行: $(tail -2 ~/biref_pour17_L3.log | tr -d '\0' | tail -1 | cut -c1-100)"

# 停旧守护(它只认 L3, 会把 L3 拉回来) + 旧录像守护
touch "$HOME/.cache/rl_correction/nightwatch.stop"
for p in $(ps -eo pid,args | grep "autorec.sh BiRef_pour17_L3" | grep -v grep | awk '{print $1}'); do
  kill "$p" 2>/dev/null
done
sleep 15
# 保险: 若守护在窗口期又把 L3 拉了起来, 清掉
for p in $(ps -eo pid,comm,args | awk '$2 ~ /python/ && /BiRef_pour17_L3/ {print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 12

log "起 L4 (2.0 联合参考: 双手同时, 左手终点=离杯5cm近点)"
SHARPA_WANDB=0 setsid nohup "$PY" -u -m tasks.pregrasp.train --headless \
  --name BiRef_pour17_L4 --clip Pour17_bottle --num_envs 1024 \
  --prior_npz tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
  --bimanual --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180 \
  --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz \
  --ff_freeze_cm 5 --ff_pull 0.08 \
  --approach --approach_only --minimal --kl_threshold 0.02 --auto_stop dry \
  --max_agent_steps 40000000 --seed 42 > ~/biref_pour17_L4.log 2>&1 < /dev/null &

# 守护切到 L4 (nightwatch 里的 ref 路径一并换)
sed -i 's/BiRef_pour17_L3/BiRef_pour17_L4/g; s/biref_pour17_L3\.log/biref_pour17_L4.log/g; s|curobo_pour17_ref\.npz|curobo_pour17_joint_ref.npz|' \
  tasks/pregrasp/nightwatch.sh
rm -f "$HOME/.cache/rl_correction/nightwatch.stop"
setsid nohup bash tasks/pregrasp/nightwatch.sh > ~/nightwatch.log 2>&1 < /dev/null &
setsid nohup bash tasks/pregrasp/autorec.sh BiRef_pour17_L4 ~/biref_pour17_L4.log \
  --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
  --bimanual --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180 \
  --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz \
  --ff_freeze_cm 5 --ff_pull 0.08 --approach --approach_only --minimal \
  > ~/autorec_l4.log 2>&1 < /dev/null &
log "L4 + 守护 + 录像 已全部拉起 ✅"
