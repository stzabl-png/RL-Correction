#!/usr/bin/env bash
# L3 -> L5 交接 (用户 2026-08-17 裁定: L3 停止之后本地起 L5, L4 只在远端跑):
#   等 L3 结束 -> L5 冒烟(64 env) -> 核验横幅/无崩 -> L5 全量 + 守护/录像切换。
# 日志: ~/l5_handoff.log; 冒烟: ~/smoke_l5.log; 全量: ~/l5_pour17.log
set -uo pipefail
cd /home/lyh/Project/RL_Correction
PY=/home/lyh/luhr/MagicSim/.venv/bin/python
WARM=logs/BiRef_pour17_L3/2026-08-17_17-14-24/stage1_nn/eval_best.pth
COMMON="--clip Pour17_bottle --prior_npz tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
  --bimanual --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180 \
  --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz \
  --ff_freeze_cm 5 --ff_pull 0.08 --approach --minimal --l5 \
  --warm_start $WARM --kl_threshold 0.02 --seed 42"
log(){ echo "[$(date +%m-%d\ %H:%M)] $*"; }

log "等待 L3 结束..."
while pgrep -f "name BiRef_pour17_L3" >/dev/null 2>&1; do sleep 60; done
log "L3 已退出: $(grep -a '确定性成功率' ~/biref_pour17_L3.log | tr -d '\0' | tail -1 | cut -c1-90)"
touch "$HOME/.cache/rl_correction/nightwatch.stop"
for p in $(ps -eo pid,args | grep "autorec.sh BiRef_pour17_L3" | grep -v grep | awk '{print $1}'); do
  kill "$p" 2>/dev/null
done
sleep 15
for p in $(ps -eo pid,comm,args | awk '$2 ~ /python/ && /BiRef_pour17_L3/ {print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 12

log "L5 冒烟 (64 env, 15 万步)..."
SHARPA_WANDB=0 PYTHONPATH=. $PY -u -m tasks.pregrasp.train --headless \
  --name SMOKE_l5 --num_envs 64 --max_agent_steps 150000 $COMMON \
  > ~/smoke_l5.log 2>&1
SM_RC=$?
sleep 10
_ok=1
grep -qa "\[L5\] 最终生效" ~/smoke_l5.log || { log "❌ 冒烟缺 L5 横幅"; _ok=0; }
grep -qa "前馈已接" ~/smoke_l5.log || { log "❌ 冒烟缺前馈横幅"; _ok=0; }
grep -qa "L5 B 侧定向重建" ~/smoke_l5.log || { log "❌ 冒烟缺 B 侧定向重建"; _ok=0; }
grep -qa "逐侧搬臂行" ~/smoke_l5.log || { log "⚠ 热启动逐侧映射没打印 (14->26 分支没走到?)"; }
grep -qa "Traceback" ~/smoke_l5.log && { log "❌ 冒烟有 Traceback"; _ok=0; }
[ "$(grep -ac 'Agent Steps' ~/smoke_l5.log)" -ge 3 ] || { log "❌ 冒烟没跑起来"; _ok=0; }
if [ "$_ok" != "1" ]; then
  log "❌❌ 冒烟不通过 (rc=$SM_RC), 不起全量。守护保持停机, 等人工处理。"
  exit 1
fi
log "✅ 冒烟通过, 起 L5 全量 (1024 env, 40M)"

SHARPA_WANDB=0 setsid nohup $PY -u -m tasks.pregrasp.train --headless \
  --name L5_pour17 --num_envs 1024 --auto_stop dry --max_agent_steps 40000000 \
  $COMMON > ~/l5_pour17.log 2>&1 < /dev/null &

sed -i 's/BiRef_pour17_L3/L5_pour17/g; s/biref_pour17_L3\.log/l5_pour17.log/g' \
  tasks/pregrasp/nightwatch.sh
sed -i 's|--curobo_ref tasks/pregrasp/priors/curobo_pour17_ref.npz --ff_freeze_cm 5 --ff_pull 0.08 \\|--curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz --ff_freeze_cm 5 --ff_pull 0.08 --l5 --warm_start '"$WARM"' \\|' \
  tasks/pregrasp/nightwatch.sh
sed -i 's/--approach --approach_only --minimal/--approach --minimal/' tasks/pregrasp/nightwatch.sh
rm -f "$HOME/.cache/rl_correction/nightwatch.stop"
setsid nohup bash tasks/pregrasp/nightwatch.sh > ~/nightwatch.log 2>&1 < /dev/null &
setsid nohup bash tasks/pregrasp/autorec.sh L5_pour17 ~/l5_pour17.log \
  --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz --prior_yaw 19.5 \
  --bimanual --prior_b tasks/pregrasp/priors/Pour17_cup.npz --prior_b_yaw 180 \
  --curobo_ref tasks/pregrasp/priors/curobo_pour17_joint_ref.npz \
  --ff_freeze_cm 5 --ff_pull 0.08 --l5 --approach --minimal \
  > ~/autorec_l5.log 2>&1 < /dev/null &
log "✅ L5 全量 + 守护 + 录像 已拉起"
