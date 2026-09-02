#!/usr/bin/env bash
# Unscrew/17 发射前把关链 (顺序不可乱): smoke_zero -> probe_pull -> build_reference v2 -> probe_acceptance
# 每级完成后点名验尸 (Isaac 常卡在 app.close), 失败即停. 日志: launch_logs/u17_stage_*.log
set -uo pipefail
cd "$(dirname "$0")/../../../.."
PY=${PY:-/home/lyh/luhr/MagicSim/.venv/bin/python}
export UNSCREW_CLIP=17 UNSCREW_DETACH=${UNSCREW_DETACH:-twist} UNSCREW_RIGHT_CL=${UNSCREW_RIGHT_CL:-1} SHARPA_WANDB=0 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=. TMPDIR=${TMPDIR:-$HOME/tmp}
mkdir -p launch_logs "$TMPDIR"
D=tasks/Unscrew/part4
reap() { P=$(ps -u $USER -o pid=,comm=,args= | awk -v pat="$1" '$2 ~ /^python/ && $0 ~ pat {print $1}'); [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 10; }
stage() { # name script pass_regex extra_args...
  local name=$1 script=$2 pass=$3; shift 3
  echo "[chain] === $name ===" ; local t0=$(date +%s)
  timeout ${STAGE_TIMEOUT:-2700} $PY -u $script "$@" --headless > launch_logs/u17_stage_${name}.log 2>&1 &
  local wp=$!; local done_pat="${DONE_PAT:-全部通过|❌ 失败|已写|★训练稳定性|机器段死线误触}"
  while kill -0 $wp 2>/dev/null; do if grep -qE "$done_pat" launch_logs/u17_stage_${name}.log; then sleep 25; break; fi; sleep 10; done
  reap "$(basename $script)"; wait $wp 2>/dev/null; local rc=$?
  echo "[chain] $name rc=$rc ($(( $(date +%s) - t0 ))s)"
  if ! grep -q "$pass" launch_logs/u17_stage_${name}.log; then echo "[chain] ❌ $name 未见通过标记 '$pass'"; grep -n "FAIL\|Traceback\|Error\|❌\|铁则\|误触" launch_logs/u17_stage_${name}.log | tail -8; return 1; fi
  echo "[chain] ✅ $name"; return 0
}
if [ "${START_FROM:-smoke}" = smoke ]; then
stage smoke "$D/C_Wiring/smoke_zero.py" "机器段死线误触 = 0" --steps 600 || exit 1
# 2026-09-01 硬门: 缝1 零动作合拢把瓶推倒 (D2) 也阻塞 —— 三发证明策略救不回 "站位前瓶已倒"
if ! grep -q "缝1 交接段死线 = 0" launch_logs/u17_stage_smoke.log; then echo "[chain] ❌ smoke: 缝1 交接段零动作死线 ≠ 0 (硬门): $(grep -o '缝1 交接段死线 = [0-9]*' launch_logs/u17_stage_smoke.log)"; grep -n "缝1 交接段死线 env" launch_logs/u17_stage_smoke.log | head -4; exit 1; fi
grep -n "静置对账\|A 静置\|死线\|G链\|缝1" launch_logs/u17_stage_smoke.log | tail -12
fi
# U10 (2026-09-02): 默认 twist 30° -> 螺纹副探针; pull 变体仍走 probe_pull
if [ "${UNSCREW_DETACH:-twist}" = pull ]; then _TPROBE=probe_pull.py; else _TPROBE=probe_thread.py; fi
case "${START_FROM:-smoke}" in smoke|pull) POUR_SQUEEZE_FF=1 stage pull "$D/B_SmokeTest/$_TPROBE" "全部通过" || exit 1;; esac
case "${START_FROM:-smoke}" in smoke|pull|v2) stage v2 "$D/B_SmokeTest/build_reference.py" "已写" || exit 1; grep -n "\[v2\]" launch_logs/u17_stage_v2.log | tail -6;; esac
stage accept "$D/B_SmokeTest/probe_acceptance.py" "=> ✅" || exit 1
grep -n "★训练稳定性\|可训练性\|告警\|站位垫\|回放" launch_logs/u17_stage_accept.log | tail -10
echo "[chain] ★ 全链通过, 可发射"
