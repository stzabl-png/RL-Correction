#!/bin/bash
# Real-time monitor for MegaSAM + HaWoR dual pipeline
# Usage: bash tools/monitor_egodex.sh

MEGA_LOG="/home/lyh/Project/Reconstruct_and_Retarget/Output/Depth/MegaSAM/Egodex/batch_log.jsonl"
HAWOR_LOG="/home/lyh/Project/Reconstruct_and_Retarget/Output/MANO/HaWoR/Egodex/batch_log.jsonl"
TOTAL=3051

while true; do
    clear

    # ── Header ─────────────────────────────────────────────
    echo "══════════════════════════════════════════════════════"
    echo "  📊 EgoDex Pipeline Monitor   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════"
    echo ""

    # ── MegaSAM ────────────────────────────────────────────
    M_OK=$(grep -c '"ok"' "$MEGA_LOG" 2>/dev/null || echo 0)
    M_FAIL=$(grep -c '"error"' "$MEGA_LOG" 2>/dev/null || echo 0)
    M_REM=$((TOTAL - M_OK))
    M_PCT=$((M_OK * 100 / TOTAL))
    M_BAR_LEN=$((M_PCT / 2))
    M_BAR=$(printf '█%.0s' $(seq 1 $M_BAR_LEN 2>/dev/null) 2>/dev/null)
    M_SPC=$(printf '░%.0s' $(seq 1 $((50 - M_BAR_LEN)) 2>/dev/null) 2>/dev/null)

    echo "  🔵 MegaSAM (Depth + Intrinsic + Pose)"
    printf "     [%s%s] %3d%%\n" "$M_BAR" "$M_SPC" "$M_PCT"
    printf "     ✅ %-5s done   ❌ %-4s fail   ⏳ %-5s left\n" "$M_OK" "$M_FAIL" "$M_REM"

    M_LAST=$(tail -1 "$MEGA_LOG" 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.readline())
    s=d.get('status','?'); q=d.get('seq','?'); t=d.get('elapsed_s','?')
    icon='✅' if s=='ok' else '❌'
    print(f'     {icon} {q}  ({t}s)')
except: print('     (waiting...)')
" 2>/dev/null)
    echo "$M_LAST"
    echo ""

    # ── HaWoR ──────────────────────────────────────────────
    H_OK=$(grep -c '"ok"' "$HAWOR_LOG" 2>/dev/null || echo 0)
    H_FAIL=$(grep -c '"error"' "$HAWOR_LOG" 2>/dev/null || echo 0)
    H_REM=$((TOTAL - H_OK))
    H_PCT=$((H_OK * 100 / TOTAL))
    H_BAR_LEN=$((H_PCT / 2))
    H_BAR=$(printf '█%.0s' $(seq 1 $H_BAR_LEN 2>/dev/null) 2>/dev/null)
    H_SPC=$(printf '░%.0s' $(seq 1 $((50 - H_BAR_LEN)) 2>/dev/null) 2>/dev/null)

    echo "  🟢 HaWoR (MANO Hand Reconstruction)"
    printf "     [%s%s] %3d%%\n" "$H_BAR" "$H_SPC" "$H_PCT"
    printf "     ✅ %-5s done   ❌ %-4s fail   ⏳ %-5s left\n" "$H_OK" "$H_FAIL" "$H_REM"

    H_LAST=$(tail -1 "$HAWOR_LOG" 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.readline())
    s=d.get('status','?'); q=d.get('seq','?'); t=d.get('elapsed_s','?')
    icon='✅' if s=='ok' else '❌'
    print(f'     {icon} {q}  ({t}s)')
except: print('     (waiting...)')
" 2>/dev/null)
    echo "$H_LAST"
    echo ""

    # ── GPU ────────────────────────────────────────────────
    echo "  🖥️  GPU Status"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader 2>/dev/null | \
        awk -F", " '{
            split($2, mu, " "); split($3, mt, " ");
            pct=int(mu[1]*100/mt[1]);
            bar_len=int(pct/2);
            bar=""; for(i=0;i<bar_len;i++) bar=bar"█";
            spc=""; for(i=0;i<50-bar_len;i++) spc=spc"░";
            printf "     VRAM [%s%s] %3d%%  (%s / %s)\n", bar, spc, pct, $2, $3;
            printf "     Util: %s   Temp: %s   Power: %s\n", $1, $4, $5;
        }'
    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Press Ctrl+C to exit"

    sleep 10
done
