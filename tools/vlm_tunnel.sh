#!/usr/bin/env bash
# UCB 8 卡机上的 VLM 服务隧道。
#
# 模型跑在 169.229.192.185 的 GPU7(Qwen3.5-27B-Int4), 只监听远端 127.0.0.1:8807,
# 所以本地必须走 ssh 端口转发才能访问。
#
#   ./tools/vlm_tunnel.sh          开隧道(已开则复用) 并自检
#   ./tools/vlm_tunnel.sh --down   关掉
#   ./tools/vlm_tunnel.sh --check  只自检, 不开
#
# 开通后管线自动可用: vlm_transparency_gate 认 VLM_API_BASE / QWEN_BASE_URL,
# 两者都不设时默认就是 http://127.0.0.1:8807/v1。
set -uo pipefail

REMOTE="${VLM_REMOTE:-yanghong@169.229.192.185}"
PORT="${VLM_PORT:-8807}"
BASE="http://127.0.0.1:${PORT}/v1"

up() { curl -s -m 4 "$BASE/models" -o /dev/null -w '%{http_code}' 2>/dev/null | grep -q 200; }

case "${1:-}" in
  --down)
    pkill -f "ssh -f -N -L ${PORT}:127.0.0.1:${PORT} ${REMOTE}" && echo "[vlm] 隧道已关闭" \
      || echo "[vlm] 没有找到隧道进程"
    exit 0 ;;
  --check) ;;
  *)
    if ! up; then
      echo "[vlm] 开隧道: ssh -f -N -L ${PORT}:127.0.0.1:${PORT} ${REMOTE}"
      ssh -f -N -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE" || {
        echo "[vlm] X ssh 失败 —— 检查免密登录: ssh $REMOTE true" >&2; exit 1; }
      sleep 2
    fi ;;
esac

if up; then
  MODEL=$(curl -s -m 4 "$BASE/models" | python3 -c \
    'import sys,json;print((json.load(sys.stdin).get("data") or [{}])[0].get("id","?"))' 2>/dev/null)
  echo "[vlm] ✓ 可用  $BASE   模型=$MODEL"
  echo "[vlm]   管线无需额外设置; 要指到别处用 export VLM_API_BASE=..."
  exit 0
fi
# ★ 不可达时**明确失败**, 不要让调用方以为通了 —— 管线那边会写 status=skipped 留痕,
#   但人在终端里必须一眼看出来是"没判", 而不是"判过了都不透明"。
cat >&2 <<EOF
[vlm] X 不可达 $BASE
      1) 隧道通吗:  ssh $REMOTE true
      2) 远端服务起了吗:  ssh $REMOTE 'curl -s -m3 127.0.0.1:${PORT}/v1/models | head -c 200'
      3) 没起就在远端开:  GPU=7 bash ~/bin/start_vlm.sh
EOF
exit 1
