#!/bin/bash
# 守夜: 每 10 分钟巡一次, 跑满 N 轮后退出(退出即唤醒我复盘)。
# 职责: ①记录各条进度 ②发现掉线/卡死 ③清僵尸 Isaac ④**不擅自重启**, 只记录
# ★必须钉死 cwd: 下面的一致性检查用的是相对路径 `find tasks/Pour/17`, 而本脚本的
#   cwd 取决于在哪儿起的它(Bash 工具的工作目录跨调用保留)。2026-08-31 实测: 从
#   A_Design 目录起的那次, find 找到 0 个文件, 于是每巡都报"三台都不一致" ——
#   **假警报比不报警更坏**, 它会让人对真警报脱敏。
cd /home/lyh/Project/RL_Correction || exit 1
ROUNDS=${1:-5}
ST=$(dirname $0)/wd_state.txt
LOG=$(dirname $0)/watchdog.log
# ★L5-33 批次: 日志名前缀不一 (seedtest_/ab3_), 直接列完整日志名
declare -A RUNS=(
  [Denso]="seedtest_s31 ab3_base_oh ab3_base_o ab3_flat_oh ab3_flat_o"
  [msc]="ab3_noise1xC ab3_noise3xC"
)
say(){ echo "$(date +%H:%M:%S) $*" >> $LOG; }
# ★状态文件必须在**本实例启动时**清空。2026-08-31 实测: 不清的话, 新实例的
#   第一巡拿来对比的是**上一个实例几十秒前**的读数(而不是 10 分钟前), 于是
#   "日志一字未增"必然成立 —— 一连报 4 条假"卡住", 而 4 条线全是好的。
#   (同一晚这已是守夜的第二个误报源, 第一个是"步数只报到整数 M"。)
: > $ST
for r in $(seq 1 $ROUNDS); do
  say "───────── 巡查 $r/$ROUNDS ─────────"
  # --- Denso / msc: AB2 线 ---
  for h in Denso msc; do
    u=$([ $h = Denso ] && echo yanghong || echo msc)
    n=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 $h "ps -u $u -o cmd= | grep -cE 'train[_]pour.py --name (AB2|AB3)'" 2>/dev/null)
    want=$(echo ${RUNS[$h]} | wc -w)
    [ -z "$n" ] && { say "  $h ★连不上"; continue; }
    for nm in ${RUNS[$h]}; do
      # ★步数只报到整数 M, 10 分钟只走约 0.4M ⟹ 光看 M 会大量误报"卡住"
      #   (2026-08-31 首轮实测: 5 巡里 13 次误报, 而线全是好的)。
      #   改成两个信号: 日志字节数=**存活**(每 epoch 都写), M=**进度**。
      #   只有**两者都不动**才算卡住。
      read cur sz <<< $(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 $h \
        "echo \$(grep -oE 'Agent Steps: [0-9]+M' ~/$nm.log 2>/dev/null | tail -1 | grep -oE '[0-9]+') \$(stat -c%s ~/$nm.log 2>/dev/null)" 2>/dev/null)
      prev=$(grep "^$h/$nm " $ST 2>/dev/null | tail -1 | awk '{print $2}')
      psz=$(grep "^$h/$nm " $ST 2>/dev/null | tail -1 | awk '{print $3}')
      if [ -z "$cur" ]; then
        say "  $h/$nm ★无步数 (日志 ${sz:-0}B) —— 建场景中或已挂"
      elif [ "$cur" = "$prev" ] && [ "$sz" = "$psz" ]; then
        say "  $h/$nm ★★卡住: ${cur}M 且日志 ${sz}B 一字未增"
      else
        say "  $h/$nm ${cur}M"
      fi
      echo "$h/$nm ${cur:-NA} ${sz:-0}" >> $ST
    done
    [ "$n" -lt "$want" ] && say "  $h ★进程 $n/$want —— **有线掉了**"
  done
  # --- 清僵尸: 非训练的 Isaac 进程跑了 >30 分钟 ---
  for h in Denso msc taitan; do
    u=$(case $h in Denso) echo yanghong;; msc) echo msc;; *) echo vision;; esac)
    z=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=20 $h "ps -u $u -o pid=,etimes=,cmd= | grep -E 'build_ref_v5|probe_|eval_pour|smoke_' | grep -v grep | awk '\$2>1800{print \$1}'" 2>/dev/null)
    for p in $z; do
      timeout 30 ssh -o BatchMode=yes $h "kill -9 $p" 2>/dev/null && say "  $h 清僵尸 pid=$p (>30分钟的非训练 Isaac)"
    done
  done
  # 本机
  for p in $(ps -u lyh -o pid=,etimes=,cmd= | grep -E 'build_ref_v5|probe_pour' | grep -v grep | awk '$2>1800{print $1}'); do
    kill -9 $p 2>/dev/null && say "  本机 清僵尸 pid=$p"
  done
  # --- ★全树一致性 (2026-08-31 血的教训): 我只同步了 pour_env.py 没同步
  #     progress_batch.py, 于是"传参的一方"更新了、"收参的一方"没有 ——
  #     taitan 两条评测跑了 50 分钟才在 TypeError 上暴露, Denso 4 条训练则是
  #     "只要不重启就没事"的定时炸弹。**永远全树同步, 别逐文件挑。**
  LOC=$(find tasks/Pour/17 -name '*.py' | grep -v __pycache__ | sort | xargs md5sum | awk '{print substr($1,1,8),$2}')
  for h in Denso msc taitan; do
    REM=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=20 $h "cd RL_Correction && find tasks/Pour/17 -name '*.py' | grep -v __pycache__ | sort | xargs md5sum | awk '{print substr(\$1,1,8),\$2}'" 2>/dev/null)
    if [ -z "$REM" ]; then say "  代码一致性/$h ★连不上"
    elif diff -q <(echo "$LOC") <(echo "$REM") >/dev/null 2>&1; then :
    else say "  代码一致性/$h ★★与本机不一致 ($(diff <(echo "$LOC") <(echo "$REM") | grep -c '^<') 个文件)"; fi
  done
  [ $r -lt $ROUNDS ] && sleep 600
done
say "守夜本轮结束 ($ROUNDS 巡)"
