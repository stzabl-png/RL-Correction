#!/usr/bin/env bash
# One-command PICO -> Vega teleop launcher (tmux).
#
# Opens a tmux session with three panes so you don't juggle terminals:
#   [PC svc ] XRoboToolkit PC service
#   [produce] scripts/teleop_pico_producer.py  (.venv-pico, records to logs/)
#   [consume] sim/teleop_vega_pico.py           (.venv-isaac, Isaac + Pink IK)
#
# Usage:
#   scripts/run_teleop.sh                # tracker mode (the final rig - just run it)
#   scripts/run_teleop.sh --probe        # legacy: probe the per-tracker channel serials
#   scripts/run_teleop.sh --controllers  # legacy controller mode (bring-up comparison)
#   scripts/run_teleop.sh --headless     # no window AND no rendering at all
#   scripts/run_teleop.sh --snap         # periodic camera snapshots (costs ~2x rtf)
#   scripts/run_teleop.sh --no-service   # skip PC service (already running)
#   scripts/run_teleop.sh --fast         # the measured real-time config (see below)
#   scripts/run_teleop.sh --hands right  # ALSO run the wuji glove -> Sharpa producer
#   scripts/run_teleop.sh --hands both   # both gloves (right :5556, left :5557)
#   scripts/run_teleop.sh --hands right --glove-mock   # fake glove: test the wiring unworn
#   scripts/run_teleop.sh -- <extra consumer args...>   # anything after -- goes to the consumer
#
# WEARING THE GLOVE AND THE HEADSET AT THE SAME TIME (--hands)
#   These are TWO INDEPENDENT PIPELINES that happen to end up in one browser
#   window: different device, different producer, different venv, different
#   port. The glove path never goes through Isaac and never touches the arm
#   path - which is deliberate, because the one rule that keeps them from
#   taking each other down is: BROADCAST ONLY, NEVER WAIT. Either can be
#   started, stopped or restarted without disturbing the other; the mirror
#   draws whatever has arrived and says so when something has not.
#
#     PICO   -> teleop_pico_producer (.venv-pico) -> :5581 -> consumer (.venv-isaac)
#                                                              -> Isaac + :5583
#     glove  -> teleop_retarget      (.venv)      -> :5556 -----------------.
#                                                                          v
#                                        viser_isaac_mirror (.venv) -> :8086
#
#   The glove needs no calibration step here, but the FIRST second of a Sharpa
#   recording is tactile zeroing - keep the fingertips off everything then.
#
# No serials needed: the PICO trackers stream as the full-body 24-joint estimate
# and the producer republishes the joints we use under fixed names
# (LWRIST/RWRIST/WAIST), which are also the consumer defaults.
# scripts/trackers.env is an OPTIONAL override. Two uses: the legacy individual
# motion-tracker channel (--probe finds the serials), and the 2026-07-29 hybrid
# rig (raw dual-wrist serials + model WAIST, both channels concurrent).
#
# Welded wheels (vega_1p_weldwheels.usd) are the DEFAULT since 2026-07-29 -
# the free-spinning parked wheels corrupted the articulation solve and wrecked
# wrist tracking (76mm -> 0.9mm after welding). VEGA_WELD_WHEELS=0 reverts.
#
# Stop everything: press the shortcut printed at launch, or `tmux kill-session -t vega`.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SESSION=vega
PICO=$ROOT/.venv-pico/bin/python
ISAAC=$ROOT/.venv-isaac/bin/python
SERVICE=/opt/apps/roboticsservice/runService.sh
TRACKERS_ENV=$ROOT/scripts/trackers.env

MODE=trackers
RUN_SERVICE=1
HEADLESS=""
SNAP=""
EXTRA=""
# --fast: the configuration measured on 2026-08-02 to take arms+head+waist from
# 0.20x real time to 0.88x with NO loss of accuracy (wrist tracking 26.4/27.4mm
# -> 26.2/28.0mm on logs/clip_headwaist.msgpack). Each flag was measured on its
# own, not guessed - the guesses were all wrong:
#
#   --physics-device cpu   physics was 77% of the frame budget at 49 ms per
#                          control step. One small articulation has nothing to
#                          amortise GPU physics' fixed per-step cost over: on
#                          CPU the same step costs 14.5 ms, and everything else
#                          halves too because the GPU<->CPU syncs go away.
#                          It also fixes the head: head_j3 settled 62.8 deg off
#                          its command on GPU, 0.1 deg off on CPU.
#   --physics-hz 50        7.1 ms instead of 14.5, tracking error unchanged.
#   --control_hz 50        60 -> 50 Hz control, one physics step per control
#                          step.
#   (--pink-substeps 2 was in here and has been REMOVED: it buys 3% of rtf
#    - 0.94x to 0.97x - and costs 2x the wrist accuracy, 30.5/23.1 mm to
#    61.9/61.6 mm on logs/playlist_all13.msgpack. It measured harmless on
#    clip_headwaist at the authored torso, which is why it got in; one clip is
#    not a measurement.)
#   --no-sym-lock          the bimanual symmetriser costs 2-3.5 ms AND, under
#                          the operator's 2026-08-02 priorities, actively hurts:
#                          two-hand relative position 9.8 -> 32.5 mm, relative
#                          orientation 0.0 -> 1.4 deg, with no gain in palm or
#                          finger orientation.
#
# Head/waist are NOT the cost: arms alone measured 0.21x, arms+head+waist 0.20x.
FAST=""
HANDS=none           # none | right | left | both  - the wuji glove -> Sharpa line
GLOVE_SRC=wuji       # --glove-mock swaps this for the synthetic glove
_WANT_HANDS=0
for arg in "$@"; do
  # `--hands right` arrives as two iterations of this loop, so remember that the
  # previous one asked for a value. `--hands=right` also works.
  if [ "$_WANT_HANDS" = 1 ]; then HANDS="$arg"; _WANT_HANDS=0; shift || true; continue; fi
  case "$arg" in
    --hands)       _WANT_HANDS=1 ;;
    --hands=*)     HANDS="${arg#*=}" ;;
    --glove-mock)  GLOVE_SRC=mock ;;
    --probe)       MODE=probe ;;
    --controllers) MODE=controllers ;;
    # --headless used to bundle --snap-dir, and --snap-dir forces
    # --enable_cameras: asking for "no window" therefore switched the CAMERA
    # renderer ON. Measured live 2026-08-03: rtf 0.37x with it, 0.85x without,
    # on a run whose whole point was that it should not be rendering anything.
    # Snapshots are now their own flag.
    --headless)    HEADLESS="--headless" ;;
    --snap)        SNAP="--snap-dir $ROOT/logs/snaps" ;;
    --no-service)  RUN_SERVICE=0 ;;
    --fast)        FAST="--physics-device cpu --physics-hz 50 --control_hz 50 --no-sym-lock" ;;
    --) shift; EXTRA="$*"; break ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
  shift || true
done

[ -x "$PICO" ]  || { echo "missing $PICO (run scripts/setup_pico_env.sh)"; exit 1; }
[ -x "$ISAAC" ] || { echo "missing $ISAAC (run scripts/setup_isaac_env.sh)"; exit 1; }
mkdir -p "$ROOT/logs"

# tracker names (tracker mode only). Defaults = the fixed names the producer
# synthesizes from the full-body joints; trackers.env optionally overrides
# them (legacy individual-tracker channel with real hardware serials).
TRK_ARGS=""
if [ "$MODE" = trackers ]; then
  TRACKER_LEFT=LWRIST
  TRACKER_RIGHT=RWRIST
  TRACKER_WAIST=WAIST
  if [ -f "$TRACKERS_ENV" ]; then
    # shellcheck disable=SC1090
    source "$TRACKERS_ENV"
    echo "[run_teleop] tracker names overridden by $TRACKERS_ENV"
  fi
  TRK_ARGS="--mode trackers --tracker-left $TRACKER_LEFT \
    --tracker-right $TRACKER_RIGHT --tracker-waist $TRACKER_WAIST"
elif [ "$MODE" = controllers ]; then
  TRK_ARGS="--mode controllers"   # consumer defaults to trackers now
fi

# CUDA health, before tmux swallows the traceback in a pane nobody is looking
# at. Killing a long-running Isaac process corrupts nvidia_uvm: nvidia-smi
# stays perfectly healthy and CUDA reports "unknown error", so the failure
# looks like a code problem for as long as it takes to remember. Four times
# now (2026-07-31, 08-01, 08-02 x2). One line, zero risk, prints the fix.
if ! $ISAAC -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
     >/dev/null 2>&1; then
  echo "[run_teleop] CUDA is not available - almost certainly the nvidia_uvm"
  echo "[run_teleop] corruption (nvidia-smi will look fine). Fix and re-run:"
  echo
  echo "    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
  echo
  exit 1
fi

REC="$ROOT/logs/pico_$(date +%Y%m%d_%H%M%S).msgpack"
PRODUCER_CMD="$PICO scripts/teleop_pico_producer.py --body-full --record $REC"
CONSUMER_CMD="OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 VEGA_WELD_WHEELS=${VEGA_WELD_WHEELS:-1} $ISAAC -u sim/teleop_vega_pico.py --source zmq $TRK_ARGS $HEADLESS $SNAP $FAST $EXTRA"
PROBE_CMD="$PICO scripts/probe_trackers.py"
VIEWER_CMD="$PICO scripts/view_skeleton.py"
# The hands-on view of what the ROBOT is doing. It mirrors the joint angles
# Isaac actually has (published on --pub-state) and computes nothing, so it
# cannot disagree with the consumer. viser_mapping_preview.py re-derives the
# mapping and the IK itself, and on 2026-08-02 that was measured landing the
# right arm on a different IK branch than Isaac - R_arm_j1 +176 deg there
# against +1.7 deg here - so it is for offline exploration, not for judging a
# live run. Look at THIS one while wearing the headset.
# --- the glove line. Independent of everything above: its own device, its own
# venv, its own port, and no handshake with the arm side in either direction.
# Nothing here may block or be blocked by the arm pipeline.
# --pinch-weight 20 --relax-distal are the SETTLED parameters (SOP.md 209) -
# not tuning knobs to fiddle with per session.
GLOVE_CMD=""
MIRROR_HANDS="--hand-right ''"      # no glove -> no hand readout, no false alarm
case "$HANDS" in
  none) ;;
  right)
    GLOVE_CMD="PYTHONPATH= $ROOT/.venv/bin/python scripts/teleop_retarget.py \
--source $GLOVE_SRC --hand right --pinch-weight 20 --relax-distal"
    MIRROR_HANDS="--hand-right tcp://127.0.0.1:5556" ;;
  left)
    # single-hand producer binds :5556 whichever hand it is; move it to :5557 so
    # left always means the same port, dual or not.
    GLOVE_CMD="PYTHONPATH= $ROOT/.venv/bin/python scripts/teleop_retarget.py \
--source $GLOVE_SRC --hand left --pinch-weight 20 --relax-distal --pub tcp://*:5557"
    MIRROR_HANDS="--hand-right '' --hand-left tcp://127.0.0.1:5557" ;;
  both)
    GLOVE_CMD="PYTHONPATH= $ROOT/.venv/bin/python scripts/teleop_retarget_dual.py \
--source $GLOVE_SRC --pinch-weight 20 --relax-distal"
    MIRROR_HANDS="--hand-right tcp://127.0.0.1:5556 --hand-left tcp://127.0.0.1:5557" ;;
  *) echo "unknown --hands '$HANDS' (want: none|right|left|both)"; exit 2 ;;
esac
if [ "$HANDS" != none ] && [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "--hands needs $ROOT/.venv (the retarget venv)"; exit 1
fi

MIRROR_CMD="$ROOT/.venv/bin/python scripts/viser_isaac_mirror.py $MIRROR_HANDS"

tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "session '$SESSION' already running. attach: tmux attach -t $SESSION"
  echo "or kill it: tmux kill-session -t $SESSION"; exit 0; }

tmux new-session -d -s "$SESSION" -n teleop -c "$ROOT"

# pane 0: PC service (or a note if skipped)
if [ "$RUN_SERVICE" = 1 ] && [ -f "$SERVICE" ]; then
  tmux send-keys -t "$SESSION":teleop.0 "echo '[PC service]'; bash $SERVICE" C-m
else
  tmux send-keys -t "$SESSION":teleop.0 "echo '[PC service] skipped (--no-service or not installed)'" C-m
fi

# pane 1: producer
tmux split-window -v -t "$SESSION":teleop -c "$ROOT"
tmux send-keys -t "$SESSION":teleop.1 "echo '[producer] recording -> $REC'; sleep 2; $PRODUCER_CMD" C-m

# pane 2: consumer OR probe
tmux split-window -h -t "$SESSION":teleop.1 -c "$ROOT"
if [ "$MODE" = probe ]; then
  tmux send-keys -t "$SESSION":teleop.2 "echo '[probe] wave one limb at a time; note the MOVING serials'; sleep 3; $PROBE_CMD" C-m
else
  tmux send-keys -t "$SESSION":teleop.2 "echo '[consumer] mode=$MODE  (waiting 6s for producer)'; sleep 6; $CONSUMER_CMD" C-m
fi

# pane 3: skeleton viewer - pops a window showing what the PICO body estimate
# thinks the operator is doing (green = live, RED = frozen). Needs a display;
# skipped when headless was requested.
if [ -z "$HEADLESS" ] && [ "$MODE" != probe ]; then
  tmux split-window -v -t "$SESSION":teleop.2 -c "$ROOT"
  tmux send-keys -t "$SESSION":teleop.3 "echo '[viewer] skeleton window opening (green=live, red=frozen)'; sleep 8; $VIEWER_CMD" C-m
fi

# pane 4: the robot-side mirror (browser, :8086). No GPU, no mapping, no IK.
if [ "$MODE" != probe ] && [ -x "$ROOT/.venv/bin/python" ]; then
  # -P -F '#{pane_id}' returns the pane that was just CREATED. Deriving it
  # instead from `list-panes | tail -1` is wrong and was: pane indices are
  # ordered by position, not by creation, so after the skeleton-viewer split
  # the "last" index pointed back at the viewer's pane and both commands got
  # typed into it - the mirror sat queued behind the viewer and never started,
  # while the pane it should have used sat idle at a shell prompt.
  MIRROR_PANE=$(tmux split-window -v -P -F '#{pane_id}' -t "$SESSION":teleop.2 -c "$ROOT")
  tmux send-keys -t "$MIRROR_PANE" \
    "echo '[mirror] robot view -> http://localhost:8086 (mirrors Isaac, computes nothing)'; sleep 12; $MIRROR_CMD" C-m
fi

# pane 5: the glove producer. Started WITHOUT waiting for anything - the two
# pipelines are independent and coupling their startup would be the first step
# toward coupling their failures.
if [ -n "$GLOVE_CMD" ]; then
  GLOVE_PANE=$(tmux split-window -v -P -F '#{pane_id}' -t "$SESSION":teleop.1 -c "$ROOT")
  # RETRY, do not fire once. fj_work_claude.md's run section states the
  # prerequisite plainly: "先接好手套、再起终端 B" - the producer needs a glove
  # that is already up, and connect() raises "Device not found" if it is not.
  # Launching it in the same breath as the arm stack broke that rule and the
  # producer died at t=0, leaving :5556 with nobody on it. Retrying keeps the
  # documented ordering without making the operator time it: bring the glove up
  # whenever, this pane picks it up. Exit 130 (Ctrl-C) means you meant to stop.
  tmux send-keys -t "$GLOVE_PANE" \
    "echo '[glove] $HANDS  source=$GLOVE_SRC  -> zmq (mirror :8086)'; \
     echo '[glove] 前提:手套已标定并佩戴、已通电、已接入网络(右 192.168.1.101,左 192.168.1.100)'; \
     echo '[glove] 标定用 SharpaPilot 完成,标定后需将其关闭,否则会占用手套接口'; \
     while :; do $GLOVE_CMD; rc=\$?; case \$rc in 0|130) break;; esac; \
       echo \"[glove] 手套程序已退出(返回码 \$rc),可能未通电、未标定或未接入网络,3 秒后重试。\"; \
       echo '[glove] 手臂一侧不受影响。可单独自检:PYTHONPATH= .venv/bin/python scripts/diag_glove.py --hand $HANDS --duration 8'; \
       sleep 3; done" C-m
fi

tmux select-layout -t "$SESSION":teleop tiled >/dev/null

cat <<EOF

  teleop session '$SESSION' launched (mode=$MODE, hands=$HANDS).
    attach : tmux attach -t $SESSION       (Ctrl-b then arrow keys to switch panes)
    stop   : tmux kill-session -t $SESSION
    logs   : $REC
    页面   : http://localhost:8086         戴着头显时观察的就是这个页面
$( [ "$HANDS" != none ] && echo "    手套   : $HANDS(来源 $GLOVE_SRC)。页面中 'Sharpa 手' 一栏应显示 ✅ 与帧数。
             显示 ⚠ 无数据时,表示手套一侧尚未启动,查看 [glove] 窗口即可,
             与手臂无关,不需要重启整个会话。" )

EOF
tmux attach -t "$SESSION"
