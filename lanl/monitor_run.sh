#!/bin/bash
# Background monitor: checks every 3 min, logs status to results/monitor_log.txt
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT_DIR/results/run_log.txt"
MLOG="$ROOT_DIR/results/monitor_log.txt"

while true; do
    TS=$(date '+%H:%M:%S')
    ALIVE=$(tmux list-sessions 2>/dev/null | grep -c rad_run || echo 0)
    DONE=$(grep -c "AUROC=" $LOG 2>/dev/null || echo 0)
    FAILED=$(grep -c "FAILED:" $LOG 2>/dev/null || echo 0)
    LAST=$(tail -3 $LOG 2>/dev/null | tr '\n' ' ')
    echo "$TS  sessions=$ALIVE  completed=$DONE  failed=$FAILED  last: $LAST" >> $MLOG

    if [[ "$ALIVE" -eq 0 ]]; then
        echo "$TS  [MONITOR] rad_run session DIED — check $LOG" >> $MLOG
    fi
    sleep 180
done
