#!/bin/bash
# Stop fable-hand Chromium if idle (no active tab / all about:blank) for IDLE_MIN minutes.
# Run from cron every 15min. Safe: only kills the fable-hand profile's chromium, not user Chrome.
set -euo pipefail
IDLE_MIN="${IDLE_MIN:-30}"
PROFILE="/home/justinbrianthompson/.fable-hand/chrome-profile"
PID=$(pgrep -f "user-data-dir=$PROFILE" -o || true)
[ -z "$PID" ] && exit 0

# Consider idle if devtools reports only about:blank / no active target.
ACTIVE=$(curl -s --max-time 2 http://localhost:9222/json 2>/dev/null | grep -c '"url":"http' || true)
if [ "$ACTIVE" -eq 0 ]; then
  START=$(ps -o lstart= -p "$PID" 2>/dev/null || true)
  [ -z "$START" ] && exit 0
  START_EPOCH=$(date -d "$START" +%s)
  NOW_EPOCH=$(date +%s)
  IDLE_SEC=$(( NOW_EPOCH - START_EPOCH ))
  if [ "$IDLE_SEC" -ge $(( IDLE_MIN * 60 )) ]; then
    pkill -f "user-data-dir=$PROFILE" || true
    echo "$(date '+%Y-%m-%d %H:%M')  fable-hand-idle-check  killed idle Chromium (idle ${IDLE_SEC}s)" >> /home/justinbrianthompson/projects/claudius-maximus/state/LOG.md
  fi
fi
