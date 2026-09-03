#!/bin/bash
set -uo pipefail
REPO=/home/justinbrianthompson/projects/alice-aegis-ca1
OUT=$REPO/scratch_ca1
LOG=$REPO/docs/ca1-ladder.log
RESULT=$REPO/docs/ca1-RESULT.txt
mkdir -p "$OUT" "$REPO/docs"
: > "$LOG"
: > "$RESULT"

EVAL="$REPO/aegis-eval/target/release/aegis-eval"
EMBED=/home/justinbrianthompson/projects/alice-aegis/falcon_e_1b_embed.bin
VOCAB=/home/justinbrianthompson/projects/alice-aegis/falcon_e_1b_vocab.bin
TXT=/home/justinbrianthompson/projects/alice-aegis/test.txt
BASE=/home/justinbrianthompson/projects/alice-aegis/falcon_e_1b_model.safetensors

FAIL=0

echo "=== CA1 driver start $(date -u) ===" >> "$LOG"

if [ ! -x "$EVAL" ]; then
  echo "building aegis-eval release..." >> "$LOG"
  (cd "$REPO/aegis-eval" && /home/justinbrianthompson/.cargo/bin/cargo build --release) >> "$LOG" 2>&1
  if [ ! -x "$EVAL" ]; then
    echo "FAIL: aegis-eval did not build" > "$RESULT"
    echo "=== CA1 driver end (build failed) $(date -u) ===" >> "$LOG"
    exit 1
  fi
fi

echo "=== ptq_ladder.py ===" >> "$LOG"
"$REPO/.venv-ca1/bin/python" "$REPO/scripts/ptq_ladder.py" "$OUT" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "FAIL: ptq_ladder.py" > "$RESULT"
  echo "=== CA1 driver end (ladder script failed) $(date -u) ===" >> "$LOG"
  exit 1
fi

echo "=== baseline float PPL ===" >> "$LOG"
"$EVAL" "$BASE" "$EMBED" "$VOCAB" "$TXT" 200 --sample >> "$LOG" 2>&1
[ $? -ne 0 ] && FAIL=1

for name in int8_w int4_w int3_w int2_w ternary_w; do
  M="$OUT/falcon_e_1b_model.${name}.safetensors"
  echo "=== $name float PPL ===" >> "$LOG"
  "$EVAL" "$M" "$EMBED" "$VOCAB" "$TXT" 200 --sample >> "$LOG" 2>&1
  [ $? -ne 0 ] && FAIL=1
  rm -f "$M"
done

echo "=== CA1 driver end $(date -u) ===" >> "$LOG"
if [ "$FAIL" -ne 0 ]; then
  echo "FAIL: one or more aegis-eval runs failed, see log" > "$RESULT"
  exit 1
fi
echo "DONE" > "$RESULT"
