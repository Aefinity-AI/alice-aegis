#!/usr/bin/env bash
# bb-1 runner (PREREG docs/hardware_logs/bb1_PREREGISTRATION_2026-10-28.md). NOT yet executed.
# Usage: bb1_run.sh <OUTDIR> ; launch under systemd-run on w2 only (idle host).
# Env: ALICE_BIN (bench_decode), ALICE_ASSETS (dir with MODEL.SAF EMBED.BIN VOCAB.BIN),
#      BITNET_DIR (bitnet.cpp build @0b341e5), GGUF, GGUF_SHA256 (full pinned hash),
#      ALICE_REPO (git checkout), BOOTS=5 RUNS=8 (1 warm-up discarded + 7 counted), NGEN=64
set -euo pipefail
OUT=${1:?outdir}; mkdir -p "$OUT"
: "${ALICE_BIN:?}" "${ALICE_ASSETS:?}" "${BITNET_DIR:?}" "${GGUF:?}" "${GGUF_SHA256:?}" "${ALICE_REPO:?}"
BOOTS=${BOOTS:-5}; RUNS=${RUNS:-8}; NGEN=${NGEN:-64}
PF="$(cd "$(dirname "$0")" && pwd)/bb1_prompt.txt"   # fixed prompt, committed (sha256 in manifest)
PROMPT=$(cat "$PF")
PIN=${PIN_CPUS:-0-3}
[ "$(sha256sum "$GGUF" | cut -d' ' -f1)" = "$GGUF_SHA256" ] || { echo "GGUF hash mismatch" >&2; exit 3; }
case "$GGUF_SHA256" in 4221b252*) ;; *) echo "GGUF is not the pinned 4221b252 model" >&2; exit 3;; esac
cp "$PF" "$OUT/prompt.txt"
sample() { # $1 label
  { echo "## $1 $(date -u +%FT%TZ)"; cat /proc/loadavg
    grep MHz /proc/cpuinfo | tr -s ' \t' ' '
    for f in /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count; do echo "$f $(cat "$f" 2>/dev/null)"; done
    echo "governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
  } >> "$OUT/sys_samples.log"; }
cat > "$OUT/manifest.json" <<JSON
{"date_utc":"$(date -u +%FT%TZ)","host":"$(hostname)","kernel":"$(uname -r)",
 "alice_commit":"$(git -C "$ALICE_REPO" rev-parse HEAD)",
 "bitnet_commit":"$(git -C "$BITNET_DIR" rev-parse HEAD 2>/dev/null || echo unknown)",
 "gguf":"$GGUF","gguf_sha256":"$GGUF_SHA256",
 "alice_assets_sha256":{"MODEL.SAF":"$(sha256sum "$ALICE_ASSETS/MODEL.SAF"|cut -d' ' -f1)","EMBED.BIN":"$(sha256sum "$ALICE_ASSETS/EMBED.BIN"|cut -d' ' -f1)","VOCAB.BIN":"$(sha256sum "$ALICE_ASSETS/VOCAB.BIN"|cut -d' ' -f1)"},
 "prompt_sha256":"$(sha256sum "$PF"|cut -d' ' -f1)",
 "prompt":$(printf '%s' "$PROMPT" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))'),
 "n_gen":$NGEN,"boots":$BOOTS,"runs_per_boot":$RUNS,"cpu_pin":"$PIN","threads":[1,4]}
JSON
run_A() { # threads log
  taskset -c "$PIN" "$ALICE_BIN" "$ALICE_ASSETS/MODEL.SAF" "$ALICE_ASSETS/EMBED.BIN" "$ALICE_ASSETS/VOCAB.BIN" \
    --threads "$1" --n-gen "$NGEN" --prompt-file "$OUT/prompt.txt" > "$2" 2>&1; }
run_B() {
  local s e; s=$(date +%s.%N)
  LD_LIBRARY_PATH="$BITNET_DIR/build-bb1/bin" taskset -c "$PIN" "$BITNET_DIR/build-bb1/bin/llama-completion" -m "$GGUF" -p "$PROMPT" -n "$NGEN" -t "$1" --temp 0 --ignore-eos -no-cnv > "$2.out" 2> "$2.err" || true
  e=$(date +%s.%N)
  { echo arm=B; echo threads="$1"; echo outer_wall_s="$(echo "$e - $s" | bc -l)"; grep -E "eval time|sample time|load time" "$2.err" | sed 's/^/bitnet_timing: /'; } > "$2"; }
# Work-identity gate (PREREG s.3), automated. A: prompt_tokens, n_gen_emitted. B: perf lines
# "prompt eval time = .. / N tokens", "eval time = .. / M runs" (M = N_gen-1, first token comes from prefill).
REF_PT=""; GATE="$OUT/gate.log"
gate() { # $1 arm $2 log ; abort on mismatch
  local pt ng
  if [ "$1" = A ]; then pt=$(sed -n 's/^prompt_tokens=//p' "$2"); ng=$(sed -n 's/^n_gen_emitted=//p' "$2")
  else pt=$(sed -n 's/.*prompt eval time.*\/ *\([0-9]*\) tokens.*/\1/p' "$2.err" | head -1)
       ng=$(sed -n 's/.*[^t] eval time.*\/ *\([0-9]*\) runs.*/\1/p' "$2.err" | grep -v '^$' | tail -1); [ -n "$ng" ] && ng=$((ng+1)); fi
  echo "$(basename "$2") arm=$1 prompt_tokens=${pt:-NA} n_gen=${ng:-NA}" >> "$GATE"
  [ -n "$REF_PT" ] || REF_PT=$pt
  if [ -z "$pt" ] || [ "$pt" != "$REF_PT" ] || [ "$ng" != "$NGEN" ]; then
    echo "GATE FAIL: work identity ($1 pt=${pt:-NA} ref=$REF_PT n_gen=${ng:-NA} want $NGEN) $2" | tee "$OUT/GATE_FAIL" >&2; exit 4; fi; }
for T in 1 4; do
  for ((b=1; b<=BOOTS; b++)); do
    for arm in A B; do          # interleaved A,B,A,B...
      sample "pre-boot arm=$arm T=$T boot=$b"
      for ((r=0; r<RUNS; r++)); do
        tag=$([ $r -eq 0 ] && echo warmup || echo counted)
        log="$OUT/${arm}_T${T}_boot${b}_run${r}_${tag}.log"
        sample "pre-run $(basename "$log")"
        if [ $arm = A ]; then run_A "$T" "$log"; else run_B "$T" "$log"; fi
        sample "post-run $(basename "$log")"
        gate "$arm" "$log"
      done
    done
  done
done
echo done > "$OUT/DONE"
