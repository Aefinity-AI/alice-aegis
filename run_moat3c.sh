#!/bin/bash
set -e
cd "$(dirname "$0")/aegis-linux"
BIN=target/release/examples/cis_bitflip_moat3b
MODEL=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/MODEL.SAF
EMBED=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/EMBED.BIN
VOCAB=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/VOCAB.BIN
PROMPT="Once upon a time in a small village, there lived a young girl named Luna. Luna was always curious and eager to explore and learn new things. She was always determined to understand the mysteries of the world around her."
OUT=../docs/hardware_logs/moat3c_stratified_bitnet2b_aefinity-box1_2026-09-15.log
MAXNEW=64

# moat-3c: extension of moat-3b (state/reports/2026-09-13-moat3b-stratified-box1.md)
# to N~=400 new trials, seed range 200..399 (non-overlapping with moat-3b's
# 100-105/201-203 seeds). Same 5-class stratification proportions as moat-3b's
# main run, doubled per-class counts (CONTROL 5->10, EMBED 50->100,
# SCALE_ATTN 40->80, SCALE_FFN 40->80, TRIT_ATTN 35->70, TRIT_FFN 35->70;
# total 410, closest doubling to the requested "~400"). No bit-position sweep
# leg this time (task scope = main stratified extension only).

echo "moat-3c stratified extension -- BitNet-2B, prompt fixed, max_new=$MAXNEW, host aefinity-box1, 2026-09-15, N~=410 (doubled per-class vs moat-3b main run), seeds 200-205 (new, non-overlapping with moat-3b's 100-105)" > "$OUT"
echo "byte->trit LUT verified from aegis-core/src/ops.rs build_unpack_lut(): 2 bits/weight LSB-first per byte, 00=0.0 01=+1.0 10=-1.0 11=undefined(decodes 0.0)" >> "$OUT"
echo "identity/correctness experiment ONLY -- no timing reported (Rule A)" >> "$OUT"
echo >> "$OUT"

echo "=== CONTROL (n=10 no-change) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB CONTROL - 0 0 $MAXNEW "$PROMPT" 10 200 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 1. EMBED-bf16 (n=100, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB EMBED - 0 2 $MAXNEW "$PROMPT" 100 201 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 2. SCALE-attn (n=80, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_ATTN ../tools/moat3b/scale_attn.csv 0 2 $MAXNEW "$PROMPT" 80 202 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 3. SCALE-ffn (n=80, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_FFN ../tools/moat3b/scale_ffn.csv 0 2 $MAXNEW "$PROMPT" 80 203 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 4. TRIT-attn (n=70, neighbour-code remap) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB TRIT_ATTN ../tools/moat3b/trit_attn.csv 0 0 $MAXNEW "$PROMPT" 70 204 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 5. TRIT-ffn (n=70, neighbour-code remap) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB TRIT_FFN ../tools/moat3b/trit_ffn.csv 0 0 $MAXNEW "$PROMPT" 70 205 >> "$OUT" 2>&1
echo >> "$OUT"

echo "ALL DONE" >> "$OUT"
