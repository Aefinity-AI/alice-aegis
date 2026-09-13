#!/bin/bash
set -e
cd "$(dirname "$0")/aegis-linux"
BIN=target/release/examples/cis_bitflip_moat3b
MODEL=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/MODEL.SAF
EMBED=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/EMBED.BIN
VOCAB=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/amdahl-links/VOCAB.BIN
PROMPT="Once upon a time in a small village, there lived a young girl named Luna. Luna was always curious and eager to explore and learn new things. She was always determined to understand the mysteries of the world around her."
OUT=../docs/hardware_logs/moat3b_stratified_bitnet2b_aefinity-box_2026-09-13.log
MAXNEW=64

echo "moat-3b stratified experiment -- BitNet-2B, prompt fixed, max_new=$MAXNEW, host aefinity-box, 2026-09-13" > "$OUT"
echo "byte->trit LUT verified from aegis-core/src/ops.rs build_unpack_lut(): 2 bits/weight LSB-first per byte, 00=0.0 01=+1.0 10=-1.0 11=undefined(decodes 0.0)" >> "$OUT"
echo "identity/correctness experiment ONLY -- no timing reported (Rule A)" >> "$OUT"
echo >> "$OUT"

echo "=== CONTROL (n=5 no-change) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB CONTROL - 0 0 $MAXNEW "$PROMPT" 5 100 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 1. EMBED-bf16 (n=50, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB EMBED - 0 2 $MAXNEW "$PROMPT" 50 101 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 2. SCALE-attn (n=40, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_ATTN ../tools/moat3b/scale_attn.csv 0 2 $MAXNEW "$PROMPT" 40 102 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 3. SCALE-ffn (n=40, bits 0-2) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_FFN ../tools/moat3b/scale_ffn.csv 0 2 $MAXNEW "$PROMPT" 40 103 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 4. TRIT-attn (n=35, neighbour-code remap) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB TRIT_ATTN ../tools/moat3b/trit_attn.csv 0 0 $MAXNEW "$PROMPT" 35 104 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== 5. TRIT-ffn (n=35, neighbour-code remap) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB TRIT_FFN ../tools/moat3b/trit_ffn.csv 0 0 $MAXNEW "$PROMPT" 35 105 >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== bit-position sweep, bits 3-5, 10 trials/bit, class 1 (EMBED) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB EMBED - 3 5 $MAXNEW "$PROMPT" 10 201 --sweep >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== bit-position sweep, bits 3-5, 10 trials/bit, class 2 (SCALE-attn) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_ATTN ../tools/moat3b/scale_attn.csv 3 5 $MAXNEW "$PROMPT" 10 202 --sweep >> "$OUT" 2>&1
echo >> "$OUT"

echo "=== bit-position sweep, bits 3-5, 10 trials/bit, class 3 (SCALE-ffn) ===" >> "$OUT"
$BIN $MODEL $EMBED $VOCAB SCALE_FFN ../tools/moat3b/scale_ffn.csv 3 5 $MAXNEW "$PROMPT" 10 203 --sweep >> "$OUT" 2>&1
echo >> "$OUT"

echo "ALL DONE" >> "$OUT"
