#!/bin/bash
# M1 Falcon-E-3B bring-up battery — sequential, niced, one heavy job at a time
cd /home/killboxincorporated
LOG=docs/hardware_logs/falcon_e_3b_bringup_2026-07-17.log
A3=falcon-e-3b-artifacts; A1=falcon-e-artifacts
{
echo "=== M1: Falcon-E-3B-Instruct bring-up ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo "artifacts: $A3 (MODEL.SAF 864351226 B, repacked from tiiuae/Falcon-E-3B-Instruct main U8-packed, --source-packing hf1bitllm --max-seq 2048)"
echo "box: contended (aegis-ppl-rerun running niced). PPL deterministic; tok/s figures contended."
echo
echo "--- 1. coherence: aegis-linux, chatml template from config ---"
for P in "What is the capital of France?" "Why is the sky blue?"; do
  echo "PROMPT: $P"
  nice -n 8 ./aegis-linux/target/release/aegis-linux $A3/MODEL.SAF $A3/EMBED.BIN $A3/VOCAB.BIN 48 "$P" 2>&1 | tail -8
  echo
done
echo "--- 2. 3B PPL on t2d_sample (1B G4a same-text: engine 4.837 / reference 4.831 @1896 tok) ---"
nice -n 8 ./aegis-eval/target/release/aegis-eval $A3/MODEL.SAF $A3/EMBED.BIN $A3/VOCAB.BIN $A1/t2d_sample.txt 1900 --sample 2>&1 | tail -8
echo
echo "--- 3. 3B PPL on WikiText-2 sample (test.txt) ---"
nice -n 8 ./aegis-eval/target/release/aegis-eval $A3/MODEL.SAF $A3/EMBED.BIN $A3/VOCAB.BIN test.txt 1900 --sample 2>&1 | tail -8
echo
echo "--- 4. 1B PPL on WikiText-2 sample (comparison baseline, same binary/mode) ---"
nice -n 8 ./aegis-eval/target/release/aegis-eval $A1/MODEL.SAF $A1/EMBED.BIN $A1/VOCAB.BIN test.txt 1900 --sample 2>&1 | tail -8
echo
echo "=== battery complete ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
} >> $LOG 2>&1
tail -50 $LOG
