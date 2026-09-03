#!/usr/bin/env bash
# Leg E-A gate: prompt-lookup speculative decoding vs sequential greedy decode
# on BitNet-2B and Falcon-E-1B, over the two fixed corpora in prompts/.
#
# Heavy lane discipline (2026-09-02 lab plan): every model run holds
# /tmp/aefinity-lab-heavy.lock and runs inside a systemd scope capped at
# CPUQuota=60% / MemoryMax=1800M, one at a time.
#
# Rule A: the harness prints COUNTS ONLY (passes, drafted, accepted,
# committed). Nothing here times anything and no rate may be derived from it.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${ART:-/home/justinbrianthompson/projects/alice-aegis}"
BIN="$REPO/aegis-linux/target/release/specdecode"
LOGDIR="$REPO/docs/hardware_logs"
DATE="${DATE:-$(date +%Y-%m-%d)}"
HOST_TAG="i5-10210U_crosvm"
MAX_NEW="${MAX_NEW:-32}"
KSET="${KSET:-2,4,8}"

run_one() {
    local model="$1" embed="$2" vocab="$3" corpus="$4" tag="$5"
    local log="$LOGDIR/ea_specdecode_${tag}_${HOST_TAG}_${DATE}.log"
    if [ -e "$log" ]; then
        echo "REFUSING to overwrite $log (hardware logs are append-only, Rule C)" >&2
        return 1
    fi
    local avail
    avail=$(free -m | awk '/^Mem:/ {print $7}')
    echo "[$(date -Is)] $tag: available RAM ${avail} MB" >&2
    {
        echo "# leg E-A — prompt-lookup speculative decode gate"
        echo "# host: penguin (i5-10210U, Comet Lake, crosvm) — Rule A: counts only, no timing"
        echo "# scope: systemd --user CPUQuota=60% MemoryMax=1800M, heavy lane lock"
        echo "# date: $DATE"
        echo "# corpus: $corpus"
        echo "# available RAM at start (MB): $avail"
        echo "# engine commit: $(git -C "$REPO" rev-parse HEAD)"
        echo
        flock /tmp/aefinity-lab-heavy.lock \
            systemd-run --user --scope --quiet \
                -p CPUQuota=60% -p MemoryMax=1800M -- \
                "$BIN" "$model" "$embed" "$vocab" "$corpus" "$MAX_NEW" "$KSET" "$tag"
        echo "# harness exit: $?"
    } > "$log" 2>&1
    echo "[$(date -Is)] $tag: wrote $log" >&2
}

run_one "$ART/aegis_pruned_model.safetensors" "$ART/aegis-forge/embed.bin" \
        "$ART/aegis-forge/vocab.bin" "$REPO/prompts/ea_natural.txt" "bitnet2b_natural"
run_one "$ART/aegis_pruned_model.safetensors" "$ART/aegis-forge/embed.bin" \
        "$ART/aegis-forge/vocab.bin" "$REPO/prompts/ea_repetitive.txt" "bitnet2b_repetitive"
run_one "$ART/falcon_e_1b_model.safetensors" "$ART/falcon_e_1b_embed.bin" \
        "$ART/falcon_e_1b_vocab.bin" "$REPO/prompts/ea_natural.txt" "falcone1b_natural"
run_one "$ART/falcon_e_1b_model.safetensors" "$ART/falcon_e_1b_embed.bin" \
        "$ART/falcon_e_1b_vocab.bin" "$REPO/prompts/ea_repetitive.txt" "falcone1b_repetitive"
echo "[$(date -Is)] all four legs done" >&2
