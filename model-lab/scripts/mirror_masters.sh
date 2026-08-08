#!/usr/bin/env bash
# mirror_masters.sh <target-dir> — mirror the irreplaceable masters to an
# external drive (run once per drive). ~14GB total. Upstream repo
# availability is not permanent; this mirror is the insurance policy.
#
# Typical target after sharing a drive with Linux from the ChromeOS Files app:
#   ./mirror_masters.sh "/mnt/chromeos/removable/TOSHIBA EXT/alice-mirror"
set -uo pipefail

TGT="${1:?usage: mirror_masters.sh <target-dir on mounted drive>}"
H=/home/killboxincorporated

mkdir -p "$TGT" || { echo "FATAL: cannot create $TGT — is the drive shared with Linux?"; exit 1; }
touch "$TGT/.write_test" && rm -f "$TGT/.write_test" \
    || { echo "FATAL: $TGT not writable"; exit 1; }

FREE_KB=$(df -k --output=avail "$TGT" | tail -1 | tr -d ' ')
[ "$FREE_KB" -gt 20000000 ] || { echo "FATAL: <20GB free on target"; exit 1; }

SRCS=(
    "$H/falcon-e-artifacts"            # 1B engine artifacts (635M)
    "$H/falcon-e-3b-artifacts"         # 3B engine artifacts (953M)
    "$H/aegis_pruned_model.safetensors" # BitNet pruned master (499M)
    "$H/aegis-forge/embed.bin"         # BitNet pruned embed
    "$H/aegis-forge/vocab.bin"         # BitNet pruned vocab
    "$H/.cache/huggingface/hub"        # HF masters: BitNet, Falcon-E-3B, corpora (2.0G)
    "$H/model-lab/data"                # corpora + evals + provenance (7.3G)
    "$H/model-lab/tinybit"             # trainer + ckpts + logs (1.2G)
    "$H/model-lab/scripts"
    "$H/model-lab/checkpoints"
    "$H/docs/hardware_logs"            # every number's provenance
    "$H/program"                       # roadmap / ledger / handoff
)

echo "== mirror start $(date -u +%FT%TZ) -> $TGT"
rc=0
for s in "${SRCS[@]}"; do
    [ -e "$s" ] || { echo "-- skip (missing): $s"; continue; }
    echo "-- rsync $s"
    rsync -a --info=progress2 "$s" "$TGT/" || { echo "!! rsync FAILED: $s"; rc=1; }
done

echo "== sha256 of model masters (compare across drives / after restore)"
{
    date -u +%FT%TZ
    for f in "$H/aegis_pruned_model.safetensors" \
             "$H/falcon-e-artifacts/MODEL.SAF" \
             "$H/falcon-e-3b-artifacts/MODEL.SAF"; do
        [ -f "$f" ] && sha256sum "$f"
    done
} | tee -a "$TGT/MIRROR_MANIFEST.txt"

echo "== mirror done rc=$rc $(date -u +%FT%TZ)"
exit "$rc"
