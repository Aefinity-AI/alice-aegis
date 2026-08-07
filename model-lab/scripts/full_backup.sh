#!/usr/bin/env bash
# full_backup.sh <target-dir> — FULL offline backup of the entire project
# (home dir minus regenerable toolchains/build dirs) to an external drive.
# Safe to re-run: rsync incremental, never deletes on target.
set -uo pipefail

TGT="${1:?usage: full_backup.sh <target-dir on mounted drive>}"
H=/home/killboxincorporated
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

mkdir -p "$TGT" || { echo "FATAL: cannot create $TGT"; exit 1; }
touch "$TGT/.wt" && rm -f "$TGT/.wt" || { echo "FATAL: $TGT not writable"; exit 1; }
FREE_KB=$(df -k --output=avail "$TGT" | tail -1 | tr -d ' ')
[ "$FREE_KB" -gt 80000000 ] || { echo "FATAL: <80GB free on target"; exit 1; }

echo "[$(ts)] full backup -> $TGT"

# regenerable-environment manifest (so the excluded toolchains can be rebuilt)
{
    echo "backup started: $(ts)"
    echo "--- rustc/cargo ---";  rustc -V 2>/dev/null; cargo -V 2>/dev/null
    echo "--- rustup toolchains ---"; rustup toolchain list 2>/dev/null
    echo "--- python venv (ranger-venv) pip freeze ---"
    "$H/ranger-venv/bin/pip" freeze 2>/dev/null
    echo "--- dpkg selections (key pkgs) ---"
    dpkg -l 2>/dev/null | grep -Ei 'python3|build-essential|clang|qemu' | awk '{print $2, $3}'
    echo "--- systemd user units enabled ---"
    systemctl --user list-unit-files --state=enabled --no-legend 2>/dev/null
} > "$TGT/ENVIRONMENT_MANIFEST.txt"

# FAT32 cannot hold files >4GB — skip them there and list them in the manifest
FSTYPE=$(df --output=fstype "$TGT" | tail -1 | tr -d ' ')
SIZECAP=()
if [ "$FSTYPE" = "vfat" ] || [ "$FSTYPE" = "msdos" ]; then
    SIZECAP=(--max-size=4095m)
    {
        echo "--- target is $FSTYPE: files >4GB SKIPPED on this drive ---"
        find "$H" -xdev -type f -size +4095M -not -path '*/target/*' \
            -not -path "$H/.rustup/*" -not -path "$H/.cargo/*" \
            -not -path "$H/ranger-venv/*" 2>/dev/null
    } >> "$TGT/ENVIRONMENT_MANIFEST.txt"
fi

# ionice idle + nice 19: invisible to the running eval/training jobs
ionice -c3 nice -n 19 rsync -a --info=stats2 "${SIZECAP[@]}" \
    --exclude='*/target/' \
    --exclude='.rustup/' \
    --exclude='.cargo/' \
    --exclude='ranger-venv/' \
    --exclude='.cache/pip/' \
    --exclude='.local/share/Trash/' \
    "$H/" "$TGT/home/"
rc=$?

{
    echo "rsync exit: $rc at $(ts)"
    echo "--- sha256 of model masters ---"
    for f in "$H/aegis_pruned_model.safetensors" \
             "$H/falcon-e-artifacts/MODEL.SAF" \
             "$H/falcon-e-3b-artifacts/MODEL.SAF"; do
        [ -f "$f" ] && sha256sum "$f"
    done
} >> "$TGT/ENVIRONMENT_MANIFEST.txt"

if [ "$rc" -eq 0 ]; then
    echo "backup completed $(ts), rsync rc=0" > "$TGT/BACKUP_COMPLETE.txt"
    echo "[$(ts)] BACKUP COMPLETE -> $TGT"
else
    echo "[$(ts)] BACKUP FAILED rc=$rc -> $TGT (re-run to resume; rsync is incremental)"
fi
exit "$rc"
