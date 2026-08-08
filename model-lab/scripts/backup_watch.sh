#!/usr/bin/env bash
# backup_watch.sh — polls for drives shared with Linux and runs a full
# project backup to each one, sequentially (avoids USB contention).
# Exits when 3 drives carry a complete backup, or after 48h.
set -uo pipefail
SCRIPT=/home/killboxincorporated/model-lab/scripts/full_backup.sh
ROOT=/mnt/chromeos/removable
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] backup watcher armed; waiting for drives under $ROOT"
END=$((SECONDS + 172800))
while [ "$SECONDS" -lt "$END" ]; do
    shopt -s nullglob
    for d in "$ROOT"/*/; do
        tgt="${d}alice-full-backup"
        [ -f "$tgt/BACKUP_COMPLETE.txt" ] && continue
        echo "[$(ts)] drive detected: $d -> backing up"
        bash "$SCRIPT" "$tgt"
    done
    shopt -u nullglob
    n=$(ls "$ROOT"/*/alice-full-backup/BACKUP_COMPLETE.txt 2>/dev/null | wc -l)
    if [ "$n" -ge 3 ]; then
        echo "[$(ts)] all 3 drives backed up — watcher done"
        exit 0
    fi
    sleep 60
done
echo "[$(ts)] watcher timed out after 48h with $n complete"
exit 1
