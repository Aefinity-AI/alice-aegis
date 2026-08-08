#!/usr/bin/env bash
# toshiba_archive_reformat.sh — user-authorized 2026-07-18:
# "move the project files and related to 1tb aegis and format the toshiba
#  drive fully" (condition verified: transmuter project IS on the Toshiba).
# Phases: A) archive Toshiba -> fedora-disk  B) verify (count+bytes+hash
# spot-check)  C) format Toshiba ext4 ONLY if verify is perfect
# D) fresh home backup to both drives.
# CANCEL ANY TIME BEFORE PHASE C: systemctl --user stop toshiba-reformat
set -uo pipefail

SRC=/mnt/external/toshiba
DST=/mnt/external/fedora-disk/toshiba-archive-2026-04-era
EXPECT_UUID="F0BE-AF61"   # recorded at first mount; format refuses any other partition
BK=/home/killboxincorporated/model-lab/scripts/full_backup.sh
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "[$(ts)] FATAL: $*"; exit 1; }

mountpoint -q "$SRC" || die "toshiba not mounted"
mountpoint -q /mnt/external/fedora-disk || die "fedora-disk not mounted"
mkdir -p "$DST" || die "cannot create $DST"

# ---- Phase A: archive copy ----
echo "[$(ts)] PHASE A: archiving Toshiba -> $DST (excl. alice-full-backup partial)"
ls -la "$SRC" > "$DST/TOSHIBA_TOP_LEVEL_INVENTORY.txt" 2>/dev/null
ionice -c3 nice -n 19 rsync -a --modify-window=2 --info=stats1 \
    --exclude='/alice-full-backup/' \
    --exclude='System Volume Information/' \
    "$SRC/" "$DST/data/"
rc=$?
[ "$rc" -eq 0 ] || [ "$rc" -eq 24 ] || die "archive rsync failed rc=$rc — NOT formatting"

# ---- Phase B: verification ----
echo "[$(ts)] PHASE B: verification"
DIFFS=$(rsync -a --dry-run --itemize-changes --modify-window=2 \
    --exclude='/alice-full-backup/' --exclude='System Volume Information/' \
    "$SRC/" "$DST/data/" | grep -cv '^\.d' || true)
echo "[$(ts)] dry-run re-sync differences: $DIFFS (must be 0)"

SRC_STATS=$(find "$SRC" -path "$SRC/alice-full-backup" -prune -o -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n, s}')
DST_STATS=$(find "$DST/data" -type f -printf '%s\n' | awk '{n++; s+=$1} END {print n, s}')
echo "[$(ts)] source files/bytes: $SRC_STATS | dest files/bytes: $DST_STATS"

HASH_FAIL=0
mapfile -t SAMPLE < <(find "$SRC" -path "$SRC/alice-full-backup" -prune -o -type f -printf '%s\t%P\n' \
    | sort -rn | awk -F'\t' 'NR<=3{print $2} NR%997==0{print $2}' | head -8)
for f in "${SAMPLE[@]}"; do
    h1=$(sha256sum "$SRC/$f" | cut -d' ' -f1)
    h2=$(sha256sum "$DST/data/$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$h1" = "$h2" ]; then
        echo "[$(ts)] hash OK: $f"
    else
        echo "[$(ts)] HASH MISMATCH: $f"
        HASH_FAIL=1
    fi
done

if [ "$DIFFS" -ne 0 ] || [ "$SRC_STATS" != "$DST_STATS" ] || [ "$HASH_FAIL" -ne 0 ]; then
    die "VERIFICATION FAILED (diffs=$DIFFS src='$SRC_STATS' dst='$DST_STATS' hashfail=$HASH_FAIL) — Toshiba NOT touched"
fi
{
    echo "verified $(ts): diffs=0, files/bytes match ($SRC_STATS), $((${#SAMPLE[@]})) spot hashes OK"
} > "$DST/ARCHIVE_VERIFIED.txt"
echo "[$(ts)] PHASE B PASSED — archive is a verified complete copy"

# ---- Phase C: format (guarded) ----
PART=$(findmnt -no SOURCE "$SRC")
DISK="/dev/$(lsblk -no pkname "$PART")"
UUID=$(sudo blkid -s UUID -o value "$PART")
TRAN=$(lsblk -dno tran "$DISK")
echo "[$(ts)] PHASE C: format target part=$PART disk=$DISK uuid=$UUID tran=$TRAN"
[ "$UUID" = "$EXPECT_UUID" ] || die "UUID mismatch ($UUID != $EXPECT_UUID) — refusing to format"
[ "$TRAN" = "usb" ] || die "not a USB disk — refusing to format"
case "$DISK" in /dev/vd*|/dev/nvme*) die "system disk pattern — refusing";; esac

sync
sudo umount "$SRC" || die "umount busy — refusing to format while in use"
sudo parted --script "$DISK" mklabel gpt mkpart primary ext4 1MiB 100% || die "parted failed"
sleep 3
NEWPART="${DISK}1"
[ -b "$NEWPART" ] || die "new partition $NEWPART missing"
sudo mkfs.ext4 -F -q -L TOSHIBA_1TB "$NEWPART" || die "mkfs failed"
sudo mount "$NEWPART" "$SRC" || die "remount failed"
sudo chown 1000:1000 "$SRC"
echo "[$(ts)] PHASE C DONE: Toshiba is now ext4 'TOSHIBA_1TB', mounted at $SRC"
df -h "$SRC" | tail -1

# ---- Phase D: fresh home backups to both drives ----
echo "[$(ts)] PHASE D: home backup -> fresh Toshiba (ext4, no 4GB limit) + fedora-disk"
bash "$BK" "$SRC/alice-full-backup"
bash "$BK" /mnt/external/fedora-disk/alice-full-backup
echo "[$(ts)] ALL PHASES COMPLETE"
