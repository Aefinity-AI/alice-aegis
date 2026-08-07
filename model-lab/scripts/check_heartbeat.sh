#!/usr/bin/env bash
# check_heartbeat.sh — report any VM suspend gaps from the heartbeat log.
# A gap >120s between beats means ChromeOS suspended the VM in that window.
LOG=/home/killboxincorporated/docs/hardware_logs/vm_heartbeat.log
[ -f "$LOG" ] || { echo "no heartbeat log yet"; exit 1; }
awk 'NR>1 && $2-prev>120 {printf "SUSPEND GAP: %d min ending %s\n", ($2-prev)/60, $1; n++}
     {prev=$2}
     END {if (!n) print "no suspend gaps — VM ran continuously";
          printf "beats: %d  first: ", NR}' "$LOG"
head -1 "$LOG" | cut -d' ' -f1
echo -n "last:  "; tail -1 "$LOG" | cut -d' ' -f1
