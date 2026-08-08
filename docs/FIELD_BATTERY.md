# A.L.I.C.E. Field Test Battery — per-machine procedure

Stick: 940MB FAT32, interactive EFI (commit 2c25d27), STARTUP.NSH fallback present,
BOOTLOG wiped clean before burn.

## 1. Boot (per machine)
- Plug stick, power on, mash the boot-menu key (F12 Dell/Acer/Lenovo, F9 HP, ESC/F2 others).
- Pick the USB entry marked **UEFI**. Machine needs 64-bit UEFI (~2012+) and 2 GB RAM
  (1.4 GB verified to fail with a clean OOM panic — that machine is below the floor; note it).
- Secure Boot must be OFF (Setup → Security). Legacy/CSM-only boxes won't boot it — also worth
  noting as a data point (firmware floor, not CPU floor).
- If it drops to a yellow `Shell>` — wait 5 s; STARTUP.NSH auto-chains into the engine.
- If it hangs/black-screens: power off, bring the stick back anyway — BOOTLOG.TXT records
  the last STAGE reached (1–7) for forensics.

## 2. At the A.L.I.C.E. prompt, in order
| Command | What it does | Time |
|---|---|---|
| `/cpuinfo` | CPU identity, SIMD level, feature flags — quick sanity | seconds |
| `/gauntlet` | THE battery: scalar-vs-AVX2 decode race, per-token-vs-batched prefill, P-state drift control, ctx slope 20/100/400, TURBO_DIAG pre/post, PEAK_MEMORY → logs `==== GAUNTLET ====` block to BOOTLOG.TXT | minutes → ~1 h on throttled/old iron |
| `/parity` | prefill/decode parity (must be 0.0 / bit-identical) → PARITY line | ~1 min |
| `/turbo` | optional: attempt P-state raise, TURBO_DIAG before/after (diagnosis-only, never overrides) | seconds |
| free question | optional coherence spot-check: `What is the capital of France?` | ~1 min |

Then power off (or `/exit`). **Do not skip /gauntlet completion** — the harvester requires the
`GAUNTLET DONE` marker.

## 3. Harvest (back on the Chromebook)
Share the USB with Linux (Files app → right-click → Share with Linux), then:

    ~/collect_gauntlet.sh /dev/sda <nickname>     # e.g. acer_core3, hp_probook_i5

- Batching multiple machines before harvesting is OK: BOOTLOG accumulates every block and each
  block self-identifies via its `GAUNTLET CPU:` line. The script auto-parses only the LAST block —
  for earlier ones, save the raw log and split by hand (or ask Claude).
- Rows land in `docs/hardware_logs/gauntlet_dataset.tsv`; raw logs beside it. Commit after each
  harvest session so every number stays traceable.

## 4. ⚡ MANDATORY BEFORE RFI SEND-OFF: energy measurement (real numbers)
The 3.31 J/token figure in the submission has no committed raw-readings log yet —
this run creates it. On the Chromebook, AFTER all field runs, with the box quiet:

    1. UNPLUG AC (battery telemetry reads 0 while plugged in)
    2. close everything; the script refuses to run if any process >25% CPU
    3. ~/measure_energy.sh          # needs the long idle window — don't touch it
    4. commit the readings log; Claude reconciles the submission numbers vs 3.31
       BEFORE the Gmail draft is sent

## Known-good reference rows
| machine | SIMD | decode | notes |
|---|---|---|---|
| Dell i5-5200U (Broadwell 2015) | 5.08× | 0.61 tok/s | firmware pinned 22% clock |
| QEMU i5-10210U host | 5.53× | 3.15 tok/s | KVM |
