# RUNCARD — paired OS-cost boots on the Dell Inspiron 15 (i5-5200U)

Protocol is locked in `docs/hardware_logs/oscost_PREREGISTRATION_2026-07-30.md`.
Read it once before starting. Total hands-on time: ~20–30 minutes.

## Prepare (once, at the Chromebook)

1. Take a **separate** USB stick (do NOT touch the GATE-1 aegis stick).
   Format it FAT32 if it isn't already.
2. Copy the two files from `scripts/linux-arm/out/esp-tree/` onto it,
   preserving paths:
   - `EFI/BOOT/BOOTX64.EFI`
   - `AEGIS_LINUX_ARM.tag` (must be in the stick's root — the init script
     finds its log target by this marker)
   (Alternative: `dd` the whole `out/aegis-linux-arm.img` to the stick —
   destroys existing contents.)

## Boot sequence (at the Dell)

Ten boots, strictly alternating, fresh power-on each time:

| # | arm | stick | what you'll see |
|---|-----|-------|-----------------|
| 1 | U | aegis stick (M7 partition) | usual A.L.I.C.E. boot + gauntlet + 3 prompts |
| 2 | L | new Linux stick | kernel messages, then STAGE banners, gauntlet runs, **powers itself off** |
| 3 | U | aegis stick | … |
| 4 | L | Linux stick | … |
| … | | | continue to 10 (5 each) |

Per boot:
1. Insert the stick for this arm, power on, F12 → pick the stick.
2. **Arm L**: hands-off — runs ~2–4 minutes and powers off by itself.
3. **Arm U**: after the gauntlet finishes and the chat prompt appears, type
   these three prompts EXACTLY (character-for-character — the pairing script
   matches exact strings), one at a time, letting each response finish:

   ```
   hello alice
   how are you today?
   continue
   ```

   Then power off. A typo'd prompt is fine to re-type correctly afterwards —
   the extra line is ignored; the exact three must each appear once.
4. Swap sticks, repeat.

If a boot fails (no video, hang > 5 min): power off, note which boot number
and arm on paper, and continue the sequence — do not retry in place, do not
reorder.

## Afterwards (back at the Chromebook)

1. Mount both sticks; copy verbatim, adding no edits:
   - `BOOTLOG.TXT` (aegis stick) → `docs/hardware_logs/oscost_U_BOOTLOG_2026-07-XX.txt`
   - `BOOTLOG_LINUX_ARM.txt` (Linux stick) → `docs/hardware_logs/oscost_L_BOOTLOG_2026-07-XX.txt`
2. Tell the session the boots are done and whether any boot failed. Analysis
   runs against the preregistration — bands are already fixed.

## Troubleshooting

- **Linux stick shows nothing on screen:** wait 30 s (quiet boot); if still
  nothing after 2 min, the kernel may not like the console setup — report it,
  don't debug at the machine.
- **"STAGE 7-equiv bd-prochot: FAILED" in the L log:** the run completed but
  clock parity is broken; the preregistration's ratio gate will handle it —
  still bring the log back.
- **Machine doesn't power off after L gauntlet:** give it 1 min, then hold
  the power button; the log is already written (sync runs before poweroff).
