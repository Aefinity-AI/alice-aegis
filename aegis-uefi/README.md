# aegis-uefi — the A.L.I.C.E. unikernel

`no_std` + `alloc` UEFI application. Boots from firmware off a FAT32 volume
with no operating system present, loads `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN`, and runs transformer inference in ring 0.

Build the production binary with `./build_hardfloat.sh` (the stock
`x86_64-unknown-uefi` target is soft-float — see the script's header and
CLAUDE.md ledger A14). Gate any `.efi` before staging it with
`../scripts/check-efi-simd.sh`.

## AEFINITY OS: `JOB.TXT` / `RESULT.TXT`

Spec and build contract: [`../program/AEFINITY_OS.md`](../program/AEFINITY_OS.md).
Phase 0 lives in `src/job.rs` (parser, runner, record, watchdog, reset) and
`src/sysinfo.rs` (read-only CPUID identity), reached from a single hook in
`src/main.rs` immediately before the interactive console.

**If the boot volume root carries a `JOB.TXT`**, this box is a headless lab
worker instead of a console: it parses the directives, runs them in file
order, writes one `RESULT.TXT` next to them, and then resets or halts.

**If it does not, nothing changes.** No `JOB.TXT` means the hook returns
immediately and the boot path is exactly what it was — which is what
`cargo xtask boot-test` still exiting 33 verifies.

### `JOB.TXT` (input)

ASCII, one `KEY value` directive per line, CRLF or LF, `#` starts a comment.
Unknown or malformed keys are written to `BOOTLOG.TXT` and ignored, so a job
file written for a later phase still runs its phase-0 parts here.

```
BUDGET 240                 # wall seconds for the whole job (default 300)
MODE oneshot               # oneshot (default) | resident (phase 2)
NET dhcp                   # or: NET static 10.0.2.15/24 10.0.2.2  (phase 1a)
REPORT http://10.0.2.2:8787/aefinity/result   # POST the record here (phase 1b)
LISTEN 4242                # resident listener port (phase 2)
TOKENS 64                  # max new tokens for following PROMPTs (cap 1024)
PROMPT The capital of France is        # generate; may repeat
BENCH 8                    # generate N tokens from the fixed bench prompt
NETCHECK 10.0.2.2:9000     # phase 1a
AFTER reset                # reset (default) | halt
```

`PROMPT` and `BENCH` run in file order and each appends a `job.N.*` block to
the record. A `PROMPT` uses the `TOKENS` value in force where it appears.

### `RESULT.TXT` (output)

`key=value` lines, LF, written **once** at the end of the job — created fresh
each time, never appended. Values are printable ASCII; `prompt` and
`response` are escaped (`\\`, `\n`, `\r`, everything else non-printable as
`\xNN`) and capped at 256 bytes. Full key list in the spec, §3.

`env=iron|vm` comes from the CPUID hypervisor bit. **Every record produced
under QEMU says `env=vm`, and that is the record telling you its `tps` is not
a measurement** (CLAUDE.md Rule A). `tsc_per_tok` is RDTSC *ticks*, not
cycles, for the same reason. `digest` is sha256 over the generated token ids
as little-endian `u32`, first 16 hex — the cross-machine witness of what was
generated, and the thing a fleet actually compares.

### Budget, watchdog and AFTER

`BUDGET` is wall-clock, read from UEFI `GetTime()`. When it is spent the
token callback stops generation at the next token and the record carries
`verdict=FAIL budget` — a short answer with an honest verdict, still written
to disk. Independently, the firmware watchdog is armed at `BUDGET + 60`, so a
hang anywhere in our own code is reset by firmware rather than leaving a
headless box wedged. `AFTER reset` issues `ResetSystem(COLD)`; `AFTER halt`
disarms the watchdog and returns to the normal boot path.

Everything stays inside boot services — the file, watchdog and reset
protocols all require them, so the unikernel never calls `ExitBootServices`.

### Gate

```bash
cargo xtask boot-test   # no JOB.TXT: unchanged boot path, QEMU exit 33
cargo xtask job-test    # stages a JOB.TXT, asserts the record and the reset
```

`job-test` boots with `-no-reboot`, so the guest's `ResetSystem` exits QEMU,
then asserts `target/esp/RESULT.TXT` has `aefinity_os=0.1`, `verdict=OK`,
`jobs=2` and `env=vm`. Structure and exit codes only — never a value.
