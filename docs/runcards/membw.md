# Memory Bandwidth Benchmark (membw) Runcard

## Overview
The `membw` benchmark measures the machine's memory read bandwidth under three distinct access patterns. This measurement closes the only claim in the 2026-07-29 DARPA audit with no source at all (TECHNICAL_REPORT.md:185 "17.3 GB/s" — unsourced until now).

The benchmark reports three separate numbers:
1. **Peak sequential read, 1 thread** — single-thread hardware ceiling in roofline sense
2. **Peak sequential read, N threads** — saturated throughput; a single-thread figure is NOT the machine ceiling
3. **Ternary weight-stream, 1 thread** — actual access pattern of ternary_matvec (what belongs in a decode roofline)

## Prerequisites: Build membw
Before running, the binary must be built. This is a one-time setup:

```bash
cd ~/aegis-linux
CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --example membw
```

Verify the binary exists:
```bash
ls -la ~/aegis-linux/target/release/examples/membw
```

## Exact Command to Run

From a **quiet terminal** (no background processes using >25% CPU):

```bash
ev run membw
```

That is all. The `ev run` wrapper handles:
- Environment fingerprinting
- CPU busyness check (will refuse if another process is at >25% CPU)
- Log capture and tee to stdout
- Runcard JSON receipt generation
- SHA256 verification of binaries and logs

## What Happens
1. `ev run membw` invokes `program/loop/runners/membw.sh`
2. `membw.sh` checks that the binary is built and calls it with default parameters:
   - Buffer size: 512 MB (or 8x largest cache, whichever is larger)
   - Passes: 5
   - Threads: `nproc` (all logical cores)
3. The binary:
   - Pre-faults the buffer into RAM
   - Performs 5 sequential read passes on each access pattern
   - Reports timing for each pass (both wall-clock seconds and GB/s)
   - Prints a SUMMARY with the best (highest GB/s) result for each pattern

## Output Locations
After a successful run, you will see output like:

```
[runcard] docs/hardware_logs/runcards/2026-08-25T1628Z-membw-05d1e672cbca.json
[runcard] log docs/hardware_logs/membw_2026-08-25_162849Z.log  sha256 abc123...  env 58b35e35d93ab954
```

Two files are created:
- **Log file**: `docs/hardware_logs/membw_<timestamp>.log` — tee'd stdout, CSV of all passes + summary
- **Runcard receipt**: `docs/hardware_logs/runcards/<runid>.json` — machine-readable envelope with:
  - Environment fingerprint (CPU, kernel, virt, git SHA, binary digests)
  - Quiet check status (must be `true` for the measurement to be valid)
  - Log file hash and size
  - Exit code

## Example Output (from log file)

```
# runid: 2026-08-25T1628Z-membw-05d1e672cbca
# started_utc: 2026-08-25T16:28:49Z
# env_hash: 58b35e35d93ab954
# cmd: bash /home/justinbrianthompson/projects/alice-aegis/program/loop/runners/membw.sh
# host: penguin | Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz | virt=kvm | git=05d1e67
# free before: 1879 MB available
# membw: buffer 512 MB, 5 passes, up to 8 threads

label,threads,pass,bytes_read,secs,GB_per_s,checksum
seq_read,1,0,536870912,0.050784,10.572,6da96b07d6000001
seq_read,1,1,536870912,0.053587,10.019,6da96b07d6000001
...
ternary_stream,1,0,536870912,0.705880,0.761,2e1
...

==== SUMMARY (best of 5 passes; report ALL THREE or none) ====
  peak sequential read, 1 thread     : 10.57 GB/s
  peak sequential read, 8 threads    : 25.28 GB/s
  ternary weight-stream, 1 thread    : 0.80 GB/s
```

## Requirements and Pitfalls

### MUST: Quiet Terminal
- **This is a sensitive measurement.** The runcard.py script checks `ps` for the busiest process.
- If ANY process (including Claude IDE, QEMU, etc.) is using >25% of a single CPU core, `ev run` will **refuse to measure** and exit with code 3.
- Solution: Close IDE tabs, stop background VMs, and run from a plain shell with no other activity.

### MUST NOT: --allow-busy
- Do **NOT** run `ev run membw --allow-busy`. This flag marks the runcard as explicitly "noisy" and the measurement becomes useless for the report.
- If you see "REFUSING TO MEASURE" error, stop the interfering process and re-run; do not use --allow-busy.

### Binary Must Exist
- If the binary is not built, `membw.sh` will print a fatal error and the build command needed.
- Rebuild with: `cd ~/aegis-linux && CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --example membw`

### Environment Stability
- The runcard checks environment before and after the run.
- If the tree is modified mid-run (files changed, binaries recompiled), the runcard will flag `env_stable: false`.
- Do not modify the tree during measurement; let it complete undisturbed.

### Buffer Size and System Memory
- Default buffer is 512 MB. The system must have this available.
- If the machine has less than 512 MB free RAM, membw.sh will proceed anyway but results will be affected by swapping.
- Check free memory before running: `free -m` should show >512 MB available.

## Customization (if needed)

The `membw.sh` runner can be customized via environment variables:
- `MB`: Buffer size in megabytes (default 512)
- `PASSES`: Number of passes for each pattern (default 5)
- `THREADS`: Number of threads to use (default `nproc`)

Example (if you wanted to change defaults, which is not recommended):
```bash
MB=256 PASSES=3 THREADS=4 ev run membw
```

However, **report only runs with default parameters** (512 MB, 5 passes, nproc threads). Any variation must be disclosed.

## Validation and Re-verification

After running, validate the runcard receipt:

```bash
ev runcard validate docs/hardware_logs/runcards/2026-08-25T1628Z-membw-*.json
```

This re-verifies that:
- The log file still exists and has not been modified
- Declared binaries are unchanged
- The machine was quiet (quiet_check.quiet == true)
- No tree changes mid-run (env_stable == true)

## How It Fits into the Evidence Workflow

1. **Measure**: `ev run membw` → produces log + runcard
2. **Extract**: Parse the SUMMARY section from the log to get the three GB/s figures
3. **Bank**: `ev claim add --id <id> --value <num> --source <log> --runid <runid>`
4. **Gate**: `ev gate docs/TECHNICAL_REPORT.md` will verify the claim has a valid runcard
5. **Report**: Only claims with valid runcards may appear in the document

## References
- **Benchmark source**: `aegis-linux/examples/membw.rs` — documents method, caveats, and why three figures matter
- **Runner script**: `program/loop/runners/membw.sh` — wraps the binary with environment checks
- **Runcard harness**: `program/loop/tools/runcard.py` — the harness that refuses to measure on a busy box
- **Existing log**: `docs/hardware_logs/membw_2026-08-25_162849Z.log` — example output for reference
- **TECHNICAL_REPORT.md:185** — the unsourced 17.3 GB/s claim this measurement closes
