# Quick Start: Run membw from a Quiet Terminal

This is the short, copy-paste version. Full details are in `membw.md`.

## One-Time Setup

```bash
cd ~/aegis-linux
CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --example membw
```

## When at a Quiet Terminal

Close Claude, close QEMU, close everything. Open a plain terminal with nothing else running. Then:

```bash
ev run membw
```

That's it. It will:
- Refuse if the machine is busy (>25% CPU on any core)
- Measure sequential read bandwidth (1 thread and N threads)
- Measure ternary weight-stream bandwidth (the actual ALICE access pattern)
- Write a log file to `docs/hardware_logs/membw_<timestamp>.log`
- Write a runcard receipt to `docs/hardware_logs/runcards/<runid>.json`
- Print a summary with three GB/s numbers (report all three or none)

## After the Run

You'll see output like:
```
[runcard] docs/hardware_logs/runcards/2026-08-25T1628Z-membw-05d1e672cbca.json
[runcard] log docs/hardware_logs/membw_2026-08-25_162849Z.log  sha256 abc123...
```

**Save the runid** (e.g., `2026-08-25T1628Z-membw-05d1e672cbca`) and the **log file path**. You'll need these to claim the numbers in the ledger.

The three numbers from the SUMMARY at the end of the log are what get banked:

```
  peak sequential read, 1 thread     : 10.57 GB/s
  peak sequential read, 8 threads    : 25.28 GB/s
  ternary weight-stream, 1 thread    : 0.80 GB/s
```

## Validate It

```bash
ev runcard validate docs/hardware_logs/runcards/<runid>.json
```

Should print "ok" and match the env_hash. If it says the machine was NOT quiet or the tree changed mid-run, the measurement is poisoned — do not bank it.

## Common Failure: "REFUSING TO MEASURE"

If you see:
```
REFUSING TO MEASURE: firefox is at 35.0% CPU (threshold 25%).
```

Stop the process and re-run:
```bash
killall firefox
ev run membw
```

Do **NOT** add `--allow-busy`. That flag is only for explicitly documenting noisy runs; this measurement must be clean.

## Useful Commands

Check if the machine is quiet before running:
```bash
ev env
```

Look for `"quiet": true` in the output.

See all recent membw runs:
```bash
ls -lh docs/hardware_logs/membw_*.log
ls docs/hardware_logs/runcards/*membw*.json
```

Extract the summary from a past run:
```bash
grep -A 10 "==== SUMMARY" docs/hardware_logs/membw_*.log
```

## Full Documentation

See `docs/runcards/membw.md` for complete details, caveats, and how it fits into the evidence workflow.
