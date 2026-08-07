#!/usr/bin/env python3
"""A.L.I.C.E. integrity tripwire.

Greps the live source tree for known classes of drift:
  P0  fabricated/mocked metrics, unconditional review verdicts,
      embedding-stride splits (the prefill-corruption bug class)
  P1  hardcoded artifact sizes, fixed TSC-clock assumptions,
      fragile boot patterns, per-call static-mut init

This is a tripwire, not a verifier: a clean pass means "none of the known
bad patterns matched," not "the code is correct." Read the code.

Usage:  python integrity_check.py <repo-root-or-file> [--include-backups]
Exit:   1 if any P0 finding, else 0.
"""

import re
import sys
from pathlib import Path

SKIP_DIR_TOKENS = ("_ALICE_1_0_BACKUP", ".gemini", "target", "scratch", ".git")
SCAN_SUFFIXES = {".rs", ".sh", ".toml", ".txt", ".md"}

# (severity, pattern, why)
CHECKS = [
    ("P0", r"Mock computation|for DARPA review",
     "Mocked metric explicitly labeled for review — replace with a real measurement."),
    ("P0", r"Phase I Grant Viable",
     "Unconditional review verdict in code. Delete or gate on real, stated criteria."),
    ("P0", r"\b14\.12\b|\b14\.58\b|\b312\.5\b",
     "Known fabricated perplexity/memory constants from the mock eval harness."),
    ("P0", r"TTFT[^\n]{0,40}14\s*ms|TPS[^\n]{0,20}84\.6|Peak RSS[^\n]{0,20}412",
     "Known fabricated CLI benchmark output (fixed TTFT/TPS/RSS)."),
    ("P0", r"emb_dim\s*\*\s*2\b|hidden(?:_size|_dim)?\s*\*\s*2\b|BF16 is 2 bytes",
     "2-byte embedding stride. Unless the WHOLE pipeline is BF16 via the single "
     "derived element size, this is the prefill-corruption bug class. Verify every "
     "reader (prefill, decode, LM head) derives stride from one init-time value."),
    ("P1", r"\b522831576\b|\b513935360\b|\b432968\b",
     "Hardcoded artifact byte size. Read FileInfo.file_size() instead; any re-forge "
     "changes these and boot dies with a size mismatch."),
    ("P1", r"2_500_000_000|2\.5\s*GHz assumption",
     "Fixed TSC-frequency assumption. Calibrate via CPUID leaf 0x15 or report "
     "cycles/token — a fixed denominator fabricates the seconds."),
    ("P1", r"find_handles::<SimpleFileSystem>",
     "Blind filesystem enumeration. Must resolve the boot volume via "
     "LoadedImage.device() or NVMe-first firmware mounts the wrong drive."),
    # Path-scoped to aegis-uefi: host-side tools (forge) legitimately write
    # lowercase files; only firmware-facing open calls must be uppercase.
    ("P1@aegis-uefi", r'"(model\.saf|embed\.bin|vocab\.bin)"',
     "Lowercase 8.3 filename in a firmware open call. Strict UEFI matches the "
     "FAT32 directory table byte-for-byte; names must be uppercase (MODEL.SAF ...)."),
    ("P1", r"static\s+mut\s+\w*(LUT|INIT|UNPACK)",
     "static-mut init flag checked in a hot path. Initialize once at engine "
     "construction."),
    ("P1", r"Ternary Quantization \(1\.58-bit\) Pipeline|Quantization & Pruning Pipeline",
     "Forge log claims quantization; the stage is vocabulary pruning (weights "
     "arrive pre-quantized). Misstatement a reviewer will catch."),
]

# Inventory (informational): every embedding/stride multiply, to eyeball consistency.
STRIDE_INVENTORY = re.compile(
    r"(?:emb_dim|hidden_size|hidden_dim)\s*\*\s*\d+|row\s*\*\s*emb_dim\s*\*\s*\d+")


def iter_files(root: Path, include_backups: bool):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in SCAN_SUFFIXES:
            continue
        parts = str(p)
        if not include_backups and any(tok in parts for tok in SKIP_DIR_TOKENS):
            continue
        yield p


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    include_backups = "--include-backups" in sys.argv
    if not args:
        print(__doc__)
        return 2
    root = Path(args[0])
    if not root.exists():
        print(f"error: {root} does not exist")
        return 2

    compiled = [(sev, re.compile(pat), why) for sev, pat, why in CHECKS]
    findings = {"P0": [], "P1": []}
    inventory = []

    for f in iter_files(root, include_backups):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for sev, rx, why in compiled:
                if "@" in sev:
                    sev, scope = sev.split("@", 1)
                    if scope not in str(f):
                        continue
                if rx.search(line):
                    findings[sev].append((f, lineno, line.strip()[:120], why))
            if STRIDE_INVENTORY.search(line):
                inventory.append((f, lineno, line.strip()[:120]))

    for sev in ("P0", "P1"):
        if findings[sev]:
            print(f"\n=== {sev} findings ({len(findings[sev])}) ===")
            for f, ln, snippet, why in findings[sev]:
                print(f"{f}:{ln}\n    {snippet}\n    why: {why}")

    if inventory:
        print(f"\n=== stride inventory ({len(inventory)} sites — check they all "
              f"derive from ONE init-time element size) ===")
        for f, ln, snippet in inventory:
            print(f"{f}:{ln}    {snippet}")

    p0, p1 = len(findings["P0"]), len(findings["P1"])
    print(f"\nSummary: {p0} P0, {p1} P1 findings.")
    if p0:
        print("P0 findings block any release, benchmark run, or review artifact.")
        return 1
    if p1:
        print("No P0s. P1 findings above are open items — fix or explicitly defer.")
    else:
        print("No known-bad patterns matched. (Tripwire only — still read the code.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
