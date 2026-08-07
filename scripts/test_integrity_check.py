#!/usr/bin/env python3
"""Self-test for integrity_check.py.

An instrument that has never been calibrated against a known signal is not an
instrument. Each case below is a real line from this project's history, or a
real line from its current source that must NOT be flagged.

    python3 scripts/test_integrity_check.py     -> exit 0 on pass
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "integrity_check.py")

# (source line, should_be_flagged, provenance)
CASES = [
    # --- real fabrications from this repo's history: MUST flag ---
    ('println!("  [BENCHMARK] Tokens Per Second (TPS): 84.6");', True,
     "antigravity-aegis, unit-before-number — missed by the first checker"),
    ('println!("  [BENCHMARK] Time to First Token (TTFT): 14ms");', True,
     "antigravity-aegis"),
    ('println!("  [BENCHMARK] Peak RSS: 412 MB");', True, "antigravity-aegis"),
    ('println!("Sparsity Protocol:   61.8034% Golden Ratio");', True,
     "numerology printed as fact"),
    ('println!("Decoded text: The capital of France is Paris");', True,
     "FAILED GEMMA4 PROJECT — the hardcoded success"),
    ('println!("  [STATUS] Memory: 34MB FibScratchPool (Locked)");', True,
     "antigravity-aegis"),
    ('println!("* INFERENCE SPEED: 84.6 cycles/token");', True, "synthetic"),
    ('println!("Phase I Grant Viable");', True, "unconditional verdict"),
    ('println!("Multiplications: ZERO");', True, "Infinity OS dashboard"),
    ('println!("Energy: 0.9 mJ per token");', True, "Infinity OS dashboard"),

    # --- legitimate lines from the current engine: MUST NOT flag ---
    ('println!("Perplexity (teacher-forced, {} tokens): {:.3}", n, ppl);', False,
     "aegis-eval — computed"),
    ('println!("Decode {} tokens in {:.3} s ({:.2} tok/s)", n, dt, tps);', False,
     "aegis-linux — computed"),
    ('print!("\\x1B[2J\\x1B[1;1H");', False,
     "ANSI screen clear — '2J' is not two joules"),
    ('let _ = write!(st, "* WALL TIME: {}.{:03} s", secs, ms);', False,
     "aegis-uefi — computed"),
    ('// Measured: 12.22 ms vs 1.94 ms matvec (6.3x)', False,
     "a comment documenting a real measurement"),
    ('//! CTZ dual-bitmap : 12.60 ms  ( 1.40 GMAC/s)', False,
     "bench header documenting a real measurement"),
    ('println!("Engine Online.");', False, "no metric at all"),
    ('boot_log(&mut root, &format!("clock {}% of nominal", pct));', False,
     "aegis-uefi — computed"),

    # --- regression tripwires: bugs already paid for once. MUST flag ---
    ('if embeddings.len() < vocab_size * emb_dim * 4 { return 0; }', True,
     "the F32 stride guard that made argmax return token 0 forever"),
    ('let size = 522831576;', True, "hardcoded artifact size"),
    ('let secs = cycles / 2_500_000_000;', True, "fixed TSC frequency assumption"),
    ('let f = root.open("model.saf")?;', True, "lowercase artifact name"),
    ('static mut UNPACK_LUT: [f32; 1024] = [0.0; 1024];', True,
     "static mut with check-then-set init — UB under threads"),
    ('let h = find_handles::<SimpleFileSystem>()?;', True,
     "blind first-filesystem lookup — mounts the internal NVMe, not the stick"),

    # --- and the stale rule's victims: correct BF16 code. MUST NOT flag ---
    ('let start = current_tok as usize * emb_dim * 2;', False,
     "BF16 stride is 2 — this is the FIX, not the bug. The old tripwire flagged it 14x"),
    ('if start + emb_dim * 2 > self.pipeline.embeddings.len() { continue; }', False,
     "BF16 bounds guard — correct"),
]


def run_on(source_line: str) -> bool:
    """True if integrity_check flags this line."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        os.makedirs(src)
        with open(os.path.join(src, "main.rs"), "w") as f:
            f.write("fn main() {\n    " + source_line + "\n}\n")
        r = subprocess.run([sys.executable, CHECK, td], capture_output=True, text=True)
        return r.returncode != 0


def main():
    failures = []
    for line, expect_flag, why in CASES:
        got = run_on(line)
        ok = got == expect_flag
        mark = "ok  " if ok else "FAIL"
        verdict = "flagged" if got else "clean"
        want = "flag" if expect_flag else "clean"
        print(f"  [{mark}] {verdict:<8} (want {want:<5}) {why}")
        if not ok:
            failures.append((line, expect_flag, got, why))

    print()
    if failures:
        print(f"{len(failures)} case(s) failed:\n")
        for line, expect, got, why in failures:
            print(f"  {line}")
            print(f"    expected {'flag' if expect else 'clean'}, got {'flag' if got else 'clean'}  ({why})\n")
        return 1
    print(f"all {len(CASES)} cases pass — the instrument is calibrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
