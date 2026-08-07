#!/usr/bin/env python3
"""Integrity gate: fail if the source prints a number it did not compute.

This is the enforcement layer `alice-launch-protocol.md` depends on and which
was never built. Its doctrine, restated: **no number gets printed that the
machine did not compute in that same run.**

For fifteen months this project shipped:
    println!("  [BENCHMARK] Tokens Per Second (TPS): 84.6");
    println!("Decoded text: The capital of France is Paris");
    // perplexity.rs returning a hardcoded 14.12 / 14.58
    "Multiplications: ZERO", "0.9 mJ per token", "Phase I Grant Viable"

Every one of those is a print macro with a metric-shaped literal and **no
format arguments** — nothing computed, nothing measured. That is the signature
this checks for.

    scripts/integrity_check.py [repo-root]     -> exit 0 clean, 1 dirty

Comments are ignored: a `//!` header documenting a measured result is exactly
what we want, and is not a print statement.
"""
import os
import re
import sys

# A print/write macro whose literal carries a unit-bearing number.
MACRO = re.compile(r'\b(println!|print!|eprintln!|write!|writeln!|format!)\s*\(')
# A metric can be written unit-after ("14ms", "412 MB", "61.8034%") or
# unit-before ("TPS: 84.6", "Peak RSS: 412"). The first version of this checker
# only matched the former, and silently missed `TPS: 84.6` — the single most
# notorious fabricated number in this repository's history. Both directions now.
METRIC_UNIT_AFTER = re.compile(
    r'\d+(?:\.\d+)?\s*'
    r'(ms\b|s\b|us\b|ns\b|tok/s|tokens?/s|TPS\b|MB\b|GB\b|KB\b|GHz\b|MHz\b'
    r'|J\b|mJ\b|W\b|%|cycles?/token|GMAC/s|ppl\b|perplexity)',
    re.IGNORECASE,
)
METRIC_UNIT_BEFORE = re.compile(
    r'(TPS|TTFT|RSS|latency|throughput|perplexity|ppl|tokens?[ /]per[ /]second'
    r'|cycles?[ /]per[ /]token|energy|power|sparsity|speed)'
    r'\s*[)\]:=]*\s*\d',
    re.IGNORECASE,
)


def looks_like_metric(s: str) -> bool:
    return bool(METRIC_UNIT_AFTER.search(s) or METRIC_UNIT_BEFORE.search(s))

# Strings that assert a result rather than report one.
DENY_PHRASES = [
    "Grant Viable",
    "READY FOR FLIGHT",
    "HIGH ASSURANCE",
    "Multiplications: ZERO",
    "BEATS GPT",
    "capital of France is Paris",   # the hardcoded println! of the Gemma era
    "mock computation",
    "for DARPA review",
    "DARPA Grant Review compliance",
]

# Regression tripwires: specific bugs this project has already paid for once.
# Inherited from an earlier `integrity_check.py`, minus one rule that had gone
# stale — it flagged `emb_dim * 2`, which was the *bug* when embeddings were F32
# and is the *fix* now that they are BF16. It fired 14 times on correct code.
# A checker that fires on correct code gets ignored, and then it protects nothing.
# The rule below tests the invariant that one was reaching for: every reader of
# the embedding table must agree on the stride.
REGRESSIONS = [
    (re.compile(r"vocab_size\s*\*\s*emb_dim\s*\*\s*4|emb_dim\s*\*\s*4\b"),
     "4-byte embedding stride. Embeddings are BF16 (2 bytes). This exact mismatch "
     "made f32_dot_argmax return token 0 forever."),
    (re.compile(r"\b522831576\b|\b513935360\b|\b432968\b|\b257310720\b"),
     "hardcoded artifact size. Read it from the file at runtime."),
    (re.compile(r"2_500_000_000|2\.5\s*GHz"),
     "fixed TSC-frequency assumption. rdtsc is invariant; use UEFI GetTime() or "
     "APERF/MPERF, or the number is wrong on every machine but one."),
    # NOTE: case-SENSITIVE on purpose. The uppercase MODEL.SAF/EMBED.BIN/VOCAB.BIN
    # are correct; only a *lowercase* name in a UEFI open path is the bug. A
    # case-insensitive version of this rule flagged the correct code — the
    # checker's own mistake, caught because it fired on known-good source.
    (re.compile(r'(?:open|from_str_with_buf|get_file_size|load_file_into)\s*\([^)]*"(?:model\.saf|embed\.bin|vocab\.bin)"'),
     "lowercase artifact name in a UEFI file-open path. FAT is case-insensitive "
     "but UEFI open() is not always; on the stick these are MODEL.SAF / EMBED.BIN "
     "/ VOCAB.BIN. (Writing lowercase files on Linux, as aegis-forge does, is fine.)"),
    (re.compile(r"static\s+mut\s+\w*(LUT|INIT|UNPACK|ACTIVE)"),
     "static mut with check-then-set init. Undefined behavior under threads; "
     "use a const table or an atomic."),
    (re.compile(r"find_handles::<SimpleFileSystem>"),
     "blind first-filesystem lookup. Use LoadedImage.device() or you will mount "
     "the internal NVMe Windows partition instead of the boot stick."),
]

SKIP_DIRS = {
    "target", ".git", "node_modules", ".cargo", ".rustup", "models",
    "hf-venv", "venv", ".claude", ".gemini",
}
# Backup trees are historical evidence, not live code. Substring match so
# both `foo_ALICE_1_0_BACKUP` and `foo_ALICE_1_0_BACKUP_2` style trees skip.
SKIP_SUBSTRINGS = ("_ALICE_1_0_BACKUP",)


ANSI = re.compile(r'\\x1[bB]\[[0-9;]*[A-Za-z]|\\r|\\n|\\t')


def strip_comments(line: str) -> str:
    """Remove // and //! and /// comment tails. Crude but sufficient: we only
    need to avoid flagging documentation of real measurements."""
    i = line.find("//")
    return line[:i] if i >= 0 else line


def strip_ansi(line: str) -> str:
    r"""Drop terminal escape sequences before unit matching.

    `print!("\x1B[2J\x1B[1;1H")` clears the screen. The "2J" in it is not two
    joules. Learned by having the checker flag its own screen-clear.
    """
    return ANSI.sub("", line)


def macro_has_args(line: str) -> bool:
    """True if the macro call passes a format argument after the literal.

    `println!("TPS: {}", tps)` computes. `println!("TPS: 84.6")` does not.
    """
    # crude: look for a comma outside the string literal
    depth = 0
    in_str = False
    esc = False
    for ch in line:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth >= 1:
            return True
    return False


def scan_file(path: str):
    findings = []
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return findings
    for n, raw in enumerate(lines, 1):
        line = strip_comments(raw)
        if not line.strip():
            continue
        for phrase in DENY_PHRASES:
            if phrase.lower() in line.lower():
                findings.append((path, n, f'asserted result: "{phrase}"', raw.strip()))
        clean = strip_ansi(line)
        if MACRO.search(clean) and looks_like_metric(clean) and not macro_has_args(clean):
            findings.append((path, n, "metric printed with no computed argument", raw.strip()))
        for rx, why in REGRESSIONS:
            if rx.search(clean):
                findings.append((path, n, f"regression: {why}", raw.strip()))
    return findings


def scan(root: str):
    """Single code path: walk to the .rs files, then hand each to scan_file so
    there is exactly one place where the rules live. (A duplicated scan loop is
    how the regression rules silently applied to files but not directories.)"""
    if os.path.isfile(root):
        return scan_file(root)
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS
            and not any(s in d for s in SKIP_SUBSTRINGS)
        ]
        # Every .rs file is in scope — benches/, tests/, examples/, build.rs
        # print numbers too. An earlier version only walked src/ and let a
        # hardcoded GMAC/s figure ship in a bench for a week.
        for fn in filenames:
            if fn.endswith(".rs"):
                findings.extend(scan_file(os.path.join(dirpath, fn)))
    return findings


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    findings = scan(root)
    if not findings:
        print("integrity_check: clean — no uncomputed metrics, no asserted results")
        return 0

    print(f"integrity_check: {len(findings)} violation(s)\n")
    for path, line, why, src in findings:
        rel = os.path.relpath(path, root)
        print(f"  {rel}:{line}")
        print(f"      {why}")
        print(f"      {src[:100]}")
        print()
    print("Every number printed must be computed in the same run.")
    print("Delete the code path, or make it measure. There is no third option.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
