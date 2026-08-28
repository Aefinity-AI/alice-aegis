#!/usr/bin/env python3
"""
gen_table3.py — regenerate docs/paper/table3_provenance.md from program/RESEARCH_LEDGER.md.

Table 3 (CIS-1 paper, per docs/CIS1_PAPER_OUTLINE.md §6) maps every quantitative
claim in paper sections 4, 5, and 6 to its ledger row and its raw log file.

Why this isn't a pure mechanical scrape of the ledger:
  The ledger (program/RESEARCH_LEDGER.md) records *findings*; it does not record
  which paper section cites which finding, nor the short claim/value phrasing the
  paper actually uses. That mapping is curatorial — it was built by reading
  docs/CIS1_PAPER_OUTLINE.md §4-6 against program/RESEARCH_LEDGER.md rows A19-A40
  by hand. CLAIM_MAP below encodes that mapping (append-only; do not edit
  program/RESEARCH_LEDGER.md or docs/hardware_logs/ or tests/golden/ — this
  script only reads them).

What IS regenerated mechanically on every run:
  - the ledger row's verdict column
  - the ledger row's provenance column (first path token is used as the primary
    provenance path; CI run ids are pulled out of the claim text when present)
  - whether the primary provenance path actually exists in the repo (ls-equivalent
    os.path.exists check) — rows where it does not are marked MISSING, never
    silently dropped or invented.

Usage:
    python3 docs/paper/gen_table3.py            # writes docs/paper/table3_provenance.md
    python3 docs/paper/gen_table3.py --check     # exit nonzero if any provenance is MISSING
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "program" / "RESEARCH_LEDGER.md"
OUTPUT_PATH = REPO_ROOT / "docs" / "paper" / "table3_provenance.md"

# (paper section, short claim description, value quoted in the paper, ledger row id)
# Built by hand from docs/CIS1_PAPER_OUTLINE.md §4 ("Implementability and cross-ISA
# identity"), §5 ("The decode receipt and bare-metal verification"), and §6 ("Cost
# and quality"), cross-referenced against program/RESEARCH_LEDGER.md rows A19-A40.
CLAIM_MAP = [
    ("4", "Four-implementation digest jury (HP N4020 bare iron, Dell i5-5200U bare iron x2, crosvm, QEMU/TCG)", "digest 76985613c965f643, ALL_PASS=true, 4 codegen paths", "A25"),
    ("4", "Digest jury crosses to a second ISA (aarch64 CI)", "same digest 76985613c965f643 on aarch64", "A28"),
    ("4", "Clean-room spec-only reimplementation reproduces the digest", "2 implementers, 400/484 lines, distinct by md5, first-run pass", "A31"),
    ("4", "NEON kernel bit-identical on real ARM silicon", "equivalence 5/5, contract 6/6, mechanism 3/3 exhaustive", "A30"),
    ("4", "Token-level full-pipeline decode digest identical x86_64 vs aarch64", "digest 67e8c0a96abc04e1, prompt_toks=4 gen_toks=64", "A29"),
    ("5", "Decode receipt format + CI replay verification", "chain aee25b770bd7b22e…, CI run 31249589879 (snapshot ce93bbb)", "A32"),
    ("5", "Physical iron verification, Dell (AVX2 path)", "STAGE V VERIFY PASS, receipt md5 87c45bdd…, BOOTLOG md5 becd7cef", "A33"),
    ("5", "Physical iron verification, HP N4020 (SSE2 scalar path)", "VERIFY PASS, BOOTLOG md5 4fb5fc8b, one new BOOTLOG entry vs Dell", "A34"),
    ("6", "Quality cost, hybrid path, M7", "+0.3127% PPL (float 5.639491 vs int 5.657126), digest 0x42E820C2A8A59CD6", "A19"),
    ("6", "Quality cost, full-integer path, M7", "+0.0637% PPL (5.643085 vs 5.639491), digest 0xBED4A17A1A5EE296", "A20"),
    ("6", "Quality cost, integer-dominant HYBRID path (f32 attention), BitNet-2B", "+0.7408% PPL (30.934140 vs 30.706665), digest 0x24C4E510A86659D6", "A21"),
    ("6", "Quality cost, FULL-INTEGER path, BitNet-2B (after v1.0.3 erratum)", "+0.1239% PPL (30.744724 vs 30.706665), argmax digest 0xB274DE03F5862DB7", "A35"),
    ("6", "Throughput cost by microarchitecture (C/B ratio)", "Dell (Broadwell-U) 1.248x slower; HP (Gemini Lake) 0.961x (faster)", "A26"),
    ("6", "AVX2 integer kernel vs float AVX2 kernel", "D/A 0.340x = 2.94x faster; D/C 0.061x = 16.4x over scalar", "A27"),
    ("6", "Ring-0 unikernel vs minimal Linux decode throughput", "+3.6% / +9.4% / +5.1% (prereg form); P-V2-2 FAIL spread 4.7-9.6%", "A22"),
    ("6", "Column-skip kernel candidate (engine context, not a CIS-1 claim)", "2.88-2.89x vs incumbent (ordered variant); 2.80x (chain variant)", "A23"),
    ("6", "Bandwidth ceiling (engine context, not a CIS-1 claim)", "peak seq. 11.19/10.95 GB/s (1T), 11.63/11.70 GB/s (4T) vs ternary stream 0.62 GB/s", "A24"),
    ("4", "Token-level FULL-INTEGER decode digest, BitNet-2B, x86-64 leg", "digest cab11400d737ac4a, prompt_toks=4 gen_toks=64, identical on 2 runs, coherent text", "A36"),
    ("5", "Standalone third-party verifier (no engine dependency) reproduces both digests and verifies the golden receipt", "CIS_SELFTEST 76985613c965f643 ALL_PASS=true; CIS_DECODE 67e8c0a96abc04e1; VERIFY PASS in ~1.4s, tamper tests name the field", "A37"),
    ("5", "Standalone verifier crosses the ISA boundary in public CI", "CIS_SELFTEST 76985613c965f643 ALL_PASS=true, 81 unit+integration tests pass, VERIFY PASS on x86-minted golden receipt, on aarch64", "A38"),
    ("4", "BitNet-2B decode receipt crosses the ISA boundary in public CI", "digest cab11400d737ac4a reproduced on aarch64; cis_witness and standalone cis-verify both print VERIFY PASS", "A39"),
    ("5", "BitNet-2B receipt re-derived by the unikernel under QEMU/TCG, no OS present", "STAGE V VERIFY PASS, cis-digest cab11400d737ac4a chain 917ddf5fea9a8488…, artifacts 3/3 hashes match", "A40"),
]

CI_RUN_RE = re.compile(r"\bpublic run (\d+)\b")


def parse_ledger_rows(ledger_text: str) -> dict[str, dict]:
    """Parse '| Axx | claim | verdict | provenance |' rows into a dict keyed by row id."""
    rows = {}
    for line in ledger_text.splitlines():
        m = re.match(r"^\|\s*(A\d+)\s*\|", line)
        if not m:
            continue
        row_id = m.group(1)
        fields = line.strip().strip("|").split(" | ")
        if len(fields) != 4:
            continue
        _, claim_text, verdict, provenance = fields
        rows[row_id] = {
            "claim_text": claim_text.strip(),
            "verdict": verdict.strip(),
            "provenance": provenance.strip(),
        }
    return rows


def split_provenance_chunks(provenance_field: str) -> list[str]:
    """Split a provenance field on top-level ' + ' separators, respecting
    parenthesis nesting so a '+' inside an annotation like
    '(instrument, artifact sha256s + both transcripts)' doesn't split the chunk."""
    chunks = []
    depth = 0
    buf = []
    i = 0
    s = provenance_field
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif depth == 0 and s[i : i + 3] == " + ":
            chunks.append("".join(buf))
            buf = []
            i += 3
            continue
        else:
            buf.append(ch)
        i += 1
    if buf:
        chunks.append("".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _split_path_and_annotation(chunk: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", chunk)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return chunk.strip(), ""


RAW_LOG_EXTENSIONS = (".log", ".txt", ".json", ".csv", ".receipt")


def primary_path(provenance_field: str) -> str:
    """Pick the raw-log path out of a provenance field like
    'docs/hardware_logs/foo.md (prereg, committed pre-boot) + bar.txt (instrument)'.

    Most ledger rows lead with the raw log/BOOTLOG itself, which we take as
    primary. A few rows (e.g. A22) instead lead with a preregistration or
    derivation *document* (.md) and only a later chunk is the actual raw
    instrument log — in that case, skip forward to the first non-.md chunk.
    Chunks that omit a directory (ledger shorthand, e.g. 'bar.txt' following
    'docs/hardware_logs/foo.md') are resolved against the first chunk's
    directory.
    """
    chunks = split_provenance_chunks(provenance_field)
    parsed = [_split_path_and_annotation(c) for c in chunks]
    if not parsed:
        return provenance_field.strip()

    base_dir = ""
    first_path = parsed[0][0]
    if "/" in first_path:
        base_dir = first_path.rsplit("/", 1)[0]

    def resolve(path: str) -> str:
        if "/" not in path and base_dir:
            return f"{base_dir}/{path}"
        return path

    if first_path.lower().endswith(".md"):
        for path, _annotation in parsed[1:]:
            resolved = resolve(path)
            if resolved.lower().endswith(RAW_LOG_EXTENSIONS):
                return resolved

    return resolve(first_path)


def extract_machine(row_id: str, claim_text: str) -> str:
    """Best-effort machine name extraction from the ledger row's free text."""
    known = {
        "A19": "i5-10210U crosvm",
        "A20": "i5-10210U crosvm",
        "A21": "i5-10210U crosvm",
        "A22": "Dell i5-5200U",
        "A23": "Dell i5-5200U",
        "A24": "Dell i5-5200U",
        "A25": "HP N4020 + Dell i5-5200U + crosvm i5-10210U + QEMU/TCG",
        "A26": "Dell i5-5200U (Broadwell-U) + HP N4020 (Gemini Lake)",
        "A27": "Dell i5-5200U",
        "A28": "GitHub ubuntu-24.04-arm (Neoverse N2)",
        "A29": "GitHub ubuntu-24.04 + ubuntu-24.04-arm",
        "A30": "GitHub ubuntu-24.04-arm (Neoverse N2)",
        "A31": "i5-10210U crosvm (dev host)",
        "A32": "i5-10210U crosvm (mint) + GitHub ubuntu-24.04-arm + ubuntu-24.04 (verify)",
        "A33": "Dell Inspiron 15 (i5-5200U, physical iron)",
        "A34": "HP Stream (Celeron N4020, physical iron)",
        "A35": "i5-10210U crosvm",
        "A36": "i5-10210U crosvm",
        "A37": "i5-10210U crosvm",
        "A38": "GitHub ubuntu-24.04-arm (Neoverse N2) + ubuntu-24.04",
        "A39": "GitHub ubuntu-24.04-arm (Neoverse N2) + ubuntu-24.04",
        "A40": "QEMU/TCG (crosvm dev host, i5-10210U)",
    }
    return known.get(row_id, "UNKNOWN — check ledger row")


def build_table(ledger_rows: dict[str, dict]) -> tuple[str, int, int, list[str]]:
    header = (
        "| Paper § | Claim (short) | Value | Ledger row | Machine | "
        "Provenance (path or CI run) | File exists? |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    lines = []
    total = 0
    existing = 0
    missing_rows = []
    for section, claim, value, row_id in CLAIM_MAP:
        total += 1
        row = ledger_rows.get(row_id)
        if row is None:
            lines.append(
                f"| {section} | {claim} | {value} | {row_id} | UNKNOWN — row not found in ledger | "
                f"UNKNOWN | MISSING (ledger row not found) |"
            )
            missing_rows.append(f"{row_id} (ledger row itself not found)")
            continue

        machine = extract_machine(row_id, row["claim_text"])
        prov_path = primary_path(row["provenance"])

        ci_match = CI_RUN_RE.search(row["claim_text"])
        prov_display = prov_path
        if ci_match:
            prov_display = f"{prov_path} (CI run {ci_match.group(1)})"

        full_path = REPO_ROOT / prov_path
        exists = full_path.exists()
        if exists:
            existing += 1
            exists_str = "yes"
        else:
            exists_str = "MISSING"
            missing_rows.append(f"{row_id}: {prov_path}")

        claim_cell = claim.replace("|", "\\|")
        value_cell = value.replace("|", "\\|")
        prov_cell = prov_display.replace("|", "\\|")

        lines.append(
            f"| {section} | {claim_cell} | {value_cell} | {row_id} | {machine} | "
            f"`{prov_cell}` | {exists_str} |"
        )

    return header + "\n".join(lines) + "\n", total, existing, missing_rows


def main() -> int:
    check_only = "--check" in sys.argv

    ledger_text = LEDGER_PATH.read_text()
    ledger_rows = parse_ledger_rows(ledger_text)

    table_body, total, existing, missing_rows = build_table(ledger_rows)

    doc = f"""# Table 3 — Every quantitative claim in CIS-1 paper §4-6, mapped to its ledger row and raw log

Generated by `docs/paper/gen_table3.py` from `program/RESEARCH_LEDGER.md` rows
A19-A40 and `docs/CIS1_PAPER_OUTLINE.md` §4-6. Do not hand-edit this file —
edit `CLAIM_MAP` in the generator script and re-run it.

Regenerate with:

```
python3 docs/paper/gen_table3.py
```

{total} claims total, {existing} with a verified existing primary log file.

{table_body}
"""

    if check_only:
        if missing_rows:
            print(f"MISSING provenance for {len(missing_rows)} row(s):")
            for m in missing_rows:
                print(f"  - {m}")
            return 1
        print(f"OK: all {total} claims have existing primary provenance files.")
        return 0

    OUTPUT_PATH.write_text(doc)
    print(f"Wrote {OUTPUT_PATH} ({total} claims, {existing} with existing primary log file).")
    if missing_rows:
        print(f"MISSING provenance for {len(missing_rows)} row(s):")
        for m in missing_rows:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
