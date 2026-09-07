#!/usr/bin/env python3
"""gen_suite.py — deterministic generator for the tool-call eval suite.

Suite version: trace-eval-suite-v1-2026-09-06
Seed: 20260906

Regenerates demo/agent-trace/eval/suite.tsv (60 items across 8 buckets, see
../../../claudius-maximus/state/reports/2026-09-06-TOOLCALL-EVAL-PLAN.md
section 2) and smoke.tsv (10 items: 2 per bucket from the 5 largest buckets).
Running this script twice must produce byte-identical output files — it uses
only python3 stdlib (random.Random with a fixed seed, deterministic item
order) and prints the sha256 of each file it writes.

Deviation from the plan: section 2 lists CALC hard-bucket ops as
"+ - * //". This tool has no `//` operator (see agent_trace.rs find_calc:
op in {+ - * / %}); this generator uses `/` (truncating, as Rust's
i64::checked_div) and `%` (remainder with the sign of the dividend, as
Rust's i64::checked_rem) instead. See eval/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

SUITE_VERSION = "trace-eval-suite-v1-2026-09-06"
SEED = 20260906

HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE.parent / "tables" / "demo.tsv"
CHAIN_TABLE_PATH = HERE.parent / "tables" / "chain.tsv"

TSV_HEADER = [
    "item_id",
    "bucket",
    "prompt_template_id",
    "prompt_text",
    "expected_tool",
    "expected_input",
    "expected_output",
    "notes",
]

I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


# ---------------------------------------------------------------------
# CALC semantics, mirroring aegis-linux/examples/agent_trace.rs eval_calc
# exactly: i64 checked arithmetic, truncating division (toward zero),
# remainder with the sign of the dividend, div-by-zero and overflow
# reported as the fixed strings "div-by-zero" / "overflow".
# ---------------------------------------------------------------------


def trunc_div(a: int, b: int) -> int:
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def trunc_rem(a: int, b: int) -> int:
    return a - b * trunc_div(a, b)


def eval_calc(a: int, op: str, b: int) -> str:
    """Return the exact string the Rust `calc` tool would record as output
    (decimal result, or "div-by-zero" / "overflow")."""
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    elif op == "/":
        if b == 0:
            return "div-by-zero"
        if a == I64_MIN and b == -1:
            return "overflow"
        r = trunc_div(a, b)
    elif op == "%":
        if b == 0:
            return "div-by-zero"
        if a == I64_MIN and b == -1:
            return "overflow"
        r = trunc_rem(a, b)
    else:
        raise ValueError(f"bad op {op!r}")
    if not (I64_MIN <= r <= I64_MAX):
        return "overflow"
    return str(r)


def calc_call(a: int, op: str, b: int) -> str:
    return f"CALC({a} {op} {b})"


def lookup_call(key: str) -> str:
    return f"LOOKUP({key})"


# ---------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------


def t1_calc_prompt(question: str) -> str:
    """T1: 3-shot Q/A CALC style, exactly matching the working prototype,
    with a 3rd shot added for robustness (plan section 3)."""
    return (
        "Q: 2 + 2\nA: CALC(2 + 2).\n"
        "Q: 10 + 10\nA: CALC(10 + 10).\n"
        "Q: 6 * 7\nA: CALC(6 * 7).\n"
        f"Q: {question}\nA:"
    )


def t2_lookup_prompt(question: str) -> str:
    """T2: 2-shot LOOKUP-flavored analog of T1, Q/A few-shot ending in
    'A:' (prior working style: 'Q: part P-100\\nA: LOOKUP(P-100).')."""
    return (
        "Q: part P-100\nA: LOOKUP(P-100).\n"
        "Q: part P-101\nA: LOOKUP(P-101).\n"
        f"Q: {question}\nA:"
    )


def load_table_keys() -> list[str]:
    keys = []
    for line in TABLE_PATH.read_text().splitlines():
        if not line:
            continue
        key = line.split("\t", 1)[0]
        keys.append(key)
    return keys


def load_table_from(path: Path) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        k, v = line.split("\t", 1)
        table[k] = v
    return table


# ---------------------------------------------------------------------
# Bucket builders. Each returns a list of row dicts. A single RNG instance,
# seeded once, is threaded through in a fixed bucket order so re-running
# this script produces byte-identical output.
# ---------------------------------------------------------------------


def build_calc_easy(rng: random.Random) -> list[dict]:
    rows = []
    ops = ["+", "-", "*"]
    for i in range(1, 16):
        a = rng.randint(0, 20)
        b = rng.randint(0, 20)
        op = rng.choice(ops)
        out = eval_calc(a, op, b)
        rows.append(
            {
                "item_id": f"calc_easy_{i:02d}",
                "bucket": "calc_easy",
                "prompt_template_id": "T1",
                "prompt_text": t1_calc_prompt(f"{a} {op} {b}"),
                "expected_tool": "CALC",
                "expected_input": calc_call(a, op, b),
                "expected_output": out,
                "notes": "a,b in [0,20], ops + - *",
            }
        )
    return rows


def build_calc_hard(rng: random.Random) -> list[dict]:
    rows = []
    ops = ["+", "-", "*", "/", "%"]
    for i in range(1, 16):
        a = rng.randint(21, 999)
        b = rng.randint(21, 999)
        op = rng.choice(ops)
        notes = "a,b in [21,999], ops + - * / % (plan's // does not exist; see README)"
        if op == "-" and i <= 5 and a > b:
            # force a negative result for a few early subtraction items,
            # per plan: "includes a few with negative results".
            a, b = b, a
            notes += "; swapped to force negative result"
        out = eval_calc(a, op, b)
        rows.append(
            {
                "item_id": f"calc_hard_{i:02d}",
                "bucket": "calc_hard",
                "prompt_template_id": "T1",
                "prompt_text": t1_calc_prompt(f"{a} {op} {b}"),
                "expected_tool": "CALC",
                "expected_input": calc_call(a, op, b),
                "expected_output": out,
                "notes": notes,
            }
        )
    return rows


def build_calc_overflow(rng: random.Random) -> list[dict]:
    # Fixed, hand-chosen boundary-adjacent triples (a, op, b): a mix of
    # large-but-in-range products/sums and genuine i64 overflow/div cases.
    # rng is threaded through (unused for values, kept for API symmetry
    # and so bucket order still consumes the same RNG stream as a no-op).
    del rng
    triples = [
        (31623, "*", 31623, "product near 10^9, in-range"),
        (44721, "*", 22360, "product near 10^9, in-range"),
        (999999999, "+", 999999999, "sum near 2e9, in-range"),
        (500000000, "%", 3, "remainder, large operand, in-range"),
        (I64_MAX, "+", 1, "i64::MAX + 1, overflow"),
        (I64_MIN, "-", 1, "i64::MIN - 1, overflow"),
        (I64_MAX, "-", -1, "i64::MAX - (-1), overflow"),
        (I64_MIN, "/", -1, "i64::MIN / -1, overflow"),
    ]
    rows = []
    for i, (a, op, b, note) in enumerate(triples, start=1):
        out = eval_calc(a, op, b)
        rows.append(
            {
                "item_id": f"calc_overflow_{i:02d}",
                "bucket": "calc_overflow",
                "prompt_template_id": "T1",
                "prompt_text": t1_calc_prompt(f"{a} {op} {b}"),
                "expected_tool": "CALC",
                "expected_input": calc_call(a, op, b),
                "expected_output": out,
                "notes": note,
            }
        )
    return rows


def build_lookup_hit(rng: random.Random, keys: list[str], table: dict) -> list[dict]:
    chosen = rng.sample(keys, 8)
    rows = []
    for i, key in enumerate(chosen, start=1):
        rows.append(
            {
                "item_id": f"lookup_hit_{i:02d}",
                "bucket": "lookup_hit",
                "prompt_template_id": "T2",
                "prompt_text": t2_lookup_prompt(f"part {key}"),
                "expected_tool": "LOOKUP",
                "expected_input": lookup_call(key),
                "expected_output": table[key],
                "notes": "key present in demo/agent-trace/tables/demo.tsv",
            }
        )
    return rows


def build_lookup_miss(rng: random.Random, keys: list[str]) -> list[dict]:
    del rng
    candidates = ["P-999", "P-000", "Z-100", "X-1", "part.local"]
    for c in candidates:
        assert c not in keys, f"miss candidate {c} unexpectedly in table"
    rows = []
    for i, key in enumerate(candidates, start=1):
        rows.append(
            {
                "item_id": f"lookup_miss_{i:02d}",
                "bucket": "lookup_miss",
                "prompt_template_id": "T2",
                "prompt_text": t2_lookup_prompt(f"part {key}"),
                "expected_tool": "LOOKUP",
                "expected_input": lookup_call(key),
                "expected_output": "NONE",
                "notes": "well-formed key, absent from table",
            }
        )
    return rows


def build_lookup_near_miss(rng: random.Random, keys: list[str]) -> list[dict]:
    del rng
    # (source real key, mutated near-miss key)
    mutations = [
        ("P-100", "p-100"),  # case-flipped
        ("P-101", "P-102"),  # one digit off (still absent from table)
        ("P-205", "Q-205"),  # one letter off
        ("P-402", "P-4023"),  # one char appended
    ]
    rows = []
    for i, (src, key) in enumerate(mutations, start=1):
        assert key not in keys, f"near-miss key {key} unexpectedly in table"
        rows.append(
            {
                "item_id": f"lookup_near_{i:02d}",
                "bucket": "lookup_near_miss",
                "prompt_template_id": "T2",
                "prompt_text": t2_lookup_prompt(f"part {key}"),
                "expected_tool": "LOOKUP",
                "expected_input": lookup_call(key),
                "expected_output": "NONE",
                "notes": f"one char/case off real key {src}; must resolve as a miss",
            }
        )
    return rows


def build_mixed(rng: random.Random, keys: list[str], table: dict) -> list[dict]:
    calc_triples = [(3, "+", 5), (12, "-", 4), (6, "*", 6)]
    lookup_keys = rng.sample(keys, 3)
    orders = ["calc_first", "lookup_first", "calc_first"]
    rows = []
    for i in range(3):
        a, op, b = calc_triples[i]
        key = lookup_keys[i]
        calc_out = eval_calc(a, op, b)
        lookup_out = table[key]
        if orders[i] == "calc_first":
            # Step 1 prompt primes both a CALC and a LOOKUP shot, ending
            # on a CALC question; step 2 depends on the model continuing
            # into a LOOKUP question after seeing TOOL[calc]=... appended
            # (aegis-linux replay_episode appends tool_result_text and
            # keeps decoding — this eval does not force step 2's text).
            prompt = (
                "Q: 2 + 2\nA: CALC(2 + 2).\n"
                f"Q: part {key}\nA: {lookup_call(key)}.\n"
                f"Q: {a} {op} {b}\nA:"
            )
            template_id = "T1T2"
            expected_tool = "CALC,LOOKUP"
            expected_input = f"{calc_call(a, op, b)},{lookup_call(key)}"
            expected_output = f"{calc_out},{lookup_out}"
        else:
            prompt = (
                f"Q: part P-100\nA: {lookup_call('P-100')}.\n"
                f"Q: 2 + 2\nA: CALC(2 + 2).\n"
                f"Q: part {key}\nA:"
            )
            template_id = "T2T1"
            expected_tool = "LOOKUP,CALC"
            expected_input = f"{lookup_call(key)},{calc_call(a, op, b)}"
            expected_output = f"{lookup_out},{calc_out}"
        rows.append(
            {
                "item_id": f"mixed_{i + 1:02d}",
                "bucket": "mixed",
                "prompt_template_id": template_id,
                "prompt_text": prompt,
                "expected_tool": expected_tool,
                "expected_input": expected_input,
                "expected_output": expected_output,
                "notes": (
                    "K=2 episode; expected_tool/input/output are comma-joined "
                    "step-1,step-2 values (see eval/README.md 'Mixed items')"
                ),
            }
        )
    return rows


def build_chain(table: dict[str, str]) -> list[dict]:
    """"chain" bucket (K=2, NOT part of the default 60-item suite): each
    item's step-1 LOOKUP hits a "superseded" row (P-901..P-906 in
    tables/chain.tsv) whose value text names a second, real key ("Superseded,
    see part P-100"). Unlike `build_mixed`, step 2's question is not primed
    by any few-shot in the initial prompt -- it only exists in the text
    `agent_trace` appends after step 1 runs (`TOOL[lookup]=<row text>`), so a
    step-2 verbatim argument here is a genuine positive case for the
    per-step verbatim rule (see `verbatim_ok_for` in agent_trace.rs and
    eval/README.md 'Chain items'). The two few-shot examples use P-402/P-403
    -- present in chain.tsv but never a chain target or superseded key --
    so the shots never leak either half of any chain answer.
    """
    shot_a, shot_b = "P-402", "P-403"
    rows = []
    for i in range(1, 7):
        key = f"P-90{i}"
        value = table[key]
        # value is exactly "Superseded, see part <target key>"
        target = value.rsplit(" ", 1)[-1]
        prompt = (
            f"Q: part {shot_a}\nA: {lookup_call(shot_a)}.\n"
            f"Q: part {shot_b}\nA: {lookup_call(shot_b)}.\n"
            f"Q: part {key}\nA:"
        )
        rows.append(
            {
                "item_id": f"chain_{i:02d}",
                "bucket": "chain",
                "prompt_template_id": "T2T2",
                "prompt_text": prompt,
                "expected_tool": "LOOKUP,LOOKUP",
                "expected_input": f"{lookup_call(key)},{lookup_call(target)}",
                "expected_output": f"{value},{table[target]}",
                "notes": (
                    f"K=2 chain: step-1 LOOKUP({key}) result names step-2's "
                    f"real question (part {target}); step-2's argument "
                    "appears only in the appended TOOL[lookup]=... text, "
                    "never in the initial prompt (see eval/README.md "
                    "'Chain items')"
                ),
            }
        )
    return rows


def build_distractor(rng: random.Random) -> list[dict]:
    del rng
    rows = [
        {
            "item_id": "distractor_01",
            "bucket": "distractor",
            "prompt_template_id": "T1",
            "prompt_text": t1_calc_prompt("what is the capital of France"),
            "expected_tool": "NONE",
            "expected_input": "",
            "expected_output": "",
            "notes": "looks like the few-shot pattern, no defined tool for the question",
        },
        {
            "item_id": "distractor_02",
            "bucket": "distractor",
            "prompt_template_id": "T1",
            "prompt_text": t1_calc_prompt("two + two"),
            "expected_tool": "NONE",
            "expected_input": "",
            "expected_output": "",
            "notes": "CALC-shaped question with non-integer operands",
        },
    ]
    return rows


BUCKET_ORDER = [
    "calc_easy",
    "calc_hard",
    "calc_overflow",
    "lookup_hit",
    "lookup_miss",
    "lookup_near_miss",
    "mixed",
    "distractor",
]

BUCKET_COUNTS = {
    "calc_easy": 15,
    "calc_hard": 15,
    "calc_overflow": 8,
    "lookup_hit": 8,
    "lookup_miss": 5,
    "lookup_near_miss": 4,
    "mixed": 3,
    "distractor": 2,
}


def build_suite() -> list[dict]:
    rng = random.Random(SEED)
    keys = load_table_keys()
    table = {}
    for line in TABLE_PATH.read_text().splitlines():
        if not line:
            continue
        k, v = line.split("\t", 1)
        table[k] = v

    rows: list[dict] = []
    rows += build_calc_easy(rng)
    rows += build_calc_hard(rng)
    rows += build_calc_overflow(rng)
    rows += build_lookup_hit(rng, keys, table)
    rows += build_lookup_miss(rng, keys)
    rows += build_lookup_near_miss(rng, keys)
    rows += build_mixed(rng, keys, table)
    rows += build_distractor(rng)

    for bucket, count in BUCKET_COUNTS.items():
        n = sum(1 for r in rows if r["bucket"] == bucket)
        assert n == count, f"bucket {bucket}: expected {count} items, built {n}"
    assert len(rows) == 60, f"expected 60 items total, built {len(rows)}"
    return rows


def tsv_escape(val: str) -> str:
    """TSV rows are one-per-line, so a literal newline in prompt_text (every
    few-shot prompt has several) must be escaped. Encoded as the two-byte
    sequence backslash-n; consumers (run_suite.sh) must reverse this before
    writing the prompt to a file for `gen`. Tabs are never expected in any
    field and are a hard error if present."""
    if "\t" in val:
        raise ValueError("field contains a literal tab")
    return val.replace("\\", "\\\\").replace("\n", "\\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ["\t".join(TSV_HEADER)]
    for r in rows:
        lines.append("\t".join(tsv_escape(r[col]) for col in TSV_HEADER))
    path.write_text("\n".join(lines) + "\n")


def build_smoke(rows: list[dict]) -> list[dict]:
    counts_desc = sorted(BUCKET_COUNTS.items(), key=lambda kv: (-kv[1], kv[0]))
    top5 = [b for b, _ in counts_desc[:5]]
    smoke = []
    for bucket in top5:
        bucket_rows = sorted(
            (r for r in rows if r["bucket"] == bucket), key=lambda r: r["item_id"]
        )
        smoke.extend(bucket_rows[:2])
    return smoke


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the tool-call eval suite. With no arguments, writes "
            "suite.tsv and smoke.tsv exactly as before (byte-identical "
            "run to run). --bucket chain writes only the 6-item 'chain' "
            "bucket (not part of the default 60-item suite) to its own "
            "file."
        )
    )
    parser.add_argument(
        "--bucket",
        choices=["chain"],
        default=None,
        help="generate only this extra bucket instead of the default suite",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="table TSV to use for --bucket chain (default: tables/chain.tsv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path for --bucket chain (default: eval/chain.tsv)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(sys.argv[1:])

    if args.bucket == "chain":
        table_path = Path(args.table) if args.table else CHAIN_TABLE_PATH
        out_path = Path(args.out) if args.out else HERE / "chain.tsv"
        table = load_table_from(table_path)
        rows = build_chain(table)
        write_tsv(out_path, rows)
        print(f"{out_path} sha256={sha256_of(out_path)} ({len(rows)} items)")
        return 0

    rows = build_suite()
    suite_path = HERE / "suite.tsv"
    write_tsv(suite_path, rows)

    smoke_rows = build_smoke(rows)
    smoke_path = HERE / "smoke.tsv"
    write_tsv(smoke_path, smoke_rows)

    print(f"suite version: {SUITE_VERSION} seed: {SEED}")
    print(f"{suite_path} sha256={sha256_of(suite_path)} ({len(rows)} items)")
    print(f"{smoke_path} sha256={sha256_of(smoke_path)} ({len(smoke_rows)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
