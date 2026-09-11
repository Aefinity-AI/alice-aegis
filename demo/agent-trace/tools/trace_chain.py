#!/usr/bin/env python3
"""Independent, inference-free implementation of FORMAT.md §2/§4/§5 (written
clean-room from the spec; not derived from agent_trace.rs). Proves receipt
self-consistency only — see FORMAT.md §7.

Usage:
    trace_chain.py <receipt> [--table FILE]
    trace_chain.py --selftest

Exit codes: 0 = MATCH, 1 = MISMATCH, 2 = FAIL/VERIFY FAIL (structural or
artifact rejection). --selftest exits 0 on n/n PASS, else 1.
"""
import sys
import hashlib
import os

LEAD_KEYS = {
    "AEGIS-TRACE", "model", "embed", "vocab", "K", "N", "prompt-hex",
    "trace-chain", "table-sha256", "suite-sha256", "commit", "host",
}

STEP_KEYS = {"toks", "tool", "in", "out", "decode-chain", "ctx", "q"}
REQUIRED_STEP_KEYS = ("toks", "tool", "in", "out", "decode-chain")


class Fail(Exception):
    """Structural rejection: exit 2, print `FAIL ...` (or `VERIFY FAIL ...`)."""


def fail(msg):
    raise Fail(msg)


def rust_debug(s):
    """Mimics Rust's `{:?}` for &str (used in the reference verifier's
    error messages): double-quoted, backslash/quote-escaped."""
    if s is None:
        return "None"
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return f'"{out}"'


def is_hex64_lower(s):
    return len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def is_hex_even_any_case(s):
    if len(s) % 2 != 0:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in s)


def unhex(s):
    """Mirrors agent_trace.rs::unhex: even length, [0-9a-fA-F] pairs."""
    if not is_hex_even_any_case(s):
        raise ValueError("bad hex")
    return bytes.fromhex(s)


def strict_uint(s, max_val, field_desc):
    """Rust <uN>::from_str: optional leading '+', then ASCII digits only,
    no other whitespace/sign, value <= max_val. No trimming."""
    body = s[1:] if s.startswith("+") else s
    if body == "" or not all(c in "0123456789" for c in body):
        raise ValueError(field_desc)
    v = int(body)
    if v > max_val:
        raise ValueError(field_desc)
    return v


def parse_receipt(text):
    """Structural parse only (FORMAT.md §2). Raises Fail on any structural
    rejection, using the reference verifier's exact wording where the
    verifier itself performs the check chain-only, and this tool's own
    `FAIL structure: missing <key> line` convention for lines the reference
    verifier only detects via artifact/replay checks that need the model
    (see FORMAT.md §2 note after the rejection list, and §7)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    header_pos = None
    for idx, line in enumerate(lines):
        key = line.split(" ", 1)[0] if " " in line else line
        if key == "AEGIS-TRACE":
            if header_pos is None:
                header_pos = idx
            else:
                fail("duplicate AEGIS-TRACE header line")
    if header_pos is None:
        fail("missing AEGIS-TRACE header line")
    if header_pos != 0:
        fail(f"AEGIS-TRACE header line at position {header_pos}, must be first")

    seen_lead = {}
    steps = []
    w_format = None
    step_ordinal = 0

    for line_no, line in enumerate(lines):
        if line.startswith("step "):
            rest = line[len("step "):]
            if ":" not in rest:
                fail(f"step {step_ordinal}: stray token {rust_debug(rest)}")
            label, fields = rest.split(":", 1)
            label = label.strip()
            fields = fields.strip()
            position = step_ordinal
            try:
                label_ok = int(label) == position
            except ValueError:
                label_ok = False
            if not label_ok:
                fail(f"step label {label} at position {position}")

            tokens = fields.split() if fields else []
            step_kv = {}
            for tok in tokens:
                if "=" not in tok:
                    fail(f"step {position}: stray token {rust_debug(tok)}")
                k, v = tok.split("=", 1)
                if k not in STEP_KEYS:
                    fail(f"step {position}: unknown field {rust_debug(k)}")
                if k in step_kv:
                    fail(f"step {position}: duplicate field {rust_debug(k)}")
                step_kv[k] = v
            for req in REQUIRED_STEP_KEYS:
                if req not in step_kv:
                    fail(
                        f"step {position}: missing one of toks= tool= in= "
                        "out= decode-chain="
                    )
            for piece in step_kv["toks"].split(","):
                if piece == "":
                    continue
                try:
                    strict_uint(piece, 2**32 - 1, "toks")
                except ValueError:
                    fail(f"step {position}: bad token id {rust_debug(piece)}")

            steps.append(step_kv)
            step_ordinal += 1
            continue

        if line.startswith("WARNING"):
            rest = line[len("WARNING"):]
            ok = False
            if rest.startswith(" step "):
                after = rest[len(" step "):]
                if ":" in after:
                    num, tail = after.split(":", 1)
                    if tail == " tool argument not found verbatim in context":
                        try:
                            int(num)
                            ok = True
                        except ValueError:
                            ok = False
            if not ok:
                fail(f"malformed WARNING line on line {line_no}")
            continue

        if " " in line:
            key, value = line.split(" ", 1)
        else:
            key, value = line, None

        if key not in LEAD_KEYS:
            fail(f"unknown line key {rust_debug(key)} on line {line_no}")
        if key in seen_lead:
            fail(f"duplicate {key} line")

        if key == "AEGIS-TRACE":
            if value == "v0":
                w_format = 1
            elif value == "v1":
                w_format = 2
            elif value == "v2":
                w_format = 3
            else:
                fail(f"unknown AEGIS-TRACE format {rust_debug(value)}")
        elif key in ("K", "N"):
            try:
                strict_uint(value if value is not None else "", 2**64 - 1, key)
            except ValueError:
                fail(f"malformed {key} {rust_debug(value)}")
        elif key == "prompt-hex":
            v = value if value is not None else ""
            try:
                b = unhex(v)
                b.decode("utf-8")
            except Exception:
                fail("malformed hex in prompt-hex")
        elif key in ("table-sha256", "suite-sha256"):
            v = value if value is not None else ""
            if not is_hex64_lower(v):
                fail(f"malformed {key} (want 64 lowercase hex)")
        elif key == "trace-chain":
            v = value if value is not None else ""
            if not is_hex64_lower(v):
                fail("malformed trace-chain (want 64 lowercase hex)")

        seen_lead[key] = value

    # cross-structural checks (verifier order; see FORMAT.md §2)
    if w_format == 1:
        for i, s in enumerate(steps):
            if "ctx" in s or "q" in s:
                fail(
                    f"receipt declares format 1 but step {i} carries "
                    "ctx=/q= (format-2 downgrade)"
                )
    if w_format is not None and w_format >= 3:
        if "commit" not in seen_lead or "host" not in seen_lead:
            fail("format-3 receipt is missing its commit or host line")
    if "K" in seen_lead:
        w_k = strict_uint(seen_lead["K"], 2**64 - 1, "K")
        if w_k != len(steps):
            fail(f"receipt claims K={w_k} but has {len(steps)} step lines")

    # The reference verifier does not reject these structurally when
    # absent (see FORMAT.md §2's note): a missing model/embed/vocab line
    # instead surfaces as `FAIL artifact: ... hash mismatch`; missing K/N
    # surfaces via header-bounds; missing prompt-hex surfaces as `prompt
    # tokenizes to zero tokens`; missing trace-chain surfaces as a final
    # `VERIFY FAIL` (empty string vs. recomputed hex). Since this tool
    # cannot run those model-dependent checks, it rejects up front with
    # its own `FAIL structure: missing <key> line`, per FORMAT.md §2.
    for required in ("model", "embed", "vocab", "K", "N", "prompt-hex", "trace-chain"):
        if required not in seen_lead:
            fail(f"missing {required} line")

    return w_format, seen_lead, steps


def fold_genesis(w_format, seen_lead, table_path):
    TRACE_DOMAIN = b"AEGIS-TRACE v0\n"
    model_sha = unhex(seen_lead["model"])
    embed_sha = unhex(seen_lead["embed"])
    vocab_sha = unhex(seen_lead["vocab"])
    if len(model_sha) != 32 or len(embed_sha) != 32 or len(vocab_sha) != 32:
        fail("malformed model/embed/vocab hash (want 32 raw bytes)")
    k_val = strict_uint(seen_lead["K"], 2**64 - 1, "K")
    n_val = strict_uint(seen_lead["N"], 2**64 - 1, "N")
    prompt = unhex(seen_lead["prompt-hex"])

    buf = bytearray()
    buf += TRACE_DOMAIN
    buf += model_sha
    buf += embed_sha
    buf += vocab_sha
    buf += k_val.to_bytes(8, "big")
    buf += n_val.to_bytes(8, "big")
    buf += len(prompt).to_bytes(8, "big")
    buf += prompt

    if "table-sha256" in seen_lead:
        declared = seen_lead["table-sha256"]
        if table_path is None:
            short = declared[:16]
            print(
                f"VERIFY FAIL — receipt declares table-sha256 {short} but "
                "no --table was given"
            )
            sys.exit(2)
        with open(table_path, "rb") as tf:
            table_bytes = tf.read()
        actual = hashlib.sha256(table_bytes).hexdigest()
        if actual != declared:
            print(
                "FAIL artifact: TABLE hash mismatch "
                f"(receipt {declared[:16]} vs local {actual[:16]})"
            )
            sys.exit(2)
        buf += unhex(declared)
        buf += len(table_bytes).to_bytes(8, "big")

    if "suite-sha256" in seen_lead:
        buf += b"SUITE"
        buf += unhex(seen_lead["suite-sha256"])

    if w_format is not None and w_format >= 3:
        commit = seen_lead["commit"]
        host = seen_lead["host"]
        if commit == "unknown":
            print("WARNING: receipt commit is unknown (provenance not pinned to code)")
        buf += b"PROV"
        for field in (commit.encode("utf-8"), host.encode("utf-8")):
            buf += len(field).to_bytes(4, "little")
            buf += field

    return hashlib.sha256(bytes(buf)).digest()


def fold_steps(genesis, steps):
    chain = genesis
    for i, s in enumerate(steps):
        step_buf = bytearray()
        step_buf += chain
        step_buf += b"TSTEP"
        step_buf += i.to_bytes(8, "big")
        # FORMAT.md §3 "Undecodable step hex": the reference verifier only
        # sees these as a VERIFY FAIL divergence at replay; a chain-only
        # tool cannot fold what it cannot decode, so it rejects up front.
        try:
            dc = unhex(s["decode-chain"])
        except ValueError:
            fail(f"step {i}: malformed hex in decode-chain")
        step_buf += dc
        name = s["tool"].encode("utf-8")
        try:
            inp = unhex(s["in"]) if s["in"] else b""
        except ValueError:
            fail(f"step {i}: malformed hex in in")
        try:
            out = unhex(s["out"]) if s["out"] else b""
        except ValueError:
            fail(f"step {i}: malformed hex in out")
        for field in (name, inp, out):
            step_buf += len(field).to_bytes(4, "little")
            step_buf += field
        chain = hashlib.sha256(bytes(step_buf)).digest()
    return chain


def run(path, table_path):
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        print("FAIL structure: receipt is not valid UTF-8")
        return 2

    try:
        w_format, seen_lead, steps = parse_receipt(text)
        genesis = fold_genesis(w_format, seen_lead, table_path)
        chain = fold_steps(genesis, steps)
    except Fail as e:
        print(f"FAIL structure: {e}")
        return 2

    got = chain.hex()
    expected = seen_lead["trace-chain"]
    if got == expected:
        print(f"MATCH {got}")
        return 0
    print(f"MISMATCH expected={expected} got={got}")
    return 1


def selftest():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # demo/agent-trace
    expected_path = os.path.join(here, "vectors", "EXPECTED.tsv")
    n = 0
    passed = 0
    with open(expected_path) as f:
        rows = [ln.rstrip("\n").split("\t") for ln in f if ln.strip()]
    header, rows = rows[0], rows[1:]
    for file_stem, table, expect, detail in rows:
        n += 1
        receipt_path = os.path.join(here, "vectors", file_stem + ".txt")
        table_path = None if table == "-" else os.path.join(here, table)
        # capture stdout of run()
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run(receipt_path, table_path)
        out = buf.getvalue().strip()

        ok = False
        if expect == "MATCH":
            ok = out == f"MATCH {detail}" and code == 0
        elif expect == "MISMATCH":
            ok = out.startswith("MISMATCH ") and code == 1
        elif expect == "REJECTED":
            ok = (out == detail) and code == 2

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status} {file_stem}: {out!r}")

    if passed == n:
        print(f"SELFTEST PASS {passed}/{n}")
        return 0
    print(f"SELFTEST FAIL {passed}/{n}")
    return 1


def main():
    args = sys.argv[1:]
    if args == ["--selftest"]:
        sys.exit(selftest())

    table_path = None
    if "--table" in args:
        i = args.index("--table")
        table_path = args[i + 1]
        del args[i : i + 2]
    if len(args) != 1:
        print("usage: trace_chain.py <receipt> [--table FILE]  |  trace_chain.py --selftest")
        sys.exit(2)

    sys.exit(run(args[0], table_path))


if __name__ == "__main__":
    main()
