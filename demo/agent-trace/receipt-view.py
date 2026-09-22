#!/usr/bin/env python3
"""receipt-view — plain-English report for one AEGIS-TRACE agent-episode receipt.

Reads a receipt written by `agent_trace gen` (see demo/agent-trace/README.md
for the format) and prints:

  - what ran: model/embed/vocab artifact hashes, host, commit, K/N, the
    initial prompt, and one line per step (tool called, its input/output).
  - whether the hash chain is intact or broken (by shelling out to the real
    `agent_trace verify` and reading its verdict — this script does not
    reimplement any hashing or replay logic).
  - if broken, the FIRST broken step verify's output points at, and what
    field(s) diverged (token ids, tool name, tool input, tool output, the
    per-step decode-chain digest, or the per-step ctx/query binding).
  - whether the check was done offline: yes, always — `agent_trace verify`
    replays the episode against local artifact files only and makes no
    network call. This script makes none either. A run that fails before
    replay (missing artifacts, wrong artifact hashes, missing binary) is
    reported as NOT verified, not as an offline pass.

This is a thin formatter around `agent_trace verify`'s own stdout. It does
not itself decide PASS/FAIL, does not hash anything, and does not replay
the episode; it only explains, in prose, what the verifier already said.
See demo/agent-trace/README.md for exactly what a PASS does and does not
prove, and demo/agent-trace/tamper/README.md for the adversarial mutation
kit this tool's own README documents four cases against.

Usage:
  receipt-view.py <receipt-file>
    [--model PATH] [--embed PATH] [--vocab PATH] [--artifacts DIR]
    [--table PATH] [--bin PATH]

Same env-var conventions as demo/agent-trace/run.sh:
  AEGIS_ARTIFACTS, AEGIS_MODEL, AEGIS_EMBED, AEGIS_VOCAB, AEGIS_TABLE,
  AEGIS_AGENT_TRACE_BIN
Defaults: AEGIS_ARTIFACTS defaults to the in-repo M7 tinybit model
(model-lab/tinybit/m7_final_gate_work/artifacts), same default as run.sh.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def default_bin():
    return os.environ.get(
        "AEGIS_AGENT_TRACE_BIN",
        os.path.join(ROOT, "aegis-linux", "target", "release", "examples", "agent_trace"),
    )


def default_artifacts_dir():
    return os.environ.get(
        "AEGIS_ARTIFACTS",
        os.path.join(ROOT, "model-lab", "tinybit", "m7_final_gate_work", "artifacts"),
    )


def hexdecode(s):
    """Decode a hex field to text where possible, else return '<n bytes, not UTF-8>'."""
    if not s:
        return ""
    try:
        b = bytes.fromhex(s)
    except ValueError:
        return f"<malformed hex: {s!r}>"
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return f"<{len(b)} bytes, not UTF-8: {s}>"


def parse_step_fields(body):
    fields = {}
    for tok in body.split():
        k, sep, v = tok.partition("=")
        if sep:
            fields[k] = v
    return fields


def parse_receipt(path):
    """Parses only the lead/step line shapes this script needs for display —
    NOT a validating parser. `agent_trace verify` is the source of truth for
    whether the receipt is well-formed; this function is best-effort display
    only and never used to decide PASS/FAIL."""
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        text = f.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    lead = {}
    steps = []
    warnings = []
    for line in lines:
        if line.startswith("step "):
            m = re.match(r"step (\d+):(.*)", line)
            if not m:
                continue
            idx = int(m.group(1))
            fields = parse_step_fields(m.group(2))
            steps.append((idx, fields))
        elif line.startswith("WARNING"):
            warnings.append(line)
        elif line.strip():
            k, sep, v = line.partition(" ")
            if sep:
                lead.setdefault(k, v)
    return lead, steps, warnings


def run_verify(bin_path, model, embed, vocab, receipt, table):
    if not os.path.isfile(bin_path):
        return None, f"agent_trace binary not found at {bin_path} (build it: demo/agent-trace/run.sh build, or set AEGIS_AGENT_TRACE_BIN)"
    for name, path in (("model", model), ("embed", embed), ("vocab", vocab)):
        if not os.path.isfile(path):
            return None, f"missing {name} artifact: {path} (this tool never downloads anything)"
    cmd = [bin_path, "verify", model, embed, vocab, receipt]
    if table:
        cmd += ["--table", table]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "agent_trace verify timed out"
    return p, None


STEP_DIV_RE = re.compile(
    r"^step (\d+) divergence: toks-match=(\w+) tool-match=(\w+) "
    r"in-match=(\w+) out-match=(\w+) decode-chain-match=(\w+)$"
)
CTXQ_RE = re.compile(r"^STEP (\d+) (CTX|QUERY) MISMATCH$")
STEPCOUNT_RE = re.compile(
    r"^VERIFY FAIL — replay produced (\d+) steps, receipt has (\d+)$"
)
WARN_MISMATCH_RE = re.compile(
    r"^VERIFY FAIL — WARNING lines claim steps (\[.*\]), replay derives (\[.*\])$"
)
STRUCTURE_RE = re.compile(r"^FAIL structure: (.*)$")
PASS_RE = re.compile(r"^VERIFY PASS")
GENERIC_FAIL_RE = re.compile(r"^VERIFY FAIL — (.*)$")

FIELD_NAMES = {
    "toks": "the step's token ids",
    "tool": "which tool was called",
    "in": "the tool's recorded input",
    "out": "the tool's recorded output",
    "decode-chain": "the step's decode-chain digest (the model's own chained decode state for that step)",
}


def diagnose(stdout_lines):
    """Scans agent_trace verify's stdout, in the order it was printed (which
    is the order the verifier itself checked things), and returns a dict
    describing the verdict and, on failure, the first broken step it names.
    Returns None fields it cannot determine rather than guessing."""
    result = {
        "verdict": None,        # "PASS" | "FAIL" | None (could not determine)
        "first_broken_step": None,
        "what_changed": None,
        "detail": None,
        "pre_replay": False,    # failed before any per-step replay was attempted
    }
    for line in stdout_lines:
        line = line.rstrip("\n")
        if PASS_RE.match(line):
            result["verdict"] = "PASS"
            return result
        m = STRUCTURE_RE.match(line)
        if m:
            result["verdict"] = "FAIL"
            result["pre_replay"] = True
            result["detail"] = f"receipt is not well-formed: {m.group(1)}"
            return result
        m = STEP_DIV_RE.match(line)
        if m and result["first_broken_step"] is None:
            step, toks_ok, tool_ok, in_ok, out_ok, dchain_ok = m.groups()
            changed = []
            for name, ok in (
                ("toks", toks_ok), ("tool", tool_ok), ("in", in_ok),
                ("out", out_ok), ("decode-chain", dchain_ok),
            ):
                if ok == "false":
                    changed.append(FIELD_NAMES[name])
            result["verdict"] = "FAIL"
            result["first_broken_step"] = int(step)
            result["what_changed"] = changed
            result["detail"] = line
        m = CTXQ_RE.match(line)
        if m and result["first_broken_step"] is None:
            step, kind = m.groups()
            result["verdict"] = "FAIL"
            result["first_broken_step"] = int(step)
            if kind == "CTX":
                result["what_changed"] = [
                    "the exact prompt text (ctx=) the model was fed at that step"
                ]
            else:
                result["what_changed"] = [
                    "the step's query text (q=) — the newly-appended prompt "
                    "material for that step (initial prompt at step 0, the "
                    "previous step's tool result at step 1+)"
                ]
            result["detail"] = line
        m = STEPCOUNT_RE.match(line)
        if m and result["first_broken_step"] is None:
            got, want = m.groups()
            result["verdict"] = "FAIL"
            result["first_broken_step"] = min(int(got), int(want))
            result["what_changed"] = [
                f"the number of steps itself (replay produced {got}, receipt claims {want}) "
                "— a step was added or dropped"
            ]
            result["detail"] = line
        m = WARN_MISMATCH_RE.match(line)
        if m and result["first_broken_step"] is None:
            claimed, derived = m.groups()
            result["verdict"] = "FAIL"
            result["what_changed"] = [
                f"the receipt's WARNING lines ({claimed}) do not match what replay "
                f"independently derives ({derived})"
            ]
            result["detail"] = line
        m = GENERIC_FAIL_RE.match(line)
        if m and result["verdict"] is None:
            # A failure before any per-step comparison was printed at all
            # (e.g. artifact hash mismatch, table/suite mismatch).
            result["verdict"] = "FAIL"
            result["pre_replay"] = True
            result["detail"] = m.group(1)
    return result


def format_report(receipt_path, lead, steps, warnings, verify_proc, diag, model, embed, vocab, verify_error):
    out = []
    out.append(f"receipt: {receipt_path}")
    out.append("")
    out.append("WHAT RAN")
    fmt = lead.get("AEGIS-TRACE", "?")
    out.append(f"  format:       {fmt}")
    if "model" in lead:
        out.append(f"  model hash:   {lead['model'][:16]}...")
    if "embed" in lead:
        out.append(f"  embed hash:   {lead['embed'][:16]}...")
    if "vocab" in lead:
        out.append(f"  vocab hash:   {lead['vocab'][:16]}...")
    if "host" in lead:
        out.append(f"  machine:      {lead['host']}")
    if "commit" in lead:
        out.append(f"  commit:       {lead['commit']}")
    if "K" in lead and "N" in lead:
        out.append(f"  steps:        K={lead['K']} rounds, up to N={lead['N']} tokens decoded per round")
    if "prompt-hex" in lead:
        prompt_text = hexdecode(lead["prompt-hex"])
        shown = prompt_text if len(prompt_text) <= 200 else prompt_text[:200] + "…"
        out.append(f"  prompt:       {shown!r}")
    if "table-sha256" in lead:
        out.append(f"  lookup table: sha256 {lead['table-sha256'][:16]}... (a LOOKUP table was used)")
    if "suite-sha256" in lead:
        out.append(f"  suite hash:   {lead['suite-sha256'][:16]}...")
    out.append("")
    out.append(f"  {len(steps)} step(s) recorded in the receipt:")
    for idx, fields in steps:
        tool = fields.get("tool", "?")
        toks = fields.get("toks", "")
        n_toks = len(toks.split(",")) if toks else 0
        line = f"    step {idx}: {n_toks} token(s) decoded, tool={tool}"
        if tool not in ("no-tool", "?"):
            tin = hexdecode(fields.get("in", ""))
            tout = hexdecode(fields.get("out", ""))
            line += f", called {tin!r} -> {tout!r}"
        out.append(line)
    if warnings:
        out.append("")
        out.append(f"  {len(warnings)} WARNING line(s) in the receipt (non-fatal, see README):")
        for w in warnings:
            out.append(f"    {w}")
    out.append("")
    out.append("HASH CHAIN")
    if verify_error:
        out.append(f"  NOT CHECKED: {verify_error}")
        out.append("")
        out.append("VERIFIED OFFLINE: no — the check could not be run at all")
        return "\n".join(out)

    verdict = diag.get("verdict")
    if verdict == "PASS":
        out.append("  INTACT — agent_trace verify replayed the episode locally and")
        out.append("  reproduced every step and the final trace-chain digest bit-for-bit.")
    elif verdict == "FAIL":
        out.append("  BROKEN — agent_trace verify rejected this receipt.")
        if diag.get("pre_replay"):
            out.append(f"  Reason: {diag.get('detail')}")
            out.append("  (This failure was caught before any step was replayed, so no")
            out.append("  specific step can be named as \"first broken\".)")
        elif diag.get("first_broken_step") is not None:
            step = diag["first_broken_step"]
            out.append(f"  First broken step: step {step}")
            changed = diag.get("what_changed") or []
            if changed:
                out.append("  What changed at that step:")
                for c in changed:
                    out.append(f"    - {c}")
            out.append(f"  (verifier line: {diag.get('detail')})")
        else:
            out.append(f"  Reason: {diag.get('detail')}")
    else:
        out.append("  UNKNOWN — agent_trace verify's output did not match any recognised")
        out.append("  PASS/FAIL pattern this tool knows how to parse. Raw output follows.")
    out.append("")
    if verdict == "PASS":
        out.append("VERIFIED OFFLINE: yes — replayed against local model/embed/vocab files,")
        out.append("no network access used.")
    elif verdict == "FAIL" and not diag.get("pre_replay"):
        out.append("VERIFIED OFFLINE: yes — the check ran fully offline and correctly")
        out.append("rejected this receipt (see HASH CHAIN above).")
    else:
        out.append("VERIFIED OFFLINE: no — verification did not complete (see HASH CHAIN above).")

    if verify_proc is not None and verify_proc.returncode not in (0, 1):
        out.append("")
        out.append(f"  note: agent_trace verify exited {verify_proc.returncode} (not the usual 0=PASS/1=FAIL)")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("receipt")
    ap.add_argument("--model")
    ap.add_argument("--embed")
    ap.add_argument("--vocab")
    ap.add_argument("--artifacts", help="directory with MODEL.SAF/EMBED.BIN/VOCAB.BIN")
    ap.add_argument("--table")
    ap.add_argument("--bin", help="path to the agent_trace binary")
    ap.add_argument("--raw", action="store_true", help="also print agent_trace verify's raw stdout/stderr")
    args = ap.parse_args()

    if not os.path.isfile(args.receipt):
        print(f"no such receipt: {args.receipt}", file=sys.stderr)
        sys.exit(2)

    artifacts_dir = args.artifacts or default_artifacts_dir()
    model = args.model or os.environ.get("AEGIS_MODEL") or os.path.join(artifacts_dir, "MODEL.SAF")
    embed = args.embed or os.environ.get("AEGIS_EMBED") or os.path.join(artifacts_dir, "EMBED.BIN")
    vocab = args.vocab or os.environ.get("AEGIS_VOCAB") or os.path.join(artifacts_dir, "VOCAB.BIN")
    table = args.table or os.environ.get("AEGIS_TABLE")
    bin_path = args.bin or default_bin()

    lead, steps, warnings = parse_receipt(args.receipt)

    proc, verify_error = run_verify(bin_path, model, embed, vocab, args.receipt, table)
    diag = {}
    if proc is not None:
        stdout_lines = proc.stdout.split("\n")
        diag = diagnose(stdout_lines)

    report = format_report(args.receipt, lead, steps, warnings, proc, diag, model, embed, vocab, verify_error)
    print(report)

    if args.raw and proc is not None:
        print("\n--- agent_trace verify raw stdout ---")
        print(proc.stdout, end="")
        if proc.stderr:
            print("--- agent_trace verify raw stderr ---")
            print(proc.stderr, end="")

    if verify_error:
        sys.exit(2)
    if diag.get("verdict") == "FAIL":
        sys.exit(1)
    if diag.get("verdict") != "PASS":
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
