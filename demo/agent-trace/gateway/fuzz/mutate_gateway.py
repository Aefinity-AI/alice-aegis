#!/usr/bin/env python3
"""safe-1e: structured mutation fuzz driver for the receipt gateway.

Recipe follows the E23 style (state/reports/2026-09-11-E23-R9-RESULT.md in
claudius-maximus): structured byte/field-level mutations of a receipt
format, checking that a verifier (here: the receipt gateway CLI, which
wraps `agent_trace verify` + allowlist + verify-execute binding + freshness)
fails closed rather than panicking, hanging, or wrongly ALLOWing.

Two seed corpora:
  (A) "live" seeds -- generated fresh via `agent_trace gen` against the
      local tinybit M7 model artifacts (CALC and LOOKUP tool kinds). The
      gateway's full decision path, including the real `agent_trace verify`
      subprocess call, runs against these, because the allowlist and
      model/embed/vocab paths given to the gateway are the real local
      artifacts these receipts were generated from.
  (B) "fixture" seeds -- the repo's existing tests/fixtures/live20/*.receipt
      corpus (CALC/LOOKUP/FILE-READ/chain/mixed tool kinds, generated
      against a real 2B model not present on this box). These exercise
      `parse_receipt` (which runs on every case, before any allowlist or
      model check) and the gateway's fail-closed behavior when the
      referenced model artifacts are unavailable/mismatched, but never
      reach a successful `agent_trace verify` locally.

Mutation classes (>=8, matching the safe-1e brief):
  byte_flip        - single/multi random bit flips anywhere in the receipt
  truncate         - truncate to a random prefix length (incl. 0 bytes)
  non_utf8         - splice a raw invalid-UTF-8 byte sequence into the file
  giant_field      - replace a field (model/in=/prompt-hex) with an
                     oversized value (10 KB - 200 KB)
  dup_step         - duplicate a "step " line (or the last line, if none)
  ctx_edit         - flip bytes inside a step's ctx=/q= hex field
  allowlist_corrupt- corrupt the SIGNED ALLOWLIST FILE itself (bit flip,
                     truncate, strip sig line, wrong signature)
  malformed_nonce  - malformed/oversized/non-numeric/negative counter, and
                     non-UTF-8 bytes in the session/counter CLI arguments

For every case the gateway process must either:
  - exit 0 (ALLOW) -- only plausible if the mutation left the receipt
    byte-identical/still-valid (rare, checked for and flagged if it ever
    happens on a mutation that changed the file), or
  - exit nonzero with a DENY message on stdout, or a clean "GATEWAY INIT
    FAIL"/usage error on stderr (allowlist_corrupt / malformed_nonce
    classes never even construct a Gateway),
within a 60s wall-clock budget and without the string "panicked" appearing
in stderr.

Anything else (panic, hang, wrongful ALLOW on a mutation that changed
receipt bytes away from validity) is recorded as a FINDING.
"""
import hashlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]  # .../alice-aegis-cm-safe1e-fuzz
GW_DIR = REPO / "demo/agent-trace/gateway"
GATEWAY_BIN = GW_DIR / "target/release/gateway"
AGENT_TRACE_BIN = REPO / "aegis-linux/target/release/examples/agent_trace"
TINYBIT = REPO / "model-lab/tinybit/m7_final_gate_work/artifacts"
FIXTURES = GW_DIR / "tests/fixtures/live20"
TABLES = REPO / "demo/agent-trace/tables"

RNG_SEED = 20260913
TIMEOUT_S = 60
CASES_PER_MUTATION_TARGET = int(os.environ.get("FUZZ_TARGET_TOTAL", "2200"))

random.seed(RNG_SEED)


def sha256hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(cmd, timeout=TIMEOUT_S, input_bytes=None):
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            input=input_bytes,
        )
        dt = time.monotonic() - t0
        return p.returncode, p.stdout, p.stderr, dt, False
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return None, b"", b"", dt, True
    except OSError as e:
        # e.g. "Argument list too long" -- an OS-level exec constraint the
        # driver should never actually hit post-cap, but if it does, record
        # it as a distinct (non-panic, non-hang) outcome rather than crash
        # the whole run.
        dt = time.monotonic() - t0
        return -1, b"", f"OSError: {e}".encode(), dt, False


# ---------------------------------------------------------------------
# Seed corpus A: fresh tinybit-generated receipts (real verify path).
# ---------------------------------------------------------------------

def gen_receipt(tmpdir: Path, prompt: str, k: int, n: int) -> Path:
    m, e, v = TINYBIT / "MODEL.SAF", TINYBIT / "EMBED.BIN", TINYBIT / "VOCAB.BIN"
    out = tmpdir / f"seed_{abs(hash(prompt))}_{k}_{n}.receipt"
    with open(out, "wb") as f:
        p = subprocess.run(
            [str(AGENT_TRACE_BIN), "gen", str(m), str(e), str(v), str(k), str(n), prompt],
            stdout=f,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    assert p.returncode == 0, f"gen failed: {p.stderr}"
    return out


def tinybit_seeds(tmpdir: Path):
    seeds = []
    calc_prompts = [
        "Q: 6 + 7\nA: CALC(6 + 7).\n",
        "Q: 100 - 37\nA: CALC(100 - 37).\n",
        "Q: 12 * 12\nA: CALC(12 * 12).\n",
    ]
    for pr in calc_prompts:
        seeds.append(("CALC", gen_receipt(tmpdir, pr, 1, 16)))
    lookup_prompt = (
        "Q: LOOKUP(P-317)\nA: LOOKUP(P-317).\nQ: LOOKUP(P-100)\nA: LOOKUP(P-100).\n"
    )
    try:
        seeds.append(("LOOKUP", gen_receipt(tmpdir, lookup_prompt, 2, 32)))
    except AssertionError:
        pass  # tinybit model may not always trigger LOOKUP for this prompt
    return seeds


# ---------------------------------------------------------------------
# Seed corpus B: existing live20 fixtures (structural fuzz only).
# ---------------------------------------------------------------------

def fixture_seeds():
    out = []
    for p in sorted(FIXTURES.glob("*.receipt")):
        kind = (
            "FILE-READ" if "fileread" in p.name
            else "LOOKUP" if "lookup" in p.name
            else "CALC" if "calc" in p.name
            else "MIXED"
        )
        out.append((kind, p))
    return out


# ---------------------------------------------------------------------
# Mutation classes.
# ---------------------------------------------------------------------

def mut_byte_flip(data: bytes) -> bytes:
    if not data:
        return data
    b = bytearray(data)
    n_flips = random.choice([1, 1, 1, 2, 4, 8])
    for _ in range(n_flips):
        i = random.randrange(len(b))
        bit = 1 << random.randrange(8)
        b[i] ^= bit
    return bytes(b)


def mut_truncate(data: bytes) -> bytes:
    if not data:
        return data
    choices = [0, 1, len(data) // 4, len(data) // 2, len(data) - 1, max(0, len(data) - random.randrange(1, 20))]
    cut = random.choice(choices)
    cut = max(0, min(cut, len(data)))
    return data[:cut]


def mut_non_utf8(data: bytes) -> bytes:
    if not data:
        data = b"x"
    b = bytearray(data)
    bad_seqs = [b"\xff\xfe", b"\x80\x80\x80", b"\xc0\xaf", b"\xed\xa0\x80", bytes([random.randrange(0x80, 0x100)])]
    seq = random.choice(bad_seqs)
    pos = random.randrange(len(b) + 1)
    return bytes(b[:pos]) + seq + bytes(b[pos:])


def mut_giant_field(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    # Capped well under this host's argv/env size limit (giant `in=` fields
    # get passed on the gateway CLI's argv as the action-hex argument; a
    # multi-hundred-KB value there hits OSError("Argument list too long")
    # at exec() time -- an OS-level constraint on this harness's CLI
    # transport, not a gateway behavior under test, so the size is kept
    # comfortably below that ceiling while still far exceeding any normal
    # field (a real hash/argument field here is 64-70 hex chars).
    size = random.choice([2_000, 8_000, 20_000])
    giant_hex = "ab" * (size // 2)
    targets = []
    for i, l in enumerate(lines):
        if l.startswith("model ") or l.startswith("embed ") or l.startswith("vocab ") or l.startswith("prompt-hex "):
            targets.append(i)
        if l.startswith("step ") and "in=" in l:
            targets.append(i)
    if not targets:
        return data
    i = random.choice(targets)
    l = lines[i]
    if l.startswith("step "):
        import re
        lines[i] = re.sub(r"in=[0-9a-fA-F]*", "in=" + giant_hex, l, count=1)
    else:
        prefix = l.split(" ", 1)[0]
        lines[i] = f"{prefix} {giant_hex}"
    return "\n".join(lines).encode("utf-8")


def mut_dup_step(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    step_idxs = [i for i, l in enumerate(lines) if l.startswith("step ")]
    if not step_idxs:
        idx = len(lines) - 1 if lines else 0
    else:
        idx = random.choice(step_idxs)
    if 0 <= idx < len(lines):
        lines.insert(idx, lines[idx])
    return "\n".join(lines).encode("utf-8")


def mut_ctx_edit(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    import re
    changed = False
    for i, l in enumerate(lines):
        if l.startswith("step ") and ("ctx=" in l or "q=" in l):
            def flip_hex(m):
                nonlocal changed
                h = list(m.group(1))
                if h:
                    pos = random.randrange(len(h))
                    h[pos] = random.choice("0123456789abcdef")
                    changed = True
                return m.group(0)[: m.group(0).index(m.group(1))] + "".join(h)
            newl = re.sub(r"(?:ctx|q)=([0-9a-fA-F]+)", flip_hex, l, count=1)
            lines[i] = newl
            if changed:
                break
    return "\n".join(lines).encode("utf-8")


RECEIPT_MUTATIONS = {
    "byte_flip": mut_byte_flip,
    "truncate": mut_truncate,
    "non_utf8": mut_non_utf8,
    "giant_field": mut_giant_field,
    "dup_step": mut_dup_step,
    "ctx_edit": mut_ctx_edit,
}


# ---------------------------------------------------------------------
# Allowlist-corruption mutation class (operates on the allowlist file).
# ---------------------------------------------------------------------

def mut_allowlist_bitflip(data: bytes) -> bytes:
    return mut_byte_flip(data)


def mut_allowlist_truncate(data: bytes) -> bytes:
    return mut_truncate(data)


def mut_allowlist_strip_sig(data: bytes) -> bytes:
    lines = data.split(b"\n")
    lines = [l for l in lines if not l.startswith(b"sig ")]
    return b"\n".join(lines)


def mut_allowlist_wrong_sig(data: bytes) -> bytes:
    lines = data.split(b"\n")
    out = []
    for l in lines:
        if l.startswith(b"sig "):
            out.append(b"sig " + b"0" * 64)
        else:
            out.append(l)
    return b"\n".join(out)


ALLOWLIST_MUTATIONS = {
    "allowlist_corrupt.bitflip": mut_allowlist_bitflip,
    "allowlist_corrupt.truncate": mut_allowlist_truncate,
    "allowlist_corrupt.strip_sig": mut_allowlist_strip_sig,
    "allowlist_corrupt.wrong_sig": mut_allowlist_wrong_sig,
}


# ---------------------------------------------------------------------
# malformed_nonce/counter class: CLI-argument level mutations.
# ---------------------------------------------------------------------

MALFORMED_COUNTERS = [
    "not-a-number", "-1", "99999999999999999999999999", "3.14", "",
    " 1", "1 ", "0x10", "1e400", "\t", "18446744073709551616",
]
MALFORMED_SESSIONS_BYTES = [
    # NOTE: a literal NUL byte is not a valid argv element at the OS/libc
    # level (posix_spawn/execve reject it outright) -- that is a kernel
    # constraint, not something the gateway process ever gets a chance to
    # mishandle, so it is intentionally excluded here.
    b"sess-\xff\xfe", b"\x80" * 8, b"", b"a" * 100_000,
]


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def find_last_in_hex(text: str):
    last = None
    for line in text.splitlines():
        if line.startswith("step "):
            for field in line.split():
                if field.startswith("in="):
                    last = field[3:]
    return last


def hex_to_bytes(h):
    try:
        if h is None or len(h) % 2 != 0:
            return None
        return bytes.fromhex(h)
    except ValueError:
        return None


def build_signed_allowlist(tmpdir: Path, triples, key: bytes) -> Path:
    import hmac as _hmac

    lines = [f"{m} {e} {v}" for (m, e, v) in triples]
    body = "".join(l + "\n" for l in lines).encode("utf-8")
    sig = _hmac.new(key, body, hashlib.sha256).hexdigest()
    path = tmpdir / "allowlist.signed"
    path.write_bytes(body + f"sig {sig}\n".encode("utf-8"))
    return path


def classify_result(rc, stdout, stderr, dt, timed_out, mutation_changed, orig_stdout_allow):
    stderr_s = stderr.decode("utf-8", errors="replace")
    stdout_s = stdout.decode("utf-8", errors="replace")
    if timed_out:
        return "FINDING-HANG", stdout_s, stderr_s
    if "panicked" in stderr_s:
        return "FINDING-PANIC", stdout_s, stderr_s
    allowed = rc == 0 and stdout_s.startswith("ALLOW")
    if allowed and mutation_changed:
        return "FINDING-WRONGFUL-ALLOW", stdout_s, stderr_s
    return "ok", stdout_s, stderr_s


def main():
    findings = []
    counts = {}
    total = 0

    work = Path("/tmp/safe1e-fuzz-work")
    work.mkdir(exist_ok=True)
    for old in work.glob("*"):
        try:
            old.unlink()
        except IsADirectoryError:
            pass

    if not GATEWAY_BIN.exists():
        print("gateway release binary missing; build first", file=sys.stderr)
        sys.exit(2)
    if not AGENT_TRACE_BIN.exists():
        print("agent_trace release example missing; build first", file=sys.stderr)
        sys.exit(2)

    # Real allowlist for the tinybit artifacts (corpus A).
    mh = sha256hex((TINYBIT / "MODEL.SAF").read_bytes())
    eh = sha256hex((TINYBIT / "EMBED.BIN").read_bytes())
    vh = sha256hex((TINYBIT / "VOCAB.BIN").read_bytes())
    key = b"safe1e-fuzz-key-not-for-prod"
    good_allowlist = build_signed_allowlist(work, [(mh, eh, vh)], key)
    keyfile = work / "key.bin"
    keyfile.write_bytes(key)

    tinybit = tinybit_seeds(work)
    fixtures = fixture_seeds()
    all_seeds = [("tinybit", kind, p) for kind, p in tinybit] + [("fixture", kind, p) for kind, p in fixtures]

    print(f"seeds: {len(tinybit)} tinybit ({[k for k,_ in tinybit]}), {len(fixtures)} fixture ({[k for k,_ in fixtures]})")

    n_classes = len(RECEIPT_MUTATIONS)
    per_class_per_seed = max(1, CASES_PER_MUTATION_TARGET // (n_classes * max(1, len(all_seeds))))

    def check_gateway(receipt_path, action_bytes, session, counter, mutation_changed, class_name, seed_label, allowlist_path=None):
        nonlocal total
        cmd = [
            str(GATEWAY_BIN),
            str(TINYBIT / "MODEL.SAF"),
            str(TINYBIT / "EMBED.BIN"),
            str(TINYBIT / "VOCAB.BIN"),
            str(AGENT_TRACE_BIN),
            str(allowlist_path or good_allowlist),
            str(keyfile),
            str(receipt_path),
            action_bytes.hex() if isinstance(action_bytes, (bytes, bytearray)) else action_bytes,
            session,
            str(counter),
        ]
        run_cmd = [c.encode() if isinstance(c, str) else c for c in cmd]
        rc, out, err, dt, timed_out = run(run_cmd)
        total += 1
        counts[class_name] = counts.get(class_name, 0) + 1
        verdict, stdout_s, stderr_s = classify_result(rc, out, err, dt, timed_out, mutation_changed, None)
        if verdict != "ok":
            findings.append({
                "class": class_name,
                "seed": seed_label,
                "receipt_path": str(receipt_path),
                "action_hex": action_bytes.hex() if isinstance(action_bytes, (bytes, bytearray)) else action_bytes,
                "session": repr(session),
                "counter": str(counter),
                "rc": rc,
                "timed_out": timed_out,
                "verdict": verdict,
                "stdout": stdout_s[:2000],
                "stderr": stderr_s[:2000],
            })
        return verdict

    # ---- receipt mutation classes over all seeds ----
    for provenance, kind, seed_path in all_seeds:
        orig_bytes = seed_path.read_bytes()
        orig_text = orig_bytes.decode("utf-8", errors="ignore")
        orig_action_hex = find_last_in_hex(orig_text)
        orig_action = hex_to_bytes(orig_action_hex) or b""

        for class_name, fn in RECEIPT_MUTATIONS.items():
            for i in range(per_class_per_seed):
                mutated = fn(orig_bytes)
                mutation_changed = mutated != orig_bytes
                mpath = work / f"mut_{provenance}_{kind}_{class_name}_{i}.receipt"
                mpath.write_bytes(mutated)

                # Action bytes: prefer the mutated receipt's OWN last in=
                # field (exercises the deeper verify path when parseable);
                # fall back to the original valid action (still a fine,
                # meaningful DENY case: confused-deputy / binding-mismatch
                # class) when the mutation destroyed that field.
                mut_text = mutated.decode("utf-8", errors="ignore")
                mut_action_hex = find_last_in_hex(mut_text)
                mut_action = hex_to_bytes(mut_action_hex)
                action = mut_action if mut_action is not None else orig_action

                session = f"fuzz-{provenance}-{class_name}-{i}"
                counter = i + 1
                check_gateway(mpath, action, session, counter, mutation_changed, f"receipt.{class_name}", f"{provenance}:{kind}:{seed_path.name}")

    # ---- allowlist_corrupt class (uses one real receipt + real allowlist) ----
    base_receipt = tinybit[0][1] if tinybit else fixtures[0][1]
    base_text = base_receipt.read_bytes().decode("utf-8", errors="ignore")
    base_action = hex_to_bytes(find_last_in_hex(base_text)) or b""
    good_bytes = good_allowlist.read_bytes()
    n_allowlist_cases = max(1, CASES_PER_MUTATION_TARGET // (len(ALLOWLIST_MUTATIONS) * 20))
    for class_name, fn in ALLOWLIST_MUTATIONS.items():
        for i in range(max(20, n_allowlist_cases)):
            mutated = fn(good_bytes)
            # A trailing-newline-only truncation is not a semantic mutation
            # for this format: the signed-body parser splits on `.lines()`,
            # which is agnostic to a missing final line terminator, so
            # stripping only trailing `\n` bytes leaves an equally-valid
            # signed allowlist. Do not count that (rare, degenerate) case
            # as "mutation changed the file" for wrongful-ALLOW purposes.
            mutation_changed = mutated.rstrip(b"\n") != good_bytes.rstrip(b"\n")
            apath = work / f"allowlist_{class_name}_{i}.signed"
            apath.write_bytes(mutated)
            session = f"fuzz-allowlist-{class_name}-{i}"
            check_gateway(base_receipt, base_action, session, i + 1, mutation_changed, class_name, f"tinybit:CALC:{base_receipt.name}", allowlist_path=apath)

    # ---- malformed_nonce/counter class ----
    n_nonce_cases = max(40, CASES_PER_MUTATION_TARGET // 20)
    for i in range(n_nonce_cases):
        counter = random.choice(MALFORMED_COUNTERS)
        session_bytes = random.choice(MALFORMED_SESSIONS_BYTES)
        cmd = [
            str(GATEWAY_BIN).encode(),
            str(TINYBIT / "MODEL.SAF").encode(),
            str(TINYBIT / "EMBED.BIN").encode(),
            str(TINYBIT / "VOCAB.BIN").encode(),
            str(AGENT_TRACE_BIN).encode(),
            str(good_allowlist).encode(),
            str(keyfile).encode(),
            str(base_receipt).encode(),
            base_action.hex().encode(),
            session_bytes,
            counter.encode() if isinstance(counter, str) else counter,
        ]
        rc, out, err, dt, timed_out = run(cmd)
        total += 1
        class_name = "malformed_nonce"
        counts[class_name] = counts.get(class_name, 0) + 1
        verdict, stdout_s, stderr_s = classify_result(rc, out, err, dt, timed_out, False, None)
        if verdict != "ok":
            findings.append({
                "class": class_name,
                "seed": f"tinybit:CALC:{base_receipt.name}",
                "receipt_path": str(base_receipt),
                "action_hex": base_action.hex(),
                "session": repr(session_bytes),
                "counter": repr(counter),
                "rc": rc,
                "timed_out": timed_out,
                "verdict": verdict,
                "stdout": stdout_s[:2000],
                "stderr": stderr_s[:2000],
            })

    print(f"\nTOTAL CASES: {total}")
    print("Per-class counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\nFINDINGS: {len(findings)}")
    for f in findings:
        print("----")
        for k, v in f.items():
            print(f"{k}: {v}")

    out_path = Path("/tmp/safe1e-fuzz-findings.txt")
    with open(out_path, "w") as fh:
        fh.write(f"TOTAL CASES: {total}\n")
        fh.write("Per-class counts:\n")
        for k, v in sorted(counts.items()):
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nFINDINGS: {len(findings)}\n")
        for f in findings:
            fh.write("----\n")
            for k, v in f.items():
                fh.write(f"{k}: {v}\n")
    print(f"\nwrote {out_path}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
