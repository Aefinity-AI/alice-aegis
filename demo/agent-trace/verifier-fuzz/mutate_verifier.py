#!/usr/bin/env python3
"""safe-1f: structured mutation fuzz driver for the receipt v2/v3 VERIFIER
itself -- the bare `agent_trace verify` CLI (aegis-linux/examples/agent_trace.rs),
not the gateway wrapper (that was safe-1e's target).

Style/lineage: same recipe as
demo/agent-trace/gateway/fuzz/mutate_gateway.py (safe-1e), adapted for the
verifier's actual CLI signature:

    agent_trace verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> receipt... \
        [--table <path>] [--suite-sha256 <64hex>] [--expect-ctx <64hex>] [--fail-fast]

Two mandated seed corpora (per the safe-1f QUEUE item):
  (A) eval-60-2b-T1-box1 -- 60 real-2B-model receipts, format "AEGIS-TRACE
      v3" (item-ctx bound). NOTE: the literal directory
      eval/receipts/eval-60-2b-T1-box1/ in this repo only has stray
      *.gen.err/*.verify.err logs committed -- the receipt bodies were never
      checked in (they point at a since-renamed sibling repo path in
      summary.tsv). The sibling directory eval-60-2b-T1-box1-v3/receipts/
      has the equivalent 60 real-2B receipts *with bodies*, committed to
      this repo, so that is used as corpus A instead. Called out again in
      the report.
  (B) demo/agent-trace/gateway/tests/fixtures/live20/*.receipt -- 20 real-
      2B-model receipts (format v2), CALC/LOOKUP/FILE-READ/chain/mixed tool
      kinds.

A third, UNMANDATED, supplementary corpus (C) is used only for the mutation
sub-cases that require a full model replay to be caught (see "REPLAY COST"
below): fresh receipts generated against the local tinybit m7 model (the
same fast fixture safe-1e used). This is NOT one of the two corpora the
QUEUE item names; it exists purely so the ctx/q content-mutation class gets
real full-replay coverage at scale, cheaply, instead of being skipped
outright. All required-corpus (A/B) case counts are reported separately
from corpus-C counts.

REPLAY COST (why corpus A/B get different treatment for different classes):
Measured on this box (2 cores, scalar build, no AVX2): a single legitimate
`agent_trace verify` of a real-2B-model receipt with exactly one decode step
(K=1) took ~150-190s wall-clock end to end (model+embed load/hash: ~7s;
everything else is the forward-pass replay). That is already >60s -- the
harness's own hang threshold -- for a *correct, unmodified* receipt. Fields
checked by verify's `parse_receipt`/pre-replay structural gate (magic line,
model/embed/vocab hex, K/N, prompt-hex, trace-chain/table-sha256/
suite-sha256/item-ctx hex format+length, WARNING lines, and the
table-sha256 cross-check against the actual --table file, and the
--expect-ctx cross-check) are all resolved BEFORE any model replay begins,
so mutations that trip any of those gates return in ~7s (dominated by
model-file load) regardless of how many receipts are checked -- and `verify`
accepts multiple receipt paths in one process invocation, so N such mutants
share ONE model load. Only mutations that leave the receipt fully
structurally valid while changing per-step content that only the trace-chain
fold can catch (i.e. ctx=/q=/decode-chain/toks/in=/out= byte edits that
happen to stay well-formed) require the full ~150-190s replay to resolve --
those are run in small, explicitly-capped numbers directly against corpus
A/B (to prove the real fail-closed path works end to end), with the bulk of
that class's case count coming from corpus C (tinybit, ~0.7s/case).

Mutation classes (10, matching/exceeding the >=8 safe-1f brief):
  byte_flip         - random single/multi-bit flip anywhere in the file
  truncate          - truncate to a random prefix length (incl. 0 bytes)
  truncated_chain   - drop the file at a cut point inside/after the
                       trace-chain line specifically (structural: missing or
                       partial trace-chain line)
  non_utf8_random   - splice a random invalid-UTF-8 byte sequence anywhere
  non_utf8_fields   - splice invalid UTF-8 into ONE instance of EVERY
                       distinct field type the format has (model/embed/
                       vocab/K/N/prompt-hex/commit/host/item-ctx/
                       table-sha256/suite-sha256/step toks/tool/in/out/
                       decode-chain/ctx/q/trace-chain)
  giant_field       - replace a header or step field with an oversized
                       (2KB-200KB) value
  dup_step          - duplicate a "step " line (giant/duplicated step lines)
  ctx_edit          - flip a hex digit inside a step's ctx=/q= field
                       (content mutation; needs full replay to catch --
                       capped on corpus A/B, bulk on corpus C)
  expect_ctx_mismatch - call verify with a deliberately wrong --expect-ctx
                       against an otherwise UNMODIFIED valid receipt
                       (checked structurally, before replay, for both v2
                       receipts with no item-ctx line and v3 receipts with
                       one)
  table_sha_edit    - corrupt the table-sha256 header line, OR pass a
                       bit-corrupted copy of the referenced --table file
                       (both resolved before replay, via the declared-vs-
                       recomputed table hash cross-check)

Acceptable outcomes ONLY:
  - "VERIFY PASS" on the unmodified original (bytes unchanged), or
  - "VERIFY FAIL"/"FAIL structure" with a message on every case where the
    receipt bytes were actually mutated,
within a 60s wall-clock budget per invocation (relaxed to 300s only for the
explicitly-capped corpus A/B full-replay ctx_edit samples, documented as
such -- see REPLAY COST above) and without "panicked" appearing in stderr.
Any panic, any hang past the stated budget, or any PASS on a mutated
receipt is a FINDING.
"""
import hashlib
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # .../alice-aegis-cm-safe1f-fuzz
BIN = REPO / "aegis-linux/target/release/examples/agent_trace"
ART = Path("/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts")
MODEL = ART / "aegis_pruned_model.cis.safetensors"
EMBED = ART / "embed.bin"
VOCAB = ART / "vocab.bin"
TABLES_DIR = REPO / "demo/agent-trace/tables"
CORPUS_A = REPO / "eval/receipts/eval-60-2b-T1-box1-v3/receipts"
CORPUS_B = REPO / "demo/agent-trace/gateway/tests/fixtures/live20"
TINYBIT = Path(
    "/home/cm/projects/alice-aegis-cm-safe1e-fuzz/model-lab/tinybit/m7_final_gate_work/artifacts"
)

RNG_SEED = 20260913
TIMEOUT_S = 60
REPLAY_TIMEOUT_S = 300
WORK = Path("/tmp/safe1f-fuzz-work")

random.seed(RNG_SEED)


def sha256hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run(cmd, timeout=TIMEOUT_S):
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr, time.monotonic() - t0, False
    except subprocess.TimeoutExpired as e:
        # Preserve whatever partial stdout/stderr the child had already
        # written before being killed (Python's TimeoutExpired carries this
        # on .stdout/.stderr when capture_output=True was used) -- an
        # earlier version of this driver discarded it here, which made
        # every batch-level timeout look like a silent, contentless hang
        # even when most of the batch had actually completed and the
        # timeout was really just the LAST (slow, full-replay) mutant in a
        # large batch running out of budget. See the safe-1f report's
        # "REPLAY COST" / byte_flip-truncated_chain-giant_field discussion.
        out = e.stdout or b""
        err = e.stderr or b""
        return None, out, err, time.monotonic() - t0, True


TABLE_BY_SHA = {}
for tf in TABLES_DIR.glob("*.tsv"):
    TABLE_BY_SHA[sha256hex(tf.read_bytes())] = tf


def table_for(receipt_text: str):
    m = re.search(r"^table-sha256 ([0-9a-f]{64})$", receipt_text, re.M)
    if not m:
        return None
    return TABLE_BY_SHA.get(m.group(1))


def suite_for(receipt_text: str):
    m = re.search(r"^suite-sha256 ([0-9a-f]{64})$", receipt_text, re.M)
    return m.group(1) if m else None


# ---------------------------------------------------------------------
# Mutation classes
# ---------------------------------------------------------------------

def mut_byte_flip(data: bytes) -> bytes:
    if not data:
        return data
    b = bytearray(data)
    for _ in range(random.choice([1, 1, 1, 2, 4, 8])):
        i = random.randrange(len(b))
        b[i] ^= 1 << random.randrange(8)
    return bytes(b)


def mut_truncate(data: bytes) -> bytes:
    if not data:
        return data
    choices = [0, 1, len(data) // 4, len(data) // 2, len(data) - 1,
               max(0, len(data) - random.randrange(1, 20))]
    cut = max(0, min(random.choice(choices), len(data)))
    return data[:cut]


def mut_truncated_chain(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    idx = text.find("trace-chain ")
    if idx == -1:
        return mut_truncate(data)
    cut = idx + random.randrange(0, len("trace-chain ") + 30)
    return text[:cut].encode("utf-8")


def mut_non_utf8_random(data: bytes) -> bytes:
    if not data:
        data = b"x"
    bad_seqs = [b"\xff\xfe", b"\x80\x80\x80", b"\xc0\xaf", b"\xed\xa0\x80",
                bytes([random.randrange(0x80, 0x100)])]
    seq = random.choice(bad_seqs)
    pos = random.randrange(len(data) + 1)
    return data[:pos] + seq + data[pos:]


FIELD_PATTERNS = [
    ("model", r"^model ([0-9a-f]+)$"),
    ("embed", r"^embed ([0-9a-f]+)$"),
    ("vocab", r"^vocab ([0-9a-f]+)$"),
    ("K", r"^K (\d+)$"),
    ("N", r"^N (\d+)$"),
    ("prompt-hex", r"^prompt-hex ([0-9a-f]+)$"),
    ("commit", r"^commit (\S+)$"),
    ("host", r"^host (\S+)$"),
    ("item-ctx", r"^item-ctx ([0-9a-f]+)$"),
    ("table-sha256", r"^table-sha256 ([0-9a-f]+)$"),
    ("suite-sha256", r"^suite-sha256 ([0-9a-f]+)$"),
    ("step-toks", r"toks=([0-9,]+)"),
    ("step-tool", r"tool=(\S+)"),
    ("step-in", r"in=([0-9a-f]*)"),
    ("step-out", r"out=([0-9a-f]*)"),
    ("step-decode-chain", r"decode-chain=([0-9a-f]+)"),
    ("step-ctx", r"(?<!item-)ctx=([0-9a-f]+)"),
    ("step-q", r"q=([0-9a-f]+)"),
    ("trace-chain", r"^trace-chain ([0-9a-f]+)$"),
]


def non_utf8_field_variants(data: bytes):
    """Yield (field_name, mutated_bytes) for every distinct field type
    found in this receipt, each spliced with an invalid-UTF-8 sequence."""
    text = data.decode("utf-8", errors="ignore")
    bad = b"\xff\xfe"
    for name, pat in FIELD_PATTERNS:
        m = re.search(pat, text, re.M)
        if not m:
            continue
        start, end = m.span(1)
        prefix = text[:start].encode("utf-8")
        mid = text[start:end].encode("utf-8")
        suffix = text[end:].encode("utf-8")
        pos = len(mid) // 2 if mid else 0
        mutated = prefix + mid[:pos] + bad + mid[pos:] + suffix
        yield name, mutated


def mut_giant_field(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    size = random.choice([2_000, 20_000, 200_000])
    giant_hex = "ab" * (size // 2)
    targets = [i for i, l in enumerate(lines)
               if l.startswith(("model ", "embed ", "vocab ", "prompt-hex "))
               or (l.startswith("step ") and "in=" in l)]
    if not targets:
        return data
    i = random.choice(targets)
    l = lines[i]
    if l.startswith("step "):
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
        return data
    idx = random.choice(step_idxs)
    lines.insert(idx, lines[idx])
    return "\n".join(lines).encode("utf-8")


def mut_ctx_edit(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    changed = False
    for i, l in enumerate(lines):
        if l.startswith("step ") and (" ctx=" in l or " q=" in l):
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


HEX_LINE_RE = re.compile(r"^(trace-chain|table-sha256|suite-sha256|item-ctx) ([0-9a-f]+)$", re.M)


def looks_still_fully_valid(data: bytes, model_sha, embed_sha, vocab_sha) -> bool:
    """Heuristic pre-check mirroring the cheap structural gates `verify`
    resolves before replay: if this returns True, the mutant likely reaches
    the (expensive) full replay path, so batch-mode callers should reroll
    rather than risk a ~150-190s-per-case stall inside a 60s-budget batch.
    False negatives (says "not valid" when it actually would be) are fine;
    false positives (says "valid" when the real parser would reject it) just
    mean an occasional case that could have stayed in the fast batch gets
    rerolled instead -- also harmless, just slightly conservative."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not text.startswith(("AEGIS-TRACE v1\n", "AEGIS-TRACE v2\n", "AEGIS-TRACE v3\n")):
        return False
    m = re.search(r"^model ([0-9a-f]+)$", text, re.M)
    if not m or m.group(1) != model_sha:
        return False
    m = re.search(r"^embed ([0-9a-f]+)$", text, re.M)
    if not m or m.group(1) != embed_sha:
        return False
    m = re.search(r"^vocab ([0-9a-f]+)$", text, re.M)
    if not m or m.group(1) != vocab_sha:
        return False
    for m in HEX_LINE_RE.finditer(text):
        if len(m.group(2)) != 64:
            return False
    for m in re.finditer(r"ctx=([0-9a-fA-F]*)", text):
        if len(m.group(1)) != 64 or not all(c in "0123456789abcdef" for c in m.group(1)):
            return False
    for m in re.finditer(r"q=([0-9a-fA-F]*)", text):
        if len(m.group(1)) != 64 or not all(c in "0123456789abcdef" for c in m.group(1)):
            return False
    for m in re.finditer(r"decode-chain=([0-9a-fA-F]*)", text):
        if len(m.group(1)) != 64 or not all(c in "0123456789abcdef" for c in m.group(1)):
            return False
    return True


REAL_MODEL_SHA = None
REAL_EMBED_SHA = None
REAL_VOCAB_SHA = None


def mut_with_reroll(fn, orig, max_tries=6):
    """Apply `fn` to `orig`, rerolling (regenerating the mutation) up to
    `max_tries` times if the result still looks fully structurally valid
    against the REAL model/embed/vocab hashes -- see
    `looks_still_fully_valid`. Used only for the cheap-batch classes
    (byte_flip / dup_step / giant_field) to keep those batches out of the
    expensive full-replay path; falls through to whatever the last attempt
    produced if it can't shake structural validity in max_tries."""
    mutated = fn(orig)
    if REAL_MODEL_SHA is None:
        return mutated
    for _ in range(max_tries):
        if not looks_still_fully_valid(mutated, REAL_MODEL_SHA, REAL_EMBED_SHA, REAL_VOCAB_SHA):
            break
        mutated = fn(orig)
    return mutated


RECEIPT_MUTATIONS = {
    "byte_flip": mut_byte_flip,
    "truncate": mut_truncate,
    "truncated_chain": mut_truncated_chain,
    "non_utf8_random": mut_non_utf8_random,
    "giant_field": mut_giant_field,
    "dup_step": mut_dup_step,
}

# classes handled specially below (not simple 1-seed-in-1-seed-out fns):
#   non_utf8_fields, ctx_edit, expect_ctx_mismatch, table_sha_edit


def classify(rc, stdout, stderr, timed_out, mutation_changed, timeout_budget):
    stderr_s = stderr.decode("utf-8", errors="replace")
    stdout_s = stdout.decode("utf-8", errors="replace")
    if timed_out:
        return "FINDING-HANG", stdout_s, stderr_s
    if "panicked" in stderr_s or "panicked" in stdout_s:
        return "FINDING-PANIC", stdout_s, stderr_s
    passed = rc == 0 and "VERIFY PASS" in stdout_s
    if passed and mutation_changed:
        return "FINDING-WRONGFUL-PASS", stdout_s, stderr_s
    if not passed and not mutation_changed:
        return "FINDING-UNEXPECTED-FAIL-ON-ORIGINAL", stdout_s, stderr_s
    if not passed and ("FAIL" not in stdout_s and "FAIL" not in stderr_s):
        return "FINDING-FAIL-WITHOUT-MESSAGE", stdout_s, stderr_s
    return "ok", stdout_s, stderr_s


class Fuzzer:
    def __init__(self):
        self.total = 0
        self.counts = {}
        self.findings = []

    def bump(self, cls):
        self.total += 1
        self.counts[cls] = self.counts.get(cls, 0) + 1

    def record(self, cls, seed, path, verdict, stdout_s, stderr_s, extra=None):
        if verdict != "ok":
            f = {
                "class": cls, "seed": seed, "path": str(path),
                "verdict": verdict, "stdout": stdout_s[:1500],
                "stderr": stderr_s[:1500],
            }
            if extra:
                f.update(extra)
            self.findings.append(f)

    def verify_batch(self, model, embed, vocab, receipt_paths, extra_args=None,
                      timeout=TIMEOUT_S):
        cmd = [str(BIN), "verify", str(model), str(embed), str(vocab)] + \
            [str(p) for p in receipt_paths] + (extra_args or [])
        return run(cmd, timeout=timeout)

    def verify_one(self, model, embed, vocab, receipt_path, extra_args=None,
                    timeout=TIMEOUT_S):
        return self.verify_batch(model, embed, vocab, [receipt_path], extra_args, timeout)


def per_receipt_table_args(text):
    t = table_for(text)
    s = suite_for(text)
    args = []
    if t:
        args += ["--table", str(t)]
    if s:
        args += ["--suite-sha256", s]
    return args


def corrupt_table_copy(table_path: Path, tmpdir: Path) -> Path:
    data = bytearray(table_path.read_bytes())
    if data:
        data[len(data) // 2] ^= 0xFF
    out = tmpdir / f"corrupt_{table_path.name}"
    out.write_bytes(bytes(data))
    return out


def main():
    if not BIN.exists():
        print("agent_trace release example missing; build first", file=sys.stderr)
        return 2
    WORK.mkdir(exist_ok=True)
    for old in WORK.glob("*"):
        try:
            old.unlink()
        except IsADirectoryError:
            pass

    global REAL_MODEL_SHA, REAL_EMBED_SHA, REAL_VOCAB_SHA
    REAL_MODEL_SHA = sha256hex(MODEL.read_bytes())
    REAL_EMBED_SHA = sha256hex(EMBED.read_bytes())
    REAL_VOCAB_SHA = sha256hex(VOCAB.read_bytes())

    fz = Fuzzer()

    corpus_a = sorted(CORPUS_A.glob("*.txt"))
    corpus_b = sorted(CORPUS_B.glob("*.receipt"))
    print(f"corpus A (eval-60-2b-T1-box1-v3): {len(corpus_a)} receipts")
    print(f"corpus B (live20): {len(corpus_b)} receipts")
    assert len(corpus_a) == 60, len(corpus_a)
    assert len(corpus_b) == 20, len(corpus_b)

    all_seeds = [("A", p) for p in corpus_a] + [("B", p) for p in corpus_b]

    # ---- 0. baseline: VERIFY PASS on a SAMPLE of unmodified originals.
    # Full-corpus baseline PASS at ~150-190s/receipt (see REPLAY COST) is
    # not re-run here for all 80 receipts: eval-60-2b-T1-box1-v3/verify.log
    # (committed, pre-existing) already records "TOTAL: 60 PASS / 0 FAIL /
    # 60 items" for the entirety of corpus A from a prior verified run on
    # this same binary lineage. This script re-confirms a handful (3 per
    # corpus) fresh, plus records the pre-existing log's result. ----
    baseline_sample = [("A", p) for p in corpus_a[:3]] + [("B", p) for p in corpus_b[:3]]
    for corp, p in baseline_sample:
        text = p.read_text(errors="ignore")
        extra = per_receipt_table_args(text)
        rc, out, err, dt, to = fz.verify_one(MODEL, EMBED, VOCAB, p, extra, timeout=REPLAY_TIMEOUT_S)
        verdict, so, se = classify(rc, out, err, to, False, REPLAY_TIMEOUT_S)
        fz.bump("baseline_pass")
        fz.record("baseline_pass", f"{corp}:{p.name}", p, verdict, so, se)

    # ---- 1..6: simple 1-in-1-out mutation classes, batched by
    # (table,suite) bucket so each class needs only ~5 verify invocations
    # (one model load each) instead of one per seed. ----
    N_PER_SEED = 20  # -> 80 seeds * 20 = 1600 cases across these 6 classes
    buckets = {}
    for corp, p in all_seeds:
        text = p.read_text(errors="ignore")
        extra = tuple(per_receipt_table_args(text))
        buckets.setdefault(extra, []).append((corp, p))

    for class_name, fn in RECEIPT_MUTATIONS.items():
        for extra, seeds_in_bucket in buckets.items():
            batch = []
            changed_flags = []
            labels = []
            for corp, p in seeds_in_bucket:
                orig = p.read_bytes()
                for i in range(N_PER_SEED):
                    if class_name in ("byte_flip", "dup_step", "giant_field"):
                        mutated = mut_with_reroll(fn, orig)
                    else:
                        mutated = fn(orig)
                    changed = mutated != orig
                    mpath = WORK / f"{class_name}_{corp}_{p.stem}_{i}.receipt"
                    mpath.write_bytes(mutated)
                    batch.append(mpath)
                    changed_flags.append(changed)
                    labels.append(f"{corp}:{p.name}")
            rc, out, err, dt, to = fz.verify_batch(MODEL, EMBED, VOCAB, batch, list(extra), timeout=max(TIMEOUT_S, 20 + len(batch) // 4))
            out_s = out.decode("utf-8", errors="replace")
            err_s = err.decode("utf-8", errors="replace")
            if to or "panicked" in err_s or "panicked" in out_s:
                for mpath in batch:
                    fz.bump(class_name)
                verdict = "FINDING-HANG" if to else "FINDING-PANIC"
                fz.findings.append({
                    "class": class_name, "seed": f"bucket {extra} (batch of {len(batch)})",
                    "path": str(batch[0]), "verdict": verdict,
                    "stdout": out_s[:1500], "stderr": err_s[:1500],
                })
                continue
            for i, mpath in enumerate(batch):
                fz.bump(class_name)
                marker = f"== {mpath}\n"
                idx = out_s.find(marker)
                seg = out_s[idx: out_s.find("\n== ", idx + 1)] if idx != -1 else out_s
                passed = "VERIFY PASS" in seg
                changed = changed_flags[i]
                if passed and changed:
                    verdict = "FINDING-WRONGFUL-PASS"
                elif not passed and "FAIL" not in seg:
                    verdict = "FINDING-FAIL-WITHOUT-MESSAGE"
                else:
                    verdict = "ok"
                fz.record(class_name, labels[i], mpath, verdict, seg, "")

    # ---- 7: non_utf8_fields (one mutant per distinct field type found),
    # batched by (table,suite) bucket, one invocation per bucket. ----
    for extra, seeds_in_bucket in buckets.items():
        batch = []
        labels = []
        for corp, p in seeds_in_bucket:
            orig = p.read_bytes()
            for name, mutated in non_utf8_field_variants(orig):
                mpath = WORK / f"non_utf8_fields_{corp}_{p.stem}_{name}.receipt"
                mpath.write_bytes(mutated)
                batch.append(mpath)
                labels.append(f"{corp}:{p.name}:{name}")
        if not batch:
            continue
        rc, out, err, dt, to = fz.verify_batch(MODEL, EMBED, VOCAB, batch, list(extra), timeout=max(TIMEOUT_S, 20 + len(batch) // 4))
        out_s = out.decode("utf-8", errors="replace")
        err_s = err.decode("utf-8", errors="replace")
        if to or "panicked" in err_s or "panicked" in out_s:
            for mpath in batch:
                fz.bump("non_utf8_fields")
            verdict = "FINDING-HANG" if to else "FINDING-PANIC"
            fz.findings.append({
                "class": "non_utf8_fields", "seed": f"bucket {extra} (batch of {len(batch)})",
                "path": str(batch[0]), "verdict": verdict,
                "stdout": out_s[:1500], "stderr": err_s[:1500],
            })
            continue
        for i, mpath in enumerate(batch):
            fz.bump("non_utf8_fields")
            marker = f"== {mpath}\n"
            idx = out_s.find(marker)
            seg = out_s[idx: out_s.find("\n== ", idx + 1)] if idx != -1 else out_s
            passed = "VERIFY PASS" in seg
            verdict = "FINDING-WRONGFUL-PASS" if passed else ("ok" if "FAIL" in seg else "FINDING-FAIL-WITHOUT-MESSAGE")
            fz.record("non_utf8_fields", labels[i], mpath, verdict, seg, "")

    # ---- 8: expect_ctx_mismatch (unmodified receipt bytes, wrong flag),
    # batched by bucket. ----
    wrong_ctx = "ab" * 32
    for extra, seeds_in_bucket in buckets.items():
        batch = [p for _, p in seeds_in_bucket]
        labels = [f"{corp}:{p.name}" for corp, p in seeds_in_bucket]
        args = list(extra) + ["--expect-ctx", wrong_ctx]
        rc, out, err, dt, to = fz.verify_batch(MODEL, EMBED, VOCAB, batch, args, timeout=max(TIMEOUT_S, 20 + len(batch) // 2))
        out_s = out.decode("utf-8", errors="replace")
        err_s = err.decode("utf-8", errors="replace")
        if to or "panicked" in err_s or "panicked" in out_s:
            for p in batch:
                fz.bump("expect_ctx_mismatch")
            verdict = "FINDING-HANG" if to else "FINDING-PANIC"
            fz.findings.append({
                "class": "expect_ctx_mismatch", "seed": f"bucket {extra} (batch of {len(batch)})",
                "path": str(batch[0]), "verdict": verdict,
                "stdout": out_s[:1500], "stderr": err_s[:1500],
            })
            continue
        for i, p in enumerate(batch):
            fz.bump("expect_ctx_mismatch")
            marker = f"== {p}\n"
            idx = out_s.find(marker)
            seg = out_s[idx: out_s.find("\n== ", idx + 1)] if idx != -1 else out_s
            if "VERIFY PASS" in seg:
                # receipt bytes untouched, but --expect-ctx was
                # deliberately wrong -- a PASS here is the finding.
                verdict = "FINDING-EXPECT-CTX-IGNORED"
            elif "FAIL" not in seg:
                verdict = "FINDING-FAIL-WITHOUT-MESSAGE"
            else:
                verdict = "ok"
            fz.record("expect_ctx_mismatch", labels[i], p, verdict, seg, "")

    # ---- 9: table_sha_edit, batched by bucket (only buckets with a
    # --table arg apply; class is N/A for calc-only receipts with no
    # table-sha256 line). ----
    corrupted_table_cache = {}
    for extra, seeds_in_bucket in buckets.items():
        if "--table" not in extra:
            continue  # no table-sha256 line in this bucket's receipts
        table_idx = extra.index("--table") + 1
        real_table = Path(extra[table_idx])
        # rebuild suite args (everything in `extra` after the table pair)
        rest = list(extra)
        del rest[table_idx - 1:table_idx + 1]
        suite_args = rest

        # (a) header-corruption batch: real --table, each receipt's own
        # table-sha256 line zeroed out.
        batch_a, labels_a = [], []
        for corp, p in seeds_in_bucket:
            text = p.read_text(errors="ignore")
            bad_header = re.sub(r"^table-sha256 [0-9a-f]{64}$",
                                 "table-sha256 " + "0" * 64, text, flags=re.M)
            mpath = WORK / f"table_sha_header_{corp}_{p.stem}.receipt"
            mpath.write_bytes(bad_header.encode("utf-8"))
            batch_a.append(mpath)
            labels_a.append(f"{corp}:{p.name}")
        args_a = ["--table", str(real_table)] + suite_args
        rc, out, err, dt, to = fz.verify_batch(MODEL, EMBED, VOCAB, batch_a, args_a, timeout=max(TIMEOUT_S, 20 + len(batch_a) // 2))
        out_s = out.decode("utf-8", errors="replace")
        err_s = err.decode("utf-8", errors="replace")
        if to or "panicked" in err_s or "panicked" in out_s:
            for mpath in batch_a:
                fz.bump("table_sha_edit")
            verdict = "FINDING-HANG" if to else "FINDING-PANIC"
            fz.findings.append({"class": "table_sha_edit.header", "seed": f"bucket {extra}",
                                 "path": str(batch_a[0]), "verdict": verdict,
                                 "stdout": out_s[:1500], "stderr": err_s[:1500]})
        else:
            for i, mpath in enumerate(batch_a):
                fz.bump("table_sha_edit")
                marker = f"== {mpath}\n"
                idx = out_s.find(marker)
                seg = out_s[idx: out_s.find("\n== ", idx + 1)] if idx != -1 else out_s
                verdict = "FINDING-WRONGFUL-PASS" if "VERIFY PASS" in seg else ("ok" if "FAIL" in seg else "FINDING-FAIL-WITHOUT-MESSAGE")
                fz.record("table_sha_edit.header", labels_a[i], mpath, verdict, seg, "")

        # (b) table-FILE-corruption batch: original receipts, unchanged,
        # against a bit-corrupted copy of the real table file.
        if str(real_table) not in corrupted_table_cache:
            corrupted_table_cache[str(real_table)] = corrupt_table_copy(real_table, WORK)
        bad_table = corrupted_table_cache[str(real_table)]
        batch_b = [p for _, p in seeds_in_bucket]
        labels_b = [f"{corp}:{p.name}" for corp, p in seeds_in_bucket]
        args_b = ["--table", str(bad_table)] + suite_args
        rc, out, err, dt, to = fz.verify_batch(MODEL, EMBED, VOCAB, batch_b, args_b, timeout=max(TIMEOUT_S, 20 + len(batch_b) // 2))
        out_s = out.decode("utf-8", errors="replace")
        err_s = err.decode("utf-8", errors="replace")
        if to or "panicked" in err_s or "panicked" in out_s:
            for p in batch_b:
                fz.bump("table_sha_edit")
            verdict = "FINDING-HANG" if to else "FINDING-PANIC"
            fz.findings.append({"class": "table_sha_edit.file", "seed": f"bucket {extra}",
                                 "path": str(batch_b[0]), "verdict": verdict,
                                 "stdout": out_s[:1500], "stderr": err_s[:1500]})
        else:
            for i, p in enumerate(batch_b):
                fz.bump("table_sha_edit")
                marker = f"== {p}\n"
                idx = out_s.find(marker)
                seg = out_s[idx: out_s.find("\n== ", idx + 1)] if idx != -1 else out_s
                verdict = "FINDING-WRONGFUL-PASS" if "VERIFY PASS" in seg else ("ok" if "FAIL" in seg else "FINDING-FAIL-WITHOUT-MESSAGE")
                fz.record("table_sha_edit.file", labels_b[i], p, verdict, seg, "")

    # ---- 10: ctx_edit -- capped full-replay samples on corpus A/B ----
    CTX_EDIT_CAP_A = 3
    CTX_EDIT_CAP_B = 3
    capped = [("A", p) for p in corpus_a[:CTX_EDIT_CAP_A]] + \
             [("B", p) for p in corpus_b[:CTX_EDIT_CAP_B]]
    for corp, p in capped:
        orig = p.read_bytes()
        text = orig.decode("utf-8", errors="ignore")
        extra = per_receipt_table_args(text)
        mutated = mut_ctx_edit(orig)
        mpath = WORK / f"ctx_edit_full_{corp}_{p.stem}.receipt"
        mpath.write_bytes(mutated)
        rc, out, err, dt, to = fz.verify_one(MODEL, EMBED, VOCAB, mpath, extra, timeout=REPLAY_TIMEOUT_S)
        fz.bump("ctx_edit.full_replay_2b")
        verdict, so, se = classify(rc, out, err, to, mutated != orig, REPLAY_TIMEOUT_S)
        fz.record("ctx_edit.full_replay_2b", f"{corp}:{p.name}", mpath, verdict, so, se)

    # ---- supplementary corpus C: tinybit, bulk ctx_edit + byte_flip full replay ----
    corpus_c_count = 0
    if TINYBIT.exists():
        tb_model, tb_embed, tb_vocab = TINYBIT / "MODEL.SAF", TINYBIT / "EMBED.BIN", TINYBIT / "VOCAB.BIN"
        calc_prompts = [
            "Q: 6 + 7\nA: CALC(6 + 7).\n",
            "Q: 100 - 37\nA: CALC(100 - 37).\n",
            "Q: 12 * 12\nA: CALC(12 * 12).\n",
            "Q: 9 - 4\nA: CALC(9 - 4).\n",
        ]
        seeds_c = []
        for pr in calc_prompts:
            out_path = WORK / f"tinybit_seed_{abs(hash(pr))}.receipt"
            with open(out_path, "wb") as f:
                p = subprocess.run([str(BIN), "gen", str(tb_model), str(tb_embed), str(tb_vocab),
                                     "1", "16", pr], stdout=f, stderr=subprocess.PIPE, timeout=30)
            if p.returncode == 0:
                seeds_c.append(out_path)
        for class_name in ["ctx_edit", "byte_flip"]:
            fn = mut_ctx_edit if class_name == "ctx_edit" else mut_byte_flip
            for seed in seeds_c:
                orig = seed.read_bytes()
                for i in range(50):
                    mutated = fn(orig)
                    changed = mutated != orig
                    mpath = WORK / f"corpusC_{class_name}_{seed.stem}_{i}.receipt"
                    mpath.write_bytes(mutated)
                    rc, out, err, dt, to = fz.verify_one(tb_model, tb_embed, tb_vocab, mpath, timeout=TIMEOUT_S)
                    fz.bump(f"corpusC.{class_name}")
                    corpus_c_count += 1
                    verdict, so, se = classify(rc, out, err, to, changed, TIMEOUT_S)
                    fz.record(f"corpusC.{class_name}", f"C:{seed.name}", mpath, verdict, so, se)
        # baseline PASS check for corpus C seeds themselves
        for seed in seeds_c:
            rc, out, err, dt, to = fz.verify_one(tb_model, tb_embed, tb_vocab, seed, timeout=TIMEOUT_S)
            fz.bump("corpusC.baseline_pass")
            verdict, so, se = classify(rc, out, err, to, False, TIMEOUT_S)
            fz.record("corpusC.baseline_pass", f"C:{seed.name}", seed, verdict, so, se)

    print(f"\nTOTAL CASES: {fz.total}  (corpus C / tinybit supplementary: {corpus_c_count + len(seeds_c) if TINYBIT.exists() else 0})")
    print("Per-class counts:")
    for k, v in sorted(fz.counts.items()):
        print(f"  {k}: {v}")
    print(f"\nFINDINGS: {len(fz.findings)}")
    for f in fz.findings:
        print("----")
        for k, v in f.items():
            print(f"{k}: {v}")

    out_path = Path("/tmp/safe1f-fuzz-findings.txt")
    with open(out_path, "w") as fh:
        fh.write(f"TOTAL CASES: {fz.total}\n")
        fh.write("Per-class counts:\n")
        for k, v in sorted(fz.counts.items()):
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nFINDINGS: {len(fz.findings)}\n")
        for f in fz.findings:
            fh.write("----\n")
            for k, v in f.items():
                fh.write(f"{k}: {v}\n")
    print(f"\nwrote {out_path}")
    return 0 if not fz.findings else 1


if __name__ == "__main__":
    sys.exit(main())
