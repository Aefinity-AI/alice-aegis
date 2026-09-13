#!/usr/bin/env python3
"""safe-2c: retrofit item-ctx binding onto the safe-2 EVAL-60 T1 (2B, box1)
receipt bundle WITHOUT re-running inference.

Genesis is a pure hash of already-recorded header fields (model/embed/vocab
sha, K, N, prompt bytes, table/suite/commit/host) — no model needed to
recompute it (see FORMAT.md sec4/tools/trace_chain.py, the existing
inference-free clean-room reference). This script:
  1. reads each ORIGINAL (untampered) safe-2 receipt,
  2. adds an `item-ctx <hex>` header line (sha256(item_id || prompt_bytes ||
     nonce)) computed from the receipt's own item id (filename stem) and its
     own prompt-hex bytes,
  3. bumps the header line AEGIS-TRACE v2 -> v3 (format 4),
  4. recomputes trace-chain by refolding genesis (now including the
     item-ctx block) + the UNCHANGED per-step decode-chain/tool/in/out
     records already in the file (pure hash math, no model replay),
  5. writes the result to a new v3 receipt set.

Source: eval-60-2b-T1-box1/receipts/*.txt in the sibling cm/safe2-verified-
evals worktree (the untampered safe-2 bundle safe-2b's tamper demo copied
from). Nonce is a fixed constant for this whole demo (not secret, not
meant to be) -- see NONCE below.
"""
import hashlib
import os
import re
import sys

SRC = "/home/cm/projects/alice-aegis-cm-safe2/eval/receipts/eval-60-2b-T1-box1/receipts"
DST = "eval/receipts/eval-60-2b-T1-box1-v3/receipts"
NONCE = b"safe2c-demo-nonce-1"
TABLE_PATH = "demo/agent-trace/tables/demo.tsv"

TRACE_DOMAIN = b"AEGIS-TRACE v0\n"


def unhex(s):
    return bytes.fromhex(s)


def item_ctx(item_id: bytes, prompt: bytes, nonce: bytes) -> bytes:
    return hashlib.sha256(item_id + prompt + nonce).digest()


def fold_genesis(
    model_sha, embed_sha, vocab_sha, k, n, prompt, commit, host, ctx,
    table_sha_and_len=None, suite_sha=None,
):
    buf = bytearray()
    buf += TRACE_DOMAIN
    buf += model_sha
    buf += embed_sha
    buf += vocab_sha
    buf += k.to_bytes(8, "big")
    buf += n.to_bytes(8, "big")
    buf += len(prompt).to_bytes(8, "big")
    buf += prompt
    if table_sha_and_len is not None:
        table_sha, table_len = table_sha_and_len
        buf += table_sha
        buf += table_len.to_bytes(8, "big")
    if suite_sha is not None:
        buf += b"SUITE"
        buf += suite_sha
    buf += b"PROV"
    for field in (commit.encode(), host.encode()):
        buf += len(field).to_bytes(4, "little")
        buf += field
    buf += b"ITEMCTX"
    buf += ctx
    return hashlib.sha256(bytes(buf)).digest()


def fold_steps(genesis, steps):
    chain = genesis
    for i, s in enumerate(steps):
        step_buf = bytearray()
        step_buf += chain
        step_buf += b"TSTEP"
        step_buf += i.to_bytes(8, "big")
        step_buf += unhex(s["decode-chain"])
        name = s["tool"].encode()
        inp = unhex(s["in"]) if s["in"] else b""
        out = unhex(s["out"]) if s["out"] else b""
        for field in (name, inp, out):
            step_buf += len(field).to_bytes(4, "little")
            step_buf += field
        chain = hashlib.sha256(bytes(step_buf)).digest()
    return chain


def process(item_id, text):
    lines = text.splitlines()
    assert lines[0] == "AEGIS-TRACE v2", (item_id, lines[0])
    lead = {}
    steps = []
    warn_lines = []
    for ln in lines[1:]:
        if ln.startswith("step "):
            rest = ln.split(":", 1)[1].strip()
            kv = dict(tok.split("=", 1) for tok in rest.split())
            steps.append(kv)
        elif ln.startswith("WARNING"):
            warn_lines.append(ln)
        elif ln.startswith("trace-chain "):
            lead["trace-chain"] = ln.split(" ", 1)[1]
        else:
            k, v = ln.split(" ", 1)
            lead[k] = v
    model_sha = unhex(lead["model"])
    embed_sha = unhex(lead["embed"])
    vocab_sha = unhex(lead["vocab"])
    k_val = int(lead["K"])
    n_val = int(lead["N"])
    prompt = unhex(lead["prompt-hex"])
    commit = lead["commit"]
    host = lead["host"]

    table_sha_and_len = None
    if "table-sha256" in lead:
        with open(TABLE_PATH, "rb") as tf:
            table_bytes = tf.read()
        actual = hashlib.sha256(table_bytes).hexdigest()
        assert actual == lead["table-sha256"], (item_id, actual, lead["table-sha256"])
        table_sha_and_len = (unhex(lead["table-sha256"]), len(table_bytes))
    suite_sha = unhex(lead["suite-sha256"]) if "suite-sha256" in lead else None

    ctx = item_ctx(item_id.encode(), prompt, NONCE)
    genesis = fold_genesis(
        model_sha, embed_sha, vocab_sha, k_val, n_val, prompt, commit, host, ctx,
        table_sha_and_len=table_sha_and_len, suite_sha=suite_sha,
    )
    trace_chain = fold_steps(genesis, steps)

    out = []
    out.append("AEGIS-TRACE v3")
    out.append(f"model {lead['model']}")
    out.append(f"embed {lead['embed']}")
    out.append(f"vocab {lead['vocab']}")
    out.append(f"K {k_val}")
    out.append(f"N {n_val}")
    if "table-sha256" in lead:
        out.append(f"table-sha256 {lead['table-sha256']}")
    if "suite-sha256" in lead:
        out.append(f"suite-sha256 {lead['suite-sha256']}")
    out.append(f"prompt-hex {lead['prompt-hex']}")
    out.append(f"commit {commit}")
    out.append(f"host {host}")
    out.append(f"item-ctx {ctx.hex()}")
    for i, s in enumerate(steps):
        fields = " ".join(f"{k}={s[k]}" for k in ("toks", "tool", "in", "out", "decode-chain", "ctx", "q") if k in s)
        out.append(f"step {i}: {fields}")
    out.extend(warn_lines)
    out.append(f"trace-chain {trace_chain.hex()}")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(DST, exist_ok=True)
    names = sorted(
        f[:-4] for f in os.listdir(SRC) if f.endswith(".txt")
    )
    assert len(names) == 60, len(names)
    for name in names:
        with open(f"{SRC}/{name}.txt") as f:
            text = f.read()
        out_text = process(name, text)
        with open(f"{DST}/{name}.txt", "w") as f:
            f.write(out_text)
    print(f"wrote {len(names)} v3 (item-ctx-bound) receipts to {DST}")


if __name__ == "__main__":
    main()
