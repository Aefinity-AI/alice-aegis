# Verified edge inference — one-command demo

`run.sh` builds and runs a chain: greedy, integer-only (CIS-1 `FullInt`)
decode of the checked-in M7 tinybit model produces a **receipt**
(artifact SHA-256s, prompt, every generated token id, an FNV-1a digest,
and a SHA-256 chain over the logits). That receipt is then replayed and
checked by **two independent verifiers** — the in-tree `cis_witness`
example and the standalone `no_std` `cis-verify` binary — against the
same three artifact files. A receipt generated on one CPU can be copied
to a different machine and verified there with no GPU and no network
access beyond the copy itself.

No files are downloaded. Everything comes from
`model-lab/tinybit/m7_final_gate_work/artifacts/` already in this repo.

## Usage

```
demo/edge-receipt/run.sh build                       # compile everything
demo/edge-receipt/run.sh gen "The quick brown fox" 32 # write a receipt
demo/edge-receipt/run.sh verify <receipt-file>        # replay + check
demo/edge-receipt/run.sh all "The quick brown fox"    # build + gen + verify
```

Receipts land in `demo/edge-receipt/out/receipt-<hostname>-<utc>.txt`
with a `.meta` sidecar (hostname, arch, CPU model, AVX2/FMA cpuinfo flag
counts, the aegis commit, and the decoded text from `cis_decode`).

## Cross-machine recipe

```
# machine A (generate)
demo/edge-receipt/run.sh build
demo/edge-receipt/run.sh gen "The quick brown fox" 32
scp demo/edge-receipt/out/receipt-A-*.txt machineB:alice-aegis/demo/edge-receipt/out/

# machine B (verify, independently, same checked-in artifacts)
demo/edge-receipt/run.sh build
demo/edge-receipt/run.sh verify demo/edge-receipt/out/receipt-A-*.txt
```

Both machines must have the identical `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN` triple (same repo checkout is sufficient — that's the point:
the hashes in the receipt are checked against whatever copy is local).

## What a PASS proves

- The three artifact files present on the verifying machine hash to
  exactly what the receipt claims.
- Replaying the greedy `FullInt` decode from the receipt's prompt on the
  verifying machine reproduces, bit-for-bit, the same sequence of
  **token ids**, the same FNV-1a digest, and the same SHA-256 logit chain
  the receipt records.
- This held across two independently-written verifier implementations
  (`cis_witness`, which shares `aegis-core`'s decode path, and
  `cis-verify`, a separate `no_std` reimplementation).

## What a PASS does NOT prove

- Nothing about GPUs, throughput, or latency — this is an integer-only
  CPU path and any timing this script prints is explicitly labelled
  "not a benchmark".
- Nothing about text-level correctness or model quality — the receipt
  and both verifiers check **token ids**, per CIS-1 spec v1.0
  (`docs/CIS-1_SPEC_v1.0.md`, §8-10). The decoded text printed by
  `cis_decode` is convenience output only, not part of what is verified.
- No claim that this is the "first" or "only" system to do bit-exact
  receipted replay — it's a demonstration of the CIS-1 receipt format
  already implemented in this repo.
- No claim about any model other than the specific artifacts hashed in the receipt.

## Swapping in the private 2B artifacts later

Point `ARTIFACTS` (edit the top of `run.sh`, or generalize it to a CLI
flag) at wherever the 2B release's `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN` are checked out. Nothing else in this demo is model-size
specific — `cis_witness` and `cis-verify` both read `aegis_config` out of
`MODEL.SAF`'s metadata at runtime. Confirm `max_new + prompt_tokens <=
max_position_embeddings` for that config before generating.

Concretely, point the three env vars at the files (names differ from the M7 set):

    AEGIS_MODEL=/path/aegis_pruned_model.cis.safetensors \
    AEGIS_EMBED=/path/embed.bin AEGIS_VOCAB=/path/vocab.bin \
    demo/edge-receipt/run.sh all "your prompt" 64

The receipt records the artifact SHA-256s, so a verifier on another machine must
use byte-identical files (check with sha256sum) or every verifier check fails at
"artifact hashes".
