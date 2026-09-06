# Verifiable agent trace — one-command demo

`run.sh` runs a small, deterministic **agent episode** over the checked-in
M7 tinybit model: K rounds (default 3) of {greedy, integer-only CIS-1
`FullInt` decode of up to N tokens (default 16), scan the decoded text for
one `CALC(<int> <op> <int>)` tool call, run the tool, append the decoded
text and the tool's result to the running prompt for the next round}. The
whole episode is hash-chained into one **AEGIS-TRACE v0** receipt: artifact
SHA-256s, K, N, the initial prompt, and one line per step (the step's token
ids, which tool ran, its input/output, and that step's own decode-chain
digest), followed by a `trace-chain` digest that folds every step together.

`verify` replays the entire episode from the receipt's header (artifacts,
K, N, initial prompt) on the local machine and compares every step and the
final trace chain to what the receipt claims — bit-for-bit. Any altered
token id, tool input, tool output, or dropped step changes the trace chain
and `verify` exits 1.

No files are downloaded. Everything comes from
`model-lab/tinybit/m7_final_gate_work/artifacts/` already in this repo
(same default as `demo/edge-receipt`).

## Usage

```
demo/agent-trace/run.sh build                          # compile agent_trace
demo/agent-trace/run.sh gen "The quick brown fox" 3 16  # write a receipt (prompt K N)
demo/agent-trace/run.sh verify <receipt-file>           # replay + check
demo/agent-trace/run.sh tamper                          # 3 adversarial mutations, each must FAIL
demo/agent-trace/run.sh all "The quick brown fox" 3 16  # build + gen + verify + tamper
```

Receipts land in `demo/agent-trace/out/trace-<hostname>-<utc>.txt`.

### Env overrides (same names as `demo/edge-receipt`)

```
AEGIS_ARTIFACTS=<dir>   # expects MODEL.SAF / EMBED.BIN / VOCAB.BIN in it
AEGIS_MODEL=<file>      # override MODEL.SAF individually
AEGIS_EMBED=<file>      # override EMBED.BIN individually
AEGIS_VOCAB=<file>      # override VOCAB.BIN individually
```

## Cross-machine recipe

```
# machine A (generate)
demo/agent-trace/run.sh build
demo/agent-trace/run.sh gen "The quick brown fox" 3 16
scp demo/agent-trace/out/trace-A-*.txt machineB:alice-aegis/demo/agent-trace/out/

# machine B (verify, independently, same checked-in artifacts)
demo/agent-trace/run.sh build
demo/agent-trace/run.sh verify demo/agent-trace/out/trace-A-*.txt
```

Both machines must have the identical `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN` triple. The commit hash and hostname printed in the receipt are
informational (Rule B provenance) and are NOT folded into the trace chain,
so a receipt generated on one machine still verifies bit-for-bit on a
different machine, commit, or host.

## What a PASS proves

- The three artifact files present on the verifying machine hash to
  exactly what the receipt claims.
- Replaying the K-step episode from the receipt's initial prompt on the
  verifying machine reproduces, bit-for-bit, every step's token ids, tool
  name/input/output, and per-step decode-chain digest, and therefore the
  final trace-chain digest.
- The `tamper` mutations (flip a token id, flip a tool-output byte, drop a
  step) are each independently rejected by `verify`.

## What a PASS does NOT prove

Non-goals of this prototype, verbatim from the design brief:

- Not a general agent framework; no network tools, no filesystem tools, no
  sampling.
- Does not attest the tool implementation beyond "it is in this pinned
  binary".
- No claims of speed, quality, or "first".

Additionally: the receipt and verifier check **token ids, tool
name/input/output bytes, and chain digests**, per the same CIS-1 decode
path `cis_witness` uses — not text-level correctness or model quality. No
claim about any model other than the specific artifacts hashed in the
receipt. No timing is printed anywhere.

## Prompting note

The `CALC` scanner only looks at the model's newly decoded text for that
step — it never re-scans the prompt or earlier steps. A prompt that ends
with an unclosed `CALC(` (for example, a truncated few-shot example) can
never register a tool call, because the decoded text that follows has no
way to complete an already-open call from the prompt side; `find_calc` only
matches a `CALC(` that starts inside the text it is given. Use few-shot
prompts that end after a complete example, not mid-call.

## Evidence so far

An M7 receipt generated on box2 (Celeron N4020, no AVX2/FMA) verified
bit-for-bit on box1 (i5-5200U) and penguin (Crostini VM); three tamper
cases rejected on both boxes. A 2B (bitnet2b artifacts) two-step episode
generated on box2 recorded one model-emitted tool=calc step with input
CALC(10 + 10) and output 20 and verified bit-for-bit on box1 (trace-chain
b3030c0f63a5b37f…); the same trace-chain was reproduced by two different
builds of the branch (94b0c6d and 78c133e) on box2. In that episode the
model echoed a few-shot example rather than solving the question asked —
this demonstrates the receipt mechanism, not model capability.

## Optional: TPM attestation of the receipt

`demo/edge-receipt/attest.sh` also accepts an agent-trace receipt (its
`trace-chain` line instead of a CIS-1 `cis-digest` line) and binds a TPM
quote to it the same way it does for a decode receipt: the TPM signs the
PCR values plus the qualifying data, which is the whole receipt digest (32
bytes for a `trace-chain`, vs. 8 bytes for a CIS-1 `cis-digest`). The same
limits apply: null hierarchy (not vendor-rooted) until BIOS TPM State is
enabled, and the quote proves firmware/PCR state bound to the receipt
digest at quote time, not the trace-chain replay itself. See
`demo/edge-receipt/README.md`'s "Optional: TPM attestation" section for
the full command reference and caveats.

## Swapping in other artifacts

Point `AEGIS_ARTIFACTS` (or the three `AEGIS_MODEL`/`AEGIS_EMBED`/
`AEGIS_VOCAB` vars) at another CIS-1 artifact set. `agent_trace` reads
`aegis_config` out of `MODEL.SAF`'s metadata at runtime, same as
`cis_witness`. Each step guards its own prompt length against
`max_position_embeddings` and panics (not a silent truncation) if a step's
prompt plus N would exceed it.
