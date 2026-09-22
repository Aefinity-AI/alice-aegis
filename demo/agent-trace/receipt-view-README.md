# receipt-view — plain-English report for an AEGIS-TRACE receipt

`receipt-view.py` (in this directory) is a single-file, stdlib-only Python
wrapper around the real verifier, `agent_trace verify` (built from
`aegis-linux/examples/agent_trace.rs`, see `demo/agent-trace/README.md`).
It does not reimplement any hashing or replay logic — it shells out to
`agent_trace verify`, parses its stdout, and reformats the verdict as
prose. If `agent_trace verify` prints something this tool's parser does not
recognise, it says so ("UNKNOWN") rather than guessing.

For a receipt, it prints:

- **what ran** — model/embed/vocab artifact hashes, host, commit, K/N, the
  initial prompt (hex-decoded), and one line per step: which tool was
  called and its decoded input/output.
- **whether the hash chain is intact or broken** — the verdict
  `agent_trace verify` reached, replaying the episode locally.
- **if broken, the first broken step and what changed** — parsed from
  verify's own `step N divergence: ...` / `STEP N CTX MISMATCH` /
  `STEP N QUERY MISMATCH` / step-count-mismatch / structural-failure lines,
  translated into plain field names (token ids, which tool ran, the tool's
  recorded input, the tool's recorded output, or the step's decode-chain
  digest).
- **whether it was verified offline** — yes/no. This tool and
  `agent_trace verify` both run entirely against local artifact files and
  make no network call; "no" is only reported when the check could not be
  completed at all (missing binary, missing/mismatched artifacts, or a
  pre-replay structural rejection).

## Usage

```
demo/agent-trace/receipt-view.py <receipt-file> \
    [--model PATH] [--embed PATH] [--vocab PATH] [--artifacts DIR] \
    [--table PATH] [--bin PATH] [--raw]
```

Same env-var conventions as `demo/agent-trace/run.sh`: `AEGIS_ARTIFACTS`,
`AEGIS_MODEL`, `AEGIS_EMBED`, `AEGIS_VOCAB`, `AEGIS_TABLE`,
`AEGIS_AGENT_TRACE_BIN`. `--raw` also prints `agent_trace verify`'s raw
stdout/stderr after the formatted report. Exit codes: `0` chain intact,
`1` chain broken (verify ran and rejected the receipt), `2` the check could
not be run at all (missing receipt/binary/artifacts), `3` verify's output
did not match any pattern this tool recognises.

Build the verifier first: `demo/agent-trace/run.sh build`.

## Mutation test (4 cases)

`demo/agent-trace/receipt-view-fixtures/` holds a real, verifying baseline
receipt (`good-receipt.receipt`) generated against the BitNet-2B artifacts at
`~/aefinity-artifacts/bitnet2b-2b-artifacts` (sha256: model
`facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db`, embed
`e32b99a25e345c65054f36dedf40329a89513cdf1a8195db64fc440fd364e077`, vocab
`5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae` — not
checked into this repo; same artifacts already referenced by this demo's
main README "Evidence so far" section), prompted with a two-shot CALC
few-shot so the model actually emits a tool call (`step 0: tool=calc,
CALC(10 + 10) -> 20`), K=2, N=16. `demo/agent-trace/receipt-view-fixtures/`
also holds four single-field mutants of that receipt, one per required
case, each editing only the field named:

| # | mutation | file | edit |
|---|---|---|---|
| 1 | changed token | `m1-changed-token.receipt` | step 0's first `toks=` id bumped by 1 |
| 2 | changed tool-result | `m2-changed-tool-result.receipt` | step 0's `out=` hex last nibble bit-flipped (`"20"` -> `"21"`) |
| 3 | flipped bit, intermediate value | `m3-flipped-bit-decode-chain.receipt` | step 0's `decode-chain=` (the per-step chained digest over the model's decode) last nibble bit-flipped |
| 4 | reordered step | `m4-reordered-step.receipt` | step 0's and step 1's entire field bodies swapped (labels left as `step 0:`/`step 1:`) |

`demo/agent-trace/receipt-view.py` was run against the good receipt and
each mutant, against the same BitNet-2B artifact triple:

```
demo/agent-trace/receipt-view.py demo/agent-trace/receipt-view-fixtures/<file>.receipt \
    --model <bitnet2b>/aegis_pruned_model.cis.safetensors \
    --embed <bitnet2b>/embed.bin --vocab <bitnet2b>/vocab.bin
```

### Result

| case | mutation applied | receipt-view verdict | first broken step named | field named | PASS (correctly caught)? | step named correctly? |
|---|---|---|---|---|---|---|
| baseline | none | INTACT | n/a | n/a | n/a | n/a |
| 1 | changed token | BROKEN | step 0 | "the step's token ids" | yes | yes |
| 2 | changed tool-result | BROKEN | step 0 | "the tool's recorded output" | yes | yes |
| 3 | flipped bit (intermediate value) | BROKEN | step 0 | "the step's decode-chain digest (the model's own chained decode state for that step)" | yes | yes |
| 4 | reordered step | BROKEN | step 0 | all five fields (toks/tool/in/out/decode-chain) — the whole step content moved | yes | yes |

All 4 mutations were correctly caught and each was correctly reported as
first broken at step 0 (the step whose content each mutation actually
changed — for case 4, swapping steps 0 and 1 means step 0's line no longer
matches what replay computes for step 0, which is what the verifier's own
`step 0 divergence: ...` line reports; `receipt-view` does not invent a
"this was a reorder" diagnosis beyond what `agent_trace verify` itself
prints — it only translates the divergence flags into field names).

Exact `receipt-view.py` output for each case (verbatim, `agent_trace
verify`'s own line quoted in each report's last parenthetical) is
reproducible by running the command above against each fixture file; the
baseline and mutant #3 outputs are, respectively:

```
$ demo/agent-trace/receipt-view.py demo/agent-trace/receipt-view-fixtures/good-receipt.receipt ...
HASH CHAIN
  INTACT — agent_trace verify replayed the episode locally and
  reproduced every step and the final trace-chain digest bit-for-bit.
VERIFIED OFFLINE: yes ...

$ demo/agent-trace/receipt-view.py demo/agent-trace/receipt-view-fixtures/m3-flipped-bit-decode-chain.receipt ...
HASH CHAIN
  BROKEN — agent_trace verify rejected this receipt.
  First broken step: step 0
  What changed at that step:
    - the step's decode-chain digest (the model's own chained decode state for that step)
  (verifier line: step 0 divergence: toks-match=true tool-match=true in-match=true out-match=true decode-chain-match=false)
VERIFIED OFFLINE: yes — the check ran fully offline and correctly
rejected this receipt (see HASH CHAIN above).
```

## What this tool does NOT do

It does not verify anything itself — `agent_trace verify` is the sole
source of truth for PASS/FAIL. It does not check attestation
(`demo/edge-receipt/attest.sh`) or bundle integrity (`run.sh
verify-bundle`); for those, run the underlying tools directly. It makes no
claim about model quality, speed, or "first" (see this demo's main
README, "What a PASS does NOT prove").

## Relation to the existing tamper kit

`demo/agent-trace/run.sh tamper` and `demo/agent-trace/tamper/mutate4.py`
already exercise much larger adversarial mutation sets (4 and hundreds of
cases respectively) against the raw verifier. The four fixtures here are a
small, human-readable set chosen to match one example of each of the four
mutation *kinds* those larger harnesses cover in bulk (single-field value
edit at the token/tool-result/digest level, and one structural edit), used
specifically to exercise and demonstrate `receipt-view`'s own plain-English
diagnosis, not to re-measure the verifier's tamper-resistance (already
measured in `demo/agent-trace/tamper/README.md`'s E23 round 4).
