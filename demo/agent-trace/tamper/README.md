# Agent-episode receipt tamper matrix

Adversarial test kit for `agent_trace`: take a receipt that verifies, mutate
it several hundred ways, and require the verifier to reject every one.

## Why round 4 exists

Round 3 caught 360/360 mutants with 0 still-verifying — but only 198 of those
rejections came from a digest or replay comparison. The other 162 were parse
rejections. A parse rejection is a real defence (a tamperer must produce a
well-formed receipt) but it is **not evidence that the hash chain binds**: it
only says the parser is strict. Counting the two together overstates what the
receipt proves.

Round 3 proposed fixing this by raising the "cryptographic share" above 80 %.
That target was withdrawn, because the share is not a property of the receipt
format at all — both counts are chosen by whoever writes the mutator, so the
share can be set to any value by adding or deleting malformed mutants. It is a
statistic about the test suite, not about the thing under test.

Round 4 replaces it with a claim that does not have that defect:

> *N* mutants that are **well-formed receipts**, each certified as such by a
> grammar checker written independently of the verifier, were every one
> rejected, and **none of them by the parser**.

The count of well-formed mutants is a lower bound you can raise by working
harder; it can never be inflated by adding junk.

## Pieces

| file | what it does |
|---|---|
| `mutate4.py` | Emits two labelled buckets from one receipt: **W**, well-formed by construction, and **M**, malformed on purpose. |
| `wellformed.py` | Independent grammar checker for AEGIS-TRACE receipts, written from the format description rather than from `agent_trace.rs`. Decides bucket membership. |
| `e23r4.sh` | The harness: generates baselines, mutates, verifies every mutant in parallel, classifies each rejection, checks nine acceptance criteria fixed before the run. |

## What makes a mutant well-formed

Every value `mutate4.py` substitutes into bucket W satisfies its field's
grammar, usually because some genuine receipt in the corpus carried exactly
that value:

- a 64-hex slot gets another 64-hex value from the corpus, or one nibble flipped;
- `prompt-hex` gets another episode's real prompt; `commit` gets 40 hex; `host`
  gets a real hostname; `N` gets a positive integer strictly below the genuine
  one, so the prompt-plus-N position bound cannot be what rejects it;
- `tool=` gets a different *valid* tool name, never a corrupted one;
- `toks=`, `in=`, `out=` get another step's real values, or a token id bumped
  by one with the list length preserved;
- structural edits — reorder, drop, duplicate, rotate — **renumber the step
  labels and correct `K`**, so the result is an internally consistent receipt
  describing a *different* episode. Round 3 made these edits without
  renumbering, which is why all 36 of them were rejected by the step-label
  check instead of by the chain.

Nothing but a digest or replay comparison can reject a receipt like that.

## What the harness refuses to let you get away with

- **No catch-all in the rejection classifier.** Round 3's ended in
  `*) why=parse-reject`, which silently filed 14 `FAIL artifact: … hash
  mismatch` rejections — genuine digest comparisons — as parse failures, and
  understated its own result. Here an unrecognised message is `unclassified`
  and `unclassified > 0` fails acceptance.
- **Bucket labels are not taken on trust.** Every emitted mutant is re-checked
  by `wellformed.py`; a disagreement between the mutator's label and the
  checker's verdict fails acceptance and is printed, never silently resolved.
- **The grammar checker is itself controlled.** It must accept all four genuine
  receipts, or the run refuses before mutating anything.
- **No mutant may be byte-identical to a genuine receipt** (criterion A9), so a
  "rejection" can never be an artefact of comparing a file to itself.
- **A coverage gap aborts the run.** If an episode that is supposed to fire a
  tool fires none, `in=`/`out=` are empty and the tool half of the claim is
  untested. That run can never be accepted, so it stops immediately.

## Running it

```sh
E23_JOBS=3 bash demo/agent-trace/tamper/e23r4.sh
```

Needs the BitNet-2B artifacts under `~/aefinity-artifacts/bitnet2b-2b-artifacts`
(override with `E23_ART`) and the kit at `~/legs/e23r4-kit` (override with
`E23_KIT`). Writes to `~/legs/e23r4-wellformed-tamper-matrix-2b/`. Prints no
timing numbers, ever, and commits nothing.

The episode spec in the harness is recovered from round 3's own baseline
receipts — prompt-hex decoded, `table-sha256` matched against the tables in the
tree — not from a driver script, because an earlier draft of that script used
different prompts and tables that produce no tool calls at all.
