# agent-trace tool-call eval — suite, runner, scorer

This directory implements the pre-registered plan at
`claudius-maximus/state/reports/2026-09-06-TOOLCALL-EVAL-PLAN.md` (sections
2-5): a 60-item deterministic suite of CALC/LOOKUP tool-call prompts, a
runner that turns each item into one AEGIS-TRACE receipt (`gen` then
`verify`, same mechanism as `demo/agent-trace/run.sh`), and a scorer that
reports Wilson-interval rates per the plan's metrics.

**Nothing is measured yet.** This directory is the harness, committed before
any run. `suite.tsv` and `smoke.tsv` are generated files (see below); no
receipts, summaries, or rate numbers are checked in.

## Files

- `gen_suite.py` — deterministic item generator (stdlib only, seed
  `20260906`). Writes `suite.tsv` (60 items) and `smoke.tsv` (10 items: 2
  each from the 5 largest buckets). Re-run any time; output is
  byte-identical run to run.
- `suite.tsv` / `smoke.tsv` — generated TSVs, header:
  `item_id bucket prompt_template_id prompt_text expected_tool expected_input expected_output notes`.
  `prompt_text` newlines are escaped as the two-byte sequence `\n` (backslash-n)
  because TSV rows are one-per-line; `run_suite.sh` reverses this escaping
  before writing the prompt to a file for `gen`.
- `test_calc_semantics.py` — stdlib `unittest` checking the suite's Python
  CALC implementation against the Rust tool's vectors.
- `run_suite.sh` — runs one TSV of items through `agent_trace gen` +
  `agent_trace verify`, writes `<outdir>/summary.tsv`, `<outdir>/RUN.txt`,
  `<outdir>/timing.tsv`, and per-item receipts/prompts. Resumable.
- `score.py` — reads a `summary.tsv` and prints per-bucket / overall rates
  with Wilson 95% intervals, plus the plan's pre-registered red flags.

## Generating the suite

```
cd demo/agent-trace/eval
python3 gen_suite.py
```

Prints the sha256 of `suite.tsv` and `smoke.tsv`. Run twice to confirm
byte-identical output (the script does not depend on anything but stdlib
`random.Random(20260906)` and a fixed generation order).

## Running the smoke suite

Uses the same artifact-path convention as `demo/agent-trace/run.sh`
(`AEGIS_ARTIFACTS=<dir>` with `MODEL.SAF`/`EMBED.BIN`/`VOCAB.BIN` inside it,
or the three `AEGIS_MODEL`/`AEGIS_EMBED`/`AEGIS_VOCAB` env vars individually).

In-repo M7 tinybit model (never emits a tool call within N=24 tokens — that
is expected and fine for a mechanism smoke test, not a rate measurement):

```
AGENT_TRACE_BIN=/path/to/aegis-linux/target/release/examples/agent_trace \
  ./run_suite.sh smoke.tsv /tmp/eval-smoke-m7
```

2B artifacts (note: the 2B artifact directory does NOT follow the
`MODEL.SAF`/`EMBED.BIN`/`VOCAB.BIN` naming `AEGIS_ARTIFACTS` expects — its
files are named `aegis_pruned_model.cis.safetensors`, `embed.bin`,
`vocab.bin` — so override the three individual variables, not
`AEGIS_ARTIFACTS`):

```
AEGIS_MODEL=~/aefinity-artifacts/bitnet2b-2b-artifacts/aegis_pruned_model.cis.safetensors \
AEGIS_EMBED=~/aefinity-artifacts/bitnet2b-2b-artifacts/embed.bin \
AEGIS_VOCAB=~/aefinity-artifacts/bitnet2b-2b-artifacts/vocab.bin \
AGENT_TRACE_BIN=/path/to/aegis-linux/target/release/examples/agent_trace \
  ./run_suite.sh smoke.tsv /tmp/eval-smoke-2b
```

`--template T3` runs the unclosed-`CALC(` ablation instead of the default T1
prompts. `--limit N` caps how many not-yet-done items are run this
invocation (the run is resumable — items already in `summary.tsv` are
skipped on a re-run).

## Scoring

```
python3 score.py /tmp/eval-smoke-m7/summary.tsv
```

Works on a partial `summary.tsv` (fewer rows than the suite/smoke file).

## Deviations from the plan (logged, not silent)

- **`//` operator does not exist.** Plan section 2 lists CALC hard-bucket
  ops as `+ - * //`. `agent_trace.rs`'s `find_calc` grammar only supports
  `+ - * / %` (see its module doc comment and `eval_calc`). `gen_suite.py`
  uses `/` (truncating toward zero, matching `i64::checked_div`) and `%`
  (remainder with the sign of the dividend, matching `i64::checked_rem`)
  instead of the plan's non-existent `//`.
- **Suite-hash folding into the receipt — DONE.** Plan section 5 noted that
  folding the eval item-suite's own sha256 into the cryptographic trace
  chain (the way `table-sha256` is folded for LOOKUP items) "if... turns
  out to need a code change, that change is a NEEDS item for alice-aegis,
  not something done inside this plan." That code change has since landed:
  `agent_trace gen`/`verify` accept an optional `--suite-sha256 <64hex>`
  that folds the given 32 bytes into the trace genesis after the table
  slot, under its own domain tag (the pre-existing table fold is
  byte-for-byte unchanged, so archived table-bound receipts keep
  verifying) and records
  a `suite-sha256 <64hex>` receipt header. `run_suite.sh` passes
  `--suite-sha256 $(sha256sum "$ITEMS")` to every `gen` and `verify` call,
  so every receipt this harness produces now binds the suite file's hash
  into its own cryptographic chain — a receipt generated under a different
  suite file will not verify unless the same hash is supplied or embedded,
  for CALC-only items as well as LOOKUP items. `RUN.txt`'s `suite-sha256`
  line remains as the human-readable out-of-band record.
- **Mixed-item (K=2) column conventions.** The plan's summary/suite TSV
  formats do not define how a 2-step mixed item's `expected_tool`/
  `tool_observed`/`arg_match`/`output_match` should represent two steps in
  one row. This harness comma-joins step-1,step-2 values (e.g.
  `expected_tool` = `CALC,LOOKUP`) in both `suite.tsv` and `run_suite.sh`'s
  `summary.tsv`. `arg_match`/`output_match` are single booleans (true only
  if *both* steps match); a finer per-step breakdown was judged unnecessary
  for a 3-item bucket.
- **Mixed items' second step is not forced.** `replay_episode` (the shared
  gen/verify core) only lets the model's own continuation, plus the fixed
  `TOOL[<name>]=<output>` text `agent_trace.rs` appends after step 1,
  determine what step 2 sees. This harness cannot and does not force the
  model into asking the second (LOOKUP or CALC) question — it primes both a
  CALC and a LOOKUP few-shot example in the initial prompt and reports
  whatever step 2 actually does. A mixed item where step 2 never asks the
  intended second question is a legitimate model-behavior finding, not a
  suite bug.
- **`well-formed rate` operationalization in `score.py`.** The plan defines
  well-formed as "fraction of emitted calls that parse under the existing
  scanner grammar... without the malformed/ambiguous failure modes already
  fixed in review." This harness has no independent access to the model's
  raw un-scanned decode text (only the receipt's already-scanned
  `tool=`/`in=`/`out=` fields), so `score.py` treats "well-formed" as
  "the scanner recognized *some* call" (`tool_observed` != `NONE` for a
  tool-expected item) — a proxy for the scanner's own accept decision, not
  a re-implementation of the grammar. `calc-error` (parsed but a
  runtime `div-by-zero`/`overflow`) counts as well-formed and as the
  correct tool, but not necessarily correct-argument/output (those still
  compare literal strings).

## What `RUN.txt` records

One header written once per `<outdir>`, before the first item: the suite
file's path and sha256, the table file's path and sha256, the three
artifact file sha256s, the `agent_trace` binary's sha256, `N`, the
`--template` in effect, `hostname`, and a UTC start timestamp. This is
provenance for the run as a whole; per-item reproducibility comes from each
item's own receipt file plus `agent_trace verify`, independent of this
report's summary numbers (plan section 5).

## Caveat: comma-joined fields

For mixed (K=2) items the `expected_tool`, `expected_input`, `expected_output`
columns in the suite and the `tool_observed`, `arg_match`, `output_match`
columns in `summary.tsv` are the two steps' values joined with a comma. Table
values themselves may contain commas, so these fields are compared as whole
strings and must not be split on `,` by any future consumer.
