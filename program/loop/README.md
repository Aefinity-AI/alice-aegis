# `ev` — the on-device experiment loop

One command. No GPU, no cloud, no model. Every verb works during a 529.

    ln -s ~/program/loop/ev ~/.local/bin/ev && ev help

## The one-sentence thesis

Every throughput number this program has **retracted** came out of a hand-run
whose stdout was pasted into a commit message. Every number that **survived** the
DARPA forensic audit came out of a script that wrote a file. The bare-metal
target has had a harness since day one (`/gauntlet` → `BOOTLOG.TXT` →
`collect_gauntlet.sh` → `gauntlet_dataset.tsv`) and its numbers held. The dev box
never had one. `ev` is the dev box's harness, plus the binding that stops a dead
number from walking back into a document.

## The four layers

| Layer | Tool | Question it answers | Model needed |
|---|---|---|---|
| RUN | `ev run`, `runcard.py` | what ran, on what, from which tree, and what came out | never |
| BIND | `ev claim`, `ledger.py`, `evidence_check.py` | does this number have a log | never |
| GATE | `ev lint`, `ev gate`, `claim_gate.sh` | does this document state a number we killed | never |
| JUDGE | `ev park`, `ev accept` | is this wording honest / is this confound controlled | yes, and **only** here |

The split is the whole design. Layers 1–3 are the program. Layer 4 is an
optional upgrade that can be queued for hours or days without stopping anything.

## Daily shape

```bash
# morning: is the machine fit to measure on?
ev env --nick chromebook-crosvm

# run something. refuses if another process holds >25% of a core.
ev run thread_sweep                 # closes 3 CRITICAL + 3 HIGH audit defects
ev run membw                        # closes the one claim with NO source at all

# bank the result
ev claim add --id A4.4t --value 3.370 --unit x --kind measured \
  --statement "decode speedup, 4 workers vs 1" \
  --scope "i5-10210U crosvm guest (topology FLATTENED - no SMT claim possible), \
BitNet-2B, int8_act+parallel, 5 rounds x 3 iters interleaved" \
  --source docs/hardware_logs/thread_sweep_<ts>.log \
  --runid <runid> \
  --ceiling "oversubscription only; median of 15; IQR 3.7%"

# before anything leaves the house
ev gate docs/TECHNICAL_REPORT.md
```

## Run state that survives a crash and an outage

```bash
ev start m7lr --prereg docs/hardware_logs/m7lr_PREREGISTRATION_2026-07-29.md \
  --question "How much of the M7 gap is the undisclosed LR cooldown?" \
  --phase arm_H:deterministic --phase arm_K:deterministic \
  --phase score:deterministic --phase band:deterministic \
  --phase scope_wording:interpretive

ev set    m7lr arm_H done --artifact <log>          # executor self-report, safe
ev verify m7lr arm_H -- python3 <checker>           # only a checker writes `verified`
ev park   m7lr scope_wording --question "..."       # writes a zero-context packet
ev resume m7lr                                      # first NON-TERMINAL phase
ev review-queue                                     # burn this down when the API is back
```

`set` physically cannot write `verified` or `accepted`. `resume` stops at a
`done`-but-unchecked phase — the one a naive resume skips.

## Pre-registration with teeth

`docs/hardware_logs/m7lr_PREREGISTRATION_2026-07-29.md` is already the best
instrument in this repo. Add one ```ev-prereg``` block (see
`templates/PREREG_EXAMPLE_m7lr.md`) and its §5 bands, §8 banned sentences and
sanity aborts become executable. **The prose does not change.**

```bash
ev prereg lock   <PREREG.md>                    # tracked + unmodified in git, or it is not a prereg
ev prereg bands  <PREREG.md> <results.json>     # exactly one band must match
ev prereg banned <PREREG.md> program/RESEARCH_LEDGER.md docs/*.md
```

## What is in this tree

```
program/loop/
  ev                          the dispatcher
  claims.jsonl                append-only machine-readable claim ledger
  seed_retractions.sh         one-time: enter the already-dead numbers
  tools/
    runcard.py                env fingerprint + capture + re-verifiable receipt
    ledger.py                 claim add / retract / verify / values
    claimlint.py              document -> ledger  (the increment over ARIS)
    evidence_check.py         claim -> source     (vendored from ARIS, MIT, unmodified)
    runq.py                   phase state machine, done != verified != accepted
    prereg.py                 band evaluator, sanity aborts, banned sentences
  runners/
    thread_sweep.sh           + thread_sweep_parse.py   the missing multicore bench
    membw.sh                  the missing bandwidth bench
  hooks/
    claim_gate.sh             PostToolUse: no edited file may state a killed number
    settings.snippet.json     how to wire it beside the existing integrity_gate.sh
  patches/
    01_gauntlet_build_identity.md   4 columns that make the bare-metal TSV A/B-able
  templates/
    PREREG_EXAMPLE_m7lr.md    the real prereg with its machine-checkable block
  state/runs/                 one JSON per experiment
  state/review_packets/        what the frontier model owes you
docs/hardware_logs/runcards/  one JSON receipt per execution
aegis-linux/examples/membw.rs the bandwidth probe
```

## Deliberate non-goals

- It does not replace `program/RESEARCH_LEDGER.md`. That file's dense adversarial
  prose is better than any schema; `claims.jsonl` is its machine-readable shadow.
- It does not replace `CLAUDE.md`, `guard.py`, or `integrity_gate.sh`.
- It does not judge prose. `ev gate` PASS means *mechanically clean*, which is
  not *reviewed*, and the tool prints that distinction every time.
