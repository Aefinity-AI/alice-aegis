# safe-9: EVAL-60 T1 through the receipt gateway

Two scripts used to re-verify all 60 EVAL-60 T1 receipts (`demo/agent-trace/eval/`)
through this repo's async receipt gateway (PR #99), using the same
RECEIPT/ACTION/SESSION/COUNTER -> `PENDING ticket=...` -> `POLL` -> ALLOW/DENY
protocol as safe-8's `demo.sh` (unmerged `cm/safe8-onecommand-demo`, aad8737).

- `run_gateway_eval.sh` — starts a `gateway serve` daemon against the real
  BitNet-2B artifacts and submits the 50 EVAL-60 T1 items whose receipt has a
  real (non-empty) tool-call action.
- `rerun_notool.sh` — same protocol for the 10 EVAL-60 T1 items whose model
  emitted no tool call at all (receipt's last step has `in=` present but
  empty); submits an explicitly empty ACTION, matching the receipt's own
  empty `last_step_input` exactly.

Both scripts hardcode `/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts` and
`/home/cm/legs/eval-60-replay-box1/receipts` (this box's paths) — adjust for
another machine. Full methodology, per-item verdicts, and the resulting score
(only over gateway-ALLOWed items, per this track's instruction) are in
`claudius-maximus/state/reports/2026-09-14-safe9-eval60-gated.md`.

Result on aefinity-box (box1), 2026-09-14: 60/60 ALLOW, 0 DENY — the gated
score is therefore identical to the ungated EVAL-60 T1 score.
