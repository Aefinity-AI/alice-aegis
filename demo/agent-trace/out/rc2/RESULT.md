# rc-2: three agent-trace episodes, hash-chained receipts, cross-machine verify

Three `AEGIS-TRACE v2` (format 3) receipts generated over the pinned bitnet-2B
artifact triple (matches `.github/workflows/bitnet2b-receipt.yml`
`artifacts-bitnet2b-2026-08-27` pins: model
`facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db`, embed
`e32b99a25e345c65054f36dedf40329a89513cdf1a8195db64fc440fd364e077`, vocab
`5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae`), one per
scenario, each ≤20 steps. Full transcripts: `logs/box1-local-verify.log`,
`logs/box2-daemon-verify.log`.

## Scenarios

| receipt | steps | shape | tool calls |
|---|---|---|---|
| `scenario1-tool-use.receipt` | 3 | few-shot `CALC` prompt | step 0: `CALC(11 - 9)` -> `2` |
| `scenario2-reasoning.receipt` | 5 | chained prime-check reasoning, no table | none (pure decode chain) |
| `scenario3-plan-execute.receipt` | 3 | stated 3-item plan, then `CALC` execution steps | steps 0-2 all call `CALC` |

## Verify results

| receipt | box1 (aefinity-box, local `agent_trace verify`) | box2 (aefinity-box2, `cm-gateway.service` daemon worker) | phone |
|---|---|---|---|
| scenario1-tool-use | **PASS** — trace-chain `2e7b42d1ce4c904b` matched | **PASS** (`ALLOW idx=2 cap=5a1de024…`, ~7.5 min) | unreachable |
| scenario2-reasoning | **PASS** — trace-chain `6e93db62c1b09e6d` matched | **DENY — gateway's 900s `--request-timeout` exceeded**, not a receipt-validity failure (see below) | unreachable |
| scenario3-plan-execute | **PASS** — trace-chain `0644c907ff3033fe` matched (2 lenient `WARNING step N` lines, receipt still verifies) | **PASS** (`ALLOW idx=6 cap=e56c6a3d…`, ~9.5 min) | unreachable |

All three receipts were generated AND independently locally verified on box1
first (`agent_trace verify`, direct — box1 has no host restriction). The
box2 leg used ONLY the daemon's single background verify worker
(`RECEIPT`/`ACTION`/`SESSION`/`COUNTER` over
`$XDG_RUNTIME_DIR/cm-gateway.sock`, `PENDING ticket=` then poll to `ALLOW`/
`DENY`), per the box2 HOST RULE (`state/QUEUE.md` line 57: box2 must never
run a 2B replay/generation directly; the daemon's one worker is the
permitted exception). Phone replay was not attempted beyond checking for
`adb` — none present on box1 or box2 (same finding as pc-2, `state/reports/2026-09-22-pc2-smollm2-360m.md`) — best-effort, not a blocker for this task.

### scenario2 box2 DENY — root cause

`agent_trace verify` for a 5-step 2B episode did not finish within the
gateway's fixed `--request-timeout 900` (15 min) on box2's Celeron N4020
(no AVX2/FMA); the daemon killed the subprocess and replied
`DENY agent_trace verify: could not run: agent_trace verify exceeded
--request-timeout (900s), killed`. scenario1 (3 steps) and scenario3
(3 steps) both finished within the same 900s budget (~7.5 min and ~9.5 min
respectively). This receipt already VERIFY PASSed bit-for-bit on box1
(`logs/box1-local-verify.log`) — the DENY here is honest evidence about
box2's wall-clock budget for a longer 2B episode under the daemon's current
fixed timeout, not a defect in the receipt or the trace-chain mechanism.
Not fixed as part of this task (out of scope — would mean either raising
`--request-timeout` on the live `cm-gateway.service` unit, which is shared
safety-track infrastructure, or shortening scenario2, which would violate
its "5-step reasoning chain" scenario intent).

## An in-flight correction, disclosed

Before using the daemon-worker path, an early attempt in this tick built
`agent_trace` directly in a `/tmp` worktree on box2 and ran `agent_trace
verify` as a bare process against the 2B triple — a violation of the box2
HOST RULE. This was caught mid-run (before any receipt in this report was
produced from it), the offending processes were killed, the temporary
worktree removed, and `cm-gateway.service`'s health was confirmed before
continuing exclusively via the daemon-socket protocol documented above. No
receipt or verify result in this report comes from that mistaken run. Full
detail in `logs/box2-daemon-verify.log`'s "CORRECTION NOTE".

## What a box2 ALLOW here proves (and does not)

Same as `state/reports/2026-09-14-safe5e-twobox-2b.md`: an `ALLOW` from the
gateway means its cheap checks (artifact-triple allowlist membership,
verify-execute action-hash binding, freshness) all passed AND its own
`agent_trace verify` subprocess, run against box2's own copy of the pinned
2B artifact triple, reproduced the exact trace-chain the receipt claims. It
does not prove anything about the quality of the model's output (see
scenario3's note on the model echoing a malformed few-shot pattern in
`logs/box1-local-verify.log`) — only that the recorded episode replays
bit-for-bit on independent hardware.

## Not attempted

- Phone replay-verify (no `adb` on box1 or box2).
- Raising box2's `cm-gateway.service` `--request-timeout` to accommodate
  longer episodes (shared safety-track infra, out of scope here).
- A LOOKUP-based (table-driven) plan-execute scenario through the box2
  daemon — the daemon's live table pin (`tiny/tables/chain.tsv`) differs
  from `demo/agent-trace/tables/demo.tsv`; this is a pre-existing,
  previously-documented gap (`state/LOG.md` 2026-09-14, safe-5e "Notes").
  scenario3 was built CALC-only instead to route around it; a
  LOOKUP-flavored draft of scenario3 (against `demo.tsv`) still exists at
  commit `e188744` on this branch and VERIFY PASSed locally on box1 if a
  future task wants to specifically re-probe the table-pin gap.
