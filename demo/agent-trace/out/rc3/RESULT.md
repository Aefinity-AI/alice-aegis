# rc-3: receipt overhead table (gen, size, verify)

box1 = aefinity-box (i5-5200U, AVX2, no BD-PROCHOT clamp evident at run time —
`/proc/cpuinfo` MHz samples 2258-2640 across cores). box2 = aefinity-box2
(Celeron N4020, no AVX2/FMA, 192.168.10.21). Scenario reused verbatim from
rc-2 (alice-aegis PR #109, `demo/agent-trace/out/rc2/scenario1-tool-use.receipt`):
`scenario1-tool-use`, K=3 steps, N=24 tokens/step budget, few-shot CALC
prompt, over the pinned bitnet2B artifact triple (matches
`.github/workflows/bitnet2b-receipt.yml` `artifacts-bitnet2b-2026-08-27`
pins: model `facb3597…`, embed `e32b99a2…`, vocab `5bde1b03…`). Regenerated
in this tick (worktree `~/projects/alice-aegis-cm-rc3`, branch
`cm/rc3-overhead-table`, base commit `e72427e`) with the exact same prompt
bytes as rc-2 — the regenerated receipt is byte-identical to rc-2's
scenario1 receipt in every field except `commit` (rebuilt from a later
commit) and the resulting `trace-chain`/`decode-chain` hashes that fold it
in (verified by diff, see logs/).

## What "with receipts" vs "without receipts" means here

`agent_trace gen` (K=3 N=24) runs the SAME underlying greedy FullInt decode
as `cis_decode`, but (a) re-prefills from scratch at the start of each of
its 3 steps instead of one continuous prefill+decode, (b) scans each step's
output for a `CALC(...)` tool call and executes it, and (c) builds+writes
an AEGIS-TRACE v2 hash-chained receipt (genesis fold, per-step
`decode-chain`/`ctx`/`q` digests, trace-chain fold). `cis_decode` does none
of that: one prefill, N=72 (=K×N) tokens straight through, one summary
digest line, no receipt. So the delta below is the cost of running this
*episode* (K-step tool-use loop) with full receipt-minting, vs. an
equivalent-token-count plain decode with NEITHER the multi-step tool-use
loop NOR the receipt. **This conflates receipt-minting overhead with the
K-step re-prefill architecture of `agent_trace`'s episode loop — it is NOT
an isolated "cost of the hash chain alone" number.** No code change was
made to instrument the receipt-writing block in isolation (out of scope for
this task); see cost-1 (`state/reports/2026-09-17-cost1-overhead.md`) for a
same-model K=1 (single-step, no multi-step re-prefill) comparison that is
closer to an isolated receipt-write cost: 31.2% overhead there, vs. the
larger delta below for this multi-step K=3 episode.

## Measured numbers

| what | machine | value | command | log |
|---|---|---|---|---|
| gen WITH receipts (agent_trace gen, K=3 N=24) | box1 | wall 44.74s (user 44.23s) | `agent_trace gen $M $E $V 3 24 "$PROMPT"` | `logs/box1-gen-with-receipt.log` |
| gen WITHOUT receipts (cis_decode, N=72, same prompt) | box1 | wall 17.84s (user 17.46s) | `cis_decode $M $E $V 72 "$PROMPT"` | `logs/box1-gen-without-receipt.log` |
| receipt size | box1 | 1599 bytes | `wc -c scenario1-tool-use.receipt` | `scenario1-tool-use.receipt` |
| verify | box1 (local, direct) | wall 40.08s (user 39.66s), PASS | `agent_trace verify $M $E $V scenario1-tool-use.receipt` | `logs/box1-verify.log` |
| verify | box2 (cm-gateway.service daemon worker — the only box2 HOST-RULE-permitted 2B path) | ALLOW (PASS), ~12m13s submit-to-ALLOW, polling-bounded | RECEIPT/POLL protocol over `$XDG_RUNTIME_DIR/cm-gateway.sock` | `logs/box2-daemon-verify.log` |
| verify | phone | unreachable, not attempted (no adb on box1/box2, same finding as pc-2/rc-2) | n/a | n/a |
| peak RSS (gen w/ receipt) | box1 | 1,406,336 KB (`/usr/bin/time -v` Maximum resident set size) | — | `logs/box1-gen-with-receipt.log` |
| peak RSS (verify) | box1 | see log (`/usr/bin/time -v`) | — | `logs/box1-verify.log` |
| energy | box1/box2 | not measured this tick (RAPL exists on box1 but was not wired into these runs) — n/a, not estimated | — | — |

Overhead (gen, box1, this episode shape): 44.74s / 17.84s = **2.51x**
(151% slower), NOT an isolated receipt-cost figure — see caveat above.

## box2 leg

Per `state/QUEUE.md` HOST RULE (box2 must never run 2B verify/generation
directly — the only permitted path is `cm-gateway.service`'s single
background verify worker): the rc-3 receipt (`scenario1-tool-use.receipt`,
byte-identical episode content to rc-2's, see diff note above) was copied
to box2 (`scp` to `/tmp/rc3-scenario1-tool-use.receipt`) and submitted over
the daemon's unix socket with the RECEIPT/ACTION/SESSION/COUNTER protocol,
then polled to completion — see `logs/box2-daemon-verify.log` for the full
transcript and result. rc-2 already ran this same scenario shape through
the same daemon and got ALLOW in ~7.5 min wall (polling-bounded, not a
precise `time` measurement, since the client only sees PENDING/ALLOW/DENY);
this tick's submission got ALLOW after ~12m13s (submitted 14:43:24Z, ALLOW
14:55:37Z, polling-bounded the same way) — slower than rc-2's figure for
the same scenario shape; no control was made for other daemon-queue
activity between the two ticks, so both numbers are reported as-observed
rather than treated as "the" box2 figure for this scenario. See
`logs/box2-daemon-verify.log` for the full transcript.

## Honest summary

Every number here names its machine and points to a committed log under
`demo/agent-trace/out/rc3/logs/`. The "with vs without receipts" gen
comparison is for the full `scenario1-tool-use` K=3 episode as actually run
in rc-2/rc-3, not an isolated ablation of the receipt-writing code path —
it also includes `agent_trace`'s per-step re-prefill loop, which
`cis_decode`'s single continuous decode does not do, so the 2.51x figure
overstates the cost attributable to receipts alone (cost-1's K=1 same-model
figure, 31.2%, is the closer isolated estimate). Receipt size (1599 bytes)
and box1 verify time (40.08s) are direct, un-caveated measurements. box2's
verify result is polling-bounded (daemon protocol has no wall-clock RPC),
consistent with rc-2's ~7.5min figure for the same scenario shape. Phone
was not reachable (no adb) — reported as unreachable, not estimated.
No RSS-vs-energy claim is made: RSS is reported (from `/usr/bin/time -v`),
energy is explicitly "not measured this tick" (the rc-3 ADDENDUM item in
`state/QUEUE.md` covers a dedicated energy-measurement pass; not done as
part of this task).
