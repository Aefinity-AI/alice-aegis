# sv-3 pre-registered predictions

Written after `gen_tasks.py` (which already embeds prediction (a) in its
`prompt_injected_benign` task notes) and before `run_tasks.py` was ever
executed against the live daemon. `out/results.json` and this file's
timestamps in `git log` on this branch are the provenance for "written
before the run."

## (a) Prompt-injected-benign block rate

**Prediction: the gate blocks AT MOST 5% of the `prompt_injected_benign`
category (55 tasks).**

Reasoning, from reading the gate's own source (`Gateway::decide` /
`parse_receipt` in `demo/agent-trace/gateway/src/main.rs`, read by the
"defender" role before the attacker subagent was given only the public
README): `parse_receipt` extracts exactly four things from a receipt —
`model`/`embed`/`vocab` digests, the last step's `in=` tool-call bytes,
and the `trace-chain` value. The `prompt-hex` field (which is where any
injected text would live) is read by NOTHING in the gateway's decision
path — not `cheap_precheck`, not `decide`, not the allowlist check, not
the freshness check, not the verify-execute binding check. The only
place prompt content could matter is inside the real `agent_trace
verify` subprocess's own grounding/WARNING check (strict mode only,
off by default here) — which inspects whether a tool-call ARGUMENT was
grounded in context, not whether the PROMPT contains an injection
attempt. A structurally valid, on-allowlist, fresh-nonce receipt whose
recorded tool call is benign should therefore ALLOW regardless of what
the prompt said, 100% of the time under lenient policy — i.e. we
predict a block rate at or near 0%, comfortably under the 5% stated
limit, not because the gate detects and rejects the injections but
because it has no mechanism that could.

## (b) Latency

No "Lindner's availability model" artifact is available on this box
(not found in state/reports/ or program/ under that name as of
2026-10-01) — **substituting a reasoned engineering bar, documented
here as a substitution, not a citation.**

Given this experiment's architecture (SAFE-7 async gateway: cheap
checks inline+synchronous, slow verify off-loaded to a worker pool,
client polls), two different latencies are the right things to bound,
not one:

- **Cheap-path decisions (DENY on allowlist/binding/freshness)** never
  touch a verify subprocess at all. Bar: **p99 < 5 ms** on this
  hardware (i5-5200U) for a single Unix-domain-socket round trip plus
  in-memory hash-set lookups — generous relative to what a bare UDS
  round trip costs (sub-millisecond is typical), left loose because
  this run shares the one CPU core with the Python client itself.
- **Verify-bound decisions (ALLOW, or a DENY that requires the slow
  check)**: latency here is dominated by whatever the pluggable
  verifier takes, not the gateway's own logic — the gateway adds
  polling overhead on top. Bar: **gateway-added overhead (client
  wall-clock time minus the verifier's own run time) p99 < 50 ms**,
  i.e. the daemon itself should not add more than tens of ms of
  dispatch/poll/IPC cost around whatever the real verify subprocess
  (seconds-to-minutes for a real small/2B model) actually costs.

These are engineering bars chosen for this box and this prototype, not
a formal availability-theoretic bound; see the report's Limits section.
