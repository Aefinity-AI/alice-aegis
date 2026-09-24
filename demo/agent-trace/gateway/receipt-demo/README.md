# tool-2: capability/receipt gateway end-to-end demo

One-command demo of the full receipt-gateway loop, built entirely on the
existing gateway crate's real code (`gateway::capability::issue`/`verify`,
`cap_issue_for_test`, `tool_shim_file`) plus one small new binary,
`offline_verify_receipts`, that reuses `capability::verify` to
independently audit a receipt log after the fact. No new crypto or
capability-verification logic is written anywhere in this demo — every
ALLOW/DENY decision is made by the same code the real gateway/shims use.

## Run it

```sh
demo/agent-trace/gateway/receipt-demo/demo.sh
```

Needs only `cargo`/`rustc` (offline, no network, no external crates —
same constraint as the rest of this crate). Runtime: builds three debug
binaries then runs them a handful of times; ~30s on a Celeron N4020C,
well under the 5-minute budget.

## What it shows (xb-3 auditor walkthrough)

1. **Case A — PASS.** A toy "agent" proposes a file-write. The gateway
   (`cap_issue_for_test`, standing in for the real gateway daemon's
   ALLOW path) issues a capability token bound to that exact action's
   hash. `tool_shim_file` — the real enforcement point — checks the
   token and only then performs the write. The attempt is appended to
   `receipts.log` as one JSON line (tool, args, capability, decision,
   timestamp).

2. **Case B — FAIL, fail-closed at the gateway.** A second proposed
   write uses a capability token with one hex character flipped
   (simulated forgery/bit-flip). `tool_shim_file` refuses
   (`SHIM REFUSE: bad mac`) *before* touching the filesystem — nothing
   is written. The refusal is logged too.

3. **Offline re-verification ("another box").** `offline_verify_receipts`
   is a separate process with no shared in-memory state with steps 1-2:
   it only reads the key file and `receipts.log` off disk, and
   independently re-derives each action hash and re-runs
   `capability::verify` itself. Run against the real log, it reproduces
   both the Case A ALLOW and the Case B DENY.

4. **Case C — FAIL, caught only by the offline step.** The demo then
   edits a *copy* of `receipts.log`, changing Case A's logged `data`
   field (simulating an attacker/bug editing the audit trail after the
   fact) while leaving the capability token, index, and expiry alone.
   `offline_verify_receipts` catches this: the logged `action_hash` no
   longer matches `sha256(logged data)`, so it reports FAIL even though
   the token string itself is untouched. This is the property that
   matters for an auditor: the log's own claims are checked against the
   receipt's cryptographic binding, not taken on faith.

The script exits 0 and prints `PASS: ...` only if all four expected
outcomes hold (A allows, B refuses, offline verify of the real log is
clean, offline verify of the tampered log fails). Any deviation prints
`FAIL: ...` lines and a non-zero exit — this is what "fail-closed" means
in practice for this demo, and is the thing to look for as an auditor
running this script.

## Files

- `demo.sh` — the one command; orchestrates the four steps above and
  prints `==>` section headers plus a final `==> RESULT` line.
- `../src/bin/offline_verify_receipts.rs` — the only new code: reads a
  key file + JSON-lines receipt log and calls the real
  `gateway::capability::verify` per entry, with its own fresh
  consumed-index set (genuinely independent of the shim's on-disk
  consumed-set files).
- Everything else it drives (`cap_issue_for_test`, `tool_shim_file`,
  `gateway::capability`) is unmodified, pre-existing gateway code.
