# SAFE-1b gateway prototype — status (2026-09-13, cm-builder subagent)

## What was found
`demo/agent-trace/gateway/src/main.rs` (commit 6597de1, "WIP, unverified
integration test") already implements all three critic-mandated properties
as hard, non-optional checks in `Gateway::decide`:

- **(a) verify->execute binding**: `sha256(action_bytes)` must equal
  `sha256(receipt's last-step `in=` bytes)` or DENY, checked before
  `agent_trace verify` is even invoked.
- **(b) freshness**: `(session, counter, action_hash)` triples are
  remembered in-memory (`Gateway.seen`); any repeat is DENY. Covers the
  original design's "replayed receipt" threat.
- **(c) log anchor + allowlist integrity**: decisions are hash-chained
  (`append_log`, each entry depends on `prev_digest`), with a periodic
  anchor line appended to a separate file every `anchor_every` decisions
  (stubbed as local file — see caveats). The allowlist is a flat file of
  `model embed vocab` triples with a trailing `sig <hex>` line
  (HMAC-SHA256, keyed, built on aegis-core's own SHA-256 so the crate
  stays dependency-free); a bad/missing/tampered signature is a hard
  `GatewayInitError` at construction time, not a silent pass-through.

Unit tests already present and all cover the required cases:

1. `allow_valid_receipt_matching_action_fresh_nonce` — baseline ALLOW.
2. `deny_chain_tampered_receipt` — threat 1 (tampered token id -> chain
   replay fails -> `agent_trace verify` FAIL).
3. `deny_artifact_triple_not_on_allowlist` — threat 2.
4. `deny_dropped_step_receipt` — threat 4 (dropped step -> parse-error DENY).
5. `deny_action_mismatch_confused_deputy` — new rule (a), the critic's
   confused-deputy case: valid+allowlisted receipt but proposed action
   differs from the receipt's own last-step bytes -> DENY.
6. `deny_replayed_receipt_same_session_counter` — new rule (b), replay
   DENY (also covers original threat-3 "replayed receipt").
7. `allow_same_session_new_counter_new_action` — rule (b) does not
   over-lock (new counter + new action in the same session still ALLOWs).
8. `deny_tampered_unsigned_allowlist_fails_to_load` — new rule (c),
   hand-edited allowlist without re-signing fails at load, not at decide.
9. `allow_correctly_signed_allowlist_permits_matching_triple` — rule (c)
   positive case.
10. `log_head_changes_with_each_decision_and_is_deterministic_given_history`
    — sanity that the log chains DENYs too, not just ALLOWs.

All 10 of the above pass (`cargo test --release`, run on this box against
the tinybit M7 fixture model — fast, seconds each).

## live20 integration test
`tests::live20_lenient_allow_count` / `tests::live20_strict_allow_count`
replay the real 20-receipt `tools-1` live corpus
(`tests/fixtures/live20/*.receipt`, verbatim copy, not fabricated here)
against the real 2B artifacts at
`/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts/`, invoking the real
`agent_trace verify` release binary once per receipt, once in lenient mode
(PASS-only) and once in strict mode (PASS AND zero `WARNING step` lines).
This is a genuine subprocess call against the full 2B model per receipt —
**not a stub or simulation** — but each `agent_trace verify` call on the
2B model takes on the order of 3-10 minutes (chain_*/mixed_* receipts are
markedly slower than calc_*), so the full 20-receipt x 2-mode run is
roughly an hour of wall-clock time.

**Caveat / what is NOT yet confirmed**: I started this run
(`cargo test --release`, detached via `setsid`, log at
`/tmp/gw_test.log` on this box, PID group leader 1097793) and it was
still in progress — at receipt 6/20 (`chain_fileread_01`) after ~35
minutes — when this subagent's budget ran out. The 10 non-live20 unit
tests above all passed. **I have not personally observed a final
ALLOW/DENY count for live20 lenient or strict mode; I am not reporting
one.** The test code itself asserts `allow == 20` (lenient) and
`allow == 13` (strict, cross-checked against the independent Python
grounding check in `state/reports/2026-09-13-tools1c-grounding-box2.md`
per the code's own comment) and will fail loudly with per-id eprintln
output if the real run doesn't match — but whether it actually does
match has not been verified in this session.

**No hangs observed**: the test makes real, bounded progress (one receipt
finishes, the next starts); it is slow, not stuck.

## What to do next (for whoever resumes this)
- Check `/tmp/gw_test.log` on this box (aefinity-box2) for whether the
  detached run finished and what it printed for
  `live20 strict allow count: N/20` (eprintln in the test) and whether
  both `live20_*_allow_count` tests report `ok`.
- If it's gone (box restarted, log rotated), rerun:
  `cd demo/agent-trace/gateway && cargo test --release -- live20 --nocapture`
  and budget ~60-90 minutes of wall-clock, ideally via a proper background
  job (not tied to a single subagent's turn budget) with a check-in step
  afterward.

## Ready to check off the QUEUE line?
**Not yet.** Code-wise, (a)/(b)/(c) and the required unit tests are done
and passing. The live20 integration test is written and running for-real
(no placeholders), but its actual ALLOW/DENY counts are unconfirmed as of
this report — that's the one open item before this line can be marked
done. Recommend a QUEUE item: "confirm live20 gateway integration test
result (lenient 20/20, strict N/20) — long-running, needs a full-length
tick or manual check of /tmp/gw_test.log on aefinity-box2."

No code changes were made in this pass — `demo/agent-trace/gateway/`
was already complete when inspected; this report documents that finding
plus the unresolved live20-completion gap.
