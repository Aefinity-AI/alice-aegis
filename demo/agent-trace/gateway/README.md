# agent-trace gateway

An async receipt-verification gateway for agent tool calls (see the module
doc comment in `src/main.rs` for the full protocol/design). A client
submits a receipt + the action it is about to dispatch over a Unix socket;
the gateway does cheap inline checks (parse, allowlist, verify-execute
binding, freshness) synchronously and replies `PENDING ticket=T...`
immediately, then a background worker pool runs the real
`agent_trace verify` against the real model and the client polls
(`POLL ticket=T...`) until it resolves to `ALLOW idx=... cap=... exp=...`
(with a capability token tool shims can independently check) or
`DENY <reason>`.

## One-command demo

`demo.sh` is a single, self-contained script that demonstrates the whole
story end to end against the REAL BitNet-2B artifacts. Set
`CM_2B_ARTIFACTS` to the directory holding
`aegis_pruned_model.cis.safetensors`, `embed.bin`, and `vocab.bin` before
running it (the artifacts are not shipped in this repo — obtain them from
wherever your BitNet-2B pruned-model export lives). It:

1. Uses a real receipt already generated against the real 2B model
   (`tests/fixtures/live20/calc_01.receipt`).
2. Starts a gateway daemon in a private scratch dir (its own socket,
   signed allowlist, and capability key — does not touch anything else
   running on this box), submits the receipt, polls until it resolves,
   and prints the real `ALLOW idx=... cap=... exp=...` line. This
   genuinely takes ~1-2 minutes wall clock because a real
   `agent_trace verify` subprocess is re-running the model in the
   background — nothing is faked or stubbed.
3. Replays the identical (session, counter, action) and shows the
   gateway DENY it immediately on freshness grounds (a tool call cannot
   be authorized twice).
4. Flips one byte inside the receipt's `decode-chain` hash (a field the
   cheap inline precheck does not look at — only the real model-based
   verify can catch it) and submits that under a fresh (session,
   counter). Shows it go `PENDING` (passes the cheap checks) and then,
   once the background worker actually re-runs the real model, resolve
   to `DENY agent_trace verify: VERIFY FAIL`.
5. Prints the exact standalone `agent_trace verify <model> <embed>
   <vocab> <receipt>` command a third party could run on ANY machine
   (a laptop, a phone — anything with no network path to this gateway at
   all) using nothing but the receipt file and the three public artifact
   files, to independently confirm the ALLOW'd receipt is genuine — and
   then actually runs it, on this box, against the exact receipt used
   above, for illustration.
6. Stops the daemon and cleans up its scratch dir on exit (including on
   failure, via a trap), regardless of success or failure of any step.

Run it with:

```
CM_2B_ARTIFACTS=/path/to/bitnet2b-2b-artifacts bash demo.sh
```

It builds the two release binaries it needs (`agent_trace` example,
`gateway`) if they are not already current — this is a no-op if you have
already built this repo. Expect the whole run to take roughly 3-4 minutes
wall clock (two real ~1-2 minute model verifications; on a slow single-core
host a 2B verify can take 6-7+ minutes — the poll deadline defaults to 900 s,
override with `DEMO_VERIFY_TIMEOUT=<seconds>`).

`demo-expected-output.log` is a captured real run to diff your own run
against. The exact capability token, `exp=` timestamp, scratch-dir PID
(`/tmp/safe8-demo-<pid>`), `CM_2B_ARTIFACTS` path (the captured log's
`agent_trace verify ...` command line shows the literal, unexpanded
`$CM_2B_ARTIFACTS` — that is how the gateway echoes it, not a bug), and
per-run wall-clock poll-attempt counts will differ between runs — everything else (the
`AEGIS-TRACE` receipt bodies, the ALLOW/DENY/DENY sequence, the
`VERIFY PASS` line) should match byte-for-byte.

## Verify on a phone

The same standalone `agent_trace verify` step 8 prints can be run on a
phone with no network path to this gateway at all:

1. Cross-compile a static verifier for aarch64 on any Linux host with the
   `aarch64-unknown-linux-gnu` target installed:
   ```
   RUSTFLAGS="-C target-feature=+crt-static" \
     cargo build --release --offline --example agent_trace \
     --target aarch64-unknown-linux-gnu
   ```
2. Push the resulting static binary, the receipt file, and the three
   artifact files to the phone (e.g. via `adb push` into
   `/data/local/tmp/`).
3. Run it on-device with absolute paths:
   ```
   adb shell /data/local/tmp/agent_trace verify \
     /data/local/tmp/aegis_pruned_model.cis.safetensors \
     /data/local/tmp/embed.bin /data/local/tmp/vocab.bin \
     /data/local/tmp/receipt.orig
   ```
   A `VERIFY PASS` line confirms the receipt bit-for-bit, entirely
   offline.

If you don't want to cross-compile yourself, the public
[`cis2-spec`](https://github.com/Aefinity-AI/cis2-spec) repo's
`kits/verification-kit-v1/` already ships prebuilt aarch64 verifier
binaries for exactly this phone-verification lane.
