# First 90 minutes

Do them in order. Nothing here needs the network, a GPU, or a frontier model.

- [ ] **0–5  Put it on PATH and look at the box.**
      `ln -s ~/program/loop/ev ~/.local/bin/ev && ev env --nick chromebook-crosvm`
      Read `quiet_check`. If a Claude session is running it will say `quiet: false`
      — that is correct, and it means step 4 must be run from a plain terminal.

- [ ] **5–15  Enter the already-dead numbers, once.**
      `cd ~/program/loop && ./seed_retractions.sh`
      Nothing here is a new judgment; every entry is a number already retracted or
      found unlogged. From this moment they cannot re-enter a file silently.

- [ ] **15–20  See what it finds. This is the payoff moment.**
      `ev lint docs/TECHNICAL_REPORT.md`          → 12 dead-number uses
      `ev lint --no-exempt docs/TECHNICAL_REPORT.md` → 24
      `ev lint aegis-core/src/ops.rs`             → ops.rs:1240, the 17.3 GB/s
                                                     nobody has ever measured,
                                                     living in a doc comment
      `ev lint program/RESEARCH_LEDGER.md`        → 0. The honest ledger is not
                                                     punished. That is the design.

- [ ] **20–25  Wire the hook.**
      Merge `hooks/settings.snippet.json` into `~/.claude/settings.json` — keep the
      existing `integrity_gate.sh` entry, add `claim_gate.sh` beside it. Self-test:
      `echo '{"tool_input":{"file_path":"'$HOME'/aegis-core/src/ops.rs"}}' \
        | ~/program/loop/hooks/claim_gate.sh; echo $?`   → 2

- [ ] **25–40  Build the two missing benchmarks.**
      `cd ~/aegis-linux && CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release \
         --example inproc_variance --example membw`
      (`membw` is new and already builds; `inproc_variance` exists since 07-29.)

- [ ] **40–70  Close the multicore hole. FROM A PLAIN TERMINAL, no Claude session.**
      `ROUNDS=5 ITERS=3 ev run thread_sweep`
      ~30 min at 64 tokens x 60 samples. It writes a log, a runcard, and a
      SUMMARY that computes the ratios from cycles/token and states plainly that
      **no SMT claim is identifiable on this host** (crosvm reports 8 sockets x
      1 core x 1 thread — siblings are invisible, so a 4t-vs-8t delta is host
      placement, not SMT). That single line resolves ledger row A4's three-way
      contradiction better than another re-measurement ever could.

- [ ] **70–80  Bank it.**
      `ev claim add --id A4.4t --value <ratio> --unit x --kind measured \
         --statement "decode speedup, 4 workers vs 1" --source <log> --runid <runid> \
         --scope "i5-10210U crosvm guest, topology flattened, BitNet-2B, int8_act" \
         --ceiling "oversubscription only; no SMT claim; median of 15, IQR <x>%"`
      Then `ev claim retract --id A4.sweep2026 --reason "superseded by a logged \
      sweep" --superseded-by A4.4t`.

- [ ] **80–90  Fix the four report lines the gate is now blocking, and commit.**
      Lines 32, 66, 207–208, 249, 275–276. Replace the dead numbers with the live
      claim ids or delete them. Re-run `ev lint docs/TECHNICAL_REPORT.md` until it
      is 0. Commit `program/loop/` and the log and the runcard **in one commit**,
      with the runid in the message.

## Tomorrow, not today
- `ev run membw` from a quiet terminal, then delete `17.3 GB/s` everywhere.
- Apply `patches/01_gauntlet_build_identity.md` before the next USB boot.
- Add the ```ev-prereg``` block to the live m7lr pre-registration.
- Populate ~40 live claims, then turn on `ev lint --strict` for the report.
