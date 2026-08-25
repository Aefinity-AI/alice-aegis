# HANDOFF 2026-08-25 — machine rebuild + membw closed

The dev box was **fully reset on 2026-08-24**. Everything local was lost: configs,
SSH keys, all repos, all Claude memory, and — importantly for this program — every
gitignored artifact, including the model weights.

## Done

- Machine rebuilt (Debian 13 Crostini). Rust 1.98 + nightly 1.100 w/ `rust-src`,
  `x86_64-unknown-uefi` on both, Go 1.27, Node 20, Python 3.13 + uv, Docker,
  VS Code, QEMU 10.0.11 + OVMF + mtools. `~/dev-setup/bootstrap.sh` (private repo
  `Ranger3143/dev-setup`) rebuilds all of it in one command.
- Skill, hooks and per-clone `pre-commit` gates reinstalled with paths corrected.
- `ev` on PATH; **`ALICE_REPO` exported** (see Traps).
- **A13.bw closed.** `ev run membw` → runid `2026-08-25T1628Z-membw-05d1e672cbca`,
  log `docs/hardware_logs/membw_2026-08-25_162849Z.log`. Banked, all
  `evidence_check: verified`:
  `A13.bw.seq1t 10.57 GB/s` · `A13.bw.seq8t 25.28 GB/s` · `A13.bw.tern1t 0.80 GB/s`.
  The retracted 17.3 GB/s matched **neither** figure it could have meant.
- `ev lint`: `ops.rs` 1 → **0**, `RESEARCH_LEDGER.md` **0**.
- Unikernel boot test: **functionally passed** — bootlog shows coherent generation
  and `MECHV2 EXACT … true` bit-exact repeats. The harness reported exit 124 only
  because of a 30-min timeout set by the operator, which cut it ~30 s short of
  writing `isa-debug-exit`. Re-run with a longer timeout for a clean PASS.

## Traps found (all were failing SILENTLY)

The repo used to live **at `$HOME`**; **109 files** still hardcode
`/home/killboxincorporated`. Two mattered:

1. `program/loop/hooks/claim_gate.sh` resolved claimlint via `$HOME`, so `[ -x ]`
   failed and the hook **exited 0** — wired and never firing. Fixed to resolve from
   `BASH_SOURCE`; a missing claimlint is now a hard exit 2 (commit `05d1e67`).
2. `runcard.py` defaults `REPO` to `$HOME`. Without `ALICE_REPO` set, runcards are
   written **outside the repo** and every runid stamps `nogit`. Now in `.bashrc`.

## Open, in priority order

1. **`aegis-uefi/build_hardfloat.sh` does not link.** `undefined symbol: wcslen`
   (15 refs). The hard-float target enables SSE, so LLVM's loop-idiom pass rewrites
   the `uefi` crate's UTF-16 scans into `wcslen` calls; the stock target is
   `+soft-float` and does not. No UEFI sysroot provides `wcslen`. **This is the only
   command whose output may be staged to a stick, and CI exercises only the
   stock+stable path — CI is green while the shippable artifact cannot be built.**
   Fix: a `wcslen` shim in `aegis-uefi` using `read_volatile`/`#[no_builtins]` (a
   naive scan loop gets idiom-recognized into a call to itself). Then add the
   hard-float build to CI.
2. **`ev run thread_sweep` cannot run.** Needs `aegis_pruned_model.safetensors` and
   `aegis-forge/{embed,vocab}.bin` — gitignored, so lost in the reset. Rebuilding
   needs `microsoft/bitnet-b1.58-2B-4T-bf16` re-downloaded (**hours** at this box's
   ~141 KB/s) and `aegis-forge/regen_vocab_embed.py` de-hardcoded.
   `docs/TECHNICAL_REPORT.md`'s remaining **2** dead-number uses are A4.sweep2026
   and stay blocked until this runs.
3. **The compute-bound argument in `ternary_matmul` needs re-deriving.** The old
   doc comment used the RETRACTED 17.3 GB/s (A13.bw, superseded) to argue the engine
   is compute-bound. The
   number is now replaced, but the *conclusion was deliberately not restated*: the
   weight-streaming roofline divides by `A13.bw.tern1t`, not a sequential peak, and
   that figure is itself a scalar LOWER BOUND, not the AVX2 kernel's rate.
4. `program/RESEARCH_LEDGER.md` and `program/ROADMAP.md` not yet updated with the
   A13.bw.* figures — left for review since that file's wording is load-bearing.
5. Nothing is pushed. Commits `05d1e67`, `0e95a92` are **local only**.

## Notes

- `ev run` refused to measure while QEMU held the box, correctly. The membw run was
  taken with the quiet-gate passed but **a Claude Code session resident** (busiest
  unrelated process 20.9% of one core); that is recorded in the runcard. If a
  reviewer objects, re-run it from a plain terminal — it takes under a minute.
- These are crosvm-guest numbers on an i5-10210U, **not baremetal** and **not a
  ceiling**. Rule A still applies: nothing from the QEMU boot test is a
  performance figure.
