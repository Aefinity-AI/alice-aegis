# CLAUDE.md — A.L.I.C.E. / Aegis

## What this is

A ternary-quantized LLM inference engine implemented as a **`no_std` Rust UEFI
unikernel**. It boots from firmware off a FAT32 USB stick with **no operating
system present** — no kernel, no scheduler, no interrupts, no page cache — and
runs transformer inference in ring 0. The same engine also builds as a Linux
userspace binary so every result can be A/B'd against a normal OS.

- **Primary target:** `x86_64-unknown-uefi`
- **Dev host:** mobile Intel i5 Chromebook (i5-10210U, Comet Lake, under crosvm).
  No GPU. Treat it as the weakest machine that must still work.
- **Bare-metal test targets:** Dell Inspiron 15 (i5-5200U, Broadwell-U, AVX2),
  HP Stream (Celeron N4020, Gemini Lake, SSE2 — scalar path only).

### Crate map

| Crate | Role |
|---|---|
| `aegis-core` | Engine: ternary kernels, attention, tokenizer, sampler. `no_std`. |
| `aegis-uefi` | The unikernel. UEFI entry, firmware bring-up, DMA, FAT32 loading. |
| `aegis-linux` | Userspace host binary wrapping the same `aegis-core`. |
| `aegis-eval` | Perplexity / eval harness. |
| `aegis-forge` | Model preparation, vocabulary pruning, quantization tooling. |
| `xtask` | Dev automation. `cargo xtask boot-test` boots the unikernel in QEMU. |

## Build and test

```bash
# host-side checks — run this before claiming anything is done
# (NO root Cargo.toml exists by design; devloop iterates the crates.
#  clippy is a RATCHET vs scripts/devloop_clippy_baseline.tsv: new debt fails,
#  documented old debt is reported until paid down.)
scripts/devloop.sh gate     # = fmt + clippy ratchet + tests + boot + figures
scripts/devloop.sh fmt      # or run any single verb: fmt|clippy|test|boot|figures

# the unikernel — PRODUCTION build. This is the ONLY command whose output may
# be staged to a stick. The stock x86_64-unknown-uefi target is SOFT-FLOAT:
# it scalarizes every AVX2 intrinsic to library calls (zero vector instructions,
# ~120-250x slower on the Dell — ledger A14, the MECH-A1 regression). The
# script builds against x86_64-uefi-hardfloat.json and hard-fails unless the
# binary contains vector instructions (objdump census).
./aegis-uefi/build_hardfloat.sh

# stock-target build (CORRECTNESS ONLY — never stage this artifact):
# cargo build --release --target x86_64-unknown-uefi -p aegis-uefi

# boot the unikernel under OVMF (correctness only — see Rule A)
cargo xtask boot-test

# staging gate: refuses any .efi with zero ymm/vfmadd instructions
scripts/check-efi-simd.sh <path-to.efi>

# claim verification
scripts/verify-figures.sh
```

`xtask boot-test` maps QEMU's `isa-debug-exit` code **33** to success. Any other
exit is a failure.

---

## Hard rules

These are not style preferences. Each one exists because violating it already
cost this program a public retraction.

### Rule A — no performance number may come from QEMU/TCG

Emulation is for **correctness only**. `-accel tcg -cpu max` does not model
cache hierarchy, memory latency, DVFS, turbo residency, or instruction
throughput. A tok/s, cycles/token, GB/s, J/token, or speedup figure taken under
TCG is meaningless and must never be recorded, quoted, or compared.

Performance numbers come from physical hardware, with the machine named.

Corollary: **RDTSC is not a cycle counter.** It advances at the invariant
nominal rate regardless of core clock, so `ticks/token` is not `cycles/token`.
Any figure derived from it (MAC/cycle, %-of-peak) must carry the measured
effective/nominal clock ratio alongside, or it is wrong by that factor.

### Rule B — no number enters `program/RESEARCH_LEDGER.md` without a matching raw log

Every figure in the ledger must be traceable to a raw log file under
`docs/hardware_logs/`. Not a commit message. Not a markdown summary. Not a
source-code comment. **Prose can repeat a number; it cannot measure one.**

A number whose only provenance is prose gets quoted by other prose and becomes
"true" by circulation. That is precisely how this program shipped an unlogged
throughput figure into its README.

Three legal parents for a number, and nothing else:
1. **Instrument** — raw output of a run, in `docs/hardware_logs/`.
2. **Derivation** — arithmetic from instrument-backed numbers, formula written down.
3. **Citation** — an external published source, cited.

Run `scripts/verify-figures.sh` before touching the ledger.

### Rule C — never edit files under `tests/golden/` or `docs/hardware_logs/`

They are **append-only**. Golden files are the definition of correct output;
hardware logs are primary evidence. Editing either destroys the only record
that can contradict us, which is the entire point of keeping them.

Add new files. Never rewrite existing ones. A `PreToolUse` guard enforces this
(`.claude/guard-ledger.sh`) — if it blocks you, the answer is a new file, not a
workaround.

### Rule D — prefer bit-exactness tests over benchmarks

A bit-exactness test is deterministic, cheap, and fails loudly. A benchmark is
noisy, environment-dependent, and fails silently by drifting.

When choosing what to write: assert that output is **byte-identical** across
the change. Bit-exactness caught the soft-float codegen bug, the LM-head size
guard, and the tokenizer byte-mapping bug. Benchmarks caught none of them.

Reserve benchmarks for questions only a benchmark can answer, and then treat
Rules A and B as binding on the result.

---

## Working conventions

- **Negative findings are deliverables.** "This approach is 6.6× slower" is a
  result. Record it with the log, do not bury it.
- **Settled negatives — do not re-propose:** the CTZ zero-multiplication kernel
  (measured slower at real 42.21% sparsity) and T-MAC-style LUT-mpGEMM `pshufb`
  (settled 2026-07-30 on memory-traffic grounds: 4 bits/weight = 1.67× decode
  traffic on a bandwidth-bound path outweighs its kernel-side throughput, which
  same-binary A/B showed is bimodal and not reliably slower — see
  `docs/hardware_logs/lut_mpgemm_ab_findings_2026-07-30.md`).
- **Name the machine.** Every measurement states which physical box produced it.
  "On the dev box" and "on bare metal" are different claims.
- **`unsafe` needs a `// SAFETY:` comment** explaining the invariant upheld.
  Firmware bring-up, MSR access, and DMA are inherently `unsafe`; justify each.
- **Smallest diff that satisfies the task.** No drive-by refactors.
- Do not claim a task complete without running the verification command.

## Repo layout notes

- `docs/hardware_logs/` — primary evidence, append-only (Rule C).
- `program/` — `ROADMAP.md`, `RESEARCH_LEDGER.md`, `SESSION_PROTOCOL.md`.
- `tests/golden/` — byte-exact reference outputs, append-only (Rule C).
- `artifacts/` — build and measurement artifacts referenced by claims.
- `scripts/` — `verify-figures.sh` and measurement helpers.
- `.claude/rules/invariants.md` — Rules A–D, terse, re-injected each session.

## Guardrails in force

| Hook | Trigger | Effect |
|---|---|---|
| `PreToolUse` | `Write`/`Edit`/`MultiEdit` | `guard-ledger.sh` blocks append-only paths |
| `PreToolUse` | `Bash` | `hooks/guard.py` blocks destructive commands (fixed-disk writes, rm -rf /, mkfs) |
| `PostToolUse` | `Write`/`Edit`/`MultiEdit` | `hooks/post_edit_rust.sh`: crate-scoped `cargo fmt` on `.rs` edits (clippy is a ratchet in `scripts/devloop.sh`, not per-edit) |
| `PostToolUse` | `Write`/`Edit`/`MultiEdit` | `hooks/integrity_gate.sh`: edited `.rs` must not print numbers it did not compute |
| `SessionStart` | new / compacted session | re-injects `.claude/rules/invariants.md` |

`git push` and `gh pr create` require explicit approval.

> Note: this file replaced an earlier generic dev-loop protocol. That version is
> recoverable at commit `5b69327`.
