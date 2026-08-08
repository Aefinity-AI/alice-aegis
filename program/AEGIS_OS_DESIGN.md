# AEGIS OS — DESIGN v1 (DRAFT)

*2026-08-08. Internal design draft for maintainer review (Justin) and the
orchestrating session. NOT a public document. Builds on
`program/SELF_SPEC_FABLE0_2026-08-01.md` (the adopted FABLE-0 architecture)
— this doc extends it toward a buildable plan; it does not reinvent it.
Rules A–D (`.claude/rules/invariants.md`) are binding on every line below.*

---

## 1. Vision

Aegis OS is a bootable personal edge AI: a USB stick (or internal drive)
that turns a commodity laptop into an offline agentic coding assistant —
plan, read files, edit, run the compiler, verify, leave a cryptographic
transcript — with no cloud dependency. The forcing reality is lived, twice
over: the 2026-07-29 Anthropic outage stalled the bare-metal program for
hours ("kills my productivity, and ruins alignment in the pipeline" —
`memory/offline-research-assistant-goal.md`), and when money flow gets
slow, API spend is the first thing that dies. Continuity of the research
program cannot depend on a subscription. The capability gap is the
*harness*, not the weights — this program already runs its own ternary
models on its own engine on its own iron.

What exists today is more than a napkin. The A.L.I.C.E. unikernel boots
from firmware off FAT32 with no OS, loads MODEL.SAF/EMBED.BIN/VOCAB.BIN,
decodes coherently, and — as of this week — *verifies*: given RECEIPT.TXT,
`aegis-uefi/src/verifier.rs` hashes the artifacts, replays the decode under
CIS-1 full-integer semantics, and recomputes the SHA-256 witness chain over
every step's full logit vector, ring 0, no OS underneath.
`scripts/make-kit-image.sh` packages that as the Provable AI Kit image.
CIS-1 itself is frozen enough to bet on: one digest (`76985613c965f643`)
identical across four x86 implementations including HP bare iron
(ledger A25, `docs/hardware_logs/hp_L_BOOTLOG_2026-08-02.txt`) and across
the ISA boundary to aarch64 (A28/A29,
`docs/hardware_logs/cis_decode_token_crossisa_ci_2026-08-07.log`), with a
full-pipeline token-level digest (`CIS_DECODE 67e8c0a96abc04e1`) pinned in
CI. Aegis OS = that engine + a body + a scaffold that does the Claude-ness.

## 2. Two-track architecture

### Track U — pure unikernel ("the courtroom body")

Extend the existing ring-0 REPL (`process_intent` loop,
`aegis-uefi/src/main.rs`) toward a minimal tool loop: model proposes a tool
call in a grammar-locked block; the unikernel executes it against FAT32
(read file, write file, append transcript); result re-enters context.
Smallest possible step — no scheduler, no interrupts, no filesystem beyond
the FAT32 support already shipped.

Honest tradeoffs:
- **No toolchain.** No processes means no rustc, no git, no grep binaries —
  every tool is engine code. A coding assistant that cannot run the
  compiler is a notebook, not an assistant.
- **Every capability is bring-up cost.** Persistent writes to FAT32,
  editor-grade file mutation, and any networking would each be a firmware
  engineering project (and Rule-C-grade risk to the evidence story if done
  sloppily).
- **What it uniquely offers:** minimal TCB, one PE image measurable into
  TPM PCR4, and the strongest demo in the program — an AI that boots,
  answers, and *proves its own replay* with nothing underneath it.

### Track L — FABLE-0 proper ("the daily body")

Per the adopted spec: minimal Linux (initramfs, tmpfs console, performance
governor — the exact environment the L sticks already boot) + the
`aegis-linux` engine + an agent scaffold in Rust userspace. Linux is not
there for comfort; it is there because it provides, for free, the four
things an agentic coding assistant is made of: processes (run rustc/cargo,
run tests), a real filesystem (the repo being edited), exec of arbitrary
tools, and optional local networking. The engine core, CIS-1 semantics,
witness chain, and model artifacts are byte-identical to Track U's.

### Recommendation — and a correction to the spec's rationale

**Track L is primary; Track U is the showpiece and reference verifier.**
This matches the FABLE-0 memory's split — but the *reason* has changed and
must be stated honestly: the spec justified Linux-as-daily-body partly with
"minimal Linux is the fast venue (ledger A13)". That sign is now
**overturned** by the preregistered MECH v2 (A22, Dell i5-5200U, two
hands-off boots, n=10x3): the ring-0 unikernel beat minimal Linux on 3/3
per-prompt decode medians, +3.7% / +10.4% / +5.4%
(`docs/hardware_logs/mech2_U_BOOTLOG_2026-08-01.txt` +
`mech2colskip_L_dell_BOOTLOG_2026-08-01.txt`, prereg
`mech2_PREREGISTRATION_2026-08-01.md`). Track L wins on **capability**
(toolchain, processes, storage), not on speed. Single-digit-percent decode
cost is a price worth paying for a body that can run `cargo test`; a
scheduler-free REPL that cannot is not an assistant. The unikernel remains
the trust story — now *also* a fast one, which strengthens the demo.

Every transcript the L body produces is verifiable by the U body: the
witness receipt (`RECEIPT.TXT`) is the interchange format, already
implemented end-to-end (aegis-linux `cis_witness gen` -> verifier.rs
replay). That loop — work on Linux, prove on bare metal — is the product.

## 3. Target iron

| Machine | Role | Facts on file |
|---|---|---|
| **Dell Inspiron 15, i5-5200U (Broadwell-U, AVX2, DDR3L)** | Primary body. Most-measured box in the program. | Peak seq read 11.19/10.95 GB/s (1T, two boots); ternary scalar stream lower bound 0.62 GB/s (A24, `docs/hardware_logs/mech2colskip_L_dell_BOOTLOG_2026-08-01.txt` + `_dell2_`). CIS AVX2 kernel 2.94x FASTER than float AVX2, bit-exact (A27, `docs/hardware_logs/cis_avx2_armD_ordercontrol_L_dell_BOOTLOG_2026-08-06.txt`). BitNet-2B decode 0.62 -> 3.03 tok/s after bd-PROCHOT fix (A12, `docs/hardware_logs/m7_BAREMETAL_bdprochot_FIXED_2026-07-29.log`; pair build-confounded per the row's own caveat). |
| **HP Stream, Celeron N4020 (Gemini Lake, SSE2)** | Scalar floor. Proves "runs on the weakest iron". | CIS digest identical to Dell (A25, `docs/hardware_logs/hp_L_BOOTLOG_2026-08-02.txt`). Integer semantics 4–14% *faster* than scalar float here (C/B 0.961x, A26, `docs/hardware_logs/cis_vs_float_L_hp_BOOTLOG_2026-08-05.txt`) — the CIS path is the right default on this class. |
| **Chromebook i5-10210U (crosvm)** | Mission control and forge. **NEVER converted to a target.** | **HARD CONSTRAINT:** converting it (dev-mode / powerwash) destroys the 24/7 ops cascade (aefinity-ops timers, collectors, session automation) and the only always-on dev seat. It is also crosvm — Rule A already bars perf numbers from it. Quality/correctness runs only. |

## 4. The six FABLE-0 gates — verbatim, with proof mapping

Gate text verbatim from `SELF_SPEC_FABLE0_2026-08-01.md` §6. Spec rule
stands: **no build hours before gates** (G2/G5/G6 still gate the build).

| # | Gate (verbatim) | Status | Proving artifact / machine / test pattern |
|---|---|---|---|
| G1 | **Bandwidth-vs-compute probe on the Dell** (1T vs 2T scaling + bitnet.cpp/ik_llama.cpp A/B on identical iron, logged) | **MEASURED** (probe leg, A24); external-engine A/B still OPEN (also a ROADMAP ⬜) | Dell i5-5200U, two cold boots: `docs/hardware_logs/mech2colskip_L_dell_BOOTLOG_2026-08-01.txt` + `_dell2_`. Pattern: interleaved captures with in-log clock-state blocks. |
| G2 | **Prefill baseline** on Dell + dev box (none exists anywhere) | **PARTIAL** — A18 carries batched-prefill 1.5605x and SIMD-prefill 6.38x on Dell iron (`docs/hardware_logs/mech11_U_BOOTLOG_2026-08-01.txt`) but flagged *unpreregistered indicators*; no dedicated tok-throughput prefill baseline log exists | Needs: named-machine (Dell) dedicated prefill run, preregistered form. Pattern: Rule-D first — batched-vs-sequential prefill byte-identity gate (the `gemm_equivalence` parity test is the template), then the timing log. |
| G3 | **TII Falcon license + AUP read** | **DONE 2026-08-01** — with amendment | Not a measurement; artifact is the G3 addendum in the spec itself. Result: **BitNet-2B-4T (MIT) is model of record for anything DARPA-/customer-facing; Falcon-E-3B is internal capability core only.** Never call Falcon-E "open source" externally. |
| G4 | **E2 integer-ppl gate** (M7 -> 2B, kill >5% rel) | **PASSED**, all legs | M7 hybrid +0.3127% (A19, `docs/hardware_logs/cis1_e2_int_vs_float_ppl_m7_i5-10210U_crosvm_2026-08-01.log`); M7 full-int +0.0637% (A20, `.../cis1_fullint_attention_ppl_m7_i5-10210U_crosvm_2026-08-01.log`); BitNet-2B +0.7408% (A21, `.../cis1_e2_bitnet2b_int_vs_float_ppl_i5-10210U_crosvm_2026-08-01.log`). Machine: i5-10210U crosvm — quality only, per Rule A. Pattern: bit-identical run digests + goldens from an independent Python big-int generator (`scripts/cis_e2_golden_gen.py`). |
| G5 | **In-ternary SFT smoke** (does tool-call tuning transfer at all?) | **OPEN** | Needs: Axolotl QAT-SFT smoke on Falcon-E (internal-only model is fine — G3 split), eval **engine-loaded** on our harness, before/after tool-call score in one log. Pattern: engine round-trip digest of the tuned checkpoint (bit-exact load) before any quality claim. Train venue: cloud/Chromebook; eval numbers quality-only unless run on named iron. |
| G6 | **Bonsai-8B on our harness** (generation-mode + format compat) | **OPEN** | Needs: repo-level license verification of its Apache 2.0 claim, repack through `aegis-forge` (watch the g128/FP16-scale convention — B4-class bug risk), then IFEval-style + tool-call + generation, engine-loaded, logged. Pattern: loader parity + golden transcript before any benchmark. |

## 5. Scaffold spec sketch (Track L userspace, Rust)

Per spec §4, with the survivability patterns this program has already
proven in production named as the things to port:

1. **Tool loop:** plan-once-then-act (one planner call -> deterministic
   executor -> one verifier call), bounded micro-ReAct (<=3–5 hops) only
   inside a failing step. Tool contract = the fable-hand pattern: CLI tools
   with JSON contracts, honest scope statements. Two-phase constrained
   decoding in-engine: free-text thought, trigger token, grammar-locked
   action block.
2. **Filesystem + editor primitives:** `read` (path, range), `write`
   (create), `edit` (exact-match replace — the Claude Code Edit contract is
   the proven shape), `exec` (cargo/rustc/test, captured output), `search`
   (ripgrep-class). Every mutation appended to the witness-chained
   transcript.
3. **Context/memory files — what "keep of yourself" means concretely.**
   Three patterns, all battle-tested in this program, port them as-is:
   - **The memory-dir pattern** (`.../memory/MEMORY.md` index + topic
     files): distilled, curated state that survives session death; the
     scaffold maintains its own equivalent on the boot volume.
   - **The OPS_BRIEF pattern** (`~/aefinity-ops/`): collectors write to an
     inbox; session start reads the brief, drains the inbox, re-arms
     timers. This is how the assistant resumes after power loss.
   - **The invariants re-injection pattern** (`.claude/rules/invariants.md`
     via SessionStart): hard rules re-entering context every session, not
     trusted to memory. Aegis OS ships its own invariants file; the
     scaffold injects it into every planner context unconditionally.
4. **Integrity reflexes enforced by the harness, not by hope:** a
   figure-guard on model output (numbers must trace to a log or citation —
   `hooks/integrity_gate.sh` is the prototype), the witness chain on every
   transcript (verifiable by Track U), negative findings filed as results.
5. **Retrieval as the knowledge organ:** static-embedding retrieval
   (model2vec-class, CPU-trivial) over the local corpus — this repo, the
   ledger, the hardware logs, cached papers. Cites sources by construction.
6. **Honest context:** treat 4–8k tokens as the working window; compaction
   and tool-result clearing from day one.

Models per G3 split: **BitNet-2B-4T** model of record (external/demo);
**Falcon-E-3B** internal capability core; **M7 (2.8MB)** boot-anywhere seed
and speculative-draft candidate (draft-retrain gate in spec §1 stands).

## 6. What this document does NOT claim

- **No performance number appears here without a `docs/hardware_logs/`
  path attached** (Rule B), and none originates from QEMU/TCG or crosvm
  (Rule A). Everything unmeasured is written as `target:` or OPEN:.
- target: usable interactive decode for an agent loop on the Dell.
  OPEN: what tok/s an agentic loop actually needs is itself unmeasured;
  prefill (G2) likely dominates — tool outputs are prompt, not decode.
- OPEN: the spec's "~10% of ceiling, 8–10x kernel headroom" derivation
  used the 25.6 GB/s *theoretical* DDR3L figure; A24 measured the Dell's
  real 1T sequential peak at ~11 GB/s. Headroom-to-the-real-roof is
  smaller than the spec implied, and A27's 2.94x already cashes part of
  it. The budget needs a re-derivation against A24, written down.
- OPEN: end-to-end token-level throughput of the CIS AVX2 kernel and
  colskip (A23, 2.88–2.89x kernel-level, same logs as A24) — kernel
  ratios are not token ratios; system A/B unwired.
- OPEN: whether in-ternary SFT transfers tool-calling gains at all (G5 —
  the whole "train my behavior in" plan hangs on it).
- OPEN: Track U tool-loop scope — which tools are worth firmware cost
  beyond read/append (write-heavy FAT32 in ring 0 is a risk decision).
- OPEN: RAM per target machine still never pinned by dmidecode (spec
  critic Q4) — one command per box, next physical session.
- No claim that a 3B ternary model "is" a Claude. The spec's honest line
  stands: the method survives the compression; the parametric knowledge
  does not.

## 7. Next three actions

1. **G2 dedicated prefill baseline on the Dell** (preregister the form
   first, per MECH v2 precedent). Byte-identity gate before timing.
   Verify: `scripts/devloop.sh gate` green before staging;
   `scripts/check-efi-simd.sh <efi>` on any staged binary; after the boot,
   log lands in `docs/hardware_logs/` and `scripts/verify-figures.sh`
   passes with the new row.
2. **G5 in-ternary SFT smoke** (Falcon-E + Axolotl, tool-call format
   corpus from our own traces). Verify: tuned checkpoint round-trips the
   engine loader bit-exactly (parity test), before/after eval in one log;
   `scripts/verify-figures.sh` before any ledger row.
3. **SCAFFOLD-0, design-only + Rule-D test first:** freeze the tool-call
   grammar and witness-chained transcript format (extend witness v1), and
   land a bit-exactness test: a canned transcript replays to an identical
   chain digest on dev box and, via the kit image, under
   `cargo xtask boot-test` (exit 33 = pass). No model build hours — this
   is harness work the gates do not block. Verify:
   `scripts/devloop.sh gate` + `cargo xtask boot-test`.

---

*Slots into ROADMAP as a new lane ("Aegis OS / FABLE-0 lane") alongside
hybrid-engine, RANGER, ALICE-evidence, model-lab; gates G1–G6 keep their
spec numbering, and the ledger's Objective 1/2 rows (A29/A30, cross-ISA
identity) are this lane's portability substrate. Review before any edit
lands in ROADMAP.md.*
