# ALICE Launch Protocol — from artifact to funding

**v0.1 — July 10, 2026 — Aefinity AI**

Doctrine: every claim traceable to a runtime computation. No number gets printed that
the machine didn't compute in that same run. The benchmark table is the keystone —
the writeup, the SBIR package, and the fellowship application all stand on it.
We do not claim "first." We claim "fastest verified," and we prove it.

---

## Phase 0 — Integrity purge + slow paperwork (start both today)

### 0.1 Mock-metric audit (P0)

Verify each of these in **current source** before assuming anything is fixed
(trust code, not changelogs). For each: locate the exact lines, then either make
the number computed at runtime or delete the code path. Nothing gets built on
top of a mocked metric.

| Location | Known offense |
|---|---|
| `aegis-eval/src/perplexity.rs` | Hardcoded 14.12 / 14.58 PPL ("mock computation for DARPA review") |
| `antigravity` CLI `/benchmark`, `/status` | Fixed TTFT 14 ms / TPS 84.6 / RSS 412 MB |
| UEFI `/grant-review` | Unconditional "Phase I Grant Viable" verdict |
| UEFI `/benchmark` | Cycles→seconds assumes fixed 2.5 GHz TSC |

**Gate:** `python scripts/integrity_check.py <repo-root>` exits 0.

### 0.2 Registrations (long-lead bureaucracy — blocks September if not started now)

- SAM.gov entity registration for Aefinity AI. Free; takes weeks; required for any
  federal award. Ignore paid "registration help" solicitations — they're scams.
- Login.gov account, then DSIP account (DoD SBIR submission portal).
- SBIR.gov company registry entry.

---

## Phase 1 — Honest instrumentation

- **Clock:** TSC calibration via CPUID leaf 0x15. If the leaf is unavailable on the
  target box, measure TSC against a firmware timer and record the method. Never
  assume a fixed clock.
- **Metrics defined once, used everywhere:**
  - TTFT — prompt submitted → first generated token.
  - Decode rate — steady-state tok/s, excluding the first few tokens.
  - Prefill rate — prompt tok/s (compute-bound; reported separately).
  - Load time — firmware handoff → weights resident (cold vs warm reported separately).
  - Peak memory — arena high-water mark.
- **Perplexity:** teacher-forced NLL on WikiText-2 using the **pruned** tokenizer
  mapping. The pruned-vocab caveat is disclosed wherever PPL is reported.
- **Bandwidth denominator:** STREAM-style triad microbenchmark at boot → measured
  GB/s. Speed claims are reported as tok/s **and** as % of the measured roofline
  ceiling. Efficiency is the honest brag.
- **Provenance rule:** every printed number points to the runtime computation that
  produced it (definition-of-done gate 4).

---

## Phase 2 — The Table

**Hardware:** document exact machine(s) — CPU model, RAM speed and **channel count**,
storage path (USB 2/3, SATA). Run the single-vs-dual-channel experiment explicitly:
the $25 second SODIMM is a headline result if it doubles decode rate as predicted.

**Contenders (same physical machine, same weights lineage):**

| System | Environment |
|---|---|
| ALICE | Bare-metal UEFI, no OS |
| bitnet.cpp | Linux, same BitNet b1.58-2B-4T weights |
| llama.cpp | Linux, BitNet support or nearest comparable quant |

**Protocol:** fixed public prompt set (~10 prompts, varied lengths); fixed context
and generation lengths; ≥5 runs per cell; report median ± spread; pinned threads;
baselines built honestly (`-O3 -march=native`, flags documented); raw logs committed
to the repo. Disclose pruned vocab (ALICE) vs full vocab (baselines) and quantify
the PPL delta. Publish losses as well as wins.

**Deliverable:** one table — TTFT / decode tok/s / prefill tok/s / peak mem / PPL /
% of measured bandwidth ceiling — plus raw logs.

---

## Phase 3 — Publication

- **Repo hygiene:** nothing references `*_ALICE_1_0_BACKUP` or scratch dirs; QEMU
  repro path (`qemu-test`) documented so anyone can verify without hardware; README
  gets from clone to boot in under ten steps.
- **Writeup:** "Booting a language model with no operating system." Lead with the
  systems contribution — the model is Microsoft's and the writeup says so plainly.
  Related work cites the prior art: the freestanding-C UEFI inference demo
  (Dell E6510, early 2026) and cllm (bare-metal C unikernel serving inference over
  HTTP). Differentiation: ternary weights + hand-written AVX2 LUT kernels +
  memory-safe `no_std` Rust + verified measurement.
- **Distribution:** public repo, writeup, boot-to-chat demo video (the money shot),
  HN / r/LocalLLaMA.

---

## Phase 4 — Applications (ride entirely on Phases 0–3)

- **Anthropic Fellows Program:** rolling applications; next cohort expected late
  Sept/Oct 2026. Target: application submitted by ~mid-August with the table and
  writeup linked. No PhD or publications required; US work authorization suffices;
  remote OK.
- **DoD SBIR / AFWERX open topics + NSF SBIR (America's Seed Fund):** verify
  currently open windows at DSIP and sbir.gov (dates shift); Aefinity AI is the
  applicant entity. Package per `references/grant-readiness.md` checklist — scope
  statement, novelty framing, presentation hygiene.
- **Positioning line:** *verified, air-gapped, OS-less inference appliance —
  minimal attack surface, portable across commodity x86.*

---

## Global gates (apply to every phase)

1. `integrity_check.py` passes, or every remaining finding is explicitly acknowledged.
2. Prefill/decode parity check after any inference-math change.
3. Boots under `qemu-test` before anything is flashed to real hardware.
4. Every printed number traces to a runtime computation.
5. Backups and scratch dirs untouched and unreferenced.

---

## Timeline (aggressive but honest)

| Window | Work |
|---|---|
| Now → Jul 25 | Phase 0 complete; registrations in flight |
| Jul 25 → Aug 10 | Phase 1 instrumentation |
| Aug 10 → Aug 22 | Phase 2 table (incl. SODIMM experiment) |
| Aug 22 → Sep 5 | Phase 3 publication |
| September | Applications out — Fellows first (rolling), SBIR per window |

## Honesty ledger (known caveats we disclose, not bury)

- Pruned-vocab perplexity is not directly comparable to full-vocab baselines; we
  quantify and state the delta.
- Old-Dell TSC behavior varies; calibration method is recorded per machine.
- bitnet.cpp baseline build friction is expected; build steps get documented so the
  comparison is reproducible.
- "First bare-metal LLM" is not claimable and never gets claimed. "Fastest verified
  bare-metal inference of a ternary LLM on commodity hardware" is the target claim,
  and only after the table exists.
