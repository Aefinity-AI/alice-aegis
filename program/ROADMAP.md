# Master Roadmap — Aefinity AI edge-inference program

**North star:** the smallest, fastest, most capable AI that boots on bare
commodity hardware — the A.L.I.C.E. unikernel — backed by research we can
defend number-by-number in front of DARPA. Facts first; a negative result
with a log beats a positive claim without one.

**Three lanes, one destination:**

```
RANGER (quantization science)──┐
                               ├──► ALICE unikernel (no_std Rust, bare metal)
Hybrid engine (ternary portfolio)─┘         ▲
                                            │
              DARPA RFI / fleet evidence ───┘  (proof it's real, on real iron)
```

- **RANGER** answers *what precision/protection tricks actually work* (and
  which published levers are artifacts) — so ALICE's future quantization
  choices are evidence-based, not vibes.
- **Hybrid engine** turns aegis-core from a one-model appliance into a
  *portfolio engine* for every viable ternary checkpoint family — model
  choice becomes a dial (small/fast ↔ big/quality) on the same verified core.
- **Fleet evidence** proves it on real hardware, honestly labeled, down to
  2GB pre-AVX2 machines.

Status markers: ✅ done+verified · 🔶 in flight · ⬜ next · ⛔ blocked
(blocker named). Every number's provenance lives in `RESEARCH_LEDGER.md`.

> **SESSION BOOT 2026-07-18:** live automation, event→action playbook, and
> next steps are in `HANDOFF_2026-07-18.md`, which is kept in the working
> repository and is deliberately **not** part of the public snapshot (it
> records operational state, not research). Read it before starting any
> model-lab work. Scheduled jobs own the box; do not start heavy compute
> manually.

---

## Phase history (simplified — what we did and what each step bought us)

### P0 — Escape the transmutation tar pit ✅
15 months of dense→ternary conversion produced zero coherent outputs.
A verified research sweep settled it: post-training ternarization collapses
modern sub-7B checkpoints, full stop; recovery needs datacenter-scale
continual pretraining. **Gain: stopped spending on a dead end; pivoted to
pre-quantized checkpoints (Microsoft BitNet), which produced first real
coherence within days.** Negative finding, fully documented (ledger B1).

### P1 — Make one model real on bare metal ✅
BitNet-2B in a from-scratch no_std Rust engine: five correctness bugs fixed
(soft-float UEFI target being the killer — every prior build had zero vector
instructions), coherent generation in userspace and bare-metal UEFI.
**Gain: the core claim exists — an LLM answering questions with no OS.**

### P2 — Make it fast, honestly ✅
Profiling showed the engine latency/ILP-bound — ≈2.3 MAC/cycle/core, ≈14% of
single-core AVX2 f32 peak (see TECHNICAL_REPORT §6; the 21.7% carried in earlier
drafts divided MACs by rdtsc ticks, which are not core cycles — this machine's
measured effective/nominal ratio is 1.53×). Ordered by payoff:
batched prefill GEMM (1.63–1.75×), thread-safe multicore (2.80 → 4.94 tok/s =
1.76×, both from committed logs),
int8 activations (the apparent quality win, PPL 12.801→12.738, is **retracted**
— one 1,842-token window, one run, no error bars, and the source record says the
sign could reverse; superseded by the full-testset measurement B9. The speed win
was correctly REJECTED — no VNNI on this silicon). Two seductive "optimizations" (CTZ
zero-skipping, T-MAC-style pshufb LUT) benchmarked and rejected with math.
**Gain: ~3× real throughput, plus a kernel playbook of what NOT to do on
AVX2 — each rejection is a defensible engineering finding.**

### P3 — Make the numbers defensible ✅
Fabricated benchmarks purged; every metric now computed at runtime with a
log. Byte-level tokenizer fix (the 14.488→12.801 pair that once evidenced it is
**retracted** — superseded, and unreproducible since its dataset file was
deleted); the standing figure is the full-testset PPL 15.825,
battery-discharge energy 3.99 J/token, peak-memory instrumentation
(≈1.35GB total on 2GB-class machines), /gauntlet self-benchmark that runs
scalar-vs-SIMD races on any box it boots. **Gain: a claims package that
survives adversarial audit — because we ran the adversarial audit ourselves
(88 agents, 77 findings, all remediated).**

### P4 — Prove it on iron + submit ✅
Three-machine fleet (Dell i5-5200U Broadwell, HP Stream N4020 pre-AVX2,
QEMU matrix 6/8), every row environment-labeled with caveats (Dell's 22%
clock stated, not hidden). DARPA-SN-26-97 RFI submitted 2026-07-14, three
days early. **Gain: bare-metal LLM inference demonstrated on 2GB commodity
hardware — the RFI's core evidence — and a fleet harness anyone can extend
with one USB stick.**

### P5 — RANGER Phase 1: quantization ground truth ✅
Real-model experiments (SmolLM2-135M, Qwen3-0.6B), adversarially verified,
including retracting our own E7 result when it proved artifactual:
- **Rotation is the invariant** — recovers ~92.5% of 4-bit activation
  collapse across both architectures. The one lever that generalizes.
- Weight-space census predicts activation outliers (cheap, static analysis)
  — but census-based *exemption* is architecture-dependent (59% recovery on
  SmolLM2, ~0–25% on Qwen). Now we know the boundary.
- Super-weight protection: null at 135M, real at 0.6B — scale-dependent.
- MSE clipping sign-flips with scale (hurts 135M, helps 0.6B) — confirmed
  on disjoint data.
**Gain: an evidence-based decision table for edge quantization — which
published levers are real, at what scale, on which architectures — instead
of trusting papers' own evals.**

### P6 — Hybrid engine: one core, a portfolio of models ✅ core / 🔶 gates
Engine generalized (SwiGLU, optional SubLN, untied LM head, config embedded
in the model file) with the BitNet anchor reproduced EXACTLY afterwards —
zero drift. Falcon-E-1B repacked and running coherently in the same binary.
Found+fixed the weight_scale convention bug (divide-vs-multiply) that
silently token-salads anything with transformers' BitLinear convention.
Caught a packing-axis error in review that would have silently scrambled
Llama3-8B weights. **Gain: aegis-core is now a ternary *platform*; adding a
model family is a repack, not a rewrite.**

---

## In flight / next (ordered, with blockers)

### Hybrid-engine lane
- ✅ **Logged re-run of the Falcon-E generation + BitNet 10.348 anchor**
  (2026-07-16 loop iteration 1: anchor reproduced EXACTLY at 10.348/1899
  tokens; Falcon-E coherent at 4.39 tok/s decode — both logs in
  docs/hardware_logs/).
- ✅ **T2d Falcon pre-tokenizer parity PASSES** (loop iteration 3):
  diagnosed the 15-token divergence (every site a broken ' \n \n' group),
  fixed whitespace-run pre-tokenization + id-0 merge guard in one change
  per the runbook, T2d now EXACT (2410/2410 ids).
- ✅ `scripts/phase0_meter_check.sh` rewritten (was pinning the superseded
  12.80 anchor AND silently running chunked mode post-merge — a broken
  gate); now --sample / reference=test.txt / EXPECTED=10.758 (re-pinned
  after the tokenizer fixes; per-token rise is a tokenization-density
  artifact, documented in the script header and run log).
- ⛔ **G4a**: Falcon-E engine-PPL within 5% of reference — T2d cleared;
  needs the reference PPL run (Colab; model too big for this box's
  transformers stack).
- ⛔ Llama3-8B-1.58 quality tier — needs 1TB drive mounted + G1 kill gate
  (measure its PPL first; its 12.2 claim was refuted 0-3).
- ⬜ maddubs int8 kernel — parked behind bench gates (`lut_mpgemm`,
  `gemm_tile`); bench first, iron second, coherence third.

### RANGER lane
- 🔶 **RECOVERED: the interrupted cloud workflow's products are on GitHub**
  (found 2026-07-16 loop iteration 3 via fetch). A cloud session
  (session_019yn6fEuV1GLreL7V9DMpsw) spent Jul 16 building experiment labs
  on the `claude/edge-quantization-hybrid-research-9clrdh` branch: 8-item
  evidence-per-FLOP roadmap (research/ROADMAP.md), Experiment A
  cross-family outlier census, C quantized-drafter speculative decoding,
  D super-weight hunt, Phase-2 QAT scaffold + CI. User merged PRs #1/#2/#4
  into origin/main; **Experiment B (KV-cache lab) merged as PR #5 on
  user's instruction (2026-07-17 04:28 UTC; local main at 64da6de) — the
  entire recovered workflow is now in main.** CI (selftests) was pending
  at merge; verify it went green. NEXT: reconcile the two research
  lineages (cloud labs ↔ local phase1-real-model E7–E12 results), then
  run the CPU-safe labs locally once the full-set PPL run frees the box.
- ⛔ **Phase 2** (4 experiments, P2.1 outlier-suppression finetune first) —
  needs one 24–48GB GPU, ~1–4 GPU-days each. Plan at research/phase2/PLAN.md.
- ⬜ RESULTS-phase1.md E8 wording fix: "top channel 2–12% of tokens"
  understates (8/30 layers exceed 12%, up to 55%); conclusion unchanged (E3).
- ⬜ Theory-doc pillar-status updates never landed in
  edge-quantization-hybrid.md (only §3.8 was fixed) — apply or close (E4).
- ⛔ Push ranger to GitHub + reconcile branches (work is on
  `phase1-real-model`; `main` is still the initial commit) — user must run
  `gh auth login` first.

### ALICE / evidence lane
- ✅ Context-limit crash fixed (2026-07-16): decode loop now clamps to the
  config window, prompt clamp config-derived (was hardcoded 1948);
  regression test proven to fail on the unfixed code.
- ✅ TECHNICAL_REPORT §2 row 4 corrected to the logged 3.60B/3.71B
  cycles/token (stale 4.01B traced only to a commit message).
- ⬜ bitnet.cpp / llama.cpp honest baseline comparison.
- ⬜ OS-vs-no-OS energy delta (the unikernel's headline efficiency claim —
  currently unmeasured).
- ⬜ forward_batch/forward_step unification (riskier; parity tests police
  the duplication meanwhile).

### Model-lab lane (NEW 2026-07-17 — train/distill/port; plan = program/MODEL_LAB.md)
- ✅ 28-agent adversarially-verified research (wf_914021b5-c1f): decision
  table + M-series pipeline + refuted-claims guardrails (ledger §M).
  Regime: Chromebook-only + free cloud, unlimited wall-clock.
- 🔶 M0 G4a gate (engine 4.837 @1896 tok logged; transformers reference
  running), M1 Falcon-E-3B port (packed downloading), M2 corpus assembly
  (downloading, text-form), M7 tinybit scaffolding + step-0 round-trip gate
  (agent building).
- ⬜ M3 engine MC harness; M4 soft-prompt (corrected recipe); M5 cloud QAT
  smokes (needs Kaggle phone-verify); M6 1–2B specialization; M7 sovereignty
  model (after full-set PPL rerun frees the box); M8 8B port; M9 gated.
- 6GB swapfile added 2026-07-17 (btrfs NOCOW, fstab) — RAM ceiling now
  6.5GB+6GB; full-testset PPL rerun relaunched under systemd-run + linger
  (attempt 1 died at session close 2026-07-16 23:47).

### Program lane
- 🔶 This scaffolding (protocol, ledger, roadmap, named audit workflow).
- ⬜ Aefinity AI website (user goal, separate track).

---

## The DARPA story (if they call)

1. **Demonstrated**: a from-scratch no_std Rust LLM unikernel booting on
   commodity 2GB hardware, no OS, 782MB weights + <600MB heap, coherent
   generation, self-benchmarking on any machine it boots. Fleet-proven on
   pre-AVX2 silicon. Submitted RFI is fully environment-labeled and
   audit-hardened.
2. **Methodology**: measurement-integrity discipline (anchored regressions,
   adversarial multi-agent audits, negative findings kept on the record) —
   directly responsive to "low-resource computing" needing trustworthy
   numbers, and we can show the audit trail.
3. **Research depth**: RANGER gives us original, replicated findings on
   which quantization levers survive contact with real models at edge
   scale — rotation-invariance, census→activation bridge, scale-dependent
   super-weights. This is the science that decides ALICE's next precision
   regime (W1.58A8 and below).
4. **Trajectory**: one verified engine now runs multiple ternary model
   families — capability becomes a configuration choice per mission
   footprint (1B fast / 2B canonical / 8B quality-tier pending gates).

---

## Session timeline (how the work actually flowed, local + cloud)

Reconstructed 2026-07-16 by the audit workflow from transcript metadata,
commit author dates, and memory provenance:

- **Jul 9–10** — LOCAL marathon: full-disk audit (real vs fabricated drawn),
  five correctness bugs → first coherent bare-metal boot, perf tier 1
  (GEMM/multicore/int8), CTZ falsified, energy method, gauntlet harness.
- **Jul 12** — LOCAL: Antigravity-transcript triage (fabrication boundary),
  meter repair, throttle diagnostics, Dell i5-5200U iron gauntlet.
- **Jul 13** — CLOUD product #1: `ternaryportfolio.patch` — 5-commit series
  (engine generalization + repack pipeline + review fixes), delivered as a
  patch because no remote exists.
- **Jul 14** — LOCAL ~20h bridged session: RFI remediation (88-agent line
  audit), logged energy runs, HP N4020 fleet datapoint, **RFI v4 submitted
  to DARPA-SN-26-97, 3 days early**. Same day, CLOUD product #2: RANGER
  theory + synthetic baseline (E1–E6) on the GitHub `claude/…` branch.
- **Jul 14–15** — LOCAL: RANGER pulled cloud→local, Phase 1a census, E7
  launched. Then the outage window: internet loss, API errors, OAuth expiry
  (three stub sessions, no work lost — checkpoints held).
- **Jul 16** — LOCAL ~16.5h ultracode day: E7 retracted as artifactual,
  E8–E12 + confirmation pass executed and committed; ternary-portfolio
  patch applied → `hybrid-engine` branch; weight_scale bug found+fixed;
  Falcon-E running. Ended on API 529.
- **Jul 16 (tonight)** — this session: two cloud ultraplan launches failed
  (401), went local; full program audit (14 agents, 102 claims checked:
  91 confirmed / 6 plausible / 5 refuted); this scaffolding created.

**Resilience note:** three network/auth outages cost zero committed work —
resume-safe checkpoints + handoff files + memory carried every line over.

*Created 2026-07-16, grounded by audit run wf_7bcb9985-6bc. Re-ground with
the `program-audit-roadmap` workflow whenever state may have drifted.*
