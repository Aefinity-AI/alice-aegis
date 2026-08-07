# AEFINITY MODEL SCALE LADDER — one family, every size this program can build

**User directive 2026-07-18:** extend the M7 techniques into "a whole
operational scale of models from this lower tier going up as high as we can
build on here." This doc is the honest engineering answer: what each rung
costs, what it can do, and where the from-scratch ceiling on this Chromebook
actually is. Companion to MODEL_LAB.md (M-series gates apply to every rung).

## The unifying idea: one family, one contract

Every rung shares the SAME verified stack — this is what makes it a product
family instead of a pile of experiments:

- **Architecture recipe:** tinybit Llama-style decoder (RMSNorm, RoPE, GQA,
  SwiGLU, SubLN, tied head), ternary QAT via BitLinear absmean+STE.
- **One family tokenizer** (current 8k BPE; may re-train at 16k for upper
  rungs — decide before A2, because it locks the family).
- **One export contract:** train.py → export_hf.py → repack_ternary.py →
  aegis engine. Round-trip parity gate MANDATORY per rung (proven at 0.20%,
  480/480 tokens, ledger M12).
- **One eval harness:** engine-side PPL + --mc (parity-gated, ledger M15) +
  the M6 domain eval once built. Every number: runtime log or it didn't
  happen.
- **One data ledger:** PROVENANCE.md (license + generator-model per subset).
- **Within-family distillation is LEGAL:** the cross-tokenizer logit-KD
  kill (ledger, wf_914021b5) applies to *foreign* teachers. Same-tokenizer
  rungs can distill into each other. Teacher logits (top-8) can be
  precomputed on free cloud and stored — 500M tokens ≈ 8–16GB, which the
  new external drives hold trivially. Upper rungs lift lower rungs.

## Measured basis (adversarially verified 2026-07-17, this box)

Ternary QAT training throughput, torch 2.13 CPU 8T: **733 tok/s @8.5M
params · 458 @21M · 138 @47M**. Params × tok/s ≈ constant (~6.5–9.6 G·tok/s)
→ extrapolation: ~65 tok/s @100M, ~30 @200M. Memory: fp32 AdamW ≈ 16
bytes/param → even 200M params fits 6.4GB+swap; **compute, not memory, is
the local binding constraint.** (30M/50M points should be re-benched 1–2h
each when the box frees — extrapolation between 21M and 47M is the softest
part of this table.)

## The ladder

| Rung | Params | Tokens (≈20/param) | Venue | Wall-clock | Expected capability (literature-anchored, NOT yet measured) | Status |
|---|---|---|---|---|---|---|
| **A0** | 3.8M | (smoke) | local | done | round-trip proof only | ✅ ledger M12 |
| **A1 "sovereign"** | 10–15M | 300–500M | local | 5–12 d | TinyStories-class: coherent simple English, narrow-domain competence; 3–8MB artifact boots 2GB pre-AVX2 iron | 🔶 twin queued (M7a), ternary next (M7) |
| **A2 "workhorse"** | 25–35M | 500–700M | local | ~2–4 wks | TinyStories++ / simple instruction following on curated narrow domain; at this scale data curation beats params (TinyStories finding) | ⬜ after A1 gate |
| **A3 "local ceiling"** | ~50M | ~1B | local | ~3 mo | upper bound of useful local from-scratch; GO only if A2 gate shows the curve still paying | ⬜ decision after A2 |
| **A4** | 90–125M | 2–2.5B | Kaggle TPU v5e-8 (=M9a) | wks (XLA smoke first) | SmolLM-135M-class IF data is good: borderline general utility, decent narrow assistant | ⛔ Kaggle verify + XLA smoke |
| **KD pass** | A4→A1/A2 | precomputed top-8 logits | cloud gen + local train | days | lifts small rungs above their from-scratch quality; logits stored on external drives | ⬜ after A4 |
| **P-tier (capable)** | 1B / 2B / 3B | fine-tune only | Kaggle GPU (=M5/M6) | GPU-hours | ports (Falcon-E-1B/3B, BitNet-2B) + our domain SFT = the flagship; from-scratch at this size is DEAD on $0 (20B+ tokens ≫ free quota) | 🔶 bases ported; SFT gated on user |

## The ceiling, stated plainly

From-scratch on this Chromebook tops out at **~50M params trained
compute-optimally (~3 months)**. 100M from scratch locally = ~1 year: dead.
Above 50M the ladder rides free cloud (A4) and pre-quantized bases (P-tier).
That's not a limitation to hide — it's the DARPA story: *a documented,
reproducible recipe for what edge-sovereign training actually costs at each
scale, with the crossover points measured, on $0 of infrastructure.*

## Sequencing & gates (extends the MODEL_LAB calendar)

1. **A1 first, nothing else local until its gate closes** (twin vs ternary,
   round-trip, engine PPL, iron boot). A1's result IS the go/no-go data for
   A2: it gives the real tok/s at 14M and the quality-per-token curve.
2. A2 GO criteria: A1 passed its twin gate AND re-benched 30M throughput
   ≥250 tok/s (else wall-clock >5 wks — park, prioritize A4/P-tier).
3. A3 GO only if A2 shows quality still scaling AND the box has no higher-
   value tenant (P-tier SFT experiments outrank it if Kaggle unlocks).
4. A4 (=M9a) and P-tier (=M5/M6) proceed in parallel on cloud, gated on the
   user's Kaggle/SageMaker account actions — they don't touch the box.
5. Every rung: checkpoint-resume discipline (SIGKILL-resume already PASS),
   heartbeat log for suspend forensics, df logged before launch.

## Risks specific to multi-week local runs

- **ChromeOS sleep/power** is the #1 killer (VM freezes with host). Power
  settings + lid + heartbeat (`check_heartbeat.sh`) are mandatory pre-launch
  checks for A2/A3. Resume-from-checkpoint is the recovery path, not
  prevention.
- Thermals: i5-10210U sustained all-core for weeks — watch for throttling in
  tok/s logs (a slow decay in step time = thermal, not code).
- 8k tokenizer may be too small for A4-scale data diversity; decide 8k vs
  16k family tokenizer BEFORE A2 so upper rungs stay KD-compatible.

*Created 2026-07-18 from measured wf_914021b5 throughput data + M7a/M12/M15
gate results. Numbers in "Expected capability" are literature anchors
(TinyStories, SmolLM, Pythia) — replace with OUR harness numbers as rungs
complete. Every completed rung gets a ledger row.*
