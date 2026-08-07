# Decision memo — 2026-07-30

Scope: partition-2 model behavior; state of the M7 twin program after the
2026-07-29 session crash; ternary-vs-fp model-type decision; unikernel-vs-Linux
posture; novelty audit. Internal numbers cite instrument logs; external claims
cite sources (verified 2026-07-30). Ledger rows implied by this memo are
drafted in the session scratchpad and are guard-gated (user applies).

---

## 1. Why the partition-2 model "kept telling stories"

**It is working exactly as trained.** Partition 2 carries the M7 sovereignty
model (MODEL.SAF 2,797,632 B — the boot log in
`docs/hardware_logs/m7_baremetal_prompts_postfix_2026-07-29.log` confirms
`model=2797632`). 100% of its 500M training tokens are TinyStories — children's
stories. It has never seen a single question-answer exchange, instruction, or
chat transcript. A base LM continues the prompt under its training
distribution, so "hello alice" → a story about Lily, and "how are you today?"
→ a story continuation. BitNet-2B (the other partition) answers because
Microsoft instruction-tuned it. This is not a bug, a quantization artifact, or
an engine defect.

**The chat-retargeted variant exists but is not the fix yet.** GATE 2 (commit
3162a2c; evidence promoted to
`docs/hardware_logs/m7_retarget_gate2_evidence_2026-07-30/`) retargeted M7 on
614,400 smoltalk tokens in ~10 min of laptop CPU; the deployable artifact
round-trips (0.95%, 471/471) and sits at
`model-lab/tinybit/m7_retarget_gate_work/artifacts/`. Tested today on
aegis-linux: it answers in chat *register* but is incoherent ("return
narrative.constrate a Pythrase for your new Apple…"). 614K tokens moved the
style, not the competence, and cost 84% general-PPL forgetting
(5.14 → 9.48). A 14M model that *answers* needs instruction data **in
pretraining** (mix TinyStories + chat-formatted data from step 0), not a
600-step afterthought. That is a training-mix decision for the next run, and
the M26 freeze finding (§3) says how to retarget cheaply once a base exists.

## 2. The 7-day training: nothing was lost

Everything from the 142.98h ternary run and its aftermath survived the crash:

| Artifact | Where | State |
|---|---|---|
| m7_ternary.pt (full ckpt + optimizer) | model-lab/tinybit/checkpoints/ + both external drives | intact |
| 2.8MB deployed artifact, round-trip 0.19% PASS | m7_final_gate_work + USB | intact |
| fp32 twin + cooled/hold ablation arms | checkpoints/ | intact |
| LR-cooldown ablation (prereg + verdicts) | docs/hardware_logs/m7lr_* | **complete — finished 12:32Z on 07-29, before the crash** |
| Retarget gate + flip/freeze probes | logs + probes/ | complete |
| lm-head 2-plane probe | m7_lmhead_argmax_probe log | complete |

The crash cost an interactive session, zero data. What the ablation cost is the
*headline*, which is the next section.

## 3. Twin-test verdict (the part nobody wants but the prereg demands)

Preregistered bands (`m7lr_PREREGISTRATION_2026-07-29.md`, git 1ee1f5c, written
before the contrast existed) against the completed run
(`m7lr_armh_verdict_2026-07-29.log`, 512 windows, Holm-corrected):

- d_gap (ternary vs twin): **+0.073351** nats/tok [+0.070252..+0.076525]
- d_cool (cooldown alone): **+0.039480** [+0.037635..+0.041389] — and this is a
  **lower bound** (arm K annealed 3.3% of training; the ternary annealed 25%)
- d_tokens: +0.003684 (the extra tokens did almost nothing)
- **R ≈ 0.54 → Band 2, cooldown-major.** Per the prereg: *"Cooldown is the
  single largest identified driver. Headline unsupportable as stated. Report
  the full decomposition; do not re-report the residual as a ternary win."*
- Residual (ternary vs cooled twin): +0.030187 [+0.027539..+0.032900, 427/512,
  p 5.00e-05] — significant, but the ternary arm carries **2.17× the
  parameters** (14,171,392 vs 6,529,920), so it is not attributable to ternary.

**Plain English: "ternary beat the fp32 twin by 6.8%" is dead as a headline.**
At least half the gap was an undisclosed learning-rate/weight-decay cooldown
the fp arm never got, and the rest is confounded by a 2.17× parameter gap. The
literature agrees this was the predictable confound: BitNet's own papers show
ternary tolerates and benefits from more aggressive LR schedules than fp
(arXiv 2310.11453; JMLR 2025). What survives: the decomposition itself, the
paired-eval infrastructure, and the prereg discipline — all reusable.

To resurrect a quality claim honestly, the run needed: parameter-matched arms,
per-arm LR sweeps, multi-seed error bars. That is a known, costed experiment —
not seven wasted days, but also not a result yet.

## 4. Model-type decision: ternary, fp32, or hybrid?

**Decision: stay ternary for deployment; train through fp shadow weights (which
QAT already does); stop claiming quality superiority until the controlled
experiment exists.** Reasoning, ours + literature (agent briefing 2026-07-30,
sources in session record):

1. **Deployment is the whole game for this program, and ternary wins it
   unconditionally.** 2 bits/weight is why 2.8MB boots on a 2GB Broadwell
   laptop at ~470-507 tok/s (m7_coherence_spotcheck_2026-07-28.log) and why the
   lut_mpgemm A/B settled on traffic grounds. Nothing in the fp column
   competes on the target iron.
2. **Quality at matched params, small scale: no one has shown ternary wins —
   including us.** TernaryLM's fp arm wins val PPL; BitNet parity starts ~3B;
   Spectra's parity is per-bit, not per-param; BitNet-Reloaded needs 2× width
   for LM parity. Our Band-2 result is consistent with all of that.
3. **Training route for <100M:** from-scratch ternary QAT is proven viable and
   nearly free at this scale (M5: measured local throughput). The
   compute-optimal published recipe is ParetoQ's ~90% fp pretrain + ~10% QAT
   anneal (arXiv 2502.02631) with optimizer-state retention (ACL Findings
   2025, arXiv 2502.11895) — worth adopting when runs get expensive (Kaggle
   lane). Pure PTQ-to-ternary below billions is contraindicated by everything
   published and by our own B1.
4. **"Ternary isn't retrainable" is half-true, and we measured the useful
   half.** Full retargeting flips ~4.2% of ternary weights and forgets 3.3×
   more than fp (commit 3162a2c negatives). But the L03 freeze control
   (m7_ternary_flip_rate_L03_2026-07-29.log) shows **freezing the ternary core
   and training only the fp periphery captures ~70% of the domain gain at ~5%
   of the forgetting (0 flips)**. Falcon-E's own recipe (shadow-weight
   finetune, full-FT only, PEFT open) confirms the field has no better answer.
   This freeze result is one of our few genuinely novel, publishable findings.

## 5. Novelty audit — what have we actually learned?

**Killed or known (do not pitch as new):**
- bd-PROCHOT / MSR 0x1FC: ~15 years of prior art (A12, already ledgered).
- "Bare metal = uniquely deterministic measurement": measured FALSE by our own
  skeptic probes — hosted `instructions:u` CV 0.000465% vs best bare-metal tick
  CV 0.0780%, 168× in favor of hosted (skeptic_variance_budget_kill,
  skeptic_novelty_instret_kill, 2026-07-29).
- "LLM boots from firmware with no OS": **scooped as of this morning.**
  NightRun (github.com/hardrave/NIGHTRUN, CNX Software 2026-07-30) is a Rust
  no_std UEFI app that stays in Boot Services, uses firmware MP services,
  runs Llama 3.2 1B/3B, Granite 3B, Qwen3 4B, MIT-licensed. Verified
  firsthand today. Two earlier demos existed (L2E 2023; freestanding-C UEFI
  chat, Mar 2026). Notably NightRun's x86 numbers are from an 8-core QEMU/KVM
  VM — the exact practice our Rule A prohibits; its only real-iron numbers are
  Pi 5 (3.0 tok/s decode, Granite 3B).
- Ternary regularization narrative, per-bit Pareto optimality, fp-then-anneal
  superiority at ≥600M: all published (TernaryLM, ParetoQ, CQPT).

**Genuinely ours (defensible, mostly unpublished territory):**
1. **Ternary plasticity instrumentation** — per-step flip rates by LR, the
   fragility surface (20.07% of latents within 10% of a boundary), and the
   freeze-control decomposition showing forgetting lives in the flips. Only
   one published flip-rate number exists anywhere (arXiv 2412.04787, ~0.05%/step).
2. **The paired bare-metal-vs-Linux measurement, same engine, same iron.** The
   research sweep found *nobody* has published a rigorous OS-cost measurement
   for LLM decode on identical hardware. We are uniquely instrumented to own
   that number — and our skeptic probes already show we'll report it honestly
   whichever way it lands.
3. **The verification methodology itself** — bit-exact golden outputs,
   raw-log-or-it-doesn't-exist provenance, preregistered bands, adversarial
   self-audit. Aligned with what DARPA is actually funding in assurance
   (CLARA-class), and it is the one asset a retraction-scarred program can
   assert without new risk.
4. CPU-only from-scratch ternary QAT at usable quality (TernaryLM needed a T4;
   we used a laptop; our 14M val PPL 5.1-5.5 class vs their 132M @ 58.42 —
   different tokenizers, so not directly comparable, but the resource claim
   stands).

## 6. Unikernel vs a purpose-built Linux

**What staying in UEFI Boot Services permanently forecloses:** GPU/NPU compute
(no drivers; AMD XDNA needs Linux ≥6.14), real networking (SNP/HTTP-boot is a
boot-image fetcher, Wi-Fi near-nonexistent), persistence beyond FAT32 reads,
ACPI power management (every machine a reverse-engineering project — our own
bd-PROCHOT proves both that it's possible and that it doesn't scale),
suspend/resume, Secure Boot in practice, and post-ExitBootServices SMP without
writing half an OS (our own doctrine note at
baremetal_speed_findings_2026-07-29.md:483 already said this).

**What it uniquely provides:** seconds to inference, zero OS jitter, one-binary
TCB, all-RAM, and a firmware-level control arm no Linux setup can be.

**Posture decision: the unikernel is the instrument, not the product.** As a
product it now has three free MIT-licensed competitors and zero deployments
anywhere; every commercial edge-AI vendor ships Linux. The user's year-ago
instinct — an AI-centered OS — matches the actual open niche: no purpose-built
minimal AI-first Linux distro exists as a product (the niche is empty; the
moat is our kernels + measurement discipline, not the distro concept).
Concrete path: keep `aegis-uefi` as the zero-OS control arm; build a
Buildroot/initramfs minimal-Linux arm (~5-15MB, boot ~2-4s, isolcpus+nohz_full;
1-2 weeks); publish the paired OS-cost measurement (§5.2) as the flagship
result. That converts the unikernel from a scooped stunt into apparatus for a
measurement nobody owns.

## 7. Actions

- [ ] USER TODAY: SageMaker Studio Lab signups close **2026-07-30** (ledger M9).
      Sign up today if that lane is wanted at all.
- [ ] USER: apply guard-gated ledger rows — A6, A7 (scratchpad drafts from this
      morning) + M24-M27, D4 (scratchpad M24-M27_proposed_rows.md).
- [ ] Next training run (Kaggle M17 lane is green, local lane free): decide
      training mix WITH instruction data from step 0; parameter-matched twin +
      per-arm LR sweep + ≥3 seeds if the quality claim is wanted.
- [ ] Engine: lm-head 2-plane screen (M27 probe: 2.0× lm-head traffic cut,
      ≤0.52% extra MAC) is the best measured, unimplemented decode win.
- [ ] Minimal-Linux arm (Buildroot) for the paired OS-cost measurement.
- [ ] Kaggle TPU: still silently downgraded as of 2026-07-30 09:11Z
      (tpu_retry.log); leave the retry unit running.
