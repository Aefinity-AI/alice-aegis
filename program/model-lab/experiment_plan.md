# M-Series Experiment Pipeline (ordered by evidence-bought / cost)

Regime: time-rich / compute-poor. The local box owns ONE heavy job at a time (measured: 6.4GiB RAM + 6GB active swap, ~30GB free disk — the "38GB free / no swap" premise is stale). Every multi-day run is preceded by an hours-scale gate. Parallelism: M0 tonight; M1+M2 this week; M3 next; M4 and M5 in week 1-2; M6 weeks 2-4; M7 owns the box for its 5-12 day window; M8 slots into any engine-work gap; M9 gated.

## M0 — G4a engine-vs-reference PPL parity (tonight, ~15 min)
- Objective: prove the engine's ternary kernels reproduce the transformers reference on identical token IDs — the anchor for every number that follows.
- Method: ranger-venv, TORCHDYNAMO_DISABLE=1; teacher-force the exact 2410 token IDs from /home/killboxincorporated/falcon-e-artifacts/ref_tokens.txt through transformers (~7 min at measured 6.4 tok/s) and through aegis-eval; log both PPLs with commands. Feed IDENTICAL IDs to both sides — the engine tokenizer diverges (2425 vs 2410 tokens) and would silently confound the gate.
- Where: Chromebook. Wall-clock: ~15 min.
- GATE: engine PPL within 5% of transformers PPL on identical IDs.
- KILL: >5% divergence -> freeze all ports and training; kernel bug hunt first.
- Unblocks: M1, M3, M4, M6, M7 — every engine-side eval and every "gains survive the port" check.

## M1 — Falcon-E-3B-Instruct zero-training port (deliverable c, fleet tier)
- Objective: highest-capability fleet-wide checkpoint at zero training; independently re-measure TII's self-reported superiority. DARPA clause: capability uplift on every fleet tier, including the 2GB pre-AVX2 boxes, with no training pipeline required.
- Method: download main packed revision (~1.0GB); also pull bfloat16 + prequantized masters (~6GB) and mirror to the external drive (availability not permanent). Repack via aegis-forge/repack_ternary.py (streams tensor-by-tensor, proven on the 1B, identical packing family). Boot in engine; measure WT2 sample PPL, tok/s (expect ~2.6 single-thread — unbenchmarked scaling estimate), and peak RSS; verify 2GB-tier fit at 2048 ctx on real hardware (~1.42GB computed, tight against unikernel overhead).
- Where: Chromebook. Wall-clock: 1-2 days.
- GATE: coherent chat output + honest WT2 PPL logged + measured RSS fits the 2GB tier at 2048 ctx.
- KILL: engine WT2/coherence clearly worse than Falcon-E-1B (superiority evidence was vendor-single-source from two different tables; the only head-to-head is 53.17 vs 51.54) -> keep as dev-box option; BitNet-2B stays canonical.
- Unblocks: M3 baselines, the fleet deployment story, and the capability bar M6 must clear.

## M2 — SFT corpus assembly + provenance ledger
- Objective: DARPA-clean training data for M5/M6.
- Method: download HuggingFaceTB/smoltalk AND smoltalk2 (successor, current SOTA mix), allenai/tulu-3-sft-mixture (drop the CC-BY-NC No Robots subset), optional 2-4GB Cosmopedia-v2 slice (<10GB total vs ~30GB free). Filter per the SmolLM2 sub-1B recipe (remove function-calling/hard examples). Re-tokenize to the target tokenizer (Falcon-E vocab 32,768 — counts shift 15-25%; size the mix AFTER re-tokenization). Build a 300-500M-token mix at ~70% domain/persona + ~30% general-instruct replay. Write a per-subset license ledger: SmolTalk is NOT blanket Apache-2.0 (only the 4 new synthetic subsets are; the OpenHermes-2.5 slice declares no license — disclose, don't hide).
- Where: Chromebook. Wall-clock: 2-4h downloads + hours of CPU filtering, inside this week.
- GATE: mix built, post-retokenization token counts recorded, provenance appendix complete.
- KILL: none (pure download); any NC/unlicensed subset that can't be cleanly dropped is excluded, never waived.
- Unblocks: M5 smokes and M6.

## M3 — Engine MC eval harness + tripwire baselines (deliverable d)
- Objective: score shipped artifacts on the bare-metal stack itself. DARPA clause: every reported number traceable to the program's own harness — no vendor self-reports.
- Method: refactor aegis-core inference.rs:457 calculate_perplexity's loop into continuation_nll(ctx, cont) (~60 LOC); add an --mc JSONL mode to aegis-eval (~100 LOC, format {context, choices[], gold}); export tinyMMLU/tinyArc/tinyHellaswag/tinyWinogrande to JSONL once. Fresh --system-site-packages venv for lm-eval (never pip-install into ranger-venv). Baseline BitNet-2B, Falcon-E-1B, and post-M1 Falcon-E-3B on both transformers side (13-22h overnight at 6.4 tok/s teacher-forced) and engine side. Adopt tripwires: per-checkpoint fixed 1900-token WT2 anchor + tinyArc + tinyWinogrande (~2-3h); generation evals (IFEval-541 ~12.5h, tinyGSM8k ~1.5h) ONLY via the engine at 4.8 tok/s — transformers decode is ~0.25 tok/s (verifier re-measured; the 0.34 figure had no log), i.e. a ~240h trap.
- Where: Chromebook. Wall-clock: 2-3 days engineering + overnight baseline runs.
- GATE: engine MC scores match transformers-side within binomial noise (±5 raw pts at n=100) on identical items.
- KILL: systematic engine-vs-transformers divergence -> fix before trusting any training number.
- Unblocks: M4c, M6 tripwires, M7 quality checks, all milestone reporting.

## M4 — Soft-prompt over frozen ternary (SALVAGED path; original claim killed 2/3 by refuters)
- Objective: cheapest fully-local specialization lever, plus a novel Compress-Then-Prompt-style quality-recovery test on a ternary unikernel. DARPA clause: field-adaptable behavior without touching weights or leaving the device.
- Method (per corrected claims — the stock recipe is WRONG for Falcon-E): M4a parity gate (hours): patch packed BitLinear's activation path with ActQuant STE (preserves the checkpoint's divide-convention weight_scale) OR invert all 168 weight_scale tensors if using AutoBitLinear(offline); TORCHDYNAMO_DISABLE=1; custom training loop (HF's quantizer flags offline mode not-trainable); verify step-0 logit equivalence vs the stock packed forward (~1e-5 rel err class). M4b smoke (2-3 days): freeze everything, train k=32 soft-prompt vectors (Adam, lr ~0.3) on a persona/format task, 0.4-1M token-passes at measured ~2.9-4.3 tok/s. M4c engine port (1 day): append 32 rows to EMBED.BIN + reserved pseudo-token ids at the exact trained template position; re-eval on engine.
- Where: Chromebook, box idle (no concurrent PPL jobs). Wall-clock: ~3-5 days total.
- GATE: M4a logits match; M4b beats BOTH the k=0 baseline AND an equal-token-budget hard prompt on the same ternary forward; M4c gains survive the engine port.
- KILL: any stage fails -> path dead (no independent replication exists anywhere; we are first, so evidence discipline is maximal). Scope claims to persona/format/routing — no new knowledge at 1.8B per Lester.
- Unblocks: a zero-cloud specialization demo independent of M5/M6.

## M5 — Cloud QAT gate battery (selects the M6 vehicle)
- Objective: burn down the two unproven steps blocking free-GPU training: falcon-sft's Triton/Turing environment pin and bitnet-sft's fp16-QAT-on-T4 stability.
- Method:
  - M5a (local, $0 GPU, run FIRST): quantize_to_1bit on the UNTOUCHED Falcon-E-1B prequantized revision on CPU; diff tensors against the main revision; then repack_ternary.py -> engine boot. Proves the export->ingest leg no third party has ever demonstrated.
  - M5b (bitnet-sft smoke): Kaggle 2xT4, plain transformers+trl on microsoft/bitnet-b1.58-2B-4T-bf16 (online QAT, code-verified at three layers); 100 SFT steps, lr ~1e-4 (10x normal — the documented community failure used 5e-7), fp16 AMP + fp32 master, grad checkpointing, 8-bit Adam; budget is tight ~14.5GB/GPU with optimizer CPU offload as fallback. Snap to ternary, pull, repack, engine PPL.
  - M5c (falcon-sft smoke): Kaggle T4, pin torch<=2.6 (bundles triton 3.2, last Turing-supporting release) + onebitllms 0.0.4 + TRL pinned around issue #10; 100 steps on Falcon-E-1B-Instruct prequantized, fp32 first, then test bf16-emulation.
- Where: Chromebook (M5a) + Kaggle (M5b/c). Wall-clock: M5a hours; M5b/c ~1-2 GPU-h each; one calendar week.
- GATE: loss decreases with no NaNs AND the exported ternary artifact repacks, boots, and shows sane engine PPL. Whichever of M5b/M5c passes (prefer better engine PPL; bitnet-sft targets the canonical model, falcon-sft the vendor toolchain) becomes the M6 vehicle.
- KILL: M5c dies if the environment pin cannot produce working Triton-on-T4 (refuter-predicted failure). M5b dies if fp16 QAT at 1e-4 is unstable and fp32 fallback doesn't fit. BOTH dying -> 1B specialization degrades to M4 soft-prompts + M7 tiny model; the local-CPU full-FT fallback stays parked.
- Unblocks: M6.

## M6 — 1-2B instruct specialization (deliverable a)
- Objective: the fleet's specialized instruct model trained on free compute. DARPA clause: supply-chain-independent model adaptation on zero-cost infrastructure with checkpoint-resume discipline.
- Method: winning M5 vehicle; 1-5M tokens of the M2 mix (70/30 domain/replay), 1-2 epochs, full-parameter QAT (LoRA unsupported/risky on ternary); checkpoints pushed to HF hub every ~2h for 12h-session resume via "Save & Run All" background execution inside the ~30 GPU-h/week quota. After training: snap -> repack -> engine; run the FULL M3 battery before/after; final numbers come from the ENGINE artifact only.
- Where: Kaggle. Wall-clock: 2-10 GPU-h compute; 1-2 calendar weeks including eval.
- GATE: M3 tripwires hold (WT2 anchor <+20% vs step-0; tiny tasks within 5 raw pts across two consecutive checkpoints); domain evals improve on the engine artifact; instruct retention (engine-side IFEval-class) within tolerance of base.
- KILL: forgetting beyond thresholds on two consecutive checkpoints -> rollback, halve LR/epochs; second failure kills the run and the negative result is recorded as a deliverable.

## M7 — From-scratch tiny ternary model (deliverable b — the sovereignty demo)
- Objective: sub-50M native-ternary storyteller trained ENTIRELY on the Chromebook and booted bare-metal on the 2GB pre-AVX2 tier. DARPA clause: full-stack sovereignty — own data -> own QAT -> own engine -> bare metal, no OS; "first" claim scoped to sub-50M + bare-metal-no-OS (TernaryLM holds 132M).
- Method: 10-15M params, llama-family config matching the engine (d 256-320, L 8-10, SwiGLU BitLinear, SubLN, fp embeddings/tied head, ctx 512-1024), custom 4-8k BPE. Budget 2x hidden width AS A FLOOR — the fp16-parity framing was refuted; target quality of a <=half-size fp16 twin. ~300-500M TinyStories tokens at MEASURED 733 tok/s (8.5M) to 458 tok/s (21M). Two-stage LR + weight-decay-to-zero (recipe is in microsoft/unilm's bitnet dir, not microsoft/BitNet). MANDATORY step-0 round-trip export test (trainer absmean scales -> repack_ternary.py -> logit match) before the multi-day run — the same scale-convention bug class has now bitten twice.
- Where: Chromebook, unattended (systemd-run --user + linger; power settings fixed first; box owned — the 47M bench already showed a memory-pressure throughput cliff).
- Wall-clock: 5-12 days.
- GATE: coherent TinyStories-class generation; on-corpus PPL tracks the half-size-fp16 expectation; packed artifact (~3-8MB) boots on real 2GB pre-AVX2 hardware.
- KILL: PPL anchor stalls/diverges mid-run, or final quality below usable coherence at full budget -> record the negative, retry ONCE at 2x width / smaller vocab, then stop.

## M8 — Llama3-8B-1.58 dev-box port (deliverable c, scale headline)
- Objective: largest drop-in ternary checkpoint on the dev box. DARPA clause: 8B-class ternary inference on commodity CPU bare-metal, $0 software stack.
- Method: checkpoint already on the external drive; land f16 embedding storage OR vocab pruning first (f32 embed+head = 4.2GB busts the 6.5GB VM; f16 -> ~4.9GB total); repack (per-tensor BitNet packing already supported). Position as scale demo — modest quality expected per settled PTQ-recovery findings.
- Where: Chromebook (dev box only). Wall-clock: 1-3 days engine work + repack/eval.
- GATE: boots in 6.5GB; honest WT2 PPL + tok/s logged.
- KILL: cannot fit even with f16 embeddings + pruning -> PARK, do not force.

## M9 — Gated scale-ups (only after M1-M8 disposition)
- M9a Track B: 90-125M ternary pretrain on Kaggle TPU v5e-8 (~1.9e18 FLOPs = 5-17 TPU-h at conservative MFU; 1-3 weeks of quota). GATE: a 1h TPU smoke proving an XLA/JAX QAT loop + checkpoint-resume, and the quota figure confirmed on the account page. KILL: v5e-8 availability droughts (user-reported) persist or the XLA port balloons. Payoff: upgrades deliverable (b) to an instruction-flavored generalist still 2GB-fleet-compatible (~31MB packed).
- M9b Bonsai-1.7B trial port, then Bonsai-8B (the 2026 ternary-native SOTA bar): exactly two bounded kernel adds (group-128 fp16 scales, Qwen3 per-head QK-norm). GATE: 1.7B trial port coherent. KILL: kernel scope exceeds the two additions (hybrid-attention variants stay out of scope). Verify 8B RSS (~5.4-5.9GB, marginal) before promising it.
- M9c TRC lane (if the user approves the ~$5-15/mo GCP exception): the only free lane for >125M or 10B-token-class training; requires preemption-tolerant GCS checkpointing from day one.