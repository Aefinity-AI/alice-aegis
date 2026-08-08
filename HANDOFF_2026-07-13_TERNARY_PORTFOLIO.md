# HANDOFF — 2026-07-13 — Ternary Model Portfolio (remote session)

> **CORRECTION (2026-07-16, post-merge):** where this document says
> "flagless `aegis-eval` is the single-sample pin again" it describes the
> patch-side evaluator, which LOST the merge (750d377 kept HEAD's). On the
> merged binary, flagless = chunked full-text; the anchor pin is `--sample`.
> `scripts/phase0_meter_check.sh` is updated to match; anchor is 10.348.

Branch: `ternary-portfolio` (5 commits on top of `6f29c6b`). Produced in a
remote container with **no model weights and no GitHub remote** — delivered
as a git bundle + patch. Every number below is a measured output from this
session; weight-dependent gates are explicitly marked NOT RUN.

## The reframe this branch implements

Dense→ternary "transmutation" is dead (15 months, zero coherent outputs;
PTQ ternarization collapses modern checkpoints per the verified research
sweep; even TII says conversion "often leads to poor performance"). The
viable project: **generalize aegis-core to run the existing ternary
checkpoint families** — Falcon-E-1B/3B (native ternary) and HF1BitLLM
Llama3-8B-1.58 (100B-token fine-tune) — which share three engine deltas:
SwiGLU, optional SubLN, untied LM head. Target roster: BitNet-2B
(canonical) + Falcon-E-1B (small/fast) + optionally Llama3-8B-1.58
(quality tier, measure first).

Also assessed and dispositioned: the "A.S.P.E.C.T." proposal. Its GPU
permutation search cannot cluster zeros within rows (wrong axis; expected
gain ~0) and its bit-serial SWAR is strictly dominated (~30-60 ops per 32
MACs vs 3 for maddubs; no bandwidth win over the existing 2-bit packing) —
both rejected with math. Its `maddubs` int8 kernel is real (standard AVX2
idiom; ternary weights make int16 saturation impossible; ops.rs:100 already
reserved the symmetric ±127 grid for it) — **parked** behind Phases 0-4
with a bench-first gate. None of it is a transmuter.

## What changed (by commit)

1. `658e44d` — Engine generalization + meter repair. Config-driven silu/
   relu2 dispatch, optional SubLN, untied BF16 lm_head, rope_theta and
   rms_norm_eps from config; config travels in MODEL.SAF safetensors
   `__metadata__["aegis_config"]` (baked BitNet config as the legacy
   fallback); `aegis-eval` chunked mode + `KVCache::reset`; test suites
   (graph_variants, reference_parity, forge_artifacts harnesses).
2. `e1ca533` — Forge repack pipeline. `repack_ternary.py` (already-ternary
   checkpoints → MODEL.SAF/EMBED.BIN/VOCAB.BIN; refuses non-ternary
   weights, biases, group scales, unknown activations; clamps the sequence
   window), self-tests, Phase-0/fixture gate scripts.
3. `d782c00` — Weights-free QEMU gauntlet (`scripts/qemu_synth_gauntlet.sh`
   + `gen_synth_checkpoint.py`): boots the unikernel on a synthetic
   silu/untied model, asserts the isa-debug-exit success signal.
4. `b5655f2` — **Review fixes** (8-angle adversarial review, verified
   findings). Headline: the hf1bitllm repack assumed `[out, in/4]`
   input-dim packing; the format is `[out/4, in]` packed along dim 0 in
   row blocks (pinned by this repo's own `correct_transmute.py`). The
   original byte-relabel would have silently scrambled every
   Llama3-8B-1.58 weight while passing every size check. Also: malformed
   metadata configs are load errors (not silent BitNet fallback); KV/arena
   window clamps (was a bare-metal fault path); flagless `aegis-eval` is
   the single-sample pin again (documented reproduction commands stay
   true; corpus mode = `--chunked`); SubLN required for relu2 artifacts;
   mispaired-artifact checks hardened to release builds; fixture dump
   excludes transformers' post-norm final entry; prefix-proportional KV
   reset; vectorized forge paths.
5. `HEAD` — one shared safetensors writer across forge/generator/tests;
   this handoff.

## Gates run HERE (synthetic artifacts, real machinery)

- `cargo test` aegis-core: 24 pass (+ ignored env-gated), single-threaded
  per gemm_equivalence's documented toggle race; also green with
  `--no-default-features`, `--features parallel`, `--features legacy_matmul`.
- aegis-eval: 3 unit tests (fold math incl. NaN poisoning); both modes run
  against a python-forged synthetic model; sample vs chunked agree exactly
  on identical input (11153.440 == 11153.440, random-weight model — the
  number is meaningless, the equality is the point).
- Forge self-tests: 33 checks incl. exhaustive 81-block pack round-trip,
  dim-0 layout unpack, numpy==pure equivalence on every fast path.
- Cross-language: python-forged artifacts load in the Rust engine,
  `prefill_decode_parity() == 0.0`; aegis-linux generation + `--parity`
  bit-identical (`0e0`).
- QEMU/OVMF boot (TCG): metadata config parsed under no_std, 43-token
  prefill + 11 generated tokens through the silu/untied graph, exit 33.
- `scripts/integrity_check.py .` clean.

## Gates NOT run (need weights / your machine) — the runbook

1. **T0a/T0b first** (everything waits on the meter):
   `scripts/phase0_meter_check.sh MODEL.SAF EMBED.BIN VOCAB.BIN sample.txt`
   → must reproduce 12.80 ± 0.01. Then full WikiText-2 via `--chunked`.
   If T0a/T0b pass but `~/test.txt` still reads ~175, the file contents or
   vocab pairing is the culprit — not the meter.
2. Falcon-E-1B: reference PPL + probes on Colab (G1 kill gate: PPL < 20);
   `scripts/dump_reference_fixtures.py` for token/hidden fixtures;
   `onebitllms` unpack → `repack_ternary.py --source-packing unpacked`;
   then `forge_artifacts` → `t2d` (tokenizer risk item — see below) →
   `t2b` → G4a (engine PPL within 5% of reference) → QEMU → iron via
   `collect_gauntlet.sh`.
3. Llama3-8B-1.58: measure first (its 12.2-PPL claim was refuted 0-3);
   port only on G1 pass, `--source-packing hf1bitllm --llama3-prune`.
4. tzervas-14B: one measurement to reject with evidence, or ledger-close
   as untested/presumed-non-viable per PT2-LLM.

## Known open items (deliberate, documented)

- ~~**tokenizer.rs merge guard drops id-0 merges**~~ **FIXED 2026-07-16**
  together with whitespace-run pre-tokenization, exactly as this runbook
  prescribed (one change, T0a + T2d re-run): T2d vs Falcon-E reference now
  EXACT (2410/2410 tokens); T0a re-pinned 10.348 → 10.758
  (wikitext2_sample1900_2026-07-16_repin.log; per-token PPL rise is a
  tokenization-density artifact, documented in the log and meter script).
- hf1bitllm layout is implemented per the vendored `correct_transmute.py`;
  first real-weight run should still confirm via T2b layer parity.
- `forward_batch`/`forward_step` remain duplicated graphs; parity tests
  police the duplication. Unification is a separate, riskier change.
- maddubs int8 kernel: parked post-G4c; bench first (`lut_mpgemm`,
  `gemm_tile`), iron numbers second, coherence gate third.
- Repo push: historical note — at the time of writing no GitHub remote existed.
  The repository is now published as a scrubbed snapshot built by
  `scripts/build-public-snapshot.sh`;
  pushing this branch and enabling repo access for remote sessions makes
  future work land as PRs instead of bundles.
