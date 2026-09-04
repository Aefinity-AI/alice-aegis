# E-S1 — Trit census of the shipped BitNet-b1.58-2B engine model (2026-09-04)

**Question.** How many bits per weight does the deployed ternary model actually need, losslessly?
**Machine.** aefinity-box (Dell, i5-5200U), leg `es1-census`, `scripts/trit_census.py` (this repo, merged 2026-09-04) on
`aegis_pruned_model.safetensors` (sha256 53586af0856e141e…; 210 body tensors, 2,084,044,800 weights). Byte-exact round-trip on every tensor: **0 mismatches**.
Counts and byte sizes only (Rule A); raw CSV/JSON in `docs/experiments/es1/`.

## Whole-model result
| storage | bits/weight | bytes |
|---|---|---|
| engine packing today (2-bit codes, 4 trits/byte) | 2.000 | 521,011,200 |
| ideal 5-trit-per-byte packing (reference) | 1.600 | 416,808,960 |
| rANS, order-0 adaptive, per tensor | 1.558 | 405,910,480 |
| rANS, order-1 (previous trit in row) | 1.551 | 404,107,430 |
| xz -9 on packed bytes | 1.547 | 403,081,744 |
| **zstd -19 on packed bytes** | **1.532** | **399,113,010** |

Whole-body trit distribution: p(0) ≈ 0.42, p(±1) ≈ 0.29 each; zeroth-order entropy H0 ≈ 1.56 bits/weight. Per-tensor p(0) ranges 0.361–0.601.

## By projection type (weight-averaged)
| type | weights | p(0) | H0 | rANS-1 | zstd-19 |
|---|---|---|---|---|---|
| q_proj | 196,608,000 | 0.4675 | 1.5275 | 1.5279 | 1.5208 |
| k_proj | 49,152,000 | 0.4465 | 1.5430 | 1.5549 | 1.5408 |
| v_proj | 49,152,000 | 0.3764 | 1.5780 | 1.5944 | 1.5702 |
| o_proj | 196,608,000 | 0.4184 | 1.5595 | 1.5618 | 1.5608 |
| gate_proj | 530,841,600 | 0.4236 | 1.5551 | 1.5447 | **1.5078** |
| up_proj | 530,841,600 | 0.4161 | 1.5586 | 1.5481 | **1.5109** |
| down_proj | 530,841,600 | 0.4131 | 1.5598 | 1.5614 | 1.5667 |

## Reading
1. **The trits are near-uniform.** A lossless container saves 23 % over the engine's current file only because that file spends 2 bits per trit; against ideal trit packing the saving is ~4 %. Pretrained ternary weights do not compress much: the plan's prediction (H0 1.45–1.55, ≤10 % vs packing) holds.
2. **Generic zstd beats a one-trit-context coder on gate/up projections** (1.508 vs 1.545): there is structure beyond adjacent-trit context — plausibly rows/columns with atypical zero fractions. Worth ~1 % overall; a row/column-adaptive model is the only remaining lossless headroom.
3. **Nothing here approaches sub-integer bits per weight.** That requires changing the model (sparsity), which is what E-S2/E-S2b (post-hoc) and E-S4c (train-time) measure. Lossless coding is a container decision, not a modelling result; because it decodes to identical trits it leaves every CIS digest unchanged.

Provenance: `docs/experiments/es1/RESULT.txt` (leg header names host + commit), `bitnet2b_census.csv` (per tensor), `bitnet2b_summary.json`, row/column zero-fraction histograms. Plan and predictions: claudius-maximus `state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md` §1.
