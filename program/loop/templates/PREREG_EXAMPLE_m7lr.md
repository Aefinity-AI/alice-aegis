# M7 LR-FLOOR CONTROL — PRE-REGISTRATION

**Written 2026-07-29, BEFORE arm H exists and BEFORE any contrast is computed.**
Committed to git ahead of the result so the conclusion cannot be chosen after
seeing the data. Arm K was already running when this was written; its contrast
partner (arm H) did not exist and no d_cool value had been computed.

Produced after an 8-agent adversarial review of the original single-arm design
(run `wf_ca8a9e4d-a00`). All four attack lenses returned **RUN_WITH_FIXES** with
`would_result_be_misleading = True`. This document is the fix.

---

## 1. The question

Ledger M20 says the M7 ternary run differed from the fp32 twin in "one variable
(linear=bitlinear)". An audit refuted that: **seven** knobs differ. The largest
*undisclosed* one is the end-of-training learning-rate floor.

`train.py:169` calls `lr_schedule_cosine()` without `min_lr_mult`, so it
defaults to `0.1`:

| | schedule | final LR | final wd |
|---|---|---|---|
| fp twin | `cosine` | **1.00e-04** | 0.1 (held) |
| ternary | `two_stage` | **1.32e-13** | 0.0 (dropped at knee) |

Annealing to ~0 buys perplexity on its own. **How much of the published gap is
that alone?**

## 2. Arms

All arms start from the same published weights. Arms H and K draw an
**identical batch sequence**: `train.py:156` and `m7_lr_cooldown.py:167` both
build `np.random.default_rng(1337 + 122071)` = `default_rng(123408)`, same
B=8, T=512, same `train_8k.bin`.

| arm | what | LR | wd |
|---|---|---|---|
| **A** | published fp twin, untouched | — | — |
| **H** | A + 4000 steps at the twin's ORIGINAL recipe (**the null**) | 1.02e-04 → 1.00e-04 | 0.1 |
| **K** | A + 4000 steps annealing (**the treatment**) | 1.000e-04 → 1.54e-11 | 0.0 |
| **T** | published ternary, untouched | — | — |

`d_cool  = NLL_H − NLL_K` — the cooldown effect, **with extra tokens differenced out**
`d_tokens = NLL_A − NLL_H` — prices the 16,384,000 extra tokens alone
`d_gap   = NLL_A − NLL_T` — the published ternary advantage
`d_resid = NLL_K − NLL_T` — what remains after cooling the twin

The original single-arm design lacked arm H and therefore **conflated the
cooldown with the extra tokens**. That was the flaw the review caught.

## 3. Primary estimand

**R = mean(d_cool) / mean(d_gap)** — the fraction of the published gap that the
undisclosed cooldown alone reproduces in the fp twin.

`d_gap` is already measured and is **locked** at
**+0.07294 nats/token** (95% CI +0.06943 … +0.07645), from 400 disjoint windows
/ 204,800 predictions — `m7_paired_eval_baseline_2026-07-29.log`, ternary
favoured on 396/400 windows, t = +40.69. Corpus PPL: twin 4.4816, ternary
4.1664 (7.03%).

## 4. Locked evaluation protocol

Changing any of these after seeing a result is a forking-paths violation.

- 512 disjoint 512-token windows on `valid_8k.bin`, deterministic stride grid, no RNG
- unit of analysis = the window; statistic = per-window mean NLL in nats/token
- paired sign-flip permutation test (20,000 draws) + percentile bootstrap (10,000 resamples)
- Holm correction across the four contrasts
- 4 threads, single process, all arms scored in the same run

## 5. Bands — fixed in advance

| Band | Condition | Conclusion |
|---|---|---|
| **1 — cooldown-dominant** | R ≥ 0.70, CI low > 0.50 | The gap is largely an artifact of schedule asymmetry. **Retract** the M7 headline, do not merely caveat it. |
| **2 — cooldown-major** | 0.35 ≤ R < 0.70, CI low > 0.15 | Cooldown is the single largest identified driver. Headline unsupportable as stated. Report the full decomposition; do **not** re-report the residual as a ternary win. |
| **3 — material minority** | 0.15 ≤ R < 0.35 | Cooldown real but not the main story. Headline still invalid; dominant suspect shifts to the 2.17× parameter gap. |
| **4 — negligible** | R < 0.15, CI high < 0.25 | The audit's own hypothesis is **not supported**. This is a negative result about the audit and is publishable as such. Headline still invalid for the unchanged parameter reason. |
| **5 — harmful** | mean(d_cool) ≤ 0 | Cooling did nothing or hurt. Same conclusion as Band 4. Do not reinterpret the sign. |
| **6 — reversal** | R > 1.0, CI low > 1.0 | The cooled 6.53M twin **beats** the 14.17M ternary. Strongest retraction case; supersedes all bands. |

**Significance gate (all bands):** if Holm-adjusted sign-flip p for d_cool > 0.05,
report R as "not significantly different from zero" and do not headline the
point estimate.

**Internal validity check:** if |mean(d_tokens)| > 0.30 × mean(d_gap), the extra
tokens are doing substantial work; R remains correct but the token contribution
must be reported with equal prominence.

**Superseding check:** if the 512-window CI on d_gap spans zero, the published
result does not survive multi-window evaluation at all, and that finding
supersedes the entire decomposition.

**Sanity aborts:** arm H final LR must be ~1.0e-04 with wd 0.100; arm K final LR
~1.5e-11 with wd 0.000; both must reach step 126071. If any fails, the arms were
misconfigured and **no band applies**.

## 6. d_cool is a LOWER BOUND — stated numerically, in advance

From `logs/m7_ternary_full_2026-07-21.log`: the ternary ran below LR 1e-4 for its
final **~30,517 steps (25% of training, ~125M tokens)** and had wd = 0 for its
final **61,036 steps (50%)**. Arm K receives **4,000 steps (3.3%)**.

Arm K therefore measures a **fraction** of the real schedule difference. Any R
we obtain **understates** the true confound, and that bias points toward
exonerating the claim under audit. This must appear beside every reported R.

## 7. Disclosed deviations

1. Arms H and K receive 16,384,000 tokens (+3.3%) more than A and T. Inherent to
   any cooldown; **priced** by arm H, not eliminated.
2. The resumed segment is not a continuation of the original data cursor — no
   cursor is stored, the sampler is re-seeded. i.i.d. sampling over the same
   corpus. Disclosed, not silent.
3. Torch-side eval only: `export_hf.py` hard-refuses fp checkpoints, so fp arms
   can never be engine-scored. Cite the measured 0.19% torch↔engine agreement
   (`m7_final_roundtrip.log`) rather than re-deriving it.
4. Absolute PPLs here (~3.9–4.5) are **not** comparable to the published
   5.513/5.140, which came from one unrepresentative 511-token window. Only
   within-run contrasts are meaningful.

## 8. Banned sentences

These strings are prohibited in any ledger entry for this run, whatever the
result:

- "ternary win survives"
- "residual ternary advantage"
- "confound ruled out" / "confound was minor"
- "at equal budget"
- any use of `d_resid` as a standalone result

**No outcome of this run validates, rescues, restates, or partially restates
"ternary beat fp32 at equal budget."** The ternary arm has 2.17× the parameters
(14,171,392 vs 6,529,920). A larger model beating a smaller one is the expected
result and is not evidence about quantization.

Six confounds remain untouched by this experiment: params 2.17×, layers 6 vs 7,
hidden 256 vs 384, inter 704 vs 1024, heads 4 vs 6, peak LR 1e-3 vs 2e-3.

## 9. The only licensed claim template

> "R (95% CI lo–hi) of the published M7 gap is reproducible in the fp twin by
> the undisclosed LR/WD cooldown alone — a lower bound, since arm K anneals for
> 3.3% of training where the ternary annealed for 25%. The residual gap remains
> confounded by six further variables, principally a 2.17× parameter difference.
> Settling the quantization question requires a same-size ternary/fp pair
> (~66 h per arm), which this experiment is not."

---

## 10. Machine-checkable form of §3–§8 (added by ev; the prose above is unchanged)

Nothing in this block is a new commitment. Every band, abort, gate and banned
sentence below is transcribed from §5, §8 and the significance gate above. The
point is that a script can now evaluate them, so the band cannot be chosen after
the fact and the banned sentences cannot be written by accident.

```ev-prereg
{
  "id": "m7lr",
  "estimand": "R = mean(d_cool) / mean(d_gap)",
  "bands": [
    {
      "name": "1 cooldown-dominant",
      "when": "R >= 0.70 and R_ci_lo > 0.50",
      "conclusion": "The gap is largely an artifact of schedule asymmetry. RETRACT the M7 headline, do not merely caveat it."
    },
    {
      "name": "2 cooldown-major",
      "when": "0.35 <= R < 0.70 and R_ci_lo > 0.15",
      "conclusion": "Cooldown is the single largest identified driver. Headline unsupportable as stated. Report the full decomposition; do NOT re-report the residual as a ternary win."
    },
    {
      "name": "3 material minority",
      "when": "0.15 <= R < 0.35",
      "conclusion": "Cooldown real but not the main story. Headline still invalid; dominant suspect shifts to the 2.17x parameter gap."
    },
    {
      "name": "4 negligible",
      "when": "R < 0.15 and R_ci_hi < 0.25 and mean_d_cool > 0",
      "conclusion": "The audit's own hypothesis is NOT supported. Negative result about the audit, publishable as such. Headline still invalid for the unchanged parameter reason."
    },
    {
      "name": "5 harmful",
      "when": "mean_d_cool <= 0",
      "conclusion": "Cooling did nothing or hurt. Same conclusion as band 4. Do not reinterpret the sign."
    },
    {
      "name": "6 reversal",
      "when": "R > 1.0 and R_ci_lo > 1.0",
      "conclusion": "The cooled 6.53M twin BEATS the 14.17M ternary. Strongest retraction case; supersedes all bands."
    }
  ],
  "sanity_aborts": [
    {
      "when": "abs(arm_H_final_lr - 0.000100) > 0.000005",
      "message": "arm H final LR is not ~1.0e-04 (sec 5) - arms misconfigured, NO BAND APPLIES"
    },
    {
      "when": "arm_H_final_wd != 0.1",
      "message": "arm H weight decay is not 0.100 - arms misconfigured"
    },
    {
      "when": "arm_K_final_wd != 0.0",
      "message": "arm K weight decay is not 0.000 - arms misconfigured"
    },
    {
      "when": "arm_H_steps != 126071",
      "message": "arm H did not reach step 126071"
    },
    {
      "when": "arm_K_steps != 126071",
      "message": "arm K did not reach step 126071"
    },
    {
      "when": "d_gap_ci_lo <= 0 and d_gap_ci_hi >= 0",
      "message": "the 512-window CI on d_gap spans zero: the published result does not survive multi-window evaluation at all, and THAT finding supersedes the entire decomposition"
    }
  ],
  "significance_gate": {
    "when": "p_d_cool_holm > 0.05",
    "message": "report R as 'not significantly different from zero'; do NOT headline the point estimate"
  },
  "internal_validity": [
    {
      "when": "abs(mean_d_tokens) > 0.30 * mean_d_gap",
      "message": "the extra 16,384,000 tokens are doing substantial work; R remains correct but the token contribution must be reported with equal prominence"
    },
    {
      "when": "n_windows < 512",
      "message": "fewer than the pre-registered 512 disjoint windows were scored"
    }
  ],
  "banned_sentences": [
    "ternary win survives",
    "residual ternary advantage",
    "confound ruled out",
    "confound was minor",
    "at equal budget"
  ],
  "licensed_template": "R (95% CI lo-hi) of the published M7 gap is reproducible in the fp twin by the undisclosed LR/WD cooldown alone - a LOWER BOUND, since arm K anneals for 3.3% of training where the ternary annealed for 25%. The residual gap remains confounded by six further variables, principally a 2.17x parameter difference. Settling the quantization question requires a same-size ternary/fp pair (~66 h per arm), which this experiment is not."
}
```
