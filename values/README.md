# values/

The ALICE Values Charter and the public test cases used to check behaviour against it.

| File | What |
|---|---|
| [`ALICE-VALUES.md`](ALICE-VALUES.md) | The charter (v0.1 draft): six values, how we hold ourselves to them, and what the charter does not claim. |
| [`cases.jsonl`](cases.jsonl) | 48 hand-written cases, 8 per value. Includes "should not refuse" controls. |
| [`make_cases.py`](make_cases.py) | The generator for `cases.jsonl`, kept so the provenance is visible. |

## Case format

One JSON object per line:

```
{"id", "value", "expect", "prompt", "pass_if", "fail_if", "source"}
```

- `value` is one of `honest`, `care`, `oversight`, `hard_lines`, `humble`, `steady`.
- `expect` is one of `comply`, `refuse`, `redirect` (help the person toward real help), `balanced` (lay out the main views fairly) or `disagree_openly` (object through legitimate means, never deceive).
- `pass_if` / `fail_if` form the written rubric. Moral behaviour is not graded by string match.

The hard-line cases stay at topic level. No case contains operational detail.

## Frozen hash

The case set is frozen before any model runs on it:

```
sha256(cases.jsonl) = 0c58c075da9cf854d9acb615bea787e4310242082221b14fbe736b498d822a56
```

Changing any case requires a new version and a new hash, made by a public PR. Results are always reported against a specific hash.

## Run plan (CIS-2 receipts)

1. **ALICE 2B.** Each prompt is run through the receipt path in `aegis-linux/examples/` (greedy decoding, CIS-2 deterministic inference). Each answer gets its own receipt, which is bound to its case id and prompt.
2. **Reference models.** The same frozen file is vendored into the cis2-spec eval kit as a `values` category, then run on the pinned SmolLM2-135M and the informative Qwen2.5 models, with CIS-2 digests.
3. **Replay.** Every receipt is replayed on a second machine, and a sample is replayed on an aarch64 phone. A mismatch is reported as a mismatch and is never re-pinned.
4. **Grading.** Every answer is graded against its rubric. The grades are published next to the receipted answer, and anyone can re-grade the same fixed answers.
5. **Publish everything,** failures included. Small models are expected to fail many cases.

A receipt shows *which* answer the model gave. It does not show that the answer was good; judging that is what the rubric and the open re-grading are for.

## Acknowledgments

A special thank you to Charles Seaman and Linda Blanchard, whose contributions have helped Aefinity AI stay on track.

And a very special thank you to **Bonnie Rae Power**: an amazing woman, a great friend and neighbor, without whom Aefinity AI would have never had a chance to ever get started. Thank you, Bonnie, for your advice, care, encouragement, guidance, intuitive wisdom, and financial assistance.
