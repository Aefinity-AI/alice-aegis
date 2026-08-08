# Attribution — falcon-e-1b test fixtures

These two fixtures are derived from third-party data and are **not** covered by
this repository's Apache-2.0 LICENSE.

| File | Source | License |
|---|---|---|
| `t2d_sample.txt` | **WikiText-2** test split (Merity et al., Salesforce Research) — a 7,325-byte excerpt | CC BY-SA (dataset card YAML says 3.0; card body says 4.0) |
| `ref_tokens.txt` | Reference tokenization of that same excerpt | CC BY-SA (derived) |

The tokenization in `ref_tokens.txt` was produced with the Falcon-E-1B tokenizer
(`tiiuae/Falcon-E-1B`, `license: other` — the TII Falcon-LLM licence, which
carries an acceptable-use policy). Only token ids are reproduced here; no Falcon
model weights are redistributed in this repository.

CC BY-SA's Collection provision means these fixtures do not relicense the
surrounding Apache-2.0 source tree. The obligations discharged here are licence
notice and attribution.

See `THIRD_PARTY_NOTICES.md` at the repository root for the full accounting.
