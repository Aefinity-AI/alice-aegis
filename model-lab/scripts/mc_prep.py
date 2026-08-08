#!/usr/bin/env python3
"""M3 prep: ARC-Easy validation -> multiple-choice items JSONL with explicit
token ids, so the Rust engine and the transformers reference consume
byte-identical inputs (lm-eval-harness continuation convention).

Conventions (matched to G4a, docs/hardware_logs/g4a_falcon_e_1b_parity_2026-07-17.log):
  * Tokenizer: tiiuae/Falcon-E-1B-Instruct (local snapshot), plain encode,
    NO BOS prepended. Falcon-E's tokenizer defines no bos_token; G4a's
    ref_tokens.txt was verified this session to equal
    tok(text, add_special_tokens=False) exactly (2410/2410 ids).
  * ctx  = "Question: " + question + "\nAnswer:"        (lm-eval arc doc_to_text)
  * continuation k = " " + choice_text[k]
  * choice_ids[k] = enc(ctx + " " + choice_k)[len(enc(ctx)):]
    (lm-eval continuation slicing; prefix mismatches are counted and reported)
  * sample: n items, random.Random(seed).sample over row indices, then sorted
    by row index for a stable file order.

Output: items JSONL, one object per line:
  {id, answer_idx, ctx_text, choice_texts, ctx_ids, choice_ids, choice_byte_lens}
choice_byte_lens[k] = UTF-8 byte length of choice_texts[k] (WITHOUT the leading
space), the lm-eval acc_norm length-normalization denominator. Precomputed here
so both consumers use identical values.
"""
import argparse
import hashlib
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = "allenai/ai2_arc"
PARQUET = "ARC-Easy/validation-00000-of-00001.parquet"
MAX_PARQUET_BYTES = 5 * 1024 * 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/home/killboxincorporated/models/falcon-e-1b-instruct")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/home/killboxincorporated/model-lab/data/evals/arc_easy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- size-check via hub API BEFORE any download (DISK LAW) ---
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.get_paths_info(REPO, [PARQUET], repo_type="dataset")
    assert len(info) == 1, f"hub returned {len(info)} entries for {PARQUET}"
    size = info[0].size
    print(f"hub size check: {PARQUET} = {size} bytes")
    if size is None or size > MAX_PARQUET_BYTES:
        print(f"ABORT: parquet size {size} exceeds {MAX_PARQUET_BYTES} cap", file=sys.stderr)
        return 1

    local = hf_hub_download(
        REPO, PARQUET, repo_type="dataset", local_dir=args.out_dir
    )
    sha = hashlib.sha256(open(local, "rb").read()).hexdigest()
    print(f"downloaded: {local} sha256={sha}")

    import pandas as pd

    df = pd.read_parquet(local)
    n_rows = len(df)
    print(f"ARC-Easy validation rows: {n_rows}")

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(n_rows), args.n))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    items_path = os.path.join(args.out_dir, f"arc_easy_val_n{args.n}_seed{args.seed}.jsonl")
    prefix_mismatches = 0
    n_choices_hist = {}
    tok_counts = []
    with open(items_path, "w") as f:
        for row_idx in picked:
            row = df.iloc[row_idx]
            question = row["question"]
            texts = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            answer_key = row["answerKey"]
            answer_idx = labels.index(answer_key)

            ctx = f"Question: {question}\nAnswer:"
            ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            choice_ids = []
            for ch in texts:
                full = tok(ctx + " " + ch, add_special_tokens=False)["input_ids"]
                if full[: len(ctx_ids)] != ctx_ids:
                    prefix_mismatches += 1
                cont = full[len(ctx_ids):]
                assert len(cont) >= 1, f"empty continuation for {row['id']!r} choice {ch!r}"
                choice_ids.append(cont)

            n_choices_hist[len(texts)] = n_choices_hist.get(len(texts), 0) + 1
            tok_counts.append(len(ctx_ids) + sum(len(c) for c in choice_ids))
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "row_idx": int(row_idx),
                        "answer_idx": answer_idx,
                        "ctx_text": ctx,
                        "choice_texts": texts,
                        "ctx_ids": ctx_ids,
                        "choice_ids": choice_ids,
                        "choice_byte_lens": [len(t.encode("utf-8")) for t in texts],
                    }
                )
                + "\n"
            )

    note_path = os.path.join(args.out_dir, "NOTE.txt")
    with open(note_path, "w") as f:
        f.write(
            "NOTE: ARC-Easy (AI2 Reasoning Challenge, allenai/ai2_arc) is licensed CC BY-SA 4.0.\n"
            f"Source file: {PARQUET} from {REPO} (validation split ONLY), sha256={sha}, {size} bytes.\n"
            f"Items file: {os.path.basename(items_path)} = n={args.n} seed={args.seed} "
            f"random.Random(seed).sample over {n_rows} validation rows, sorted by row index.\n"
            "Tokenizer: Falcon-E-1B-Instruct local snapshot, plain encode, no BOS (G4a convention).\n"
        )

    print(f"items: {items_path}")
    print(f"note:  {note_path}")
    print(f"n_choices histogram: {n_choices_hist}")
    print(f"prefix mismatches (enc(ctx+cont) not extending enc(ctx)): {prefix_mismatches}")
    print(
        f"token load: total={sum(tok_counts)} min={min(tok_counts)} "
        f"max={max(tok_counts)} mean={sum(tok_counts)/len(tok_counts):.1f} per item (ctx+all choices)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
