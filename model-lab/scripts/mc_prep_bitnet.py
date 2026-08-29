#!/usr/bin/env python3
"""BitNet-2B pruned-vocab-aware M3 prep: ARC-Easy validation -> MC items JSONL
with explicit token ids in the PRUNED vocab id space (the space aegis-forge's
regen_vocab_embed.py built vocab.bin/embed.bin in), so the Rust engine can
score them against the same rows used for the Falcon-E baselines (ledger M16)
without re-tokenizing on the engine side.

Closes the M3/MODEL_LAB.md tripwire: "BitNet MC baseline needs pruned-vocab-
aware mc_prep."

Tokenizer: the BitNet-2B-4T checkpoint's own tokenizer (Llama-3 BPE,
vocab_size 128256), loaded from the repo-root tokenizer.json/tokenizer_config
json snapshot (the same files aegis-forge/regen_vocab_embed.py reads),
add_special_tokens=False (matches the Falcon-E convention: no BOS).

Pruned id remap (must exactly match aegis-forge/regen_vocab_embed.py):
  old_id < 50000                      -> new_id = old_id            (kept as-is)
  old_id in added_tokens (specials)   -> new_id = 50000 + k          (128000+k -> 50000+k)
  old_id in [50000, 128000) and NOT a
    special (i.e. base BPE token whose
    id is >= 50000)                   -> DROPPED FROM THE VOCAB (the pruning
                                          kept only the first 50000 base ids)

A row whose ctx or ANY choice tokenizes to an id in the dropped set cannot be
represented in the pruned-vocab model at all; such rows are excluded from the
BitNet items file and counted/reported (never silently coerced to another
id — that would corrupt the eval, not measure it). Item **ids** (not row
indices) are preserved so per-item comparison against the Falcon-E M16/M3
n=570 results (docs/hardware_logs/m3_mc_full570_falcon_e_{1b,3b}_2026-07-18.log)
remains possible on the intersection of surviving ids.
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


def build_remap(tokenizer_json_path, full_vocab=False):
    """Reproduce aegis-forge's id remap exactly, without touching
    vocab.bin/embed.bin (read-only derivation from tokenizer.json).

    full_vocab=True reproduces repack_ternary.py's full_vocab_id_space():
    the identity map over the dense [0, n) id space (no pruning at all —
    the forge built WITHOUT --llama3-prune)."""
    tk = json.load(open(tokenizer_json_path))
    base_vocab = tk["model"]["vocab"]  # str -> old_id
    added = tk.get("added_tokens", [])

    if full_vocab:
        entries = set(base_vocab.values()) | {t["id"] for t in added}
        n = max(entries) + 1
        missing = [i for i in range(n) if i not in entries]
        assert not missing, f"vocab has id gaps (first: {missing[:5]})"
        remap = {i: i for i in range(n)}
        print(f"remap: IDENTITY over full vocab space [0,{n}) (no pruning)")
        return remap, n

    base_kept = sorted(oid for oid in base_vocab.values() if oid < 50000)
    assert len(base_kept) == 50000, f"expected 50000 base tokens, got {len(base_kept)}"
    for i, oid in enumerate(base_kept):
        assert oid == i, f"base id gap at {i} (old id {oid})"

    specials = sorted(t["id"] for t in added)
    remap = {oid: oid for oid in range(50000)}
    for k, oid in enumerate(specials):
        remap[oid] = 50000 + k
    new_vocab_size = 50000 + len(specials)
    print(f"remap: {len(remap)} old ids map into pruned space [0,{new_vocab_size})")
    return remap, new_vocab_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("AEGIS_TOKENIZER_DIR", str(
            __import__("pathlib").Path(__file__).resolve().parents[2]
        )),
        help="dir containing tokenizer.json/tokenizer_config.json (default: repo root)",
    )
    ap.add_argument("--n", type=int, default=570)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-vocab", action="store_true",
                     help="target the UNPRUNED (no --llama3-prune) forge: identity id "
                          "remap, every row expected representable (asserted)")
    ap.add_argument(
        "--out-dir",
        default=str(__import__("pathlib").Path(__file__).resolve().parents[1]
                    / "data/evals/arc_easy"),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer_json_path = os.path.join(args.tokenizer, "tokenizer.json")
    remap, new_vocab_size = build_remap(tokenizer_json_path, full_vocab=args.full_vocab)

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

    local = hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=args.out_dir)
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

    suffix = "_bitnet_fullvocab" if args.full_vocab else "_bitnet"
    items_path = os.path.join(
        args.out_dir, f"arc_easy_val_n{args.n}_seed{args.seed}{suffix}.jsonl"
    )
    dropped_path = os.path.join(
        args.out_dir, f"arc_easy_val_n{args.n}_seed{args.seed}{suffix}_dropped.jsonl"
    )
    prefix_mismatches = 0
    n_kept = 0
    n_dropped = 0
    tok_counts = []
    with open(items_path, "w") as f, open(dropped_path, "w") as fdrop:
        for row_idx in picked:
            row = df.iloc[row_idx]
            question = row["question"]
            texts = list(row["choices"]["text"])
            labels = list(row["choices"]["label"])
            answer_key = row["answerKey"]
            answer_idx = labels.index(answer_key)

            ctx = f"Question: {question}\nAnswer:"
            ctx_ids_old = tok(ctx, add_special_tokens=False)["input_ids"]
            choice_ids_old = []
            bad = False
            reason = None
            for ch in texts:
                full = tok(ctx + " " + ch, add_special_tokens=False)["input_ids"]
                if full[: len(ctx_ids_old)] != ctx_ids_old:
                    prefix_mismatches += 1
                cont = full[len(ctx_ids_old):]
                if len(cont) < 1:
                    bad, reason = True, "empty continuation"
                    cont = [0]
                choice_ids_old.append(cont)

            all_old_ids = list(ctx_ids_old) + [t for c in choice_ids_old for t in c]
            unrepresentable = sorted({t for t in all_old_ids if t not in remap})
            if unrepresentable:
                bad, reason = True, f"{len(unrepresentable)} old ids not in pruned vocab: {unrepresentable[:10]}"

            if bad:
                n_dropped += 1
                fdrop.write(json.dumps({"id": row["id"], "row_idx": int(row_idx), "reason": reason}) + "\n")
                continue

            ctx_ids = [remap[t] for t in ctx_ids_old]
            choice_ids = [[remap[t] for t in c] for c in choice_ids_old]

            n_kept += 1
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

    if args.full_vocab:
        assert n_dropped == 0, (
            f"--full-vocab expects every row representable (identity remap over the dense "
            f"unpruned vocab) but {n_dropped} were dropped — investigate before trusting "
            f"the full-vocab baseline"
        )
    note_path = os.path.join(args.out_dir, "NOTE_bitnet_fullvocab.txt" if args.full_vocab else "NOTE_bitnet.txt")
    with open(note_path, "w") as f:
        f.write(
            "NOTE: ARC-Easy (AI2 Reasoning Challenge, allenai/ai2_arc) is licensed CC BY-SA 4.0.\n"
            f"Source file: {PARQUET} from {REPO} (validation split ONLY), sha256={sha}, {size} bytes.\n"
            f"Items file: {os.path.basename(items_path)} = n={args.n} seed={args.seed} "
            f"random.Random(seed).sample over {n_rows} validation rows, sorted by row index "
            f"(SAME row selection as arc_easy_val_n{args.n}_seed{args.seed}.jsonl, the Falcon-E "
            "M16/M3 items file — item ids are comparable across the two files).\n"
            "Tokenizer: BitNet-2B-4T checkpoint's own tokenizer (Llama-3 BPE, vocab_size 128256), "
            "repo-root tokenizer.json/tokenizer_config.json snapshot, plain encode, no BOS "
            "(add_special_tokens=False).\n"
            "Ids are remapped into the aegis-forge pruned vocab space (regen_vocab_embed.py): "
            "old_id<50000 unchanged, old added-special-token id 128000+k -> 50000+k; any row "
            "needing an old id in [50000,128000) that is NOT an added special token cannot be "
            f"represented in the pruned (50,256-of-128,256) vocab and was DROPPED. kept={n_kept} "
            f"dropped={n_dropped} (see {os.path.basename(dropped_path)}).\n"
        )

    print(f"items:   {items_path}")
    print(f"dropped: {dropped_path} ({n_dropped} items)")
    print(f"note:    {note_path}")
    print(f"kept={n_kept} dropped={n_dropped} of {len(picked)} picked")
    print(f"prefix mismatches (enc(ctx+cont) not extending enc(ctx)): {prefix_mismatches}")
    if tok_counts:
        print(
            f"token load: total={sum(tok_counts)} min={min(tok_counts)} "
            f"max={max(tok_counts)} mean={sum(tok_counts)/len(tok_counts):.1f} per item (ctx+all choices)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
