#!/usr/bin/env python3
"""M2 corpus statistics: enumerate subsets in local parquets, row counts,
sampled dual-tokenizer token estimates (Falcon-E 32,768 / Llama-3 128,256).
Sampling is honest-labeled: estimates carry sample size; ledger rows must say
'sampled estimate'. Single-thread, run nice'd — the twin owns the box."""
import glob, json, os, random, sys, time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pyarrow.parquet as pq
from tokenizers import Tokenizer

H = "/home/killboxincorporated"
OUT = f"{H}/model-lab/data/m2_corpus_stats.json"
SAMPLE_PER_SUBSET = 400
random.seed(42)

tok_falcon = Tokenizer.from_file(f"{H}/models/falcon-e-1b-instruct/tokenizer.json")  # 32,768; 1B/3B share it
tok_llama = Tokenizer.from_file(glob.glob(
    f"{H}/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/tokenizer.json")[0])  # 128,256
print(f"tokenizers: falcon vocab {tok_falcon.get_vocab_size()}, llama vocab {tok_llama.get_vocab_size()}", flush=True)

def row_text(row):
    msgs = row.get("messages")
    if msgs is not None and not isinstance(msgs, float):
        try:
            return "\n".join(m.get("content", "") or "" for m in msgs)
        except Exception:
            return str(msgs)
    return str(row.get("text", ""))

def scan(name, parquet_glob, group_col=None):
    """Group rows by subset (parent dir name, or group_col value)."""
    stats = {}
    files = sorted(glob.glob(parquet_glob, recursive=True))
    print(f"[{name}] {len(files)} parquet files", flush=True)
    for fp in files:
        subset_from_path = os.path.basename(os.path.dirname(fp))
        pf = pq.ParquetFile(fp)
        n = pf.metadata.num_rows
        cols = [c for c in ("messages", "text", group_col) if c and c in pf.schema_arrow.names]
        if group_col and group_col in pf.schema_arrow.names:
            tbl = pf.read(columns=[group_col])
            import collections
            counts = collections.Counter(tbl.column(group_col).to_pylist())
            for src, c in counts.items():
                s = stats.setdefault(src, {"rows": 0, "sample_texts": []})
                s["rows"] += c
            # sample rows with their group for token estimates
            full = pf.read(columns=cols).to_pylist()
            for row in random.sample(full, min(len(full), SAMPLE_PER_SUBSET * 4)):
                s = stats[row[group_col]]
                if len(s["sample_texts"]) < SAMPLE_PER_SUBSET:
                    s["sample_texts"].append(row_text(row))
            del full
        else:
            s = stats.setdefault(subset_from_path, {"rows": 0, "sample_texts": []})
            s["rows"] += n
            if len(s["sample_texts"]) < SAMPLE_PER_SUBSET:
                want = SAMPLE_PER_SUBSET - len(s["sample_texts"])
                head = next(pf.iter_batches(batch_size=max(want * 3, 256), columns=cols)).to_pylist()
                s["sample_texts"].extend(row_text(r) for r in random.sample(head, min(len(head), want)))
        print(f"  {fp.split('/')[-2]}/{os.path.basename(fp)}: {n} rows", flush=True)
    # tokenize samples -> estimates
    out = {}
    for subset, s in sorted(stats.items()):
        texts = s["sample_texts"]
        if texts:
            fa = [len(e.ids) for e in tok_falcon.encode_batch(texts)]
            ll = [len(e.ids) for e in tok_llama.encode_batch(texts)]
            mean_fa, mean_ll = sum(fa) / len(fa), sum(ll) / len(ll)
        else:
            mean_fa = mean_ll = 0
        out[subset] = {
            "rows": s["rows"], "sampled": len(texts),
            "mean_tok_falcon": round(mean_fa, 1), "mean_tok_llama3": round(mean_ll, 1),
            "est_total_tok_falcon_M": round(s["rows"] * mean_fa / 1e6, 1),
            "est_total_tok_llama3_M": round(s["rows"] * mean_ll / 1e6, 1),
        }
        print(f"  => {subset}: rows={s['rows']} est_falcon={out[subset]['est_total_tok_falcon_M']}M est_llama={out[subset]['est_total_tok_llama3_M']}M", flush=True)
    return out

t0 = time.time()
result = {
    "method": f"row counts exact (parquet metadata); token counts = sampled estimate ({SAMPLE_PER_SUBSET}/subset, seed 42) x rows",
    "tulu-3-sft-mixture": scan("tulu3", f"{H}/model-lab/data/tulu-3-sft-mixture/**/*.parquet", group_col="source"),
    "smoltalk": scan("smoltalk", f"{H}/model-lab/data/smoltalk/**/*.parquet"),
}
json.dump(result, open(OUT, "w"), indent=1)
print(f"DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)
