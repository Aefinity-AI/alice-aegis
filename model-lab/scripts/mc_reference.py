#!/usr/bin/env python3
"""M3 reference side: transformers Falcon-E-1B-Instruct (bf16, packed
checkpoint — same load as G4a g4a_reference_ppl.py) scoring the SAME items
JSONL the engine scored, on the EXACT same token ids.

Per item: ONE teacher-forced forward over all K choices batched, each row =
ctx_ids + choice_ids[k], right-padded with id 0 to the longest row. Causal
attention means real positions never attend to trailing pads, and default
position_ids are arange, so right-padded rows score identically to
per-sequence forwards (validated this session vs the sequential path on the
3-item smoke). Per choice k: NLL summed over ONLY the continuation positions
(predictions of tokens x[C..L_k-1], i.e. logit rows C-1..L_k-2). Scores and
tie-breaking match aegis-eval/src/mc.rs exactly:
  acc      : pred = argmin_k cont_nll[k]            (raw sum)
  acc_norm : pred = argmax_k(-cont_nll[k] / utf8_byte_len(choice_text_k))
  ties break toward the lower choice index (strict > replacement).

Usage: mc_reference.py <items.jsonl> <results_out.jsonl>
"""
import json
import os
import resource
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch

torch.set_num_threads(4)

MODEL_DIR = "/home/killboxincorporated/models/falcon-e-1b-instruct"
NORM_DEF = (
    "acc: pred = argmin_k cont_nll[k] (raw sum over continuation tokens); "
    "acc_norm: pred = argmax_k(-cont_nll[k] / utf8_byte_len(choice_text_k)); "
    "ties -> lower index"
)


def argbest(scores):
    best = 0
    for k in range(1, len(scores)):
        if scores[k] > scores[best]:
            best = k
    return best


def main():
    items_path, out_path = sys.argv[1], sys.argv[2]
    items = [json.loads(l) for l in open(items_path) if l.strip()]

    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
    model.eval()
    print(f"load: {time.time()-t0:.1f}s", flush=True)

    out = open(out_path, "w")
    out.write(
        json.dumps(
            {
                "header": "mc_reference.py transformers bf16",
                "items": items_path,
                "normalization": NORM_DEF,
            }
        )
        + "\n"
    )

    n = n_raw = n_norm = 0
    tokens_forwarded = 0
    t0 = time.time()
    for item in items:
        ctx = item["ctx_ids"]
        C = len(ctx)
        rows = [ctx + cont for cont in item["choice_ids"]]
        lens = [len(r) for r in rows]
        lmax = max(lens)
        x = torch.zeros((len(rows), lmax), dtype=torch.long)  # pad id 0 (ignored)
        for k, r in enumerate(rows):
            x[k, : len(r)] = torch.tensor(r, dtype=torch.long)
        with torch.no_grad():
            logits = model(x, use_cache=False).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        cont_nll, cont_per_tok = [], []
        for k, cont in enumerate(item["choice_ids"]):
            tgt = x[k, 1 : lens[k]]
            nll_all = -logp[k, torch.arange(lens[k] - 1), tgt]
            s = nll_all[C - 1 :].sum().item()  # continuation terms only
            cont_nll.append(s)
            cont_per_tok.append(s / len(cont))
            tokens_forwarded += lens[k]

        byte_lens = item["choice_byte_lens"]
        pred_raw = argbest([-v for v in cont_nll])
        pred_norm = argbest([-v / b for v, b in zip(cont_nll, byte_lens)])
        correct_raw = pred_raw == item["answer_idx"]
        correct_norm = pred_norm == item["answer_idx"]
        n += 1
        n_raw += correct_raw
        n_norm += correct_norm

        out.write(
            json.dumps(
                {
                    "id": item["id"],
                    "answer_idx": item["answer_idx"],
                    "choice_nll": [round(v, 6) for v in cont_nll],
                    "choice_nll_per_token": [round(v, 6) for v in cont_per_tok],
                    "choice_cont_tokens": [len(c) for c in item["choice_ids"]],
                    "choice_byte_lens": byte_lens,
                    "pred_raw": pred_raw,
                    "pred_norm": pred_norm,
                    "correct_raw": bool(correct_raw),
                    "correct_norm": bool(correct_norm),
                }
            )
            + "\n"
        )
        out.flush()
        print(
            f"[{n}] {item['id']} pred_raw={pred_raw} pred_norm={pred_norm} "
            f"gold={item['answer_idx']} nll=[{','.join(f'{v:.6f}' for v in cont_nll)}] | "
            f"running acc {n_raw/n:.3f} acc_norm {n_norm/n:.3f} | {time.time()-t0:.0f}s",
            flush=True,
        )

    acc, acc_norm = n_raw / n, n_norm / n
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out.write(
        json.dumps(
            {
                "summary": True,
                "n": n,
                "acc": round(acc, 6),
                "acc_norm": round(acc_norm, 6),
                "tokens_forwarded": tokens_forwarded,
                "wall_s": round(dt, 1),
            }
        )
        + "\n"
    )
    out.close()
    print("-" * 50)
    print(
        f"MC reference summary: n={n} acc={acc:.4f} ({n_raw}/{n}) "
        f"acc_norm={acc_norm:.4f} ({n_norm}/{n})"
    )
    print(
        f"tokens forwarded: {tokens_forwarded} | wall {dt:.1f}s "
        f"({tokens_forwarded/dt:.2f} tok/s) | peak RSS {rss:.2f}GB"
    )
    print(f"normalization: {NORM_DEF}")


if __name__ == "__main__":
    main()
