#!/usr/bin/env python3
"""Dump reference-stack fixtures for the T2b/T2d parity tests.

Runs the candidate checkpoint through HuggingFace transformers and writes:

  tokens.txt   one decimal token id per line (T2d tokenization parity)
  hidden.bin   raw little-endian float32, layers x (tokens x hidden),
               the residual stream AFTER each decoder layer, pre final
               norm (T2b per-layer parity)

Run this once per candidate on a machine that can hold the model
(Colab free tier works for Falcon-E-1B and Llama3-8B-1.58), commit the
fixtures next to the eval sample, then on the engine side:

  AEGIS_MODEL=MODEL.SAF AEGIS_EMBED=EMBED.BIN AEGIS_VOCAB=VOCAB.BIN \
  AEGIS_EVAL_TEXT=sample.txt AEGIS_REF_TOKENS=tokens.txt \
  AEGIS_REF_HIDDEN=hidden.bin \
  cargo test --test reference_parity -- --ignored

Usage:
  python dump_reference_fixtures.py MODEL_ID_OR_PATH sample.txt outdir \
      [--max-tokens N] [--revision REV]
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="HF model id or local path")
    ap.add_argument("sample", help="eval text file (the engine strips non-ASCII; so do we)")
    ap.add_argument("outdir")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="short prompt is enough; per-layer states get big fast")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(args.sample, encoding="utf-8") as f:
        text = "".join(c for c in f.read() if c.isascii())

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                        trust_remote_code=True)
    # add_special_tokens=False: the engine's eval path feeds raw text with no
    # BOS; the reference must tokenize identically or T2d is comparing
    # different sequences.
    ids = tok(text, add_special_tokens=False)["input_ids"][: args.max_tokens]

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "tokens.txt"), "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    print(f"tokens.txt: {len(ids)} ids")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float32,  # engine computes in f32; compare like with like
    )
    model.eval()

    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)

    # hidden_states[0] is the embedding output and [i+1] is the residual
    # stream entering layer i+1 (= after layer i) — but transformers appends
    # the LAST entry AFTER the final model.norm (see the vendored
    # modeling_bitnet.py: all_hidden_states += (hidden_states,) follows
    # self.norm), while the engine captures pre-final-norm residuals.
    # Dump [1:-1]: layers 0..N-2, the entries both stacks define identically.
    # The last decoder layer is covered by the logit/PPL gates instead.
    layers = [h[0].to(torch.float32).numpy() for h in out.hidden_states[1:-1]]
    stacked = np.stack(layers)  # (num_layers - 1) x tokens x hidden
    stacked.astype("<f4").tofile(os.path.join(args.outdir, "hidden.bin"))
    print(f"hidden.bin: {stacked.shape[0]} layers (= num_layers - 1; the final "
          f"entry of hidden_states is post-norm and is deliberately excluded) "
          f"x {stacked.shape[1]} tokens x {stacked.shape[2]} hidden ({stacked.nbytes} bytes)")
    print("NOTE: if the model ties embeddings and the engine build prunes the "
          "vocab, dump with the SAME pruned tokenizer or T2d will fail by design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
