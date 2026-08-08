#!/usr/bin/env python3
"""train_tokenizer.py — byte-level BPE (HF `tokenizers`) for tinybit.

  * ByteLevel pre-tokenizer + ByteLevel decoder, GPT-2-style vocab strings
    (space -> 'Ġ', newline -> 'Ċ', ...). This is what makes encode the exact
    inverse of the engine's byte_to_unicode map (aegis-core tokenizer.rs), so
    the two stacks agree on token strings by construction.
  * <|endoftext|> is added as the FIRST special token, so it takes id 0. The
    engine loader drops any BPE merge containing token id 0; keeping id 0 a
    special (which never appears in merges) sidesteps that entirely.
  * Dense ids 0..V-1 (special + 256-byte alphabet + merges), asserted.
  * add_prefix_space=False: the engine never prepends a leading space, so we
    must not either, or the first word would gain a phantom 'Ġ'.

Usage:
  python3 train_tokenizer.py [--input FILE] [--vocab-size 8192]
                             [--max-bytes N] [--out tokenizer.json]
"""
import argparse
import os
import sys

DEFAULT_INPUT = "/home/killboxincorporated/model-lab/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
EOT = "<|endoftext|>"


def text_iterator(path: str, max_bytes: int, chunk_lines: int = 10000):
    """Stream up to `max_bytes` of the file, yielding lists of lines. max_bytes<=0
    reads the whole file. Byte-budgeted so a huge corpus does not blow RSS."""
    read = 0
    batch = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            batch.append(line)
            read += len(line.encode("utf-8"))
            if len(batch) >= chunk_lines:
                yield batch
                batch = []
            if max_bytes > 0 and read >= max_bytes:
                break
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="only train on the first N bytes of --input (0 = all). "
                         "Use a subset (e.g. 100000000 = 100MB) for speed.")
    ap.add_argument("--out", default=None,
                    help="output tokenizer.json (default: alongside this script)")
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer.json")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[EOT],                         # id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, dense
        show_progress=True,
    )

    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")
    size = os.path.getsize(args.input)
    budget = args.max_bytes if args.max_bytes > 0 else size
    print(f"[tok] training vocab={args.vocab_size} on {min(budget, size)/1e6:.1f}MB "
          f"of {args.input}", flush=True)

    tok.train_from_iterator(text_iterator(args.input, args.max_bytes), trainer=trainer)

    # ---- invariants the whole pipeline depends on ----
    eot_id = tok.token_to_id(EOT)
    if eot_id != 0:
        sys.exit(f"FATAL: {EOT} got id {eot_id}, expected 0 "
                 "(engine drops id-0 merges — id 0 must be the special)")
    V = tok.get_vocab_size()
    vocab = tok.get_vocab()
    ids = sorted(vocab.values())
    if ids != list(range(V)):
        missing = [i for i in range(V) if i not in set(ids)][:5]
        sys.exit(f"FATAL: ids are not dense 0..{V-1} (first gaps: {missing})")

    tok.save(args.out)
    print(f"[tok] saved {args.out}: vocab_size={V}, {EOT}=id 0 (dense ids verified)",
          flush=True)

    # quick smoke check: round-trip a sample string
    sample = "Once upon a time, there was a little cat.\n"
    enc = tok.encode(sample)
    dec = tok.decode(enc.ids)
    print(f"[tok] smoke: {len(enc.ids)} tokens; decode-eq={dec == sample}", flush=True)


if __name__ == "__main__":
    main()
