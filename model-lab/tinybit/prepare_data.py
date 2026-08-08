#!/usr/bin/env python3
"""prepare_data.py — tokenize a text file into a packed uint16 .bin memmap.

Documents are split on the literal '<|endoftext|>' separator that TinyStories
already carries; each document is byte-level-BPE encoded and terminated with the
<|endoftext|> id (0) so the training loop sees explicit document boundaries.
Output is a flat little-endian uint16 stream (vocab < 65536), memmap-friendly.

Usage:
  python3 prepare_data.py [--input FILE] [--tokenizer tokenizer.json]
                          [--out train.bin] [--max-bytes N]
"""
import argparse
import os
import sys

import numpy as np

DEFAULT_INPUT = "/home/killboxincorporated/model-lab/data/TinyStories/TinyStoriesV2-GPT4-train.txt"
SEP = "<|endoftext|>"


def read_documents(path: str, max_bytes: int):
    """Yield document strings split on SEP, streaming with a carry buffer so we
    never hold the whole (2GB) file in memory."""
    carry = ""
    read = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(4 * 1024 * 1024)
            if not block:
                break
            read += len(block.encode("utf-8"))
            carry += block
            parts = carry.split(SEP)
            carry = parts.pop()          # last piece may be incomplete
            for doc in parts:
                yield doc
            if max_bytes > 0 and read >= max_bytes:
                break
    if carry.strip():
        yield carry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--tokenizer", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tokenizer.json"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "train.bin"))
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="only read the first N bytes of --input (0 = all)")
    ap.add_argument("--batch", type=int, default=1000, help="docs per encode_batch")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id(SEP)
    if eot is None:
        sys.exit(f"{SEP} not in tokenizer")
    V = tok.get_vocab_size()
    if V > 65536:
        sys.exit(f"vocab {V} > 65536: uint16 packing impossible")

    total = 0
    n_docs = 0
    buf = []
    with open(args.out, "wb") as out:
        def flush(docs):
            nonlocal total, n_docs
            if not docs:
                return
            encs = tok.encode_batch(docs, add_special_tokens=False)
            arrs = []
            for e in encs:
                arrs.append(np.asarray(e.ids, dtype=np.uint16))
                arrs.append(np.asarray([eot], dtype=np.uint16))
            flat = np.concatenate(arrs) if arrs else np.zeros(0, dtype=np.uint16)
            out.write(flat.tobytes())
            total += int(flat.size)
            n_docs += len(docs)

        for doc in read_documents(args.input, args.max_bytes):
            doc = doc.strip()
            if not doc:
                continue
            buf.append(doc)
            if len(buf) >= args.batch:
                flush(buf)
                buf = []
                if n_docs % 20000 == 0:
                    print(f"[data] {n_docs} docs, {total} tokens", flush=True)
        flush(buf)

    print(f"[data] wrote {args.out}: {total} uint16 tokens from {n_docs} docs "
          f"({total*2} bytes)", flush=True)


if __name__ == "__main__":
    main()
