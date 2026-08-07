#!/usr/bin/env python3
"""roundtrip_gate.py — the STEP-0 GATE.

Proves the tinybit training/export stack agrees with the A.L.I.C.E. Rust engine:

  1. build a tiny config, train it briefly (just enough for non-degenerate weights)
  2. (a) torch-side teacher-forced PPL on a FIXED held-out slice, QAT forward
  3. (b) export via export_hf.py
  4. (c) repack:  aegis-forge/repack_ternary.py EXPORT OUT --max-seq 512
  5. (d) engine:  aegis-eval MODEL.SAF EMBED.BIN VOCAB.BIN heldout.txt K --sample
  6. (e) compare the two PPLs; PASS if relative diff <= 3%.

The held-out slice is built by tokenizing valid text with OUR tokenizer, taking
the first K (<= ctx) ids, and decoding them back to an ASCII text file. That file
is what the engine tokenizes itself; we then check the engine's token count == K
(the tokenization-parity guard the spec requires) before trusting the PPLs.

Everything is teed to roundtrip_gate.log with commands and numbers.
"""
import argparse
import os
import re
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from model import TinyBitConfig, TinyBitModel, teacher_forced_ppl
from train import TokenDataset, train_loop
from export_hf import export_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
REPACKER = "/home/killboxincorporated/aegis-forge/repack_ternary.py"
AEGIS_EVAL = "/home/killboxincorporated/aegis-eval/target/release/aegis-eval"
VALID_TXT = "/home/killboxincorporated/model-lab/data/TinyStories/TinyStoriesV2-GPT4-valid.txt"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def build_heldout(tok, ctx, log, k_target=480, char_budget=4000):
    """Return (ids_list, text) for a clean, ASCII, in-window held-out slice."""
    with open(VALID_TXT, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read(200000)
    # skip the leading partial story, then drop separators; take a clean run
    docs = [d for d in raw.split("<|endoftext|>") if len(d.strip()) > 500]
    chunk = docs[1] if len(docs) > 1 else docs[0]
    chunk = "".join(c for c in chunk if ord(c) < 128).strip()[:char_budget]
    ids = tok.encode(chunk, add_special_tokens=False).ids
    K = min(k_target, ctx - 16, len(ids))
    ids = ids[:K]
    text = tok.decode(ids)                      # ByteLevel decode -> raw bytes
    text = "".join(c for c in text if ord(c) < 128)
    # canonical re-encode check: the file the engine reads must map back to `ids`
    re_ids = tok.encode(text, add_special_tokens=False).ids
    if re_ids != ids:
        log(f"  [heldout] NOTE: decode->encode not identity "
            f"({len(ids)} vs {len(re_ids)}); using re-encoded ids for parity")
        ids = re_ids
    log(f"  [heldout] {len(text)} chars -> {len(ids)} tokens (ctx={ctx})")
    return ids, text


def run(cmd, log):
    log("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        log("    | " + line)
    if p.returncode != 0:
        log(f"  [ERROR] exit {p.returncode}")
    return p.returncode, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "train.bin"))
    ap.add_argument("--tokenizer", default=os.path.join(HERE, "tokenizer.json"))
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workdir", default=os.path.join(HERE, "gate_work"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(1337)
    np.random.seed(1337)
    try:
        os.nice(10)
    except OSError:
        pass

    log = Tee(os.path.join(HERE, "roundtrip_gate.log"))
    t_start = time.time()
    log("=" * 70)
    log(f"tinybit round-trip gate | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"torch {torch.__version__} | threads {args.threads} | tol {args.tol*100:.0f}%")
    log("=" * 70)

    for pth in (REPACKER, AEGIS_EVAL, args.tokenizer, args.data):
        if not os.path.exists(pth):
            log(f"FATAL missing prerequisite: {pth}")
            log("Run train_tokenizer.py (vocab 4096) and prepare_data.py first.")
            sys.exit(2)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    V = tok.get_vocab_size()

    cfg = TinyBitConfig(
        vocab_size=V, hidden_size=256, intermediate_size=640,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0,
        hidden_act="silu", tie_word_embeddings=True, use_subln=True)
    model = TinyBitModel(cfg)
    log(f"config: hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
        f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
        f"kv={cfg.num_key_value_heads} vocab={cfg.vocab_size} ctx={cfg.max_position_embeddings}")
    log(f"params: {model.num_params()/1e6:.2f}M")

    ids, heldout_text = build_heldout(tok, cfg.max_position_embeddings, log)
    os.makedirs(args.workdir, exist_ok=True)
    heldout_path = os.path.join(args.workdir, "heldout.txt")
    with open(heldout_path, "w") as f:
        f.write(heldout_text)
    val_ids = torch.tensor(ids, dtype=torch.long)

    # ---- train ----
    log(f"\n[1] training {args.steps} steps (B={args.batch} T={args.block} lr={args.lr})")
    dataset = TokenDataset(args.data, args.block)
    t0 = time.time()
    train_loop(model, cfg, dataset, val_ids,
               total_steps=args.steps, batch_size=args.batch, peak_lr=args.lr,
               warmup_steps=max(10, args.steps // 10), base_wd=0.1, grad_clip=1.0,
               eval_every=max(1, args.steps // 3), ckpt_path=None,
               log_every=25, seed=1337)
    train_dt = time.time() - t0
    log(f"[1] training done in {train_dt:.1f}s")

    # ---- (a) torch PPL (QAT forward) ----
    torch_ppl = teacher_forced_ppl(model, val_ids)
    log(f"\n[2a] torch QAT teacher-forced PPL ({len(ids)} tokens, {len(ids)-1} preds): {torch_ppl:.4f}")

    # ---- (b) export ----
    export_dir = os.path.join(args.workdir, "export")
    log(f"\n[2b] export -> {export_dir}")
    export_checkpoint(model, cfg, args.tokenizer, export_dir, verbose=False)
    log(f"  exported model.safetensors + config.json + tokenizer.json")

    # ---- (c) repack ----
    out_dir = os.path.join(args.workdir, "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    log(f"\n[2c] repack -> {out_dir}")
    rc, _ = run(["nice", "-n", "10", sys.executable, REPACKER,
                 export_dir, out_dir, "--max-seq", "512"], log)
    if rc != 0:
        log("GATE FAIL: repacker rejected the export"); log.close(); sys.exit(1)

    saf = os.path.join(out_dir, "MODEL.SAF")
    emb = os.path.join(out_dir, "EMBED.BIN")
    vcb = os.path.join(out_dir, "VOCAB.BIN")

    # ---- (d) engine eval ----
    log(f"\n[2d] engine eval (aegis-eval, --sample)")
    rc, out = run(["nice", "-n", "10", AEGIS_EVAL, saf, emb, vcb,
                   heldout_path, str(cfg.max_position_embeddings), "--sample"], log)
    if rc != 0:
        log("GATE FAIL: aegis-eval errored"); log.close(); sys.exit(1)

    m_tok = re.search(r"->\s*(\d+)\s*tokens", out)
    m_ppl = re.search(r"Perplexity \(teacher-forced,\s*(\d+)\s*tokens\):\s*([\d.]+)", out)
    if not (m_tok and m_ppl):
        log("GATE FAIL: could not parse engine output"); log.close(); sys.exit(1)
    engine_ntok = int(m_tok.group(1))
    engine_scored = int(m_ppl.group(1))
    engine_ppl = float(m_ppl.group(2))

    # ---- (e) compare ----
    log("\n" + "=" * 70)
    log("[3] VERDICT")
    log(f"  token count   : torch={len(ids)}  engine={engine_ntok}  "
        f"(engine PPL line reports {engine_scored})")
    count_ok = (engine_ntok == len(ids))
    if not count_ok:
        log(f"  [WARN] token counts differ by {abs(engine_ntok-len(ids))} "
            "-- tokenization drift (engine BPE vs HF regex). Investigate before "
            "trusting the PPL delta; see README caveats.")
    log(f"  torch  PPL    : {torch_ppl:.4f}")
    log(f"  engine PPL    : {engine_ppl:.4f}")
    rel = abs(torch_ppl - engine_ppl) / max(engine_ppl, 1e-9)
    log(f"  relative diff : {rel*100:.2f}%   (tolerance {args.tol*100:.0f}%)")
    log(f"  peak RSS      : {peak_rss_gb():.2f} GB")
    log(f"  wall clock    : {time.time()-t_start:.1f}s")

    verdict = "PASS" if (rel <= args.tol and count_ok) else "FAIL"
    log(f"\n  >>> GATE {verdict} <<<")
    if verdict != "PASS" and rel <= args.tol and not count_ok:
        log("  (PPL within tolerance but token counts disagree -> not a clean pass)")
    log("=" * 70)
    log.close()
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
