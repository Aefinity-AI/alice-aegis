#!/usr/bin/env python3
"""m7_lr_cooldown.py — the M7 LR-floor control experiment.

WHY THIS EXISTS
---------------
Ledger M20 claims the M7 ternary run differed from the m7a fp32 twin in "one
variable (linear=bitlinear)". An audit refuted that: SEVEN knobs differ. The
largest UNDISCLOSED confound is the end-of-training learning-rate floor.

    twin    : sched=cosine    -> lr_schedule_cosine(..., min_lr_mult=0.1)
              => stopped at 1e-3 * 0.1 = 1.00e-04, weight decay held at 0.1
    ternary : sched=two_stage -> lr_schedule(..., min_lr_mult=0.0)
              => annealed to 1.32e-13, weight decay dropped to 0 at the knee

Annealing LR to ~0 is worth a substantial perplexity drop on its own. So the
twin was never given the cooldown the ternary got, and part (possibly all) of
the ternary's apparent win may be that alone.

WHAT THIS DOES
--------------
Resumes the FINISHED twin checkpoint and gives it the cooldown it never had:
cosine-anneal lr from its stopping point (1.00e-04) to exactly 0, with weight
decay set to 0, for --steps additional steps. Everything else is held fixed —
same data file, same batch/ctx, same thread count, same seed lineage.

SAFETY
------
The original checkpoint is NEVER modified. It is copied to a new path first and
all writes go to the copy. train.py is not edited (the round-trip gate depends
on it); this script only imports from it.

HONEST CAVEAT (state this with any result)
------------------------------------------
The cooled twin also receives --steps * B * T EXTRA TRAINING TOKENS that the
ternary arm did not get at that point. That biases the comparison IN THE TWIN'S
FAVOUR, i.e. against the ternary claim. This makes the control conservative: if
the ternary still wins after the twin is both cooled AND given extra tokens,
the result is stronger than before, not weaker.

This experiment does NOT and CANNOT settle the parameter-count confound: the
ternary model has 14,171,392 params vs the twin's 6,529,920 (2.17x). A true
single-variable answer requires a same-size ternary/fp pair trained from
scratch, which is a ~130h commitment. This control is the cheap first cut.
"""
import argparse
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import TinyBitModel, TinyBitConfig  # noqa: E402
from train import (TokenDataset, make_optimizer, set_optim_hparams,  # noqa: E402
                   evaluate_val_ppl, save_ckpt, build_val_ids)


def cooldown_lr(step_in_cooldown, cooldown_steps, lr_start):
    """Cosine anneal lr_start -> exactly 0 across the cooldown.

    Matches the SHAPE of the ternary arm's stage-2 tail (cosine to a zero
    floor) rather than inventing a new schedule.
    """
    prog = step_in_cooldown / max(1, cooldown_steps)
    return 0.5 * lr_start * (1.0 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-ckpt", default=os.path.join(HERE, "checkpoints", "m7a_twin.pt"),
                    help="finished twin checkpoint (READ ONLY, never written)")
    ap.add_argument("--out-ckpt", default=os.path.join(HERE, "checkpoints", "m7a_twin_cooled.pt"),
                    help="new checkpoint written by this run")
    ap.add_argument("--data", default=os.path.join(HERE, "train_8k.bin"))
    ap.add_argument("--val-data", default=os.path.join(HERE, "valid_8k.bin"))
    ap.add_argument("--steps", type=int, default=4000, help="cooldown length in steps")
    ap.add_argument("--lr-start", type=float, default=1.0e-4,
                    help="LR the twin actually stopped at (1e-3 peak * min_lr_mult 0.1)")
    ap.add_argument("-B", "--batch", type=int, default=8)
    ap.add_argument("-T", "--block", type=int, default=512)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-tokens", type=int, default=512,
                    help="matches the original runs' in-training probe (512)")
    ap.add_argument("--threads", type=int, default=4,
                    help="4 is the MEASURED optimum on this box; 8T was slower")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true",
                    help="load, print the plan and the first 5 LR values, then exit")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.src_ckpt):
        sys.exit(f"[cooldown] FATAL: source checkpoint not found: {args.src_ckpt}")

    # ---- rebuild the model from the checkpoint's OWN stored config -----------
    ckpt = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    cfg_d = ckpt["config"]
    src_step = ckpt["step"]
    val_hist = list(ckpt.get("val_hist", []))
    cfg = TinyBitConfig(**cfg_d)

    if cfg.linear != "fp":
        sys.exit(f"[cooldown] FATAL: expected an fp checkpoint, got linear={cfg.linear!r}. "
                 f"This control only applies to the fp32 twin.")

    model = TinyBitModel(cfg)
    model.load_state_dict(ckpt["model"])
    n_params = model.num_params()

    print(f"[cooldown] source      : {args.src_ckpt}", flush=True)
    print(f"[cooldown] source step : {src_step}", flush=True)
    print(f"[cooldown] params      : {n_params} ({n_params/1e6:.3f}M) linear={cfg.linear}", flush=True)
    print(f"[cooldown] arch        : layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"inter={cfg.intermediate_size} heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} vocab={cfg.vocab_size}", flush=True)
    print(f"[cooldown] val_hist    : {len(val_hist)} entries, last {val_hist[-1]:.4f}"
          if val_hist else "[cooldown] val_hist    : empty", flush=True)
    print(f"[cooldown] PLAN        : {args.steps} steps, lr {args.lr_start:.3e} -> 0 (cosine), "
          f"wd 0.0 (was 0.1), B={args.batch} T={args.block} threads={args.threads}", flush=True)
    print(f"[cooldown] extra tokens: {args.steps * args.batch * args.block:,} "
          f"(CONSERVATIVE: favours the twin)", flush=True)
    print(f"[cooldown] output      : {args.out_ckpt}", flush=True)

    lrs = [cooldown_lr(i, args.steps, args.lr_start) for i in range(min(5, args.steps))]
    print(f"[cooldown] first LRs   : {', '.join(f'{v:.4e}' for v in lrs)}", flush=True)
    print(f"[cooldown] final LR    : {cooldown_lr(args.steps - 1, args.steps, args.lr_start):.4e}",
          flush=True)

    if args.dry_run:
        print("[cooldown] --dry-run: exiting before any training or write", flush=True)
        return

    # ---- copy the checkpoint; original stays untouched -----------------------
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    shutil.copy2(args.src_ckpt, args.out_ckpt)
    print(f"[cooldown] copied source -> output (original untouched)", flush=True)

    # ---- optimizer: restore AdamW moments, then force wd=0 -------------------
    optimizer = make_optimizer(model, args.lr_start, 0.0)
    if "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
        print("[cooldown] optimizer moments restored from checkpoint", flush=True)
    rng_state = ckpt.get("rng")
    if rng_state:
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["python"])
        print("[cooldown] RNG state restored", flush=True)
    del ckpt

    dataset = TokenDataset(args.data, args.block)
    val_ids = build_val_ids(args.val_data, min(args.val_tokens, cfg.max_position_embeddings))
    rng = np.random.default_rng(args.seed + src_step)

    baseline = evaluate_val_ppl(model, val_ids)
    print(f"[cooldown] BASELINE val_ppl (before any cooldown step): {baseline:.4f}", flush=True)
    model.train()

    t0 = time.time()
    for i in range(args.steps):
        lr = cooldown_lr(i, args.steps, args.lr_start)
        set_optim_hparams(optimizer, lr, 0.0)

        x, y = dataset.get_batch(args.batch, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if i % args.log_every == 0 or i == args.steps - 1:
            print(f"cool {i:5d}/{args.steps} | loss {loss.item():.4f} | lr {lr:.3e} | "
                  f"wd 0.000 | gnorm {float(gnorm):.2f} | {time.time()-t0:.1f}s", flush=True)

        if (i + 1) % args.eval_every == 0:
            vppl = evaluate_val_ppl(model, val_ids)
            model.train()
            val_hist.append(vppl)
            print(f"  [val] cool {i+1} | val_ppl {vppl:.4f} | baseline {baseline:.4f} | "
                  f"delta {vppl - baseline:+.4f}", flush=True)

        if (i + 1) % args.save_every == 0:
            save_ckpt(args.out_ckpt, model, optimizer, src_step + i + 1, cfg, val_hist)

    final = evaluate_val_ppl(model, val_ids)
    save_ckpt(args.out_ckpt, model, optimizer, src_step + args.steps, cfg, val_hist)

    print("", flush=True)
    print("=" * 66, flush=True)
    print("M7 LR-COOLDOWN CONTROL — RESULT (single 512-token probe)", flush=True)
    print("=" * 66, flush=True)
    print(f"  twin val_ppl BEFORE cooldown : {baseline:.4f}", flush=True)
    print(f"  twin val_ppl AFTER  cooldown : {final:.4f}", flush=True)
    print(f"  change                       : {final - baseline:+.4f} "
          f"({100.0*(baseline-final)/baseline:+.2f}% better)", flush=True)
    print(f"  wall clock                   : {time.time()-t0:.1f}s", flush=True)
    print(f"  checkpoint                   : {args.out_ckpt}", flush=True)
    print("", flush=True)
    print("  NOTE: this 512-token probe is the SAME underpowered metric the", flush=True)
    print("  original runs used. The decisive number comes from the paired", flush=True)
    print("  multi-window eval (m7_paired_eval.py). Do not conclude from this.", flush=True)
    print("[cooldown] done", flush=True)


if __name__ == "__main__":
    main()
