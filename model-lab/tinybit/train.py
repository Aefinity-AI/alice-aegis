#!/usr/bin/env python3
"""train.py — CPU pretraining loop for tinybit (BitNet-style QAT).

Recipe (BitNet b1.58 training tips):
  * AdamW, gradient clipping.
  * Two-stage LR: linear warmup -> cosine decay to a stage-2 knee, then a second
    cosine at a lower peak; weight-decay is dropped to 0 at the stage-2 boundary.
  * Packed uint16 token dataset (memmap), random contiguous windows of length T.
  * Checkpoint save/resume: model + optim + step + RNG (torch/numpy/python).
  * Val-PPL anchor on a FIXED held-out slice every --eval-every steps, with a
    tripwire that flags when val PPL rises for 2 consecutive checks.
  * Plain-text, flush=True progress (systemd-journal friendly). No wandb/network.

Importable helpers (reused by roundtrip_gate.py): lr_schedule, TokenDataset,
make_optimizer, evaluate_val_ppl, train_loop.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

from model import TinyBitModel, TinyBitConfig, teacher_forced_ppl


# ---------------------------------------------------------------------------
# two-stage LR schedule
# ---------------------------------------------------------------------------
def lr_schedule(step, total_steps, warmup_steps, peak_lr,
                stage2_frac=0.5, stage2_lr_mult=0.1, min_lr_mult=0.0):
    """Linear warmup -> cosine to the stage-2 knee (peak*stage2_lr_mult) ->
    second cosine down to peak*min_lr_mult. Returns the LR for `step`."""
    stage2_start = int(stage2_frac * total_steps)
    knee = peak_lr * stage2_lr_mult
    floor = peak_lr * min_lr_mult
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    if step < stage2_start:
        prog = (step - warmup_steps) / max(1, stage2_start - warmup_steps)
        return knee + 0.5 * (peak_lr - knee) * (1 + math.cos(math.pi * prog))
    prog = (step - stage2_start) / max(1, total_steps - stage2_start)
    return floor + 0.5 * (knee - floor) * (1 + math.cos(math.pi * prog))


def wd_for_step(step, total_steps, base_wd, stage2_frac=0.5):
    """Weight decay is active in stage 1, then 0 in stage 2 (BitNet tip)."""
    return base_wd if step < int(stage2_frac * total_steps) else 0.0


def lr_schedule_cosine(step, total_steps, warmup_steps, peak_lr, min_lr_mult=0.1):
    """Standard fp pretraining schedule: linear warmup -> single cosine decay to
    peak*min_lr_mult. Used by the fp arm (sched="cosine"); weight decay stays
    CONSTANT under this schedule (the stage-2 wd drop is a ternary-QAT trick and
    does not apply to fp training)."""
    floor = peak_lr * min_lr_mult
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return floor + 0.5 * (peak_lr - floor) * (1 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class TokenDataset:
    """memmap uint16 token stream; samples random contiguous (T+1)-windows."""

    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) < block_size + 1:
            raise ValueError(f"{bin_path} has {len(self.data)} tokens, need > {block_size}")

    def get_batch(self, batch_size, rng: np.random.Generator):
        ix = rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1:i + 1 + self.block_size].astype(np.int64) for i in ix])
        return torch.from_numpy(x), torch.from_numpy(y)


def make_optimizer(model, lr, wd):
    """Two param groups: matrices decay, norms/embeddings do not."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed_tokens" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8)


def set_optim_hparams(optimizer, lr, wd):
    optimizer.param_groups[0]["lr"] = lr
    optimizer.param_groups[0]["weight_decay"] = wd
    optimizer.param_groups[1]["lr"] = lr
    optimizer.param_groups[1]["weight_decay"] = 0.0


@torch.no_grad()
def evaluate_val_ppl(model, val_ids: torch.Tensor) -> float:
    return teacher_forced_ppl(model, val_ids)


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def save_ckpt(path, model, optimizer, step, cfg, val_hist):
    # atomic: write to .tmp then rename, so a kill mid-save can never leave a
    # corrupt checkpoint behind on a multi-day resumable run
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "step": step,
        "config": cfg.as_dict(),
        "val_hist": val_hist,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    rng = ckpt.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return ckpt["step"], ckpt.get("val_hist", [])


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_loop(model, cfg, dataset, val_ids, *, total_steps, batch_size, peak_lr,
               warmup_steps, base_wd, grad_clip, eval_every, ckpt_path,
               log_every=10, seed=1337, start_step=0, optimizer=None,
               val_hist=None, save_every=None, sched="two_stage"):
    rng = np.random.default_rng(seed + start_step)
    if optimizer is None:
        optimizer = make_optimizer(model, peak_lr, base_wd)
    if val_hist is None:
        val_hist = []
    if save_every is None:
        save_every = eval_every
    rising = 0
    t0 = time.time()
    model.train()

    for step in range(start_step, total_steps):
        if sched == "cosine":
            lr = lr_schedule_cosine(step, total_steps, warmup_steps, peak_lr)
            wd = base_wd                       # constant wd for fp recipes
        else:                                  # "two_stage" (ternary-QAT recipe)
            lr = lr_schedule(step, total_steps, warmup_steps, peak_lr)
            wd = wd_for_step(step, total_steps, base_wd)
        set_optim_hparams(optimizer, lr, wd)

        x, y = dataset.get_batch(batch_size, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_every == 0 or step == total_steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{total_steps} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | wd {wd:.3f} | gnorm {float(gnorm):.2f} | "
                  f"{dt:.1f}s", flush=True)

        if val_ids is not None and (step + 1) % eval_every == 0:
            vppl = evaluate_val_ppl(model, val_ids)
            model.train()
            prev = val_hist[-1] if val_hist else None
            val_hist.append(vppl)
            tag = ""
            if prev is not None and vppl > prev:
                rising += 1
                if rising >= 2:
                    tag = "  [TRIPWIRE] val PPL rose 2 consecutive checks"
            else:
                rising = 0
            print(f"  [val] step {step+1} | val_ppl {vppl:.3f}"
                  f"{' (prev %.3f)' % prev if prev is not None else ''}{tag}", flush=True)

        if ckpt_path and (step + 1) % save_every == 0:
            save_ckpt(ckpt_path, model, optimizer, step + 1, cfg, val_hist)

    if ckpt_path:
        save_ckpt(ckpt_path, model, optimizer, total_steps, cfg, val_hist)
    return optimizer, val_hist


def build_val_ids(bin_path, n_tokens, offset_frac=0.9):
    """A FIXED held-out slice taken from deep in the token stream (default 90%
    in) so it does not overlap the early windows the loop samples most."""
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    start = int(len(data) * offset_frac)
    ids = np.asarray(data[start:start + n_tokens], dtype=np.int64)
    return torch.from_numpy(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--config", default=None,
                    help="JSON file of defaults (keys = argparse dests, e.g. "
                         "configs/m7a_twin.json); explicit CLI flags override it, "
                         "keys starting with '_' are comments")
    ap.add_argument("--data", default=os.path.join(here, "train.bin"))
    ap.add_argument("--val-data", default=None, help="defaults to --data")
    ap.add_argument("--ckpt", default=os.path.join(here, "checkpoints", "tinybit.pt"))
    ap.add_argument("--resume", action="store_true")
    # model
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--inter", type=int, default=640)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--rms-eps", type=float, default=1e-5)
    ap.add_argument("--no-subln", action="store_true")
    ap.add_argument("--linear", choices=("bitlinear", "fp"), default="bitlinear",
                    help="bitlinear = ternary QAT (engine-exportable); fp = plain "
                         "fp32 nn.Linear twin (NOT exportable to the engine)")
    # training
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("-B", "--batch", type=int, default=8)
    ap.add_argument("-T", "--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--sched", choices=("two_stage", "cosine"), default="two_stage",
                    help="two_stage = BitNet QAT recipe (wd drops to 0 at the knee); "
                         "cosine = standard fp recipe (constant wd)")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=None,
                    help="checkpoint cadence in steps (defaults to --eval-every)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-tokens", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)

    # --config JSON supplies defaults; explicit CLI flags still win
    pre, _ = ap.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            file_cfg = {k: v for k, v in json.load(f).items()
                        if not k.startswith("_")}
        valid = {a.dest for a in ap._actions}
        unknown = sorted(set(file_cfg) - valid)
        if unknown:
            sys.exit(f"[train] unknown keys in {pre.config}: {unknown}")
        ap.set_defaults(**file_cfg)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    cfg = TinyBitConfig(
        vocab_size=args.vocab_size, hidden_size=args.hidden,
        intermediate_size=args.inter, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.ctx, rms_norm_eps=args.rms_eps,
        rope_theta=args.rope_theta, hidden_act="silu",
        tie_word_embeddings=True, use_subln=not args.no_subln,
        linear=args.linear)

    model = TinyBitModel(cfg)
    print(f"[train] model params: {model.num_params()} "
          f"({model.num_params()/1e6:.2f}M) | linear={cfg.linear} "
          f"| sched={args.sched} | B={args.batch} T={args.block} "
          f"threads={args.threads}", flush=True)

    dataset = TokenDataset(args.data, args.block)
    val_ids = build_val_ids(args.val_data or args.data,
                            min(args.val_tokens, args.ctx))

    optimizer = make_optimizer(model, args.lr, args.wd)
    start_step, val_hist = 0, []
    if args.resume and os.path.exists(args.ckpt):
        start_step, val_hist = load_ckpt(args.ckpt, model, optimizer)
        print(f"[train] resumed from step {start_step}", flush=True)

    train_loop(model, cfg, dataset, val_ids,
               total_steps=args.steps, batch_size=args.batch, peak_lr=args.lr,
               warmup_steps=args.warmup, base_wd=args.wd, grad_clip=args.grad_clip,
               eval_every=args.eval_every, ckpt_path=args.ckpt,
               seed=args.seed, start_step=start_step, optimizer=optimizer,
               val_hist=val_hist, save_every=args.save_every,
               log_every=args.log_every, sched=args.sched)
    print("[train] done", flush=True)


if __name__ == "__main__":
    main()
