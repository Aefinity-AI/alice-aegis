#!/usr/bin/env python3
"""m7_ternary_retarget.py — does on-device retargeting survive the ternary export gate?

THE QUESTION (pre-registered, see the banner printed at run time)
----------------------------------------------------------------
A 19-agent workflow measured, on the fp32 twin, that ~600k tokens of out-of-domain
data re-aims the model at an unseen domain: domain NLL improved by +3.0100 nats
(constant-LR arm H, 96/96 windows won) for a general-corpus cost of -0.1853 nats.

That result was measured on the fp32 TWIN, which is NOT deployable — export_hf.py
refuses fp checkpoints by design, so arm H can never reach the engine. The entire
"re-aim a sovereign model in the field" story therefore rests on an UNTESTED
assumption: that the same procedure applied to the deployable TERNARY artifact
still passes the 0.19% torch->engine round-trip gate.

If it does not, the capability evaporates. This script tests exactly that.

DESIGN
------
  source : checkpoints/m7_ternary.pt (14,171,392 params, ternary QAT) - COPIED, never written
  arm    : constant LR (the workflow measured constant-LR BEATING the anneal for
           specialization by 0.4168 nats, so constant is the correct recipe here),
           weight decay 0, on smoltalk_train_8k.bin
  budget : matched in TOKENS to the fp experiment (614,400), at reduced batch so
           peak RSS stays inside the machine's headroom
  then   : export_hf.py -> aegis-forge repack -> engine, and re-run the round-trip gate

PRE-REGISTERED READING (fixed before the run)
---------------------------------------------
  PASS   : round-trip rel diff <= 3% (the standing tolerance) AND full token parity
           -> on-device ternary retargeting is real and deployable.
  FAIL   : rel diff > 3% or token parity broken
           -> the capability does NOT survive to the engine. Report it as a KILL and
              do not claim field-retargeting anywhere outside the repo.
  Domain improvement is SECONDARY here and is reported for information only; this
  run is powered to answer the export question, not to re-measure the fp result.

  Baseline to beat, from docs/hardware_logs/m7_final_roundtrip_2026-07-27.log:
      torch 3.6808 vs engine 3.6880 = 0.19%, 471/471 tokens.
"""
import argparse
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

SP = "/tmp/claude-1000/-home-killboxincorporated/d75b4760-6465-4d8e-bf5f-32d865609e80/scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-ckpt", default=os.path.join(HERE, "checkpoints", "m7_ternary.pt"))
    ap.add_argument("--out-ckpt", default=os.path.join(HERE, "checkpoints", "m7_ternary_retargeted.pt"))
    ap.add_argument("--data", default=os.path.join(SP, "smoltalk_train_8k.bin"))
    ap.add_argument("--domain-val", default=os.path.join(SP, "smoltalk_val_8k.bin"))
    ap.add_argument("--general-val", default=os.path.join(HERE, "valid_8k.bin"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1.0e-4, help="CONSTANT — the arm that won")
    ap.add_argument("-B", "--batch", type=int, default=2, help="small batch to respect the RAM budget")
    ap.add_argument("-T", "--block", type=int, default=512)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    for p in (args.src_ckpt, args.data, args.domain_val, args.general_val):
        if not os.path.exists(p):
            sys.exit(f"[retarget] FATAL: missing {p}")

    ckpt = torch.load(args.src_ckpt, map_location="cpu", weights_only=False)
    cfg = TinyBitConfig(**ckpt["config"])
    if cfg.linear != "bitlinear":
        sys.exit(f"[retarget] FATAL: expected a TERNARY checkpoint, got linear={cfg.linear!r}")
    src_step = ckpt["step"]
    model = TinyBitModel(cfg)
    model.load_state_dict(ckpt["model"])

    tokens = args.steps * args.batch * args.block
    print("=" * 70, flush=True)
    print("M7 TERNARY RETARGETING — does the capability survive the export gate?", flush=True)
    print("=" * 70, flush=True)
    print(f"  source      : {args.src_ckpt} (step {src_step})", flush=True)
    print(f"  params      : {model.num_params():,} linear={cfg.linear}", flush=True)
    print(f"  arch        : L{cfg.num_hidden_layers} h{cfg.hidden_size} i{cfg.intermediate_size} "
          f"heads{cfg.num_attention_heads} vocab{cfg.vocab_size}", flush=True)
    print(f"  recipe      : CONSTANT lr {args.lr:.2e}, wd 0 "
          f"(constant beat the anneal by 0.4168 nats in the fp experiment)", flush=True)
    print(f"  budget      : {args.steps} steps x B{args.batch} x T{args.block} = {tokens:,} tokens", flush=True)
    print(f"  domain data : {args.data}", flush=True)
    print(f"  PRE-REG     : PASS iff round-trip rel diff <= 3% AND full token parity.", flush=True)
    print(f"                Baseline 0.19% (471/471) from m7_final_roundtrip_2026-07-27.log", flush=True)
    print(f"  output      : {args.out_ckpt}", flush=True)
    if args.dry_run:
        print("[retarget] --dry-run: nothing written", flush=True); return

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    shutil.copy2(args.src_ckpt, args.out_ckpt)
    print("[retarget] copied source -> output (original untouched)", flush=True)

    optimizer = make_optimizer(model, args.lr, 0.0)
    if "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
        print("[retarget] optimizer moments restored", flush=True)
    del ckpt

    dataset = TokenDataset(args.data, args.block)
    dval = build_val_ids(args.domain_val, min(512, cfg.max_position_embeddings))
    gval = build_val_ids(args.general_val, min(512, cfg.max_position_embeddings))
    rng = np.random.default_rng(args.seed + src_step)

    d0 = evaluate_val_ppl(model, dval); g0 = evaluate_val_ppl(model, gval)
    print(f"[retarget] BEFORE  domain_ppl {d0:.4f}   general_ppl {g0:.4f}", flush=True)
    model.train()

    t0 = time.time()
    for i in range(args.steps):
        set_optim_hparams(optimizer, args.lr, 0.0)
        x, y = dataset.get_batch(args.batch, rng)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if i % 25 == 0 or i == args.steps - 1:
            print(f"rt {i:5d}/{args.steps} | loss {loss.item():.4f} | lr {args.lr:.2e} | "
                  f"gnorm {float(gn):.2f} | {time.time()-t0:.1f}s", flush=True)
        if (i + 1) % args.eval_every == 0:
            d = evaluate_val_ppl(model, dval); g = evaluate_val_ppl(model, gval); model.train()
            print(f"  [val] {i+1} | domain {d:.4f} ({d-d0:+.4f}) | general {g:.4f} ({g-g0:+.4f})",
                  flush=True)

    d1 = evaluate_val_ppl(model, dval); g1 = evaluate_val_ppl(model, gval)
    save_ckpt(args.out_ckpt, model, optimizer, src_step + args.steps, cfg, [])
    print("", flush=True)
    print("=" * 70, flush=True)
    print("RETARGET COMPLETE (secondary numbers — the gate is what decides)", flush=True)
    print(f"  domain  ppl {d0:.4f} -> {d1:.4f}   ({100*(d0-d1)/d0:+.2f}%)", flush=True)
    print(f"  general ppl {g0:.4f} -> {g1:.4f}   ({100*(g0-g1)/g0:+.2f}%)", flush=True)
    print(f"  wall {time.time()-t0:.1f}s | checkpoint {args.out_ckpt}", flush=True)
    print("  NEXT: export -> repack -> engine round-trip gate. That is the verdict.", flush=True)
    print("[retarget] done", flush=True)


if __name__ == "__main__":
    main()
