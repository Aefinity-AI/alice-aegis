#!/usr/bin/env python3
"""m7_final_roundtrip.py — the FINAL-ARTIFACT round-trip gate for the completed
M7 ternary run (ledger M20/M21 context). Identical procedure to
roundtrip_gate.py steps 2a-3, but loads the TRAINED checkpoint instead of
training a throwaway model. PASS = torch QAT PPL vs engine PPL within 3% on
the same held-out ids with exact tokenization parity."""
import json, os, re, sys, time

import torch
from roundtrip_gate import Tee, build_heldout, run, REPACKER, AEGIS_EVAL
from model import TinyBitConfig, TinyBitModel, teacher_forced_ppl
from export_hf import export_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
# CKPT/WORK are overridable by environment variable so this gate can be pointed at
# ANY ternary checkpoint (e.g. a retargeted one) without editing or copying the
# script. Unset -> byte-identical behaviour to every previous run of this gate.
CKPT = os.environ.get("M7_GATE_CKPT") or os.path.join(HERE, "checkpoints", "m7_ternary.pt")
TOKJ = os.path.join(HERE, "tokenizer_8k.json")  # the tokenizer that built train_8k.bin — NOT the 4k smoke tokenizer
WORK = os.environ.get("M7_GATE_WORK") or os.path.join(HERE, "m7_final_gate_work")
TOL = 0.03

torch.set_num_threads(4)
log = Tee(os.path.join(HERE, "m7_final_roundtrip.log"))
t0 = time.time()
log("=" * 70)
log(f"M7 FINAL-ARTIFACT round-trip gate | {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

cj = json.load(open(os.path.join(HERE, "configs", "m7_ternary.json")))
cfg = TinyBitConfig(
    vocab_size=cj["vocab_size"], hidden_size=cj["hidden"],
    intermediate_size=cj["inter"], num_hidden_layers=cj["layers"],
    num_attention_heads=cj["heads"], num_key_value_heads=cj["kv_heads"],
    max_position_embeddings=cj["ctx"], rms_norm_eps=cj["rms_eps"],
    rope_theta=cj["rope_theta"], hidden_act="silu",
    tie_word_embeddings=True, use_subln=True)
model = TinyBitModel(cfg)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
log(f"loaded {CKPT} @ step {ck['step']} | params {model.num_params()/1e6:.2f}M")

from tokenizers import Tokenizer
tok = Tokenizer.from_file(TOKJ)
ids, heldout_text = build_heldout(tok, cfg.max_position_embeddings, log)
os.makedirs(WORK, exist_ok=True)
heldout_path = os.path.join(WORK, "heldout.txt")
open(heldout_path, "w").write(heldout_text)
val_ids = torch.tensor(ids, dtype=torch.long)

torch_ppl = teacher_forced_ppl(model, val_ids)
log(f"[a] torch QAT teacher-forced PPL ({len(ids)} tokens): {torch_ppl:.4f}")

export_dir = os.path.join(WORK, "export")
log(f"[b] export -> {export_dir}")
export_checkpoint(model, cfg, TOKJ, export_dir, verbose=False)

out_dir = os.path.join(WORK, "artifacts")
os.makedirs(out_dir, exist_ok=True)
log(f"[c] repack -> {out_dir}")
rc, _ = run(["nice", "-n", "10", sys.executable, REPACKER,
             export_dir, out_dir, "--max-seq", "512"], log)
if rc != 0:
    log("GATE FAIL: repacker rejected export"); log.close(); sys.exit(1)
saf_mb = os.path.getsize(os.path.join(out_dir, "MODEL.SAF")) / 1e6
log(f"    MODEL.SAF = {saf_mb:.2f} MB (the deployable sovereignty artifact)")

log("[d] engine eval")
rc, out = run(["nice", "-n", "10", AEGIS_EVAL,
               os.path.join(out_dir, "MODEL.SAF"),
               os.path.join(out_dir, "EMBED.BIN"),
               os.path.join(out_dir, "VOCAB.BIN"),
               heldout_path, str(cfg.max_position_embeddings), "--sample"], log)
if rc != 0:
    log("GATE FAIL: aegis-eval errored"); log.close(); sys.exit(1)

m_tok = re.search(r"->\s*(\d+)\s*tokens", out)
m_ppl = re.search(r"Perplexity \(teacher-forced,\s*(\d+)\s*tokens\):\s*([\d.]+)", out)
if not (m_tok and m_ppl):
    log("GATE FAIL: could not parse engine output"); log.close(); sys.exit(1)
engine_ntok, engine_ppl = int(m_tok.group(1)), float(m_ppl.group(2))

log("=" * 70)
count_ok = engine_ntok == len(ids)
rel = abs(torch_ppl - engine_ppl) / max(engine_ppl, 1e-9)
log(f"token count : torch={len(ids)} engine={engine_ntok} {'OK' if count_ok else 'MISMATCH'}")
log(f"torch  PPL  : {torch_ppl:.4f}")
log(f"engine PPL  : {engine_ppl:.4f}")
log(f"rel diff    : {rel*100:.2f}% (tol {TOL*100:.0f}%)")
log(f"wall clock  : {time.time()-t0:.1f}s")
verdict = "PASS" if (rel <= TOL and count_ok) else "FAIL"
log(f">>> M7 FINAL ROUND-TRIP GATE {verdict} <<<")
log.close()
sys.exit(0 if verdict == "PASS" else 1)
