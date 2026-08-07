#!/usr/bin/env python3
"""G4a reference side: teacher-forced PPL of Falcon-E-1B-Instruct (transformers
bitnet integration, packed checkpoint) on the EXACT token ids the engine scored.
Convention matched to aegis-core calculate_perplexity: predictions for positions
1..K-1 (K-1 CE terms), PPL = exp(mean NLL). Engine side scored K=1896 ids."""
import os, sys, time, resource
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
torch.set_num_threads(4)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1896
ids = [int(l) for l in open("/home/killboxincorporated/falcon-e-artifacts/ref_tokens.txt") if l.strip()][:K]
assert len(ids) == K, f"only {len(ids)} ids"

from transformers import AutoModelForCausalLM
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "/home/killboxincorporated/models/falcon-e-1b-instruct", dtype=torch.bfloat16)
model.eval()
print(f"load: {time.time()-t0:.1f}s", flush=True)

x = torch.tensor([ids], dtype=torch.long)
t0 = time.time()
with torch.no_grad():
    out = model(x)
logits = out.logits.float()  # [1, K, V]
logp = torch.log_softmax(logits[0, :-1], dim=-1)
tgt = x[0, 1:]
nll = -logp[torch.arange(K - 1), tgt]
ppl = torch.exp(nll.mean()).item()
dt = time.time() - t0
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"transformers reference PPL (teacher-forced, {K} tokens, {K-1} predictions): {ppl:.3f}")
print(f"forward wall: {dt:.1f}s ({K/dt:.2f} tok/s), peak RSS {rss:.2f}GB")
