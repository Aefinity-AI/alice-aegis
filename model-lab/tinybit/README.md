# tinybit — from-scratch ternary (BitNet-style QAT) LM trainer for ALICE

`tinybit` trains a tiny llama-family decoder with ternary (1.58-bit) QAT weights
and per-token int8 activations, then exports checkpoints that flow through the
**existing** `aegis-forge/repack_ternary.py` repacker into the **existing** Rust
inference engine (`aegis-core` / `aegis-eval`). Nothing in `aegis-forge`,
`aegis-core`, or `aegis-eval` is modified — tinybit conforms to their contracts.

The whole point is the **round-trip gate** (`roundtrip_gate.py`): a model trained
in torch must score the *same* perplexity when run by the Rust engine. If the two
agree within tolerance, the training/export/repack/inference chain is proven
end-to-end and larger runs can be trusted.

## Files

| File | Role |
|---|---|
| `train_tokenizer.py` | Byte-level BPE (HF `tokenizers`), ByteLevel pre-tok + decoder, dense ids 0..V-1, `<\|endoftext\|>` pinned to id 0. |
| `prepare_data.py` | Tokenize a text file → packed `uint16` memmap `.bin` with `<\|endoftext\|>` doc separators. |
| `model.py` | Llama decoder matching the engine exactly: RMSNorm, RoPE (rotate_half), GQA, SwiGLU, SubLN, tied embeddings, 7× `BitLinear` QAT. |
| `train.py` | CPU pretraining loop: AdamW, two-stage or cosine LR, grad clip, checkpoint/resume, val-PPL anchor + tripwire, `--config` JSON support. |
| `export_hf.py` | Export a checkpoint to an HF dir the repacker accepts (`--source-packing unpacked`). Refuses fp checkpoints. |
| `roundtrip_gate.py` | **The step-0 gate.** Train briefly → torch PPL → export → repack → engine PPL → PASS/FAIL. |
| `param_count.py` | Build the model a config JSON describes; print the exact runtime param count. |
| `configs/` | Run configs (JSON of `train.py` argparse dests; CLI flags override; `_`-keys are comments). |
| `launch_m7a_twin.sh` | Guarded systemd-run launcher for the full M7a twin run (refuses while `aegis-ppl-rerun` is active or disk < 8 GB). |

## Engine-parity contract (verified against aegis-core, 2026-07-17)

`model.py` is built to reproduce the engine's forward pass numerically:

- **RMSNorm** `y = x / sqrt(mean(x²) + eps) · w` (`ops.rs rmsnorm_scalar`).
- **RoPE** HF-Llama rotate_half, `freq = theta^(-2d/head_dim)` (`attention.rs`);
  half-split, **not** interleaved.
- **SiLU** `x·sigmoid(x)`; **SwiGLU** `down(silu(gate)·up)`.
- **GQA** `kv_head = q_head // (n_heads / n_kv_heads)`.
- **SubLN** (BitNet b1.58): `attn_sub_norm` on the concatenated attention output
  **before** `o_proj`; `ffn_sub_norm` on `silu(gate)·up` **before** `down_proj`.
  Default ON.
- **BitLinear** (all 7 projections), QAT with straight-through estimator:
  - weight, per-tensor: `gamma = mean(|w|)`, `w_q = round(w/gamma).clamp(-1,1)`,
    `w_eff = w + (w_q·gamma − w).detach()`.
  - activation, per-token absmax int8 (mirrors `ops.rs quantize_activations_int8`,
    which is a **default** engine feature): `s = 127/absmax`, round, clamp
    `[-127,127]`, dequantize. Applied to each BitLinear's **input**, exactly where
    the engine applies it (rmsnorm → quant_act → ternary matmul).
- Embeddings / norms / LM head are **fp32** in tinybit; the engine reads them as
  **BF16** (and the LM head is the tied embedding table). This is the main,
  documented source of torch-vs-engine drift, bounded by the 3 % gate tolerance.
- **No biases** anywhere; embeddings tied to the LM head.

### The scale convention (has caused two prior bugs — get it right)

The engine **multiplies** the ternary dot by a stored per-tensor scale.
`repack_ternary.py` writes the **reciprocal** of the `weight_scale` it reads. So
`export_hf.py` stores `weight_scale = 1/gamma` (divide convention); the repacker
writes `1/(1/gamma) = gamma`; the engine multiplies by `gamma`. Effective weight
`= w_q · gamma`, identical to the QAT forward. `export_hf.py` asserts the
round-trip `1/stored_scale ≈ mean(|w|)`.

### Divisibility

`hidden_size`, `intermediate_size`, and `kv_dim = (hidden/heads)·kv_heads` must all
be divisible by 4 (the repacker packs 4 ternary weights per byte along the input
dim). `TinyBitConfig.validate_engine_constraints()` enforces this.

## How to run

```bash
source /home/killboxincorporated/ranger-venv/bin/activate
cd /home/killboxincorporated/model-lab/tinybit

# 1. tokenizer (byte-level BPE); --max-bytes streams a subset for speed
python3 train_tokenizer.py --vocab-size 4096 --max-bytes 40000000

# 2. packed training data (uint16 memmap)
python3 prepare_data.py --max-bytes 150000000 --out train.bin

# 3. THE GATE: train briefly → torch PPL → export → repack → engine PPL → verdict
python3 roundtrip_gate.py            # writes roundtrip_gate.log, exits 0 on PASS

# --- or a longer standalone pretraining run ---
python3 train.py --steps 2000 -B 8 -T 512 --layers 8 --heads 8 --kv-heads 4 \
    --data train.bin --ckpt checkpoints/tinybit.pt          # resume with --resume

# --- export any checkpoint by hand, then repack + eval ---
python3 export_hf.py checkpoints/tinybit.pt /path/export_dir
python3 /home/killboxincorporated/aegis-forge/repack_ternary.py \
    /path/export_dir /path/artifacts --max-seq 512
/home/killboxincorporated/aegis-eval/target/release/aegis-eval \
    /path/artifacts/MODEL.SAF /path/artifacts/EMBED.BIN /path/artifacts/VOCAB.BIN \
    heldout.txt 512 --sample
```

## Gate result (2026-07-17, this machine)

Config: `hidden=256 inter=640 layers=4 heads=4 kv=2 vocab=4096 ctx=512`, 3.81M
params. 250 steps, B=8, T=512, lr=3e-3, 2 threads, `nice -n 10`. Held-out slice:
1907 ASCII chars → 480 tokens (both stacks agree exactly).

| quantity | value |
|---|---|
| torch QAT teacher-forced PPL (480 tok, 479 preds) | **77.5159** |
| engine PPL (`aegis-eval --sample`, 480 tok) | **77.6730** |
| relative diff | **0.20 %**  (tolerance 3 %) |
| token-count parity (torch vs engine) | **480 == 480** |
| peak RSS | 1.88 GB |
| wall clock (train + export + repack + eval) | 574 s |
| **verdict** | **>>> GATE PASS <<<** |

Full transcript in `roundtrip_gate.log`; repacked artifacts in
`gate_work/artifacts/` (`MODEL.SAF` 73 tensors / 707,900 B, `EMBED.BIN`
4096×256 BF16, `VOCAB.BIN` 4096 tokens / 3839 merges). Reproduce with
`python3 roundtrip_gate.py` (deterministic, seed 1337).

The 0.20 % gap is the expected fp32-vs-BF16 embedding/norm/LM-head drift and sits
far inside tolerance: the train → export → repack → engine chain is proven
end-to-end.


## M7 / M7a twin experiment (full-precision mode)

`TinyBitConfig.linear` selects the weight-precision path of the 7 projections:

- `"bitlinear"` (default) — BitLinear ternary QAT + per-token int8 activation
  fake-quant, exactly as before. Engine-exportable.
- `"fp"` — plain **fp32 `nn.Linear`**, no bias, no quantization anywhere.
  **Not** engine-exportable: `export_hf.py` refuses fp checkpoints with a clear
  error (ternary packing is meaningless for fp weights).

**Single-variable rule.** The switch changes *only* the projection precision.
RMSNorm, RoPE, GQA, SwiGLU, SubLN, tied embedding head, and the init
distribution (normal, std 0.02) are the identical code path in both modes
(`make_linear` in `model.py`), so a ternary arm and an fp twin differ in exactly
one variable: weight precision. Do not add mode-specific architecture tweaks.

**Per-arm recipes.** Each arm gets its own sane recipe — fp recipes and
ternary-QAT recipes legitimately differ, and handicapping either arm with the
other's recipe would corrupt the comparison:

| arm | schedule (`--sched`) | weight decay |
|---|---|---|
| ternary (`m7_ternary.json`) | `two_stage` (warmup → cosine to knee at 50%, second cosine) | 0.1, dropped to 0 in stage 2 (BitNet tip) |
| fp twin (`m7a_twin.json`) | `cosine` (warmup → single cosine to 0.1×peak) | 0.1, constant |

Both use AdamW betas (0.9, 0.95), grad clip 1.0 (`make_optimizer`; norms and
embeddings never decay).

**Configs** (exact runtime counts from `param_count.py`, logged in
`logs/m7_param_counts_2026-07-18.log`):

| config | mode | hidden/inter/layers/heads/kv | params |
|---|---|---|---|
| `configs/m7_ternary.json` | bitlinear | 384/1024/7/6/2 | **14,171,392** (14.17M) |
| `configs/m7a_twin.json` | fp | 256/704/6/4/2 | **6,529,920** (6.53M, ~0.46× of M7) |

```bash
# param counts (runtime, not formulas)
python3 param_count.py configs/m7_ternary.json configs/m7a_twin.json

# smoke: config supplies defaults, CLI overrides win
python3 train.py --config configs/m7a_twin.json --steps 200 --warmup 20 --threads 4

# full twin run (guarded; DO NOT start while aegis-ppl-rerun is active)
./launch_m7a_twin.sh
```

**Smoke result (2026-07-18, this machine, aegis-ppl-rerun eval active
throughout — see `logs/m7a_twin_smoke_2026-07-18.log` for every number):**
twin 200-step run on train_8k.bin: loss 8.61 → 4.32 (first-10 vs last-10 mean),
SIGKILL at step ~106 + `--resume` from the step-100 checkpoint continued the
loss curve with no jump (post-resume minus pre-stop = −0.12); val PPL
192.2 → 114.8. Throughput measured: **1174–1310 tok/s at 4 threads**, but only
**620 tok/s in the 8-thread burst** — 8T was *slower* than 4T under that day's
conditions (HT oversubscription against the running eval + swap traffic).
Re-check 8T vs 4T once the box is idle before committing to a multi-day run.
M7 ternary arm: 10-step crash test passed (loss 9.12 → 6.96), 472 tok/s at 4T
as a 10-step sample only.

## Known caveats

- **Engine ctx window clamps PPL.** `aegis-core calculate_perplexity` truncates
  the token sequence to `max_position_embeddings`. With the gate's `ctx=512`, the
  held-out slice must be ≤ 512 tokens. The gate therefore scores ~480 tokens, not
  the "~2000" the spec mentions for larger-window configs — a deliberate deviation
  forced by the engine window. Raise `--ctx` (and re-export) to score longer
  slices; the arena/KV grow with it.
- **BF16 embeddings/norms/LM-head drift.** torch computes in fp32; the engine
  reads embeddings, norms, and the tied LM head as BF16. This is the dominant term
  in the torch-vs-engine PPL gap and is expected, not a bug.
- **Tokenization parity is by construction, not by proof.** The engine reimplements
  byte-level BPE and only *approximates* the HF ByteLevel regex pre-tokenizer
  (whitespace-run handling differs). The gate writes the held-out slice as an ASCII
  text file, lets the engine tokenize it, and checks the engine's token count
  equals tinybit's before comparing PPLs. Clean single-spaced English (TinyStories)
  matches; heavy/irregular whitespace can drift. Keeping `<|endoftext|>` at id 0
  (a special, never in merges) avoids the engine's id-0-merge-drop entirely.
- **Non-ASCII is dropped** by the engine's evaluator; held-out slices are
  ASCII-filtered on both sides so the two stacks see identical bytes.
- **CPU-only, small.** Trained with `torch.set_num_threads(2)` and `nice -n 10`
  so it coexists with other jobs on this box. Measured peak RSS for the gate was
  1.88 GB (dominated by the torch training graph at B=8/T=512); drop `--batch` or
  `--block` if you need a tighter footprint.
- **Disk hygiene.** This box's data volume was at 0 bytes free when the gate was
  first built; ~2 GB was reclaimed by clearing the regenerable pip wheel cache
  (`~/.cache/pip`) — no user data, models, or datasets were touched. Keep an eye
  on free space before large runs; `gate_work/` and `train.bin` are regenerable
  scratch and safe to delete.
