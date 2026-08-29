# C2a — BitNet-2B ARC-Easy MC baseline: run map (2026-08-29)

Pre-reg: claudius-maximus `state/reports/2026-08-29-COMPACT-AXIS-PREREG.md` (Leg C2a) +
Amendment 1 (capability numbers measured on the FULL-128k-vocab forge; pruned n=119 is
supplementary).

## Artifacts

- **Pruned-vocab (A35/A36 baseline, 50,256-of-128,256):** `aegis_pruned_model.safetensors`
  (sha256 `53586af0...`), `aegis-forge/embed.bin` (`e32b99a2...`), `aegis-forge/vocab.bin`
  (`5bde1b03...`) — symlinked into this worktree from `~/projects/alice-aegis` (gitignored,
  restored via `~/dev-setup/restore-weights.sh`).
- **Full-vocab (this task, no `--llama3-prune`):** built with
  `python3 aegis-forge/repack_ternary.py ~/projects/alice-aegis aegis-forge/fullvocab_out
  --source-packing hf1bitllm` from the same base checkpoint (`~/projects/alice-aegis/model.safetensors`,
  hf1bitllm-packed) →
  `aegis-forge/fullvocab_out/{MODEL.SAF,EMBED.BIN,VOCAB.BIN}` (gitignored, not checked in;
  MODEL.SAF 521,953,187 B / EMBED.BIN 656,670,720 B / VOCAB.BIN 128,256 tokens, 280,147
  merges). Reproducible from the command above (deterministic given the same source
  checkpoint + tokenizer.json).

## Items files

- `model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42_bitnet.jsonl` — pruned-vocab
  remap, **119/570** ARC-Easy validation ids representable (451 dropped, logged in
  `..._bitnet_dropped.jsonl`) — the pruning's crude id<50000 cutoff makes most
  science-question vocabulary unrepresentable, not just non-ASCII text.
- `model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42_bitnet_fullvocab.jsonl` —
  full-vocab identity remap, **570/570** representable (asserted by
  `mc_prep_bitnet.py --full-vocab`, zero drops).
- Both built by `model-lab/scripts/mc_prep_bitnet.py` (`--full-vocab` flag selects the
  identity remap over the dense unpruned id space instead of the pruned remap), same
  ARC-Easy validation rows (sha256 `ed890ff1...`) as the Falcon-E-1B/3B M16/M3 baseline
  items file (`arc_easy_val_n570_seed42.jsonl`) — ids are directly comparable across files.

## Engine

`aegis-eval/src/mc.rs` / `main.rs`: `--mc <items> --mc-out <out> [--mc-cis-full [items]
--mc-cis-full-out <out>]` — one process, one model load, float MC pass then (if requested)
the CIS-1 FULL-INTEGER MC pass on the same items via `CisEngine::calculate_perplexity_int`.

## Runs (systemd --user units, nice 10, sequential to bound peak RAM to one resident model)

1. **`c2a-mc`** (launched 09:20 CDT): pruned-vocab, n=119, float then full-int, single
   process.
   - float out: `model-lab/data/evals/arc_easy/results/bitnet2b_float_n119.jsonl`
   - full-int out: `model-lab/data/evals/arc_easy/results/bitnet2b_fullint_n119.jsonl`
   - raw stdout: `docs/hardware_logs/.c2a_mc_n119_run.log.tmp` (rename once complete)
2. **`c2a-mc-fullvocab`** (launched 10:26 CDT): waits for `c2a-mc` to exit
   (`aegis-forge/run_c2a_fullvocab.sh` polls `systemctl --user is-active --quiet
   c2a-mc.service` every 60s), then runs full-vocab, n=570, float then full-int, single
   process.
   - float out: `model-lab/data/evals/arc_easy/results/bitnet2b_float_n570_fullvocab.jsonl`
   - full-int out: `model-lab/data/evals/arc_easy/results/bitnet2b_fullint_n570_fullvocab.jsonl`
   - raw stdout: `docs/hardware_logs/.c2a_mc_fullvocab_n570_run.log.tmp` (rename once
     complete)

Both `--mc-out`/`--mc-cis-full-out` files are JSONL: one `{"header":...}` line, one
`{"id":...,"pred_raw":...,"pred_norm":...,"correct_raw":...,"correct_norm":...}` line per
item, one `{"summary":true,"n":...,"acc":...,"acc_norm":...}` line at the end.

Per-item wall-clock in the raw logs is incidental engine output, not a reported result
(Rule A — no timing claims in the ledger row or any external document drawn from these
runs).

Falcon-E-1B/3B per-item predictions for McNemar comparison:
`docs/hardware_logs/m3_mc_full570_falcon_e_{1b,3b}_2026-07-18.log` (console logs with
`[n] <id> pred_raw=.. pred_norm=.. gold=..` lines; no separate JSONL results file exists
for those two runs).
