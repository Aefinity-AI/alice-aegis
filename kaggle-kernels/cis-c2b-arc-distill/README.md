# cis-c2b-arc-distill

Kaggle kernel source for pre-registered leg **C2b** (compact-capable axis) —
logit-distillation SFT of BitNet-2B-4T toward a Llama-3.2-3B-Instruct-family
teacher on ARC-Easy/ARC-Challenge/OpenBookQA (open-license only).

PRIVATE per the 2026-08-28 publish policy (new technique / product-shaped).
This branch (`cm/c2b-distill-kernel`) is a local/private branch — do not
merge to alice-aegis main, do not open a PR, do not add to the public paper.

Full writeup, data/eval plan, teacher cascade, memory-fix history (v1 OOM ->
v2 OOM -> v3 two-phase redesign), gate criteria, and re-forge/eval commands:
`claudius-maximus/state/reports/2026-08-29-C2b-kernel.md` and
`claudius-maximus/state/reports/2026-08-29-COMPACT-AXIS-PREREG.md` (leg C2b
+ Amendment 1).

Pushed to Kaggle as `aefinityaiinc/cis-c2b-arc-distill` (private, 2x T4 +
internet). This copy (version 3 source) is committed here for off-box
provenance per reset-guard doctrine; it was authored and pushed to Kaggle in
an earlier session on 2026-08-29 and was already `RUNNING` (confirmed again
at commit time via `kaggle kernels status`) — this commit does not restart
or modify the live run.

Fetch outputs once complete:
```
kaggle kernels status aefinityaiinc/cis-c2b-arc-distill
kaggle kernels output aefinityaiinc/cis-c2b-arc-distill -p ~/projects/kaggle-kernels/cis-c2b-arc-distill/out
```
