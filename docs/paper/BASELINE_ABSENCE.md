# Baseline absence statement: external-engine decode comparison

Stage 1 deliverable D4. Written 2026-10-27 on branch `stage1/system-paper`
(base `d3700f0d`). No measurement was taken for this document. Nothing under
`docs/hardware_logs/` was added or changed (Rule C).

## 1. What is absent

No like-for-like `bitnet.cpp` (or `llama.cpp`) decode-throughput log on a named
physical box exists in this repository. Claims-ledger row **C047** (VERIFIED,
absence) records this with the check

```
git ls-files docs/hardware_logs | grep -iE 'bitnet.?cpp|ggml|llama|ik_llama'   # must print nothing
```

which printed nothing at `d3700f0d` and prints nothing at this commit. That
check matches file names only. A content search of the same tree,

```
git grep -n -iE 'bitnet.?cpp|ggml|llama|ik_llama' -- docs/hardware_logs
```

returns four hits at this commit, all the same comment line in
`docs/hardware_logs/2026-10-16-cis1-2b-x86/` (`job.sh` and the three `run.sh`
scripts): a note that those ALICE 2B receipt runs were "paired with" external-engine
rows taken on another host. The paired rows, and any external-engine output, are not
in this repository; that tree's `README.md` states that it is not a benchmark and that
wall times are not quoted. The absence claim, stated so that it can be falsified, is:
no file in this repository prints an external-engine decode-throughput figure.

The gap is tracked in the repository at `program/ROADMAP.md:170` (open item),
`program/AEGIS_OS_DESIGN.md:111` (gate G1: external-engine A/B still OPEN) and
`docs/TECHNICAL_REPORT.md:374` (future work). No row in
`program/RESEARCH_LEDGER.md` carries this comparison (see also claims-ledger
row C091). The rows that govern this document are C046 and C047.

## 2. Why no measurement was taken for the paper

Stage 1 writes up what exists. A like-for-like pair is two new measurements on
the same named box under the same operating-system condition, and the ALICE
side of that pair has no instrument today:

- The `aegis-linux` host binary (`aegis-linux/src/main.rs`) takes positional
  arguments plus `--parity`. It has no integer-path mode, so its
  `Wall time ... tok/s` line reports the f32-activation path.
- The integer-path entry points (`aegis-linux/examples/cis_decode.rs`,
  `cis_witness.rs`, the `arm-digest` workflow) emit digests and token counts
  and no wall-clock field. Claims-ledger row **C046** (VERIFIED, absence):
  token-level throughput on the integer path is unmeasured; its check is

```
grep -rilE 'CIS_DECODE|mode=fullint' docs/hardware_logs | xargs -r grep -ilE 'tok/s|tokens/s|ticks/token'   # must print nothing
```

Producing the ALICE side therefore means a new timing instrument plus a
preregistered run on physical hardware (Rules A and B). That is new capability,
outside Stage 1. The paper keeps the absence as a limitation.

Figures that do exist and must not be mistaken for this baseline:

- The README decode figures (rows C003, C004) come from the f32-activation
  build on the i5-10210U development host. No external engine ran.
- The per-run `tok/s` lines in the Dell and HP minimal-Linux boot logs come
  from the `aegis-linux` harness binary described above, which has no
  integer-path mode. No external engine ran.
- The ring-0 versus minimal-Linux decode comparison (A22, the `oscost_*` logs,
  model pinned in `docs/hardware_logs/oscost_ANALYSIS_2026-07-31.md`) has its
  Linux arm from that same harness and its ring-0 arm from the UEFI unikernel,
  whose GAUNTLET lines report `path=AVX2+FMA`, a float path. Both arms are
  float; neither is an external engine.
- The kernel-level integer-versus-float ratios on the Dell and HP (A26, A27;
  rows C037 to C045, C092, C093) are matmul-kernel captures inside the same
  logs, not token-level decode. No external engine ran.

## 3. What exists instead, and why it is not a baseline

`.github/workflows/e17-2-bitnetcpp-crossisa.yml` is a `workflow_dispatch`,
identity-only cross-ISA check of upstream `bitnet.cpp` (pinned commit, official
`i2_s` GGUF) on GitHub-hosted x86_64 and aarch64 runners. Its header reads
"IDENTITY ONLY — Rule A: no timing numbers"; it strips server timings by design
(`scripts/e17_2_strip_timings.py`); it writes its logs on the runner and uploads
them as workflow artifacts; it has no commit step, so no log from it exists
under `docs/hardware_logs/` (the C047 check above). GitHub-hosted runners are
shared cloud machines, not named physical boxes (Rule A). Nothing in it runs
ALICE, and nothing from it is a number with a Rule B parent.

## 4. What the paper may say

- Identity and conformance claims from VERIFIED claims-ledger rows only (D1).
- External throughput figures from Wang et al. (arXiv:2410.16144) and the
  `bitnet.cpp` report (arXiv:2502.11880) as citations of those papers' own
  measurements on their own machines, never as a comparison with ALICE and
  never in the same sentence as an ALICE figure.
- The existing limitation wording (no token-level throughput figure for the
  integer path) at `docs/paper/paper.md` lines 131, 409, 516 and 899: keep it,
  do not duplicate it.
- In Limitations: a like-for-like external-engine comparison on a named
  physical box has not been run; this document states why.
- The kernel-level ratios (A26, A27) only if their rows are VERIFIED, labelled
  kernel-level and with the machine named.

## 5. What the paper may not say

- Any ALICE-versus-`bitnet.cpp` or ALICE-versus-`llama.cpp` throughput,
  latency, energy or speedup figure or ratio, in either direction.
- Any number from a GitHub-hosted runner or from QEMU/TCG (Rule A).
- Any characterisation of `bitnet.cpp` behaviour or output taken from the
  cross-ISA workflow.
- "Parity with", "faster than" or "comparable to" an external engine.

## 6. Lines in `docs/paper/paper.md` for D5 to adjust (line numbers at `d3700f0d`)

| Lines | Content | D5 action |
|---|---|---|
| 20, 86, 844 | AVX2 integer-versus-float kernel ratio (A27) | keep only if the row is VERIFIED; label kernel-level, Dell |
| 147 | "parity with FP16 LLaMA baselines" | cite as the model authors' claim |
| 154 to 157 | Wang et al. speedup ranges | citation wording, their machines |
| 266 | `bitnet.cpp` reference entry | keep |
| 723 to 728, 763 to 767, 796 to 804 | Cost section and summary table (A22, A26, A27) | label each row f32-activation harness or kernel-level; none is an external-engine comparison |
| 776 | NOMINAL speedup caveat | keep |
| 1067 | table row for A27 | kernel-level label |
| 131, 409, 473, 516, 899 | existing limitation wording | keep |

## 7. How this absence is closed later (outside Stage 1)

A registered, preregistered run on the Dell Inspiron 15 (i5-5200U, Broadwell-U):
a pinned `bitnet.cpp` build with the official `i2_s` GGUF, and an ALICE
integer-path decode with a wall-clock instrument added for that purpose; same
prompt, same token count, same operating-system condition (bare-metal ALICE
against Linux `bitnet.cpp` is not like-for-like), thread counts stated, boots
per arm as in `docs/hardware_logs/oscost_PREREGISTRATION_2026-07-30.md`, both
logs as new files under `docs/hardware_logs/` (Rule C), ledger rows only after
`scripts/verify-figures.sh`. Until then C047 stands and this document is the
paper's statement of absence.
