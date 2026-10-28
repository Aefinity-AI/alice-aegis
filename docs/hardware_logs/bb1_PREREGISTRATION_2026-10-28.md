# bb-1 EXTERNAL-ENGINE BASELINE — PRE-REGISTRATION (claims row C047)

**Written 2026-10-28, BEFORE any bb-1 measurement.** Rules, bands and exclusions below are fixed now and apply to both arms identically. Whatever is measured is reportable, including NOT-MET.

## 1. Question
On one legacy x86 host, what is wall-clock decode throughput of the ALICE integer-path decode relative to an external ternary engine (bitnet.cpp) on the same BitNet-b1.58-2B-class model, same prompt, same generated-token count, same OS condition?

## 2. Arms
| arm | stack |
|---|---|
| **B** | bitnet.cpp pinned at commit 0b341e5, unmodified-source AVX2 build (the "unpatched" build in /home/cm/legs/w2-bitnet-box1), official i2_s GGUF, sha256 prefix 4221b252 (full hash recorded at pre-run and must match the pinned value before any run) |
| **A** | ALICE integer-path decode (aegis-core, commit pinned in the run manifest before boot 1), same model weights converted by the repo's own pipeline, driven by the NEW wall-clock instrument (section 4) |

Patched bitnet.cpp build, if used, is a separate declared arm B' reported beside B, never substituted for B.

## 3. Protocol (locked)
- Host: w2 (Dell, i5-5200U, Broadwell-U, AVX2), Linux userspace under systemd-run, no other lab job, governor performance, bd-PROCHOT/throttle state read back before each boot.
- Identical across arms: prompt text (fixed string recorded in manifest), prompt token count, generated token count N_gen = 64 (greedy, temperature 0), same OS condition (same kernel, same boot, same cgroup limits, same core pinning).
- Threads: 1 and 4, both reported; no pooling across thread counts.
- Boots per arm: per oscost_PREREGISTRATION_2026-07-30 = interleaved A, B, A, B ... N = 5 boots per arm per thread count minimum, fresh process per run; within a boot, 1 discarded warm-up run then 7 counted fresh-process runs.
- Load average, per-core MHz and thermal-throttle counters sampled before and after every run and logged. Raw logs copied verbatim to docs/hardware_logs/, never edited (Rule C).
- Work identity gate: both arms must emit the same prompt token count and N_gen tokens. Output-text identity is reported but not required (different quantization kernels may differ); if prompt/N_gen differ, no timing comparison is valid.

## 4. Instrument
Decode tokens/s = N_gen / wall seconds of the decode loop only (monotonic clock, excludes model load, includes sampling), measured identically on both arms. Arm B uses bitnet.cpp's own eval timing AND an outer wall clock around the process; the outer-wall value is primary. Arm A's instrument is new code added to a bench harness; it is reviewed and committed BEFORE boot 1, and its first output is a dry-run that is not counted.

## 5. Metrics
Primary: per boot, per thread count, median over counted runs of decode tok/s; ratio R = tok/s(A) / tok/s(B), pooled median over boots with percentile-bootstrap CI over boots (10,000 resamples). Secondary: prefill tok/s, within-boot CV, peak RSS, clock %-of-nominal.
Clock parity gate: if adjacent A/B boots differ in mean MHz by >2%, those boots are excluded and the exclusion reported.

## 6. Rules fixed in advance (symmetric)
Evaluated separately at threads=1 and threads=4, on the pooled R with its CI:
| outcome | condition | statement allowed |
|---|---|---|
| MET | CI lower bound >= 0.90 | "ALICE decode is within 10% of or above bitnet.cpp at this thread count on this host and model" |
| NOT-MET | CI upper bound < 0.90 | "ALICE decode is more than 10% slower than bitnet.cpp ..." reported as-is |
| INCONCLUSIVE | otherwise | "not distinguishable at N=5/boots" ; extension of N must be declared before unblinding |
Faster-than direction: if CI lower bound > 1.00, report as ahead, same wording care. Valid runs only after the work-identity gate. Failed boots count and are listed; no run is dropped except by the stated parity/identity gates. No threshold, prompt or arm is changed after data.

## 7. Provenance (Rule B)
Every number in any ledger row cites a raw log path, git commit of each engine, GGUF/weights hash, host, date. Ledger batch and both logs are new files under docs/hardware_logs/.

## 8. NOT claimed
Not a claim about any other host, CPU, model size or quantization; not multi-host; not energy; not quality/perplexity; not a comparison to bitnet.cpp's patched or GPU paths; not a bare-metal result; no tok/s from emulators. Result applies to Broadwell-U AVX2, one model, N_gen=64.
Banned wording: first, novel, world's, unique.

## 9. Pre-result clarifications (2026-10-28, before any bb-1 measurement; design unchanged)
- **N_gen / rate definition.** N_gen = 64 generated tokens on both arms (arm B run with --ignore-eos, arm A ignore_eos=true). Token 1 is produced at the end of prefill on both engines, so the timed decode loop is 63 forward steps (bitnet.cpp reports "eval time ... / 63 runs"). Decode tok/s = 63 / decode seconds (ALICE field `tok_s_decode`; bitnet.cpp eval-time rate), the same definition on both arms. `tok_s_ngen` (64/s) is logged but not used. Section 4's "N_gen / wall seconds" is read in this sense.
- **Prompt.** Fixed string in scripts/bb1_prompt.txt (sha256 in manifest). Prompt token count must be equal on both arms; the runner checks every run and aborts with GATE_FAIL on any mismatch of prompt tokens or N_gen (gate.log lists every run).
- **Arm B flags.** `llama-completion -m GGUF -p PROMPT -n 64 -t T --temp 0 --ignore-eos -no-cnv` (clarification 2026-10-28: the llama-cli target of the pinned tree is conversation-only and ignores -no-cnv; llama-completion is built from the same tree at 0b341e5, build dir build-bb1, same I2_S kernels; the flags are the same).
- **Open reading (not resolved here).** Section 4 says outer process wall is primary for B, but that includes model load and prefill, which arm A excludes. The runner records both; which is primary for R must be fixed before boot 1 (the like-for-like value is bitnet.cpp eval time).

- **s9 clarification (2026-10-28, pre-registration, before any counted run).** (1) Arm-B primary metric = bitnet.cpp eval-time decode tok/s (63 runs), like-for-like with ALICE `tok_s_decode`; arm-B outer wall (incl. load+prefill) is recorded as secondary, not primary. Section 4's outer-wall wording is superseded in this sense. (2) ALICE `--n-gen K` means K decode steps (K+1 tokens emitted, measured on w2: n-gen 64 gave n_gen_emitted=65), so the runner passes `--n-gen NGEN-1` = 63 to emit 64 tokens, matching arm B.
