# MECH v1 analysis — H1 console / H2 turbo-bin decomposition of Band 3

**Date:** 2026-08-01. **Boot machine:** Dell Inspiron 15 (i5-5200U,
Broadwell-U), bare metal, hands-off per
`docs/RUNCARD_MECH_2026-08-01.md`. **Data pull machine:** dev Chromebook
(i5-10210U, crosvm), sticks mounted read-only.

## Provenance

| Item | Value |
|---|---|
| U stick BOOTLOG.TXT | grew 44,612 → 53,439 B; first 44,612 B md5 `6906af8068ee8acf232787e7646666fe` == archived `oscost_U_BOOTLOG_2026-07-31.txt` (prefix untouched) |
| New U evidence | `docs/hardware_logs/mech_U_BOOTLOG_2026-08-01.txt` (8,827 B, md5 `f4aff7606e2e59bc35ff97e1070f199c`) |
| L stick BOOTLOG_LINUX_ARM.txt | grew 113,419 → 242,955 B; first 113,419 B md5 `17c802798759f26990b456fd5cfd0959` == archived `oscost_L_BOOTLOG_2026-07-31.txt` |
| New L evidence | `docs/hardware_logs/mech_L_BOOTLOG_2026-08-01.txt` (129,536 B) — **unpreregistered L replicate**, stability check only |
| Binary under test | `BOOTX64.EFI` md5 `431ff3a8246559a0b12e6f640dd86c0a` (pinned in runcard §1) |

## Bit-exactness gate (Rule D): PASS

All three prompts produced byte-identical RESPONSE text and equal token
counts across LOUD / QUIET / QUIET2 (85 / 257 / 214 tokens; verified
programmatically, not by eye). The run is valid.
(Footnote: "how are you today?" counts 257 callbacks against MECH_MAX=256 —
the callback tally appears to include one terminal fragment; identical in
all three passes, so it cancels in every contrast.)

`MSR_TURBO_RATIO_LIMIT raw=0x000019191919191b` → 1C bin = 0x1b = **27**,
2C bin = 0x19 = **25**. Matches the expected 27/25 for this part.

## Raw contrasts (ticks/token, invariant-TSC, from mech_U_BOOTLOG)

| Prompt | LOUD | QUIET | QUIET2 | H1 share | H2 share |
|---|---|---|---|---|---|
| hello alice (85 tok) | 1,650,870,240 | 1,650,538,760 | 1,529,135,891 | 0.020% | 7.355% |
| how are you today? (257) | 1,665,892,785 | 1,665,584,262 | 1,543,070,737 | 0.019% | 7.356% |
| continue (214) | 1,658,263,357 | 1,657,424,927 | 1,535,480,864 | 0.051% | 7.357% |

H1 share = (LOUD−QUIET)/LOUD; H2 share = (QUIET−QUIET2)/QUIET.
Clock: LOUD/QUIET **113%**, QUIET2 **122%** of nominal (engine-attached
perf-snapshot ratio, Rule A corollary satisfied).

## H1 — console-in-the-stopwatch: DEAD

The preregistered share formula gives 0.02–0.05%, but that denominator is
not the phenomenon's denominator: the MECH loop times `process_intent`,
which costs 1.65–1.67 G ticks/token — **119–250× the 07-31 gauntlet decode
path** on the same machine (6,590,463 / 12,457,798 / 13,890,704 ticks/token,
`oscost_U_BOOTLOG_2026-07-31.txt` lines 423/430/440). The honest H1 verdict
therefore uses **absolute console cost per token** (LOUD−QUIET):

| Prompt | Console ticks/token | ÷ 07-31 U gauntlet ticks/token | Share of protocol cost |
|---|---|---|---|
| hello alice | 331,480 | 6,590,463 | **5.0%** |
| how are you today? | 308,523 | 12,457,798 | **2.5%** |
| continue | 838,430 | 13,890,704 | **6.0%** |

Even the worst case (≤6%) cannot produce the +89% per-token cost gradient
(6.59 M → 12.46 M ticks/token) that motivated H1, nor the −39.3% pooled
Band-3 gap. **The console hypothesis is dead.** Decision table row 3 of the
runcard fires: the gradient evidence was misleading; proceed to H3.

## H2 — idle-core turbo bin: CONFIRMED, fully clock-explained

- Time ratio QUIET2/QUIET = **0.9264** on all three prompts (0.92645 /
  0.92645 / 0.92642); the ratio-bin prediction 25/27 = **0.9259**.
- `TURBO_DIAG mech-postpark`: cur_ratio **25 → 27** after MWAIT-C6 parking
  of 3 APs; engine clock reading 113% → 122% of nominal.
- Cross-arm corroboration: the L replicate ends `final cur_freq: 2700000
  kHz` — Linux reaches ratio 27 naturally because cpuidle parks idle cores;
  the unikernel only gets it by explicit AP-PARK.

The entire 7.36% H2 effect is the 1-core turbo bin. Mechanism understood.
**Adopt AP-PARK in the production boot path** (it is already in this
binary; it should run at boot, not only inside MECH).

## Gap accounting

With Δ_OS = −39.3% pooled (L = 0.607 × U, `oscost_ANALYSIS_2026-07-31.md`)
and parking improving U by factor 0.9264: residual gap =
(0.607 − 0.9264) / 0.9264 = **−34.5%**. Parking closes ~5 points of the
39.3-point gap; the rest is **unexplained**. Per the runcard decision
rules: **H3 next — UEFI memory caching attributes (MTRR/PAT audit)** —
and −39.3% (now −34.5% residual) stays **unpublishable** until explained.

## Side finding MECH-A1 (new, needs its own investigation)

The `process_intent` path costs 119–250× the gauntlet harness decode path
per token on identical hardware in the same boot lineage (1.65 G vs
6.6–13.9 M ticks/token). MECH's absolute throughput is therefore **not
comparable** to any protocol number; only its within-boot contrasts are
valid (which is all this analysis uses). Why that path is two orders of
magnitude slower is a real question — likely no KV reuse / full re-prefill
per emitted token — but it is **not investigated here**.

## L replicate (unpreregistered — stability check only, no ledger claim)

Decode-only medians, n=8 fresh-process runs each, from
`mech_L_BOOTLOG_2026-08-01.txt`: hello alice **367.83 tok/s**, how are you
today? **320.71 tok/s**, continue **332.60 tok/s**. Wall-derived cost at
2.2 GHz nominal: 5.98 / 6.86 / 6.61 M ticks/token — consistent with the
07-31 L arm (e.g. 7.27 M for the long prompt), ~6% faster (this boot held
2.7 GHz throughout). The Band-3 sign is unchanged by replication.

## Verdicts

1. **H1 dead** — console writes are ≤6% of the 07-31 per-token cost;
   cannot explain the gradient or the gap.
2. **H2 real and closed** — 7.36% = the 25→27 turbo bin; park APs at boot
   from now on.
3. **Band 3 residual −34.5% unexplained** — H3 (MTRR/PAT) is next; still
   unpublishable.
4. **MECH-A1** — process_intent is 119–250× slower than harness decode;
   separate investigation required.
