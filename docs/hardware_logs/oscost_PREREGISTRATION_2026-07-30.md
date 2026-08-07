# PAIRED OS-COST MEASUREMENT — PRE-REGISTRATION

**Written 2026-07-30, BEFORE any paired boot has occurred on real iron.**
Committed ahead of the result so the conclusion cannot be chosen after seeing
the data. The literature sweep of 2026-07-30 (session record) found no
published rigorous bare-metal-vs-Linux LLM decode comparison on identical
hardware; whatever this measures is reportable, including a null.

## 1. Question

What does an operating system cost (or buy) single-thread ternary LLM decode
on legacy x86 iron, holding engine, weights, machine, and clock state fixed?

## 2. Arms

| arm | stack | boot vehicle |
|---|---|---|
| **U** | aegis-uefi unikernel, UEFI Boot Services, no OS | existing M7 USB partition (GATE-1 stick) |
| **L** | minimal Linux: Debian kernel 6.12.94+deb13-amd64 (stock binary), ~10MB initramfs, busybox, no systemd/no daemons/no swap/no other processes; `mitigations=off`, governor `performance` | scripts/linux-arm payload (UKI BOOTX64.EFI), separate USB stick |

Both arms: same physical machine (Dell Inspiron 15, i5-5200U, Broadwell-U,
AVX2), same engine source (aegis-core, git a249b2c lineage), same model
artifacts, single thread, bd-PROCHOT cleared with read-back before measuring
(unikernel STAGE 7 / Linux msrtool clearbit 0x1FC bit 0).

Artifact identity (md5, m7_final_gate_work/artifacts, pinned now):
```
53235f594ca3df50785cda6538d17075  MODEL.SAF
297315890a7fa2aa8efcf068240fa2d9  EMBED.BIN
03301400fff883d86b37520cbe135533  VOCAB.BIN
```

## 3. Protocol (locked)

- **Interleaved boots: U, L, U, L, … N = 5 boots per arm minimum** (10 boots).
  Interleaving controls thermal/time drift. Fresh power-on per boot.
- Arm U emits its gauntlet + the three PROMPT runs to BOOTLOG.TXT (unchanged
  firmware binary — the same one that produced
  m7_baremetal_prompts_postfix_2026-07-29.log). Arm L auto-runs
  L1 (7 fresh-process, 64 new tokens, "Once upon a time"),
  L2 (in-process n=20), L3 (the same three prompts as U) to
  BOOTLOG_LINUX_ARM.txt.
- Nothing on either stick is edited by hand, ever (Rule C). Sticks come back
  and both logs are copied verbatim into docs/hardware_logs/.

## 4. Primary estimand

Per boot b, per prompt p ∈ {"hello alice", "how are you today?", "continue"}:
r_{b,p} = decode ticks/token as printed by the engine.

**Δ_OS = pooled median over boots of (r^L − r^U) / r^U**, reported per-prompt
and pooled, percentile-bootstrap CI over boots (10,000 resamples).

Ticks are RDTSC on both arms; both engines print clock %-of-nominal. Ratio
parity gate: if per-boot clock ratios differ by >2% between adjacent U/L
boots, those boots are excluded and the exclusion reported (Rule A corollary).

## 5. Secondary estimands (reported, never headlined over the primary)

- Prefill cycles/token, same pairing.
- Within-boot variance: L2 in-process CV vs arm U's within-boot spread.
- **Output identity: the three prompts' generated token sequences must be
  IDENTICAL across arms** (greedy decode, same weights). This is the
  bit-exactness gate (Rule D): if outputs differ, work-identity is broken and
  NO timing comparison is valid until explained.
- Boot-to-gauntlet wall time per arm (coarse, from log ordering/user watch).

## 6. Bands — fixed in advance

| band | condition (pooled Δ_OS) | conclusion |
|---|---|---|
| 1 — null | \|Δ\| < 2% | OS cost is below noise: unikernel has no throughput case; its value is boot time/TCB/control only. Publish the null. |
| 2 — modest | 2% ≤ \|Δ\| < 10% | Real but modest; report with mechanism hypotheses; no architecture decision changes on this alone. |
| 3 — large | \|Δ\| ≥ 10% | Investigate mechanism BEFORE publishing (clock parity, SMM stealing, C-state/P-state policy, hidden throttle). A large number without a mechanism is not a result. |
| sign | Δ < 0 (Linux FASTER) | Report as-is. Plausible (OS P-state/turbo management can beat firmware defaults). Do not suppress. |

Significance: if the bootstrap CI on pooled Δ_OS spans zero, report "not
distinguishable from zero at N=5/arm" and (optionally) extend N — extension
must be declared before unblinding the extended data.

## 7. Disclosed limitations, in advance

1. Arm L is a PURPOSE-BUILT minimal Linux (mitigations=off, no daemons,
   initramfs-resident artifacts, performance governor). This measures the
   floor of what Linux costs, not what stock Ubuntu costs. A stock-distro arm
   is a separate future measurement; conclusions here must say "minimal
   Linux", never just "Linux".
2. Single-thread only. Multicore OS-cost is a different experiment (and no
   multicore number may be quoted anyway until A4's bench exists).
3. M7-class model (14.17M, 2.8MB, compute-heavy per byte relative to 2B).
   The 2B-model pairing (bandwidth-bound regime) is a follow-up; do not
   extrapolate this result to BitNet-2B.
4. RTC on the Dell may be wrong; logs are ordered by append sequence, not
   wall-clock claims.

## 8. Banned sentences (whatever the result)

- "The unikernel is Nx faster than Linux" (unqualified — arm L is minimal, not stock)
- "The OS costs nothing" (if CI merely spans zero — that is "not distinguishable at this N")
- Any tok/s or ticks figure from the QEMU correctness test of either arm (Rule A)
