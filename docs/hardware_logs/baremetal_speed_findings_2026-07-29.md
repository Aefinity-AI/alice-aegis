# Bare-metal speed findings — 2026-07-29

Workflow wf_601f0c1b-306: 11 agents, 248 tool calls, 1.14M tokens.

## Throttle verdict: **REAL_THROTTLE** (confidence HIGH)

### Reasoning
```
REAL THROTTLE, and it is provable without APERF/MPERF at all. That last point matters: the artifact hypothesis attacks the instrument, so I built the decisive proof from an instrument the hypothesis cannot touch — token throughput on byte-identical code.

[MEASURED-BY-ME-TODAY, from existing logs] THE MSR-FREE PROOF. Compare the Dell i5-5200U against the HP Stream N4020 on the *scalar* path (byte-identical code; the N4020 has no AVX2 so both machines run the same SSE2 fallback). The N4020's clock is independently trustworthy because on that machine the same metric read 99/100/245% — it varies, so it is not stuck-at-22.

  Dell  : 17,162,364,212 ticks/tok / 2.1975 GHz TSC = 7.810 s/tok = 0.12804 tok/s
          (docs/hardware_logs/gauntlet_dell_i5-5200U_2026-07-12_005500.txt:12)
  N4020 : 6,323,622,274 ticks/tok / 1.0930 GHz TSC = 5.785 s/tok = 0.17285 tok/s
          (docs/hardware_logs/gauntlet_hp_stream_n4020_2026-07-14_141531.txt:36, clock 100%)

  N4020 work rate = 0.17285 / 1.0930 GHz = 0.15814 tok/s per GHz.
  Grant Broadwell ZERO IPC advantage over Goldmont Plus — an absurdly charitable
  floor, since Broadwell is 4-wide OoO with 2x the ROB — and solve for the Dell's clock:
      0.12804 / 0.15814 = 0.8097 GHz.

  The Dell was running at NO MORE THAN 810 MHz. That is a hard ceiling derived
  from token counts and wall-clock seconds only. Nominal is 2200 MHz. The
  throttle is real no matter what any MSR says.

  Inverting the other way: at the claimed 2.2 GHz the implied Broadwell/Goldmont+
  IPC ratio would be 0.368 — Broadwell would have to be 2.7x *worse* per clock
  than a Celeron. Physically impossible. At the APERF/MPERF-implied 484 MHz the
  ratio is 1.673, which is exactly the expected figure for that generation pair.

[MEASURED-BY-ME-TODAY] THE 22% RESOLVES TO AN EXACT INTEGER P-STATE. MPERF ticks at
the max-non-turbo ratio; the log's own TSC rate confirms that is 2.1975 GHz = ratio 22,
bus 100 MHz. cpu.rs:271 does `da*100/dm` with floor division, so "22" means
actual/nominal in [0.22, 0.23), i.e. ratio in [4.84, 5.06). The only integer in that
window is 5. Enumerated today: ratio 4 prints 18, ratio 5 prints 22, ratio 6 prints 27.
Ratio 5 x 100 MHz = 500 MHz = Broadwell-U LFM exactly. An artifact has no reason to
land on the architectural minimum ratio and nothing else.

[INFERRED] ARTIFACT HYPOTHESES, EACH KILLED BY ARITHMETIC:
 (a) "It's TSC-vs-wall-clock confusion." No — cpu.rs:265-272 never touches TSC or
     wall clock. Inputs are rdmsr(0xE8)/rdmsr(0xE7) only (cpu.rs:41-42, 248-260).
     The tick source and the clock% source share zero inputs; they are merely
     printed on the same line (main.rs:547 rdtsc vs main.rs:549/554 perf_snapshot).
     The invariant TSC ticking at 2.1754-2.1975 GHz on a throttled core is exactly
     correct behavior, documented in the code's own header at main.rs:8-11.
 (b) "Counter wraparound / 32-bit truncation." Killed by interval-length invariance:
     nine gauntlet segments spanning 30 s to 688 s all report exactly 22%. A
     wrap artifact would produce essentially random ratios across a 23x range of
     interval lengths.
 (c) "Stuck-at value / broken read." Killed by the same binary emitting 99%, 100%,
     and 245% on the N4020, and `?` under a hypervisor (correctly — cpu.rs:251
     early-returns None when is_hypervisor()). Histogram over all of
     docs/hardware_logs/: 30x `?`, 21x 22%, 4x 99%, 4x 245%, 2x 100%.
 (d) "Arithmetic overflow." Killed: longest segment gives da ~ 3.3e11, da*100 ~ 3.3e13,
     five orders below u64::MAX. saturating_mul cannot saturate.
 (e) "C-state residency inflating it." APERF and MPERF both freeze outside C0, so the
     ratio is the mean frequency *while running*. And a UEFI compute loop never halts.

[MEASURED-BY-ME-TODAY] THE METRIC PREDICTS THROUGHPUT TO 1.6% ACROSS A 2.4x SWING.
Controlled same-machine A/B on the N4020: clock% 99 -> 245 (2.475x) while measured
ticks/tok went 6,323,664,454 -> 2,604,789,228 (2.428x). TSC rate over the same pair
moved 1.0932 -> 1.0969 GHz (+0.34%, i.e. blind, as an invariant TSC must be).
A metric that forecasts real throughput to 1.6% is not an artifact.

CAVEAT I will not hide: n=1 machine for the entire 22% observation. Every one of the
21 "22%" readings comes from one Dell Inspiron 15. The N4020 file's first block is a
byte-identical concatenation of the Dell run (BOOTLOG.TXT appends at EOF forever,
main.rs:26) — its line 11 even says "i5-5200U". So this is one sick laptop, not a
Broadwell-family finding.
```

### Arithmetic
```
THE NUMBER THAT DECIDES IT (MSR-free, computed today with python3):
  N4020 verified work rate:  0.17285 tok/s / 1.0930 GHz = 0.15814 tok/s per GHz
  Dell observed work rate:   0.12804 tok/s
  Dell clock upper bound assuming Broadwell IPC == Goldmont+ IPC:
      0.12804 / 0.15814 = 0.8097 GHz  <<  2.2 GHz nominal
  Implied IPC ratio at 2.2 GHz: 0.368  (impossible)
  Implied IPC ratio at 0.484 GHz: 1.673 (correct for the generation pair)

P-STATE RATIO IDENTIFICATION (floor division at cpu.rs:271):
  ratio 4 -> 18.18% -> prints 18
  ratio 5 -> 22.73% -> prints 22   <-- unique solution
  ratio 6 -> 27.27% -> prints 27
  ratio 5 x 100 MHz bus = 500 MHz = Broadwell-U LFM

TSC IS INNOCENT (Dell log lines 12-13):
  45,604,947,327 t / 20.964 s = 2.1754 GHz = 98.9% of 2.20 GHz nominal
  408,278,464,193 t / 185.791 s = 2.1975 GHz = 99.9% of nominal
  Same 22% reported for both. Invariant TSC, exactly as main.rs:8-11 documents.

THE NULL IS BIT-LEVEL, NOT NOISY (Dell log lines 17 vs 19):
  PSTATE_run2_control  3,548,334,308 ticks/tok
  PSTATE_run3_turbo    3,548,347,118 ticks/tok
  delta = +0.00036%. Since TSC is invariant, ticks/tok is inversely proportional
  to real core frequency. The frequency did not move by one part in 275,000.
  The 0.61 -> 0.62 tok/s wobble is timer quantization: every "total" field is an
  integer second (70.000/69.000/69.000) because UEFI GetTime returns nanoseconds=0
  on this firmware.

MECHANISM DISCRIMINATOR — HARD FLOOR, NOT POWER EQUILIBRIUM:
  Dell log:12 scalar path -> clock 22%
  Dell log:13 AVX2+FMA path -> clock 22%
  Both floor to ratio 5 exactly. AVX2 FMA draws substantially more power per clock
  than the scalar path. Under a RAPL PL1 or TM2 *power equilibrium*, the two legs
  would settle at different frequencies. They did not. This is a pin at the
  architectural minimum, which is what PROCHOT# assertion and TM1 forced-LFM do,
  and what a power-equilibrium clamp does not do.

ROOFLINE — the workload was compute-bound at 484 MHz, so frequency mattered:
  Traffic 780 MB/token (MODEL.SAF 522,831,576 + EMBED.BIN 257,310,720, zero reuse)
  At 0.62 tok/s that is 483 MB/s — ~5% of any plausible DDR3L-1600 dual-channel rate.
  MACs/token ~ 2.213e9 (2.084 G ternary per benches/ctz_vs_simd.rs:25 + 128.7 M LM head)
    at 484 MHz: 2.213e9 / (3.548e9 x 0.22) = 2.84 MAC/cycle = 17.7% of AVX2 peak (16)
    at 2.2 GHz: 2.213e9 / 3.548e9      = 0.62 MAC/cycle =  3.9% of AVX2 peak
  3.9% of compute peak AND 5% of bandwidth peak means bound by nothing — self-
  contradictory. And on-machine SIMD gain is 17,162,364,212/3,548,301,534 = 4.837x;
  a memory-bound kernel does not gain 4.8x from vectorization. The 17.7% figure is
  internally consistent; the 3.9% figure is not.
```

### Why the P-state fix failed
```
The fix did not fail because it was wrong or because it did not execute. It failed because IA32_PERF_CTL is a *request to an arbiter*, and something with higher authority than EIST was holding the package at LFM.

WHAT WE KNOW EXECUTED. Dell log:18 prints "GAUNTLET TURBO: legacy ratio=27 (~2700MHz)". That string is emitted only on the Ok(Boost::LegacySpeedStep) arm (main.rs:622), which is reachable only after every gate in cpu.rs:288-350 passed: msrs_safe(), !has_hwp() (correct — Broadwell is pre-HWP), has_eist(), has_platform_info(), IA32_MISC_ENABLE[16] SET (cpu.rs:323-325 returns Err otherwise), and MISC_ENABLE[38] clear (cpu.rs:337-342 — otherwise target would have been base 22, not turbo 27). The wrmsr at cpu.rs:348 happened. So "EIST is off" and "the code is broken" are both RULED OUT — and the same binary produced 2.428x on the N4020, so the mechanism itself is proven working.

WHY THE ASK WAS IGNORED. On Intel client parts the PCU resolves the final core ratio as the MINIMUM over every clamp: the EIST/HWP request, thermal (TM1/TM2), RAPL PL1/PL2, electrical/VR limits, and PROCHOT#. IA32_PERF_CTL sits at the *top* of that min(), so writing it can never override anything below it. The code has no way to notice this because request_max_performance() NEVER READS BACK. cpu.rs:346-349 writes 0x199 and immediately returns a Boost struct built from the value it *intended* to write. It never re-reads IA32_PERF_CTL (0x199) or IA32_PERF_STATUS (0x198). The HWP branch is the same — it reads back only IA32_PM_ENABLE (0x770), never IA32_HWP_REQUEST (0x774). So "legacy ratio=27 (~2700MHz)" is a log of a REQUEST, not a measurement; and "~2700MHz" is ratio x 100, since mhz_from_ratio() falls back to a hardcoded 100 MHz bus when CPUID leaf 0x16 is absent (cpu.rs:80-83), which it is on Broadwell — confirmed by "bus=0MHz" in the log banner.

RANKED MECHANISM HYPOTHESES (none measured — see below):
 1. [SPECULATION, highest prior] External bi-directional PROCHOT. MSR_POWER_CTL[0] set
    and PROCHOT# asserted by the EC — the classic 2015 Dell Inspiron degraded-battery /
    non-OEM-charger clamp. Fits every fact: it forces exactly LFM (ratio 5, observed);
    it is workload-independent (scalar and AVX2 both at 22%, observed); it predates all
    P-state code (22% already present in dell_pstate_first_reading_2026-07-09.txt,
    3 days before any turbo write, and cpu.rs itself first appeared 2026-07-09 in adda212);
    and it is completely immune to a legitimate EIST request. The code's own doc comment
    at cpu.rs:176-182 names this exact failure mode.
 2. [SPECULATION] SMM re-writing IA32_PERF_CTL back down. There is no OS under UEFI, but
    SMM is live. Distinguishable from #1 by the pre/post diag pair.
 3. [INFERRED, WEAKENED] RAPL PL1 at a configurable-TDP-down level, or TM1/TM2 thermal.
    Argued against above: a power-equilibrium clamp settles at different frequencies for
    scalar vs AVX2, and both legs pinned at exactly ratio 5. Thermal is further weakened
    because the 2026-07-09 reading was 22% after only 18 seconds of work.
 4. [INFERRED, NOT the explanation but a real latent bug] cpu.rs:347 masks only bits 15:8
    (`rdmsr(IA32_PERF_CTL) & !0xFF00`) and never tests IA32_PERF_CTL[32] (IDA/turbo
    disengage). If firmware left that bit set, a request for ratio 27 is silently clamped
    to base 22. That would have shown ~100% of nominal, not 22%, so it is not the Dell's
    problem — but it will bite some future fleet machine.

THE INSTRUMENT THAT WOULD ANSWER THIS ALREADY EXISTS AND HAS NEVER RUN. cpu::throttle_diag()
(cpu.rs:203-240) reads IA32_PERF_STATUS ratio, IA32_PERF_CTL ratio, MISC_ENABLE[16]/[38],
IA32_CLOCK_MODULATION, IA32_THERM_STATUS (hot / prochot_now / prochot_log / degrees below
TjMax) and MSR_POWER_CTL[0]. It is called pre/post the write at main.rs:619/626 (gauntlet)
and main.rs:813/832 (/turbo). It was committed in ec1c139 on 2026-07-12 20:58 — AFTER the
Dell run at 00:55 that same day — and the N4020 stick on 07-14 was still the pre-diag Jul-12
binary. `grep -rn TURBO_DIAG docs/hardware_logs/` returns nothing. Zero hardware coverage.
The cause of the single most important negative result in this program has never been
measured, and the code to measure it has been sitting unused for 17 days.
```

### Honest headroom
```
HONEST BAND: 3.6x - 5.3x on the Dell's AVX2 decode, central estimate ~4.4x, i.e. 0.62 -> 2.2-3.3 tok/s, most likely ~2.7 tok/s. NOT the 5.58x that naive 484->2700 MHz arithmetic gives. And it is worth substantially less than that headline in program terms, for four reasons stated below.

DERIVATION (computed today):
  Sustained clock is the first uncertainty. The code requests ratio 27 (2.7 GHz 1-core
  turbo); a 15W Broadwell-U running one core of continuous AVX2 FMA for 70 s may or may
  not hold it, and may settle at base 2.2 GHz. So clock multiplier is 4.55x-5.58x.
  Scaling efficiency is the second. Linear clock->throughput scaling is empirically
  validated at 98.4% (N4020: 2.428x measured vs 2.475x clock) — but that was the SCALAR
  path, which is 4.84x less memory-dense per cycle than the Dell's AVX2 path. DRAM
  latency does not scale with core clock, so I derate to 0.80-0.95.
    2.2 GHz x 0.80 = 3.64x -> 2.25 tok/s (1.76 GB/s)
    2.2 GHz x 0.95 = 4.32x -> 2.68 tok/s (2.09 GB/s)
    2.7 GHz x 0.80 = 4.46x -> 2.77 tok/s (2.16 GB/s)
    2.7 GHz x 0.95 = 5.30x -> 3.29 tok/s (2.56 GB/s)
  Bandwidth does not clip the single-core case: 2.6 GB/s is ~30% of a single Broadwell-U
  core's latency-bound streaming ceiling (~10 line-fill buffers x 64 B / ~80 ns ~ 8 GB/s).
  CROSS-CHECK: the QEMU i5-10210U host reached 3.15 tok/s AVX2
  (docs/hardware_logs/gauntlet_dataset.tsv row qemu_i5-10210U_host_v2). A Broadwell at
  2.2-2.7 GHz landing at 2.2-3.3 tok/s against a Comet Lake at 3.15 is coherent.

WHY THIS IS WORTH LESS THAN IT LOOKS:
 1. PROBABILITY-WEIGHTED. If the clamp is external bd-PROCHOT driven by a bad battery or
    a non-OEM charger, there may be NO software fix at all — and clearing MSR_POWER_CTL[0]
    is a deliberate thermal-safety override, not a drive-by patch. My honest odds that
    ~4.4x is recoverable in software on this machine: roughly 40-50%. Expected value
    ~2x, not 4.4x.
 2. n=1 MACHINE. All 21 "22%" readings are one Dell Inspiron 15. The N4020 was already at
    99-100% of nominal — there is no phantom 4x hiding there. This is a broken-laptop
    recovery, not a fleet-wide multiplier. Presenting it as fleet headroom would be the
    overclaim the program's own doctrine forbids.
 3. IT DOES NOT MULTIPLY WITH MULTICORE. 4 cores x 5.58x would be 13.8 tok/s = 10.8 GB/s,
    which exceeds achievable DDR3L-1600 dual-channel (~9-10 GB/s). The combined ceiling is
    bandwidth-bound at roughly 11-13 tok/s, not 22x. The clock fix and any future SMP work
    are largely alternatives, not factors.
 4. THE CERTAIN WIN IS SEPARATE AND SMALLER-SOUNDING BUT REAL. Verified by grep today:
    request_max_performance() has exactly three call sites — main.rs:620 (/gauntlet),
    :750 (/autotest), :814 (/turbo). It is NOT on the boot path (main.rs:251-459,
    STAGE 1..7, which auto-runs inference). A normal boot on ANY fleet machine never asks
    the CPU to leave the firmware's default P-state. On the N4020 that is a MEASURED
    2.428x being forfeited on every ordinary boot, for the cost of one line. That is the
    finding to bank. It requires no hypothesis, no new hardware trip, and no argument.

LEDGER CORRECTION OWED: program/RESEARCH_LEDGER.md:21 (A10) cites the Dell's 0.61 tok/s
without recording that it was taken at 22% of nominal clock. That number is a FLOOR
imposed by a platform defect, not the machine's capability, and it should be annotated as
such. The Dell half of A10 should also be recorded as a documented NEGATIVE result — the
P-state request was correctly issued, verifiably reached IA32_PERF_CTL, and provably had
zero effect — which is a stronger and more honest finding than silence.
```

### What would settle it
```
HIGH confidence applies to the VERDICT (real throttle, ratio 5 / ~484 MHz). Confidence in the specific MECHANISM (bd-PROCHOT) is LOW-MEDIUM — it is a ranked hypothesis, never measured. Confidence in the headroom band is MEDIUM.

THE EXPERIMENT, on the Dell i5-5200U Inspiron 15, one boot, no code changes required:

STEP 0 (prep, 2 minutes). The current tree already contains the diagnostic (ec1c139,
2026-07-12 20:58). Confirm it is in the build: `git merge-base --is-ancestor ec1c139 HEAD`.
Rebuild the .efi and reflash the stick — the 2026-07-12 Dell binary predates the diag.
FIRST: delete or rename BOOTLOG.TXT on the stick. boot_log() seeks to
0xFFFF_FFFF_FFFF_FFFF and appends (main.rs:20-34), which already produced one mislabeled
log (the N4020 file's lines 1-24 are the Dell run, and its line 11 names the wrong CPU).

STEP 1. Boot the Dell. At the REPL run `/turbo`. That path calls log_throttle_diag pre
(main.rs:813) and post (main.rs:832, after a 100 ms settle) around the write. Then run
`/gauntlet` for the throughput legs.

STEP 2. Read the two TURBO_DIAG lines out of BOOTLOG.TXT. DECISION TABLE:
  - post: ctl_ratio=27, status_ratio=5, prochot_now=1
      -> CONFIRMED external bi-directional PROCHOT. Mechanism settled. Go to Step 3.
  - post: ctl_ratio=27, status_ratio=5, prochot_now=0, bd_prochot_enabled=false,
    clock_mod bit 4 clear, temp_below_tjmax large
      -> not PROCHOT, not thermal, not duty-cycle. Remaining suspect is RAPL PL1
         (MSR_PKG_POWER_LIMIT 0x610), which the codebase never reads. Add that read
         gated on family/model.
  - post: ctl_ratio reverted to 5 (or anything < 27)
      -> SMM is clobbering IA32_PERF_CTL. Test by writing 0x199 in a tight loop around
         the benchmark and seeing whether status_ratio ever rises.
  - post: clock_mod & 0x10 set
      -> IA32_CLOCK_MODULATION duty-cycle throttling. Write 0 to it and re-measure.
  - post: status_ratio=27 but the gbench line still says clock 22%
      -> ONLY in this case is the metric itself suspect. I consider this outcome very
         unlikely given the MSR-free bound above, but it is the falsifier and it should
         be stated in advance.

STEP 3 (zero code, decides whether ANY software fix is possible). Run the identical stick
under three power configurations: (a) OEM 65W Dell adapter + healthy battery,
(b) OEM adapter, battery physically removed, (c) as-found. If 22% becomes ~100% under
(a) or (b), the finding is PLATFORM HEALTH, not software — no MSR work will ever fix it,
and the program should stop budgeting for it. This is the single cheapest experiment in
the whole set and it should be run FIRST, before Step 1, because a positive result makes
Steps 1-2 unnecessary.

STEP 4 (n=1 fix). Run the same stick on any second Broadwell-U machine. This separates
"Broadwell needs a different mechanism" from "this one Dell is sick." Without it the
entire 22% claim rests on one laptop.

THREE CODE CHANGES WORTH MAKING BEFORE THE TRIP (all small, all independently justified):
 1. Add a read-back to request_max_performance(): after the wrmsr at cpu.rs:348, re-read
    IA32_PERF_STATUS and return the ACTUAL ratio, not the requested one. Same for the HWP
    branch (read 0x774, not just 0x770). Today the log records an intention and prints it
    as if it were a measurement — a Prime-Directive-1 violation.
 2. Log raw da and dm alongside the percentage at cpu.rs:271. Costs one format string and
    removes the last degree of freedom (absolute MPERF is currently never captured, so
    C0 residency cannot be independently bounded).
 3. Call request_max_performance() on the BOOT path. It is currently only at main.rs:620,
    750, 814. This is the one certain win: a measured 2.428x on the N4020 that every
    normal boot currently forfeits.

NOT worth doing yet: MSR_CORE_PERF_LIMIT_REASONS (0x64F) would name the clamping agent
outright on Broadwell, but it does not exist on Goldmont Plus and an unhandled #GP in a
UEFI app with no IDT is a dead machine. Gate it on family/model and verify the bit layout
against the SDM before shipping it to anyone else's laptop.
```

## Ranked proposals

### [P01-PSTATE-ON-BOOT] Call request_max_performance() on the boot path (with a real read-back), not only from /gauntlet, /autotest, /turbo
- category throttle | confidence HIGH | effort 1-2 h (one call + log line; the read-back rewrite of cpu.rs:288-351 is the other hour) | needs iron: True
- **gain**: 2.428x decode on the HP Stream N4020, MEASURED-IN-AN-EXISTING-LOG, not projected: docs/hardware_logs/gauntlet_hp_stream_n4020_2026-07-14_141531.txt lines 41-43 show PSTATE_run2_control at 6,323,664,454 ticks/tok and PSTATE_run3_turbo at 2,604,789,228 ticks/tok after the identical MSR write (0.17 -> 0.42 tok/s, APERF/MPERF clock 99% -> 245%). That entire 2.428x is currently forfeited on every non-/gauntlet boot of that machine. On the Dell i5-5200U the same write measurably did NOTHING (3,548,334,308 -> 3,548,347,118 ticks/tok, +0.00036%), so expected gain there is 1.00x until P02 resolves the clamp. QEMU: 1.00x, msrs_safe() returns false under a hypervisor. Fleet-weighted expectation: 1 of 3 known machines gets 2.4x, cost is one line.
- **verify**: Boot the current stick on the HP Stream N4020 and, WITHOUT typing /gauntlet or /turbo, run /benchmark from the REPL. Record tok/s. Reflash with the boot-path call and repeat. GATE: post-change /benchmark tok/s must be >= 2.0x the pre-change figure on the N4020, and the new BOOTLOG line must print a read-back ratio from IA32_PERF_STATUS, not the requested ratio. FAILS IF: the boot-path write is issued before firmware has finished its own P-state setup and gets overwritten (then status_ratio in the read-back will be below the requested ratio and the benchmark will not move) - this is a real possibility and is exactly why the read-back is part of the same proposal.
- **risk**: LOW-MEDIUM. Raising the P-state raises package power on 15W and 6W parts that will now hold turbo through a 782 MB load plus 565 MB of heap zeroing. Thermal shutdown on the N4020 (6W Celeron, passively cooled HP Stream) is the plausible failure. Mitigate by placing the call AFTER STAGE 4 file loading, not before, so the USB read still runs at firmware default. Also: any bug in the read-back path is a rdmsr on 0x198, which exists wherever EIST exists - already gated by has_eist() at cpu.rs:150.

### [P02-TURBO-DIAG-DELL] Run the throttle diagnostic that has existed for 17 days and never once executed, plus the zero-code power-configuration A/B
- category instrumentation | confidence HIGH | effort 2 h hardware session, 0 h engineering for STEP 0; +1 h to rebuild and reflash for STEP 1 | needs iron: True
- **gain**: No direct speedup. It decides whether a 3.6x-5.3x is recoverable at all, and my honest odds that it is recoverable IN SOFTWARE are 40-50%. The MSR-free bound is that the Dell was running at NO MORE THAN 810 MHz against a 2200 MHz nominal (N4020 verified work rate 0.15814 tok/s per GHz on byte-identical scalar code; Dell observed 0.12804 tok/s; 0.12804/0.15814 = 0.8097 GHz even granting Broadwell zero IPC advantage over Goldmont Plus). Decision value: without this, every hour spent on Dell P-state work is a coin flip; with it, the branch is known. Cost is one hardware session and no engineering.
- **verify**: GATE: BOOTLOG.TXT must contain two lines matching 'TURBO_DIAG pre:' and 'TURBO_DIAG post:'. Decision table stated in advance so it can falsify me: (i) post shows ctl_ratio=27, status_ratio=5, prochot_now=1 -> external bi-directional PROCHOT confirmed, mechanism settled; (ii) ctl_ratio=27, status_ratio=5, prochot_now=0, bd_prochot=false, clock_mod bit 4 clear, large temp margin -> none of the instrumented causes; remaining suspect is RAPL PL1 (MSR_PKG_POWER_LIMIT 0x610), which this codebase never reads; (iii) ctl_ratio reverted below 27 -> SMM is clobbering IA32_PERF_CTL; (iv) clock_mod & 0x10 set -> duty-cycle throttling, write 0 and re-measure; (v) status_ratio=27 but gbench still prints clock 22% -> ONLY in this case is the APERF/MPERF metric itself suspect. I consider (v) very unlikely given the MSR-free 810 MHz bound, and I am naming it in advance as the falsifier.
- **risk**: LOW. Read-only MSR reads, all CPUID-gated (has_eist cpu.rs:150, has_therm cpu.rs:172, msrs_safe cpu.rs:74). Do NOT add MSR_CORE_PERF_LIMIT_REASONS (0x64F) to this trip: it does not exist on Goldmont Plus and an unhandled #GP in a UEFI app with no IDT is a dead machine (the code says so at cpu.rs:163-165).

### [P03-BOOT-KILL-TYPEMATIC] Delete the 2.12 s typewriter-effect banner from the boot path
- category boot | confidence HIGH | effort 0.25 h | needs iron: True
- **gain**: 2.115 s off every boot, MEASURED-BY-ME-TODAY by character count from source (423 chars x 5 ms), independent of any hardware unknown. Additional unquantified saving from removing 423 firmware console glyph writes. This is the largest boot saving in the whole set that carries zero correctness risk.
- **verify**: Requires P04 to have landed first, otherwise it is unverifiable. GATE: with stage timestamps in BOOTLOG.TXT, the wall-clock delta from '==== A.L.I.C.E. BOOT ====' to 'STAGE 1' plus 'STAGE 6' to REPL-ready must drop by >= 1.8 s on the same machine, same USB stick, cold boot. FAILS IF: the firmware console is so slow that glyph rendering, not the stall, dominated - in which case the delta will be well under 1.8 s and the honest write-up is 'the stall was not the cost, the console was'.
- **risk**: NONE technical. Cosmetic only - it removes the demo aesthetic. Keep it behind a feature flag rather than deleting it if the banner is used in grant-review video.

### [P04-BOOT-INSTRUMENTATION] Timestamp all nine boot stages and log installed/contiguous memory - without this every boot claim in the program is unfalsifiable
- category instrumentation | confidence HIGH | effort 1.5-2 h | needs iron: True
- **gain**: Zero speedup - and I am ranking it 4th anyway because it is the precondition for P03, P06, P09 and for retiring a reviewer-facing overclaim. Today the premise 'file loading dominates boot' is PLAUSIBLE BUT UNPROVEN: the 8.12 s of pure sleep plus 0.5-3 s of bounce memcpy plus 73 progress-bar redraws could plausibly rival a fast USB3 load. Second deliverable: NO LOG IN THE TREE RECORDS ANY MACHINE'S INSTALLED RAM (`grep -rniE '\bram\b|conventional' docs/hardware_logs/gauntlet_*.txt` -> zero hits), yet program/ROADMAP.md:74-76 and :197 assert 'bare-metal LLM inference demonstrated on 2GB commodity hardware', contradicted by program/MODEL_LAB.md:121 which lists the 2GB boot as pending. This change produces the evidence that either supports or retires that sentence.
- **verify**: GATE: a fresh BOOTLOG.TXT from a cold boot on the Dell must contain nine STAGE lines each with a monotonically non-decreasing wall-second and TSC field, and a STAGE 3 line reporting total conventional bytes and largest contiguous extent. Cross-check: sum of stage deltas must equal the total boot wall time to within 1 s. FAILS IF: UEFI GetTime returns nanoseconds=0 on the fleet firmware (it does - every 'total' field in the Dell gauntlet log is an integer second: 70.000/69.000/69.000), so wall_seconds() has 1 s granularity and CANNOT resolve stages shorter than a second. That is precisely why the TSC delta must be logged alongside; if only wall seconds are logged this gate fails and the instrumentation is useless.
- **risk**: LOW. More BOOTLOG writes on a path where each boot_log() already opens CreateReadWrite, seeks EOF, writes, flushes and closes - three or more USB BOT round trips per line. Do not add more than the nine that already exist.

### [P05-MEM-CHUNKED-PREFILL] Chunk prefill and size the arena batch buffers to the chunk, not to max_seq_len - recovers 218 MB with a provable exact-parity gate
- category memory | confidence HIGH | effort 3-4 h | needs iron: False
- **gain**: 218,365,952 B = 218.4 MB recovered (249,561,088 - 31,195,136), MEASURED-BY-ME-TODAY by reading the allocation sizes out of arena.rs against aegis-forge/aegis_pruned_config.json. Working heap high-water drops from 571.9 MB to ~354 MB. Combined with P11 (BF16 KV) it drops to ~196 MB, which is what makes P07 (dropping the hardcoded 700 MiB heap) possible. Net effect on total physical reservation, currently 1,532,686,336 B: down to roughly 1.05 GB. That is the difference between 'might boot on a 2GB box' and 'boots with headroom'. Throughput effect: expected NEUTRAL to slightly positive (smaller working set, better L2/L3 reuse in the batched GEMM); do not claim a speedup.
- **verify**: GATE 1 (exact, could genuinely fail): run aegis-core/tests/reference_parity.rs before and after; the generated token id sequence must be BIT-IDENTICAL, not merely close - the program already holds a 471/471 token parity result (program/RESEARCH_LEDGER.md:101, docs/hardware_logs/m7_final_roundtrip_2026-07-27.log) so this is a real regression gate. GATE 2: run a prompt LONGER than 256 tokens through aegis-linux and confirm identical output to the unchunked build - this is the case that will expose an off-by-one in seq_pos_start. GATE 3: GAUNTLET PEAK_MEMORY on real iron must drop from 571,912,597 bytes to under 360,000,000. FAILS IF: any caller of forward_batch assumes it can read logits or intermediate state for arbitrary positions after the call returns - I checked the attention path but did NOT audit forward_batch_with_capture's capture callback, which may want whole-sequence hidden states.
- **risk**: MEDIUM. The capture path (the `capture` argument at inference.rs:173) is used by graph/activation tooling and may legitimately need whole-sequence buffers; if so, keep the large buffers behind a feature the UEFI build does not enable. Prefill wall time may rise slightly because per-chunk overhead is paid ceil(n/256) times.

### [P06-BOOT-STALL-AB] A/B the six unexplained 1-second USB stalls - 6 s of every boot that nobody can currently justify
- category boot | confidence MEDIUM | effort 0.5 h code, 2 h hardware (needs several cold boots per machine per variant) | needs iron: True
- **gain**: 6.0 s off every boot if they are removable, MEASURED-BY-ME-TODAY as a count of stall sites in source. Combined with P03 that is 8.115 s of pure sleep removed from a boot whose total duration is currently unknown (see P04). If they are NOT removable, the deliverable is a documented answer to a question that has been open since the file was written - a negative result the doctrine counts as a win.
- **verify**: GATE: five consecutive cold boots of the no-stall build on EACH of the Dell and the N4020 must all reach 'STAGE 6: engine online' with no 'STAGE 4x FAILED' line, AND the P04 timestamps must show the expected ~6 s reduction. FAILS IF: even one boot in five shows a truncated read - load_file_into already catches this correctly at main.rs:213-220 ('[ERR: truncated read N of M bytes]'), so the failure is detected rather than silent. IMPORTANT: a flaky-USB regression may not reproduce on the first try, which is exactly why the gate is five boots per machine and not one.
- **risk**: MEDIUM-HIGH on correctness, LOW on safety. If the stalls are load-bearing for a specific XHCI/EHCI BOT state machine, removing them produces intermittent truncated reads. The truncation check at main.rs:213-220 turns that into a visible error rather than garbage inference, so the failure mode is loud. Ship the stalls ON by default and OFF behind the feature until five-boot data exists for every fleet machine.

### [P07-MEM-HEAP-SIZING] Size the large heap from the engine's actual requirement instead of a hardcoded 700 MiB, and fix the silent under-allocation that will bite exactly on a 2GB box
- category memory | confidence HIGH | effort 2 h | needs iron: False
- **gain**: After P05+P11 land: total physical reservation drops from 1,532,686,336 B (weights 781,905,920 + small heap 16,777,216 + large heap 734,003,200) to roughly 1,050,000,000 B - about 480 MB returned to the system on exactly the constrained machines where it matters. Standalone (without P05/P11) the saving is only ~130 MB, which is why this is ranked below P05. Second deliverable: converts a silent boot-time OOM into a legible failure.
- **verify**: GATE 1 (QEMU, no iron needed): run run_qemu.sh with -m 1536 and then -m 1280. The pre-change build must fail (or OOM); the post-change build must reach 'STAGE 6: engine online' at 1536M. GATE 2: deliberately fragment by requesting a target the map cannot satisfy and confirm the new code prints a STAGE 5 failure line instead of panicking in the allocator. GATE 3: GAUNTLET PEAK_MEMORY must be unchanged (this proposal changes reservation, not usage) - if PEAK_MEMORY moves, something else broke. FAILS IF: the derived target underestimates a transient the arena high-water does not capture (format! strings, tokenizer BTreeMap) - the 6,348,293 B residual between my computed 565,564,304 B and the logged 571,912,597 B is exactly that, so the slack must exceed it.
- **risk**: MEDIUM. Under-sizing the heap turns a working machine into a boot failure. The 15% slack and GATE 3 are the mitigations. Do not land this before P05, or the derived target will just reproduce 700 MiB.

### [P08-MP-SERVICES-PROBE] Boot-time probe of EFI_MP_SERVICES_PROTOCOL - the 20-line go/no-go for the single largest compute multiplier
- category multicore | confidence HIGH | effort 1-2 h | needs iron: True
- **gain**: Zero speedup. It is ranked here because it costs 1-2 hours and it is the entire risk gate on P10, which is a 20-30 hour item with a 2.1-2.3x payoff. If the protocol is absent on the Dell or the N4020 firmware, P10 must either be abandoned or rescoped to hand-written INIT-SIPI-SIPI (which is a different and much larger project - and would require writing an IDT, since there is none today). Answering that for 2 hours before committing 30 is the correct sequencing.
- **verify**: GATE: BOOTLOG.TXT on the Dell must contain a line reporting >= 2 processors (i5-5200U is 2C/4T) and on the N4020 >= 2 (Celeron N4020 is 2C/2T). FAILS IF: locate_protocol returns NOT_FOUND on either machine - which is a genuine possibility and is the whole point of running it. Also verify under OVMF first (run_qemu.sh with -smp 4) so a locate failure on iron is distinguishable from a code bug.
- **risk**: LOW. Read-only protocol queries. The one hazard is that locate_protocol on absent firmware protocols must be handled as Err, not unwrapped - a panic here halts the machine (panic handler at main.rs:954 is a `loop { _mm_pause() }`).

### [P09-BOOT-DIRECT-READ] Read directly into 64 KiB-aligned sub-4GB weight buffers and delete the 782 MB bounce memcpy
- category boot | confidence MEDIUM | effort 3 h | needs iron: True
- **gain**: 0.46-3.07 s, MEASURED-BY-ME-TODAY as a proxy on this i5-10210U at 2112 MHz: a 64 KiB-chunked copy of 781.9 MB ran at 142.5 MB/s cold (5.487 s), 254.9 MB/s (3.068 s), and 1691.4 MB/s warm-source (0.462 s) across three reps. The realistic UEFI case is the warm-source figure (the 64 KiB bounce stays in L2), so 0.46 s at 2.1 GHz - but the fleet boxes are slower and, per the Dell logs, may be running at 484 MHz during boot, which scales that to ~2 s. Do NOT lead with this: it is very likely an order of magnitude below the USB read time, which P04 will finally quantify.
- **verify**: GATE 1: with P04 timestamps, STAGE 3->4a and 4a->4b deltas must drop measurably on the Dell. GATE 2 (the one that matters): SHA-256 the loaded MODEL.SAF in memory against the on-disk file - if P13 has landed, this is free; otherwise compare the first and last 4 KiB plus the byte count. Silent corruption from a bad DMA target is the failure mode and it must not be detectable only as bad inference. FAILS IF: the firmware's FAT/USB stack refuses or corrupts reads into a caller-supplied buffer that is not its own bounce-friendly memory - genuinely possible on old EDK2 builds, and the reason the fallback path stays.
- **risk**: MEDIUM-HIGH. A DMA target that violates a controller constraint produces silently wrong weights, not an error. Never ship this without a hash check of the loaded bytes. Keep the bounce path as the fallback and log which path was taken.

### [P10-MP-SMP-MATVEC] Row-parallel ternary_matvec across firmware-started APs - the largest compute multiplier available, and the largest risk
- category multicore | confidence MEDIUM | effort 20-30 h | needs iron: True
- **gain**: 2.14x at 4 threads on the dev box - and I am deliberately quoting 2.14x, not the 8.25 tok/s / 2.27x figure in program/RESEARCH_LEDGER.md:15 and README.md:21-22, because that figure has NO LOG (`grep -rn '8.25' docs/hardware_logs/` -> no match) and the only primary source, commit 154f00a's own message, measures 8 threads at 5.40 tok/s, SLOWER than 4 threads at 5.61. Fleet reality is worse than the dev box: the i5-5200U is 2C/4T and the N4020 is 2C/2T, so the realistic bare-metal ceiling is ~1.8-2.0x, not 4x. AND IT DOES NOT MULTIPLY WITH P01. Arithmetic intensity of the decode path is 5.69 FLOP/byte; machine balance 32 FLOP/cycle x f / BW crosses over at f = 1.42 GHz (8 GB/s) to 1.78 GHz (10 GB/s). Above that clock the workload becomes bandwidth-bound, and the bandwidth roof on 778.3 MB/token traffic is 10.3-12.8 tok/s. So P01 at full clock plus 4 cores is ~10-13 tok/s combined, NOT 5.58 x 4 = 22x. P01 and P10 are substantially alternatives, not factors. Anyone presenting them as multiplicative is overclaiming.
- **verify**: GATE 1: aegis-core/tests/reference_parity.rs must produce a BIT-IDENTICAL token sequence with the parallel path on - floating-point row partitioning does not change per-row results, so exact parity is the correct gate and any drift means a genuine race. GATE 2: aegis-core/tests/thread_safety.rs must pass. GATE 3 (iron): /benchmark on the Dell with APs enabled must beat single-core by >= 1.6x. FAILS IF: firmware AP dispatch latency dominates (measure it explicitly with a null-work startup_this_ap timed by rdtsc BEFORE writing the kernel split - if a round trip costs more than ~200 us, the per-token dispatch budget is blown and the persistent-spin design is mandatory rather than optional).
- **risk**: HIGH. Per-core XCR0 is a hard-hang failure mode with no debugger. Firmware AP support is unverified (P08 gates this). The measured ceiling is lower than the headline because of the bandwidth crossover above. Do not start before P08 returns >= 2 processors on real fleet iron.

### [P11-BF16-KV-CACHE] Store the KV cache in BF16 - halves 314.6 MB and halves attention memory traffic
- category memory | confidence MEDIUM | effort 4-6 h | needs iron: False
- **gain**: 157,286,400 B recovered. Secondary and possibly larger: attention KV read traffic halves, and attention is the least efficient kernel in the engine at ~1.16 MAC/cycle = 7.3% of AVX2 peak versus the matvec's 17.7% (derived from the near-linear CTX curve in gauntlet_dell_i5-5200U_2026-07-12_005500.txt: 3,545,464,382 t/tok at CTX_20, 3,568,870,988 at CTX_100, 3,661,843,164 at CTX_400 = ~600K TSC ticks per context position, consistent across both intervals). At CTX_400 attention is 3.7% of a token so halving its traffic buys under 2%; at the full 2048 window attention is 26% of a token, so this is worth up to ~10% there. Quote the memory saving, not the speed.
- **verify**: GATE 1 (this one will probably NOT hold, and that is the point): reference_parity.rs bit-identical token ids. BF16 truncation of K and V changes attention scores in the last 16 mantissa bits, so exact parity may legitimately break. GATE 2 (the real gate): WikiText-2 perplexity must not degrade by more than 0.5% against the f32-KV baseline of 12.738 recorded in aegis-core/Cargo.toml's int8_act feature comment, AND the M21 round-trip tolerance must hold (torch 3.6808 vs engine 3.6880 = 0.19%, tol 3%, docs/hardware_logs/m7_final_roundtrip_2026-07-27.log). GATE 3: a 200+ token generation must remain coherent - compare against docs/hardware_logs/m7_coherence_spotcheck_2026-07-28.log. FAILS IF: PPL moves more than 0.5% or long-context generation degrades, in which case this is a documented negative result and the memory has to come from P05 alone.
- **risk**: MEDIUM-HIGH numerically. NO PARITY TEST CURRENTLY COVERS REDUCED-PRECISION KV, and this project's worst historical bug was exactly a KV/embedding dtype mismatch. Land P05 first (218 MB, no numerical risk) and only reach for this if the 2GB gate still fails.

### [P12-ATTN-HEAD-GROUP-BLOCKING] Hoist the 4 query heads that share a KV head into the inner loop - 4x less attention KV traffic
- category engine | confidence MEDIUM | effort 4-5 h | needs iron: True
- **gain**: Up to 4x reduction in attention KV load traffic. Bounded honestly: at the measured CTX_400 operating point attention is 3.7% of a token (612,520 TSC ticks/position x 227 mean positions against a 3.52e9-tick base), so the token-level gain is at most ~2.8%. At the full 2048 window attention is 26% of a token, so the gain rises to ~15-19%. This is a max-context optimization, not a short-prompt one, and should be sold that way. It also compounds with P11 (halved element width on top of 4x fewer reads).
- **verify**: GATE 1: reference_parity.rs must be BIT-IDENTICAL. Reordering which q-head consumes a K row does not change any individual dot product or the softmax input order, so exact parity is the right gate and any drift is a real bug. GATE 2: /gauntlet CTX_400 ticks/tok on the Dell must drop by >= 2%, and a synthetic CTX_2000 leg must drop by >= 10%. FAILS IF: the 4-way score fan-out spills the q vectors out of registers and the gain is eaten by reload traffic - very plausible, since 4 q-heads x 128 f32 = 2 KB of live state against 16 YMM registers. Measure before committing.
- **risk**: MEDIUM. Attention correctness is the easiest thing in a transformer to break subtly (softmax over the wrong range, wrong kv_h mapping). GATE 1 is exact so it will catch it. Do NOT do this at the same time as P11 - land one, verify, then the other, or a parity failure is uninterpretable.

### [P13-ARTIFACT-INTEGRITY-AND-PROVENANCE] SHA-256 the three artifacts at load, and embed the git commit in the binary - the minimum credible contested-environment increment
- category security | confidence HIGH | effort 5-6 h | needs iron: True
- **gain**: No speedup. Capability. It closes three of the five contested-environment gaps in one change: no tamper evidence, no supply-chain provenance in the binary, and no way to prove to a relying party what code and what weights actually booted. It also makes P09 (direct DMA read) safe to ship, because the hash becomes the corruption detector. For a DARPA-facing package, 'every benchmark line names its own commit and its own weight digest' is a stronger sentence than any tok/s number.
- **verify**: GATE 1: flip one byte in MODEL.SAF on the stick with a hex editor and boot - the machine must print a STAGE 4 digest-mismatch failure and refuse to run inference. This gate can genuinely fail (a streaming hash that resets on a short read, or a build.rs digest computed from a stale artifact, both produce false passes). GATE 2: the digest printed at boot must equal `sha256sum` of the file on the build host. GATE 3: boot time increase must be under 1.5 s - measure with P04 timestamps. FAILS IF: SHA-256 at 782 MB costs more than the budget on a 484 MHz Broadwell (a scalar no_std SHA-256 runs ~150-250 MB/s at 2 GHz, so ~4-6 s at full clock and far worse if the Dell throttle is unresolved) - in which case gate it behind a `/verify` command rather than the boot path, and say so.
- **risk**: MEDIUM. The boot-time cost is the real risk and it interacts with P01: at 484 MHz a full-artifact SHA-256 could add 20+ s. Sequence this AFTER P01 and P04 so the cost is measurable and the clock is known.

### [P14-AVX2-QUANTIZE-ACT] Vectorize quantize_activations_int8 - MARGINAL, ~1% of decode, but free and bit-exact
- category engine | confidence HIGH | effort 0.5 h | needs iron: False
- **gain**: ~1.0% of decode. MEASURED-BY-ME-TODAY on this i5-10210U with a Rust harness using the actual libm 0.2.16 crate (source at /tmp/claude-1000/-home-killboxincorporated/d75b4760-6465-4d8e-bf5f-32d865609e80/scratchpad/qbench): n=2560 libm-scalar 21.258 us/call vs avx2 1.322 us/call (16.08x); n=6912 libm-scalar 70.370 us/call vs avx2 3.776 us/call (18.64x); maxabsdiff 0e0 on both, i.e. BIT-IDENTICAL output. Per token that is 30 x (3 x 21.258 + 70.370) = 4.02 ms scalar vs 0.232 ms AVX2 = 3.79 ms saved. Against the userspace 1-thread baseline of 382 ms/token (A4's 2.62 tok/s) that is 0.99%. I am explicitly correcting the recon's 1.4-1.5% figure downward: it used a C/glibc roundf proxy, which is slower than the Rust libm implementation actually shipped. CAVEAT: the box had load average 9.7-10.0 during the run (the user's own python3 jobs), so absolutes are contended; the ratio was measured back-to-back under identical contention. ALSO: zero benefit on the N4020, which has no AVX2 and takes the SSE2 scalar fallback for everything.
- **verify**: GATE 1: reference_parity.rs bit-identical - my bench already shows maxabsdiff 0e0 on both sizes, so anything else means the port is wrong. GATE 2: /benchmark tok/s on the Dell must improve by >= 0.5%. This gate WILL PROBABLY FAIL to resolve: 1% is inside the timer quantization of the UEFI benchmark path (UEFI GetTime returns nanoseconds=0 on this firmware, so every 'total' field in the Dell log is an integer second - 70.000/69.000/69.000). Use the ticks/tok field from /gauntlet instead, which has TSC resolution and did resolve a 0.00036% difference in the P-state null. If the ticks/tok delta is under 0.5%, record it as a null.
- **risk**: LOW. Requires _mm256_round_ps to match libm::roundf's ties-away-from-zero for the values actually seen. My bench found zero difference across 20,000 calls on both sizes, but that is not a proof for all inputs - _MM_FROUND_TO_NEAREST_INT is ties-to-EVEN while roundf is ties-AWAY-from-zero, and they differ at exact .5 values. Activations landing exactly on .5 after scaling by 127/absmax are rare but not impossible. GATE 1 across the full parity fixture is what catches it.

### [P15-DEAD-ZERO-FILLS] EXPECT TO FAIL: remove the seven dead arena zero-fills per layer
- category engine | confidence HIGH | effort 0.5 h | needs iron: False
- **gain**: I EXPECT THIS TO MEASURE AS ZERO and I am including it so the list is credible. 3.04 MB of stores per token, all into buffers already resident in L1/L2 (largest is 27 KB), at a plausible L2 store bandwidth of 20+ GB/s is ~0.15 ms against a 382 ms token = 0.04%. That is two orders of magnitude below the gauntlet's own run-to-run drift. Realistic outcome: an unmeasurable null and a slightly cleaner hot loop. It is on the list because it costs 30 minutes and because leaving dead stores in a hot path that a future optimizer will re-derive is its own tax.
- **verify**: GATE 1 (the only gate that matters): reference_parity.rs bit-identical. If ANY of the seven was actually load-bearing - e.g. a short-vector tail in the scalar kernel that does not write every row - parity breaks immediately and loudly. That is a gate that can genuinely fail and it is the reason to do this with the test rather than by inspection. GATE 2: /gauntlet ticks/tok on the Dell. PREDICTION ON THE RECORD: the delta will be under 0.3% and should be published as a null, not as a win.
- **risk**: LOW-MEDIUM. The hazard is that ternary_matvec's row loop has a `while row < dim_out` tail (ops.rs:565-575) - if any dim_out were not covered by both the 4-wide and 1-wide paths, a removed fill would leak stale data. Parity catches it.

### [P16-PREFETCH-TERNARY-MATVEC] EXPECT TO FAIL: software prefetch in the ternary matvec
- category engine | confidence MEDIUM | effort 3 h | needs iron: True
- **gain**: NEAR ZERO at current bare-metal clocks, and I am proposing it as a proposal-to-reject. Three independent measurements say so. (1) MEASURED-IN-AN-EXISTING-SESSION on this box: DRAM-resident vs L3-resident matvec slowdown was only 1.329x and 1.183x across two runs, meaning memory traffic costs at most 20-30% of the kernel today and prefetch can recover only a fraction of that. (2) Roofline from the Dell log: at 484 MHz the kernel runs at 2.84 MAC/cycle = 17.7% of AVX2 peak (16 MAC/cycle from 2 FMA ports x 8 lanes), while achieved bandwidth is 483 MB/s - roughly 5% of any plausible DDR3L-1600 ceiling. Compute-bound by a factor of ~3.5x. (3) The 4.837x SIMD gain measured on the Dell (17,162,364,212 scalar vs 3,548,301,534 AVX2 ticks/tok) is itself proof: a memory-bound kernel does not gain 4.8x from vectorization. NOTE THE CONDITION UNDER WHICH THIS FLIPS: arithmetic intensity is 5.69 FLOP/byte and machine balance crosses over at 1.42-1.78 GHz. If P01 lands and the Dell reaches 2.2-2.7 GHz, the workload becomes bandwidth-bound and prefetch becomes worth exactly one experiment. Not before.
- **verify**: GATE: /gauntlet ticks/tok on the Dell must improve by >= 3%. PREDICTION ON THE RECORD: it will improve by under 1% at 484 MHz, and I expect a small REGRESSION from the extra instruction slots in an already instruction-bound loop. Re-run the same gate only after P01 has been verified to raise the sustained clock above ~1.8 GHz. Also do not trust benches/gemm_tile.rs for this: it uses a single (2560, 6912) = 4.42 MB weight matrix that fits this box's 6 MB L3 and is reused across reps, so it is L3-resident and cannot see a DRAM effect at all - a DRAM-resident rewrite of the same matvec measured 2.07-2.92 GMAC/s against gemm_tile's reported 9.67 GMAC/s.
- **risk**: LOW technically, MEDIUM to the schedule - it is the kind of change that feels productive and measures nothing. Do it only after P01 and only if the post-P01 roofline says bandwidth-bound.

### [P17-HONEST-OS-SCOPE] Retire the 'first UEFI AI-centered operating system' claim and ship the two findings that survive adversarial search
- category os-feature | confidence HIGH | effort 3-4 h | needs iron: False
- **gain**: No speedup. Removes the single highest-probability way this package gets discredited in review - a claim of precedence that one search falsifies - and replaces it with two claims that are defensible number-by-number. Under the program's own honesty doctrine ('log path or cut it', 'negative results published as wins') this is not optional; it is a correction that is currently owed.
- **verify**: GATE: hand the revised claim to an adversarial reader with instructions to falsify it in 30 minutes of search. It passes only if no counterexample surfaces for the narrowed claim. Note the ONE clause that currently survives search - no bare-metal/UEFI TERNARY (BitNet b1.58) inference implementation surfaced; every Rust BitNet implementation found (oxbitnet/wgpu, 0xBitNet/WebGPU, bitnet-rust/Metal) is OS-hosted, Anima runs GGUF, marvin-42 runs an unspecified hosted format. That is ABSENCE OF EVIDENCE, not proof of absence, and must be worded as such. FAILS IF: the reader finds a bare-metal ternary implementation, in which case the remaining defensible claim is provenance alone.
- **risk**: NONE technical. Organizational only: someone has to be willing to delete a marketing sentence. RESEARCH_LEDGER.md:87 already carries the 'unclaimed first (absence-of-evidence caveat)' hedge for M7; this converts that hedge into positive evidence of prior art for the bare-metal-boot half.

### [P18-COLD-BOOT-HYGIENE] Zeroize weights and heap on /exit, and stop placing them at deterministic physical addresses
- category security | confidence MEDIUM | effort 2-3 h | needs iron: False
- **gain**: No speedup. Closes contested-environment gap 2 (cold-boot exposure). The threat model has to be chosen first - the three plausible ones need different mitigations: a tampered artifact on a trusted machine (P13 hash-at-load), a trusted artifact on a compromised machine (measured boot + TPM), or adversary recovery of the stick (encrypted weights + secure erase). This proposal only addresses the third and only for DRAM, not for the stick itself.
- **verify**: GATE 1: under OVMF with a debug memory dump, physical pages that held MODEL.SAF must read as zero after /exit. GATE 2: the wipe must complete in under 2 s - 782 MB of stores at 484 MHz on a throttled Broadwell may not, and if it does not, the honest answer is to wipe only the KV cache and the tokenizer state (which hold the conversation) rather than the weights (which are public on the stick anyway). FAILS IF: firmware BDS reclaims and reuses the pages before the wipe completes, or the wipe is optimized out - use a volatile write loop or core::ptr::write_volatile, not slice::fill.
- **risk**: LOW-MEDIUM. write_volatile over 780 MB on a throttled core could look like a hang; print progress. Randomizing placement interacts badly with P09 (which needs 64 KiB-aligned sub-4GB targets) - if both land, the randomized picker must still honour those constraints.

## Adversarial reviews

### HARDWARE REALITY — Intel P-state/MSR/firmware semantics for Broadwell-U (i5-5200U, 06_3D), Goldmont Plus (N4020, 06_7A) and Comet Lake (i5-10210U, 06_8E) specifically. Method: read aegis-uefi/src/cpu.rs and main.rs line-by-line, recompute every number in the diagnosis from the raw log fields, and attempt direct MSR verification on this box.

VERIFICATION BUDGET — what I could and could not measure:
- /dev/cpu/0/msr and /dev/cpu/0/cpuid both exist but return EACCES (Permission denied). Verified by attempted read of 0xCE/0x198/0x199/0x1A0/0xE7/0xE8/0x770/0x771/0x774/0x1FC/0x19A/0x19C/0x1AD/0x610/0x64F.
- Independently, this box is a KVM GUEST: `lscpu` shows the `hypervisor` flag and "Hypervisor vendor: KVM". So cpu::msrs_safe() (cpu.rs:74) returns FALSE here by construction and the entire P-state subsystem no-ops on this machine. There is also no /sys/devices/system/cpu/cpu0/cpufreq (no cpufreq driver in the guest).
- CONSEQUENCE, stated plainly: ZERO of my MSR-semantics claims below are MEASURED-BY-ME-TODAY. They are VENDOR-DOC (Intel SDM Vol.4 MSR tables, recalled, not re-read here) or INFERRED-FROM-EXISTING-LOGS. Anything I assert about bit layouts or MSR existence should be checked against the SDM before it is shipped to someone else's laptop. What IS measured-by-me-today is the source reading and the arithmetic.

MEASURED-BY-ME-TODAY (python3, from raw log fields): Dell TSC 408278464193/185.791 = 2.1975 GHz; N4020 TSC 148318110954/135.695 = 1.0930 GHz; Dell scalar 7.810 s/tok vs N4020 scalar 5.785 s/tok = Dell 1.350x SLOWER in wall time on a byte-identical instruction stream; N4020 work rate 0.15814 tok/s/GHz; implied Dell clock ceiling 0.8097 GHz; implied Broadwell/Goldmont+ IPC ratio 1.675 at 22% vs 0.368 at 100%; floor-division table ratio 4->18, 5->22, 6->27. Every figure in the diagnosis reproduces. I could not break the core verdict.
- verdict on diagnosis: **OVERSTATED**
- The VERDICT (real throttle, core running at roughly one-fifth of nominal) is bulletproof and I could not dent it. The MECHANISM RANKING rests on a hardware-physics argument that is wrong, and that error propagates into the instrument list for the field trip.

OBJECTION 1 — THE MECHANISM DISCRIMINATOR IS PHYSICALLY INVALID. The diagnosis's "MECHANISM DISCRIMINATOR — HARD FLOOR, NOT POWER EQUILIBRIUM" argues: scalar (log:12) and AVX2+FMA (log:13) both floor at exactly ratio 5; AVX2 FMA draws more power per clock; a RAPL PL1 or TM2 power equilibrium would settle the two legs at DIFFERENT frequencies; they did not; therefore it is a hard pin (PROCHOT/TM1-forced-LFM), not a power clamp.

That inference is invalid because LFM IS A RAIL, NOT AN EQUILIBRIUM POINT. The P-state range has a floor: on Intel client parts the PCU cannot select a ratio below the max-efficiency ratio. Going below LFM requires a different mechanism entirely (T-states via IA32_CLOCK_MODULATION, or automatic thermal duty-cycling). So a PL1 clamp that is merely SEVERE — set low enough that even the scalar leg's power exceeds it at LFM — saturates at LFM for BOTH workloads, for exactly the same reason a hard pin does. At the rail, the scalar-vs-AVX2 test has ZERO discriminating power. PROCHOT-forced-LFM and PL1-at-rail are indistinguishable by it.

This is not a nitpick: this argument is what demotes RAPL PL1 to hypothesis #3 ("argued against") and promotes bd-PROCHOT to #1. Strip it out and the ranking is not established. The diagnosis's own honest self-assessment ("Confidence in the specific MECHANISM is LOW-MEDIUM") is correct; the arithmetic section presents the discriminator as if it were a settled finding, and it is not one.

Corollary the diagnosis also gets backwards: it notes "on-machine SIMD gain is 4.837x; a memory-bound kernel does not gain 4.8x from vectorization." True and useful for the roofline, but under a frequency clamp at a rail, the full 4.837x SIMD gain is EXACTLY what you expect from a power clamp too — since the frequency does not move, the vector path just does more work per cycle at the same clock. It is not evidence either way on mechanism.

OBJECTION 2 — THE INSTRUMENT LIST FOLLOWS FROM THE WRONG RANKING, AND THE DECISION TABLE HAS A BRANCH WITH NO POSITIVE TEST. I read cpu::throttle_diag() (cpu.rs:203-240) in full. It reads exactly: IA32_PERF_STATUS 0x198, IA32_PERF_CTL 0x199, IA32_MISC_ENABLE 0x1A0, IA32_CLOCK_MODULATION 0x19A, IA32_THERM_STATUS 0x19C, MSR_POWER_CTL 0x1FC. It does NOT read MSR_CORE_PERF_LIMIT_REASONS (0x64F), MSR_PKG_POWER_LIMIT (0x610), MSR_RAPL_POWER_UNIT (0x606), or MSR_TEMPERATURE_TARGET (0x1A2).

The diagnosis's own decision-table branch (ii) — "ctl_ratio=27, status_ratio=5, prochot_now=0, bd_prochot=false, clkmod clear, large temp margin" — concludes "remaining suspect is RAPL PL1" BY EXCLUSION, with no positive test on the trip. Given Objection 1 removes the evidence against RAPL, branch (ii) is materially likely, and it ends a hardware session with no answer and a request for a second trip.

0x64F is precisely the register that resolves this ambiguity: on Haswell/Broadwell client it reports, per-bit and with sticky log copies, whether the core frequency was clipped by PROCHOT, thermal, core power limiting, package PL1, package PL2, VR thermal alert, electrical design point, or max-turbo-limit. [VENDOR-DOC, RECALLED — verify bit layout against SDM Vol.4 before use.] The diagnosis defers it ("not worth doing yet") on #GP grounds. But the safe-gating pattern is already in this file and already load-bearing: cpu.rs:163-167 gates MSR_TURBO_RATIO_LIMIT (0x1AD) on CPUID.06H:EAX[1] with an explicit comment that an unhandled #GP in a UEFI app with no IDT is a dead machine. For 0x64F the equivalent gate is a CPUID.01H family/model check for Broadwell client — the same thing turbostat has done for a decade. Deferring the one register that answers the question, on a trip whose entire purpose is to answer that question, is the wrong call. Add 0x64F + 0x610 + 0x606, family/model-gated, Broadwell-only, before the stick is built.

OBJECTION 3 — throttle_diag CANNOT VERIFY THE DIAGNOSIS'S OWN HEADLINE CLAIM. "ratio 5 x 100 MHz = 500 MHz = Broadwell-U LFM exactly" requires the max-efficiency ratio, which lives in MSR_PLATFORM_INFO[47:40]. cpu.rs:50 documents that field in a comment and NO CODE ANYWHERE READS IT — grep confirms MSR_PLATFORM_INFO is read at cpu.rs:96 and cpu.rs:327 and both take only >>8 & 0xFF (the max-NON-turbo ratio). So after running P02 you will have status_ratio=5 and no on-machine reference for what this part's LFM actually is. "Broadwell-U LFM = 500 MHz" is currently VENDOR-DOC/recalled and unverified; some U-series parts in that era have LFM 8 (800 MHz). It is one extra shift on an MSR the code ALREADY reads. Add it, and print base and min ratio in the TURBO_DIAG line.

OBJECTION 4 — "EXACT INTEGER P-STATE" IS OVER-READ. APERF/MPERF gives a TIME-AVERAGE core frequency over the interval, not an instantaneous P-state. "The only integer in [4.84, 5.06) is 5" presupposes the pin it is trying to demonstrate; a core alternating between ratio 4 and ratio 6 averages into the same bucket. Worse, actual_pct_of_nominal (cpu.rs:265-272) floor-divides to integer percent, which at these magnitudes is ~4.5% resolution — it cannot distinguish 22.0% from 22.9%, i.e. 484 MHz from 504 MHz. The claim "an artifact has no reason to land on the architectural minimum ratio and nothing else" is rhetorically strong and evidentially empty, because the instrument cannot resolve "the architectural minimum" from a 4.5%-wide band around it.

The REAL evidence is the invariance: exactly 22 across nine segments spanning 30 s to 688 s, on two different code paths. That is strong and sufficient. Drop the integer-uniqueness argument rather than defending it — and note the diagnosis's own remediation item 2 (log raw da and dm) makes the whole question moot for one format string.

OBJECTION 5 — TWO MODEL-SPECIFIC MSRs ARE GATED ON THE WRONG CPUID BITS, WHICH IS A FLEET SAFETY ISSUE, NOT A STYLE ISSUE.
(a) MSR_POWER_CTL (0x1FC) is gated at cpu.rs:234 on has_platform_info(), which cpu.rs:154-157 DEFINES AS has_aperf_mperf() = CPUID.06H:ECX[0]. APERF/MPERF availability does not imply 0x1FC exists — 0x1FC is model-specific, not architectural, and CPUID has no bit for it. On Goldmont Plus it probably exists (Silvermont-lineage SDM tables list MSR_POWER_CTL with bit 0 bi-directional PROCHOT) but "probably" is not the standard this very file applies to 0x1AD three functions earlier.
(b) IA32_CLOCK_MODULATION (0x19A) is gated at cpu.rs:227 on has_therm() = CPUID.01H:EDX[29] (Thermal Monitor). Per the SDM, EDX[29] gates IA32_THERM_STATUS; on-demand clock modulation is gated by CPUID.01H:EDX[22] (ACPI/thermal-and-software-controlled-clock). Different bit. On mainstream Core parts both are set so this has never bitten — but the stated fleet is Atom-class, and this is exactly the class of mistake the file's own 0x1AD comment exists to prevent.
Both are one-line fixes. Neither is theoretical on a target with no IDT.

OBJECTION 6 — THE COMET LAKE CROSS-CHECK CARRIES NO INFORMATION AS WRITTEN. The headroom section anchors its band with "the QEMU i5-10210U host reached 3.15 tok/s AVX2 ... a Broadwell at 2.2-2.7 GHz landing at 2.2-3.3 against a Comet Lake at 3.15 is coherent." Verified in docs/hardware_logs/gauntlet_dataset.tsv row qemu_i5-10210U_host_v2: 3.15 is real. But that run is a KVM guest — perf_snapshot() early-returns on is_hypervisor() (cpu.rs:255) — so the HOST's actual frequency during that run was never measured and appears nowhere in the log. The i5-10210U spans 1.6 GHz base to 4.2 GHz 1-core turbo, a 2.6x range. A number with 2.6x of unmeasured freedom is consistent with essentially any Broadwell extrapolation and therefore corroborates nothing.

It can be rescued using the diagnosis's own method — the scalar column is the same instruction stream on all three machines. qemu scalar 0.57 tok/s; Dell scalar consumes 3.776e9 ACTUAL core cycles/token (17.162e9 TSC ticks x 0.22, computed today). If Comet Lake's per-clock advantage over Broadwell on that scalar kernel is ~1.15x, the host was running near 1.9 GHz, which then puts Comet Lake at ~1.3x Broadwell per clock on the AVX2 kernel — plausible for Skylake-core + DDR4 vs Broadwell + DDR3L, and it does support the band. State it that way or delete the cross-check.

OBJECTION 7 — A TOP-RANKED CONFOUND IS NEVER NAMED ANYWHERE: WAS THE DELL ON AC POWER? Nothing in gauntlet_dell_i5-5200U_2026-07-12_005500.txt records the power source, no code captures it, and neither the diagnosis nor P02 names AC-vs-battery as a variable (P02's three configurations all assume an adapter is attached). Many platforms under UEFI with no OS leave the core at LFM by firmware policy when on battery, and Dell specifically detects adapter identity over the 1-Wire pin in the barrel connector and derates the CPU when it reads a non-genuine or under-wattage adapter — with a POST message saying so, which is free evidence requiring no MSR at all. This is the cheapest possible explanation for the entire finding and it has never been controlled for.

WHAT I TRIED TO BREAK AND COULD NOT — stated because these were my best shots:
(a) I suspected the fractional prefill seconds made the TSC-rate derivation circular, since the code computes p_secs = secs * (p_ticks/dt) at main.rs:551-556 rather than measuring it. It is not circular: p_ticks/p_secs reduces algebraically to dt_total/secs_total, a genuine TSC-versus-wall measurement. It does mean the diagnosis's two quoted TSC rates (2.1754 and 2.1975 GHz) are NOT independent samples, and each carries the +/-0.5 s quantization of its segment total (+/-0.7% on the 70 s segments, +/-0.12% on the 421 s one). The conclusion "TSC ticks at ~nominal, TSC is innocent" survives comfortably.
(b) The diagnosis's explanation of the 0.61-vs-0.62 wobble as one-second timer quantization is CONFIRMED by the code: secs comes from wall_seconds() (integer on this firmware — every total field is X.000) and d_secs = secs x d_ticks/dt; 69 vs 70 s is 1.4%, exactly the wobble.
(c) P02's central premise is true: `git merge-base --is-ancestor ec1c139 HEAD` returns YES, and `grep -rn TURBO_DIAG docs/hardware_logs/` returns nothing. The instrument exists in the tree and has never executed on any machine.
(d) The provenance note at the end of gauntlet_hp_stream_n4020_2026-07-14_141531.txt independently confirms the byte-identical-binary claim (same 314,368-byte EFI), which is what makes the cross-machine scalar comparison legitimate. That comparison is the strongest thing in the diagnosis and it holds.

### MEASUREMENT & INSTRUMENTATION ADVERSARY — I did not re-litigate the physics. I asked one question of every claim: with the instruments that exist in this tree today, could the claimed number actually be produced, attributed to a machine, and survive a hostile reader? I read aegis-uefi/src/{main.rs,cpu.rs,allocator.rs}, aegis-core/src/{arena.rs,inference.rs}, both gauntlet logs, gauntlet_dataset.tsv, and aegis-core/tests/. Everything below is either a line I read or arithmetic I ran on this box.

WHAT I VERIFIED AND CONFIRMED (the diagnosis's own numbers survived my attack):
- 423 typematic chars x 5 ms = 2.115 s exactly. Counted from main.rs:282,285,293-296,297,443-446 with python3. P03's number is right.
- Six from_secs(1) boot-path stalls confirmed: main.rs:158 (inside load_file_into, x3 files) + :398 + :406 + :414. Exactly 6.000 s. P06's count is right (references/uefi-boot.md's "three" is wrong).
- Arena batch buffers: 30,464 f32/position x 4 B x 2048 = 249,561,088 B. Recomputed from arena.rs:57-66 against aegis_pruned_config.json (hidden 2560, kv_dim 5x128=640, inter 6912). P05's 249.6 MB is exact.
- The 700 MiB hardcode and the silent-underallocation bug are real: allocator.rs:122-128 `while total_allocated < target` with `target = 700*1024*1024`, and allocator.rs:146 `if heap_index >= 16 { break; }` inside the Ok arm with no post-loop check.
- The MSR-free 810 MHz bound survives quantization attack. I checked whether the "185.791 s" and "135.695 s" prefill figures are independent measurements — they are NOT (gbench computes p_secs = secs * p_ticks/dt, so TSC_rate reduces algebraically to dt/secs), so the whole bound rests on integer-second wall times: 421 s and 310 s. Quantization error is +/-0.12% and +/-0.16%. The bound holds.

THE SINGLE MOST IMPORTANT INSTRUMENT FACT NOBODY IN THIS PACKAGE STATED:
ticks/tok is ~1000x more precise than tok/s, and every proposal gates on the wrong one.
  Dell PSTATE_run1/2/3 ticks/tok: 3,548,319,387 / 3,548,334,308 / 3,548,347,118 -> spread 27,731 = 7.8e-6 (CV < 0.001%).
  N4020 SIMD_scalar/SIMD_native/run2: 6,323,622,274 / 6,323,678,937 / 6,323,664,454 -> spread 56,663 = 9.0e-6.
  Meanwhile tok/s over the same runs wobbles 0.61/0.62 and 0.59-0.62 — a ~1.7% quantization jitter, because tok/s is derived from wall_seconds() which is INTEGER-SECOND on this firmware (every `total` field in both logs is X.000s).
RULE THE PROGRAM SHOULD ADOPT: no performance gate may be stated in tok/s. State it in ticks/tok. tok/s is a presentation number; ticks/tok is the measurement. gauntlet_dataset.tsv already half-knows this — its r_simd/r_turbo/r_batch columns are computed from ticks (Dell r_simd 4.837 = 17162364212/3548301534, N4020 r_turbo 2.428 = 6323664454/2604789228), while the displayed tok/s columns are the quantized ones. Nobody wrote that down.

TWO LATENT INSTRUMENT BUGS THAT WILL SILENTLY FABRICATE A NUMBER:
1. wall_seconds() (main.rs:36-45) has NO DATE FIELD — it is seconds-since-midnight only. Every consumer guards with `(Some(a), Some(b)) if b >= a => b - a, _ => 0.0`. A run that crosses midnight therefore reports secs = 0.0, which makes tok/s = 0.00 AND p_secs/d_secs = 0.0 with no error line. The Dell gauntlet started 00:55 and ran ~48 min of logged segment time; it missed this by under an hour. Any future overnight fleet session hits it.
2. /benchmark (main.rs:854-903) counts the callback UNFILTERED. forward_batch emits one "[SYSTEM] Analyzing N tokens..." (inference.rs:180-181) and process_intent emits one "[PERFORMANCE] Average Cycles/Token" (inference.rs:891-897). /gauntlet filters both (main.rs:552) and the plain REPL path filters both (main.rs:915); /benchmark does not. So /benchmark reports 52 tokens for a 50-token run and its ticks/token and tok/s are ~4% optimistic and NOT comparable to any GAUNTLET row. P01 gates on exactly this command.

ON A4 (the audit's "no log at all" finding): A4 is not merely under-evidenced, its evidence column literally reads "commit msgs; post-ugrep clean re-measurement" (program/RESEARCH_LEDGER.md:15) — the only ledger row citing commit messages rather than a file in docs/hardware_logs/. Nothing in P01-P07 is multicore, so none of these proposals repairs or worsens A4. But the same failure mode is being re-created by P01: see below.
- verdict on diagnosis: **SOUND**
- The verdict (real throttle, ratio 5, ~484 MHz) is correct and I could not break it. Four objections, none fatal, all measurement-level:

(1) ARTIFACT-KILLER (c) IS LOGICALLY INVALID AS ARGUED. The diagnosis kills "stuck-at value / broken read" with "the same binary emitting 99%, 100%, and 245% on the N4020". That is CROSS-MACHINE evidence against a MACHINE-SPECIFIC hypothesis. A Dell-local fault — this one CPU's MPERF miscounting, this one firmware's MSR virtualization — is entirely consistent with a different laptop reading correctly. The histogram (30x `?`, 21x 22%, 4x 99%, 4x 245%, 2x 100%) does not exclude it either, because all 21 "22%" and all the non-22 values come from disjoint machines. What actually kills (c) is the MSR-free token-throughput bound, which touches no MSR at all. The diagnosis HAS that argument; it should retire (c) and let the bound carry it alone, otherwise a hostile reviewer finds a broken link in a chain that did not need it.

(2) "PREDICTS THROUGHPUT TO 1.6%" IS AN APPLES/ORANGES COMPARISON, AND THE ERROR BAR IS WRONG. Two problems. First, cpu.rs:271 floor-divides, so "99%" means [99,100) and "245%" means [245,246); the clock ratio is 2.4747-2.4848, not a point value of 2.475. Against a ticks ratio of 2.4277 the agreement is 1.9-2.4%, not 1.6%. Second and worse: clock% is APERF/MPERF over the WHOLE gbench segment (prefill + decode + warm transitions, main.rs:549/554), while 2.4277 is decode-only (engine.last_decode_cycles / last_decode_steps). They are not the same interval. The conclusion survives — nothing else could make an artifact track real work to within a few percent across a 2.4x swing — but the stated precision is manufactured.

(3) THE QEMU CROSS-CHECK FOR THE HEADROOM BAND IS NOT ADMISSIBLE. The diagnosis anchors "2.2-3.3 tok/s on a Broadwell is coherent" against qemu_i5-10210U_host_v2 = 3.15 tok/s (gauntlet_dataset.tsv). That row is a wall-derived, integer-second-quantized tok/s taken inside a hypervisor where msrs_safe() is false (clock column is "?"), on a host whose Linux governor was free to boost a Comet Lake to 4.2 GHz, with a TSC that KVM may offset or scale. It cannot bound a bare-metal Broadwell prediction. Drop it or label it SPECULATION. The band 3.6x-5.3x stands on its own derivation.

(4) THE FALSIFIER THE DIAGNOSIS NAMES IS REACHABLE BY A SAMPLING ARTIFACT — see my P02 objection. Outcome (v) ("status_ratio=27 but gbench still says 22% -> the metric is suspect") can be produced by a load-dependent clamp plus an idle-time sample, with the metric working perfectly. As written, the pre-registered falsifier can fire falsely.

WHAT I TRIED AND FAILED TO BREAK, and should be said out loud because it is the strongest thing in the package: the Dell's null is real at the bit level. 3,548,334,308 -> 3,548,347,118 is +0.00036% on an instrument whose demonstrated run-to-run CV is 7.8e-6. The P-state write did nothing, and that is measured, not inferred.

### NO_STD / NO-OS CONSTRAINTS — with one correction to the brief itself that reframes half the review.

[MEASURED-BY-ME-TODAY] THE BRIEF'S PREMISE IS HALF WRONG, AND IT MATTERS. `grep -rn "exit_boot_services|ExitBootServices" aegis-uefi/ --include=*.rs` returns ZERO hits. A.L.I.C.E. never leaves UEFI Boot Services. It is not a machine-owning unikernel; it is a long-lived UEFI application. So the brief's "no scheduler, no drivers, no interrupts configured" is false by construction: firmware's IDT is installed, firmware's timer interrupt is firing, firmware's XHCI/BOT/FAT32 drivers are doing every byte of the 782 MB load, firmware's console driver renders every glyph, and SMM is live and rendezvouses all logical processors on every SMI. What is genuinely absent is an OSPM/cpufreq governor, a scheduler, threads, and any idle path. Every proposal below must be judged against "firmware is running underneath us," not "nothing is running."

Concretely verified:
- allocator.rs is the ONLY allocator: MultiHeap over 16 `linked_list_allocator::Heap` slots, `spin::Mutex`, backed by `uefi::boot::allocate_pages`. On exhaustion `alloc` returns `null_mut()` (allocator.rs:56) -> Rust `handle_alloc_error` -> panic_handler (main.rs:942) -> `loop { _mm_pause() }`. There is no OOM killer and no watchdog (it is explicitly disabled at main.rs:253). A panic on this target is a bricked session requiring a hard power cycle, and BOOTLOG.TXT does NOT get the line — the panic handler writes to stdout only and has no `Directory` handle. Any proposal whose failure mode is "panic" is a proposal whose failure mode is "dead laptop with no evidence."
- The arena is NOT an arena. arena.rs uses `vec![0.0f32; n]` — global-allocator `Vec`s. The doc comment at inference.rs:133 says "Zero-allocation arena protects us from OOM"; the code allocates 564,133,888 B through the heap. Trust code, not comments.
- SMP: pool.rs:11-13 is `use std::sync::{Arc, Condvar, Mutex}` / `std::thread`, gated on `feature = "parallel"`, which is NOT in aegis-core's default features (`default = ["int8_act"]`, Cargo.toml:21) and whose own header (pool.rs:8-9) says "the no_std UEFI unikernel never sees this file." There is zero multicore code in the UEFI build. Any future SMP work is a rewrite, not a feature flag: a Condvar pool cannot be ported (no scheduler; workers must spin on `spin::Mutex`/atomics).
- AP startup, since the brief asks: raw INIT-SIPI-SIPI is the WRONG answer while Boot Services are alive. Firmware owns the APs, parks them, and requires them for SMM rendezvous on every SMI; hijacking one with your own GDT/stack while SMIs still arrive is how you get an SMM rendezvous timeout / machine check. The sanctioned path exists and is already in the dependency: uefi 0.38 ships `proto/pi/mp.rs` (EFI_MP_SERVICES_PROTOCOL, `StartupAllAPs`) — verified present in the vendored crate. INIT-SIPI-SIPI only becomes correct AFTER ExitBootServices, and at that instant you lose SimpleFileSystem (no BOOTLOG.TXT), the text console, `boot::stall`, `allocate_pages`, and `get_time` — i.e. the entire evidence pipeline this program's doctrine is built on. An SMP proposal that calls ExitBootServices is a proposal to make its own result unrecordable. It should be rejected on doctrine grounds before it is evaluated on engineering grounds. (Separately, the diagnosis's own roofline already caps 4-core at ~11-13 tok/s bandwidth-bound, so SMP is not the multiplier it looks like.)

Nothing in P01-P07 asks for APs. The proposals that are secretly asking for an OS are subtler: P05 assumes a fault is recoverable (it is not), and P07 assumes the heap size can be computed before the thing that knows the size has been constructed.

[NOTE] The proposal list arrived truncated mid-P07 ("...WITHOUT checking to"). P08-P12 are referenced (P09, P11) but were not delivered. I reviewed P01-P06 in full and P07 on its visible text only.
- verdict on diagnosis: **SOUND**
- The verdict is right and I could not break it. I re-derived the load-bearing numbers today from the primary logs and they hold:

VERIFIED BY ME TODAY:
- Dell scalar 17,162,364,212 ticks/tok and AVX2 3,548,301,534 ticks/tok, both at "clock 22%" — docs/hardware_logs/gauntlet_dell_i5-5200U_2026-07-12_005500.txt lines 12-13. The bit-level null: PSTATE_run2_control 3,548,334,308 vs PSTATE_run3_turbo 3,548,347,118 = +0.00036%. Confirmed.
- N4020 hwp=false turbo=true, scalar 6,323,664,454 -> 2,604,789,228 after `GAUNTLET TURBO: legacy ratio=28`, clock 99% -> 245%. Confirmed at docs/hardware_logs/gauntlet_hp_stream_n4020_2026-07-14_141531.txt:41-43.
- The floor-division argument at cpu.rs:271 (`da.saturating_mul(100)/dm`) and the ratio-5 uniqueness: confirmed by reading the source.
- The stuck-at rebuttal: same binary emits 99/100/245/`?`. Confirmed.

TWO OBJECTIONS, NEITHER FATAL:

(1) THE MSR-FREE PROOF RESTS ON THE SCALAR LEG, BUT THE ROOFLINE SANITY CHECK WAS RUN ON THE AVX2 LEG. The 810 MHz bound is computed by normalizing tok/s by GHz — which is only legitimate if the scalar kernel is compute-bound on BOTH machines. If either scalar leg were memory-bound, throughput would not scale with core clock and dividing by GHz would be meaningless. The diagnosis proves compute-boundedness for the AVX2 path (17.7% of AVX2 peak) and never closes it for the scalar path, which is the one the whole argument stands on. I closed it: Dell scalar 0.12804 tok/s x 780 MB/tok = 100 MB/s; N4020 scalar 0.17285 x 780 = 135 MB/s. Both are ~1-2% of any plausible DDR3L-1600 or DDR4-2400 rate. Massively compute-bound, normalization valid, conclusion stands. But the diagnosis should not have shipped the bound without that line — it is the single step an adversary would attack first.

(2) "1.673 IS EXACTLY THE EXPECTED IPC RATIO FOR THAT GENERATION PAIR" IS SPECULATION LABELLED AS CORROBORATION. There is no citation, no measurement, and no reason a ternary LUT kernel's IPC ratio should match any published Broadwell/Goldmont+ figure. The <=810 MHz BOUND is hard and survives without it. The 484 MHz POINT ESTIMATE is corroborated only by the MSR — which is precisely the instrument the artifact hypothesis attacks. Those two claims have different evidentiary status and the writeup merges them. Relabel: bound = MEASURED, point estimate = MSR-DEPENDENT.

MINOR, no-OS-specific: the RAPL discriminator ("scalar and AVX2 would settle at different frequencies under a power equilibrium") is the weaker of the two arguments available. The stronger one is that no PL1 setting on a 15 W Broadwell-U — not even configurable-TDP-down at 7.5 W — produces 484 MHz on a single-core workload; at ratio 5 the package is drawing single-digit watts. Also: the bus is 100 MHz by ASSUMPTION, not measurement — the Dell log's own banner reads `base=0MHz max=0MHz bus=0MHz`, so CPUID leaf 0x16 is absent and cpu.rs:82 falls back to a hardcoded 100. The TSC cross-check (2.1975/22 = 99.9 MHz) rescues it, but that should be stated as the reason rather than left implicit.

  - REJECTS P05-MEM-CHUNKED-PREFILL: The MEMORY ANALYSIS IS EXACTLY RIGHT and I reproduced every byte of it. The MECHANISM AS SPECIFIED WILL BRICK A FLEET MACHINE AND BREAKS ITS OWN GATE 1. Both halves matter.

What holds. Recomputed from arena.rs:57-66 against aegis_pruned_config.json (hidden 2560, kv 5x128=640, intermediate 6912, max_position_embeddings 2048): 30,464 f32/position = 121,856 B/pos x 2048 = 249,561,088 B. KV cache from kvcache.rs:19-22 is 2 x 30 x 2048 x 640 x 4 = 314,572,800 B. Sum 564,133,888 against a measured PEAK_MEMORY of 571,912,597 (identical in both hardware logs) — a residual of 7,778,709 B for logits, s
    -> Keep the memory analysis, move the mechanism, and rewrite GATE 1.

1. CHUNK INSIDE forward_batch_with_capture, NOT IN CALLERS. Make the function itself loop over `batch_tokens.chunks(PREFILL_CHUNK)`, advancing seq_pos_start by the chunk length. All four call sites become safe by construction, no caller can get it wrong, and no future caller can reintroduce the hazard. This is the whole fix and it 

### MISSION FIT — does this work serve a DDIL/contested-environment decision-support tool, or is it engineering for its own sake? Judged against the actually-submitted document (docs/RFI_SN_26_97_ALICE_DRAFT_V2.md, sent 2026-07-14) rather than against the roadmap's self-description, because the RFI is what a reviewer will hold us to.
- verdict on diagnosis: **SOUND**
- The technical diagnosis is the best-argued document in this program and I could not break it. I spot-checked the load-bearing arithmetic myself: 17,162,364,212 / 2.1975e9 = 7.810 s/tok = 0.1280 tok/s and 6,323,622,274 / 1.0930e9 = 5.785 s/tok = 0.1729 tok/s, both consistent with docs/hardware_logs/gauntlet_dataset.tsv rows dell_i5-5200U_inspiron15 (SIMD_scalar 0.12) and hp_stream_n4020 (0.17); 0.12804/0.15814 = 0.8097 GHz. The IPC-parity floor is genuinely charitable (Goldmont Plus is 3-wide with 2x128b load; Broadwell is 4-wide with 2x256b — on scalar SSE2 Broadwell should be 1.3-1.7x, not 1.0x), so the 810 MHz ceiling holds. The self-deflation in real_headroom_estimate (n=1 machine, does not multiply with multicore, 40-50% odds of a software fix) is exactly the discipline this program needs. I am not going to manufacture a technical objection where none exists.

MY OBJECTION IS THAT THE DIAGNOSIS NEVER COMPUTES ITS OWN MISSION DENOMINATOR, AND THE DENOMINATOR IS DEVASTATING.

[MEASURED-IN-AN-EXISTING-LOG + arithmetic by me] The mission is stated at docs/RFI_SN_26_97_ALICE_DRAFT_V2.md:38 — "type a question, read the answer." A useful decision-support answer is ~150 tokens. On the primary bare-metal machine at the logged 0.61 tok/s that is 246 seconds. Grant the diagnosis its own central estimate of 4.4x, in full, with no probability weighting: 2.7 tok/s, 56 seconds per answer. That is the *best case* of the entire throttle lane. Fifty-six seconds is still not a tool an isolated operator uses under stress; it is a tool they abandon. The diagnosis calls the P-state work "the single most important negative result in this program." Measured against the mission it is a 4x on a configuration that is ~50x away from usable, on one laptop that is probably just sick.

[VERIFIED BY ME TODAY] Meanwhile the program already holds a strictly dominating alternative it never mentions. Falcon-E-1B artifacts on this box: MODEL.SAF 507M + EMBED.BIN 128M + VOCAB.BIN 652K = ~635 MB versus BitNet's 522,831,576 + 257,310,720 + 1,759,936 = 782 MB. Ledger B3 logs it at 4.39 tok/s decode versus BitNet's 2.62 tok/s single-thread (A4) — 1.68x faster AND 147 MB smaller, on the same binary, with T2d tokenizer parity PASS (B6) and G4a engine-vs-reference parity at 0.12% (B8). It is done. It needs a repack and a USB stick. The throttle lane is a 40-50%-probability 4.4x on one machine; the portfolio swap is a certain 1.7x on every machine, and it also moves the 2 GB memory gate. The diagnosis's own "certain win" framing was applied to the boot-path P-state call — correctly — but the actually-certain win is one lane over and unmentioned.

SECOND OBJECTION, SPECIFIC AND UNADDRESSED. The diagnosis recommends calling request_max_performance() on the boot path as its one no-argument-needed action. It never asks whether an unconditional Ring-0 write to IA32_PERF_CTL on every boot is compatible with the mission claims the same program submitted. RFI §3a:36-39 sells "nothing to misconfigure under stress" and "no step at which deployment can be done wrong"; RFI §5 names firmware diversity as "empirically our dominant deployment-failure class (four distinct per-machine firmware quirks required targeted fixes to date)." An unconditional power-management write, on a fleet whose firmware behavior is enumerated on exactly two machines, executed on hardware the RFI describes as "scavenged," is the single most attack-surface-adding and most physically-damaging thing in the whole proposal set. The measured 2.428x comes from the HP Stream N4020 — a 6 W fanless passively-cooled Celeron, i.e. the machine in the fleet with the least thermal headroom. Pinning that part at 245% of nominal through a 782 MB load plus 700 MiB of heap zeroing, with no thermal abort, is how you get a thermal shutdown mid-answer. In DDIL terms that is not a slow answer, it is a mission kill, and it is worse than the problem being solved.

[SPECULATION, but it is the right prior] Third: the diagnosis's mechanism ranking puts external bd-PROCHOT on a 2015 Dell Inspiron first, and its own Step 3 (swap charger, pull battery) can settle it in twenty minutes with zero code. It then buries that twenty-minute test at position 3 of 4 in what_would_settle_it while placing three code changes ahead of it. It says "it should be run FIRST" and then does not order the plan that way. If the answer is a degraded battery, every hour of P-state engineering budgeted against this machine is wasted, and — this is the part nobody has said out loud — RFI §5 risk (1) told DARPA "the Dell's firmware ignores the same request, so per-family handling remains engineering work." If Step 3 comes back clean under an OEM adapter, that submitted sentence is wrong and we owe a correction at the workshop. The cheapest experiment in the set is also the only one with a documentation obligation attached, and it is ranked third.

  - REJECTS P06-BOOT-STALL-AB: CUT THIS FIRST. It is the only proposal in the set that attacks the program's own self-declared number-one deployment risk in exchange for a mission-irrelevant gain, and its gate is structurally incapable of proving what it needs to prove.

One: the target. docs/RFI_SN_26_97_ALICE_DRAFT_V2.md §5 risk (2) tells DARPA that firmware diversity across OEM UEFI implementations is 'empirically our dominant deployment-failure class (four distinct per-machine firmware quirks required targeted fixes to date).' The stalls are, by the code's own comment at main.rs:157, a USB-controller workaround — i.e. t
    -> CUT as proposed. Replace with a strictly smaller change that is not an A/B at all: delete the three post-load stalls at aegis-uefi/src/main.rs:398, :406, :414, keep the pre-read stall at :158. Justification is semantic, not empirical — the pre-read stall already provides the 'settle before touching the controller' behavior both comments describe, and the :414 stall precedes no read whatsoever. Exp

