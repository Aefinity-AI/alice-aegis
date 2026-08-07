//! Direct measurement of what the CPU is actually doing, from inside the
//! unikernel. Written to test one hypothesis:
//!
//!   Bare-metal inference is slow because there is no operating system to
//!   raise the processor's P-state. The firmware leaves the core near its
//!   minimum clock and nothing ever asks it to go faster.
//!
//! `rdtsc` cannot see this. On modern Intel the TSC is *invariant*: it ticks at
//! the nominal frequency regardless of the core's real speed, so a throttled
//! core simply appears to need more "cycles" per token. To see the truth you
//! must read the hardware's own accounting.
//!
//! IA32_APERF (MSR 0xE8) counts at the **actual** core clock.
//! IA32_MPERF (MSR 0xE7) counts at the **nominal** clock (TSC rate).
//! Their ratio over an interval IS the P-state, measured, not inferred.
//!
//! UEFI applications run in ring 0, so `rdmsr`/`wrmsr` are available.

use core::arch::x86_64::__cpuid;
use core::arch::x86_64::__cpuid_count;

/// Crate-visible so the H3 probe (h3.rs) can read the architectural MTRR/PAT
/// MSRs. Callers own the existence proof: rdmsr of an absent MSR raises #GP,
/// which with no exception handler is a dead machine (see has_turbo()).
#[inline]
pub(crate) unsafe fn rdmsr(msr: u32) -> u64 {
    let lo: u32;
    let hi: u32;
    core::arch::asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack));
    ((hi as u64) << 32) | (lo as u64)
}

#[inline]
unsafe fn wrmsr(msr: u32, val: u64) {
    core::arch::asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") val as u32,
        in("edx") (val >> 32) as u32,
        options(nomem, nostack)
    );
}

const IA32_MPERF: u32 = 0xE7;
const IA32_APERF: u32 = 0xE8;
const IA32_PM_ENABLE: u32 = 0x770;
const IA32_HWP_CAPABILITIES: u32 = 0x771;
const IA32_HWP_REQUEST: u32 = 0x774;

// Model-specific. Present on every modern Intel core, but reading them on a
// non-Intel CPU or inside a hypervisor can raise #GP, which on bare metal is
// fatal. Both are gated behind `msrs_safe()`.
const MSR_PLATFORM_INFO: u32 = 0xCE; // [15:8] max non-turbo ratio, [47:40] min ratio
const MSR_TURBO_RATIO_LIMIT: u32 = 0x1AD; // [7:0] 1-core turbo ratio
const IA32_PERF_STATUS: u32 = 0x198; // [15:8] current ratio
const IA32_PERF_CTL: u32 = 0x199; // [15:8] requested ratio (legacy SpeedStep)
const IA32_MISC_ENABLE: u32 = 0x1A0; // [16] EIST enable, [38] turbo disable
const IA32_CLOCK_MODULATION: u32 = 0x19A; // [4] on-demand throttle enable, [3:0] duty
const IA32_THERM_STATUS: u32 = 0x19C; // [0] hot, [2] PROCHOT#/FORCEPR, [3] its sticky log, [22:16] deg below TjMax
const MSR_POWER_CTL: u32 = 0x1FC; // [0] bi-directional PROCHOT enable (Nehalem+)

/// CPUID.1:ECX[31] — set by every mainstream hypervisor.
pub fn is_hypervisor() -> bool {
    unsafe { __cpuid(1).ecx & (1 << 31) != 0 }
}

pub fn is_intel() -> bool {
    unsafe {
        let r = __cpuid(0);
        // "Genu" "ntel" "ineI"
        r.ebx == 0x756E_6547 && r.edx == 0x4965_6E69 && r.ecx == 0x6C65_746E
    }
}

/// Whether it is safe to touch Intel's model-specific power MSRs.
/// Under a hypervisor these are usually unimplemented and `rdmsr` faults.
pub fn msrs_safe() -> bool {
    is_intel() && !is_hypervisor()
}

/// Bus clock in MHz, needed to turn a P-state ratio into a frequency.
/// Modern Intel uses a 100 MHz bus; CPUID leaf 0x16 reports it when available.
fn bus_mhz() -> u32 {
    let (_, _, bus) = frequencies_mhz();
    if bus != 0 { bus } else { 100 }
}

/// (current, max_non_turbo, max_turbo_1core) P-state ratios. None if unreadable.
/// `max_turbo_1core` is 0 when the part has no Turbo Boost — its MSR does not
/// exist there and must not be read. Multiply by `bus_mhz()` for MHz.
pub fn ratios() -> Option<(u8, u8, u8)> {
    if !msrs_safe() {
        return None;
    }
    unsafe {
        // Each MSR is read only when the CPU says it exists. A #GP here would be
        // fatal, and this runs on borrowed machines.
        let cur = if has_eist() {
            ((rdmsr(IA32_PERF_STATUS) >> 8) & 0xFF) as u8
        } else {
            0
        };
        let base = if has_platform_info() {
            ((rdmsr(MSR_PLATFORM_INFO) >> 8) & 0xFF) as u8
        } else {
            0
        };
        let turbo = if has_turbo() {
            (rdmsr(MSR_TURBO_RATIO_LIMIT) & 0xFF) as u8
        } else {
            0
        };
        if cur == 0 && base == 0 {
            return None;
        }
        Some((cur, base, turbo))
    }
}

pub fn mhz_from_ratio(ratio: u8) -> u32 {
    ratio as u32 * bus_mhz()
}

/// 48-character CPU brand string from CPUID 0x80000002..0x80000004.
pub fn brand_string(buf: &mut [u8; 48]) -> &str {
    unsafe {
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return "unknown";
        }
        let mut i = 0;
        for leaf in 0x8000_0002u32..=0x8000_0004 {
            let r = __cpuid(leaf);
            for reg in [r.eax, r.ebx, r.ecx, r.edx] {
                buf[i..i + 4].copy_from_slice(&reg.to_le_bytes());
                i += 4;
            }
        }
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(48);
    core::str::from_utf8(&buf[..end])
        .unwrap_or("unknown")
        .trim()
}

/// (base MHz, max MHz, bus MHz) from CPUID leaf 0x16. Zeros if unsupported.
/// This gives the nominal TSC frequency without asking anyone to read a BIOS screen.
pub fn frequencies_mhz() -> (u32, u32, u32) {
    unsafe {
        if __cpuid(0).eax < 0x16 {
            return (0, 0, 0);
        }
        let r = __cpuid(0x16);
        (r.eax & 0xFFFF, r.ebx & 0xFFFF, r.ecx & 0xFFFF)
    }
}

/// CPUID.06H:ECX[0] — hardware coordination feedback (APERF/MPERF) available.
pub fn has_aperf_mperf() -> bool {
    unsafe { __cpuid_count(6, 0).ecx & 1 != 0 }
}

/// CPUID.06H:EAX[7] — Hardware-Managed P-states (Speed Shift) available.
pub fn has_hwp() -> bool {
    unsafe { __cpuid_count(6, 0).eax & (1 << 7) != 0 }
}

/// CPUID.01H:ECX[7] — Enhanced SpeedStep. IA32_PERF_STATUS (0x198) and
/// IA32_PERF_CTL (0x199) are architectural *only when this bit is set*.
pub fn has_eist() -> bool {
    unsafe { __cpuid(1).ecx & (1 << 7) != 0 }
}

/// MSR_PLATFORM_INFO (0xCE) arrived with Nehalem, alongside APERF/MPERF.
/// Use the latter as the availability proxy: a Core 2 has neither.
pub fn has_platform_info() -> bool {
    has_aperf_mperf()
}

/// CPUID.06H:EAX[1] — Intel Turbo Boost available.
///
/// This gate is load-bearing: `MSR_TURBO_RATIO_LIMIT` (0x1AD) **does not exist**
/// on parts without Turbo Boost, and `rdmsr` on a nonexistent MSR raises #GP.
/// A UEFI application installs no exception handler, so that is a dead machine —
/// and this code is about to run on other people's laptops.
pub fn has_turbo() -> bool {
    unsafe { __cpuid_count(6, 0).eax & (1 << 1) != 0 }
}

/// CPUID.01H:EDX[29] — Thermal Monitor. Gates IA32_THERM_STATUS and
/// IA32_CLOCK_MODULATION, which do not exist without it.
fn has_therm() -> bool {
    unsafe { __cpuid(1).edx & (1 << 29) != 0 }
}

pub struct ThermDiag {
    /// PROCHOT#/FORCEPR asserted right now — includes an *external* assertion
    /// (charger/battery/EC) when bi-directional PROCHOT is enabled. This is the
    /// classic reason an old laptop sits at minimum clock no matter what
    /// IA32_PERF_CTL requests.
    pub prochot_now: bool,
    /// Sticky since last clear: PROCHOT fired at some point this boot.
    pub prochot_log: bool,
    /// Core at thermal limit right now.
    pub hot_now: bool,
    /// Digital readout: degrees Celsius below TjMax (0 = at the limit).
    pub temp_below_tjmax: u8,
}

/// Everything that can clamp the clock from OUTSIDE the EIST request path.
/// `request_max_performance()` proves the ask was made; this proves whether
/// something else is vetoing it.
pub struct ThrottleDiag {
    pub status_ratio: u8, // IA32_PERF_STATUS[15:8] — what the core is doing
    pub ctl_ratio: u8,    // IA32_PERF_CTL[15:8] — what was last requested
    pub eist_enabled: bool,
    pub turbo_disabled: bool, // IA32_MISC_ENABLE[38]
    pub clock_mod: u8,        // IA32_CLOCK_MODULATION[4:0]; 0x10.. = throttling
    pub therm: Option<ThermDiag>,
    pub bd_prochot_enabled: Option<bool>, // MSR_POWER_CTL[0]; None pre-Nehalem
}

pub fn throttle_diag() -> Option<ThrottleDiag> {
    if !msrs_safe() || !has_eist() {
        return None;
    }
    unsafe {
        let status_ratio = ((rdmsr(IA32_PERF_STATUS) >> 8) & 0xFF) as u8;
        let ctl_ratio = ((rdmsr(IA32_PERF_CTL) >> 8) & 0xFF) as u8;
        let misc = rdmsr(IA32_MISC_ENABLE);
        let (clock_mod, therm) = if has_therm() {
            let ts = rdmsr(IA32_THERM_STATUS);
            (
                (rdmsr(IA32_CLOCK_MODULATION) & 0x1F) as u8,
                Some(ThermDiag {
                    hot_now: ts & 1 != 0,
                    prochot_now: ts & (1 << 2) != 0,
                    prochot_log: ts & (1 << 3) != 0,
                    temp_below_tjmax: ((ts >> 16) & 0x7F) as u8,
                }),
            )
        } else {
            (0, None)
        };
        let bd_prochot_enabled = if has_platform_info() {
            Some(rdmsr(MSR_POWER_CTL) & 1 != 0)
        } else {
            None
        };
        Some(ThrottleDiag {
            status_ratio,
            ctl_ratio,
            eist_enabled: misc & (1 << 16) != 0,
            turbo_disabled: misc & (1u64 << 38) != 0,
            clock_mod,
            therm,
            bd_prochot_enabled,
        })
    }
}

/// Outcome of trying to switch off bi-directional PROCHOT.
pub enum BdProchotClear {
    /// MSR_POWER_CTL[0] was already 0 — nothing to do, and nothing was written.
    AlreadyDisabled,
    /// Written and confirmed by read-back. `prochot_was_asserted` records whether
    /// PROCHOT# was actually being pulled at the time, which is the difference
    /// between "fixed a live throttle" and "pre-emptively disabled a signal".
    Cleared { prochot_was_asserted: bool },
    /// The write did not stick: firmware or hardware is holding the bit set.
    /// Reported rather than assumed — the whole point of the read-back.
    WouldNotClear,
}

/// Switch off bi-directional PROCHOT (MSR_POWER_CTL[0]).
///
/// WHY THIS EXISTS. Measured on the Dell i5-5200U, 2026-07-29, bare metal:
///     cur_ratio=5 req_ratio=27 eist=1 turbo_dis=0 clkmod=0x00
///     prochot_now=1 prochot_log=1 hot=0 temp=Tj-72C bdprochot_en=1
/// EIST is on, turbo is not disabled, clock modulation is off and the core is
/// 72 C BELOW its thermal limit — yet it sits at ratio 5 (~500 MHz on a 2200 MHz
/// part) and ignores a ratio-27 request. The only remaining explanation is the
/// last three fields: something OUTSIDE the CPU is asserting PROCHOT#, and with
/// bi-directional PROCHOT enabled the core obeys that pin and pins itself to its
/// minimum ratio. IA32_PERF_CTL is a request to an arbiter; the pin outranks it.
/// Clearing this bit makes the core stop honouring the external assertion.
///
/// SAFETY — READ THIS BEFORE ENABLING IT ANYWHERE ELSE.
/// bd-PROCHOT exists so that a laptop's charger, battery or voltage regulator can
/// tell the CPU to back off when IT is in trouble, not when the die is hot. On a
/// machine whose VRM or battery is genuinely failing, disabling it removes a real
/// protection and lets the CPU draw power the platform may not be able to deliver.
/// It is defensible HERE only because the same diagnostic proves the assertion is
/// not thermal (hot=0, Tj-72C). It is a diagnostic and demo lever, not something
/// to ship blind on unknown hardware. Every attempt is logged with its read-back.
pub fn clear_bd_prochot() -> Result<BdProchotClear, &'static str> {
    if !msrs_safe() {
        return Err("not a bare-metal Intel CPU (hypervisor or non-Intel)");
    }
    // MSR_POWER_CTL is Nehalem+. has_platform_info() is the same gate the
    // diagnostic uses to decide whether reading it is safe; reading or writing
    // an absent MSR raises #GP, which on bare metal is fatal.
    if !has_platform_info() {
        return Err("pre-Nehalem: MSR_POWER_CTL absent");
    }
    unsafe {
        let before = rdmsr(MSR_POWER_CTL);
        if before & 1 == 0 {
            return Ok(BdProchotClear::AlreadyDisabled);
        }
        // Was the pin actually being pulled? Only meaningful with Thermal Monitor.
        let asserted = has_therm() && (rdmsr(IA32_THERM_STATUS) & (1 << 2)) != 0;

        // Preserve every other bit — C1E lives at [1] and the upper bits are
        // model-specific. Clear exactly one bit.
        wrmsr(MSR_POWER_CTL, before & !1u64);

        if rdmsr(MSR_POWER_CTL) & 1 != 0 {
            return Ok(BdProchotClear::WouldNotClear);
        }
        Ok(BdProchotClear::Cleared {
            prochot_was_asserted: asserted,
        })
    }
}

#[derive(Clone, Copy)]
pub struct PerfSnapshot {
    pub aperf: u64,
    pub mperf: u64,
}

pub fn perf_snapshot() -> Option<PerfSnapshot> {
    // CPUID must claim the counters AND we must not be under a hypervisor that
    // advertises them without implementing the MSRs.
    if !has_aperf_mperf() || is_hypervisor() {
        return None;
    }
    unsafe {
        Some(PerfSnapshot {
            aperf: rdmsr(IA32_APERF),
            mperf: rdmsr(IA32_MPERF),
        })
    }
}

/// Actual clock as a percentage of nominal, over the interval between two
/// snapshots. 100 means "running at the nominal/base frequency"; below that the
/// core is throttled; above it, turbo.
pub fn actual_pct_of_nominal(a: PerfSnapshot, b: PerfSnapshot) -> Option<u64> {
    let da = b.aperf.wrapping_sub(a.aperf);
    let dm = b.mperf.wrapping_sub(a.mperf);
    if dm == 0 {
        return None;
    }
    Some(da.saturating_mul(100) / dm)
}

/// Ask the processor to run at its highest performance level, via Intel's
/// Hardware-Managed P-states (Speed Shift). This is, in essence, the one job an
/// operating system's cpufreq governor performs that a bare-metal application
/// otherwise never does.
///
/// Returns Err with a reason when unsupported, so the caller can report honestly
/// rather than silently doing nothing.
pub enum Boost {
    /// Skylake and later: Hardware-Managed P-states (Speed Shift).
    Hwp { highest: u8 },
    /// Sandy Bridge .. Broadwell: legacy Enhanced SpeedStep via IA32_PERF_CTL.
    LegacySpeedStep { ratio: u8, mhz: u32 },
}

pub fn request_max_performance() -> Result<Boost, &'static str> {
    if !msrs_safe() {
        return Err("not a bare-metal Intel CPU (hypervisor or non-Intel)");
    }

    if has_hwp() {
        unsafe {
            // Enabling HWP is a one-way latch until reset; harmless to set twice.
            wrmsr(IA32_PM_ENABLE, 1);
            if rdmsr(IA32_PM_ENABLE) & 1 == 0 {
                return Err("IA32_PM_ENABLE would not latch");
            }
            // Highest_Performance is bits [7:0] of IA32_HWP_CAPABILITIES.
            let highest = (rdmsr(IA32_HWP_CAPABILITIES) & 0xFF) as u8;
            if highest == 0 {
                return Err("HWP capabilities report highest performance = 0");
            }
            // IA32_HWP_REQUEST: Min[7:0], Max[15:8], Desired[23:16], EPP[31:24].
            // Pin min and max to the top, leave Desired = 0 (hardware picks within
            // the window), Energy-Performance Preference = 0 (maximum performance).
            let req = (highest as u64) | ((highest as u64) << 8);
            wrmsr(IA32_HWP_REQUEST, req);
            return Ok(Boost::Hwp { highest });
        }
    }

    // No HWP — pre-Skylake. Use legacy Enhanced SpeedStep. This is precisely the
    // mechanism a Linux cpufreq governor drives, and which nothing drives here.
    if !has_eist() {
        return Err("no HWP and no Enhanced SpeedStep (CPUID.01H:ECX[7] clear)");
    }
    if !has_platform_info() {
        return Err("pre-Nehalem: MSR_PLATFORM_INFO absent, cannot read base ratio");
    }
    unsafe {
        let misc = rdmsr(IA32_MISC_ENABLE);
        if misc & (1 << 16) == 0 {
            return Err("EIST disabled in firmware (IA32_MISC_ENABLE[16] clear)");
        }
        let base = ((rdmsr(MSR_PLATFORM_INFO) >> 8) & 0xFF) as u8;

        // MSR_TURBO_RATIO_LIMIT exists only when Turbo Boost does. Reading it
        // otherwise raises #GP and kills the machine.
        let turbo_1c = if has_turbo() {
            (rdmsr(MSR_TURBO_RATIO_LIMIT) & 0xFF) as u8
        } else {
            0
        };
        // Firmware can disable turbo independently: IA32_MISC_ENABLE[38].
        let turbo_disabled = misc & (1u64 << 38) != 0;

        // Prefer the single-core turbo ratio when it is both available and
        // permitted; otherwise the maximum non-turbo ratio, which is still an
        // enormous step up from whatever minimum the firmware left us at.
        let target = if turbo_1c > base && !turbo_disabled {
            turbo_1c
        } else {
            base
        };
        if target == 0 {
            return Err("could not read a usable P-state ratio");
        }
        // IA32_PERF_CTL[15:8] = requested ratio. Preserve the other bits.
        let ctl = rdmsr(IA32_PERF_CTL) & !0xFF00;
        wrmsr(IA32_PERF_CTL, ctl | ((target as u64) << 8));
        Ok(Boost::LegacySpeedStep {
            ratio: target,
            mhz: mhz_from_ratio(target),
        })
    }
}

/// CPUID.01H:ECX[3] — MONITOR/MWAIT available. Gates the AP parking path.
pub fn has_mwait() -> bool {
    __cpuid(1).ecx & (1 << 3) != 0
}

/// Raw MSR_TURBO_RATIO_LIMIT (0x1AD). Byte 0 is the 1-core turbo ratio, byte 1
/// the 2-core ratio, and so on up the core counts. The 2026-07-31 protocol
/// boots requested byte 0 (27) and were granted byte 1's value (25) on the
/// i5-5200U — evidence that a second core was still counted as active. Turbo
/// bin accounting counts cores in C0/C1; only C3-or-deeper retires a core.
pub fn turbo_ratio_limit_raw() -> Option<u64> {
    if !msrs_safe() || !has_turbo() {
        return None;
    }
    // SAFETY: gated on msrs_safe() (Intel, bare metal) and has_turbo() —
    // MSR_TURBO_RATIO_LIMIT exists only on Turbo Boost parts (see has_turbo).
    unsafe { Some(rdmsr(MSR_TURBO_RATIO_LIMIT)) }
}

/// Monitor target for parked APs. A static so it outlives every AP stack;
/// nothing ever writes it, so a parked AP sleeps until a hardware wake
/// (SMI/INIT), then immediately re-arms.
static AP_PARK_LINE: u64 = 0;

/// AP parking procedure handed to EFI_MP_SERVICES.StartupAllAPs. Never returns.
///
/// WHY THIS EXISTS. Turbo ratio bins count cores sitting in C0/C1 as active.
/// Firmware parks APs in HLT (C1) or a spin loop, so even a single-threaded
/// workload is capped at the multi-core turbo bin — measured on the Dell
/// i5-5200U as cur_ratio=25 against a req_ratio=27 on every 2026-07-31 boot.
/// MWAIT with the C6 hint (0x20 on Broadwell, per Linux intel_idle's bdw
/// table) retires the core from turbo accounting. This is the exact job
/// Linux's cpuidle/intel_idle driver does, and the reason the minimal-Linux
/// arm reached the full 2.7 GHz 1-core turbo while the unikernel never did.
pub extern "efiapi" fn ap_park_mwait_c6(_arg: *mut core::ffi::c_void) {
    // SAFETY: runs only on an AP dispatched by MP services, ring 0, with this
    // application in full control of the machine. CLI is safe — this AP is
    // never given work again. MONITOR/MWAIT is gated by has_mwait() at the
    // dispatch site. The monitored line is a static that no one writes, so the
    // AP stays in C6 until a hardware wake event, then re-arms in the loop.
    // The procedure intentionally never returns; the dispatch is non-blocking.
    unsafe {
        core::arch::asm!("cli", options(nomem, nostack));
        let line = &AP_PARK_LINE as *const u64 as usize;
        loop {
            core::arch::asm!(
                "monitor",
                in("rax") line,
                in("ecx") 0u32,
                in("edx") 0u32,
                options(nostack)
            );
            core::arch::asm!(
                "mwait",
                in("eax") 0x20u32, // C6 hint on Broadwell (intel_idle bdw table)
                in("ecx") 0u32,    // no extensions: IF=0 stays asleep on masked interrupts
                options(nostack)
            );
        }
    }
}
