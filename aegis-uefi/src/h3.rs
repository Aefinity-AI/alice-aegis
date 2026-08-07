//! H3 probe: memory-attribute audit for the MECH-A1 anomaly.
//!
//! MECH-A1: `process_intent` measured ~1.65-1.67 GIGAticks/token under UEFI on
//! the Dell i5-5200U (mech_U_BOOTLOG_2026-08-01.txt) while the SAME function on
//! the SAME machine under minimal Linux took 6.56M cycles/token, and the SAME
//! function under UEFI on the previous (oscost 07-31) binary took 6.59-13.9M
//! ticks/token. Hypothesis H3: the bigger MECH binary shifted where the
//! firmware placed our allocations, and this boot the model / arena / KV
//! buffers landed in a physical range whose effective memory type is not
//! write-back (UC/WT/WC) — every load then goes to DRAM and decode craters.
//!
//! This block dumps the evidence, one "H3 "-prefixed line at a time, into
//! BOOTLOG.TXT via `boot_log` so a stick that comes home can be read:
//!  - IA32_MTRRCAP, IA32_MTRR_DEF_TYPE, every variable-range MTRR base/mask
//!    pair (guarded by MTRRCAP.VCNT), the fixed-range MTRRs, IA32_PAT;
//!  - the UEFI memory-map descriptors overlapping each range the engine
//!    actually touches (loaded image, MODEL.SAF, EMBED.BIN, VOCAB.BIN, heaps);
//!  - a static UC/WC/WT/WB verdict per range (`mtrr_decode`, host-unit-tested).
//!
//! Read-only: this probe writes no MSR and changes nothing. It is gated on
//! CPUID feature bits, not `cpu::msrs_safe()`: MTRR (CPUID.01H:EDX[12]) and
//! PAT (CPUID.01H:EDX[16]) are ARCHITECTURAL — a hypervisor that advertises
//! them must implement the MSRs — unlike the model-specific power MSRs in
//! cpu.rs, which fault under QEMU. So it also runs under `cargo xtask
//! boot-test` (correctness only, Rule A).

use core::arch::x86_64::__cpuid;
use core::fmt::Write as _;

use alloc::format;
use alloc::string::String;

use crate::boot_log;
use crate::cpu;
use crate::mtrr_decode::{self, RangeVerdict, VarMtrr};

const IA32_MTRRCAP: u32 = 0xFE;
const IA32_MTRR_PHYSBASE0: u32 = 0x200; // pairs: PHYSBASEn = 0x200+2n, PHYSMASKn = 0x201+2n
const IA32_PAT: u32 = 0x277;
const IA32_MTRR_DEF_TYPE: u32 = 0x2FF;

/// The eleven fixed-range MTRRs covering the first 1 MiB (SDM Vol 3A §12.11.2.2).
const MTRR_FIX: [(u32, &str); 11] = [
    (0x250, "64K@00000"),
    (0x258, "16K@80000"),
    (0x259, "16K@A0000"),
    (0x268, "4K@C0000"),
    (0x269, "4K@C8000"),
    (0x26A, "4K@D0000"),
    (0x26B, "4K@D8000"),
    (0x26C, "4K@E0000"),
    (0x26D, "4K@E8000"),
    (0x26E, "4K@F0000"),
    (0x26F, "4K@F8000"),
];

// `__cpuid` is a safe intrinsic on this toolchain — no `unsafe` needed here.

/// CPUID.01H:EDX[12] — MTRRs implemented (gates 0xFE, 0x200.., 0x2FF).
fn has_mtrr() -> bool {
    __cpuid(1).edx & (1 << 12) != 0
}

/// CPUID.01H:EDX[16] — PAT implemented (gates IA32_PAT, 0x277).
fn has_pat() -> bool {
    __cpuid(1).edx & (1 << 16) != 0
}

/// Physical address width from CPUID.80000008H:EAX[7:0]; 36 is the
/// architectural default when the leaf is absent (pre-2003 parts).
fn phys_bits() -> u8 {
    if __cpuid(0x8000_0000).eax >= 0x8000_0008 {
        (__cpuid(0x8000_0008).eax & 0xFF) as u8
    } else {
        36
    }
}

/// PAT entry field decode (SDM Vol 3A, Table 12-10). 7 = UC- (minus): UC that
/// an MTRR of type WC may override, unlike plain UC.
fn pat_entry_name(e: u8) -> &'static str {
    match e & 7 {
        0 => "UC",
        1 => "WC",
        4 => "WT",
        5 => "WP",
        6 => "WB",
        7 => "UC-",
        _ => "??",
    }
}

fn verdict_str(v: RangeVerdict) -> String {
    match v {
        RangeVerdict::Uniform(t) => format!("{} (uniform)", mtrr_decode::type_name(t)),
        RangeVerdict::Mixed { types_seen } => {
            let mut s = String::from("MIXED[");
            let mut sep = "";
            for t in 0..8u8 {
                if types_seen & (1 << t) != 0 {
                    let _ = write!(s, "{}{}", sep, mtrr_decode::type_name(t));
                    sep = ",";
                }
            }
            s.push(']');
            s
        }
        RangeVerdict::Undecodable => {
            String::from("UNDECODABLE (non-contiguous mask or undefined overlap)")
        }
    }
}

/// Dump MTRR/PAT state and, for each `(label, start, len)` physical range,
/// the overlapping UEFI memory-map descriptors plus the effective-type
/// verdict. Called once, right after STAGE 7; safe on every path (bare metal,
/// QEMU, hypervisor) because every MSR read is CPUID-gated.
pub fn log_h3_probe(
    root: &mut uefi::proto::media::file::Directory,
    ranges: &[(String, usize, usize)],
) {
    let pb = phys_bits();
    boot_log(
        root,
        &format!(
            "H3 probe: mtrr={} pat={} phys_bits={} hypervisor={}",
            has_mtrr() as u8,
            has_pat() as u8,
            pb,
            cpu::is_hypervisor() as u8
        ),
    );

    // ---- MTRR state ---------------------------------------------------------
    let mut vars = [VarMtrr { base: 0, mask: 0 }; mtrr_decode::MAX_VARS];
    let mut nvar = 0usize;
    let mut def_type = 0u64;
    let mut mtrr_state_read = false;

    if has_mtrr() {
        // SAFETY: CPUID.01H:EDX[12] is set, so IA32_MTRRCAP (0xFE) and
        // IA32_MTRR_DEF_TYPE (0x2FF) exist; rdmsr of an implemented MSR in
        // ring 0 does not fault. Reads only — nothing is written.
        let cap = unsafe { cpu::rdmsr(IA32_MTRRCAP) };
        // SAFETY: as above.
        def_type = unsafe { cpu::rdmsr(IA32_MTRR_DEF_TYPE) };
        mtrr_state_read = true;

        let vcnt = (cap & 0xFF) as usize;
        boot_log(
            root,
            &format!(
                "H3 MTRRCAP=0x{:X} (vcnt={} fix={} wc={} smrr={})",
                cap,
                vcnt,
                (cap >> 8) & 1,
                (cap >> 10) & 1,
                (cap >> 11) & 1
            ),
        );
        boot_log(
            root,
            &format!(
                "H3 MTRR_DEF_TYPE=0x{:X} (enable={} fixed_enable={} default={})",
                def_type,
                (def_type >> 11) & 1,
                (def_type >> 10) & 1,
                mtrr_decode::type_name((def_type & 0xFF) as u8)
            ),
        );

        nvar = vcnt.min(mtrr_decode::MAX_VARS);
        if vcnt > nvar {
            boot_log(
                root,
                &format!(
                    "H3 MTRR WARNING: vcnt={} exceeds probe cap {}; later pairs unread",
                    vcnt, nvar
                ),
            );
        }
        for (i, slot) in vars[..nvar].iter_mut().enumerate() {
            // SAFETY: MTRRCAP.VCNT reports `vcnt` variable-range pairs and
            // i < vcnt, so IA32_MTRR_PHYSBASEi/PHYSMASKi exist. Reads only.
            let base = unsafe { cpu::rdmsr(IA32_MTRR_PHYSBASE0 + 2 * i as u32) };
            // SAFETY: as above.
            let mask = unsafe { cpu::rdmsr(IA32_MTRR_PHYSBASE0 + 2 * i as u32 + 1) };
            *slot = VarMtrr { base, mask };
            if mask & mtrr_decode::MTRR_VALID_BIT == 0 {
                boot_log(
                    root,
                    &format!(
                        "H3 MTRR var{}: base=0x{:016X} mask=0x{:016X} (disabled)",
                        i, base, mask
                    ),
                );
                continue;
            }
            let t = mtrr_decode::type_name((base & 0xFF) as u8);
            match mtrr_decode::var_span(slot, pb) {
                Some((lo, size)) => boot_log(
                    root,
                    &format!(
                        "H3 MTRR var{}: base=0x{:016X} mask=0x{:016X} -> {} 0x{:X}+0x{:X}",
                        i, base, mask, t, lo, size
                    ),
                ),
                None => boot_log(
                    root,
                    &format!(
                        "H3 MTRR var{}: base=0x{:016X} mask=0x{:016X} -> {} NON-CONTIGUOUS mask",
                        i, base, mask, t
                    ),
                ),
            }
        }

        if (cap >> 8) & 1 == 1 {
            for chunk in MTRR_FIX.chunks(4) {
                let mut line = String::from("H3 MTRR fixed:");
                for (msr, label) in chunk {
                    // SAFETY: MTRRCAP[8] confirms the fixed-range MTRRs exist.
                    // Reads only.
                    let v = unsafe { cpu::rdmsr(*msr) };
                    let _ = write!(line, " {}=0x{:016X}", label, v);
                }
                boot_log(root, &line);
            }
        }
    } else {
        boot_log(root, "H3 MTRR: CPUID reports no MTRR support");
    }

    // ---- PAT ------------------------------------------------------------------
    if has_pat() {
        // SAFETY: CPUID.01H:EDX[16] is set, so IA32_PAT (0x277) exists. Read only.
        let pat = unsafe { cpu::rdmsr(IA32_PAT) };
        let mut names = String::new();
        for i in 0..8 {
            let _ = write!(
                names,
                "{}{}",
                if i > 0 { " " } else { "" },
                pat_entry_name(((pat >> (8 * i)) & 0xFF) as u8)
            );
        }
        boot_log(root, &format!("H3 PAT=0x{:016X} entries=[{}]", pat, names));
        // Firmware page tables leave the PTE PAT/PCD/PWT bits clear, i.e. PAT
        // entry 0. If entry 0 is not WB, EVERY mapping is degraded regardless
        // of MTRRs — that alone would explain MECH-A1.
        if pat & 0x7 != 6 {
            boot_log(
                root,
                "H3 PAT WARNING: entry0 != WB — firmware page-table default is not write-back",
            );
        }
    } else {
        boot_log(root, "H3 PAT: CPUID reports no PAT support");
    }

    // ---- Per-range verdicts + UEFI memory map ---------------------------------
    // Map att bits are CAPABILITIES ("this region could be mapped WB"), not the
    // current setting; the governing setting is the MTRR verdict above.
    use uefi::mem::memory_map::MemoryMap;
    let mmap = uefi::boot::memory_map(uefi::boot::MemoryType::LOADER_DATA).ok();
    if mmap.is_none() {
        boot_log(root, "H3 map: memory_map() failed; att lines unavailable");
    }

    for (label, start, len) in ranges {
        let s = *start as u64;
        let e = s.saturating_add(*len as u64);
        let verdict = if !mtrr_state_read {
            String::from("n/a (no MTRR state)")
        } else {
            let mut v = verdict_str(mtrr_decode::decode_range_type(
                def_type,
                &vars[..nvar],
                pb,
                s,
                e,
            ));
            if s < 0x10_0000 && (def_type >> 10) & 1 == 1 {
                // Below 1 MiB the fixed-range MTRRs govern; the pure decode
                // covers variable ranges only. Flag rather than mis-state.
                v.push_str(" [below-1MiB portion governed by fixed MTRRs]");
            }
            v
        };
        boot_log(
            root,
            &format!(
                "H3 range {}: 0x{:X}-0x{:X} ({} B) mtrr={}",
                label, s, e, len, verdict
            ),
        );

        if let Some(ref m) = mmap {
            let mut covered = 0u64;
            for d in m.entries() {
                let dlo = d.phys_start;
                let dhi = d
                    .phys_start
                    .saturating_add(d.page_count.saturating_mul(4096));
                if dlo < e && dhi > s {
                    boot_log(
                        root,
                        &format!(
                            "H3 map {}: 0x{:X}-0x{:X} type={:?} att=0x{:X}",
                            label,
                            dlo.max(s),
                            dhi.min(e),
                            d.ty,
                            d.att.bits()
                        ),
                    );
                    covered += dhi.min(e) - dlo.max(s);
                }
            }
            if covered < e - s {
                boot_log(
                    root,
                    &format!(
                        "H3 map {} WARNING: {} of {} bytes in no descriptor",
                        label,
                        (e - s) - covered,
                        e - s
                    ),
                );
            }
        }
    }
    boot_log(
        root,
        "H3 note: att bits are UEFI capability flags, not settings; effective type = MTRR verdict (PAT entry 0 assumed for firmware page tables)",
    );
}
