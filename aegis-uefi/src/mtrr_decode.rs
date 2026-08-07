//! Static decode of the effective MTRR memory type for a physical range.
//!
//! Pure integer logic — no MSR access, no UEFI calls — so the H3 probe's
//! verdicts ("this buffer is WB / this buffer is UC") rest on code that can be
//! unit-tested on the host rather than only exercised on someone's laptop:
//!
//!     rustc --edition 2024 --test aegis-uefi/src/mtrr_decode.rs \
//!         -o /tmp/mtrr_decode_test && /tmp/mtrr_decode_test
//!
//! Semantics per Intel SDM Vol 3A §12.11 "MTRRs":
//!  - MTRRs disabled (IA32_MTRR_DEF_TYPE.E = 0): all memory is UC.
//!  - Where no variable range matches, the default type applies.
//!  - Overlapping variable ranges: UC wins over everything; WT wins over WB;
//!    any other combination of distinct types is undefined behaviour of the
//!    processor and is reported as `Undecodable`, never guessed.
//!  - Fixed-range MTRRs (< 1 MiB) are NOT modelled here; the caller flags
//!    ranges that dip below 1 MiB and prints the fixed MSRs raw.

/// MTRR memory type encodings (SDM Vol 3A, Table 12-8).
pub const MTRR_UC: u8 = 0;
pub const MTRR_WC: u8 = 1;
pub const MTRR_WT: u8 = 4;
pub const MTRR_WP: u8 = 5;
pub const MTRR_WB: u8 = 6;

/// IA32_MTRR_PHYSMASKn bit 11 — this variable range is enabled.
pub const MTRR_VALID_BIT: u64 = 1 << 11;

/// IA32_MTRR_DEF_TYPE bit 11 — MTRRs enabled at all.
pub const DEF_TYPE_ENABLE_BIT: u64 = 1 << 11;

/// Hard cap on variable-range pairs the probe reads (MTRRCAP.VCNT is 8 on
/// QEMU, 10 on Broadwell; 255 is architecturally possible but never seen).
pub const MAX_VARS: usize = 16;

pub fn type_name(t: u8) -> &'static str {
    match t {
        MTRR_UC => "UC",
        MTRR_WC => "WC",
        MTRR_WT => "WT",
        MTRR_WP => "WP",
        MTRR_WB => "WB",
        _ => "??",
    }
}

/// One variable-range MTRR pair, raw as read from IA32_MTRR_PHYSBASEn/PHYSMASKn.
#[derive(Clone, Copy)]
pub struct VarMtrr {
    pub base: u64,
    pub mask: u64,
}

/// Verdict for a physical range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RangeVerdict {
    /// The whole range resolves to one MTRR type.
    Uniform(u8),
    /// Different parts of the range resolve to different types.
    /// Bit `t` of `types_seen` is set when effective type `t` occurs.
    Mixed { types_seen: u8 },
    /// A valid MTRR has a non-contiguous mask, or overlapping ranges combine
    /// types the SDM calls undefined. The raw MSRs must be read by a human.
    Undecodable,
}

/// The physical span a VALID variable MTRR matches: `(start, size)`.
/// `None` when the mask is non-contiguous (matches a scattered set of
/// addresses — legal in silicon, never emitted by sane firmware).
/// The caller must have checked `MTRR_VALID_BIT` already.
pub fn var_span(v: &VarMtrr, phys_bits: u8) -> Option<(u64, u64)> {
    let phys_mask: u64 = if phys_bits >= 64 {
        !0
    } else {
        (1u64 << phys_bits) - 1
    };
    let m = v.mask & phys_mask & !0xFFF;
    if m == 0 {
        // Degenerate mask: every address matches (addr & 0 == base & 0).
        let size = if phys_bits >= 64 {
            u64::MAX
        } else {
            1u64 << phys_bits
        };
        return Some((0, size));
    }
    let k = m.trailing_zeros(); // >= 12 (low 12 bits cleared), < 64 (m != 0)
    if m != phys_mask & (!0u64 << k) {
        return None; // holes in the mask: not a single power-of-two block
    }
    Some((v.base & m, 1u64 << k))
}

/// Effective MTRR memory type over the physical range `[start, end)`, given
/// the raw IA32_MTRR_DEF_TYPE value and the raw variable-range pairs.
pub fn decode_range_type(
    def_type: u64,
    vars: &[VarMtrr],
    phys_bits: u8,
    start: u64,
    end: u64,
) -> RangeVerdict {
    if start >= end {
        return RangeVerdict::Undecodable;
    }
    if def_type & DEF_TYPE_ENABLE_BIT == 0 {
        return RangeVerdict::Uniform(MTRR_UC); // MTRRs disabled: everything UC
    }
    let default_t = (def_type & 0xFF) as u8;

    // Clip every valid variable range against [start, end).
    let mut spans = [(0u64, 0u64, 0u8); MAX_VARS]; // (lo, hi, type)
    let mut nspans = 0usize;
    for v in vars.iter().take(MAX_VARS) {
        if v.mask & MTRR_VALID_BIT == 0 {
            continue;
        }
        let (lo, size) = match var_span(v, phys_bits) {
            Some(s) => s,
            None => return RangeVerdict::Undecodable,
        };
        let hi = lo.saturating_add(size);
        let (clo, chi) = (lo.max(start), hi.min(end));
        if clo < chi {
            spans[nspans] = (clo, chi, (v.base & 0xFF) as u8);
            nspans += 1;
        }
    }
    if nspans == 0 {
        return RangeVerdict::Uniform(default_t);
    }

    // Boundary points partition [start, end) into subintervals inside which
    // MTRR coverage cannot change; a point test then decides each interval.
    let mut pts = [0u64; 2 * MAX_VARS + 2];
    let mut np = 0usize;
    pts[np] = start;
    np += 1;
    pts[np] = end;
    np += 1;
    for &(lo, hi, _) in &spans[..nspans] {
        pts[np] = lo;
        np += 1;
        pts[np] = hi;
        np += 1;
    }
    for i in 1..np {
        // insertion sort — np <= 34, no allocator needed
        let mut j = i;
        while j > 0 && pts[j - 1] > pts[j] {
            pts.swap(j - 1, j);
            j -= 1;
        }
    }

    let mut first: Option<u8> = None;
    let mut seen: u8 = 0;
    for w in 0..np - 1 {
        let (p, q) = (pts[w], pts[w + 1]);
        if p >= q {
            continue; // duplicate boundary
        }
        // Combine the types of every span covering this subinterval.
        let mut covered = false;
        let mut has_uc = false;
        let mut has_wt = false;
        let mut other: Option<u8> = None;
        let mut conflict = false;
        for &(lo, hi, t) in &spans[..nspans] {
            if p >= lo && p < hi {
                covered = true;
                match t {
                    MTRR_UC => has_uc = true,
                    MTRR_WT => has_wt = true,
                    t2 => match other {
                        None => other = Some(t2),
                        Some(o) if o == t2 => {}
                        Some(_) => conflict = true,
                    },
                }
            }
        }
        let eff = if !covered {
            default_t
        } else if has_uc {
            MTRR_UC // UC beats everything
        } else if conflict {
            return RangeVerdict::Undecodable; // SDM: undefined combination
        } else if has_wt {
            match other {
                None | Some(MTRR_WB) => MTRR_WT, // WT alone, or WT+WB -> WT
                Some(_) => return RangeVerdict::Undecodable, // WT+WC etc: undefined
            }
        } else {
            other.unwrap_or(default_t)
        };
        seen |= 1u8 << (eff & 7);
        if first.is_none() {
            first = Some(eff);
        }
    }
    match first {
        None => RangeVerdict::Undecodable,
        Some(f) if seen == 1u8 << (f & 7) => RangeVerdict::Uniform(f),
        Some(_) => RangeVerdict::Mixed { types_seen: seen },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PB: u8 = 39; // Broadwell-U physical address width
    const DEF_UC_ON: u64 = DEF_TYPE_ENABLE_BIT; // E=1, default UC (typical firmware)
    const DEF_WB_ON: u64 = DEF_TYPE_ENABLE_BIT | MTRR_WB as u64;

    fn pmask() -> u64 {
        (1u64 << PB) - 1
    }

    /// Canonical var MTRR: power-of-two `size` covering `[base, base+size)`.
    fn var(base: u64, size: u64, t: u8) -> VarMtrr {
        assert!(size.is_power_of_two() && base % size == 0);
        VarMtrr {
            base: base | t as u64,
            mask: (pmask() & !(size - 1)) | MTRR_VALID_BIT,
        }
    }

    #[test]
    fn disabled_mtrrs_mean_uc_everywhere() {
        let vars = [var(0, 0x8000_0000, MTRR_WB)];
        assert_eq!(
            decode_range_type(MTRR_WB as u64, &vars, PB, 0x7000_0000, 0x7100_0000),
            RangeVerdict::Uniform(MTRR_UC)
        );
    }

    #[test]
    fn no_matching_var_gives_default() {
        let vars = [var(0, 0x10_0000, MTRR_WB)];
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0x4000_0000, 0x4001_0000),
            RangeVerdict::Uniform(MTRR_UC)
        );
        assert_eq!(
            decode_range_type(DEF_WB_ON, &[], PB, 0x4000_0000, 0x4001_0000),
            RangeVerdict::Uniform(MTRR_WB)
        );
    }

    #[test]
    fn typical_firmware_layout_dram_is_wb() {
        // Default UC; two WB vars covering 0..2G and 4G..8G (hole for MMIO).
        let vars = [
            var(0, 0x8000_0000, MTRR_WB),
            var(0x1_0000_0000, 0x1_0000_0000, MTRR_WB),
        ];
        // A model buffer well inside DRAM: uniform WB.
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0x7000_0000, 0x702A_B000),
            RangeVerdict::Uniform(MTRR_WB)
        );
        // A buffer inside the MMIO hole: uniform UC (the H3 smoking gun).
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0xC000_0000, 0xC010_0000),
            RangeVerdict::Uniform(MTRR_UC)
        );
    }

    #[test]
    fn straddling_a_boundary_is_mixed() {
        let vars = [var(0, 0x8000_0000, MTRR_WB)];
        let v = decode_range_type(DEF_UC_ON, &vars, PB, 0x7FF0_0000, 0x8010_0000);
        assert_eq!(
            v,
            RangeVerdict::Mixed {
                types_seen: (1 << MTRR_UC) | (1 << MTRR_WB)
            }
        );
    }

    #[test]
    fn uc_var_wins_over_wb_var() {
        let vars = [
            var(0, 0x8000_0000, MTRR_WB),
            var(0x4000_0000, 0x1000_0000, MTRR_UC),
        ];
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0x4800_0000, 0x4900_0000),
            RangeVerdict::Uniform(MTRR_UC)
        );
    }

    #[test]
    fn wt_over_wb_gives_wt() {
        let vars = [
            var(0, 0x8000_0000, MTRR_WB),
            var(0x4000_0000, 0x1000_0000, MTRR_WT),
        ];
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0x4800_0000, 0x4900_0000),
            RangeVerdict::Uniform(MTRR_WT)
        );
    }

    #[test]
    fn undefined_overlap_is_undecodable() {
        // WC over WB is undefined by the SDM — must refuse, not guess.
        let vars = [
            var(0, 0x8000_0000, MTRR_WB),
            var(0x4000_0000, 0x1000_0000, MTRR_WC),
        ];
        assert_eq!(
            decode_range_type(DEF_UC_ON, &vars, PB, 0x4800_0000, 0x4900_0000),
            RangeVerdict::Undecodable
        );
    }

    #[test]
    fn non_contiguous_mask_is_undecodable() {
        // Mask with a real hole ABOVE the trailing-zero run: bit 12 set,
        // bit 13 clear, bits [PB-1:14] set. (Clearing bit 12 alone would just
        // be a legal 8 KiB range mask.)
        let bad = VarMtrr {
            base: MTRR_WB as u64,
            mask: ((pmask() & !0xFFF) & !(1 << 13)) | MTRR_VALID_BIT,
        };
        assert_eq!(
            decode_range_type(DEF_UC_ON, &[bad], PB, 0x1000, 0x2000),
            RangeVerdict::Undecodable
        );
    }

    #[test]
    fn zero_mask_matches_everything() {
        let all = VarMtrr {
            base: MTRR_WT as u64,
            mask: MTRR_VALID_BIT,
        };
        assert_eq!(
            decode_range_type(DEF_WB_ON, &[all], PB, 0x4000_0000, 0x4001_0000),
            RangeVerdict::Uniform(MTRR_WT)
        );
        assert_eq!(var_span(&all, PB), Some((0, 1u64 << PB)));
    }

    #[test]
    fn invalid_vars_are_ignored() {
        let mut v = var(0, 0x8000_0000, MTRR_UC);
        v.mask &= !MTRR_VALID_BIT;
        assert_eq!(
            decode_range_type(DEF_WB_ON, &[v], PB, 0x1000_0000, 0x1100_0000),
            RangeVerdict::Uniform(MTRR_WB)
        );
    }

    #[test]
    fn var_span_decodes_range() {
        let v = var(0x1_0000_0000, 0x4000_0000, MTRR_WB);
        assert_eq!(var_span(&v, PB), Some((0x1_0000_0000, 0x4000_0000)));
    }

    #[test]
    fn empty_range_is_undecodable() {
        assert_eq!(
            decode_range_type(DEF_WB_ON, &[], PB, 0x1000, 0x1000),
            RangeVerdict::Undecodable
        );
    }
}
