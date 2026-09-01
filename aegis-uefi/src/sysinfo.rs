//! Read-only machine identity for the AEFINITY OS `RESULT.TXT` record.
//!
//! Spec: `program/AEFINITY_OS.md` §3/§5. Three facts, all straight out of
//! CPUID, none of them derived and none of them invented:
//!
//! * [`env`] — CPUID.1:ECX bit 31, the architectural "hypervisor present"
//!   bit. It is what stamps `env=vm` on every record produced under QEMU, so
//!   CLAUDE.md Rule A ("no performance number may come from QEMU/TCG") is
//!   enforced by the artifact rather than by whoever reads it later.
//! * [`cpu_brand`] — CPUID leaves 0x8000_0002..=0x8000_0004, the processor
//!   brand string, so a result names the machine that produced it.
//! * [`cpuid_sig`] — CPUID.1:EAX, the family/model/stepping signature word,
//!   which identifies the silicon even when two boxes share a brand string.
//!
//! Nothing here writes an MSR, allocates, or measures anything.

use core::arch::x86_64::__cpuid;

/// Where this boot is running, as the firmware-visible CPU reports it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Env {
    /// No hypervisor bit: physical silicon.
    Iron,
    /// CPUID.1:ECX[31] set: a hypervisor is present (QEMU/TCG, KVM, crosvm…).
    Vm,
}

impl Env {
    /// The exact token written to `RESULT.TXT` (`env=iron` / `env=vm`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Env::Iron => "iron",
            Env::Vm => "vm",
        }
    }
}

/// CPUID.1:ECX bit 31 — the hypervisor-present bit.
///
/// The bit is architecturally reserved-zero on physical hardware and set by
/// every mainstream hypervisor, so this is a hint, not a proof: a hypervisor
/// that hides itself reads as `Iron`. It is used here only to mark records
/// that must never be quoted as performance, which is the safe direction —
/// the failure mode is a VM claiming iron, and that is why the *number* still
/// has to come from a named physical box (Rule A) rather than from this bit.
pub fn env() -> Env {
    // Leaf 1 is architecturally defined on every x86_64 CPU; `__cpuid` reads
    // registers only, touches no memory, and is safe on this target.
    let ecx = __cpuid(1).ecx;
    if ecx & (1 << 31) != 0 {
        Env::Vm
    } else {
        Env::Iron
    }
}

/// CPUID.1:EAX — the family/model/stepping signature word, verbatim.
pub fn cpuid_sig() -> u32 {
    // See `env`: leaf 1 is always present, and this is a register read only.
    __cpuid(1).eax
}

/// Processor brand string from CPUID leaves 0x8000_0002..=0x8000_0004,
/// trimmed. Returns `"unknown"` when the CPU does not implement the extended
/// leaves. The caller owns the 48-byte buffer so this stays allocation-free.
pub fn cpu_brand(buf: &mut [u8; 48]) -> &str {
    // Leaf 0x8000_0000 reports the highest extended leaf supported; the brand
    // leaves are only read when it says they exist. Every byte written lands
    // inside the caller's 48-byte buffer (3 leaves x 4 registers x 4 = 48).
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
    let end = buf.iter().position(|&b| b == 0).unwrap_or(48);
    core::str::from_utf8(&buf[..end])
        .unwrap_or("unknown")
        .trim()
}
