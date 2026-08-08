//! msrtool — minimal MSR read/clear-bit tool for the minimal-Linux arm.
//!
//! The Dell i5-5200U's platform asserts bi-directional PROCHOT
//! (MSR_POWER_CTL 0x1FC bit 0), clamping the core to ratio 5 (~500 MHz).
//! The unikernel clears it on the boot path (STAGE 7, commit f337137). The
//! Linux arm of the paired OS-cost measurement must do the same, or the
//! comparison measures the platform quirk, not the OS. This tool mirrors
//! STAGE 7's read → clear → read-back-verify sequence through /dev/cpu/N/msr
//! (requires the `msr` kernel module).
//!
//! Usage:
//!   msrtool read <msr-hex>                # print per-cpu values
//!   msrtool clearbit <msr-hex> <bit>      # clear bit on every cpu, verify
//!
//! Exit: 0 on success (read-back confirms), 1 on any failure.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::process::exit;

fn cpu_list() -> Vec<u32> {
    let mut cpus: Vec<u32> = fs::read_dir("/dev/cpu")
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse().ok()))
                .collect()
        })
        .unwrap_or_default();
    cpus.sort_unstable();
    cpus
}

fn rdmsr(cpu: u32, msr: u64) -> Result<u64, String> {
    let path = format!("/dev/cpu/{cpu}/msr");
    let mut f = OpenOptions::new()
        .read(true)
        .open(&path)
        .map_err(|e| format!("{path}: {e}"))?;
    f.seek(SeekFrom::Start(msr))
        .map_err(|e| format!("{path} seek: {e}"))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)
        .map_err(|e| format!("{path} read: {e}"))?;
    Ok(u64::from_le_bytes(buf))
}

fn wrmsr(cpu: u32, msr: u64, val: u64) -> Result<(), String> {
    let path = format!("/dev/cpu/{cpu}/msr");
    let mut f = OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|e| format!("{path}: {e}"))?;
    f.seek(SeekFrom::Start(msr))
        .map_err(|e| format!("{path} seek: {e}"))?;
    f.write_all(&val.to_le_bytes())
        .map_err(|e| format!("{path} write: {e}"))?;
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let usage = "usage: msrtool read <msr-hex> | msrtool clearbit <msr-hex> <bit>";
    let cmd = a.get(1).map(String::as_str).unwrap_or("");
    let msr = a
        .get(2)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
    let (Some(msr), true) = (msr, cmd == "read" || cmd == "clearbit") else {
        eprintln!("{usage}");
        exit(1);
    };

    let cpus = cpu_list();
    if cpus.is_empty() {
        eprintln!("msrtool: no /dev/cpu/*/msr (is the msr module loaded?)");
        exit(1);
    }

    match cmd {
        "read" => {
            for c in &cpus {
                match rdmsr(*c, msr) {
                    Ok(v) => println!("cpu{c} msr {msr:#x} = {v:#018x}"),
                    Err(e) => {
                        eprintln!("msrtool: {e}");
                        exit(1);
                    }
                }
            }
        }
        "clearbit" => {
            let Some(bit) = a
                .get(3)
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|b| *b < 64)
            else {
                eprintln!("{usage}");
                exit(1);
            };
            let mask = !(1u64 << bit);
            let mut ok = true;
            for c in &cpus {
                let before = match rdmsr(*c, msr) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("msrtool: {e}");
                        exit(1);
                    }
                };
                if let Err(e) = wrmsr(*c, msr, before & mask) {
                    eprintln!("msrtool: {e}");
                    exit(1);
                }
                let after = rdmsr(*c, msr).unwrap_or(u64::MAX);
                let cleared = after & (1u64 << bit) == 0;
                ok &= cleared;
                println!(
                    "cpu{c} msr {msr:#x} bit {bit}: {before:#018x} -> {after:#018x} cleared={cleared}"
                );
            }
            if !ok {
                eprintln!("msrtool: read-back did NOT confirm clear");
                exit(1);
            }
        }
        _ => unreachable!(),
    }
}
