//! xtask — dev automation for A.L.I.C.E. / Aegis.
//!
//! `cargo xtask boot-test` builds the UEFI unikernel, stages an ESP, and boots it
//! under OVMF in QEMU, mapping the `isa-debug-exit` success code to process exit 0.
//!
//! IMPORTANT (CLAUDE.md Rule A): this harness runs under `accel=tcg`, i.e. pure
//! emulation. It is a CORRECTNESS gate only. No timing, throughput, or cycle
//! figure produced under it may be recorded or quoted — TCG models neither the
//! cache hierarchy nor DVFS. Performance numbers come from physical hardware.
//!
//! No external crates: argument parsing is hand-rolled so this works offline.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// QEMU exit status that means the unikernel signalled success via
/// `isa-debug-exit`. The device returns `(value << 1) | 1`, so the engine writing
/// 16 to port 0xf4 surfaces here as 33.
const QEMU_SUCCESS_STATUS: i32 = 33;

const OVMF_CODE: &str = "/usr/share/OVMF/OVMF_CODE_4M.fd";
const OVMF_VARS: &str = "/usr/share/OVMF/OVMF_VARS_4M.fd";

/// Spec target. Override with AEGIS_UEFI_TARGET when building the custom
/// hard-float target (`x86_64-uefi-hardfloat.json`), which emits real SIMD where
/// the stock UEFI target is soft-float.
const DEFAULT_TARGET: &str = "x86_64-unknown-uefi";

const UEFI_CRATE_MANIFEST: &str = "aegis-uefi/Cargo.toml";

/// Model artifacts staged next to BOOTX64.EFI so the unikernel can load them.
/// This is the M7 set whose md5s are pinned in
/// docs/hardware_logs/oscost_PREREGISTRATION_2026-07-30.md. Override the
/// directory with AEGIS_BOOT_ASSETS.
const DEFAULT_ASSET_DIR: &str = "model-lab/tinybit/m7_final_gate_work/artifacts";
const BOOT_ASSETS: [&str; 3] = ["MODEL.SAF", "EMBED.BIN", "VOCAB.BIN"];

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let flags: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();

    match cmd {
        "boot-test" => match boot_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: boot-test failed: {e}");
                ExitCode::from(1)
            }
        },
        "" | "-h" | "--help" | "help" => {
            usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("xtask: unknown subcommand `{other}`\n");
            usage();
            ExitCode::from(2)
        }
    }
}

fn usage() {
    eprintln!(
        "xtask — A.L.I.C.E. / Aegis dev automation

USAGE:
    cargo xtask boot-test [--ci] [--debug]

SUBCOMMANDS:
    boot-test    Build the UEFI unikernel, stage an ESP, boot under OVMF in QEMU.
                 QEMU exit {QEMU_SUCCESS_STATUS} maps to success (0); anything else to failure (1).

FLAGS:
    --ci         Non-interactive: fail fast with diagnostics instead of prompting.
    --debug      Build the debug profile instead of release.

ENVIRONMENT:
    AEGIS_UEFI_TARGET   Rust target triple (default: {DEFAULT_TARGET}).
    QEMU                qemu binary (default: qemu-system-x86_64).

NOTE: boot-test runs under accel=tcg. It is a CORRECTNESS gate only. Per
CLAUDE.md Rule A, no performance figure may be taken from it."
    );
}

fn repo_root() -> PathBuf {
    // xtask/Cargo.toml lives one level below the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn boot_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");
    let root = repo_root();
    let target = env::var("AEGIS_UEFI_TARGET").unwrap_or_else(|_| DEFAULT_TARGET.to_string());
    let profile = if debug { "debug" } else { "release" };

    println!("== xtask boot-test ==");
    println!("   repo    : {}", root.display());
    println!("   target  : {target}");
    println!("   profile : {profile}");
    println!("   mode    : {}", if ci { "CI" } else { "local" });
    println!("   NOTE    : accel=tcg — correctness only, never a perf measurement.");
    println!();

    // ---- 1. Build the unikernel -------------------------------------------
    let manifest = root.join(UEFI_CRATE_MANIFEST);
    if !manifest.exists() {
        return Err(format!(
            "cannot find {}. Expected the UEFI crate at {UEFI_CRATE_MANIFEST}.",
            manifest.display()
        ));
    }

    println!("[1/4] building unikernel");
    let mut build = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    build
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target")
        .arg(&target)
        // Without qemu-test the binary is the interactive console: it waits on
        // the keyboard forever and never writes isa-debug-exit, so the harness
        // can only ever time out.
        .args(["--features", "qemu-test"]);
    if !debug {
        build.arg("--release");
    }
    let status = build
        .status()
        .map_err(|e| format!("could not launch cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }

    // ---- 2. Locate the .efi ------------------------------------------------
    let target_dir = root.join("aegis-uefi/target").join(&target).join(profile);
    let efi = find_efi(&target_dir)
        .ok_or_else(|| format!("no .efi produced under {}", target_dir.display()))?;
    println!("      -> {}", efi.display());

    // ---- 3. Stage the ESP --------------------------------------------------
    println!("[2/4] staging ESP");
    let out = root.join("target");
    let esp_boot = out.join("esp/EFI/BOOT");
    fs::create_dir_all(&esp_boot)
        .map_err(|e| format!("cannot create {}: {e}", esp_boot.display()))?;
    let bootx64 = esp_boot.join("BOOTX64.EFI");
    fs::copy(&efi, &bootx64).map_err(|e| format!("cannot stage {}: {e}", bootx64.display()))?;
    println!("      -> {}", bootx64.display());

    // Model artifacts go in the ESP root, where the unikernel's FAT32 loader
    // looks for them (aegis-uefi/src/main.rs load path).
    let asset_dir = env::var("AEGIS_BOOT_ASSETS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root.join(DEFAULT_ASSET_DIR));
    for name in BOOT_ASSETS {
        let src = asset_dir.join(name);
        if !src.exists() {
            return Err(format!(
                "missing model artifact {}. Set AEGIS_BOOT_ASSETS to a directory \
                 holding {BOOT_ASSETS:?}.",
                src.display()
            ));
        }
        let dst = out.join("esp").join(name);
        fs::copy(&src, &dst).map_err(|e| format!("cannot stage {}: {e}", dst.display()))?;
        println!("      -> {}", dst.display());
    }

    // ---- 4. Writable OVMF vars --------------------------------------------
    // The vars pflash is written by the firmware, so it must be a private copy;
    // pointing QEMU at the system file would fail read-only or mutate the host.
    println!("[3/4] copying OVMF vars");
    for p in [OVMF_CODE, OVMF_VARS] {
        if !Path::new(p).exists() {
            return Err(format!(
                "missing {p}. Install OVMF (apt-get install -y ovmf)."
            ));
        }
    }
    let vars = out.join("OVMF_VARS_4M.fd");
    fs::copy(OVMF_VARS, &vars).map_err(|e| format!("cannot copy OVMF vars: {e}"))?;
    println!("      -> {}", vars.display());

    // ---- 5. Boot -----------------------------------------------------------
    println!("[4/4] booting under OVMF");
    let qemu = env::var("QEMU").unwrap_or_else(|_| "qemu-system-x86_64".into());
    let esp_arg = format!("format=raw,file=fat:rw:{}", out.join("esp").display());
    let code_arg = format!("if=pflash,format=raw,readonly=on,file={OVMF_CODE}");
    let vars_arg = format!("if=pflash,format=raw,file={}", vars.display());

    let mut qemu_cmd = Command::new(&qemu);
    qemu_cmd
        .current_dir(&root)
        .args(["-machine", "q35,accel=tcg"])
        .args(["-cpu", "max"])
        // 4 vCPUs so OVMF publishes MP services and the MECH AP-park path is
        // exercised. Correctness only (Rule A) — TCG timing means nothing.
        .args(["-smp", "4"])
        .args(["-m", "2048"])
        .args(["-drive", &code_arg])
        .args(["-drive", &vars_arg])
        .args(["-drive", &esp_arg])
        .args(["-device", "isa-debug-exit,iobase=0xf4,iosize=0x04"])
        .args(["-serial", "stdio"])
        .args(["-display", "none"])
        .arg("-no-reboot");

    println!("      $ {qemu} -machine q35,accel=tcg -cpu max -m 2048 ...");
    println!();

    let status = qemu_cmd
        .status()
        .map_err(|e| format!("could not launch {qemu}: {e}. Install qemu-system-x86."))?;

    println!();
    match status.code() {
        Some(QEMU_SUCCESS_STATUS) => {
            println!(
                "== PASS == QEMU exited {QEMU_SUCCESS_STATUS} (isa-debug-exit success signal)"
            );
            Ok(ExitCode::SUCCESS)
        }
        Some(c) => {
            eprintln!(
                "== FAIL == QEMU exited {c}, expected {QEMU_SUCCESS_STATUS}.\n\
                 The unikernel did not reach its success signal. BOOTLOG.TXT on the\n\
                 staged ESP holds the stage checkpoints reached before it stopped."
            );
            Ok(ExitCode::from(1))
        }
        None => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            Ok(ExitCode::from(1))
        }
    }
}

/// Pick the freshest `.efi` in `dir`, ignoring the `deps/` copies cargo leaves.
fn find_efi(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(dir).ok()? {
        let path = match entry {
            Ok(e) => e.path(),
            Err(_) => continue,
        };
        if path.extension().and_then(|e| e.to_str()) != Some("efi") {
            continue;
        }
        let mtime = match path.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        // Explicit match rather than map_or/is_none_or: neither spelling is
        // lint-clean across the clippy versions this repo builds under.
        let newer = match &best {
            Some((t, _)) => mtime > *t,
            None => true,
        };
        if newer {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p)
}
