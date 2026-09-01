//! xtask — dev automation for A.L.I.C.E. / Aegis.
//!
//! `cargo xtask boot-test` builds the UEFI unikernel, stages an ESP, and boots it
//! under OVMF in QEMU, mapping the `isa-debug-exit` success code to process exit 0.
//!
//! `cargo xtask job-test` stages the same ESP plus a `JOB.TXT` and asserts the
//! AEFINITY OS phase-0 contract (program/AEFINITY_OS.md §6): the guest runs the
//! job, writes `RESULT.TXT` into the `fat:rw:` directory, and resets — which
//! under `-no-reboot` exits QEMU 0.
//!
//! IMPORTANT (CLAUDE.md Rule A): this harness runs under `accel=tcg`, i.e. pure
//! emulation. It is a CORRECTNESS gate only. No timing, throughput, or cycle
//! figure produced under it may be recorded or quoted — TCG models neither the
//! cache hierarchy nor DVFS. Performance numbers come from physical hardware.
//!
//! `cargo xtask job-budget-test` is the regression gate for budget enforcement:
//! it stages a job whose `PROMPT` cannot be prefilled inside its `BUDGET` and
//! asserts the guest still writes a `RESULT.TXT` saying `verdict=FAIL budget`,
//! rather than being killed by the firmware watchdog with nothing on the volume.
//!
//! No external crates: argument parsing is hand-rolled so this works offline.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::time::{Duration, Instant};

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
        "job-test" => match job_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: job-test failed: {e}");
                ExitCode::from(1)
            }
        },
        "job-budget-test" => match job_budget_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: job-budget-test failed: {e}");
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
    cargo xtask <boot-test|job-test|job-budget-test> [--ci] [--debug]

SUBCOMMANDS:
    boot-test    Build the UEFI unikernel, stage an ESP, boot under OVMF in QEMU.
                 QEMU exit {QEMU_SUCCESS_STATUS} maps to success (0); anything else to failure (1).
                 No JOB.TXT is staged, so this is the unchanged boot path.

    job-test     AEFINITY OS phase 0 (program/AEFINITY_OS.md §6). Stages the same
                 ESP plus a JOB.TXT (BUDGET {JOB_BUDGET_S} / PROMPT / BENCH {JOB_BENCH_TOKENS} /
                 TOKENS {JOB_TOKENS} / AFTER reset) and boots it. PASS = QEMU exits within
                 {JOB_TIMEOUT_S}s because the guest reset, AND target/esp/RESULT.TXT parses
                 with verdict=OK, jobs=2, env=vm, aefinity_os=0.1. The record is
                 printed. Structure and exit codes only — never a timing figure.

    job-budget-test
                 Budget-enforcement regression gate. Stages a JOB.TXT with a
                 {BUDGET_PROMPT_BYTES}-byte PROMPT, TOKENS {BUDGET_TOKENS} and BUDGET {BUDGET_FAIL_S} — a job whose
                 prompt cannot be prefilled inside its budget. PASS = QEMU exits
                 within {BUDGET_TIMEOUT_S}s because the guest reset, AND RESULT.TXT exists with
                 a verdict of `FAIL budget`. Before the prefill deadline check
                 this case produced no record at all: the firmware watchdog reset
                 the box mid-prefill and the volume stayed silent.

FLAGS:
    --ci         Non-interactive: fail fast with diagnostics instead of prompting.
    --debug      Build the debug profile instead of release.

ENVIRONMENT:
    AEGIS_UEFI_TARGET   Rust target triple (default: {DEFAULT_TARGET}).
    QEMU                qemu binary (default: qemu-system-x86_64).

NOTE: every gate here runs under accel=tcg. They are CORRECTNESS gates only.
Per CLAUDE.md Rule A, no performance figure may be taken from any of them —
which is also why a job-test RESULT.TXT says env=vm."
    );
}

fn repo_root() -> PathBuf {
    // xtask/Cargo.toml lives one level below the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

// ---------------------------------------------------------------------------
// Shared harness: stage an ESP, run QEMU against it.
// ---------------------------------------------------------------------------

/// A staged ESP, ready to boot: the repo root, the `fat:rw:` directory QEMU
/// exposes to the guest (guest writes land here on the host), and the private
/// OVMF vars copy.
struct Esp {
    root: PathBuf,
    dir: PathBuf,
    vars: PathBuf,
}

/// What QEMU did.
enum Outcome {
    Exited(i32),
    /// Terminated by a signal, with no exit code.
    Signalled,
    /// Still running when the harness deadline passed; killed.
    TimedOut,
}

/// Build the unikernel (qemu-test feature), stage BOOTX64.EFI plus the model
/// artifacts into `target/esp`, and copy the OVMF vars.
///
/// `job_txt` writes `JOB.TXT` into the ESP root. `None` removes any JOB.TXT a
/// previous run left there, so `boot-test` always exercises the no-job boot
/// path regardless of what ran before it. A stale `RESULT.TXT` is removed
/// either way — a gate that can pass on the previous run's output is not a
/// gate.
fn stage(label: &str, ci: bool, debug: bool, job_txt: Option<&str>) -> Result<Esp, String> {
    let root = repo_root();
    let target = env::var("AEGIS_UEFI_TARGET").unwrap_or_else(|_| DEFAULT_TARGET.to_string());
    let profile = if debug { "debug" } else { "release" };

    println!("== xtask {label} ==");
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
        // can only ever time out. job-test reuses the same binary — its JOB.TXT
        // hook runs before the qemu-test block, so the job's ResetSystem wins.
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

    // Guest-written artifacts from any previous run, cleared case-insensitively
    // (see `find_ci`) so a stale record cannot pass a later gate.
    let esp_dir = out.join("esp");
    if let Some(stale) = find_ci(&esp_dir, "RESULT.TXT") {
        let _ = fs::remove_file(stale);
    }
    // The guest's in-progress marker (aegis-uefi job::WIP_NAME). Left behind,
    // it would make a later run look like it died mid-step.
    if let Some(stale) = find_ci(&esp_dir, "RESULT.WIP") {
        let _ = fs::remove_file(stale);
    }
    if let Some(stale) = find_ci(&esp_dir, "JOB.TXT") {
        let _ = fs::remove_file(stale);
    }
    if let Some(body) = job_txt {
        let job_path = esp_dir.join("JOB.TXT");
        fs::write(&job_path, body)
            .map_err(|e| format!("cannot stage {}: {e}", job_path.display()))?;
        println!("      -> {}", job_path.display());
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

    Ok(Esp {
        root,
        dir: out.join("esp"),
        vars,
    })
}

/// Boot a staged ESP under OVMF. `extra` is appended verbatim (net devices,
/// hostfwd, …). `timeout` of `None` waits forever, which is what `boot-test`
/// has always done — the unikernel signals through isa-debug-exit.
fn qemu(esp: &Esp, extra: &[String], timeout: Option<Duration>) -> Result<Outcome, String> {
    let qemu_bin = env::var("QEMU").unwrap_or_else(|_| "qemu-system-x86_64".into());
    let esp_arg = format!("format=raw,file=fat:rw:{}", esp.dir.display());
    let code_arg = format!("if=pflash,format=raw,readonly=on,file={OVMF_CODE}");
    let vars_arg = format!("if=pflash,format=raw,file={}", esp.vars.display());

    let mut cmd = Command::new(&qemu_bin);
    cmd.current_dir(&esp.root)
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
        // A guest ResetSystem terminates QEMU (exit 0) instead of rebooting
        // into the same job forever.
        .arg("-no-reboot");
    cmd.args(extra);

    println!("      $ {qemu_bin} -machine q35,accel=tcg -cpu max -m 2048 ...");
    println!();

    match timeout {
        None => {
            let status = cmd.status().map_err(|e| {
                format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86.")
            })?;
            Ok(match status.code() {
                Some(c) => Outcome::Exited(c),
                None => Outcome::Signalled,
            })
        }
        Some(limit) => {
            let mut child: Child = cmd.spawn().map_err(|e| {
                format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86.")
            })?;
            let deadline = Instant::now() + limit;
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return Ok(match status.code() {
                            Some(c) => Outcome::Exited(c),
                            None => Outcome::Signalled,
                        });
                    }
                    Ok(None) => {}
                    Err(e) => return Err(format!("waiting on {qemu_bin} failed: {e}")),
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(Outcome::TimedOut);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// boot-test
// ---------------------------------------------------------------------------

fn boot_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    let esp = stage("boot-test", ci, debug, None)?;

    // ---- 5. Boot -----------------------------------------------------------
    println!("[4/4] booting under OVMF");
    let outcome = qemu(&esp, &[], None)?;

    println!();
    match outcome {
        Outcome::Exited(QEMU_SUCCESS_STATUS) => {
            println!(
                "== PASS == QEMU exited {QEMU_SUCCESS_STATUS} (isa-debug-exit success signal)"
            );
            Ok(ExitCode::SUCCESS)
        }
        Outcome::Exited(c) => {
            eprintln!(
                "== FAIL == QEMU exited {c}, expected {QEMU_SUCCESS_STATUS}.\n\
                 The unikernel did not reach its success signal. BOOTLOG.TXT on the\n\
                 staged ESP holds the stage checkpoints reached before it stopped."
            );
            Ok(ExitCode::from(1))
        }
        Outcome::Signalled => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            Ok(ExitCode::from(1))
        }
        Outcome::TimedOut => {
            eprintln!("== FAIL == QEMU did not exit before the harness deadline.");
            Ok(ExitCode::from(1))
        }
    }
}

// ---------------------------------------------------------------------------
// job-test — AEFINITY OS phase 0 gate (program/AEFINITY_OS.md §6)
// ---------------------------------------------------------------------------

/// Wall budget written into the staged JOB.TXT. The guest arms the firmware
/// watchdog at BUDGET + 60, so this also leaves room under `JOB_TIMEOUT_S`
/// for the watchdog to be the *second* line of defence.
///
/// Spec §6 says 180. That is an iron-calibrated figure, and a wall-clock
/// budget is exactly the kind of number CLAUDE.md Rule A says cannot cross
/// the emulation boundary: measured on the dev box 2026-08-31, this same
/// two-directive job under accel=tcg needed more than 180 s (the record came
/// back `verdict=FAIL budget`, stopped four tokens into BENCH), so at 180 the
/// gate is a race against whatever else the box is doing rather than a check
/// on the unikernel. The budget path itself is not going untested by raising
/// it — that run is what proved the callback stops generation and the record
/// says so. This value keeps the budget a real guard while making the gate
/// deterministic.
const JOB_BUDGET_S: u64 = 900;
/// `TOKENS` for the staged PROMPT.
const JOB_TOKENS: usize = 16;
/// `BENCH n`.
const JOB_BENCH_TOKENS: usize = 8;
/// Harness deadline for the whole boot: firmware, model load, the job, and
/// the reset. Spec §6 says 300; on the dev box the model load alone is ~90 s
/// under accel=tcg before a directive runs, so 300 leaves the gate timing out
/// on a busy box rather than reporting anything about the code. Raised for
/// the same reason as `JOB_BUDGET_S`, and it stays a real deadline: a guest
/// that never resets still fails here.
const JOB_TIMEOUT_S: u64 = 1200;

fn job_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    let job_txt = format!(
        "# staged by `cargo xtask job-test` — AEFINITY OS phase 0 gate\n\
         BUDGET {JOB_BUDGET_S}\n\
         MODE oneshot\n\
         TOKENS {JOB_TOKENS}\n\
         PROMPT The capital of France is\n\
         BENCH {JOB_BENCH_TOKENS}\n\
         AFTER reset\n"
    );

    let esp = stage("job-test", ci, debug, Some(&job_txt))?;

    println!("[4/4] booting under OVMF with JOB.TXT (AFTER reset, -no-reboot)");
    let outcome = qemu(&esp, &[], Some(Duration::from_secs(JOB_TIMEOUT_S)))?;

    println!();
    match outcome {
        // The guest asked the firmware to reset; -no-reboot turns that into a
        // clean QEMU exit. The exact code is not the assertion — RESULT.TXT is.
        Outcome::Exited(c) => println!("   QEMU exited {c} (guest ResetSystem under -no-reboot)"),
        Outcome::Signalled => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            return Ok(ExitCode::from(1));
        }
        Outcome::TimedOut => {
            eprintln!(
                "== FAIL == the guest did not reset within {JOB_TIMEOUT_S}s.\n\
                 BOOTLOG.TXT on the staged ESP holds the stage checkpoints reached."
            );
            return Ok(ExitCode::from(1));
        }
    }

    // `fat:rw:` writes land in the host directory, so the guest's RESULT.TXT is
    // readable here directly — under whatever case vvfat gave it (`find_ci`).
    let Some(result_path) = find_ci(&esp.dir, "RESULT.TXT") else {
        eprintln!(
            "== FAIL == no RESULT.TXT under {}. The guest reset without writing a record;\n\
             BOOTLOG.TXT there records whether it tried.",
            esp.dir.display()
        );
        return Ok(ExitCode::from(1));
    };
    let body = match fs::read_to_string(&result_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("== FAIL == cannot read {} ({e}).", result_path.display());
            return Ok(ExitCode::from(1));
        }
    };

    println!("---- {} ----", result_path.display());
    print!("{body}");
    println!("---- end ----");
    println!();

    // Structure only (Rule A): key presence and exact expected values. No
    // number in this record is read as a measurement, and `env=vm` is the
    // assertion that says so.
    let expect: [(&str, &str); 4] = [
        ("aefinity_os", "0.1"),
        ("verdict", "OK"),
        ("jobs", "2"),
        ("env", "vm"),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (key, want) in expect {
        match record_value(&body, key) {
            Some(got) if got == want => println!("   ok   {key}={got}"),
            Some(got) => failures.push(format!("{key}={got}, expected {want}")),
            None => failures.push(format!("{key} missing")),
        }
    }

    println!();
    if failures.is_empty() {
        println!("== PASS == RESULT.TXT satisfies the phase-0 contract.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

// ---------------------------------------------------------------------------
// job-budget-test — budget enforcement during prefill (regression gate)
// ---------------------------------------------------------------------------

/// `BUDGET` for the failing job. Small enough that a prompt of
/// `BUDGET_PROMPT_BYTES` cannot be prefilled inside it under TCG, large enough
/// that the *pre-step* budget check (which runs before the engine is entered)
/// does not trip first on an idle box — this gate is about the checkpoints
/// inside the engine. Either path still produces the asserted record, so a
/// loaded box degrades the gate's specificity, never its verdict.
const BUDGET_FAIL_S: u64 = 5;
/// `TOKENS` for it: the spec's hard cap, so nothing about this job is short.
const BUDGET_TOKENS: usize = 1024;
/// Length of the staged `PROMPT`, the spec §4 per-line cap. This is the
/// largest prompt a legal `JOB.TXT` may carry, which is the point.
const BUDGET_PROMPT_BYTES: usize = 4096;
/// Harness deadline. Generous: firmware, the model load and the guest's own
/// watchdog (BUDGET + 60) all fit inside it, so a guest that never resets
/// fails here rather than hanging the gate.
const BUDGET_TIMEOUT_S: u64 = 600;

fn job_budget_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    // One line, ASCII, no '#' (which would start a comment) and no newline.
    let mut prompt = String::new();
    while prompt.len() < BUDGET_PROMPT_BYTES {
        prompt.push_str("the quick brown fox jumps over the lazy dog and then keeps going ");
    }
    prompt.truncate(BUDGET_PROMPT_BYTES);

    // ZORP is deliberate: an unknown key must be logged and ignored, not
    // turned into a parse failure that would mask what this gate measures.
    let job_txt = format!(
        "# staged by `cargo xtask job-budget-test` — budget enforcement gate\n\
         BUDGET {BUDGET_FAIL_S}\n\
         MODE oneshot\n\
         TOKENS {BUDGET_TOKENS}\n\
         ZORP this key does not exist\n\
         PROMPT {prompt}\n\
         AFTER reset\n"
    );

    let esp = stage("job-budget-test", ci, debug, Some(&job_txt))?;

    println!("[4/4] booting under OVMF with an unmeetable BUDGET (AFTER reset, -no-reboot)");
    let outcome = qemu(&esp, &[], Some(Duration::from_secs(BUDGET_TIMEOUT_S)))?;

    println!();
    match outcome {
        Outcome::Exited(c) => println!("   QEMU exited {c} (guest ResetSystem under -no-reboot)"),
        Outcome::Signalled => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            return Ok(ExitCode::from(1));
        }
        Outcome::TimedOut => {
            eprintln!(
                "== FAIL == the guest did not reset within {BUDGET_TIMEOUT_S}s.\n\
                 A budget-stopped job must still write its record and reset."
            );
            return Ok(ExitCode::from(1));
        }
    }

    // The BOOTLOG lines are the diagnosis when this gate fails: they say
    // whether the stop happened in prefill or in decode.
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (JOB/RESET lines) ----");
        for line in text.lines() {
            if line.contains("JOB:") || line.contains("RESET:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    let Some(result_path) = find_ci(&esp.dir, "RESULT.TXT") else {
        eprintln!(
            "== FAIL == no RESULT.TXT under {}. A job that blows its budget must still\n\
             leave a record — silence is the failure this gate exists to catch.",
            esp.dir.display()
        );
        return Ok(ExitCode::from(1));
    };
    let body = match fs::read_to_string(&result_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("== FAIL == cannot read {} ({e}).", result_path.display());
            return Ok(ExitCode::from(1));
        }
    };

    println!("---- {} ----", result_path.display());
    print!("{body}");
    println!("---- end ----");
    println!();

    let mut failures: Vec<String> = Vec::new();
    for (key, want) in [("aefinity_os", "0.1"), ("env", "vm")] {
        match record_value(&body, key) {
            Some(got) if got == want => println!("   ok   {key}={got}"),
            Some(got) => failures.push(format!("{key}={got}, expected {want}")),
            None => failures.push(format!("{key} missing")),
        }
    }
    match record_value(&body, "verdict") {
        Some(got) if got.starts_with("FAIL budget") => println!("   ok   verdict={got}"),
        Some(got) => failures.push(format!(
            "verdict={got}, expected a verdict starting `FAIL budget`"
        )),
        None => failures.push("verdict missing".to_string()),
    }

    println!();
    if failures.is_empty() {
        println!("== PASS == the budget stopped the job and the record says so.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// First `key=value` line for `key` in a RESULT.TXT body.
fn record_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    body.lines().find_map(|line| {
        let (k, v) = line.split_once('=')?;
        if k == key {
            Some(v.trim_end())
        } else {
            None
        }
    })
}

/// Find `name` in `dir` ignoring ASCII case.
///
/// FAT is case-insensitive, and QEMU's `fat:rw:` writes a guest-created short
/// name back into the host directory in lower case — the guest's `RESULT.TXT`
/// lands as `result.txt`. A case-sensitive host lookup therefore reports "the
/// guest never wrote a record" for a record that is sitting right there, which
/// is exactly the wrong diagnosis to hand someone debugging a headless box.
fn find_ci(dir: &Path, name: &str) -> Option<PathBuf> {
    let want = name.to_ascii_lowercase();
    for entry in fs::read_dir(dir).ok()? {
        let path = match entry {
            Ok(e) => e.path(),
            Err(_) => continue,
        };
        let matches = path
            .file_name()
            .and_then(|f| f.to_str())
            .is_some_and(|f| f.to_ascii_lowercase() == want);
        if matches {
            return Some(path);
        }
    }
    None
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
