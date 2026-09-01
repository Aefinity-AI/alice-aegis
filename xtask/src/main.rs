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
//! `cargo xtask net-test` is the AEFINITY OS phase-1a gate (spec §6): it stands
//! up a TCP listener on the host, boots the guest with a virtio NIC and a
//! `JOB.TXT` carrying `NET static` + `NETCHECK`, and asserts that the guest's
//! own TCP/IP stack reached the host and announced its MAC.
//!
//! `cargo xtask os-test` is the AEFINITY OS phase-1b gate (spec §6): it runs a
//! minimal HTTP/1.1 server on the host, boots the guest with a virtio NIC and
//! a `JOB.TXT` carrying `PROMPT`/`BENCH` plus `REPORT <url>` pointing at that
//! server, and asserts that the guest POSTed its `RESULT.TXT` bytes there
//! before resetting.
//!
//! No external crates: argument parsing is hand-rolled so this works offline.

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
        "net-test" => match net_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: net-test failed: {e}");
                ExitCode::from(1)
            }
        },
        "os-test" => match os_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: os-test failed: {e}");
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
    cargo xtask <boot-test|job-test|net-test|os-test|job-budget-test> [--ci] [--debug] [--dhcp] [--pcap]

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

    net-test     AEFINITY OS phase 1a (program/AEFINITY_OS.md §6). Adds
                 `-netdev user,id=n0 -device virtio-net-pci,netdev=n0
                 -device virtio-rng-pci`, opens a host TCP listener on an
                 ephemeral 127.0.0.1 port, and stages a JOB.TXT with
                 `NET static {NET_GUEST_CIDR} {NET_HOST_IP}` and
                 `NETCHECK {NET_HOST_IP}:<port>`. QEMU's user networking maps a
                 guest connection to {NET_HOST_IP} onto the host loopback, so the
                 listener is what the guest reaches. PASS = the harness receives
                 a line beginning `HELLO ` followed by a 17-character MAC, AND
                 RESULT.TXT carries job.1.ok=true with a matching mac=/ip=.

    os-test      AEFINITY OS phase 1b (program/AEFINITY_OS.md §6). Runs a minimal
                 HTTP/1.1 server on an ephemeral 127.0.0.1 port, and stages a
                 JOB.TXT with `NET static {NET_GUEST_CIDR} {NET_HOST_IP}` / `PROMPT` /
                 `BENCH {JOB_BENCH_TOKENS}` / `TOKENS {JOB_TOKENS}` /
                 `REPORT http://{NET_HOST_IP}:<port>/aefinity/result`. PASS = the
                 harness receives a POST whose body parses with aefinity_os=0.1,
                 env=vm, jobs=2, verdict=OK, AND QEMU exits (guest ResetSystem).
                 The on-disk RESULT.TXT is not required to carry `report=` (spec
                 §6: it predates the POST) — only the received body is asserted.
                 The received body is printed either way.

    job-budget-test
                 Budget-enforcement regression gate. Stages a JOB.TXT with a
                 {BUDGET_PROMPT_BYTES}-byte PROMPT, TOKENS {BUDGET_TOKENS} and BUDGET {BUDGET_FAIL_S} — a job whose
                 prompt cannot be prefilled inside its budget. PASS = QEMU exits
                 within {BUDGET_TIMEOUT_S}s because the guest reset, AND RESULT.TXT exists with
                 a verdict of `FAIL budget`. Before the prefill deadline check
                 this case produced no record at all: the firmware watchdog reset
                 the box mid-prefill and the volume stayed silent.

    Both job gates also assert the guest's BOOTLOG line `RESULT.WIP cleared=true`:
    spec §3 makes the marker's presence mean \"this box did not finish\", so the
    guest must have confirmed it off the volume. QEMU's `fat:rw:` does not commit
    the unlink to the host mirror, so target/esp may still show the file; that is
    noted, not failed (see `wip_cleared_check`).

FLAGS:
    --ci         Non-interactive: fail fast with diagnostics instead of prompting.
    --debug      Build the debug profile instead of release.
    --dhcp       net-test only: stage `NET dhcp` instead of `NET static`, so the
                 directive spec §2 makes the default is exercised too. QEMU's
                 slirp answers the DISCOVER; the gate then asserts that a lease
                 was taken on 10.0.2.0/24, not which address slirp chose.

    --pcap       net-test only: also write every frame on the guest's NIC to
                 target/net.pcap (`-object filter-dump`). Read it with
                 `tcpdump -r target/net.pcap`. The first thing to look for when
                 the guest never connects is whether QEMU's slirp answered the
                 guest's ARP for {NET_HOST_IP}.

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
    // BOOTLOG.TXT is appended to, never rewritten, so across runs it becomes a
    // pile of blocks — and `wip_cleared_check` and net-test both *assert* on
    // lines they find in it. Measured on 2026-09-01: a net-test run whose
    // RESULT.TXT committed to the host mirror had its BOOTLOG appends not
    // commit at all, so the harness read four earlier runs and reported a
    // `RESULT.WIP cleared=true` that belonged to none of them. A gate that can
    // pass on a previous run's evidence is not a gate. Starting each run from
    // an empty log makes a missing line show up as a missing line.
    if let Some(stale) = find_ci(&esp_dir, "BOOTLOG.TXT") {
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
    match wip_cleared_check(&esp) {
        Ok(line) => println!("   ok   {line}"),
        Err(why) => failures.push(why),
    }
    wip_host_note(&esp);

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
// net-test — AEFINITY OS phase 1a gate (program/AEFINITY_OS.md §6)
// ---------------------------------------------------------------------------

/// The guest's address on QEMU's user network, in the form the `NET static`
/// directive takes. 10.0.2.15/24 is what slirp hands out and therefore the one
/// static address that works without a DHCP server.
const NET_GUEST_CIDR: &str = "10.0.2.15/24";
/// The gateway on QEMU's user network. slirp answers ARP for it and proxies a
/// TCP connection to it onto the **host loopback** on the same port, which is
/// why the harness listener binds 127.0.0.1 and not 0.0.0.0.
const NET_HOST_IP: &str = "10.0.2.2";
/// `BUDGET` in the staged JOB.TXT. Same value and same reason as
/// [`JOB_BUDGET_S`]: a wall-clock budget calibrated on iron does not survive
/// the crossing into TCG (CLAUDE.md Rule A), and the model load alone eats
/// most of a spec-sized budget before a directive runs.
const NET_BUDGET_S: u64 = 900;
/// Harness deadline for the whole boot — firmware, model load, NIC bring-up,
/// the NETCHECK and the reset. Same value and same reason as
/// [`JOB_TIMEOUT_S`]; a guest that never resets still fails here.
const NET_TIMEOUT_S: u64 = 1200;
/// The connection deadline, applied twice: the harness keeps accepting for
/// this long after QEMU exits, and once a connection arrives this is the read
/// timeout on it.
///
/// Spec §6 states the gate as "receives `HELLO <mac>` within 60 s". That is a
/// figure for a booted box; here the accept runs *concurrently* with a boot
/// that spends minutes in the model load under TCG, so the accept window is
/// [`NET_TIMEOUT_S`] and this is the tail — how long a connection that has
/// already been made gets to finish, and how long a late one gets to arrive.
const NET_ACCEPT_S: u64 = 90;
/// Most a single `HELLO` line can be. The guest sends 24 bytes; anything past
/// this is a peer the gate should not be reading from.
const NET_HELLO_MAX: usize = 4096;

/// What the host listener saw.
#[derive(Default)]
struct Caught {
    /// Bytes read from the first connection, up to [`NET_HELLO_MAX`].
    bytes: Vec<u8>,
    /// Whether a connection was accepted at all.
    connected: bool,
    /// Anything that went wrong on the host side.
    note: Option<String>,
}

fn net_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");
    let pcap = flags.contains(&"--pcap");
    // Spec §2 makes `NET dhcp` the default, and §6 shapes this gate around
    // `NET static` — so without this flag the default path ships unexercised.
    // QEMU's slirp runs a DHCP server on the same user netdev, so covering it
    // costs one directive and one relaxed assertion.
    let dhcp = flags.contains(&"--dhcp");

    // Port 0 lets the kernel choose; the guest is told the result. Binding
    // loopback rather than 0.0.0.0 is deliberate: slirp proxies the guest's
    // connection to NET_HOST_IP onto 127.0.0.1, and a gate that listened on
    // every interface would also be reachable from off the box.
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("cannot bind a host listener: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("cannot read the listener address: {e}"))?
        .port();
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("cannot set the listener non-blocking: {e}"))?;

    let caught = Arc::new(Mutex::new(Caught::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let accept = {
        let caught = Arc::clone(&caught);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let hard_deadline = Instant::now() + Duration::from_secs(NET_TIMEOUT_S);
            loop {
                match listener.accept() {
                    Ok((mut sock, peer)) => {
                        let _ = sock.set_read_timeout(Some(Duration::from_secs(NET_ACCEPT_S)));
                        let mut buf = Vec::new();
                        // Read until the guest closes, the read times out, or
                        // the cap is hit. `read_to_end` on a socket with a read
                        // timeout returns what it has when the timeout fires.
                        let mut chunk = [0u8; 1024];
                        loop {
                            match sock.read(&mut chunk) {
                                Ok(0) => break,
                                Ok(n) => {
                                    buf.extend_from_slice(&chunk[..n]);
                                    if buf.len() >= NET_HELLO_MAX {
                                        break;
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                        let mut c = match caught.lock() {
                            Ok(c) => c,
                            Err(p) => p.into_inner(),
                        };
                        c.connected = true;
                        c.bytes = buf;
                        c.note = Some(format!("connection from {peer}"));
                        return;
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => {
                        let mut c = match caught.lock() {
                            Ok(c) => c,
                            Err(p) => p.into_inner(),
                        };
                        c.note = Some(format!("accept failed: {e}"));
                        return;
                    }
                }
                if stop.load(Ordering::Relaxed) || Instant::now() >= hard_deadline {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        })
    };

    let net_line = if dhcp {
        "NET dhcp".to_string()
    } else {
        format!("NET static {NET_GUEST_CIDR} {NET_HOST_IP}")
    };
    let job_txt = format!(
        "# staged by `cargo xtask net-test` — AEFINITY OS phase 1a gate\n\
         BUDGET {NET_BUDGET_S}\n\
         MODE oneshot\n\
         {net_line}\n\
         NETCHECK {NET_HOST_IP}:{port}\n\
         AFTER reset\n"
    );

    let esp = stage("net-test", ci, debug, Some(&job_txt))?;

    // virtio-net-pci is the NIC Debian's OVMF has a driver for (VirtioNetDxe),
    // and it publishes EFI_SIMPLE_NETWORK and nothing above it — which is the
    // whole reason the unikernel carries smoltcp. virtio-rng-pci is present
    // because OVMF's own entropy path wants an RNG whenever a NIC is attached.
    let mut extra = vec![
        "-netdev".to_string(),
        "user,id=n0".to_string(),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];
    let pcap_path = esp.root.join("target/net.pcap");
    if pcap {
        extra.push("-object".to_string());
        extra.push(format!(
            "filter-dump,id=f0,netdev=n0,file={}",
            pcap_path.display()
        ));
    }

    println!("[4/4] booting under OVMF with a virtio NIC");
    println!("      host listener : 127.0.0.1:{port} (guest dials {NET_HOST_IP}:{port})");
    if pcap {
        println!("      frame dump    : {}", pcap_path.display());
    }
    let outcome = qemu(&esp, &extra, Some(Duration::from_secs(NET_TIMEOUT_S)))?;

    // QEMU is gone, so nothing more can connect; give the accept thread the
    // tail deadline and then take what it has.
    stop.store(true, Ordering::Relaxed);
    let _ = accept.join();
    let caught = match caught.lock() {
        Ok(c) => Caught {
            bytes: c.bytes.clone(),
            connected: c.connected,
            note: c.note.clone(),
        },
        Err(p) => {
            let c = p.into_inner();
            Caught {
                bytes: c.bytes.clone(),
                connected: c.connected,
                note: c.note.clone(),
            }
        }
    };

    println!();
    match outcome {
        Outcome::Exited(c) => println!("   QEMU exited {c} (guest ResetSystem under -no-reboot)"),
        Outcome::Signalled => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            return Ok(ExitCode::from(1));
        }
        Outcome::TimedOut => {
            eprintln!(
                "== FAIL == the guest did not reset within {NET_TIMEOUT_S}s.\n\
                 BOOTLOG.TXT on the staged ESP holds the NET: lines it reached."
            );
            return Ok(ExitCode::from(1));
        }
    }

    // The guest's own account of what it did, printed before any verdict: on a
    // failure these lines say whether the NIC came up, what MAC it read and
    // whether the connect or the write was what failed.
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (NET/NETCHECK lines) ----");
        for line in text.lines() {
            if line.contains("NET:") || line.contains("NETCHECK:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    if let Some(note) = &caught.note {
        println!("   note {note}");
    }

    let mut failures: Vec<String> = Vec::new();

    // ---- the assertion the phase exists for -------------------------------
    let text = String::from_utf8_lossy(&caught.bytes).to_string();
    if !caught.connected {
        failures.push(format!(
            "the guest never connected to {NET_HOST_IP}:{port}. Re-run with --pcap and \
             read target/net.pcap: QEMU's slirp must answer the guest's ARP for {NET_HOST_IP}"
        ));
    } else {
        println!("---- received on 127.0.0.1:{port} ----");
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
        println!("---- end ----");
    }

    let hello = text.lines().find(|l| l.starts_with("HELLO "));
    let mut guest_mac: Option<String> = None;
    match hello {
        Some(line) => {
            let mac = line["HELLO ".len()..].trim_end();
            if is_mac(mac) {
                println!("   ok   HELLO line carries a 17-character MAC {mac}");
                guest_mac = Some(mac.to_string());
            } else {
                failures.push(format!(
                    "the HELLO line carries {mac:?}, which is not a 17-character \
                     xx:xx:xx:xx:xx:xx MAC"
                ));
            }
        }
        None if caught.connected => {
            failures.push("the connection carried no line starting `HELLO `".to_string());
        }
        None => {}
    }

    // ---- and the record the guest left behind -----------------------------
    match find_ci(&esp.dir, "RESULT.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        Some(body) => {
            println!();
            println!("---- RESULT.TXT ----");
            print!("{body}");
            println!("---- end ----");
            for (key, want) in [
                ("aefinity_os", "0.1"),
                ("env", "vm"),
                ("jobs", "1"),
                ("job.1.kind", "netcheck"),
                ("job.1.ok", "true"),
                ("verdict", "OK"),
            ] {
                match record_value(&body, key) {
                    Some(got) if got == want => println!("   ok   {key}={got}"),
                    Some(got) => failures.push(format!("{key}={got}, expected {want}")),
                    None => failures.push(format!("{key} missing")),
                }
            }
            // The address the JOB.TXT asked for must be the address the record
            // reports; a guest that fell back to something else got its packets
            // through by luck, not by the directive. Under --dhcp the address
            // is the server's to choose, so the assertion is that there is one
            // and that it is on slirp's network — asserting slirp's particular
            // choice would be a gate on QEMU, not on the unikernel.
            match (record_value(&body, "ip"), dhcp) {
                (Some(got), false) => {
                    let want_ip = NET_GUEST_CIDR.split('/').next().unwrap_or("");
                    if got == want_ip {
                        println!("   ok   ip={got}");
                    } else {
                        failures.push(format!("ip={got}, expected {want_ip}"));
                    }
                }
                (Some(got), true) => {
                    if got != "none" && got.starts_with("10.0.2.") {
                        println!("   ok   ip={got} (on slirp's network)");
                    } else {
                        failures.push(format!("ip={got}, expected a DHCP lease on 10.0.2.0/24"));
                    }
                }
                (None, _) => failures.push("ip missing".to_string()),
            }
            // The address alone proves nothing about how it was obtained:
            // slirp's first DHCP lease is 10.0.2.15, the same address
            // NET_GUEST_CIDR asks for statically. `net=` is what separates a
            // working DHCP client from a silent fallback.
            let want_net = if dhcp { "dhcp" } else { "static" };
            match record_value(&body, "net") {
                Some(got) if got == want_net => println!("   ok   net={got}"),
                Some(got) => failures.push(format!("net={got}, expected {want_net}")),
                None => failures.push("net missing".to_string()),
            }
            // The MAC on the wire and the MAC in the record are two independent
            // readings of the same NIC. They have to agree, or one of them is
            // being made up.
            match (record_value(&body, "mac"), guest_mac.as_deref()) {
                (Some(rec), Some(wire)) if rec == wire => {
                    println!("   ok   mac={rec} (record == wire)")
                }
                (Some(rec), Some(wire)) => {
                    failures.push(format!("mac={rec} in RESULT.TXT but {wire} on the wire"))
                }
                (Some(rec), None) if is_mac(rec) => {
                    println!("   ok   mac={rec} (no wire MAC to compare)")
                }
                (Some(rec), None) => failures.push(format!("mac={rec} is not a MAC")),
                (None, _) => failures.push("mac missing".to_string()),
            }
        }
        None => failures.push(format!(
            "no RESULT.TXT under {} — the guest reset without writing a record",
            esp.dir.display()
        )),
    }

    match wip_cleared_check(&esp) {
        Ok(line) => println!("   ok   {line}"),
        Err(why) => failures.push(why),
    }
    wip_host_note(&esp);

    println!();
    if failures.is_empty() {
        println!("== PASS == the guest's own TCP/IP stack reached the host and named its NIC.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        if !pcap {
            eprintln!("   hint: re-run with --pcap to capture the frames.");
        }
        Ok(ExitCode::from(1))
    }
}

/// Exactly `xx:xx:xx:xx:xx:xx`, lower- or upper-case hex: 17 characters, six
/// hex pairs, five colons. The gate asserts the shape, never the value — the
/// MAC is whatever QEMU assigned.
fn is_mac(s: &str) -> bool {
    if s.len() != 17 {
        return false;
    }
    let mut parts = 0;
    for part in s.split(':') {
        parts += 1;
        if part.len() != 2 || !part.bytes().all(|b| b.is_ascii_hexdigit()) {
            return false;
        }
    }
    parts == 6
}

// ---------------------------------------------------------------------------
// os-test — AEFINITY OS phase 1b gate (program/AEFINITY_OS.md §6)
// ---------------------------------------------------------------------------

/// Bytes of request head (method line + headers) the harness buffers before
/// giving up looking for the blank line that ends them. `net/http.rs` sends
/// six short headers; this is the cap on a peer that never sends the blank
/// line at all.
const OS_HEAD_MAX: usize = 8 * 1024;
/// Total request bytes (head + body) the harness accepts. The guest's
/// `RESULT.TXT` is capped at 64 KiB on its own side (spec §4, `job.rs`
/// `JOB_MAX_BYTES`); this is double that so a legal POST is never what trips
/// the cap.
const OS_BODY_MAX: usize = 128 * 1024;
/// Read timeout once a connection is accepted. The boot itself already
/// happened by the time the guest dials in, so this only bounds a stalled
/// write, not the model load.
const OS_READ_S: u64 = 60;
/// The reply body the harness's HTTP server sends back. `net::http::post`'s
/// only caller (`job::dispatch`) reads the status code, not this text; kept
/// short and ASCII so a `tcpdump`/log read is not surprised by it.
const OS_REPLY_BODY: &str = "ok";

/// What the harness's HTTP server saw.
#[derive(Default)]
struct CaughtBody {
    /// The POST body, once a full request has been read.
    body: Vec<u8>,
    /// Whether a connection was accepted at all (a connection that never
    /// produced a parseable request still sets this).
    connected: bool,
    note: Option<String>,
}

fn os_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    // Port 0 lets the kernel choose; the guest's JOB.TXT is told the result.
    // Loopback rather than 0.0.0.0 for the same reason as net-test: slirp
    // proxies the guest's connection to NET_HOST_IP onto 127.0.0.1, and a
    // listener on every interface would also be reachable from off the box.
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("cannot bind a host listener: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("cannot read the listener address: {e}"))?
        .port();
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("cannot set the listener non-blocking: {e}"))?;

    let caught = Arc::new(Mutex::new(CaughtBody::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let server = {
        let caught = Arc::clone(&caught);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            // The whole boot, not just the accept: REPORT is the last thing
            // the guest does before AFTER, so a slow model load pushes the
            // connection attempt out nearly as far as JOB_TIMEOUT_S itself.
            let hard_deadline = Instant::now() + Duration::from_secs(JOB_TIMEOUT_S);
            loop {
                match listener.accept() {
                    Ok((mut sock, peer)) => {
                        let _ = sock.set_read_timeout(Some(Duration::from_secs(OS_READ_S)));
                        match read_http_request(&mut sock) {
                            Ok(body) => {
                                let reply = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{OS_REPLY_BODY}",
                                    OS_REPLY_BODY.len()
                                );
                                let _ = sock.write_all(reply.as_bytes());
                                let _ = sock.flush();
                                let _ = sock.shutdown(std::net::Shutdown::Both);
                                let mut c = match caught.lock() {
                                    Ok(c) => c,
                                    Err(p) => p.into_inner(),
                                };
                                c.connected = true;
                                c.body = body;
                                c.note = Some(format!("POST from {peer}"));
                                return;
                            }
                            Err(e) => {
                                // A connection that did not carry a parseable
                                // request must not consume the only chance to
                                // see the real POST — keep listening.
                                let mut c = match caught.lock() {
                                    Ok(c) => c,
                                    Err(p) => p.into_inner(),
                                };
                                c.connected = true;
                                c.note = Some(format!("connection from {peer} but {e}"));
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => {
                        let mut c = match caught.lock() {
                            Ok(c) => c,
                            Err(p) => p.into_inner(),
                        };
                        c.note = Some(format!("accept failed: {e}"));
                        return;
                    }
                }
                if stop.load(Ordering::Relaxed) || Instant::now() >= hard_deadline {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        })
    };

    // BUDGET/TOKENS/BENCH reuse job-test's constants (JOB_BUDGET_S,
    // JOB_TOKENS, JOB_BENCH_TOKENS): the same wall-clock-under-TCG reasoning
    // documented on JOB_BUDGET_S applies verbatim to a job that also does
    // PROMPT/BENCH before REPORT, and this box is shared with other gates —
    // one pair of constants for "a job that generates tokens under TCG" is
    // enough.
    let job_txt = format!(
        "# staged by `cargo xtask os-test` — AEFINITY OS phase 1b gate\n\
         BUDGET {JOB_BUDGET_S}\n\
         MODE oneshot\n\
         NET static {NET_GUEST_CIDR} {NET_HOST_IP}\n\
         TOKENS {JOB_TOKENS}\n\
         PROMPT The capital of France is\n\
         BENCH {JOB_BENCH_TOKENS}\n\
         REPORT http://{NET_HOST_IP}:{port}/aefinity/result\n\
         AFTER reset\n"
    );

    let esp = stage("os-test", ci, debug, Some(&job_txt))?;

    // Same NIC as net-test: virtio-net-pci is what Debian's OVMF has a
    // driver for (VirtioNetDxe/EFI_SIMPLE_NETWORK), and virtio-rng-pci is
    // present because OVMF's entropy path wants an RNG whenever a NIC is
    // attached.
    let extra = vec![
        "-netdev".to_string(),
        "user,id=n0".to_string(),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];

    println!("[4/4] booting under OVMF with a virtio NIC");
    println!(
        "      host HTTP server : 127.0.0.1:{port}/aefinity/result (guest posts to {NET_HOST_IP}:{port})"
    );
    let outcome = qemu(&esp, &extra, Some(Duration::from_secs(JOB_TIMEOUT_S)))?;

    // QEMU is gone, so nothing more can connect; give the server thread the
    // tail deadline and then take what it has.
    stop.store(true, Ordering::Relaxed);
    let _ = server.join();
    let caught = match caught.lock() {
        Ok(c) => CaughtBody {
            body: c.body.clone(),
            connected: c.connected,
            note: c.note.clone(),
        },
        Err(p) => {
            let c = p.into_inner();
            CaughtBody {
                body: c.body.clone(),
                connected: c.connected,
                note: c.note.clone(),
            }
        }
    };

    println!();
    match outcome {
        Outcome::Exited(c) => println!("   QEMU exited {c} (guest ResetSystem under -no-reboot)"),
        Outcome::Signalled => {
            eprintln!("== FAIL == QEMU terminated by a signal with no exit code.");
            return Ok(ExitCode::from(1));
        }
        Outcome::TimedOut => {
            eprintln!(
                "== FAIL == the guest did not reset within {JOB_TIMEOUT_S}s.\n\
                 BOOTLOG.TXT on the staged ESP holds the JOB/NET/REPORT lines it reached."
            );
            return Ok(ExitCode::from(1));
        }
    }

    // The guest's own account of what it did: on a failure these lines say
    // whether the NIC came up, and whether REPORT connected, sent, or failed.
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (JOB/NET/REPORT lines) ----");
        for line in text.lines() {
            if line.contains("JOB:") || line.contains("NET:") || line.contains("REPORT:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    if let Some(note) = &caught.note {
        println!("   note {note}");
    }

    let mut failures: Vec<String> = Vec::new();

    // ---- the assertion the phase exists for -------------------------------
    let body_text = String::from_utf8_lossy(&caught.body).to_string();
    if !caught.connected {
        failures.push(format!(
            "the guest never connected to {NET_HOST_IP}:{port} — no REPORT POST arrived"
        ));
    } else if caught.body.is_empty() {
        failures.push(
            "the guest connected but the harness never read a complete POST (see the note above)"
                .to_string(),
        );
    } else {
        println!("---- received POST body (/aefinity/result) ----");
        print!("{body_text}");
        if !body_text.ends_with('\n') {
            println!();
        }
        println!("---- end ----");
    }

    for (key, want) in [
        ("aefinity_os", "0.1"),
        ("env", "vm"),
        ("jobs", "2"),
        ("verdict", "OK"),
    ] {
        match record_value(&body_text, key) {
            Some(got) if got == want => println!("   ok   {key}={got}"),
            Some(got) => failures.push(format!("POST body: {key}={got}, expected {want}")),
            None => failures.push(format!("POST body: {key} missing")),
        }
    }

    // The on-disk record, printed for a human reading the gate output. Not
    // asserted key-by-key: spec §6 says explicitly that `report=` on it is
    // not required to reflect the POST outcome, because the file predates
    // the POST (job.rs writes it, then reports it — never the reverse). What
    // is asserted is only that it exists, because a guest that reset without
    // ever writing one failed a step long before REPORT.
    match find_ci(&esp.dir, "RESULT.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        Some(disk) => {
            println!();
            println!(
                "---- {}/RESULT.TXT (on-disk, predates the POST) ----",
                esp.dir.display()
            );
            print!("{disk}");
            println!("---- end ----");
        }
        None => failures.push(format!(
            "no RESULT.TXT under {} — the guest reset without writing a record",
            esp.dir.display()
        )),
    }

    match wip_cleared_check(&esp) {
        Ok(line) => println!("   ok   {line}"),
        Err(why) => failures.push(why),
    }
    wip_host_note(&esp);

    println!();
    if failures.is_empty() {
        println!("== PASS == the guest POSTed RESULT.TXT to the host collector and reset.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// Read one HTTP/1.1 request off `sock`: the head (request line + headers, up
/// to the blank line), then exactly `Content-Length` more bytes as the body.
/// Returns the body.
///
/// `Transfer-Encoding: chunked` is not decoded: `net::http::post` (spec §5)
/// never sends it, so a peer that does is not phase 1b's client and gets an
/// error here rather than a wrong body.
fn read_http_request(sock: &mut std::net::TcpStream) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        if let Some(i) = find_subslice(&buf, b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() >= OS_HEAD_MAX {
            return Err(format!(
                "request head exceeded {OS_HEAD_MAX} bytes with no blank line"
            ));
        }
        let n = sock
            .read(&mut chunk)
            .map_err(|e| format!("read failed: {e}"))?;
        if n == 0 {
            return Err("peer closed before sending a complete request head".to_string());
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..head_end]);
    let content_length: usize = head
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.trim().eq_ignore_ascii_case("content-length") {
                v.trim().parse().ok()
            } else {
                None
            }
        })
        .ok_or_else(|| "request carried no Content-Length header".to_string())?;
    if content_length > OS_BODY_MAX {
        return Err(format!(
            "Content-Length {content_length} exceeds the {OS_BODY_MAX}-byte cap"
        ));
    }

    let mut body = buf[head_end..].to_vec();
    while body.len() < content_length {
        let n = sock
            .read(&mut chunk)
            .map_err(|e| format!("read failed: {e}"))?;
        if n == 0 {
            return Err(format!(
                "peer closed after {} of {content_length} body bytes",
                body.len()
            ));
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(body)
}

/// First index of `needle` in `hay`, or `None`.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
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
    match wip_cleared_check(&esp) {
        Ok(line) => println!("   ok   {line}"),
        Err(why) => failures.push(why),
    }
    wip_host_note(&esp);

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

/// Assert that the guest cleared its in-progress marker, reading the guest's
/// own `BOOTLOG.TXT` line rather than the host mirror of the ESP.
///
/// `aegis-uefi`'s `job::confirm_wip_cleared` writes
/// `JOB: RESULT.WIP cleared=<bool> (open=… dir=… retried=…)` after the settle
/// stall, from two independent probes of the volume it is holding: an `open`
/// that distinguishes `NOT_FOUND` from unreadable, and a walk of the root
/// directory. That line is the claim spec §3 rests on — a `RESULT.WIP` on a
/// stick that comes home means the box did not finish — so it is what this
/// gate checks.
///
/// It is checked there and not with `ls target/esp` because the difference is
/// measured, not assumed: on 2026-09-01, in the same run, both guest probes
/// reported the marker absent while `target/esp/result.wip` was still in the
/// host directory, and the guest's *later* writes (the BOOTLOG lines below
/// it, `RESULT.TXT`) committed through fine. QEMU's `fat:rw:` backend commits
/// guest writes back to the host directory; it does not commit the unlink.
/// A real stick has no such mirror — the FAT directory the firmware walks is
/// the medium — so a host-side `ls` here would assert a QEMU property, not a
/// unikernel one. The host state is still printed, because someone reading
/// this output will see the file and deserves to be told why.
fn wip_cleared_check(esp: &Esp) -> Result<String, String> {
    let Some(log) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok())
    else {
        return Err("no BOOTLOG.TXT on the staged ESP — cannot confirm RESULT.WIP".to_string());
    };
    let Some(line) = log.lines().rfind(|l| l.contains("RESULT.WIP cleared=")) else {
        return Err(
            "BOOTLOG.TXT has no `RESULT.WIP cleared=` line — the guest never confirmed              the marker was off the volume"
                .to_string(),
        );
    };
    let line = line.trim().to_string();
    if line.contains("cleared=true") {
        Ok(line)
    } else {
        Err(format!(
            "the guest could not clear its in-progress marker: {line}"
        ))
    }
}

/// Report the host mirror's view of the marker (see [`wip_cleared_check`]).
fn wip_host_note(esp: &Esp) {
    if let Some(p) = find_ci(&esp.dir, "RESULT.WIP") {
        println!("   note {} is still in the host mirror.", p.display());
        println!("        QEMU `fat:rw:` commits guest writes but not the unlink;");
        println!("        the guest's BOOTLOG line above is the statement about the volume.");
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
