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
//! `cargo xtask resident-test` is the AEFINITY OS phase-2 gate (spec §4/§6):
//! the guest boots into `MODE resident` and becomes a TCP job server, and the
//! harness — the client, for once — talks the §4 line protocol to it through
//! QEMU's `hostfwd`: READY, `PING`/`PONG`, two `JOB` blocks on one connection,
//! a garbage line that must come back `ERR unknown`, and `REBOOT`.
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
        "lab-test" => match lab_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: lab-test failed: {e}");
                ExitCode::from(1)
            }
        },
        "files-test" => match files_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: files-test failed: {e}");
                ExitCode::from(1)
            }
        },
        "files-soak" => match files_soak(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: files-soak failed: {e}");
                ExitCode::from(1)
            }
        },
        "resident-test" => match resident_test(&flags) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("xtask: resident-test failed: {e}");
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
    cargo xtask <boot-test|job-test|net-test|os-test|resident-test|files-test|files-soak|lab-test|job-budget-test>
                [--ci] [--debug] [--dhcp] [--pcap] [--vvfat]

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

    resident-test
                 AEFINITY OS phase 2 (program/AEFINITY_OS.md §4/§6). Stages a
                 JOB.TXT with `MODE resident` / `NET static {NET_GUEST_CIDR} {NET_HOST_IP}` /
                 `LISTEN {RESIDENT_GUEST_PORT}`, boots with a virtio NIC and
                 `-netdev user,...,hostfwd=tcp:127.0.0.1:<port>-:{RESIDENT_GUEST_PORT}`, then talks the
                 §4 line protocol to the guest while it runs: retry until the
                 READY banner (≤{RESIDENT_READY_S}s, each attempt ≤{RESIDENT_ATTEMPT_S}s), PING->PONG, a
                 two-directive JOB, a second JOB on the same connection, a
                 garbage line, one {RESIDENT_OVERLONG_BYTES}-byte line with no newline in it, and
                 REBOOT. PASS = both RESULTs carry
                 verdict=OK / env=vm with jobs=2 then jobs=1, garbage answered
                 `ERR unknown`, the over-long line answered `ERR too-large` and
                 then dropped with the listener coming back,
                 REBOOT answered `BYE`, RESULT.TXT on the volume
                 holds the LAST job, and QEMU exits because the guest reset.
                 Both RESULTs are printed. Structure and exit codes only.

    files-soak
                 AEFINITY OS phase 4 INTEGRITY gate (design §7 / §8). {SOAK_CYCLES} PUT/RM/RELOAD
                 cycles in ONE boot, mixing the three ordinary SOAK*.BIN names
                 with the three artifacts, and after EVERY cycle a fresh
                 guest-side `SHA` of every file present — checked against the
                 host's own digest of the bytes that were sent, both by the
                 client-visible name (through §8's pointer) and by the physical
                 A/B half. First mismatch fails the gate and names the cycle.
                 Also asserts, every cycle, `HEALTH parts=0 degraded=none` and
                 that `RM` of an A/B half is `ERR protected`; late in the run,
                 that `RM MODEL.SAF`-style artifact removal takes both halves
                 and leaves no orphaned pointer.

                 The boot volume is a REAL FAT32 image (`mformat` + `mcopy`,
                 `-drive format=raw`), not QEMU's `fat:rw:`. vvfat synthesises a
                 FAT filesystem over a host directory and is documented-fragile
                 under create/rename/delete churn — it fails this soak inside
                 `block/vvfat.c`, which is a fact about the emulator and not
                 about the file plane. `--vvfat` selects it anyway, to
                 reproduce that. See AEFINITY_OS_STATUS.md §9.

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

    --vvfat      files-soak only: boot the soak on QEMU's `fat:rw:` directory
                 mapping instead of a real FAT image. Expected to fail, and the
                 point of running it is to see HOW: a digest that moves under
                 churn, or `block/vvfat.c` asserting and taking QEMU down.

    --pcap       net-test only: also write every frame on the guest's NIC to
                 target/net.pcap (`-object filter-dump`). Read it with
                 `tcpdump -r target/net.pcap`. The first thing to look for when
                 the guest never connects is whether QEMU's slirp answered the
                 guest's ARP for {NET_HOST_IP}.

ENVIRONMENT:
    AEGIS_SOAK_CYCLES   files-soak cycle count (default: {SOAK_CYCLES}).
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

/// Build the QEMU invocation every gate shares, returning the binary name and
/// the ready-to-run `Command`.
///
/// Split out of [`qemu`] so a gate that has to *talk to* the guest while it
/// runs (`resident-test`) spawns exactly the same machine as the gates that
/// only wait for it to exit. One definition of the machine, not two.
fn qemu_command(esp: &Esp, extra: &[String]) -> (String, Command) {
    let esp_arg = format!("format=raw,file=fat:rw:{}", esp.dir.display());
    qemu_command_on(esp, extra, &esp_arg)
}

/// [`qemu_command`] with the boot volume named explicitly.
///
/// `fat:rw:` is QEMU's `vvfat`, which synthesises a FAT filesystem over a host
/// directory on the fly. It is the right thing for a gate that wants to read
/// the guest's writes back on the host, and it is **not** a filesystem: the
/// mapping between directory entries and host files is rebuilt as the guest
/// mutates it, and heavy create/rename/delete churn is a documented weak
/// spot — `block/vvfat.c` will assert and take QEMU down with it.
/// `files-soak` therefore points this at a real FAT image (`mformat` + `mcopy`)
/// by default, so a digest mismatch under churn is evidence about the guest
/// rather than about the emulator. See `AEFINITY_OS_STATUS.md` §9.
fn qemu_command_on(esp: &Esp, extra: &[String], drive: &str) -> (String, Command) {
    let qemu_bin = env::var("QEMU").unwrap_or_else(|_| "qemu-system-x86_64".into());
    let esp_arg = drive.to_string();
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
    (qemu_bin, cmd)
}

/// Boot a staged ESP under OVMF. `extra` is appended verbatim (net devices,
/// hostfwd, …). `timeout` of `None` waits forever, which is what `boot-test`
/// has always done — the unikernel signals through isa-debug-exit.
fn qemu(esp: &Esp, extra: &[String], timeout: Option<Duration>) -> Result<Outcome, String> {
    let (qemu_bin, mut cmd) = qemu_command(esp, extra);
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

// ---------------------------------------------------------------------------
// resident-test — AEFINITY OS phase 2 gate (program/AEFINITY_OS.md §4/§6)
// ---------------------------------------------------------------------------

/// The port the guest listens on, written into the staged `JOB.TXT` as
/// `LISTEN` and used as the guest side of QEMU's `hostfwd`. Spec §2's default.
const RESIDENT_GUEST_PORT: u16 = 4242;
/// `BUDGET` written into each `JOB` body sent over the socket.
///
/// Spec §6 sketches this gate with a small budget. That is an iron figure and
/// it does not cross into TCG (CLAUDE.md Rule A): the same two-directive job
/// under `accel=tcg` needed more than 180 s when phase 0 measured it, which is
/// why [`JOB_BUDGET_S`] is what it is. This gate uses the same constant for
/// the same reason — a budget the emulator cannot meet turns a protocol gate
/// into a budget gate, and `job-budget-test` already covers the budget.
const RESIDENT_BUDGET_S: u64 = JOB_BUDGET_S;
/// `TOKENS` for the first job's `PROMPT`, and `BENCH n` for its second
/// directive — the two-step shape spec §6 asks for, so the record comes back
/// with `jobs=2`.
const RESIDENT_TOKENS: usize = JOB_TOKENS;
/// `BENCH n` in the first job.
const RESIDENT_BENCH_TOKENS: usize = JOB_BENCH_TOKENS;
/// `TOKENS` for the second job. Deliberately tiny: the second job's purpose is
/// to prove the server survived the first one and is still serving the same
/// connection, not to run the work again.
const RESIDENT_TOKENS_2: usize = 4;

/// Hard deadline on the whole QEMU run: firmware, the model load, the listener
/// coming up, two socket-delivered jobs and the reset. Larger than
/// [`JOB_TIMEOUT_S`] because this gate runs *two* jobs on one boot where
/// `job-test` runs one; it stays a real deadline, and a guest that never
/// resets still fails here.
const RESIDENT_TIMEOUT_S: u64 = 2700;
/// How long the harness keeps retrying for the READY banner.
///
/// Spec §6 says "retry ≤60 s" and this gate was briefed at 90 s. Both are
/// figures for a box that is already up. Here the retry runs *concurrently
/// with the boot*, and under TCG the model load alone takes minutes before
/// `job::dispatch` is reached — so the outer window is sized to the boot and
/// the 90 s figure survives as [`RESIDENT_ATTEMPT_S`], the window one
/// connection attempt gets. Rule A: this is a harness bound, not a
/// measurement, and nothing is recorded from it.
const RESIDENT_READY_S: u64 = 1200;
/// How long a single connection attempt waits for the banner before it is
/// dropped and retried.
///
/// QEMU's `hostfwd` accepts the harness's TCP connection on the host side
/// immediately and only then tries to reach the guest, so a `connect` that
/// succeeds proves nothing about the guest — the banner does. An attempt that
/// does not produce one is a dead forward and is retried on a fresh socket.
const RESIDENT_ATTEMPT_S: u64 = 90;
/// How long a short reply (`PONG`, `RUNNING`, `BYE`, `ERR unknown`) is given.
const RESIDENT_IO_S: u64 = 120;
/// How long a `RESULT` is given to come back after `RUNNING`. This is the
/// window the actual inference runs in, under emulation.
const RESIDENT_JOB_S: u64 = 1500;
/// How long QEMU gets to exit after the guest answered `BYE` to `REBOOT`.
const RESIDENT_EXIT_S: u64 = 240;
/// Bytes in the deliberately over-long command line, with no `\n` anywhere in
/// it. Comfortably past the server's 64 KiB line cap (spec §4) so the case is
/// unambiguous, and past the network stack's own per-read cap too — the two
/// are the same number, and an over-long line has to be answered
/// `ERR too-large` whichever of them notices first.
const RESIDENT_OVERLONG_BYTES: usize = 70 * 1024;
/// How long the over-long line is given to go out before the harness stops
/// pushing and reads the answer. The guest stops reading once it has decided
/// the line is over the cap, so the tail of this write is *expected* to stall
/// or fail; that is the case being tested, not a harness fault.
const RESIDENT_OVERLONG_WRITE_S: u64 = 30;
/// How long the guest gets to come back up on the listener after it drops the
/// over-long client. No boot happens here — only `tcp_close` and a re-listen.
const RESIDENT_RELISTEN_S: u64 = 180;

/// A line-oriented client for the resident protocol (spec §4).
///
/// The server writes whole lines but is free to coalesce them into one
/// segment, so the harness buffers exactly the way the server does: a
/// residual buffer, and a `read_line` that only ever consumes up to the next
/// `\n`. Reading with `read_to_string` here would block until the guest closed
/// the connection, which in resident mode it never does.
struct ResidentConn {
    stream: std::net::TcpStream,
    pending: Vec<u8>,
}

impl ResidentConn {
    fn connect(port: u16) -> Result<ResidentConn, String> {
        let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
        let stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .map_err(|e| format!("connect 127.0.0.1:{port}: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay: {e}"))?;
        Ok(ResidentConn {
            stream,
            pending: Vec::new(),
        })
    }

    fn send(&mut self, msg: &str) -> Result<(), String> {
        use std::io::Write;
        self.stream
            .write_all(msg.as_bytes())
            .map_err(|e| format!("write {msg:?}: {e}"))?;
        self.stream.flush().map_err(|e| format!("flush: {e}"))
    }

    /// Next line, without its terminator. `wait` bounds the whole call.
    fn read_line(&mut self, wait: Duration) -> Result<String, String> {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(i) = self.pending.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = self.pending.drain(..=i).collect();
                let text = String::from_utf8_lossy(&line[..line.len() - 1])
                    .trim_end_matches('\r')
                    .to_string();
                return Ok(text);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!("no line within {}s", wait.as_secs()));
            }
            // Cap each blocking read so the deadline is checked regularly and
            // a stalled guest is reported as a stall rather than a hang.
            let slice = left.min(Duration::from_secs(5));
            self.stream
                .set_read_timeout(Some(slice))
                .map_err(|e| format!("set_read_timeout: {e}"))?;
            let mut buf = [0u8; 4096];
            match self.stream.read(&mut buf) {
                Ok(0) => return Err("the guest closed the connection".to_string()),
                Ok(n) => self.pending.extend_from_slice(&buf[..n]),
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(format!("read: {e}")),
            }
        }
    }

    /// Read the `RESULT\n<body>END\n` block of spec §4, returning the body.
    /// `RUNNING` is consumed first because the server always sends it.
    fn read_result(&mut self, hard_deadline: Instant) -> Result<String, String> {
        self.read_result_within(hard_deadline, RESIDENT_JOB_S)
    }

    /// [`ResidentConn::read_result`] with a caller-chosen ceiling on the wait
    /// for the `RESULT` line.
    ///
    /// Phase 5's `VERIFY` replays 64 decode steps through the scalar
    /// full-integer engine under TCG, which is slower than anything phase 2
    /// asked a guest to do. The bound is a hang detector and nothing else — no
    /// gate times anything (Rule A).
    fn read_result_within(&mut self, hard_deadline: Instant, job_s: u64) -> Result<String, String> {
        let first = self.read_line(left(hard_deadline, RESIDENT_IO_S)?)?;
        if first != "RUNNING" {
            return Err(format!("expected RUNNING, got {first:?}"));
        }
        let head = self.read_line(left(hard_deadline, job_s)?)?;
        if head != "RESULT" {
            return Err(format!("expected RESULT, got {head:?}"));
        }
        let mut body = String::new();
        loop {
            let line = self.read_line(left(hard_deadline, RESIDENT_IO_S)?)?;
            if line == "END" {
                return Ok(body);
            }
            if body.len() > 256 * 1024 {
                return Err("RESULT body ran past 256 KiB with no END".to_string());
            }
            body.push_str(&line);
            body.push('\n');
        }
    }
}

/// `want` seconds, or whatever is left before the run's hard deadline —
/// whichever is shorter. An expired deadline is an error, not a zero wait, so
/// the gate says the run overran instead of reporting the next read as a
/// protocol failure.
fn left(hard_deadline: Instant, want: u64) -> Result<Duration, String> {
    let remaining = hard_deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(format!(
            "the run passed its {RESIDENT_TIMEOUT_S}s hard deadline"
        ));
    }
    Ok(remaining.min(Duration::from_secs(want)))
}

/// A `JOB … END` block for the wire (spec §4).
fn resident_job_block(tokens: usize, bench: Option<usize>, prompt: &str) -> String {
    let mut s = format!(
        "JOB\n\
         BUDGET {RESIDENT_BUDGET_S}\n\
         TOKENS {tokens}\n\
         PROMPT {prompt}\n"
    );
    if let Some(n) = bench {
        s.push_str(&format!("BENCH {n}\n"));
    }
    s.push_str("END\n");
    s
}

fn resident_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    // Ask the kernel for a free loopback port, then let go of it: QEMU's
    // hostfwd has to be the one bound to it. The gap is a race in principle;
    // in practice nothing else on this box claims an ephemeral port in the
    // microseconds between, and a hostfwd that cannot bind makes QEMU exit
    // immediately with a message rather than pass silently.
    let host_port = {
        let probe = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("cannot pick a host port: {e}"))?;
        probe
            .local_addr()
            .map_err(|e| format!("cannot read the probe address: {e}"))?
            .port()
    };

    let job_txt = format!(
        "# staged by `cargo xtask resident-test` — AEFINITY OS phase 2 gate\n\
         MODE resident\n\
         NET static {NET_GUEST_CIDR} {NET_HOST_IP}\n\
         LISTEN {RESIDENT_GUEST_PORT}\n"
    );

    let esp = stage("resident-test", ci, debug, Some(&job_txt))?;

    let extra = vec![
        "-netdev".to_string(),
        format!("user,id=n0,hostfwd=tcp:127.0.0.1:{host_port}-:{RESIDENT_GUEST_PORT}"),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];

    println!("[4/4] booting under OVMF with a virtio NIC and a host forward");
    println!(
        "      hostfwd : 127.0.0.1:{host_port} -> guest {NET_GUEST_CIDR} port {RESIDENT_GUEST_PORT}"
    );
    let (qemu_bin, mut cmd) = qemu_command(&esp, &extra);
    println!("      $ {qemu_bin} -machine q35,accel=tcg -cpu max -m 2048 ...");
    println!();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86."))?;

    // One hard deadline over the whole exchange, so a guest that answers each
    // step just inside its own window cannot walk the gate past every bound in
    // small steps.
    let hard_deadline = Instant::now() + Duration::from_secs(RESIDENT_TIMEOUT_S);
    let run = resident_exchange(&mut child, host_port, hard_deadline);

    // Whatever happened, QEMU must not be left running.
    let outcome = match &run {
        Ok(_) => wait_for_exit(&mut child, Duration::from_secs(RESIDENT_EXIT_S)),
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            Outcome::TimedOut
        }
    };

    println!();
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (RESIDENT/NET lines) ----");
        for line in text.lines() {
            if line.contains("RESIDENT:") || line.contains("NET:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    let session = match run {
        Ok(s) => s,
        Err(e) => {
            eprintln!("== FAIL == {e}");
            eprintln!(
                "   The BOOTLOG lines above are the guest's own account of how far it got.\n\
                 No `RESIDENT: listening` line means the NIC or the address never came up."
            );
            return Ok(ExitCode::from(1));
        }
    };

    let mut failures: Vec<String> = Vec::new();

    println!("   ok   banner {}", session.banner);
    // Structure only (Rule A). `env=vm` in the banner is the same statement
    // the record makes: nothing this gate sees may be quoted as performance.
    if !session.banner.contains(" env=vm ") && !session.banner.ends_with(" env=vm") {
        failures.push(format!(
            "the READY banner does not carry env=vm: {:?}",
            session.banner
        ));
    }
    if !session.banner.contains("cpu=") {
        failures.push(format!(
            "the READY banner does not carry cpu=: {:?}",
            session.banner
        ));
    }

    for (n, body) in session.results.iter().enumerate() {
        println!();
        println!("---- RESULT {} (over the socket) ----", n + 1);
        print!("{body}");
        println!("---- end ----");
    }
    println!();

    let expects: [(&str, &str); 4] = [
        ("aefinity_os", "0.1"),
        ("verdict", "OK"),
        ("jobs", "2"),
        ("env", "vm"),
    ];
    match session.results.first() {
        Some(body) => {
            for (key, want) in expects {
                match record_value(body, key) {
                    Some(got) if got == want => println!("   ok   job 1 {key}={got}"),
                    Some(got) => failures.push(format!("job 1 {key}={got}, expected {want}")),
                    None => failures.push(format!("job 1 {key} missing")),
                }
            }
        }
        None => failures.push("the first JOB produced no RESULT".to_string()),
    }
    // The second job is the proof the server survived the first: same
    // connection, same engine, one directive.
    match session.results.get(1) {
        Some(body) => {
            for (key, want) in [("verdict", "OK"), ("jobs", "1"), ("env", "vm")] {
                match record_value(body, key) {
                    Some(got) if got == want => println!("   ok   job 2 {key}={got}"),
                    Some(got) => failures.push(format!("job 2 {key}={got}, expected {want}")),
                    None => failures.push(format!("job 2 {key} missing")),
                }
            }
        }
        None => failures.push("the second JOB produced no RESULT".to_string()),
    }

    // `RESULT.TXT` on the volume is the other half of the phase-2 contract:
    // a Debian harvest of the same disk has to be able to see what the box
    // last did without having been the client. It must be the *last* job's
    // record, not the first.
    match find_ci(&esp.dir, "RESULT.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        Some(body) => {
            println!();
            println!("---- RESULT.TXT (on the volume) ----");
            print!("{body}");
            println!("---- end ----");
            // vvfat commits a guest write but not a guest unlink (the same
            // mechanism `wip_host_note` documents), so the host mirror of a
            // record that replaced a longer one can show the older record's
            // tail after the newer one. `record_value` reads the first
            // `key=` line, which is the record the guest actually wrote.
            if body.matches("\nverdict=").count() > 1 {
                println!("   note the host mirror of RESULT.TXT carries a previous record's tail.");
                println!(
                    "        QEMU `fat:rw:` commits the guest's write but not its unlink, so a"
                );
                println!(
                    "        shorter record written over a longer one leaves the old bytes past"
                );
                println!("        its end in the mirror only. The assertions below read the first");
                println!("        record, which is the one the guest wrote.");
            }
            match record_value(&body, "jobs") {
                Some("1") => println!("   ok   the volume holds the last job's record (jobs=1)"),
                Some(got) => failures.push(format!(
                    "RESULT.TXT on the volume says jobs={got}; the last job ran one directive"
                )),
                None => failures.push("RESULT.TXT on the volume has no jobs= key".to_string()),
            }
        }
        None => failures.push(format!(
            "no RESULT.TXT under {} — the resident server never wrote the last job \
             to the volume",
            esp.dir.display()
        )),
    }

    match outcome {
        Outcome::Exited(c) => {
            println!("   ok   QEMU exited {c} (guest ResetSystem under -no-reboot)")
        }
        Outcome::Signalled => {
            failures.push("QEMU terminated by a signal with no exit code".to_string())
        }
        Outcome::TimedOut => failures.push(format!(
            "QEMU did not exit within {RESIDENT_EXIT_S}s of the guest answering BYE to REBOOT"
        )),
    }

    println!();
    if failures.is_empty() {
        println!("== PASS == the resident server served two jobs, refused garbage, and reset.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// What the harness got out of one resident session.
struct ResidentSession {
    banner: String,
    results: Vec<String>,
}

/// Connect to the forwarded resident port and wait for the READY banner,
/// retrying until `wait_s` seconds have passed or the run's hard deadline is
/// reached.
///
/// Used twice: once for the guest's first boot into the listener, and once
/// after the over-long-line case, which by spec §4 costs the client its
/// connection and so has to be reconnected before the run can go on.
fn resident_ready(
    child: &mut Child,
    host_port: u16,
    wait_s: u64,
    hard_deadline: Instant,
) -> Result<(ResidentConn, String), String> {
    let deadline = (Instant::now() + Duration::from_secs(wait_s)).min(hard_deadline);
    let mut attempt = 0usize;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!(
                "QEMU exited ({status}) before the guest ever answered on 127.0.0.1:{host_port}"
            ));
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "no READY banner on 127.0.0.1:{host_port} within {wait_s}s \
                 ({attempt} attempts)"
            ));
        }
        attempt += 1;
        match ResidentConn::connect(host_port) {
            Ok(mut c) => match c.read_line(Duration::from_secs(RESIDENT_ATTEMPT_S)) {
                Ok(line) if line.starts_with("AEFINITY-OS ") && line.contains(" READY ") => {
                    println!("   attempt {attempt}: {line}");
                    return Ok((c, line));
                }
                // A stale forward the guest accepted before this attempt was
                // made is being served; it will be dropped when its peer (the
                // previous attempt's socket) is seen to be gone. Retry.
                Ok(line) => println!("   attempt {attempt}: got {line:?}, retrying"),
                Err(e) => println!("   attempt {attempt}: {e}"),
            },
            Err(e) => println!("   attempt {attempt}: {e}"),
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

/// Drive the protocol of spec §4 end to end against the running guest.
fn resident_exchange(
    child: &mut Child,
    host_port: u16,
    hard_deadline: Instant,
) -> Result<ResidentSession, String> {
    // ---- connect and wait for READY ---------------------------------------
    let (mut conn, banner) = resident_ready(child, host_port, RESIDENT_READY_S, hard_deadline)?;

    // ---- PING -> PONG -----------------------------------------------------
    conn.send("PING\n")?;
    let pong = conn.read_line(left(hard_deadline, RESIDENT_IO_S)?)?;
    if pong != "PONG" {
        return Err(format!("PING answered {pong:?}, expected PONG"));
    }
    println!("   ok   PING -> PONG");

    // ---- a second concurrent connection must be refused (spec §4) --------
    // Done while the first connection is idle, which is the case that matters:
    // the server has to answer a second peer out of its own wait loop, not
    // only when the first peer next says something.
    match ResidentConn::connect(host_port) {
        Ok(mut second) => match second.read_line(left(hard_deadline, RESIDENT_IO_S)?) {
            Ok(line) if line == "BUSY" => println!("   ok   second connection -> BUSY"),
            Ok(line) => {
                return Err(format!(
                    "a second concurrent connection was answered {line:?}, expected BUSY"
                ));
            }
            Err(e) => return Err(format!("a second concurrent connection got no BUSY: {e}")),
        },
        Err(e) => return Err(format!("could not open a second connection: {e}")),
    }

    let mut results = Vec::new();

    // ---- job 1: two directives -------------------------------------------
    println!("   sending JOB 1 (PROMPT + BENCH {RESIDENT_BENCH_TOKENS})");
    conn.send(&resident_job_block(
        RESIDENT_TOKENS,
        Some(RESIDENT_BENCH_TOKENS),
        "The capital of France is",
    ))?;
    results.push(conn.read_result(hard_deadline)?);
    println!("   ok   JOB 1 answered");

    // ---- job 2: the server survived job 1 --------------------------------
    println!("   sending JOB 2 on the same connection");
    conn.send(&resident_job_block(RESIDENT_TOKENS_2, None, "Hello"))?;
    results.push(conn.read_result(hard_deadline)?);
    println!("   ok   JOB 2 answered — the server survived a job");

    // ---- garbage -> ERR unknown ------------------------------------------
    conn.send("ZORP not a command\n")?;
    let err = conn.read_line(left(hard_deadline, RESIDENT_IO_S)?)?;
    if err != "ERR unknown" {
        return Err(format!("garbage answered {err:?}, expected `ERR unknown`"));
    }
    println!("   ok   garbage -> ERR unknown");

    // ---- one over-long line -> ERR too-large (spec §4) --------------------
    let mut conn = resident_overlong(child, host_port, conn, hard_deadline)?;

    // ---- REBOOT -> BYE ----------------------------------------------------
    conn.send("REBOOT\n")?;
    let bye = conn.read_line(left(hard_deadline, RESIDENT_IO_S)?)?;
    if bye != "BYE" {
        return Err(format!("REBOOT answered {bye:?}, expected BYE"));
    }
    println!("   ok   REBOOT -> BYE");
    drop(conn);

    Ok(ResidentSession { banner, results })
}

/// Send one command line longer than the 64 KiB cap of spec §4 and assert the
/// server answers `ERR too-large` instead of dropping the peer in silence.
///
/// This exists because it once did drop it in silence. The network stack's own
/// per-read cap and the server's line cap are the same number, so the stack hit
/// its cap first and reported the generic "closed", which the line reader could
/// not tell from the peer having vanished — the client got zero bytes back and
/// no way to know why. The regression is invisible to every other step in this
/// gate, so it gets its own.
///
/// The write runs on its own thread: the guest stops reading at the cap, so the
/// tail of an over-long line has nowhere to go, and the answer has to be read
/// while the write is still stuck. Both fds have a timeout, so neither can hang
/// the gate.
///
/// By spec the connection is spent afterwards, so this returns a fresh one.
fn resident_overlong(
    child: &mut Child,
    host_port: u16,
    mut conn: ResidentConn,
    hard_deadline: Instant,
) -> Result<ResidentConn, String> {
    println!("   sending one {RESIDENT_OVERLONG_BYTES}-byte line with no newline in it");
    let mut writer = conn
        .stream
        .try_clone()
        .map_err(|e| format!("could not clone the connection to write on: {e}"))?;
    writer
        .set_write_timeout(Some(Duration::from_secs(RESIDENT_OVERLONG_WRITE_S)))
        .map_err(|e| format!("set_write_timeout: {e}"))?;
    let pusher = std::thread::spawn(move || {
        use std::io::Write;
        // A stalled or refused write is the expected ending here, so the
        // result is deliberately dropped: what this case asserts is what came
        // *back*, which the reader below has.
        let blob = vec![b'X'; RESIDENT_OVERLONG_BYTES];
        let _ = writer.write_all(&blob);
    });

    let answer = conn.read_line(left(hard_deadline, RESIDENT_IO_S)?);
    let _ = pusher.join();
    match answer {
        Ok(line) if line == "ERR too-large" => {
            println!("   ok   over-long line -> ERR too-large");
        }
        Ok(line) => {
            return Err(format!(
                "an over-long line was answered {line:?}, expected `ERR too-large`"
            ));
        }
        Err(e) => {
            return Err(format!(
                "an over-long line got no `ERR too-large` back: {e} — the server must \
                 answer before it drops the peer (spec §4)"
            ));
        }
    }

    // Spec §4: "the connection is dropped". Either ending is that drop — a
    // clean close, or a reset once the guest tears down a socket the harness
    // is still pushing into. What must not happen is the server carrying on
    // taking commands on a connection it has already refused.
    match conn.read_line(left(hard_deadline, RESIDENT_IO_S)?) {
        Ok(line) => {
            return Err(format!(
                "the server kept serving after `ERR too-large` and sent {line:?}"
            ));
        }
        Err(e) => println!("   ok   connection dropped after ERR too-large ({e})"),
    }
    drop(conn);

    let (conn, banner) = resident_ready(child, host_port, RESIDENT_RELISTEN_S, hard_deadline)?;
    println!("   ok   the server took a new connection afterwards: {banner}");
    Ok(conn)
}

/// Wait for a spawned QEMU to exit, killing it if it overruns.
fn wait_for_exit(child: &mut Child, limit: Duration) -> Outcome {
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return match status.code() {
                    Some(c) => Outcome::Exited(c),
                    None => Outcome::Signalled,
                };
            }
            Ok(None) => {}
            Err(_) => return Outcome::Signalled,
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Outcome::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(200));
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

// ---------------------------------------------------------------------------
// files-test — AEFINITY OS phase 4 (design §7)
// ---------------------------------------------------------------------------

/// Hard deadline over the whole phase-4 exchange.
///
/// Larger than `resident-test`'s because this gate moves 64 MiB across the
/// wire, writes it to FAT, reads it back for a digest, and then hashes it a
/// third time for the `SHA` verb — under TCG, on a shared 6 GB box. Rule A:
/// this is a harness bound, not a measurement, and nothing is recorded from it.
const FILES_TIMEOUT_S: u64 = 3600;
/// How long the harness keeps retrying for the READY banner.
const FILES_READY_S: u64 = 1200;
/// How long a short reply (`OK`, `ERR …`, `SEND`, a `LS` header) is given.
const FILES_IO_S: u64 = 180;
/// How long one bulk transfer step is given: the 64 MiB `PUT` payload, its
/// readback digest, or a `SHA` over it.
const FILES_BULK_S: u64 = 1500;
/// How long QEMU gets to exit after the guest answers `BYE`.
const FILES_EXIT_S: u64 = 240;
/// The small round-trip payload: 256 KiB of deterministic pseudorandom bytes.
const FILES_SMALL_BYTES: usize = 256 * 1024;
/// The large payload (design §7): 64 MiB. A 256 KiB transfer proves nothing
/// about a 1.8 GB one — this is the case that exercises §4.2's listener window
/// and §8's per-chunk watchdog re-arm across a readback.
const FILES_BIG_BYTES: usize = 64 * 1024 * 1024;
/// The shared secret staged into `JOB.TXT`, so the gate can prove design
/// §1.2's rule that a `TOKEN` gates **reads** as well as writes.
const FILES_TOKEN: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
/// A name one byte over `NAME_MAX_BYTES` (31), for the `ERR bad-name` case.
const FILES_LONG_NAME: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA.BIN";
/// A declared length over `PUT_MAX_BYTES` (2 GiB): 3 GiB.
const FILES_OVER_LEN: u64 = 3 * 1024 * 1024 * 1024;

/// A minimal FIPS 180-4 sha256, host side.
///
/// Deliberately a **second implementation**, not a call into `aegis-core`: the
/// whole point of `SHA <NAME>` in this gate is that the guest's digest of the
/// bytes on its volume equals a digest computed by something that is not the
/// code under test (CLAUDE.md Rule D). It is cross-checked once per run
/// against `sha256sum` where that binary exists, so a bug in *this* function
/// cannot make the gate pass either.
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    len: usize,
    total: u64,
}

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            len: 0,
            total: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        while !data.is_empty() {
            let take = (64 - self.len).min(data.len());
            self.buf[self.len..self.len + take].copy_from_slice(&data[..take]);
            self.len += take;
            data = &data[take..];
            if self.len == 64 {
                let block = self.buf;
                self.block(&block);
                self.len = 0;
            }
        }
    }

    fn block(&mut self, b: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut h = self.state;
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ ((!h[4]) & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA_K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);
            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(t1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = t1.wrapping_add(t2);
        }
        for (i, v) in h.iter().enumerate() {
            self.state[i] = self.state[i].wrapping_add(*v);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bits = self.total * 8;
        self.update(&[0x80]);
        while self.len != 56 {
            self.update(&[0u8]);
        }
        let block = {
            let mut b = self.buf;
            b[56..64].copy_from_slice(&bits.to_be_bytes());
            b
        };
        self.block(&block);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.state[i].to_be_bytes());
        }
        out
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex_of(&h.finalize())
}

/// The raw 32-byte digest, for a container header that carries bytes rather
/// than hex (design §2.1's `AEFCORP1`).
fn sha256_raw(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

fn hex_of(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Deterministic pseudorandom bytes — an xorshift64* stream, so the payload is
/// the same on every run and a `GET` that comes back byte-identical is a
/// bit-exactness assertion (Rule D), not a coincidence.
fn pseudorandom(n: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut x = seed | 1;
    while out.len() < n {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let v = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.truncate(n);
    out
}

impl ResidentConn {
    /// Write raw bytes (a `DATA`/`PUT` payload). Separate from [`Self::send`]
    /// because a payload is not a line and must not be `&str`.
    fn send_bytes(&mut self, data: &[u8]) -> Result<(), String> {
        use std::io::Write;
        self.stream
            .write_all(data)
            .map_err(|e| format!("write {} bytes: {e}", data.len()))?;
        self.stream.flush().map_err(|e| format!("flush: {e}"))
    }

    /// Read exactly `n` raw bytes, draining anything `read_line` over-read
    /// first — the same residual-buffer discipline the server itself keeps.
    fn read_exact(&mut self, n: usize, wait: Duration) -> Result<Vec<u8>, String> {
        let deadline = Instant::now() + wait;
        while self.pending.len() < n {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(format!(
                    "only {} of {n} payload bytes within {}s",
                    self.pending.len(),
                    wait.as_secs()
                ));
            }
            self.stream
                .set_read_timeout(Some(left.min(Duration::from_secs(5))))
                .map_err(|e| format!("set_read_timeout: {e}"))?;
            let mut buf = vec![0u8; 64 * 1024];
            match self.stream.read(&mut buf) {
                Ok(0) => return Err("the guest closed the connection mid-payload".to_string()),
                Ok(k) => self.pending.extend_from_slice(&buf[..k]),
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => return Err(format!("read: {e}")),
            }
        }
        Ok(self.pending.drain(..n).collect())
    }

    /// Read a `DATA <len> <sha16>\n<bytes>END\n` frame (design §1.1) and hand
    /// back the payload, having checked both halves of the header against it.
    fn read_data_frame(&mut self, wait: Duration, bulk: Duration) -> Result<Vec<u8>, String> {
        let head = self.read_line(wait)?;
        let mut it = head.split_whitespace();
        match (it.next(), it.next(), it.next(), it.next()) {
            (Some("DATA"), Some(len), Some(sha16), None) => {
                let len: usize = len
                    .parse()
                    .map_err(|_| format!("DATA length {len:?} does not parse"))?;
                let body = self.read_exact(len, bulk)?;
                let tail = self.read_exact(4, wait)?;
                if tail != b"END\n" {
                    return Err(format!("DATA frame did not end in END: {tail:?}"));
                }
                let got = sha256_hex(&body);
                if !got.starts_with(sha16) {
                    return Err(format!(
                        "DATA header sha16={sha16} but the payload hashes to {got}"
                    ));
                }
                Ok(body)
            }
            _ => Err(format!("expected a DATA frame, got {head:?}")),
        }
    }
}

/// One request/response step, printed as it goes so a failing gate says which
/// verb failed rather than only that something did.
fn files_step(
    conn: &mut ResidentConn,
    hard: Instant,
    cmd: &str,
    want: &str,
) -> Result<String, String> {
    conn.send(&format!("{cmd}\n"))?;
    let line = conn.read_line(left(hard, FILES_IO_S)?)?;
    if line == want || (want.ends_with('*') && line.starts_with(&want[..want.len() - 1])) {
        println!("   ok   {cmd}  ->  {line}");
        Ok(line)
    } else {
        Err(format!("{cmd} answered {line:?}, expected {want:?}"))
    }
}

/// Drive one `PUT`: header, `SEND`, payload, trailer, answer.
///
/// `trailer` is what goes after the payload — `b"END\n"` for a well-formed
/// frame, and something else for design §1.4's `bad-frame` case.
fn files_put(
    conn: &mut ResidentConn,
    hard: Instant,
    name: &str,
    payload: &[u8],
    declared_sha: &str,
    trailer: &[u8],
) -> Result<String, String> {
    conn.send(&format!("PUT {name} {} {declared_sha}\n", payload.len()))?;
    let first = conn.read_line(left(hard, FILES_IO_S)?)?;
    if first != "SEND" {
        // A refusal before `SEND` is a legitimate answer (§1.4: nothing
        // written, connection kept), so it is handed back rather than treated
        // as a protocol error — the caller decides whether it wanted one.
        return Ok(first);
    }
    conn.send_bytes(payload)?;
    conn.send_bytes(trailer)?;
    conn.read_line(left(hard, FILES_BULK_S)?)
}

/// `SHA <NAME>` → the 64-hex digest the guest computed over its own copy.
fn files_sha(conn: &mut ResidentConn, hard: Instant, name: &str) -> Result<(u64, String), String> {
    conn.send(&format!("SHA {name}\n"))?;
    let line = conn.read_line(left(hard, FILES_BULK_S)?)?;
    let mut it = line.split_whitespace();
    match (it.next(), it.next(), it.next(), it.next(), it.next()) {
        (Some("SHA"), Some(got), Some(size), Some(hex), None) if got == name => {
            let size: u64 = size
                .parse()
                .map_err(|_| format!("SHA size {size:?} does not parse"))?;
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(format!("SHA digest {hex:?} is not 64 hex"));
            }
            Ok((size, hex.to_string()))
        }
        _ => Err(format!("SHA {name} answered {line:?}")),
    }
}

fn files_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    let host_port = {
        let probe = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("cannot pick a host port: {e}"))?;
        probe
            .local_addr()
            .map_err(|e| format!("cannot read the probe address: {e}"))?
            .port()
    };

    // `TOKEN` is staged so the gate can prove design §1.2's rule that a token
    // gates reads too. Every verb below therefore runs after an `AUTH`.
    let job_txt = format!(
        "# staged by `cargo xtask files-test` — AEFINITY OS phase 4 gate\n\
         MODE resident\n\
         NET static {NET_GUEST_CIDR} {NET_HOST_IP}\n\
         LISTEN {RESIDENT_GUEST_PORT}\n\
         TOKEN {FILES_TOKEN}\n"
    );

    let esp = stage("files-test", ci, debug, Some(&job_txt))?;

    // Files a previous run left in the mirror would be inside the vvfat image
    // this run boots, and a 64 MiB stray would eat the volume the gate needs.
    for stray in [
        "TEST.BIN",
        "BIG.BIN",
        "BAD.BIN",
        "STAGE.PRT",
        "CURRENT.TXT",
        // The A/B halves a previous run's pointer swap left behind. A stale
        // `MODEL.NEW` with a stale `CURRENT.TXT` would boot this run pointed
        // at the previous run's bytes.
        "MODEL.NEW",
        "EMBED.NEW",
        "VOCAB.NEW",
        // `files-soak` shares this ESP, and vvfat commits its writes to the
        // host mirror. Its files are inert here, but a gate that boots with
        // another gate's leftovers is not booting what it staged.
        "SOAK1.BIN",
        "SOAK2.BIN",
        "SOAK3.BIN",
    ] {
        if let Some(p) = find_ci(&esp.dir, stray) {
            let _ = fs::remove_file(p);
        }
    }

    // The host's own digest of the staged MODEL.SAF, which the guest's
    // `SHA MODEL.SAF` has to equal.
    let model_path = find_ci(&esp.dir, "MODEL.SAF")
        .ok_or_else(|| format!("no MODEL.SAF staged under {}", esp.dir.display()))?;
    let model_bytes =
        fs::read(&model_path).map_err(|e| format!("reading {}: {e}", model_path.display()))?;
    let model_sha = sha256_hex(&model_bytes);
    let model_len = model_bytes.len() as u64;
    println!("[3.5/4] host sha256 of the staged MODEL.SAF: {model_sha}");
    match Command::new("sha256sum").arg(&model_path).output() {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let want = text.split_whitespace().next().unwrap_or("");
            if want == model_sha {
                println!("        cross-checked against sha256sum ✓");
            } else {
                return Err(format!(
                    "the harness sha256 ({model_sha}) disagrees with sha256sum ({want}) — \
                     the harness is wrong, not the guest"
                ));
            }
        }
        _ => println!("        (sha256sum not available; harness digest uncross-checked)"),
    }

    let extra = vec![
        "-netdev".to_string(),
        format!("user,id=n0,hostfwd=tcp:127.0.0.1:{host_port}-:{RESIDENT_GUEST_PORT}"),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];

    println!("[4/4] booting under OVMF with a virtio NIC and a host forward");
    let (qemu_bin, mut cmd) = qemu_command(&esp, &extra);
    println!("      $ {qemu_bin} -machine q35,accel=tcg -cpu max -m 2048 ...");
    println!();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86."))?;

    let hard = Instant::now() + Duration::from_secs(FILES_TIMEOUT_S);
    let run = files_exchange(
        &mut child,
        host_port,
        hard,
        &model_sha,
        model_len,
        &model_bytes,
    );

    let outcome = match &run {
        Ok(_) => wait_for_exit(&mut child, Duration::from_secs(FILES_EXIT_S)),
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            Outcome::TimedOut
        }
    };

    println!();
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (RESIDENT/JOB/RELOAD lines) ----");
        for line in text.lines() {
            if line.contains("RESIDENT:") || line.contains("RELOAD:") || line.contains("JOB:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    let mut failures: Vec<String> = match run {
        Ok(f) => f,
        Err(e) => {
            eprintln!("== FAIL == {e}");
            eprintln!("   The BOOTLOG lines above are the guest's own account of how far it got.");
            return Ok(ExitCode::from(1));
        }
    };

    match outcome {
        Outcome::Exited(c) => {
            println!("   ok   QEMU exited {c} (guest ResetSystem under -no-reboot)")
        }
        Outcome::Signalled => {
            failures.push("QEMU terminated by a signal with no exit code".to_string())
        }
        Outcome::TimedOut => failures.push(format!(
            "QEMU did not exit within {FILES_EXIT_S}s of the guest answering BYE to REBOOT"
        )),
    }

    println!();
    if failures.is_empty() {
        println!(
            "== PASS == the file plane served AUTH/LS/STAT/SHA/GET/PUT/RM/RELOAD/HEALTH/RUNID,"
        );
        println!("           refused every malformed case by its §1.3 slug, and reset.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// Drive design §1.2's verbs end to end against the running guest.
///
/// **Every assertion is on the guest's own `LS`/`SHA`/`STAT`, never on the
/// host mirror** (design §7): QEMU's `fat:rw:` commits a guest write but not a
/// guest unlink, so the staged directory cannot answer "is it gone".
///
/// Returns the list of assertion failures — an empty list is a pass. A
/// protocol-level error (a connection that dies, a reply that never comes) is
/// an `Err` instead, because from there the remaining steps mean nothing.
fn files_exchange(
    child: &mut Child,
    host_port: u16,
    hard: Instant,
    model_sha: &str,
    model_len: u64,
    model_bytes: &[u8],
) -> Result<Vec<String>, String> {
    let (mut c, banner) = resident_ready(child, host_port, FILES_READY_S, hard)?;
    let mut fail: Vec<String> = Vec::new();
    println!("   ok   banner {banner}");
    // Rule A: `env=vm` in the banner is the same statement the record makes.
    if !banner.contains(" env=vm ") {
        fail.push(format!(
            "the READY banner does not carry env=vm: {banner:?}"
        ));
    }
    if !banner.contains(" caps=") {
        fail.push(format!("the READY banner does not carry caps=: {banner:?}"));
    }

    // ---- §1.2: a TOKEN gates reads, not only writes ----------------------
    files_step(&mut c, hard, "GET BOOTLOG.TXT", "ERR auth")?;
    files_step(&mut c, hard, "PING", "PONG")?;
    files_step(
        &mut c,
        hard,
        &format!("AUTH {}", "f".repeat(64)),
        "ERR auth",
    )?;
    files_step(&mut c, hard, &format!("AUTH {FILES_TOKEN}"), "OK")?;

    // ---- LS --------------------------------------------------------------
    c.send("LS\n")?;
    let head = c.read_line(left(hard, FILES_IO_S)?)?;
    let mut it = head.split_whitespace();
    let n: usize = match (it.next(), it.next(), it.next(), it.next()) {
        (Some("LS"), Some(n), Some(state), None) if state == "ok" || state == "truncated" => n
            .parse()
            .map_err(|_| format!("LS count {n:?} does not parse"))?,
        _ => return Err(format!("LS header was {head:?}")),
    };
    let mut names: Vec<String> = Vec::new();
    for _ in 0..n {
        let line = c.read_line(left(hard, FILES_IO_S)?)?;
        names.push(
            line.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string(),
        );
    }
    let end = c.read_line(left(hard, FILES_IO_S)?)?;
    if end != "END" {
        return Err(format!("LS did not end in END: {end:?}"));
    }
    println!("   ok   LS  ->  {head}  [{}]", names.join(" "));
    for want in ["MODEL.SAF", "EMBED.BIN", "VOCAB.BIN", "JOB.TXT"] {
        if !names.iter().any(|x| x == want) {
            fail.push(format!("LS does not list {want}"));
        }
    }

    // ---- STAT / SHA of MODEL.SAF against the host's own digest ------------
    files_step(
        &mut c,
        hard,
        "STAT MODEL.SAF",
        &format!("STAT MODEL.SAF {model_len}"),
    )?;
    let (size, hex) = files_sha(&mut c, hard, "MODEL.SAF")?;
    if size != model_len || hex != model_sha {
        fail.push(format!(
            "SHA MODEL.SAF = {size}/{hex}; the host staged {model_len}/{model_sha}"
        ));
    } else {
        println!("   ok   SHA MODEL.SAF matches the host's digest of the staged file");
    }

    // ---- GET of a file the guest wrote itself ----------------------------
    c.send("GET BOOTLOG.TXT\n")?;
    let log = c.read_data_frame(left(hard, FILES_IO_S)?, left(hard, FILES_BULK_S)?)?;
    println!(
        "   ok   GET BOOTLOG.TXT  ->  DATA frame of {} bytes, header sha16 matches",
        log.len()
    );

    // ---- PUT 256 KiB, then GET it back byte-identical (Rule D) -----------
    let small = pseudorandom(FILES_SMALL_BYTES, 0x5eed_0001);
    let small_sha = sha256_hex(&small);
    let ans = files_put(&mut c, hard, "TEST.BIN", &small, &small_sha, b"END\n")?;
    let want = format!("OK TEST.BIN {} {}", small.len(), &small_sha[..16]);
    if ans != want {
        return Err(format!("PUT TEST.BIN answered {ans:?}, expected {want:?}"));
    }
    println!("   ok   PUT TEST.BIN  ->  {ans}");
    c.send("GET TEST.BIN\n")?;
    let back = c.read_data_frame(left(hard, FILES_IO_S)?, left(hard, FILES_BULK_S)?)?;
    if back == small {
        println!(
            "   ok   GET TEST.BIN returned {} byte-identical bytes",
            back.len()
        );
    } else {
        fail.push(format!(
            "GET TEST.BIN returned {} bytes that are not byte-identical to what was PUT",
            back.len()
        ));
    }

    // ---- PUT 64 MiB, then SHA it (design §7's real case) -----------------
    let big = pseudorandom(FILES_BIG_BYTES, 0x5eed_0002);
    let big_sha = sha256_hex(&big);
    println!(
        "   ..   PUT BIG.BIN ({} bytes) — this is the slow one",
        big.len()
    );
    let ans = files_put(&mut c, hard, "BIG.BIN", &big, &big_sha, b"END\n")?;
    let want = format!("OK BIG.BIN {} {}", big.len(), &big_sha[..16]);
    if ans != want {
        return Err(format!("PUT BIG.BIN answered {ans:?}, expected {want:?}"));
    }
    println!("   ok   PUT BIG.BIN  ->  {ans}");
    let (bsize, bhex) = files_sha(&mut c, hard, "BIG.BIN")?;
    if bsize != big.len() as u64 || bhex != big_sha {
        fail.push(format!(
            "SHA BIG.BIN = {bsize}/{bhex}; the harness sent {}/{big_sha}",
            big.len()
        ));
    } else {
        println!("   ok   SHA BIG.BIN matches what the harness sent");
    }
    drop(big);

    // ---- §1.4: a wrong declared digest ------------------------------------
    let other = pseudorandom(FILES_SMALL_BYTES, 0x5eed_0003);
    let wrong = "0".repeat(64);
    let ans = files_put(&mut c, hard, "TEST.BIN", &other, &wrong, b"END\n")?;
    if ans != "ERR digest-mismatch" {
        fail.push(format!(
            "a PUT with a wrong declared digest answered {ans:?}, expected ERR digest-mismatch"
        ));
    } else {
        println!("   ok   PUT with a wrong digest  ->  {ans}");
    }
    // §1.4: the target is untouched and the stage is gone.
    let (_, still) = files_sha(&mut c, hard, "TEST.BIN")?;
    if still != small_sha {
        fail.push(format!(
            "after a rejected PUT, SHA TEST.BIN is {still}; it should still be {small_sha}"
        ));
    } else {
        println!("   ok   TEST.BIN still holds the bytes of the PUT that succeeded");
    }

    // ---- §1.4: a payload whose trailer is not END\n -----------------------
    let ans = files_put(
        &mut c,
        hard,
        "BAD.BIN",
        &other,
        &sha256_hex(&other),
        b"XXXX",
    )?;
    if ans != "ERR bad-frame" {
        fail.push(format!(
            "a PUT with a wrong trailer answered {ans:?}, expected ERR bad-frame"
        ));
    } else {
        println!("   ok   PUT with a wrong trailer  ->  {ans}");
    }

    // ---- §1.3: names, protection, lengths ---------------------------------
    files_step(&mut c, hard, "PUT ../X 4 00", "ERR bad-name")?;
    files_step(
        &mut c,
        hard,
        &format!("PUT {FILES_LONG_NAME} 4 {}", "0".repeat(64)),
        "ERR bad-name",
    )?;
    files_step(
        &mut c,
        hard,
        &format!("PUT BOOTLOG.TXT 4 {}", "0".repeat(64)),
        "ERR protected",
    )?;
    files_step(
        &mut c,
        hard,
        &format!("PUT HUGE.BIN {FILES_OVER_LEN} {}", "0".repeat(64)),
        "ERR bad-len",
    )?;

    // ---- §8: the A/B halves are not names a client may touch --------------
    //
    // `MODEL.NEW` and friends are internal: the protocol exposes `MODEL.SAF`
    // and the box decides which half the bytes land on. A client that could
    // name a half could delete the one `CURRENT.TXT` designates, and the
    // pointer's canonical-name fallback would then answer every read verb out
    // of the other, stale half reporting ordinary success. Both halves are
    // refused, always — the inactive one too, because "you may delete this
    // one but not that one, and which is which depends on how many times you
    // have uploaded" is a footgun with no use case behind it.
    for half in ["MODEL.NEW", "EMBED.NEW", "VOCAB.NEW"] {
        files_step(
            &mut c,
            hard,
            &format!("PUT {half} 4 {}", "0".repeat(64)),
            "ERR protected",
        )?;
        files_step(&mut c, hard, &format!("RM {half}"), "ERR protected")?;
    }

    // ---- §8: a PUT of an artifact swaps the pointer ------------------------
    //
    // The same bytes go back up, so the digests do not move and the `RELOAD`
    // assertion further down still has something to compare. What *does* move
    // is `CURRENT.TXT`: after this, `model=MODEL.NEW`, and `MODEL.NEW` is the
    // live model. So the `RM` below is a delete of the file the box would
    // boot from, and it must still be refused.
    println!("   ..   PUT MODEL.SAF ({model_len} bytes) — the §8 pointer swap");
    let ans = files_put(&mut c, hard, "MODEL.SAF", model_bytes, model_sha, b"END\n")?;
    let want = format!("OK MODEL.SAF {model_len} {}", &model_sha[..16]);
    if ans != want {
        return Err(format!("PUT MODEL.SAF answered {ans:?}, expected {want:?}"));
    }
    println!("   ok   PUT MODEL.SAF  ->  {ans}");
    files_step(&mut c, hard, "RM MODEL.NEW", "ERR protected")?;
    println!("   ok   RM of the now-LIVE half is still refused");

    // The blocker's shape in one cycle: a commit-time readback that verified,
    // then a fresh guest-side `SHA` of the same file. They must agree.
    let (size, hex) = files_sha(&mut c, hard, "MODEL.SAF")?;
    if size != model_len || hex != model_sha {
        fail.push(format!(
            "after the pointer swap, SHA MODEL.SAF = {size}/{hex}; the same bytes went up \
             and the commit-time readback verified, so it must still be {model_len}/{model_sha}"
        ));
    } else {
        println!("   ok   SHA MODEL.SAF after the swap still matches the host's digest");
    }
    files_step(
        &mut c,
        hard,
        "STAT MODEL.SAF",
        &format!("STAT MODEL.SAF {model_len}"),
    )?;

    // ---- LS is free of strays, HEALTH says parts=0 ------------------------
    c.send("LS\n")?;
    let head = c.read_line(left(hard, FILES_IO_S)?)?;
    let n: usize = head
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("LS header was {head:?}"))?;
    let mut after: Vec<String> = Vec::new();
    for _ in 0..n {
        let line = c.read_line(left(hard, FILES_IO_S)?)?;
        after.push(
            line.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string(),
        );
    }
    let end = c.read_line(left(hard, FILES_IO_S)?)?;
    if end != "END" {
        return Err(format!("LS did not end in END: {end:?}"));
    }
    println!("   ok   LS after the failures  ->  [{}]", after.join(" "));
    for stray in ["STAGE.PRT", "BAD.BIN", "HUGE.BIN"] {
        if after.iter().any(|x| x == stray) {
            fail.push(format!("LS shows a stray {stray} after an aborted PUT"));
        }
    }

    c.send("HEALTH\n")?;
    let health = c.read_line(left(hard, FILES_IO_S)?)?;
    println!("   ok   HEALTH  ->  {health}");
    for key in [
        "up=",
        "served=",
        "last=",
        "wd=",
        "heapfree=",
        "model=",
        "reloads=",
        "parts=",
        "degraded=",
        "env=",
    ] {
        if !health.contains(key) {
            fail.push(format!("HEALTH has no {key} field: {health:?}"));
        }
    }
    // §8: `degraded=pointer` is how the canonical-name fallback stops being
    // silent. Nothing here has deleted a half out from under `CURRENT.TXT`,
    // so it must be `none` — and if it is not, every digest above was read
    // out of the wrong half of an A/B pair.
    if !health.contains("degraded=none") {
        fail.push(format!(
            "HEALTH does not say degraded=none: {health:?}. The pointer designates an \
             artifact file that is not on the volume."
        ));
    }
    if !health.contains("parts=0") {
        fail.push(format!(
            "HEALTH does not say parts=0 after the aborted PUTs: {health:?}"
        ));
    }
    if !health.contains("env=vm") {
        fail.push(format!("HEALTH does not carry env=vm: {health:?}"));
    }
    let health_model = health
        .split_whitespace()
        .find_map(|t| t.strip_prefix("model="))
        .unwrap_or_default()
        .to_string();

    // ---- RM, then STAT says it is gone (guest-side, never the mirror) -----
    files_step(&mut c, hard, "RM TEST.BIN", "OK TEST.BIN")?;
    files_step(&mut c, hard, "STAT TEST.BIN", "ERR not-found")?;
    files_step(&mut c, hard, "RM BIG.BIN", "OK BIG.BIN")?;

    // ---- RUNID: NEW, a job, then REPLAY of the same id --------------------
    files_step(&mut c, hard, "RUNID files-gate-1", "NEW")?;
    c.send(&resident_job_block(4, None, "The capital of France is"))?;
    let first = c.read_result(hard)?;
    println!();
    println!("---- RESULT (RUNID files-gate-1) ----");
    print!("{first}");
    println!("---- end ----");
    for (k, want) in [
        ("verdict", "OK"),
        ("env", "vm"),
        ("run_id", "files-gate-1"),
        ("replay", "false"),
    ] {
        match record_value(&first, k) {
            Some(v) if v == want => println!("   ok   record {k}={v}"),
            Some(v) => fail.push(format!("record {k}={v}, expected {want}")),
            None => fail.push(format!("record has no {k} key")),
        }
    }
    for k in [
        "artifacts",
        "model_sha",
        "reloads",
        "uptime_s",
        "served",
        "files",
        "merge_key",
    ] {
        match record_value(&first, k) {
            Some(v) if !v.is_empty() => println!("   ok   record carries {k}={v}"),
            _ => fail.push(format!("record has no {k} key")),
        }
    }
    if let Some(mk) = record_value(&first, "merge_key") {
        if mk.len() != 16 || !mk.bytes().all(|b| b.is_ascii_hexdigit()) {
            fail.push(format!("merge_key={mk} is not 16 hex characters"));
        }
    }
    let merge_key = record_value(&first, "merge_key")
        .unwrap_or_default()
        .to_string();

    files_step(&mut c, hard, "RUNID files-gate-1", "REPLAY")?;
    c.send(&resident_job_block(
        4,
        None,
        "This body must be drained, not run",
    ))?;
    let replayed = c.read_result(hard)?;
    match record_value(&replayed, "replay") {
        Some("true") => println!("   ok   the replayed record says replay=true"),
        other => fail.push(format!(
            "the replayed record says replay={other:?}, expected true"
        )),
    }
    if record_value(&replayed, "merge_key") == Some(merge_key.as_str()) {
        println!("   ok   the replayed record is the cached one (same merge_key)");
    } else {
        fail.push("the replayed record is not the cached one — the job was re-run".to_string());
    }

    // ---- RELOAD, then a job still runs ------------------------------------
    c.send("RELOAD\n")?;
    let line = c.read_line(left(hard, FILES_IO_S)?)?;
    if line != "RELOADING" {
        return Err(format!("RELOAD answered {line:?}, expected RELOADING"));
    }
    let done = c.read_line(left(hard, FILES_BULK_S)?)?;
    println!("   ok   RELOAD  ->  {done}");
    let mut it = done.split_whitespace();
    match (
        it.next(),
        it.next(),
        it.next(),
        it.next(),
        it.next(),
        it.next(),
    ) {
        (Some("OK"), Some("reload"), Some(m), Some(e), Some(v), None)
            if m.starts_with("model=") && e.starts_with("embed=") && v.starts_with("vocab=") =>
        {
            let m = m.trim_start_matches("model=");
            if m != health_model {
                fail.push(format!(
                    "RELOAD reports model={m} but HEALTH reported model={health_model}; \
                     nothing on the volume changed, so the digests must agree"
                ));
            } else {
                println!("   ok   RELOAD reports the same three digests the box already had");
            }
        }
        _ => fail.push(format!("RELOAD finished with {done:?}")),
    }
    c.send(&resident_job_block(4, None, "After the reload"))?;
    let after_reload = c.read_result(hard)?;
    match record_value(&after_reload, "verdict") {
        Some("OK") => println!("   ok   a job after RELOAD still runs (verdict=OK)"),
        other => fail.push(format!("the job after RELOAD ended verdict={other:?}")),
    }
    match record_value(&after_reload, "reloads") {
        Some("1") => println!("   ok   the record after RELOAD says reloads=1"),
        other => fail.push(format!(
            "after one RELOAD the record says reloads={other:?}"
        )),
    }

    // ---- REBOOT ------------------------------------------------------------
    c.send("REBOOT\n")?;
    let bye = c.read_line(left(hard, FILES_IO_S)?)?;
    if bye != "BYE" {
        return Err(format!("REBOOT answered {bye:?}, expected BYE"));
    }
    println!("   ok   REBOOT  ->  BYE");
    Ok(fail)
}

// ---------------------------------------------------------------------------
// files-soak — the phase 4 integrity gate under sustained churn
// ---------------------------------------------------------------------------

/// Default cycles. Fifteen because the failure this gate exists to catch was
/// first seen around the tenth PUT/RM cycle of one boot; a gate that stopped at
/// ten would have been green on the day the bug was found.
const SOAK_CYCLES: usize = 15;
/// Every `SOAK_RELOAD_EVERY`-th cycle also does a `RELOAD`, so the engine
/// rebuild path is inside the churn rather than beside it.
const SOAK_RELOAD_EVERY: usize = 5;
/// The three ordinary files the cycles create, delete and recreate. Three so a
/// name is always being reused while two others live, which is the directory
/// pattern that broke `vvfat`.
const SOAK_ORDINARY: [&str; 3] = ["SOAK1.BIN", "SOAK2.BIN", "SOAK3.BIN"];
/// Size of the FAT image the raw-image path formats, in bytes. §8 requires
/// ≥ 2× the artifact set resident at once; the staged M7 set is ~9 MB, so
/// 256 MiB is that with room for the soak files and no reason to be tighter.
const SOAK_IMG_BYTES: u64 = 256 * 1024 * 1024;
/// Bytes per sector for the formatted image.
const SOAK_SECTOR: u64 = 512;

/// `files-soak` — 15 PUT/RM/RELOAD cycles in one boot, with a fresh guest-side
/// `SHA` of every file present after every cycle.
///
/// The quick gates (`files-test`) prove each verb once. This one proves the
/// volume still holds what it said it held after the directory has been
/// churned, which is the only way the phase 4 file plane's central claim — a
/// commit-time readback digest means the bytes are there — can be believed
/// beyond a single transfer.
///
/// **The boot volume is a real FAT image by default** (`mformat` + `mcopy`,
/// `-drive format=raw`). `--vvfat` selects QEMU's `fat:rw:` directory mapping
/// instead, which is what the other gates use and what this gate was written
/// to indict: it fails this soak, and it fails it inside `block/vvfat.c`.
fn files_soak(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");
    let vvfat = flags.contains(&"--vvfat");
    let cycles: usize = env::var("AEGIS_SOAK_CYCLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SOAK_CYCLES);

    let host_port = {
        let probe = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("cannot pick a host port: {e}"))?;
        probe
            .local_addr()
            .map_err(|e| format!("cannot read the probe address: {e}"))?
            .port()
    };

    let job_txt = format!(
        "# staged by `cargo xtask files-soak` — AEFINITY OS phase 4 integrity gate\n\
         MODE resident\n\
         NET static {NET_GUEST_CIDR} {NET_HOST_IP}\n\
         LISTEN {RESIDENT_GUEST_PORT}\n\
         TOKEN {FILES_TOKEN}\n"
    );

    let esp = stage("files-soak", ci, debug, Some(&job_txt))?;

    // Anything a previous run left is inside the volume this run boots, and a
    // stale `CURRENT.TXT` next to a stale half would start the soak already
    // pointed at last run's bytes.
    let mut strays: Vec<String> = vec![
        "STAGE.PRT".into(),
        "CURRENT.TXT".into(),
        "MODEL.NEW".into(),
        "EMBED.NEW".into(),
        "VOCAB.NEW".into(),
        "TEST.BIN".into(),
        "BIG.BIN".into(),
    ];
    strays.extend(SOAK_ORDINARY.iter().map(|s| (*s).to_string()));
    for stray in &strays {
        if let Some(p) = find_ci(&esp.dir, stray) {
            let _ = fs::remove_file(p);
        }
    }

    // The host's own digest of each staged artifact. Every `SHA` the guest
    // answers with is checked against these, so "the file is intact" is a
    // statement about bytes the harness knows, not about self-consistency.
    let mut artifacts: Vec<(String, Vec<u8>, String)> = Vec::new();
    for name in BOOT_ASSETS {
        let p = find_ci(&esp.dir, name)
            .ok_or_else(|| format!("no {name} staged under {}", esp.dir.display()))?;
        let bytes = fs::read(&p).map_err(|e| format!("reading {}: {e}", p.display()))?;
        let sha = sha256_hex(&bytes);
        println!("       host sha256 {name:<10} {} bytes  {sha}", bytes.len());
        artifacts.push((name.to_string(), bytes, sha));
    }

    // `JOB.TXT` is the canary: nothing in the soak ever writes to it, so its
    // digest is the one that can only change if the volume corrupted a file
    // no verb touched.
    let job_path = find_ci(&esp.dir, "JOB.TXT")
        .ok_or_else(|| format!("no JOB.TXT staged under {}", esp.dir.display()))?;
    let job_bytes =
        fs::read(&job_path).map_err(|e| format!("reading {}: {e}", job_path.display()))?;
    let untouched = (
        "JOB.TXT".to_string(),
        job_bytes.len() as u64,
        sha256_hex(&job_bytes),
    );
    println!(
        "       host sha256 {:<10} {} bytes  {}",
        untouched.0, untouched.1, untouched.2
    );

    // ---- the boot volume ---------------------------------------------------
    let drive = if vvfat {
        println!();
        println!("      volume  : QEMU vvfat (fat:rw:) — the FRAGILE path, by request");
        format!("format=raw,file=fat:rw:{}", esp.dir.display())
    } else {
        let img = esp.root.join("target").join("soak.img");
        build_fat_image(&esp.dir, &img)?;
        println!(
            "      volume  : real FAT32 image {} (no vvfat)",
            img.display()
        );
        format!("format=raw,file={}", img.display())
    };

    let extra = vec![
        "-netdev".to_string(),
        format!("user,id=n0,hostfwd=tcp:127.0.0.1:{host_port}-:{RESIDENT_GUEST_PORT}"),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];

    println!("[4/4] booting for {cycles} PUT/RM/RELOAD cycles");
    println!();
    let (qemu_bin, mut cmd) = qemu_command_on(&esp, &extra, &drive);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86."))?;

    let hard = Instant::now() + Duration::from_secs(FILES_TIMEOUT_S);
    let run = soak_exchange(&mut child, host_port, hard, &artifacts, &untouched, cycles);

    let outcome = match &run {
        Ok(_) => wait_for_exit(&mut child, Duration::from_secs(FILES_EXIT_S)),
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            Outcome::TimedOut
        }
    };

    println!();
    println!("---- BOOTLOG.TXT (RESIDENT/RELOAD/ERROR lines) ----");
    for line in soak_bootlog(&esp, vvfat).lines() {
        if line.contains("RESIDENT:") || line.contains("RELOAD:") || line.contains("ERROR") {
            println!("{line}");
        }
    }
    println!("---- end ----");
    println!();

    let mut failures: Vec<String> = match run {
        Ok(f) => f,
        Err(e) => {
            eprintln!("== FAIL == {e}");
            if vvfat {
                eprintln!();
                eprintln!(
                    "   This is the --vvfat run. QEMU's vvfat rebuilds its directory mapping as\n   \
                     the guest mutates it and is documented-fragile under create/rename/delete\n   \
                     churn; `block/vvfat.c:1901: get_cluster_count_for_direntry: Assertion\n   \
                     'mapping->mode & MODE_DELETED' failed` is the emulator, not the guest.\n   \
                     Re-run WITHOUT --vvfat (a real FAT image) before believing anything here\n   \
                     about the file plane."
                );
            }
            return Ok(ExitCode::from(1));
        }
    };

    match outcome {
        Outcome::Exited(c) => println!("   ok   QEMU exited {c} (guest ResetSystem)"),
        Outcome::Signalled => {
            failures.push("QEMU terminated by a signal with no exit code".to_string())
        }
        Outcome::TimedOut => failures.push(format!(
            "QEMU did not exit within {FILES_EXIT_S}s of the guest answering BYE"
        )),
    }

    println!();
    if failures.is_empty() {
        println!("== PASS == {cycles} PUT/RM/RELOAD cycles in one boot, and after every one of");
        println!("           them a fresh guest-side SHA of every file present matched the");
        println!("           host's digest of the bytes that were sent.");
        if !vvfat {
            println!();
            println!("           Volume: a real FAT32 image, not vvfat. The same soak under");
            println!("           --vvfat is a known emulator failure (AEFINITY_OS_STATUS.md §9)");
            println!("           and says nothing about the guest.");
        }
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// Read `BOOTLOG.TXT` back off whichever volume the run used.
///
/// vvfat commits guest writes to the host mirror, so the staged directory has
/// it. A raw image has to be opened with `mcopy`.
fn soak_bootlog(esp: &Esp, vvfat: bool) -> String {
    if vvfat {
        return find_ci(&esp.dir, "BOOTLOG.TXT")
            .and_then(|p| fs::read_to_string(p).ok())
            .unwrap_or_default();
    }
    let img = esp.root.join("target").join("soak.img");
    match Command::new("mcopy")
        .args(["-n", "-i"])
        .arg(&img)
        .args(["::BOOTLOG.TXT", "-"])
        .output()
    {
        Ok(out) => String::from_utf8_lossy(&out.stdout).into_owned(),
        Err(e) => format!("(mcopy could not read BOOTLOG.TXT out of the image: {e})"),
    }
}

/// Format a real FAT32 image and copy the staged ESP into it.
///
/// `mformat`, not `mkfs.fat`: mtools writes the filesystem into a plain file
/// with no loop device and no privileges, which is the whole reason this path
/// is usable in a gate. The image carries no partition table — EDK2's FAT
/// driver binds to the block device directly — so `mcopy` and OVMF are looking
/// at the same volume with no offset arithmetic between them.
fn build_fat_image(esp_dir: &Path, img: &Path) -> Result<(), String> {
    let sectors = SOAK_IMG_BYTES / SOAK_SECTOR;
    let _ = fs::remove_file(img);
    if let Some(parent) = img.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    println!(
        "[3.5/4] formatting a real FAT32 volume: {} ({} MiB)",
        img.display(),
        SOAK_IMG_BYTES / (1024 * 1024)
    );
    // -C creates the file at -T sectors; -F forces FAT32; the geometry is the
    // conventional 64 heads / 32 sectors so the total is a whole number of
    // cylinders and mformat does not have to guess.
    let out = Command::new("mformat")
        .args(["-i"])
        .arg(img)
        .args([
            "-C",
            "-T",
            &sectors.to_string(),
            "-h",
            "64",
            "-s",
            "32",
            "-F",
            "-v",
            "ALICEUEFI",
            "::",
        ])
        .output()
        .map_err(|e| format!("mformat: {e}. Install mtools."))?;
    if !out.status.success() {
        return Err(format!(
            "mformat failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    // Copy ONLY what this gate staged, by name. `target/esp/` is shared with
    // the vvfat gates, and vvfat commits guest writes but not guest unlinks,
    // so it leaves strays (e.g. a lowercase `vocab.bin`) that collide
    // case-insensitively under mcopy and break this gate with an empty
    // stderr. A whitelist keeps the raw-image gate hermetic. `-s` recurses,
    // so EFI/BOOT/BOOTX64.EFI lands where the firmware looks for it.
    let mut entries: Vec<PathBuf> = Vec::new();
    for name in BOOT_ASSETS.iter().copied().chain(["JOB.TXT", "EFI"]) {
        if let Some(p) = find_ci(esp_dir, name) {
            entries.push(p);
        }
    }
    entries.sort();
    if entries.is_empty() {
        return Err(format!("{} is empty — nothing to boot", esp_dir.display()));
    }
    let mut copy = Command::new("mcopy");
    copy.args(["-s", "-i"]).arg(img);
    for e in &entries {
        copy.arg(e);
    }
    copy.arg("::");
    let out = copy
        .output()
        .map_err(|e| format!("mcopy: {e}. Install mtools."))?;
    if !out.status.success() {
        return Err(format!(
            "mcopy into {} failed: {}",
            img.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

/// What the harness believes is on the volume: client-visible name → (len, sha).
type Expect = Vec<(String, u64, String)>;

/// Drive the cycles and verify after every one.
///
/// Returns the assertion failures. A digest mismatch is **not** one of them —
/// it is an `Err`, because once the volume has answered with bytes it was not
/// given there is nothing informative left to assert and the cycle number is
/// the finding.
fn soak_exchange(
    child: &mut Child,
    host_port: u16,
    hard: Instant,
    artifacts: &[(String, Vec<u8>, String)],
    untouched: &(String, u64, String),
    cycles: usize,
) -> Result<Vec<String>, String> {
    let (mut c, banner) = resident_ready(child, host_port, FILES_READY_S, hard)?;
    let mut fail: Vec<String> = Vec::new();
    println!("   ok   banner {banner}");
    files_step(&mut c, hard, &format!("AUTH {FILES_TOKEN}"), "OK")?;

    // The artifacts are on the volume already, at their canonical names.
    let mut expect: Expect = artifacts
        .iter()
        .map(|(n, b, s)| (n.clone(), b.len() as u64, s.clone()))
        .collect();
    expect.push(untouched.clone());
    // Cycle 0: the baseline. If this does not hold, nothing after it means
    // anything and the churn is not what broke it.
    soak_verify(&mut c, hard, &expect, 0)?;

    let mut reload_digests: Option<String> = None;

    for cycle in 1..=cycles {
        println!();
        println!("== cycle {cycle}/{cycles} ==");

        // ---- an ordinary file, at a size that is not a whole number of
        // ---- XFER_CHUNKs, so the tail chunk moves from cycle to cycle.
        let name = SOAK_ORDINARY[cycle % SOAK_ORDINARY.len()];
        let len = 64 * 1024 * (1 + (cycle * 7) % 40) + cycle * 37;
        let body = pseudorandom(len, 0x50a4_0000 + cycle as u64);
        let sha = sha256_hex(&body);
        let ans = files_put(&mut c, hard, name, &body, &sha, b"END\n")?;
        let want = format!("OK {name} {len} {}", &sha[..16]);
        if ans != want {
            return Err(format!(
                "cycle {cycle}: PUT {name} answered {ans:?}, expected {want:?}"
            ));
        }
        println!("   ok   PUT {name} ({len} bytes)  ->  {ans}");
        soak_remember(&mut expect, name, len as u64, &sha);

        // ---- and delete the one two cycles back, so entries are being
        // ---- created and released at the same time.
        if cycle >= 2 {
            let old = SOAK_ORDINARY[(cycle - 2) % SOAK_ORDINARY.len()];
            if expect.iter().any(|(n, _, _)| n == old) {
                files_step(&mut c, hard, &format!("RM {old}"), &format!("OK {old}"))?;
                files_step(&mut c, hard, &format!("STAT {old}"), "ERR not-found")?;
                expect.retain(|(n, _, _)| n != old);
            }
        }

        // ---- one artifact per cycle, rotating: the pointer swaps every time,
        // ---- so `CURRENT.TXT` is rewritten inside the churn too. The same
        // ---- bytes go up each time, so the expected digest never moves and a
        // ---- mismatch is unambiguous.
        let (aname, abytes, asha) = &artifacts[cycle % artifacts.len()];
        let ans = files_put(&mut c, hard, aname, abytes, asha, b"END\n")?;
        let want = format!("OK {aname} {} {}", abytes.len(), &asha[..16]);
        if ans != want {
            return Err(format!(
                "cycle {cycle}: PUT {aname} answered {ans:?}, expected {want:?}"
            ));
        }
        println!("   ok   PUT {aname} ({} bytes)  ->  {ans}", abytes.len());
        // §8: the half the pointer is not on is still not a name a client may
        // touch, and the half it IS on is the live model — both refused.
        let half = format!("{}.NEW", aname.split('.').next().unwrap_or(aname));
        files_step(&mut c, hard, &format!("RM {half}"), "ERR protected")?;

        // ---- an artifact RM, once, late: `remove_artifact` must clear the
        // ---- pointer BEFORE the bytes go, or the canonical-name fallback
        // ---- serves the stale other half and calls it success.
        if cycle == cycles.saturating_sub(1) {
            let (vname, vbytes, vsha) = &artifacts[2];
            files_step(&mut c, hard, &format!("RM {vname}"), &format!("OK {vname}"))?;
            files_step(&mut c, hard, &format!("STAT {vname}"), "ERR not-found")?;
            let vhalf = format!("{}.NEW", vname.split('.').next().unwrap_or(vname));
            files_step(&mut c, hard, &format!("STAT {vhalf}"), "ERR not-found")?;
            println!("   ok   RM {vname} took BOTH halves and left no orphan pointer");
            let ans = files_put(&mut c, hard, vname, vbytes, vsha, b"END\n")?;
            if !ans.starts_with(&format!("OK {vname} ")) {
                return Err(format!(
                    "cycle {cycle}: re-PUT of {vname} after its RM answered {ans:?}"
                ));
            }
            println!("   ok   re-PUT {vname}  ->  {ans}");
        }

        // ---- a RELOAD inside the churn, not beside it -----------------------
        if cycle % SOAK_RELOAD_EVERY == 0 {
            c.send("RELOAD\n")?;
            let line = c.read_line(left(hard, FILES_IO_S)?)?;
            if line != "RELOADING" {
                return Err(format!(
                    "cycle {cycle}: RELOAD answered {line:?}, expected RELOADING"
                ));
            }
            let done = c.read_line(left(hard, FILES_BULK_S)?)?;
            println!("   ok   RELOAD  ->  {done}");
            if !done.starts_with("OK reload ") {
                fail.push(format!("cycle {cycle}: RELOAD finished with {done:?}"));
            } else {
                let digests = done.trim_start_matches("OK reload ").to_string();
                match &reload_digests {
                    None => reload_digests = Some(digests),
                    Some(first) if *first == digests => {
                        println!("   ok   RELOAD reports the same three digests as the first one");
                    }
                    Some(first) => fail.push(format!(
                        "cycle {cycle}: RELOAD reports {digests}, the first RELOAD reported \
                         {first}. The same bytes were re-uploaded every cycle, so the engine \
                         has been rebuilt from something the harness never sent."
                    )),
                }
            }
        }

        // ---- and now the point of the whole gate ---------------------------
        soak_verify(&mut c, hard, &expect, cycle)?;

        let health = soak_health(&mut c, hard)?;
        if !health.contains("parts=0") {
            fail.push(format!(
                "cycle {cycle}: HEALTH says {health:?}, not parts=0"
            ));
        }
        if !health.contains("degraded=none") {
            fail.push(format!(
                "cycle {cycle}: HEALTH says {health:?}. degraded=pointer means CURRENT.TXT \
                 designates a file that is gone and every read above came from the other half."
            ));
        }
    }

    println!();
    c.send("REBOOT\n")?;
    let bye = c.read_line(left(hard, FILES_IO_S)?)?;
    if bye != "BYE" {
        return Err(format!("REBOOT answered {bye:?}, expected BYE"));
    }
    println!("   ok   REBOOT  ->  BYE");
    Ok(fail)
}

fn soak_health(c: &mut ResidentConn, hard: Instant) -> Result<String, String> {
    c.send("HEALTH\n")?;
    c.read_line(left(hard, FILES_IO_S)?)
}

/// Record (or replace) what a name is expected to hold.
fn soak_remember(expect: &mut Expect, name: &str, len: u64, sha: &str) {
    expect.retain(|(n, _, _)| n != name);
    expect.push((name.to_string(), len, sha.to_string()));
}

/// `SHA` every file the harness put there, plus every physical name `LS`
/// reports, and compare against the host's digests.
///
/// Both directions matter. Checking only what the harness expects would miss a
/// stray `STAGE.PRT`; checking only what `LS` says would miss a file that
/// vanished. The artifacts are queried by their **client-visible** name so the
/// query goes through §8's pointer — which is exactly the resolution a `RELOAD`
/// and a boot use, so a wrong answer here is a wrong model there.
fn soak_verify(
    c: &mut ResidentConn,
    hard: Instant,
    expect: &Expect,
    cycle: usize,
) -> Result<(), String> {
    c.send("LS\n")?;
    let head = c.read_line(left(hard, FILES_IO_S)?)?;
    let n: usize = head
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("cycle {cycle}: LS header was {head:?}"))?;
    let mut present: Vec<String> = Vec::new();
    for _ in 0..n {
        let line = c.read_line(left(hard, FILES_IO_S)?)?;
        present.push(
            line.split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string(),
        );
    }
    let end = c.read_line(left(hard, FILES_IO_S)?)?;
    if end != "END" {
        return Err(format!("cycle {cycle}: LS did not end in END: {end:?}"));
    }
    if present.iter().any(|x| x == "STAGE.PRT") {
        return Err(format!(
            "cycle {cycle}: LS shows a STAGE.PRT — a transfer committed and left its stage"
        ));
    }
    // No orphans: every LS entry must be a name this harness expects, a known
    // A/B half of one, or a server-owned file. A stray entry that is not
    // STAGE.PRT would otherwise survive every per-name SHA below unnoticed.
    for x in &present {
        let known = expect.iter().any(|(n, _, _)| n == x)
            || expect.iter().any(|(n, _, _)| {
                n.rsplit_once('.')
                    .is_some_and(|(stem, _)| format!("{stem}.NEW") == *x)
            })
            || [
                "BOOTLOG.TXT",
                "RESULT.TXT",
                "RESULT.WIP",
                "CURRENT.TXT",
                "JOB.TXT",
                "EFI",
                "BOOTX64.EFI",
            ]
            .contains(&x.as_str());
        if !known {
            return Err(format!(
                "cycle {cycle}: LS shows an orphan entry {x:?} that no verb should have left"
            ));
        }
    }

    let mut checked = 0usize;
    for (name, len, sha) in expect {
        let (got_len, got_sha) = files_sha(c, hard, name)?;
        if got_len != *len || &got_sha != sha {
            return Err(format!(
                "cycle {cycle}: SHA {name} = {got_len}/{got_sha}, but the harness sent \
                 {len}/{sha} and the commit-time readback verified it. The volume has \
                 changed underneath a file nobody wrote to."
            ));
        }
        checked += 1;
    }

    // The A/B halves, by physical name. Both halves hold the same bytes here
    // (the same artifact is re-uploaded every cycle), so either one differing
    // is corruption and not a version skew.
    for (name, len, sha) in expect {
        let Some(stem) = name
            .strip_suffix(".SAF")
            .or_else(|| name.strip_suffix(".BIN"))
        else {
            continue;
        };
        let half = format!("{stem}.NEW");
        if !present.contains(&half) {
            continue;
        }
        let (got_len, got_sha) = files_sha(c, hard, &half)?;
        if got_len != *len || &got_sha != sha {
            return Err(format!(
                "cycle {cycle}: SHA {half} = {got_len}/{got_sha}, expected {len}/{sha}. \
                 The inactive half of {name}'s A/B pair is not the bytes that were \
                 committed to it."
            ));
        }
        checked += 1;
    }

    println!(
        "   ok   cycle {cycle}: {checked} fresh SHA(s) all match  [{}]",
        present.join(" ")
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// `lab-test` — AEFINITY OS phase 5 (design §7)
// ---------------------------------------------------------------------------

/// Whole-run ceiling. `VERIFY GOOD.RCP` replays 64 decode steps through the
/// CIS-1 full-integer engine — scalar integer arithmetic with no SIMD
/// dispatch — under TCG, and `EVAL` runs three more scored positions over the
/// same path. Generous on purpose: this bounds a hang, it does not time
/// anything (Rule A).
const LAB_TIMEOUT_S: u64 = 10800;
/// The guest has to load the artifacts and bring the NIC up before it answers.
const LAB_READY_S: u64 = 1200;
/// One short request/response on an idle guest.
const LAB_IO_S: u64 = 240;
/// One `JOB` — the `RESULT` line after `RUNNING`.
const LAB_JOB_S: u64 = 5700;
/// After `BYE`, how long QEMU gets to exit.
const LAB_EXIT_S: u64 = 240;
/// Tokens in the staged corpus. `EVAL TINY.BIN 0:4` takes the first four, one
/// window (W = min(2048, max_position_embeddings) is far larger), and scores
/// three of them — the first token of a window is context, not a prediction.
const LAB_CORPUS_NTOK: usize = 8;
/// The receipt fixture. It is the repository's own witness v1 golden for the
/// M7 tinybit model, and `xtask stage` puts exactly those three artifacts on
/// the ESP — so the receipt's `model`/`embed`/`vocab` lines already name the
/// bytes the guest will be holding. Nothing is generated, and `tests/golden/`
/// is read, never written (CLAUDE.md Rule C).
const LAB_RECEIPT_GOLDEN: &str = "tests/golden/witness_v1_m7_once64.receipt";
/// The `BUDGET` every job in this gate carries. `run_job` refuses to *start* a
/// step once the budget is spent, so it has to cover a 64-step full-integer
/// replay plus an `EVAL` behind it on an emulated CPU. It bounds a hang; it
/// measures nothing (Rule A).
const LAB_BUDGET_S: u64 = 5400;
/// `MEMBW <mib>` the gate asks for. Small: the assertion is that a digest
/// comes back and that the bandwidth field is gated, not how fast it was.
const LAB_MEMBW_MIB: u32 = 16;
/// The `BUDGET` job 5 runs its `EVAL` under, chosen so the budget is spent
/// *inside* the step rather than before it.
///
/// `run_job` re-reads the clock at the top of every step and refuses to
/// **start** one whose budget is already gone, so the only way to reach
/// `lab::eval`'s own window-boundary check with nothing left is for the
/// step's setup to outlast the budget. Two bounds have to hold at once:
///
/// * the step is *entered* — only one `BOOTLOG` line separates `run_job`'s
///   clock read from the pre-step check, so the elapsed it sees is 0 and any
///   budget ≥ 1 lets the step start;
/// * the first window is *never reached* — the setup has to run past two
///   whole seconds, which is why job 5 evaluates [`LAB_STOP_CORPUS_NTOK`]
///   tokens' worth of container rather than `TINY.BIN`'s eight.
///
/// The firmware clock OVMF exposes ticks in whole seconds, so nothing finer
/// than a second can be relied on here. `BOOTLOG.TXT`'s
/// `JOB: eval setup ready in <ms> ms` line is the measured margin, written
/// down by the run itself: if that number ever falls near this budget on some
/// faster host, raise [`LAB_STOP_CORPUS_NTOK`], not this.
const LAB_STOP_BUDGET_S: u64 = 2;
/// Tokens in `STOP.BIN`, the corpus job 5 stops inside of — 128 MiB of
/// payload.
///
/// `load_corpus` streams the **whole** payload before the first window: one
/// sha256 over it and one bounds check per id, whatever `lo:hi` asks to
/// score. That pass is the only part of `EVAL`'s setup that scales with
/// anything the gate controls, so it is what puts the budget boundary inside
/// the step. `EVAL STOP.BIN 0:4` still scores only four tokens, so a run that
/// does *not* stop (the diagnosis for a too-small corpus) costs one window,
/// not thirty-three million.
///
/// Calibrated, not guessed: on this box under TCG a 16 MiB container measured
/// `eval setup ready in 1000 ms` and this one measures 5000 ms, against a
/// 2 s budget. That margin is what the gate has; the guest writes the number
/// into `BOOTLOG.TXT` on every run, so a host fast enough to shrink it says
/// so in the log before the assertion fails.
const LAB_STOP_CORPUS_NTOK: usize = 32 * 1024 * 1024;

/// An `AEFCORP1` header (design §2.1) over an already-built payload.
///
/// 64 bytes, all integers little-endian: magic, `version` = 1,
/// `token_width` = 4, `ntok`, `vocab_size`, one reserved zero word, then the
/// sha256 of the payload. One function so the two callers below cannot drift
/// into describing two different containers.
fn lab_corpus_header(vocab_size: u32, ntok: usize, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    out.extend_from_slice(b"AEFCORP1");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(ntok as u64).to_le_bytes());
    out.extend_from_slice(&vocab_size.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&sha256_raw(payload));
    out
}

/// Build an `AEFCORP1` container over `ids`; the payload is `ids` as u32 LE.
fn lab_corpus(vocab_size: u32, ids: &[u32]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(ids.len() * 4);
    for &id in ids {
        payload.extend_from_slice(&id.to_le_bytes());
    }
    let mut out = lab_corpus_header(vocab_size, ids.len(), &payload);
    out.extend_from_slice(&payload);
    out
}

/// The same container, written straight onto the volume by repeating `ids`
/// until it holds `ntok` tokens.
///
/// Streamed rather than returned: job 5's corpus is 128 MiB, and building it
/// through [`lab_corpus`] would hold three copies of that in host memory on a
/// 6 GB box to hand one of them back.
fn lab_corpus_stage(path: &Path, vocab_size: u32, ids: &[u32], ntok: usize) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("staging {}: {e}", path.display());
    let mut payload = Vec::with_capacity(ntok * 4);
    for i in 0..ntok {
        payload.extend_from_slice(&ids[i % ids.len()].to_le_bytes());
    }
    let header = lab_corpus_header(vocab_size, ntok, &payload);
    let f = fs::File::create(path).map_err(fail)?;
    let mut w = std::io::BufWriter::new(f);
    w.write_all(&header).map_err(fail)?;
    w.write_all(&payload).map_err(fail)?;
    w.flush().map_err(fail)
}

/// `vocab_size` out of a `MODEL.SAF`'s safetensors `__metadata__`.
///
/// The header is an 8-byte LE length followed by that many bytes of JSON, and
/// `aegis_config` inside it is a JSON *string* whose quotes are escaped — so
/// the value reads as `\"vocab_size\":8192` in the raw bytes. xtask has no
/// dependencies by design (`xtask/Cargo.toml`), so this scans for the key and
/// takes the digits that follow rather than pulling in a JSON parser for one
/// integer.
fn model_vocab_size(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 8 {
        return None;
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().ok()?) as usize;
    let head = bytes.get(8..8 + n)?;
    let at = find_subslice(head, b"vocab_size")? + b"vocab_size".len();
    let rest = &head[at..];
    let start = rest.iter().position(|b| b.is_ascii_digit())?;
    let end = start
        + rest[start..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
    core::str::from_utf8(&rest[start..end]).ok()?.parse().ok()
}

/// A `JOB … END` block for the wire out of a body fragment.
fn lab_job_block(body: &str) -> String {
    lab_job_block_budget(LAB_BUDGET_S, body)
}

/// The same, with a `BUDGET` of its own — for the one job that is *about* the
/// budget running out.
fn lab_job_block_budget(budget_s: u64, body: &str) -> String {
    format!("JOB\nBUDGET {budget_s}\n{body}END\n")
}

fn lab_test(flags: &[&str]) -> Result<ExitCode, String> {
    let ci = flags.contains(&"--ci");
    let debug = flags.contains(&"--debug");

    let host_port = {
        let probe = TcpListener::bind("127.0.0.1:0")
            .map_err(|e| format!("cannot pick a host port: {e}"))?;
        probe
            .local_addr()
            .map_err(|e| format!("cannot read the probe address: {e}"))?
            .port()
    };

    // No `TOKEN`: `files-test` already proves the auth gate, and a second copy
    // of that proof here would only lengthen a gate that is about phase 5.
    let job_txt = format!(
        "# staged by `cargo xtask lab-test` — AEFINITY OS phase 5 gate\n\
         MODE resident\n\
         NET static {NET_GUEST_CIDR} {NET_HOST_IP}\n\
         LISTEN {RESIDENT_GUEST_PORT}\n"
    );

    let esp = stage("lab-test", ci, debug, Some(&job_txt))?;

    // Anything a previous run (or another gate sharing this ESP) left behind
    // is inside the image this run boots. `RECEIPT.TXT` in particular would
    // put the guest into boot-time verifier mode and it would never reach the
    // job hook at all.
    for stray in [
        "TINY.BIN",
        "STOP.BIN",
        "BADMAG.BIN",
        "GOOD.RCP",
        "BAD.RCP",
        "RECEIPT.TXT",
        "STAGE.PRT",
        "CURRENT.TXT",
        "MODEL.NEW",
        "EMBED.NEW",
        "VOCAB.NEW",
        "TEST.BIN",
        "BIG.BIN",
        "BAD.BIN",
        "SOAK1.BIN",
        "SOAK2.BIN",
        "SOAK3.BIN",
    ] {
        if let Some(p) = find_ci(&esp.dir, stray) {
            let _ = fs::remove_file(p);
        }
    }

    // ---- the fixtures -----------------------------------------------------
    let model_path = find_ci(&esp.dir, "MODEL.SAF")
        .ok_or_else(|| format!("no MODEL.SAF staged under {}", esp.dir.display()))?;
    let model_bytes =
        fs::read(&model_path).map_err(|e| format!("reading {}: {e}", model_path.display()))?;
    let vocab_size = model_vocab_size(&model_bytes).ok_or_else(|| {
        "could not read vocab_size out of the staged MODEL.SAF metadata".to_string()
    })?;
    println!("[3.5/4] staged model vocab_size = {vocab_size}");

    let root = repo_root();
    let good = fs::read(root.join(LAB_RECEIPT_GOLDEN))
        .map_err(|e| format!("reading {LAB_RECEIPT_GOLDEN}: {e}"))?;
    // "GOOD with a flipped byte" (design §7). The flip lands in the receipt's
    // `model` hash line, so the guest refuses it at artifact binding — the
    // first thing `verifier::run` checks — and answers `pass=false` without
    // spending a second 64-step replay under TCG. What the gate asserts is
    // `pass=false`, and a receipt that does not name these artifacts is
    // exactly a receipt this box must not pass.
    let mut bad = good.clone();
    let flip = find_subslice(&bad, b"model ")
        .map(|i| i + 6)
        .ok_or_else(|| "the golden receipt has no `model ` line to corrupt".to_string())?;
    bad[flip] = if bad[flip] == b'a' { b'b' } else { b'a' };

    // The first four ids are what `EVAL TINY.BIN 0:4` scores. They are taken
    // from the golden receipt's own `token-ids`, so every one is a legal id
    // for this vocabulary by construction.
    let ids: Vec<u32> = [12u32, 407, 283, 259, 397, 484, 408, 411]
        .into_iter()
        .take(LAB_CORPUS_NTOK)
        .collect();
    let corpus = lab_corpus(vocab_size, &ids);
    // A corpus whose only defect is its magic: design §7's `bad-corpus` case.
    let mut bad_corpus = corpus.clone();
    bad_corpus[0] = b'X';
    // Job 5's corpus: the same eight legal ids, repeated until the streamed
    // validation pass is long enough to outlast `LAB_STOP_BUDGET_S`. The ids
    // are legal for this vocabulary for the same reason `TINY.BIN`'s are.
    // Written onto the ESP directly rather than `PUT` down the socket: the
    // assertion it serves is about `EVAL`'s budget, and phase 4's delivery
    // path is already proved by the four fixtures above — sending 128 MiB a
    // second time would only make the gate longer.
    let stop_path = esp.dir.join("STOP.BIN");
    lab_corpus_stage(&stop_path, vocab_size, &ids, LAB_STOP_CORPUS_NTOK)?;
    println!(
        "[3.5/4] staged STOP.BIN, {} tokens ({} MiB of payload)",
        LAB_STOP_CORPUS_NTOK,
        LAB_STOP_CORPUS_NTOK / (1024 * 1024 / 4)
    );

    let extra = vec![
        "-netdev".to_string(),
        format!("user,id=n0,hostfwd=tcp:127.0.0.1:{host_port}-:{RESIDENT_GUEST_PORT}"),
        "-device".to_string(),
        "virtio-net-pci,netdev=n0".to_string(),
        "-device".to_string(),
        "virtio-rng-pci".to_string(),
    ];

    println!("[4/4] booting under OVMF with a virtio NIC and a host forward");
    let (qemu_bin, mut cmd) = qemu_command(&esp, &extra);
    println!("      $ {qemu_bin} -machine q35,accel=tcg -cpu max -m 2048 ...");
    println!();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not launch {qemu_bin}: {e}. Install qemu-system-x86."))?;

    let hard = Instant::now() + Duration::from_secs(LAB_TIMEOUT_S);
    let run = lab_exchange(
        &mut child,
        host_port,
        hard,
        &corpus,
        &bad_corpus,
        &good,
        &bad,
    );

    let outcome = match &run {
        Ok(_) => wait_for_exit(&mut child, Duration::from_secs(LAB_EXIT_S)),
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            Outcome::TimedOut
        }
    };

    println!();
    if let Some(text) = find_ci(&esp.dir, "BOOTLOG.TXT").and_then(|p| fs::read_to_string(p).ok()) {
        println!("---- BOOTLOG.TXT (RESIDENT/JOB lines) ----");
        for line in text.lines() {
            if line.contains("RESIDENT:") || line.contains("JOB:") {
                println!("{line}");
            }
        }
        println!("---- end ----");
        println!();
    }

    let mut failures: Vec<String> = match run {
        Ok(f) => f,
        Err(e) => {
            eprintln!("== FAIL == {e}");
            eprintln!("   The BOOTLOG lines above are the guest's own account of how far it got.");
            return Ok(ExitCode::from(1));
        }
    };

    match outcome {
        Outcome::Exited(c) => {
            println!("   ok   QEMU exited {c} (guest ResetSystem under -no-reboot)")
        }
        Outcome::Signalled => {
            failures.push("QEMU terminated by a signal with no exit code".to_string())
        }
        Outcome::TimedOut => failures.push(format!(
            "QEMU did not exit within {LAB_EXIT_S}s of the guest answering BYE to REBOOT"
        )),
    }

    println!();
    if failures.is_empty() {
        println!("== PASS == the lab plane served CPUID/VERIFY/EVAL/MEMBW as JOB.TXT");
        println!("           directives over PUT-delivered files, and reset.");
        Ok(ExitCode::SUCCESS)
    } else {
        for f in &failures {
            eprintln!("== FAIL == {f}");
        }
        Ok(ExitCode::from(1))
    }
}

/// Drive design §7's phase-5 assertions against the running guest.
///
/// The corpus and both receipts arrive by `PUT` before any `JOB` is sent, so
/// the gate exercises phases 4 and 5 **together** — which is the point of
/// running them in one gate rather than two. Everything is asserted on the
/// record the guest sends back down the socket; no value is asserted (Rule A).
fn lab_exchange(
    child: &mut Child,
    host_port: u16,
    hard: Instant,
    corpus: &[u8],
    bad_corpus: &[u8],
    good: &[u8],
    bad: &[u8],
) -> Result<Vec<String>, String> {
    let (mut c, banner) = resident_ready(child, host_port, LAB_READY_S, hard)?;
    let mut fail: Vec<String> = Vec::new();
    println!("   ok   banner {banner}");

    // ---- phase 4: put the fixtures on the volume -------------------------
    for (name, bytes) in [
        ("TINY.BIN", corpus),
        ("BADMAG.BIN", bad_corpus),
        ("GOOD.RCP", good),
        ("BAD.RCP", bad),
    ] {
        let sha = sha256_hex(bytes);
        let line = files_put(&mut c, hard, name, bytes, &sha, b"END\n")?;
        if !line.starts_with(&format!("OK {name} {}", bytes.len())) {
            return Err(format!("PUT {name} answered {line:?}"));
        }
        println!("   ok   PUT {name} ({} bytes)  ->  {line}", bytes.len());
    }

    // ---- job 1: the §7 body ---------------------------------------------
    c.send(&lab_job_block(
        "SEED 1\nCPUID\nVERIFY GOOD.RCP\nVERIFY BAD.RCP\nEVAL TINY.BIN 0:4\n",
    ))?;
    let body = c.read_result_within(hard, LAB_JOB_S)?;
    println!("---- RESULT.TXT (job 1) ----\n{body}---- end ----");

    let want = [
        ("jobs", "4"),
        ("env", "vm"),
        ("verdict", "OK"),
        ("job.1.kind", "cpuid"),
        ("job.2.kind", "verify"),
        ("job.3.kind", "verify"),
        ("job.4.kind", "eval"),
        ("job.2.pass", "true"),
        ("job.3.pass", "false"),
        ("job.4.ntok", "3"),
        ("job.1.rate_valid", "false"),
        ("job.4.rate_valid", "false"),
        ("seed", "1"),
    ];
    for (k, v) in want {
        match record_value(&body, k) {
            Some(got) if got == v => println!("   ok   {k}={got}"),
            Some(got) => fail.push(format!("{k} is {got:?}, expected {v:?}")),
            None => fail.push(format!("{k} is absent from the record")),
        }
    }
    // `nll_q16` must be present and parse as a u64. Its VALUE is never
    // asserted: it is an integer produced by the engine, and pinning it here
    // would make the gate a golden test of the model rather than of the
    // record's structure (Rule A / design §7).
    match record_value(&body, "job.4.nll_q16").map(|v| v.parse::<u64>()) {
        Some(Ok(_)) => println!("   ok   job.4.nll_q16 present and parses as u64"),
        Some(Err(_)) => fail.push("job.4.nll_q16 does not parse as a u64".to_string()),
        None => fail.push("job.4.nll_q16 is absent".to_string()),
    }
    // No `MEMBW` step ran, so the field must be absent everywhere; where it
    // does appear (job 3 below) it must read `n/a` under `env=vm`.
    for n in 1..=4 {
        match record_value(&body, &format!("job.{n}.membw_mibs")) {
            None | Some("n/a") => {}
            Some(v) => fail.push(format!(
                "job.{n}.membw_mibs is {v:?} on a record with no MEMBW step"
            )),
        }
    }
    println!("   ok   membw_mibs absent from every block of a job with no MEMBW");
    // `artifacts=` is `<model16>/<embed16>/<vocab16>`; `merge_key` is 16
    // lowercase hex (design §3/§3.1).
    match record_value(&body, "artifacts") {
        Some(a) if a.split('/').count() == 3 && a.split('/').all(is_hex16) => {
            println!("   ok   artifacts={a}")
        }
        Some(a) => fail.push(format!("artifacts={a:?} is not <16hex>/<16hex>/<16hex>")),
        None => fail.push("artifacts= is absent".to_string()),
    }
    match record_value(&body, "merge_key") {
        Some(k) if is_hex16(k) => println!("   ok   merge_key={k}"),
        Some(k) => fail.push(format!("merge_key={k:?} is not 16 lowercase hex")),
        None => fail.push("merge_key= is absent".to_string()),
    }

    // ---- job 2: a corpus whose magic is wrong ----------------------------
    c.send(&lab_job_block("EVAL BADMAG.BIN 0:4\n"))?;
    let body = c.read_result_within(hard, LAB_JOB_S)?;
    match record_value(&body, "job.1.err") {
        Some("bad-corpus") => println!("   ok   corrupted magic  ->  job.1.err=bad-corpus"),
        Some(e) => fail.push(format!("a corrupted-magic corpus answered err={e:?}")),
        None => fail.push("a corrupted-magic corpus produced no job.1.err".to_string()),
    }
    if record_value(&body, "job.1.nll_q16").is_some() {
        fail.push("a refused corpus still reported an nll_q16".to_string());
    }

    // ---- job 3: MEMBW, digest emitted and bandwidth gated ----------------
    c.send(&lab_job_block(&format!("MEMBW {LAB_MEMBW_MIB}\n")))?;
    let body = c.read_result_within(hard, LAB_JOB_S)?;
    match record_value(&body, "job.1.digest") {
        Some(d) if is_hex16(d) => println!("   ok   MEMBW digest={d}"),
        Some(d) => fail.push(format!("MEMBW digest {d:?} is not 16 lowercase hex")),
        None => fail.push("MEMBW produced no job.1.digest".to_string()),
    }
    match record_value(&body, "job.1.membw_mibs") {
        Some("n/a") => println!("   ok   job.1.membw_mibs=n/a under env=vm (Rule A)"),
        Some(v) => fail.push(format!(
            "job.1.membw_mibs is {v:?} on an env=vm record — Rule A says n/a"
        )),
        None => fail.push("MEMBW produced no job.1.membw_mibs".to_string()),
    }

    // ---- job 4: STRICT on stops at the first failing step ----------------
    c.send(&lab_job_block("STRICT on\nVERIFY BAD.RCP\nCPUID\n"))?;
    let body = c.read_result_within(hard, LAB_JOB_S)?;
    match record_value(&body, "verdict") {
        Some("FAIL verify") => println!("   ok   STRICT on  ->  verdict=FAIL verify"),
        Some(v) => fail.push(format!(
            "STRICT on gave verdict={v:?}, expected FAIL verify"
        )),
        None => fail.push("the STRICT record carries no verdict".to_string()),
    }
    match record_value(&body, "jobs") {
        Some("1") => println!("   ok   the step behind the failure was not run (jobs=1)"),
        Some(v) => fail.push(format!("STRICT on ran {v} steps, expected 1")),
        None => fail.push("the STRICT record carries no jobs=".to_string()),
    }

    // ---- job 5: the budget runs out before EVAL's first window -----------
    // A step the budget stopped is the *job's* verdict, not just the step's:
    // design §5 files a `verdict=OK` record as a completed unit, and an
    // `nll_q16` folded over zero scored positions must never be one. The case
    // that is easy to get wrong is this one — the stop lands before any
    // window folded, so `job.1.partial` reads 0, exactly as a completed
    // EVAL's does, and only `job.1.err` tells them apart.
    c.send(&lab_job_block_budget(
        LAB_STOP_BUDGET_S,
        "EVAL STOP.BIN 0:4\n",
    ))?;
    let body = c.read_result_within(hard, LAB_JOB_S)?;
    println!("---- RESULT.TXT (job 5, BUDGET {LAB_STOP_BUDGET_S}) ----\n{body}---- end ----");
    for (k, v) in [
        ("jobs", "1"),
        ("job.1.kind", "eval"),
        ("job.1.err", "budget"),
        ("job.1.partial", "0"),
        ("job.1.ntok", "0"),
        ("job.1.pass", "false"),
        ("verdict", "FAIL budget"),
    ] {
        match record_value(&body, k) {
            Some(got) if got == v => println!("   ok   {k}={got}"),
            Some(got) => fail.push(format!(
                "budget-stopped EVAL: {k} is {got:?}, expected {v:?}"
            )),
            None => fail.push(format!(
                "budget-stopped EVAL: {k} is absent from the record"
            )),
        }
    }

    // ---- and out ---------------------------------------------------------
    c.send("REBOOT\n")?;
    let bye = c.read_line(left(hard, LAB_IO_S)?)?;
    if bye != "BYE" {
        return Err(format!("REBOOT answered {bye:?}, expected BYE"));
    }
    println!("   ok   REBOOT  ->  BYE");
    Ok(fail)
}

/// 16 lowercase hex characters — the shape of `merge_key`, of each third of
/// `artifacts=`, and of the `MEMBW` checksum.
fn is_hex16(s: &str) -> bool {
    s.len() == 16
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
