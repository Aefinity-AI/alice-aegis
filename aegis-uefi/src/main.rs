#![no_std]
#![no_main]

extern crate alloc;

use alloc::format;
use core::fmt::Write;

mod allocator;
mod console;
mod cpu;
mod files;
mod font;
#[cfg(feature = "gop")]
mod gop;
mod h3;
mod job;
// AEFINITY OS phase 5: the lab job kinds (CPUID/VERIFY/EVAL/MEMBW/MECH).
// `mech` is the diagnostic block this file used to carry inline.
mod lab;
mod mtrr_decode;
// AEFINITY OS phase 1a: the unikernel's own TCP/IP stack over
// EFI_SIMPLE_NETWORK. Reached only from `job.rs` (a NETCHECK/REPORT directive
// asks for it); a boot with no JOB.TXT never touches it.
mod net;
// AEFINITY OS phase 2: the resident TCP job server (spec §4/§5). Reached only
// from `job::dispatch` when JOB.TXT says `MODE resident`.
mod reload;
mod server;
mod sysinfo;
mod verifier;
// Provides `wcslen`, which LLVM emits for the uefi crate's UTF-16 scans once
// SSE is on (hard-float target). No UEFI sysroot defines it. See wcs.rs.
mod wcs;

// Deleted load_file to eliminate dynamic allocations

/// Append a checkpoint line to BOOTLOG.TXT on the boot volume. Each call
/// opens/flushes/closes so the line survives a crash immediately after —
/// post-mortem: plug the stick into another machine and read how far boot got.
fn boot_log(root: &mut uefi::proto::media::file::Directory, msg: &str) {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    let mut buf = [0u16; 32];
    if let Ok(cstr) = uefi::CStr16::from_str_with_buf("BOOTLOG.TXT", &mut buf) {
        if let Ok(h) = root.open(cstr, FileMode::CreateReadWrite, FileAttribute::empty()) {
            if let Ok(FileType::Regular(mut f)) = h.into_type() {
                let _ = f.set_position(0xFFFF_FFFF_FFFF_FFFF); // seek to EOF
                let _ = f.write(msg.as_bytes());
                let _ = f.write(b"\r\n");
                let _ = f.flush();
                f.close();
            }
        }
    }
}

/// Wall-clock seconds since midnight, from UEFI Runtime Services GetTime().
/// Independent of the TSC, which is invariant and therefore cannot see a
/// throttled core. Returns None if the firmware refuses.
fn wall_seconds() -> Option<f64> {
    let t = uefi::runtime::get_time().ok()?;
    Some(
        t.hour() as f64 * 3600.0
            + t.minute() as f64 * 60.0
            + t.second() as f64
            + t.nanosecond() as f64 / 1e9,
    )
}

/// Log why the clock is (or is not) where we asked it to be. One line per
/// call, tagged, into BOOTLOG.TXT — readable after the stick comes home.
fn log_throttle_diag(root: &mut uefi::proto::media::file::Directory, tag: &str) {
    match cpu::throttle_diag() {
        Some(d) => {
            let therm = match &d.therm {
                Some(t) => format!(
                    "prochot_now={} prochot_log={} hot={} temp=Tj-{}C",
                    t.prochot_now as u8, t.prochot_log as u8, t.hot_now as u8, t.temp_below_tjmax
                ),
                None => alloc::string::String::from("therm=n/a"),
            };
            let bd = match d.bd_prochot_enabled {
                Some(true) => "1",
                Some(false) => "0",
                None => "n/a",
            };
            boot_log(
                root,
                &format!(
                    "TURBO_DIAG {}: cur_ratio={} req_ratio={} eist={} turbo_dis={} clkmod={:#04x} {} bdprochot_en={}",
                    tag,
                    d.status_ratio,
                    d.ctl_ratio,
                    d.eist_enabled as u8,
                    d.turbo_disabled as u8,
                    d.clock_mod,
                    therm,
                    bd
                ),
            );
        }
        None => boot_log(
            root,
            &format!(
                "TURBO_DIAG {}: unavailable (hypervisor/non-Intel/no EIST)",
                tag
            ),
        ),
    }
}

/// H2 experiment: retire the application processors from turbo-bin accounting
/// by parking them in MWAIT C6 via EFI_MP_SERVICES — the one job of an OS
/// cpuidle driver, done once, pre-ExitBootServices. Non-blocking dispatch: the
/// parking procedure never returns. Every outcome is logged; failure is a
/// finding, not an error.
fn park_aps_for_turbo(root: &mut uefi::proto::media::file::Directory) {
    use uefi::proto::pi::mp::MpServices;
    if !cpu::has_mwait() {
        boot_log(root, "MECH AP-PARK: skipped (no MONITOR/MWAIT)");
        return;
    }
    let handle = match uefi::boot::get_handle_for_protocol::<MpServices>() {
        Ok(h) => h,
        Err(e) => {
            boot_log(root, &format!("MECH AP-PARK: no MP services ({:?})", e));
            return;
        }
    };
    // SAFETY: GET_PROTOCOL open — read-only sharing with whatever agent
    // already holds MP services; we never close or uninstall it.
    let mp = match unsafe {
        uefi::boot::open_protocol::<MpServices>(
            uefi::boot::OpenProtocolParams {
                handle,
                agent: uefi::boot::image_handle(),
                controller: None,
            },
            uefi::boot::OpenProtocolAttributes::GetProtocol,
        )
    } {
        Ok(p) => p,
        Err(e) => {
            boot_log(root, &format!("MECH AP-PARK: open failed ({:?})", e));
            return;
        }
    };
    let count = match mp.get_number_of_processors() {
        Ok(c) => c,
        Err(e) => {
            boot_log(root, &format!("MECH AP-PARK: count failed ({:?})", e));
            return;
        }
    };
    boot_log(
        root,
        &format!(
            "MECH AP-PARK: {} logical processors, {} enabled",
            count.total, count.enabled
        ),
    );
    if count.enabled <= 1 {
        boot_log(root, "MECH AP-PARK: no APs to park");
        return;
    }
    // SAFETY: plain event (type 0, no notify) used only to make the dispatch
    // non-blocking; it is intentionally leaked — the APs never signal it.
    let event = match unsafe {
        uefi::boot::create_event(
            uefi::boot::EventType::empty(),
            uefi::boot::Tpl::APPLICATION,
            None,
            None,
        )
    } {
        Ok(ev) => ev,
        Err(e) => {
            boot_log(root, &format!("MECH AP-PARK: event failed ({:?})", e));
            return;
        }
    };
    match mp.startup_all_aps(
        false,
        cpu::ap_park_mwait_c6,
        core::ptr::null_mut(),
        Some(event),
        None,
    ) {
        Ok(()) => boot_log(
            root,
            &format!(
                "MECH AP-PARK: dispatched MWAIT-C6 park to {} AP(s)",
                count.enabled - 1
            ),
        ),
        Err(e) => boot_log(
            root,
            &format!("MECH AP-PARK: startup_all_aps failed ({:?})", e),
        ),
    }
}

fn typematic_print(msg: &str, delay_ms: u64) {
    let _ = console::with_console(|st| {
        for c in msg.chars() {
            let _ = st.write_char(c);
            if delay_ms > 0 {
                let _ = uefi::boot::stall(core::time::Duration::from_millis(delay_ms));
            }
        }
        core::fmt::Result::Ok(())
    });
}

fn get_file_size(root: &mut uefi::proto::media::file::Directory, path: &str) -> Option<usize> {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    let mut buf = [0u16; 128];
    let cstr = uefi::CStr16::from_str_with_buf(path, &mut buf).ok()?;
    let file_handle = root
        .open(cstr, FileMode::Read, FileAttribute::empty())
        .ok()?;
    let mut file = match file_handle.into_type().ok()? {
        FileType::Regular(f) => f,
        _ => return None,
    };
    let mut info_buf = [0u8; 256];
    let info = file
        .get_info::<uefi::proto::media::file::FileInfo>(&mut info_buf)
        .ok()?;
    let size = info.file_size() as usize;
    file.close();
    Some(size)
}

fn load_file_into(
    root: &mut uefi::proto::media::file::Directory,
    path: &str,
    bounce_slice: &mut [u8],
    dest_slice: &mut [u8],
) -> bool {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};

    let mut buf = [0u16; 128];
    let cstr = match uefi::CStr16::from_str_with_buf(path, &mut buf) {
        Ok(c) => c,
        Err(_) => {
            let _ = console::with_console(|st| {
                let _ = st.write_str(" [ERR: CStr16 fail] ");
                core::fmt::Result::Ok(())
            });
            return false;
        }
    };

    let file_handle = match root.open(cstr, FileMode::Read, FileAttribute::empty()) {
        Ok(f) => f,
        Err(e) => {
            let _ = console::with_console(|st| {
                use core::fmt::Write;
                let _ = write!(st, " [ERR: open fail {:?}] ", e.status());
                core::fmt::Result::Ok(())
            });
            return false;
        }
    };

    let mut file = match file_handle.into_type() {
        Ok(FileType::Regular(f)) => f,
        _ => {
            let _ = console::with_console(|st| {
                let _ = st.write_str(" [ERR: not regular file] ");
                core::fmt::Result::Ok(())
            });
            return false;
        }
    };

    let mut info_buf = [0u8; 256];
    let info = match file.get_info::<uefi::proto::media::file::FileInfo>(&mut info_buf) {
        Ok(i) => i,
        Err(e) => {
            let _ = console::with_console(|st| {
                use core::fmt::Write;
                let _ = write!(st, " [ERR: get_info fail {:?}] ", e.status());
                core::fmt::Result::Ok(())
            });
            return false;
        }
    };
    let size = info.file_size() as usize;
    if size > dest_slice.len() {
        let _ = console::with_console(|st| {
            use core::fmt::Write;
            let _ = write!(st, " [ERR: size mismatch {} > {}] ", size, dest_slice.len());
            core::fmt::Result::Ok(())
        });
        return false;
    }

    let _ = console::with_console(|st| {
        let _ = st.clear(); // CLEAR THE SCREEN TO PREVENT HARDWARE SCROLL BUG!
        use core::fmt::Write;
        let _ = write!(st, "  -> Reading {} ({} bytes) [", path, size);
        core::fmt::Result::Ok(())
    });

    // Let the USB controller's internal state machine flush and breathe between files
    let _ = uefi::boot::stall(core::time::Duration::from_secs(1)); // 1 second

    let bounce_ptr = bounce_slice.as_mut_ptr();
    let ptr = dest_slice.as_mut_ptr();
    let mut bytes_read = 0;

    while bytes_read < size {
        let chunk_size = core::cmp::min(bounce_slice.len(), size - bytes_read);
        let chunk = &mut bounce_slice[0..chunk_size];

        match file.read(chunk) {
            Ok(read_len) => {
                if read_len == 0 {
                    break;
                }
                unsafe {
                    core::ptr::copy_nonoverlapping(bounce_ptr, ptr.add(bytes_read), read_len);
                }
                bytes_read += read_len;

                // Pet the watchdog and print progress every ~10MB to prevent hardware watchdog reset
                if bytes_read % (10 * 1024 * 1024) < chunk_size {
                    let _ = uefi::boot::set_watchdog_timer(0, 0, None);
                    let pct = (bytes_read * 100) / size;
                    let _ = console::with_console(|st| {
                        use core::fmt::Write;
                        let _ = write!(st, "\r  -> Reading {} [", path);
                        let bar_len = 20;
                        let filled = (pct * bar_len) / 100;
                        for i in 0..bar_len {
                            if i < filled {
                                let _ = st.write_char('=');
                            } else if i == filled {
                                let _ = st.write_char('>');
                            } else {
                                let _ = st.write_char(' ');
                            }
                        }
                        let mb_read = bytes_read / (1024 * 1024);
                        let mb_total = size / (1024 * 1024);
                        let _ = write!(st, "] {}% ({} MB/{} MB)   ", pct, mb_read, mb_total);
                        core::fmt::Result::Ok(())
                    });
                }
            }
            Err(e) => {
                let _ = console::with_console(|st| {
                    use core::fmt::Write;
                    let _ = write!(
                        st,
                        " [ERR: read fail at byte {} - {:?}] ",
                        bytes_read,
                        e.status()
                    );
                    core::fmt::Result::Ok(())
                });
                file.close(); // Prevent handle leak on error
                return false;
            }
        }
    }

    file.close(); // VERY IMPORTANT: Close the file to free FAT32 subsystem resources!

    // A short read (flaky USB, BOT stall) must NOT be reported as success —
    // silently truncated weights produce garbage inference with no error.
    if bytes_read != size {
        let _ = console::with_console(|st| {
            use core::fmt::Write;
            let _ = write!(
                st,
                " [ERR: truncated read {} of {} bytes] ",
                bytes_read, size
            );
            core::fmt::Result::Ok(())
        });
        return false;
    }

    let _ = console::with_console(|st| {
        let _ = st.write_str("OK]\r\n");
        core::fmt::Result::Ok(())
    });
    true
}

#[cfg(feature = "qemu-test")]
pub unsafe fn exit_uefi_test_runner(success: bool) -> ! {
    let code: u8 = if success { 0x10 } else { 0x11 };
    core::arch::asm!("out dx, al", in("dx") 0xf4u16, in("al") code, options(nomem, nostack, preserves_flags));
    loop {}
}

#[cfg(not(feature = "qemu-test"))]
pub unsafe fn exit_uefi_test_runner(_success: bool) {}

fn fatal_error(msg: &str) -> uefi::Status {
    let _ = console::with_console(|st| {
        let _ = st.write_str(msg);
        let _ = st.write_str("\r\nRebooting in 10 seconds...\r\n");
        core::fmt::Result::Ok(())
    });

    unsafe {
        exit_uefi_test_runner(false);
    }

    let _ = uefi::boot::stall(core::time::Duration::from_secs(10)); // 10 seconds
    uefi::Status::ABORTED
}

#[uefi::entry]
fn main() -> uefi::Status {
    // Disable UEFI Watchdog Timer (prevents 5-minute timeout reboots)
    let _ = uefi::boot::set_watchdog_timer(0, 0, None);

    // 1. Initialize a small 32MB block for early `find_handles` Vec allocations and JSON parsing
    allocator::init_uefi_alloc_small();

    // Select the boot console: GOP framebuffer if a suitable mode exists,
    // else the firmware's 80x25 text console (console::with_console handles
    // the fallback transparently for every print site below).
    console::init();

    // (Removed duplicate init_uefi_alloc_large call here to prevent OOM before file load)

    // Enable AVX-2 in UEFI (Firmware puts CPU in Long Mode but leaves AVX disabled)
    unsafe {
        let cpuid_res = core::arch::x86_64::__cpuid(1);
        let cpuid_ecx = cpuid_res.ecx;
        // Bit 26 = XSAVE capability. (Bit 27 is OSXSAVE *already enabled by the
        // OS* — firmware boots with it clear, so gating on 27 always skipped this block.)
        let xsave_supported = (cpuid_ecx & (1 << 26)) != 0;
        let avx_supported = (cpuid_ecx & (1 << 28)) != 0;

        if xsave_supported && avx_supported {
            let mut cr4: usize;
            core::arch::asm!("mov {}, cr4", out(reg) cr4);
            cr4 |= (1 << 9) | (1 << 10) | (1 << 18); // Added OSFXSR (9) and OSXMMEXCPT (10) for safety
            core::arch::asm!("mov cr4, {}", in(reg) cr4);

            let mut eax: u32;
            let mut edx: u32;
            core::arch::asm!("xgetbv", in("ecx") 0, out("eax") eax, out("edx") edx);
            eax |= 0b110; // SSE (1) and AVX (2)
            core::arch::asm!("xsetbv", in("ecx") 0, in("eax") eax, in("edx") edx);
        }

        typematic_print("[SYS] Tactically Questioning Silicon...\r\n", 5);
        let mut bbuf = [0u8; 48];
        let brand = cpu::brand_string(&mut bbuf);
        typematic_print(&format!("[SYS] -> Model: {}\r\n", brand), 5);

        let sse41 = (cpuid_ecx & (1 << 19)) != 0;
        let fma = (cpuid_ecx & (1 << 12)) != 0;
        let avx2_supported = (core::arch::x86_64::__cpuid_count(7, 0).ebx & (1 << 5)) != 0;

        let mut msg = alloc::string::String::new();
        let _ = write!(
            msg,
            "[SYS] -> SSE4.1: [ {} ]\r\n",
            if sse41 { "DETECTED" } else { "MISSING " }
        );
        let _ = write!(
            msg,
            "[SYS] -> AVX:    [ {} ]\r\n",
            if avx_supported {
                "DETECTED"
            } else {
                "MISSING "
            }
        );
        let _ = write!(
            msg,
            "[SYS] -> AVX2:   [ {} ]\r\n",
            if avx2_supported {
                "DETECTED"
            } else {
                "MISSING "
            }
        );
        let _ = write!(
            msg,
            "[SYS] -> FMA3:   [ {} ]\r\n",
            if fma { "DETECTED" } else { "MISSING " }
        );
        typematic_print(&msg, 5);
        typematic_print(
            &format!(
                "[SYS] Locking inference kernel to native {}.\r\n",
                aegis_core::ops::simd_level_name()
            ),
            5,
        );
    }

    let _ = console::with_console(|st| {
        let _ = st.clear();
        let _ = st.write_str("\r\n==================================================\r\n");
        let _ = st.write_str(" A.L.I.C.E. UNIKERNEL INFERENCE ENGINE (UEFI APP)\r\n");
        let _ = st.write_str("==================================================\r\n\n");
        let _ = st.write_str("[SYSTEM] Pre-allocating DMA Bounce Buffer...\r\n");
        core::fmt::Result::Ok(())
    });

    let bounce_alloc_size = 128 * 1024; // 128KB total allocation to guarantee we can find a 64KB-aligned segment
    let bounce_pages = bounce_alloc_size / 4096;
    let bounce_addr = match uefi::boot::allocate_pages(
        uefi::boot::AllocateType::MaxAddress(0xFFFFFFFF),
        uefi::boot::MemoryType::LOADER_DATA,
        bounce_pages,
    ) {
        Ok(a) => a,
        Err(_) => return fatal_error("FATAL: Could not allocate DMA Bounce Buffer!\r\n"),
    };

    // Find the 64KB aligned physical boundary within the 128KB allocation
    let mut aligned_ptr = bounce_addr.as_ptr() as usize;
    let remainder = aligned_ptr % (64 * 1024);
    if remainder != 0 {
        aligned_ptr += (64 * 1024) - remainder;
    }
    // Create a 64KB bounce slice perfectly aligned to 64KB physical boundary (Cannot cross boundaries!)
    let bounce_slice =
        unsafe { core::slice::from_raw_parts_mut(aligned_ptr as *mut u8, 64 * 1024) };

    let _ = console::with_console(|st| {
        let _ = st.write_str("[SYSTEM] Pre-allocating Tensor Space...\r\n");
        core::fmt::Result::Ok(())
    });

    // Get the handle of the USB drive we actually booted from (Bypasses internal NVMe EFI partitions)
    use uefi::proto::loaded_image::LoadedImage;
    use uefi::proto::media::fs::SimpleFileSystem;

    let image_handle = uefi::boot::image_handle();
    let loaded_image = match uefi::boot::open_protocol_exclusive::<LoadedImage>(image_handle) {
        Ok(li) => li,
        Err(_) => return fatal_error("FATAL: Could not open LoadedImage protocol\r\n"),
    };

    let device_handle = match loaded_image.device() {
        Some(d) => d,
        None => return fatal_error("FATAL: Boot device handle not found\r\n"),
    };

    let mut sfs = match uefi::boot::open_protocol_exclusive::<SimpleFileSystem>(device_handle) {
        Ok(s) => s,
        Err(_) => return fatal_error("FATAL: Could not open SimpleFileSystem on boot device\r\n"),
    };
    let mut root = match sfs.open_volume() {
        Ok(v) => v,
        Err(_) => return fatal_error("FATAL: Could not open root volume on boot device\r\n"),
    };

    boot_log(&mut root, "==== A.L.I.C.E. BOOT ====");
    boot_log(
        &mut root,
        "STAGE 1: boot volume opened, AVX enable attempted",
    );

    let model_size = get_file_size(&mut root, "MODEL.SAF").unwrap_or(0);
    let emb_size = get_file_size(&mut root, "EMBED.BIN").unwrap_or(0);
    let vocab_size = get_file_size(&mut root, "VOCAB.BIN").unwrap_or(0);

    if model_size == 0 || emb_size == 0 || vocab_size == 0 {
        boot_log(&mut root, "STAGE 2 FAILED: file sizes unreadable");
        return fatal_error("FATAL: Could not read file sizes from USB FAT32 Partition!\r\n");
    }
    boot_log(
        &mut root,
        &format!(
            "STAGE 2: sizes OK model={} embed={} vocab={}",
            model_size, emb_size, vocab_size
        ),
    );

    let model_pages = (model_size + 4095) / 4096;
    let emb_pages = (emb_size + 4095) / 4096;
    let vocab_pages = (vocab_size + 4095) / 4096;
    let model_addr = match allocator::allocate_huge_pages(model_pages) {
        Ok(a) if a.as_ptr() as usize != 0 => a,
        _ => {
            boot_log(
                &mut root,
                "STAGE 3 FAILED: contiguous alloc for MODEL.SAF (memory map too fragmented / not enough RAM)",
            );
            return fatal_error(
                "FATAL: Could not allocate memory for Model Tensors or received Zero Page!\r\n",
            );
        }
    };
    let model_slice =
        unsafe { core::slice::from_raw_parts_mut(model_addr.as_ptr() as *mut u8, model_size) };

    let emb_addr = match allocator::allocate_huge_pages(emb_pages) {
        Ok(a) if a.as_ptr() as usize != 0 => a,
        _ => {
            return fatal_error(
                "FATAL: Could not allocate memory for Embeddings or received Zero Page!\r\n",
            );
        }
    };
    let embeddings_slice =
        unsafe { core::slice::from_raw_parts_mut(emb_addr.as_ptr() as *mut u8, emb_size) };

    let vocab_addr = match allocator::allocate_huge_pages(vocab_pages) {
        Ok(a) if a.as_ptr() as usize != 0 => a,
        _ => {
            return fatal_error(
                "FATAL: Could not allocate memory for Vocabulary or received Zero Page!\r\n",
            );
        }
    };
    let vocab_slice =
        unsafe { core::slice::from_raw_parts_mut(vocab_addr.as_ptr() as *mut u8, vocab_size) };

    let _ = console::with_console(|st| {
        let _ = st.write_str("[SYSTEM] Locating weights on USB FAT32 Partition...\r\n");
        core::fmt::Result::Ok(())
    });

    boot_log(&mut root, "STAGE 3: tensor memory allocated");
    // H3: record where the firmware actually put each buffer, at load time —
    // the physical placement is the variable under test (h3.rs).
    boot_log(
        &mut root,
        &format!(
            "H3 buffers: MODEL.SAF=0x{:X}+{} EMBED.BIN=0x{:X}+{} VOCAB.BIN=0x{:X}+{}",
            model_addr.as_ptr() as usize,
            model_size,
            emb_addr.as_ptr() as usize,
            emb_size,
            vocab_addr.as_ptr() as usize,
            vocab_size
        ),
    );

    // 1. Largest: Model Tensors
    if !load_file_into(&mut root, "MODEL.SAF", bounce_slice, model_slice) {
        boot_log(&mut root, "STAGE 4 FAILED: MODEL.SAF read error");
        return fatal_error("FATAL: Could not load MODEL.SAF\r\n");
    }
    let _ = uefi::boot::stall(core::time::Duration::from_secs(1)); // 1-second stall to flush USB hardware state machine
    boot_log(&mut root, "STAGE 4a: MODEL.SAF loaded");

    // 2. Embeddings
    if !load_file_into(&mut root, "EMBED.BIN", bounce_slice, embeddings_slice) {
        boot_log(&mut root, "STAGE 4b FAILED: EMBED.BIN read error");
        return fatal_error("FATAL: Could not load EMBED.BIN\r\n");
    }
    let _ = uefi::boot::stall(core::time::Duration::from_secs(1)); // 1-second stall to flush USB hardware state machine
    boot_log(&mut root, "STAGE 4b: EMBED.BIN loaded");

    // 3. Smallest: Vocabulary
    if !load_file_into(&mut root, "VOCAB.BIN", bounce_slice, vocab_slice) {
        boot_log(&mut root, "STAGE 4c FAILED: VOCAB.BIN read error");
        return fatal_error("FATAL: Could not load VOCAB.BIN\r\n");
    }
    let _ = uefi::boot::stall(core::time::Duration::from_secs(1));
    boot_log(&mut root, "STAGE 4c: VOCAB.BIN loaded");

    // 4d. Witness receipt, if the stick carries one (Provable AI Kit mode).
    // Loaded while the bounce buffer is still alive; acted on after the
    // working heap comes up. Uses the early heap — receipts are a few KB.
    let mut receipt_bytes: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    if let Some(rsize) = get_file_size(&mut root, "RECEIPT.TXT")
        && rsize > 0
        && rsize <= 64 * 1024
    {
        receipt_bytes.resize(rsize, 0);
        if load_file_into(&mut root, "RECEIPT.TXT", bounce_slice, &mut receipt_bytes) {
            boot_log(
                &mut root,
                "STAGE 4d: RECEIPT.TXT loaded — verifier mode armed",
            );
        } else {
            receipt_bytes.clear();
            boot_log(
                &mut root,
                "STAGE 4d: RECEIPT.TXT present but unreadable — ignoring",
            );
        }
    }

    // 5. Initialize the massive second block LAST so large files got priority for contiguous physical RAM
    allocator::init_uefi_alloc_large();
    boot_log(&mut root, "STAGE 5: working heap online");

    // Free bounce buffer
    unsafe {
        let _ = uefi::boot::free_pages(bounce_addr, bounce_pages);
    }

    let _ = console::with_console(|st| {
        let _ = st.write_str("\r\n[SYSTEM] Files loaded to RAM. Initializing Tensor Arena...\r\n");
        core::fmt::Result::Ok(())
    });

    // ---- Provable AI Kit: verifier mode --------------------------------
    // A receipt on the stick turns this boot into a witness verification:
    // replay the receipt's decode through the CIS-1 full-integer engine and
    // report PASS/FAIL. Identity only — no REPL, no timing (Rule A).
    if !receipt_bytes.is_empty() {
        let _ = console::with_console(|st| {
            let _ = st.write_str(
                "\r\n[AEGIS] RECEIPT.TXT found — witness verifier mode (CIS-1 FullInt)\r\n",
            );
            core::fmt::Result::Ok(())
        });
        // No watchdog is armed on this path — it was disabled above, before
        // the artifacts were loaded — so the re-arm hook (design §8) is a
        // no-op here and this boot behaves exactly as it did before phase 5.
        let verdict = verifier::run(
            model_slice,
            embeddings_slice,
            vocab_slice,
            &receipt_bytes,
            &mut || {},
        );
        let _ = console::with_console(|st| {
            for line in verdict.detail.lines() {
                let _ = st.write_str(line);
                let _ = st.write_str("\r\n");
            }
            core::fmt::Result::Ok(())
        });
        // Attribution (ledger A33/A34): the previous log carried no CPUID at
        // all, so a PASS could not be tied to a specific physical machine
        // from the log alone. This line changes nothing about the verify
        // computation above — it only records what CPU produced this PASS.
        let mut cpuid_vendor_buf = [0u8; 12];
        let cpuid_vendor = cpu::vendor_string(&mut cpuid_vendor_buf);
        let mut cpuid_brand_buf = [0u8; 48];
        let cpuid_brand = cpu::brand_string(&mut cpuid_brand_buf);
        let (cpuid_family, cpuid_model, cpuid_stepping) = cpu::family_model_stepping();
        let (cpuid_avx2, cpuid_fma, cpuid_sse2) = cpu::identity_feats();
        boot_log(
            &mut root,
            &format!(
                "CPUID: vendor={} brand=\"{}\" family={} model={} stepping={} feats=<avx2:{},fma:{},sse2:{}>",
                cpuid_vendor,
                cpuid_brand,
                cpuid_family,
                cpuid_model,
                cpuid_stepping,
                cpuid_avx2 as u8,
                cpuid_fma as u8,
                cpuid_sse2 as u8,
            ),
        );
        boot_log(
            &mut root,
            &format!(
                "STAGE V: witness verify {} — {}",
                if verdict.pass { "PASS" } else { "FAIL" },
                verdict.detail.lines().last().unwrap_or("")
            ),
        );
        // SAFETY: qemu-test builds write the exit code to QEMU's
        // isa-debug-exit port and never return; production builds compile
        // this to a no-op and fall through to the on-screen halt below.
        unsafe {
            exit_uefi_test_runner(verdict.pass);
        }
        #[allow(unreachable_code)]
        {
            let _ = console::with_console(|st| {
                let _ = st.write_str("\r\nVerification complete. It is safe to power off.\r\n");
                core::fmt::Result::Ok(())
            });
            loop {
                uefi::boot::stall(core::time::Duration::from_secs(60));
            }
        }
    }

    let mut engine = match aegis_core::inference::TernaryInferenceEngine::new(
        embeddings_slice,
        model_slice,
        vocab_slice,
    ) {
        Ok(e) => e,
        Err(err) => {
            boot_log(&mut root, &format!("STAGE 6 FAILED: engine init: {}", err));
            let _ = console::with_console(|st| {
                let _ = st.write_str("FATAL: Engine failed: ");
                let _ = st.write_str(&err);
                let _ = st.write_str("\r\n");
                core::fmt::Result::Ok(())
            });
            return uefi::Status::ABORTED;
        }
    };
    // CPUID says what the CPU can do; cfg! says what this binary was compiled
    // to do. A soft-float build (stock x86_64-unknown-uefi target) prints
    // codegen=softfloat here and self-identifies in every bootlog — the
    // MECH-A1 regression (ledger A14) was invisible precisely because this
    // line used to report capability only.
    boot_log(
        &mut root,
        &format!(
            "STAGE 6: engine online, SIMD={}, codegen={}",
            aegis_core::ops::simd_level_name(),
            if cfg!(target_feature = "sse2") {
                "hardfloat"
            } else {
                "softfloat"
            }
        ),
    );

    // ---- STAGE 7: take over the job an OS cpufreq governor would do ----------
    // Nothing here ran on the boot path before 2026-07-29. request_max_performance()
    // existed but was only reachable from /gauntlet, /turbo and /autotest, so a
    // normal boot never asked the CPU for full speed at all.
    //
    // On the Dell i5-5200U the 2026-07-29 bare-metal diagnostic measured
    // cur_ratio=5 (~500 MHz on a 2200 MHz part) with eist=1, turbo_dis=0,
    // clkmod=0x00, hot=0 and temp=Tj-72C — not thermal, not disabled, request
    // delivered and ignored — alongside prochot_now=1 and bdprochot_en=1. An
    // external assertion was outranking the P-state request. Clear that first,
    // THEN ask for max performance; the order matters, because a boost request
    // into an asserted PROCHOT is exactly what produced the 0.61 -> 0.62 tok/s
    // null result the program chased for seventeen days.
    //
    // Every step is logged with a before/after read-back so a stick that comes
    // home can be read as evidence rather than as an assumption.
    log_throttle_diag(&mut root, "boot-pre");

    match cpu::clear_bd_prochot() {
        Ok(cpu::BdProchotClear::AlreadyDisabled) => {
            boot_log(
                &mut root,
                "STAGE 7 bd-prochot: already disabled, no write issued",
            );
        }
        Ok(cpu::BdProchotClear::Cleared {
            prochot_was_asserted,
        }) => {
            boot_log(
                &mut root,
                &format!(
                    "STAGE 7 bd-prochot: CLEARED (read-back confirms 0); prochot_was_asserted={}",
                    prochot_was_asserted as u8
                ),
            );
        }
        Ok(cpu::BdProchotClear::WouldNotClear) => {
            boot_log(
                &mut root,
                "STAGE 7 bd-prochot: WRITE DID NOT STICK — bit still set after write",
            );
        }
        Err(why) => {
            boot_log(&mut root, &format!("STAGE 7 bd-prochot: skipped ({})", why));
        }
    }

    match cpu::request_max_performance() {
        Ok(cpu::Boost::Hwp { highest }) => {
            boot_log(
                &mut root,
                &format!("STAGE 7 pstate: HWP engaged, highest={}", highest),
            );
        }
        Ok(cpu::Boost::LegacySpeedStep { ratio, mhz }) => {
            boot_log(
                &mut root,
                &format!(
                    "STAGE 7 pstate: legacy SpeedStep ratio={} (~{}MHz)",
                    ratio, mhz
                ),
            );
        }
        Err(why) => {
            boot_log(&mut root, &format!("STAGE 7 pstate: not applied ({})", why));
        }
    }

    log_throttle_diag(&mut root, "boot-post");

    // ---- H3 probe: memory-attribute audit -------------------------------------
    // Dumps MTRR/PAT state and the UEFI-map view of exactly the physical
    // ranges the engine touches, so a UC/WT/WC-mapped buffer — the H3
    // explanation for the Band-3 residual — is visible in BOOTLOG.TXT
    // instead of inferred. Read-only; runs on every path including qemu-test.
    {
        use alloc::string::String;
        use alloc::vec::Vec;
        let mut ranges: Vec<(String, usize, usize)> = Vec::new();
        let (img_base, img_len) = loaded_image.info();
        ranges.push((String::from("image"), img_base as usize, img_len as usize));
        ranges.push((
            String::from("MODEL.SAF"),
            model_addr.as_ptr() as usize,
            model_size,
        ));
        ranges.push((
            String::from("EMBED.BIN"),
            emb_addr.as_ptr() as usize,
            emb_size,
        ));
        ranges.push((
            String::from("VOCAB.BIN"),
            vocab_addr.as_ptr() as usize,
            vocab_size,
        ));
        let mut heaps = [(0usize, 0usize); 16];
        let nheaps = allocator::heap_regions(&mut heaps);
        for (i, &(lo, sz)) in heaps[..nheaps].iter().enumerate() {
            ranges.push((format!("heap{}", i), lo, sz));
        }
        h3::log_h3_probe(&mut root, &ranges);
    }

    let _ = console::with_console(|st| {
        let _ = write!(st, "[SYSTEM] Engine Online.\r\n\n");
        core::fmt::Result::Ok(())
    });

    #[cfg(not(feature = "qemu-test"))]
    {
        typematic_print("[OPSEC] Operating System: None\r\n", 5);
        typematic_print("[OPSEC] Network Stack:    None\r\n", 5);
        typematic_print("[OPSEC] Background Procs: 0\r\n", 5);
        typematic_print(
            "[OPSEC] Attack Surface:   Minimal (Ring 0 Unikernel)\r\n\n",
            5,
        );
    }

    // ---- AEFINITY OS phase 0: the JOB.TXT hook -----------------------------
    // The single hook the spec allows (program/AEFINITY_OS.md §5): a boot
    // volume carrying JOB.TXT makes this box a headless lab worker. Parse the
    // directives, run them, write RESULT.TXT, then ResetSystem (AFTER reset,
    // which does not return) or fall back through here (AFTER halt).
    //
    // WITH NO JOB.TXT `load` returns None and nothing below changes: the
    // MECH experiment, the qemu-test signal and the interactive console are
    // all exactly what they were. `cargo xtask boot-test` still exiting 33 is
    // the check on that.
    //
    // Placement. The spec says "immediately before the interactive console",
    // which would put this after the MECH block. It goes BEFORE MECH instead,
    // for two reasons, and the gate is the evidence for the second:
    //   1. A box that was handed a job should do the job. MECH is an
    //      unconditional hands-off diagnostic experiment that generates ~100
    //      tokens across nine runs before the console is ever reached; on a
    //      lab worker that is time charged to nobody's job and, in resident
    //      mode later, would delay every reboot-and-serve cycle.
    //   2. Under QEMU/TCG on the dev box MECH alone runs for >11 minutes
    //      (measured 2026-08-31, 13m20s wall and still inside MECH v2), so
    //      behind it the §6 `job-test` deadline of 300 s cannot be met by any
    //      job, however small. Correctness gate, not a timing claim (Rule A).
    // It is still ONE hook, still after the engine is loaded and before the
    // interactive console (spec §2), and still before the qemu-test inference
    // block below — which matters because job-test reuses the qemu-test
    // binary and that block ends the guest through isa-debug-exit, which
    // would pre-empt the job's ResetSystem.
    if let Some(job) = job::load(&mut root) {
        // AEFINITY OS phase 4 (design §4.1): the job path takes an
        // `EngineSlot`, not a `&mut Engine`. The slot **owns** the engine, so
        // `RELOAD` can drop it, refill the slabs allocated at STAGE 3 and
        // build a new one — which is the only way a box can report a fresh
        // `model_sha` and be telling the truth about what it is inferring
        // with. Everything below stays inside this one hook, so the
        // no-`JOB.TXT` boot path is byte-for-byte what it was.
        //
        // SAFETY (the three `Slab::adopt` calls): each address is exactly what
        // `allocator::allocate_huge_pages` returned above, each capacity is
        // that call's page count in bytes, and each length is the artifact
        // size already loaded into it. None of the three is ever freed — this
        // unikernel never releases tensor memory — and from here the slot is
        // their only writer, which is what `adopt` is promised.
        let slabs = unsafe {
            reload::Slabs {
                model: reload::Slab::adopt(model_addr, model_pages * 4096, model_size),
                embed: reload::Slab::adopt(emb_addr, emb_pages * 4096, emb_size),
                vocab: reload::Slab::adopt(vocab_addr, vocab_pages * 4096, vocab_size),
            }
        };
        // The artifact digests are streamed once, here, because every record
        // and every `HEALTH` reply for the rest of this uptime quotes them.
        // The watchdog is re-armed per chunk (design §8): hashing 1.83 GB can
        // outlast any single window.
        let mut slot = reload::EngineSlot::adopt(engine, slabs, &mut || {
            job::arm_watchdog(files::FILES_WD_S);
        });
        job::arm_watchdog(0);
        job::dispatch(&job, &mut root, &mut slot);
        // `dispatch` only returns for `AFTER halt` in oneshot mode. Take the
        // engine back so the boot path below is the one it would have had.
        match slot.take_engine() {
            Some(e) => engine = e,
            None => {
                boot_log(&mut root, "JOB: the engine slot came back empty");
                return fatal_error("FATAL: engine slot empty after the job\r\n");
            }
        }
    }

    // ---- MECH: OS-advantage mechanism experiment, hands-off, one boot ------
    // AEFINITY OS phase 5 (design §4): the block that used to sit inline here
    // now lives in `lab::mech`, so the dispatcher can run it as a `MECH`
    // directive — and, per design §2, run it **last** whatever order the
    // `JOB.TXT` gave, because it is ~24 minutes under TCG and must never
    // starve real work. This call is a pure move: same output, same order,
    // same place in the boot path, so a stick with no `JOB.TXT` behaves
    // exactly as it did (`cargo xtask boot-test`, 33 checks).
    lab::mech(&mut root, &mut engine);

    #[cfg(feature = "qemu-test")]
    {
        // Run REAL autonomous inference; only claim success if tokens were generated.
        let _ = console::with_console(|st| {
            let _ =
                st.write_str("[TEST] Prompt: What is the capital of France?\r\n[TEST] Response: ");
            core::fmt::Result::Ok(())
        });
        let mut token_count: usize = 0;
        let response = engine.process_intent("What is the capital of France?", 48, |token_str| {
            token_count += 1;
            let _ = console::with_console(|st| {
                let _ = st.write_str(token_str);
                core::fmt::Result::Ok(())
            });
        });
        let ok = token_count > 1 && !response.is_empty();
        boot_log(
            &mut root,
            &format!(
                "STAGE 7: inference complete, {} tokens generated",
                token_count
            ),
        );
        let _ = console::with_console(|st| {
            let _ = write!(st, "\r\n\r\n[TEST] Generated {} tokens.\r\n", token_count);
            if ok {
                let _ = st.write_str("[TEST SUCCESS] Autonomous Inference Completed.\r\n");
            } else {
                let _ = st.write_str("[TEST FAILED] No tokens generated.\r\n");
            }
            core::fmt::Result::Ok(())
        });
        unsafe {
            exit_uefi_test_runner(ok);
        }
    }

    #[cfg(not(feature = "qemu-test"))]
    // Bounded by default. The old value was 8192, which on a machine with no way
    // to interrupt generation meant a stray prompt could run for hours.
    let mut max_new_tokens: usize = 256;

    #[cfg(not(feature = "qemu-test"))]
    loop {
        let _ = console::with_console(|st| {
            let _ = st.write_str("A.L.I.C.E.> ");
            core::fmt::Result::Ok(())
        });
        let mut input_buffer = [0u8; 512];
        let mut len = 0;

        loop {
            let key_opt = uefi::system::with_stdin(|stdin| stdin.read_key());
            if let Ok(Some(key)) = key_opt {
                match key {
                    uefi::proto::console::text::Key::Printable(c) => {
                        let c_char: char = c.into();
                        if c_char == '\r' || c_char == '\n' {
                            let _ = console::with_console(|st| {
                                let _ = st.write_char('\r');
                                let _ = st.write_char('\n');
                                core::fmt::Result::Ok(())
                            });
                            break;
                        } else if c_char == '\x08' {
                            // Step back over a whole UTF-8 character, not one byte,
                            // or the buffer stops being valid UTF-8.
                            while len > 0 {
                                len -= 1;
                                if input_buffer[len] & 0xC0 != 0x80 {
                                    break;
                                } // not a continuation byte
                            }
                            let _ = console::with_console(|st| {
                                let _ = st.write_str("\x08 \x08");
                                core::fmt::Result::Ok(())
                            });
                        } else {
                            // Gate on the ENCODED length: a Char16 encodes to up to 3
                            // UTF-8 bytes, so `len < capacity` alone can overflow.
                            let mut buf = [0u8; 4];
                            let enc = c_char.encode_utf8(&mut buf);
                            if len + enc.len() <= input_buffer.len() {
                                input_buffer[len..len + enc.len()].copy_from_slice(enc.as_bytes());
                                len += enc.len();
                                let _ = console::with_console(|st| {
                                    let _ = st.write_char(c_char);
                                    core::fmt::Result::Ok(())
                                });
                            }
                        }
                    }
                    _ => {}
                }
            } else {
                core::arch::x86_64::_mm_pause();
            }
        }

        if len > 0 {
            if let Ok(cmd) = core::str::from_utf8(&input_buffer[..len]) {
                if cmd == "/exit" {
                    break;
                }

                // A whitespace-only prompt gives the model nothing to condition on.
                // It then emits whatever is most probable given no content, which for
                // a code-heavy corpus is import blocks and licence headers — and with
                // no EOS in sight it runs to the token cap. Refuse it explicitly
                // instead of letting the machine appear to have gone haywire.
                if cmd.trim().is_empty() {
                    let _ = console::with_console(|st| {
                        let _ = st.write_str("(empty prompt ignored — type a question)\r\n");
                        core::fmt::Result::Ok(())
                    });
                    continue;
                }

                if cmd == "/gauntlet" {
                    // Race every measurable approach against itself on THIS
                    // silicon, in one boot. See docs/EXPERIMENT_GAUNTLET.md.
                    // Each segment is its own control; cheapest first so a partial
                    // run still yields the early rows. All toggles are reset after.
                    macro_rules! gbench {
                        ($label:expr, $prompt:expr, $max:expr) => {{
                            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
                            let w0 = wall_seconds();
                            let p0 = cpu::perf_snapshot();
                            let mut n = 0u64;
                            engine.process_intent($prompt, $max, |t| {
                                if !t.starts_with("[SYSTEM]") && !t.contains("[PERFORMANCE]") { n += 1; }
                            });
                            let p1 = cpu::perf_snapshot();
                            let w1 = wall_seconds();
                            let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
                            let secs = match (w0, w1) { (Some(a), Some(b)) if b >= a => b - a, _ => 0.0 };
                            let clk = match (p0, p1) { (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b), _ => None };
                            // Split the segment's wall time across prefill/decode in
                            // proportion to TSC ticks (invariant-rate), so the reported
                            // tok/s is decode-only: a fixed prefill cost amortized over
                            // more output tokens otherwise fakes a speedup with length.
                            let (p_ticks, p_toks) = (engine.last_prefill_cycles, engine.last_prefill_tokens);
                            let (d_ticks, d_steps) = (engine.last_decode_cycles, engine.last_decode_steps);
                            let p_secs = if dt > 0 { secs * (p_ticks as f64 / dt as f64) } else { 0.0 };
                            let d_secs = if dt > 0 { secs * (d_ticks as f64 / dt as f64) } else { 0.0 };
                            let d_tps = if d_secs > 0.0 { d_steps as f64 / d_secs } else { 0.0 };
                            boot_log(&mut root, &format!(
                                "GAUNTLET {}: {} tok, path={}, prefill {} tok {}t {}.{:03}s, decode {} tok {} ticks/tok {}.{:02} tok/s, total {}.{:03}s, clock {}",
                                $label, n, aegis_core::ops::active_path_name(),
                                p_toks, p_ticks,
                                p_secs as u64, ((p_secs * 1000.0) as u64) % 1000,
                                d_steps, if d_steps > 0 { d_ticks / d_steps } else { 0 },
                                d_tps as u64, ((d_tps * 100.0) as u64) % 100,
                                secs as u64, ((secs * 1000.0) as u64) % 1000,
                                match clk { Some(p) => format!("{}%", p), None => alloc::string::String::from("?") }
                            ));
                            let _ = console::with_console(|st| { let _ = write!(st, "  [{}] decode {}.{:02} tok/s, prefill {}.{:03}s\r\n", $label, d_tps as u64, ((d_tps*100.0) as u64)%100, p_secs as u64, ((p_secs*1000.0) as u64)%1000); core::fmt::Result::Ok(()) });
                        }};
                    }
                    let say = |s: &str| {
                        let _ = console::with_console(|st| {
                            let _ = st.write_str(s);
                            core::fmt::Result::Ok(())
                        });
                    };
                    let ESSAY = "Write a comprehensive and detailed essay about the future of artificial intelligence in aerospace.";

                    // Segment 0: identify
                    boot_log(&mut root, "==== GAUNTLET ====");
                    let mut bbuf = [0u8; 48];
                    let brand = cpu::brand_string(&mut bbuf);
                    let (fb, fm, fbus) = cpu::frequencies_mhz();
                    boot_log(
                        &mut root,
                        &format!(
                            "GAUNTLET CPU: {} | base={}MHz max={}MHz bus={}MHz | simd={} hwp={} turbo={} baremetal={}",
                            brand,
                            fb,
                            fm,
                            fbus,
                            aegis_core::ops::simd_level_name(),
                            cpu::has_hwp(),
                            cpu::has_turbo(),
                            cpu::msrs_safe()
                        ),
                    );

                    // Segment 1: warmup (unrecorded)
                    say("[GAUNTLET] warmup...\r\n");
                    {
                        let mut _n = 0u64;
                        engine.process_intent("Hello.", 8, |_t| {
                            _n += 1;
                        });
                    }

                    // Segment 2: SIMD value — scalar vs native, same silicon
                    say("[GAUNTLET] seg2 SIMD value (scalar vs vector)...\r\n");
                    aegis_core::ops::set_force_scalar(true);
                    gbench!("SIMD_scalar", ESSAY, 30);
                    aegis_core::ops::set_force_scalar(false);
                    gbench!("SIMD_native", ESSAY, 30);

                    // Segment 3: batching value — per-token prefill vs batched GEMM
                    // (prefill dominates a long prompt; use a long prompt, few new tokens)
                    say("[GAUNTLET] seg3 batching value (prefill)...\r\n");
                    let LONGP = "the quick brown fox jumps over the lazy dog and runs far into the deep forest again and again and again and again and again and again";
                    aegis_core::ops::set_force_legacy_prefill(true);
                    gbench!("PREFILL_pertoken", LONGP, 1);
                    aegis_core::ops::set_force_legacy_prefill(false);
                    gbench!("PREFILL_batched", LONGP, 1);

                    // Segment 4: P-state — drift control, then turbo
                    say("[GAUNTLET] seg4 P-state...\r\n");
                    gbench!("PSTATE_run1", ESSAY, 30);
                    gbench!("PSTATE_run2_control", ESSAY, 30);
                    log_throttle_diag(&mut root, "pre");
                    match cpu::request_max_performance() {
                        Ok(cpu::Boost::Hwp { highest }) => boot_log(
                            &mut root,
                            &format!("GAUNTLET TURBO: HWP highest={}", highest),
                        ),
                        Ok(cpu::Boost::LegacySpeedStep { ratio, mhz }) => boot_log(
                            &mut root,
                            &format!("GAUNTLET TURBO: legacy ratio={} (~{}MHz)", ratio, mhz),
                        ),
                        Err(e) => boot_log(&mut root, &format!("GAUNTLET TURBO FAILED: {}", e)),
                    }
                    let _ = uefi::boot::stall(core::time::Duration::from_millis(100));
                    log_throttle_diag(&mut root, "post");
                    gbench!("PSTATE_run3_turbo", ESSAY, 30);

                    // Segment 5: context slope — tok/s vs generation length
                    say("[GAUNTLET] seg5 context slope...\r\n");
                    gbench!("CTX_20", ESSAY, 20);
                    gbench!("CTX_100", ESSAY, 100);
                    gbench!("CTX_400", ESSAY, 400);

                    let peak_mb = crate::allocator::get_peak_memory() as f64 / (1024.0 * 1024.0);
                    boot_log(
                        &mut root,
                        &format!(
                            "GAUNTLET PEAK_MEMORY: {} bytes ({}.{:02} MB) [Arena High-Water Mark]",
                            crate::allocator::get_peak_memory(),
                            peak_mb as u64,
                            ((peak_mb * 100.0) as u64) % 100
                        ),
                    );
                    boot_log(&mut root, "GAUNTLET DONE");
                    say("\r\n[GAUNTLET] complete. Type /exit and remove the stick.\r\n");
                    continue;
                }

                if cmd == "/parity" {
                    // Prefill/decode parity on THIS machine's silicon: the same
                    // tokens through forward_batch and forward_step must agree.
                    // Evidence for the grant package; logged like everything else.
                    let toks = engine.tokenizer.encode("The quick brown fox jumps over the lazy dog. Paris is the capital of France.");
                    let diff = engine.prefill_decode_parity(&toks);
                    let verdict = if diff == 0.0 {
                        "PASS (bit-identical)"
                    } else if diff <= 1e-4 {
                        "PASS (within 1e-4 FP tolerance)"
                    } else {
                        "FAIL"
                    };
                    boot_log(
                        &mut root,
                        &format!(
                            "PARITY: {} tok, path={}, max|batch-step|={:e}, {}",
                            toks.len(),
                            aegis_core::ops::active_path_name(),
                            diff,
                            verdict
                        ),
                    );
                    let _ = console::with_console(|st| {
                        let _ = write!(st, "  [PARITY] max diff {:e} -> {}\r\n", diff, verdict);
                        core::fmt::Result::Ok(())
                    });
                    continue;
                }

                if cmd.starts_with("/translate ") {
                    let phrase = &cmd[11..];
                    let prompt = alloc::format!(
                        "Translate the following phonetic Arabic into English, or English into phonetic Arabic:\n\n{}\n\nTranslation:\n",
                        phrase
                    );
                    let _ = console::with_console(|st| {
                        let _ = st.write_str("\r\n[ALICE TRANSLATING...]\r\n");
                        core::fmt::Result::Ok(())
                    });

                    engine.process_intent(&prompt, 128, |t| {
                        let _ = console::with_console(|st| {
                            let _ = st.write_str(t);
                            core::fmt::Result::Ok(())
                        });
                    });

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("\r\n\n");
                        core::fmt::Result::Ok(())
                    });
                    continue;
                }

                if cmd == "/autotest" {
                    // The whole P-state protocol, unattended. Seven laptops must run
                    // an identical sequence; a human typing it seven times will not.
                    macro_rules! bench {
                        ($label:expr) => {{
                            let t0 = unsafe { core::arch::x86_64::_rdtsc() };
                            let w0 = wall_seconds();
                            let p0 = cpu::perf_snapshot();
                            let mut n = 0u64;
                            engine.process_intent(
                                "Write a comprehensive and detailed essay about the future of artificial intelligence in aerospace.",
                                50,
                                |t| { n += 1; let _ = console::with_console(|st| { let _ = st.write_str(t); core::fmt::Result::Ok(()) }); },
                            );
                            let p1 = cpu::perf_snapshot();
                            let w1 = wall_seconds();
                            let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
                            let secs = match (w0, w1) { (Some(a), Some(b)) if b >= a => b - a, _ => 0.0 };
                            let tps = if secs > 0.0 { n as f64 / secs } else { 0.0 };
                            let clk = match (p0, p1) { (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b), _ => None };
                            boot_log(&mut root, &format!(
                                "AUTOTEST {}: {} tokens, {} ticks/token, wall {}.{:03}s, {}.{:02} tok/s, clock {}",
                                $label, n,
                                if n > 0 { dt / n } else { 0 },
                                secs as u64, ((secs * 1000.0) as u64) % 1000,
                                tps as u64, ((tps * 100.0) as u64) % 100,
                                match clk { Some(p) => format!("{}%", p), None => alloc::string::String::from("?") }
                            ));
                        }};
                    }

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("\r\n[AUTOTEST] 1/6 identifying CPU...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    let mut bbuf = [0u8; 48];
                    let brand = cpu::brand_string(&mut bbuf);
                    let (fb, fm, fbus) = cpu::frequencies_mhz();
                    let a = cpu::perf_snapshot();
                    let _ = uefi::boot::stall(core::time::Duration::from_secs(1));
                    let b = cpu::perf_snapshot();
                    let idle = match (a, b) {
                        (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b),
                        _ => None,
                    };
                    boot_log(&mut root, "==== AUTOTEST ====");
                    boot_log(
                        &mut root,
                        &format!(
                            "AUTOTEST CPU: {} | base={}MHz max={}MHz bus={}MHz | hwp={} aperf={} baremetal={} | {} | idle_clock={}",
                            brand,
                            fb,
                            fm,
                            fbus,
                            cpu::has_hwp(),
                            cpu::has_aperf_mperf(),
                            cpu::msrs_safe(),
                            match cpu::ratios() {
                                Some((c, b2, t)) =>
                                    format!("ratios cur={} base={} turbo1c={}", c, b2, t),
                                None => alloc::string::String::from("ratios=unreadable"),
                            },
                            match idle {
                                Some(p) => format!("{}%", p),
                                None => alloc::string::String::from("?"),
                            }
                        ),
                    );

                    // A QEMU dry run showed the SECOND benchmark is ~6% faster than the
                    // first even when /turbo fails outright — frequency ramp, TLB, branch
                    // predictors. Without controlling for that, a real turbo effect below
                    // ~10% would be indistinguishable from the artifact.
                    //
                    // So: warm up (unlogged), then run RUN1 and RUN2 with NOTHING changed
                    // between them. Their ratio is the drift. Only then raise the P-state
                    // and run RUN3. The turbo effect is RUN3/RUN2, and it must exceed the
                    // measured drift to mean anything.
                    let _ = console::with_console(|st| {
                        let _ = st.write_str("[AUTOTEST] 2/6 warmup (not recorded)...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    {
                        let mut _n = 0u64;
                        engine.process_intent("Hello.", 8, |_t| {
                            _n += 1;
                        });
                    }

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("[AUTOTEST] 3/6 RUN1 (baseline)...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    bench!("RUN1_baseline");

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("\r\n[AUTOTEST] 4/6 RUN2 (repeat, nothing changed -> measures drift)...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    bench!("RUN2_control");

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("\r\n[AUTOTEST] 5/6 requesting max P-state...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    match cpu::request_max_performance() {
                        Ok(cpu::Boost::Hwp { highest }) => boot_log(
                            &mut root,
                            &format!("AUTOTEST TURBO: HWP ok, highest={}", highest),
                        ),
                        Ok(cpu::Boost::LegacySpeedStep { ratio, mhz }) => boot_log(
                            &mut root,
                            &format!("AUTOTEST TURBO: legacy ratio={} (~{}MHz)", ratio, mhz),
                        ),
                        Err(e) => boot_log(&mut root, &format!("AUTOTEST TURBO FAILED: {}", e)),
                    }

                    let _ = console::with_console(|st| {
                        let _ = st.write_str("[AUTOTEST] 6/6 RUN3 (after turbo)...\r\n");
                        core::fmt::Result::Ok(())
                    });
                    bench!("RUN3_turbo");

                    boot_log(
                        &mut root,
                        "AUTOTEST DONE  (effect = RUN3/RUN2; drift = RUN2/RUN1; effect must exceed drift)",
                    );
                    let _ = console::with_console(|st| {
                        let _ = st.write_str(
                            "\r\n[AUTOTEST] complete. Type /exit, then remove the stick.\r\n",
                        );
                        core::fmt::Result::Ok(())
                    });
                    continue;
                }

                if cmd == "/cpuinfo" {
                    let mut buf = [0u8; 48];
                    let brand = cpu::brand_string(&mut buf);
                    let (base, max, bus) = cpu::frequencies_mhz();

                    // Measure the actual P-state over a 1-second stall.
                    let a = cpu::perf_snapshot();
                    let _ = uefi::boot::stall(core::time::Duration::from_secs(1));
                    let b = cpu::perf_snapshot();
                    let pct = match (a, b) {
                        (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b),
                        _ => None,
                    };

                    let ratio_str = match cpu::ratios() {
                        Some((cur, base_r, turbo)) => format!(
                            "  P-state ratio: current {} ({} MHz), base {} ({} MHz), 1-core turbo {} ({} MHz)\r\n",
                            cur,
                            cpu::mhz_from_ratio(cur),
                            base_r,
                            cpu::mhz_from_ratio(base_r),
                            turbo,
                            cpu::mhz_from_ratio(turbo)
                        ),
                        None => alloc::string::String::from(
                            "  P-state ratio: unreadable (hypervisor or non-Intel)\r\n",
                        ),
                    };
                    let msg = format!(
                        "CPU: {}\r\n  base {} MHz, max {} MHz, bus {} MHz\r\n  APERF/MPERF: {}   HWP: {}   bare-metal Intel: {}\r\n{}  actual clock: {} of nominal\r\n",
                        brand,
                        base,
                        max,
                        bus,
                        if cpu::has_aperf_mperf() { "yes" } else { "no" },
                        if cpu::has_hwp() { "yes" } else { "no" },
                        if cpu::msrs_safe() { "yes" } else { "no" },
                        ratio_str,
                        match pct {
                            Some(p) => format!("{}%", p),
                            None => alloc::string::String::from("unmeasurable"),
                        },
                    );
                    let _ = console::with_console(|st| {
                        let _ = st.write_str(&msg);
                        core::fmt::Result::Ok(())
                    });
                    boot_log(
                        &mut root,
                        &format!(
                            "CPUINFO: {} | cpuid base={}MHz max={}MHz bus={}MHz | hwp={} aperf={} | {} | idle_clock={}%",
                            brand,
                            base,
                            max,
                            bus,
                            cpu::has_hwp(),
                            cpu::has_aperf_mperf(),
                            match cpu::ratios() {
                                Some((c, b, t)) =>
                                    format!("ratios cur={} base={} turbo1c={}", c, b, t),
                                None => alloc::string::String::from("ratios=unreadable"),
                            },
                            match pct {
                                Some(p) => p,
                                None => 0,
                            }
                        ),
                    );
                    continue;
                }

                if cmd == "/turbo" {
                    log_throttle_diag(&mut root, "pre");
                    match cpu::request_max_performance() {
                        Ok(cpu::Boost::Hwp { highest }) => {
                            let m = format!(
                                "HWP (Speed Shift): requested max performance, level {}.\r\n  Now re-run /benchmark and compare.\r\n",
                                highest
                            );
                            let _ = console::with_console(|st| {
                                let _ = st.write_str(&m);
                                core::fmt::Result::Ok(())
                            });
                            boot_log(
                                &mut root,
                                &format!("TURBO: HWP max requested, highest_perf={}", highest),
                            );
                        }
                        Ok(cpu::Boost::LegacySpeedStep { ratio, mhz }) => {
                            let m = format!(
                                "Legacy SpeedStep: requested ratio {} (~{} MHz).\r\n  Now re-run /benchmark and compare.\r\n",
                                ratio, mhz
                            );
                            let _ = console::with_console(|st| {
                                let _ = st.write_str(&m);
                                core::fmt::Result::Ok(())
                            });
                            boot_log(
                                &mut root,
                                &format!("TURBO: legacy SpeedStep ratio={} (~{}MHz)", ratio, mhz),
                            );
                        }
                        Err(e) => {
                            let m = format!("Cannot raise P-state: {}\r\n", e);
                            let _ = console::with_console(|st| {
                                let _ = st.write_str(&m);
                                core::fmt::Result::Ok(())
                            });
                            boot_log(&mut root, &format!("TURBO FAILED: {}", e));
                        }
                    }
                    let _ = uefi::boot::stall(core::time::Duration::from_millis(100));
                    log_throttle_diag(&mut root, "post");
                    continue;
                }

                if let Some(n) = cmd.strip_prefix("/tokens ") {
                    if let Ok(v) = n.trim().parse::<usize>() {
                        if v > 0 && v <= 4096 {
                            max_new_tokens = v;
                            let _ = console::with_console(|st| {
                                let _ = write!(st, "max_new_tokens = {}\r\n", v);
                                core::fmt::Result::Ok(())
                            });
                            continue;
                        }
                    }
                    let _ = console::with_console(|st| {
                        let _ = st.write_str("usage: /tokens <1..4096>\r\n");
                        core::fmt::Result::Ok(())
                    });
                    continue;
                }

                if cmd == "/benchmark" {
                    let _ = console::with_console(|st| {
                        let _ = st.write_str(
                            "[BENCHMARK] Warming up Matrix Core... Running speed test.\r\n",
                        );
                        core::fmt::Result::Ok(())
                    });

                    let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };
                    let t_wall0 = wall_seconds();
                    let p0 = cpu::perf_snapshot();
                    let mut token_count = 0;
                    engine.process_intent("Write a comprehensive and detailed essay about the future of artificial intelligence in aerospace.", 50, |t| {
                        token_count += 1;
                        let _ = console::with_console(|st| { let _ = st.write_str(t); core::fmt::Result::Ok(()) });
                    });
                    let p1 = cpu::perf_snapshot();
                    let t_wall1 = wall_seconds();
                    let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };

                    let diff = end_tsc - start_tsc;
                    let cycles_per_token = if token_count > 0 {
                        diff / token_count as u64
                    } else {
                        0
                    };
                    let secs = match (t_wall0, t_wall1) {
                        (Some(a), Some(b)) if b >= a => b - a,
                        _ => 0.0,
                    };
                    let clock_pct = match (p0, p1) {
                        (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b),
                        _ => None,
                    };

                    let tps = if secs > 0.0 {
                        token_count as f64 / secs
                    } else {
                        0.0
                    };
                    let clock_str = match clock_pct {
                        Some(p) => format!("{}%", p),
                        None => alloc::string::String::from("unmeasurable"),
                    };
                    let _ = console::with_console(|st| {
                        let msg = alloc::format!(
                            "\r\n\n[BENCHMARK RESULTS]\r\n* Tokens: {}\r\n* TSC ticks: {} ({} ticks/token)\r\n* WALL TIME: {}.{:03} s -> {}.{:02} tok/s\r\n* ACTUAL CLOCK: {} of nominal\r\n\n",
                            token_count,
                            diff,
                            cycles_per_token,
                            secs as u64,
                            ((secs * 1000.0) as u64) % 1000,
                            tps as u64,
                            ((tps * 100.0) as u64) % 100,
                            clock_str
                        );
                        let _ = st.write_str(&msg);
                        core::fmt::Result::Ok(())
                    });
                    // Persist the measurement to the stick — a benchmark nobody
                    // recorded is a benchmark that did not happen.
                    boot_log(
                        &mut root,
                        &format!(
                            "BENCHMARK: {} tokens, {} ticks, {} ticks/token, wall {}.{:03}s, {}.{:02} tok/s, clock {} of nominal",
                            token_count,
                            diff,
                            cycles_per_token,
                            secs as u64,
                            ((secs * 1000.0) as u64) % 1000,
                            tps as u64,
                            ((tps * 100.0) as u64) % 100,
                            clock_str
                        ),
                    );
                    continue;
                }

                // Record prompt, response, and speed to BOOTLOG.TXT on the boot
                // volume. Real hardware has no serial console to scrape, and an
                // unrecorded generation cannot be cited later.
                let _ = console::with_console(|st| {
                    let _ = write!(st, "[prompt: {:?}, max {} tokens]\r\n", cmd, max_new_tokens);
                    core::fmt::Result::Ok(())
                });
                boot_log(&mut root, &format!("PROMPT: {:?}", cmd));
                let t0 = unsafe { core::arch::x86_64::_rdtsc() };
                let w0 = wall_seconds();
                let g0 = cpu::perf_snapshot();
                let mut ntok: u64 = 0;
                let response = engine.process_intent(cmd, max_new_tokens, |token_str| {
                    if !token_str.starts_with("[SYSTEM]") && !token_str.contains("[PERFORMANCE]") {
                        ntok += 1;
                    }
                    let _ = console::with_console(|st| {
                        let _ = st.write_str(token_str);
                        core::fmt::Result::Ok(())
                    });
                });
                let g1 = cpu::perf_snapshot();
                let w1 = wall_seconds();
                let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
                let gsecs = match (w0, w1) {
                    (Some(a), Some(b)) if b >= a => b - a,
                    _ => 0.0,
                };
                let gtps = if gsecs > 0.0 {
                    ntok as f64 / gsecs
                } else {
                    0.0
                };
                let gclock = match (g0, g1) {
                    (Some(a), Some(b)) => cpu::actual_pct_of_nominal(a, b),
                    _ => None,
                };
                boot_log(&mut root, &format!("RESPONSE: {}", response));
                boot_log(
                    &mut root,
                    &format!(
                        "  ({} tokens, {} ticks, {} ticks/token, wall {}.{:03}s, {}.{:02} tok/s, clock {} of nominal)",
                        ntok,
                        dt,
                        if ntok > 0 { dt / ntok } else { 0 },
                        gsecs as u64,
                        ((gsecs * 1000.0) as u64) % 1000,
                        gtps as u64,
                        ((gtps * 100.0) as u64) % 100,
                        match gclock {
                            Some(p) => format!("{}%", p),
                            None => alloc::string::String::from("?"),
                        }
                    ),
                );
                let _ = console::with_console(|st| {
                    let _ = st.write_str("\r\n\r\n");
                    core::fmt::Result::Ok(())
                });
            }
        }
    }

    uefi::Status::SUCCESS
}

#[panic_handler]
fn panic_handler(info: &core::panic::PanicInfo) -> ! {
    let _ = console::with_console(|st| {
        let _ = st.write_str("\r\n\r\n*** KERNEL PANIC ***\r\n");
        use core::fmt::Write;
        let _ = write!(st, "{}\r\n", info);
        core::fmt::Result::Ok(())
    });

    #[cfg(feature = "qemu-test")]
    unsafe {
        exit_uefi_test_runner(false);
    }

    loop {
        core::arch::x86_64::_mm_pause();
    }
}
