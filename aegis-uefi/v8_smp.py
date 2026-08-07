import os
import subprocess

smp_rs_content = """use uefi::proto::pi::mp::MpServices;
use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use core::ffi::c_void;

pub static AP_COUNT: AtomicUsize = AtomicUsize::new(0);

pub struct MatVecJob {
    pub output: *mut f32,
    pub input: *const f32,
    pub weights: *const u8,
    pub dim_out: usize,
    pub dim_in: usize,
    pub chunks_done: AtomicUsize,
    pub total_chunks: usize,
}
unsafe impl Send for MatVecJob {}
unsafe impl Sync for MatVecJob {}

pub static CURRENT_JOB: AtomicPtr<MatVecJob> = AtomicPtr::new(core::ptr::null_mut());

extern "efiapi" fn ap_park(_arg: *mut c_void) {
    AP_COUNT.fetch_add(1, Ordering::SeqCst);
    loop {
        let job_ptr = CURRENT_JOB.load(Ordering::Acquire);
        if !job_ptr.is_null() {
            unsafe {
                let job = &*job_ptr;
                // AP processes chunks here in future
                job.chunks_done.fetch_add(1, Ordering::SeqCst);
                
                // Wait for job to be cleared by BSP
                while CURRENT_JOB.load(Ordering::Acquire) == job_ptr {
                    core::hint::spin_loop();
                }
            }
        } else {
            core::hint::spin_loop();
        }
    }
}

pub fn init() -> Result<usize, uefi::Status> {
    if let Ok(mp_handle) = uefi::boot::get_handle_for_protocol::<MpServices>() {
        if let Ok(mp) = uefi::boot::open_protocol_exclusive::<MpServices>(mp_handle) {
            let info = mp.get_number_of_processors().unwrap();
            
            // In uefi crate, startup_all_aps might require unsafe or mut
            // Let's attempt the standard signature.
            let _ = mp.startup_all_aps(false, ap_park as extern "efiapi" fn(*mut c_void), core::ptr::null_mut(), None, None);
            
            return Ok(info.total);
        }
    }
    Ok(1)
}
"""

with open("src/smp.rs", "w") as f:
    f.write(smp_rs_content)

res = subprocess.run(["cargo", "check", "--target", "x86_64-unknown-uefi"], capture_output=True, text=True)
if res.returncode == 0:
    print("SMP compiled successfully.")
else:
    print("SMP compilation failed:")
    print(res.stderr)
