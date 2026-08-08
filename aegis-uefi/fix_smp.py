import os

smp_rs = """use uefi::proto::pi::mp::{MpServices, ProcedureArgument};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub static AP_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static AP_AWAKE: AtomicBool = AtomicBool::new(false);

extern "efiapi" fn ap_entry_point(_arg: ProcedureArgument) {
    AP_COUNT.fetch_add(1, Ordering::SeqCst);
    loop {
        if AP_AWAKE.load(Ordering::Acquire) {
            core::hint::spin_loop();
        } else {
            core::hint::spin_loop();
        }
    }
}

pub fn init() -> Result<usize, uefi::Status> {
    if let Ok(mp_handle) = uefi::boot::get_handle_for_protocol::<MpServices>() {
        if let Ok(mut mp) = uefi::boot::open_protocol_exclusive::<MpServices>(mp_handle) {
            let info = mp.get_number_of_processors().unwrap();
            return Ok(info.total);
        }
    }
    Ok(1)
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/smp.rs", "w") as f:
    f.write(smp_rs)
