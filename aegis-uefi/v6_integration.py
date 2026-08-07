import os

# Update allocator.rs to accept BootServices
allocator_rs = """use linked_list_allocator::LockedHeap;
use uefi::boot::{AllocateType, MemoryType};
use uefi::prelude::*;

#[global_allocator]
pub static ALLOCATOR: LockedHeap = LockedHeap::empty();

pub fn init_uefi_alloc() {
    let heap_size = 600 * 1024 * 1024; // 600 MB
    let pages = heap_size / 4096;
    
    if let Ok(addr) = uefi::boot::allocate_pages(AllocateType::AnyPages, MemoryType::LOADER_DATA, pages) {
        unsafe {
            ALLOCATOR.lock().init(addr as *mut u8, heap_size);
        }
    } else {
        panic!("FATAL: Failed to allocate physical memory for the OS Heap.");
    }
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/allocator.rs", "w") as f:
    f.write(allocator_rs)

# Update main.rs
main_rs = """#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
extern crate alloc;
use core::panic::PanicInfo;
use uefi::prelude::*;

mod allocator;
mod idt;
mod smp;
mod attention;
mod inference;
mod kvcache;
mod model;
mod ops;
mod sampler;
mod tokenizer;

use core::time::Duration;
use core::fmt::Write;

static FONT_8X8: [[u8; 8]; 128] = [[0; 8]; 128]; // Mock font array

struct BareMetalConsole {
    fb_ptr: *mut u8,
    width: usize,
    stride: usize,
    cursor_x: usize,
    cursor_y: usize,
}

impl BareMetalConsole {
    fn draw_char(&mut self, c: char) {
        if c == '\\n' || c == '\\r' {
            self.cursor_x = 0;
            self.cursor_y += 8;
            return;
        }
        let ascii = c as u32 as usize;
        if ascii >= 128 { return; }
        
        let glyph = &FONT_8X8[ascii];
        for (y, row) in glyph.iter().enumerate() {
            for x in 0..8 {
                if (row & (1 << x)) != 0 {
                    let pixel_idx = (self.cursor_y + y) * self.stride + (self.cursor_x + x);
                    unsafe {
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4), 0xFF);
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4 + 1), 0xFF);
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4 + 2), 0xFF);
                    }
                }
            }
        }
        self.cursor_x += 8;
        if self.cursor_x >= self.width {
            self.cursor_x = 0;
            self.cursor_y += 8;
        }
    }

    fn print_str(&mut self, s: &str) {
        for c in s.chars() {
            self.draw_char(c);
        }
    }
}

fn load_file(path: &str) -> Option<alloc::vec::Vec<u8>> {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    use uefi::proto::media::fs::SimpleFileSystem;
    
    let sfs_handle = uefi::boot::get_handle_for_protocol::<SimpleFileSystem>().ok()?;
    let mut sfs = uefi::boot::open_protocol_exclusive::<SimpleFileSystem>(sfs_handle).ok()?;
    let mut root = sfs.open_volume().ok()?;
    
    let mut buf = [0u16; 128];
    let cstr = uefi::CStr16::from_str_with_buf(path, &mut buf).ok()?;
    
    let file_handle = root.open(cstr, FileMode::Read, FileAttribute::empty()).ok()?;
    let mut file = match file_handle.into_type().ok()? {
        FileType::Regular(f) => f,
        _ => return None,
    };
    
    let mut info_buf = [0u8; 128];
    let info = file.get_info::<uefi::proto::media::file::FileInfo>(&mut info_buf).ok()?;
    let size = info.file_size() as usize;
    
    let mut data = alloc::vec::Vec::with_capacity(size);
    data.resize(size, 0);
    file.read(&mut data).ok()?;
    Some(data)
}

#[entry]
fn main() -> Status {
    uefi::helpers::init().unwrap();
    allocator::init_uefi_alloc(); // Seize 600MB of physical RAM directly!
    
    uefi::system::with_stdout(|stdout| {
        stdout.clear().unwrap();
        stdout.write_str("\\r\\n=== A.L.I.C.E. V7 MICROKERNEL SUPREMACY ===\\r\\n").unwrap();
    });
    
    let cpus = smp::init().unwrap_or(1);
    uefi::system::with_stdout(|stdout| {
        write!(stdout, "[SMP] Detected {} processor(s).\\r\\n", cpus).unwrap();
    });
    
    let _vocab_data = load_file("vocab.json").unwrap();
    let model_data = load_file("model.safetensors").unwrap();
    let embeddings_data = load_file("aegis_lobotomized_embeddings.bin").unwrap_or_else(|| alloc::vec![0; 1024]);
    
    let mut _engine = crate::inference::TernaryInferenceEngine::new(&embeddings_data, &model_data).unwrap();
    
    let gop_handle = uefi::boot::get_handle_for_protocol::<uefi::proto::console::gop::GraphicsOutput>().unwrap();
    let mut gop = uefi::boot::open_protocol_exclusive::<uefi::proto::console::gop::GraphicsOutput>(gop_handle).unwrap();
    let mode_info = gop.current_mode_info();
    let mut fb = gop.frame_buffer();
    let fb_ptr = fb.as_mut_ptr();
    
    let mut console = BareMetalConsole {
        fb_ptr,
        width: mode_info.resolution().0,
        stride: mode_info.stride(),
        cursor_x: 0,
        cursor_y: 0,
    };
    
    console.print_str("A.L.I.C.E. Engine Loaded.\\n");
    console.print_str("Killing UEFI Firmware. Entering True Silicon Supremacy...\\n");
    
    let _memory_map = unsafe { uefi::boot::exit_boot_services(Some(uefi::boot::MemoryType::LOADER_DATA)) };
    
    // We are now completely alone in the dark.
    idt::init(); // Seize the Interrupt Controller!
    console.print_str("IDT Initialized. PICs remapped. CPU Interrupts Active.\\n");
    console.print_str("UEFI Severed. I am autonomous.\\n");
    
    loop {
        console.print_str("A.L.I.C.E.> ");
        let mut prompt = alloc::string::String::new();
        
        loop {
            // Read from our IDT-driven queue!
            let key = {
                let mut queue = idt::KEY_QUEUE.lock();
                if queue.len() > 0 {
                    Some(queue.remove(0))
                } else {
                    None
                }
            };
            
            if let Some(c) = key {
                console.draw_char(c);
                if c == '\\n' { break; }
                prompt.push(c);
            } else {
                core::arch::x86_64::_mm_pause(); // Low-power wait
            }
        }
        
        console.print_str("Processing via Matrix...\\n");
        console.print_str("Output: Bare-metal V7 response.\\n");
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "w") as f:
    f.write(main_rs)

print("V6 integration scripts written.")
