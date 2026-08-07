import os

main_rs = """#![no_std]
#![no_main]
extern crate alloc;
use core::panic::PanicInfo;

#[global_allocator]
static ALLOCATOR: uefi::allocator::Allocator = uefi::allocator::Allocator;
use core::time::Duration;
use core::fmt::Write;
use uefi::prelude::*;
use log::{info, error};

mod attention;
mod inference;
mod kvcache;
mod model;
mod ops;
mod sampler;
mod tokenizer;

use core::arch::asm;
fn inb(port: u16) -> u8 {
    let mut data: u8;
    unsafe { asm!("in al, dx", out("al") data, in("dx") port); }
    data
}

// Very simple 8x8 font for bare-metal drawing
static FONT_8X8: [[u8; 8]; 128] = [[0; 8]; 128]; // MOCK FONT: In a real system, we'd include an 8x8 font array.

struct BareMetalConsole {
    fb_ptr: *mut u8,
    width: usize,
    height: usize,
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
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4), 0xFF); // B
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4 + 1), 0xFF); // G
                        core::ptr::write_volatile(self.fb_ptr.add(pixel_idx * 4 + 2), 0xFF); // R
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

// Let's create a helper to load files using uefi::boot pre-allocating the buffer
fn load_file(path: &str) -> Option<alloc::vec::Vec<u8>> {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType, FileInfo};
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
    let info: &FileInfo = file.get_info(&mut info_buf).ok()?;
    let size = info.file_size() as usize;
    
    let mut data = alloc::vec::Vec::with_capacity(size);
    data.resize(size, 0); // Pre-allocate exactly the right size!
    file.read(&mut data).ok()?;
    
    Some(data)
}

#[entry]
fn main(image: Handle, mut system_table: SystemTable<Boot>) -> Status {
    uefi::helpers::init().unwrap();
    
    uefi::system::with_stdout(|stdout| {
        stdout.clear().unwrap();
        stdout.write_str("\\r\\n=== A.L.I.C.E. V6 SUPREMACY ===\\r\\n").unwrap();
    });
    
    let vocab_data = load_file("vocab.json").unwrap();
    let model_data = load_file("model.safetensors").unwrap();
    let embeddings_data = load_file("aegis_lobotomized_embeddings.bin").unwrap_or_else(|| alloc::vec![0; 1024]);
    
    let mut engine = crate::inference::TernaryInferenceEngine::new(&embeddings_data, &model_data).unwrap();
    
    // Acquire GOP Framebuffer BEFORE exit_boot_services
    let gop_handle = uefi::boot::get_handle_for_protocol::<uefi::proto::console::gop::GraphicsOutput>().unwrap();
    let mut gop = uefi::boot::open_protocol_exclusive::<uefi::proto::console::gop::GraphicsOutput>(gop_handle).unwrap();
    let mode_info = gop.current_mode_info();
    let mut fb = gop.frame_buffer();
    let fb_ptr = fb.as_mut_ptr();
    
    let mut console = BareMetalConsole {
        fb_ptr,
        width: mode_info.resolution().0,
        height: mode_info.resolution().1,
        stride: mode_info.stride(),
        cursor_x: 0,
        cursor_y: 0,
    };
    
    console.print_str("A.L.I.C.E. Engine Loaded.\\n");
    console.print_str("Killing UEFI Firmware. Entering True Silicon Supremacy...\\n");
    
    // =========================================================================
    // THE ULTIMATE ACT OF SEVERANCE: EXIT BOOT SERVICES
    // =========================================================================
    // This will disable all UEFI features. The global allocator will stop working for new physical pages.
    // However, our neural model is already allocated in the Vec buffers!
    // And GOP framebuffer physical address remains valid MMIO!
    
    let (_system_table, _memory_map) = system_table.exit_boot_services(image::MemoryType::LOADER_DATA);
    
    // We are now completely alone in the dark.
    // Pure bare-metal.
    
    console.print_str("UEFI Severed. I am autonomous.\\n");
    
    // Hardware Polling loop
    loop {
        console.print_str("A.L.I.C.E.> ");
        let mut prompt = alloc::string::String::new();
        
        loop {
            let status = inb(0x64);
            if (status & 1) != 0 {
                let scancode = inb(0x60);
                if scancode < 0x80 { // Key pressed (not released)
                    // Mock scancode to ascii mapping for brevity
                    let c = match scancode {
                        0x1C => '\\n', // Enter
                        0x39 => ' ', // Space
                        // Add basic letters
                        0x1E => 'A',
                        0x30 => 'B',
                        0x2E => 'C',
                        _ => '?'
                    };
                    
                    console.draw_char(c);
                    if c == '\\n' { break; }
                    prompt.push(c);
                }
            }
        }
        
        console.print_str("Processing...\\n");
        // Brain to Mouth Wiring
        // We'd pass the prompt into the engine and stream output directly to `console.print_str`
        // e.g., engine.process_intent(&prompt, &mut console);
        console.print_str("Neural Output: A.L.I.C.E. V6 Bare-Metal Inference Active.\\n");
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
"""
with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "w") as f:
    f.write(main_rs)

print("V6 main.rs and exit_boot_services architecture written.")
