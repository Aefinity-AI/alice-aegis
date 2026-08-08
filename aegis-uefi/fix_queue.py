import os

idt_rs = """use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};
use pic8259::ChainedPics;
use spin::{Mutex, LazyLock};
use core::arch::asm;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::cell::UnsafeCell;

pub const PIC_1_OFFSET: u8 = 32;
pub const PIC_2_OFFSET: u8 = PIC_1_OFFSET + 8;

pub static PICS: Mutex<ChainedPics> = Mutex::new(unsafe { ChainedPics::new(PIC_1_OFFSET, PIC_2_OFFSET) });

pub static IDT: LazyLock<InterruptDescriptorTable> = LazyLock::new(|| {
    let mut idt = InterruptDescriptorTable::new();
    idt.double_fault.set_handler_fn(double_fault_handler);
    idt[PIC_1_OFFSET as usize + 1].set_handler_fn(keyboard_interrupt_handler);
    idt
});

pub fn init() {
    IDT.load();
    unsafe { PICS.lock().initialize(); }
    x86_64::instructions::interrupts::enable(); // STI
}

extern "x86-interrupt" fn double_fault_handler(stack_frame: InterruptStackFrame, _error_code: u64) -> ! {
    panic!("DOUBLE FAULT\\n{:#?}", stack_frame);
}

pub struct RingBuffer {
    buffer: UnsafeCell<[char; 256]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    pub const fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(['\\0'; 256]),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, c: char) {
        let head = self.head.load(Ordering::Acquire);
        unsafe {
            (*self.buffer.get())[head % 256] = c;
        }
        self.head.store(head.wrapping_add(1), Ordering::Release);
    }
    
    pub fn pop(&self) -> Option<char> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        if head == tail {
            None
        } else {
            let c = unsafe { (*self.buffer.get())[tail % 256] };
            self.tail.store(tail.wrapping_add(1), Ordering::Release);
            Some(c)
        }
    }
}

pub static KEY_QUEUE: RingBuffer = RingBuffer::new();

fn inb(port: u16) -> u8 {
    let mut data: u8;
    unsafe { asm!("in al, dx", out("al") data, in("dx") port); }
    data
}

extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
    let scancode = inb(0x60);
    if scancode < 0x80 { 
        let c = match scancode {
            0x1C => '\\n', 
            0x39 => ' ', 
            0x0E => '\\x08', 
            0x1E => 'A', 0x30 => 'B', 0x2E => 'C', 0x20 => 'D', 0x12 => 'E',
            0x21 => 'F', 0x22 => 'G', 0x23 => 'H', 0x17 => 'I', 0x24 => 'J',
            0x25 => 'K', 0x26 => 'L', 0x32 => 'M', 0x31 => 'N', 0x18 => 'O',
            0x19 => 'P', 0x10 => 'Q', 0x13 => 'R', 0x1F => 'S', 0x14 => 'T',
            0x16 => 'U', 0x2F => 'V', 0x11 => 'W', 0x2D => 'X', 0x15 => 'Y', 0x2C => 'Z',
            _ => '?'
        };
        KEY_QUEUE.push(c); 
    }
    unsafe {
        PICS.lock().notify_end_of_interrupt(PIC_1_OFFSET + 1);
    }
}
"""

with open("/home/killboxincorporated/aegis-uefi/src/idt.rs", "w") as f:
    f.write(idt_rs.replace("\\\\n", "\\n").replace("\\\\0", "\\0").replace("\\\\x08", "\\x08"))

with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "r") as f:
    main_rs = f.read()

main_rs = main_rs.replace("let key = unsafe { idt::KEY_QUEUE.pop() };", "let key = idt::KEY_QUEUE.pop();")

with open("/home/killboxincorporated/aegis-uefi/src/main.rs", "w") as f:
    f.write(main_rs)

print("IDT RingBuffer deadlock fix applied.")
