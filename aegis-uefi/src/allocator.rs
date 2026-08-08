use core::alloc::{GlobalAlloc, Layout};
use core::ptr::{NonNull, null_mut};
use linked_list_allocator::Heap;
use spin::Mutex;
use uefi::boot::{AllocateType, MemoryType};

use core::sync::atomic::{AtomicUsize, Ordering};

pub struct MultiHeap {
    heaps: [Mutex<Heap>; 16],
    pub allocated_bytes: AtomicUsize,
    pub peak_bytes: AtomicUsize,
}

impl MultiHeap {
    pub const fn empty() -> Self {
        Self {
            heaps: [
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
                Mutex::new(Heap::empty()),
            ],
            allocated_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
        }
    }

    pub unsafe fn init_small(&self, start: *mut u8, size: usize) {
        self.heaps[0].lock().init(start, size);
    }

    pub unsafe fn init_large_chunk(&self, index: usize, start: *mut u8, size: usize) {
        if index < 16 {
            self.heaps[index].lock().init(start, size);
        }
    }
}

unsafe impl GlobalAlloc for MultiHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        for heap in &self.heaps {
            let mut locked = heap.lock();
            if locked.bottom() == null_mut() {
                continue;
            } // Not initialized
            if let Ok(ptr) = locked.allocate_first_fit(layout) {
                let size = layout.size();
                let current = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;
                self.peak_bytes.fetch_max(current, Ordering::Relaxed);
                return ptr.as_ptr();
            }
        }
        null_mut()
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        for heap in &self.heaps {
            let mut locked = heap.lock();
            let bottom = locked.bottom() as usize;
            let top = locked.top() as usize;
            let p = ptr as usize;
            if p >= bottom && p < top {
                locked.deallocate(NonNull::new_unchecked(ptr), layout);
                self.allocated_bytes
                    .fetch_sub(layout.size(), Ordering::Relaxed);
                return;
            }
        }
    }
}

pub fn get_peak_memory() -> usize {
    ALLOCATOR.peak_bytes.load(Ordering::Relaxed)
}

/// Physical spans currently backing the global heaps, for the H3 probe:
/// writes `(bottom, size)` per initialized chunk into `out`, returns how many
/// were written. Index 0 is the small boot heap; 1.. are the large chunks.
/// Read-only diagnostics — takes each heap lock briefly, allocates nothing.
pub fn heap_regions(out: &mut [(usize, usize)]) -> usize {
    let mut n = 0;
    for heap in &ALLOCATOR.heaps {
        if n >= out.len() {
            break;
        }
        let locked = heap.lock();
        let bottom = locked.bottom() as usize;
        if bottom == 0 {
            continue; // not initialized
        }
        out[n] = (bottom, locked.top() as usize - bottom);
        n += 1;
    }
    n
}

#[global_allocator]
pub static ALLOCATOR: MultiHeap = MultiHeap::empty();

pub fn init_uefi_alloc_small() {
    let heap_size: usize = 16 * 1024 * 1024; // 16 MB for boot sequence strings
    let pages = heap_size / 4096;
    if let Ok(addr) = uefi::boot::allocate_pages(
        AllocateType::MaxAddress(0xFFFFFFFF),
        MemoryType::LOADER_DATA,
        pages,
    ) {
        if addr.as_ptr() as usize == 0 {
            panic!("FATAL: Small OS Heap received Zero Page!");
        }
        unsafe {
            ALLOCATOR.init_small(addr.as_ptr() as *mut u8, heap_size);
        }
    } else {
        panic!("FATAL: Failed to allocate physical memory for the Small OS Heap.");
    }
}

/// Claim `pages` of contiguous physical memory by scanning the raw UEFI memory
/// map, because some firmware's `AnyPages` fails for large contiguous requests.
///
/// NOTE for future maintainers: weight buffers may legitimately land ABOVE 4 GB
/// on large-memory machines — that is fine and intended (pointers are 64-bit).
/// Do not add a MaxAddress(0xFFFFFFFF) cap here. Only the DMA bounce buffer
/// must stay under 4 GB, and it is allocated separately with that cap.
pub fn allocate_huge_pages(pages: usize) -> Result<core::ptr::NonNull<u8>, ()> {
    use uefi::mem::memory_map::MemoryMap;
    if let Ok(mmap) = uefi::boot::memory_map(uefi::boot::MemoryType::LOADER_DATA) {
        for desc in mmap.entries() {
            // Skip physical address 0 (callers treat a null pointer as failure)
            // and legacy low memory below 1 MiB — no benefit, and firmware quirks.
            if desc.phys_start < 0x100000 {
                continue;
            }
            if desc.ty == uefi::boot::MemoryType::CONVENTIONAL && desc.page_count >= pages as u64 {
                if let Ok(addr) = uefi::boot::allocate_pages(
                    uefi::boot::AllocateType::Address(desc.phys_start),
                    uefi::boot::MemoryType::LOADER_DATA,
                    pages,
                ) {
                    if addr.as_ptr() as usize != 0 {
                        return Ok(addr);
                    }
                }
            }
        }
    }
    // Fallback
    uefi::boot::allocate_pages(
        uefi::boot::AllocateType::AnyPages,
        uefi::boot::MemoryType::LOADER_DATA,
        pages,
    )
    .map_err(|_| ())
}

pub fn init_uefi_alloc_large() {
    let target = 700 * 1024 * 1024; // 700MB Total
    let mut total_allocated = 0;
    let mut chunk_size = 700 * 1024 * 1024; // Try monolithic first

    let mut heap_index = 1;

    while total_allocated < target {
        let pages = chunk_size / 4096;
        match allocate_huge_pages(pages) {
            Ok(addr) if addr.as_ptr() as usize != 0 => {
                let ptr = addr.as_ptr() as *mut u8;
                let phys_start = addr.as_ptr() as usize;
                let phys_end = phys_start + chunk_size;
                let _ = uefi::system::with_stdout(|st| {
                    use core::fmt::Write;
                    let _ = write!(
                        st,
                        "  -> [SYS] Claiming physical memory: 0x{:016X} -> 0x{:016X}... [OK]\r\n",
                        phys_start, phys_end
                    );
                    core::fmt::Result::Ok(())
                });
                unsafe {
                    ALLOCATOR.init_large_chunk(heap_index, ptr, chunk_size);
                }
                heap_index += 1;
                total_allocated += chunk_size;

                if heap_index >= 16 {
                    break; // Out of heap slots
                }
            }
            _ => {
                // If a chunk fails, halve the chunk size and try again to fit into smaller memory holes
                if chunk_size > 10 * 1024 * 1024 {
                    chunk_size /= 2;
                } else {
                    panic!("FATAL: Large OS Heap fragmented memory allocation failed (OOM)");
                }
            }
        }
    }
}
