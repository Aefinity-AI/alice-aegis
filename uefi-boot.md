# UEFI boot & hardware ground truth

This is the strongest part of the project and the easiest to break by "cleaning
up." Each pattern below was earned by debugging real firmware (Acer Core 3 DDR5
test box, plus an older Dell). Before simplifying anything that looks odd here,
read why it exists.

## Boot sequence (aegis-uefi/src/main.rs)

1. Disable the UEFI watchdog (`set_watchdog_timer(0,0,None)`) — otherwise firmware
   reboots the box after ~5 minutes of "unresponsive" app during weight loading.
2. Init the **small** (32MB) heap only — the large heap comes *after* file loading
   so the huge contiguous regions go to weights first.
3. Enable AVX2: set CR4 bits 9/10/18 (OSFXSR, OSXMMEXCPT, OSXSAVE), then
   `xgetbv`/`xsetbv` to enable SSE+AVX state.
   **Required guard:** check CPUID.1:ECX (OSXSAVE, AVX) and CPUID.7 (AVX2) before
   touching `xsetbv` — on a CPU/firmware without XSAVE this `#UD`s at boot. If the
   guard is missing, add it; fail with a readable message, not a triple fault.
4. Allocate the DMA bounce buffer (see below).
5. Resolve the boot volume via `LoadedImage` (see below), open root.
6. Allocate weight buffers via the custom physical allocator; load `MODEL.SAF`,
   `EMBED.BIN`, `VOCAB.BIN` (largest first).
7. Close root, init the **large** heap, free the bounce buffer, construct the
   engine, enter the CLI loop (or `qemu-test` auto-run).

## The four hardware bugs and their shipped fixes

**1. XHCI 64KB DMA limit.** Modern XHCI USB drops the connection on any single
bulk transfer > 64KB, and panics if a read crosses a 64KB physical boundary.
Fix: bounce buffer is exactly 64KB; allocate 128KB (`MaxAddress(0xFFFFFFFF)` to
stay below 4GB for DMA), compute the 64KB-aligned pointer inside it, and slice
64KB there. The over-allocate-then-align math guarantees the slice never crosses a
boundary. Do not "optimize" the buffer size up.

**2. FAT32 8.3 exact matching.** The firmware's SimpleFileSystem does raw
byte-for-byte matching against directory entries, which are stored uppercase.
Request `MODEL.SAF`, never `model.saf`, and keep the image build script
(`build_usb_img.sh` / `mcopy`) writing uppercase names. Filenames must stay 8.3.

**3. Boot-device enumeration order.** NVMe initializes before USB; blindly taking
the first `SimpleFileSystem` handle mounts the internal Windows EFI partition.
Fix: `uefi::boot::image_handle()` → `LoadedImage` protocol → `.device()` — mount
exactly the volume we booted from. Never revert to `find_handles(...)[0]`.

**4. Broken firmware `AnyPages`.** The Acer refuses large `AnyPages` allocations
regardless of free RAM (DDR5 UMA fragmentation + buggy allocator), all the way
down to 10MB chunks. Fix: `allocator.rs::allocate_huge_pages()` walks the raw UEFI
memory map, finds `EfiConventionalMemory` regions, and locks them with
`AllocateType::Address(phys_start)`. All big allocations (weights, embeddings, KV
arena heap) go through this path.

## Required robustness upgrades (do these when touching the loader)

- **File sizes from `FileInfo`, never hardcoded.** Historical code hardcoded
  `model_size = 522831576`, `emb_size = 513935360`, `vocab_size = 432968`; any
  re-forge changes these and boot dies with a size mismatch. `load_file_into`
  already fetches `FileInfo` — allocate from `file_size()` and drop the constants.
  This is the #1 boot-fragility item. (It also feeds prime directive 3: the
  loaded byte length is what the engine derives the embedding dtype from.)
- **Allocator target selection.** `AllocateType::Address(desc.phys_start)` on the
  first fitting region can collide with firmware's own use of low conventional
  memory. Prefer the *largest* region, skip the lowest 1–2MB, and loop over
  candidate regions with a fallback before panicking.
- **The three 1-second `stall`s** between file loads exist to let the USB
  state machine settle. If you can prove them unnecessary on the target box,
  remove them; if they're load-bearing, keep them and comment *which controller*
  needs them. Don't leave them unexplained either way.
- **Allocator locking:** single global lock over 16 heaps, linear scan on every
  alloc/dealloc. Fine single-core; a known hazard if multicore ever happens —
  note it, don't fix it prematurely.

## Scope and portability

Boot support is **modern x86_64 UEFI**, full stop. The vendored `rusty-loader`
carries aarch64/riscv64 code paths, but `aegis-uefi` is hard x86_64 (AVX enable,
`_rdtsc`, x86 intrinsics throughout `ops.rs`). Say "boot compatible" only with
that scoping in review materials.

## Testing changes

Boot changes are validated in QEMU first via the `qemu-test` feature (auto-runs an
inference then exits through `exit_uefi_test_runner`), then on real hardware.
Never hand the user a USB image for the Acer without a green QEMU run. When a
QEMU-passing change fails on hardware, suspect the four bug classes above first —
firmware strictness is the usual culprit.
