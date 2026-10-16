# 2B image re-check with the final heap code (commit 3af5c24)

Same pin (aa0309d) and kit as `pin-plus-heap-patch/`, with `aegis-uefi/src/allocator.rs` from 3af5c24
(heap = clamp(free conventional memory - 512 MB, 700 MB, 2 GB)). Only BOOTX64.EFI was replaced in the kit
image (`mcopy -o`). Host w1, KVM, 8G VM; command as in `../pin-plus-heap-patch/boot.cmd` with `kit.img`.

Result: qemu rc=33 (success), one 2 GB heap claim, `VERIFY PASS`, cis-digest c404a984d548715b,
chain 7bfc1c74cbb8fe5b.., identical to the earlier run. EFI and image sha256 in build-and-sha.txt.
Correctness only; no timing.
