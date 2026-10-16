# CIS-1 BitNet 2B-4T: bootable ALICE image booted in QEMU/KVM (2026-10-16)

Chain step (a) bootable image, linked to (b)/(c) in `../2026-10-16-cis1-2b-x86/`.
Host: w1 (Intel Core i9-9980HK, 16 threads, QEMU 10.0.13 Debian, KVM, kernel 6.12.94). w1 chosen because w2 load was 0.64 (>0.5) at 16:05Z with the timing leg running; w1 load was 0.00. Not hq.
Correctness/identity only (Rule A). Emulated platform (KVM virtualized PC, UEFI OVMF), NOT physical iron. Wall time (times.txt) is not a benchmark.

## Result
`pin-plus-heap-patch/`: the image boots under OVMF, loads the 2B trio, and prints (serial.log / BOOTLOG.TXT):

    artifacts: 3/3 hashes match the receipt
    receipt cis-digest c404a984d548715b chain 7bfc1c74cbb8fe5b..
    local   cis-digest c404a984d548715b chain 7bfc1c74cbb8fe5b..
    VERIFY PASS — this machine reproduced all 8 decode steps' full logit vectors bit-for-bit, with no OS underneath
    qemu rc=33 (isa-debug-exit 0x10 = success)

The "local" digest/chain were recomputed inside the image by replaying the decode under CIS-1 FullInt. They equal the digest/chain in the Linux-harness receipt (c404a984d548715b / 7bfc1c74cbb8fe5b...). The image prints digests, not the sha256 of the receipt file; a5103369... is the sha256 of the RECEIPT.TXT input placed on the image (see image.sha256), which is identical to the w2 `receipt-R1.txt`.

## Which: (i) or (ii)?  (ii)
The image only VERIFIES a receipt minted by the Linux harness (`aegis-uefi/src/verifier.rs`, entered only when RECEIPT.TXT exists on the volume; otherwise it runs the REPL). It does not mint a receipt itself. Not shown: an image that mints. Cost to mint (read-only estimate from code, not implemented): verifier.rs already runs the same prefill + greedy-argmax decode loop that `aegis-linux/examples/cis_witness.rs gen` uses; missing pieces are a prompt source (e.g. PROMPT.TXT), a witness-v1 text writer (cis_witness.rs ~60 lines; BOOTLOG.TXT write path already exists), and a mint-mode flag. Order of 100-150 lines of Rust, no new cryptography; a few hours plus a boot test. Unmeasured.

## Pin status: pin does NOT run 2B verify as-is; one-constant patch needed
Pin `aa0309d532cf` builds the image (`aegis-uefi/build_hardfloat.sh --qemu-test`: fma=161 ymm=504 census OK; log in `pin-unpatched-oom/build-hardfloat.log`) but boots to a KERNEL PANIC: `memory allocation of 157286400 bytes failed` right after "witness verifier mode" (isa-debug-exit rc=35 = 0x11 failure; `pin-unpatched-oom/`: gop build @4G, no-gop build serial log @8G). Cause: `aegis-uefi/src/allocator.rs init_uefi_alloc_large()` fixes the working heap at 700 MiB; the 2B CisModel conversion needs more. The 2026-08-27 QEMU-TCG 2B pass (`../unikernel_bitnet2b_verify_qemu_tcg_2026-08-27.log`) predates the 2026-09-06 CIS perf commits (e.g. 88f48eb pre-converts the LM head) and ran on an older build. Likely related, not verified by bisect.
Patch used (`pin-plus-heap-patch/allocator-heap-2GiB.patch`): 700 MiB -> 2048 MiB, nothing else. alice-aegis main is identical to the pin for aegis-uefi/aegis-core, so main has the same issue. Committed in this branch as a separate commit on top of the pin.

## Build and boot (exact)
- Source: `git archive aa0309d532cf` + allocator patch, extracted to `~/cis1-img-build` on w1. Image build was on w1 (rust nightly, `-Zbuild-std`).
- EFI: `cargo +nightly build --release --target ./x86_64-uefi-hardfloat.json -Zjson-target-spec -Zbuild-std=core,alloc -Zbuild-std-features=compiler-builtins-mem --no-default-features --features qemu-test` (no `gop` so the console is on serial). EFI sha256 `888472b1caa1aa2a9bdfa23dfd52e7cae5442d0e922799150d89f07de11b5006`. (The hardfloat script run with default features is in build-hardfloat-patched.log; the shipped EFI is the no-gop variant, so the serial log shows the console.)
- Image: `AEGIS_KIT_ASSETS=<dir with MODEL.SAF EMBED.BIN VOCAB.BIN symlinks to the 2B artifacts> AEGIS_KIT_RECEIPT=<w2 receipt-R1.txt> AEGIS_KIT_SIZE_MB=1024 scripts/make-kit-image.sh nogop2.efi cis1-2b-kit.img` (make-kit-image.log). Image sha256 `73a7661502eaa21b7f7ae4c6b30979dc356353f9c1556f3491c40bc92d44b02d` (FAT timestamps make it non-reproducible bit for bit). The image itself (781 MB) is not committed.
- Boot: `pin-plus-heap-patch/boot.cmd`; qemu rc 33; start/end in times.txt.
- Artifact shas on w1: `host-and-artifacts.txt` (match facb3597...). 

## Not shown
- No physical-iron boot of the 2B model (iron boots exist only for the M7 model).
- Not the no-AVX2 variant (this boot is KVM `-cpu host` on AVX2: CPUID line in BOOTLOG shows avx2:1,fma:1). The scalar_only build was not run with 2B here.
- The image does not mint (ii) and does not print the receipt file's sha256.
- Not a bisect of which commit grew the heap need.

## sha256 of every file in this directory (except this README)
    8573eaf32586e6fc26ea9a0c187a3fd4e62914ed9073e009ab3258cad52de9bb  ./host-and-artifacts.txt
    3d5ef4d20579295e77cad802bbc65f7995e7fb54ac302910962afb33634c3e77  ./pin-plus-heap-patch/allocator-heap-2GiB.patch
    5632f1a63810b0411071c6d6e91e9c89a26244c1783f84deea08799a3ba8a946  ./pin-plus-heap-patch/boot.cmd
    4cbaddd4feac1b012b7bc46b161d6ef5a6f2297e1b11bd34a716e99f5df369ba  ./pin-plus-heap-patch/BOOTLOG.TXT
    cdc8cd4da5c6575329b0bf6f8d4b0c948bd65823828220101bf42503e0d9da1b  ./pin-plus-heap-patch/build-hardfloat-patched.log
    06c26794907e66e08ded8010518e6eda40980460a9cf84292887e8db09879ac7  ./pin-plus-heap-patch/image.sha256
    7a611866e531a1f5a1987e5e4f789ec00d4dcdd2ce4804094b8eb90c3dbc09ab  ./pin-plus-heap-patch/make-kit-image.log
    fe330a94a357657c083e93e9f9cd4b3e456695c254dc6087cc9287164a6ace69  ./pin-plus-heap-patch/serial.log
    94e0517e42b51755657e1237adf9fc19ef1c6bbec99bec797340dbc41ec22ff1  ./pin-plus-heap-patch/times.txt
    98cdda68dfcf00516e2b25d1a0fe0ddef91f16270f8df7f48fde9e3e4ec9b1b0  ./pin-unpatched-oom/BOOTLOG-gop-4G.TXT
    27b5d92c08453f1dd433a1b0d0a57805a0a9e50a35cf64b3518359733d2b111e  ./pin-unpatched-oom/build-hardfloat.log
    b41eaa5cacce8ad23f1aad6deff6fc5162dc8df99bf4556dca0a79e5c0564de5  ./pin-unpatched-oom/qemu-gop-4G.out
    2074f6e30893c6181ed5e086f8c5b74d19cae8b5763ca3810a11a9c32c6f55bd  ./pin-unpatched-oom/serial-nogop-8G.log
