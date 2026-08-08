#!/bin/bash
# RETIRED. This script built with the stock x86_64-unknown-uefi target, which
# is SOFT-FLOAT: the binary contains zero vector instructions and inference
# runs ~100x slower (see build_hardfloat.sh for the full story). It also
# produced a 2.5GB GPT image with stale artifact names instead of the current
# 940MB FAT32 layout (MODEL.SAF / EMBED.BIN / VOCAB.BIN / STARTUP.NSH).
#
# The canonical image builder is ../build_usb_img.sh (repo root):
#   1. ./build_hardfloat.sh              # hard-float EFI + AVX sanity gate
#   2. cd .. && ./build_usb_img.sh       # 940MB FAT32 image + startup.nsh
echo "RETIRED: use build_hardfloat.sh then ../build_usb_img.sh (repo root)." >&2
echo "This script produced soft-float (~100x slower) binaries. See comments." >&2
exit 1
