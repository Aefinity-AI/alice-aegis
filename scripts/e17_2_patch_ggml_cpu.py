#!/usr/bin/env python3
"""E17.2 build-portability fix for microsoft/BitNet's vendored llama.cpp.

Upstream ggml-cpu.c (ggml_compute_forward_mul_mat) declares `src1_cont`
only inside an `#if GGML_USE_LLAMAFILE` block, but references it
unconditionally a few lines past the matching `#endif`, in the I2_S GEMM
fast path. Whenever GGML_USE_LLAMAFILE is not defined for this
translation unit, this is an "undeclared identifier 'src1_cont'" compile
error -- reproduced on both x86_64 and aarch64 (arch-independent; not an
ISA or compiler-strictness issue). Fix: hoist the declaration out of the
#if block so it is always in scope. Semantics are unchanged (same
expression, now unconditionally evaluated -- ggml_is_contiguous(src1) is
a cheap, side-effect-free check regardless of the macro).

Usage: python3 e17_2_patch_ggml_cpu.py <path-to-ggml-cpu.c>
"""
import sys

path = sys.argv[1]
src = open(path).read()

old = (
    "#if GGML_USE_LLAMAFILE\n"
    "    // broadcast factors\n"
    "    const int64_t r2 = ne12 / ne02;\n"
    "    const int64_t r3 = ne13 / ne03;\n"
    "\n"
    "    const bool src1_cont = ggml_is_contiguous(src1);\n"
    "\n"
    "    if (src1_cont) {"
)
new = (
    "    // E17.2 patch (alice-aegis): hoisted out of the GGML_USE_LLAMAFILE guard below --\n"
    "    // used unconditionally a few lines further down in the I2_S GEMM fast path, but\n"
    "    // upstream only ever declared it inside this #if block. Undeclared-identifier build\n"
    "    // failure whenever GGML_USE_LLAMAFILE is not defined for this TU. Semantics unchanged.\n"
    "    const bool src1_cont = ggml_is_contiguous(src1);\n"
    "#if GGML_USE_LLAMAFILE\n"
    "    // broadcast factors\n"
    "    const int64_t r2 = ne12 / ne02;\n"
    "    const int64_t r3 = ne13 / ne03;\n"
    "\n"
    "    if (src1_cont) {"
)

if old not in src:
    print("PATCH TARGET NOT FOUND -- upstream ggml-cpu.c may have changed since this was written", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
open(path, "w").write(src)
print(f"patched {path}: hoisted src1_cont declaration out of GGML_USE_LLAMAFILE guard")
