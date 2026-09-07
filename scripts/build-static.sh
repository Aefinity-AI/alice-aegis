#!/usr/bin/env bash
# scripts/build-static.sh — build the agent_trace verifier as ONE fully static
# executable (musl libc, no shared-library dependencies), so a receipt can be
# verified on any Linux box of the same ISA by copying a single file: no Rust
# toolchain, no distro packages, no container runtime required.
#
#   scripts/build-static.sh                # host ISA (x86_64 or aarch64)
#   scripts/build-static.sh aarch64        # cross-build for aarch64 (needs a
#                                          # linker that can target it; on an
#                                          # x86_64 host rust-lld is used)
#
# Output: dist/agent_trace-<target>  (+ .sha256), printed with size and `file`.
# The verifier's only crate dependency is pure Rust (libm), which is why this
# works without a C cross toolchain. Runtime CPU dispatch (AVX2 or scalar) is
# unchanged by static linking: the same binary runs on an AVX2 box and on a
# scalar Celeron.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
ARCH="${1:-$(uname -m)}"
case "$ARCH" in
    x86_64|amd64) TARGET=x86_64-unknown-linux-musl ;;
    aarch64|arm64) TARGET=aarch64-unknown-linux-musl ;;
    *) echo "unsupported arch '$ARCH' (x86_64 or aarch64)" >&2; exit 2 ;;
esac
rustup target add "$TARGET" >/dev/null
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
LINKER_ARGS=()
if [ "$TARGET" = aarch64-unknown-linux-musl ] && [ "$(uname -m)" != aarch64 ]; then
    # Cross from x86_64: let rustc's bundled lld link the self-contained musl CRT.
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C link-self-contained=yes"
fi
( cd "$ROOT/aegis-linux" && cargo build --release --target "$TARGET" --example agent_trace "${LINKER_ARGS[@]}" )
SRC="$ROOT/aegis-linux/target/$TARGET/release/examples/agent_trace"
mkdir -p "$ROOT/dist"
OUT="$ROOT/dist/agent_trace-$TARGET"
cp "$SRC" "$OUT"
( cd "$ROOT/dist" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )
echo "built: $OUT"
ls -l "$OUT" | awk '{print "size:", $5, "bytes"}'
file "$OUT" 2>/dev/null || true
# The whole point: refuse to ship anything that still needs a dynamic loader.
if command -v ldd >/dev/null 2>&1 && [ "$(uname -m)" = "${ARCH/amd64/x86_64}" ] && ldd "$OUT" 2>&1 | grep -q '=>'; then
    echo "ERROR: $OUT is dynamically linked" >&2; exit 1
fi
cat "$OUT.sha256"
