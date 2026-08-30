#!/usr/bin/env python3
"""Rasterize DejaVu Sans Mono into a fixed 16x32 1-bit-per-pixel bitmap font
for the aegis-uefi GOP console, and emit it as a Rust source file.

Covers ASCII 32..126 inclusive (95 glyphs). Each glyph is FONT_W x FONT_H
pixels, packed 1 bit per pixel, row-major, MSB-first within each row's byte
stream (2 bytes per row for a 16px-wide glyph -> 64 bytes per glyph).

Run:
    python3 scripts/gen_font.py
or, if Pillow isn't installed system-wide:
    uv run --with pillow python scripts/gen_font.py

Font: DejaVu Sans Mono (Bitstream Vera / DejaVu license — permits embedding
and redistribution, including in modified/rasterized form; see
THIRD_PARTY_NOTICES.md).
"""

import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print(
        "PIL/Pillow not available. Re-run as:\n"
        "  uv run --with pillow python scripts/gen_font.py",
        file=sys.stderr,
    )
    sys.exit(1)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_W = 16
FONT_H = 32
FIRST = 32
LAST = 126  # inclusive
COUNT = LAST - FIRST + 1

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "aegis-uefi", "src", "font.rs")


def rasterize_glyph(font: "ImageFont.FreeTypeFont", ch: str) -> bytes:
    """Return FONT_H rows of ceil(FONT_W/8) bytes each, 1 = lit pixel."""
    img = Image.new("L", (FONT_W, FONT_H), 0)
    draw = ImageDraw.Draw(img)
    # Center the glyph in the cell; DejaVu Sans Mono is already fixed-width,
    # but different sizes render with slightly different bearings.
    bbox = draw.textbbox((0, 0), ch, font=font)
    gw = bbox[2] - bbox[0]
    gh = bbox[3] - bbox[1]
    x = (FONT_W - gw) // 2 - bbox[0]
    y = (FONT_H - gh) // 2 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)

    row_bytes = (FONT_W + 7) // 8
    out = bytearray(row_bytes * FONT_H)
    px = img.load()
    for row in range(FONT_H):
        for col in range(FONT_W):
            if px[col, row] >= 128:
                byte_idx = row * row_bytes + (col // 8)
                bit = 7 - (col % 8)
                out[byte_idx] |= 1 << bit
    return bytes(out)


def main() -> None:
    if not os.path.exists(FONT_PATH):
        print(f"font not found: {FONT_PATH}", file=sys.stderr)
        sys.exit(1)

    # Pick a point size that comfortably fills a 16x32 cell without clipping
    # most glyphs (DejaVu Sans Mono's em is taller than its typical x-height).
    point_size = 24
    font = ImageFont.truetype(FONT_PATH, point_size)

    glyphs = []
    for code in range(FIRST, LAST + 1):
        glyphs.append(rasterize_glyph(font, chr(code)))

    row_bytes = (FONT_W + 7) // 8
    bytes_per_glyph = row_bytes * FONT_H

    lines = []
    lines.append("// GENERATED FILE — do not hand-edit.")
    lines.append("// Produced by scripts/gen_font.py from DejaVu Sans Mono")
    lines.append(f"// ({FONT_PATH}), rasterized at {point_size}pt into a")
    lines.append(f"// {FONT_W}x{FONT_H} 1-bit-per-pixel cell, ASCII {FIRST}..{LAST}.")
    lines.append("// DejaVu fonts are derived from Bitstream Vera; both licenses permit")
    lines.append("// embedding/redistribution in original or modified (rasterized) form.")
    lines.append("// See THIRD_PARTY_NOTICES.md.")
    lines.append("")
    lines.append(f"pub const FONT_W: usize = {FONT_W};")
    lines.append(f"pub const FONT_H: usize = {FONT_H};")
    lines.append(f"pub const FONT_ROW_BYTES: usize = {row_bytes};")
    lines.append(f"pub const FONT_FIRST: u32 = {FIRST};")
    lines.append(f"pub const FONT_LAST: u32 = {LAST};")
    lines.append(f"pub const FONT_COUNT: usize = {COUNT};")
    lines.append(f"pub const FONT_BYTES_PER_GLYPH: usize = {bytes_per_glyph};")
    lines.append("")
    lines.append(
        f"pub static FONT_DATA: [[u8; FONT_BYTES_PER_GLYPH]; FONT_COUNT] = ["
    )
    for g in glyphs:
        byte_list = ", ".join(f"0x{b:02x}" for b in g)
        lines.append(f"    [{byte_list}],")
    lines.append("];")
    lines.append("")
    lines.append("/// Returns the glyph bitmap for `ch`, falling back to a blank cell")
    lines.append("/// (space) for anything outside FONT_FIRST..=FONT_LAST.")
    lines.append("pub fn glyph(ch: char) -> &'static [u8; FONT_BYTES_PER_GLYPH] {")
    lines.append("    let code = ch as u32;")
    lines.append("    if code >= FONT_FIRST && code <= FONT_LAST {")
    lines.append("        &FONT_DATA[(code - FONT_FIRST) as usize]")
    lines.append("    } else {")
    lines.append("        &FONT_DATA[0] // space")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {OUT_PATH} ({COUNT} glyphs, {bytes_per_glyph} bytes/glyph, "
          f"{COUNT * bytes_per_glyph} bytes total)")


if __name__ == "__main__":
    main()
