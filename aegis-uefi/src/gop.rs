//! Framebuffer console on top of UEFI's Graphics Output Protocol (GOP).
//!
//! Draws fixed-cell text using the embedded bitmap font (`font.rs`), with a
//! cursor, newline handling, and scroll-on-overflow — enough to be a drop-in
//! replacement for the firmware's 80x25 `SimpleTextOutput` console. Falls
//! back to that text console at runtime if GOP isn't present or `set_mode`
//! fails (see `console.rs`).

use crate::font;
use uefi::proto::console::gop::{GraphicsOutput, PixelFormat};

/// A raw framebuffer pointer plus enough metadata to address it, wrapped so
/// it can live in a global static. UEFI is single-core at this point in boot
/// (no APs have been started when the console is selected), so there is no
/// concurrent access to guard against; nothing here is safe to share across
/// real threads.
///
/// SAFETY: `base` points at firmware-mapped framebuffer memory for the
/// lifetime of the boot session (until ExitBootServices, which this app
/// never survives past). We only ever touch it from the single boot thread.
pub struct GopConsole {
    base: *mut u8,
    fb_size: usize,
    stride_px: usize, // pixels per scanline, from ModeInfo::stride()
    width_px: usize,
    height_px: usize,
    bgr: bool,
    scale: usize,
    cols: usize,
    rows: usize,
    cursor_x: usize,
    cursor_y: usize,
    fg: u32,
    bg: u32,
    /// Last border drawn (x, y, w, h, color); re-applied after every clear so panels survive `.clear()`.
    border: Option<(usize, usize, usize, usize, u32)>,
}

// SAFETY: see struct doc — single-threaded boot-time use only.
unsafe impl Send for GopConsole {}

const DEFAULT_FG: u32 = 0x00E0_E0E0; // light gray, easy on the eyes on black
const DEFAULT_BG: u32 = 0x0000_0000; // black

impl GopConsole {
    /// Locate the GOP, pick the highest-resolution Rgb/Bgr (non-BltOnly,
    /// non-Bitmask) mode, set it, and build a console over it.
    ///
    /// Returns `None` if no GOP handle exists, no suitable mode is found, or
    /// `set_mode` fails — callers must fall back to the text console.
    pub fn init(scale: usize) -> Option<Self> {
        let handle = uefi::boot::get_handle_for_protocol::<GraphicsOutput>().ok()?;
        let mut gop = uefi::boot::open_protocol_exclusive::<GraphicsOutput>(handle).ok()?;

        let mut best: Option<uefi::proto::console::gop::Mode> = None;
        let mut best_area = 0usize;
        for mode in gop.modes() {
            let info = mode.info();
            match info.pixel_format() {
                PixelFormat::Rgb | PixelFormat::Bgr => {}
                PixelFormat::Bitmask | PixelFormat::BltOnly => continue,
            }
            let (w, h) = info.resolution();
            let area = w * h;
            if area > best_area {
                best_area = area;
                best = Some(mode);
            }
        }

        let mode = best?;
        gop.set_mode(&mode).ok()?;

        let info = mode.info();
        let (width_px, height_px) = info.resolution();
        let stride_px = info.stride();
        let bgr = matches!(info.pixel_format(), PixelFormat::Bgr);

        let mut fb = gop.frame_buffer();
        let base = fb.as_mut_ptr();
        let fb_size = fb.size();

        let scale = scale.max(1);
        let cell_w = font::FONT_W * scale;
        let cell_h = font::FONT_H * scale;
        let cols = width_px / cell_w;
        let rows = height_px / cell_h;
        if cols == 0 || rows == 0 {
            return None;
        }

        let mut console = GopConsole {
            base,
            fb_size,
            stride_px,
            width_px,
            height_px,
            bgr,
            scale,
            cols,
            rows,
            cursor_x: 0,
            cursor_y: 0,
            fg: DEFAULT_FG,
            bg: DEFAULT_BG,
            border: None,
        };
        console.clear_screen();
        // A single-pixel panel border around the whole screen so the boot
        // console reads as a designed UI rather than a raw text dump.
        let border_color = 0x0060_6060;
        console.draw_border(0, 0, width_px, height_px, border_color);
        Some(console)
    }

    /// Native panel resolution in pixels, used to pick a text scale factor
    /// before committing to a cell layout.
    pub fn dims(&self) -> (usize, usize) {
        (self.width_px, self.height_px)
    }

    #[inline]
    fn pack(&self, color: u32) -> u32 {
        // color is 0x00RRGGBB regardless of the panel's native order; swap
        // channels here so callers never think about BGR vs RGB.
        if self.bgr {
            let r = (color >> 16) & 0xFF;
            let g = (color >> 8) & 0xFF;
            let b = color & 0xFF;
            (r << 16) | (g << 8) | b // stored as B,G,R,X in memory below
        } else {
            color
        }
    }

    /// SAFETY: `x < width_px` and `y < height_px` must hold; caller checks.
    #[inline]
    unsafe fn put_pixel_raw(&mut self, x: usize, y: usize, color: u32) {
        let packed = self.pack(color);
        let offset = (y * self.stride_px + x) * 4;
        if offset + 4 > self.fb_size {
            return;
        }
        // Framebuffer is 32-bit-per-pixel Rgb/Bgr (checked at init). Byte
        // order in memory is little-endian: byte0=B(or R), byte1=G, byte2=R(or B).
        let b0 = (packed & 0xFF) as u8;
        let b1 = ((packed >> 8) & 0xFF) as u8;
        let b2 = ((packed >> 16) & 0xFF) as u8;
        unsafe {
            let p = self.base.add(offset);
            core::ptr::write_volatile(p, b0);
            core::ptr::write_volatile(p.add(1), b1);
            core::ptr::write_volatile(p.add(2), b2);
        }
    }

    fn put_pixel(&mut self, x: usize, y: usize, color: u32) {
        if x >= self.width_px || y >= self.height_px {
            return;
        }
        // SAFETY: bounds just checked.
        unsafe { self.put_pixel_raw(x, y, color) };
    }

    /// Fill the whole framebuffer with `bg`.
    pub fn clear_with(&mut self, bg: u32) {
        self.bg = bg;
        self.clear_screen();
    }

    fn clear_screen(&mut self) {
        for y in 0..self.height_px {
            for x in 0..self.width_px {
                self.put_pixel(x, y, self.bg);
            }
        }
        self.cursor_x = 0;
        self.cursor_y = 0;
        // Re-apply the persistent border, if any (it was just painted over).
        if let Some((bx, by, bw, bh, bc)) = self.border {
            for i in 0..bw {
                self.put_pixel(bx + i, by, bc);
                self.put_pixel(bx + i, by + bh - 1, bc);
            }
            for j in 0..bh {
                self.put_pixel(bx, by + j, bc);
                self.put_pixel(bx + bw - 1, by + j, bc);
            }
        }
    }

    /// Draw a 1px-wide rectangular border/panel outline.
    pub fn draw_border(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.border = Some((x, y, w, h, color));
        for i in 0..w {
            self.put_pixel(x + i, y, color);
            self.put_pixel(x + i, y + h - 1, color);
        }
        for j in 0..h {
            self.put_pixel(x, y + j, color);
            self.put_pixel(x + w - 1, y + j, color);
        }
    }

    /// Draw one glyph cell at text-cell coordinates (x_cell, y_cell).
    pub fn put_char(&mut self, x_cell: usize, y_cell: usize, ch: char, fg: u32, bg: u32) {
        let glyph = font::glyph(ch);
        let cell_w = font::FONT_W * self.scale;
        let cell_h = font::FONT_H * self.scale;
        let base_x = x_cell * cell_w;
        let base_y = y_cell * cell_h;

        for row in 0..font::FONT_H {
            let byte0 = glyph[row * font::FONT_ROW_BYTES];
            let byte1 = if font::FONT_ROW_BYTES > 1 {
                glyph[row * font::FONT_ROW_BYTES + 1]
            } else {
                0
            };
            for col in 0..font::FONT_W {
                let bit_set = if col < 8 {
                    (byte0 >> (7 - col)) & 1 != 0
                } else {
                    (byte1 >> (7 - (col - 8))) & 1 != 0
                };
                let color = if bit_set { fg } else { bg };
                for sy in 0..self.scale {
                    for sx in 0..self.scale {
                        self.put_pixel(base_x + col * self.scale + sx, base_y + row * self.scale + sy, color);
                    }
                }
            }
        }
    }

    fn newline(&mut self) {
        self.cursor_x = 0;
        self.cursor_y += 1;
        if self.cursor_y >= self.rows {
            self.scroll_one_row();
            self.cursor_y = self.rows - 1;
        }
    }

    /// Move all text rows up by one cell row, clearing the last row.
    fn scroll_one_row(&mut self) {
        let cell_h = font::FONT_H * self.scale;
        let row_stride_bytes = self.stride_px * 4;
        let move_bytes = row_stride_bytes * (self.height_px - cell_h);
        // SAFETY: src/dst are both within the framebuffer allocation
        // (fb_size >= stride_px*4*height_px); regions may overlap, hence copy
        // (memmove semantics) rather than copy_nonoverlapping. This is a
        // linear framebuffer, not cached device memory with side effects, so
        // a bulk byte copy is standard practice for scrolling text consoles.
        unsafe {
            let dst = self.base;
            let src = self.base.add(row_stride_bytes * cell_h);
            core::ptr::copy(src, dst, move_bytes);
        }
        // Clear the newly-exposed bottom row.
        for y in (self.height_px - cell_h)..self.height_px {
            for x in 0..self.width_px {
                self.put_pixel(x, y, self.bg);
            }
        }
    }

    fn put_str_at_cursor(&mut self, s: &str) {
        for ch in s.chars() {
            match ch {
                '\n' => self.newline(),
                '\r' => {} // CRLF pairs are common in this codebase; no-op on CR
                _ => {
                    if self.cursor_x >= self.cols {
                        self.newline();
                    }
                    let (x, y, fg, bg) = (self.cursor_x, self.cursor_y, self.fg, self.bg);
                    self.put_char(x, y, ch, fg, bg);
                    self.cursor_x += 1;
                }
            }
        }
    }
}

impl core::fmt::Write for GopConsole {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.put_str_at_cursor(s);
        Ok(())
    }

    fn write_char(&mut self, c: char) -> core::fmt::Result {
        let mut buf = [0u8; 4];
        self.put_str_at_cursor(c.encode_utf8(&mut buf));
        Ok(())
    }
}
