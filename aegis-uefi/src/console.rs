//! Console abstraction: routes the unikernel's status/print output either to
//! the firmware's `SimpleTextOutput` (80x25 text) or to the GOP framebuffer
//! console (`gop.rs`), chosen once at boot.
//!
//! `main.rs` calls `console::init()` early, then every print site calls
//! `console::with_console(|st| { ... })` — the same closure shape as the
//! `uefi::system::with_stdout` calls it replaces, so existing message text is
//! untouched. If GOP init fails for any reason, we fall back to the text
//! console automatically; nothing else in the boot path needs to know.

use core::fmt::Write;
use spin::Mutex;

#[cfg(feature = "gop")]
use crate::gop::GopConsole;

/// Common surface both backends provide. A thin superset of `core::fmt::Write`
/// so callers that also need `.clear()` (as several boot-status prints do)
/// have it available without matching on which backend is live.
pub trait ConsoleWriter: Write {
    fn clear(&mut self);
}

/// Adapter around the firmware's `Output` protocol (borrowed for the
/// duration of a single `with_stdout` callback — it cannot be stored, so this
/// wrapper only ever exists on the stack inside `with_console`).
struct TextAdapter<'a>(&'a mut uefi::proto::console::text::Output);

impl Write for TextAdapter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0.write_str(s)
    }
    fn write_char(&mut self, c: char) -> core::fmt::Result {
        self.0.write_char(c)
    }
}

impl ConsoleWriter for TextAdapter<'_> {
    fn clear(&mut self) {
        let _ = self.0.clear();
    }
}

#[cfg(feature = "gop")]
impl ConsoleWriter for GopConsole {
    fn clear(&mut self) {
        self.clear_with(0x0000_0000);
    }
}

#[cfg(feature = "gop")]
static GOP_CONSOLE: Mutex<Option<GopConsole>> = Mutex::new(None);

/// Try to bring up the GOP console. Call once, early in `main()`, before any
/// other status output. No-op (and never selected) if the `gop` feature is
/// off, no GOP handle exists, or mode-setting fails — the text console is
/// used transparently in all of those cases.
///
/// Returns `true` if the GOP console is now active.
pub fn init() -> bool {
    #[cfg(feature = "gop")]
    {
        // 1080p+ panels get a 2x scale so 16x32 glyphs stay crisp instead of
        // going tiny; anything smaller keeps 1x.
        if let Some(probe) = GopConsole::init(1) {
            let (_w, h) = probe_resolution(&probe);
            let scale = if h >= 1080 { 2 } else { 1 };
            drop(probe);
            if let Some(console) = GopConsole::init(scale) {
                *GOP_CONSOLE.lock() = Some(console);
                return true;
            }
        }
    }
    false
}

#[cfg(feature = "gop")]
fn probe_resolution(c: &GopConsole) -> (usize, usize) {
    c.dims()
}

/// Call `f` with whichever console backend is active. Mirrors the shape of
/// `uefi::system::with_stdout` so call sites are unchanged apart from the
/// function path.
pub fn with_console<F>(mut f: F)
where
    F: FnMut(&mut dyn ConsoleWriter) -> core::fmt::Result,
{
    #[cfg(feature = "gop")]
    {
        let mut guard = GOP_CONSOLE.lock();
        if let Some(gop) = guard.as_mut() {
            let _ = f(gop);
            return;
        }
    }
    let _ = uefi::system::with_stdout(|st| {
        let mut adapter = TextAdapter(st);
        f(&mut adapter)
    });
}
