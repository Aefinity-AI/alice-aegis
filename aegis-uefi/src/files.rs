//! AEFINITY OS phase 4 — the file plane.
//!
//! Contract: `program/AEFINITY_OS_FLEET_DESIGN.md` §1.1 (the `DATA` frame),
//! §1.2 (verbs), §1.3 (caps and byte-exact `ERR` slugs), §1.4 (the abort
//! table), §7 (`files-test`), §8 (iron safety). Spec `program/AEFINITY_OS.md`
//! §9 phase 4.
//!
//! This module owns **the boot volume**, not the wire. It validates names,
//! lists, stats, streams a sha256, opens a file for a streamed read, stages a
//! streamed write into `STAGE.PRT`, verifies the staged bytes by reading them
//! back, and commits by rename. `server.rs` owns the socket and drives these;
//! keeping the two apart is what makes the §1.4 abort table implementable —
//! every failure below leaves either the old file or nothing, never half of a
//! new one.
//!
//! # What this file is careful about
//!
//! - **No verb ever returns a partial success** (§1.3). `PUT` never opens the
//!   target: it writes `STAGE.PRT`, closes it, re-opens it, streams a sha256
//!   over the **readback**, and only then deletes the target and renames the
//!   stage over it. A power loss anywhere before the rename leaves the old
//!   file; anywhere after it leaves the new one.
//! - **The watchdog is re-armed by the caller, per chunk, on every long
//!   loop** (§8) — the transfer, the readback *and* `RELOAD`'s load. A 1.83 GB
//!   readback at FAT read speed can itself outlast `FILES_WD_S`, so a
//!   watchdog armed only around the network half would reset a healthy box.
//!   Every streaming entry point here therefore takes a `rearm` hook and
//!   calls it once per chunk.
//! - **Nothing here measures anything** (CLAUDE.md Rule A). The only numbers
//!   this module produces are byte counts and digests.
//! - **No `unwrap`/`expect` on a firmware result.** Every `open`, `read`,
//!   `write`, `flush`, `get_info` and `set_info` is matched, and the failure
//!   becomes an `ERR` slug the operator can act on.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use uefi::proto::media::file::Directory;
use uefi::proto::media::file::{
    File, FileAttribute, FileInfo, FileMode, FileSystemInfo, FileType, RegularFile,
};

// ---------------------------------------------------------------------------
// Caps (design §1.3) — byte-exact, and the reason each number is that number
// ---------------------------------------------------------------------------

/// Longest legal `<NAME>`, in bytes.
///
/// 31, not 32: every path in this crate builds its `CStr16` in a `[u16; 32]`
/// (`job.rs`'s `write_named`, `read_small_file`, and this module), and the
/// terminating NUL takes one unit. 31 ASCII bytes is the true ceiling and a
/// 32-byte name is `ERR bad-name`, not a truncation.
pub const NAME_MAX_BYTES: usize = 31;
/// Entries one `LS` will list before the header says `truncated` (§1.3).
pub const LS_MAX_ENTRIES: usize = 256;
/// Largest `PUT`, in bytes: 2 GiB. `MODEL.SAF` is 1.83 G.
pub const PUT_MAX_BYTES: u64 = 2_147_483_648;
/// Largest `GET`, in bytes. Framed and streamed, never buffered whole.
pub const GET_MAX_BYTES: u64 = 2_147_483_648;
/// One transfer chunk, re-exported from the transport that defines it: it
/// equals `net::RECV_MAX_BYTES`, the cap on a single `tcp_recv_until`, so a
/// chunk is the largest unit both halves of a transfer agree on. One
/// definition, because two that drifted apart would desynchronise a stream.
pub use crate::net::XFER_CHUNK;
/// Watchdog window armed for the duration of a `GET`/`PUT`/`SHA`/`RELOAD`,
/// **re-armed every chunk** (§8). It bounds *no progress*, not total duration.
pub const FILES_WD_S: u64 = 300;
/// The one fixed staging name (§1.3).
///
/// Not `<NAME>.PART`: a per-name suffix breaks 8.3 and can exceed
/// [`NAME_MAX_BYTES`] for a target that is itself legal. Only one `PUT` can be
/// in flight anyway — one client at a time — so a single fixed name is enough
/// and makes `sweep_parts` a one-line boot rule and `HEALTH parts=` a bit.
pub const STAGE_NAME: &str = "STAGE.PRT";

/// Names a client may read but never write (§1.3), plus the two this build
/// adds.
///
/// `BOOTLOG.TXT`, `RESULT.TXT`, `RESULT.WIP` and `STAGE.PRT` are §1.3's list:
/// they are the box's own account of itself, and a client that could overwrite
/// them could forge it.
///
/// `BOOTX64.EFI` and `CURRENT.TXT` are **additions beyond §1.3**, recorded in
/// `docs/AEFINITY_OS_STATUS.md`. A `PUT` of the loader that is interrupted
/// between the commit's delete and its rename leaves a box that will not boot
/// and that nobody can dial to fix — the exact failure the fleet has no
/// recovery path for short of a stick. Design §9 already leaves large-file
/// provisioning to the Debian side; this makes the loader part of that rule
/// instead of a footgun. `CURRENT.TXT` is the artifact pointer the OS itself
/// writes (§8), so a client writing it directly could point boot at a file
/// whose bytes were never verified.
///
/// [`crate::reload::ALTERNATES`] — `MODEL.NEW`, `EMBED.NEW`, `VOCAB.NEW` — are
/// protected too, by [`is_protected`] rather than by being listed here, so the
/// A/B halves have exactly one definition (`reload.rs`). They are **internal
/// names**: the protocol only ever exposes the canonical `MODEL.SAF` and the
/// box decides which half a `PUT` lands on (§8's pointer swap). A client that
/// could name a half directly could delete the one `CURRENT.TXT` designates,
/// and `current_names`' fallback would then answer `STAT`/`SHA`/`GET`/`RELOAD`
/// from the *other*, stale half with no error at all — a silent downgrade of
/// the model the box serves, which is exactly the failure the swap exists to
/// prevent.
const PROTECTED: [&str; 6] = [
    "BOOTLOG.TXT",
    "RESULT.TXT",
    "RESULT.WIP",
    STAGE_NAME,
    "BOOTX64.EFI",
    "CURRENT.TXT",
];

// ---------------------------------------------------------------------------
// Errors (design §1.3) — the slug list is exhaustive and byte-exact
// ---------------------------------------------------------------------------

/// Why a file-plane operation did not happen.
///
/// §1.3's slug list also names `exists`, `bad-runid`, `bad-corpus`,
/// `reload-size`, `reload-engine` and `auth`. The last five belong to
/// `reload.rs` and `server.rs`, which render them directly; `exists` has **no
/// producer in phase 4** — no verb here refuses a name for already being
/// taken, because `PUT` replaces and `RM` does not create — so it is not a
/// variant of this enum rather than a variant nothing can build.
///
/// The `slug` is what goes on the wire after `ERR `, byte for byte. Design
/// §1.3: a slug not on this list means the box is unhealthy, not that the job
/// failed — so the variants are deliberately few and each one tells an
/// operator a different thing to do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FileErr {
    /// The name is not 1–[`NAME_MAX_BYTES`] bytes of `[A-Z0-9._-]`, starts
    /// with `.`, or contains a path separator or `..`.
    BadName,
    /// The verb's arguments did not parse (wrong count, digest not 64 hex).
    BadArgs,
    /// `<len>` did not parse, or is above [`PUT_MAX_BYTES`].
    BadLen,
    /// The declared payload did not end in `END\n`.
    BadFrame,
    /// No such file on the boot volume root.
    NotFound,
    /// One of [`PROTECTED`].
    Protected,
    /// The advisory free-space check said the write cannot fit (§8: best
    /// effort — exhaustion may instead surface as [`FileErr::Io`]).
    NoSpace,
    /// The readback digest did not equal the declared one.
    DigestMismatch,
    /// The firmware took fewer bytes than were handed to it.
    ShortWrite,
    /// The firmware returned fewer bytes than the file claims to hold.
    ShortRead,
    /// A `STAGE.PRT` is in the way of an operation that needs the volume
    /// quiescent (`RELOAD`, §4.1).
    BusyFile,
    /// A firmware call failed in a way with no more specific slug.
    Io,
    /// A firmware call failed where no I/O should have been attempted at all
    /// (a `CStr16` that will not build, a root that will not enumerate).
    FwError,
}

impl FileErr {
    /// The byte-exact §1.3 slug, without the `ERR ` prefix.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            FileErr::BadName => "bad-name",
            FileErr::BadArgs => "bad-args",
            FileErr::BadLen => "bad-len",
            FileErr::BadFrame => "bad-frame",
            FileErr::NotFound => "not-found",
            FileErr::Protected => "protected",
            FileErr::NoSpace => "no-space",
            FileErr::DigestMismatch => "digest-mismatch",
            FileErr::ShortWrite => "short-write",
            FileErr::ShortRead => "short-read",
            FileErr::BusyFile => "busy-file",
            FileErr::Io => "io",
            FileErr::FwError => "fw-error",
        }
    }
}

// ---------------------------------------------------------------------------
// Names (design §1.3)
// ---------------------------------------------------------------------------

/// Validate and normalise a client-supplied `<NAME>`.
///
/// 1–[`NAME_MAX_BYTES`] bytes of `[A-Za-z0-9._-]`, upper-cased on the way
/// through, not starting with `.`, and with no `..` anywhere in it. There is
/// no directory syntax in the protocol at all, so rejecting `/`, `\` and `:`
/// by not being in the allowed set is the whole of the traversal defence: the
/// boot-volume root is the only reachable directory because no name that names
/// another one can be built.
pub fn validate_name(raw: &str) -> Result<String, FileErr> {
    let s = raw.trim();
    if s.is_empty() || s.len() > NAME_MAX_BYTES {
        return Err(FileErr::BadName);
    }
    if s.starts_with('.') {
        return Err(FileErr::BadName);
    }
    if s.contains("..") {
        return Err(FileErr::BadName);
    }
    for b in s.bytes() {
        let ok = b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-';
        if !ok {
            return Err(FileErr::BadName);
        }
    }
    let mut out = s.to_string();
    out.make_ascii_uppercase();
    Ok(out)
}

/// Is this (already validated, upper-cased) name one the box writes for
/// itself?
///
/// [`PROTECTED`] plus the three artifact alternates, which `reload.rs` owns.
#[must_use]
pub fn is_protected(name: &str) -> bool {
    PROTECTED.contains(&name) || crate::reload::ALTERNATES.contains(&name)
}

/// Build a `CStr16` for a root-relative name in a caller-owned buffer.
///
/// The buffer is `[u16; 32]` everywhere in this crate, which is exactly why
/// [`NAME_MAX_BYTES`] is 31.
fn cstr<'b>(name: &str, buf: &'b mut [u16; 32]) -> Result<&'b uefi::CStr16, FileErr> {
    uefi::CStr16::from_str_with_buf(name, buf).map_err(|_| FileErr::BadName)
}

// ---------------------------------------------------------------------------
// The DMA-reachable bounce buffer
// ---------------------------------------------------------------------------

/// A 64 KiB, 64 KiB-aligned staging buffer below 4 GB, freed on drop.
///
/// The same reasoning as `main.rs`'s boot-time bounce buffer and `net`'s
/// `DmaPool`: a firmware block driver may DMA straight into the buffer it is
/// handed, and a heap frame is under no obligation to be 32-bit reachable or
/// to avoid straddling a 64 KiB boundary. Under QEMU with `-m 2048` every
/// address is low, so this is a hardware-only failure mode the gates cannot
/// see — which is why it is here rather than a `Vec`.
pub struct Bounce {
    base: core::ptr::NonNull<u8>,
    pages: usize,
    aligned: *mut u8,
}

impl Bounce {
    /// Claim one. `None` means the firmware would not give us low memory,
    /// which every caller reports as [`FileErr::Io`].
    pub fn new() -> Option<Bounce> {
        // 128 KiB so a 64 KiB-aligned 64 KiB window is guaranteed inside it.
        let pages = (128 * 1024) / 4096;
        let base = uefi::boot::allocate_pages(
            uefi::boot::AllocateType::MaxAddress(0xFFFF_FFFF),
            uefi::boot::MemoryType::LOADER_DATA,
            pages,
        )
        .ok()?;
        let mut p = base.as_ptr() as usize;
        let rem = p % (64 * 1024);
        if rem != 0 {
            p += (64 * 1024) - rem;
        }
        Some(Bounce {
            base,
            pages,
            aligned: p as *mut u8,
        })
    }

    /// The 64 KiB window.
    pub fn buf(&mut self) -> &mut [u8] {
        // SAFETY: `aligned` points into the 128 KiB allocation `base`/`pages`
        // describe, at an offset of less than 64 KiB, so a 64 KiB window from
        // it stays inside the allocation. Nothing else holds a reference to
        // those bytes: the pool is one allocation owned by this struct and
        // handed out only through this `&mut self` borrow.
        unsafe { core::slice::from_raw_parts_mut(self.aligned, XFER_CHUNK) }
    }
}

impl Drop for Bounce {
    fn drop(&mut self) {
        // SAFETY: `base` and `pages` are exactly what `allocate_pages`
        // returned and nothing else has freed any of it. No firmware call is
        // outstanding against it — every reader and writer in this module
        // closes its file before the `Bounce` goes out of scope.
        unsafe {
            let _ = uefi::boot::free_pages(self.base, self.pages);
        }
    }
}

// ---------------------------------------------------------------------------
// Reading the volume
// ---------------------------------------------------------------------------

/// One `LS` line: a regular file in the boot volume root.
pub struct Entry {
    pub name: String,
    pub size: u64,
}

/// List regular files in the boot volume root (design §1.2/§1.3).
///
/// Returns the entries and whether the listing was cut at
/// [`LS_MAX_ENTRIES`]. Directories, the volume label and `.`/`..` are skipped
/// and **not counted** — §1.3 says `<n>` counts the entry lines that follow.
/// A name the protocol could not express (non-ASCII, over
/// [`NAME_MAX_BYTES`]) is skipped too: listing a name a client cannot then
/// `GET` would be a lie about what the box will serve.
pub fn ls(root: &mut Directory) -> Result<(Vec<Entry>, bool), FileErr> {
    root.reset_entry_readout().map_err(|_| FileErr::FwError)?;
    let mut out: Vec<Entry> = Vec::new();
    loop {
        let info = match root.read_entry_boxed() {
            Ok(None) => return Ok((out, false)),
            Ok(Some(i)) => i,
            Err(_) => return Err(FileErr::Io),
        };
        if info.is_directory() {
            continue;
        }
        let Some(name) = ascii_name(info.file_name()) else {
            continue;
        };
        if validate_name(&name).is_err() {
            continue;
        }
        out.push(Entry {
            name,
            size: info.file_size(),
        });
        if out.len() >= LS_MAX_ENTRIES {
            // §1.3: more than the cap is never an error — a box with 300
            // files is still a usable box — so the header says `truncated`
            // and the listing stops here.
            return Ok((out, true));
        }
    }
}

/// A firmware-returned name as ASCII, or `None` when it is not one the
/// protocol can carry.
fn ascii_name(n: &uefi::CStr16) -> Option<String> {
    let mut s = String::new();
    for c in n.iter() {
        let v = u16::from(*c);
        if v == 0 || v > 0x7f {
            return None;
        }
        s.push(v as u8 as char);
        if s.len() > NAME_MAX_BYTES {
            return None;
        }
    }
    if s.is_empty() { None } else { Some(s) }
}

/// Open a validated name for reading.
fn open_read(root: &mut Directory, name: &str) -> Result<RegularFile, FileErr> {
    let mut buf = [0u16; 32];
    let c = cstr(name, &mut buf)?;
    let handle = match root.open(c, FileMode::Read, FileAttribute::empty()) {
        Ok(h) => h,
        Err(e) if e.status() == uefi::Status::NOT_FOUND => return Err(FileErr::NotFound),
        Err(_) => return Err(FileErr::Io),
    };
    match handle.into_type() {
        Ok(FileType::Regular(f)) => Ok(f),
        // A directory is not `NOT_FOUND`, but it is not a file this protocol
        // can serve either, and there is no slug for "is a directory".
        Ok(_) => Err(FileErr::NotFound),
        Err(_) => Err(FileErr::Io),
    }
}

/// Size of `name` on the boot volume root (design §1.2 `STAT`).
pub fn stat(root: &mut Directory, name: &str) -> Result<u64, FileErr> {
    let mut f = open_read(root, name)?;
    let mut info_buf = [0u8; 512];
    let size = match f.get_info::<FileInfo>(&mut info_buf) {
        Ok(i) => i.file_size(),
        Err(_) => {
            f.close();
            return Err(FileErr::Io);
        }
    };
    f.close();
    Ok(size)
}

/// Stream a sha256 over `name`, re-arming the watchdog once per chunk.
///
/// Returns `(size, digest)`. This is both the `SHA` verb and the readback half
/// of `PUT`: the same code proves what is on the volume in both cases, so a
/// client's `SHA` after a `PUT` cannot disagree with the `OK` the `PUT`
/// returned for any reason other than the volume changing underneath.
///
/// `rearm` is called before every chunk (§8). It is a hook rather than a
/// direct `set_watchdog_timer` call so the caller — which knows whether it is
/// inside a job, a transfer or a reload — owns the window.
pub fn sha_named(
    root: &mut Directory,
    name: &str,
    rearm: &mut dyn FnMut(),
) -> Result<(u64, [u8; 32]), FileErr> {
    let mut bounce = Bounce::new().ok_or(FileErr::Io)?;
    let mut f = open_read(root, name)?;
    let mut info_buf = [0u8; 512];
    let size = match f.get_info::<FileInfo>(&mut info_buf) {
        Ok(i) => i.file_size(),
        Err(_) => {
            f.close();
            return Err(FileErr::Io);
        }
    };

    let mut h = aegis_core::witness::Sha256::new();
    let mut done: u64 = 0;
    while done < size {
        rearm();
        let want = core::cmp::min(XFER_CHUNK as u64, size - done) as usize;
        let buf = &mut bounce.buf()[..want];
        match f.read(buf) {
            Ok(0) => {
                f.close();
                return Err(FileErr::ShortRead);
            }
            Ok(n) => {
                h.update(&buf[..n]);
                done += n as u64;
            }
            Err(_) => {
                f.close();
                return Err(FileErr::Io);
            }
        }
    }
    f.close();
    if done != size {
        return Err(FileErr::ShortRead);
    }
    Ok((size, h.finalize()))
}

/// A file opened for a streamed read (`GET`'s second pass).
///
/// Design §1.4: `GET` reads the file twice. The header's `sha16` comes from a
/// full [`sha_named`] pass *before* the header goes on the wire, so a header
/// that arrives is a promise the bytes were readable once. This is the second
/// pass, and it has no way to fail politely — a header already sent cannot be
/// retracted, so the caller's only honest answer to an error here is to close.
pub struct Reader {
    file: RegularFile,
    pub size: u64,
    read: u64,
}

impl Reader {
    /// Open `name` for streaming and report its size.
    pub fn open(root: &mut Directory, name: &str) -> Result<Reader, FileErr> {
        let mut file = open_read(root, name)?;
        let mut info_buf = [0u8; 512];
        let size = match file.get_info::<FileInfo>(&mut info_buf) {
            Ok(i) => i.file_size(),
            Err(_) => {
                file.close();
                return Err(FileErr::Io);
            }
        };
        Ok(Reader {
            file,
            size,
            read: 0,
        })
    }

    /// Next chunk into `buf`, at most `buf.len()` bytes. `Ok(0)` only ever
    /// means the file is finished.
    pub fn next(&mut self, buf: &mut [u8]) -> Result<usize, FileErr> {
        if self.read >= self.size {
            return Ok(0);
        }
        let want = core::cmp::min(buf.len() as u64, self.size - self.read) as usize;
        match self.file.read(&mut buf[..want]) {
            Ok(0) => Err(FileErr::ShortRead),
            Ok(n) => {
                self.read += n as u64;
                Ok(n)
            }
            Err(_) => Err(FileErr::Io),
        }
    }

    /// Close the underlying handle. Consuming, so a `Reader` cannot outlive
    /// the firmware resource it holds.
    pub fn close(self) {
        self.file.close();
    }
}

// ---------------------------------------------------------------------------
// Free space (design §8)
// ---------------------------------------------------------------------------

/// Free bytes on the boot volume, as the firmware reports them.
///
/// **Advisory** (§8): the UEFI `FileSystemInfo` free-space number is not
/// reliable on every vendor FAT driver — some report it wrong, some not at
/// all. `None` means "the firmware would not say", and a caller must treat
/// that as permission to try, not as a refusal: exhaustion then surfaces
/// mid-stream as a short write, which is `ERR io` with the stage deleted.
/// `no-space` is best effort; `io` is the guarantee.
pub fn free_space(root: &mut Directory) -> Option<u64> {
    let mut buf = [0u8; 1024];
    root.get_info::<FileSystemInfo>(&mut buf)
        .ok()
        .map(|i| i.free_space())
}

// ---------------------------------------------------------------------------
// Writing: STAGE.PRT, readback, commit (design §8)
// ---------------------------------------------------------------------------

/// Is a `STAGE.PRT` sitting on the volume? (`HEALTH parts=`.)
pub fn stage_present(root: &mut Directory) -> bool {
    let mut buf = [0u16; 32];
    let Ok(c) = cstr(STAGE_NAME, &mut buf) else {
        return false;
    };
    match root.open(c, FileMode::Read, FileAttribute::empty()) {
        Ok(h) => {
            h.close();
            true
        }
        Err(_) => false,
    }
}

/// Delete `STAGE.PRT` at boot (design §8, "Orphans").
///
/// Returns `true` when there was one to sweep. The caller logs it: a box that
/// reappears with a stale stage is `SUSPECT` until an operator clears it, and
/// `BOOTLOG.TXT` is the only place that fact survives a reset.
pub fn sweep_parts(root: &mut Directory) -> bool {
    if !stage_present(root) {
        return false;
    }
    let _ = delete(root, STAGE_NAME);
    true
}

/// Delete a name from the boot volume root.
pub fn delete(root: &mut Directory, name: &str) -> Result<(), FileErr> {
    let mut buf = [0u16; 32];
    let c = cstr(name, &mut buf)?;
    let handle = match root.open(c, FileMode::ReadWrite, FileAttribute::empty()) {
        Ok(h) => h,
        Err(e) if e.status() == uefi::Status::NOT_FOUND => return Err(FileErr::NotFound),
        Err(_) => return Err(FileErr::Io),
    };
    match handle.into_type() {
        Ok(FileType::Regular(f)) => f.delete().map_err(|_| FileErr::Io),
        Ok(_) => Err(FileErr::NotFound),
        Err(_) => Err(FileErr::Io),
    }
}

/// Write a small file whole, replacing whatever was there.
///
/// **Not** the `STAGE.PRT` dance, deliberately. This exists for one caller —
/// `reload::set_pointer` writing `CURRENT.TXT` (§8) — and routing that write
/// through the stage would cost a second create/write/delete/rename cycle on
/// *every* artifact `PUT`, immediately after the first one, against the same
/// two directory entries. Directory churn is the one thing a FAT32 volume with
/// no journal has no defence against, and halving it is worth more here than a
/// readback would be, because §8 already gives `CURRENT.TXT` the only recovery
/// rule it needs: boot and `RELOAD` **fall back to the canonical names when it
/// is absent or unparsable**. A torn 48-byte write is therefore a recoverable
/// state by construction, whereas a torn `MODEL.SAF` is not — which is why one
/// is staged and the other is not.
///
/// Delete-then-create rather than open-and-overwrite: `CreateReadWrite` on an
/// existing file does not truncate, so a shorter body would leave the previous
/// tail past its end and `read_pointer` would parse two generations of keys.
pub fn write_small(root: &mut Directory, name: &str, bytes: &[u8]) -> Result<(), FileErr> {
    match delete(root, name) {
        Ok(()) | Err(FileErr::NotFound) => {}
        Err(e) => return Err(e),
    }
    let mut buf = [0u16; 32];
    let c = cstr(name, &mut buf)?;
    let handle = root
        .open(c, FileMode::CreateReadWrite, FileAttribute::empty())
        .map_err(|_| FileErr::Io)?;
    let mut file = match handle.into_type() {
        Ok(FileType::Regular(f)) => f,
        _ => return Err(FileErr::Io),
    };
    let wrote = file.write(bytes).map_err(|_| FileErr::ShortWrite);
    let flushed = file.flush().is_ok();
    file.close();
    wrote?;
    if flushed { Ok(()) } else { Err(FileErr::Io) }
}

/// The staging file of a `PUT`, open for writing.
///
/// The target is **never opened** while a `PUT` is in flight. Every byte goes
/// to [`STAGE_NAME`]; the target only changes at [`commit`], after the
/// readback digest matched. That is the whole of §8's half-written-file rule,
/// and it is why an aborted `PUT` costs a stale `STAGE.PRT` and nothing else.
pub struct Stage {
    file: RegularFile,
    written: u64,
}

impl Stage {
    /// Create (or truncate) `STAGE.PRT`.
    ///
    /// A stale stage is deleted first rather than written over: opening
    /// `CreateReadWrite` on a longer previous stage and writing a shorter
    /// payload would leave the old tail past the end, and the readback would
    /// then hash bytes this transfer never sent.
    pub fn create(root: &mut Directory) -> Result<Stage, FileErr> {
        let _ = delete(root, STAGE_NAME);
        let mut buf = [0u16; 32];
        let c = cstr(STAGE_NAME, &mut buf)?;
        let handle = root
            .open(c, FileMode::CreateReadWrite, FileAttribute::empty())
            .map_err(|_| FileErr::Io)?;
        match handle.into_type() {
            Ok(FileType::Regular(file)) => Ok(Stage { file, written: 0 }),
            _ => Err(FileErr::Io),
        }
    }

    /// Append a chunk. A firmware that took fewer bytes than it was handed is
    /// [`FileErr::ShortWrite`] — most often a full volume (§8), which is why
    /// `no-space` is advisory and this is the guarantee.
    pub fn write(&mut self, data: &[u8]) -> Result<(), FileErr> {
        match self.file.write(data) {
            Ok(()) => {
                self.written += data.len() as u64;
                Ok(())
            }
            Err(e) => {
                self.written += *e.data() as u64;
                Err(FileErr::ShortWrite)
            }
        }
    }

    /// Flush and close. The readback in the caller re-opens the name, so the
    /// close has to have happened first — a readback through the same handle
    /// would be answered from firmware state, not from the medium.
    pub fn finish(self) -> Result<u64, FileErr> {
        let mut file = self.file;
        let flushed = file.flush().is_ok();
        let n = self.written;
        file.close();
        if flushed { Ok(n) } else { Err(FileErr::Io) }
    }

    /// Abandon the transfer: close the handle and delete the stage.
    ///
    /// Every §1.4 abort path ends here, so "the stage is gone" is one line of
    /// code rather than a rule each caller has to remember.
    pub fn abandon(self, root: &mut Directory) {
        let mut file = self.file;
        let _ = file.flush();
        file.close();
        let _ = delete(root, STAGE_NAME);
    }
}

/// Commit the stage onto `target`: delete the target, then rename.
///
/// Delete-then-rename is the smallest commit FAT32 offers — there is no
/// journal and no atomic replace — and the order matters: the delete happens
/// **only after** the readback digest matched (the caller enforces that), so
/// the window in which neither file exists is as short as the firmware can
/// make it and is only ever entered with verified bytes ready to take the
/// name. A power loss inside it leaves neither, which `LS` shows and a
/// re-`PUT` fixes; §8's artifact pointer swap exists precisely so the three
/// files boot depends on never enter that window at all.
pub fn commit(root: &mut Directory, target: &str) -> Result<(), FileErr> {
    match delete(root, target) {
        Ok(()) | Err(FileErr::NotFound) => {}
        Err(e) => return Err(e),
    }
    rename_stage(root, target)
}

/// Rename `STAGE.PRT` to `target` through `set_info::<FileInfo>`.
///
/// UEFI has no rename call: changing the `FileName` field of a `FileInfo` and
/// setting it back is the rename, and the target must not already exist.
fn rename_stage(root: &mut Directory, target: &str) -> Result<(), FileErr> {
    let mut namebuf = [0u16; 32];
    let c = cstr(STAGE_NAME, &mut namebuf)?;
    let handle = match root.open(c, FileMode::ReadWrite, FileAttribute::empty()) {
        Ok(h) => h,
        Err(e) if e.status() == uefi::Status::NOT_FOUND => return Err(FileErr::NotFound),
        Err(_) => return Err(FileErr::Io),
    };
    let mut file = match handle.into_type() {
        Ok(FileType::Regular(f)) => f,
        _ => return Err(FileErr::Io),
    };

    // Read the current info so every field but the name is carried over
    // unchanged: `SetInfo` replaces the whole structure, and inventing
    // timestamps here would rewrite the file's history as a side effect of
    // renaming it.
    let mut read_buf = [0u8; 512];
    let (size, phys, create, access, modify, attr) = match file.get_info::<FileInfo>(&mut read_buf)
    {
        Ok(i) => (
            i.file_size(),
            i.physical_size(),
            *i.create_time(),
            *i.last_access_time(),
            *i.modification_time(),
            i.attribute(),
        ),
        Err(_) => {
            file.close();
            return Err(FileErr::Io);
        }
    };

    let mut tobuf = [0u16; 32];
    let to = match cstr(target, &mut tobuf) {
        Ok(t) => t,
        Err(e) => {
            file.close();
            return Err(e);
        }
    };
    // `FileInfo::new` builds in place and needs 8-byte alignment, which the
    // `u64` wrapper below guarantees without a runtime alignment dance.
    let mut store = InfoBuf([0u64; 64]);
    let bytes = store.bytes();
    let info = match FileInfo::new(bytes, size, phys, create, access, modify, attr, to) {
        Ok(i) => i,
        Err(_) => {
            file.close();
            return Err(FileErr::FwError);
        }
    };
    let res = file.set_info(info);
    file.close();
    res.map_err(|_| FileErr::Io)
}

/// 512 bytes of 8-byte-aligned scratch for [`FileInfo::new`].
///
/// A `FileInfo` header is 80 bytes and the name is at most
/// [`NAME_MAX_BYTES`] + 1 UCS-2 units, so 512 is comfortable. The `u64` array
/// is the alignment: `FileInfo` contains `u64` fields and `FileInfo::new`
/// rejects a misaligned buffer rather than fixing it.
#[repr(C)]
struct InfoBuf([u64; 64]);

impl InfoBuf {
    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: `[u64; 64]` is 512 initialised bytes with alignment 8, and
        // `u8` has no alignment or validity requirement that `u64` does not
        // already satisfy. The borrow is exclusive and lives only as long as
        // `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(self.0.as_mut_ptr().cast::<u8>(), 512) }
    }
}

// ---------------------------------------------------------------------------
// Hex
// ---------------------------------------------------------------------------

/// Full 64-character lower-case hex of a sha256.
#[must_use]
pub fn hex64(d: &[u8; 32]) -> String {
    let mut out = [0u8; 64];
    let n = aegis_core::witness::hex_lower(d, &mut out);
    String::from_utf8_lossy(&out[..n]).into_owned()
}

/// First 16 hex characters of a sha256 — the `sha16` of §1.1's `DATA` frame.
#[must_use]
pub fn hex16(d: &[u8; 32]) -> String {
    let mut s = hex64(d);
    s.truncate(16);
    s
}

/// Is `s` exactly 64 lower-case hex characters? The `PUT` digest argument
/// (§1.2) is checked before `SEND`, so a malformed one costs no bytes on the
/// wire (§1.4).
#[must_use]
pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}
