//! AEFINITY OS phase 4 — `EngineSlot` and the `RELOAD` verb.
//!
//! Contract: `program/AEFINITY_OS_FLEET_DESIGN.md` §4.1 (the ownership change
//! `RELOAD` forces), §8 (double capacity, the artifact pointer swap, the
//! watchdog rule), §1.2/§1.3 (`ERR reload-size`, `ERR reload-engine`,
//! `ERR busy-file`).
//!
//! # Why this module exists at all
//!
//! In v0.1 `main.rs` allocated three slabs at boot and handed
//! `&mut TernaryInferenceEngine` down through `job::dispatch` →
//! `dispatch_resident` → `server::run` → `serve` → `do_job`. The engine
//! borrows those slabs for the server's whole life, so there is no point at
//! which their bytes may be replaced: overwriting a slab under a live engine
//! would leave a box reporting a fresh `model_sha` while still inferring
//! against layout state derived from the old bytes — a wrong answer with a
//! correct-looking provenance line, which is the worst failure this program
//! can ship.
//!
//! [`EngineSlot`] **owns** the engine as an `Option`, so `RELOAD` can drop it,
//! refill the slabs, and build a new one. The three states are: engine
//! present (serving), engine absent (inside `reload`, and nowhere else), and
//! engine absent after a failed rebuild — which is not a state the box stays
//! in, because §4.1 requires a cold reset instead. **`RELOAD` never leaves a
//! box serving an engine it cannot describe.**
//!
//! # What this file is careful about
//!
//! - **Capacity is checked before anything is touched** (§4.1 step 2). A file
//!   bigger than its slab is `ERR reload-size` with the old engine still
//!   live. Growing a slab means a `REBOOT`, and saying so beats faulting in
//!   ring 0 — `main.rs`'s `STAGE 3 FAILED: contiguous alloc` is what a
//!   re-allocation attempt on a fragmented map actually looks like.
//! - **The watchdog is re-armed once per chunk of the load loop** (§8), the
//!   same rule the transfer and the readback follow.
//! - Rule A: nothing here measures anything.

use core::sync::atomic::{AtomicBool, Ordering};

use alloc::format;
use alloc::string::{String, ToString};

use aegis_core::inference::TernaryInferenceEngine;
use uefi::proto::media::file::Directory;

use crate::files::{self, Bounce, FileErr, XFER_CHUNK};

/// The canonical artifact names, and the `CURRENT.TXT` key that may point
/// each one somewhere else (§8).
const ARTIFACTS: [(&str, &str); 3] = [
    ("model", "MODEL.SAF"),
    ("embed", "EMBED.BIN"),
    ("vocab", "VOCAB.BIN"),
];

/// The `.NEW` half of each artifact's A/B pair, in [`ARTIFACTS`] order.
///
/// These are **internal names**, not part of the protocol surface: a client
/// names `MODEL.SAF` and §8's pointer swap decides which half the bytes land
/// on. `files::is_protected` consults this array so both `PUT` and `RM` refuse
/// them — a client that could name a half directly could delete the live one
/// and be silently served the stale one afterwards. One definition, here,
/// because a second list that drifted would re-open exactly that hole.
pub const ALTERNATES: [&str; 3] = ["MODEL.NEW", "EMBED.NEW", "VOCAB.NEW"];

/// The artifact pointer file of §8.
///
/// `model=MODEL.SAF|MODEL.NEW` (and the same for `embed`/`vocab`), one key per
/// line. Boot and `RELOAD` read it and **fall back to the canonical name when
/// it is absent or unparsable** — the smallest commit available on a
/// filesystem with no journal. Power loss before the pointer is written ⇒ old
/// weights; after ⇒ new; during ⇒ the fallback.
pub const CURRENT_NAME: &str = "CURRENT.TXT";

/// Why a `RELOAD` did not happen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReloadErr {
    /// An artifact is larger than the slab allocated for it at boot. **No
    /// state is touched** — the box is still serving the engine it had.
    Size,
    /// A file would not read, or is not there.
    File(FileErr),
    /// The bytes loaded but `TernaryInferenceEngine::new` refused them. The
    /// RAM copy is junk and the caller must cold-reset (§4.1 step 3).
    Engine,
}

impl ReloadErr {
    /// The byte-exact §1.3 slug, without the `ERR ` prefix.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            ReloadErr::Size => "reload-size",
            ReloadErr::File(e) => e.slug(),
            ReloadErr::Engine => "reload-engine",
        }
    }
}

/// One pinned, page-aligned buffer the engine reads from.
///
/// `capacity` is what the boot-time `allocate_huge_pages` claimed; `len` is
/// how many of those bytes are the current artifact. A reload may shrink
/// `len` freely and may never grow it past `capacity`.
pub struct Slab {
    base: core::ptr::NonNull<u8>,
    capacity: usize,
    len: usize,
}

impl Slab {
    /// Adopt a boot-time allocation.
    ///
    /// # Safety
    ///
    /// `base` must be the start of an allocation of at least `capacity`
    /// bytes that lives for the rest of the box's uptime and that nothing
    /// else writes to, and `len` must be at most `capacity`. `main.rs`'s
    /// three `allocate_huge_pages` regions are exactly that: they are never
    /// freed, and after this call the slot is their only writer.
    pub unsafe fn adopt(base: core::ptr::NonNull<u8>, capacity: usize, len: usize) -> Slab {
        Slab {
            base,
            capacity,
            len: len.min(capacity),
        }
    }

    /// The bytes the engine reads.
    ///
    /// The lifetime is the caller's to choose because the allocation outlives
    /// every borrow of it — see [`Slab::adopt`]'s safety contract.
    fn as_slice<'a>(&self) -> &'a [u8] {
        // SAFETY: `base`/`len` describe an initialised sub-range of the
        // allocation `adopt` was promised, which is never freed and which
        // this slot is the only writer of. No `&mut` to the same bytes is
        // live: `fill_from` takes `&mut self` and returns before any slice
        // handed out here is created.
        unsafe { core::slice::from_raw_parts(self.base.as_ptr(), self.len) }
    }

    /// Replace the contents from `name` on the boot volume.
    ///
    /// `rearm` is called once per [`XFER_CHUNK`] (§8): a 1.83 GB load through
    /// a FAT driver can outlast any single watchdog window, so the window is
    /// refreshed by progress rather than sized for the whole file.
    fn fill_from(
        &mut self,
        root: &mut Directory,
        name: &str,
        bounce: &mut Bounce,
        rearm: &mut dyn FnMut(),
    ) -> Result<(), ReloadErr> {
        let mut reader = files::Reader::open(root, name).map_err(ReloadErr::File)?;
        let size = reader.size as usize;
        if size > self.capacity {
            reader.close();
            return Err(ReloadErr::Size);
        }
        let mut done = 0usize;
        loop {
            rearm();
            let n = match reader.next(bounce.buf()) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => {
                    reader.close();
                    return Err(ReloadErr::File(e));
                }
            };
            // SAFETY: `done + n <= size <= capacity`, so the destination
            // range is inside the adopted allocation, and `bounce`'s window
            // is a separate allocation entirely — the two cannot overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bounce.buf().as_ptr(),
                    self.base.as_ptr().add(done),
                    n,
                );
            }
            done += n;
        }
        reader.close();
        if done != size {
            return Err(ReloadErr::File(FileErr::ShortRead));
        }
        self.len = done;
        Ok(())
    }
}

/// The three slabs a box boots with.
pub struct Slabs {
    pub model: Slab,
    pub embed: Slab,
    pub vocab: Slab,
}

/// Full sha256 of each resident buffer, lower-case hex.
///
/// `RESULT.TXT`'s `artifacts=` (design §3) is the first 16 of each of these,
/// and `merge_key` (§3.1) is built from the full 64. They are computed by
/// streaming over the **resident slices**, never re-read from FAT: the claim
/// is about what the engine is actually using.
#[derive(Clone)]
pub struct Digests {
    pub model: String,
    pub embed: String,
    pub vocab: String,
}

impl Digests {
    fn of(slabs: &Slabs, rearm: &mut dyn FnMut()) -> Digests {
        Digests {
            model: stream_hex(slabs.model.as_slice(), rearm),
            embed: stream_hex(slabs.embed.as_slice(), rearm),
            vocab: stream_hex(slabs.vocab.as_slice(), rearm),
        }
    }

    /// `<model16>/<embed16>/<vocab16>` — the `artifacts=` value of §3.
    #[must_use]
    pub fn artifacts_line(&self) -> String {
        format!(
            "{}/{}/{}",
            short(&self.model),
            short(&self.embed),
            short(&self.vocab)
        )
    }
}

/// First 16 characters of a 64-hex digest.
#[must_use]
pub fn short(hex: &str) -> &str {
    if hex.len() >= 16 { &hex[..16] } else { hex }
}

/// sha256 of a slice, re-arming per chunk so a 1.83 GB hash cannot trip the
/// watchdog it is running under (§8).
fn stream_hex(data: &[u8], rearm: &mut dyn FnMut()) -> String {
    let mut h = aegis_core::witness::Sha256::new();
    for part in data.chunks(XFER_CHUNK) {
        rearm();
        h.update(part);
    }
    files::hex64(&h.finalize())
}

/// The engine and the memory it reads, owned together.
///
/// The lifetime is the slabs' — which, per [`Slab::adopt`], is the box's
/// uptime. Nothing here borrows from `self`, so the engine can be dropped and
/// rebuilt without the borrow checker having to be argued with.
pub struct EngineSlot<'a> {
    slabs: Slabs,
    engine: Option<TernaryInferenceEngine<'a>>,
    reloads: u64,
    digests: Digests,
}

impl<'a> EngineSlot<'a> {
    /// Take ownership of the engine `main.rs` built and the slabs it reads.
    ///
    /// The digests are computed here, once, because every record and every
    /// `HEALTH` reply for the rest of this uptime quotes them.
    pub fn adopt(
        engine: TernaryInferenceEngine<'a>,
        slabs: Slabs,
        rearm: &mut dyn FnMut(),
    ) -> EngineSlot<'a> {
        let digests = Digests::of(&slabs, rearm);
        EngineSlot {
            slabs,
            engine: Some(engine),
            reloads: 0,
            digests,
        }
    }

    /// The engine, for a caller that is going to run a job with it.
    ///
    /// `None` is only reachable inside [`EngineSlot::reload`], which never
    /// returns to a caller in that state — a failed rebuild cold-resets.
    pub fn engine_mut(&mut self) -> Option<&mut TernaryInferenceEngine<'a>> {
        self.engine.as_mut()
    }

    /// Give the engine back to the boot path (`AFTER halt`, where
    /// `job::dispatch` returns and `main.rs` carries on).
    pub fn take_engine(&mut self) -> Option<TernaryInferenceEngine<'a>> {
        self.engine.take()
    }

    /// Digests of the three resident buffers.
    #[must_use]
    pub fn digests(&self) -> &Digests {
        &self.digests
    }

    /// How many successful reloads this uptime has seen.
    #[must_use]
    pub fn reloads(&self) -> u64 {
        self.reloads
    }

    /// Drop the engine, refill the slabs from the volume, and rebuild.
    ///
    /// Order, and why it is this order (§4.1):
    ///
    /// 1. Refuse while a `STAGE.PRT` exists — a reload racing an interrupted
    ///    `PUT` would load whichever half of the world happened to be on the
    ///    volume. `ERR busy-file`.
    /// 2. Check every file against its slab **first**. Too big is
    ///    `ERR reload-size` with nothing touched, so the caller can answer and
    ///    keep serving.
    /// 3. Only then drop the engine and overwrite the slabs. From here the
    ///    RAM copy is not the engine's any more, so there is no going back.
    /// 4. Rebuild. On failure the caller must cold-reset: the box holds bytes
    ///    it cannot describe, and Debian is the recovery partition.
    pub fn reload(
        &mut self,
        root: &mut Directory,
        rearm: &mut dyn FnMut(),
    ) -> Result<(), ReloadErr> {
        if files::stage_present(root) {
            return Err(ReloadErr::File(FileErr::BusyFile));
        }
        let names = current_names(root);

        // ---- 1. capacity, before anything is touched ----------------------
        for (i, name) in names.iter().enumerate() {
            let cap = match i {
                0 => self.slabs.model.capacity,
                1 => self.slabs.embed.capacity,
                _ => self.slabs.vocab.capacity,
            };
            match files::stat(root, name) {
                Ok(size) if size as usize <= cap => {}
                Ok(_) => return Err(ReloadErr::Size),
                Err(e) => return Err(ReloadErr::File(e)),
            }
        }

        // ---- 2. the point of no return ------------------------------------
        self.engine = None;
        let mut bounce = match Bounce::new() {
            Some(b) => b,
            None => return Err(ReloadErr::File(FileErr::Io)),
        };
        self.slabs
            .model
            .fill_from(root, &names[0], &mut bounce, rearm)?;
        self.slabs
            .embed
            .fill_from(root, &names[1], &mut bounce, rearm)?;
        self.slabs
            .vocab
            .fill_from(root, &names[2], &mut bounce, rearm)?;
        drop(bounce);

        // ---- 3. rebuild ----------------------------------------------------
        rearm();
        let engine = match TernaryInferenceEngine::new(
            self.slabs.embed.as_slice(),
            self.slabs.model.as_slice(),
            self.slabs.vocab.as_slice(),
        ) {
            Ok(e) => e,
            Err(why) => {
                // The box is holding bytes it cannot describe. Say why while
                // there is still a volume to say it on — the caller's next
                // move is a cold reset (§4.1 step 3), which takes BOOTLOG.TXT
                // with it as the only record of what happened.
                crate::boot_log(root, &format!("RELOAD: engine rebuild failed: {why}"));
                return Err(ReloadErr::Engine);
            }
        };
        self.engine = Some(engine);
        self.digests = Digests::of(&self.slabs, rearm);
        self.reloads += 1;
        Ok(())
    }
}

/// Has the pointer been seen designating a file that is not on the volume?
///
/// A `static` rather than a field on `Srv` because the fallback is decided in
/// [`current_names`], which boot, `RELOAD` and every read verb all reach
/// without a server in scope. Only the **transition** is logged: the fallback
/// is consulted several times per verb, and an `ERROR` line per `GET` would
/// bury the one that mattered under its own repetitions.
static POINTER_DEGRADED: AtomicBool = AtomicBool::new(false);

/// Which file each artifact key currently points at (§8), plus whether the
/// canonical-name fallback had to be used.
///
/// Absent, unreadable or unparsable `CURRENT.TXT` ⇒ the canonical names, and
/// that is **not** degraded: §8 says in as many words that boot falls back
/// when the pointer is absent or unparsable, so a box that has never swapped
/// is in its normal state, not a damaged one.
///
/// Degraded is the narrower case: the pointer parses, names a legal file, and
/// **that file is not there**. That cannot happen from a torn write — a torn
/// `CURRENT.TXT` is unparsable, not confidently wrong — so it means the file
/// the box was serving went away underneath the pointer. Falling back then
/// answers `STAT`/`SHA`/`GET`/`RELOAD` out of the *other*, stale half of the
/// A/B pair and reports ordinary success, which is the silent model downgrade
/// §8's swap exists to make impossible. The fallback still happens (serving
/// something beats faulting), but it stops being silent.
fn names_and_degraded(root: &mut Directory) -> ([String; 3], bool) {
    let mut out = [
        ARTIFACTS[0].1.to_string(),
        ARTIFACTS[1].1.to_string(),
        ARTIFACTS[2].1.to_string(),
    ];
    let Some(text) = read_pointer(root) else {
        return (out, false);
    };
    for line in text.lines() {
        let Some((k, v)) = line.trim().split_once('=') else {
            continue;
        };
        let Ok(name) = files::validate_name(v) else {
            continue;
        };
        for (i, (key, _)) in ARTIFACTS.iter().enumerate() {
            if k.trim().eq_ignore_ascii_case(key) {
                out[i] = name.clone();
            }
        }
    }
    // A pointer at a file that is not there is as unusable as no pointer at
    // all, and §8's rule is that the fallback is the canonical name. Checking
    // it here means one place decides, and boot, `RELOAD` and every read verb
    // all get the same answer.
    let mut degraded = false;
    for (i, (_, canonical)) in ARTIFACTS.iter().enumerate() {
        if out[i] != *canonical && files::stat(root, &out[i]).is_err() {
            degraded = true;
            out[i] = canonical.to_string();
        }
    }
    (out, degraded)
}

/// Which file each artifact key currently points at (§8).
///
/// Logs the first crossing into (and back out of) the degraded state defined
/// by [`names_and_degraded`]; [`pointer_degraded`] is the same answer for
/// `HEALTH`.
pub fn current_names(root: &mut Directory) -> [String; 3] {
    let (names, degraded) = names_and_degraded(root);
    note_degraded(root, degraded);
    names
}

/// `HEALTH degraded=` — recomputed, not cached, because the whole point is to
/// report the volume's state at the moment a scheduler asks.
pub fn pointer_degraded(root: &mut Directory) -> bool {
    let (_, degraded) = names_and_degraded(root);
    note_degraded(root, degraded);
    degraded
}

/// `BOOTLOG.TXT` on a change of state, and nothing on a repeat.
fn note_degraded(root: &mut Directory, degraded: bool) {
    if POINTER_DEGRADED.swap(degraded, Ordering::Relaxed) == degraded {
        return;
    }
    if degraded {
        crate::boot_log(
            root,
            "ERROR RELOAD: CURRENT.TXT designates an artifact file that is not on the \
             volume — falling back to the canonical name, which may be older bytes than \
             the pointer was written for. HEALTH now says degraded=pointer.",
        );
    } else {
        crate::boot_log(
            root,
            "RELOAD: CURRENT.TXT designates files that are all present again — \
             HEALTH degraded=none.",
        );
    }
}

/// Read `CURRENT.TXT` as text. Small by construction: three key lines.
fn read_pointer(root: &mut Directory) -> Option<String> {
    let mut reader = files::Reader::open(root, CURRENT_NAME).ok()?;
    if reader.size > 4096 {
        reader.close();
        return None;
    }
    let mut buf = alloc::vec![0u8; reader.size as usize];
    let mut got = 0usize;
    while got < buf.len() {
        match reader.next(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(_) => {
                reader.close();
                return None;
            }
        }
    }
    reader.close();
    buf.truncate(got);
    core::str::from_utf8(&buf).ok().map(ToString::to_string)
}

/// Point one artifact key at `name` (§8's pointer swap).
///
/// Written whole, every time, with the other two keys carried over: a
/// `CURRENT.TXT` holding one key would silently reset the other two to their
/// canonical names on the next boot.
///
/// Written **in place** through [`files::write_small`], not through
/// `STAGE.PRT`. The staged path would put a second create/write/delete/rename
/// cycle immediately after the one that just committed 1.8 GB, on the same two
/// directory entries, on a filesystem with no journal — and it would buy
/// nothing, because §8 gives this file the recovery rule a stage would
/// otherwise provide: absent or unparsable falls back to the canonical names.
/// See `files::write_small` for the full argument.
pub fn set_pointer(root: &mut Directory, key: &str, name: &str) -> Result<(), FileErr> {
    let names = current_names(root);
    let mut body = String::new();
    for (i, (k, _)) in ARTIFACTS.iter().enumerate() {
        let v = if k.eq_ignore_ascii_case(key) {
            name
        } else {
            &names[i]
        };
        body.push_str(&format!("{k}={v}\n"));
    }
    files::write_small(root, CURRENT_NAME, body.as_bytes())
}

/// Where a client-visible name actually lives right now (§8's pointer swap).
///
/// The pointer is **invisible to clients**: `STAT`/`SHA`/`GET MODEL.SAF` all
/// resolve through it, so `SHA` after a `PUT` returns the bytes that `PUT`
/// just verified even though they are physically in the other half of the A/B
/// pair. Without this a host tool that verifies its own upload — which
/// design §6 says `cm-os-fs` does on every `put` — would read the *previous*
/// model and report a mismatch on a perfectly good transfer.
///
/// Any name that is not one of the three canonical artifacts resolves to
/// itself, which is every ordinary file.
pub fn resolve(root: &mut Directory, name: &str) -> String {
    let Some(i) = ARTIFACTS.iter().position(|(_, c)| *c == name) else {
        return name.to_string();
    };
    let names = current_names(root);
    names[i].clone()
}

/// Where a `PUT` of a canonical artifact name should land: the half of the A/B
/// pair the pointer is **not** currently on (§8).
///
/// `MODEL.SAF` and `MODEL.NEW` alternate. Writing the inactive half means the
/// active artifact is never opened for writing at all, so the double-capacity
/// requirement (§8: `ALICE_UEFI` must hold ≥ 2× the artifact set) is what pays
/// for a commit that is a single pointer write instead of a delete-then-rename
/// window over the file the box boots from.
///
/// `None` for an ordinary file, which commits by rename as usual.
pub fn artifact_target(root: &mut Directory, name: &str) -> Option<(&'static str, String)> {
    let i = ARTIFACTS.iter().position(|(_, c)| *c == name)?;
    let (key, canonical) = ARTIFACTS[i];
    let alternate = alternate_name(canonical);
    let current = current_names(root);
    let dest = if current[i] == canonical {
        alternate.to_string()
    } else {
        canonical.to_string()
    };
    Some((key, dest))
}

/// The `.NEW` half of an artifact's A/B pair. Fixed 8.3 names, so both halves
/// are inside [`crate::files::NAME_MAX_BYTES`] by construction.
fn alternate_name(canonical: &str) -> &'static str {
    match ARTIFACTS.iter().position(|(_, c)| *c == canonical) {
        Some(i) => ALTERNATES[i],
        // Unreachable through any caller here — all three take an index into
        // ARTIFACTS first — but a wrong answer would be a wrong *file*, so the
        // fallback is the name that is never the live half of anything.
        None => ALTERNATES[2],
    }
}

/// Delete a client-visible artifact name **without ever orphaning the pointer**
/// (§8).
///
/// `None` when `name` is not one of the three canonical artifacts; the caller
/// then does an ordinary delete.
///
/// The failure this exists to prevent: `current_names` falls back to the
/// canonical name whenever `CURRENT.TXT` designates a file that is not there.
/// That fallback is right for a pointer torn by power loss and *wrong* for one
/// a client emptied out — deleting the designated half while the pointer still
/// names it makes `STAT`/`SHA`/`GET`/`RELOAD` answer from the other, stale half
/// with no error, so an operator who deleted a model would be served the
/// previous one and told it was fine. `RM MODEL.NEW` is refused outright
/// (`files::is_protected`); this handles `RM MODEL.SAF`, which is legal and
/// must mean *the artifact is gone*, not *the other half is now live*.
///
/// Order matters: the pointer stops designating anything **before** any bytes
/// go, so there is no instant at which it names a file that has been deleted.
pub fn remove_artifact(root: &mut Directory, name: &str) -> Option<Result<(), FileErr>> {
    let i = ARTIFACTS.iter().position(|(_, c)| *c == name)?;
    let (key, canonical) = ARTIFACTS[i];
    let alternate = alternate_name(canonical);

    if current_names(root)[i] != canonical
        && let Err(e) = set_pointer(root, key, canonical)
    {
        return Some(Err(e));
    }

    // Both halves go. A client asked for the artifact to be gone; leaving the
    // inactive half behind would make the next `PUT`'s A/B choice depend on a
    // file the client believes it deleted.
    let a = files::delete(root, canonical);
    let b = files::delete(root, alternate);
    Some(match (a, b) {
        (Err(FileErr::NotFound), Err(FileErr::NotFound)) => Err(FileErr::NotFound),
        (Err(e), _) if e != FileErr::NotFound => Err(e),
        (_, Err(e)) if e != FileErr::NotFound => Err(e),
        _ => Ok(()),
    })
}
