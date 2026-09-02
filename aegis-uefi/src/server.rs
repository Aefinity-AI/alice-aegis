//! AEFINITY OS phase 2 — the resident TCP job server.
//!
//! Spec: `program/AEFINITY_OS.md` §4 (the wire protocol), §5 (this file), §6
//! (`resident-test`).
//!
//! A stick whose `JOB.TXT` says `MODE resident` does not run a job and reset.
//! It brings the NIC up, listens on `LISTEN` (default 4242), and serves one
//! client at a time over a line-oriented protocol:
//!
//! ```text
//! S: AEFINITY-OS 0.1 READY env=<iron|vm> cpu=<brand>\n
//! C: PING\n                       S: PONG\n
//! C: JOB\n<JOB.TXT body>\nEND\n   S: RUNNING\n … S: RESULT\n<RESULT.TXT body>END\n
//! C: REBOOT\n                     S: BYE\n  → ResetSystem(COLD)
//! C: HALT\n                       S: BYE\n  → close listener, park
//! ```
//!
//! Phase 4 (`program/AEFINITY_OS_FLEET_DESIGN.md` §1) adds the file plane and
//! the fleet verbs on top of exactly that, without changing any of it:
//!
//! ```text
//! C: AUTH <64hex>\n              S: OK\n | ERR auth\n
//! C: LS\n                        S: LS <n> <ok|truncated>\n {<NAME> <size>\n}*n END\n
//! C: STAT <NAME>\n               S: STAT <NAME> <size>\n | ERR <e>\n
//! C: SHA <NAME>\n                S: SHA <NAME> <size> <64hex>\n | ERR <e>\n
//! C: GET <NAME>\n                S: DATA <len> <sha16>\n<bytes>END\n | ERR <e>\n
//! C: PUT <NAME> <len> <64hex>\n  S: SEND\n | ERR <e>\n
//! C: <len raw bytes>END\n        S: OK <NAME> <len> <sha16>\n | ERR <e>\n
//! C: RM <NAME>\n                 S: OK <NAME>\n | ERR <e>\n
//! C: RELOAD\n                    S: RELOADING\n … OK reload model=… | ERR <e>\n
//! C: HEALTH\n                    S: HEALTH up=… served=… last=… …\n
//! C: RUNID <id>\n                S: NEW\n | REPLAY\n
//! ```
//!
//! Three rules bind all of them and are worth stating once:
//!
//! - **`AUTH` gates everything but `PING` and `AUTH`** when `JOB.TXT` carries
//!   a `TOKEN` (§1.2) — reads included, because `GET MODEL.SAF` is
//!   exfiltration and a `HEALTH` reply publishes artifact digests. Absent a
//!   `TOKEN` the box is open, which is exactly v0.1 behaviour, so no v0.1
//!   gate changes.
//! - **No verb ever returns a partial success** (§1.3), and **no frame is
//!   ever truncated on `GET`** (§1.4): once a `DATA` header is on the wire the
//!   server cannot retract it, so the only honest failure left is to close.
//! - **The watchdog is armed for the duration of a transfer and re-armed per
//!   chunk** — during the transfer, during the sha256 readback, and during
//!   `RELOAD`'s load loop (§8) — and disarmed on every exit path.
//!
//! Anything else gets `ERR unknown\n`; a job body over the caps of §4 gets
//! `ERR too-large\n` and the connection is dropped; a second concurrent peer
//! gets `BUSY\n` and is closed.
//!
//! # What this file is careful about
//!
//! - **The box must stay reachable.** Every failure path here ends back at
//!   `accept`, not in a wedge: a peer that vanishes mid-job, a send that
//!   fails, a body that overruns its cap, a client that connects and says
//!   nothing. The only two ways out are `REBOOT` and `HALT`, both asked for.
//! - **The watchdog is off while idle and on while working** (spec §4). A
//!   resident box waiting for work must never be reset for waiting; a job
//!   that hangs somewhere this code cannot see must be. `run` arms
//!   `BUDGET + 60` at `JOB` and disarms the moment `run_job` returns.
//! - **`RESULT.TXT` is written to the volume as well as to the socket.** The
//!   socket answer is for the client; the file is for whoever pulls the stick
//!   or boots Debian on the same disk later and wants to know what the box
//!   last did.
//! - **Nothing here measures anything** (CLAUDE.md Rule A). The timeouts are
//!   protocol bounds. The record carries `env=vm` under QEMU exactly as the
//!   oneshot path's does.
//! - No `unwrap` on a firmware call, and every state change gets a
//!   `BOOTLOG.TXT` line, because a headless box in a rack has no other voice.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use uefi::proto::media::file::Directory;

use crate::files::{self, FileErr};
use crate::job::{self, Job, Mode};
use crate::net::{Net, NetError, TcpHandle, Until};
use crate::reload::{EngineSlot, ReloadErr};

/// Protocol version in the READY banner (spec §4). Same `0.1` the record's
/// `aefinity_os=` key carries — one version for the whole OS surface.
const PROTO_VERSION: &str = "0.1";

/// Whole job body cap (spec §4: "Job bodies are capped at 64 KiB").
const BODY_MAX_BYTES: usize = 64 * 1024;
/// Single `PROMPT` line cap (spec §4: "`PROMPT` lines at 4 KiB"). Measured on
/// the value, not the whole line: the directive keyword is not the prompt.
const PROMPT_MAX_BYTES: usize = 4 * 1024;
/// A line longer than a whole legal body is over the cap whatever it is.
///
/// This is the accumulated-across-reads check; a single read that fills the
/// stack's own `RECV_MAX_BYTES` without a newline is caught one layer down and
/// arrives here as [`NetError::TooMuchData`]. Both land on
/// [`LineErr::TooLong`], so the answer is `ERR too-large` either way and does
/// not depend on which cap happens to be the smaller number.
const LINE_MAX_BYTES: usize = BODY_MAX_BYTES;

/// How long one read pass waits for bytes before it goes round again. Not a
/// deadline: an idle client simply loops here, which is what "a resident box
/// idles indefinitely" (spec §4) means in code.
const READ_SLICE_MS: u64 = 2_000;
/// How long a short reply (`PONG`, `BYE`, `ERR …`, the banner) is given to
/// reach the peer.
const SEND_MS: u64 = 15_000;
/// How long the `RESULT` body is given. Larger than [`SEND_MS`] because the
/// body is the one message that can be tens of kilobytes.
const SEND_RESULT_MS: u64 = 60_000;
/// Watchdog window covering the post-job record: the `RESULT.TXT` write to the
/// boot volume and the `RESULT` send that follows it.
///
/// The job's own window (`budget_s + WATCHDOG_MARGIN_S`) may be entirely spent
/// by the time the job ends, so this is armed fresh rather than inherited.
/// Sized for what is actually left to do: [`SEND_RESULT_MS`] (60 s) for the
/// send, plus the FAT write, plus slack. It is deliberately not `0` — a FAT
/// write that hangs on a real stick, a USB controller that stops answering,
/// a firmware `Flush()` that never returns, is precisely the failure a
/// headless box in a rack has no other way out of.
const RECORD_WATCHDOG_S: u64 = 120;
/// The four-way close. Short: the exchange is over by then, and a peer that
/// will not finish the close must not hold the listener.
const CLOSE_MS: u64 = 2_000;
/// How long the accept loop stalls between passes. Same order as the stack's
/// own poll sleep — long enough not to be a spin, short enough that a client
/// does not wait on us.
const ACCEPT_SLEEP_MS: u64 = 20;

/// A connection that has sent nothing for this long is dropped and the box
/// goes back to `accept`.
///
/// Spec §4 says a resident box idles indefinitely, and it does — as a *box*.
/// This bounds one silent *connection*, which is a different thing: smoltcp
/// has one socket per listener, so a peer that connects, says nothing and
/// never closes (a host that lost power, a forwarder whose far end went away)
/// would otherwise hold the only slot and make the box unreachable while it
/// is still perfectly healthy. Fifteen minutes is far longer than any client
/// exchange and far shorter than "for ever". The drop is logged.
const IDLE_DROP_MS: u64 = 15 * 60 * 1000;

// ---------------------------------------------------------------------------
// Phase 4 constants (design §1.3)
// ---------------------------------------------------------------------------

/// `(run_id, rendered RESULT.TXT)` pairs the box remembers (design §1.3).
///
/// Eight is the retention, not a queue depth: the ring is the at-most-once
/// primitive, and a controller that has more than eight units in flight
/// against **one** box has already broken the one-client-at-a-time rule.
const RECORD_MAX: usize = 8;
/// Bytes retained per record ⇒ the ring is at most 64 KiB of heap.
///
/// A rendered record above this (reachable only through many steps'
/// `job.N.detail`) is retained **truncated**, with a final `truncated=true`
/// line. The job is still not re-run — at-most-once is preserved — and the
/// replay is honest about being incomplete (§1.4).
const RECORD_MAX_BYTES: usize = 8_192;
/// Bounds **no progress** on a transfer, not its duration (§1.3).
const XFER_STALL_MS: u64 = 30_000;
/// How long one chunk of a `GET` is given to reach the peer. The transfer as a
/// whole is unbounded by design — a 1.83 GB file over a slow link is slow, not
/// broken — and it is progress, not elapsed time, that the box insists on.
const XFER_SEND_MS: u64 = 30_000;

/// The capability token in the READY banner (design §1).
///
/// §1 fixes the banner as `caps=files,lab`. This build ships **`files` only**:
/// phase 5 has not landed, and a box that advertises `lab` before `lab.rs`
/// exists would be telling a scheduler it can take `EVAL` work it will answer
/// `ERR unknown` to. Phase 5 changes this one constant. Recorded in
/// `docs/AEFINITY_OS_STATUS.md` as a deliberate deviation.
const CAPS: &str = "files";

// ---------------------------------------------------------------------------
// Entry point (spec §5)
// ---------------------------------------------------------------------------

/// Serve jobs over TCP until a client says `REBOOT` or `HALT`. Never returns.
///
/// Takes the [`Net`] by value: `HALT` has to shut the NIC down and release the
/// exclusive SNP open before parking, and `REBOOT` has to do the same before
/// `ResetSystem`. Both are `Net`'s `Drop`, which needs ownership.
pub fn run(mut net: Net, root: &mut Directory, slot: &mut EngineSlot<'_>, cfg: &Job) -> ! {
    let port = cfg.listen;

    // Idle: no watchdog. `dispatch_resident` already disarmed it; this is the
    // statement that the idle state owns that fact, not the caller.
    job::arm_watchdog(0);

    let mut listener = match net.tcp_listen(port) {
        Ok(h) => h,
        Err(e) => {
            crate::boot_log(
                root,
                &format!("RESIDENT: cannot listen on {port}: {}", e.as_str()),
            );
            say(&format!(
                "[AEFINITY OS] RESIDENT: cannot listen on {port}\r\n"
            ));
            park(net, root);
        }
    };

    // Everything the box knows about itself for as long as the listener lives.
    // `up_base_ms` is taken from the stack's own clock at this point and not
    // before: `HEALTH up=` is the **listener's** uptime, which is precisely
    // what design §1.4's ring-loss rule needs — a scheduler that sees `up=`
    // go backwards knows the `RUNID` ring is gone, whether the box rebooted or
    // only the server did.
    let mut srv = Srv {
        root,
        slot,
        cfg,
        ring: Vec::new(),
        served: 0,
        up_base_ms: net.now_ms(),
        wd_s: 0,
        last: String::from("none"),
    };
    if srv.cfg.token.is_some() {
        srv.log("RESIDENT: TOKEN present — every verb but PING/AUTH is gated");
    }

    loop {
        // ---- accept ------------------------------------------------------
        while !net.tcp_accepted(&listener) {
            stall(ACCEPT_SLEEP_MS);
        }
        srv.served += 1;
        let served = srv.served;
        srv.log(&format!(
            "RESIDENT: connection {served} accepted on port {port}"
        ));

        // One backlog socket, listening on the same port for the whole time
        // the connection above is being served. It is what makes the `BUSY`
        // answer of spec §4 possible — without it a second peer's SYN goes
        // unanswered and the client sees a hang rather than a refusal — and
        // when this connection ends it is already listening, so it becomes
        // the next listener with no gap in which the box is undialable.
        let mut spare = net.tcp_listen(port).ok();
        if spare.is_none() {
            srv.log("RESIDENT: no backlog socket — a second peer will not get BUSY");
        }

        let outcome = serve(&mut net, &listener, &mut spare, port, &mut srv);

        net.tcp_close(listener, CLOSE_MS);
        srv.log(&format!("RESIDENT: connection {served} closed"));

        match outcome {
            Outcome::Next => {
                listener = match spare.take() {
                    Some(s) => s,
                    None => match net.tcp_listen(port) {
                        Ok(h) => h,
                        Err(e) => {
                            srv.log(&format!(
                                "RESIDENT: cannot re-listen on {port}: {} — parking",
                                e.as_str()
                            ));
                            park(net, srv.root);
                        }
                    },
                };
            }
            Outcome::Reboot => {
                if let Some(s) = spare.take() {
                    net.tcp_close(s, CLOSE_MS);
                }
                srv.log("RESIDENT: REBOOT — resetting");
                // Shut the NIC down and release the SNP open before handing
                // the machine to the firmware.
                drop(net);
                job::after(job::After::Reset, srv.root);
                // `After::Reset` does not return. If a firmware ignored
                // ResetSystem, parking is the honest ending — the client has
                // already been told BYE and this box must not carry on
                // serving as if nothing was asked of it.
                srv.log("RESIDENT: firmware did not reset — parking");
                park_no_net(srv.root);
            }
            Outcome::Halt => {
                if let Some(s) = spare.take() {
                    net.tcp_close(s, CLOSE_MS);
                }
                srv.log("RESIDENT: HALT — listener closed, parking");
                park(net, srv.root);
            }
        }
    }
}

/// Everything the listener owns for as long as it lives.
///
/// It exists because phase 4's verbs need five long-lived things at once — the
/// volume, the engine slot, the job config, the `RUNID` ring and the counters
/// `HEALTH` reports — and threading five more parameters through `serve` and
/// every verb would have made the argument lists unreadable without making
/// anything safer.
///
/// **The ring is RAM only** (design §1.4). A watchdog reset, `REBOOT` or power
/// loss empties it, so a re-issued id afterwards answers `NEW` and the job runs
/// a second time. At-most-once holds *within one box uptime*, nothing more, and
/// `HEALTH up=` is how a scheduler detects that the guarantee lapsed.
struct Srv<'r, 'e> {
    root: &'r mut Directory,
    slot: &'r mut EngineSlot<'e>,
    cfg: &'r Job,
    /// `(run_id, rendered record)`, oldest first, at most [`RECORD_MAX`].
    ring: Vec<(String, String)>,
    /// Connections accepted since the listener came up.
    served: u64,
    /// The stack clock at listener start, for `HEALTH up=`.
    up_base_ms: i64,
    /// The watchdog window currently armed, for `HEALTH wd=`. `0` is off.
    wd_s: u64,
    /// `HEALTH last=` — `OK` | `FAIL <reason>` | `none`.
    last: String,
}

impl Srv<'_, '_> {
    fn log(&mut self, msg: &str) {
        crate::boot_log(self.root, msg);
    }

    /// Arm the firmware watchdog and remember what was armed, so `HEALTH wd=`
    /// reports the box's actual state rather than a constant.
    fn arm(&mut self, secs: u64) {
        self.wd_s = secs;
        job::arm_watchdog(secs);
    }

    /// Seconds since the listener came up.
    fn uptime_s(&self, net: &Net) -> u64 {
        let d = net.now_ms() - self.up_base_ms;
        if d <= 0 { 0 } else { (d as u64) / 1000 }
    }

    /// Remember a finished record against its `run_id`.
    ///
    /// Over [`RECORD_MAX_BYTES`] the body is kept truncated on a line boundary
    /// with a final `truncated=true`, because §1.4 is explicit that a large
    /// record must not become a reason to re-run a job: at-most-once wins, and
    /// the replay says it is incomplete.
    fn remember(&mut self, id: &str, body: &str) {
        let stored = if body.len() <= RECORD_MAX_BYTES {
            body.to_string()
        } else {
            let mut cut = RECORD_MAX_BYTES;
            while cut > 0 && body.as_bytes()[cut - 1] != b'\n' {
                cut -= 1;
            }
            let mut t = body[..cut].to_string();
            t.push_str("truncated=true\n");
            t
        };
        self.ring.retain(|(k, _)| k != id);
        if self.ring.len() >= RECORD_MAX {
            self.ring.remove(0);
        }
        self.ring.push((id.to_string(), stored));
    }

    /// The record for `id`, with `replay=` flipped to `true`.
    ///
    /// The stored body was rendered `replay=false` because it *was* a real run.
    /// Flipping the one line on the way out is what makes the served copy
    /// honest about being a replay without keeping two renderings of the same
    /// record in a 64 KiB ring.
    fn replay_of(&self, id: &str) -> Option<String> {
        let (_, body) = self.ring.iter().find(|(k, _)| k == id)?;
        Some(body.replace("\nreplay=false\n", "\nreplay=true\n"))
    }
}

/// Per-connection state. Reset on every accept, which is what makes `AUTH`
/// a property of the connection and not of the box.
struct Session {
    /// `true` once a correct `AUTH` arrived, or when no `TOKEN` is configured.
    authed: bool,
    /// The `RUNID` in force for the next `JOB`, and whether it was a replay.
    runid: Option<(String, bool)>,
}

/// Why [`serve`] gave the connection back.
enum Outcome {
    /// The exchange ended; go back to `accept`.
    Next,
    /// The client asked for `REBOOT`.
    Reboot,
    /// The client asked for `HALT`.
    Halt,
}

/// Serve one connection: banner, then commands until the client leaves or asks
/// the box to stop.
fn serve(
    net: &mut Net,
    conn: &TcpHandle,
    spare: &mut Option<TcpHandle>,
    port: u16,
    srv: &mut Srv<'_, '_>,
) -> Outcome {
    let served = srv.served;
    if send(net, conn, &banner(), SEND_MS).is_err() {
        srv.log("RESIDENT: peer went away before the READY banner");
        return Outcome::Next;
    }
    srv.log(&format!("RESIDENT: READY sent to connection {served}"));

    // Absent a `TOKEN` the box is open — exactly v0.1 behaviour (design §1.2),
    // which is why no v0.1 gate had to change for phase 4.
    let mut sess = Session {
        authed: srv.cfg.token.is_none(),
        runid: None,
    };

    let mut rd = Lines::new();
    loop {
        let line = match rd.read_line(net, conn, spare, port, srv.root) {
            Ok(l) => l,
            Err(LineErr::Closed) => {
                srv.log("RESIDENT: peer closed the connection");
                return Outcome::Next;
            }
            Err(LineErr::Idle) => {
                srv.log(&format!(
                    "RESIDENT: peer silent for {IDLE_DROP_MS} ms — dropping"
                ));
                return Outcome::Next;
            }
            Err(LineErr::TooLong) => {
                srv.log("RESIDENT: command line over the cap — ERR too-large");
                let _ = send(net, conn, "ERR too-large\n", SEND_MS);
                return Outcome::Next;
            }
        };

        // A bare CRLF or a blank keep-alive line is not a command and is not
        // an error either; answering `ERR unknown` to one would make every
        // client that ends its writes with a spare newline look broken.
        if line.trim().is_empty() {
            continue;
        }

        let trimmed = line.trim();
        let (head, rest) = split_verb(trimmed);
        let mut key = [0u8; 8];
        let verb = upper(head, &mut key);

        // Design §1.2: with a `TOKEN` set, **every** verb but `PING` and
        // `AUTH` answers `ERR auth` until this connection has authenticated —
        // reads included, because `GET MODEL.SAF` is exfiltration and a
        // `HEALTH` reply publishes artifact digests. The check is here, once,
        // rather than in each verb, so a verb added later cannot forget it.
        if !sess.authed && !matches!(verb, "PING" | "AUTH") {
            srv.log(&format!("RESIDENT: {verb} refused — not authenticated"));
            if send(net, conn, "ERR auth\n", SEND_MS).is_err() {
                return Outcome::Next;
            }
            continue;
        }

        let step = match verb {
            "PING" => reply(net, conn, "PONG\n"),
            "AUTH" => do_auth(net, conn, srv, &mut sess, rest),
            "LS" => do_ls(net, conn, srv),
            "STAT" => do_stat(net, conn, srv, rest),
            "SHA" => do_sha(net, conn, srv, rest),
            "GET" => do_get(net, conn, srv, rest),
            "PUT" => do_put(net, conn, srv, &mut rd, rest),
            "RM" => do_rm(net, conn, srv, rest),
            "RELOAD" => do_reload(net, conn, srv),
            "HEALTH" => do_health(net, conn, srv),
            "RUNID" => do_runid(net, conn, srv, &mut sess, rest),
            "JOB" => {
                let out = do_job(net, conn, spare, port, srv, &mut sess, &mut rd);
                match out {
                    JobOutcome::Served => Step::Continue,
                    JobOutcome::Dropped => Step::Drop,
                }
            }
            "REBOOT" => {
                srv.log("RESIDENT: REBOOT requested");
                let _ = send(net, conn, "BYE\n", SEND_MS);
                return Outcome::Reboot;
            }
            "HALT" => {
                srv.log("RESIDENT: HALT requested");
                let _ = send(net, conn, "BYE\n", SEND_MS);
                return Outcome::Halt;
            }
            _ => {
                srv.log(&format!("RESIDENT: unknown command {:?}", clip(&line, 64)));
                reply(net, conn, "ERR unknown\n")
            }
        };
        match step {
            Step::Continue => {}
            Step::Drop => return Outcome::Next,
            Step::Reset => {
                // Design §4.1: a `RELOAD` that could not rebuild the engine
                // leaves the box holding bytes it cannot describe. It must not
                // serve, and there is no self-healing inside the unikernel —
                // the cold reset spends `BootNext` and lands in Debian, which
                // is the recovery partition. The client already has its
                // `ERR reload-engine`; `do_reload` sends it before returning
                // here, precisely so this path can be unconditional.
                srv.log("RESIDENT: RELOAD left no engine — cold reset");
                if let Some(sp) = spare.take() {
                    net.tcp_close(sp, CLOSE_MS);
                }
                job::arm_watchdog(0);
                job::after(job::After::Reset, srv.root);
                srv.log("RESIDENT: firmware did not reset after a failed RELOAD — parking");
                park_no_net(srv.root);
            }
        }
    }
}

/// What one verb left the connection in.
enum Step {
    /// Keep serving this client.
    Continue,
    /// The connection is finished with (§1.4's `closed` column, or the peer
    /// left).
    Drop,
    /// The box cannot go on serving; cold-reset it.
    Reset,
}

/// Split a command line into its verb and the rest, on the first run of
/// whitespace. A verb with no argument gets an empty `rest`.
fn split_verb(line: &str) -> (&str, &str) {
    match line.find(char::is_whitespace) {
        Some(i) => (&line[..i], line[i..].trim()),
        None => (line, ""),
    }
}

/// Send a short answer; a peer that has gone away ends the connection.
fn reply(net: &mut Net, conn: &TcpHandle, msg: &str) -> Step {
    if send(net, conn, msg, SEND_MS).is_err() {
        Step::Drop
    } else {
        Step::Continue
    }
}

/// `ERR <slug>\n` for a file-plane failure.
fn reply_err(net: &mut Net, conn: &TcpHandle, e: FileErr) -> Step {
    reply(net, conn, &format!("ERR {}\n", e.slug()))
}

// ---------------------------------------------------------------------------
// Phase 4 verbs (design §1.2), in the order §1.2 lists them
// ---------------------------------------------------------------------------

/// `AUTH <64hex>` → `OK` | `ERR auth`.
///
/// The compare is constant-time in the bytes: a byte-at-a-time `==` that
/// returns early leaks the length of the matching prefix to anyone who can
/// time the answer, and a box on a lab LAN is exactly where that matters. The
/// *lengths* still differ observably, which is inherent and not secret — the
/// token's length is a configuration fact, not a credential.
///
/// A failed `AUTH` does not close the connection. Closing would make an
/// operator with a stale token watch a box refuse to talk at all; refusing
/// each verb by name is both kinder and more debuggable, and the attempt is in
/// `BOOTLOG.TXT` either way.
fn do_auth(
    net: &mut Net,
    conn: &TcpHandle,
    srv: &mut Srv<'_, '_>,
    sess: &mut Session,
    arg: &str,
) -> Step {
    let Some(want) = srv.cfg.token.clone() else {
        // No TOKEN configured: the box is open, and `AUTH` is then a no-op
        // that succeeds. Answering `ERR auth` here would make a client that
        // always authenticates fail against an open box for no reason.
        sess.authed = true;
        return reply(net, conn, "OK\n");
    };
    if ct_eq(arg.as_bytes(), want.as_bytes()) {
        sess.authed = true;
        srv.log("RESIDENT: AUTH ok");
        reply(net, conn, "OK\n")
    } else {
        srv.log("RESIDENT: AUTH failed");
        reply(net, conn, "ERR auth\n")
    }
}

/// Byte-wise equality that always looks at every byte it can.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff: u8 = u8::from(a.len() != b.len());
    let n = core::cmp::min(a.len(), b.len());
    for i in 0..n {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// `LS` → `LS <n> <ok|truncated>\n{<NAME> <size>\n}*n END\n`.
///
/// `<n>` counts the entry lines that follow and does **not** count `END`
/// (§1.3). More than `LS_MAX_ENTRIES` real entries is never an error — a box
/// with 300 files is still a usable box — so the header says `truncated` and
/// the listing stops.
fn do_ls(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>) -> Step {
    let (entries, truncated) = match files::ls(srv.root) {
        Ok(v) => v,
        Err(e) => return reply_err(net, conn, e),
    };
    let mut out = format!(
        "LS {} {}\n",
        entries.len(),
        if truncated { "truncated" } else { "ok" }
    );
    for e in &entries {
        out.push_str(&format!("{} {}\n", e.name, e.size));
    }
    out.push_str("END\n");
    reply(net, conn, &out)
}

/// `STAT <NAME>` → `STAT <NAME> <size>` | `ERR <e>`.
fn do_stat(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>, arg: &str) -> Step {
    let name = match files::validate_name(arg) {
        Ok(n) => n,
        Err(e) => return reply_err(net, conn, e),
    };
    let on_disk = crate::reload::resolve(srv.root, &name);
    match files::stat(srv.root, &on_disk) {
        Ok(size) => reply(net, conn, &format!("STAT {name} {size}\n")),
        Err(e) => reply_err(net, conn, e),
    }
}

/// `SHA <NAME>` → `SHA <NAME> <size> <64hex>` | `ERR <e>`.
///
/// Streaming, and watchdog-guarded: a 1.83 GB hash at FAT read speed can
/// outlast any single window, so the window is short and every chunk refreshes
/// it (§8). Disarmed on every exit path below.
fn do_sha(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>, arg: &str) -> Step {
    let name = match files::validate_name(arg) {
        Ok(n) => n,
        Err(e) => return reply_err(net, conn, e),
    };
    let on_disk = crate::reload::resolve(srv.root, &name);
    srv.arm(files::FILES_WD_S);
    let out = files::sha_named(srv.root, &on_disk, &mut wd_rearm());
    srv.arm(0);
    match out {
        Ok((size, d)) => reply(
            net,
            conn,
            &format!("SHA {name} {size} {}\n", files::hex64(&d)),
        ),
        Err(e) => reply_err(net, conn, e),
    }
}

/// `GET <NAME>` → `DATA <len> <sha16>\n<bytes>END\n` | `ERR <e>`.
///
/// **The file is read twice** (design §1.4). The header's `sha16` comes from a
/// full pass *before* the header goes on the wire, so a header that arrives is
/// a promise the bytes were readable once. After that the server cannot
/// retract, and **no frame is ever truncated**: an error during the second
/// pass closes the connection, and the client sees a short read with no
/// `END\n` — which is unambiguous, because the length was declared.
fn do_get(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>, arg: &str) -> Step {
    let name = match files::validate_name(arg) {
        Ok(n) => n,
        Err(e) => return reply_err(net, conn, e),
    };
    let on_disk = crate::reload::resolve(srv.root, &name);

    srv.arm(files::FILES_WD_S);
    let head = files::sha_named(srv.root, &on_disk, &mut wd_rearm());
    let (size, digest) = match head {
        Ok(v) => v,
        Err(e) => {
            srv.arm(0);
            return reply_err(net, conn, e);
        }
    };
    if size > files::GET_MAX_BYTES {
        srv.arm(0);
        srv.log(&format!(
            "RESIDENT: GET {name} over the frame cap — ERR too-large"
        ));
        return reply(net, conn, "ERR too-large\n");
    }

    let mut reader = match files::Reader::open(srv.root, &on_disk) {
        Ok(r) => r,
        Err(e) => {
            srv.arm(0);
            return reply_err(net, conn, e);
        }
    };
    // Past this line the header is committed and the only failure is a close.
    let header = format!("DATA {size} {}\n", files::hex16(&digest));
    if send(net, conn, &header, SEND_MS).is_err() {
        reader.close();
        srv.arm(0);
        srv.log("RESIDENT: peer went away before the DATA header");
        return Step::Drop;
    }

    let mut bounce = match files::Bounce::new() {
        Some(b) => b,
        None => {
            reader.close();
            srv.arm(0);
            srv.log("RESIDENT: GET could not claim a bounce buffer — closing");
            return Step::Drop;
        }
    };
    let mut sent: u64 = 0;
    loop {
        job::arm_watchdog(files::FILES_WD_S);
        let n = match reader.next(bounce.buf()) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                reader.close();
                srv.arm(0);
                srv.log(&format!(
                    "RESIDENT: GET {name} failed mid-frame ({}) after {sent} bytes — closing",
                    e.slug()
                ));
                return Step::Drop;
            }
        };
        if net
            .tcp_send_slice(conn, &bounce.buf()[..n], XFER_SEND_MS, &mut wd_rearm())
            .is_err()
        {
            reader.close();
            srv.arm(0);
            srv.log(&format!(
                "RESIDENT: GET {name} send failed after {sent} bytes — closing"
            ));
            return Step::Drop;
        }
        sent += n as u64;
    }
    reader.close();
    let done = sent == size && send(net, conn, "END\n", SEND_MS).is_ok();
    srv.arm(0);
    if done {
        srv.log(&format!("RESIDENT: GET {name} {size} bytes sent"));
        Step::Continue
    } else {
        srv.log(&format!(
            "RESIDENT: GET {name} ended short at {sent} of {size} — closing"
        ));
        Step::Drop
    }
}

/// `PUT <NAME> <len> <64hex>` → `SEND`, then the payload, then
/// `OK <NAME> <len> <sha16>` | `ERR <e>`.
///
/// Declare-then-stream (§1.2): the name, the length and the free space are
/// checked **before** the client burns 1.8 GB on the wire, and the full 64-hex
/// digest is what lets an operator prove what landed.
///
/// The abort table (§1.4) in one place:
///
/// | condition | wire | connection | on-disk |
/// |---|---|---|---|
/// | header malformed | `ERR bad-len` / `bad-args` / `bad-name` before `SEND` | kept | nothing written |
/// | trailer is not `END\n`, or the readback digest differs | `ERR bad-frame` / `digest-mismatch` after the full `len` is drained | kept | stage deleted, target untouched |
/// | peer closes mid-payload | `ERR bad-frame` best effort | **closed** — resync is impossible once the byte count is unknown | stage deleted |
/// | no progress for `XFER_STALL_MS` | `ERR io` best effort | **closed** | stage deleted; the watchdog is disarmed and the box goes back to `accept` — **it does not reboot** |
/// | free space exhausts mid-stream | `ERR io` | closed | stage deleted |
fn do_put(
    net: &mut Net,
    conn: &TcpHandle,
    srv: &mut Srv<'_, '_>,
    rd: &mut Lines,
    arg: &str,
) -> Step {
    // ---- header ---------------------------------------------------------
    let mut it = arg.split_whitespace();
    let (Some(raw_name), Some(raw_len), Some(raw_hex), None) =
        (it.next(), it.next(), it.next(), it.next())
    else {
        return reply_err(net, conn, FileErr::BadArgs);
    };
    let name = match files::validate_name(raw_name) {
        Ok(n) => n,
        Err(e) => return reply_err(net, conn, e),
    };
    if files::is_protected(&name) {
        srv.log(&format!("RESIDENT: PUT {name} refused — protected"));
        return reply_err(net, conn, FileErr::Protected);
    }
    let Ok(len) = raw_len.parse::<u64>() else {
        return reply_err(net, conn, FileErr::BadLen);
    };
    if len > files::PUT_MAX_BYTES {
        return reply_err(net, conn, FileErr::BadLen);
    }
    if !files::is_hex64(raw_hex) {
        return reply_err(net, conn, FileErr::BadArgs);
    }
    let want_hex = raw_hex.to_ascii_lowercase();
    // Advisory (§8): some vendor FAT drivers report free space wrongly or not
    // at all, so `None` is permission to try. `no-space` is best effort; `io`
    // on a short write is the guarantee.
    if let Some(free) = files::free_space(srv.root)
        && free < len
    {
        srv.log(&format!(
            "RESIDENT: PUT {name} refused — {len} bytes into {free} free"
        ));
        return reply_err(net, conn, FileErr::NoSpace);
    }

    if send(net, conn, "SEND\n", SEND_MS).is_err() {
        return Step::Drop;
    }
    srv.log(&format!("RESIDENT: PUT {name} {len} bytes — staging"));

    // ---- stream into STAGE.PRT -------------------------------------------
    srv.arm(files::FILES_WD_S);
    let mut stage = match files::Stage::create(srv.root) {
        Ok(st) => st,
        Err(e) => {
            srv.arm(0);
            return reply_err(net, conn, e);
        }
    };
    let mut wire = aegis_core::witness::Sha256::new();
    let mut got: u64 = 0;
    let mut stalled = 0u64;
    while got < len {
        job::arm_watchdog(files::FILES_WD_S);
        let want = core::cmp::min(len - got, files::XFER_CHUNK as u64) as usize;
        match rd.take_bytes(net, conn, want, XFER_STALL_MS) {
            Ok(buf) => {
                stalled = 0;
                wire.update(&buf);
                if let Err(e) = stage.write(&buf) {
                    // §1.4: a short write is where a full volume actually
                    // shows up. Nothing partial is committed, and the stream
                    // is now un-resyncable because the rest of the payload is
                    // still coming.
                    stage.abandon(srv.root);
                    srv.arm(0);
                    srv.log(&format!(
                        "RESIDENT: PUT {name} write failed at {got} ({}) — closing",
                        e.slug()
                    ));
                    let _ = send(net, conn, "ERR io\n", SEND_MS);
                    return Step::Drop;
                }
                got += buf.len() as u64;
            }
            Err(NetError::Timeout) => {
                stalled = stalled.saturating_add(XFER_STALL_MS);
                if stalled >= XFER_STALL_MS {
                    stage.abandon(srv.root);
                    srv.arm(0);
                    srv.log(&format!(
                        "RESIDENT: PUT {name} made no progress for {XFER_STALL_MS} ms \
                         at {got} of {len} — closing"
                    ));
                    let _ = send(net, conn, "ERR io\n", SEND_MS);
                    return Step::Drop;
                }
            }
            Err(_) => {
                stage.abandon(srv.root);
                srv.arm(0);
                srv.log(&format!(
                    "RESIDENT: PUT {name} peer left at {got} of {len} — closing"
                ));
                let _ = send(net, conn, "ERR bad-frame\n", SEND_MS);
                return Step::Drop;
            }
        }
    }

    // ---- the trailer -----------------------------------------------------
    // The declared length is the frame; `END\n` is a resync marker (§1.1). A
    // wrong trailer means the peer's idea of the payload and ours differ, but
    // the byte count is still known, so the connection is **kept** and the
    // client is told which of the two failed.
    let trailer = match rd.take_exact(net, conn, 4, XFER_STALL_MS) {
        Ok(t) => t,
        Err(_) => {
            stage.abandon(srv.root);
            srv.arm(0);
            srv.log(&format!(
                "RESIDENT: PUT {name} trailer never arrived — closing"
            ));
            let _ = send(net, conn, "ERR bad-frame\n", SEND_MS);
            return Step::Drop;
        }
    };
    if trailer != b"END\n" {
        stage.abandon(srv.root);
        srv.arm(0);
        srv.log(&format!(
            "RESIDENT: PUT {name} trailer was not END — ERR bad-frame"
        ));
        return reply_err(net, conn, FileErr::BadFrame);
    }

    if let Err(e) = stage.finish() {
        let _ = files::delete(srv.root, files::STAGE_NAME);
        srv.arm(0);
        return reply_err(net, conn, e);
    }

    // ---- readback verify --------------------------------------------------
    // The digest that decides is the one read back **off the volume**, not the
    // one computed from the wire: the wire digest proves the transfer, and only
    // a readback proves the write. Both are logged when they disagree, because
    // "the network corrupted it" and "the medium corrupted it" want different
    // repairs.
    let wire_hex = files::hex64(&wire.finalize());
    let back = files::sha_named(srv.root, files::STAGE_NAME, &mut wd_rearm());
    let (back_size, back_digest) = match back {
        Ok(v) => v,
        Err(e) => {
            let _ = files::delete(srv.root, files::STAGE_NAME);
            srv.arm(0);
            return reply_err(net, conn, e);
        }
    };
    let back_hex = files::hex64(&back_digest);
    if back_size != len || !back_hex.eq_ignore_ascii_case(&want_hex) {
        let _ = files::delete(srv.root, files::STAGE_NAME);
        srv.arm(0);
        srv.log(&format!(
            "RESIDENT: PUT {name} digest mismatch — declared {want_hex}, wire {wire_hex}, \
             readback {back_hex} ({back_size} of {len} bytes)"
        ));
        return reply_err(net, conn, FileErr::DigestMismatch);
    }

    // ---- commit ------------------------------------------------------------
    let commit = match crate::reload::artifact_target(srv.root, &name) {
        // §8: the three artifacts commit by **pointer swap**, never by
        // delete-then-rename, because the file boot depends on must never
        // spend a moment not existing. The staged bytes take the inactive half
        // of the A/B pair and `CURRENT.TXT` is then rewritten to point at it.
        Some((key, dest)) => files::commit(srv.root, &dest)
            .and_then(|()| crate::reload::set_pointer(srv.root, key, &dest)),
        None => files::commit(srv.root, &name),
    };
    srv.arm(0);
    match commit {
        Ok(()) => {
            srv.log(&format!("RESIDENT: PUT {name} committed, sha {back_hex}"));
            reply(
                net,
                conn,
                &format!("OK {name} {len} {}\n", files::hex16(&back_digest)),
            )
        }
        Err(e) => {
            let _ = files::delete(srv.root, files::STAGE_NAME);
            srv.log(&format!(
                "RESIDENT: PUT {name} commit failed ({}) — stage removed",
                e.slug()
            ));
            reply_err(net, conn, e)
        }
    }
}

/// `RM <NAME>` → `OK <NAME>` | `ERR <e>`.
fn do_rm(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>, arg: &str) -> Step {
    let name = match files::validate_name(arg) {
        Ok(n) => n,
        Err(e) => return reply_err(net, conn, e),
    };
    if files::is_protected(&name) {
        return reply_err(net, conn, FileErr::Protected);
    }
    // An artifact name goes through `reload::remove_artifact`, which clears
    // the pointer *before* it deletes either half (§8). A plain
    // `delete(resolve(name))` here would leave `CURRENT.TXT` designating a
    // file that is gone, and `current_names`' fallback would then serve the
    // stale other half as if it were the artifact the operator just removed.
    let removed = match crate::reload::remove_artifact(srv.root, &name) {
        Some(r) => r,
        None => files::delete(srv.root, &name),
    };
    match removed {
        Ok(()) => {
            srv.log(&format!("RESIDENT: RM {name}"));
            reply(net, conn, &format!("OK {name}\n"))
        }
        Err(e) => reply_err(net, conn, e),
    }
}

/// `RELOAD` → `RELOADING` … `OK reload model=… embed=… vocab=…` | `ERR <e>`.
///
/// The one genuinely dangerous verb (design §4.1). It refuses while a
/// `STAGE.PRT` exists, checks every artifact against its slab **before**
/// touching anything, and only then drops the engine. A rebuild that fails
/// leaves the box holding bytes it cannot describe, and the answer to that is
/// a cold reset — never carrying on.
fn do_reload(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>) -> Step {
    if send(net, conn, "RELOADING\n", SEND_MS).is_err() {
        return Step::Drop;
    }
    srv.log("RESIDENT: RELOAD requested");
    srv.arm(files::FILES_WD_S);
    let out = srv.slot.reload(srv.root, &mut wd_rearm());
    srv.arm(0);
    match out {
        Ok(()) => {
            let d = srv.slot.digests();
            let line = format!(
                "OK reload model={} embed={} vocab={}\n",
                crate::reload::short(&d.model),
                crate::reload::short(&d.embed),
                crate::reload::short(&d.vocab)
            );
            let n = srv.slot.reloads();
            srv.log(&format!("RESIDENT: RELOAD ok, reloads={n}"));
            reply(net, conn, &line)
        }
        Err(e) => {
            srv.log(&format!("RESIDENT: RELOAD failed — {}", e.slug()));
            // `ERR reload-size` and `ERR busy-file` are refusals *before* the
            // point of no return, so the old engine is still live and the box
            // keeps serving. Anything that emptied the slot is terminal.
            if srv.slot.engine_mut().is_none() {
                let _ = send(net, conn, &format!("ERR {}\n", e.slug()), SEND_MS);
                return Step::Reset;
            }
            let slug = match e {
                ReloadErr::File(f) => f.slug(),
                other => other.slug(),
            };
            reply(net, conn, &format!("ERR {slug}\n"))
        }
    }
}

/// `HEALTH` → the one line a scheduler reads **before** dispatching work
/// (design §1.2): a `model=` mismatch, `parts=1` or `last=FAIL` are reasons not
/// to send anything.
///
/// Rule A: every field here is a count, a digest, a byte total or an
/// identifier. None of it is a rate, and `env=` travels with it so nothing
/// downstream can mistake a TCG box for an iron one.
fn do_health(net: &mut Net, conn: &TcpHandle, srv: &mut Srv<'_, '_>) -> Step {
    let up = srv.uptime_s(net);
    let served = srv.served;
    let last = srv.last.clone();
    let wd = if srv.wd_s == 0 {
        String::from("off")
    } else {
        format!("{}", srv.wd_s)
    };
    let heapfree = free_pool_bytes();
    let model = crate::reload::short(&srv.slot.digests().model).to_string();
    let reloads = srv.slot.reloads();
    let parts = u8::from(files::stage_present(srv.root));
    let env = crate::sysinfo::env().as_str();
    // §8's pointer fallback, made visible. `pointer` means `CURRENT.TXT`
    // names an artifact file that is not on the volume, so every read verb is
    // being answered from the other, older half of the A/B pair. The box is
    // still serving — that is why this is a field and not a refusal — but a
    // scheduler that sends work to it is getting the previous model's answers
    // under this boot's `model=`, and `parts=`/`last=` would not have said so.
    let degraded = if crate::reload::pointer_degraded(srv.root) {
        "pointer"
    } else {
        "none"
    };
    reply(
        net,
        conn,
        &format!(
            "HEALTH up={up} served={served} last={last} wd={wd} heapfree={heapfree} \
             model={model} reloads={reloads} parts={parts} degraded={degraded} env={env}\n"
        ),
    )
}

/// Bytes of `CONVENTIONAL` memory the firmware still has, for `HEALTH
/// heapfree=`.
///
/// A fact about the box, not a measurement (Rule A): design §4.2 sizes the
/// listener buffers at 640 KiB of pool while a client is connected and says
/// the cost is visible here, so it has to be visible here.
fn free_pool_bytes() -> u64 {
    use uefi::mem::memory_map::MemoryMap;
    let Ok(map) = uefi::boot::memory_map(uefi::boot::MemoryType::LOADER_DATA) else {
        return 0;
    };
    map.entries()
        .filter(|d| d.ty == uefi::boot::MemoryType::CONVENTIONAL)
        .map(|d| d.page_count * 4096)
        .sum()
}

/// `RUNID <id>` → `NEW` | `REPLAY` | `ERR bad-runid`.
///
/// The at-most-once primitive, and the most important addition for a fleet
/// (design §1.2). A controller whose TCP died after `RUNNING` but before
/// `RESULT` retries blind without double-spending a twenty-minute shard.
///
/// After `REPLAY` the client **still sends its `JOB … END`**: the server drains
/// and discards it and then answers the cached record (§1.4). One code path
/// reads `JOB` in every case, and the stream is never left half-framed.
fn do_runid(
    net: &mut Net,
    conn: &TcpHandle,
    srv: &mut Srv<'_, '_>,
    sess: &mut Session,
    arg: &str,
) -> Step {
    if !job::valid_runid(arg) {
        return reply(net, conn, "ERR bad-runid\n");
    }
    let known = srv.ring.iter().any(|(k, _)| k == arg);
    sess.runid = Some((arg.to_string(), known));
    if known {
        srv.log(&format!("RESIDENT: RUNID {arg} already served — REPLAY"));
        reply(net, conn, "REPLAY\n")
    } else {
        srv.log(&format!("RESIDENT: RUNID {arg} — NEW"));
        reply(net, conn, "NEW\n")
    }
}

/// The watchdog re-arm hook handed to every streaming call in `files`/`reload`
/// (design §8).
///
/// A closure rather than a method because the callee already holds the `&mut
/// Directory` these calls need; a hook that borrowed the server too would not
/// compile, and threading the window through as a number would put the rule in
/// two places.
fn wd_rearm() -> impl FnMut() {
    || {
        job::arm_watchdog(files::FILES_WD_S);
    }
}

/// The `AEFINITY-OS 0.1 READY env=… cpu=…` line of spec §4.
///
/// `env` and `cpu` come straight out of CPUID via `sysinfo`, the same two
/// facts `RESULT.TXT` carries, so a client knows before it sends a job
/// whether anything that box produces may be quoted as a measurement — under
/// QEMU this reads `env=vm` and the answer is no (Rule A).
fn banner() -> String {
    let mut brand = [0u8; 48];
    // Design §1: the banner keeps its version and gains one token, so every
    // v0.1 client assertion still passes. `caps=` goes **after** `cpu=`, which
    // means a client that read `cpu=` to end-of-line now reads the caps token
    // too — v0.1's own gate matches on ` env=vm ` and `cpu=`, both unaffected,
    // and no v0.1 client in this program parses the brand that way.
    format!(
        "AEFINITY-OS {PROTO_VERSION} READY env={} cpu={} caps={CAPS}\n",
        crate::sysinfo::env().as_str(),
        crate::sysinfo::cpu_brand(&mut brand)
    )
}

/// Whether [`do_job`] left the connection usable.
enum JobOutcome {
    /// The job ran and the client got its answer; keep serving this client.
    Served,
    /// The connection is finished with — over a cap, or the peer left.
    Dropped,
}

/// `JOB\n<body>\nEND\n` (spec §4).
#[allow(clippy::too_many_arguments)]
fn do_job(
    net: &mut Net,
    conn: &TcpHandle,
    spare: &mut Option<TcpHandle>,
    port: u16,
    srv: &mut Srv<'_, '_>,
    sess: &mut Session,
    rd: &mut Lines,
) -> JobOutcome {
    let cfg_listen = srv.cfg.listen;
    let cfg_net = srv.cfg.net.clone();
    // ---- collect the body -------------------------------------------------
    let mut body = String::new();
    loop {
        let line = match rd.read_line(net, conn, spare, port, srv.root) {
            Ok(l) => l,
            Err(LineErr::Closed) => {
                srv.log("RESIDENT: peer closed inside a JOB body");
                return JobOutcome::Dropped;
            }
            Err(LineErr::Idle) => {
                srv.log("RESIDENT: peer went silent inside a JOB body");
                return JobOutcome::Dropped;
            }
            Err(LineErr::TooLong) => {
                srv.log("RESIDENT: JOB body line over the cap");
                let _ = send(net, conn, "ERR too-large\n", SEND_MS);
                return JobOutcome::Dropped;
            }
        };
        let trimmed = line.trim();
        let mut key = [0u8; 8];
        if upper(trimmed, &mut key) == "END" {
            break;
        }
        if let Some(why) = over_cap(&body, &line) {
            srv.log(&format!("RESIDENT: JOB body rejected — {why}"));
            let _ = send(net, conn, "ERR too-large\n", SEND_MS);
            return JobOutcome::Dropped;
        }
        body.push_str(&line);
        body.push('\n');
    }

    // ---- the RUNID ring, before anything is run (design §1.4) -------------
    // The body above was read whichever way this goes: one code path reads
    // `JOB` in every case, so the stream is never left half-framed. On a
    // replay the body is simply discarded and the cached record goes back.
    if let Some((id, true)) = sess.runid.clone() {
        sess.runid = None;
        // `RUNNING` goes out here too, so the `JOB` exchange has the same shape
        // whichever way it went: a controller retrying blind after a dead TCP
        // is exactly the client that must not need two code paths, and it was
        // already told `REPLAY` by the verb above. The record itself carries
        // `replay=true`, which is where the honesty about not re-running lives
        // — `RUNNING` here means "the JOB exchange has begun", which is true.
        let _ = send(net, conn, "RUNNING\n", SEND_MS);
        return match srv.replay_of(&id) {
            Some(cached) => {
                srv.log(&format!(
                    "RESIDENT: RUNID {id} replayed — job body drained, not run"
                ));
                let mut msg = String::with_capacity(cached.len() + 16);
                msg.push_str("RESULT\n");
                msg.push_str(&cached);
                msg.push_str("END\n");
                if send(net, conn, &msg, SEND_RESULT_MS).is_err() {
                    JobOutcome::Dropped
                } else {
                    JobOutcome::Served
                }
            }
            None => {
                // The id was in the ring when `RUNID` answered and is not now.
                // Only an eviction between the two can do that, and the honest
                // answer is that this box cannot serve the replay it promised.
                srv.log(&format!("RESIDENT: RUNID {id} evicted before its replay"));
                let _ = send(net, conn, "ERR not-found\n", SEND_MS);
                JobOutcome::Served
            }
        };
    }

    // ---- parse, with the §4 exclusions applied ----------------------------
    let mut sub = job::parse(&body);
    // Spec §4: "A job body's MODE/LISTEN/NET lines are ignored in resident
    // mode." They are dropped here rather than at the parser so a stray
    // `MODE resident` in a body cannot recurse, and so the box keeps the
    // address and port the stick's own JOB.TXT gave it.
    sub.mode = Mode::Oneshot;
    sub.listen = cfg_listen;
    sub.net = cfg_net;
    // Integration note (v0.1): phase 1b's `REPORT <url>` is performed by the
    // ONESHOT dispatch path only. A resident job's result goes back down the
    // socket the client is already holding (spec §4), and the POST client
    // would need the NIC this listener has opened exclusively. Rather than
    // accept the directive and quietly not do it, it is dropped here and
    // named in BOOTLOG.TXT, the same way §4 handles MODE/LISTEN/NET.
    if let Some(url) = sub.report.take() {
        srv.log(&format!(
            "RESIDENT: JOB ignored REPORT {url:?} — oneshot-only in v0.1"
        ));
    }
    srv.log(&format!(
        "RESIDENT: JOB parsed — {} directives, {} steps, budget={}s",
        sub.directives,
        sub.steps.len(),
        sub.budget_s
    ));
    let unknown = sub.unknown.clone();
    for k in &unknown {
        srv.log(&format!("RESIDENT: JOB ignored directive {k:?}"));
    }

    if send(net, conn, "RUNNING\n", SEND_MS).is_err() {
        // The peer is gone, but the job was asked for and the box is not going
        // to pretend it was not: spec's resident contract is that a job in
        // flight finishes and leaves a record. Fall through and run it.
        srv.log("RESIDENT: peer gone before RUNNING — running the job anyway");
    }

    // ---- run it -----------------------------------------------------------
    let wd = sub.budget_s.saturating_add(job::WATCHDOG_MARGIN_S);
    srv.arm(wd);
    srv.log(&format!("RESIDENT: watchdog armed at {wd}s for JOB"));

    // NOTE: a body carrying `NETCHECK` will ask `run_job` to bring up a second
    // `Net`, and the SNP handle this server is holding is opened exclusively —
    // so that directive fails with `no nic` and says so in the record. It is
    // reported, not silently wrong. A resident box's network check is the
    // connection the client is already talking on.
    // `run_job` hands back whatever NIC it brought up for the body's own
    // directives — phase 1b's REPORT reuses it in the oneshot path. Here the
    // listener already holds the SNP open exclusively, so per the NOTE above
    // this is `None`; dropping it is what releases the open in the case where
    // the firmware did hand out a second one.
    if srv.slot.engine_mut().is_none() {
        // See `job::dispatch`: the slot is only ever empty inside
        // `EngineSlot::reload`, which cold-resets rather than returning here.
        srv.arm(0);
        srv.log("RESIDENT: no engine in the slot — cannot run the JOB");
        let _ = send(net, conn, "ERR reload-engine\n", SEND_MS);
        return JobOutcome::Dropped;
    }
    // `srv.root` and `srv.slot` are two disjoint fields, but `run_job` wants
    // both at once and the borrow checker cannot see through a method. The
    // reborrow is spelled out rather than worked around with a clone, because
    // there is nothing here to clone. Phase 5: `run_job` takes the whole slot,
    // not just the engine — `VERIFY` and `EVAL` read the resident artifact
    // slices, which only the slot can hand out.
    let root: &mut Directory = srv.root;
    let slot: &mut EngineSlot<'_> = srv.slot;
    let (mut rec, sub_net) = job::run_job(&sub, root, slot);
    drop(sub_net);
    // The identity of the NIC this server is running over. `run_job` fills
    // these from the `Net` it brings up itself, and in resident mode it has
    // none — the address is ours, so it is ours to report.
    rec.mac = net.mac_string();
    rec.ip = net.ip_string();
    rec.net = net.how().to_string();
    // Design §1.2: a `RUNID` in force names the record, so the id a controller
    // dedupes on and the id in the record it stores are the same string.
    let run_id = sess.runid.take().map(|(id, _)| id);
    if let Some(id) = &run_id {
        rec.run_id = id.clone();
    }
    // Design §3's additions. `resident` is `Some` here by construction — this
    // is the resident path — so `uptime_s`/`served` are real, and `replay` is
    // `false` because this record was produced by actually running the job.
    let up = srv.uptime_s(net);
    let served = srv.served;
    rec.fleet = Some(job::fleet_info(
        srv.slot,
        srv.root,
        Some((up, served)),
        false,
        &sub,
    ));
    let out = rec.render();
    if let Some(id) = &run_id {
        srv.remember(id, &out);
    }
    srv.last = if rec.verdict == "OK" {
        String::from("OK")
    } else {
        rec.verdict.clone()
    };

    // ---- the record: to the volume and to the socket ----------------------
    // The watchdog stays armed across both of these, and is disarmed only
    // once the record is on the volume and the answer is on the wire. It used
    // to be disarmed here, before the write — which left the one FAT write
    // this whole mode exists to produce running with no backstop at all.
    srv.arm(RECORD_WATCHDOG_S);
    srv.log(&format!(
        "RESIDENT: watchdog re-armed at {RECORD_WATCHDOG_S}s for the record"
    ));

    if job::write_result_txt(srv.root, &out) {
        job::clear_wip(srv.root);
        srv.log(&format!(
            "RESIDENT: RESULT.TXT written, verdict={}",
            rec.verdict
        ));
    } else {
        srv.log("RESIDENT: RESULT.TXT could not be written");
    }

    let mut msg = String::with_capacity(out.len() + 16);
    msg.push_str("RESULT\n");
    msg.push_str(&out);
    // `ResultRecord::render` always ends in a newline, so `END` starts its own
    // line without a separator being guessed at here.
    msg.push_str("END\n");
    let sent = send(net, conn, &msg, SEND_RESULT_MS);

    // Everything that had to reach the volume has reached it, and the answer
    // is on the wire or gone. Both BOOTLOG lines are written while the
    // watchdog is still armed — every FAT write this function makes is
    // covered — and only then does the box go back to idle, where "no
    // watchdog" is the contract (spec §4): a resident worker waiting for its
    // next client must not reset itself for being idle.
    let outcome = if sent.is_err() {
        srv.log("RESIDENT: peer went away before the RESULT — record is on the volume");
        JobOutcome::Dropped
    } else {
        srv.log("RESIDENT: RESULT sent");
        JobOutcome::Served
    };
    srv.log("RESIDENT: JOB done, disarming watchdog");
    srv.arm(0);
    outcome
}

/// Why a job body may not take this line, or `None` if it may (spec §4).
fn over_cap(body: &str, line: &str) -> Option<String> {
    if body.len() + line.len() + 1 > BODY_MAX_BYTES {
        return Some(format!("body over {BODY_MAX_BYTES} bytes"));
    }
    // The cap is on the prompt, not on the line: `PROMPT ` is the directive.
    let trimmed = line.trim_start();
    let mut key = [0u8; 8];
    let (head, rest) = match trimmed.find(char::is_whitespace) {
        Some(i) => (&trimmed[..i], trimmed[i..].trim()),
        None => (trimmed, ""),
    };
    if upper(head, &mut key) == "PROMPT" && rest.len() > PROMPT_MAX_BYTES {
        return Some(format!(
            "PROMPT of {} bytes over {PROMPT_MAX_BYTES}",
            rest.len()
        ));
    }
    None
}

/// Answer a second concurrent peer with `BUSY\n` and close it (spec §4), then
/// put a fresh backlog socket in its place so a third peer gets the same
/// answer rather than silence.
fn busy_check(net: &mut Net, spare: &mut Option<TcpHandle>, port: u16, root: &mut Directory) {
    let accepted = match spare.as_ref() {
        Some(s) => net.tcp_accepted(s),
        None => return,
    };
    if !accepted {
        return;
    }
    let Some(busy) = spare.take() else {
        return;
    };
    crate::boot_log(root, "RESIDENT: second connection — BUSY");
    let _ = net.tcp_send_all(&busy, b"BUSY\n", SEND_MS);
    net.tcp_close(busy, CLOSE_MS);
    *spare = net.tcp_listen(port).ok();
    if spare.is_none() {
        crate::boot_log(root, "RESIDENT: backlog socket could not be reopened");
    }
}

// ---------------------------------------------------------------------------
// Line reading
// ---------------------------------------------------------------------------

/// Why a line did not arrive.
enum LineErr {
    /// The peer closed, or the connection was reset.
    Closed,
    /// Nothing at all for [`IDLE_DROP_MS`].
    Idle,
    /// A single line longer than [`LINE_MAX_BYTES`].
    TooLong,
}

/// A line reader over one connection.
///
/// [`Net::tcp_recv_until`] stops *at* a delimiter but returns everything it
/// had drained from the socket, which for a client that writes a whole job
/// body in one call is many lines. Keeping the remainder here is what makes
/// the protocol line-oriented rather than write-oriented: a client may send
/// `JOB\n…\nEND\n` as one write or as twenty and the server reads the same
/// lines either way.
struct Lines {
    pending: Vec<u8>,
    /// Milliseconds of consecutive silence, for [`IDLE_DROP_MS`].
    idle_ms: u64,
}

impl Lines {
    fn new() -> Lines {
        Lines {
            pending: Vec::new(),
            idle_ms: 0,
        }
    }

    /// Take up to `max` raw bytes of a declared payload (design §1.1).
    ///
    /// This is the other half of what makes the protocol line-oriented **and**
    /// binary: `read_line` leaves whatever it over-read in `pending`, and so
    /// does this, so a client is free to write `PUT …\n<payload>END\n` as one
    /// segment or as a thousand and the server reads the same bytes either
    /// way. Without it a payload that arrived in the same segment as its
    /// header would be lost and the stream would desynchronise on the first
    /// batched write — which is what every efficient client does.
    ///
    /// Never more than one [`files::XFER_CHUNK`] leaves the socket per call,
    /// so a 2 GiB `PUT` is bounded memory: one chunk in flight, hashed and
    /// written, then dropped.
    ///
    /// `Err(NetError::Timeout)` means *no progress*, which is what the
    /// caller's stall bound is measured in.
    fn take_bytes(
        &mut self,
        net: &mut Net,
        conn: &TcpHandle,
        max: usize,
        timeout_ms: u64,
    ) -> Result<Vec<u8>, NetError> {
        if self.pending.is_empty() {
            let want = core::cmp::min(max, files::XFER_CHUNK);
            let (got, res) = net.tcp_recv_exact(conn, want, timeout_ms);
            let empty = got.is_empty();
            self.pending.extend_from_slice(&got);
            if empty {
                res?;
                return Err(NetError::Timeout);
            }
        }
        let n = core::cmp::min(max, self.pending.len());
        self.idle_ms = 0;
        Ok(self.pending.drain(..n).collect())
    }

    /// Take exactly `n` bytes, looping until they are all here.
    ///
    /// Only ever called for the 4-byte `END\n` trailer, which is small enough
    /// that "loop until it is all here or the peer stalls" is the whole of the
    /// error handling. `Err` is the caller's cue to abandon the stage and
    /// close: a trailer that never arrives means the byte count is no longer
    /// known, and §1.4 says resync is impossible from there.
    fn take_exact(
        &mut self,
        net: &mut Net,
        conn: &TcpHandle,
        n: usize,
        timeout_ms: u64,
    ) -> Result<Vec<u8>, NetError> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let part = self.take_bytes(net, conn, n - out.len(), timeout_ms)?;
            out.extend_from_slice(&part);
        }
        Ok(out)
    }

    /// Next line, without its terminator. `\r\n` and `\n` are both accepted —
    /// spec §2 says a job body may use either, and the command lines above it
    /// come from the same clients.
    fn read_line(
        &mut self,
        net: &mut Net,
        conn: &TcpHandle,
        spare: &mut Option<TcpHandle>,
        port: u16,
        root: &mut Directory,
    ) -> Result<String, LineErr> {
        loop {
            if let Some(i) = self.pending.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.pending.drain(..=i).collect();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.idle_ms = 0;
                // Lossy on purpose: a client that sends a stray non-UTF-8 byte
                // gets `ERR unknown` for that line, not a dead box. Firmware
                // data may never reach a `from_utf8().unwrap()`.
                return Ok(String::from_utf8_lossy(&line).into_owned());
            }
            if self.pending.len() > LINE_MAX_BYTES {
                return Err(LineErr::TooLong);
            }

            // Answer a second peer here rather than between commands (spec
            // §4). Checking only where a command is dispatched would make the
            // `BUSY` answer depend on the *first* client saying something: a
            // client that connects and waits would see silence, not a
            // refusal, for as long as the first one stayed quiet. This is the
            // wait, so this is where anyone else knocking gets an answer.
            //
            // It still cannot run *during* a job — `run_job` does not poll the
            // stack — so a peer that arrives mid-job is told BUSY when the job
            // ends. That is a delay, not a hang: its handshake completes, its
            // answer is late.
            busy_check(net, spare, port, root);

            let (got, res) = net.tcp_recv_until(conn, Until::Delim(b'\n'), READ_SLICE_MS);
            let empty = got.is_empty();
            self.pending.extend_from_slice(&got);
            match res {
                // A delimiter arrived; the loop above will cut the line.
                Ok(()) => self.idle_ms = 0,
                Err(NetError::Timeout) => {
                    if empty {
                        self.idle_ms = self.idle_ms.saturating_add(READ_SLICE_MS);
                        if self.idle_ms >= IDLE_DROP_MS {
                            return Err(LineErr::Idle);
                        }
                    } else {
                        // Partial line: the client is mid-write, not silent.
                        self.idle_ms = 0;
                    }
                }
                // The stack refused to hold any more for one read and the
                // delimiter still had not arrived, so this is one line longer
                // than the stack itself will buffer — over the cap by
                // definition, and the peer is owed `ERR too-large` (spec §4)
                // rather than a silent drop. Distinguishing it from a real
                // close is the whole reason `TooMuchData` exists: the two are
                // numerically identical here (`RECV_MAX_BYTES` ==
                // `LINE_MAX_BYTES`) and a shared error variant made an
                // over-long line indistinguishable from a peer that vanished.
                Err(NetError::TooMuchData) => return Err(LineErr::TooLong),
                // Closed or reset. Whatever is in `pending` has no terminator,
                // so there is no line to hand back and the connection is over.
                Err(_) => return Err(LineErr::Closed),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn send(net: &mut Net, conn: &TcpHandle, msg: &str, timeout_ms: u64) -> Result<(), NetError> {
    net.tcp_send_all(conn, msg.as_bytes(), timeout_ms)
}

fn stall(ms: u64) {
    uefi::boot::stall(core::time::Duration::from_millis(ms));
}

/// ASCII-uppercase `s` into `buf`. A token that does not fit, or is not ASCII,
/// comes back unchanged and therefore matches no command — which is the
/// `ERR unknown` arm, exactly where it belongs.
fn upper<'b>(s: &'b str, buf: &'b mut [u8; 8]) -> &'b str {
    if s.len() > buf.len() || !s.is_ascii() {
        return s;
    }
    let n = s.len();
    buf[..n].copy_from_slice(s.as_bytes());
    buf[..n].make_ascii_uppercase();
    core::str::from_utf8(&buf[..n]).unwrap_or(s)
}

/// First `max` bytes of `s`, on a character boundary — for a BOOTLOG line that
/// must not carry a client's whole payload.
fn clip(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn say(msg: &str) {
    crate::console::with_console(|st| {
        let _ = st.write_str(msg);
        core::fmt::Result::Ok(())
    });
}

/// Shut the NIC down, then park (spec §4: "park with hlt/pause loop").
fn park(net: Net, root: &mut Directory) -> ! {
    // Dropping the `Net` shuts the NIC down and releases the exclusive SNP
    // open, so a box that halted is not still holding a protocol nobody can
    // take back.
    drop(net);
    park_no_net(root)
}

/// The park loop itself.
fn park_no_net(root: &mut Directory) -> ! {
    // Idle for ever is only safe with the watchdog off (spec §4).
    job::arm_watchdog(0);
    crate::boot_log(root, "RESIDENT: parked");
    say("\r\n[AEFINITY OS] RESIDENT: halted — parked.\r\n");
    loop {
        // SAFETY: `hlt` is a ring-0 instruction and this unikernel runs in
        // ring 0 under boot services (it never calls ExitBootServices). It
        // reads no memory and touches no stack, which is what the options
        // assert.
        //
        // What it does here, precisely: `hlt` halts the core until the next
        // interrupt, NMI or SMI. Under boot services the firmware's own timer
        // interrupt is normally unmasked, so the core wakes every tick and the
        // loop parks it again — and if a firmware has masked interrupts, `hlt`
        // halts until an NMI or SMI instead. It does *not* "return at once",
        // as the comment here used to claim.
        //
        // Either behaviour is correct for this loop, because parking for ever
        // is the whole intent: `HALT` (spec §4) is defined as a box that stops
        // until someone power-cycles or resets it, the NIC is already down and
        // the watchdog is already off. The `pause` below is for the waking
        // case, where it keeps the spin cheap; on a masked-interrupt firmware
        // it is simply never reached.
        unsafe {
            core::arch::asm!("hlt", options(nomem, nostack, preserves_flags));
        }
        core::arch::x86_64::_mm_pause();
    }
}
