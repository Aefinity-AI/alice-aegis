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

use aegis_core::inference::TernaryInferenceEngine;
use uefi::proto::media::file::Directory;

use crate::job::{self, Job, Mode};
use crate::net::{Net, NetError, TcpHandle, Until};

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
// Entry point (spec §5)
// ---------------------------------------------------------------------------

/// Serve jobs over TCP until a client says `REBOOT` or `HALT`. Never returns.
///
/// Takes the [`Net`] by value: `HALT` has to shut the NIC down and release the
/// exclusive SNP open before parking, and `REBOOT` has to do the same before
/// `ResetSystem`. Both are `Net`'s `Drop`, which needs ownership.
pub fn run(
    mut net: Net,
    root: &mut Directory,
    engine: &mut TernaryInferenceEngine,
    cfg: &Job,
) -> ! {
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

    let mut served: u64 = 0;
    loop {
        // ---- accept ------------------------------------------------------
        while !net.tcp_accepted(&listener) {
            stall(ACCEPT_SLEEP_MS);
        }
        served += 1;
        crate::boot_log(
            root,
            &format!("RESIDENT: connection {served} accepted on port {port}"),
        );

        // One backlog socket, listening on the same port for the whole time
        // the connection above is being served. It is what makes the `BUSY`
        // answer of spec §4 possible — without it a second peer's SYN goes
        // unanswered and the client sees a hang rather than a refusal — and
        // when this connection ends it is already listening, so it becomes
        // the next listener with no gap in which the box is undialable.
        let mut spare = net.tcp_listen(port).ok();
        if spare.is_none() {
            crate::boot_log(
                root,
                "RESIDENT: no backlog socket — a second peer will not get BUSY",
            );
        }

        let outcome = serve(
            &mut net, &listener, &mut spare, port, root, engine, cfg, served,
        );

        net.tcp_close(listener, CLOSE_MS);
        crate::boot_log(root, &format!("RESIDENT: connection {served} closed"));

        match outcome {
            Outcome::Next => {
                listener = match spare.take() {
                    Some(s) => s,
                    None => match net.tcp_listen(port) {
                        Ok(h) => h,
                        Err(e) => {
                            crate::boot_log(
                                root,
                                &format!(
                                    "RESIDENT: cannot re-listen on {port}: {} — parking",
                                    e.as_str()
                                ),
                            );
                            park(net, root);
                        }
                    },
                };
            }
            Outcome::Reboot => {
                if let Some(s) = spare.take() {
                    net.tcp_close(s, CLOSE_MS);
                }
                crate::boot_log(root, "RESIDENT: REBOOT — resetting");
                // Shut the NIC down and release the SNP open before handing
                // the machine to the firmware.
                drop(net);
                job::after(job::After::Reset, root);
                // `After::Reset` does not return. If a firmware ignored
                // ResetSystem, parking is the honest ending — the client has
                // already been told BYE and this box must not carry on
                // serving as if nothing was asked of it.
                crate::boot_log(root, "RESIDENT: firmware did not reset — parking");
                park_no_net(root);
            }
            Outcome::Halt => {
                if let Some(s) = spare.take() {
                    net.tcp_close(s, CLOSE_MS);
                }
                crate::boot_log(root, "RESIDENT: HALT — listener closed, parking");
                park(net, root);
            }
        }
    }
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
#[allow(clippy::too_many_arguments)]
fn serve(
    net: &mut Net,
    conn: &TcpHandle,
    spare: &mut Option<TcpHandle>,
    port: u16,
    root: &mut Directory,
    engine: &mut TernaryInferenceEngine,
    cfg: &Job,
    served: u64,
) -> Outcome {
    if send(net, conn, &banner(), SEND_MS).is_err() {
        crate::boot_log(root, "RESIDENT: peer went away before the READY banner");
        return Outcome::Next;
    }
    crate::boot_log(
        root,
        &format!("RESIDENT: READY sent to connection {served}"),
    );

    let mut rd = Lines::new();
    loop {
        let line = match rd.read_line(net, conn, spare, port, root) {
            Ok(l) => l,
            Err(LineErr::Closed) => {
                crate::boot_log(root, "RESIDENT: peer closed the connection");
                return Outcome::Next;
            }
            Err(LineErr::Idle) => {
                crate::boot_log(
                    root,
                    &format!("RESIDENT: peer silent for {IDLE_DROP_MS} ms — dropping"),
                );
                return Outcome::Next;
            }
            Err(LineErr::TooLong) => {
                crate::boot_log(root, "RESIDENT: command line over the cap — ERR too-large");
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

        let mut key = [0u8; 8];
        match upper(line.trim(), &mut key) {
            "PING" => {
                if send(net, conn, "PONG\n", SEND_MS).is_err() {
                    crate::boot_log(root, "RESIDENT: PONG could not be sent");
                    return Outcome::Next;
                }
            }
            "JOB" => match do_job(net, conn, spare, port, root, engine, cfg, &mut rd) {
                JobOutcome::Served => {}
                JobOutcome::Dropped => return Outcome::Next,
            },
            "REBOOT" => {
                crate::boot_log(root, "RESIDENT: REBOOT requested");
                let _ = send(net, conn, "BYE\n", SEND_MS);
                return Outcome::Reboot;
            }
            "HALT" => {
                crate::boot_log(root, "RESIDENT: HALT requested");
                let _ = send(net, conn, "BYE\n", SEND_MS);
                return Outcome::Halt;
            }
            _ => {
                crate::boot_log(
                    root,
                    &format!("RESIDENT: unknown command {:?}", clip(&line, 64)),
                );
                if send(net, conn, "ERR unknown\n", SEND_MS).is_err() {
                    return Outcome::Next;
                }
            }
        }
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
    format!(
        "AEFINITY-OS {PROTO_VERSION} READY env={} cpu={}\n",
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
    root: &mut Directory,
    engine: &mut TernaryInferenceEngine,
    cfg: &Job,
    rd: &mut Lines,
) -> JobOutcome {
    // ---- collect the body -------------------------------------------------
    let mut body = String::new();
    loop {
        let line = match rd.read_line(net, conn, spare, port, root) {
            Ok(l) => l,
            Err(LineErr::Closed) => {
                crate::boot_log(root, "RESIDENT: peer closed inside a JOB body");
                return JobOutcome::Dropped;
            }
            Err(LineErr::Idle) => {
                crate::boot_log(root, "RESIDENT: peer went silent inside a JOB body");
                return JobOutcome::Dropped;
            }
            Err(LineErr::TooLong) => {
                crate::boot_log(root, "RESIDENT: JOB body line over the cap");
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
            crate::boot_log(root, &format!("RESIDENT: JOB body rejected — {why}"));
            let _ = send(net, conn, "ERR too-large\n", SEND_MS);
            return JobOutcome::Dropped;
        }
        body.push_str(&line);
        body.push('\n');
    }

    // ---- parse, with the §4 exclusions applied ----------------------------
    let mut sub = job::parse(&body);
    // Spec §4: "A job body's MODE/LISTEN/NET lines are ignored in resident
    // mode." They are dropped here rather than at the parser so a stray
    // `MODE resident` in a body cannot recurse, and so the box keeps the
    // address and port the stick's own JOB.TXT gave it.
    sub.mode = Mode::Oneshot;
    sub.listen = cfg.listen;
    sub.net = cfg.net.clone();
    // Integration note (v0.1): phase 1b's `REPORT <url>` is performed by the
    // ONESHOT dispatch path only. A resident job's result goes back down the
    // socket the client is already holding (spec §4), and the POST client
    // would need the NIC this listener has opened exclusively. Rather than
    // accept the directive and quietly not do it, it is dropped here and
    // named in BOOTLOG.TXT, the same way §4 handles MODE/LISTEN/NET.
    if let Some(url) = sub.report.take() {
        crate::boot_log(
            root,
            &format!("RESIDENT: JOB ignored REPORT {url:?} — oneshot-only in v0.1"),
        );
    }
    crate::boot_log(
        root,
        &format!(
            "RESIDENT: JOB parsed — {} directives, {} steps, budget={}s",
            sub.directives,
            sub.steps.len(),
            sub.budget_s
        ),
    );
    for k in &sub.unknown {
        crate::boot_log(root, &format!("RESIDENT: JOB ignored directive {k:?}"));
    }

    if send(net, conn, "RUNNING\n", SEND_MS).is_err() {
        // The peer is gone, but the job was asked for and the box is not going
        // to pretend it was not: spec's resident contract is that a job in
        // flight finishes and leaves a record. Fall through and run it.
        crate::boot_log(
            root,
            "RESIDENT: peer gone before RUNNING — running the job anyway",
        );
    }

    // ---- run it -----------------------------------------------------------
    let wd = sub.budget_s.saturating_add(job::WATCHDOG_MARGIN_S);
    if job::arm_watchdog(wd) {
        crate::boot_log(root, &format!("RESIDENT: watchdog armed at {wd}s for JOB"));
    } else {
        crate::boot_log(root, "RESIDENT: firmware refused the watchdog for JOB");
    }

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
    let (mut rec, sub_net) = job::run_job(&sub, root, engine);
    drop(sub_net);
    // The identity of the NIC this server is running over. `run_job` fills
    // these from the `Net` it brings up itself, and in resident mode it has
    // none — the address is ours, so it is ours to report.
    rec.mac = net.mac_string();
    rec.ip = net.ip_string();
    rec.net = net.how().to_string();
    let out = rec.render();

    // ---- the record: to the volume and to the socket ----------------------
    // The watchdog stays armed across both of these, and is disarmed only
    // once the record is on the volume and the answer is on the wire. It used
    // to be disarmed here, before the write — which left the one FAT write
    // this whole mode exists to produce running with no backstop at all.
    if job::arm_watchdog(RECORD_WATCHDOG_S) {
        crate::boot_log(
            root,
            &format!("RESIDENT: watchdog re-armed at {RECORD_WATCHDOG_S}s for the record"),
        );
    } else {
        crate::boot_log(
            root,
            "RESIDENT: firmware refused the watchdog for the record — writing unguarded",
        );
    }

    if job::write_result_txt(root, &out) {
        job::clear_wip(root);
        crate::boot_log(
            root,
            &format!("RESIDENT: RESULT.TXT written, verdict={}", rec.verdict),
        );
    } else {
        crate::boot_log(root, "RESIDENT: RESULT.TXT could not be written");
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
        crate::boot_log(
            root,
            "RESIDENT: peer went away before the RESULT — record is on the volume",
        );
        JobOutcome::Dropped
    } else {
        crate::boot_log(root, "RESIDENT: RESULT sent");
        JobOutcome::Served
    };
    crate::boot_log(root, "RESIDENT: JOB done, disarming watchdog");
    job::arm_watchdog(0);
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
