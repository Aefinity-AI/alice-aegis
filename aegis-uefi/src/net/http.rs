//! AEFINITY OS phase 1b — a minimal HTTP/1.1 POST client over [`crate::net::Net`].
//!
//! Spec: `program/AEFINITY_OS.md` §1 (phase 1b), §5 (this file).
//!
//! `job.rs` uses exactly one function here, [`post`], to send `RESULT.TXT`'s
//! bytes to the `REPORT <url>` a `JOB.TXT` names. v0.1 is deliberately small:
//! no TLS, no redirects, no chunked transfer-encoding, no DNS (the host in a
//! `REPORT` url must be a dotted IPv4 address — spec §5). None of that is a
//! gap this file works around; it is the v0.1 scope line.
//!
//! Rule A note: nothing here measures anything. A status code is a fact about
//! the collector's reply, not a performance figure.

use alloc::format;
use alloc::string::{String, ToString};

use crate::net::{Net, NetError, Until, parse_ipv4};

/// Why [`post`] did not return a status code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HttpErr {
    /// The url did not parse: no `http://` scheme, no host, or a bad port.
    BadUrl,
    /// The host is not a dotted IPv4 address. v0.1 has no resolver (spec §5),
    /// so a hostname is a configuration error, not a network failure.
    NoDns,
    /// The connect, the write, or the read failed at the network layer.
    Net(NetError),
    /// The peer replied, but the bytes read back do not start with a
    /// parseable `HTTP/<version> <code> ...` status line.
    BadResponse,
}

impl HttpErr {
    /// The short form that goes into `RESULT.TXT` / `BOOTLOG.TXT` — spec §5
    /// (`job.rs`) writes exactly `REPORT: fail {reason}` from this, and
    /// `NoDns` is what the spec names verbatim: "REPORT: fail no-dns".
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            HttpErr::BadUrl => "bad-url",
            HttpErr::NoDns => "no-dns",
            HttpErr::Net(e) => e.as_str(),
            HttpErr::BadResponse => "bad-response",
        }
    }
}

/// A parsed `http://host[:port]/path` (spec §5).
struct Url {
    host: core::net::Ipv4Addr,
    port: u16,
    /// Always starts with `/` — `http://host` alone means `/`.
    path: String,
}

/// Parse `"http://host[:port]/path"`. `host` must be a dotted IPv4 address;
/// anything else (a hostname, an IPv6 literal) is [`HttpErr::NoDns`] — v0.1
/// carries no resolver, so that is a configuration fact, not a transient
/// network failure.
fn parse_url(url: &str) -> Result<Url, HttpErr> {
    let rest = url.strip_prefix("http://").ok_or(HttpErr::BadUrl)?;
    if rest.is_empty() {
        return Err(HttpErr::BadUrl);
    }
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if hostport.is_empty() {
        return Err(HttpErr::BadUrl);
    }
    let (host_str, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => {
            let port: u16 = p.parse().map_err(|_| HttpErr::BadUrl)?;
            (h, port)
        }
        None => (hostport, 80u16),
    };
    if port == 0 {
        return Err(HttpErr::BadUrl);
    }
    // A dotted-quad host is the only form v0.1 speaks; anything else is a
    // hostname this stack has no resolver for (spec §5: "host must be a
    // dotted IPv4 for v0.1; a hostname -> Err logged as REPORT: fail no-dns").
    let host = parse_ipv4(host_str).ok_or(HttpErr::NoDns)?;
    Ok(Url {
        host,
        port,
        path: path.to_string(),
    })
}

/// The `Host:` header value: bare address for the default port, `addr:port`
/// otherwise — the usual HTTP/1.1 convention, and harmless either way since
/// the collector this talks to does not route on it.
fn host_header(u: &Url) -> String {
    if u.port == 80 {
        format!("{}", u.host)
    } else {
        format!("{}:{}", u.host, u.port)
    }
}

/// The first line of `resp`, parsed as an HTTP status line
/// (`HTTP/<version> <code> <reason>`). Only the code is kept — spec §5 says
/// `post` returns the status code, nothing else about the response.
fn parse_status(resp: &[u8]) -> Option<u16> {
    let text = core::str::from_utf8(resp).ok()?;
    let line = text.lines().next()?;
    let mut parts = line.split_whitespace();
    let version = parts.next()?;
    if !version.starts_with("HTTP/") {
        return None;
    }
    parts.next()?.parse::<u16>().ok()
}

/// `POST url` with `body`, HTTP/1.1, `Connection: close` (spec §5).
///
/// `token`, when present, is sent as `X-Aefinity-Token` — the shared secret a
/// collector may require before it will accept a write (spec §7 host tooling;
/// `JOB.TXT`'s `TOKEN` line).
///
/// `timeout_ms` bounds **the whole exchange**: connect, write, read and the
/// close together, on one deadline taken before the first of them, with each
/// phase given only what is left of it.
///
/// That wording used to be a claim rather than a fact. Each phase was handed
/// the same `timeout_ms` afresh, so a collector that stalled every phase cost
/// `4 x timeout_ms` — 120 s against the 60 s of watchdog margin `job.rs`
/// carves this out of, which turns a finished job into a firmware cold reset.
/// One deadline is the fix; the arithmetic is in `job::REPORT_TIMEOUT_MS`.
///
/// A phase reached with the deadline already past still runs, with a
/// zero-length budget: `Net`'s waits poll once and return `Timeout`, so the
/// exchange unwinds and closes instead of hanging. On success, the peer's
/// whole reply (headers and body, up to `net::Net`'s own receive cap) has
/// already been drained: `Connection: close` means the socket is not reused,
/// so there is nothing left for a caller to read afterwards.
pub fn post(
    net: &mut Net,
    url: &str,
    body: &[u8],
    token: Option<&str>,
    timeout_ms: u64,
) -> Result<u16, HttpErr> {
    let u = parse_url(url)?;

    // One deadline for everything below. `now_ms` is monotone (it clamps), so
    // `remaining` only ever shrinks and can never hand a phase more than the
    // caller allowed.
    let deadline = net.now_ms().saturating_add(timeout_ms as i64);
    let remaining = |net: &Net| -> u64 { deadline.saturating_sub(net.now_ms()).max(0) as u64 };

    let handle = net
        .tcp_connect(u.host, u.port, remaining(net))
        .map_err(HttpErr::Net)?;

    let mut req = String::new();
    req.push_str(&format!("POST {} HTTP/1.1\r\n", u.path));
    req.push_str(&format!("Host: {}\r\n", host_header(&u)));
    req.push_str("Content-Type: text/plain\r\n");
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n");
    req.push_str("User-Agent: aefinity-os/0.1\r\n");
    // `TOKEN <value>` in JOB.TXT, when the collector requires one. `job::parse`
    // has already refused any value with a control character in it, so this
    // cannot break the header framing; the check is repeated here because the
    // consequence of being wrong is request smuggling, not a bad header.
    if let Some(t) = token.filter(|t| !t.is_empty() && !t.bytes().any(|b| b < 0x20 || b == 0x7F)) {
        req.push_str(&format!("X-Aefinity-Token: {t}\r\n"));
    }
    req.push_str("\r\n");

    let mut wire = req.into_bytes();
    wire.extend_from_slice(body);

    if let Err(e) = net.tcp_send_all(&handle, &wire, remaining(net)) {
        net.tcp_close(handle, remaining(net));
        return Err(HttpErr::Net(e));
    }

    // Read the status line, and drain the rest of the reply until the peer
    // closes or the timeout expires (spec §5). A single `Until::Close` read
    // does both: the status line is the first line of whatever comes back,
    // and draining is what `Until::Close` already means.
    let (resp, drained) = net.tcp_recv_until(&handle, Until::Close, remaining(net));
    net.tcp_close(handle, remaining(net));

    // Bytes already on hand are read either way: a peer that sent a full
    // reply and then let the idle timer fire (rather than sending FIN) still
    // handed over a parseable status line, and that is what matters here.
    match parse_status(&resp) {
        Some(code) => Ok(code),
        None => match drained {
            Ok(()) => Err(HttpErr::BadResponse),
            Err(e) => Err(HttpErr::Net(e)),
        },
    }
}
