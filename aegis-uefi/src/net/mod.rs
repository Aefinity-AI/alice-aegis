//! AEFINITY OS phase 1a — the unikernel's own TCP/IP stack.
//!
//! Spec: `program/AEFINITY_OS.md` §1 (phase 1a), §2 (`NET`), §5 (this file).
//!
//! Why the OS carries a stack at all: Debian's OVMF 2025.02 ships **no** EDK2
//! network stack. The only thing `VirtioNetDxe` publishes is
//! `EFI_SIMPLE_NETWORK` — a raw send/receive/get-status packet interface — so
//! everything above the wire (ARP, IPv4, ICMP, UDP, DHCP, TCP) is ours. That
//! is what `smoltcp` is here for, and [`SnpDevice`] is the one adapter between
//! the two: it implements [`smoltcp::phy::Device`] over the firmware protocol.
//!
//! Everything stays inside boot services (spec §5): SNP, the file protocol,
//! the watchdog and `ResetSystem` all disappear at `ExitBootServices`, which
//! this unikernel therefore never calls.
//!
//! # Public surface
//!
//! Phase 1b (`net/http.rs`) and phase 2 (`server.rs`) are built on exactly
//! this, by other builders, so it is listed here in one place and kept small:
//!
//! | item | what it does |
//! |---|---|
//! | [`Net::bring_up`] | find the NIC, start it, take an address (static or DHCP), never hang |
//! | [`Net::poll`] | pump the stack once; call it whenever you are waiting |
//! | [`Net::tcp_connect`] | open a client connection, bounded by a timeout |
//! | [`Net::tcp_listen`] | open a listening socket on a port (phase 2) |
//! | [`Net::tcp_accepted`] | has a peer completed a handshake on it yet (phase 2) |
//! | [`Net::tcp_send_all`] | write a whole buffer, bounded by a timeout |
//! | [`Net::tcp_recv_until`] | read until a delimiter, a byte count, or close |
//! | [`Net::tcp_close`] | close one connection and release its buffers |
//! | [`Net::ip`] / [`Net::ip_string`] | the address in force, or `none` |
//! | [`Net::mac`] / [`Net::mac_string`] | the NIC's hardware address |
//! | [`Net::how`] | `dhcp` \| `static` \| `none` — where the address came from |
//! | [`parse_host_port`], [`parse_ipv4`], [`parse_cidr`] | `JOB.TXT` string forms |
//!
//! # Rules this file is written under
//!
//! - **Never hang the boot.** Every wait in here is bounded by a caller-given
//!   timeout, and a NIC that will not come up, a DHCP server that never
//!   answers and a peer that never accepts all end as a logged failure with
//!   the box still running. A headless lab worker that wedges on the network
//!   is worse than one that reports `ip=none`.
//! - **No `unwrap` on a firmware call.** Every SNP call is matched; failures
//!   are counted or logged and degraded past.
//! - **No panics reachable from firmware data.** `smoltcp` panics on a
//!   non-unicast hardware address and on a prefix length above 32, so both are
//!   checked here before the value is handed over — `panic = "abort"` on this
//!   target means a panic is a dead box in a rack.
//! - Rule A: nothing in this file measures anything. The clock exists to drive
//!   protocol timers, and no duration it produces may be quoted as a
//!   measurement.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::net::Ipv4Addr;
use core::time::Duration as CoreDuration;

use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet};
use smoltcp::phy::{Device, DeviceCapabilities, Medium};
use smoltcp::socket::{dhcpv4, tcp};
use smoltcp::time::{Duration as SmolDuration, Instant};
use smoltcp::wire::{EthernetAddress, HardwareAddress, IpAddress, IpCidr, IpEndpoint, Ipv4Cidr};

use uefi::boot::ScopedProtocol;
use uefi::proto::media::file::Directory;
use uefi::proto::network::snp::{NetworkState, ReceiveFlags, SimpleNetwork};

use crate::job::NetCfg;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/// How long `NET dhcp` is given before the box gives up and carries on with no
/// address (spec §2/§5: "DHCP with 10 s cap").
pub const DHCP_TIMEOUT_MS: u64 = 10_000;
/// Per-socket receive buffer. One MSS of slack over a 1500-byte path, which is
/// all phase 1a's `HELLO` exchange and phase 1b's HTTP response need.
const TCP_RX_BYTES: usize = 4096;
/// Per-socket transmit buffer.
const TCP_TX_BYTES: usize = 4096;
/// Frames handed to the NIC and not yet recycled. Each one is a live DMA
/// source, so the count is also the bound on how much memory an unresponsive
/// NIC can pin: `MAX_PENDING_TX * (MTU)`, about 12 KiB.
const MAX_PENDING_TX: usize = 8;
/// The backstop that tears down a connection nobody is talking on.
///
/// Deliberately **not** the caller's operation timeout. Arming the socket's
/// idle timer with the same value a caller passes to [`Net::tcp_connect`]
/// makes the backstop race every subsequent wait on that connection and win:
/// the first net-test run ended its exchange with a RST five seconds after
/// the connect, because the 5 s idle timer expired at the same moment the 5 s
/// wait-for-close did. A backstop that fires before the primary is not a
/// backstop. This is longer than any single operation the OS performs and
/// exists only so a peer that vanishes cannot leave a socket wedged on a box
/// nobody can log into.
const TCP_IDLE_TIMEOUT_MS: u64 = 60_000;
/// How long each polling loop sleeps between passes. Short enough that a
/// handshake is not sleep-bound, long enough that the loop is not a spin.
const POLL_SLEEP_MS: u64 = 2;
/// Bytes `tcp_recv_until` will accumulate before it stops, whatever the
/// caller asked for. A peer that never sends the delimiter must not be able to
/// exhaust the heap of a box that cannot be logged into.
const RECV_MAX_BYTES: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a network operation did not happen. Small on purpose: every variant
/// here ends up as a `RESULT.TXT` reason string or a `BOOTLOG.TXT` line, and a
/// reason a collector cannot act on is not worth a variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NetError {
    /// No `EFI_SIMPLE_NETWORK` handle, or the firmware would not give us one.
    NoNic,
    /// The NIC is up but the interface has no IP address (DHCP failed, or the
    /// `JOB.TXT` `NET` line was unusable).
    NoAddress,
    /// The peer refused the connection, or reset it.
    Refused,
    /// The connection closed before the operation finished.
    Closed,
    /// The caller's timeout expired first.
    Timeout,
    /// A `JOB.TXT` value did not parse (`NET static …`, `NETCHECK host:port`).
    BadAddress,
}

impl NetError {
    /// The short form that goes into `RESULT.TXT` / `BOOTLOG.TXT`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            NetError::NoNic => "no nic",
            NetError::NoAddress => "no ip",
            NetError::Refused => "refused",
            NetError::Closed => "closed",
            NetError::Timeout => "timeout",
            NetError::BadAddress => "bad address",
        }
    }
}

/// What [`Net::tcp_recv_until`] is waiting for.
///
/// `Delim` and `Len` carry `#[allow(dead_code)]` because phase 1a's only
/// reader waits for the peer to close: an HTTP reader (phase 1b) ends its
/// headers on a delimiter and its body on a length, and a resident server
/// (phase 2) reads a line at a time. They are part of the surface those
/// builders were promised, not code phase 1a exercises — which is why the
/// allow says so instead of the variants being quietly deleted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum Until {
    /// Stop after this byte is seen (it is included in the returned data).
    Delim(u8),
    /// Stop once this many bytes have arrived.
    Len(usize),
    /// Read until the peer closes, or the timeout expires.
    Close,
}

/// A connection opened by [`Net::tcp_connect`].
///
/// Not `Copy`: closing a handle removes the socket, and a second handle to a
/// removed socket would index a slot that has been reused.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TcpHandle(SocketHandle);

// ---------------------------------------------------------------------------
// String forms used by JOB.TXT (spec §2)
// ---------------------------------------------------------------------------

/// `a.b.c.d`.
#[must_use]
pub fn parse_ipv4(s: &str) -> Option<Ipv4Addr> {
    s.trim().parse::<Ipv4Addr>().ok()
}

/// `a.b.c.d/len`. Rejects a prefix above 32 rather than letting
/// `Ipv4Cidr::new` panic on it — firmware-side, a panic is an abort.
#[must_use]
pub fn parse_cidr(s: &str) -> Option<Ipv4Cidr> {
    let (addr, len) = s.trim().split_once('/')?;
    let addr = parse_ipv4(addr)?;
    let len: u8 = len.trim().parse().ok()?;
    if len > 32 {
        return None;
    }
    Some(Ipv4Cidr::new(addr, len))
}

/// `host:port`, host in dotted-quad form (no DNS: the stack has no resolver
/// and spec §2 gives addresses, not names).
#[must_use]
pub fn parse_host_port(s: &str) -> Option<(Ipv4Addr, u16)> {
    let (host, port) = s.trim().rsplit_once(':')?;
    let addr = parse_ipv4(host)?;
    let port: u16 = port.trim().parse().ok()?;
    if port == 0 {
        return None;
    }
    Some((addr, port))
}

/// `xx:xx:xx:xx:xx:xx`, lower case, always 17 characters.
///
/// Deliberately not `EthernetAddress`'s own `Display`, which separates with
/// `-`. The colon form is what every other tool on the fleet prints and what
/// the phase-1a gate matches on.
#[must_use]
pub fn mac_string(mac: &EthernetAddress) -> String {
    let b = mac.0;
    format!(
        "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5]
    )
}

// ---------------------------------------------------------------------------
// Clock (spec §5)
// ---------------------------------------------------------------------------

/// Where the millisecond count comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// UEFI Runtime Services `GetTime()`, via [`crate::wall_seconds`].
    Wall,
    /// RDTSC, scaled by a one-off calibration against `boot::stall`.
    Tsc,
}

/// The monotonic millisecond clock `smoltcp` drives its timers from.
///
/// Spec §5 asks for `wall_seconds()` with an RDTSC fallback calibrated once
/// against the UEFI stall, and that is what this is — with one addition that
/// the firmware forces. `GetTime()` is specified to report a nanosecond field,
/// but a firmware backed by the CMOS RTC advances it only once a second, and a
/// TCP stack driven by a clock that moves in one-second steps retransmits and
/// backs off on the wrong schedule. So the calibration measures the wall
/// clock's *resolution* over the same 10 ms stall that calibrates the TSC, and
/// picks the finer of the two. Which one was picked is logged, because it
/// changes how the timers behave and a post-mortem should not have to guess.
///
/// Rule A: this is a protocol clock. Its output is never a measurement, and
/// the TSC scaling here is a nominal-rate estimate, not a cycle count.
struct Clock {
    source: Source,
    /// `wall_seconds()` at calibration, for the `Wall` source.
    base_wall: f64,
    /// RDTSC at calibration, for the `Tsc` source.
    base_tsc: u64,
    /// RDTSC ticks per millisecond. Never zero (see [`Clock::calibrate`]).
    tsc_per_ms: u64,
    /// Highest millisecond value handed out so far. `smoltcp` subtracts
    /// `Instant`s, so time going backwards — a midnight rollover, a firmware
    /// whose RTC is stepped — would underflow a `Duration`. Clamping is
    /// cheaper than auditing every subtraction in a dependency.
    last_ms: Cell<i64>,
}

impl Clock {
    /// Measure both sources once and keep the usable one.
    fn calibrate() -> Clock {
        let w0 = crate::wall_seconds();
        // SAFETY: RDTSC is unprivileged and has no operands or side effects.
        // It is architecturally present on every x86_64 CPU this unikernel can
        // boot on, so there is no feature to test first.
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        uefi::boot::stall(CoreDuration::from_micros(10_000));
        // SAFETY: as above.
        let t1 = unsafe { core::arch::x86_64::_rdtsc() };
        let w1 = crate::wall_seconds();

        // 10 ms of stall, so ticks/10 is ticks per millisecond.
        let tsc_per_ms = t1.saturating_sub(t0) / 10;

        // Did the wall clock resolve anything finer than a second across the
        // stall? A firmware reporting whole seconds shows either 0.0 or 1.0
        // here; one with a real nanosecond field shows about 0.01.
        let wall_fine = matches!((w0, w1), (Some(a), Some(b)) if b > a && b - a < 1.0);

        let source = if wall_fine {
            Source::Wall
        } else if tsc_per_ms > 0 {
            Source::Tsc
        } else {
            // Neither source is usable as-is. Prefer the wall clock if it
            // exists at all (one-second granularity still lets TCP make
            // progress); otherwise the TSC with a floor of 1 tick/ms, which
            // runs fast but is at least monotonic. Both are logged.
            match w0 {
                Some(_) => Source::Wall,
                None => Source::Tsc,
            }
        };

        Clock {
            source,
            base_wall: w0.unwrap_or(0.0),
            base_tsc: t0,
            tsc_per_ms: tsc_per_ms.max(1),
            last_ms: Cell::new(0),
        }
    }

    /// Milliseconds since [`Clock::calibrate`], monotonically.
    fn now_ms(&self) -> i64 {
        let raw = match self.source {
            Source::Wall => match crate::wall_seconds() {
                Some(w) => {
                    let mut d = w - self.base_wall;
                    if d < 0.0 {
                        // `wall_seconds` is seconds since midnight; a boot that
                        // crosses midnight reads smaller than it started.
                        d += 86_400.0;
                    }
                    (d * 1000.0) as i64
                }
                // The firmware answered once and refuses now. Hold the clock
                // rather than jump it to zero.
                None => self.last_ms.get(),
            },
            Source::Tsc => {
                // SAFETY: RDTSC, as in `calibrate`.
                let t = unsafe { core::arch::x86_64::_rdtsc() };
                (t.saturating_sub(self.base_tsc) / self.tsc_per_ms) as i64
            }
        };
        if raw > self.last_ms.get() {
            self.last_ms.set(raw);
        }
        self.last_ms.get()
    }

    fn now(&self) -> Instant {
        Instant::from_millis(self.now_ms())
    }

    fn source_str(&self) -> &'static str {
        match self.source {
            Source::Wall => "wall",
            Source::Tsc => "tsc",
        }
    }
}

// ---------------------------------------------------------------------------
// SnpDevice — smoltcp::phy::Device over EFI_SIMPLE_NETWORK (spec §5)
// ---------------------------------------------------------------------------

/// The NIC, as `smoltcp` sees it.
///
/// # Transmit ownership
///
/// `EFI_SIMPLE_NETWORK.Transmit()` is **asynchronous**: it queues the caller's
/// buffer and returns, and the buffer stays a live DMA source until the driver
/// hands the pointer back through `GetStatus()`. Freeing it before then is a
/// use-after-free that the NIC, not the CPU, commits — silent corruption of
/// whatever the allocator handed out next.
///
/// So a transmitted frame is boxed and parked in `pending` keyed by its
/// address, and only dropped once `get_recycled_transmit_buffer_status`
/// returns that address. `pending` is capped at [`MAX_PENDING_TX`]; when it is
/// full the frame is not queued at all and `tx_dropped` counts it. Dropping a
/// frame is legal — TCP retransmits, ARP retries — and it is the only choice
/// here that is both bounded and sound.
pub struct SnpDevice {
    snp: ScopedProtocol<SimpleNetwork>,
    /// Ethernet frame size the NIC reports: media header + payload.
    mtu: usize,
    /// Reused receive staging buffer, so a poll that finds nothing allocates
    /// nothing.
    rx: Vec<u8>,
    /// Frames the NIC still owns, keyed by buffer address.
    pending: Vec<(usize, Box<[u8]>)>,
    /// Frames never handed to the NIC (queue full, or the firmware refused).
    tx_dropped: u64,
    /// Receives that failed for a reason other than "nothing to read".
    rx_errors: u64,
}

impl SnpDevice {
    /// Locate the first NIC, start it, initialize it, and set the receive
    /// filters this stack needs (unicast for our own traffic, broadcast for
    /// ARP and DHCP).
    ///
    /// Every step is logged and every failure returns `None` rather than
    /// unwinding: on a headless box the `BOOTLOG.TXT` line is the only
    /// evidence anyone will ever have about why the network is not there.
    fn open(root: &mut Directory) -> Option<SnpDevice> {
        let handles = match uefi::boot::find_handles::<SimpleNetwork>() {
            Ok(h) => h,
            Err(e) => {
                crate::boot_log(root, &format!("NET: no SimpleNetwork handles ({e:?})"));
                return None;
            }
        };
        let Some(&handle) = handles.first() else {
            crate::boot_log(root, "NET: no SimpleNetwork handles");
            return None;
        };
        if handles.len() > 1 {
            crate::boot_log(
                root,
                &format!("NET: {} NICs present, using the first", handles.len()),
            );
        }

        let snp = match uefi::boot::open_protocol_exclusive::<SimpleNetwork>(handle) {
            Ok(p) => p,
            Err(e) => {
                crate::boot_log(
                    root,
                    &format!("NET: SimpleNetwork open_protocol_exclusive failed ({e:?})"),
                );
                return None;
            }
        };

        // START. A firmware that already started the interface answers
        // ALREADY_STARTED, which is the state we want, not an error.
        match snp.start() {
            Ok(()) => {}
            Err(e) if e.status() == uefi::Status::ALREADY_STARTED => {}
            Err(e) => {
                crate::boot_log(root, &format!("NET: snp.start() failed ({e:?})"));
                return None;
            }
        }

        // INITIALIZE with no extra buffers: the driver's own defaults are what
        // every SNP consumer uses, and asking for more is a request a driver
        // may refuse outright.
        match snp.initialize(0, 0) {
            Ok(()) => {}
            Err(e) => {
                // Already initialized is fine; anything else is not, unless the
                // mode says the interface is up regardless.
                if snp.mode().state != NetworkState::INITIALIZED {
                    crate::boot_log(root, &format!("NET: snp.initialize() failed ({e:?})"));
                    return None;
                }
                crate::boot_log(
                    root,
                    &format!("NET: snp.initialize() said {e:?}, interface already initialized"),
                );
            }
        }

        // Unicast for our own traffic, broadcast for ARP and for the DHCP
        // offer, which is addressed to the broadcast MAC before we have an IP.
        // A driver that will not take the filters is logged and used anyway:
        // many report the setting as already in force.
        let want = ReceiveFlags::UNICAST | ReceiveFlags::BROADCAST;
        if let Err(e) = snp.receive_filters(want, ReceiveFlags::empty(), false, None) {
            crate::boot_log(
                root,
                &format!("NET: receive_filters(unicast|broadcast) refused ({e:?}) — continuing"),
            );
        }

        let mode = snp.mode();
        // Ethernet MTU as smoltcp defines it: the whole frame, header included,
        // without the FCS. SNP reports the two halves separately.
        let mtu = mode.media_header_size as usize + mode.max_packet_size as usize;
        if mtu < 64 {
            crate::boot_log(
                root,
                &format!("NET: NIC reports an unusable MTU of {mtu} bytes"),
            );
            return None;
        }
        // A NIC that can detect link and says there is none still gets a stack:
        // under QEMU the link comes up microseconds after the device does, and
        // a box that refused to network because it looked at the wrong
        // microsecond would be unreachable for the rest of its life.
        if bool::from(mode.media_present_supported) && !bool::from(mode.media_present) {
            crate::boot_log(
                root,
                "NET: NIC reports no link — bringing the stack up anyway",
            );
        }

        // A few bytes of slack: some drivers hand back the FCS with the frame.
        let rx = vec![0u8; mtu + 8];
        crate::boot_log(
            root,
            &format!(
                "NET: SNP up, mtu={mtu} (hdr={} payload={}) state={:?}",
                mode.media_header_size, mode.max_packet_size, mode.state
            ),
        );

        Some(SnpDevice {
            snp,
            mtu,
            rx,
            pending: Vec::new(),
            tx_dropped: 0,
            rx_errors: 0,
        })
    }

    /// The NIC's current hardware address, as an Ethernet address.
    ///
    /// SNP carries a 32-byte address field for media that need one; Ethernet
    /// uses the first six. `hw_address_size` is checked rather than assumed —
    /// a non-Ethernet NIC would otherwise be read as six bytes of something
    /// else and become a plausible-looking wrong MAC.
    fn mac(&self) -> Option<EthernetAddress> {
        let mode = self.snp.mode();
        if mode.hw_address_size != 6 {
            return None;
        }
        let o = mode.current_address.0;
        Some(EthernetAddress([o[0], o[1], o[2], o[3], o[4], o[5]]))
    }

    /// Take back every transmit buffer the driver has finished with.
    fn reclaim(&mut self) {
        // Bounded: one pass can only return as many buffers as we handed out.
        for _ in 0..=MAX_PENDING_TX {
            match self.snp.get_recycled_transmit_buffer_status() {
                Ok(Some(p)) => {
                    let addr = p.as_ptr() as usize;
                    if let Some(i) = self.pending.iter().position(|(a, _)| *a == addr) {
                        // Dropping the Box here is the point: the driver has
                        // said it is done reading from it.
                        self.pending.swap_remove(i);
                    }
                }
                // No buffer waiting, or the driver will not say. Either way
                // there is nothing to reclaim this pass.
                _ => break,
            }
        }
    }

    /// Queue one complete Ethernet frame.
    fn send(&mut self, buf: Box<[u8]>) {
        self.reclaim();
        if self.pending.len() >= MAX_PENDING_TX {
            // Give the driver a moment to finish one, then look again.
            uefi::boot::stall(CoreDuration::from_millis(1));
            self.reclaim();
        }
        if self.pending.len() >= MAX_PENDING_TX {
            self.tx_dropped += 1;
            return;
        }
        let addr = buf.as_ptr() as usize;
        // header_size 0: the frame already carries its Ethernet header, which
        // smoltcp wrote. Passing a non-zero header size would ask the driver to
        // overwrite it from the src/dst/protocol arguments instead.
        if self.snp.transmit(0, &buf, None, None, None).is_err() {
            // Rejected, so the driver never took the pointer and `buf` is ours
            // to drop at the end of this scope.
            self.tx_dropped += 1;
            return;
        }
        self.pending.push((addr, buf));
    }
}

impl Drop for SnpDevice {
    /// Stop the NIC before the transmit buffers go away.
    ///
    /// Order matters and it is the reason this impl exists: `Shutdown()`
    /// disables the receive and transmit queues, so after it returns the
    /// driver is no longer reading from anything in `pending`. Rust drops the
    /// struct's fields *after* this body, which is exactly the order the
    /// hardware needs. Dropping `pending` first would hand the allocator
    /// memory a live DMA engine still points at.
    fn drop(&mut self) {
        let _ = self.snp.shutdown();
        let _ = self.snp.stop();
    }
}

/// A received frame, already copied off the driver's buffer.
///
/// Owning the bytes is what lets `Device::receive` hand out a receive token
/// and a transmit token at once: only the transmit half borrows the device.
pub struct SnpRxToken {
    frame: Vec<u8>,
}

impl smoltcp::phy::RxToken for SnpRxToken {
    fn consume<R, F>(self, f: F) -> R
    where
        F: FnOnce(&[u8]) -> R,
    {
        f(&self.frame)
    }
}

/// Permission to send exactly one frame.
pub struct SnpTxToken<'a> {
    dev: &'a mut SnpDevice,
}

impl smoltcp::phy::TxToken for SnpTxToken<'_> {
    fn consume<R, F>(self, len: usize, f: F) -> R
    where
        F: FnOnce(&mut [u8]) -> R,
    {
        let mut buf: Box<[u8]> = vec![0u8; len].into_boxed_slice();
        let r = f(&mut buf);
        self.dev.send(buf);
        r
    }
}

impl Device for SnpDevice {
    type RxToken<'a> = SnpRxToken;
    type TxToken<'a> = SnpTxToken<'a>;

    fn receive(&mut self, _now: Instant) -> Option<(Self::RxToken<'_>, Self::TxToken<'_>)> {
        self.reclaim();
        let n = match self.snp.receive(&mut self.rx, None, None, None, None) {
            Ok(n) => n,
            Err(e) => {
                // NOT_READY is the normal "no packet waiting" answer and is not
                // an error; BUFFER_TOO_SMALL means the driver had a frame
                // larger than the MTU it advertised, which is worth counting.
                if e.status() != uefi::Status::NOT_READY {
                    self.rx_errors += 1;
                }
                return None;
            }
        };
        if n == 0 || n > self.rx.len() {
            return None;
        }
        let frame = self.rx[..n].to_vec();
        Some((SnpRxToken { frame }, SnpTxToken { dev: self }))
    }

    fn transmit(&mut self, _now: Instant) -> Option<Self::TxToken<'_>> {
        self.reclaim();
        if self.pending.len() >= MAX_PENDING_TX {
            // Refusing the token is better than accepting it and dropping the
            // frame: smoltcp will simply try again on the next poll.
            return None;
        }
        Some(SnpTxToken { dev: self })
    }

    // `DeviceCapabilities` is `#[non_exhaustive]`, so the struct-update form
    // `DeviceCapabilities { medium, ..Default::default() }` is rejected outside
    // smoltcp and field assignment is the only way to build one.
    #[allow(clippy::field_reassign_with_default)]
    fn capabilities(&self) -> DeviceCapabilities {
        let mut caps = DeviceCapabilities::default();
        caps.medium = Medium::Ethernet;
        caps.max_transmission_unit = self.mtu;
        caps.max_burst_size = Some(MAX_PENDING_TX);
        caps
    }
}

// ---------------------------------------------------------------------------
// Net (spec §5)
// ---------------------------------------------------------------------------

/// The unikernel's network, brought up and ready to use.
///
/// One interface over one NIC, one socket set, one clock. Dropping it shuts
/// the NIC down (see [`SnpDevice::drop`]) and releases the exclusive protocol
/// open, which is what makes it safe to reset the machine afterwards.
pub struct Net {
    dev: SnpDevice,
    iface: Interface,
    sockets: SocketSet<'static>,
    clock: Clock,
    mac: EthernetAddress,
    ip: Option<Ipv4Cidr>,
    how: &'static str,
    dhcp: Option<SocketHandle>,
    /// Rotating source of ephemeral local ports.
    next_port: u16,
    /// Sockets opened by [`Net::tcp_listen`], with the port each was told to
    /// listen on.
    ///
    /// smoltcp's `reset()` clears a socket's `listen_endpoint`, so a listener
    /// that a peer aborted mid-handshake forgets which port it was serving.
    /// A resident box has to re-listen after every close (spec §4), and it
    /// cannot ask the socket where it was listening — so the port is kept
    /// here instead. A `Vec` and not a map: a resident server holds two of
    /// these (the one being served and one backlog slot for the `BUSY`
    /// answer), and a linear scan of two entries needs no hashing.
    listeners: Vec<(SocketHandle, u16)>,
}

impl Net {
    /// Find the NIC and take an address, per the `JOB.TXT` `NET` directive.
    ///
    /// Returns `None` only when there is no usable NIC at all. **A failure to
    /// get an address is not a failure to bring up**: the caller gets a `Net`
    /// with [`Net::ip`] `None`, `RESULT.TXT` records `ip=none`, and the box
    /// carries on with the rest of its job. Spec §5: never hang boot.
    ///
    /// Spec §2 says `NET dhcp` "falls back to static if given". A `JOB.TXT`
    /// `NET` line is one directive in one of two forms, so a job that asks for
    /// DHCP has not given a static address to fall back to — the fallback is
    /// reachable only once the job format carries both, and this function
    /// applies whichever form the config holds rather than pretending
    /// otherwise.
    pub fn bring_up(cfg: &NetCfg, root: &mut Directory) -> Option<Net> {
        let mut dev = SnpDevice::open(root)?;

        let Some(mac) = dev.mac() else {
            crate::boot_log(
                root,
                "NET: NIC is not Ethernet (hw_address_size != 6) — no stack",
            );
            return None;
        };
        // smoltcp panics on a non-unicast hardware address, and `panic =
        // "abort"` here means a dead box. Check it ourselves.
        if !mac.is_unicast() {
            crate::boot_log(
                root,
                &format!(
                    "NET: NIC reports a non-unicast MAC {} — no stack",
                    mac_string(&mac)
                ),
            );
            return None;
        }
        crate::boot_log(root, &format!("NET: mac={}", mac_string(&mac)));

        let clock = Clock::calibrate();
        crate::boot_log(
            root,
            &format!(
                "NET: clock source={} tsc_per_ms={}",
                clock.source_str(),
                clock.tsc_per_ms
            ),
        );

        // A fresh seed per boot keeps TCP local ports and sequence numbers from
        // repeating across the reboot cycle a resident box lives in.
        // SAFETY: RDTSC is unprivileged, has no operands and no side effects.
        let seed = unsafe { core::arch::x86_64::_rdtsc() };
        let mut config = Config::new(HardwareAddress::Ethernet(mac));
        config.random_seed = seed;
        let now = clock.now();
        let iface = Interface::new(config, &mut dev, now);

        let mut net = Net {
            dev,
            iface,
            sockets: SocketSet::new(Vec::new()),
            clock,
            mac,
            ip: None,
            how: "none",
            dhcp: None,
            // Start somewhere in the ephemeral range, seeded per boot.
            next_port: 49152 + (seed % 16000) as u16,
            listeners: Vec::new(),
        };

        match cfg {
            NetCfg::Static { cidr, gateway } => match (parse_cidr(cidr), parse_ipv4(gateway)) {
                (Some(c), Some(g)) => {
                    net.apply_address(c, Some(g));
                    net.how = "static";
                }
                _ => {
                    crate::boot_log(
                        root,
                        &format!("NET: NET static {cidr} {gateway} did not parse — ip=none"),
                    );
                }
            },
            NetCfg::Dhcp => {
                if net.run_dhcp(root) {
                    net.how = "dhcp";
                } else {
                    crate::boot_log(
                        root,
                        "NET: DHCP got no lease and the job gave no static address",
                    );
                }
            }
        }

        crate::boot_log(
            root,
            &format!("NET: ip={} ({})", net.ip_string(), net.how()),
        );
        Some(net)
    }

    /// Pump the stack once: move frames in and out, run protocol timers, and
    /// apply any DHCP state change.
    ///
    /// Call it in any loop that is waiting on the network. It never blocks.
    pub fn poll(&mut self) {
        let now = self.clock.now();
        self.iface.poll(now, &mut self.dev, &mut self.sockets);
        self.drain_dhcp();
    }

    /// The NIC's hardware address.
    #[must_use]
    pub fn mac(&self) -> EthernetAddress {
        self.mac
    }

    /// The NIC's hardware address as `xx:xx:xx:xx:xx:xx`.
    #[must_use]
    pub fn mac_string(&self) -> String {
        mac_string(&self.mac())
    }

    /// The address in force, if the interface has one.
    #[must_use]
    pub fn ip(&self) -> Option<Ipv4Cidr> {
        self.ip
    }

    /// The address in force as `a.b.c.d`, or `none` — the `RESULT.TXT` form
    /// (spec §3).
    #[must_use]
    pub fn ip_string(&self) -> String {
        match self.ip {
            Some(c) => {
                let o = c.address().octets();
                format!("{}.{}.{}.{}", o[0], o[1], o[2], o[3])
            }
            None => String::from("none"),
        }
    }

    /// Where the address came from: `dhcp`, `static`, or `none`.
    #[must_use]
    pub fn how(&self) -> &'static str {
        self.how
    }

    /// Frames the driver would not take, and receives that failed for a reason
    /// other than "nothing waiting". Both are diagnostics for `BOOTLOG.TXT`;
    /// neither is a measurement.
    #[must_use]
    pub fn counters(&self) -> (u64, u64) {
        (self.dev.tx_dropped, self.dev.rx_errors)
    }

    /// Open a listening TCP socket on `port` (spec §4, phase 2).
    ///
    /// The socket is placed in `LISTEN` and stays there until a peer completes
    /// a handshake on it; [`Net::tcp_accepted`] is how the caller finds out.
    /// Once accepted, the same handle *is* the connection — smoltcp has no
    /// separate accepted socket — so a server that wants to keep serving opens
    /// a second listener as its backlog slot and promotes it when the first
    /// connection ends. That is also what makes the `BUSY` answer of spec §4
    /// possible: a second peer completes its handshake on the backlog socket
    /// instead of being left unanswered.
    ///
    /// **No idle backstop is armed here**, unlike [`Net::tcp_connect`]. Spec
    /// §4: "a resident box idles indefinitely". smoltcp's socket timeout
    /// aborts a connection only when there is unacknowledged data in the
    /// transmit buffer (or keep-alive is on), so it would not fire on a quiet
    /// connection anyway — but a listener that expires while nobody is talking
    /// to it is exactly the failure a resident box must not have, so the
    /// timer is left off and the *server* bounds its own idle, where a
    /// BOOTLOG line can say what happened.
    pub fn tcp_listen(&mut self, port: u16) -> Result<TcpHandle, NetError> {
        if port == 0 {
            return Err(NetError::BadAddress);
        }
        let mut sock = tcp::Socket::new(
            tcp::SocketBuffer::new(vec![0u8; TCP_RX_BYTES]),
            tcp::SocketBuffer::new(vec![0u8; TCP_TX_BYTES]),
        );
        sock.set_timeout(None);
        if sock.listen(port).is_err() {
            return Err(NetError::BadAddress);
        }
        let handle = self.sockets.add(sock);
        self.listeners.push((handle, port));
        // One pass so a SYN already sitting in the NIC's receive queue is
        // answered before the caller's first `tcp_accepted`.
        self.poll();
        Ok(TcpHandle(handle))
    }

    /// Has a peer completed a handshake on this listener?
    ///
    /// Polls the stack, so it is also the pump a server's accept loop needs;
    /// call it in a loop with a short stall between passes.
    ///
    /// A listener whose handshake was aborted (a peer that sent SYN and then
    /// RST, or a slirp forward whose far end went away) lands back in
    /// `Closed`, which in smoltcp is a *dead* socket, not a listening one.
    /// This re-arms it on the port it was opened with — the "re-listen after
    /// each close" of spec §4 — so a box nobody is talking to cannot quietly
    /// stop being reachable.
    pub fn tcp_accepted(&mut self, h: &TcpHandle) -> bool {
        self.poll();
        let state = self.sockets.get::<tcp::Socket>(h.0).state();
        match state {
            // `CloseWait` too: a peer that connects and immediately sends FIN
            // has still been accepted, and the caller must be told so it can
            // close the socket rather than wait on a connection that is over.
            tcp::State::Established | tcp::State::CloseWait => true,
            tcp::State::Closed => {
                self.relisten(h);
                false
            }
            _ => false,
        }
    }

    /// Put a dead listener back into `LISTEN` on the port it was opened with.
    fn relisten(&mut self, h: &TcpHandle) {
        let Some(&(_, port)) = self.listeners.iter().find(|(sh, _)| *sh == h.0) else {
            return;
        };
        let _ = self.sockets.get_mut::<tcp::Socket>(h.0).listen(port);
    }

    /// Open a TCP connection.
    ///
    /// `timeout_ms` bounds the **handshake** only. The connection that comes
    /// back carries its own [`TCP_IDLE_TIMEOUT_MS`] backstop, so a caller is
    /// free to hold it open for longer than it took to make.
    pub fn tcp_connect(
        &mut self,
        ip: Ipv4Addr,
        port: u16,
        timeout_ms: u64,
    ) -> Result<TcpHandle, NetError> {
        if self.ip.is_none() {
            return Err(NetError::NoAddress);
        }
        if port == 0 {
            return Err(NetError::BadAddress);
        }

        let mut sock = tcp::Socket::new(
            tcp::SocketBuffer::new(vec![0u8; TCP_RX_BYTES]),
            tcp::SocketBuffer::new(vec![0u8; TCP_TX_BYTES]),
        );
        // The wedged-socket backstop, independent of `timeout_ms` — see
        // TCP_IDLE_TIMEOUT_MS for why the two must not be the same value.
        sock.set_timeout(Some(SmolDuration::from_millis(TCP_IDLE_TIMEOUT_MS)));
        let handle = self.sockets.add(sock);

        let local = self.take_local_port();
        {
            let cx = self.iface.context();
            let s = self.sockets.get_mut::<tcp::Socket>(handle);
            let remote = IpEndpoint::new(IpAddress::Ipv4(ip), port);
            if s.connect(cx, remote, local).is_err() {
                self.sockets.remove(handle);
                return Err(NetError::BadAddress);
            }
        }

        let deadline = self.clock.now_ms() + timeout_ms as i64;
        loop {
            self.poll();
            match self.sockets.get::<tcp::Socket>(handle).state() {
                tcp::State::Established => return Ok(TcpHandle(handle)),
                // The handshake ended without a connection: refused, reset, or
                // timed out inside smoltcp.
                tcp::State::Closed => {
                    self.sockets.remove(handle);
                    return Err(NetError::Refused);
                }
                _ => {}
            }
            if self.clock.now_ms() >= deadline {
                self.sockets.remove(handle);
                return Err(NetError::Timeout);
            }
            uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
        }
    }

    /// Write every byte of `data`, then wait for the stack to hand it all to
    /// the NIC. Gives up after `timeout_ms`.
    pub fn tcp_send_all(
        &mut self,
        h: &TcpHandle,
        data: &[u8],
        timeout_ms: u64,
    ) -> Result<(), NetError> {
        let deadline = self.clock.now_ms() + timeout_ms as i64;
        let mut off = 0usize;

        while off < data.len() {
            self.poll();
            {
                let s = self.sockets.get_mut::<tcp::Socket>(h.0);
                if !s.may_send() {
                    return Err(NetError::Closed);
                }
                match s.send_slice(&data[off..]) {
                    Ok(n) => off += n,
                    Err(_) => return Err(NetError::Closed),
                }
            }
            if off < data.len() {
                if self.clock.now_ms() >= deadline {
                    return Err(NetError::Timeout);
                }
                uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
            }
        }

        // Queued is not sent. Drain the socket's transmit buffer so a caller
        // that closes immediately afterwards does not close over unsent bytes.
        loop {
            self.poll();
            if self.sockets.get::<tcp::Socket>(h.0).send_queue() == 0 {
                return Ok(());
            }
            if !self.sockets.get::<tcp::Socket>(h.0).may_send() {
                return Err(NetError::Closed);
            }
            if self.clock.now_ms() >= deadline {
                return Err(NetError::Timeout);
            }
            uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
        }
    }

    /// Read until `until` is satisfied, the peer closes, or `timeout_ms`
    /// expires.
    ///
    /// Bytes already received are always returned, even on `Err`: a truncated
    /// answer plus an honest error is worth more to a collector than nothing.
    /// That is why the error type carries the buffer with it.
    pub fn tcp_recv_until(
        &mut self,
        h: &TcpHandle,
        until: Until,
        timeout_ms: u64,
    ) -> (Vec<u8>, Result<(), NetError>) {
        let deadline = self.clock.now_ms() + timeout_ms as i64;
        let mut out: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 1024];

        loop {
            self.poll();

            // Drain everything the socket is holding before deciding anything.
            loop {
                let s = self.sockets.get_mut::<tcp::Socket>(h.0);
                if !s.can_recv() {
                    break;
                }
                match s.recv_slice(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => out.extend_from_slice(&chunk[..n]),
                    Err(_) => break,
                }
                if out.len() >= RECV_MAX_BYTES {
                    return (out, Err(NetError::Closed));
                }
            }

            match until {
                Until::Delim(d) => {
                    if out.contains(&d) {
                        return (out, Ok(()));
                    }
                }
                Until::Len(n) => {
                    if out.len() >= n {
                        return (out, Ok(()));
                    }
                }
                Until::Close => {}
            }

            // The receive half has ended and we have drained what was in it.
            //
            // *Why* it ended is a different question from *that* it ended, and
            // the two must not be reported as one. A peer that sent FIN leaves
            // the socket in CloseWait (or, once we answer, LastAck / Closing /
            // TimeWait). A connection torn down by this stack's own idle
            // timeout — `set_timeout` in `tcp_connect`, armed with the caller's
            // timeout — or by a reset goes straight to Closed without the peer
            // having said anything. Only the first is "the peer closed", and
            // `Until::Close` is only *satisfied* by the first.
            let s = self.sockets.get::<tcp::Socket>(h.0);
            if !s.may_recv() {
                let peer_closed = matches!(
                    s.state(),
                    tcp::State::CloseWait
                        | tcp::State::LastAck
                        | tcp::State::Closing
                        | tcp::State::TimeWait
                );
                return match (until, peer_closed) {
                    (Until::Close, true) => (out, Ok(())),
                    _ => (out, Err(NetError::Closed)),
                };
            }
            if self.clock.now_ms() >= deadline {
                return (out, Err(NetError::Timeout));
            }
            uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
        }
    }

    /// Close a connection and release its buffers.
    ///
    /// Waits up to `timeout_ms` for the four-way close so the peer sees a FIN
    /// rather than a reset, then removes the socket either way — a handle that
    /// outlived its socket is a bug the caller cannot see, so `TcpHandle` is
    /// consumed here.
    ///
    /// `TimeWait` counts as closed, and that is not a shortcut. The side that
    /// closes first never reaches `Closed`: it sits in `TimeWait` for 2 MSL,
    /// so a loop waiting for `Closed` burns its whole timeout on a connection
    /// that shut down perfectly and then aborts it. The frame dump from the
    /// run before this comment existed shows the cost — FIN, ACK, FIN, ACK,
    /// and then a gratuitous RST pair two seconds later. Both FINs have been
    /// exchanged and acknowledged by `TimeWait`; there is nothing left to wait
    /// for. The wait that remains is for a peer that never answers our FIN,
    /// and for that one the abort below is the right ending.
    pub fn tcp_close(&mut self, h: TcpHandle, timeout_ms: u64) {
        let deadline = self.clock.now_ms() + timeout_ms as i64;
        self.sockets.get_mut::<tcp::Socket>(h.0).close();
        loop {
            self.poll();
            if matches!(
                self.sockets.get::<tcp::Socket>(h.0).state(),
                tcp::State::Closed | tcp::State::TimeWait
            ) {
                break;
            }
            if self.clock.now_ms() >= deadline {
                self.sockets.get_mut::<tcp::Socket>(h.0).abort();
                self.poll();
                break;
            }
            uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
        }
        self.sockets.remove(h.0);
        // If this was a listener, its port bookkeeping goes with it. Leaving
        // the entry behind would let a later socket reuse the slot's index and
        // inherit a port it was never opened on.
        self.listeners.retain(|(sh, _)| *sh != h.0);
    }

    // -- internals ----------------------------------------------------------

    /// Next ephemeral local port. Wraps inside 49152..=65535 (IANA dynamic
    /// range) so a long-lived resident box never walks out of it.
    fn take_local_port(&mut self) -> u16 {
        const LOW: u16 = 49152;
        let p = self.next_port;
        // `checked_add` rather than `>= u16::MAX`: at the top of the range the
        // increment would overflow, and this says so without a comparison
        // against the type's maximum.
        self.next_port = match self.next_port.checked_add(1) {
            Some(next) if next >= LOW => next,
            _ => LOW,
        };
        p
    }

    /// Put an address and default route on the interface.
    fn apply_address(&mut self, cidr: Ipv4Cidr, gateway: Option<Ipv4Addr>) {
        self.iface.update_ip_addrs(|addrs| {
            addrs.clear();
            // Capacity is smoltcp's IFACE_MAX_ADDR_COUNT and we just cleared
            // it, so this cannot fail; ignoring the result keeps the panic-free
            // rule without pretending to handle an impossible case.
            let _ = addrs.push(IpCidr::Ipv4(cidr));
        });
        match gateway {
            Some(g) => {
                let _ = self.iface.routes_mut().add_default_ipv4_route(g);
            }
            None => {
                let _ = self.iface.routes_mut().remove_default_ipv4_route();
            }
        }
        self.ip = Some(cidr);
    }

    /// Run a DHCPv4 client for up to [`DHCP_TIMEOUT_MS`]. Returns whether a
    /// lease was taken.
    fn run_dhcp(&mut self, root: &mut Directory) -> bool {
        let handle = self.sockets.add(dhcpv4::Socket::new());
        self.dhcp = Some(handle);

        let deadline = self.clock.now_ms() + DHCP_TIMEOUT_MS as i64;
        loop {
            self.poll();
            if self.ip.is_some() {
                return true;
            }
            if self.clock.now_ms() >= deadline {
                break;
            }
            uefi::boot::stall(CoreDuration::from_millis(POLL_SLEEP_MS));
        }

        crate::boot_log(
            root,
            &format!("NET: no DHCP lease within {DHCP_TIMEOUT_MS} ms — continuing without one"),
        );
        self.sockets.remove(handle);
        self.dhcp = None;
        false
    }

    /// Apply whatever the DHCP client decided since the last poll.
    fn drain_dhcp(&mut self) {
        let Some(handle) = self.dhcp else {
            return;
        };
        // The event borrows the socket, so copy the two fields out and let the
        // borrow end before touching the interface.
        let change = match self.sockets.get_mut::<dhcpv4::Socket>(handle).poll() {
            None => return,
            Some(dhcpv4::Event::Deconfigured) => None,
            Some(dhcpv4::Event::Configured(c)) => Some((c.address, c.router)),
        };
        match change {
            Some((cidr, router)) => {
                self.apply_address(cidr, router);
                self.how = "dhcp";
            }
            None => {
                self.iface.update_ip_addrs(|addrs| addrs.clear());
                let _ = self.iface.routes_mut().remove_default_ipv4_route();
                self.ip = None;
            }
        }
    }
}
