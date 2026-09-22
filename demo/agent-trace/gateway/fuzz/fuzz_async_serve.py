#!/usr/bin/env python3
"""safe-7c: mutation/protocol fuzz driver for the NEW async `gateway serve`
wire protocol (PENDING ticket=<id> / POLL ticket=<id> -> ALLOW|DENY),
extending the safe-1e mutation-fuzz kit
(`demo/agent-trace/gateway/fuzz/mutate_gateway.py`, see
`state/reports/2026-09-13-safe1e-gateway-fuzz-box2.md` in claudius-maximus)
from the old synchronous CLI protocol to the new socket-based daemon.

Talks directly to a real `gateway serve` daemon subprocess over its unix
socket (raw bytes, not line-buffered subprocess-per-case like the old sync
kit -- there is no separate CLI invocation per decision anymore, the daemon
is long-lived). Five case classes, matching the safe-7c brief:

  1. malformed_request   - byte-flip/truncate/non-utf8/giant-field/dup-line/
                            crlf/garbage-prefix mutations of a DECIDE
                            request line-block, sent over a fresh
                            connection each case.
  2. malformed_poll      - POLL requests with unknown/empty/oversized/
                            non-ascii/reused/cross-daemon-restart ticket
                            values.
  3. mass_enqueue_no_poll- thousands of DECIDE requests, each a fresh valid
                            request (unique SESSION id) that is never
                            POLLed, checking the daemon survives (no crash,
                            no hang) and that its ticket-table memory
                            footprint is bounded (not naively proportional
                            to case count forever) after the SAFE-7c fix.
  4. concurrent_replay   - N client threads submit the identical
                            (session, counter, receipt, action) concurrently
                            while the daemon is up; exactly one may reach
                            PENDING/ALLOW, the rest must DENY (freshness).
  5. worker_crash        - a DECIDE goes PENDING against a deliberately
                            slow `agent_trace` stand-in; the OS-level verify
                            subprocess and its whole descendant tree is
                            SIGKILLed while still running; POLL must
                            eventually resolve to DENY, never hang forever.

For EVERY case, the only acceptable daemon behaviors are:
  - a reply that starts with "PENDING ticket=", "ALLOW ", or "DENY " on the
    connection that sent the case (read with a bounded timeout), AND
  - the daemon process itself still running afterward (a case that kills
    the whole daemon, or that leaves "panicked" anywhere in its stderr, is
    the worst possible finding: a strictly worse regression than any single
    connection just getting a wrong reply).
A reply that never arrives inside the per-case read timeout, or a
"panicked" in the daemon's stderr, or the daemon exiting, is a FINDING.
Class 4 additionally checks the ALLOW/PENDING-vs-DENY distribution
(never more than one live outcome per (session, counter)); class 5
additionally checks the specific ticket actually resolves (not just that
the daemon survives).

No wall-clock/performance numbers are recorded or printed by this script
per program rules (state/reports numbers are pass/fail counts and, for
class 3, approximate memory-footprint deltas only -- not timings).
"""
import hashlib
import os
import random
import re
import selectors
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
GW_DIR = REPO / "demo/agent-trace/gateway"
GATEWAY_BIN = GW_DIR / "target/release/gateway"
AGENT_TRACE_BIN = REPO / "aegis-linux/target/release/examples/agent_trace"
TINYBIT = REPO / "model-lab/tinybit/m7_final_gate_work/artifacts"

RNG_SEED = 20260914
random.seed(RNG_SEED)

# Set FUZZ_QUICK=1 for a fast smoke-test pass with tiny case counts (used to
# sanity-check the harness itself before a full >=2000-case run).
QUICK = os.environ.get("FUZZ_QUICK") == "1"

CASE_READ_TIMEOUT_S = 5.0     # per-case reply-wait budget (class 1/2/3/4)
WORKER_CRASH_RESOLVE_BOUND_S = 15.0  # class 5: bound to observe eventual DENY
DAEMON_START_BOUND_S = 10.0


def sha256hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------
# Daemon lifecycle
# ---------------------------------------------------------------------

def build_signed_allowlist(path: Path, triples, key: bytes):
    import hmac as _hmac
    lines = [f"{m} {e} {v}" for (m, e, v) in triples]
    body = "".join(l + "\n" for l in lines).encode("utf-8")
    sig = _hmac.new(key, body, hashlib.sha256).hexdigest()
    path.write_bytes(body + f"sig {sig}\n".encode("utf-8"))


def artifact_hexes():
    mh = sha256hex((TINYBIT / "MODEL.SAF").read_bytes())
    eh = sha256hex((TINYBIT / "EMBED.BIN").read_bytes())
    vh = sha256hex((TINYBIT / "VOCAB.BIN").read_bytes())
    return mh, eh, vh


def gen_receipt(work: Path, prompt: str, k: int, n: int, tag: str) -> Path:
    out = work / f"seed_{tag}.receipt"
    with open(out, "wb") as f:
        p = subprocess.run(
            [str(AGENT_TRACE_BIN), "gen", str(TINYBIT / "MODEL.SAF"), str(TINYBIT / "EMBED.BIN"),
             str(TINYBIT / "VOCAB.BIN"), str(k), str(n), prompt],
            stdout=f, stderr=subprocess.PIPE, timeout=60,
        )
    assert p.returncode == 0, f"gen failed: {p.stderr}"
    return out


def last_step_in_hex(receipt_path: Path) -> str:
    last = None
    for line in receipt_path.read_text().splitlines():
        if line.startswith("step "):
            for field in line.split():
                if field.startswith("in="):
                    last = field[3:]
    assert last is not None
    return last


def make_fast_pass_stub(work: Path) -> Path:
    """A fake `agent_trace` stand-in that resolves instantly and always
    reports PASS, so class 3/4 can exercise thousands of decisions worth of
    protocol/bookkeeping behavior without thousands of real (slower) model
    verify subprocess spawns. Never used for a case whose *decision outcome*
    correctness matters -- only ticket-table bookkeeping and freshness
    dedup, both independent of verify's actual pass/fail bytes."""
    path = work / "fast_pass_stub.sh"
    path.write_text("#!/bin/sh\necho 'VERIFY PASS'\n")
    path.chmod(0o755)
    return path


def make_slow_wrapper(work: Path, sleep_secs: int) -> Path:
    path = work / f"slow_agent_trace_{sleep_secs}_{uuid.uuid4().hex[:8]}.sh"
    path.write_text(f"#!/bin/sh\nsleep {sleep_secs}\nexec \"{AGENT_TRACE_BIN}\" \"$@\"\n")
    path.chmod(0o755)
    return path


class Daemon:
    def __init__(self, work: Path, agent_trace_bin: Path, verify_workers=1,
                 request_timeout=None, tag="d"):
        self.work = work
        (mh, eh, vh) = artifact_hexes()
        self.key = b"safe7c-fuzz-key-not-for-prod"
        self.allowlist = work / f"allowlist_{tag}.signed"
        build_signed_allowlist(self.allowlist, [(mh, eh, vh)], self.key)
        self.keyfile = work / f"key_{tag}.bin"
        self.keyfile.write_bytes(self.key)
        self.cap_key_file = work / f"capkey_{tag}.bin"
        self.cap_key_file.write_bytes(os.urandom(32))
        self.socket_path = work / f"gw_{tag}_{uuid.uuid4().hex[:8]}.sock"
        cmd = [
            str(GATEWAY_BIN), "serve",
            str(TINYBIT / "MODEL.SAF"), str(TINYBIT / "EMBED.BIN"), str(TINYBIT / "VOCAB.BIN"),
            str(agent_trace_bin), str(self.allowlist), str(self.keyfile),
            "--socket", str(self.socket_path),
            "--cap-key-file", str(self.cap_key_file),
            "--verify-workers", str(verify_workers),
        ]
        if request_timeout is not None:
            cmd += ["--request-timeout", str(request_timeout)]
        # IMPORTANT harness-correctness note (found while running this
        # fuzzer, NOT a gateway bug): `gateway serve` logs one `eprintln!`
        # line per accepted connection ("gateway serve: accepted #N"). With
        # `stdout=PIPE`/`stderr=PIPE` and nothing draining those pipes while
        # the daemon runs, the OS pipe buffer (~64KB) fills after a few
        # thousand connections and the daemon's own blocking write to
        # stderr then blocks -- freezing its single-threaded accept loop
        # entirely (looks exactly like a hung/wedged daemon from the
        # outside, but is a fuzz-harness artifact: piping to files that are
        # never read, not anything `gateway serve` itself does wrong).
        # Redirecting to real files avoids this entirely.
        self.stdout_path = work / f"stdout_{tag}.log"
        self.stderr_path = work / f"stderr_{tag}.log"
        self._stdout_f = open(self.stdout_path, "wb")
        self._stderr_f = open(self.stderr_path, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self._stdout_f, stderr=self._stderr_f)
        t0 = time.monotonic()
        while not self.socket_path.exists():
            if time.monotonic() - t0 > DAEMON_START_BOUND_S:
                raise RuntimeError("daemon did not create its socket in time")
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"daemon exited early: rc={self.proc.returncode} "
                    f"stderr={self.stderr_path.read_bytes()!r}"
                )
            time.sleep(0.05)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self):
        if self.alive():
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        try:
            self._stdout_f.close()
        except Exception:
            pass
        try:
            self._stderr_f.close()
        except Exception:
            pass
        out = self.stdout_path.read_bytes() if self.stdout_path.exists() else b""
        err = self.stderr_path.read_bytes() if self.stderr_path.exists() else b""
        return out, err

    def descendant_pids(self):
        """All live descendant PIDs of the daemon (children, grandchildren,
        ...), for killing an in-flight verify subprocess regardless of
        whether it's currently the wrapper shell, `sleep`, or the real
        agent_trace binary."""
        by_ppid = {}
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():
                continue
            try:
                stat = (p / "stat").read_text()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            # pid (comm) state ppid ...
            rparen = stat.rfind(")")
            fields = stat[rparen + 2:].split()
            ppid = int(fields[1])
            by_ppid.setdefault(ppid, []).append(int(p.name))
        out = []
        frontier = [self.proc.pid]
        while frontier:
            nxt = []
            for pid in frontier:
                for child in by_ppid.get(pid, []):
                    out.append(child)
                    nxt.append(child)
            frontier = nxt
        return out


def send_raw(socket_path: Path, data: bytes, read_timeout=CASE_READ_TIMEOUT_S):
    """Connect, send raw bytes, half-close (mirrors a client that sent a
    truncated/malformed request and disconnected), then read a reply with a
    bounded timeout. Returns (reply_bytes_or_None, timed_out: bool,
    conn_error: Exception|None)."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(read_timeout)
        s.connect(str(socket_path))
        s.sendall(data)
        try:
            s.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        chunks = []
        try:
            while True:
                b = s.recv(65536)
                if not b:
                    break
                chunks.append(b)
        except socket.timeout:
            s.close()
            return (b"".join(chunks) if chunks else None, True, None)
        s.close()
        return (b"".join(chunks), False, None)
    except OSError as e:
        return (None, False, e)


def decide_bytes(receipt: Path, action_hex: str, session: str, counter) -> bytes:
    return (
        f"RECEIPT {receipt}\nACTION {action_hex}\nSESSION {session}\nCOUNTER {counter}\n\n"
    ).encode("utf-8")


def poll_bytes(ticket: str) -> bytes:
    return f"POLL ticket={ticket}\n\n".encode("utf-8", errors="surrogateescape")


VALID_OUTCOME_RE = re.compile(rb"^(PENDING\b|ALLOW |DENY )")


def classify(reply, timed_out, conn_err, daemon_alive_after):
    if not daemon_alive_after:
        return "FINDING-DAEMON-DIED"
    if timed_out:
        return "FINDING-HANG"
    if conn_err is not None:
        return "FINDING-CONN-ERROR"
    if reply is None or reply == b"":
        # A clean early close with no bytes at all (e.g. genuinely empty
        # request) is only acceptable if it came from a case that sent
        # literally nothing; conservatively require *some* reply for any
        # case that sent >0 bytes, since every code path in
        # handle_serve_conn_async does write a line before returning.
        return "FINDING-NO-REPLY"
    if not VALID_OUTCOME_RE.match(reply):
        return "FINDING-BAD-OUTCOME"
    return "ok"


# ---------------------------------------------------------------------
# Mutation classes for class 1 (malformed_request)
# ---------------------------------------------------------------------

def mut_byte_flip(data: bytes) -> bytes:
    if not data:
        return data
    b = bytearray(data)
    for _ in range(random.choice([1, 1, 2, 4, 8])):
        i = random.randrange(len(b))
        b[i] ^= 1 << random.randrange(8)
    return bytes(b)


def mut_truncate(data: bytes) -> bytes:
    if not data:
        return data
    cut = random.choice([0, 1, len(data) // 3, len(data) // 2, len(data) - 1])
    return data[: max(0, min(cut, len(data)))]


def mut_non_utf8(data: bytes) -> bytes:
    seqs = [b"\xff\xfe", b"\x80\x80\x80", b"\xc0\xaf", b"\xed\xa0\x80", bytes([random.randrange(0x80, 0x100)])]
    pos = random.randrange(len(data) + 1)
    return data[:pos] + random.choice(seqs) + data[pos:]


def mut_giant_field(data: bytes) -> bytes:
    text = data.decode("utf-8", errors="ignore")
    lines = text.split("\n")
    size = random.choice([4_000, 20_000, 100_000])
    giant = "a" * size
    targets = [i for i, l in enumerate(lines) if l.startswith(("RECEIPT ", "ACTION ", "SESSION ", "COUNTER "))]
    if not targets:
        return data
    i = random.choice(targets)
    prefix = lines[i].split(" ", 1)[0]
    lines[i] = f"{prefix} {giant}"
    return "\n".join(lines).encode("utf-8")


def mut_dup_line(data: bytes) -> bytes:
    lines = data.decode("utf-8", errors="ignore").split("\n")
    if not lines:
        return data
    i = random.randrange(len(lines))
    lines.insert(i, lines[i])
    return "\n".join(lines).encode("utf-8")


def mut_crlf(data: bytes) -> bytes:
    return data.replace(b"\n", b"\r\n")


def mut_garbage_prefix(data: bytes) -> bytes:
    garbage = bytes(random.randrange(256) for _ in range(random.choice([1, 8, 64])))
    return garbage + data


def mut_extra_garbage_after(data: bytes) -> bytes:
    garbage = bytes(random.randrange(256) for _ in range(random.choice([1, 8, 64, 4000])))
    return data + garbage


def mut_missing_fields(data: bytes) -> bytes:
    lines = [l for l in data.decode("utf-8", errors="ignore").split("\n") if l.strip()]
    if not lines:
        return data
    drop = random.randrange(len(lines))
    del lines[drop]
    return ("\n".join(lines) + "\n\n").encode("utf-8")


REQUEST_MUTATIONS = {
    "byte_flip": mut_byte_flip,
    "truncate": mut_truncate,
    "non_utf8": mut_non_utf8,
    "giant_field": mut_giant_field,
    "dup_line": mut_dup_line,
    "crlf": mut_crlf,
    "garbage_prefix": mut_garbage_prefix,
    "extra_garbage_after": mut_extra_garbage_after,
    "missing_field": mut_missing_fields,
}


MALFORMED_TICKETS = [
    "", " ", "T", "T0", "T-1", "T99999999999999999999999999999999",
    "../../../etc/passwd", "T0" + chr(0) + "T1", "'; DROP TABLE tickets; --",
    " ", "a" * 100_000, "T0" + chr(0xFFFD), "ticket=nested",
    "T0 T1", "T0\nPOLL ticket=T1",
]
MALFORMED_TICKET_BYTES = [
    b"T\xff\xfe", b"\x80" * 8, b"", b"a" * 200_000,
]


def main():
    total = 0
    findings = []
    counts = {}

    work = Path("/tmp/safe7c-fuzz-work")
    work.mkdir(exist_ok=True)
    for old in work.glob("*"):
        try:
            old.unlink()
        except IsADirectoryError:
            pass

    if not GATEWAY_BIN.exists() or not AGENT_TRACE_BIN.exists():
        print("gateway/agent_trace release binaries missing; build first", file=sys.stderr)
        return 2

    def record(class_name, detail, verdict):
        nonlocal total
        total += 1
        counts[class_name] = counts.get(class_name, 0) + 1
        if verdict != "ok":
            findings.append({"class": class_name, "verdict": verdict, **detail})

    receipt = gen_receipt(work, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16, "class1")
    action_hex = last_step_in_hex(receipt)

    # ---------------- class 1: malformed_request ----------------
    fast_bin = make_fast_pass_stub(work)
    d1 = Daemon(work, fast_bin, verify_workers=1, tag="c1")
    try:
        n_per_mutation = 3 if QUICK else 90  # 9 classes * 90 = 810 cases
        for class_name, fn in REQUEST_MUTATIONS.items():
            for i in range(n_per_mutation):
                base = decide_bytes(receipt, action_hex, f"c1-{class_name}-{i}", i + 1)
                mutated = fn(base)
                reply, timed_out, err = send_raw(d1.socket_path, mutated)
                verdict = classify(reply, timed_out, err, d1.alive())
                record(f"malformed_request.{class_name}", {
                    "mutated_len": len(mutated), "reply": (reply or b"")[:300],
                }, verdict)
                if verdict == "FINDING-DAEMON-DIED":
                    break
            if not d1.alive():
                break
        _out, err1 = d1.stop()
        if b"panicked" in err1:
            findings.append({"class": "malformed_request", "verdict": "FINDING-PANIC-STDERR",
                              "stderr": err1[:2000]})
    finally:
        if d1.alive():
            d1.stop()

    # ---------------- class 2: malformed_poll ----------------
    d2 = Daemon(work, fast_bin, verify_workers=1, tag="c2")
    try:
        # A ticket that resolves, to test idempotent re-poll + cross-restart.
        reply, timed_out, err = send_raw(d2.socket_path, decide_bytes(receipt, action_hex, "c2-real", 1))
        verdict = classify(reply, timed_out, err, d2.alive())
        record("malformed_poll.setup_decide", {"reply": (reply or b"")[:300]}, verdict)
        real_ticket = None
        if reply and reply.startswith(b"PENDING ticket="):
            real_ticket = reply[len(b"PENDING ticket="):].strip().decode()

        for i, t in enumerate(MALFORMED_TICKETS * (1 if QUICK else 6)):  # 15*6 = 90
            reply, timed_out, err = send_raw(d2.socket_path, poll_bytes(t))
            verdict = classify(reply, timed_out, err, d2.alive())
            record("malformed_poll.string", {"ticket": repr(t)[:200], "reply": (reply or b"")[:300]}, verdict)

        for i, tb in enumerate(MALFORMED_TICKET_BYTES * (1 if QUICK else 20)):  # 4*20 = 80
            raw = b"POLL ticket=" + tb + b"\n\n"
            reply, timed_out, err = send_raw(d2.socket_path, raw)
            verdict = classify(reply, timed_out, err, d2.alive())
            record("malformed_poll.bytes", {"ticket_bytes": tb[:64], "reply": (reply or b"")[:300]}, verdict)

        # Idempotent re-poll of a real (eventually resolved) ticket: poll it
        # repeatedly and make sure it never reports anything other than
        # PENDING-then-monotonically-the-same-Done-line (never flips back
        # to PENDING once Done, never a different ALLOW/DENY the second
        # time).
        if real_ticket:
            seen_done = None
            for _ in range(3 if QUICK else 40):
                reply, timed_out, err = send_raw(d2.socket_path, poll_bytes(real_ticket))
                verdict = classify(reply, timed_out, err, d2.alive())
                if reply and reply.startswith((b"ALLOW ", b"DENY ")):
                    if seen_done is None:
                        seen_done = reply
                    elif reply != seen_done:
                        verdict = "FINDING-NONDETERMINISTIC-REPOLL"
                record("malformed_poll.repoll_idempotent", {"reply": (reply or b"")[:300]}, verdict)
                time.sleep(0.05)

        # Cross-restart: a ticket minted by a NOW-DEAD daemon must DENY
        # "unknown ticket" on a fresh daemon instance, never hang/PENDING
        # forever.
        old_ticket = real_ticket
        d2.stop()
        d3 = Daemon(work, fast_bin, verify_workers=1, tag="c2b")
        try:
            if old_ticket:
                reply, timed_out, err = send_raw(d3.socket_path, poll_bytes(old_ticket))
                verdict = classify(reply, timed_out, err, d3.alive())
                if verdict == "ok" and not reply.startswith(b"DENY"):
                    verdict = "FINDING-TICKET-SURVIVED-RESTART"
                record("malformed_poll.cross_restart", {"reply": (reply or b"")[:300]}, verdict)
        finally:
            d3.stop()
    finally:
        if d2.alive():
            d2.stop()

    # ---------------- class 3: mass_enqueue_no_poll ----------------
    # Deliberately exceeds MAX_TICKETS (4096, src/main.rs) so a live daemon
    # run actually exercises the eviction path fixed in this branch, not
    # just the Rust-level unit test.
    d4 = Daemon(work, fast_bin, verify_workers=2, tag="c3")
    try:
        n_enqueue = 30 if QUICK else 4700

        def rss_kb(pid):
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
            except Exception:
                return None
            return None

        rss_before = rss_kb(d4.proc.pid)
        checkpoints = []
        for i in range(n_enqueue):
            reply, timed_out, err = send_raw(d4.socket_path,
                                              decide_bytes(receipt, action_hex, f"c3-mass-{i}", 1),
                                              read_timeout=CASE_READ_TIMEOUT_S)
            verdict = classify(reply, timed_out, err, d4.alive())
            record("mass_enqueue_no_poll", {"i": i, "reply": (reply or b"")[:200]}, verdict)
            if verdict == "FINDING-DAEMON-DIED":
                break
            if i in (999, 2499, 4699):
                checkpoints.append((i + 1, rss_kb(d4.proc.pid)))
        rss_after = rss_kb(d4.proc.pid)
        print(f"class3 memory footprint (KB, approximate, not a timing measurement): "
              f"before={rss_before} checkpoints={checkpoints} after_all_enqueued={rss_after}")
    finally:
        if d4.alive():
            d4.stop()

    # ---------------- class 4: concurrent_replay ----------------
    d5 = Daemon(work, fast_bin, verify_workers=1, tag="c4")
    try:
        n_groups = 2 if QUICK else 25
        group_size = 4 if QUICK else 12  # 25*12 = 300 connection attempts
        for g in range(n_groups):
            session = f"c4-group-{g}"
            barrier = threading.Barrier(group_size)
            results = [None] * group_size

            def worker(idx):
                barrier.wait()
                reply, timed_out, err = send_raw(
                    d5.socket_path, decide_bytes(receipt, action_hex, session, 1)
                )
                results[idx] = (reply, timed_out, err)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(group_size)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=CASE_READ_TIMEOUT_S + 2)

            live_count = 0
            deny_count = 0
            bad = 0
            for (reply, timed_out, err) in results:
                verdict = classify(reply, timed_out, err, d5.alive())
                if verdict != "ok":
                    bad += 1
                    continue
                if reply.startswith((b"PENDING ", b"ALLOW ")):
                    live_count += 1
                elif reply.startswith(b"DENY"):
                    deny_count += 1
            group_verdict = "ok"
            if bad:
                group_verdict = "FINDING-BAD-REPLY-IN-GROUP"
            elif live_count != 1:
                group_verdict = "FINDING-DOUBLE-LIVE-OUTCOME" if live_count > 1 else "FINDING-ZERO-LIVE-OUTCOME"
            record("concurrent_replay.group", {
                "group": g, "live_count": live_count, "deny_count": deny_count, "bad": bad,
            }, group_verdict)
            for i in range(group_size):
                # Count each connection attempt as its own case too, for the
                # total-case tally (matches the brief's "N clients submit
                # concurrently" -- each client connection is one case).
                reply, timed_out, err = results[i]
                verdict = classify(reply, timed_out, err, d5.alive())
                record("concurrent_replay.connection", {"group": g, "idx": i}, verdict)
    finally:
        if d5.alive():
            d5.stop()

    # ---------------- class 5: worker_crash mid-verify ----------------
    n_crash_cases = 2 if QUICK else 25
    for i in range(n_crash_cases):
        sleep_secs = random.choice([2, 3, 4])
        slow_bin = make_slow_wrapper(work, sleep_secs)
        d6 = Daemon(work, slow_bin, verify_workers=1, tag=f"c5-{i}")
        try:
            r = gen_receipt(work, f"Q: {i} + {i}\nA: CALC({i} + {i}).\n", 1, 16, f"crash{i}")
            ah = last_step_in_hex(r)
            reply, timed_out, err = send_raw(d6.socket_path, decide_bytes(r, ah, f"c5-{i}", 1))
            verdict = classify(reply, timed_out, err, d6.alive())
            if verdict != "ok" or not reply.startswith(b"PENDING ticket="):
                record("worker_crash.enqueue", {"reply": (reply or b"")[:200]}, "FINDING-ENQUEUE-FAILED" if verdict == "ok" else verdict)
                continue
            ticket = reply[len(b"PENDING ticket="):].strip().decode()

            # Give the worker a brief moment to actually spawn the
            # subprocess before we go hunting for descendants.
            time.sleep(0.2)
            killed_any = False
            for _ in range(20):
                pids = d6.descendant_pids()
                if pids:
                    for pid in pids:
                        try:
                            os.kill(pid, 9)
                            killed_any = True
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(0.1)

            # Poll until resolved or the bound is hit.
            resolved = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < WORKER_CRASH_RESOLVE_BOUND_S:
                reply, timed_out, err = send_raw(d6.socket_path, poll_bytes(ticket))
                verdict = classify(reply, timed_out, err, d6.alive())
                if verdict != "ok":
                    break
                if reply != b"PENDING\n":
                    resolved = reply
                    break
                time.sleep(0.2)

            if verdict != "ok":
                final_verdict = verdict
            elif not killed_any:
                final_verdict = "FINDING-COULD-NOT-LOCATE-VERIFY-SUBPROCESS"
            elif resolved is None:
                final_verdict = "FINDING-HANG-AFTER-WORKER-KILL"
            elif resolved.startswith(b"ALLOW"):
                final_verdict = "FINDING-ALLOW-AFTER-KILLED-VERIFY"
            elif resolved.startswith(b"DENY"):
                final_verdict = "ok"
            else:
                final_verdict = "FINDING-BAD-OUTCOME"
            record("worker_crash", {"resolved": (resolved or b"")[:200], "killed_any": killed_any}, final_verdict)
        finally:
            if d6.alive():
                d6.stop()

    print(f"\nTOTAL CASES: {total}")
    print("Per-class counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"\nFINDINGS: {len(findings)}")
    for f in findings:
        print("----")
        for k, v in f.items():
            print(f"{k}: {v}")

    out_path = Path("/tmp/safe7c-fuzz-findings.txt")
    with open(out_path, "w") as fh:
        fh.write(f"TOTAL CASES: {total}\n")
        fh.write("Per-class counts:\n")
        for k, v in sorted(counts.items()):
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nFINDINGS: {len(findings)}\n")
        for f in findings:
            fh.write("----\n")
            for k, v in f.items():
                fh.write(f"{k}: {v}\n")
    print(f"\nwrote {out_path}")
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
