# AEFINITY OS v0.1 — status

*Integration branch `os/aefinity-os-v0.1` (phases 0 + 1a + 1b + 2 merged).
Written 2026-09-01. Build contract: `program/AEFINITY_OS.md`. Hard rules: `CLAUDE.md` A–D.*

This file says what the OS **does**, on what evidence, and what it has not been
shown to do. It is deliberately short on adjectives. Every gate below runs under
QEMU/TCG, so per **Rule A** it is a correctness gate and nothing else: the gates
assert exit codes and record *structure*. No gate asserts a timing value, and no
timing value from any of them may be quoted anywhere.

---

## 1. What is in v0.1

| phase | branch merged | what it added |
|---|---|---|
| 0 JOB | `os/p0-job` (via `os/p1a-net`) | `JOB.TXT` autorun, `RESULT.TXT`, budget accounting across prefill and decode, firmware watchdog, `ResetSystem`, `RESULT.WIP` in-progress marker |
| 1a NET | `os/p1a-net` | `net/mod.rs`: `EFI_SIMPLE_NETWORK` → `smoltcp::phy::Device`, static IP and DHCP, TCP connect/send/recv/close, `NETCHECK` directive |
| 1b REPORT | `os/p1b-report` | `net/http.rs`: HTTP/1.1 `POST` of the exact `RESULT.TXT` bytes to `REPORT <url>` |
| 2 RESIDENT | `os/p2-resident` | `server.rs`: `MODE resident` TCP listener speaking the spec §4 line protocol, per-job watchdog re-arm, `REBOOT`/`HALT` |

Phase 3 FLEET is host-side and manual on iron; it is not in this branch.

The unikernel is `no_std` + `alloc`. Dependencies added for the OS work are
`smoltcp 0.12.0` with `default-features = false` and exactly the seven features
spec §5 names. Everything stays inside boot services — the OS never calls
`ExitBootServices`, because SNP, the file protocol, the watchdog and
`ResetSystem` all need them.

### The no-`JOB.TXT` path is unchanged

A stick with no `JOB.TXT` boots exactly as it did before this work. `main.rs`
gained three lines across all four phases — one `mod server;` declaration — and
the single `job::load` / `job::dispatch` hook that phase 0 placed *before* the
MECH diagnostic block. `job::load` returning `None` is the old path. That is
what `boot-test` re-asserts.

---

## 2. Exact commands

All of them, in order, from the repo root. This box has 6 GB of RAM and shares
8 cores with other agents, so every cargo and every QEMU invocation is wrapped —
this is not optional, it is how the gates were actually run:

```bash
W='flock /tmp/aefinity-os.lock nice -n 10 env CARGO_BUILD_JOBS=4'

# host-side format gate (all six crates)
$W scripts/devloop.sh fmt

# production build + staging gate
$W ./aegis-uefi/build_hardfloat.sh
$W scripts/check-efi-simd.sh aegis-uefi/target/x86_64-uefi-hardfloat/release/aegis-uefi.efi

# the five QEMU gates, ONE AT A TIME (the flock enforces it)
cd xtask
$W cargo run --release -- boot-test        # no JOB.TXT — the unchanged path
$W cargo run --release -- job-test         # phase 0
$W cargo run --release -- net-test         # phase 1a
$W cargo run --release -- os-test          # phase 1b
$W cargo run --release -- resident-test    # phase 2
$W cargo run --release -- job-budget-test  # phase 0 budget regression
```

Optional flags: `--ci` (build only, no boot), `--debug` (keep intermediates),
`--dhcp` (`net-test` only — stages `NET dhcp` instead of `NET static` and
relaxes the address assertion), `--pcap` (`net-test` only — writes
`target/net.pcap`).

These gates are slow because TCG is slow, not because anything is wrong.
`boot-test` runs the MECH diagnostic block on every boot; end-to-end it is tens
of minutes. `JOB_BUDGET_S = 900` with a `JOB_TIMEOUT_S = 1200` deadline (and
`RESIDENT_TIMEOUT_S = 2700`) are sized for *this shared box under emulation* and
say nothing about how long the work takes on hardware.

---

## 3. What the gates actually assert

Structure and exit codes. Nothing else.

- **`boot-test`** — no `JOB.TXT` is staged. PASS = QEMU's `isa-debug-exit`
  reports 33. This is the regression gate on the untouched boot path.
- **`job-test`** — stages `BUDGET 900 / MODE oneshot / TOKENS 16 / PROMPT The
  capital of France is / BENCH 8 / AFTER reset`, boots with `-no-reboot`.
  PASS = QEMU exits (the guest called `ResetSystem`) **and** the `RESULT.TXT`
  that vvfat left in `target/esp/` parses with `aefinity_os=0.1`, `verdict=OK`,
  `jobs=2`, `env=vm`, **and** `BOOTLOG.TXT` carries the guest's own
  `JOB: RESULT.WIP cleared=true (open=… dir=… retried=…)` line.
- **`net-test`** — adds `-netdev user,id=n0 -device virtio-net-pci,netdev=n0
  -device virtio-rng-pci`, opens a host listener on an ephemeral 127.0.0.1
  port, stages `NET static 10.0.2.15/24 10.0.2.2` and `NETCHECK 10.0.2.2:<port>`.
  PASS = the harness receives `HELLO <17-char MAC>\n` and `RESULT.TXT` carries
  `job.1.ok=true` with a matching `mac=`/`ip=`.
- **`os-test`** — same NIC, plus `REPORT http://10.0.2.2:<port>/aefinity/result`
  against a minimal HTTP/1.1 server in the harness. PASS = the received POST
  body parses with `aefinity_os=0.1`, `env=vm`, `jobs=2`, `verdict=OK`, and QEMU
  exits. Per spec §6 the on-disk record is **not** required to carry `report=`:
  it was written before the POST, and the OS deliberately does not rewrite it.
- **`resident-test`** — `MODE resident / NET static / LISTEN 4242`, booted with
  `hostfwd=tcp:127.0.0.1:<port>-:4242`. The harness is the client: it retries
  until the `AEFINITY-OS 0.1 READY` banner, then `PING`→`PONG`, a two-directive
  `JOB`, a second `JOB` on the same connection, a garbage line (must come back
  `ERR unknown`), a 70 KiB line with no newline in it (must come back
  `ERR too-large`, be dropped, and the listener must come back), and `REBOOT`
  (must come back `BYE`, and QEMU must exit).
- **`job-budget-test`** — a 4096-byte `PROMPT` with `TOKENS 1024` and
  `BUDGET 5`: a job that provably cannot finish prefill inside its budget.
  PASS = the guest still resets **and** leaves `verdict=FAIL budget` on the
  volume. Before the in-prefill deadline check this case left no record at all —
  the firmware watchdog reset the box mid-prefill and the volume stayed silent.

---

## 4. Gate results

All six gates below were run on **penguin** (this dev machine: ChromeOS Crostini
container, Debian 13, 6 GB RAM, 8 shared cores) on **2026-09-01**, from the
integration branch `os/aefinity-os-v0.1`, each one alone under
`flock /tmp/aefinity-os.lock`. Every one under `-machine q35,accel=tcg -cpu max
-m 2048` against `/usr/share/OVMF/OVMF_CODE_4M.fd`.

| gate | result | what it proves |
|---|---|---|
| `boot-test` | **PASS** — QEMU exited 33 | the no-`JOB.TXT` boot path is unchanged by all four phases |
| `job-test` | **PASS** | `JOB.TXT` autorun → `RESULT.TXT` (`verdict=OK`, `jobs=2`, `env=vm`) → `ResetSystem` |
| `net-test` | **PASS** | the guest's own smoltcp stack over SNP reached the host and named its NIC |
| `os-test` | **PASS** | the guest POSTed its `RESULT.TXT` bytes to a host HTTP server, then reset |
| `resident-test` | **PASS** | the §4 line protocol end to end: `READY`, `PING`/`PONG`, `BUSY`, two jobs on one connection, `ERR unknown`, `ERR too-large` + relisten, `REBOOT` |
| `job-budget-test` | **PASS** | a job that cannot finish prefill inside its `BUDGET` still leaves `verdict=FAIL budget` on the volume |
| `scripts/devloop.sh fmt` | **PASS** — all six crates | |
| `build_hardfloat.sh` + `check-efi-simd.sh` | **PASS** — `xmm=9946 ymm=504 vfmadd=210` | the production `.efi` is hard-float, not the soft-float regression |

The instruction-census numbers above are the staging gate's own output — a count
of opcodes in a binary, not a measurement of anything. There is no timing claim
in this document.

### `boot-test`

```
== PASS == QEMU exited 33 (isa-debug-exit success signal)
```

### `job-test`

```
   ok   aefinity_os=0.1
   ok   verdict=OK
   ok   jobs=2
   ok   env=vm
   ok   JOB: RESULT.WIP cleared=true (open=absent dir=absent retried=false)
   note target/esp/result.wip is still in the host mirror.
        QEMU `fat:rw:` commits guest writes but not the unlink;
        the guest's BOOTLOG line above is the statement about the volume.

== PASS == RESULT.TXT satisfies the phase-0 contract.
```

### `net-test`

```
   ok   aefinity_os=0.1
   ok   env=vm
   ok   jobs=1
   ok   job.1.kind=netcheck
   ok   job.1.ok=true
   ok   verdict=OK
   ok   ip=10.0.2.15
   ok   net=static
   ok   mac=52:54:00:12:34:56 (record == wire)
   ok   JOB: RESULT.WIP cleared=true (open=absent dir=absent retried=false)

== PASS == the guest's own TCP/IP stack reached the host and named its NIC.
```

### `os-test`

```
   note POST from 127.0.0.1:40568
---- received POST body (/aefinity/result) ----
   ok   aefinity_os=0.1
   ok   env=vm
   ok   jobs=2
   ok   verdict=OK
   ok   JOB: RESULT.WIP cleared=true (open=absent dir=absent retried=false)

== PASS == the guest POSTed RESULT.TXT to the host collector and reset.
```

### `resident-test`

```
   attempt 2: AEFINITY-OS 0.1 READY env=vm cpu=QEMU TCG CPU version 2.5+
   ok   PING -> PONG
   ok   second connection -> BUSY
   ok   JOB 1 answered
   ok   JOB 2 answered — the server survived a job
   ok   garbage -> ERR unknown
   ok   over-long line -> ERR too-large
   ok   connection dropped after ERR too-large (the guest closed the connection)
   ok   the server took a new connection afterwards: AEFINITY-OS 0.1 READY env=vm …
   ok   REBOOT -> BYE
   ok   job 1 verdict=OK / jobs=2 / env=vm
   ok   job 2 verdict=OK / jobs=1 / env=vm
   ok   the volume holds the last job's record (jobs=1)
   ok   QEMU exited 0 (guest ResetSystem under -no-reboot)

== PASS == the resident server served two jobs, refused garbage, and reset.
```

### `job-budget-test`

```
   ok   aefinity_os=0.1
   ok   env=vm
   ok   verdict=FAIL budget
   ok   JOB: RESULT.WIP cleared=true (open=absent dir=absent retried=false)

== PASS == the budget stopped the job and the record says so.
```

### One thing worth noticing, and not over-reading

`job-test`, `os-test` and `resident-test` all ran the same `PROMPT The capital
of France is` / `BENCH 8` pair, in three different harnesses across three
separate boots, and every one of them produced `job.2.digest=5c473f57998d5752`
for the bench step (and `86fe3a7315bf7fbb` for the prompt step wherever
`TOKENS 16` was in force). That is the `digest` key doing the job spec §3 gives
it — a witness of *what was generated* — and it is the same-machine,
same-binary case. The cross-machine claim it is built for needs a second
machine and has not been made.


---

## 5. Untested on iron

**No claim in this document has been reproduced on physical hardware.** First
light with a flashed stick is Justin's step and is not claimed here. The
following are the specific things emulation cannot stand in for:

- **Any performance figure.** There is none in this branch and there must not
  be one until a named physical box produces it (Rules A and B). `RESULT.TXT`
  carries `env=vm` under QEMU precisely so a record cannot be mistaken later.
- **The NIC.** The gates exercise `virtio-net-pci` behind OVMF's `VirtioNetDxe`.
  A real box's SNP is a different driver over different silicon; MTU, multicast
  filter setup, receive-filter defaults and the transmit-completion path are all
  places where a real adapter can differ.
- **DHCP against a real server.** `net-test --dhcp` exercises slirp's built-in
  DHCP server, which is not a real DHCP server on a real LAN.
- **`ResetSystem` and the firmware watchdog.** Under `-no-reboot` a reset is
  observed as a QEMU exit. A real firmware's cold reset, and a real
  `SetWatchdogTimer` actually firing, are untested.
- **The FAT32 write path on a real stick.** Under QEMU it is vvfat over a host
  directory (see the vvfat note in §6).
- **`AFTER halt` / protocol `HALT`.** The gates use `reset` and `REBOOT`. The
  park path is compiled and reachable but not asserted by any gate.
- **Two boxes.** The `job.N.digest` cross-machine comparison — the whole point
  of the digest — needs two machines. It has one.

---

## 6. Known gaps and honest caveats

1. **`REPORT` is oneshot-only in v0.1.** A `REPORT <url>` line inside a
   socket-delivered resident job body is *dropped and named in `BOOTLOG.TXT*`
   (`RESIDENT: JOB ignored REPORT … — oneshot-only in v0.1`), the same way
   spec §4 already handles `MODE`/`LISTEN`/`NET`. A resident job's record goes
   back down the socket the client is holding, and the POST client would need
   the NIC the listener has opened exclusively. It is refused out loud rather
   than accepted and quietly not done.
2. **A resident job cannot `NETCHECK`.** Same cause: the listener holds the SNP
   handle exclusively, so `run_job` cannot bring up a second `Net`. The
   directive fails with `no nic` and says so in the record. It is reported, not
   silently wrong.
3. **`RESULT.WIP` residue in the harness's host mirror.** After a job that
   *did* finish, `target/esp/result.wip` can still be present on the host. That
   is QEMU's `fat:rw:` (vvfat) committing the guest's writes back to the host
   directory and not committing the unlink; the guest confirms the delete off
   the FAT directory it actually walks, from two independent probes, and the
   gates assert the guest's own `cleared=true` BOOTLOG line rather than an
   `ls` of the host directory. A stick has no host mirror.
4. **HTTP client is minimal by design** (spec §5): HTTP/1.1, `Connection:
   close`, **no TLS**, no redirects, no chunked request bodies, and **no
   resolver** — the host in a `REPORT` url must be a dotted IPv4 address, and a
   hostname comes back `REPORT: fail no-dns`. The collector is expected to be
   on a trusted lab network at a known address.
5. **One client at a time** in resident mode (spec §4). A second connection
   gets `BUSY\n` and is closed.
6. **Clippy is a stale ratchet.** `scripts/devloop.sh clippy` fails identically
   on `main` (115 findings against a baseline of 48) and did so before this
   work. Nothing here was rebaselined; the merged files add no new warnings.
   Paying that debt down is its own task.
7. **`aegis-uefi` still emits pre-existing build warnings** (unnecessary
   `unsafe` blocks in `cpu.rs`, two non-snake-case locals in the MECH block).
   They predate this branch and are untouched.
8. **The gate timeouts are box-shaped, not spec-shaped.** `JOB_BUDGET_S`,
   `JOB_TIMEOUT_S`, `RESIDENT_TIMEOUT_S` and friends are sized so this shared
   6 GB box finishes under TCG. They are harness constants and carry no meaning
   about the workload.

---

## 7. Host side

Host tooling for §7 of the spec — `cm-os-collector`, `cm-os-job` (including
`--matrix`), and the `cm-uefi-qemu` / `cm-uefi-run` wiring — lives in the
`claudius-maximus` repo on branch `cm/aefinity-os-host`, with a stdlib-only
`python3 -m unittest tests.test_os_host` suite. It is reviewed and tested
against fake servers; it has never spoken to a real booted stick.

---

## 8. Critic fixes 2026-09-01

An adversarial review of this branch (`docs/AEFINITY_OS_CRITIC_v0.1.md`,
verbatim) returned **SHIP-WITH-FIXES**: merge as a QEMU-verified harness, but
do not flash a stick or expose a resident box on a LAN until fixes 1–4 land.
All five landed. Nothing below was verified on iron; every gate is still TCG,
still correctness-only (Rule A).

| # | finding | what changed |
|---|---|---|
| 1 | TX frames could be DMA'd from above 4 GB — `SnpTxToken::consume` boxed the frame on the global heap, and `allocate_huge_pages` deliberately does not cap the large chunks | `net/mod.rs`: `DmaPool`, one `MaxAddress(0xFFFF_FFFF)` `LOADER_DATA` allocation of 9 frame-sized slots (8 transmit + 1 receive), free list, slots reclaimed by pointer match on `get_recycled_transmit_buffer_status`. Receive stages into the pinned slot too, because a driver may serve `Receive()` by DMA into the caller's buffer. No slot = frame dropped, which is what `MAX_PENDING_TX` already meant. |
| 2 | `REPORT` could blow the watchdog margin — `http::post` gave connect, send, recv and both closes a fresh `timeout_ms` each, ~120 s against 60 s of margin | `net/http.rs`: one deadline taken before the connect, each phase gets what is left; `Net::now_ms` exposed for it. `job::REPORT_TIMEOUT_MS`'s comment now carries the arithmetic — 30 s `REPORT` + 4 s `settle_volume` = ~34 s of the 60 s margin. |
| 3 | Resident mode disarmed the watchdog before the FAT write | `server.rs`: the watchdog is re-armed at `RECORD_WATCHDOG_S` (120 s) before `write_result_txt` and disarmed only after the `RESULT` send and both BOOTLOG lines. Idle still runs unguarded, per §4. |
| 4 | The collector was an unauthenticated write endpoint on every interface | `claudius-maximus` `cm/aefinity-os-host`: `--bind` defaults to `127.0.0.1`, `--token`/`CM_COLLECTOR_TOKEN` gates POST with `X-Aefinity-Token` (401 otherwise), `cpu_brand`/`verdict`/`run_id` stripped of control characters and pipes before `LOG.md` and `inbox.txt`, bad `Content-Length` is 400, the unit passes `--bind ${CM_COLLECTOR_BIND}`, and `docs/UEFI-REMOTE-LANE.md` gains "Exposing the collector on a LAN". Here: a `TOKEN <value>` line in `JOB.TXT` (`Job::token`) is sent as `X-Aefinity-Token` on the `REPORT` POST. |
| 5 | Two `unsafe` blocks with no `// SAFETY:`, and a wrong `hlt` comment | `job.rs`: both rdtsc sites carry one. `server.rs`: `park_no_net` no longer claims `hlt` "returns at once" with interrupts masked — it halts until an NMI or SMI, which is correct for a park loop, and the comment says why. |

### Spec deltas this introduced

- **`TOKEN <value>`** is a new `JOB.TXT` directive, not in §2's table. A value
  containing a control character is refused at parse time and logged, because
  the value is pasted into an HTTP header.

### What remains — nothing here has run on iron

The critic's ordered list, minus what landed:

1. **Hardware first light.** No box has booted this. SNP against a real NIC,
   real-LAN DHCP, a real `SetWatchdogTimer` firing, a real cold reset, FAT32 on
   a stick — all unobserved. Fixes 1 and 3 in particular exist because QEMU at
   `-m 2048` and OVMF's cooperative firmware **cannot** exhibit the failures
   they prevent: the fixes are argued from the UEFI spec and the hardware's
   addressing limits, not from a run. Log first light to
   `docs/hardware_logs/` (Rule C: new files only).
2. **The cross-machine `job.N.digest` comparison** on a second box — the only
   claim this design exists to make — is unmade.
3. **Phase-4 auth on the §4 protocol.** The resident listener is still
   unauthenticated by design: any host that can reach `LISTEN` gets `READY`,
   can run arbitrary prompts, and can `HALT` the box into a state needing a
   physical power cycle. `TOKEN` currently authenticates the box *to* the
   collector, not a client to the box. The phase-4 item is an `AUTH <token>`
   verb honoured before any other command, plus a source-address allowlist.
4. **Unexamined firmware behaviours** the critic named and this pass did not
   touch: `open_protocol_exclusive` forcibly disconnecting a real firmware's
   own network stack and never restoring it; `find_handles().first()` picking
   an arbitrary NIC on a multi-NIC box; a resident DHCP lease that expires
   with no re-acquire; `write_named` deleting before creating, so power loss
   between the two leaves neither record.
5. **The clippy ratchet is still stale** (115 against a baseline of 48, on
   `main` as much as here). Not rebaselined by this pass; the counts are
   unchanged by it.
6. **"OS" is still the name.** The critic's §5 stands: a hostile reviewer will
   call this a firmware application — no scheduler, no processes, no memory
   protection, one thread, and every I/O through boot services it never exits.
   The defensible description is a single-purpose appliance image with its own
   TCP/IP stack, a job protocol and a remote lifecycle. Renaming it in
   reviewer-facing prose is an open decision, not a code change.

---

## 9. Phase 4 — FILES

Branch `os/p4-files`, built against
`program/AEFINITY_OS_FLEET_DESIGN.md` (§1 protocol, §3 record, §4 modules,
§4.1 `RELOAD`, §4.2 transport, §7 gate, §8 iron safety, §10 sequencing).
Everything below ran under QEMU/TCG on `penguin` and **nothing here has run on
iron.**

### What phase 4 adds

A **file plane** and the fleet verbs, on top of v0.1's resident protocol
without changing any of it:

```
AUTH <64hex>            LS                     STAT <NAME>
SHA <NAME>              GET <NAME>             PUT <NAME> <len> <64hex>
RM <NAME>               RELOAD                 HEALTH        RUNID <id>
```

plus the `DATA <len> <sha16>\n<bytes>END\n` frame (§1.1) — the only binary on
the wire, no base64.

New modules: `aegis-uefi/src/files.rs` (the volume) and
`aegis-uefi/src/reload.rs` (`EngineSlot`, the artifact pointer). `server.rs`
dispatches; `net/mod.rs` gains `tcp_recv_exact`, `tcp_send_slice` and §4.2's
listener buffers; `job.rs` gains §3's appended record keys and §3.1's
`merge_key`; `main.rs` changes only inside its one existing job hook.

### The three invariants, and where they are enforced

- **`AUTH` gates every verb but `PING`/`AUTH`** when `JOB.TXT` carries a
  `TOKEN` — **reads included**, because `GET MODEL.SAF` is exfiltration and a
  `HEALTH` reply publishes artifact digests. The check is one arm of the
  dispatch loop, so a verb added later cannot forget it. This closes item 3 of
  §8's "what remains" list above. Absent a `TOKEN` the box is open, which is
  exactly v0.1 behaviour — which is why no v0.1 gate changed.
- **No verb returns a partial success** (§1.3). `PUT` never opens the target:
  stage to `STAGE.PRT`, check the trailer, re-open and stream a sha256 over
  the **readback**, and only then commit. Every abort path in §1.4's table
  deletes the stage and leaves the target untouched.
- **No frame is ever truncated on `GET`** (§1.4). The file is read twice: the
  header's `sha16` comes from a full pass *before* the header goes out, so a
  header that arrives is a promise the bytes were readable once. After that the
  server cannot retract, and the only honest failure left is to close.

### What works under QEMU

`cargo xtask files-test` (design §7), PASS, 2 m 46 s wall:

```
ok   banner AEFINITY-OS 0.1 READY env=vm cpu=QEMU TCG CPU version 2.5+ caps=files
ok   GET BOOTLOG.TXT  ->  ERR auth            (TOKEN set, not yet authenticated)
ok   AUTH <wrong>  ->  ERR auth
ok   AUTH <token>  ->  OK
ok   LS  ->  LS 5 ok  [MODEL.SAF EMBED.BIN VOCAB.BIN JOB.TXT BOOTLOG.TXT]
ok   STAT MODEL.SAF  ->  STAT MODEL.SAF 2797632
ok   SHA MODEL.SAF matches the host's digest of the staged file
ok   GET BOOTLOG.TXT  ->  DATA frame, header sha16 matches the payload
ok   PUT TEST.BIN (256 KiB)  ->  OK …;  GET TEST.BIN byte-identical
ok   PUT BIG.BIN (64 MiB)  ->  OK …;   SHA BIG.BIN matches what was sent
ok   PUT with a wrong digest  ->  ERR digest-mismatch, TEST.BIN unchanged
ok   PUT with a wrong trailer ->  ERR bad-frame
ok   PUT ../X  ->  ERR bad-name;  32-byte name  ->  ERR bad-name
ok   PUT BOOTLOG.TXT  ->  ERR protected
ok   PUT of a 3 GiB declared length  ->  ERR bad-len (before any byte crosses)
ok   LS free of strays;  HEALTH … parts=0 … env=vm
ok   RM TEST.BIN;  STAT TEST.BIN  ->  ERR not-found
ok   RUNID NEW → job → RUNID REPLAY returns the cached record, replay=true
ok   RELOAD  ->  OK reload model=… embed=… vocab=…; a job after it still runs
ok   REBOOT  ->  BYE;  QEMU exited 0
```

Regression, same tree: `resident-test` PASS (4 m 16 s), `job-test` PASS
(3 m 45 s). `scripts/devloop.sh fmt` clean across all six crates.
`./aegis-uefi/build_hardfloat.sh` + `scripts/check-efi-simd.sh` PASS
(`ymm=504 vfmadd=210`). Clippy is unchanged against `os/aefinity-os-v0.1`:
aegis-uefi **115 → 115**, xtask **0 → 0** (the ratchet baseline of 48 is still
stale, as §8 item 5 records; the *count* is what this phase held).

`boot-test` was **not run** in this pass — it takes ~28 minutes because of the
MECH block and phase 4 does not touch the no-`JOB.TXT` path. `git diff` over
`main.rs` is three hunks: two `mod` lines and the inside of the existing job
hook. Someone should still run it before the merge.

Every assertion above is on the guest's own `LS`/`SHA`/`STAT`, never the host
mirror: QEMU `fat:rw:` commits a guest write but **not** a guest unlink, so
the staged directory cannot answer "is it gone".

### The `RELOAD` hazard statement

`RELOAD` is the one genuinely dangerous verb, and it is dangerous in a way the
gate cannot show.

The engine borrows the three slabs allocated at STAGE 3 for the server's whole
life. Overwriting a slab under a live engine would leave a box reporting a
fresh `model_sha` while still inferring against layout state derived from the
old bytes — a wrong answer wearing a correct-looking provenance line, which is
the worst failure this program can ship. So §4.1's ownership change:
`EngineSlot` **owns** the engine as an `Option`, and `RELOAD`

1. refuses while a `STAGE.PRT` exists (`ERR busy-file`);
2. checks every artifact against its slab capacity **first** — bigger is
   `ERR reload-size` with **nothing touched**, and growing a slab means a
   `REBOOT`, because a re-allocation attempt on a fragmented map is
   `main.rs`'s `STAGE 3 FAILED: contiguous alloc` in ring 0;
3. only then drops the engine, refills the slabs (watchdog re-armed per chunk)
   and rebuilds;
4. on a failed rebuild **cold-resets the box**, spending `BootNext` and
   landing in Debian. It never returns to serving.

**The hazard.** Step 3 has no undo. Between the drop and a successful rebuild
the box holds bytes it cannot describe, and the only exit is the reset in step
4. Under QEMU the gate exercises the *success* path only — it reloads the same
three files and asserts the same three digests come back — so the failure path
is **argued, not observed**: no run has yet dropped an engine and failed to
rebuild it. On iron that reset costs a boot cycle and a `BootNext`, and a box
whose Debian side is not reachable is then a box that needs hands. Two
consequences for operators:

- `PUT MODEL.SAF` + `REBOOT` — the already-tested boot loader — stays the
  **default** path. `RELOAD` is an opt-in fast path for iteration.
- Do not `RELOAD` a box you cannot power-cycle or reach over ssh on its Debian
  side. `BOXES.md` saying `power-cycle: human` is the relevant line.

### Deliberate deviations from the design, and why

| design says | this build | why |
|---|---|---|
| banner `caps=files,lab` | `caps=files` | phase 5 has not landed; a box advertising `lab` would be telling a scheduler it can take `EVAL` work it will answer `ERR unknown` to. One constant changes in phase 5. |
| §1.4: after `REPLAY` the client sends `JOB <len>\n<body>END\n` | the v0.1 `JOB\n<body>\nEND\n` framing, drained and discarded | §1 says `JOB` is unchanged from v0.1, so §1.4's `<len>` form is read as a slip. One code path reads `JOB` in every case and the stream is never left half-framed, which is what §1.4 is actually asking for. |
| §1.2 shows no `RUNNING` on a replay | `RUNNING` is sent on the replay path too | a controller retrying blind after a dead TCP is exactly the client that must not need two code paths; it was already told `REPLAY` by the verb. `replay=true` in the record is where the honesty about not re-running lives. |
| §1.3 protects `BOOTLOG.TXT`, `RESULT.TXT`, `RESULT.WIP`, `STAGE.PRT` | those **plus `BOOTX64.EFI` and `CURRENT.TXT`** | a `PUT` of the loader interrupted between the commit's delete and its rename leaves a box that will not boot and that nobody can dial to fix. §9 already leaves large-file provisioning to Debian; this makes the loader part of that rule. `CURRENT.TXT` is the artifact pointer the OS writes, so a client writing it could point boot at unverified bytes. |
| §1.3's slug list includes `exists` | no producer | no verb here refuses a name for already being taken — `PUT` replaces and `RM` does not create. Left out rather than shipped as a variant nothing can build. |
| §3.1 defines `<step-input>` for `cpuid`/`mech`/`membw`/`verify`/`eval` | phase 4 also defines it for `prompt`/`bench`/`netcheck` | those three are the kinds that exist today and §3.1's table does not cover them. `prompt` → the prompt text, `bench` → the token count, `netcheck` → the target. Empty would have been simpler and wrong: two different prompts would then share a `merge_key`, which is the one thing the key exists to prevent. The five lab kinds land in phase 5 unchanged. |

### Untested on iron — the phase-4 list

Everything in §5 above still stands. Phase 4 adds:

1. **The 1.83 GB case.** The gate moves 64 MiB. §4.2's window arithmetic and
   §8's claim that a full-size readback can outlast a single watchdog window
   are both about a file 28× larger, on a USB FAT32 stick rather than QEMU
   vvfat. Neither has been observed.
2. **The artifact pointer swap** (§8). `PUT MODEL.SAF` writes the inactive half
   of an A/B pair and rewrites `CURRENT.TXT`; boot and `RELOAD` read it and
   fall back to the canonical name when it is absent, unparsable, or points at
   a file that is not there. **No gate exercises this** — `files-test` PUTs
   ordinary files, because a 1.8 GB PUT is not a gate on a 6 GB box — so the
   swap is code that has been reasoned about and compiled, not code that has
   run. It is the highest-value thing to exercise on first light.
3. **`ERR no-space` and `ERR io` on a full volume.** §8 says the UEFI
   `FileSystemInfo` free-space number is advisory on vendor FAT drivers, so
   exhaustion may surface as a short write instead. `no-space` is best effort;
   `io` is the guarantee. Neither has been provoked.
4. **The `RELOAD` failure path**, above.
5. **`XFER_STALL_MS` (no progress for 30 s ⇒ close, stage deleted, and the box
   does **not** reboot).** Reasoned from §1.4, never provoked: the harness is a
   cooperative client on loopback.
6. **`sweep_parts` at boot.** The code deletes a stale `STAGE.PRT` and logs it,
   but no run has yet been interrupted mid-`PUT` and rebooted, so `parts=1` has
   only ever been observed as `parts=0`.
7. **The `RUNID` ring across a reboot.** The ring is RAM only by design: a
   reset empties it and a re-issued id then answers `NEW` and runs a second
   time. At-most-once holds within one box uptime, nothing more. `HEALTH up=`
   going backwards is how a scheduler is supposed to notice; that host-side
   rule is phase 6 and does not exist yet.
