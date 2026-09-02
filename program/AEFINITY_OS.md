# AEFINITY OS — the headless, AI-centred OS for AI-driven development and lab testing

*Spec v0.1 — 2026-08-31. This is the build contract for the `os/*` branches.
Project owner: Justin B. Thompson. Builder/operator: Claudius Maximus.*

## 0. One paragraph
Aefinity OS is the A.L.I.C.E. unikernel grown into a **resident, remotely driven
lab worker**: it boots from firmware with no operating system, brings up the
NIC through `EFI_SIMPLE_NETWORK` and its **own** Rust TCP/IP stack (smoltcp),
runs jobs handed to it by file or by socket, writes a machine-readable
`RESULT.TXT`, reports it over the network, and resets or waits for the next job.
Debian on the same disk is demoted to a **recovery partition** (see
`claudius-maximus/docs/UEFI-REMOTE-LANE.md`). Every number it produces carries
`env=iron|vm` so Rule A is enforced by the artifact, not by discipline.

This supersedes nothing in `program/AEGIS_OS_DESIGN.md` (Track U/L, FABLE-0
gates); it is the **worker body** that lane needs and it lands first because it
is harness work the model gates do not block.

## 1. Phases (each phase = one verified, mergeable branch)
| phase | branch | delivers | gate (all under `cargo xtask …`, QEMU/OVMF, correctness only) |
|---|---|---|---|
| **0 JOB** | `os/p0-job` | `JOB.TXT` autorun, `RESULT.TXT`, budget, firmware watchdog, `ResetSystem` | `boot-test` still 33; new `job-test`: JOB.TXT with `PROMPT`/`BENCH`, RESULT.TXT appears, guest resets (`-no-reboot` ⇒ QEMU exits 0) |
| **1a NET** | `os/p1a-net` | `net/` module: SNP device ↔ smoltcp, static IP from JOB.TXT and DHCP, ARP/ICMP/TCP/UDP working | `net-test`: guest opens TCP to host `10.0.2.2:<port>` (QEMU user net) and sends `HELLO <mac>\n`; harness asserts receipt |
| **1b REPORT** | `os/p1b-report` | HTTP/1.1 POST of `RESULT.TXT` to `REPORT <url>`; `BOOTLOG` records `REPORT ok/fail` | `os-test`: full oneshot job with `REPORT`; harness HTTP server receives the body; body parses; guest resets |
| **2 RESIDENT** | `os/p2-resident` | `MODE resident`: TCP listener, line protocol §4, per-job watchdog re-arm, `REBOOT`/`HALT` | `resident-test`: harness connects via hostfwd, sends a JOB, receives RESULT, sends REBOOT, guest resets |
| **3 FLEET** | host side | `cm-os-job --matrix` over `BOXES.md`, cross-CPU `RESULT.TXT` diff | manual on iron |

Host-side tooling (collector, job client, QEMU harness flags, install hook)
lives in `claudius-maximus` and is built in parallel (§6).

## 2. JOB.TXT — the job description (input)
ASCII, one directive per line, `KEY value`, CRLF or LF, `#` starts a comment.
Unknown keys are logged to `BOOTLOG.TXT` and ignored. Read from the boot volume
root by the unikernel **after** the engine is loaded and **before** the
interactive console; if absent, behaviour is exactly as today (interactive).

> **Erratum, 2026-09-01 (phase 0 as built).** v0.1 said "immediately before the
> interactive console", which in `aegis-uefi/src/main.rs` would place the hook
> *after* the unconditional MECH diagnostic block. It is placed **before MECH**
> instead. Two reasons: (1) a box that was handed a job should do the job — MECH
> is a hands-off experiment that generates ~100 tokens across nine runs and is
> charged to nobody's job, and in resident mode (phase 2) it would sit in every
> reboot-and-serve cycle; (2) under QEMU/TCG on the dev box MECH alone runs for
> tens of minutes, so behind it no job could meet the §6 `job-test` deadline.
> It is still ONE hook, still after the engine is loaded and before the
> interactive console. `JOB.TXT` absent ⇒ `job::load` returns `None` and MECH,
> the `qemu-test` block and the console are byte-for-byte what they were, which
> is what `cargo xtask boot-test` still exiting 33 checks.

```
BUDGET 240                 # seconds for the whole job (default 300). Watchdog armed at BUDGET+60.
                           # Enforced before each step AND at coarse checkpoints inside the
                           # engine in BOTH phases (prefill and decode), so a long PROMPT
                           # with a small BUDGET still yields a RESULT.TXT with
                           # verdict=FAIL budget rather than a watchdog reset and silence.
MODE oneshot               # oneshot (default) | resident
NET static 10.0.2.15/24 10.0.2.2     # or: NET dhcp   (default: dhcp; falls back to static if given)
REPORT http://10.0.2.2:8787/aefinity/result   # optional. POST RESULT.TXT here before AFTER.
LISTEN 4242                # resident mode TCP port (default 4242)
TOKENS 64                  # max new tokens for PROMPT (default 64, hard cap 1024)
PROMPT The capital of France is      # generate; may repeat; run in order
BENCH 64                   # generate N tokens from the fixed bench prompt; records tps
AFTER reset                # reset (default) | halt   — what to do when the job list is done (oneshot)
```
Directives run in file order. `PROMPT`/`BENCH` may appear multiple times; each
appends a `job.N.*` block to RESULT.TXT.

## 3. RESULT.TXT — the result record (output)
`key=value` lines, LF, written **once** at the end of the job (not appended),
opened with `FileMode::CreateReadWrite`, flushed, closed. Values are ASCII;
`response` is escaped (`\n`, `\r`, `\\`, non-ASCII → `\xNN`) and truncated to
256 bytes.

```
aefinity_os=0.1
run_id=<contents of RUN.ID line run_id=…, or none>
env=iron|vm                 # CPUID.1:ECX[31] hypervisor bit → vm. NEVER quote tps when env=vm (Rule A).
cpu_brand=<CPUID brand string, trimmed>
cpuid_sig=<eax of CPUID.1, hex>
mac=<xx:xx:xx:xx:xx:xx or none>
ip=<a.b.c.d or none>
budget_s=240
jobs=2
job.1.kind=prompt
job.1.prompt=<escaped>
job.1.tokens=17
job.1.wall_ms=812
job.1.tps=20.93
job.1.tsc_per_tok=123456789
job.1.digest=<sha256 of the generated token-id sequence as little-endian u32s, first 16 hex>
job.1.response=<escaped, ≤256 bytes>
job.2.kind=bench
...
report=ok|fail <reason>|none
verdict=OK|FAIL <reason>
```
**`RESULT.WIP`** (added 2026-09-01) is written to the same directory *before*
each step starts, in the same format, with `verdict=FAIL incomplete: step N did
not return`, and deleted once `RESULT.TXT` is on the volume. It exists for the
one case the software budget cannot cover: the firmware watchdog resets the box
without running any of our code, so the only record that can survive is one
written beforehand. A stick that comes home carrying a `RESULT.WIP` is a box
that did not finish, and the file names the step it was in; `BOOTLOG.TXT` says
whether it was still in prefill or already decoding. `RESULT.TXT` is
authoritative whenever both are present, and the controller stages a volume
with neither file on it.

> **Erratum, 2026-09-01 (the delete's confirmation).** As first built, the
> delete was issued and the name re-opened in the next statement, and
> `BOOTLOG.TXT` reported `cleared=` from that. That is the hand-off, not the
> medium — the same distinction `settle_volume` exists for on the write side —
> and the check also read "empty" and "unreadable" as "gone". The delete is now
> issued when `RESULT.TXT` lands and *confirmed* after the settle stall, from
> two independent probes (an `open` that distinguishes `NOT_FOUND` from
> unreadable, and a walk of the root directory), with one retry if either still
> sees it. The line reads
> `JOB: RESULT.WIP cleared=<bool> (open=… dir=… retried=…)`.
>
> A gate-environment caveat, measured on 2026-09-01 and not a property of the
> unikernel: under `cargo xtask job-test` / `job-budget-test` the host mirror of
> the staged ESP still shows `result.wip` after a run in which both guest probes
> reported it absent and the guest's later writes committed through. QEMU's
> `fat:rw:` (vvfat) backend commits guest writes back to the host directory and
> does not commit the unlink. A stick has no such mirror — the FAT directory the
> firmware walks is the medium — so the gates assert the guest's BOOTLOG line
> and print the host state as a note. "A stick that comes home carrying a
> `RESULT.WIP`" is a statement about a stick, and it is unverified on iron until
> hardware first light.

`digest` is the CIS-style witness of *what was generated*; identical
digests across two machines for the same JOB.TXT are the fleet check in §1/3.

## 4. Resident protocol (MODE resident) — TCP, line-oriented, port `LISTEN`
```
S: AEFINITY-OS 0.1 READY env=<iron|vm> cpu=<brand>\n
C: PING\n                       S: PONG\n
C: JOB\n<JOB.TXT body>\nEND\n   S: RUNNING\n … S: RESULT\n<RESULT.TXT body>END\n
C: REBOOT\n                     S: BYE\n  → ResetSystem(COLD)
C: HALT\n                       S: BYE\n  → close listener, park with hlt/pause loop
```
One client at a time; a second connection gets `BUSY\n` and is closed. A job
body's `MODE`/`LISTEN`/`NET` lines are ignored in resident mode. Watchdog is
re-armed to `BUDGET+60` at `JOB` and to 0 (disabled) while idle — a resident box
idles indefinitely; a hung *job* is reset by firmware. Job bodies are capped at
64 KiB; `PROMPT` lines at 4 KiB.

## 5. Unikernel implementation map (`aegis-uefi/src`)
| file | owner phase | contents |
|---|---|---|
| `job.rs` | 0 | `Job` parser, `run_job(&Job, &mut root, &mut engine) -> ResultRecord`, `write_result_txt`, `RESULT.WIP` progress marker, budget accounting (pre-step + in-engine, both phases), `arm_watchdog(secs)`, `after(Reset|Halt)`, `escape()` |
| `sysinfo.rs` | 0 | `env()` (hypervisor bit), `cpu_brand()`, `cpuid_sig()` — read-only CPUID, no numbers invented |
| `net/mod.rs` | 1a | `SnpDevice` implementing `smoltcp::phy::Device` over `uefi::proto::network::snp::SimpleNetwork`; `Net::bring_up(cfg) -> Net` (static or DHCP with 10 s cap); `Net::poll(now)`; `Clock` from `wall_seconds()` (fallback: rdtsc calibrated against the UEFI stall) |
| `net/http.rs` | 1b | `post(&mut Net, url, body, timeout) -> Result<u16 status, Err>` — HTTP/1.1, `Connection: close`, no TLS, no redirects |
| `server.rs` | 2 | resident listener per §4, uses `job::run_job` |
| `main.rs` | 0 | **one** hook, after the engine is loaded and before the MECH diagnostic block and the interactive loop (see the §2 erratum): `if let Some(job) = job::load(&mut root) { job::dispatch(job, &mut root, &mut engine) }` — `dispatch` never returns in oneshot/reset and only returns in oneshot/halt (falls through to MECH and then the existing park loop) |

Rules that bind every builder:
- `no_std` + `alloc`; `uefi = 0.38` (`uefi::boot::set_watchdog_timer`, `uefi::runtime::reset`, `uefi::proto::network::snp`). smoltcp `0.12+` with features `["medium-ethernet","proto-ipv4","proto-dhcpv4","socket-tcp","socket-udp","socket-dhcpv4","alloc"]`, `default-features = false`. Add nothing else.
- Everything stays **in boot services** (never `ExitBootServices`) — SNP, file, watchdog and reset all need it.
- Build with `./aegis-uefi/build_hardfloat.sh` (production) and the stock target for `xtask` tests; both must compile. `scripts/devloop.sh fmt` clean. Do not touch `tests/golden/`, `docs/hardware_logs/`, or the ledger (Rule B/C).
- `BOOTLOG.TXT` gets a line at every stage transition (`JOB: parsed N directives`, `NET: ip=…`, `REPORT: ok 200`, `RESET: job complete`).
- No new performance claims anywhere. Under QEMU the RESULT carries `env=vm`; tests assert structure, never values.
- **Box constraints**: this dev machine has 6 GB RAM. Wrap every `cargo` and every `qemu` invocation in `flock /tmp/aefinity-os.lock …`, set `CARGO_BUILD_JOBS=4`, run `nice -n 10`. One heavy process at a time.
- Work in a sibling worktree `~/projects/alice-aegis-os-<phase>` on branch `os/<phase>` (never switch branches in `~/projects/alice-aegis`). Commit small, push the branch, never push `main`.

## 6. xtask gates (`xtask/src/main.rs`)
Reuse the existing `boot_test` staging (tiny assets from `model-lab/tinybit/m7_final_gate_work/artifacts`, `-m 2048`, TCG, `-serial stdio`, OVMF 4M). New subcommands share a helper `stage(job_txt: Option<&str>) -> EspDir` and a `qemu(args…, timeout)` runner:
- `job-test` — stages `JOB.TXT` (`BUDGET 180`, `PROMPT`, `BENCH 8`, `AFTER reset`), runs with `-no-reboot`; PASS = QEMU exits (guest reset) **and** `target/esp/RESULT.TXT` exists with `verdict=OK`, `jobs=2`, `env=vm`. (`fat:rw:` writes land in the host dir.)
- `job-budget-test` — stages a job whose `PROMPT` cannot be prefilled inside its `BUDGET`; PASS = QEMU exits (guest reset) **and** `RESULT.TXT` exists with a `verdict` of `FAIL budget`. This is the regression gate for budget enforcement during prefill: before it, such a job was killed by the firmware watchdog with no record written at all.
- `net-test` — adds `-netdev user,id=n0 -device virtio-net-pci,netdev=n0 -device virtio-rng-pci`; starts a TCP listener on an ephemeral host port; stages `JOB.TXT` with `NET static 10.0.2.15/24 10.0.2.2` and a test directive `NETCHECK 10.0.2.2:<port>`; PASS = listener receives `HELLO <mac>\n` within 60 s.
- `os-test` — like `net-test` but `REPORT http://10.0.2.2:<port>/aefinity/result`; harness runs a minimal HTTP server; PASS = POST body parses with `verdict=OK` and `report=ok 200` in the on-disk RESULT is **not** required (it was written before the POST) — assert on the received body.
- `resident-test` — `MODE resident`, `-netdev user,…,hostfwd=tcp:127.0.0.1:<port>-:4242`; harness connects (retry ≤60 s), expects `READY`, sends `PING`→`PONG`, sends a JOB, receives RESULT with `verdict=OK`, sends `REBOOT`, expects QEMU exit.
All gates honour Rule A: exit codes and structure only.

## 7. Host side (`claudius-maximus`, branch `cm/aefinity-os-host`)
- `bin/cm-os-collector` — Python 3 stdlib HTTP server (default `:8787`, path `/aefinity/result`): stores body to `state/reports/uefi/<run_id or timestamp>/RESULT.TXT`, appends a ledger row to `state/LOG.md` (host named from `cpu_brand`/`run_id`), inbox line, ntfy. `systemd/cm-os-collector.service` (user unit).
- `bin/cm-os-job HOST[:PORT] JOB.TXT [--out DIR]` — resident-protocol client (§4); prints RESULT; `--matrix` runs against every `usb-boot ok` row in `state/BOXES.md` and prints a `job.N.digest` comparison table.
- `bin/cm-uefi-qemu` — add `--net` (virtio-net user netdev + virtio-rng, `hostfwd` for 4242), `--job FILE`, `--tiny` (use tinybit assets).
- `bin/cm-uefi-run` — writes `REPORT http://<controller>:8787/aefinity/result` into JOB.TXT unless the job already has one; controller address from `/etc/cm/box.conf` `CM_CONTROLLER`.
- `docs/UEFI-REMOTE-LANE.md` — roadmap table (§1) + the OVMF finding: Debian's OVMF 2025.02 ships **no** EDK2 network stack (only `VirtioNetDxe` SNP + iPXE ROM), which is why the OS owns its stack.

## 8. Definition of done (for this build)
`cargo xtask boot-test job-test net-test os-test resident-test` all pass on
penguin under `flock`; `build_hardfloat.sh` produces an `.efi` that passes
`scripts/check-efi-simd.sh`; branches pushed; `PR: os/p0-job → main` etc.
opened with the gate output pasted; host tooling pushed; roadmap merged.
Hardware first-light is Justin's step with the flashed stick and is **not**
claimed here.

## 9. Roadmap beyond v0.1 (added 2026-08-31 after Justin's fleet directive)
Justin: "always be able to remote into any of them and run all the labs, pull data, build
projects, even pool CPUs for AI building." Mapped honestly onto the two bodies of each box:

| phase | body | delivers | why it is possible |
|---|---|---|---|
| **4 FILES** | unikernel | resident protocol verbs `GET <path>` / `PUT <path> <len>` / `LS` / `SHA <path>` over the boot volume — pull `BOOTLOG.TXT`, push a new `JOB.TXT`, **swap `MODEL.SAF` remotely and `RELOAD`** without touching the stick | FAT32 write path already exists (`boot_log`); smoltcp TCP from 1a |
| **5 LAB** | unikernel | job kinds beyond PROMPT/BENCH: `EVAL <corpus>` (perplexity slices), `VERIFY <receipt>` (existing `verifier.rs`), `MEMBW`, `CPUID`, `MECH` — the existing lab suites become directives, results in `RESULT.TXT` with digests | suites exist in `aegis-uefi`/`aegis-eval`; this is plumbing |
| **6 POOL** | fleet | `cm-os-job --pool`: shard a job list (prompts, eval slices, receipt batches, data-gen seeds) across every resident box, stream results to the collector, merge with per-shard digests; scheduler is host-side Python first, then a Rust `aefinity-ctl` | resident boxes are stateless workers; embarrassingly parallel work needs no interconnect |
| **7 TRAIN-POOL** | Debian side first | pooled CPU *training* (own-model directive): torch/DDP over the boxes' Debian side via ssh + gloo, driven by `cm`; the unikernel joins only when `aegis-core` grows a backward pass (research item, not plumbing) | no_std training kernels are a separate research track; do not block the fleet on it |
| **build projects** | Debian side | `cm-build` already runs sonnet builders on any ssh box; boxes register in `BOXES.md`; the unikernel has **no compiler** and should not grow one | keep the OS small; Debian is the recovery *and* the compiler |

Order after v0.1 lands: 4 → 5 → 6 (each a verified branch with an xtask gate), 7 in parallel on the
Debian side once two boxes exist.

Design for phases 4–6: **`program/AEFINITY_OS_FLEET_DESIGN.md`** (judge-panel synthesis, revised after
critic review 2026-09-01; drafts and the critic pass in `program/fleet-design-drafts/`). It is the build
contract for `os/p4-files`, `os/p5-lab`, and the host-side `cm-os-pool`.

**Status.** Phase **5 LAB is built** on `os/p5-lab` (stacked on `os/p4-files`): `CPUID`, `VERIFY
<NAME>`, `EVAL <NAME> <lo>:<hi>`, `MEMBW <mib>` and `MECH` are `JOB.TXT` directives, the fleet
directives `RUNID`/`TAG`/`SHARD`/`SEED`/`STRICT` parse, and `RESULT.TXT` carries §3's `job.N.*`
fields plus `merge_key`. Gate: `cargo xtask lab-test`, under QEMU/TCG only. `EVAL`'s `nll_q16` is
exact-integer and therefore comparable across boxes **by construction**; `membw_mibs` is iron-only
and reads `n/a` on every record this branch has produced. What is and is not shown is in
`docs/AEFINITY_OS_STATUS.md` §10 — nothing above is claimed on iron.
