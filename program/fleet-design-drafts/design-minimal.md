# AEFINITY OS phases 4–6 — the PROTOCOL-MINIMALIST design

*Angle: the smallest set of verbs and job kinds that delivers FILES, LAB and POOL with
the fewest new failure modes; host-side Python first; a gate per verb. Designed against
the shipped v0.1 code in `~/projects/alice-aegis-os-int`, not the spec text alone.*

**Three cuts that define this design.**

1. **`RELOAD` does not exist.** Spec §9 asks to "swap `MODEL.SAF` remotely and `RELOAD`".
   `main.rs:564-663` loads the three artifacts into contiguous `allocate_pages` regions at
   boot and hands slices to the engine for the process's life. A resident `RELOAD` means
   re-allocating ~1.9 GB of contiguous physical pages in a heap a job has already
   fragmented — the exact failure `STAGE 3 FAILED: contiguous alloc` names. The capability
   is delivered instead by `PUT MODEL.SAF` then the existing `REBOOT`, on code that already
   has a passing gate. One fewer verb, zero new failure modes.
2. **Phase 5 adds no verbs at all.** Lab kinds are `JOB.TXT` directives; `JOB` already
   carries arbitrary bodies over the socket. LAB is a §2/§3 change plus step runners.
3. **Phase 6 adds no unikernel code at all.** POOL is a host-side scheduler over verbs
   that phases 2 and 4 already shipped. Nothing in `aegis-uefi` changes for POOL.

---

## 1. Protocol additions to §4

Five new commands. All are line-oriented, ASCII, `\n`-terminated, uppercase keyword,
single-space separated — the shape `serve()` already parses via `upper()`/`Lines`
(`server.rs:227-306`). All obey the existing one-client-at-a-time rule; a second peer
still gets `BUSY\n`.

```
C: LS\n
S: LS <n>\n
S: <NAME> <size>\n      × n            # ascending byte order of NAME
S: END\n

C: SHA <NAME>\n
S: SHA <NAME> <size> <64-hex>\n        # full sha256, lowercase hex

C: GET <NAME>\n
S: DATA <size> <16-hex>\n              # 16-hex = first 16 chars of the sha256
S: <size raw bytes>
S: END\n

C: PUT <NAME> <size> <64-hex>\n
S: SEND\n
C: <size raw bytes>
C: END\n
S: OK <NAME> <size> <16-hex>\n

C: DEL <NAME>\n
S: OK <NAME>\n
```

**Framing rule (the only new one).** After a `DATA` or `SEND` header line, exactly
`<size>` raw octets cross the wire, then the literal `END\n`. No base64: `MODEL.SAF`
would inflate 33 % and there is no encoder in the unikernel. `Lines` already buffers a
remainder (`server.rs:553+`); it gains one method, `take_bytes(n)`, draining `pending`
first and then reading through `Net::tcp_recv_until(Until::Len(k), …)` — `Until::Len`
exists today under `#[allow(dead_code)]` (`net/mod.rs:175-186`). The trailing `END\n` is
a resync marker, not a delimiter: a client that miscounts is caught at the next line read
and dropped.

**Names.** `<NAME>` is 1–32 bytes of `[A-Z0-9._-]`, no `/` `\` `:`, not starting `.`,
upper-cased before use. Boot-volume root only — there is no traversal because there is no
directory syntax. `BOOTLOG.TXT`, `RESULT.TXT`, `RESULT.WIP` and any `*.PART` are readable
but `PUT`/`DEL` refuse them: the box's own record is not client-writable.

**Size caps (byte-exact constants).**

| constant | value | why |
|---|---|---|
| `LS_MAX_ENTRIES` | `256` | one FAT32 root listing, bounded reply |
| `NAME_MAX_BYTES` | `32` | matches `write_named`'s `[u16; 32]` name buffer |
| `GET_MAX_BYTES` | `2_147_483_648` | 2 GiB; the `ALICE_UEFI` partition is 4 GB |
| `PUT_MAX_BYTES` | `2_147_483_648` | same |
| `XFER_CHUNK` | `65_536` | one `RECV_MAX_BYTES` slice (`net/mod.rs:119`) |
| `XFER_STALL_MS` | `30_000` | no progress for this long ⇒ `ERR io`, drop |
| `FILES_WD_S` | `300` | watchdog armed during a transfer, re-armed each chunk |

**Error strings.** Every error is exactly `ERR <slug>\n`, slug lowercase kebab, and the
connection is dropped after any `ERR` except `ERR no-file` and `ERR bad-name` (which are
answers, not faults, so the client may keep going). New slugs, complete list:
`ERR bad-name`, `ERR bad-args`, `ERR no-file`, `ERR too-large`(existing),
`ERR put-sha`, `ERR put-short`, `ERR put-write`, `ERR put-commit`, `ERR no-space`,
`ERR io`, `ERR protected`. `ERR unknown` keeps its meaning. **No verb ever returns a
partial success**: `PUT` either lands the whole file or leaves the target untouched.

---

## 2. `JOB.TXT` additions to §2 (phase 5)

Five directives, each one appending a `job.N.*` block exactly as `PROMPT`/`BENCH` do.

```
CPUID                      # dump CPUID identity; ok=true always; no timing
VERIFY <NAME>              # replay a receipt file through verifier::run
EVAL <NAME> <slices>       # perplexity over the first <slices> slices of a corpus file
MEMBW <mib>                # memory-traffic diagnostic over <mib> MiB (default 64)
MECH                       # the existing MECH diagnostic block
ARTIFACTS on|off           # default on: record artifact digests (see §3)
```

`EVAL`/`VERIFY` take a `<NAME>` under the §1 name rules, so a corpus or receipt arrives by
`PUT` and LAB composes with FILES without either knowing about the other. `MECH` and
`MEMBW` produce *numbers*: recorded beside the existing `env=iron|vm` key, and **no gate
asserts a value from them** (Rule A). `MECH` runs ~24 min under TCG, so no gate stages it.

## 3. `RESULT.TXT` additions to §3

Two new global keys and one new per-step key. Nothing else.

```
artifacts=MODEL.SAF:<16hex>,EMBED.BIN:<16hex>,VOCAB.BIN:<16hex>   # or "off"
files=<n>                        # entries seen in the root at record time
job.N.detail=<escaped, ≤1024 bytes>
```

`artifacts=` is what makes POOL trustworthy: two shards' digests may be compared only when
the boxes agree on this line. Computed once at first use by streaming
`aegis_core::witness::Sha256` over the already-resident slices — no re-read from FAT.
`job.N.detail` carries what a kind needs beyond the 256-byte `response` (`EVAL`'s
per-slice nll list, `VERIFY`'s last detail line, `CPUID`'s feature string). All other keys
are reused unchanged; `EVAL`'s `digest` is sha256 over the per-slice integer nll
accumulators, which is what makes a slice comparable across silicon.

---

## 4. Unikernel modules touched (§5 map)

| file | phase | change | rough LOC |
|---|---|---|---|
| `server.rs` | 4 | five verb arms in `serve()`; `Lines::take_bytes`; `xfer` helpers; watchdog arm/disarm around transfers | +330 |
| **`files.rs`** (new) | 4 | name validation, `ls()`, `sha_named()` (streaming, `XFER_CHUNK` reads), `put_stream()` → `<NAME>.PART` → commit, `del_named()`, `sweep_parts()` | +270 |
| `net/mod.rs` | 4 | nothing new; `Until::Len` loses its `#[allow(dead_code)]` | +5 |
| `job.rs` | 5 | five `Step` variants + parse arms; `artifacts=`/`files=`/`detail` in `ResultRecord::render`; call `files::sweep_parts` from `load()` | +190 |
| **`lab.rs`** (new) | 5 | one runner per kind; `verify` wraps `verifier::run`, `mech` wraps the existing block moved out of `main.rs` behind a function, `eval` wraps `aegis-eval`'s slice loop | +310 |
| `main.rs` | 4,5 | `mod files; mod lab;` and extracting the MECH block into `lab::mech()` | +6, −180 |
| `xtask/src/main.rs` | 4,5,6 | `files-test`, `lab-test`, `pool-test` | +460 |

No new crate dependencies. Everything stays in boot services.

---

## 5. Host tooling (`claudius-maximus`, branch `cm/aefinity-os-fleet`)

Python 3 stdlib only, matching `bin/cm-os-job`'s existing style and its
`tests/test_os_host.py` suite.

- **`bin/cm-os-files HOST[:PORT] <ls|get|put|sha|del>`** — one subcommand per verb, plus
  `push-model DIR` (`PUT` the three artifacts, `SHA` each back, `REBOOT`) and
  `pull-logs DIR`. Every `get` is verified against the header 16-hex before the host file
  is renamed into place; every `put` re-`SHA`s the box copy afterwards.
- **`bin/cm-os-pool PLAN.json [--boxes state/BOXES.md] [--replicate N] [--resume]`** —
  the phase-6 scheduler (§6 below). Emits `state/pools/<pool_id>/plan.json` and
  `shards/<id>/RESULT.TXT`.
- **`bin/cm-os-collector`** — role unchanged and deliberately dumb: an append-only sink at
  `/aefinity/result` that stores a body, writes a ledger row, pushes ntfy. It is **not** in
  the POOL control path. Because a resident job's `REPORT` is refused in v0.1 (the listener
  holds SNP exclusively), the scheduler POSTs each shard's `RESULT.TXT` to the collector
  *on the box's behalf* with `X-Aefinity-Via: cm-os-pool` and `X-Aefinity-Box: <name>`.
  The collector gains one header-aware ledger line (+60 LOC). A known v0.1 limitation
  becomes a host-side detail instead of a unikernel change.
- **`bin/cm-os-job`** unchanged; `--matrix` keeps working and is the degenerate 1-shard
  pool.

---

## 6. Fleet scheduler semantics (POOL)

**Plan.** A pool run is a JSON plan: `{pool_id, artifacts_expect, budget_s, shards:[{id,
body}]}` where `body` is a literal `JOB.TXT` fragment. `id` is a small integer; the plan
is the unit of reproducibility and is written before any dispatch.

**Sharding.** Explicit and dumb: `--split prompts FILE` (`--per-shard K` to batch),
`--split eval NAME K` (slice ranges), `--split seeds A..B`. No auto-balancing — shards are
equal-count, not equal-time, because Rule A forbids timing a box in order to schedule it.

**Dispatch.** Work-stealing over a queue of shard ids. One in-flight shard per box (the
protocol allows one client). A box takes the next shard when its previous connection
closes. Per-shard deadline = `budget_s + 90`.

**What counts as agreement.** Two results for the same shard *agree* iff (a) their
`artifacts=` lines are byte-identical, (b) their `verdict=` are both `OK`, and (c) the
ordered list of `job.N.kind|job.N.digest` pairs is byte-identical. `wall_ms`, `tps`,
`tsc_per_tok`, `mac`, `ip`, `cpu_brand` are excluded — those differ by construction and
none of them may become a claim. With `--replicate 2` a shard is `AGREED`, `SPLIT`
(digests differ — the interesting result, never silently resolved) or `PARTIAL`.

**Merge digest.** `pool_digest = sha256(for each shard id ascending: "<id>\n" +
"<kind>:<digest>\n"…)`, first 16 hex, printed with `artifacts=` and the box roster.
Identical `pool_digest` from two disjoint box sets over one plan is the fleet-scale form of
the CIS-2 cross-ISA claim — a determinism result, never a speed one.

**Retries.** A shard is re-dispatched at most twice. Transport failures (connect refused,
`BUSY`, timeout, connection reset) re-queue to a *different* box. A protocol-level `ERR`
or a `verdict=FAIL <reason>` is **deterministic** and is not retried anywhere — it is
recorded and the shard ends `FAILED`. Three consecutive transport failures quarantine a
box for the rest of the run (logged to `state/NEEDS.md`: "box <name> unreachable").

**Partial results.** Each completed shard hits disk the moment it lands. `--resume`
re-reads `plan.json` + `shards/` and dispatches only `PENDING`/`FAILED`. Shards are pure
functions of `(body, artifacts)`, so duplicate execution is harmless and the merge keeps
the first agreeing answer per id. A pool finishing with any shard not `DONE` exits
non-zero and prints the shard table.

**The scheduler never reboots a box.** `--reboot-on-quarantine` is opt-in and off by
default. Autonomy does not extend to power-cycling Justin's laptops without being asked.

---

## 7. xtask gates (structure and exit codes only)

- **`files-test`** (phase 4) — resident QEMU as `resident-test` sets up. Harness:
  `PUT TEST.BIN 262144 <sha>` of a pseudorandom blob → expect `OK TEST.BIN 262144 <16hex>`;
  `LS` → the entry is listed with size `262144`; `SHA TEST.BIN` → 64-hex equals the
  host's; `GET TEST.BIN` → the 262144 bytes are **byte-identical** to what was sent (Rule
  D: bit-exactness, not a rate); `PUT TEST.BIN 16 <wrong-sha>` → `ERR put-sha` and
  `SHA TEST.BIN` still reports the old, correct digest; `PUT ../X 1 <sha>` →
  `ERR bad-name`; `PUT BOOTLOG.TXT 1 <sha>` → `ERR protected`; `DEL TEST.BIN` → `OK`,
  then `SHA TEST.BIN` → `ERR no-file`; `REBOOT` → `BYE`, QEMU exits.
- **`lab-test`** (phase 5) — resident job body `CPUID` + `VERIFY RECEIPT.TXT` +
  `EVAL TINY.TXT 2` (both files `PUT` first, so the gate exercises 4 and 5 together).
  PASS = `jobs=3`, `job.1.kind=cpuid`, `job.2.kind=verify`, `job.3.kind=eval`, all
  `job.N.ok=true`, `artifacts=` present and non-empty, `verdict=OK`, `env=vm`.
  **No `MECH`, no `MEMBW`** in the gate — 24 minutes under TCG, and their output is
  numeric.
- **`pool-test`** (phase 6) — boots one resident guest, writes a 4-shard plan, runs
  `cm-os-pool --replicate 1` against a one-row `BOXES.md`. PASS = exit 0, four
  `shards/<id>/RESULT.TXT` each `verdict=OK`, a `pool_digest` line printed, and a second
  `--resume` run exiting 0 having dispatched zero shards. Multi-box semantics (agreement,
  SPLIT, quarantine, mid-shard disappearance) are covered by `python3 -m unittest
  tests.test_pool` against in-process fake resident servers — the scheduler must not need
  iron to be tested.

---

## 8. Explicitly left to the Debian side

Compilers and `cargo` (the unikernel will not grow one); `git`; `aegis-forge` model
conversion and quantization; corpus and receipt *generation*; TLS and DNS (the HTTP
client is address-only by design); any artifact larger than the `ALICE_UEFI` partition;
long-term storage and backups; `efibootmgr`/`BootNext` and `cm-uefi-harvest`; smart-plug
power cycling; and the whole of **phase 7 TRAIN-POOL** (torch/DDP over ssh + gloo). The
unikernel's job is to be a stateless worker that can be filled, asked, and read.

## 9. Sequencing and the first five commits

Order: **4 → 5 → 6**, one branch each (`os/p4-files`, `os/p5-lab`,
`cm/aefinity-os-fleet`), each merged only with its gate output pasted into the PR. Work in
sibling worktrees; every `cargo`/`qemu` under `flock /tmp/aefinity-os.lock nice -n 10 env
CARGO_BUILD_JOBS=4`.

1. **`files.rs` + name rules + `sweep_parts`, no protocol.** Pure functions with unit
   coverage on the name validator; `job::load` calls `sweep_parts` and logs it.
   Verification: `scripts/devloop.sh fmt`, `boot-test` still 33.
2. **`LS` / `SHA` / `DEL` in `serve()`** — the three verbs with no bulk framing. Small,
   and they make the box inspectable before anything can write to it.
3. **`Lines::take_bytes` + `GET`** — read path first: a bug here cannot corrupt the
   volume.
4. **`PUT` with `.PART` + commit + watchdog petting**, and `xtask files-test` in the same
   commit (a write verb does not land without its gate).
5. **`bin/cm-os-files` + `push-model`/`pull-logs`**, with `tests/test_os_host.py`
   extended against a fake server. Phase 4 closes here; phase 5 opens with `lab.rs` and
   the `MECH` extraction from `main.rs`.

## 10. Iron safety analysis

- **Watchdog.** Idle stays at 0 (a resident box may wait for ever). `GET`/`PUT` arm
  `FILES_WD_S = 300` and **re-arm every `XFER_CHUNK`**, so a 1.8 GB transfer never
  reaches the timer while a hang inside the FAT write path always does. Disarmed the
  instant the verb returns. `JOB` keeps `budget + WATCHDOG_MARGIN_S` unchanged.
- **Hangs.** `XFER_STALL_MS` bounds *no-progress*, not total duration: a slow link is not
  punished, a dead peer is. Every `ERR` and stall path returns to `accept`, never a wedge —
  the invariant `server.rs` already documents.
- **Half-written files.** `PUT` writes `<NAME>.PART` and commits by deleting the target
  then renaming via `set_info::<FileInfo>`. A firmware that refuses the rename yields
  `ERR put-commit`, `.PART` survives and the target is untouched. Delete-then-rename is the
  one ordering that could lose a good file, so the delete happens only after the `.PART`
  sha256 has been verified by re-reading it off the volume.
- **Power loss during `PUT`.** Worst case: the old target intact plus an orphan `.PART`.
  `sweep_parts()` deletes every `*.PART` at boot and logs each one to `BOOTLOG.TXT`. If
  power dies inside the commit window the box may come back with no `MODEL.SAF`; the boot
  path already fails loudly (`STAGE 4 FAILED`) and BootNext has been spent, so the box
  lands in Debian and `cm-os-files push-model` fixes it remotely. No state is
  unrecoverable without walking to the box.
- **A box that disappears mid-shard.** It finishes the job anyway (the server deliberately
  runs a job whose peer has gone) and leaves `RESULT.TXT` on the volume; the scheduler
  re-queues the shard elsewhere and `pull-logs` recovers the orphan later. Duplicate
  execution is harmless because shards are pure.
- **Rule A.** `files-test`, `lab-test` and `pool-test` assert existence, equality and
  exit codes. `MECH`/`MEMBW` are reachable but ungated, and `artifacts=`/`env=` travel
  with every record so no number can later be mistaken for an iron number.
