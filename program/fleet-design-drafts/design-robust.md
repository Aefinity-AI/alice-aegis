# AEFINITY OS phases 4–6 — the FLEET-OPERATOR design

*Angle: design from the controller down. Boxes are cattle. The collector is the
source of truth. Every wire verb exists because a scheduler on penguin needs it
to decide "retry, reassign, or believe the answer."*

Built against the real v0.1 API in `~/projects/alice-aegis-os-int`
(`job.rs` 1373 LOC, `server.rs` 710, `net/mod.rs` 1277, `net/http.rs` 168), not
the spec alone. Rule A binds: no gate below asserts a timing value.

---

## 1. Protocol additions to §4

v0.1 is line-oriented ASCII, one client at a time, `BUSY\n` for a second peer,
`ERR unknown\n`, `ERR too-large\n`. Phases 4–6 add **one framing primitive** and
seven verbs. Nothing else about §4 changes — same banner, same `PING`/`JOB`/
`REBOOT`/`HALT`, same `ERR` vocabulary style (lowercase, hyphenated, no period).

### 1.1 The `DATA` frame (the only binary on the wire)

```
DATA <len> <sha16>\n<exactly len raw bytes>
```

`<len>` decimal ASCII, no leading zeros, `0 ≤ len ≤ 2147483648` (2 GiB — `MODEL.SAF`
is 1.83 G). `<sha16>` = first 16 hex chars of sha256 over those bytes, lowercase.
No trailing newline after the payload: the length is the frame, and a trailing
byte would make a resumed transfer ambiguous. Both directions use it.

### 1.2 Phase 4 FILES verbs

```
C: LS\n
S: FILE <NAME> <size> <attr>\n   (repeated, NAME is 8.3 upper-case ASCII)
S: END\n

C: SHA <NAME>\n                  S: SHA <NAME> <sha64>\n   | ERR <e>\n
C: STAT <NAME>\n                 S: STAT <NAME> <size>\n   | ERR not-found\n

C: GET <NAME>\n
S: DATA <len> <sha16>\n<bytes>   | ERR <e>\n

C: PUT <NAME> <len> <sha64>\n
S: SEND\n                        | ERR <e>\n
C: <exactly len raw bytes>
S: OK <NAME> <sha16>\n           | ERR digest-mismatch\n | ERR short-write\n

C: RM <NAME>\n                   S: OK <NAME>\n | ERR not-found\n | ERR busy-file\n
C: RELOAD\n                      S: RELOADING\n … S: OK reload <sha16>\n | ERR <e>\n
```

`PUT` is **declare-then-stream**: the server pre-checks name, length and free
space *before* the client burns 1.8 GB on the wire. `sha64` is the full 64-hex
digest — the whole point is that the operator can prove what landed.

`RELOAD` re-runs `load_file_into` for `MODEL.SAF`/`EMBED.BIN`/`VOCAB.BIN` into the
already-allocated tensor arena and answers with the sha16 of the model actually
resident. It refuses (`ERR size-changed`) if the new file does not fit the arena
`main.rs` allocated at boot: growing the arena means a `REBOOT`, and saying so beats
faulting in ring 0.

### 1.3 Phase 6 POOL verbs

```
C: HEALTH\n
S: HEALTH up=<s> served=<n> last=<OK|FAIL <r>|none> wd=<off|<s>> \
   heapfree=<bytes> rx=<n> tx=<n> model=<sha16> env=<iron|vm>\n

C: RUNID <id>\n                  S: NEW\n | REPLAY\n
```

`RUNID` before a `JOB` is the **at-most-once** primitive, and the single most
important addition for a fleet. The server keeps the last 8 `(run_id, rendered
RESULT.TXT)` pairs in a ring; a known id answers `REPLAY` and the following
`JOB … END` is **not run** — the cached record comes back as `RESULT\n<body>END\n`
with `replay=true`. A controller whose TCP died after `RUNNING` but before `RESULT`
can retry blind without double-spending a 20-minute shard.

`<id>` is `[A-Za-z0-9._-]{1,64}`; anything else is `ERR bad-runid`.

### 1.4 Caps and error strings (byte-exact)

| cap | value |
|---|---|
| command line | 64 KiB (unchanged, `LINE_MAX_BYTES`) |
| `LS` listing | 512 entries, then `END` (FAT32 root is small) |
| `PUT` len | ≤ 2 GiB **and** ≤ free clusters − 16 MiB reserve |
| `GET` len | no cap; framed |
| name | 1–12 chars, `[A-Z0-9._]`, no `/`, no `\`, no `..` |
| RUNID ring | 8 entries |

New `ERR` strings, exhaustive: `ERR bad-name`, `ERR not-found`, `ERR exists`,
`ERR no-space`, `ERR digest-mismatch`, `ERR short-write`, `ERR short-read`,
`ERR size-changed`, `ERR busy-file`, `ERR bad-runid`, `ERR bad-len`,
`ERR fw-error`. Plus the two v0.1 already has. A client that sees a string not
on this list must treat the box as unhealthy, not as failed-this-job.

---

## 2. JOB.TXT additions (§2)

```
RUNID pool-2026-09-04-a.s03   # idempotency key; same as the RUNID verb
TAG mech-sweep                # free-form, echoed into RESULT and the ledger
SHARD 3/8                     # this box's slice of a pool run; informational to the box,
                              #   load-bearing to the collector
SEED 12345                    # sampler seed; makes digest comparison meaningful
EVAL WIKI2.BIN 0:512          # perplexity over a byte/token slice of a corpus on the volume
VERIFY RECEIPT.TXT            # existing verifier.rs run as a job step
CPUID                         # dump leaves as key=value; no timing
MEMBW                         # structural: bytes touched, not GB/s (Rule A)
MECH                          # the existing diagnostic block, as a directive
STRICT on                     # any step failure ⇒ stop and verdict=FAIL <step-kind>
```

`SHARD i/n`: `1 ≤ i ≤ n ≤ 4096`. The box does not compute its own slice —
the controller writes the concrete work into the body. `SHARD` exists so the
record can be attributed without trusting the collector's bookkeeping alone.

`MECH` is placed **after** every other step by the dispatcher regardless of file
order: it runs ~24 min under TCG and must never starve the real work. Same reasoning
that put v0.1's job hook before the MECH block in `main.rs`.

---

## 3. RESULT.TXT additions (§3)

Appended to the existing key order, never reordering v0.1 keys:

```
run_id=pool-2026-09-04-a.s03
tag=mech-sweep
shard=3/8
seed=12345
replay=false                 # true ⇒ served from the RUNID ring, not re-run
model_sha=<sha16 of MODEL.SAF as resident>
uptime_s=41233
served=17                    # jobs this box has served since boot
job.N.err=<none|short string> # why a step produced no digest
job.N.partial=<0|k>           # k tokens/items completed of the requested count
job.N.ppl=<fixed2>            # EVAL only
job.N.items=<n>               # EVAL/VERIFY only
job.N.pass=<true|false>       # VERIFY only
merge_key=<sha16 of (seed, model_sha, step kinds, step inputs)>
```

`merge_key` is the fleet's join column: two records may be compared only if
their `merge_key` matches. It deliberately excludes `cpu_brand`, `env` and
`shard` — those are what the comparison is *about*.

`job.N.partial` is what makes partial results usable: a shard killed by budget
at token 40 of 64 still reports 40 and a digest **over those 40**, so a
re-run on another box can be checked for prefix agreement instead of thrown away.

---

## 4. Unikernel modules (§5 map) and rough LOC

| file | phase | change | LOC |
|---|---|---|---|
| `files.rs` (new) | 4 | name validation, `ls`, `stat`, `sha_file` (streaming sha256 through the existing bounce buffer), `get_stream`, `put_stream` with `.PART` staging + readback verify + rename via `set_info::<FileInfo>`, free-space query | ~460 |
| `net/mod.rs` | 4 | `tcp_recv_exact(&h, &mut [u8], timeout)` and `tcp_send_slice` chunked at the existing socket buffer size; watchdog re-arm hook per chunk | ~90 |
| `server.rs` | 4,6 | `LS/STAT/SHA/GET/PUT/RM/RELOAD/HEALTH/RUNID` dispatch, `DATA` framing, RUNID ring (`[(String, String); 8]`) | ~300 |
| `lab.rs` (new) | 5 | `Step::{Eval, Verify, Cpuid, Membw, Mech}` execution; wraps `verifier::run`, the MECH block lifted out of `main.rs`, and an `aegis-eval`-shaped perplexity slice loop | ~380 |
| `job.rs` | 4,5,6 | parse `RUNID/TAG/SHARD/SEED/STRICT/EVAL/VERIFY/CPUID/MEMBW/MECH`; new `StepResult` fields; `merge_key`; MECH reordering; `StepErr` | ~250 |
| `main.rs` | 4 | expose the tensor arena slices to `RELOAD` (one struct, passed to `dispatch`); **no new hook** | ~40 |

Total ≈ 1520 LOC across two new files and four edits. `no_std` + `alloc`, uefi
0.38, smoltcp 0.12, **no new dependency** — sha256 already exists (`token_digest`,
`verifier.rs`).

---

## 5. Fleet scheduler semantics (POOL)

The scheduler is host-side Python in `claudius-maximus`, single-process, state in
SQLite at `state/pool/<pool_id>.db`. It never holds work in memory only.

**Sharding.** A pool run is a list of *units* (a prompt, an eval slice, a receipt),
numbered 1..n at submit time and written to the DB before any box is contacted.
Shard size is chosen so one shard's `BUDGET` is ≤ 900 s; fast boxes get more shards,
not bigger ones. Nothing splits at the token level.

**Leasing.** A shard is `pending → leased(box, t0, deadline=BUDGET+120) → done|failed`.
The lease is host-side bookkeeping; the box's own mutual exclusion is v0.1's
`BUSY`. On deadline expiry the shard returns to `pending` with `attempts += 1`,
and — critically — the **same `RUNID`** goes back out. If the original box was
merely slow, its eventual answer is deduped by the RUNID ring; if it was reassigned
and both answer, the two records must agree or the pool is flagged `SPLIT`.

**Retries.** 3 attempts; a box is re-offered a shard it failed only if no other
`HEALTHY` box exists. Backoff 5 / 30 / 120 s. Box-level `ERR`s (`fw-error`,
`no-space`) mark the box `SUSPECT` at once; job-level verdicts (`FAIL budget`) do not.

**Box health.** A poller hits `HEALTH` every 30 s. States: `HEALTHY` → `SUSPECT`
(one timeout or a box-level ERR) → `DOWN` (3 consecutive) → `RECOVER` (drive the
Debian side per §8) → `HEALTHY` after two clean `HEALTH` + a `PING`. `DOWN` boxes'
leases are expired at once rather than waiting out the deadline. `model_sha`
mismatch across boxes is a hard stop before any pool run starts.

**Partial results.** A record with `verdict=FAIL budget` and `job.N.partial=k>0`
is stored, marked `PARTIAL`, and the shard is re-queued with the remainder only.
Partials count toward the merge as prefix evidence, never as a completed unit.

**Digest merge.** Per unit the collector stores `(unit_id, box, digest, partial)`.
The pool digest is `sha256` over the units' digests in unit order, first 16 hex,
and it is only computed when every unit is `done`. Any unit missing ⇒ the pool has
**no digest**, and the report says which units are missing. There is no
"digest of what we got".

**Agreement.** Agreement is a claim about *replication*, so it needs ≥2 boxes on
the same unit with the same `merge_key`. `AGREE` = byte-identical `job.N.digest`.
`DISAGREE` = both complete, digests differ — this is the CIS-style finding and it
is a **deliverable, not an error**: it is written to `state/pool/<id>/DISAGREE.md`
with both records verbatim and pushed to the phone. `INCONCLUSIVE` = a partial on
either side. One box alone is `UNREPLICATED`, never `AGREE`.

**Collector as source of truth.** Boxes are stateless: `RESULT.TXT` on a volume is
a courtesy copy for whoever pulls the stick. The DB row plus the raw record under
`state/reports/uefi/<run_id>/RESULT.TXT` is the record of what happened, appended
never rewritten, and every ledger row cites that path (Rule B).

---

## 6. Host tooling (`claudius-maximus`, branch `cm/aefinity-fleet`)

- `bin/cm-os-fs HOST[:PORT] ls|stat|sha|get|put|rm|reload [NAME] [FILE]` — the §1.2
  client. `put` verifies by re-`SHA`ing after the box answers `OK`. Refuses to put
  `MODEL.SAF` unless `--i-know` or the sha matches a manifest.
- `bin/cm-os-pool submit UNITS.txt --boxes state/BOXES.md --budget 600 [--replicate 2]`,
  `cm-os-pool status <pool_id>`, `cm-os-pool merge <pool_id>` — the scheduler above.
- `bin/cm-fleet-health [--watch]` — one line per box: state, `model_sha`, `served`,
  `up`, last verdict. Exit 1 if any box is `DOWN`.
- `bin/cm-os-collector` (extend) — accept the new keys, index by `merge_key`, write
  `state/pool/<id>.db`, and on pool completion emit `state/reports/pool/<id>/SUMMARY.md`
  plus an ntfy push.
- **Phone view** (one screen, via ntfy + the existing inbox): `pool <id>: 61/64 done,
  2 boxes, 1 DISAGREE, 0 down`, then on completion `pool <id> COMPLETE digest=…
  AGREE 63/64 DISAGREE 1` with the disagreement's two `cpu_brand`s. Anything requiring
  a decision goes to `NEEDS.md`, e.g. `box hp-stream needs a power button`.

---

## 7. xtask gates (structure and exit codes only)

- **`files-test`** (phase 4) — resident guest with `hostfwd`. Harness: `LS` contains
  `MODEL.SAF`; `SHA JOB.TXT` matches the host's own sha of the staged bytes;
  `GET BOOTLOG.TXT` returns a `DATA` frame whose length and sha16 match the payload;
  `PUT NEWJOB.TXT` of 4 KiB then `SHA NEWJOB.TXT` matches; a `PUT` with a deliberately
  wrong sha returns `ERR digest-mismatch` **and** `LS` afterwards shows neither
  `NEWBAD.TXT` nor a stray `.PART`; `PUT` with `len` 3 GiB returns `ERR bad-len`
  before any bytes are sent; `RM` then `STAT` returns `ERR not-found`. PASS = all
  assertions plus `REBOOT` → `BYE` → QEMU exits.
- **`lab-test`** (phase 5) — oneshot `JOB.TXT` with `VERIFY RECEIPT.TXT`, `CPUID`,
  `EVAL` over a tiny staged corpus, `SEED 1`. PASS = `jobs=3`, each block carries a
  `job.N.kind`, `VERIFY` carries `job.N.pass`, `EVAL` carries `job.N.ppl` **parsing
  as a number** (value never asserted), `merge_key` present and non-empty, `verdict=OK`,
  guest resets. `MECH` is excluded from the gate by name — 24 min under TCG.
- **`pool-test`** (phase 6) — **host-side only, no second guest** (6 GB box). One real
  resident guest plus an in-process fake box that speaks §4. Harness asserts: 8 units
  shard across 2 boxes; killing the fake mid-shard re-leases that shard within
  `deadline`; the re-issued `RUNID` to the real guest returns `REPLAY` on a duplicate
  send; the merged digest is present only when all 8 units are `done`; removing one
  unit's record makes `merge` exit non-zero and name the missing unit. PASS = the
  scheduler's exit codes, not the guest's numbers.

---

## 8. Iron safety analysis

- **Watchdog during a 1.8 GB `PUT`.** A transfer outlasts any job budget, so `put_stream`
  re-arms `arm_watchdog(180)` every 16 MiB. A stalled link resets the box within 180 s and it
  comes back (BootNext spent → Debian) instead of hanging with the watchdog disabled. The idle
  `arm_watchdog(0)` is restored on every exit path.
- **Half-written files.** `PUT` never writes the target. It writes `<NAME>.PART`, flushes,
  closes, re-opens, streams a sha over the readback, and only then deletes the target and
  renames. A crash anywhere leaves the old target intact and a `.PART` visible to `LS`.
  `RELOAD` refuses while any `.PART` exists (`ERR busy-file`).
- **Power loss during `PUT`.** Same story; the scheduler additionally treats a box that
  reappears with a `.PART` as `SUSPECT` until `cm-os-fs rm` clears it. FAT32 has no journal, so a
  torn directory entry is possible — recovery is `fsck.vfat` from Debian (§9), never in-kernel.
- **A box that disappears mid-shard.** Lease deadline expires; shard re-leased with the same
  `RUNID`; the lost box, if it returns, is deduped by the ring or contradicted by digest.
  Its `RESULT.TXT` is harvested by the existing Debian harvest service and reconciled by
  `run_id` — a late record never overwrites a completed unit, it is filed as `LATE`.
- **Hangs the watchdog cannot catch.** A hang inside firmware (an SNP transmit that never
  completes) with the watchdog disabled is the one unrecoverable class. Mitigation is unchanged
  from `UEFI-REMOTE-LANE.md`: the smart plug, and `BOXES.md` saying `power-cycle: human` until
  one exists. The scheduler prints that per box before a pool run.
- **`RESULT.WIP`.** Unchanged and now fleet-visible: a box whose `LS` shows `RESULT.WIP`
  did not finish its last job, and `HEALTH`'s `last=` says so before any work is sent.
- **The vvfat caveat** (guest unlinks not committed to the host mirror) means `files-test`
  asserts the *guest's* `LS`, never the host directory.

---

## 9. What is explicitly left to the Debian side

Compilation of anything (the unikernel grows no compiler); `cm-build` sonnet builders;
torch/DDP CPU training (phase 7); TLS, DNS, NTP; large-file *distribution* (rsync/torrent
of `MODEL.SAF` to the `ALICE_UEFI` partition — `PUT` is for a box already up, not for
provisioning 8 boxes); `fsck.vfat` and partition repair; power-cycling and `efibootmgr`;
log rotation and long-term storage; anything needing more than 8.3 filenames; and every
recovery path once a box is `DOWN`. The unikernel stays small on purpose.

---

## 10. Sequencing and the first five commits

Order: **4 → 5 → 6**, each a branch off `os/aefinity-os-v0.1`, gate green before the
next starts. Phase 4 first: `PUT`/`RELOAD` removes the physical stick from the loop, and
every later phase's iteration cost drops behind it.

1. `os/p4-files: net: tcp_recv_exact/tcp_send_slice + DATA framing` — `net/mod.rs`
   only, plus a unit test on the frame parser. No protocol surface yet.
2. `os/p4-files: files.rs — name validation, ls/stat/sha over the boot volume` —
   read-only verbs, `LS`/`STAT`/`SHA` wired into `server.rs`. Gate: first half of
   `files-test`.
3. `os/p4-files: GET + PUT with .PART staging and readback verify` — the write path,
   watchdog re-arm, free-space check, `RM`. Gate: `files-test` complete.
4. `os/p4-files: RELOAD against the boot-time tensor arena` — `main.rs` arena handle,
   `model_sha` in `HEALTH` and `RESULT.TXT`. Gate: `files-test` + `boot-test` still 33.
5. `cm/aefinity-fleet: bin/cm-os-fs + HEALTH/RUNID client and BOXES.md health column` —
   the host half, unittest against a fake server, so phase 5 is developed against a
   controller that can already push a corpus to a box.

Then `os/p5-lab` (lab.rs, `lab-test`), then `os/p6-pool` (host-only scheduler,
`pool-test`), then the two-box `AGREE`/`DISAGREE` run that the digest was built for —
which needs iron, and is Justin's step.
