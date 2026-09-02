# AEFINITY OS phases 4–6 — SYSTEMS-FRONTIER design

*Angle: push what a no-OS fleet can do. Remote model swap + `RELOAD` without reboot,
streamed tokens, box-to-box collective reduction over our own smoltcp, a Rust
`aefinity-ctl`, and an honest path to pooled training. Written against the v0.1 code in
`~/projects/alice-aegis-os-int` (`server.rs`, `job.rs`, `net/mod.rs`), not against the spec
alone. Rule A binds every gate: structure and exit codes only.*

**The load-bearing observation.** v0.1 refuses `REPORT` inside a socket-delivered job because
`run_job` would need a second `Net` and the listener holds the SNP exclusively. That is not a
network limitation — it is a *layering* one. `server::run` **owns** the `Net` by value and can
`tcp_connect` on it. So every outbound thing a resident box wants to do (report, stream, reduce
with a peer) belongs in `server.rs`, above `run_job`, using the listener's own stack. Phases 4–6
are built on that one move.

---

## 1. Protocol additions to §4

Banner becomes (deliberate break; the only v0.1 client is `xtask`, updated in the same commit):

```
AEFINITY-OS 0.2 READY env=<iron|vm> cpu=<brand> caps=files,lab,pool\n
```

Commands are ASCII, CRLF or LF, one per line, keyword ASCII-case-insensitive, arguments
case-sensitive. `Lines` in `server.rs` already gives this; binary payloads are read with
`Until::Len(n)` on the same reader (drain `Lines::pending` first).

```
C: AUTH <64 hex>\n                       S: OK\n | ERR auth\n
C: LS\n                                  S: LS <n>\n {<name> <size>\n}*n END\n
C: SHA <name>\n                          S: SHA <64 hex> <size>\n
C: GET <name>\n                          S: DATA <len>\n<len bytes>END\n
C: PUT <name> <len> <64 hex>\n           S: SEND\n → C: <len bytes> → S: OK <64 hex> <len>\n
C: RM <name>\n                           S: OK\n
C: RELOAD\n                              S: RELOADING\n … S: OK model=<16 hex> embed=<16 hex> vocab=<16 hex>\n
C: STREAM on|off\n                       S: OK\n
C: POOL <epoch> <rank> <size> <next> <op> <len>\n<len bytes>END\n
                                         S: POOLED <epoch> <op> <len>\n<len bytes>END\n
```

- **Names**: `[A-Z0-9_][A-Z0-9_.-]{0,11}`, no `/` or `\`, root of the boot volume only. This is
  what `write_named`'s 32-`u16` buffer already assumes. Violation → `ERR badarg`.
- **Sizes**: `LS` ≤ 128 entries then `END`. `GET` ≤ 64 MiB, streamed from the file in 64 KiB
  chunks (never buffered whole). `PUT` ≤ 2 GiB (`MODEL.SAF` is 1.83 GB), streamed to disk in
  64 KiB chunks, sha256 computed incrementally with `aegis_core::witness::Sha256`. `POOL`
  payload ≤ 64 KiB. Command line cap stays `LINE_MAX_BYTES`; body cap stays 64 KiB.
- **AUTH**: `JOB.TXT` may carry `TOKEN <64 hex>`. If present, the mutating verbs
  (`PUT`/`RM`/`RELOAD`/`REBOOT`/`HALT`/`JOB`) return `ERR auth` until a constant-time-compared
  `AUTH` succeeds on that connection; absent `TOKEN`, everything is open (v0.1 behaviour). A
  fleet accepting unauthenticated remote weight swaps is a footgun. Sixteen lines, not a
  security programme, and no substitute for a trusted segment.
- **STREAM on**: during a subsequent `JOB`, each generated token is emitted as
  `TOK <step> <id> <escaped-piece>\n` before `RESULT`. Escaping is `job::escape` (already
  exists), piece capped at 64 bytes. Backpressure policy: 250 ms send timeout; on any error
  streaming is disabled for the rest of the job, the job **continues**, and the record carries
  `job.N.stream=dropped`. A peer that stops reading must never wedge a box.
- **Error strings**, byte-exact, lowercase, one table shared by both sides:
  `ERR unknown` `ERR too-large` `ERR badarg` `ERR auth` `ERR notfound` `ERR io` `ERR digest`
  `ERR nospace` `ERR reload-size` `ERR reload-engine` `ERR reload-missing` `ERR peer`
  `ERR timeout` `ERR pool-op` `ERR pool-rank`. `BUSY\n` is unchanged and is not an `ERR`.
- **POOL**: `<next>` is `host:port` or `-` for the last rank. `<op>` ∈ `xor32 | sum64 | min64 |
  max64`. Payload is a vector of LE `u64` (`xor32` = LE `u32`), length a multiple of the element
  size. Semantics: rank *r* folds the received vector with its own local vector for `<epoch>`,
  forwards to `<next>`; the last rank sends the result back down the chain it arrived on. A
  linear chain, not a ring: no leader election, and at fleet sizes ≤ 8 the hop count is the
  honest cost. Local vectors are produced by the preceding `JOB` (`job.N.digest` as four LE u32
  for `xor32`; `LOGITS` counters for `sum64`). Peer connect/send/recv timeout 30 s → `ERR peer`.

## 2. `JOB.TXT` additions to §2

```
TOKEN <64 hex>              # optional shared secret for the mutating verbs
STREAM on                   # resident default for tokens (client STREAM overrides)
EVAL <name> <lo>:<hi>       # perplexity over token slice [lo,hi) of a corpus file
VERIFY <name>               # verifier.rs against a receipt on the volume
LOGITS <k>                  # per-step top-k logit statistics, k ≤ 16
CPUID                       # feature/leaf dump, no measurement
MEMBW <mib>                 # deterministic touch pattern; rate is recorded, never trusted under vm
MECH <n>                    # existing diagnostic; NEVER in a gate (~24 min under TCG)
POOLVEC <op>                # publish this job's fold vector for a later POOL epoch
```

`Step` gains `Eval{name,lo,hi}`, `Verify{name}`, `Logits{k}`, `Cpuid`, `MemBw{mib}`, `Mech{n}`,
`PoolVec{op}`. `MODE`/`LISTEN`/`NET`/`TOKEN` stay ignored in a socket-delivered body (§4 rule,
already enforced in `do_job`).

## 3. `RESULT.TXT` additions to §3

```
aefinity_os=0.2                     # collectors accept 0.1|0.2
artifacts=<model16>/<embed16>/<vocab16>    # sha256-16 of the three loaded buffers
reloads=<n>                         # how many RELOADs since boot — a reloaded box is not a fresh box
job.N.kind=eval|verify|logits|cpuid|membw|mech|poolvec
job.N.corpus=WIKI.BIN
job.N.slice=0:256
job.N.ntok=256
job.N.nll_q16=<u64>                 # sum of -log p in Q16 fixed point. Integer: bit-comparable across boxes
job.N.logit_digest=<16 hex>
job.N.pass=true|false               # verify
job.N.rate_valid=true|false         # false whenever env=vm — Rule A enforced by the artifact
job.N.stream=sent|dropped|off
pool.epoch=<n>  pool.rank=<r>  pool.size=<s>  pool.op=<op>  pool.digest=<16 hex>
```

`nll_q16` as an integer is deliberate: perplexity as a float is not bit-comparable across boxes,
and cross-box *agreement* is the whole point of the fleet.

## 4. Unikernel modules touched (§5 map)

| file | change | LOC |
|---|---|---|
| `server.rs` | verb dispatch for `AUTH/LS/SHA/GET/PUT/RM/RELOAD/STREAM/POOL`; binary read/write on the existing `Lines`; token sink wired into `do_job`; per-verb watchdog | +650 |
| `files.rs` (new, phase 4) | FAT32 helpers: `list`, `sha_file` (64 KiB streaming), `read_chunks`, `write_streaming` to `<NAME>.PRT`, `rename`, `CURRENT.TXT` pointer read/write, stale-`.PRT` sweep | +320 |
| `reload.rs` (new, phase 4) | `Slabs` (raw ptr + len + capacity pages for the three buffers, moved out of `main.rs`), `reload(&mut Slabs, root) -> Result<Engine, ReloadErr>` | +180 |
| `main.rs` | build `Slabs` instead of bare slices, pass to `job::dispatch`; sweep stale `.PRT`; honour `CURRENT.TXT`. Still one hook, still before MECH | +60 |
| `job.rs` | new `Step` variants + parse + `StepResult` fields + `render`; `TokenSink` trait threaded into the decode loop; `rate_valid` | +400 |
| `lab.rs` (new, phase 5) | `EVAL` (streamed corpus, Q16 NLL), `LOGITS`, `CPUID`, `MEMBW`; `VERIFY` is a 20-line call into the existing `verifier::run` | +380 |
| `pool.rs` (new, phase 6) | fold ops, chain forward/return over the server's `Net`, epoch state | +260 |
| `net/mod.rs` | none expected; `Until::Len` and `tcp_connect` already exist | ~0 |

`RELOAD` is the one genuinely dangerous addition. Constraints: the buffers are
`allocate_huge_pages` slabs sized at boot, and `TernaryInferenceEngine` borrows them, so a
reload must drop the engine first. Therefore: if the new file exceeds the slab capacity →
`ERR reload-size` (no state touched). Otherwise drop engine, `load_file_into` the slab,
`Engine::new`; on failure the RAM copy is now junk, so the box answers `ERR reload-engine`,
logs it, and calls `ResetSystem(COLD)` — which lands in Debian via `BootNext`, a known state.
`RELOAD` never leaves a box serving with an engine it cannot describe.

## 5. Host tooling

`aefinity-ctl` — new Rust crate in `alice-aegis` (own `Cargo.toml`, added to
`scripts/devloop.sh`'s crate list; there is no root workspace by design). `std::net` only, no
async, ~1200 LOC. Verbs mirror §1: `ping`, `job [--stream]`, `ls`, `get`, `put`, `sha`,
`reload`, `pool`, `status`, plus `--boxes state/BOXES.md`, `--matrix`, `--pool`. No Python, no
venv — it runs from any box.

`claudius-maximus` keeps the operational surface: `cm-os-job` gains `--pool N`,
`--replicate K`, `--shard-file`; `cm-os-swap <box> MODEL.SAF` = `put` + `sha` + `reload` + a
one-token smoke `JOB`, refusing if `artifacts=` does not change. `cm-os-fleet status` renders
the ledger view.

**The collector stays dumb and append-only.** It receives POSTs (oneshot lane) and shard results
pushed by `aefinity-ctl` (resident lane), writes
`state/reports/uefi/<pool_id>/shard-<NNNN>/RESULT.TXT` as `tmp` + `rename`, appends one ledger
row per shard carrying `env=` and `rate_valid=`, and writes `POOL.json` (merge state) the same
way. It computes the merge but decides nothing: it never averages, never retries, never
schedules. Scheduling is the ctl's job; adjudication is a human's.

## 6. POOL scheduler semantics

- **Unit**: one shard = a directive list. `unit_digest = sha256(canonical directive text)` —
  trailing whitespace stripped, LF, keywords upper-cased. Deterministic and content-addressed.
- **Sharding**: prompts / eval slices / receipt batches split by count into `N` units;
  `--replicate K` emits each unit `K` times with distinct target boxes.
- **Leases**: a unit is leased to a box for `BUDGET + 90 s`. Lease expiry ⇒ requeue elsewhere.
  Delivery is at-least-once and every v0.2 unit is a pure function of (artifacts, directives),
  so re-execution is safe. Max 3 attempts, then the unit is `unassigned` and named in `NEEDS.md`.
- **Partial results are the normal case.** `POOL.json` always carries
  `complete=<k>/<n> failed=<f> agree=<m>/<r>`; a pool is publishable partial, and the merge is
  defined over whatever completed.
- **Digest merge**: dedupe by `shard_id` (first success wins), then
  `pool_digest = XOR over shards of sha256(shard_id_le32 || unit_digest || step_digest)`, first
  16 hex. Order-independent, which matters because shards land out of order; duplicate-sensitive,
  which is why dedupe precedes the fold.
- **Agreement** means: for one `unit_digest`, two boxes with *different* `cpuid_sig` produced
  identical `job.N.digest` (and identical `nll_q16` for `EVAL`). Only that is agreement. Equal
  `tps` is not agreement; unequal `tps` is not disagreement. A digest mismatch is a **finding** —
  recorded with both records, never averaged away, never retried into silence.

## 7. Iron safety

- **Watchdog**: idle 0 (unchanged). `PUT` re-arms 120 s **per chunk** — a transfer that stalls is
  a hang; a slow but live transfer is not. `RELOAD` arms 180 s. `POOL` arms `budget+60`. Every
  verb disarms on return. `arm_watchdog(0)` before any park, as today.
- **Half-written files**: `PUT` writes `<NAME>.PRT`, verifies sha256, then swaps. FAT32 rename is
  a directory-entry write and is not atomic against power loss, so the swap is a **pointer swap**:
  `CURRENT.TXT` holds `model=MODEL.SAF` or `model=MODEL.NEW`, and boot reads it, falling back to
  `MODEL.SAF` when absent or unparsable. Writing one ≤64-byte file is the smallest commit
  available on this filesystem. Power loss before it ⇒ old weights; after ⇒ new; during ⇒
  fallback. `.PRT` files are swept at boot and logged.
- **Power loss mid-`PUT`**: the target is never opened; worst case is a stale `.PRT`.
- **Box disappears mid-shard**: lease expires, unit requeues. A zombie box may finish and report
  later; the collector dedupes on `(pool_id, shard_id)` and flags a *conflicting* duplicate
  (different `result_digest`) as a finding rather than dropping it. Crucially, a box whose
  watchdog fires **leaves the fleet**: firmware reset consumes `BootNext` and lands in Debian, so
  the scheduler must treat "reset" as departure and let the Debian side re-launch via
  `cm-uefi-run`. There is no self-healing inside the unikernel and there should not be.
- **Hangs**: `IDLE_DROP_MS` already bounds a silent connection. `POOL` adds a peer timeout so a
  missing neighbour cannot hold a rank. Every failure path in `server.rs` still ends at `accept`.

## 8. Explicitly left to the Debian side

Compilation and builds (`cm-build`; the unikernel has no compiler and must not grow one).
Gradient training (torch/DDP over ssh). Model prep and quantization (`aegis-forge`). Corpus
tokenization — the unikernel reads pre-tokenized `u32` corpora only. Staging artifacts over the
`PUT` cap. TLS, WAN exposure, off-segment auth. Power cycling, boot order, harvest, long-term
storage. **Pooled training,
honestly**: the fleet can do gradient-*free* optimisation today — evolution strategies need only
forward passes and a seed, so each box scores a perturbation `θ + σ·ε(seed)` over an `EVAL`
slice, returns `nll_q16`, and the host applies the update and pushes new weights with
`PUT`+`RELOAD`. That is a real distributed training loop with no backward pass anywhere. It is
also sample-inefficient, and calling it "pooled training" without that sentence would be a lie.
A backward pass in `aegis-core` is a research track; do not block phases 4–6 on it.

## 9. Sequencing and the first five commits

Order: 4 → 5 → 6a (host-rooted star reduction, reusing existing verbs, zero unikernel risk) →
6b (`POOL` chain). Each a branch `os/p4-files`, `os/p5-lab`, `os/p6-pool` with one gate.

**Gates** (structure and exit codes only, all under `flock`, none touching `MECH`):
- `files-test` — resident guest; harness does `LS` (asserts a staged name appears), `SHA` (equals
  host-computed sha256), `GET` (bytes equal), `PUT` 3 KiB (`OK` + digest echo), `PUT` with a
  wrong digest (`ERR digest` exactly), `RM`, `RELOAD` (`OK model=…` with the same three digests),
  then a `JOB` that still answers `verdict=OK`. Asserts on the guest's `SHA`, never on the host
  mirror — vvfat does not commit guest unlinks.
- `lab-test` — `CPUID`, `VERIFY GOOD.RCP`, `VERIFY BAD.RCP`, `EVAL TINY.BIN 0:4`. Asserts
  `job.2.pass=true`, `job.3.pass=false`, `job.4.nll_q16` present and parses as `u64`,
  `rate_valid=false`, `env=vm`. No value is asserted.
- `pool-test` — two guests on `-netdev socket,mcast=…` (a shared L2, no host stack), `-m 1536`
  each, ranks 0/1, `xor32` over two known payloads; asserts `POOLED` bytes equal the
  host-computed XOR and both QEMUs exit.

First five commits:
1. `p4: move the three artifact slabs into reload::Slabs` — pure refactor of `main.rs`,
   `boot-test` still 33. Nothing else can start until the buffers are addressable after boot.
2. `p4: files.rs — LS/SHA/GET over the boot volume` + read-only verbs in `server.rs`; gate
   `files-test` asserting `LS`/`SHA`/`GET` only.
3. `p4: PUT with .PRT staging, CURRENT.TXT pointer swap, stale sweep`; extend `files-test` with
   the good and the `ERR digest` transfers.
4. `p4: RELOAD` — drop engine, reload slab, rebuild, `reloads=` in the record, reset-on-failure;
   extend `files-test` with `RELOAD` + a following `JOB`.
5. `p4: AUTH + TOKEN, error-string table, banner 0.2 + caps=`; update `resident-test`'s banner
   assertion in the same commit.

Then phase 5 opens with `job.rs` `Step` variants and `lab.rs`, and phase 6 opens with the
`TokenSink`/streaming commit — because streaming is the smallest thing that proves the
server-owns-the-`Net` layering works before `POOL` depends on it.
