# AEFINITY OS phases 4–6 — FILES, LAB, POOL

*Build contract v0.3-draft — 2026-09-01. Extends `program/AEFINITY_OS.md` (spec v0.1)
§2–§7 without breaking any of it; referenced from spec §9. Project owner: Justin B.
Thompson. Builder/operator: Claudius Maximus.*

Written against the shipped v0.1 code in `~/projects/alice-aegis-os-int` (`job.rs` 1425 LOC,
`main.rs` 1884, `net/mod.rs` 1475, `server.rs` 757), not the spec text alone. Every line
number below was re-checked against the current tree on 2026-09-01. **Rule A binds every
gate: structure and exit codes only, never a value.**

## 0. One paragraph
Phases 4–6 turn a resident box into something you can fill, ask and read without touching
the stick — and turn N of them into one lab. Phase 4 (FILES) adds a file plane plus health
and idempotency verbs. Phase 5 (LAB) adds no protocol: the lab suites become `JOB.TXT`
directives. Phase 6 (POOL) adds **no unikernel code at all** — a host-side scheduler over
verbs phases 2 and 4 already shipped. The v0.1 principle holds: the unikernel is a stateless
worker, the controller is the source of truth, Debian is the recovery partition and the
compiler.

**Honest scope.** POOL is an eval / determinism / verification harness. It is **not
progress on model training**: no gradient is computed anywhere in phases 4–6, no weights are
learned, and a green pool run improves no benchmark. Its value is that a number produced on
one box can be *reproduced* on another, and that a disagreement becomes a filed finding
instead of noise. The evolution-strategies path in §9 is **research** — unvalidated, and
labelled as such everywhere it appears.

---

## 1. Protocol additions (§4-ext)

The banner keeps its version and gains one token, so every v0.1 client assertion still
passes:

```
S: AEFINITY-OS 0.1 READY env=<iron|vm> cpu=<brand> caps=files,lab\n
```

One framing primitive and nine verbs. `PING`/`JOB`/`REBOOT`/`HALT`, `BUSY\n`, `ERR
unknown\n`, `ERR too-large\n`, `LINE_MAX_BYTES` (`server.rs:68`, = `BODY_MAX_BYTES` =
64 KiB, `server.rs:57`) and the one-client-at-a-time rule are unchanged.

### 1.1 The `DATA` frame (the only binary on the wire)

```
DATA <len> <sha16>\n<exactly len raw bytes>END\n
```

`<len>` decimal ASCII, no leading zeros; `<sha16>` = first 16 lowercase hex of the sha256
over those bytes. No base64 (33 % inflation, no encoder in the unikernel). The length is the
frame; the trailing `END\n` is a **resync marker, not a delimiter**. `Lines`
(`server.rs:583`, `pending: Vec<u8>`) gains `take_bytes(n)`: drain `pending`, then read
through `Net::tcp_recv_until(Until::Len(k), …)` — `Until::Len` exists today under the
`#[allow(dead_code)]` at `net/mod.rs:180` (enum at `:181`, `Len` at `:184`), put there for
exactly this caller.

### 1.2 Phase 4 verbs

```
C: AUTH <64hex>\n                S: OK\n | ERR auth\n
C: LS\n                          S: LS <n> <ok|truncated>\n {<NAME> <size>\n}*n END\n
C: STAT <NAME>\n                 S: STAT <NAME> <size>\n | ERR not-found\n
C: SHA <NAME>\n                  S: SHA <NAME> <size> <64hex>\n | ERR <e>\n
C: GET <NAME>\n                  S: DATA <len> <sha16>\n<bytes>END\n | ERR <e>\n
C: PUT <NAME> <len> <64hex>\n    S: SEND\n | ERR <e>\n
C: <exactly len raw bytes>END\n  S: OK <NAME> <len> <sha16>\n | ERR <e>\n
C: RM <NAME>\n                   S: OK <NAME>\n | ERR <e>\n
C: RELOAD\n                      S: RELOADING\n … S: OK reload model=<sha16> embed=<sha16> vocab=<sha16>\n | ERR <e>\n
C: HEALTH\n                      S: HEALTH up=<s> served=<n> last=<OK|FAIL <r>|none> wd=<off|<s>> \
                                    heapfree=<bytes> model=<sha16> reloads=<n> parts=<0|1> env=<iron|vm>\n
C: RUNID <id>\n                  S: NEW\n | REPLAY\n
```

**`PUT` is declare-then-stream**: the server pre-checks name, length and free space
*before* the client burns 1.8 GB on the wire, and the full 64-hex digest is what lets the
operator prove what landed.

**`AUTH`.** `JOB.TXT` may carry `TOKEN <64hex>`. When it is present **every verb except
`PING` and `AUTH` answers `ERR auth`** until a constant-time-compared `AUTH` succeeds on
that connection — reads included, because `GET MODEL.SAF` is exfiltration and a `HEALTH`
reply publishes artifact digests. (An earlier draft left reads open; that was wrong.)
Absent `TOKEN`, everything is open — exact v0.1 behaviour, so no v0.1 gate changes.

**`RUNID` is the at-most-once primitive**, the most important addition for a fleet. The
server rings the last `RECORD_MAX = 8` `(run_id, rendered RESULT.TXT)` pairs; a known id
answers `REPLAY`, the following `JOB … END` is **drained but not run**, and the cached
record comes back as `RESULT\n<body>END\n` with `replay=true`. A controller whose TCP died
after `RUNNING` but before `RESULT` retries blind without double-spending a 20-minute shard.
`<id>` is `[A-Za-z0-9._-]{1,64}`; anything else is `ERR bad-runid`. **`HEALTH`** is read
*before* dispatching: a `model=` mismatch, `parts=1` or `last=FAIL` are reasons not to send
work.

### 1.3 Caps and error strings (byte-exact)

| constant | value | why |
|---|---|---|
| `NAME_MAX_BYTES` | **31** | `write_named` builds a `CStr16` in `[0u16; 32]` (`job.rs:695`, `from_str_with_buf` at `:696`); the NUL takes one unit, so 31 ASCII bytes is the true ceiling |
| `LS_MAX_ENTRIES` | 256 | one bounded FAT32 root listing, then `END` |
| `PUT_MAX_BYTES` | 2 147 483 648 | 2 GiB; `MODEL.SAF` is 1.83 G |
| `GET_MAX_BYTES` | 2 147 483 648 | framed, streamed, never buffered whole |
| `XFER_CHUNK` | 65 536 | equals `RECV_MAX_BYTES` (`net/mod.rs:125`), the cap on one `tcp_recv_until` |
| `XFER_STALL_MS` | 30 000 | bounds *no progress*, not total duration |
| `FILES_WD_S` | 300 | watchdog during transfer **and readback**, re-armed per chunk |
| `RECORD_MAX` | 8 | retained records |
| `RECORD_MAX_BYTES` | 8 192 | per retained record ⇒ ring ≤ 64 KiB of heap |
| `STAGE_NAME` | `STAGE.PRT` | the one fixed 8.3 staging name |
| `EVAL_WINDOW` | 2 048 | tokens; see §2.1 |

`<NAME>`: 1–31 bytes of `[A-Z0-9._-]`, upper-cased, not starting `.`, no `/` `\` `:` `..`,
boot-volume root only — no traversal, because there is no directory syntax. `BOOTLOG.TXT`,
`RESULT.TXT`, `RESULT.WIP` (`job.rs:669`) and `STAGE.PRT` are readable but never writable
(`ERR protected`).

Staging is **one fixed name, `STAGE.PRT`**, not `<NAME>.PART`: a per-name suffix breaks 8.3
and can exceed 31 bytes for a legal target. Only one PUT can be in flight anyway (one client
at a time), so `sweep_parts()` becomes "delete `STAGE.PRT` at boot" and `HEALTH parts=` is
0 or 1.

New `ERR` slugs, exhaustive: `bad-name`, `bad-args`, `bad-len`, `bad-runid`, `bad-frame`,
`bad-corpus`, `not-found`, `exists`, `protected`, `no-space`, `digest-mismatch`,
`short-write`, `short-read`, `busy-file`, `reload-size`, `reload-engine`, `auth`, `io`,
`fw-error`, plus v0.1's `unknown` and `too-large`. A slug not on this list means the box is
unhealthy, not that the job failed. **Invariant: no verb ever returns a partial success.**

### 1.4 Abort and failure table (normative)

| condition | wire | connection | on-disk |
|---|---|---|---|
| `DATA`/PUT payload arrives, but the trailing 4 bytes are not `END\n`, or the readback digest ≠ the declared `<64hex>` | `ERR bad-frame` / `ERR digest-mismatch` — **after** the full `len` bytes are drained, **before** any commit | kept | `STAGE.PRT` deleted; target untouched |
| PUT header itself is malformed (`<len>` unparsable, > `PUT_MAX_BYTES`, digest not 64 hex) | `ERR bad-len` / `ERR bad-args` before `SEND` | kept | nothing written |
| PUT payload declared well-formed but the stream desynchronises (peer sends fewer bytes then closes) | `ERR bad-frame` best-effort | **closed** — resync is impossible once the byte count is unknown | `STAGE.PRT` deleted |
| PUT makes no progress for `XFER_STALL_MS` | `ERR io` best-effort (the send may itself fail) | **closed** | `STAGE.PRT` deleted; server disarms the watchdog, clears BUSY, returns to `accept` — **it does not reboot** |
| GET fails after the header is on the wire (short FAT read, send timeout) | nothing — a truncated frame is never emitted | **closed immediately** | none |
| free space exhausts mid-PUT | `ERR io` (see §8 on why not always `no-space`) | closed | `STAGE.PRT` deleted |

**GET reads the file twice.** The header `sha16` is computed by a full read pass *before*
the header is sent, so a header that arrives is a promise the bytes were readable once.
After that the server cannot retract, and the only honest failure is to close; the client
sees a short read with no `END\n`.

**`REPLAY` sequencing.** After `REPLAY`, the client **still sends `JOB <len>\n<body>END\n`**.
The server drains and discards it (bounded by `BODY_MAX_BYTES`, so the drain is cheap),
then answers the cached record. One code path reads `JOB` in every case, and the stream is
never left half-framed. A client that skips the body is answered `ERR unknown` for whatever
it sends next.

**The `RUNID` ring is RAM only.** A watchdog reset, `REBOOT`, or power loss empties it, so a
re-issued id afterwards answers `NEW` and **the job runs a second time**. At-most-once holds
*within one box uptime*, nothing more. Host-side rule: the scheduler records `up=` and
`served=` from the `HEALTH` taken at lease time; if a later `HEALTH` shows `up=` decreased,
the ring is presumed lost, the unit is re-run, and any duplicate record is reconciled by
`run_id` with the larger `uptime_s` filed `LATE` rather than overwriting the completed unit.

**`RECORD_MAX_BYTES`.** A rendered record above 8 192 bytes (reachable only via many steps'
`job.N.detail`) is retained truncated at that bound with a final line `truncated=true`. The
job is still not re-run — at-most-once is preserved, and the replay is honest about being
incomplete.

**`LS`.** `<n>` counts the entry lines that follow and **does not count `END`**.
Directories, volume-label and `.`/`..` entries are skipped and not counted. More than
`LS_MAX_ENTRIES` real entries ⇒ the first 256 are listed and the header reads
`LS 256 truncated`; it is never an error, because a box with 300 files is still usable.

---

## 2. `JOB.TXT` additions (§2-ext)

```
TOKEN <64hex>       # optional shared secret gating every verb but PING/AUTH
RUNID <id>          # idempotency key; same grammar as the RUNID verb
TAG <text>          # free-form, echoed into RESULT and the ledger
SHARD <i>/<n>       # this box's slice of a pool run; 1 ≤ i ≤ n ≤ 4096
SEED <u64>          # sampler seed; makes digest comparison meaningful
STRICT on           # any step failure ⇒ stop, verdict=FAIL <step-kind>
CPUID               # identity/leaf dump; no measurement
VERIFY <NAME>       # replay a receipt through verifier::run (verifier.rs:112)
EVAL <NAME> <lo>:<hi>   # integer NLL over token slice [lo,hi) of a corpus on the volume
MEMBW <mib>         # deterministic touch pattern over <mib> MiB
MECH                # the existing diagnostic block, as a directive
```

`SHARD` is informational to the box and load-bearing to the collector: the box never
computes its own slice — the controller writes concrete work into the body — so a record
can be attributed without trusting the collector's bookkeeping alone. `EVAL`/`VERIFY` take
a `<NAME>` under §1.3 rules, so a corpus or receipt arrives by `PUT` and LAB composes with
FILES without either knowing about the other.

**`MECH` is moved last by the dispatcher regardless of file order** — it runs ~24 min under
TCG and must never starve real work; the same reasoning put v0.1's job hook before the MECH
block in `main.rs`.

### 2.1 `EVAL`, fully specified

**Corpus container (`.BIN`, 64-byte header, all integers little-endian):**

| off | size | field |
|---|---|---|
| 0 | 8 | magic, ASCII `AEFCORP1` (`41 45 46 43 4F 52 50 31`) |
| 8 | 4 | `version` u32 = 1 |
| 12 | 4 | `token_width` u32 = 4 (u32 ids; no other width is accepted) |
| 16 | 8 | `ntok` u64 |
| 24 | 4 | `vocab_size` u32 |
| 28 | 4 | reserved u32 = 0 |
| 32 | 32 | sha256 of the payload |
| 64 | 4·`ntok` | token ids, u32 LE |

Validation, in order, all ⇒ `ERR bad-corpus`: magic; `version == 1`; `token_width == 4`;
`file_size == 64 + 4·ntok`; `vocab_size == engine.config.vocab_size`; payload sha256 match
and every id `< vocab_size`, both in one streamed pass, once per `EVAL` step, before any
forward pass.

**Window and stride.** `W = min(EVAL_WINDOW, config.max_position_embeddings)`
(`model.rs:42`). The slice `[lo, hi)` is cut into **non-overlapping** windows of `W` tokens;
stride **= W**. Each window resets the KV cache (`reset_prefix`) and runs positions `0..len`;
the first token of a window is context, not a scored position, so a window contributes
`len − 1` scored positions. A final short window of `< 2` tokens is dropped. `ntok` in the
record is the total scored positions. A sliding stride would score some positions with more
context than others and make the number depend on a parameter nobody would remember, so
there is no stride knob.

**`nll_q16` is defined over the CIS-1 full-integer path and is bit-exact by construction.**
It does **not** call `CisEngine::calculate_perplexity_int` (`cis_infer.rs:1315`) — that
function computes its NLL with `libm::exp`/`libm::log` on f64 and is *not* cross-box
comparable. EVAL reuses that function's exact-integer half and replaces its float half:

1. `logits_int()` (`cis_infer.rs:1291`) fills `logits: &[i64]` — exact integers — and yields
   the logit unit as an **exact rational** `ActScale {num, den}` at Q.`F` (`F = 20`,
   `cis_infer.rs:47`). EVAL takes the rational, never the `f64` the function currently
   returns.
2. `m = max(logits)` (exact i64 compare). Gaps `d_j = m − L_j ≥ 0`.
3. `g_j = qs.rescale(d_j)` into the Q.24 score grid (`SCORE_F = 24`, `cis_infer.rs:53`),
   where `qs = QScale64::from_ratio(...)` (`cis_infer.rs:209`) is built from that exact
   rational by exact long division with round-to-nearest-even. No f32 or f64 anywhere.
4. `e_j = exp_neg_q31(g_j << 8, lut)` (`cis_attn.rs:135`) — literally SOFTMAX-I step 2
   (`cis_attn.rs:173`), Q0.31, the max element contributing exactly `2^31`.
5. `S = Σ e_j` in i64, ascending vocabulary index. Exact: `V = 128 256 < 2^20` and
   `V · 2^31 < 2^51`.
6. `nll_t` in Q.32 nats `= (g_t << 8) + rne_div(log2_u64_q32(S) − (31 << 32) << 32,
   LOG2E_Q32)`, where the first term is the target's gap promoted Q.24→Q.32 by an exact
   shift, and the second is `ln(S / 2^31)` obtained from a new **LOG2-I over u64** — the
   same normative shift-and-square procedure as `log2_q32_f32` (`cis_attn.rs:208`) applied
   to an integer mantissa — divided by the pinned `LOG2E_Q32 = 6196328019`
   (`cis_attn.rs:37`) with `rne_div`.
7. **Accumulation order is fixed**: `Σ nll_t` over scored positions in ascending
   (window, position) order, held in `i128` at Q.32, with exactly **one** final rounding:
   `nll_q16 = rne_div(total_q32, 1 << 16)` as u64.

Two consequences. **Underflow is declared, not accidental**: a logit more than ~21.49 nats
below the max contributes `e_j = 0` exactly (the `n ≥ 31` early return in `exp_neg_q31`) —
identical on every box, and since step 6 never divides by `e_t`, a target in that tail still
gets a finite, correct-by-definition NLL. **LOG2-I is new normative code** (~40 LOC in
`cis_attn.rs`) and must land with goldens from the independent big-integer generator
`scripts/cis_e2_golden_gen.py`, never from the Rust under test — the rule the rest of
`cis_infer.rs`'s goldens already follow.

Perplexity itself is `exp(nll_q16 / 2^16 / ntok)`, rendered **host-side** for humans. It is
an f64 render of two integers; the integers are the record.

---

## 3. `RESULT.TXT` additions (§3-ext)

Appended after the existing keys; v0.1 key order is never disturbed.

```
run_id=pool-2026-09-04-a.s03
tag=mech-sweep      shard=3/8      seed=12345
replay=false                       # true ⇒ served from the RUNID ring, not re-run
artifacts=<model16>/<embed16>/<vocab16>    # sha256-16 of the three resident buffers
reloads=<n>
uptime_s=<s>   served=<n>   files=<n>
job.N.kind=cpuid|verify|eval|membw|mech
job.N.rate_valid=true|false        # false whenever env=vm — Rule A enforced by the artifact
job.N.nll_q16=<u64>   job.N.ntok=<n>       # EVAL only; §2.1
job.N.digest=<64hex>               # WitnessChain over exact i64 logits (witness.rs:189)
job.N.membw_mibs=<n|n/a>           # gated exactly like tps: n/a unless rate_valid=true
job.N.pass=<true|false>   job.N.items=<n>
job.N.partial=<0|k>                # k of the requested count actually completed
job.N.err=<none|short slug>
job.N.detail=<escaped, ≤1024 bytes>
merge_key=<16hex; §3.1>
```

**`artifacts=`** makes POOL trustworthy — two shards are comparable only when their boxes
agree on this line; computed once by streaming `aegis_core::witness::Sha256`
(`witness.rs:22`) over the resident slices, never re-read from FAT. **`job.N.digest`** for
`eval` is a `WitnessChain` fold of `(token_id, &[i64] logits)` per step — existing machinery,
and stronger evidence than the scalar. For `membw` it is the touch pattern's checksum, which
*is* comparable across boxes; only the bandwidth number is gated.

### 3.1 `merge_key` — byte-exact input serialization

`merge_key` = first 16 lowercase hex of sha256 over the concatenation of these
**NUL-terminated ASCII fields, in this order, with no other separator**:

```
"v1"                                        \0
<model 64hex> \0 <embed 64hex> \0 <vocab 64hex> \0     # full digests, not sha16
<env>                                       \0         # "iron" | "vm"
<seed decimal, or "none">                   \0
  for each step, dispatch order, N ascending:
    <N decimal> \0 <kind> \0 <step-input> \0
"END"                                       \0
```

`<step-input>` by kind: `cpuid` → empty; `mech` → empty; `membw` → `<mib decimal>`;
`verify` → `<NAME>:<receipt file 64hex>`; `eval` →
`<NAME>:<corpus payload 64hex>:<lo>:<hi>:<W>`. All decimals are ASCII, unsigned, no leading
zeros; all hex is lowercase. It **excludes** `cpu_brand`, `shard`, `tag` and `run_id` —
those are what the comparison is about — and **includes `env`**, so an `iron` record and a
TCG record can never share a merge key and can never be presented as replicating each other.

---

## 4. Unikernel modules (§5-ext) and LOC

| file | phase | change | LOC |
|---|---|---|---|
| `net/mod.rs` | 4 | `tcp_recv_exact`, `tcp_send_slice` chunked at `XFER_CHUNK`, watchdog re-arm hook per chunk, listener socket buffers (§4.2); `Until::Len` loses its `#[allow(dead_code)]` | +105 |
| `files.rs` (new) | 4 | name validation, `ls`, `stat`, `sha_named` (streaming sha256 through the existing bounce buffer), `get_stream`, `put_stream` → `STAGE.PRT` → readback-verify → commit, `del_named`, free-space query, `sweep_parts()` | +480 |
| `reload.rs` (new) | 4 | `Slabs`, `EngineSlot { slabs, engine: Option<TernaryInferenceEngine> }`, `reload(&mut EngineSlot, &mut Directory)`, `CURRENT.TXT` read/write | +190 |
| `server.rs` | 4 | dispatch for `AUTH/LS/STAT/SHA/GET/PUT/RM/RELOAD/HEALTH/RUNID`, `DATA` framing, `Lines::take_bytes`, the §1.4 abort table, RUNID ring | +360 |
| `job.rs` | 4,5 | §2-ext directives; new `StepResult` fields; `merge_key` (§3.1); `artifacts=`; MECH reordering; `StepErr`; call `files::sweep_parts` from `load()` | +270 |
| `lab.rs` (new) | 5 | `Step::{Cpuid, Verify, Eval, MemBw, Mech}`; `verify` wraps `verifier::run`; `mech` wraps the block lifted out of `main.rs`; `eval` = corpus validation + windowing + NLL-I | +430 |
| `aegis-core/cis_attn.rs` | 5 | `log2_u64_q32` (LOG2-I over an integer) + goldens | +40 |
| `main.rs` | 4,5 | build an `EngineSlot` instead of bare slices; honour `CURRENT.TXT`; `mod files; mod lab; mod reload;`; MECH block extracted | +55, −180 |
| `xtask/src/main.rs` | 4,5,6 | `files-test`, `lab-test`, `pool-test` | +500 |

≈1 930 unikernel LOC plus 500 of gate. `no_std` + `alloc`, `uefi 0.38`, `smoltcp 0.12`,
**no new dependency** — sha256 already exists (`witness.rs`, used at `verifier.rs:18`).
**Phase 6 adds zero unikernel LOC.**

### 4.1 `RELOAD` — the one genuinely dangerous verb, and the ownership change it forces

Today `main.rs` allocates three `allocate_huge_pages` slabs at boot (`main.rs:583`, `:598`,
`:609`) and hands `&mut TernaryInferenceEngine` down through `job::dispatch →
dispatch_resident → server::run → serve → do_job`; the engine borrows those slabs for the
server's whole life. Overwriting a slab under a live engine would let a box report a fresh
`model_sha` while still inferring against layout state derived from the old bytes.
Therefore:

1. `server::run` (`server.rs:120`) and `job::dispatch` take `&mut EngineSlot`, not
   `&mut Engine`. The slot **owns** the engine as an `Option`, so it can be dropped.
2. `RELOAD` checks the new file against slab capacity first: bigger ⇒ `ERR reload-size`,
   **no state touched**. Growing a slab means a `REBOOT`, and saying so beats faulting in
   ring 0 — `main.rs:588`'s `STAGE 3 FAILED: contiguous alloc` hazard.
3. Otherwise `slot.engine = None` (drop), `load_file_into` the slab (watchdog re-armed per
   chunk), `Engine::new`, `reloads += 1`. On failure the RAM copy is junk, so the box answers
   `ERR reload-engine`, logs it and calls `ResetSystem(COLD)` — spending `BootNext`, landing
   in Debian. **`RELOAD` never leaves a box serving an engine it cannot describe.**
4. `RELOAD` refuses while `STAGE.PRT` exists (`ERR busy-file`).

`PUT MODEL.SAF` + `REBOOT` — the already-tested boot loader — stays the *default* path in
the host tool; `RELOAD` is an opt-in fast path for iteration without a boot cycle.

### 4.2 Transport sizing (why 4 KiB is not enough)

`TCP_RX_BYTES` and `TCP_TX_BYTES` are both `4096` today (`net/mod.rs:87`, `:89`), used for
the listener at `net/mod.rs:1097–1098` and the client socket at `:1166–1167`. A 4 KiB
receive window stalls the peer for an ACK roughly every 4 KiB: at a 1 ms LAN RTT that is
~4 MB/s, and a 1.83 GB `PUT` spends **over seven minutes in pure window stall**. Phase 4
adds `TCP_RX_LISTEN_BYTES = 262_144` and `TCP_TX_LISTEN_BYTES = 65_536`, used **only** at
`:1097–1098`; `tcp_connect` keeps 4 KiB because collector POSTs are small. Cost: 320 KiB per
listening socket, ×2 with the backlog socket = 640 KiB of UEFI pool, visible in `HEALTH
heapfree=`. `XFER_CHUNK` stays 65 536 because `RECV_MAX_BYTES` (`net/mod.rs:125`) caps one
`tcp_recv_until` at that; the larger buffer just means a full chunk is usually resident when
the read runs.

---

## 5. POOL scheduler semantics (phase 6, host-side only)

Single-process Python 3 stdlib in `claudius-maximus`; state in SQLite at
`state/pool/<pool_id>.db`, written before any box is contacted.

**Plan.** `{pool_id, artifacts_expect, budget_s, units:[{id, body}]}`, `body` a literal
`JOB.TXT` fragment. The plan is the unit of reproducibility.

**Sharding** is explicit and dumb: `--split prompts FILE [--per-shard K]`,
`--split eval NAME K`, `--split seeds A..B`. Shards are equal-**count**, not equal-time —
Rule A forbids timing a box in order to schedule it — and sized so one `BUDGET` is ≤ 900 s.
Nothing splits at the token level.

**Leasing.** `pending → leased(box, t0, deadline = BUDGET + 120) → done | failed`. Host
bookkeeping only; the box's own mutual exclusion is v0.1's `BUSY`. On expiry the unit
returns to `pending`, `attempts += 1`, and **the same `RUNID` goes back out**: a merely-slow
original is deduped by the ring — *unless* the box rebooted, which §1.4's `up=` rule
detects — and if both answer, the records must agree or the pool is flagged `SPLIT`.

**Retries.** 3 attempts, backoff 5/30/120 s; transport failures re-queue to a *different*
box. Box-level `ERR`s (`fw-error`, `no-space`, `auth`) mark the box `SUSPECT` at once. A
**job-level** `verdict=FAIL <reason>` is deterministic and never retried: it is recorded and
the unit ends `FAILED`.

**Box health.** `HEALTH` poll every 30 s: `HEALTHY` → `SUSPECT` (one timeout or box-level
ERR) → `DOWN` (3 consecutive) → `RECOVER` (Debian side, §9) → `HEALTHY` after two clean
`HEALTH` plus a `PING`. `DOWN` expires leases at once. An `artifacts=` mismatch across boxes
is a **hard stop before a pool run starts**.

**Partial results.** Every completed unit hits disk as it lands. A record with
`verdict=FAIL budget` and `job.N.partial=k>0` — k of the requested count done, with a digest
**over those k** — is stored `PARTIAL` and the remainder re-queued; partials are prefix
evidence, never a completed unit. `--resume` dispatches only `PENDING`/`FAILED`.

**Merge digest.** `pool_digest = sha256(for each unit id ascending: "<id>\n" +
"<kind>:<digest>\n"…)`, first 16 hex, computed **only when every unit is `done` and every
contributing record carries `env=iron`**. Any unit missing ⇒ *no digest*, and the report
names what is missing; there is no "digest of what we got". Identical `pool_digest` from two
disjoint box sets over one plan is the fleet-scale form of the CIS-2 cross-ISA claim — a
determinism result, never a speed one.

**Agreement** is a claim about replication: ≥2 boxes, same `merge_key` (which now pins
`env`, §3.1), *different* `cpuid_sig`. `AGREE` = byte-identical `job.N.digest` (and
`nll_q16` for `EVAL`). **Every participant must be `env=iron`.** A `env=vm` record may
appear in a pool run, but it is filed `LAB-ONLY`: never `AGREE`, never `DISAGREE`, never in
`pool_digest`, never in a published table. A TCG guest is an emulator, and two emulators
agreeing is a statement about QEMU. `DISAGREE` = both iron, both complete, digests differ —
the CIS-style finding and a **deliverable, not an error**: both records written verbatim to
`state/pool/<id>/DISAGREE.md` and pushed to the phone, never averaged away, never retried
into silence. `INCONCLUSIVE` = a partial on either side. One box alone is `UNREPLICATED`.
Equal `tps` is not agreement; unequal `tps` is not disagreement.

**The scheduler never power-cycles a box.** `--reboot-on-quarantine` is opt-in, off by
default: autonomy does not extend to rebooting Justin's laptops unasked.

---

## 6. Host tooling (§7-ext) — `claudius-maximus`, branch `cm/aefinity-fleet`

- **`bin/cm-os-fs HOST[:PORT] ls|stat|sha|get|put|rm|reload|health [NAME] [FILE]`** — the
  §1.2 client. Every `get` is verified against the header `sha16` before the host file is
  renamed into place; every `put` re-`SHA`s the box copy. `push-model DIR` = `PUT`×3 +
  `SHA`×3 + `REBOOT` (or `--reload`), refusing if `artifacts=` does not change, and refusing
  `MODEL.SAF` without `--i-know` unless the sha matches a manifest.
- **`bin/cm-corpus pack IN.txt OUT.BIN`** — tokenizes on Debian and writes the §2.1
  container; the unikernel never tokenizes.
- **`bin/cm-os-pool submit PLAN.json --boxes state/BOXES.md --budget 600 [--replicate 2]
  [--resume]`**, plus `status` and `merge` — §5.
- **`bin/cm-fleet-health [--watch]`** — one line per box: state, `artifacts`, `served`, `up`,
  `parts`, last verdict; exit 1 if any box is `DOWN`. Carries the `TOKEN` when one is set.
- **`bin/cm-os-collector`** (extend, +90 LOC) — dumb, append-only, **not in the POOL control
  path**. A resident job's `REPORT` is refused in v0.1 (the listener holds the SNP
  exclusively), so the scheduler POSTs each unit's `RESULT.TXT` *on the box's behalf* with
  `X-Aefinity-Via: cm-os-pool` and `X-Aefinity-Box: <name>` — a known v0.1 limitation handled
  host-side instead of by a unikernel change.
- **`bin/cm-os-job`** unchanged; `--matrix` is the degenerate 1-unit pool.
- **Phone view** (ntfy + inbox): progress, then completion with digest and AGREE/DISAGREE
  counts (§11). Anything needing a decision goes to `NEEDS.md`.

A Rust `aefinity-ctl` (spec §9) waits until the Python semantics stop moving.

---

## 7. xtask gates (§6-ext)

All under `flock /tmp/aefinity-os.lock nice -n 10 env CARGO_BUILD_JOBS=4`, asserting
structure and exit codes only, none staging `MECH`.

- **`files-test`** (phase 4) — resident guest with `hostfwd`, as `resident-test` sets up.
  `LS` contains `MODEL.SAF` and its header parses as `LS <n> ok`; `SHA JOB.TXT` equals the
  host's sha of the staged bytes; `GET BOOTLOG.TXT` returns a `DATA` frame whose `len` and
  `sha16` match; `PUT TEST.BIN` (256 KiB pseudorandom) then `GET TEST.BIN` returns
  **byte-identical** bytes (Rule D); **`PUT BIG.BIN` of 64 MiB (67 108 864) then `SHA
  BIG.BIN`** — the case that exercises §4.2's window and the readback re-arm, ~90 s under
  TCG, because a 256 KiB transfer proves nothing about a 1.8 GB one; a `PUT` with a wrong digest ⇒ `ERR digest-mismatch`, `SHA TEST.BIN` still the
  old correct digest, `LS` free of strays and `HEALTH parts=0`; a `PUT` whose payload ends
  in the wrong 4 bytes ⇒ `ERR bad-frame`; `PUT ../X` ⇒ `ERR bad-name`; a 32-byte name ⇒
  `ERR bad-name`; `PUT BOOTLOG.TXT` ⇒ `ERR protected`; `PUT len=3 GiB` ⇒ `ERR bad-len`
  before any bytes cross; `RM` then `STAT` ⇒ `ERR not-found`; `RELOAD` ⇒ `OK reload …` with
  the same three digests and a following `JOB` still `verdict=OK`; with `TOKEN` set, `GET`
  before `AUTH` ⇒ `ERR auth`; `HEALTH` parses. PASS = all of that plus `REBOOT` → `BYE` →
  QEMU exits. **Every assertion is on the guest's own `LS`/`SHA`, never the host mirror —
  QEMU vvfat does not commit guest unlinks.**
- **`lab-test`** (phase 5) — oneshot body `SEED 1`, `CPUID`, `VERIFY GOOD.RCP`,
  `VERIFY BAD.RCP`, `EVAL TINY.BIN 0:4`, corpus and receipts `PUT` first so the gate
  exercises 4 and 5 together. PASS = `jobs=4`, every block carries `job.N.kind`,
  `job.2.pass=true`, `job.3.pass=false`, `job.4.nll_q16` present and **parsing as u64**,
  `job.4.ntok=3`, `rate_valid=false`, `membw_mibs` absent or `n/a`, `artifacts=` and
  `merge_key` non-empty and 16 hex, `verdict=OK`, `env=vm`, guest resets. A corpus with a
  corrupted magic ⇒ `job.N.err=bad-corpus`. **No value is asserted.** The LOG2-I goldens are
  a plain `cargo test -p aegis-core` unit, not a gate.
- **`pool-test`** (phase 6) — **host-side, no second guest** (this box has 6 GB): one real
  resident guest plus an in-process fake box speaking §4. Asserts 8 units shard across 2
  boxes; killing the fake mid-unit re-leases within `deadline`; a duplicate send of the
  re-issued `RUNID` returns `REPLAY`; a fake box whose `up=` goes backwards makes the
  scheduler re-run rather than trust the ring; `merge` prints a `pool_digest` only when all
  8 are `done`; an `env=vm` record never counts as `AGREE`; deleting one record makes
  `merge` exit non-zero and **name** the missing unit; a second `--resume` exits 0 having
  dispatched nothing. PASS = the scheduler's exit codes, not the guest's numbers.
  Agreement/SPLIT/quarantine are also covered by `python3 -m unittest tests.test_pool`
  against fake servers — the scheduler must not need iron to be tested.

---

## 8. Iron safety

- **Watchdog.** Idle stays 0 (a resident box may wait forever). `GET`/`PUT` arm
  `FILES_WD_S = 300` and **re-arm every `XFER_CHUNK` — during the transfer, during the
  sha256 readback, and during `RELOAD`'s load loop**. A 1.83 GB readback at FAT read speed
  can itself exceed 300 s, so arming only for the network half would reset a healthy box.
  Every verb disarms on every exit path (`server.rs:512`). `XFER_STALL_MS` bounds *no
  progress*, not duration.
- **Half-written files.** `PUT` never opens the target. It writes `STAGE.PRT`, flushes,
  closes, re-opens, streams a sha256 over the **readback**, and only then commits. For
  ordinary files the commit is delete-then-rename via `set_info::<FileInfo>`, and the delete
  happens only after the readback digest matched. For the three artifacts the commit is a
  **pointer swap**: `CURRENT.TXT` holds `model=MODEL.SAF|MODEL.NEW`, boot reads it and falls
  back to `MODEL.SAF` when absent or unparsable — the smallest commit available on a
  filesystem with no journal. Power loss before it ⇒ old weights; after ⇒ new; during ⇒
  fallback.
- **Double capacity is a hard requirement for artifact swap.** `MODEL.SAF` and its staged
  copy must be resident simultaneously, so `ALICE_UEFI` must hold **≥ 2× the artifact set**
  (2 × 1.83 GiB + slack ⇒ plan on 8 GB of ESP, not 4). `PUT` pre-checks free clusters and
  answers `ERR no-space`, but the UEFI `FileSystemInfo` free-space number is **advisory on
  vendor FAT drivers** and some report it wrong or not at all. So exhaustion may instead
  surface mid-stream as a short write, which is reported `ERR io` with `STAGE.PRT` deleted.
  `no-space` is best effort; `io` is the guarantee. Never a partial commit either way.
- **Orphans.** `sweep_parts()` deletes `STAGE.PRT` at boot and logs it to `BOOTLOG.TXT`;
  `HEALTH parts=1` means a box reappeared with a stale stage and it is `SUSPECT` until
  `cm-os-fs rm` clears it. A torn FAT32 directory entry is still possible; recovery is
  `fsck.vfat` from Debian, never in-kernel.
- **A box that disappears mid-shard.** It finishes the job anyway and leaves `RESULT.TXT` on
  the volume. The lease expires, the unit is re-leased with the same `RUNID`, and the lost
  box — if it returns without having rebooted — is deduped by the ring or contradicted by
  digest; `cm-uefi-harvest` reconciles by `run_id`, filing a late record `LATE` rather than
  overwriting a completed unit. A box whose watchdog fires **leaves the fleet**: the reset
  spends `BootNext` and lands in Debian, and its `RUNID` ring is gone (§1.4). There is no
  self-healing inside the unikernel and there should not be.
- **`RESULT.WIP`** is unchanged and now fleet-visible: a box whose `LS` shows it did not
  finish its last job, and `HEALTH`'s `last=` says so before any work is sent.
- **The unrecoverable class** is a firmware hang (an SNP transmit that never completes) with
  the watchdog disabled. Mitigation is unchanged from `UEFI-REMOTE-LANE.md`: a smart plug,
  and `BOXES.md` saying `power-cycle: human` until one exists — printed per box before a
  pool run.
- **Rule A.** No gate asserts a value; `rate_valid=false` travels with every record made
  under `env=vm`; `membw_mibs` is gated identically to `tps`; and `env` is inside
  `merge_key`, so no VM number can later be mistaken for an iron one even by accident.

---

## 9. Explicitly left to the Debian side

Compilation of anything (`cm-build`; the unikernel has no compiler and must not grow one);
`git`; `aegis-forge` conversion/quantization; **corpus tokenization** (`cm-corpus pack` —
the unikernel reads §2.1 containers only) and receipt *generation*; TLS, DNS, NTP;
**large-file provisioning** — rsync of `MODEL.SAF` onto `ALICE_UEFI`, because `PUT` is for a
box already up, not for standing up eight; `fsck.vfat` and partition repair;
`efibootmgr`/`BootNext`, `cm-uefi-run`, `cm-uefi-harvest`, power cycling; log rotation;
anything needing more than 8.3 names; every recovery path once a box is `DOWN`; and all of
**phase 7 TRAIN-POOL**. Deferred inside the unikernel, with reasons: token `STREAM` (it
would prove the server-owns-the-`Net` layering but adds a failure surface phases 4–6 do not
need) and box-to-box `POOL` reduction (real, but no embarrassingly parallel workload needs
it, and it is untestable on a 6 GB dev box without two concurrent guests).

**Research, not plan:** pooled optimisation without gradients. Evolution strategies need
only forward passes and a seed, so each box could score `θ + σ·ε(seed)` over an `EVAL` slice
and return `nll_q16` while the host applies the update and pushes weights with
`PUT`+`RELOAD`. It is sample-inefficient by orders of magnitude at 2B parameters, has never
been run here, and is listed as an idea these primitives happen to enable. Nothing in phases
4–6 depends on it, and no deliverable claims it.

---

## 10. Sequencing and the first five commits per phase

Order **4 → 5 → 6**, one branch each (`os/p4-files`, `os/p5-lab`, `cm/aefinity-fleet`), in
sibling worktrees, each gate green and pasted into the PR before the next. Phase 4 first:
`PUT` removes the stick from the loop.

**Phase 4 — FILES**
1. `p4: reload.rs — Slabs/EngineSlot, engine owned not borrowed` — pure refactor of
   `main.rs` and the `dispatch`/`server::run` signatures; `boot-test` still 33.
2. `p4: net: tcp_recv_exact / tcp_send_slice + listener buffer sizing + DATA framing +
   Lines::take_bytes` — frame parser unit-tested, no protocol surface yet.
3. `p4: files.rs — names (31), ls/stat/sha, sweep_parts` + read-only verbs in `serve()`.
   Gate: first half of `files-test`.
4. `p4: GET, then PUT with STAGE.PRT + readback verify + CURRENT.TXT swap + RM` with
   `xtask files-test` (including the 64 MiB case) **in the same commit**.
5. `p4: RELOAD, HEALTH, RUNID ring + up= rule, AUTH-gates-everything` + `artifacts=`/
   `reloads=`. Gate: `files-test` complete, `boot-test` 33.

**Phase 5 — LAB**
1. `p5: extract the MECH block from main.rs into lab::mech()` — pure move, `boot-test` 33.
2. `p5: job.rs — SEED/TAG/SHARD/STRICT/RUNID parse, StepResult fields, merge_key (§3.1),
   MECH-last`.
3. `p5: lab.rs — CPUID + VERIFY (wraps verifier::run)`; gate `lab-test` steps 1–3.
4. `p5: cis_attn::log2_u64_q32 + goldens, then lab.rs EVAL — corpus container, windowing,
   NLL-I` + `job.N.partial`; gate `lab-test` complete.
5. `p5: MEMBW (checksum comparable, bandwidth gated) + rate_valid + job.N.detail`.

**Phase 6 — POOL** (no unikernel code)
1. `cm: bin/cm-os-fs + cm-corpus pack + HEALTH/RUNID client` (lands during phase 4).
2. `cm: pool plan schema, SQLite state, sharding, plan-before-dispatch`.
3. `cm: leasing, RUNID re-dispatch + up= ring-loss rule, retry/backoff, box health`.
4. `cm: merge, iron-only agreement, DISAGREE.md, collector POST-on-behalf headers`.
5. `cm: xtask pool-test + tests/test_pool.py against fake resident servers`.

Then the two-box `AGREE`/`DISAGREE` run the digest was built for — that needs iron, and it
is Justin's step.

---

## 11. What Justin will see

**Every digest, count and number in this section is SPECIMEN output — hand-written to show
the shape of the interface. Nothing here has been measured, and no number below should be
quoted anywhere.**

**1. Flash a box.** The v0.1 stick, plus `JOB.TXT`:

```
MODE resident
LISTEN 4242
NET dhcp
TOKEN 9f2c…                 # optional; omit and the box is open on the LAN
BUDGET 600
```

Boot it. The screen shows the usual STAGE lines, then `RESIDENT: listening on 4242`, and
nothing happens until someone asks. Walk away.

**2. From penguin, confirm it is alive and fill it — no stick.** *(SPECIMEN)*

```
$ cm-fleet-health
dell-5200u   HEALTHY  up=00:41:12  served=0  parts=0  artifacts=4b1f…/8ac0…/22de…  last=none
hp-n4020     HEALTHY  up=00:39:58  served=0  parts=0  artifacts=4b1f…/8ac0…/22de…  last=none
thinkpad-x2  HEALTHY  up=00:12:03  served=0  parts=0  artifacts=4b1f…/8ac0…/22de…  last=none

$ cm-corpus pack ./corpora/wiki2.txt ./corpora/WIKI2.BIN
AEFCORP1 v1 ntok=2097152 vocab=128256 payload=1f7c…            (SPECIMEN)
$ cm-os-fs dell-5200u put WIKI2.BIN ./corpora/WIKI2.BIN
SEND … 8388672 bytes … OK WIKI2.BIN 8388672 c3a91f0b7d5e4412   (SPECIMEN)
verify: SHA WIKI2.BIN == local sha256 ✓
```

**3. Run the lab across three boxes.** *(SPECIMEN)*

```
$ cm-os-pool submit plans/eval-wiki2.json --boxes state/BOXES.md --budget 600 --replicate 2
pool 2026-09-04-a: 24 units, 3 boxes (all env=iron), replicate=2 → 48 dispatches
artifacts guard: all boxes 4b1f…/8ac0…/22de… ✓
[ 12%] leased u07→hp-n4020  u08→dell-5200u  u09→thinkpad-x2
[ 46%] hp-n4020 timeout on u11 → re-leased to thinkpad-x2 (same RUNID)
[ 47%] hp-n4020 answered u11 late → REPLAY deduped, filed LATE
[100%] 24/24 done, 0 failed
```

**4. The digest table** *(SPECIMEN — illustrative values, nothing measured)* — the only
table that means anything, and it has no `tps` column because Rule A says a QEMU or an
un-named box may not produce one:

```
unit  kind   nll_q16      digest            dell-5200u  hp-n4020  thinkpad-x2  verdict
u01   eval   1844674407   3f9a1c22e0d47b18      x            x                 AGREE
u02   eval   1902388110   88b2ce01a7f34d90      x                       x      AGREE
…
u17   eval   1771003942   d10c88ff2b9e6a41      x            x                 DISAGREE
                          a4471e09c3b2d558                                     ← hp-n4020

pool 2026-09-04-a COMPLETE                                          (SPECIMEN)
pool_digest = 6c2e08b41d7fa930      (24/24 units, all env=iron, artifacts 4b1f…/8ac0…/22de…)
AGREE 23/24   DISAGREE 1   UNREPLICATED 0   LAB-ONLY 0
finding: state/pool/2026-09-04-a/DISAGREE.md  (Broadwell-U AVX2 vs Gemini Lake SSE2)
```

The phone gets one line: `pool 2026-09-04-a COMPLETE digest=6c2e08b4 AGREE 23/24
DISAGREE 1`. The disagreement is not a bug report — it is the deliverable, the fleet-scale
form of the CIS-2 cross-ISA question, with both records verbatim in a file. And it is a
statement about *reproducibility*, not about how good the model is.
