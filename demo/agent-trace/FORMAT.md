# AEGIS-TRACE wire format (normative)

Source of truth: `aegis-linux/examples/agent_trace.rs`. This document lets a
third party (a) parse a receipt exactly as `agent_trace verify`'s strict
parser does, and (b) recompute `trace-chain` from the receipt text (plus the
declared table file, if any) **without running any model**. Section 7 states
precisely what such a recompute does and does not prove.

All hashes are SHA-256. All hex is lowercase, even-length, two chars/byte.
All multi-byte integers folded into a digest are noted BE (big-endian) or LE
(little-endian) explicitly — the two folds in this document use different
endianness on purpose (see §4, §5).

## 1. Scope & versions

| Header line | Format # | Adds vs. previous | commit/host in genesis? |
|---|---|---|---|
| `AEGIS-TRACE v0` | 1 | base: header + step(`toks,tool,in,out,decode-chain`) + `trace-chain` | no (lines absent) |
| `AEGIS-TRACE v1` | 2 | per-step `ctx=`/`q=` fields; `NOTE: format-1 receipt, per-step query binding not present` printed by `verify` for format 1 | no (lines absent) |
| `AEGIS-TRACE v2` | 3 | `commit`/`host` lead lines, and — unlike format 1/2, where those two lines if present are decorative — folded into trace genesis (see §4) | **yes** |

`gen` (as of the current binary) always emits `AEGIS-TRACE v2` (format 3).
`verify` accepts all three. `table-sha256` and `suite-sha256` lead lines are
each optional in every format, independent of the format number (table/suite
support predates format 2 and is not itself a format bump).

`gen` refuses to emit `commit unknown` (a format-3 receipt whose commit
could not be resolved) unless `AEGIS_ALLOW_UNKNOWN_COMMIT=1` is set in its
environment (PR #72, main `dbfb051`) — so an `unknown` commit line in the
wild means that escape hatch was used deliberately, not a build-detection
bug.

## 2. Text grammar

UTF-8 text, one logical record per line (`\n`; `str::lines()` splitting), no
trailing-content requirements beyond what's below. A file that is not valid
UTF-8 is rejected before any line is parsed (`FAIL structure: receipt is not
valid UTF-8`; an unreadable path is `FAIL structure: cannot read receipt
...`) — the verifier never panics on receipt bytes. There is no required
overall line order except: **the `AEGIS-TRACE <ver>` line must be line 0**
(`FAIL structure: missing AEGIS-TRACE header line` if absent anywhere;
`FAIL structure: AEGIS-TRACE header line at position {n}, must be first` if
present but not first). Beyond that, `verify`'s parser is a line-by-line
key scan — it does **not** enforce ordering among the remaining lead lines,
step lines, or WARNING lines (canonical `gen` output order, for interop, is
given below; a strict-but-order-insensitive parser must still accept any
ordering `verify` accepts).

**Canonical form.** A receipt must be byte-for-byte canonical or `verify`
rejects it with `FAIL structure: ...` before any artifact/replay check runs:
- No blank lines anywhere — leading, between records, or trailing before
  EOF (`FAIL structure: receipt contains a blank line`).
- The file ends with exactly one LF: `gen` always terminates the last line,
  so a receipt whose final byte is not `\n` is not the bytes that were
  attested (`FAIL structure: receipt does not end with a newline`); a second
  trailing LF is the blank-line case above.
- No CR bytes (CRLF line endings are not canonical, even though `\r` would
  otherwise silently ride along as trailing content on whatever `str::lines()`
  treats as the preceding line): `FAIL structure: receipt contains a CR
  byte (CRLF line endings are not canonical)`.
- Every integer field (`K`, `N`, the `step <label>:` label, each `toks=`
  token id, and each `WARNING step <i>:` index) must be canonical decimal:
  ASCII digits only, no leading `+`/`-`, no leading zero except the single
  digit `0` itself, and no surrounding whitespace. `"007"`, `"+3"`, `"3 "`,
  and `" 0 "` are all rejected even though Rust's integer `FromStr` would
  accept some of them — the reference verifier parses every one of these
  fields through a dedicated `parse_canonical_uint` helper, never bare
  `.parse()`, for exactly this reason.
- `commit` and `host` line values must be non-empty (`FAIL structure:
  commit line has an empty value` / `FAIL structure: host line has an
  empty value`).

**Canonical `gen` line order:**
```
AEGIS-TRACE v2
model <64 hex>
embed <64 hex>
vocab <64 hex>
K <int>
N <int>
table-sha256 <64 hex>        # only if --table was given
suite-sha256 <64 hex>        # only if --suite-sha256 was given
prompt-hex <hex of prompt bytes>
commit <string>              # format 3 only
host <string>                # format 3 only
step 0: toks=<u32,u32,...> tool=<name> in=<hex> out=<hex> decode-chain=<64 hex> ctx=<64 hex> q=<64 hex>
...                            # ctx=/q= only in format 2/3
[WARNING step <i>: tool argument not found verbatim in context]   # 0+ lines, format 2/3 only
trace-chain <64 hex>
```

**Lead-line grammar:** `<key> <rest-of-line>` split on the *first* space
(`splitn(2, ' ')`); a line with no space (`key.is_empty()` false but no
value) still routes on `key`. Allowed lead keys (`LEAD_KEYS`, exactly these
12, plus the special `WARNING` key handled separately):
`AEGIS-TRACE, model, embed, vocab, K, N, prompt-hex, trace-chain,
table-sha256, suite-sha256, commit, host`.
Any other key on a non-`step`/non-`WARNING` line is rejected. Every key in
`LEAD_KEYS` may appear **at most once**; a WARNING line does not count
against that (it is not checked for duplication as a *key*, but the
resolved `(step)` set is checked against replay — see §3).

**Step-line grammar:** a line matching `^step `; parsed as
`"step " <label> ":" <fields>`, `label`/`fields` split on the *first* `:`.
- `label` must be canonical decimal (see "Canonical form" above — no
  surrounding whitespace, no leading zero/`+`) parsing as a `usize` equal
  to the step's 0-based ordinal position among step lines encountered so
  far in the file (NOT necessarily its numeric value if the label lies —
  `verify` compares the label's parsed number to that ordinal). A label
  like `" 0 "` used to be accepted via `.trim()` before parsing; it is now
  non-canonical and rejected. Mismatch (numeric or unparsable):
  `FAIL structure: step label {n_or_text} at position {position}`.
- `fields` (trimmed) is split on ASCII whitespace into tokens; each token
  must be `key=value` (`split_once('=')`); a token with no `=` is
  `FAIL structure: step {position}: stray token {field:?}`.
- Recognized step keys: `toks, tool, in, out, decode-chain` (format 1+,
  mandatory) and `ctx, q` (format 2/3 only, present in the line — see the
  cross-check below). Any other key: `FAIL structure: step {position}:
  unknown field {other:?}`. Any key repeated on the same step line:
  `FAIL structure: step {position}: duplicate field {name:?}`.
- `toks=<v>`: `v` split on `,`, empty sub-tokens skipped, each remaining
  piece parsed as `u32` (same rule as `K`/`N` above but `u32::from_str`,
  range 0..=4294967295); a bad or out-of-range piece is `FAIL structure:
  step {position}: bad token id {t:?}`.
- Missing any of `toks=`/`tool=`/`in=`/`out=`/`decode-chain=` on a step
  line: `FAIL structure: step {position}: missing one of toks= tool= in=
  out= decode-chain=`.
- `in=`/`out=`/`decode-chain=`/`ctx=`/`q=` values are opaque hex strings at
  parse time (compared as strings against a locally recomputed `hex(...)`
  later, not hex-decoded by the structural parser itself, except where
  noted in §3/§4).

**Field-specific lead-line checks:**
- `AEGIS-TRACE <v>`: `v0`→1, `v1`→2, `v2`→3, else `FAIL structure: unknown
  AEGIS-TRACE format {other:?}`. A second occurrence anywhere:
  `FAIL structure: duplicate AEGIS-TRACE header line`.
- `K <v>` / `N <v>`: `v` is **not trimmed** — it is everything after the
  first space (`splitn(2, ' ')` on the whole line, so a trailing space
  stays part of `v`) and must be canonical decimal per the "Canonical
  form" rules above, parsed via `parse_canonical_uint`: ASCII digits only,
  no leading `+`/`-`, no leading zero except `0` itself, no surrounding
  whitespace. `K 1 ` (trailing space), `K 007`, and `K +3` are therefore
  all `FAIL structure: malformed K "1 "` / `"007"` / `"+3"` (confirmed on
  box1 for the trailing-space case) — else `FAIL structure: malformed K
  {v:?}` / `FAIL structure: malformed N {v:?}`.
- `prompt-hex <v>`: `v` must be valid even-length hex whose decoded bytes
  are valid UTF-8, else `FAIL structure: malformed hex in prompt-hex`
  (same message for both the hex-decode failure and the UTF-8 failure).
- `table-sha256 <v>` / `suite-sha256 <v>`: `v` must be exactly 64 lowercase
  hex chars (`[0-9a-f]`), else `FAIL structure: malformed table-sha256
  (want 64 lowercase hex)` / `FAIL structure: malformed suite-sha256 (want
  64 lowercase hex)`.
- `trace-chain <v>`: the reference `verify` has **no structural check** on
  this line — a malformed or truncated value simply fails to equal the
  recomputed chain at the final comparison (`VERIFY FAIL`), same as any
  other mismatch. A chain-only tool SHOULD nonetheless reject anything but
  64 lowercase hex chars up front, with `FAIL structure: malformed
  trace-chain (want 64 lowercase hex)`, since it can never *usefully*
  recompute against a value that isn't a hash.
- Any lead key not in `LEAD_KEYS` and not `WARNING`: `FAIL structure:
  unknown line key {key:?} on line {line_no}`.
- A `LEAD_KEYS` key seen a second time: `FAIL structure: duplicate {key}
  line`.
- `WARNING <rest>`: the whole line must equal
  `WARNING step <i>: tool argument not found verbatim in context` for some
  parseable `usize` `<i>`; anything else: `FAIL structure: malformed
  WARNING line on line {line_no}`.

**Cross-structural checks (after the full line scan):**
- Format 1 receipt (`w_format == 1`) with any step carrying `ctx=`/`q=`:
  `FAIL structure: receipt declares format 1 but step {i} carries ctx=/q=
  (format-2 downgrade)`.
- Format ≥ 3 receipt missing `commit` or `host` line: `FAIL structure:
  format-3 receipt is missing its commit or host line`.
- `K` value ≠ number of step lines actually present: `FAIL structure:
  receipt claims K={w_k} but has {n_steps} step lines`.
- Header bounds (`validate_receipt_header`/`check_header_bounds`, requires
  a parsed model+tokenizer, so this step is not chain-only): `K must be >=
  1`; `N must be > 0`; `prompt tokenizes to zero tokens`; `prompt tokens
  ({p}) + N ({n}) exceeds max_position_embeddings ({m})` — each wrapped as
  `FAIL structure: {reason}`.

These are the only rejections the *structural* parser emits. Beyond
structure, `verify` also does artifact/replay checks that require the
model (`FAIL artifact: MODEL/EMBED/VOCAB/TABLE hash mismatch ...`,
`VERIFY FAIL — ...` for missing/mismatched `--table`/`--suite-sha256`, and
per-step/`trace-chain` `VERIFY FAIL` divergences) — out of scope for a
chain-only recompute except as noted in §7.

**Missing required lead lines are *not* structural rejections.** The
structural parser above never checks a required key's mere *presence* by
itself (only duplication, and cross-checks like the K/step-count match);
absence instead surfaces later, once `verify` tries to use the field
against the model:
| Missing line | Where it surfaces | Message |
|---|---|---|
| `model` / `embed` / `vocab` | artifact check | `FAIL artifact: MODEL\|EMBED\|VOCAB hash mismatch (receipt  vs local …)` (empty receipt-side hash) |
| `K` / `N` | header-bounds check | `FAIL structure: K must be >= 1` / `FAIL structure: N must be > 0`, or (if some step lines exist anyway) `FAIL structure: receipt claims K=0 but has {n} step lines` |
| `prompt-hex` | header-bounds check | `FAIL structure: prompt tokenizes to zero tokens` |
| `trace-chain` | final chain comparison | `VERIFY FAIL` (empty string does not equal the recomputed hex) |

A chain-only tool cannot run the model-dependent artifact/header-bounds
checks above, so it SHOULD instead reject a missing required line up
front, with `FAIL structure: missing <key> line` (one message per key:
`model`, `embed`, `vocab`, `K`, `N`, `prompt-hex`, `trace-chain`).

## 3. Field semantics

| Field | Meaning | Bytes folded (where applicable) |
|---|---|---|
| `model`/`embed`/`vocab` | sha256 of the exact MODEL.SAF/EMBED.BIN/VOCAB.BIN file bytes | genesis, raw 32 bytes each |
| `K` | episode step count | genesis, as BE u64 |
| `N` | tokens decoded per step | genesis, as BE u64 |
| `prompt-hex` | hex of the UTF-8 initial prompt bytes | genesis, as (len BE u64, then raw bytes) |
| `table-sha256` | sha256 of the `--table` file's exact bytes (present iff `gen` was given `--table`) | genesis, raw 32 bytes, then table byte length as BE u64 (table_len is **not** itself a printed field — see §4) |
| `suite-sha256` | caller-supplied 32-byte digest, e.g. of an eval-suite TSV (present iff `--suite-sha256` was given) | genesis, tag `b"SUITE"` then raw 32 bytes |
| `commit`/`host` | build git commit (40 hex or `unknown`) / `hostname` output, format 3 only | genesis, tag `b"PROV"` then each as (LE u32 len, bytes), commit first then host |
| `step i: toks=` | comma-separated greedy-decoded token ids (u32) for that step | **not folded into trace-chain**; only the *step's* `decode-chain` (a `WitnessChain` digest over tokens+logits) is folded — see §5. Cannot be recomputed without inference. |
| `tool` | one of `no-tool`, `calc`, `calc-error`, `lookup`, `file-read` — see below | step fold, as UTF-8 bytes of the name |
| `in` | the tool call's *matched call text* (e.g. `CALC(3 + 4)`), or empty for `no-tool` | step fold, raw bytes (hex-decoded from `in=`) |
| `out` | the tool's result bytes: decimal result (`calc`), fixed error string (`calc-error`: `overflow`\|`div-by-zero`\|`bad-op`), table value or `NONE` (`lookup`), table value or `NOT-FOUND` (`file-read`), or empty (`no-tool`) | step fold, raw bytes (hex-decoded from `out=`) |
| `decode-chain` | that step's `WitnessChain` digest (folds each generated token id + its full i64 logit vector) | step fold, raw 32 bytes. **Cannot be recomputed without replaying inference** (needs the model's actual logits). |
| `ctx` (format 2/3) | sha256 of the exact prompt text fed to the model for that step (the full accumulated running prompt) | **not folded into trace-chain**; `verify` recomputes and string-compares it independently |
| `q` (format 2/3) | sha256 of that step's "query text": the initial prompt at step 0, else the immediately preceding step's tool-result text (`"\nTOOL[{name}]={output}\n"`) | **not folded into trace-chain**; same as `ctx` |
| `trace-chain` | final folded chain over genesis + every step | see §4/§5 |

**Hex-value comparison rule.** `in=`/`out=`/`decode-chain=`/`ctx=`/`q=`
values are opaque strings to the structural parser (§2) — `verify` never
hex-decode-and-compares them at parse time. Instead, at replay time
`verify` recomputes each field itself and compares it **as a string**
against `hex(recomputed_bytes)`, where `hex()` always emits lowercase.
Consequences:
- Uppercase hex, or hex that is otherwise valid but not what `verify`'s
  own `hex(...)` would print, is **not** a structural error — it produces
  a `VERIFY FAIL` divergence at replay, same as any other wrong value.
- A chain-only tool (no replay available) MUST hex-decode these fields
  itself using the same rule as `unhex` (line 437 of `agent_trace.rs`:
  even length, `[0-9a-fA-F]` pairs) in order to fold them into the
  genesis/step hashes at all, and SHOULD additionally require lowercase,
  to match what a real `verify` run would accept without a `VERIFY FAIL`.
- An empty value (`in=` / `out=`) is legal — it is the `no-tool` case
  (zero-length field, contributes only its `len_le(0)` prefix to the step
  fold, §5).
- **Undecodable step hex.** If `in=`, `out=` or `decode-chain=` fails the
  `unhex` rule (odd length, or a non-`[0-9a-fA-F]` character), the
  reference `verify` reports nothing structural — the value is simply
  unequal to its own recomputed hex, so the run ends in `VERIFY FAIL` at
  replay. A chain-only tool has no replay value to compare against and
  cannot fold bytes it cannot decode, so it MUST reject up front, with
  `FAIL structure: step {position}: malformed hex in {field}` where
  `{field}` is one of `in`, `out`, `decode-chain` (wording mirrors the
  `prompt-hex` rejection in §2), exit 2. (`ctx=`/`q=` are not folded, so a
  chain-only tool never decodes them and MUST NOT reject on them.)
  Vector: `vectors/malformed-step-hex.txt`.

Tool-kind byte rules for `in`/`out` (all are ASCII/UTF-8 bytes of the shown
text, hex-decoded from the receipt's `in=`/`out=` fields):
- `no-tool`: `in` = empty, `out` = empty (the scanner found no `CALC(`/
  `LOOKUP(` call it could parse).
- `calc`: `in` = the exact matched substring `CALC(<...>)`, `out` = the
  checked-arithmetic result formatted as a base-10 signed integer.
- `calc-error`: `in` = the exact matched `CALC(<...>)` substring, `out` =
  one of the fixed ASCII strings `overflow`, `div-by-zero`, `bad-op`.
- `lookup`: `in` = the exact matched substring `LOOKUP(<key>)`, `out` =
  the table's value string for `key`, or the literal `NONE` on a miss.
- `file-read`: `in` = the exact matched substring `FILE-READ(<key>)`,
  `out` = the same `--table`'s value string for `key`, or the literal
  `NOT-FOUND` on a miss. `file-read` shares `lookup`'s table binding (no
  separate flag or hash line) but is a distinct tool identity with its own
  miss literal, so `tool`/`out` alone (without `in`) still disambiguate the
  two. A multi-tool episode may carry `calc`/`calc-error`, `lookup`, and
  `file-read` steps in any combination — the step fold (§5) is per-step and
  tool-name-agnostic, so nothing else about the format changes.

**`--fail-fast` (verify only).** The reference `verify` accepts an
optional `--fail-fast` switch that diffs each step against the receipt as
soon as it is replayed and stops at the first divergent step instead of
replaying all `K` steps first. On divergence it prints the same
`step {i} divergence: ...`/ctx/q-mismatch lines this section already
specifies, followed by `VERIFY FAIL — replay diverged from the receipt
(fail-fast after step {i})`. This early-stop message is an **optional**
verifier behaviour — a purely local performance optimization for
receipts tampered early, not part of the wire format. On a receipt that
verifies, `--fail-fast` output is byte-identical to full-mode output; on a
divergent receipt the early stop means the `receipt trace-chain`/`local
trace-chain` lines and any `WARNING` lines for earlier steps are not
printed. The normative
verify output remains the full-report form this section describes
(every step diffed, ending in the plain `VERIFY FAIL — replay diverged
from the receipt` or `VERIFY PASS` line); a conformant chain-only tool
is never required to implement `--fail-fast` or match its wording.

## 4. Genesis fold (exact byte layout)

`TRACE_DOMAIN` is the fixed constant `b"AEGIS-TRACE v0\n"` (15 bytes,
**always this literal string regardless of the receipt's declared format**
— it is not re-derived from the header line's version text).

| # | Bytes | Always present? | Notes |
|---|---|---|---|
| 1 | `TRACE_DOMAIN` = `AEGIS-TRACE v0\n` (15 bytes, incl. trailing `\n`) | yes | domain separation constant, literal |
| 2 | `model_sha` (32 raw bytes) | yes | hex-decode the `model` line |
| 3 | `embed_sha` (32 raw bytes) | yes | hex-decode the `embed` line |
| 4 | `vocab_sha` (32 raw bytes) | yes | hex-decode the `vocab` line |
| 5 | `k` as **BE u64** (8 bytes) | yes | from `K` line |
| 6 | `n` as **BE u64** (8 bytes) | yes | from `N` line |
| 7 | `prompt.len()` as **BE u64** (8 bytes) | yes | byte length of the UTF-8 prompt (from `prompt-hex`) |
| 8 | `prompt` bytes (raw, `prompt.len()` bytes) | yes | hex-decode `prompt-hex` |
| 9 | `table_sha` (32 raw bytes) then `table_len` as **BE u64** (8 bytes) | only if receipt has a `table-sha256` line | `table_len` = the declared table **file's** byte length — it is **not printed anywhere in the receipt**; a chain-only recompute of a table-bound receipt needs the actual table file to get `table_len` (and to confirm `table_sha`). No tag byte precedes this block. |
| 10 | `b"SUITE"` (5 bytes) then `suite_sha` (32 raw bytes) | only if receipt has a `suite-sha256` line | tag present; sits after the table slot whether or not #9 is present |
| 11 | `b"PROV"` (4 bytes) then, for each of `commit` then `host` in that order: `(field.len() as u32).to_le_bytes()` (4 bytes, **LE**) followed by the field's raw UTF-8 bytes | only for format ≥ 3 (`AEGIS-TRACE v2`) — never present for format 1/2 even if `commit`/`host` lines happen to be printed | length prefix is **LE u32**, unlike the BE u64 lengths in #5-#7 |

Block #9's table_sha/table_len ordering and lack of a tag is fixed on
purpose (documented as load-bearing in the source) so pre-suite-hash
table-bound receipts keep verifying byte-for-byte after suite/PROV support
was added. A genesis with none of #9-#11 (a bare table-less, suite-less,
format 1/2 receipt) is byte-identical to the very first (pre-LOOKUP)
genesis fold.

`genesis = sha256(concat(applicable blocks 1..11 in order))`.

**Table-bound receipts without `--table`.** A receipt declaring
`table-sha256` but verified without `--table` cannot have block #9
recomputed (`table_len` is not printed anywhere) — `verify` prints
`VERIFY FAIL — receipt declares table-sha256 <first 16 hex> but no
--table was given` and exits non-zero; a chain-only tool MUST print the
same line (with the receipt's own first-16-hex prefix) and exit non-zero
rather than attempt genesis without the table. With `--table FILE`
supplied, the tool MUST check `sha256(FILE) == table-sha256` — a mismatch
is `FAIL artifact: TABLE hash mismatch (receipt <16hex> vs local
<16hex>)` — and use `FILE`'s exact byte length as `table_len`.

## 5. Step fold (exact byte layout)

For step `i` (0-based), given the running chain value `chain` (genesis for
`i=0`, else the previous step's fold result):

| Bytes | Notes |
|---|---|
| `chain` (32 raw bytes) | running trace-chain value so far |
| `b"TSTEP"` (5 bytes) | literal tag |
| `step` as **BE u64** (8 bytes) | the step's 0-based index `i` |
| `decode_chain_digest` (32 raw bytes) | hex-decode the step's `decode-chain=` field |
| for each of `tool_name`, `tool_input`, `tool_output` in that order: `(field.len() as u32).to_le_bytes()` (4 bytes, **LE**) then the field's raw bytes | `tool_name` = UTF-8 bytes of `tool`; `tool_input`/`tool_output` = hex-decoded `in=`/`out=` |

`chain_{i} = sha256(concat(chain_{i-1 or genesis}, TSTEP, be_u64(i), decode_chain_i, len_le(name)‖name, len_le(input)‖input, len_le(output)‖output))`

After all `K` steps, `trace-chain = chain_{K-1}`.

`ctx=`/`q=` and the `WARNING` lines are **never** folded into any of this —
they are checked by `verify` as independent equalities against its own
replay, not part of the chain math (see §3).

## 6. Worked example

Receipt: `vectors/fmt3-k3-763658a.txt` (copied from
`cm-box1:/home/cm/verify-pr70/src/demo/agent-trace/out/trace-aefinity-box-20260910T212140Z.txt`),
format 3 (`AEGIS-TRACE v2`), K=3, N=16, no table, no suite.

Header fields (hex, 32 bytes each unless noted):
```
model   facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db
embed   e32b99a25e345c65054f36dedf40329a89513cdf1a8195db64fc440fd364e077
vocab   5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae
K       3            -> BE u64 0000000000000003
N       16           -> BE u64 0000000000000010
prompt  "The quick brown fox" (19 bytes) -> len BE u64 0000000000000013
commit  763658a2b8c440f094d1f9d47464d822178f761c  (40 ASCII bytes)
host    aefinity-box                                (12 ASCII bytes)
```
No `table-sha256`/`suite-sha256` line, so genesis blocks #9/#10 (§4) are
absent. Format is 3, so block #11 IS present:
```
b"PROV" ‖ u32le(40) ‖ "763658a2b8c440f094d1f9d47464d822178f761c" ‖ u32le(12) ‖ "aefinity-box"
```
Genesis preimage = block1(15B) ‖ model(32B) ‖ embed(32B) ‖ vocab(32B) ‖
k(8B) ‖ n(8B) ‖ promptlen(8B) ‖ prompt(19B) ‖ PROV-block(4+4+40+4+12=64B)
= 15+32+32+32+8+8+8+19+64 = 218 bytes total, hashed once with sha256.

Three step folds follow (§5) using each step's `tool=no-tool` (so
`in`/`out` are both empty, contributing only their zero-length prefixes)
and `decode-chain` value from the receipt.

Genesis digest (recomputed per §4, sha256 of the 218-byte preimage above):
`a7146ce1cf587fd6535681d980e7f0e8ee18d8e8af11f0172a77131000e8a414`. Folding
the three step records (§5) over that genesis in order reproduces the
receipt's own final line, `trace-chain
2278dc97974f34bab86cbe0a4172ad7a50ed4ecdaa545e0d4171abbb1e8f7029` — verified
by `tools/trace_chain.py` (see §8).

## 7. What a chain-only recompute proves / does not prove

A chain-only recompute (this document's §4/§5 math, applied to the receipt
text alone, plus the table file when `table-sha256` is declared) proves:
- The receipt is **internally self-consistent**: its own `trace-chain`
  line is the correct fold of its own header fields and its own
  `decode-chain`/`tool`/`in`/`out` fields, exactly as printed.
- If a table is declared, that the declared `table-sha256` matches the
  actual table file (when the file is supplied) — but table_len can only
  be checked the same way, by having the file.

It does **not** and cannot prove:
- That `toks=` is what the model actually generated (tokens are never
  folded into the chain at all).
- That `decode-chain=` is a genuine `WitnessChain` digest of real model
  logits for that prompt — a fabricated 32-byte value folds in exactly
  the same way as a real one; only re-running inference (`decode_step`,
  §3) and rebuilding `WitnessChain` from the actual logits can check this.
- That `ctx=`/`q=` are correct (they are not folded into `trace-chain` at
  all; `verify` checks them by independent replay comparison, not chain
  math).
- That the tool outputs (`out=`) are correct results of running the
  declared tool on `in=` — a chain-only check can *recompute* `calc`/
  `lookup` outputs deterministically from `in=` (and the table, for
  `lookup`) and compare, since those are pure functions with no model
  dependency, but this is an extra check beyond the fold itself, not
  something the fold enforces.

A `PASS` from `agent_trace verify` proves strictly more: it re-derives
`toks`/`decode-chain`/`ctx`/`q` from an actual model replay (needs
MODEL.SAF/EMBED.BIN/VOCAB.BIN) and requires bit-for-bit agreement with
every receipt field, not just chain self-consistency.

**`commit unknown` warning.** A format-3 receipt whose `commit` line
reads exactly `unknown` still folds normally into genesis (block #11, §4)
and can still `MATCH`/`PASS` — `unknown` is a legal (if unpinned) provenance
value, not a structural or verify error. `verify` prints, without changing
the verdict: `WARNING: receipt commit is unknown (provenance not pinned to
code)`. A chain-only tool SHOULD print the same warning on the same
condition. (`gen` itself refuses to emit `unknown` unless
`AEGIS_ALLOW_UNKNOWN_COMMIT=1` — see §1 — so seeing this warning in
practice means that escape hatch was used.)

## 8. Test vectors

From `vectors/fmt3-k3-763658a.txt` (see §6), reproduced by
`tools/trace_chain.py`:
```
$ python3 tools/trace_chain.py vectors/fmt3-k3-763658a.txt
MATCH 2278dc97974f34bab86cbe0a4172ad7a50ed4ecdaa545e0d4171abbb1e8f7029
```

The full vector set — real receipts across format 2/3, a table-bound
receipt, two hand-tampered (MISMATCH) receipts, and six structurally
malformed (REJECTED) receipts — lives in `vectors/`, indexed by
`vectors/EXPECTED.tsv` and described in `vectors/README.md`. Run the whole
set with `python3 tools/trace_chain.py --selftest`.
