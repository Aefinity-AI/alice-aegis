# Verified edge inference — one-command demo

`run.sh` builds and runs a chain: greedy, integer-only (CIS-1 `FullInt`)
decode of the checked-in M7 tinybit model produces a **receipt**
(artifact SHA-256s, prompt, every generated token id, an FNV-1a digest,
and a SHA-256 chain over the logits). That receipt is then replayed and
checked by **two independent verifiers** — the in-tree `cis_witness`
example and the standalone `no_std` `cis-verify` binary — against the
same three artifact files. A receipt generated on one CPU can be copied
to a different machine and verified there with no GPU and no network
access beyond the copy itself.

No files are downloaded. Everything comes from
`model-lab/tinybit/m7_final_gate_work/artifacts/` already in this repo.

## Usage

```
demo/edge-receipt/run.sh build                       # compile everything
demo/edge-receipt/run.sh gen "The quick brown fox" 32 # write a receipt
demo/edge-receipt/run.sh verify <receipt-file>        # replay + check
demo/edge-receipt/run.sh all "The quick brown fox"    # build + gen + verify
```

Receipts land in `demo/edge-receipt/out/receipt-<hostname>-<utc>.txt`
with a `.meta` sidecar (hostname, arch, CPU model, AVX2/FMA cpuinfo flag
counts, the aegis commit, and the decoded text from `cis_decode`).

## Cross-machine recipe

```
# machine A (generate)
demo/edge-receipt/run.sh build
demo/edge-receipt/run.sh gen "The quick brown fox" 32
scp demo/edge-receipt/out/receipt-A-*.txt machineB:alice-aegis/demo/edge-receipt/out/

# machine B (verify, independently, same checked-in artifacts)
demo/edge-receipt/run.sh build
demo/edge-receipt/run.sh verify demo/edge-receipt/out/receipt-A-*.txt
```

Both machines must have the identical `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN` triple (same repo checkout is sufficient — that's the point:
the hashes in the receipt are checked against whatever copy is local).

## What a PASS proves

- The three artifact files present on the verifying machine hash to
  exactly what the receipt claims.
- Replaying the greedy `FullInt` decode from the receipt's prompt on the
  verifying machine reproduces, bit-for-bit, the same sequence of
  **token ids**, the same FNV-1a digest, and the same SHA-256 logit chain
  the receipt records.
- This held across two independently-written verifier implementations
  (`cis_witness`, which shares `aegis-core`'s decode path, and
  `cis-verify`, a separate `no_std` reimplementation).

## What a PASS does NOT prove

- Nothing about GPUs, throughput, or latency — this is an integer-only
  CPU path and any timing this script prints is explicitly labelled
  "not a benchmark".
- Nothing about text-level correctness or model quality — the receipt
  and both verifiers check **token ids**, per CIS-1 spec v1.0
  (`docs/CIS-1_SPEC_v1.0.md`, §8-10). The decoded text printed by
  `cis_decode` is convenience output only, not part of what is verified.
- No claim that this is the "first" or "only" system to do bit-exact
  receipted replay — it's a demonstration of the CIS-1 receipt format
  already implemented in this repo.
- No claim about any model other than the specific artifacts hashed in the receipt.

## Swapping in the private 2B artifacts later

Point `ARTIFACTS` (edit the top of `run.sh`, or generalize it to a CLI
flag) at wherever the 2B release's `MODEL.SAF` / `EMBED.BIN` /
`VOCAB.BIN` are checked out. Nothing else in this demo is model-size
specific — `cis_witness` and `cis-verify` both read `aegis_config` out of
`MODEL.SAF`'s metadata at runtime. Confirm `max_new + prompt_tokens <=
max_position_embeddings` for that config before generating.

Concretely, point the three env vars at the files (names differ from the M7 set):

    AEGIS_MODEL=/path/aegis_pruned_model.cis.safetensors \
    AEGIS_EMBED=/path/embed.bin AEGIS_VOCAB=/path/vocab.bin \
    demo/edge-receipt/run.sh all "your prompt" 64

The receipt records the artifact SHA-256s, so a verifier on another machine must
use byte-identical files (check with sha256sum) or every verifier check fails at
"artifact hashes".

## Optional: TPM attestation of the receipt (`attest.sh`)

`attest.sh` is an independent, optional add-on. It does not change the
receipt or either verifier. The receipt answers "what was computed"; a TPM
quote answers a different question: "which TPM, holding which firmware/PCR
state, signed a statement that includes this receipt's digest, at the time
the quote was taken." A PASS on one says nothing about the other.

```
demo/edge-receipt/attest.sh quote  <receipt.txt> <outdir>    # take a TPM quote over the receipt's cis-digest
demo/edge-receipt/attest.sh verify <receipt.txt> <attestdir> # offline check, no TPM required
demo/edge-receipt/attest.sh selftest [seed-dir]              # tamper-detection self-check
```

`quote` reads either a CIS-1 decode receipt's 16-hex-char `cis-digest` line
or an agent-trace receipt's 64-hex-char `trace-chain` line (see
`demo/agent-trace/README.md`) and uses it as the quote's qualifying data
(nonce; for `trace-chain` this is its first 16 hex chars, keeping the
tpm2_quote wire format unchanged), so the resulting quote is bound to that
specific receipt. It writes `<outdir>/<receipt-basename>.attest/`
containing the AK public key, the quote message/signature/PCR values, an
optional BIOS event log, and an `ATTEST.txt` summary (format line, receipt
digest, receipt kind [`cis` or `trace`], PCR list, hierarchy, file hashes,
TPM manufacturer, host, time).

`verify` needs no TPM: it recomputes the nonce from the receipt, checks the
`ATTEST.txt` file hashes against the actual files, and runs
`tpm2_checkquote` (which needs just the AK public key, the quote files, and
`tpm2-tools` — not a TPM device) to confirm the signature and PCR values are
internally consistent. This is how attestations get checked on a
non-TPM machine such as a CI runner or a reviewer's laptop.

### What this adds

- Cryptographic evidence that a specific TPM signed a statement over the
  receipt's digest and a snapshot of PCRs 0, 2, 4, 7 (firmware/bootloader
  measurements) at quote time.

### What this does NOT prove (read before relying on this)

- **Null hierarchy is not vendor-rooted.** When the endorsement hierarchy
  is unavailable (as on today's test box), `attest.sh` falls back to a
  primary key under the TPM's null hierarchy. That key is genuinely
  TPM-resident and PCR-bound, but it carries no manufacturer certificate
  chain — nothing ties it back to "this specific physical TPM chip" for a
  third party. `hierarchy null` in `ATTEST.txt` says so explicitly.
- **An EK-certified AK (`hierarchy endorsement`) is stronger** but requires
  the endorsement hierarchy to be provisioned, which on Intel PTT
  typically happens online (the EK certificate is fetched from Intel's
  provisioning service). `attest.sh` prefers this path automatically when
  available and records which one it used.
- **PCR values do not attest to the receipt's actual decode logic.** They
  attest to firmware/bootloader measurements up to the point the quote was
  taken. Nothing here re-derives the receipt's token/digest chain — that is
  what `run.sh verify` already does, separately.
- **This covers the Linux side today.** The unikernel side
  (`aegis-uefi`) does not yet call into a TPM; that is planned, not done.
- No GPU is involved in any part of this attestation flow, and none is claimed.
- **`attest.sh` also accepts `demo/agent-trace` receipts** (a `trace-chain`
  line instead of `cis-digest`), binding a TPM quote to an agent-trace
  receipt the same way; the same limits above apply verbatim — null
  hierarchy until BIOS TPM State is enabled, and the quote proves
  firmware/PCR state bound to the receipt digest, not a vendor-rooted key.

### Self-test

`attest.sh selftest [seed-dir]` proves the verifier actually detects
tampering rather than rubber-stamping: it takes a known-good
receipt+attestation pair (generated fresh from a TPM, or from a
pre-generated seed directory containing `receipt.txt` and `attest/` — for
use on machines with no TPM), confirms it verifies OK, then (1) locates
PCR0's actual 32-byte digest inside `quote.pcrs` (by reading it back out
of `tpm2_checkquote`'s own output on the good file, so the flipped byte
is guaranteed to land on real digest content), flips one byte of it, and
confirms `verify` reports `ATTEST-FAIL` with the rejection coming from
`tpm2_checkquote`'s signature/PCR-digest check itself — not from the
`ATTEST.txt` file-hash bookkeeping, which is regenerated over the
tampered file first so it can't short-circuit the crypto check — and
(2) flips one hex character of the receipt's
`cis-digest` and confirms `verify` reports `ATTEST-FAIL` on that too.
