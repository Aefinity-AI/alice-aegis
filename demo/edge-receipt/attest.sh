#!/usr/bin/env bash
# demo/edge-receipt/attest.sh — OPTIONAL TPM attestation for a CIS-1 decode
# receipt (`cis-digest`) or a demo/agent-trace receipt (`trace-chain`).
#
# The receipt (produced by run.sh or demo/agent-trace/run.sh) proves *what
# was computed*: given these artifact files, this exact decode (or agent
# episode) reproduces this exact token/digest chain. This script adds an
# independent, optional layer that proves *which TPM/firmware state signed
# a quote* at the moment the receipt's digest was fed to the TPM as
# qualifying data (nonce). The two checks are unrelated: a PASS on
# cis_witness/cis-verify or demo/agent-trace's verify says nothing about
# the TPM, and a PASS on `attest.sh verify` says nothing about the decode
# or trace replay itself. See README.md for exactly what this does and
# does NOT prove.
#
# Subcommands:
#   attest.sh quote  <receipt.txt> <outdir>     # generate a TPM quote over the receipt digest
#   attest.sh verify <receipt.txt> <attestdir>  # offline-checkable verification (no TPM needed)
#   attest.sh selftest                          # tamper-detection self-check (needs a TPM once, to seed fixtures)
set -euo pipefail

PCR_LIST="sha256:0,2,4,7"
FORMAT_LINE="format AEGIS-ATTEST v0"

die() { echo "attest.sh: error: $*" >&2; exit 1; }

need_bin() {
    command -v "$1" >/dev/null 2>&1 || die "required binary '$1' not found on PATH"
}

# Extract the receipt digest from either a CIS-1 decode receipt
# (`cis-digest <16hex>`) or an agent-trace receipt (`trace-chain <64hex>`).
# Prints "<nonce> <kind> <full>" where:
#   nonce - the 16-hex-char TPM quote qualifying data (wire format unchanged:
#           for cis-digest this IS the whole digest; for trace-chain it is
#           the first 16 hex chars of the 64-hex chain digest).
#   kind  - "cis" or "trace".
#   full  - the whole digest as it appears in the receipt (== nonce for cis;
#           the full 64-hex trace-chain for trace), recorded in ATTEST.txt
#           so tampering anywhere in the trace-chain, not just its first 16
#           hex chars, is caught by `verify`.
receipt_nonce() {
    local receipt="$1"
    local line d
    line="$(grep -m1 '^cis-digest ' "$receipt" || true)"
    if [ -n "$line" ]; then
        d="$(echo "$line" | awk '{print $2}')"
        echo "$d" | grep -Eq '^[0-9a-f]{16}$' || die "cis-digest '$d' is not 16 lowercase hex chars"
        echo "$d cis $d"
        return
    fi
    line="$(grep -m1 '^trace-chain ' "$receipt" || true)"
    if [ -n "$line" ]; then
        d="$(echo "$line" | awk '{print $2}')"
        echo "$d" | grep -Eq '^[0-9a-f]{64}$' || die "trace-chain '$d' is not 64 lowercase hex chars"
        echo "${d:0:16} trace $d"
        return
    fi
    die "no 'cis-digest' or 'trace-chain' line found in $receipt"
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

# ---------------------------------------------------------------- quote ---

cmd_quote() {
    local receipt="${1:?usage: attest.sh quote <receipt.txt> <outdir>}"
    local outdir="${2:?usage: attest.sh quote <receipt.txt> <outdir>}"
    [ -f "$receipt" ] || die "receipt not found: $receipt"

    for b in tpm2_createprimary tpm2_create tpm2_load tpm2_readpublic tpm2_quote tpm2_getcap sha256sum; do
        need_bin "$b"
    done
    [ -e /dev/tpmrm0 ] || die "no /dev/tpmrm0 (no TPM resource manager device on this host)"

    local nonce kind full
    read -r nonce kind full <<< "$(receipt_nonce "$receipt")"

    local base attestdir
    base="$(basename "$receipt")"
    attestdir="$outdir/${base}.attest"
    mkdir -p "$attestdir"

    local work
    work="$(mktemp -d)"
    # shellcheck disable=SC2064  # intentional: expand $work now, not at RETURN time
    trap "rm -rf '$work'" RETURN

    local hierarchy="null"
    local prim_ctx="$work/prim.ctx"
    local ak_ctx="$work/ak.ctx"
    local ak_pub="$work/ak.pub"
    local ak_priv="$work/ak.priv"
    local ak_name="$work/ak.name"

    # Prefer an EK-certified AK if the endorsement hierarchy is usable;
    # fall back to a null-hierarchy AK (not vendor-rooted, but still a
    # genuine TPM-resident signing key bound to the current PCR state).
    if tpm2_createek -c "$work/ek.ctx" -G rsa -u "$work/ek.pub" >/dev/null 2>&1; then
        if tpm2_createak -C "$work/ek.ctx" -c "$ak_ctx" -G rsa -g sha256 -s rsassa \
            -u "$ak_pub" -r "$ak_priv" -n "$ak_name" >/dev/null 2>&1; then
            hierarchy="endorsement"
        fi
    fi

    if [ "$hierarchy" = "null" ]; then
        tpm2_createprimary -C n -G rsa -c "$prim_ctx" >/dev/null
        tpm2_create -C "$prim_ctx" -G rsa2048:rsassa:null -g sha256 \
            -a "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|restricted|sign|noda" \
            -u "$ak_pub" -r "$ak_priv" >/dev/null
        tpm2_load -C "$prim_ctx" -u "$ak_pub" -r "$ak_priv" -c "$ak_ctx" -n "$ak_name" >/dev/null
    fi

    tpm2_readpublic -c "$ak_ctx" -f pem -o "$attestdir/ak.pem" >/dev/null

    tpm2_quote -c "$ak_ctx" -l "$PCR_LIST" -q "$nonce" \
        -m "$attestdir/quote.msg" -s "$attestdir/quote.sig" -o "$attestdir/quote.pcrs" \
        -g sha256 >/dev/null

    local eventlog_sha="none"
    if [ -r /sys/kernel/security/tpm0/binary_bios_measurements ]; then
        if cp /sys/kernel/security/tpm0/binary_bios_measurements "$attestdir/eventlog.bin" 2>/dev/null; then
            eventlog_sha="$(sha256_of "$attestdir/eventlog.bin")"
        fi
    fi

    local manufacturer
    manufacturer="$(tpm2_getcap properties-fixed 2>/dev/null | awk '/TPM2_PT_MANUFACTURER:/{getline; print; exit}' | awk '{print $2}')"
    manufacturer="${manufacturer:-unknown}"

    {
        echo "$FORMAT_LINE"
        echo "receipt-digest $full"
        echo "receipt-kind $kind"
        echo "pcr-list $PCR_LIST"
        echo "hierarchy $hierarchy"
        echo "ak-sha256 $(sha256_of "$attestdir/ak.pem")"
        echo "quote-sha256 $(sha256_of "$attestdir/quote.msg")"
        echo "sig-sha256 $(sha256_of "$attestdir/quote.sig")"
        echo "pcrs-sha256 $(sha256_of "$attestdir/quote.pcrs")"
        echo "eventlog-sha256 $eventlog_sha"
        echo "tpm-manufacturer $manufacturer"
        echo "host $(hostname)"
        echo "time-utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$attestdir/ATTEST.txt"

    echo "wrote $attestdir" >&2
    echo "$attestdir"
}

# --------------------------------------------------------------- verify ---

cmd_verify() {
    local receipt="${1:?usage: attest.sh verify <receipt.txt> <attestdir>}"
    local attestdir="${2:?usage: attest.sh verify <receipt.txt> <attestdir>}"
    need_bin tpm2_checkquote
    need_bin sha256sum

    [ -f "$receipt" ] || { echo "VERDICT: ATTEST-FAIL (receipt not found: $receipt)"; return 1; }
    [ -f "$attestdir/ATTEST.txt" ] || { echo "VERDICT: ATTEST-FAIL (no ATTEST.txt in $attestdir)"; return 1; }

    local nonce_kind_full
    if ! nonce_kind_full="$(receipt_nonce "$receipt")"; then
        echo "VERDICT: ATTEST-FAIL (could not read receipt digest)"
        return 1
    fi
    local nonce kind full
    read -r nonce kind full <<< "$nonce_kind_full"

    get_field() { grep -m1 "^$1 " "$attestdir/ATTEST.txt" | cut -d' ' -f2-; }

    local fmt att_digest att_kind pcr_list hierarchy ak_sha quote_sha sig_sha pcrs_sha eventlog_sha
    fmt="$(get_field format)"; fmt="format $fmt"
    att_digest="$(get_field receipt-digest)"
    att_kind="$(get_field receipt-kind)"
    pcr_list="$(get_field pcr-list)"
    hierarchy="$(get_field hierarchy)"
    ak_sha="$(get_field ak-sha256)"
    quote_sha="$(get_field quote-sha256)"
    sig_sha="$(get_field sig-sha256)"
    pcrs_sha="$(get_field pcrs-sha256)"
    eventlog_sha="$(get_field eventlog-sha256)"

    if [ "$fmt" != "$FORMAT_LINE" ]; then
        echo "VERDICT: ATTEST-FAIL (unrecognized ATTEST.txt format: '$fmt')"
        return 1
    fi

    if [ "$att_kind" != "$kind" ]; then
        echo "VERDICT: ATTEST-FAIL (receipt kind $kind does not match ATTEST.txt receipt-kind $att_kind)"
        return 1
    fi

    if [ "$att_digest" != "$full" ]; then
        echo "VERDICT: ATTEST-FAIL (receipt digest $full does not match ATTEST.txt receipt-digest $att_digest)"
        return 1
    fi

    for pair in "ak.pem:$ak_sha" "quote.msg:$quote_sha" "quote.sig:$sig_sha" "quote.pcrs:$pcrs_sha"; do
        local fname="${pair%%:*}"
        local expect="${pair#*:}"
        [ -f "$attestdir/$fname" ] || { echo "VERDICT: ATTEST-FAIL (missing $fname in $attestdir)"; return 1; }
        local got
        got="$(sha256_of "$attestdir/$fname")"
        if [ "$got" != "$expect" ]; then
            echo "VERDICT: ATTEST-FAIL ($fname sha256 mismatch: ATTEST.txt says $expect, file is $got)"
            return 1
        fi
    done

    if [ "$eventlog_sha" != "none" ]; then
        if [ ! -f "$attestdir/eventlog.bin" ]; then
            echo "VERDICT: ATTEST-FAIL (ATTEST.txt records an eventlog-sha256 but eventlog.bin is missing)"
            return 1
        fi
        local got_ev
        got_ev="$(sha256_of "$attestdir/eventlog.bin")"
        if [ "$got_ev" != "$eventlog_sha" ]; then
            echo "VERDICT: ATTEST-FAIL (eventlog.bin sha256 mismatch)"
            return 1
        fi
    fi

    local checkquote_out
    if ! checkquote_out="$(tpm2_checkquote -u "$attestdir/ak.pem" \
        -m "$attestdir/quote.msg" -s "$attestdir/quote.sig" -f "$attestdir/quote.pcrs" \
        -g sha256 -q "$nonce" 2>&1)"; then
        echo "VERDICT: ATTEST-FAIL (tpm2_checkquote rejected the quote/signature/PCR set)"
        echo "$checkquote_out" >&2
        return 1
    fi

    echo "pcr-list: $pcr_list  hierarchy: $hierarchy  ak-sha256: $ak_sha"
    echo "PCR values (from quote.pcrs):"
    echo "$checkquote_out" | sed -n '/pcrs:/,$p'

    echo "VERDICT: ATTEST-OK (receipt-kind $kind, receipt digest $full matches signed quote, $hierarchy hierarchy, $pcr_list)"
}

# -------------------------------------------------------------- selftest --

# Read PCR0's 32-byte digest as tpm2_checkquote itself reports it (the
# same "  0 : 0x<hex>" line cmd_verify prints), so we know exactly which
# bytes in quote.pcrs are the real PCR0 digest rather than guessing at
# the TPML_PCR_SELECTION / TPML_DIGEST binary layout.
read_pcr0_hex() {
    local attestdir="$1" nonce="$2"
    tpm2_checkquote -u "$attestdir/ak.pem" -m "$attestdir/quote.msg" \
        -s "$attestdir/quote.sig" -f "$attestdir/quote.pcrs" -g sha256 -q "$nonce" 2>/dev/null \
        | awk '/^ *0 : 0x/{print $3; exit}'
}

# Flip one byte inside PCR0's actual 32-byte digest value within a
# quote.pcrs file, by locating that exact byte string (looked up via
# read_pcr0_hex on the still-good file) and flipping its first byte.
# This guarantees the tamper lands on real digest content, never on
# padding, and that tpm2_checkquote's own signature/PCR-digest check
# (not just the ATTEST.txt file-hash check) is what rejects it.
flip_pcr_byte() {
    local file="$1" pcr0_hex="$2"
    python3 - "$file" "$pcr0_hex" <<'PYEOF'
import sys
path, hexval = sys.argv[1], sys.argv[2]
if hexval.lower().startswith("0x"):
    hexval = hexval[2:]
target = bytes.fromhex(hexval)
with open(path, "rb") as f:
    data = bytearray(f.read())
idx = data.find(target)
if idx < 0:
    sys.exit("flip_pcr_byte: PCR0 digest bytes not found in quote.pcrs")
data[idx] ^= 0xFF
with open(path, "wb") as f:
    f.write(data)
PYEOF
}

cmd_selftest() {
    local seed="${1:-}"
    local work
    work="$(mktemp -d)"
    # shellcheck disable=SC2064  # intentional: expand $work now, not at RETURN time
    trap "rm -rf '$work'" RETURN

    local receipt="$work/receipt.txt"
    local goodattest="$work/good.attest"

    if [ -n "$seed" ]; then
        [ -d "$seed" ] || die "selftest seed dir not found: $seed"
        cp "$seed/receipt.txt" "$receipt"
        cp -r "$seed/attest" "$goodattest"
    else
        [ -e /dev/tpmrm0 ] || die "no TPM on this host; run 'attest.sh selftest <seed-dir>' with a pre-generated fixture (receipt.txt + attest/) instead"
        echo "cis-digest $(head -c8 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$receipt"
        local d
        d="$(cmd_quote "$receipt" "$work")"
        mv "$d" "$goodattest"
    fi

    local nonce kind full
    read -r nonce kind full <<< "$(receipt_nonce "$receipt")"

    echo "== selftest: unmodified attestation must verify OK ==" >&2
    cmd_verify "$receipt" "$goodattest" >&2 || die "selftest: baseline attestation did not verify (fixtures are bad)"

    echo "== selftest: flipping PCR0's digest must fail verification (via tpm2_checkquote) ==" >&2
    local pcr0_hex
    pcr0_hex="$(read_pcr0_hex "$goodattest" "$nonce")"
    [ -n "$pcr0_hex" ] || die "selftest: could not read PCR0's digest from the good attestation"

    local tampered_pcr="$work/tampered_pcr.attest"
    cp -r "$goodattest" "$tampered_pcr"
    flip_pcr_byte "$tampered_pcr/quote.pcrs" "$pcr0_hex"
    # Regenerate pcrs-sha256 over the tampered file so ATTEST.txt's own
    # file-integrity check (which is not the property under test here)
    # doesn't short-circuit before tpm2_checkquote gets to reject the
    # tampered PCR digest itself.
    sed -i "s/^pcrs-sha256 .*/pcrs-sha256 $(sha256_of "$tampered_pcr/quote.pcrs")/" "$tampered_pcr/ATTEST.txt"

    local verify_out
    if verify_out="$(cmd_verify "$receipt" "$tampered_pcr" 2>&1)"; then
        echo "$verify_out" >&2
        die "selftest FAILED: tampered PCR value verified as OK"
    fi
    echo "$verify_out" >&2
    case "$verify_out" in
        *tpm2_checkquote*) ;;
        *) die "selftest FAILED: tampered PCR was rejected for the wrong reason (expected a tpm2_checkquote failure): $verify_out" ;;
    esac
    echo "selftest: tampered PCR digest correctly rejected by tpm2_checkquote" >&2

    echo "== selftest: flipping the receipt digest must fail verification ==" >&2
    local tampered_receipt="$work/tampered-receipt.txt"
    # Toggle the last hex character of the cis-digest line.
    awk '{
        if ($1 == "cis-digest") {
            d = $2
            last = substr(d, length(d), 1)
            rest = substr(d, 1, length(d)-1)
            newlast = (last == "0") ? "1" : "0"
            print $1, rest newlast
        } else { print }
    }' "$receipt" > "$tampered_receipt"
    if cmd_verify "$tampered_receipt" "$goodattest" >&2; then
        die "selftest FAILED: tampered receipt digest verified as OK"
    fi
    echo "selftest: tampered receipt digest correctly rejected" >&2

    if [ -z "$seed" ]; then
        echo "== selftest: synthetic trace-chain (agent-trace) receipt must verify OK ==" >&2
        local trace_receipt="$work/trace-receipt.txt"
        echo "trace-chain $(head -c32 /dev/urandom | od -An -tx1 | tr -d ' \n')" > "$trace_receipt"
        local trace_attestdir
        trace_attestdir="$(cmd_quote "$trace_receipt" "$work")"

        cmd_verify "$trace_receipt" "$trace_attestdir" >&2 \
            || die "selftest: baseline trace-chain attestation did not verify"

        echo "== selftest: flipping the trace-chain digest's last hex char must fail verification ==" >&2
        local tampered_trace="$work/tampered-trace-receipt.txt"
        awk '{
            if ($1 == "trace-chain") {
                d = $2
                last = substr(d, length(d), 1)
                rest = substr(d, 1, length(d)-1)
                newlast = (last == "0") ? "1" : "0"
                print $1, rest newlast
            } else { print }
        }' "$trace_receipt" > "$tampered_trace"
        if cmd_verify "$tampered_trace" "$trace_attestdir" >&2; then
            die "selftest FAILED: tampered trace-chain digest verified as OK"
        fi
        echo "selftest: tampered trace-chain digest correctly rejected" >&2
    fi

    echo "SELFTEST: PASS"
}

# ----------------------------------------------------------------- main --

main() {
    local cmd="${1:-}"
    case "$cmd" in
        quote)  shift; cmd_quote "$@" ;;
        verify) shift; cmd_verify "$@" ;;
        selftest) shift; cmd_selftest "$@" ;;
        *)
            echo "usage: attest.sh {quote <receipt.txt> <outdir> | verify <receipt.txt> <attestdir> | selftest [seed-dir]}" >&2
            exit 2
            ;;
    esac
}

main "$@"
