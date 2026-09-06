#!/usr/bin/env bash
# demo/agent-trace/run.sh — one-command "verified agent episode" demo.
#
# Produces an AEGIS-TRACE v0 receipt for a small, deterministic K-step agent
# episode (greedy CIS-1 FullInt decode + a scan for one `calc` tool call,
# repeated K times over the M7 tinybit model), then replays and verifies
# that receipt on this machine. A PASS means: given these three artifact
# files and this receipt, any conforming machine reproduces the exact same
# per-step token ids, tool outcomes, and trace chain. See README.md for
# what this does and does NOT prove.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# Artifacts: default to the in-repo M7 tinybit model, same default as
# demo/edge-receipt. Override with AEGIS_ARTIFACTS=<dir> (expects
# MODEL.SAF/EMBED.BIN/VOCAB.BIN) or the three files individually via
# AEGIS_MODEL / AEGIS_EMBED / AEGIS_VOCAB.
ARTIFACTS="${AEGIS_ARTIFACTS:-$ROOT/model-lab/tinybit/m7_final_gate_work/artifacts}"
MODEL="${AEGIS_MODEL:-$ARTIFACTS/MODEL.SAF}"
EMBED="${AEGIS_EMBED:-$ARTIFACTS/EMBED.BIN}"
VOCAB="${AEGIS_VOCAB:-$ARTIFACTS/VOCAB.BIN}"
OUT="$HERE/out"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export CARGO_BUILD_JOBS
# Optional LOOKUP table: unset by default (episode never scans for
# LOOKUP(...) at all). Set AEGIS_TABLE=<path> to fold a table into the
# genesis and enable the lookup tool; demo/agent-trace/tables/demo.tsv is a
# ready-made example.
TABLE="${AEGIS_TABLE:-}"
TABLE_ARGS=()
if [ -n "$TABLE" ]; then
    TABLE_ARGS=(--table "$TABLE")
fi

run() { echo "+ $*" >&2; "$@"; }

need_artifacts() {
    for f in "$MODEL" "$EMBED" "$VOCAB"; do
        if [ ! -f "$f" ]; then
            echo "missing artifact: $f (this demo never downloads anything — it uses the checked-in M7 model)" >&2
            exit 1
        fi
    done
}

cmd_build() {
    need_artifacts
    echo "== build: agent_trace (aegis-linux, release, CARGO_BUILD_JOBS=$CARGO_BUILD_JOBS) ==" >&2
    ( cd "$ROOT/aegis-linux" && run cargo build --release --example agent_trace )
    echo "build done." >&2
}

agent_trace_bin() { echo "$ROOT/aegis-linux/target/release/examples/agent_trace"; }

cmd_gen() {
    need_artifacts
    local prompt="${1:-Once upon a time}"
    local k="${2:-3}"
    local n="${3:-16}"
    mkdir -p "$OUT"
    local hostname_s; hostname_s="$(hostname)"
    local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
    local receipt="$OUT/trace-${hostname_s}-${ts}.txt"

    echo "== gen: agent_trace on $hostname_s (K=$k N=$n) ==" >&2
    run "$(agent_trace_bin)" gen "$MODEL" "$EMBED" "$VOCAB" "$k" "$n" "$prompt" "${TABLE_ARGS[@]}" > "$receipt"
    echo "wrote $receipt"
    echo "$receipt"
}

cmd_verify() {
    need_artifacts
    local receipt="${1:?usage: run.sh verify <receipt-file>}"
    if [ ! -f "$receipt" ]; then
        echo "no such receipt: $receipt" >&2
        exit 2
    fi
    echo "== verify: agent_trace verify ==" >&2
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" "${TABLE_ARGS[@]}"
}

# Three adversarial mutations of a known-good receipt, each of which MUST
# make `verify` print VERIFY FAIL and exit 1: flip a token id, flip a
# tool-output byte, drop a step line entirely.
cmd_tamper() {
    need_artifacts
    mkdir -p "$OUT"
    local good
    good="$(cmd_gen "The quick brown fox" 3 16 | tail -1)"
    echo "== tamper baseline: verifying the untouched receipt first (must PASS) ==" >&2
    if ! cmd_verify "$good" >/dev/null 2>&1; then
        echo "baseline receipt does not verify — cannot run tamper tests" >&2
        exit 1
    fi
    echo "baseline PASS" >&2

    local overall=0

    echo ""
    echo "== tamper 1/3: flip a token id ==" >&2
    local t1="$OUT/tamper-flip-token.txt"
    awk '
        BEGIN{done=0}
        /^step 0: / && done==0 {
            match($0, /toks=[0-9]+/)
            tok=substr($0, RSTART+5, RLENGTH-5)
            newtok = tok + 1
            line=$0
            sub("toks=" tok, "toks=" newtok, line)
            print line
            done=1
            next
        }
        {print}
    ' "$good" > "$t1"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t1"
    rc1=$?
    set -e
    if [ "$rc1" -ne 0 ]; then echo "tamper 1 (flip token id): FAIL as expected (exit $rc1)"; else echo "tamper 1 (flip token id): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    echo "== tamper 2/3: flip a tool-output byte ==" >&2
    # A step's tool output is legitimately empty when the tool is "no-tool"
    # (the default prompt rarely provokes the model into emitting a CALC(...)
    # call within N=16 tokens) — there is no byte to flip in an empty field.
    # So: if the first step's out= is non-empty, flip its first hex nibble;
    # if it is empty, inject one fake output byte (0x39, ASCII '9') instead.
    # Either way this mutates the tool-output field the trace chain folds in,
    # which must make verify FAIL.
    local t2="$OUT/tamper-flip-output.txt"
    awk '
        BEGIN{done=0}
        /^step / && done==0 {
            if (match($0, /out=[0-9a-f]*/)) {
                val=substr($0, RSTART+4, RLENGTH-4)
                line=$0
                if (length(val) > 0) {
                    first=substr(val,1,1)
                    if (first=="0") { newfirst="1" } else { newfirst="0" }
                    newval = newfirst substr(val,2)
                } else {
                    newval = "39"
                }
                sub("out=" val, "out=" newval, line)
                print line
                done=1
                next
            }
        }
        {print}
    ' "$good" > "$t2"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t2"
    rc2=$?
    set -e
    if [ "$rc2" -ne 0 ]; then echo "tamper 2 (flip tool-output byte): FAIL as expected (exit $rc2)"; else echo "tamper 2 (flip tool-output byte): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    echo "== tamper 3/3: drop a step ==" >&2
    local t3="$OUT/tamper-drop-step.txt"
    grep -v '^step 1: ' "$good" > "$t3"
    set +e
    "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$t3"
    rc3=$?
    set -e
    if [ "$rc3" -ne 0 ]; then echo "tamper 3 (drop a step): FAIL as expected (exit $rc3)"; else echo "tamper 3 (drop a step): DID NOT FAIL — BUG"; overall=1; fi

    echo ""
    if [ "$overall" -eq 0 ]; then
        echo "TAMPER SELFTEST: PASS — all three mutations were rejected"
    else
        echo "TAMPER SELFTEST: FAIL — see BUG lines above" >&2
        exit 1
    fi
}

attest_sh() { echo "$ROOT/demo/edge-receipt/attest.sh"; }

cmd_attest() {
    local receipt="${1:?usage: run.sh attest <receipt> <outdir>}"
    local outdir="${2:?usage: run.sh attest <receipt> <outdir>}"
    echo "== attest: TPM quote bound to trace-chain digest ==" >&2
    run "$(attest_sh)" quote "$receipt" "$outdir"
}

# Runs both independent checks: agent_trace verify (replays the episode)
# and attest.sh verify (checks the TPM quote against the receipt digest,
# offline, no TPM needed). Prints one VERIFY: and one ATTEST: line and
# exits nonzero if either check fails.
cmd_verify_attested() {
    need_artifacts
    local receipt="${1:?usage: run.sh verify-attested <receipt> <attestdir>}"
    local attestdir="${2:?usage: run.sh verify-attested <receipt> <attestdir>}"
    if [ ! -f "$receipt" ]; then
        echo "no such receipt: $receipt" >&2
        exit 2
    fi

    local overall=0

    echo "== verify-attested: agent_trace verify ==" >&2
    if "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" "${TABLE_ARGS[@]}"; then
        echo "VERIFY: PASS"
    else
        echo "VERIFY: FAIL"
        overall=1
    fi

    echo "== verify-attested: attest.sh verify ==" >&2
    if "$(attest_sh)" verify "$receipt" "$attestdir"; then
        echo "ATTEST-OK"
    else
        echo "ATTEST-FAIL"
        overall=1
    fi

    return "$overall"
}

sha256_field() {
    # sha256sum a file, print just the hex digest.
    sha256sum "$1" | awk '{print $1}'
}

# Extracts "<name> <value>" from a MANIFEST.txt, tolerating "no match"
# under `set -e` (a plain `grep -m1 ... | awk ...` assignment would abort
# the whole script on no-match, since grep exits 1 and pipefail is on).
# Prints the value (possibly empty) and always returns 0.
manifest_field() {
    local name="$1" file="$2"
    grep -m1 "^$name " "$file" 2>/dev/null | awk '{print $2}' || true
}

# A missing or malformed MANIFEST metadata field is a corrupt/foreign
# bundle, not "artifacts differ" — but it is grouped under the same exit
# code (3) as an artifact-triple mismatch, since both mean "cannot trust
# the artifact-identity claim in this bundle" and both must be caught
# before any replay is attempted.
require_manifest_hex() {
    local name="$1" val="$2"
    if [ -z "$val" ]; then
        echo "MANIFEST malformed: missing $name" >&2
        exit 3
    fi
    if ! [[ "$val" =~ ^[0-9a-f]{64}$ ]]; then
        echo "MANIFEST malformed: $name is not 64 lowercase hex chars: '$val'" >&2
        exit 3
    fi
}

require_manifest_flag() {
    local name="$1" val="$2"
    if [ -z "$val" ]; then
        echo "MANIFEST malformed: missing $name" >&2
        exit 3
    fi
    if [ "$val" != "yes" ] && [ "$val" != "no" ]; then
        echo "MANIFEST malformed: $name is not yes|no: '$val'" >&2
        exit 3
    fi
}

# Validates every member of a tar bundle BEFORE extraction: no absolute
# paths, no `..` path-traversal segments, and every member must be a
# plain file or directory (no symlinks, devices, etc). Exits 2 (labeled)
# on the first violation, or if `tar tvf` itself fails to list the file.
check_bundle_members() {
    local bundle="$1"
    local listing tvf_err
    tvf_err="$(mktemp)"
    # stderr kept separate from the listing: tar prints warnings (e.g.
    # "Removing leading `../' from member names") on stderr that must
    # never be parsed as a listing line.
    if ! listing="$(tar tvf "$bundle" 2>"$tvf_err")"; then
        echo "bundle: extraction failed (tar could not list $bundle)" >&2
        cat "$tvf_err" >&2
        rm -f "$tvf_err"
        exit 2
    fi
    rm -f "$tvf_err"
    local line type path
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        type="${line:0:1}"
        if [ "$type" = "l" ]; then
            # symlink listing lines look like "lrwxrwxrwx ... name -> target";
            # the member name is before " -> ", not the last field (which is
            # the (possibly attacker-controlled) link target).
            path="${line%% -> *}"
            path="${path##* }"
        else
            path="${line##* }"
        fi
        path="${path#./}"
        path="${path%/}"
        [ -n "$path" ] || continue
        case "$type" in
            -|d) ;;
            *)
                echo "bundle: unsafe member $path (not a regular file or directory)" >&2
                exit 2
                ;;
        esac
        case "$path" in
            /*)
                echo "bundle: unsafe member $path (absolute path)" >&2
                exit 2
                ;;
        esac
        case "/$path/" in
            *"/../"*)
                echo "bundle: unsafe member $path (path traversal)" >&2
                exit 2
                ;;
        esac
    done <<< "$listing"
}

# Packs a receipt (plus, optionally, its LOOKUP table and/or its TPM
# attestation directory) into one tar file for a single-scp cross-machine
# handoff. See README.md "Bundle format" for exactly what is in it and
# what verify-bundle checks.
cmd_pack() {
    need_artifacts
    local receipt="${1:?usage: run.sh pack <receipt> [attestdir]}"
    local attestdir="${2:-}"
    if [ ! -f "$receipt" ]; then
        echo "no such receipt: $receipt" >&2
        exit 2
    fi
    mkdir -p "$OUT"

    local base; base="$(basename "$receipt")"
    local stem="${base%.txt}"
    local work; work="$(mktemp -d)"
    trap 'rm -rf "$work"' RETURN

    cp "$receipt" "$work/receipt.txt"

    local has_table=no
    if [ -n "$TABLE" ]; then
        if [ ! -f "$TABLE" ]; then
            echo "AEGIS_TABLE is set but not found: $TABLE" >&2
            exit 2
        fi
        if ! grep -q '^table-sha256 ' "$receipt"; then
            echo "note: AEGIS_TABLE is set but $receipt has no table-sha256 line — this receipt's episode never consulted a table; agent_trace verify will print its own stderr note and ignore --table (see README 'LOOKUP tool'); packing the table anyway for provenance" >&2
        fi
        cp "$TABLE" "$work/table.tsv"
        has_table=yes
    fi

    local has_attest=no
    if [ -n "$attestdir" ]; then
        if [ ! -d "$attestdir" ]; then
            echo "no such attest dir: $attestdir" >&2
            exit 2
        fi
        mkdir -p "$work/attest"
        cp -a "$attestdir"/. "$work/attest/"
        has_attest=yes
    fi

    {
        # sha256sum-style lines for every bundle member (paths relative to
        # the bundle root), in a fixed, sorted order.
        ( cd "$work" && find . -type f ! -name MANIFEST.txt | sed 's|^\./||' | sort )
    } | while IFS= read -r rel; do
        sha256_field "$work/$rel" | awk -v p="$rel" '{print $1"  "p}'
    done > "$work/MANIFEST.txt"

    {
        echo "artifact-model-sha256 $(sha256_field "$MODEL")"
        echo "artifact-embed-sha256 $(sha256_field "$EMBED")"
        echo "artifact-vocab-sha256 $(sha256_field "$VOCAB")"
        echo "generator-host $(hostname)"
        echo "packed-utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "has-table $has_table"
        echo "has-attest $has_attest"
    } >> "$work/MANIFEST.txt"

    local bundle="$OUT/${stem}.bundle.tar"
    if [ -e "$bundle" ] && [ "${AEGIS_FORCE:-0}" != "1" ]; then
        echo "refusing to overwrite existing bundle: $bundle (two receipts share a basename — set AEGIS_FORCE=1 to overwrite)" >&2
        exit 2
    fi
    ( cd "$work" && run tar cf "$bundle" --sort=name --numeric-owner --owner=0 --group=0 . )
    echo "wrote $bundle" >&2
    echo "$bundle"
}

# Shared helper: prints "VERIFY: PASS"/"VERIFY: FAIL" for an agent_trace
# verify call, returns its exit status.
_do_agent_trace_verify() {
    local receipt="$1"; shift
    if "$(agent_trace_bin)" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" "$@"; then
        echo "VERIFY: PASS"
        return 0
    else
        echo "VERIFY: FAIL"
        return 1
    fi
}

# Extracts a bundle produced by `pack` and runs, in order: bundle member
# safety check (no absolute/traversal paths, files/dirs only), MANIFEST
# member check (every listed member's hash matches, and no extra unlisted
# files exist), artifact-triple check (BEFORE any replay), agent_trace
# verify (using ONLY the bundle's own table.tsv, never $AEGIS_TABLE — the
# bundle is authoritative about what it was generated against), and
# attest.sh verify if the bundle carries a quote.
#
# Exit codes: 0 pass; 1 bundle/MANIFEST integrity failure (member hash
# mismatch or unlisted member) or a failed replay/attest check; 2 usage
# error, unsafe tar member, or extraction failure; 3 artifact-triple
# mismatch or malformed/missing MANIFEST metadata field — always before
# any replay is attempted.
cmd_verify_bundle() {
    need_artifacts
    local bundle="${1:?usage: run.sh verify-bundle <bundle.tar>}"
    if [ ! -f "$bundle" ]; then
        echo "no such bundle: $bundle" >&2
        exit 2
    fi

    local base; base="$(basename "$bundle")"
    local dir="$OUT/bundle-${base%.tar}"
    rm -rf "$dir"
    mkdir -p "$dir"

    echo "== verify-bundle: bundle member safety check ==" >&2
    check_bundle_members "$bundle"
    echo "bundle members: safe (no absolute paths, no traversal, files/dirs only)" >&2

    local tar_err; tar_err="$(mktemp)"
    if ! tar xf "$bundle" -C "$dir" 2>"$tar_err"; then
        echo "bundle: extraction failed" >&2
        cat "$tar_err" >&2
        rm -f "$tar_err"
        exit 2
    fi
    rm -f "$tar_err"

    echo "== verify-bundle: MANIFEST member check ==" >&2
    if [ ! -f "$dir/MANIFEST.txt" ]; then
        echo "no MANIFEST.txt in bundle" >&2
        exit 1
    fi
    local mismatch="" listed_tmp actual_tmp
    while IFS= read -r line; do
        case "$line" in
            "") continue ;;
            artifact-*|generator-host*|packed-utc*|has-table*|has-attest*) continue ;;
        esac
        local exp_hash exp_path
        exp_hash="$(echo "$line" | awk '{print $1}')"
        exp_path="$(echo "$line" | awk '{print $2}')"
        [ -f "$dir/$exp_path" ] || { mismatch="$exp_path (missing)"; break; }
        local got_hash; got_hash="$(sha256_field "$dir/$exp_path")"
        if [ "$got_hash" != "$exp_hash" ]; then
            mismatch="$exp_path"
            break
        fi
    done < "$dir/MANIFEST.txt"
    if [ -n "$mismatch" ]; then
        echo "MANIFEST mismatch: $mismatch" >&2
        exit 1
    fi
    echo "MANIFEST: all members match" >&2

    echo "== verify-bundle: unlisted member check ==" >&2
    listed_tmp="$(mktemp)"; actual_tmp="$(mktemp)"
    grep -E '^[0-9a-f]{64}  ' "$dir/MANIFEST.txt" | awk '{print $2}' | sort > "$listed_tmp"
    ( cd "$dir" && find . -type f ! -name MANIFEST.txt | sed 's|^\./||' | sort ) > "$actual_tmp"
    local unlisted; unlisted="$(comm -13 "$listed_tmp" "$actual_tmp" || true)"
    rm -f "$listed_tmp" "$actual_tmp"
    if [ -n "$unlisted" ]; then
        echo "MANIFEST: unlisted member $(echo "$unlisted" | head -1)" >&2
        exit 1
    fi
    echo "MANIFEST: no unlisted members" >&2

    echo "== verify-bundle: artifact-triple check ==" >&2
    local exp_model exp_embed exp_vocab got_model got_embed got_vocab
    exp_model="$(manifest_field artifact-model-sha256 "$dir/MANIFEST.txt")"
    exp_embed="$(manifest_field artifact-embed-sha256 "$dir/MANIFEST.txt")"
    exp_vocab="$(manifest_field artifact-vocab-sha256 "$dir/MANIFEST.txt")"
    require_manifest_hex artifact-model-sha256 "$exp_model"
    require_manifest_hex artifact-embed-sha256 "$exp_embed"
    require_manifest_hex artifact-vocab-sha256 "$exp_vocab"
    got_model="$(sha256_field "$MODEL")"
    got_embed="$(sha256_field "$EMBED")"
    got_vocab="$(sha256_field "$VOCAB")"
    if [ "$got_model" != "$exp_model" ]; then
        echo "artifact mismatch: MODEL differs (expected $exp_model, local $got_model) — this machine does not hold the same artifact triple as the generator" >&2
        exit 3
    fi
    if [ "$got_embed" != "$exp_embed" ]; then
        echo "artifact mismatch: EMBED differs (expected $exp_embed, local $got_embed) — this machine does not hold the same artifact triple as the generator" >&2
        exit 3
    fi
    if [ "$got_vocab" != "$exp_vocab" ]; then
        echo "artifact mismatch: VOCAB differs (expected $exp_vocab, local $got_vocab) — this machine does not hold the same artifact triple as the generator" >&2
        exit 3
    fi
    echo "artifact triple matches" >&2

    local has_table has_attest
    has_table="$(manifest_field has-table "$dir/MANIFEST.txt")"
    has_attest="$(manifest_field has-attest "$dir/MANIFEST.txt")"
    require_manifest_flag has-table "$has_table"
    require_manifest_flag has-attest "$has_attest"

    echo "== verify-bundle: agent_trace verify ==" >&2
    local overall=0
    if [ "$has_table" = "yes" ]; then
        # Bundle table, not $AEGIS_TABLE: the bundle is authoritative about
        # what it was generated against.
        _do_agent_trace_verify "$dir/receipt.txt" --table "$dir/table.tsv" || overall=1
    else
        _do_agent_trace_verify "$dir/receipt.txt" || overall=1
    fi

    if [ "$has_attest" = "yes" ]; then
        echo "== verify-bundle: attest.sh verify ==" >&2
        if "$(attest_sh)" verify "$dir/receipt.txt" "$dir/attest"; then
            echo "ATTEST-OK"
        else
            echo "ATTEST-FAIL"
            overall=1
        fi
    else
        echo "ATTEST: none (bundle carries no quote)"
    fi

    return "$overall"
}

cmd_all() {
    local prompt="${1:-Once upon a time}"
    local k="${2:-3}"
    local n="${3:-16}"
    cmd_build
    local receipt
    receipt="$(cmd_gen "$prompt" "$k" "$n" | tail -1)"
    cmd_verify "$receipt"
    cmd_tamper
    if [ -e /dev/tpmrm0 ] && command -v tpm2_quote >/dev/null 2>&1; then
        cmd_attest "$receipt" "$OUT"
    else
        echo "attest: skipped (no TPM)" >&2
    fi
}

case "${1:-}" in
    build)           shift; cmd_build "$@" ;;
    gen)             shift; cmd_gen "$@" ;;
    verify)          shift; cmd_verify "$@" ;;
    tamper)          shift; cmd_tamper "$@" ;;
    attest)          shift; cmd_attest "$@" ;;
    verify-attested) shift; cmd_verify_attested "$@" ;;
    pack)            shift; cmd_pack "$@" ;;
    verify-bundle)   shift; cmd_verify_bundle "$@" ;;
    all)             shift; cmd_all "$@" ;;
    *)
        echo "usage: $0 {build|gen [prompt] [K] [N]|verify <receipt>|tamper|attest <receipt> <outdir>|verify-attested <receipt> <attestdir>|pack <receipt> [attestdir]|verify-bundle <bundle.tar>|all [prompt] [K] [N]}" >&2
        exit 2
        ;;
esac
