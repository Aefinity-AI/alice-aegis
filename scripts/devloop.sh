#!/usr/bin/env bash
# devloop.sh — the pre-claim gate for A.L.I.C.E. / Aegis.
# Run `scripts/devloop.sh gate` before claiming ANY task complete (CLAUDE.md:
# "Do not claim a task complete without running the verification command").
#
# This repo has NO root Cargo.toml by design (aegis-uefi is no_std/UEFI and
# must not share feature unification with the std crates), so every cargo
# verb here iterates the crates explicitly.
#
# Clippy is a RATCHET, not a wall: four crates carry documented pre-existing
# lint debt under -D warnings (see devloop_clippy_baseline.tsv). The ratchet
# fails only when a crate's error count EXCEEDS its recorded baseline — new
# debt is blocked, old debt is reported until deliberately paid down.
# `clippy --rebaseline` rewrites the baseline; do that only after either
# paying debt down (counts shrink) or a reviewed decision to accept new debt.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO/scripts/devloop_clippy_baseline.tsv"
CRATES=(aegis-core aegis-eval aegis-forge aegis-linux aegis-uefi xtask)
TEST_CRATES=(aegis-core aegis-eval aegis-forge aegis-linux xtask) # aegis-uefi is firmware; the boot verb covers it

clippy_flags() {
    # aegis-uefi only lints for its firmware target; --all-targets would try
    # to build a host test harness for a no_std UEFI binary.
    if [ "$1" = "aegis-uefi" ]; then
        echo "--target x86_64-unknown-uefi"
    else
        echo "--all-targets"
    fi
}

clippy_count() {
    ( cd "$REPO/$1" && cargo clippy $(clippy_flags "$1") -- -D warnings 2>&1 ) \
        | grep -E '^error(\[[A-Z0-9]+\])?' \
        | grep -v 'could not compile' \
        | grep -vc 'aborting due to'
}

verb="${1:-gate}"; shift || true
rc=0

case "$verb" in
fmt)
    for c in "${CRATES[@]}"; do
        if ( cd "$REPO/$c" && cargo fmt -- --check >/dev/null 2>&1 ); then
            echo "fmt   $c: PASS"
        else
            echo "fmt   $c: FAIL (run: cd $c && cargo fmt)"
            rc=1
        fi
    done
    ;;
clippy)
    rebase=0
    [ "${1:-}" = "--rebaseline" ] && rebase=1
    declare -A base=()
    if [ -f "$BASELINE" ]; then
        while IFS=$'\t' read -r k v; do base["$k"]="$v"; done < "$BASELINE"
    fi
    out=""
    for c in "${CRATES[@]}"; do
        n=$(clippy_count "$c")
        b="${base[$c]:-0}"
        if [ "$rebase" -eq 1 ]; then
            out+="$c\t$n\n"
            echo "clippy $c: $n error(s) — baselined"
        elif [ "$n" -gt "$b" ]; then
            echo "clippy $c: $n error(s) > baseline $b — NEW DEBT, fix before claiming done"
            rc=1
        elif [ "$n" -lt "$b" ]; then
            echo "clippy $c: $n error(s) < baseline $b — debt paid down! run 'clippy --rebaseline' to lock it in"
        else
            echo "clippy $c: $n error(s) (= baseline)"
        fi
    done
    [ "$rebase" -eq 1 ] && printf "%b" "$out" > "$BASELINE" && echo "baseline written: $BASELINE"
    ;;
test)
    for c in "${TEST_CRATES[@]}"; do
        # aegis-core owns process-global kernel toggles (force_scalar); its
        # suite is only coherent single-threaded — see the comment on
        # force_scalar_toggle_is_correct_and_reversible in
        # aegis-core/tests/gemm_equivalence.rs. Parallel harness = flaky
        # ULP mismatches when the toggle flips mid-computation.
        harness_args=""
        [ "$c" = "aegis-core" ] && harness_args="--test-threads=1"
        out=$(cd "$REPO/$c" && cargo test --quiet -- $harness_args 2>&1)
        st=$?
        if [ "$st" -eq 0 ]; then
            echo "test  $c: PASS"
            # surface cargo's own per-binary "test result: ok. N passed; ..."
            # lines instead of just printing a bare PASS with no numbers.
            echo "$out" | grep '^test result:' | sed 's/^/      /'
        else
            echo "test  $c: FAIL (run: cd $c && cargo test -- $harness_args)"
            echo "$out" | grep '^test result:' | sed 's/^/      /'
            rc=1
        fi
    done
    ;;
boot)
    # Correctness ONLY (Rule A): no number from this run may ever be quoted.
    ( cd "$REPO/xtask" && cargo run --quiet -- boot-test ) || rc=1
    ;;
figures)
    bash "$REPO/scripts/verify-figures.sh" || rc=1
    ;;
preflight)
    # verify-figures asks "is the LEDGER substantiated?"; this asks "is this
    # DOCUMENT safe to send?" — retracted figures, PII, emulation-derived
    # performance. Added 2026-08-05 after an audit found three unsupported
    # claims in funder-facing docs that verify-figures does not scan.
    bash "$REPO/scripts/preflight-outbound.sh" || rc=1
    ;;
gate)
    overall=0
    for v in fmt clippy test boot figures preflight; do
        echo "== devloop $v =="
        bash "$REPO/scripts/devloop.sh" "$v" || overall=1
    done
    if [ "$overall" -eq 0 ]; then
        echo "== GATE: PASS =="
    else
        echo "== GATE: FAIL — do not claim done =="
    fi
    exit "$overall"
    ;;
*)
    echo "usage: devloop.sh [fmt|clippy [--rebaseline]|test|boot|figures|preflight|gate]" >&2
    exit 2
    ;;
esac
exit "$rc"
