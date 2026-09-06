#!/usr/bin/env bash
# demo/agent-trace/eval/run_suite.sh — run a tool-call eval suite TSV
# through agent_trace gen + verify, one AEGIS-TRACE episode per item, and
# append one row per item to <outdir>/summary.tsv.
#
# Usage:
#   run_suite.sh <items.tsv> <outdir> [--template T1|T3] [--bin <agent_trace>] [--limit N] [--attest]
#
# --attest: after each item's receipt is generated and verified, also runs
# demo/agent-trace/run.sh attest <receipt> <outdir>/attest/<item_id> (the
# existing TPM-quote-over-the-full-receipt-digest attestation; see
# demo/edge-receipt/attest.sh). Never touches an item's verify_result column
# (attestation is an independent, optional check). Adds an eleventh
# attest_rc column to summary.tsv (only when --attest is given — without
# the flag, summary.tsv/RUN.txt/timing.tsv are byte-identical to a run
# without this flag) and three lines to RUN.txt: attest-requested,
# attest-count, attest-fail. See eval/README.md.
#
# Env:
#   AEGIS_MODEL / AEGIS_EMBED / AEGIS_VOCAB   — artifact file paths, OR
#   AEGIS_ARTIFACTS=<dir>                     — expects MODEL.SAF/EMBED.BIN/VOCAB.BIN
#     (same convention as demo/agent-trace/run.sh)
#   AGENT_TRACE_BIN   — override for the agent_trace binary (same as --bin)
#   N                 — tokens per decode step, default 24 (plan section 3)
#
# Table: any item whose expected_tool involves LOOKUP (LOOKUP alone, or a
# comma-joined mixed pair containing LOOKUP) is run with
# --table demo/agent-trace/tables/demo.tsv. CALC-only and distractor items
# run with no table (matching run.sh's table-less default: LOOKUP( is not
# even scanned for without --table).
#
# --template T3 is an ablation: every item's prompt has its trailing "A:"
# replaced with "A: CALC(" (unclosed), per the plan's T3 definition. This is
# applied to every item uniformly (including LOOKUP-only items), which is a
# generalization beyond the plan's literal "same content as T1" wording;
# logged here as a deviation, not silently decided — see eval/README.md.
#
# Resumable: an item_id already present in <outdir>/summary.tsv is skipped.
# Never aborts the whole run on one item's gen/verify failure — records
# verify_result FAIL (or a receipt-parse failure) for that item and
# continues to the next.
#
# Every gen/verify call passes --suite-sha256 <sha256 of ITEMS>, so the
# suite file's hash is folded into each receipt's trace genesis (agent_trace
# --suite-sha256; see eval/README.md) — a receipt from this run only
# verifies against the same suite TSV bytes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
TABLE_FILE="$HERE/../tables/demo.tsv"

usage() {
    echo "usage: run_suite.sh <items.tsv> <outdir> [--template T1|T3] [--bin <agent_trace>] [--limit N] [--attest]" >&2
    exit 2
}

ITEMS="${1:?$(usage)}"
OUTDIR="${2:?$(usage)}"
shift 2

TEMPLATE="T1"
BIN_OVERRIDE=""
LIMIT=0
ATTEST=0

while [ $# -gt 0 ]; do
    case "$1" in
        --template)
            TEMPLATE="${2:?--template needs an argument}"
            shift 2
            ;;
        --bin)
            BIN_OVERRIDE="${2:?--bin needs an argument}"
            shift 2
            ;;
        --limit)
            LIMIT="${2:?--limit needs an argument}"
            shift 2
            ;;
        --attest)
            ATTEST=1
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            ;;
    esac
done

case "$TEMPLATE" in
    T1|T3) ;;
    *) echo "invalid --template: $TEMPLATE (want T1 or T3)" >&2; exit 2 ;;
esac

if [ ! -f "$ITEMS" ]; then
    echo "no such items file: $ITEMS" >&2
    exit 2
fi

ARTIFACTS="${AEGIS_ARTIFACTS:-$ROOT/model-lab/tinybit/m7_final_gate_work/artifacts}"
MODEL="${AEGIS_MODEL:-$ARTIFACTS/MODEL.SAF}"
EMBED="${AEGIS_EMBED:-$ARTIFACTS/EMBED.BIN}"
VOCAB="${AEGIS_VOCAB:-$ARTIFACTS/VOCAB.BIN}"

AGENT_TRACE_BIN="${AGENT_TRACE_BIN:-$ROOT/aegis-linux/target/release/examples/agent_trace}"
if [ -n "$BIN_OVERRIDE" ]; then
    AGENT_TRACE_BIN="$BIN_OVERRIDE"
fi
N="${N:-24}"
RUN_SH="$ROOT/demo/agent-trace/run.sh"

for f in "$MODEL" "$EMBED" "$VOCAB"; do
    if [ ! -f "$f" ]; then
        echo "missing artifact: $f" >&2
        exit 1
    fi
done
if [ ! -x "$AGENT_TRACE_BIN" ]; then
    echo "agent_trace binary not found or not executable: $AGENT_TRACE_BIN" >&2
    exit 1
fi
if [ ! -f "$TABLE_FILE" ]; then
    echo "missing table file: $TABLE_FILE" >&2
    exit 1
fi
if [ "$ATTEST" -eq 1 ] && [ ! -x "$RUN_SH" ]; then
    echo "--attest given but run.sh not found or not executable: $RUN_SH" >&2
    exit 1
fi

SUMMARY="$OUTDIR/summary.tsv"
TIMING="$OUTDIR/timing.tsv"
RUNTXT="$OUTDIR/RUN.txt"
ATTESTLOG="$OUTDIR/attest.tsv"

SUMMARY_HEADER_NOATTEST='item_id	bucket	tool_expected	tool_observed	arg_match	output_match	trace_chain	receipt_path	verify_result	box'
SUMMARY_HEADER_ATTEST="${SUMMARY_HEADER_NOATTEST}	attest_rc"
if [ "$ATTEST" -eq 1 ]; then
    EXPECTED_SUMMARY_HEADER="$SUMMARY_HEADER_ATTEST"
else
    EXPECTED_SUMMARY_HEADER="$SUMMARY_HEADER_NOATTEST"
fi

# Resuming into an outdir whose existing summary.tsv was produced with a
# different --attest setting would append rows with a different column
# count than its own header (a ragged TSV silently corrupting the file).
# Check before touching anything in $OUTDIR (mkdir included).
if [ -f "$SUMMARY" ]; then
    existing_header="$(head -n1 "$SUMMARY")"
    if [ "$existing_header" != "$EXPECTED_SUMMARY_HEADER" ]; then
        echo "run_suite.sh: refusing to resume into $OUTDIR: existing summary.tsv header does not match this invocation's --attest setting." >&2
        echo "  existing header  ($SUMMARY): $existing_header" >&2
        echo "  this invocation's header : $EXPECTED_SUMMARY_HEADER" >&2
        exit 2
    fi
fi

mkdir -p "$OUTDIR/receipts" "$OUTDIR/prompts"
if [ "$ATTEST" -eq 1 ]; then
    mkdir -p "$OUTDIR/attest"
fi

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

# Bound into every receipt's genesis via agent_trace's --suite-sha256 (see
# eval/README.md) so a receipt generated under a different suite file can
# never verify against this one, even if every other input matches.
SUITE_SHA256="$(sha256_file "$ITEMS")"

if [ ! -f "$SUMMARY" ]; then
    printf '%s\n' "$EXPECTED_SUMMARY_HEADER" > "$SUMMARY"
fi
if [ ! -f "$TIMING" ]; then
    printf 'item_id\tgen_s\tverify_s\n' > "$TIMING"
fi
if [ "$ATTEST" -eq 1 ] && [ ! -f "$ATTESTLOG" ]; then
    printf 'item_id\tattest_rc\n' > "$ATTESTLOG"
fi
if [ ! -f "$RUNTXT" ]; then
    {
        echo "suite-file $ITEMS"
        echo "suite-sha256 $SUITE_SHA256"
        echo "table-file $TABLE_FILE"
        echo "table-sha256 $(sha256_file "$TABLE_FILE")"
        echo "model-sha256 $(sha256_file "$MODEL")"
        echo "embed-sha256 $(sha256_file "$EMBED")"
        echo "vocab-sha256 $(sha256_file "$VOCAB")"
        echo "binary-sha256 $(sha256_file "$AGENT_TRACE_BIN")"
        echo "N $N"
        echo "template $TEMPLATE"
        echo "host $(hostname)"
        echo "start-utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$RUNTXT"
fi

# unescape gen_suite.py's tsv_escape() encoding (backslash-n -> newline,
# backslash-backslash -> backslash) for one field, read on stdin.
unescape_field() {
    python3 -c '
import sys
s = sys.stdin.read()
out = []
i = 0
while i < len(s):
    if s[i] == "\\" and i + 1 < len(s):
        if s[i+1] == "n":
            out.append("\n"); i += 2; continue
        if s[i+1] == "\\":
            out.append("\\"); i += 2; continue
    out.append(s[i]); i += 1
sys.stdout.write("".join(out))
'
}

apply_template() {
    # $1 = prompt text (already unescaped, real newlines). T3: replace the
    # final trailing "A:" with "A: CALC(" (unclosed).
    local prompt="$1"
    if [ "$TEMPLATE" = "T3" ]; then
        if [[ "$prompt" == *$'\n'A: ]]; then
            prompt="${prompt%A:}A: CALC("
        fi
    fi
    printf '%s' "$prompt"
}

item_already_done() {
    local id="$1"
    awk -F'\t' -v id="$id" 'NR>1 && $1==id {found=1} END{exit !found}' "$SUMMARY"
}

# Parse one "step N: toks=.. tool=.. in=.. out=.. decode-chain=.." line into
# tool/in/out (in/out are hex; decode to ascii here).
parse_step_line() {
    local line="$1"
    local tool="" in_hex="" out_hex=""
    for field in $line; do
        case "$field" in
            tool=*) tool="${field#tool=}" ;;
            in=*) in_hex="${field#in=}" ;;
            out=*) out_hex="${field#out=}" ;;
        esac
    done
    python3 -c '
import sys
tool, in_hex, out_hex = sys.argv[1], sys.argv[2], sys.argv[3]
def unhex(h):
    if not h:
        return ""
    try:
        return bytes.fromhex(h).decode("utf-8", "replace")
    except ValueError:
        return "<bad-hex>"
print(tool)
print(unhex(in_hex))
print(unhex(out_hex))
' "$tool" "$in_hex" "$out_hex"
}

tsv_field_escape() {
    # collapse tabs/newlines in a value before writing it into our own
    # tab-separated summary/timing files.
    printf '%s' "$1" | tr '\t\n' '  '
}

count=0
host="$(hostname)"

# read item rows (skip header) with IFS=tab, 8 columns
tail -n +2 "$ITEMS" | while IFS=$'\t' read -r item_id bucket tmpl_id prompt_text_esc exp_tool exp_input exp_output notes; do
    : "$tmpl_id" "$notes"  # unused fields, read for TSV column alignment
    if [ "$LIMIT" -gt 0 ] && [ "$count" -ge "$LIMIT" ]; then
        break
    fi
    if item_already_done "$item_id"; then
        continue
    fi
    count=$((count + 1))

    prompt_text="$(printf '%s' "$prompt_text_esc" | unescape_field)"
    prompt_text="$(apply_template "$prompt_text")"

    k=1
    case "$exp_tool" in
        *,*) k=2 ;;
    esac

    table_args=()
    case "$exp_tool" in
        LOOKUP|*LOOKUP*) table_args=(--table "$TABLE_FILE") ;;
    esac

    prompt_file="$OUTDIR/prompts/${item_id}.txt"
    printf '%s' "$prompt_text" > "$prompt_file"

    receipt="$OUTDIR/receipts/${item_id}.txt"

    gen_ok=1
    gen_s_start=$(date +%s.%N)
    if ! "$AGENT_TRACE_BIN" gen "$MODEL" "$EMBED" "$VOCAB" "$k" "$N" "$prompt_text" "${table_args[@]}" --suite-sha256 "$SUITE_SHA256" > "$receipt" 2> "$OUTDIR/receipts/${item_id}.gen.err"; then
        gen_ok=0
    fi
    gen_s_end=$(date +%s.%N)
    gen_s=$(awk -v a="$gen_s_start" -v b="$gen_s_end" 'BEGIN{printf "%.3f", b-a}')

    verify_result="FAIL"
    verify_s="0.000"
    tool_observed="none"
    trace_chain=""
    if [ "$gen_ok" -eq 1 ]; then
        verify_s_start=$(date +%s.%N)
        verify_out="$("$AGENT_TRACE_BIN" verify "$MODEL" "$EMBED" "$VOCAB" "$receipt" "${table_args[@]}" --suite-sha256 "$SUITE_SHA256" 2> "$OUTDIR/receipts/${item_id}.verify.err")" || true
        verify_s_end=$(date +%s.%N)
        verify_s=$(awk -v a="$verify_s_start" -v b="$verify_s_end" 'BEGIN{printf "%.3f", b-a}')
        if printf '%s' "$verify_out" | grep -q "VERIFY PASS"; then
            verify_result="PASS"
        else
            verify_result="FAIL"
        fi

        trace_chain="$(grep -m1 '^trace-chain ' "$receipt" | awk '{print $2}' || true)"

        # Collect step tool/in/out for step 0 (and step 1 for mixed items).
        step0_line="$(grep -m1 '^step 0:' "$receipt" || true)"
        step1_line="$(grep -m1 '^step 1:' "$receipt" || true)"
        tool0="none"; in0=""; out0=""
        if [ -n "$step0_line" ]; then
            mapfile -t s0 < <(parse_step_line "$step0_line")
            tool0="${s0[0]:-none}"; in0="${s0[1]:-}"; out0="${s0[2]:-}"
        fi
        if [ "$k" -eq 2 ] && [ -n "$step1_line" ]; then
            mapfile -t s1 < <(parse_step_line "$step1_line")
            tool1="${s1[0]:-none}"; in1="${s1[1]:-}"; out1="${s1[2]:-}"
            tool_observed="${tool0},${tool1}"
            observed_input="${in0},${in1}"
            observed_output="${out0},${out1}"
        else
            tool_observed="$tool0"
            observed_input="$in0"
            observed_output="$out0"
        fi
    else
        observed_input=""
        observed_output=""
    fi

    # Map receipt tool names (calc/calc-error/lookup/no-tool, comma-joined
    # for k=2) to the suite's expected_tool vocabulary (CALC/LOOKUP/NONE)
    # for comparison, case-insensitively.
    norm_tool() {
        python3 -c '
import sys
s = sys.argv[1]
def one(t):
    t = t.strip().lower()
    if t in ("calc", "calc-error"):
        return "CALC"
    if t == "lookup":
        return "LOOKUP"
    return "NONE"
print(",".join(one(t) for t in s.split(",")))
' "$1"
    }

    tool_observed_norm="$(norm_tool "$tool_observed")"

    arg_match="false"
    output_match="false"
    if [ "$gen_ok" -eq 1 ]; then
        if [ "$tool_observed_norm" = "$exp_tool" ] || { [ "$exp_tool" = "NONE" ] && [ "$tool_observed_norm" = "NONE" ]; }; then
            if [ "$observed_input" = "$exp_input" ]; then
                arg_match="true"
            fi
            if [ "$observed_output" = "$exp_output" ]; then
                output_match="true"
            fi
        elif [ "$exp_tool" = "NONE" ] && [ "$exp_input" = "" ] && [ "$exp_output" = "" ]; then
            arg_match="false"
            output_match="false"
        fi
    fi

    # --attest: independent of verify_result — a TPM quote over this
    # item's own receipt digest (see demo/edge-receipt/attest.sh and
    # demo/agent-trace/run.sh's cmd_attest), stored under
    # <outdir>/attest/<item_id>/. Only attempted when gen produced a
    # receipt at all (gen_ok=1); an attest failure never changes
    # verify_result above. attest_rc: "skip" (no receipt to attest),
    # "0" (attest.sh quote succeeded), or the nonzero exit code.
    attest_rc=""
    if [ "$ATTEST" -eq 1 ]; then
        if [ "$gen_ok" -eq 1 ]; then
            attest_item_dir="$OUTDIR/attest/${item_id}"
            mkdir -p "$attest_item_dir"
            attest_status=0
            "$RUN_SH" attest "$receipt" "$attest_item_dir" \
                > "$OUTDIR/receipts/${item_id}.attest.out" \
                2> "$OUTDIR/receipts/${item_id}.attest.err" || attest_status=$?
            attest_rc="$attest_status"
        else
            attest_rc="skip"
        fi
    fi

    {
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
            "$(tsv_field_escape "$item_id")" \
            "$(tsv_field_escape "$bucket")" \
            "$(tsv_field_escape "$exp_tool")" \
            "$(tsv_field_escape "$tool_observed_norm")" \
            "$arg_match" \
            "$output_match" \
            "$(tsv_field_escape "$trace_chain")" \
            "$(tsv_field_escape "$receipt")" \
            "$verify_result" \
            "$host"
        if [ "$ATTEST" -eq 1 ]; then
            printf '\t%s' "$(tsv_field_escape "$attest_rc")"
        fi
        printf '\n'
    } >> "$SUMMARY"

    printf '%s\t%s\t%s\n' "$item_id" "$gen_s" "$verify_s" >> "$TIMING"

    if [ "$ATTEST" -eq 1 ]; then
        printf '%s\t%s\n' "$(tsv_field_escape "$item_id")" "$(tsv_field_escape "$attest_rc")" >> "$ATTESTLOG"
    fi

    echo "item $item_id: gen_ok=$gen_ok verify=$verify_result tool_observed=$tool_observed_norm attest_rc=${attest_rc:-n/a}" >&2
done

if [ "$ATTEST" -eq 1 ] && [ -f "$ATTESTLOG" ]; then
    attest_count="$(tail -n +2 "$ATTESTLOG" | wc -l | tr -d ' ')"
    attest_fail="$(tail -n +2 "$ATTESTLOG" | awk -F'\t' '$2!="0"{c++} END{print c+0}')"
    # Idempotent: strip any attest-* lines from a prior invocation of this
    # script against the same outdir (a resumed run with --attest may run
    # this block more than once), then append fresh totals covering every
    # item recorded in $ATTESTLOG so far (not just this invocation's items).
    grep -v '^attest-\(requested\|count\|fail\) ' "$RUNTXT" > "$RUNTXT.tmp" || true
    mv "$RUNTXT.tmp" "$RUNTXT"
    {
        echo "attest-requested yes"
        echo "attest-count $attest_count"
        echo "attest-fail $attest_fail"
    } >> "$RUNTXT"
fi

echo "done: wrote/updated $SUMMARY" >&2
