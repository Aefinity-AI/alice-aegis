#!/bin/bash
# E23 round 4 — AGENT-EPISODE RECEIPT tamper matrix, rebuilt so that the
# evidence is replay-derived rather than parser-derived.
#
# Round 3 (box1, BitNet-2B) caught 360/360 mutants with 0 still-verifying,
# but only about half were rejected by a digest or replay comparison; the
# rest were parse-rejects, which prove the parser is strict, not that the
# chain binds. Round 4 changes three things:
#
#  1. The mutator (mutate4.py) partitions its output. Bucket W is
#     well-formed BY CONSTRUCTION: every substituted value satisfies its
#     field's grammar (usually because some genuine receipt in the corpus
#     carried exactly that value), and every structural edit renumbers the
#     step labels and corrects K, so the mutant is an internally consistent
#     receipt describing a DIFFERENT episode. Nothing but a digest or a
#     replay comparison can reject it. Bucket M is malformed on purpose and
#     keeps round 3's parser-strictness coverage, reported separately.
#
#  2. Membership in W is certified by wellformed.py, an INDEPENDENT grammar
#     checker written from the receipt format description rather than from
#     the Rust. Any mutant whose assigned bucket disagrees with that checker
#     is a defect and fails acceptance; it is never silently reassigned.
#
#  3. The rejection-reason classifier reads the verifier's actual message
#     strings and has NO catch-all. Round 3's classifier ended in
#     `*) why=parse-reject`, which silently bucketed 14 `FAIL artifact:
#     <NAME> hash mismatch` rejections -- genuine digest comparisons -- as
#     parse-rejects and understated its own result. Here an unrecognised
#     message is `unclassified` and unclassified>0 fails acceptance.
#
# Round 3 also reported a "cryptographic share" percentage. That ratio is
# not a meaningful quantity: both buckets are entirely under the mutator
# author's control, so the share can be moved to any value by adding or
# removing malformed mutants. Round 4 therefore states an absolute count
# instead.
#
# ACCEPTANCE CRITERIA (fixed BEFORE the run):
#   A1  bucket-vs-checker disagreements                = 0
#   A2  well-formed mutants that still verify          = 0
#   A3  malformed mutants that still verify            = 0
#   A4  well-formed mutants rejected by the PARSER     = 0
#   A5  rejections the classifier could not name       = 0
#   A6  broken positive controls                       = 0
#   A7  COVERAGE GAP lines (an episode that fired no tool) = 0
#   A8  well-formed mutants exercised                  >= 400
#   A9  mutants that duplicate a genuine receipt       = 0
#
# NO TIMING NUMBERS in RESULT.txt (Rule A). Nothing is committed.
set -uo pipefail

NAME=e23r4-wellformed-tamper-matrix-2b
OUT="${LEG_OUT:-$HOME/legs/$NAME}"
RESULT="$OUT/RESULT.txt"
RAW="$OUT/raw"; rm -rf "$OUT"; mkdir -p "$OUT" "$RAW"
log() { echo "$*" | tee -a "$RESULT"; }
refuse() { log "REFUSE: $*"; log "=== $NAME refused $(date -u +%FT%TZ)"; exit 0; }

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
log "=== $NAME start $(date -u +%FT%TZ) host=$(hostname) arch=$(uname -m)"
log "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
log "simd: $(grep -o -m1 -E 'avx2|sse4_2' /proc/cpuinfo | sort -u | tr '\n' ' ')"

REPO="$HOME/projects/alice-aegis"
[ -d "$REPO" ] || refuse "no alice-aegis at $REPO"
command -v cargo >/dev/null || refuse "cargo not on PATH"
cd "$REPO" || refuse "cannot cd $REPO"
git stash -u >/dev/null 2>&1
REF="${E23_REF:-cm/trace-format3-provenance-and-strict-parse}"
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*' 2>>"$RAW/git.log" || log "warn: git fetch failed, using local tree"
git checkout -q -B e23r4-under-test "origin/$REF" 2>>"$RAW/git.log" || refuse "cannot check out origin/$REF"
log "verifier under test: origin/$REF"
log "alice-aegis @ $(git rev-parse HEAD)"

demo/agent-trace/run.sh build >"$RAW/build.log" 2>&1 || refuse "build failed (raw/build.log)"
BIN="$REPO/aegis-linux/target/release/examples/agent_trace"
[ -x "$BIN" ] || refuse "no agent_trace binary at $BIN"

A="${E23_ART:-$HOME/aefinity-artifacts/bitnet2b-2b-artifacts}"
MODEL="$A/aegis_pruned_model.cis.safetensors"; EMBED="$A/embed.bin"; VOCAB="$A/vocab.bin"
for f in "$MODEL" "$EMBED" "$VOCAB"; do [ -f "$f" ] || refuse "missing artifact $f"; done
log "model under test: $(basename "$MODEL") ($(stat -c%s "$MODEL") bytes)"
log "  sha256 model=$(sha256sum "$MODEL" | cut -c1-16)... embed=$(sha256sum "$EMBED" | cut -c1-16)... vocab=$(sha256sum "$VOCAB" | cut -c1-16)..."

KIT="${E23_KIT:-$HOME/legs/e23r4-kit}"
for f in mutate4.py wellformed.py; do [ -f "$KIT/$f" ] || refuse "missing $KIT/$f"; done
cp "$KIT/mutate4.py" "$KIT/wellformed.py" "$OUT/"
log "mutator:  mutate4.py  sha256 $(sha256sum "$OUT/mutate4.py" | cut -c1-16)..."
log "grammar:  wellformed.py sha256 $(sha256sum "$OUT/wellformed.py" | cut -c1-16)..."
log ""

EP="$OUT/episodes"; mkdir -p "$EP"
declare -A EPTBL=()
VOUT=""
verify_ok() {
  local args=("$MODEL" "$EMBED" "$VOCAB" "$1")
  [ -n "${2:-}" ] && args+=(--table "$2")
  VOUT=$("$BIN" verify "${args[@]}" 2>&1)
}

# The episode spec below is recovered from round 3's own baseline receipts
# (prompt-hex decoded, table-sha256 matched against the tables in the tree),
# not from the round-3 driver script kept in scratch -- that script was an
# earlier draft using "part P-402" prompts and demo.tsv/chain.tsv, and the
# run that produced the accepted round-3 result used numeric part numbers
# against parts-numeric.tsv / parts-numeric-chain.tsv instead. Verified:
#   round-3 lookup receipt table-sha256 c9b9c92d... == parts-numeric.tsv
#   round-3 chain  receipt table-sha256 fc539f59... == parts-numeric-chain.tsv
# Using the draft's prompts would have produced no tool calls and a coverage
# gap, which is exactly the failure round 3 existed to close.
TBL="$REPO/demo/agent-trace/tables/parts-numeric.tsv"
CHAINTBL="$REPO/demo/agent-trace/tables/parts-numeric-chain.tsv"
declare -a EPS=(); GAPS=0
while IFS='|' read -r tag prompt k n table; do
  [ -n "$tag" ] || continue
  prompt=$(printf '%b' "$prompt")
  if [ -n "$table" ] && [ ! -f "$table" ]; then log "episode $tag: table $table missing - skipped"; continue; fi
  args=("$MODEL" "$EMBED" "$VOCAB" "$k" "$n" "$prompt")
  [ -n "$table" ] && args+=(--table "$table")
  if ! "$BIN" gen "${args[@]}" >"$EP/$tag.txt" 2>"$RAW/gen-$tag.log" || [ ! -s "$EP/$tag.txt" ]; then
    log "episode $tag: generation FAILED (raw/gen-$tag.log)"; continue; fi
  if verify_ok "$EP/$tag.txt" "$table"; then
    tools=$(grep -o 'tool=[a-z-]*' "$EP/$tag.txt" | sort | uniq -c | tr '\n' ' ')
    steps=$(grep -c '^step ' "$EP/$tag.txt")
    log "episode $tag: steps=$steps baseline verify PASS  tools: $tools"
    if [ "$tag" != notool ] && ! grep -q 'tool=\(calc\|lookup\)' "$EP/$tag.txt"; then
      log "  COVERAGE GAP: episode $tag fired no tool - in=/out= are empty, so this"
      log "  episode does NOT exercise the tool-input/tool-output half of the claim"
      GAPS=$((GAPS+1))
      refuse "coverage gap on episode $tag - a gapped run cannot be ACCEPTED, stopping now"
    fi
    EPTBL[$tag]="$table"; EPS+=("$tag")
  else
    log "episode $tag: baseline receipt does NOT verify - excluded"
  fi
done <<EPSPEC
notool|The quick brown fox|3|16|
calc|Q: What is 7 + 5?\nA: CALC(7 + 5)\nQ: What is 12 + 30?\nA: CALC(|3|16|
lookup|Q: What is part 401?\nA: LOOKUP(401)\nQ: What is part 402?\nA: LOOKUP(|3|16|$TBL
chain|Q: What is part 406?\nA: LOOKUP(406)\nTOOL[lookup]=Superseded, see part 403\nA: LOOKUP(403)\nTOOL[lookup]=Gasket, O-ring, fuel line\nQ: What is part 401?\nA: LOOKUP(|3|16|$CHAINTBL
EPSPEC
[ ${#EPS[@]} -gt 0 ] || refuse "no episode produced a verifying baseline receipt"

# The grammar checker must accept every genuine receipt, or it is not a
# grammar checker and bucket W means nothing.
CHK=$(cd "$OUT" && python3 wellformed.py "$EP"/*.txt)
echo "$CHK" >"$RAW/baseline-grammar.txt"
BADBASE=$(echo "$CHK" | grep -vc 'WELL-FORMED$')
log ""
log "grammar-checker control: $(echo "$CHK" | grep -c 'WELL-FORMED$')/$(echo "$CHK" | wc -l) genuine receipts accepted (need all)"
[ "$BADBASE" -eq 0 ] || refuse "wellformed.py rejects a genuine receipt (raw/baseline-grammar.txt)"
log ""

# ---- worker: one mutant. No catch-all: an unrecognised message is named ----
cat > "$OUT/one.sh" <<'WRK'
#!/bin/bash
IFS=$'\t' read -r name cls bucket path <<< "$1"
args=("$MODEL" "$EMBED" "$VOCAB" "$path"); [ -n "${T:-}" ] && args+=(--table "$T")
VOUT=$("$BIN" verify "${args[@]}" 2>&1); rc=$?
if [ $rc -eq 0 ]; then why=STILL-VERIFIES; kind=hole
elif [[ "$VOUT" == *"FAIL structure:"* ]];            then why=structure-reject;      kind=parser
elif [[ "$VOUT" == *"hash mismatch"* ]];              then why=artifact-hash-mismatch; kind=digest
elif [[ "$VOUT" == *" CTX MISMATCH"* ]];              then why=ctx-mismatch;          kind=replay
elif [[ "$VOUT" == *" QUERY MISMATCH"* ]];            then why=query-mismatch;        kind=replay
elif [[ "$VOUT" == *"WARNING lines claim steps"* ]];  then why=warning-set-mismatch;  kind=replay
elif [[ "$VOUT" == *"replay diverged"* ]];            then why=replay-divergence;     kind=replay
elif [[ "$VOUT" == *"suite-sha256 mismatch"* ]];      then why=suite-mismatch;        kind=digest
elif [[ "$VOUT" == *"but no --table was given"* ]];   then why=table-missing;         kind=parser
else why=unclassified; kind=unclassified
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$cls" "$bucket" "$rc" "$why" "$kind"
WRK
chmod +x "$OUT/one.sh"
export MODEL EMBED VOCAB BIN

JOBS="${E23_JOBS:-3}"
TW=0; TM=0; HOLEW=0; HOLEM=0; PARSEW=0; UNCL=0; PCBAD=0; DISAGREE=0; DUPGEN=0
: > "$OUT/HOLES.tsv"; : > "$OUT/DISAGREE.tsv"; : > "$OUT/ALL.tsv"
for tag in "${EPS[@]}"; do
  M="$OUT/mut-$tag"; rm -rf "$M"
  counts=$(cd "$OUT" && python3 mutate4.py "$EP/$tag.txt" "$M" "$EP"/*.txt 2>>"$RAW/mutate-$tag.log")
  [ -n "$counts" ] || { log "episode $tag: mutator failed (raw/mutate-$tag.log)"; continue; }
  nw=${counts% *}; nm=${counts#* }
  T="${EPTBL[$tag]}"; export T

  # A1: every mutant's assigned bucket must agree with the independent checker.
  d=$(cd "$OUT" && python3 - "$M/INDEX.tsv" 2>>"$OUT/DISAGREE.tsv" <<'PY'
import sys
sys.path.insert(0, '.')
from wellformed import check
bad = 0
for row in open(sys.argv[1]):
    name, cls, bucket, path = row.rstrip('\n').split('\t')
    with open(path, encoding='utf-8', errors='surrogateescape') as fh:
        r = check(fh.read())
    if (r is None) != (bucket == 'W'):
        print(f'{name}\t{cls}\t{bucket}\t{r}', file=sys.stderr); bad += 1
print(bad)
PY
)
  DISAGREE=$((DISAGREE + ${d:-999}))

  # A9: no mutant may be byte-identical to a genuine receipt.
  dup=0
  for g in "$EP"/*.txt; do
    gs=$(sha256sum "$g" | cut -d' ' -f1)
    while IFS=$'\t' read -r _ _ _ p; do
      [ "$(sha256sum "$p" | cut -d' ' -f1)" = "$gs" ] && dup=$((dup+1))
    done < "$M/INDEX.tsv"
  done
  DUPGEN=$((DUPGEN + dup))

  cp "$EP/$tag.txt" "$M/PC.identity.txt"
  if verify_ok "$M/PC.identity.txt" "$T"; then pc=PASS; else pc="FAIL <<< POSITIVE CONTROL BROKEN"; PCBAD=$((PCBAD+1)); fi

  xargs -d '\n' -P "$JOBS" -I{} "$OUT/one.sh" {} < "$M/INDEX.tsv" > "$M/RESULTS.tsv"
  awk -F'\t' -v t="$tag" '{print t"\t"$0}' "$M/RESULTS.tsv" >> "$OUT/ALL.tsv"
  hw=$(awk -F'\t' '$3=="W" && $4==0' "$M/RESULTS.tsv" | wc -l)
  hm=$(awk -F'\t' '$3=="M" && $4==0' "$M/RESULTS.tsv" | wc -l)
  pw=$(awk -F'\t' '$3=="W" && $6=="parser"' "$M/RESULTS.tsv" | wc -l)
  uc=$(awk -F'\t' '$6=="unclassified"' "$M/RESULTS.tsv" | wc -l)
  awk -F'\t' -v t="$tag" '$4==0 {print t"\t"$1"\t"$2"\t"$3}' "$M/RESULTS.tsv" >> "$OUT/HOLES.tsv"
  log "episode $tag: well-formed=$nw malformed=$nm  bucket-disagreements=${d:-?}  dup-of-genuine=$dup  positive-control=$pc"
  log "  well-formed still-verifying=$hw   well-formed rejected by PARSER=$pw   unclassified=$uc"
  log "  well-formed, by rejection kind:"
  awk -F'\t' '$3=="W" && $4!=0 {k[$6"/"$5]++} END {for (x in k) printf "    %-38s n=%s\n", x, k[x]}' "$M/RESULTS.tsv" | sort | tee -a "$RESULT"
  log "  malformed, by rejection kind:"
  awk -F'\t' '$3=="M" && $4!=0 {k[$6"/"$5]++} END {for (x in k) printf "    %-38s n=%s\n", x, k[x]}' "$M/RESULTS.tsv" | sort | tee -a "$RESULT"
  TW=$((TW+nw)); TM=$((TM+nm)); HOLEW=$((HOLEW+hw)); HOLEM=$((HOLEM+hm))
  PARSEW=$((PARSEW+pw)); UNCL=$((UNCL+uc))
done

log ""
log "--- WELL-FORMED CORPUS, every episode, by mutation class ---"
awk -F'\t' '$4=="W" {n[$3]++; if($5==0) h[$3]++; if($7=="parser") p[$3]++}
  END {for (c in n) printf "  %-34s n=%-4s still-verifies=%-3s parser-rejected=%s\n", c, n[c], (h[c]?h[c]:0), (p[c]?p[c]:0)}' \
  "$OUT/ALL.tsv" | sort | tee -a "$RESULT"
log ""
log "--- MALFORMED CORPUS, every episode, by mutation class ---"
awk -F'\t' '$4=="M" {n[$3]++; if($5==0) h[$3]++}
  END {for (c in n) printf "  %-34s n=%-4s still-verifies=%s\n", c, n[c], (h[c]?h[c]:0)}' \
  "$OUT/ALL.tsv" | sort | tee -a "$RESULT"
log ""
log "TOTAL well-formed=$TW  malformed=$TM  still-verifying: W=$HOLEW M=$HOLEM"

if [ $((HOLEW+HOLEM)) -gt 0 ]; then
  log ""
  log "MUTANTS THAT STILL VERIFY (each is a hole in the tamper-evidence claim):"
  while IFS=$'\t' read -r tag name cls bucket; do
    log "  [$bucket] $tag  $name  [$cls]"
    diff "$EP/$tag.txt" "$OUT/mut-$tag/$name.txt" | sed 's/^/      /' | head -6 | tee -a "$RESULT" >/dev/null
  done < "$OUT/HOLES.tsv"
fi
if [ "$DISAGREE" -gt 0 ]; then
  log ""
  log "BUCKET DISAGREEMENTS (mutator says one thing, independent grammar says another):"
  sed 's/^/  /' "$OUT/DISAGREE.tsv" | tee -a "$RESULT" >/dev/null
fi
if [ "$UNCL" -gt 0 ]; then
  log ""
  log "UNCLASSIFIED REJECTIONS (the classifier saw a message it cannot name):"
  awk -F'\t' '$7=="unclassified" {print "  "$1"  "$2"  ["$3"]"}' "$OUT/ALL.tsv" | tee -a "$RESULT" >/dev/null
fi

log ""
log "--- ACCEPTANCE (criteria fixed before the run) ---"
log "  A1 bucket-vs-grammar disagreements   = $DISAGREE        (required 0)"
log "  A2 well-formed still verifying       = $HOLEW        (required 0)"
log "  A3 malformed still verifying         = $HOLEM        (required 0)"
log "  A4 well-formed rejected by PARSER    = $PARSEW        (required 0)"
log "  A5 unclassified rejections           = $UNCL        (required 0)"
log "  A6 broken positive controls          = $PCBAD        (required 0)"
log "  A7 COVERAGE GAP lines                = $GAPS        (required 0)"
log "  A8 well-formed mutants exercised     = $TW       (required >= 400)"
log "  A9 mutants duplicating a genuine receipt = $DUPGEN    (required 0)"
if [ "$DISAGREE" -eq 0 ] && [ "$HOLEW" -eq 0 ] && [ "$HOLEM" -eq 0 ] && [ "$PARSEW" -eq 0 ] \
   && [ "$UNCL" -eq 0 ] && [ "$PCBAD" -eq 0 ] && [ "$GAPS" -eq 0 ] && [ "$TW" -ge 400 ] \
   && [ "$DUPGEN" -eq 0 ]; then
  log "  VERDICT: ACCEPTED"
else
  log "  VERDICT: NOT ACCEPTED"
fi
log "=== $NAME done $(date -u +%FT%TZ)"
