#!/usr/bin/env bash
# build-public-snapshot.sh — construct the scrubbed public snapshot branch.
#
# The public repo (github.com/Aefinity-AI/alice-aegis) is NOT this working
# history. It is a small orphan snapshot: no shared ancestry, no working-branch
# commits, none of the internal operational documents. Until 2026-08-05 that
# snapshot was assembled by hand, which is how a pre-publish runbook and a
# successor-session handoff ended up published.
#
# This script rebuilds it mechanically and refuses to produce output that fails
# the gates below. It does NOT push. It writes a local branch for review.
#
#   ./scripts/build-public-snapshot.sh          # build + gate + report
#   git show --stat public-snapshot             # review before pushing
#
# The working tree, HEAD, and the normal index are never modified: the snapshot
# is assembled through a temporary index file.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH="public-snapshot"

# ── Files that must never enter the public tree ───────────────────────────────
# Each entry is a git pathspec. Add to this list rather than relying on memory.
EXCLUDE=(
  # Internal operational documents — addressed to the author or a successor
  # session, not to a reader of the repository.
  "docs/PUSH_TO_PRIVATE_REPO.md"      # pre-publish runbook; narrates the credential incident
  "program/HANDOFF_2026-07-18.md"     # names the private Kaggle secret store and its read path
  "PRE_REVIEW_SCRUB_LIST.md"          # enumerated credential locations

  # Contains the author's home address and phone inside an ASCII85+Flate stream,
  # where no text-based scrub will find them. Regenerate from the redacted
  # markdown if a public copy is ever wanted.
  "docs/Aefinity_AI_ALICE_response_DARPA-SN-26-97.pdf"

  # Superseded full-tree copies.
  "*_ALICE_1_0_BACKUP/*"
)

echo "==> selecting files"
mapfile -t ALL < <(git ls-files)
KEEP=()
DROPPED=()
for f in "${ALL[@]}"; do
  skip=""
  for pat in "${EXCLUDE[@]}"; do
    # shellcheck disable=SC2053
    if [[ "$f" == $pat ]]; then skip=1; break; fi
  done
  if [ -n "$skip" ]; then DROPPED+=("$f"); else KEEP+=("$f"); fi
done
echo "    ${#KEEP[@]} files kept, ${#DROPPED[@]} excluded"
for d in "${DROPPED[@]}"; do echo "      - $d"; done

fail=0
note() { echo "    ✗ $1"; fail=1; }

# ── Gate 1: no secret-shaped material ────────────────────────────────────────
echo "==> gate: secret-shaped material"
for f in "${KEEP[@]}"; do
  [ -f "$f" ] || continue
  [ "$(stat -c%s "$f")" -gt 2000000 ] && continue
  if grep -qE 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|-----BEGIN [A-Z ]*PRIVATE KEY-----' "$f" 2>/dev/null; then
    note "secret-shaped material in $f"
  fi
done
[ $fail -eq 0 ] && echo "    ✓ clean"

# ── Gate 2: no personal contact details ──────────────────────────────────────
# The DARPA response legitimately carries these; the public repo must not.
echo "==> gate: personal contact details (text)"
# Scoped to the author's own details on purpose. A generic phone/address
# pattern also matches Microsoft's public corporate contact in their EU AI Act
# card (data_summary_card.md), which is third-party text we must retain verbatim
# under MIT — stripping it would be a licence violation, not a privacy win.
pii_re='[0-9]{3,5} County Road [0-9]+|\(?409\)? ?[-. ]?656[-. ]2416'
for f in "${KEEP[@]}"; do
  [ -f "$f" ] || continue
  case "$f" in *.md|*.txt|*.rs|*.py|*.sh|*.toml|*.json|*.yml|*.yaml)
    grep -qE "$pii_re" "$f" 2>/dev/null && note "contact details in $f" ;;
  esac
done

# ── Gate 3: PDFs, decoded ────────────────────────────────────────────────────
# A PDF's text lives inside filtered streams. grep over the raw bytes finds
# nothing even when the address is on page 1 — this decodes before scanning.
echo "==> gate: personal contact details (PDF streams, decoded)"
for f in "${KEEP[@]}"; do
  case "$f" in *.pdf) ;; *) continue ;; esac
  [ -f "$f" ] || continue
  if python3 - "$f" <<'PY'
import sys, re, zlib, base64
d = open(sys.argv[1], 'rb').read()
out = []
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', d, re.S):
    t = m.group(1).strip()
    for how in ('a85+flate', 'a85', 'flate', 'raw'):
        try:
            if how.startswith('a85'):
                b = t[2:] if t.startswith(b'<~') else t
                b = b[:-2] if b.endswith(b'~>') else b
                x = base64.a85decode(b, adobe=False)
                if how == 'a85+flate':
                    x = zlib.decompress(x)
            elif how == 'flate':
                x = zlib.decompress(t)
            else:
                x = t
            out.append(x); break
        except Exception:
            continue
blob = b'\n'.join(out).decode('latin-1')
txt = ' '.join(re.sub(r'\\(.)', r'\1', s[1:-1])
               for s in re.findall(r'\((?:[^()\\]|\\.)*\)', blob))
bad = re.search(r'[Cc]ounty\s*[Rr]oad|\(?\d{3}\)?\s*[-. ]\s*\d{3}\s*[-. ]\s*\d{4}', txt)
sys.exit(0 if bad else 1)
PY
  then note "contact details inside PDF streams: $f"; fi
done
[ $fail -eq 0 ] && echo "    ✓ clean"

# ── Gate 4: licensing files present ──────────────────────────────────────────
# NOTE: do not test membership with `printf ... | grep -qx`. Under `set -o
# pipefail`, grep -q exits the moment it matches, printf then dies of SIGPIPE
# (141), and pipefail reports the whole pipeline as failed — so a file that IS
# present reports as missing, racily depending on buffer timing. Pure bash here.
echo "==> gate: licensing"
in_keep() {
  local needle="$1" f
  for f in "${KEEP[@]}"; do [ "$f" = "$needle" ] && return 0; done
  return 1
}
for req in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do
  in_keep "$req" || note "missing $req"
done
if in_keep LICENSE && grep -q "Microsoft Corporation" LICENSE 2>/dev/null; then
  note "root LICENSE still names Microsoft as copyright holder"
fi
[ $fail -eq 0 ] && echo "    ✓ ok"

if [ $fail -ne 0 ]; then
  echo ""
  echo "  ✗ gates failed — no snapshot written."
  exit 1
fi

# ── Assemble the orphan commit ───────────────────────────────────────────────
# A temporary index keeps the real index and working tree untouched.
echo "==> writing $BRANCH"
TMPIDX="$(mktemp)"; trap 'rm -f "$TMPIDX"' EXIT
export GIT_INDEX_FILE="$TMPIDX"
git read-tree --empty
printf '%s\0' "${KEEP[@]}" | git update-index --add -z --stdin
TREE="$(git write-tree)"
MSG="A.L.I.C.E. / Aegis — public snapshot $(date -u +%Y-%m-%d)

Scrubbed snapshot built by scripts/build-public-snapshot.sh.
Excludes internal operational documents; carries no working history.
Licensed Apache-2.0 (see LICENSE, NOTICE, THIRD_PARTY_NOTICES.md)."
COMMIT="$(git commit-tree "$TREE" -m "$MSG")"
unset GIT_INDEX_FILE
git branch -f "$BRANCH" "$COMMIT"

echo ""
echo "  ✓ $BRANCH -> $(git rev-parse --short "$COMMIT")  (${#KEEP[@]} files, orphan commit)"
echo ""
echo "  Review:  git show --stat $BRANCH"
echo "  Publish: ALICE_ALLOW_PUBLIC_PUSH=1 git push --force origin $BRANCH:main"
echo "           (the pre-push guard blocks this without that variable, by design)"
