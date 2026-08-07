#!/usr/bin/env python3
"""ledger.py — the machine-readable claim ledger. append-only, retraction-aware.

program/RESEARCH_LEDGER.md stays exactly as it is. It is better prose than any
schema will ever hold — row A4 does more honest adversarial work than a template
can. What it CANNOT do is be checked by a script, and that is the only thing
missing: a retraction written in prose does not propagate. This program wrote
"the multicore claim of 8.25 tok/s has no log anywhere" into commit e659a1f and
then shipped 8.25 in five places of a DARPA document, plus in the engine's own
default worker count (aegis-core/src/ops.rs:621, changed by the same unlogged
commit 254ba43 that produced 8.25, and never reverted when 8.25 was retracted).

claims.jsonl is the machine-readable shadow of the .md ledger. One JSON object
per line, append-only. Two rules give it all its power:

  1. A claim may only be `live` if it names a runcard OR a primary log path,
     AND evidence_check finds its value in that path. Anything else is
     `commit-only`, `inferred`, or `unlogged` — statuses that claimlint refuses
     to let into an external document.
  2. Retraction is an APPEND, never an edit. `retract` writes a new record with
     status=retracted; the original stays. This is why the value 8.25 can never
     quietly come back: it is permanently in the file as retracted, and
     claimlint fails any document containing it.

Statuses (they are the claim CEILING, and the ceiling travels with the number):
  measured      value appears in a named primary log/runcard, verified by evidence_check
  derived       arithmetic over `derived_from` claim ids; the derivation is recorded
  commit-only   the only record is a commit message  -> banned from external docs
  inferred      an explanation/attribution, not an instrument reading
  unlogged      asserted, no source found            -> banned from external docs
  superseded    replaced; `superseded_by` names the replacement
  retracted     withdrawn; `reason` mandatory        -> claimlint hard-fails on it

Usage:
  ledger.py add   --id A4.4t --value 2.14 --unit x --kind measured \
                  --statement "decode speedup, 4 threads vs 1" \
                  --scope "i5-10210U crosvm, BitNet-2B, int8_act, parallel" \
                  --source docs/hardware_logs/thread_sweep_2026-07-30.log \
                  --runid 2026-07-30T0412Z-thread_sweep-a249b2c \
                  --ceiling "single run, no CI, one host"
  ledger.py retract --id A4.8t --reason "no log anywhere; commit e659a1f said so" \
                  --superseded-by A4.4t
  ledger.py verify [--strict]        # evidence_check every live claim
  ledger.py list [--status ...] [--grep ...]
  ledger.py values --status retracted,superseded,unlogged,commit-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_check import check_claim  # noqa: E402

REPO = Path(os.environ.get("ALICE_REPO", Path.home())).resolve()
LEDGER = Path(__file__).resolve().parent.parent / "claims.jsonl"

KINDS = {"measured", "derived", "commit-only", "inferred", "unlogged"}
DEAD = {"retracted", "superseded"}
# Statuses/kinds that may never appear in a document intended for anyone outside
# this house. `derived` is allowed only when every parent is allowed.
EXTERNAL_OK = {"measured", "derived"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for i, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARNING: claims.jsonl:{i} unparseable ({e}); skipped", file=sys.stderr)
    return out


def append(rec: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def current() -> dict[str, dict]:
    """id -> latest record. Later lines win; earlier lines are the audit trail."""
    cur: dict[str, dict] = {}
    for r in load():
        cur[r["id"]] = r
    return cur


def cmd_add(a) -> int:
    if a.kind not in KINDS:
        print(f"FATAL: --kind must be one of {sorted(KINDS)}", file=sys.stderr)
        return 2
    if not a.scope.strip():
        print("FATAL: --scope is mandatory and may not be empty.\n"
              "  'measured on the Dell' becoming 'on real hardware' is how this\n"
              "  program published a platform PROCHOT clamp as a property of real\n"
              "  hardware for 20 days. Name the host, the model, and the build.",
              file=sys.stderr)
        return 2
    if a.kind == "measured" and not (a.source or a.runid):
        print("FATAL: kind=measured requires --source (a primary log) or --runid.",
              file=sys.stderr)
        return 2
    rec = {
        "id": a.id, "status": "live", "kind": a.kind,
        "value": str(a.value), "unit": a.unit or "",
        "statement": a.statement, "scope": a.scope.strip(),
        "source": a.source or "", "runid": a.runid or "",
        "ceiling": a.ceiling or "", "derived_from": a.derived_from or [],
        "supersedes": a.supersedes or [], "superseded_by": None,
        "added_utc": _now(), "reason": "",
    }
    if a.kind == "measured" and a.source:
        res = check_claim(rec["value"], rec["source"], str(REPO))
        rec["evidence_check"] = res["status"]
        if res["status"] != "verified" and not a.force:
            print(f"REFUSED: {a.value!r} is not in {a.source!r} ({res['status']}).\n"
                  f"  {res['detail']}\n"
                  f"  Fix the citation, or add it with a truthful --kind "
                  f"(commit-only / inferred / unlogged), or --force and explain.",
                  file=sys.stderr)
            return 1
    for sid in rec["supersedes"]:
        append({"id": sid, "status": "superseded", "superseded_by": a.id,
                "reason": f"superseded by {a.id}", "added_utc": _now(),
                "kind": (current().get(sid) or {}).get("kind", "unlogged"),
                "value": (current().get(sid) or {}).get("value", ""),
                "unit": (current().get(sid) or {}).get("unit", ""),
                "statement": (current().get(sid) or {}).get("statement", ""),
                "scope": (current().get(sid) or {}).get("scope", ""),
                "source": (current().get(sid) or {}).get("source", "")})
    append(rec)
    print(json.dumps(rec, indent=2))
    return 0


def cmd_retract(a) -> int:
    cur = current()
    old = cur.get(a.id)
    if old is None and not a.force:
        print(f"FATAL: no claim {a.id!r} in the ledger. Retracting something that was\n"
              f"  never entered means the number lives only in prose — add it first\n"
              f"  with --kind commit-only/unlogged so the retraction has a subject,\n"
              f"  or pass --force with --value.", file=sys.stderr)
        return 2
    if not a.reason.strip():
        print("FATAL: --reason is mandatory. A retraction without a stated basis is "
              "a deletion, and deletions are how history gets rewritten.", file=sys.stderr)
        return 2
    rec = dict(old or {})
    rec.update({"id": a.id, "status": "retracted", "reason": a.reason.strip(),
                "superseded_by": a.superseded_by or None, "retracted_utc": _now(),
                "added_utc": _now()})
    if a.value:
        rec["value"] = str(a.value)
    if not rec.get("value"):
        print("FATAL: the retracted record has no --value, so claimlint cannot "
              "recognise the number in a document. Give it one.", file=sys.stderr)
        return 2
    append(rec)
    print(f"retracted {a.id} (value {rec['value']}). "
          f"Any document containing that number now fails `ev lint`.")
    return 0


def cmd_verify(a) -> int:
    bad = 0
    for cid, r in sorted(current().items()):
        if r["status"] != "live":
            continue
        if r["kind"] not in ("measured",):
            print(f"note {cid:<14} kind={r['kind']:<12} (no source expected)")
            continue
        res = check_claim(r["value"], r["source"], str(REPO))
        if res["status"] == "verified":
            print(f"ok   {cid:<14} {r['value']:>12} {r['unit']:<8} {r['source']}")
        else:
            bad += 1
            print(f"FAIL {cid:<14} {r['value']:>12} {r['unit']:<8} {res['status']}: {res['detail']}")
    print(f"\n{bad} live measured claim(s) failed evidence_check.")
    return 1 if bad else 0


def cmd_list(a) -> int:
    want = set(a.status.split(",")) if a.status else None
    for cid, r in sorted(current().items()):
        if want and r["status"] not in want:
            continue
        if a.grep and a.grep.lower() not in json.dumps(r).lower():
            continue
        flag = {"live": " ", "retracted": "X", "superseded": "S"}.get(r["status"], "?")
        print(f"{flag} {cid:<14} {r['value']:>12} {r.get('unit',''):<8} "
              f"{r['status']:<11} {r.get('kind',''):<12} {r.get('statement','')[:60]}")
    return 0


def cmd_values(a) -> int:
    """The banned-value list claimlint consumes. Machine output, one per line."""
    want = set(a.status.split(","))
    seen = set()
    for cid, r in current().items():
        hit = r["status"] in want or r.get("kind") in want
        if hit and r.get("value") and r["value"] not in seen:
            seen.add(r["value"])
            print(f"{r['value']}\t{cid}\t{r['status']}/{r.get('kind','')}\t"
                  f"{(r.get('reason') or r.get('statement') or '')[:70]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)

    a1 = sub.add_parser("add")
    a1.add_argument("--id", required=True)
    a1.add_argument("--value", required=True)
    a1.add_argument("--unit", default="")
    a1.add_argument("--kind", required=True, choices=sorted(KINDS))
    a1.add_argument("--statement", required=True)
    a1.add_argument("--scope", required=True)
    a1.add_argument("--source", default="")
    a1.add_argument("--runid", default="")
    a1.add_argument("--ceiling", default="")
    a1.add_argument("--derived-from", nargs="*", dest="derived_from")
    a1.add_argument("--supersedes", nargs="*")
    a1.add_argument("--force", action="store_true")
    a1.set_defaults(fn=cmd_add)

    a2 = sub.add_parser("retract")
    a2.add_argument("--id", required=True)
    a2.add_argument("--reason", required=True)
    a2.add_argument("--superseded-by", dest="superseded_by", default="")
    a2.add_argument("--value", default="")
    a2.add_argument("--force", action="store_true")
    a2.set_defaults(fn=cmd_retract)

    a3 = sub.add_parser("verify")
    a3.add_argument("--strict", action="store_true")
    a3.set_defaults(fn=cmd_verify)

    a4 = sub.add_parser("list")
    a4.add_argument("--status", default="")
    a4.add_argument("--grep", default="")
    a4.set_defaults(fn=cmd_list)

    a5 = sub.add_parser("values")
    a5.add_argument("--status", default="retracted,superseded,unlogged,commit-only")
    a5.set_defaults(fn=cmd_values)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
