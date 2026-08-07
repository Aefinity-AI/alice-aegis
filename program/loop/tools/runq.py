#!/usr/bin/env python3
"""runq.py — resumable experiment state, designed for a frontier model that is
sometimes simply not there.

A multi-hour 529 stalled this program on 2026-07-29. The fix is NOT a local
reviewer model on a 6 GB Chromebook. The fix is to notice that the model was
never needed for the measurement — only for the interpretation — and to stop
letting the second block the first.

So every phase is declared with a GATE TYPE at declaration time (ARIS's
Type-A/Type-B taxonomy, applied to the phase rather than to a sentence):

  gate=deterministic   "could a dumb script with no taste answer this?"
                       Yes: the runner exited 0, the runcard validates, the
                       arms are work-identical, the metric recomputes from the
                       log, evidence_check finds the value. TERMINAL at
                       `verified`. A model NEVER touches it. This is most of the
                       program: every gauntlet run, every sweep, every PPL run,
                       every energy run.
  gate=interpretive    A judgment: is this scope wording honest, does this band
                       assignment hold, is this paragraph misleading, is the
                       confound really controlled. TERMINAL only at `accepted`,
                       and `accepted` requires a recorded reviewer + verdict id.
                       During an outage these go to `awaiting-review` WITH A
                       SELF-CONTAINED PACKET WRITTEN TO DISK, so the outage is a
                       queue and not a stall.

Status set (structurally enforced, not merely documented):
  pending running done failed skipped   <- `set` may write these, and only these
  verified                              <- only `verify`, and only with a named
                                           deterministic checker + its exit code
  awaiting-review                       <- only `park`, and it must write a packet
  accepted                              <- only `accept`, and only with reviewer
                                           + verdict_id, and only from done/
                                           verified/awaiting-review

`resume` resolves FORWARD to the first NON-TERMINAL phase — never the first
non-`done`. A phase the executor called done but that never passed its checker
is re-validated on resume, never silently skipped. A loop may DRIVE resume; it
may not ACQUIT a phase past itself.

State: program/loop/state/runs/<run>.json      packets: state/review_packets/
Pure stdlib. Every command works with the network down.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "state" / "runs"
PACKETS = ROOT / "state" / "review_packets"
REPO = Path(os.environ.get("ALICE_REPO", Path.home())).resolve()

EXECUTOR = {"pending", "running", "done", "failed", "skipped"}
GATES = {"deterministic", "interpretive"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(run: str) -> Path:
    safe = "".join(c for c in run if c.isalnum() or c in "-_.")
    if safe != run or run in (".", ".."):
        raise SystemExit(f"FATAL: bad run id {run!r} (use [A-Za-z0-9-_.])")
    return RUNS / f"{run}.json"


@contextmanager
def _locked(run: str):
    p = _path(run)
    p.parent.mkdir(parents=True, exist_ok=True)
    lk = p.with_suffix(".lock")
    with open(lk, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load(run: str) -> dict:
    p = _path(run)
    if not p.exists():
        raise SystemExit(f"FATAL: no run {run!r} at {p}")
    return json.loads(p.read_text())


def _save(run: str, st: dict) -> None:
    st["updated"] = _now()
    p = _path(run)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2) + "\n")
    tmp.replace(p)          # atomic: a crash mid-write never leaves a torn file


def _find(st: dict, name: str) -> dict:
    for ph in st["phases"]:
        if ph["name"] == name:
            return ph
    raise SystemExit(f"FATAL: run {st['run']!r} has no phase {name!r} "
                     f"(phases: {[p['name'] for p in st['phases']]})")


def _terminal(ph: dict) -> bool:
    if ph["status"] in ("accepted", "skipped"):
        return True
    return ph["status"] == "verified" and ph["gate"] == "deterministic"


# ---------------------------------------------------------------- commands
def cmd_start(a) -> int:
    phases = []
    for spec in a.phase:
        name, _, gate = spec.partition(":")
        gate = gate or "deterministic"
        if gate not in GATES:
            raise SystemExit(f"FATAL: phase {name!r} gate {gate!r} not in {sorted(GATES)}")
        phases.append({"name": name, "gate": gate, "status": "pending",
                       "artifact": "", "runcard": "", "checker": "",
                       "verdict_id": "", "reviewer": "", "updated": _now()})
    st = {"run": a.run, "prereg": a.prereg or "", "created": _now(),
          "question": a.question or "", "phases": phases}
    with _locked(a.run):
        if _path(a.run).exists() and not a.force:
            raise SystemExit(f"FATAL: run {a.run!r} exists (use --force to reset)")
        _save(a.run, st)
    cmd_status(a)
    return 0


def cmd_set(a) -> int:
    if a.status not in EXECUTOR:
        raise SystemExit(
            f"FATAL: `set` may only write {sorted(EXECUTOR)}.\n"
            f"  'verified' needs a named deterministic checker (`runq.py verify`).\n"
            f"  'accepted' needs a recorded reviewer + verdict id (`runq.py accept`).\n"
            f"  The executor cannot acquit its own work — that is the whole point.")
    with _locked(a.run):
        st = _load(a.run)
        ph = _find(st, a.phase)
        ph["status"] = a.status
        if a.artifact:
            ph["artifact"] = a.artifact
        if a.runcard:
            ph["runcard"] = a.runcard
        ph["updated"] = _now()
        _save(a.run, st)
    return cmd_status(a)


def cmd_verify(a) -> int:
    """Run a deterministic checker and record the result. The checker's EXIT CODE
    decides; this tool cannot be talked into a pass."""
    with _locked(a.run):
        st = _load(a.run)
        ph = _find(st, a.phase)
        if ph["status"] not in ("done", "verified", "awaiting-review", "failed"):
            raise SystemExit(f"FATAL: phase {a.phase!r} is {ph['status']!r}; run it first.")
        print(f"$ {' '.join(a.cmd)}")
        rc = subprocess.run(a.cmd, cwd=str(REPO)).returncode
        ph["checker"] = " ".join(a.cmd)
        ph["checker_rc"] = rc
        ph["checked_utc"] = _now()
        if rc == 0:
            ph["status"] = "verified"
            ph["review_independence"] = "deterministic"
            print(f"\nVERIFIED {a.phase} (checker exit 0)"
                  + ("" if ph["gate"] == "deterministic"
                     else "  — gate=interpretive, so a reviewer verdict is still owed"))
        else:
            ph["status"] = "failed"
            print(f"\nFAILED {a.phase}: checker exit {rc}. Phase is NOT verified.")
        ph["updated"] = _now()
        _save(a.run, st)
    return 0 if rc == 0 else 1


def cmd_park(a) -> int:
    """Park an interpretive phase and WRITE ITS PACKET. This is what turns an
    API outage from a stall into a queue."""
    with _locked(a.run):
        st = _load(a.run)
        ph = _find(st, a.phase)
        if ph["gate"] != "interpretive":
            raise SystemExit(
                f"FATAL: {a.phase!r} is gate=deterministic. It does not need a reviewer — "
                f"write its checker and run `runq.py verify`. Parking a mechanical phase "
                f"for a model is how an outage becomes a stall.")
        PACKETS.mkdir(parents=True, exist_ok=True)
        pk = PACKETS / f"{a.run}__{a.phase}.md"
        body = [
            f"# REVIEW PACKET — {a.run} / {a.phase}",
            f"", f"Written {_now()} because a reviewer was unavailable or not yet called.",
            f"ZERO-CONTEXT BY CONSTRUCTION: everything the reviewer needs is below.",
            f"Do NOT paste conversation history, prior verdicts, or your own summary of",
            f"the result — a leading question buys a confirming answer.",
            f"", f"## The question", f"", a.question or ph.get("question") or "(none stated)",
            f"", f"## Pre-registration (binding; written before the result)", f"",
        ]
        pre = st.get("prereg")
        if pre and (REPO / pre).is_file():
            body += ["```", (REPO / pre).read_text(errors="replace"), "```"]
        else:
            body += ["(no pre-registration file recorded for this run — say so in the verdict)"]
        body += ["", "## Artifact under review", ""]
        for art in filter(None, [ph.get("artifact"), a.attach and ",".join(a.attach)]):
            for one in str(art).split(","):
                p = REPO / one.strip()
                body += [f"### {one.strip()}", "```"]
                body += [p.read_text(errors="replace")[:120000] if p.is_file()
                         else f"(MISSING FILE: {one.strip()})", "```", ""]
        body += ["## Required verdict shape", "",
                 "One of: SUPPORTED / SUPPORTED_WITH_SCOPE_FIX / NOT_SUPPORTED /",
                 "MISLEADING_AS_WRITTEN. Then: the single most load-bearing unstated",
                 "assumption, and the one sentence you would delete.", "",
                 "Record it with:",
                 f"  ev accept {a.run} {a.phase} --reviewer <model-or-human> "
                 f"--verdict-id <thread-or-file>", ""]
        pk.write_text("\n".join(body))
        ph["status"] = "awaiting-review"
        ph["packet"] = str(pk.relative_to(ROOT))
        ph["updated"] = _now()
        _save(a.run, st)
    print(f"parked {a.phase} -> {pk}")
    print("The measurement is done and banked. Nothing downstream of a DETERMINISTIC")
    print("phase is blocked by this. Run `ev review-queue` when the API is back.")
    return 0


def cmd_accept(a) -> int:
    if not a.reviewer or not a.verdict_id:
        raise SystemExit("FATAL: accept requires --reviewer AND --verdict-id. "
                         "A phase cannot be accepted without recording who acquitted it.")
    with _locked(a.run):
        st = _load(a.run)
        ph = _find(st, a.phase)
        if ph["status"] not in ("done", "verified", "awaiting-review", "accepted"):
            raise SystemExit(f"FATAL: phase is {ph['status']!r} — cannot accept a phase "
                             f"that has not completed execution.")
        ph.update({"status": "accepted", "reviewer": a.reviewer,
                   "verdict_id": a.verdict_id, "verdict": a.verdict or "",
                   "review_independence": ("deterministic" if a.reviewer.startswith("deterministic:")
                                           else "named-reviewer"),
                   "accepted_utc": _now(), "updated": _now()})
        _save(a.run, st)
    print(f"accepted {a.phase} (reviewer={a.reviewer}, verdict_id={a.verdict_id})")
    return 0


def cmd_resume(a) -> int:
    st = _load(a.run)
    for ph in st["phases"]:
        if not _terminal(ph):
            print(json.dumps(ph, indent=2))
            hint = {
                "pending": "not started — run it",
                "running": "was interrupted (crash/OOM/session close). Re-run it; "
                           "a half-run leaves no runcard, so nothing is corrupted.",
                "done": "EXECUTED BUT NEVER CHECKED. Its checker is owed. This is the "
                        "phase a naive resume would have skipped.",
                "failed": "checker rejected it. Fix the run, not the checker.",
                "verified": "mechanically verified but gate=interpretive: a reviewer "
                            "verdict is owed. `ev park` writes the packet.",
                "awaiting-review": "packet written, waiting on a reviewer. "
                                   "See " + str(ph.get("packet", "")),
            }.get(ph["status"], "")
            print(f"\nRESUME AT: {ph['name']}  ({ph['status']}, gate={ph['gate']})\n{hint}")
            return 0
    print(f"run {a.run}: complete — every phase terminal.")
    return 0


def cmd_status(a) -> int:
    st = _load(a.run)
    glyph = {"pending": "·", "running": "▶", "done": "✓unchecked", "failed": "✗",
             "verified": "✓verified", "awaiting-review": "⧗review", "accepted": "★accepted",
             "skipped": "—skipped"}
    print(f"run {st['run']}  prereg={st.get('prereg') or '(none)'}  updated {st.get('updated','?')}")
    if st.get("question"):
        print(f"  Q: {st['question']}")
    for ph in st["phases"]:
        t = "T" if _terminal(ph) else " "
        print(f"  [{t}] {glyph.get(ph['status'],'?'):<11} {ph['name']:<26} "
              f"gate={ph['gate']:<13} {ph.get('artifact','') or ph.get('packet','')}")
    return 0


def cmd_queue(a) -> int:
    """Everything waiting on a reviewer, across every run. Burn this list down
    when the API comes back."""
    n = 0
    for f in sorted(RUNS.glob("*.json")):
        st = json.loads(f.read_text())
        for ph in st["phases"]:
            if ph["status"] == "awaiting-review":
                n += 1
                print(f"{st['run']:<28} {ph['name']:<26} {ph.get('packet','')}")
    print(f"\n{n} packet(s) awaiting a verdict."
          + ("" if n else "  Nothing is blocked on the network."))
    return 0


def cmd_list(a) -> int:
    for f in sorted(RUNS.glob("*.json")):
        st = json.loads(f.read_text())
        tot = len(st["phases"])
        done = sum(1 for p in st["phases"] if _terminal(p))
        nxt = next((p["name"] for p in st["phases"] if not _terminal(p)), "-")
        print(f"{st['run']:<30} {done}/{tot} terminal   next: {nxt}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = ap.add_subparsers(dest="c", required=True)

    p = s.add_parser("start"); p.add_argument("run")
    p.add_argument("--phase", action="append", required=True,
                   metavar="NAME[:deterministic|interpretive]")
    p.add_argument("--prereg", default=""); p.add_argument("--question", default="")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_start)

    p = s.add_parser("set"); p.add_argument("run"); p.add_argument("phase")
    p.add_argument("status"); p.add_argument("--artifact", default="")
    p.add_argument("--runcard", default=""); p.set_defaults(fn=cmd_set)

    p = s.add_parser("verify"); p.add_argument("run"); p.add_argument("phase")
    p.add_argument("cmd", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_verify)

    p = s.add_parser("park"); p.add_argument("run"); p.add_argument("phase")
    p.add_argument("--question", default=""); p.add_argument("--attach", action="append")
    p.set_defaults(fn=cmd_park)

    p = s.add_parser("accept"); p.add_argument("run"); p.add_argument("phase")
    p.add_argument("--reviewer", required=True); p.add_argument("--verdict-id",
                   dest="verdict_id", required=True)
    p.add_argument("--verdict", default=""); p.set_defaults(fn=cmd_accept)

    for name, fn in (("resume", cmd_resume), ("status", cmd_status)):
        p = s.add_parser(name); p.add_argument("run"); p.set_defaults(fn=fn)
    s.add_parser("review-queue").set_defaults(fn=cmd_queue)
    s.add_parser("runs").set_defaults(fn=cmd_list)

    a = ap.parse_args()
    if getattr(a, "cmd", None) and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
