#!/usr/bin/env python3
"""runcard.py — every number this program publishes must come out of here.

THE FINDING THIS EXISTS TO FIX (verified 2026-07-29, not asserted):

  Every throughput/kernel number that survived the DARPA forensic audit came
  out of a SCRIPT THAT WROTE A FILE:
      599 M cycles/token   aegis-uefi/qemu_success_2026-07-09.log
      3.60/3.71 B cyc/tok  aegis-uefi/matrix_logs/Nehalem_q35_sata*.log
      4.94 / 2.80 tok/s    measure_energy.sh -> energy_run_*.log
      15.825 / 16.124 PPL  aegis-eval -> wikitext2_full_ppl_*.log
      the whole gauntlet    BOOTLOG.TXT -> collect_gauntlet.sh -> *.tsv

  Every number that was RETRACTED came out of a HAND-RUN whose stdout was
  pasted into a commit message and never written to disk:
      8.25 / 3.64 / 6.00 / 7.91 tok/s, 2.27x, "SMT ~5%"   (commit 254ba43)
      9.86 / 17.13 / 8.28 GMAC/s                          (commit ffcdc4c)
      293 M -> 165 M cycles/token                         (commit 254ba43)
      17.3 GB/s bandwidth                                 (no source at all)
      3.31 J/token                                        (readings never logged)
      12.801 / 12.738 / 14.488 PPL                        (prose only)

  The correlation is 100%. The disease is not dishonesty and not carelessness.
  It is that the dev box had NO HARNESS while the bare-metal target did.

So: a runcard is the receipt for one execution of one registered runner. It
records what ran, on what, from which tree, against which artifacts, and what
came out — and it refuses to be written when the machine was not quiet or the
run produced no output. A metric that has no runcard has no business in
claims.jsonl, and claims.jsonl is the only thing a document may cite.

Pure stdlib. No network. No model. Works during a 529.

  runcard.py env [--nick NAME]
      Print the environment block (and its env_hash) as JSON. Free, read-only.

  runcard.py capture --runner NAME --log PATH [--bin PATH]... [--artifact PATH]...
                     [--nick NAME] [--allow-busy] [--note TEXT] -- CMD [ARGS...]
      Run CMD, tee its stdout+stderr to PATH, write
      docs/hardware_logs/runcards/<runid>.json. Exit code is CMD's.

  runcard.py validate RUNCARD...
      Re-verify a runcard against the filesystem: log still present and still
      hashing to the recorded digest, declared binaries unchanged, schema sane.
      Exit 1 on any failure. This is what makes a runcard a receipt rather than
      a note: it is re-checked, never remembered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(os.environ.get("ALICE_REPO", Path.home())).resolve()
RUNCARD_DIR = REPO / "docs" / "hardware_logs" / "runcards"
SCHEMA = "runcard/1"

# A competing process at this CPU share invalidates any timing measurement on
# this box. 25% is one full core of four. This threshold is not a guess: an
# 8-hour runaway `ugrep` held a core on 2026-07-09 and poisoned an entire day
# of benchmarks, which is the direct cause of commit 254ba43 and therefore of
# every retracted throughput number in this program's history.
BUSY_PCT = 25.0


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: str | Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    """Distinct (physical id, core id) pairs. Needed because the ONE decision
    this program got wrong from an unlogged measurement was logical-vs-physical
    worker count (aegis-core/src/ops.rs:621), so a runcard must record both."""
    try:
        pairs, cur = set(), {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" in line:
                k, v = (x.strip() for x in line.split(":", 1))
                cur[k] = v
            elif cur:
                if "core id" in cur:
                    pairs.add((cur.get("physical id", "0"), cur["core id"]))
                cur = {}
        if cur and "core id" in cur:
            pairs.add((cur.get("physical id", "0"), cur["core id"]))
        return len(pairs) or None
    except OSError:
        return None


def _virt() -> str:
    """crosvm / kvm / none. The dev box is a crosvm guest; the test targets have
    no OS at all. A number that does not say which is not comparable to one that
    does — this program already published a QEMU log as if it were bare metal
    (ledger A1)."""
    out = _run(["systemd-detect-virt"])
    if out:
        return out
    try:
        flags = Path("/proc/cpuinfo").read_text()
        if "hypervisor" in flags:
            return "hypervisor-flag-set"
    except OSError:
        pass
    return "none"


def _mem_total_kb() -> int | None:
    try:
        m = re.search(r"MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text())
        return int(m.group(1)) if m else None
    except OSError:
        return None


def _busiest() -> tuple[float, str]:
    """(pcpu, comm) of the busiest process that is not us. Same check
    measure_energy.sh already does at line 24 — promoted from one script to
    every run."""
    out = _run(["ps", "-eo", "pcpu,comm", "--sort=-pcpu"])
    me = {"ps", "runcard.py", "python3", "sh", "bash"}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pct = float(parts[0])
        except ValueError:
            continue
        comm = parts[1].strip()
        if comm in me:
            continue
        return pct, comm
    return 0.0, ""


def env_block(nick: str | None = None, bins: list[str] | None = None,
              artifacts: list[str] | None = None) -> dict:
    """The canonical environment fingerprint. env_hash is a content hash over
    the identity-bearing subset, so 'did the environment change?' becomes a
    string comparison instead of a memory. Two runcards with different
    env_hash values are NOT an A/B — this is the exact defect that made ledger
    A12's headline pair (0.62 -> 3.03 tok/s) CONFOUNDED: the arms were a
    2026-07-12 and a 2026-07-29 build, 8 commits apart, compared as if the
    only difference were the MSR write."""
    git = lambda *a: _run(["git", "-C", str(REPO), *a])
    dirty = bool(git("status", "--porcelain"))
    pcpu, comm = _busiest()
    env = {
        "host_nick": nick or os.environ.get("ALICE_HOST_NICK") or platform.node(),
        "cpu": _cpu_model(),
        "nproc_logical": os.cpu_count(),
        "nproc_physical": _physical_cores(),
        "virt": _virt(),
        "baremetal": _virt() == "none",
        "kernel": platform.release(),
        "mem_total_kb": _mem_total_kb(),
        "git_sha": git("rev-parse", "HEAD")[:12] or None,
        "git_describe": git("describe", "--always", "--dirty") or None,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "tree_dirty": dirty,
        "aegis_threads_env": os.environ.get("AEGIS_THREADS"),
        "rustc": _run(["rustc", "--version"]) or None,
        "binaries": {b: sha256_file(b) for b in (bins or [])},
        "artifacts": {a: sha256_file(a) for a in (artifacts or [])},
        "quiet_check": {"busiest_pcpu": pcpu, "busiest_comm": comm,
                        "threshold_pct": BUSY_PCT, "quiet": pcpu < BUSY_PCT},
    }
    # env_hash deliberately EXCLUDES quiet_check (a transient observation, not
    # an identity) and host_nick (a label; the cpu/kernel/virt triple is the
    # fact). It INCLUDES tree_dirty and every binary/artifact digest, because
    # those are precisely what makes two runs incomparable.
    ident = {k: env[k] for k in ("cpu", "nproc_logical", "nproc_physical", "virt",
                                 "kernel", "mem_total_kb", "git_sha", "tree_dirty",
                                 "rustc", "binaries", "artifacts")}
    env["env_hash"] = hashlib.sha256(
        json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return env


def cmd_env(a) -> int:
    print(json.dumps(env_block(a.nick, a.bin, a.artifact), indent=2))
    return 0


def cmd_capture(a) -> int:
    if not a.cmd:
        print("FATAL: nothing to run (put the command after `--`)", file=sys.stderr)
        return 2
    env = env_block(a.nick, a.bin, a.artifact)
    if not env["quiet_check"]["quiet"] and not a.allow_busy:
        q = env["quiet_check"]
        print(f"REFUSING TO MEASURE: {q['busiest_comm']} is at {q['busiest_pcpu']}% CPU "
              f"(threshold {BUSY_PCT}%).\n"
              f"  A busy core is how this program lost 20 days of numbers. Stop it and\n"
              f"  re-run, or pass --allow-busy to record a run explicitly labelled noisy.",
              file=sys.stderr)
        return 3

    runid = f"{time.strftime('%Y-%m-%dT%H%MZ', time.gmtime())}-{a.runner}-{env['git_sha'] or 'nogit'}"
    runid = re.sub(r"[^A-Za-z0-9._:-]", "_", runid)
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)

    started = _now()
    t0 = time.monotonic()
    with open(log, "wb") as lf:
        hdr = (f"# runid: {runid}\n# started_utc: {started}\n"
               f"# env_hash: {env['env_hash']}\n# cmd: {' '.join(a.cmd)}\n"
               f"# host: {env['host_nick']} | {env['cpu']} | virt={env['virt']} "
               f"| git={env['git_describe']}\n").encode()
        lf.write(hdr)
        lf.flush()
        # tee: the operator watches, the file records. Never one without the other.
        proc = subprocess.Popen(a.cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            lf.write(chunk)
            lf.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        rc = proc.wait()
    dur = round(time.monotonic() - t0, 3)
    ended = _now()

    post = env_block(a.nick, a.bin, a.artifact)
    card = {
        "schema": SCHEMA,
        "runid": runid,
        "runner": a.runner,
        "cmd": a.cmd,
        "started_utc": started,
        "ended_utc": ended,
        "wall_s": dur,
        "exit_code": rc,
        "env": env,
        # Re-fingerprint after the run. If the tree or a binary changed WHILE
        # the run was in flight, the run measured something the runcard cannot
        # name, and that must be visible rather than inferred.
        "env_hash_post": post["env_hash"],
        "env_stable": post["env_hash"] == env["env_hash"],
        "log": str(log.relative_to(REPO)) if log.is_absolute() and REPO in log.parents else str(log),
        "log_sha256": sha256_file(log),
        "log_bytes": log.stat().st_size if log.exists() else 0,
        "quiet_at_end": post["quiet_check"],
        "note": a.note or "",
        "metrics": {},   # filled by the runner's own parser, via `ev metric`
    }
    if card["log_bytes"] <= len(hdr):
        print("FATAL: the run produced no output. A runcard with an empty log is "
              "not evidence; not writing one.", file=sys.stderr)
        return 4

    RUNCARD_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNCARD_DIR / f"{runid}.json"
    out.write_text(json.dumps(card, indent=2) + "\n")
    print(f"\n[runcard] {out.relative_to(REPO)}", file=sys.stderr)
    print(f"[runcard] log {card['log']}  sha256 {(card['log_sha256'] or '')[:12]}  "
          f"env {env['env_hash']}  exit {rc}"
          f"{'' if card['env_stable'] else '  *** TREE CHANGED MID-RUN ***'}", file=sys.stderr)
    return rc


def cmd_validate(a) -> int:
    bad = 0
    for p in a.runcards:
        try:
            c = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"FAIL {p}: unreadable ({e})")
            bad += 1
            continue
        errs = []
        if c.get("schema") != SCHEMA:
            errs.append(f"schema {c.get('schema')!r} != {SCHEMA!r}")
        logp = Path(c.get("log", ""))
        if not logp.is_absolute():
            logp = REPO / logp
        if not logp.is_file():
            errs.append(f"log missing: {c.get('log')}")
        elif sha256_file(logp) != c.get("log_sha256"):
            errs.append(f"log MODIFIED since capture: {c.get('log')}")
        for b, want in (c.get("env", {}).get("binaries") or {}).items():
            got = sha256_file(b)
            if got is None:
                errs.append(f"declared binary gone: {b}")
            elif want and got != want:
                errs.append(f"binary changed since run: {b}")
        if not c.get("env_stable", True):
            errs.append("tree changed mid-run (env_stable=false)")
        if not (c.get("env", {}).get("quiet_check", {}) or {}).get("quiet", True):
            errs.append("machine was NOT quiet (timings from this run are noisy)")
        if errs:
            bad += 1
            print(f"FAIL {Path(p).name}")
            for e in errs:
                print(f"     - {e}")
        else:
            print(f"ok   {Path(p).name}  env={c['env']['env_hash']} exit={c['exit_code']}")
    return 1 if bad else 0


def cmd_metric(a) -> int:
    """Attach a parsed metric to an existing runcard. The runner computes it
    from its OWN log; this only records it, so a metric can never exist without
    the log line that produced it."""
    p = RUNCARD_DIR / f"{a.runid}.json"
    if not p.is_file():
        matches = sorted(RUNCARD_DIR.glob(f"*{a.runid}*.json"))
        if len(matches) != 1:
            print(f"FATAL: no unique runcard for {a.runid!r} ({len(matches)} matches)",
                  file=sys.stderr)
            return 2
        p = matches[0]
    c = json.loads(p.read_text())
    for kv in a.set:
        k, _, v = kv.partition("=")
        if not k or not _:
            print(f"FATAL: --set needs key=value, got {kv!r}", file=sys.stderr)
            return 2
        c.setdefault("metrics", {})[k.strip()] = v.strip()
    p.write_text(json.dumps(c, indent=2) + "\n")
    print(json.dumps(c["metrics"], indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="sub", required=True)

    e = sub.add_parser("env", help="print the environment fingerprint")
    e.add_argument("--nick")
    e.add_argument("--bin", action="append", default=[])
    e.add_argument("--artifact", action="append", default=[])
    e.set_defaults(fn=cmd_env)

    c = sub.add_parser("capture", help="run a command and write its runcard")
    c.add_argument("--runner", required=True, help="short runner name, e.g. thread_sweep")
    c.add_argument("--log", required=True, help="where the tee'd log goes")
    c.add_argument("--nick")
    c.add_argument("--bin", action="append", default=[], help="binary to fingerprint (repeatable)")
    c.add_argument("--artifact", action="append", default=[], help="model/data file to fingerprint")
    c.add_argument("--allow-busy", action="store_true")
    c.add_argument("--note", default="")
    c.add_argument("cmd", nargs=argparse.REMAINDER)
    c.set_defaults(fn=cmd_capture)

    v = sub.add_parser("validate", help="re-verify runcards against the filesystem")
    v.add_argument("runcards", nargs="+")
    v.set_defaults(fn=cmd_validate)

    m = sub.add_parser("metric", help="attach parsed metrics to a runcard")
    m.add_argument("runid")
    m.add_argument("--set", action="append", required=True, metavar="KEY=VALUE")
    m.set_defaults(fn=cmd_metric)

    a = ap.parse_args()
    if getattr(a, "cmd", None) and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
