#!/usr/bin/env python3
"""prereg.py — make the pre-registration MECHANICAL.

docs/hardware_logs/m7lr_PREREGISTRATION_2026-07-29.md is the best artifact this
program has produced. It states the arms, locks the evaluation protocol, fixes
six numbered outcome bands IN ADVANCE, lists sanity aborts, names a significance
gate, bans nine specific sentences, and supplies the one licensed claim template.
It is a better instrument than anything in ARIS.

Nothing checks it. Every one of those commitments is prose, and prose is honoured
by memory. Bands 1-6 have numeric thresholds a script can evaluate. The nine
banned sentences are literal strings a script can grep. The sanity aborts are
assertions. So: add ONE fenced block to the file and the whole document becomes
enforceable, without changing a word of the prose a human reads.

  prereg.py lock  PREREG.md
      Refuse unless the file is tracked by git and unmodified. A band table
      edited after the result is seen is a forking-paths violation, and the only
      thing that makes it visible is a commit that predates the result.

  prereg.py bands PREREG.md RESULTS.json
      Evaluate the band conditions against the result. Exactly one band must
      match. Zero matches or two matches is NOT a judgment call to be resolved
      by whoever is holding the keyboard — it is an interpretive obligation, and
      the tool says so and exits non-zero so `ev park` writes a packet.

  prereg.py banned PREREG.md FILE...
      Fail if any banned sentence appears in the ledger row, commit message, or
      document written about this run. §8's list is the sharpest anti-laundering
      device in this repo and it currently has no teeth.

The block to add (anywhere in the .md; the prose stays exactly as it is):

```ev-prereg
{
  "id": "m7lr",
  "estimand": "R",
  "bands": [
    {"name": "1 cooldown-dominant", "when": "R >= 0.70 and R_ci_lo > 0.50",
     "conclusion": "Retract the M7 headline, do not merely caveat it."},
    {"name": "2 cooldown-major", "when": "0.35 <= R < 0.70 and R_ci_lo > 0.15",
     "conclusion": "Headline unsupportable as stated; report full decomposition."},
    {"name": "3 material minority", "when": "0.15 <= R < 0.35",
     "conclusion": "Headline still invalid; dominant suspect is the 2.17x param gap."},
    {"name": "4 negligible", "when": "R < 0.15 and R_ci_hi < 0.25",
     "conclusion": "Audit hypothesis not supported; publishable as a negative result."},
    {"name": "5 harmful", "when": "mean_d_cool <= 0",
     "conclusion": "Cooling did nothing or hurt. Do not reinterpret the sign."},
    {"name": "6 reversal", "when": "R > 1.0 and R_ci_lo > 1.0",
     "conclusion": "Cooled twin beats the ternary. Supersedes all bands."}
  ],
  "sanity_aborts": [
    {"when": "abs(arm_H_final_lr - 1.0e-4) > 1.0e-5", "message": "arm H final LR is not ~1.0e-4"},
    {"when": "arm_H_final_wd != 0.1", "message": "arm H wd is not 0.100"},
    {"when": "arm_K_final_wd != 0.0", "message": "arm K wd is not 0.000"},
    {"when": "arm_H_steps != 126071 or arm_K_steps != 126071", "message": "an arm did not reach step 126071"}
  ],
  "significance_gate": {"when": "p_d_cool > 0.05",
    "message": "report R as not significantly different from zero; do not headline the point estimate"},
  "internal_validity": [
    {"when": "abs(mean_d_tokens) > 0.30 * mean_d_gap",
     "message": "extra tokens are doing substantial work; report the token contribution with equal prominence"}
  ],
  "banned_sentences": [
    "ternary win survives", "residual ternary advantage", "confound ruled out",
    "confound was minor", "at equal budget"
  ]
}
```
Pure stdlib. Works offline. Exit 0 clean, 1 finding, 2 usage.
"""
from __future__ import annotations

import argparse
import ast
import json
import operator as op
import re
import subprocess
import sys
from pathlib import Path

BLOCK = re.compile(r"```ev-prereg\s*\n(.*?)\n```", re.DOTALL)

# Safe expression evaluator. `eval()` on a file that a band condition lives in
# would let a pre-registration execute code; a pre-registration is data.
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod}
_CMP = {ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge,
        ast.Eq: op.eq, ast.NotEq: op.ne}


def _ev(node, env: dict):
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"only numeric constants allowed, got {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_ev(node.left, env), _ev(node.right, env))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        v = _ev(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else (not v if isinstance(node.op, ast.Not) else v)
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v, env) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        for o, comp in zip(node.ops, node.comparators):
            if type(o) not in _CMP:
                raise ValueError(f"comparison {type(o).__name__} not allowed")
            right = _ev(comp, env)
            if not _CMP[type(o)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs":
        return abs(_ev(node.args[0], env))
    raise ValueError(f"expression node {type(node).__name__} not allowed in a pre-registration")


def safe_eval(expr: str, env: dict):
    return _ev(ast.parse(expr, mode="eval"), env)


def load_block(md: Path) -> dict:
    m = BLOCK.search(md.read_text(encoding="utf-8", errors="replace"))
    if not m:
        raise SystemExit(
            f"FATAL: {md} has no ```ev-prereg block. The prose commitments are not\n"
            f"  machine-checkable until they are also data. See this file's docstring\n"
            f"  for the block to paste; the prose does not change.")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise SystemExit(f"FATAL: ev-prereg block in {md} is not valid JSON: {e}")


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def cmd_lock(a) -> int:
    md = Path(a.prereg)
    blk = load_block(md)
    tracked = _git("ls-files", "--error-unmatch", str(md))
    dirty = _git("status", "--porcelain", "--", str(md))
    log = _git("log", "-1", "--format=%H %ci", "--", str(md))
    ok = True
    print(f"prereg  : {md}")
    print(f"id      : {blk.get('id','(unset)')}")
    print(f"bands   : {len(blk.get('bands', []))}")
    print(f"aborts  : {len(blk.get('sanity_aborts', []))}")
    print(f"banned  : {len(blk.get('banned_sentences', []))} sentence(s)")
    if not tracked:
        print("FAIL    : NOT TRACKED BY GIT. A pre-registration that is not committed "
              "before the result exists is not a pre-registration.")
        ok = False
    if dirty:
        print(f"FAIL    : MODIFIED SINCE COMMIT ({dirty.strip()}). Bands edited after a "
              f"result is a forking-paths violation. Commit or revert.")
        ok = False
    if log:
        print(f"commit  : {log}")
    for b in blk.get("bands", []):
        try:
            ast.parse(b["when"], mode="eval")
        except SyntaxError as e:
            print(f"FAIL    : band {b['name']!r} condition does not parse: {e}")
            ok = False
    print("LOCKED" if ok else "NOT LOCKED")
    return 0 if ok else 1


def cmd_bands(a) -> int:
    blk = load_block(Path(a.prereg))
    res = json.loads(Path(a.results).read_text())
    env = {k: v for k, v in res.items() if isinstance(v, (int, float, bool))}
    missing_note = []

    print(f"result variables in scope: {sorted(env)}\n")

    hard = 0
    for chk, label in ((blk.get("sanity_aborts", []), "SANITY ABORT"),
                       (blk.get("internal_validity", []), "INTERNAL VALIDITY")):
        for c in chk:
            try:
                hit = safe_eval(c["when"], env)
            except KeyError as e:
                missing_note.append(f"{label} {c['when']!r}: result has no {e.args[0]!r}")
                continue
            except ValueError as e:
                print(f"FATAL: {label} {c['when']!r}: {e}")
                return 2
            if hit:
                print(f"{label} TRIPPED: {c['message']}\n  condition: {c['when']}")
                hard += 1 if label == "SANITY ABORT" else 0
    if hard:
        print("\nNO BAND APPLIES — the arms were misconfigured. Fix the run; do not "
              "interpret the numbers.")
        return 1

    sg = blk.get("significance_gate")
    if sg:
        try:
            if safe_eval(sg["when"], env):
                print(f"SIGNIFICANCE GATE TRIPPED: {sg['message']}\n  condition: {sg['when']}\n")
        except KeyError as e:
            missing_note.append(f"SIGNIFICANCE GATE: result has no {e.args[0]!r}")

    matched = []
    for b in blk.get("bands", []):
        try:
            if safe_eval(b["when"], env):
                matched.append(b)
        except KeyError as e:
            missing_note.append(f"band {b['name']!r}: result has no {e.args[0]!r}")
        except ValueError as e:
            print(f"FATAL: band {b['name']!r}: {e}")
            return 2

    for n in missing_note:
        print(f"  note: {n}")
    print()
    if len(matched) == 1:
        b = matched[0]
        print(f"BAND: {b['name']}\n  condition : {b['when']}\n  conclusion: {b['conclusion']}")
        if blk.get("licensed_template"):
            print(f"\nOnly licensed claim shape:\n{blk['licensed_template']}")
        return 0
    if not matched:
        print("NO BAND MATCHED. The pre-registration did not anticipate this outcome.\n"
              "  This is an INTERPRETIVE obligation, not a free choice: park it\n"
              "  (`ev park <run> interpret`) and record which band you extend and why,\n"
              "  in a commit that also shows the result. Do not pick the nearest band.")
        return 1
    print(f"{len(matched)} BANDS MATCHED — the band table is not mutually exclusive:")
    for b in matched:
        print(f"  - {b['name']}: {b['when']}")
    print("  Fix the table in a commit that predates any further result, or park it.")
    return 1


def cmd_banned(a) -> int:
    blk = load_block(Path(a.prereg))
    banned = [s.lower() for s in blk.get("banned_sentences", [])]
    if not banned:
        print("no banned_sentences in this pre-registration.")
        return 0
    hits = 0
    for f in a.files:
        p = Path(f)
        if not p.is_file():
            print(f"skip (no such file): {f}")
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            low = line.lower()
            for s in banned:
                if s in low:
                    hits += 1
                    print(f"{f}:{i}: BANNED SENTENCE {s!r}\n     >>> {line.strip()[:160]}")
    print(f"\n{hits} banned-sentence use(s). "
          + ("The pre-registration prohibits these WHATEVER the result."
             if hits else "clean."))
    return 1 if hits else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = ap.add_subparsers(dest="c", required=True)
    p = s.add_parser("lock"); p.add_argument("prereg"); p.set_defaults(fn=cmd_lock)
    p = s.add_parser("bands"); p.add_argument("prereg"); p.add_argument("results")
    p.set_defaults(fn=cmd_bands)
    p = s.add_parser("banned"); p.add_argument("prereg"); p.add_argument("files", nargs="+")
    p.set_defaults(fn=cmd_banned)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
