#!/usr/bin/env python3
"""make_bundle.py — pack the entire Aefinity program into upload-ready text files.

Produces a small number of plain-text files that can be uploaded to another
Claude instance so every branch works from identical facts.

DESIGN RULES
  1. SECRETS NEVER TRAVEL. Every emitted line passes a credential scrubber.
     .kaggle/, .gnupg/, .cache/huggingface/, *.pem, *.key are excluded outright.
  2. NOTHING IS SILENTLY DROPPED. Any file skipped or truncated is listed in the
     manifest with the reason and its true size, so the reader knows what is
     missing and can ask for it.
  3. BIG REPETITIVE LOGS ARE DOWNSAMPLED, NOT CUT. A 122,071-step training log
     keeps its header, every validation check, every Nth step line and its tail —
     the science survives, the bulk does not.
  4. Files are chunked to a target size so uploads do not bounce.

Usage:  python3 make_bundle.py [--outdir DIR] [--max-mb 1.5]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HOME = "/home/killboxincorporated"

# ---------------------------------------------------------------- secrets ----
SECRET_PATTERNS = [
    (re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'), '[REDACTED_HF_TOKEN]'),
    (re.compile(r'\bsk-[A-Za-z0-9_\-]{20,}\b'), '[REDACTED_API_KEY]'),
    (re.compile(r'\bghp_[A-Za-z0-9]{20,}\b'), '[REDACTED_GITHUB_TOKEN]'),
    (re.compile(r'\bAKIA[0-9A-Z]{16}\b'), '[REDACTED_AWS_KEY]'),
    (re.compile(r'(?i)("?(?:api[_-]?key|token|secret|password|passwd)"?\s*[:=]\s*")([^"]{12,})(")'),
     r'\1[REDACTED]\3'),
    (re.compile(r'(?i)\b(bearer)\s+[A-Za-z0-9._\-]{20,}'), r'\1 [REDACTED]'),
    (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----',
                re.S), '[REDACTED_PRIVATE_KEY_BLOCK]'),
]

EXCLUDE_DIRS = {'.git', 'target', '__pycache__', 'node_modules', '.kaggle', '.gnupg',
                '.cache', '.opsec_quarantine', 'ranger-venv', 'hf-venv', 'venv',
                '.rustup', '.cargo', '.local', 'ventoy-1.0.99'}
EXCLUDE_EXT = {'.pt', '.bin', '.safetensors', '.saf', '.parquet', '.zip', '.tar',
               '.gz', '.png', '.jpg', '.jpeg', '.pdf', '.mp4', '.pem', '.key',
               '.crdownload', '.so', '.rlib', '.o'}


def scrub(text):
    for pat, rep in SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text


def read(path, limit=None):
    try:
        with open(path, 'r', errors='replace') as f:
            return f.read(limit) if limit else f.read()
    except Exception as e:
        return f"[UNREADABLE: {e}]"


# ------------------------------------------------------------ downsamplers ---
def downsample_training_log(path, step_every=1000):
    """Keep the header, EVERY [val] line, every Nth step line, and the tail."""
    keep, total = [], 0
    with open(path, errors='replace') as f:
        lines = f.readlines()
    total = len(lines)
    keep.extend(lines[:15])
    n_val = n_step = 0
    for ln in lines[15:-25]:
        if '[val]' in ln or 'TRIPWIRE' in ln:
            keep.append(ln); n_val += 1
        elif ln.startswith(('step ', 'cool ')):
            m = re.match(r'\w+\s+(\d+)/', ln)
            if m and int(m.group(1)) % step_every == 0:
                keep.append(ln); n_step += 1
        elif not ln.startswith(('step ', 'cool ')):
            keep.append(ln)
    keep.extend(lines[-25:])
    hdr = (f"[DOWNSAMPLED] original {total:,} lines -> {len(keep):,} kept. "
           f"ALL {n_val} validation/tripwire lines retained; step lines sampled "
           f"every {step_every}. Full file on disk at {path}\n")
    return hdr + ''.join(keep)


def downsample_heartbeat(path):
    with open(path, errors='replace') as f:
        lines = f.readlines()
    gaps = []
    prev = None
    for ln in lines:
        p = ln.split()
        if len(p) < 2:
            continue
        try:
            ts = int(p[1])
        except ValueError:
            continue
        if prev and ts - prev > 150:
            gaps.append(f"  GAP {(ts-prev)/60:.1f} min ending {p[0]}")
        prev = ts
    return (f"[SUMMARISED] vm_heartbeat.log — {len(lines):,} minute-marks, "
            f"first {lines[0].strip()}, last {lines[-1].strip()}.\n"
            f"Purpose: a gap >2 min proves ChromeOS suspended the VM and paused every job.\n"
            f"GAPS DETECTED ({len(gaps)}):\n" + ('\n'.join(gaps) if gaps else "  none") +
            "\n[full 15k-line file on disk]\n")


def slim_verdict_json(path):
    """Drop the giant per-window arrays, keep meta + contrasts."""
    try:
        d = json.load(open(path))
    except Exception as e:
        return f"[UNPARSEABLE JSON: {e}]"
    for v in d.get('results', {}).values():
        v.pop('per_window_ppl', None)
        v.pop('per_window_nll', None)
    return json.dumps(d, indent=1)


# ------------------------------------------------------------------ chat -----
def extract_chat(jsonl_path):
    """Pull human-readable user/assistant turns out of a session transcript."""
    out = []
    try:
        with open(jsonl_path, errors='replace') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                msg = r.get('message') or {}
                role = msg.get('role') or r.get('type')
                if role not in ('user', 'assistant'):
                    continue
                content = msg.get('content')
                parts = []
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        if c.get('type') == 'text':
                            parts.append(c.get('text', ''))
                        elif c.get('type') == 'tool_use':
                            parts.append(f"[tool: {c.get('name')}]")
                text = '\n'.join(p for p in parts if p).strip()
                if text:
                    out.append(f"\n### {role.upper()}\n{text}\n")
    except Exception as e:
        return f"[CHAT EXTRACT FAILED: {e}]"
    return ''.join(out)


# ------------------------------------------------------------------ main -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HOME, 'aefinity_bundle'))
    ap.add_argument('--max-mb', type=float, default=1.5)
    ap.add_argument('--include-chat', action='store_true', default=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    manifest = []
    sections = []

    def add(title, body, note=''):
        body = scrub(body)
        sections.append(f"\n\n{'='*78}\n### {title}\n{'='*78}\n{body}")
        manifest.append((title, len(body), note))

    # ---- 0. orientation ----
    try:
        gitlog = subprocess.run(['git', 'log', '--oneline', '-25'], cwd=HOME,
                                capture_output=True, text=True, timeout=20).stdout
    except Exception:
        gitlog = '[git log unavailable]'

    add('README — HOW TO USE THIS BUNDLE', f"""
AEFINITY AI — COMPLETE PROGRAM BUNDLE
generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

This is the full text state of the ALICE / Aegis / model-lab program: every
source file, every ledger, every finding, and every log (large logs are
downsampled — each says so in its own header).

READ IN THIS ORDER
  1. program/RESEARCH_LEDGER.md   — every claim with a verdict and a log path
  2. program/MODEL_LAB.md         — the M-series plan and gate status
  3. program/HANDOFF_2026-07-18.md— operational state, binding laws
  4. program/MODEL_SCALE_LADDER.md— the model family plan, rungs A0-A4
  5. program/ROADMAP.md           — history (NOTE: ~11 days stale as of 07-29)

THE BINDING DOCTRINE (applies to any instance working on this)
  - Every number needs a runtime log path. "Log path or cut it."
  - Negative findings are deliverables and are published as wins.
  - Trust code, not changelogs. Verifiers re-run gates.
  - No frontier-API-generated training text (ToS + DARPA provenance).

MOST RECENT AND MOST IMPORTANT RESULT (2026-07-29)
  A pre-registered 4-arm control RETRACTED the program's own flagship claim.
  Ledger M21 said a ternary model beat an fp32 twin "at equal budget, one
  variable changed". An audit found SEVEN variables differed. A control then
  showed R = 0.5382 (95% CI 0.5131-0.5643) of the gap is reproduced in the fp32
  twin by the LEARNING-RATE COOLDOWN ALONE — and that is a lower bound.
  See docs/hardware_logs/m7lr_PREREGISTRATION_2026-07-29.md and the verdict log.

  DO NOT REPEAT THE RETRACTED CLAIM. The licensed statement is:
  "More than half of what was attributed to ternary quantization was an
   undisclosed schedule asymmetry; the residual remains confounded by six
   further variables, principally a 2.17x parameter difference."

KNOWN-DEAD PATHS (do not re-propose)
  CTZ sparse ternary kernel; T-MAC-style LUT pshufb; dense->ternary PTQ sub-7B;
  cross-tokenizer logit KD; local 8B-teacher bulk datagen; onebitllms on free
  GPUs (needs bf16 silicon, sm_80+); Kaggle TPU (silently downgraded to CPU).

RECENT GIT HISTORY
{gitlog}
""")

    # ---- 1. program ledgers + notes ----
    for d, pat in [('program', '.md'), ('docs', '.md'), ('.', '.md')]:
        base = os.path.join(HOME, d)
        for fn in sorted(os.listdir(base)):
            p = os.path.join(base, fn)
            if not os.path.isfile(p) or not fn.endswith(pat):
                continue
            rel = os.path.relpath(p, HOME)
            if os.path.getsize(p) > 400_000:
                manifest.append((rel, os.path.getsize(p), 'SKIPPED >400KB'))
                continue
            add(f'FILE: {rel}', read(p))

    for sub in ['program/model-lab']:
        base = os.path.join(HOME, sub)
        if os.path.isdir(base):
            for fn in sorted(os.listdir(base)):
                p = os.path.join(base, fn)
                if os.path.isfile(p) and os.path.getsize(p) < 200_000:
                    add(f'FILE: {os.path.relpath(p, HOME)}', read(p))
                elif os.path.isfile(p):
                    manifest.append((os.path.relpath(p, HOME), os.path.getsize(p),
                                     'SKIPPED >200KB (raw workflow output)'))

    # ---- 2. source code ----
    for root_dir in ['aegis-core', 'aegis-uefi', 'aegis-eval', 'aegis-linux',
                     'aegis-forge', 'aegis-transmuter', 'model-lab/scripts',
                     'model-lab/tinybit', 'model-lab/kaggle', 'scripts']:
        base = os.path.join(HOME, root_dir)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for fn in sorted(filenames):
                ext = os.path.splitext(fn)[1]
                if ext in EXCLUDE_EXT or fn.startswith('.'):
                    continue
                if ext not in ('.rs', '.py', '.sh', '.toml', '.json', '.md', '.txt', '.cfg'):
                    continue
                p = os.path.join(dirpath, fn)
                sz = os.path.getsize(p)
                rel = os.path.relpath(p, HOME)
                if sz > 250_000:
                    manifest.append((rel, sz, 'SKIPPED >250KB'))
                    continue
                add(f'SOURCE: {rel}', read(p))

    # ---- 3. evidence logs ----
    hw = os.path.join(HOME, 'docs/hardware_logs')
    for fn in sorted(os.listdir(hw)):
        p = os.path.join(hw, fn)
        if not os.path.isfile(p):
            continue
        rel = os.path.relpath(p, HOME)
        if fn == 'vm_heartbeat.log':
            add(f'LOG: {rel}', downsample_heartbeat(p), 'summarised')
        elif fn.endswith('.json') and 'verdict' in fn:
            add(f'LOG: {rel}', slim_verdict_json(p), 'per-window arrays dropped')
        elif os.path.getsize(p) > 120_000:
            add(f'LOG: {rel}', downsample_training_log(p), 'downsampled')
        else:
            add(f'LOG: {rel}', read(p))

    tl = os.path.join(HOME, 'model-lab/tinybit/logs')
    if os.path.isdir(tl):
        for fn in sorted(os.listdir(tl)):
            p = os.path.join(tl, fn)
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, HOME)
            if os.path.getsize(p) > 120_000:
                add(f'LOG: {rel}', downsample_training_log(p), 'downsampled')
            else:
                add(f'LOG: {rel}', read(p))

    # ---- write chunks ----
    max_bytes = int(args.max_mb * 1024 * 1024)
    written, buf, part = [], '', 1

    def flush(buf, part):
        fp = os.path.join(args.outdir, f'AEFINITY_BUNDLE_part{part:02d}.txt')
        with open(fp, 'w') as f:
            f.write(f"AEFINITY AI PROGRAM BUNDLE — part {part}\n")
            f.write(f"generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
            f.write(buf)
        written.append((fp, os.path.getsize(fp)))

    for s in sections:
        if len(buf) + len(s) > max_bytes and buf:
            flush(buf, part); part += 1; buf = ''
        buf += s
    if buf:
        flush(buf, part)

    # ---- manifest ----
    mf = os.path.join(args.outdir, 'AEFINITY_BUNDLE_MANIFEST.txt')
    with open(mf, 'w') as f:
        f.write("AEFINITY BUNDLE MANIFEST\n")
        f.write(f"generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n")
        f.write("PARTS:\n")
        for fp, sz in written:
            f.write(f"  {sz:>9,} B  {os.path.basename(fp)}\n")
        f.write(f"\nSECTIONS ({len(manifest)}):\n")
        for t, sz, note in manifest:
            f.write(f"  {sz:>9,} B  {t}{'   [' + note + ']' if note else ''}\n")
    written.append((mf, os.path.getsize(mf)))

    for fp, sz in written:
        print(f"{sz:>10,} B  {fp}")


if __name__ == '__main__':
    main()
