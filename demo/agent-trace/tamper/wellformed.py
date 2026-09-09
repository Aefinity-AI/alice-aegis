"""Independent well-formedness checker for an AEGIS-TRACE receipt.

Written from the receipt format description, NOT from the Rust verifier, so
that "is this mutant a grammatically valid receipt?" is decided by something
other than the binary under test. It is deliberately allowed to be STRICTER
than the verifier (a receipt it accepts must be one the verifier can parse;
the converse is not required), because the corpus it gates is used to make a
claim about cryptographic rejection, and a false "well-formed" would weaken
that claim while a false "malformed" only shrinks it.

Rules (each returns a reason string on violation):
  R1  line 0 is exactly "AEGIS-TRACE v0" | "v1" | "v2"
  R2  every other line is a step line, a known lead-key line, or a WARNING line
  R3  no duplicate lead-key line
  R4  step labels are 0..S-1 in file order
  R5  K parses, K >= 1, K == S
  R6  N parses, N >= 1
  R7  prompt-hex is even-length lowercase hex, decodes to non-empty valid UTF-8
  R8  model/embed/vocab present and each 64 lowercase hex
  R9  table-sha256/suite-sha256, when present, 64 lowercase hex
  R10 format v2 carries commit (40 lowercase hex) and a non-empty host
  R11 step fields are exactly {toks,tool,in,out,decode-chain} plus {ctx,q}
      for v1/v2 (forbidden for v0); no stray tokens, no duplicate keys
  R12 toks is a non-empty comma-separated list of u32
  R13 decode-chain/ctx/q are 64 lowercase hex; in/out are even-length
      lowercase hex, possibly empty
  R14 tool is one of no-tool, calc, lookup, calc-error
  R15 the file ends with exactly one newline and carries no blank lines
  R16 a trace-chain line is present (a receipt without one is not a receipt)
"""
import re, sys

LEAD = ("AEGIS-TRACE model embed vocab K N prompt-hex trace-chain "
        "table-sha256 suite-sha256 commit host").split()
FORMATS = {"v0": 1, "v1": 2, "v2": 3}
TOOLS = {"no-tool", "calc", "lookup", "calc-error"}
HEX64 = re.compile(r'\A[0-9a-f]{64}\Z')
HEX40 = re.compile(r'\A[0-9a-f]{40}\Z')
HEXANY = re.compile(r'\A([0-9a-f]{2})*\Z')
WARN = re.compile(r'\AWARNING step ([0-9]+): tool argument not found verbatim in context\Z')


def check(text):
    """Return None if `text` is a well-formed receipt, else a reason string."""
    if not text.endswith('\n') or text.endswith('\n\n'):
        return "R15 file must end with exactly one newline"
    lines = text[:-1].split('\n')
    if any(l == '' for l in lines):
        return "R15 blank line"
    if not lines:
        return "R2 empty file"

    m = re.fullmatch(r'AEGIS-TRACE (v[0-9]+)', lines[0])
    if not m or m.group(1) not in FORMATS:
        return f"R1 bad magic line {lines[0]!r}"
    fmt = FORMATS[m.group(1)]

    lead, steps = {}, []
    for i, line in enumerate(lines):
        if line.startswith('step '):
            steps.append((i, line))
            continue
        if line.startswith('WARNING'):
            if not WARN.fullmatch(line):
                return f"R2 malformed WARNING line {i}"
            continue
        key, _, val = line.partition(' ')
        if key not in LEAD:
            return f"R2 unknown line key {key!r} on line {i}"
        if key in lead:
            return f"R3 duplicate {key} line"
        lead[key] = val
    if 'AEGIS-TRACE' not in lead or lines[0] != 'AEGIS-TRACE ' + lead['AEGIS-TRACE']:
        return "R1 magic line not first"

    for pos, (_, line) in enumerate(steps):
        label = line[len('step '):].split(':', 1)[0]
        if label != str(pos):
            return f"R4 step label {label!r} at position {pos}"

    for key in ('K', 'N'):
        if key not in lead:
            return f"R5 missing {key} line"
        if not re.fullmatch(r'[0-9]+', lead[key]):
            return f"R5 malformed {key} {lead[key]!r}"
    K, N = int(lead['K']), int(lead['N'])
    if K < 1:
        return "R5 K must be >= 1"
    if K != len(steps):
        return f"R5 K={K} but {len(steps)} step lines"
    if N < 1:
        return "R6 N must be >= 1"

    ph = lead.get('prompt-hex')
    if ph is None:
        return "R7 missing prompt-hex"
    if not HEXANY.fullmatch(ph) or ph == '':
        return "R7 prompt-hex is not non-empty even-length lowercase hex"
    try:
        if bytes.fromhex(ph).decode('utf-8') == '':
            return "R7 prompt-hex decodes to the empty string"
    except UnicodeDecodeError:
        return "R7 prompt-hex is not valid UTF-8"

    for key in ('model', 'embed', 'vocab'):
        if key not in lead:
            return f"R8 missing {key} line"
        if not HEX64.fullmatch(lead[key]):
            return f"R8 {key} is not 64 lowercase hex"
    for key in ('table-sha256', 'suite-sha256'):
        if key in lead and not HEX64.fullmatch(lead[key]):
            return f"R9 {key} is not 64 lowercase hex"
    if 'trace-chain' not in lead:
        return "R16 missing trace-chain line"
    if not HEX64.fullmatch(lead['trace-chain']):
        return "R9 trace-chain is not 64 lowercase hex"

    if fmt >= 3:
        if not HEX40.fullmatch(lead.get('commit', '')):
            return "R10 commit is not 40 lowercase hex"
        if not lead.get('host'):
            return "R10 empty or missing host"

    want = {'toks', 'tool', 'in', 'out', 'decode-chain'}
    if fmt >= 2:
        want |= {'ctx', 'q'}
    for pos, (_, line) in enumerate(steps):
        body = line.split(':', 1)[1].strip() if ':' in line else ''
        got = {}
        for field in body.split():
            if '=' not in field:
                return f"R11 step {pos}: stray token {field!r}"
            k, _, v = field.partition('=')
            if k in got:
                return f"R11 step {pos}: duplicate field {k!r}"
            got[k] = v
        if set(got) != want:
            return f"R11 step {pos}: fields {sorted(got)} != {sorted(want)}"
        ids = [t for t in got['toks'].split(',') if t != '']
        if not ids:
            return f"R12 step {pos}: empty toks"
        for t in ids:
            if not re.fullmatch(r'[0-9]+', t) or int(t) > 0xFFFFFFFF:
                return f"R12 step {pos}: bad token id {t!r}"
        for k in ('decode-chain',) + (('ctx', 'q') if fmt >= 2 else ()):
            if not HEX64.fullmatch(got[k]):
                return f"R13 step {pos}: {k} is not 64 lowercase hex"
        for k in ('in', 'out'):
            if not HEXANY.fullmatch(got[k]):
                return f"R13 step {pos}: {k} is not even-length lowercase hex"
        if got['tool'] not in TOOLS:
            return f"R14 step {pos}: unknown tool {got['tool']!r}"
    return None


if __name__ == '__main__':
    for p in sys.argv[1:]:
        with open(p, 'r', encoding='utf-8', errors='surrogateescape') as fh:
            r = check(fh.read())
        print(f"{p}\t{'WELL-FORMED' if r is None else r}")
