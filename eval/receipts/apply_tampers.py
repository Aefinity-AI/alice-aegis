#!/usr/bin/env python3
"""safe-2b tamper demo: apply 4 distinct tampers to 4 distinct EVAL-60 T1
receipts (2B model, box1), one tamper type each. Run from repo root:
    python3 eval/receipts/apply_tampers.py

Source: freshly copied, untampered receipts at
eval-60-2b-T1-box1-tampered/receipts/*.txt (60 files, copied verbatim from
the safe-2 bundle). This script mutates 4 of them in place.
"""
import re
import sys

RCPT_DIR = "eval/receipts/eval-60-2b-T1-box1-tampered/receipts"


def read(item):
    with open(f"{RCPT_DIR}/{item}.txt") as f:
        return f.read()


def write(item, text):
    with open(f"{RCPT_DIR}/{item}.txt", "w") as f:
        f.write(text)


def tamper1_flip_output_token(item):
    """Flip one token id in the step's toks= list (a generated output
    token), leaving everything else — including trace-chain — untouched."""
    text = read(item)

    def repl(m):
        toks = m.group(1).split(",")
        # flip the last token id to something different
        old = toks[-1]
        new = str(int(old) + 1)
        toks[-1] = new
        return "toks=" + ",".join(toks)

    new_text, n = re.subn(r"toks=([0-9,]+)", repl, text, count=1)
    assert n == 1, item
    write(item, new_text)
    return f"{item}: flipped last toks= token id (output token)"


def tamper2_change_tool_arg(item):
    """Change one hex byte of the tool's in= (argument) field."""
    text = read(item)

    def repl(m):
        hexstr = m.group(1)
        # flip the last hex digit
        last = hexstr[-1]
        newlast = "0" if last != "0" else "1"
        return "in=" + hexstr[:-1] + newlast

    new_text, n = re.subn(r"in=([0-9a-f]+)", repl, text, count=1)
    assert n == 1, item
    write(item, new_text)
    return f"{item}: flipped last hex digit of in= (tool arg field)"


def tamper3_truncate_chain(item):
    """Remove the last `step N:` line, leaving the (now-stale) trace-chain
    line that was computed over all steps."""
    text = read(item)
    lines = text.splitlines(keepends=True)
    step_idx = [i for i, l in enumerate(lines) if l.startswith("step ")]
    assert len(step_idx) >= 2, item
    del lines[step_idx[-1]]
    write(item, "".join(lines))
    return f"{item}: removed last step line (truncated {len(step_idx)}->{len(step_idx)-1} steps)"


def tamper4_replay_wrong_id(dest_item, src_item):
    """Overwrite dest_item's receipt with src_item's receipt content
    verbatim (replay-under-wrong-id)."""
    text = read(src_item)
    write(dest_item, text)
    return f"{dest_item}: replaced with verbatim receipt content from {src_item} (replay-under-wrong-id)"


def main():
    log = []
    log.append(tamper1_flip_output_token("calc_easy_02"))
    log.append(tamper2_change_tool_arg("calc_easy_03"))
    log.append(tamper3_truncate_chain("mixed_02"))
    log.append(tamper4_replay_wrong_id("lookup_hit_01", "lookup_hit_02"))
    for line in log:
        print(line)


if __name__ == "__main__":
    main()
