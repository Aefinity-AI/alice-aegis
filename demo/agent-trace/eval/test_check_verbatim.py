#!/usr/bin/env python3
"""test_check_verbatim.py — stdlib unittest for check_verbatim.check_receipt_text
using the exact shapes seen in the EVAL-60 T1 receipts (2026-09-06)."""
import unittest

from check_verbatim import check_receipt_text


def receipt(prompt: str, tool: str, arg: str) -> str:
    return (
        "AEGIS-TRACE v0\nK 1\nN 24\n"
        f"prompt-hex {prompt.encode().hex()}\n"
        f"step 0: toks=1,2,3 tool={tool} in={arg.encode().hex()} out=\n"
        "trace-chain 00\n"
    )


SHOTS = "Q: 2 + 2\nA: CALC(2 + 2).\nQ: 10 + 10\nA: CALC(10 + 10).\n"
LSHOTS = "Q: part P-100\nA: LOOKUP(P-100).\nQ: part P-101\nA: LOOKUP(P-101).\n"


class VerbatimTest(unittest.TestCase):
    def test_calc_verbatim_ok(self):
        r = receipt(SHOTS + "Q: 585 + 895\nA:", "calc", "CALC(585 + 895)")
        self.assertEqual(check_receipt_text(r)[3], "ok")

    def test_shot_copy_flagged(self):
        r = receipt(SHOTS + "Q: two + two\nA:", "calc", "CALC(2 + 2)")
        self.assertEqual(check_receipt_text(r)[3], "FLAG")

    def test_key_snap_flagged(self):
        r = receipt(LSHOTS + "Q: part p-100\nA:", "lookup", "LOOKUP(P-100)")
        self.assertEqual(check_receipt_text(r)[3], "FLAG")

    def test_key_snap_other_letter_flagged(self):
        r = receipt(LSHOTS + "Q: part Q-205\nA:", "lookup", "LOOKUP(P-205)")
        self.assertEqual(check_receipt_text(r)[3], "FLAG")

    def test_lookup_verbatim_ok(self):
        r = receipt(LSHOTS + "Q: part P-4023\nA:", "lookup", "LOOKUP(P-4023)")
        self.assertEqual(check_receipt_text(r)[3], "ok")

    def test_no_tool_is_dash(self):
        r = receipt(SHOTS + "Q: 585 % 895\nA:", "no-tool", "")
        self.assertEqual(check_receipt_text(r)[3], "-")

    def test_last_query_used_not_shots(self):
        # argument matches a SHOT query but not the last one -> FLAG
        r = receipt(SHOTS + "Q: 7 * 8\nA:", "calc", "CALC(10 + 10)")
        self.assertEqual(check_receipt_text(r)[3], "FLAG")

    def test_empty_arg_flagged(self):
        r = receipt(SHOTS + "Q: 7 * 8\nA:", "calc", "CALC()")
        self.assertEqual(check_receipt_text(r)[3], "FLAG")


if __name__ == "__main__":
    unittest.main()
