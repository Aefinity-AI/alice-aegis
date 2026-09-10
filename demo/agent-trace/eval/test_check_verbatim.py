#!/usr/bin/env python3
"""test_check_verbatim.py — stdlib unittest for check_verbatim.check_receipt_text
using the exact shapes seen in the EVAL-60 T1 receipts (2026-09-06)."""
import unittest

from check_verbatim import check_receipt_steps, check_receipt_text


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


def episode(prompt: str, steps) -> str:
    """Multi-step receipt. `steps` is a list of (tool, arg, out) triples."""
    body = "AEGIS-TRACE v2\nK %d\nN 12\n" % len(steps)
    body += f"prompt-hex {prompt.encode().hex()}\n"
    for i, (tool, arg, out) in enumerate(steps):
        body += (
            f"step {i}: toks=1,2,3 tool={tool} in={arg.encode().hex()} "
            f"out={out.encode().hex()} decode-chain=00 ctx=00 q=00\n"
        )
    return body + "trace-chain 00\n"


class MultiStepTest(unittest.TestCase):
    """K>1: every step is checked, with the weaker post-step-0 rule."""

    PROMPT = LSHOTS + "Q: part P-901\nA:"

    def test_step0_rule_is_unchanged_when_more_steps_follow(self):
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", "see P-902"),
                                  ("lookup", "LOOKUP(P-902)", "widget")])
        self.assertEqual([s[4] for s in check_receipt_steps(r)], ["ok", "ok-tool"])

    def test_later_step_argument_from_a_tool_result_is_ok_tool(self):
        # The legitimate chaining case: step 1 asks for the key step 0 returned.
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", "see P-902"),
                                  ("lookup", "LOOKUP(P-902)", "widget")])
        step1 = check_receipt_steps(r)[1]
        self.assertEqual(step1[4], "ok-tool")
        self.assertIn("P-902", step1[3])

    def test_later_step_argument_invented_is_flagged(self):
        # P-777 appears in no prompt query and in no earlier tool result.
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", "see P-902"),
                                  ("lookup", "LOOKUP(P-777)", "")])
        self.assertEqual([s[4] for s in check_receipt_steps(r)], ["ok", "FLAG"])

    def test_later_step_may_use_any_prompt_query_not_just_the_last(self):
        # Documented permissiveness: after step 0 the current query is unknown,
        # so an argument from an earlier Q: line is accepted.
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", ""),
                                  ("lookup", "LOOKUP(P-100)", "")])
        self.assertEqual(check_receipt_steps(r)[1][4], "ok")

    def test_known_limit_later_step_key_snap_is_not_caught(self):
        # Pinned so the limit stays a choice rather than an accident: a step-1
        # snap from a lowercase key to a shot key reads as 'ok' because P-100
        # is present in the prompt. The same snap at step 0 is FLAGged.
        r = episode(LSHOTS + "Q: part p-100\nA:",
                    [("no-tool", "", ""), ("lookup", "LOOKUP(P-100)", "")])
        self.assertEqual(check_receipt_steps(r)[1][4], "ok")
        r0 = receipt(LSHOTS + "Q: part p-100\nA:", "lookup", "LOOKUP(P-100)")
        self.assertEqual(check_receipt_text(r0)[3], "FLAG")

    def test_no_tool_at_a_later_step_is_dash(self):
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", "widget"),
                                  ("no-tool", "", "")])
        self.assertEqual([s[4] for s in check_receipt_steps(r)], ["ok", "-"])

    def test_tool_result_of_a_later_step_cannot_justify_that_same_step(self):
        # Only results already recorded BEFORE the step may justify it.
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", ""),
                                  ("lookup", "LOOKUP(P-555)", "P-555 widget")])
        self.assertEqual(check_receipt_steps(r)[1][4], "FLAG")

    def test_check_receipt_text_still_reports_step_zero(self):
        r = episode(self.PROMPT, [("lookup", "LOOKUP(P-901)", "see P-902"),
                                  ("lookup", "LOOKUP(P-902)", "")])
        self.assertEqual(check_receipt_text(r)[:2], ("lookup", "P-901"))


if __name__ == "__main__":
    unittest.main()
