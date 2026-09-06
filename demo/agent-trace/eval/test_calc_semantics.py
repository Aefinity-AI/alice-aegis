#!/usr/bin/env python3
"""test_calc_semantics.py — self-test for gen_suite.eval_calc against the
Rust-side vectors it must match (aegis-linux/examples/agent_trace.rs
eval_calc + its #[test] fixtures). stdlib unittest only.
"""
import unittest

from gen_suite import I64_MAX, I64_MIN, eval_calc


class CalcSemanticsTest(unittest.TestCase):
    def test_basic_add(self):
        self.assertEqual(eval_calc(3, "+", 4), "7")

    def test_basic_sub(self):
        self.assertEqual(eval_calc(3, "-", 4), "-1")

    def test_basic_mul(self):
        self.assertEqual(eval_calc(3, "*", 4), "12")

    def test_trunc_div_positive(self):
        self.assertEqual(eval_calc(7, "/", 2), "3")

    def test_trunc_div_negative_dividend(self):
        self.assertEqual(eval_calc(-7, "/", 2), "-3")

    def test_rem_sign_of_dividend(self):
        # 7 % -2 = 1 in Rust (remainder takes the sign of the dividend, 7).
        self.assertEqual(eval_calc(7, "%", -2), "1")

    def test_div_by_zero(self):
        self.assertEqual(eval_calc(1, "/", 0), "div-by-zero")

    def test_rem_by_zero(self):
        self.assertEqual(eval_calc(1, "%", 0), "div-by-zero")

    def test_add_overflow(self):
        self.assertEqual(eval_calc(I64_MAX, "+", 1), "overflow")

    def test_sub_overflow(self):
        self.assertEqual(eval_calc(I64_MIN, "-", 1), "overflow")

    def test_div_min_by_neg_one_overflow(self):
        self.assertEqual(eval_calc(I64_MIN, "/", -1), "overflow")

    def test_rem_min_by_neg_one_overflow(self):
        # Rust's checked_rem also returns None for i64::MIN % -1 (the
        # implied divide overflows), matching checked_div's overflow case.
        self.assertEqual(eval_calc(I64_MIN, "%", -1), "overflow")

    def test_mul_overflow(self):
        self.assertEqual(eval_calc(I64_MAX, "*", 2), "overflow")

    def test_negative_dividend_negative_divisor(self):
        # -7 / -2 = 3 (truncating toward zero), -7 % -2 = -1 (sign of
        # dividend, -7).
        self.assertEqual(eval_calc(-7, "/", -2), "3")
        self.assertEqual(eval_calc(-7, "%", -2), "-1")


if __name__ == "__main__":
    unittest.main()
