"""Pure math for E-S1/E-S3: sub-integer bits/weight for ternary weights, and
the bytes-per-decode-step accounting from plan section 3
(state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md in claudius-maximus).

No I/O, no numpy dependency beyond what's already stdlib-adjacent — this
module is pure math so it can be unit-tested standalone and imported by
trit_census.py without pulling in the weight-loading machinery.
"""

from __future__ import annotations

import math

# --- section 0: the entropy math ------------------------------------------

# The reference "packed" baseline the plan's CSV column is named after: 5
# ternary trits fit in one byte (3**5 = 243 <= 256), i.e. 8/5 = 1.6
# bits/weight. NOTE: this is NOT what aegis-core actually ships — see
# ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT below and the E-S1 report.
REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT = 8.0 / 5.0  # = 1.6

# What aegis-core's UNPACK_LUT (aegis-core/src/ops.rs) actually implements:
# 2-bit codes, 4 trits per byte (one code, 0b11, is unused/undefined-maps-to-0).
# Found by reading aegis-core/src/ops.rs::build_unpack_lut and
# aegis-forge's upstream repacker (correct_transmute.py / local_transmute.py).
ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT = 2.0


def h_binary(p: float) -> float:
    """Binary entropy h(p) in bits. h(0) = h(1) = 0 by convention."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def h_ternary_even(p0: float) -> float:
    """Zeroth-order entropy of a ternary weight with P(0)=p0 and the
    nonzero mass split evenly between +1/-1: H(p0) = h(p0) + (1-p0).
    This is the closed form in plan section 0, and the plan's worked table
    (p0=0.90->0.569, 0.95->0.336, 0.98->0.161) is asserted against it below."""
    return h_binary(p0) + (1.0 - p0)


def h_ternary(p_minus: float, p_zero: float, p_plus: float) -> float:
    """General order-0 entropy -sum p*log2(p) for an arbitrary (possibly
    uneven) ternary split. Used whenever the +1/-1 split isn't exactly even,
    per the plan's instruction."""
    total = p_minus + p_zero + p_plus
    if total <= 0:
        return 0.0
    out = 0.0
    for p in (p_minus, p_zero, p_plus):
        p = p / total
        if p > 0:
            out -= p * math.log2(p)
    return out


# --- section 3: bytes-per-decode-step accounting ---------------------------


def bytes_packed(n_weights: int, bits_per_weight: float = REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT) -> float:
    """Bytes a decode step must stream under a fixed-rate packed scheme
    (default: the 1.6 bit/weight 5-trit/byte reference; pass
    ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT=2.0 for the format aegis-core ships)."""
    return n_weights * bits_per_weight / 8.0


def bytes_coded(n_weights: int, h_bits_per_weight: float) -> float:
    """Bytes under an entropy-coded scheme that still touches every weight
    (all N symbols are decoded — no skipping), at H bits/weight."""
    return n_weights * h_bits_per_weight / 8.0


def index_cost_bits_per_nonzero_approx(p0: float) -> float:
    """The plan's literal formula (section 3): bits per nonzero ≈
    log2(1/(1-p0)) for the gap/run-length code, +1 for the sign.

    HONESTY NOTE (verified while implementing, not asserted by the plan):
    this is the small-(1-p0) limit of the *exact* geometric-gap entropy
    h(p0)/(1-p0) (see index_cost_bits_per_nonzero_exact below) — the two
    agree closely once p0 is large (sparse: the regime the plan is actually
    about) but the approximation UNDER-states the true index cost at
    moderate p0. See approx_formula_valid_above_p0() for exactly where the
    approximation is trustworthy. Kept verbatim because the plan specifies
    it and E-S2/E-S3 are meant to use the preregistered formula; the exact
    version is provided alongside for the honesty check the project's rules
    require whenever a formula is used outside quick sanity range.
    """
    if p0 >= 1.0:
        return float("inf")
    return math.log2(1.0 / (1.0 - p0)) + 1.0


def index_cost_bits_per_nonzero_exact(p0: float) -> float:
    """Exact cost per nonzero to communicate the zero/nonzero mask via an
    optimal code for the gap (run-length) between successive nonzeros, plus
    1 sign bit. The gap is geometric with success probability r=1-p0, whose
    entropy is h(r)/r bits per gap (h = binary entropy); amortized over the
    n_nonzero gaps this exactly reproduces the mask's aggregate entropy
    h(p0) bits/weight (see skip_bits_per_weight_exact's docstring)."""
    if p0 >= 1.0:
        return float("inf")
    r = 1.0 - p0
    return h_binary(r) / r + 1.0


def bytes_skip_kernel(n_weights: int, p0: float, exact: bool = True) -> float:
    """Bytes a zero-skipping kernel must read: only the nonzero positions,
    each paying the index+sign cost above. `exact=False` uses the plan's
    literal approximate formula instead."""
    fn = index_cost_bits_per_nonzero_exact if exact else index_cost_bits_per_nonzero_approx
    n_nonzero = n_weights * (1.0 - p0)
    return n_nonzero * fn(p0) / 8.0


def skip_bits_per_weight(p0: float, exact: bool = True) -> float:
    """bytes_skip_kernel expressed as an amortized bits/weight rate, for
    apples-to-apples comparison against a fixed packed/coded rate.

    With exact=True this is IDENTICAL to h_ternary_even(p0): splitting a
    ternary symbol into "is it zero" (entropy h(p0)) + "sign, if not"
    (1 bit, even split) is itself a lossless, entropy-optimal decomposition.
    That equivalence is asserted in the unit tests below — it means an
    ideal skip-index scheme and an ideal entropy coder of the raw ternary
    stream cost the same bytes; the practical difference between them is
    architectural (random-access skipping vs. sequential decode), not
    informational."""
    fn = index_cost_bits_per_nonzero_exact if exact else index_cost_bits_per_nonzero_approx
    return (1.0 - p0) * fn(p0)


# The plan's approximate per-nonzero index cost (section 3) turns out, on
# checking it against the exact geometric-gap entropy, to under-state the
# true cost by an amount that is CONSTANT in absolute bits as p0 -> 1 — not
# vanishing the way a "small correction" would. Derivation: exact index
# cost (excluding the sign bit) is h(r)/r for r=1-p0; as r->0,
# h(r)/r = log2(1/r) + (1-r)/r * (-log2(1-r)) -> log2(1/r) + log2(e),
# while the plan's formula is just log2(1/r). The gap -> log2(e) ≈ 1.443
# bits per nonzero, not 0. It only vanishes in RELATIVE terms, and only
# once p0 is many nines from 1 (numerically: rel_tol=5% needs p0 with
# roughly 8 nines — see the unit test) — i.e. never, for any p0 a real
# ternary model exhibits. Treat the plan's formula as an order-of-magnitude
# engineering shortcut, not a byte-accurate estimator; prefer the exact
# formula (equivalently, H(p0) directly) whenever the byte count matters.
INDEX_COST_APPROX_ASYMPTOTIC_GAP_BITS = math.log2(math.e)  # ~1.4427


def approx_formula_valid_above_p0(rel_tol: float = 0.05, step: float = 1e-4,
                                   p0_ceiling: float = 1.0 - 1e-12) -> float | None:
    """Smallest p0 (scanning downward from just under 1.0) at which the
    plan's approximate formula stays within rel_tol of the exact one, or
    None if no p0 <= p0_ceiling reaches that tolerance (see module note
    above — for rel_tol=5% this is the expected outcome; the gap is an
    additive constant, not a vanishing one, over any p0 a real model has)."""
    p0 = p0_ceiling
    last_good = None
    while p0 > 0.0:
        exact = skip_bits_per_weight(p0, exact=True)
        approx = skip_bits_per_weight(p0, exact=False)
        if exact <= 0:
            break
        rel_err = abs(approx - exact) / exact
        if rel_err <= rel_tol:
            last_good = p0
        else:
            break
        p0 -= step
    return last_good


def crossover_p0(target_bits_per_weight: float, exact: bool = True,
                  lo: float = 1e-9, hi: float = 1.0 - 1e-9):
    """The p0 at which a zero-skipping kernel's bytes/weight equals
    target_bits_per_weight, or None with the sign of the (constant-sign)
    difference if no such p0 exists in (0,1) — which legitimately happens:
    e.g. the reference 1.6 bit/weight packing already exceeds max ternary
    entropy (log2(3)=1.585), so an entropy-competitive scheme beats it at
    EVERY p0, not just above some threshold. Returns (p0_or_None, always_below)."""
    f = lambda p: skip_bits_per_weight(p, exact=exact) - target_bits_per_weight
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None, flo < 0  # True => skip beats target everywhere; False => never
    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = f(mid)
        if fm == 0.0:
            return mid, None
        if (flo < 0) == (fm < 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (lo + hi) / 2.0, None


# --- unit tests -------------------------------------------------------------

_PLAN_TABLE = {
    1 / 3: 1.585,
    0.50: 1.500,
    0.70: 1.181,
    0.80: 0.922,
    0.90: 0.569,
    0.95: 0.336,
    0.9535: 0.314,
    0.97: 0.224,
    0.98: 0.161,
    0.9803: 0.158,
}


def _run_tests():
    failures = []

    # 1. h_ternary_even reproduces every point in the plan's section-0 table.
    # The two "marketing" points (0.9535->0.314, 0.9803->0.158) are on a
    # steep part of the curve (dH/dp0 ~ -5 to -8 there) and the plan only
    # gives p0 to 4 significant figures, so a p0 rounding of ~1e-4 there
    # shows up as an H error of several e-3 — wider tolerance for those two,
    # documented rather than silently loosened for everything.
    _WIDE_TOL_POINTS = {0.9535, 0.9803}
    for p0, expected in _PLAN_TABLE.items():
        got = h_ternary_even(p0)
        tol = 0.005 if p0 in _WIDE_TOL_POINTS else 0.001
        if abs(got - expected) > tol:
            failures.append(f"h_ternary_even({p0})={got:.4f} != {expected} (tol {tol})")

    # 2. h_ternary with an even split must agree with h_ternary_even.
    for p0 in (0.5, 0.8, 0.95, 0.98):
        nz = (1 - p0) / 2
        got_general = h_ternary(nz, p0, nz)
        got_even = h_ternary_even(p0)
        if abs(got_general - got_even) > 1e-9:
            failures.append(f"h_ternary even-split mismatch at p0={p0}: {got_general} vs {got_even}")

    # 3. h_ternary with an uneven split must NOT equal the even-split formula,
    # and must still sit in [0, log2(3)].
    got = h_ternary(0.01, 0.90, 0.09)
    if not (0.0 <= got <= math.log2(3) + 1e-9):
        failures.append(f"h_ternary uneven split out of range: {got}")
    if abs(got - h_ternary_even(0.90)) < 1e-6:
        failures.append("h_ternary uneven split accidentally matched even-split formula")

    # 4. Degenerate cases.
    if h_binary(0.0) != 0.0 or h_binary(1.0) != 0.0:
        failures.append("h_binary boundary case failed")
    if h_ternary_even(1.0) != 0.0:
        failures.append(f"h_ternary_even(1.0) should be 0, got {h_ternary_even(1.0)}")
    if h_ternary(0, 0, 0) != 0.0:
        failures.append("h_ternary(0,0,0) should be 0")

    # 5. index_cost / skip-kernel sanity: more zeros -> fewer nonzeros to
    # index but each costs more bits per gap; net bytes must still fall
    # monotonically as p0 -> 1 for the exact formula (it equals H(p0), which
    # is monotone decreasing for p0 > 1/3).
    n = 1_000_000
    b_skip_low = bytes_skip_kernel(n, 0.5, exact=True)
    b_skip_high = bytes_skip_kernel(n, 0.95, exact=True)
    if not (b_skip_high < b_skip_low):
        failures.append("skip-kernel(exact) bytes did not decrease with higher p0")

    # 6. bytes_coded at H(p0) must always be <= bytes_packed at the 2.0
    # bit/weight ACTUAL aegis format (entropy coding cannot cost more than
    # the fixed-rate format it is losslessly compressing) — true for every
    # p0 since max ternary entropy log2(3)=1.585 < 2.0.
    for p0 in (0.01, 1 / 3, 0.5, 0.7, 0.9, 0.95, 0.98):
        h = h_ternary_even(p0)
        bc = bytes_coded(n, h)
        bp_actual = bytes_packed(n, ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT)
        if bc > bp_actual + 1e-6:
            failures.append(f"coded bytes exceeded actual-packing bytes at p0={p0}: {bc} > {bp_actual}")

    # 7. The exact skip-kernel formula must be EXACTLY the ternary entropy
    # (a real derivation, not a coincidence: zero/nonzero mask entropy
    # h(p0) + 1 sign bit per nonzero * (1-p0) == h(p0) + (1-p0) == H(p0)).
    for p0 in (0.1, 1 / 3, 0.6, 0.9, 0.95, 0.98, 0.9803):
        exact_skip = skip_bits_per_weight(p0, exact=True)
        h = h_ternary_even(p0)
        if abs(exact_skip - h) > 1e-9:
            failures.append(f"exact skip formula != H(p0) at p0={p0}: {exact_skip} vs {h}")

    # 8. Neither the exact nor the plan's approximate skip formula ever
    # exceeds the 1.6 bit/weight reference packing rate: that reference is
    # ITSELF above the max possible ternary entropy (log2(3)=1.585 < 1.6),
    # so nothing entropy-competitive can lose to it, at any p0. This is a
    # real (if slightly deflating) finding for E-S3: against that
    # reference there is no interior "crossover" — variable-rate coding
    # always wins, by construction of the reference being non-optimal.
    p_ref, always_below_ref = crossover_p0(REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT, exact=True)
    if p_ref is not None or not always_below_ref:
        failures.append(
            f"expected 'always below 1.6 bit/weight, no interior crossover', got "
            f"p_ref={p_ref} always_below={always_below_ref}"
        )

    # 9. Against the ACTUAL shipped aegis-core format (2.0 bit/weight, which
    # DOES exceed log2(3)), the same holds: entropy-competitive skip/coding
    # beats it everywhere too (2.0 is even further from optimal than 1.6).
    p_actual, always_below_actual = crossover_p0(ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT, exact=True)
    if p_actual is not None or not always_below_actual:
        failures.append(
            f"expected 'always below 2.0 bit/weight (actual aegis packing)', got "
            f"p_actual={p_actual} always_below={always_below_actual}"
        )

    # 10. At 5% relative tolerance the crossover sits at r=1-p0 ~ 1e-8 (many
    # nines) — checked analytically (a linear p0-space scan can't resolve a
    # region this thin): just below that r the tolerance fails, just above
    # it passes, matching the asymptotic estimate r* = 2**-(C/tol-1-C).
    C = INDEX_COST_APPROX_ASYMPTOTIC_GAP_BITS
    r_star = 2 ** (-(C / 0.05 - 1 - C))
    relerr_just_looser = (
        abs(skip_bits_per_weight(1 - r_star * 3, exact=False) - skip_bits_per_weight(1 - r_star * 3, exact=True))
        / skip_bits_per_weight(1 - r_star * 3, exact=True)
    )
    relerr_just_tighter = (
        abs(skip_bits_per_weight(1 - r_star / 3, exact=False) - skip_bits_per_weight(1 - r_star / 3, exact=True))
        / skip_bits_per_weight(1 - r_star / 3, exact=True)
    )
    if not (relerr_just_looser > 0.05 > relerr_just_tighter):
        failures.append(
            f"5%-tolerance boundary not where predicted: r*={r_star:.3e}, "
            f"relerr(3r*)={relerr_just_looser:.4f}, relerr(r*/3)={relerr_just_tighter:.4f}"
        )
    if r_star > 1e-6:
        failures.append(f"expected the 5%-tolerance crossover to need r=1-p0 <~1e-6 or smaller, got {r_star:.3e}")

    p_valid_loose = approx_formula_valid_above_p0(rel_tol=0.15, step=1e-3, p0_ceiling=1.0 - 1e-9)
    if p_valid_loose is None or not (0.0 < p_valid_loose < 1.0):
        failures.append(f"expected some p0 within 15% rel tolerance, got {p_valid_loose}")

    # 11. The exact-vs-approx per-nonzero gap converges to log2(e), the
    # derived asymptotic constant, as p0 -> 1 — not to 0.
    r = 1e-9
    gap = index_cost_bits_per_nonzero_exact(1 - r) - index_cost_bits_per_nonzero_approx(1 - r)
    if abs(gap - INDEX_COST_APPROX_ASYMPTOTIC_GAP_BITS) > 0.01:
        failures.append(
            f"exact-approx gap did not converge to log2(e)={INDEX_COST_APPROX_ASYMPTOTIC_GAP_BITS:.4f}: got {gap:.4f}"
        )
    # And at a realistic model p0 (0.95), the approx formula understates
    # cost by roughly that same constant (confirms it's not negligible in
    # the regime this project actually measures).
    gap_realistic = index_cost_bits_per_nonzero_exact(0.95) - index_cost_bits_per_nonzero_approx(0.95)
    if not (1.0 < gap_realistic < 1.7):
        failures.append(f"expected ~log2(e) gap at p0=0.95, got {gap_realistic:.4f}")

    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)

    print("bytes_per_token.py: all unit tests passed")
    print("  crossover vs 1.6 bit/weight (5-trit/byte reference)  : none — skip/coded always below (ref > log2(3))")
    print("  crossover vs 2.0 bit/weight (actual aegis-core pack) : none — skip/coded always below")
    print(
        f"  plan's approx index-cost formula: NO p0 reaches <=5% rel. error "
        f"(gap -> log2(e)={INDEX_COST_APPROX_ASYMPTOTIC_GAP_BITS:.4f} bits/nonzero, additive not vanishing); "
        f"15% tolerance first reached at p0={p_valid_loose:.6f}"
    )
    for p0, expected in sorted(_PLAN_TABLE.items()):
        print(f"  H({p0:.4f}) = {h_ternary_even(p0):.4f}  (plan table: {expected})")


if __name__ == "__main__":
    _run_tests()
