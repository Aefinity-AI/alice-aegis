#!/usr/bin/env python3
"""CIS-1 E2 golden generator — INDEPENDENT big-int reference.

Generates the unit golden constants for aegis-core/src/cis_infer.rs from the
same spec text (docs/CIS-1_SPEC_DRAFT_v0.1.md + the v0.2 E2 implementation
notes), using Python arbitrary-precision integers only. The Rust module under
test never produces these constants; two implementations agreeing on the same
bits is the CIS-1 conformance model in miniature (same discipline as
scripts-era cis.rs goldens).

Run: python3 scripts/cis_e2_golden_gen.py
"""

import struct

# ---------------------------------------------------------------------------
# Primitives (mirror the spec, not the Rust)
# ---------------------------------------------------------------------------


def rne_div(num, den):
    """Round-to-nearest-even division, den > 0, exact (spec section 2)."""
    assert den > 0
    q, r = divmod(num, den)  # Python floor-division == div_euclid for den > 0
    if 2 * r > den or (2 * r == den and q & 1):
        q += 1
    return q


def bf16_to_fixed(bits, frac):
    """Exact BF16 -> signed fixed-point with `frac` fractional bits, RNE."""
    sign = -1 if (bits >> 15) & 1 else 1
    exp = (bits >> 7) & 0xFF
    man = bits & 0x7F
    assert exp != 0xFF, "inf/nan not representable"
    if exp == 0:
        m, e = man, 1 - 127 - 7  # subnormal
    else:
        m, e = man | 0x80, exp - 127 - 7
    sh = e + frac
    v = m << sh if sh >= 0 else rne_div(m, 1 << -sh)
    return sign * v


def f32_to_fixed(bits, frac):
    """Exact f32 -> signed fixed-point with `frac` fractional bits, RNE."""
    sign = -1 if (bits >> 31) & 1 else 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    assert exp != 0xFF, "inf/nan not representable"
    if exp == 0:
        m, e = man, 1 - 127 - 23
    else:
        m, e = man | 0x800000, exp - 127 - 23
    sh = e + frac
    v = m << sh if sh >= 0 else rne_div(m, 1 << -sh)
    return sign * v


def f32_to_ratio(bits):
    """Exact f32 decomposition: value = (-1)^neg * m * 2^e, m odd (0 -> (0,0))."""
    neg = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    assert exp != 0xFF
    if exp == 0:
        m, e = man, 1 - 127 - 23
    else:
        m, e = man | 0x800000, exp - 127 - 23
    if m == 0:
        return (0, 0, 0)
    while m % 2 == 0:
        m //= 2
        e += 1
    return (neg, m, e)


def qscale64_from_ratio(num, den):
    """num/den (num >= 0, den > 0) as m*2^e with m in [2^62, 2^63), RNE at 63
    significant bits. num == 0 -> (0, 0)."""
    assert den > 0 and num >= 0
    if num == 0:
        return (0, 0)
    k = 62 - (num.bit_length() - den.bit_length())
    while True:
        m = rne_div(num << k, den) if k >= 0 else rne_div(num, den << -k)
        if m < 1 << 62:
            k += 1
        elif m > (1 << 63) - 1:  # includes the m == 2^63 rounding carry
            k -= 1
        else:
            return (m, -k)


def rescale(x, m, e):
    """rne(x * m * 2^e), exact."""
    p = x * m
    return p << e if e >= 0 else rne_div(p, 1 << -e)


def isqrt(x):
    import math

    return math.isqrt(x)


GQ = 20  # gain fixed-point fractional bits
F = 20  # residual fixed-point fractional bits


def normq(h, g):
    """Fused integer RMSNorm + per-token absmax i8 quantization.

    codes_i = rne(h_i*g_i * 127 / max|h*g|)  (the rms cancels in the ratio)
    scale   = num/den with num = A*n, den = 127*(t << GQ), t = isqrt(s2*n)
    """
    n = len(h)
    u = [hi * gi for hi, gi in zip(h, g)]
    a = max(abs(x) for x in u)
    if a == 0:
        return [0] * n, 0, 1
    codes = [max(-127, min(127, rne_div(x * 127, a))) for x in u]
    s2 = sum(x * x for x in h)
    t = max(isqrt(s2 * n), 1)
    return codes, a * n, 127 * (t << GQ)


def quantq(h):
    """Per-token absmax i8 quantization of a Q.F fixed-point vector (no norm)."""
    a = max(abs(x) for x in h)
    if a == 0:
        return [0] * len(h), 0, 1
    codes = [max(-127, min(127, rne_div(x * 127, a))) for x in h]
    return codes, a, 127 << F


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data):
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


# Knuth MMIX LCG, identical to cis.rs / cis_infer.rs test helpers.
MASK64 = (1 << 64) - 1


def lcg_next(state):
    state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
    return state, state


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


# ---------------------------------------------------------------------------
# Emit goldens
# ---------------------------------------------------------------------------

print("== bf16_to_fixed(bits, 20) ==")
for bits in [0x3F80, 0xBF80, 0x0000, 0x8000, 0x3FC0, 0x4248, 0x0001, 0x8001, 0x3881, 0x3883, 0x2E00]:
    print(f"  (0x{bits:04X}, {bf16_to_fixed(bits, 20)}),")

print("== f32_to_fixed(bits, 20) ==")
for bits in [
    f32_bits(1.0),
    f32_bits(-1.0),
    0x00000000,
    0x80000000,
    f32_bits(0.75),
    f32_bits(123.456),
    0x40800001,  # tie: m odd, e = -21 -> x.5 exactly
    0x40800003,
    f32_bits(-2.5e-7),
    # E2-7 recalibration: 2B-scale hybrid magnitudes (old sh<=20 guard fired
    # on real BitNet-2B relu^2 MLP products; new guard is sh<=26 / |v|<2^50).
    f32_bits(3.0e8),
    f32_bits(-1.07e9),
]:
    print(f"  (0x{bits:08X}, {f32_to_fixed(bits, 20)}),")

print("== f32_to_ratio ==")
for v in [1.0, -1.0, 0.0, 0.05, 6.25, -0.001375, 3.0]:
    bits = f32_bits(v)
    neg, m, e = f32_to_ratio(bits)
    print(f"  (0x{bits:08X}, {neg}, {m}, {e}),  // {v}")

print("== qscale64_from_ratio ==")
for num, den in [(1, 3), (5, 4), (7, 1), (1, 1), (127 * 300, 2**40), (2**90 + 12345, 3**30), (0, 5)]:
    m, e = qscale64_from_ratio(num, den)
    print(f"  ({num}, {den}, {m}, {e}),")

print("== rescale ==")
cases = [
    (1000, qscale64_from_ratio(1, 3)),
    (-1000, qscale64_from_ratio(1, 3)),
    (123456789, qscale64_from_ratio(355, 113)),
    (-7, qscale64_from_ratio(2**64, 10**9)),
    (3, qscale64_from_ratio(1, 2)),  # tie: 1.5 -> 2
    (1, qscale64_from_ratio(1, 2)),  # tie: 0.5 -> 0
]
for x, (m, e) in cases:
    print(f"  ({x}, {m}, {e}, {rescale(x, m, e)}),")

print("== normq n=16, seed 0xA1CE5EED00000005 ==")
state = 0xA1CE5EED00000005
h = []
for _ in range(16):
    state, d = lcg_next(state)
    h.append((d >> 20) % (1 << 31) - (1 << 30))  # ~ +/- 2^30
g = []
for _ in range(16):
    state, d = lcg_next(state)
    g.append((d >> 40) % (1 << 20) + (1 << 19))  # Q.20 gains in [0.5, 1.5)
codes, num, den = normq(h, g)
print(f"  h = {h}")
print(f"  g = {g}")
print(f"  codes = {codes}")
print(f"  num = {num}")
print(f"  den = {den}")

print("== normq n=16 WIDE HEADROOM (E2-7, +/-2^49), seed 0xA1CE5EED00000009 ==")
state = 0xA1CE5EED00000009
h = []
for _ in range(16):
    state, d = lcg_next(state)
    h.append((d >> 14) % (1 << 50) - (1 << 49))  # ~ +/- 2^49 (new bound is 2^50)
g = []
for _ in range(16):
    state, d = lcg_next(state)
    g.append((d >> 40) % (1 << 20) + (1 << 19))
codes, num, den = normq(h, g)
print(f"  codes = {codes}")
print(f"  num = {num}")
print(f"  den = {den}")

print("== quantq n=16, seed 0xA1CE5EED00000006 ==")
state = 0xA1CE5EED00000006
h = []
for _ in range(16):
    state, d = lcg_next(state)
    h.append((d >> 22) % (1 << 30) - (1 << 29))
codes, num, den = quantq(h)
print(f"  h = {h}")
print(f"  codes = {codes}")
print(f"  num = {num}")
print(f"  den = {den}")

print("== all-zero normq ==")
codes, num, den = normq([0] * 8, [1 << 20] * 8)
print(f"  codes = {codes}, num = {num}, den = {den}")

print("== i64 head dot, seed 0xA1CE5EED00000007 ==")
state = 0xA1CE5EED00000007
a = []
for _ in range(32):
    state, d = lcg_next(state)
    a.append((d >> 40) % 255 - 127)  # i8 in [-127, 127]
emb = []
for _ in range(32):
    state, d = lcg_next(state)
    emb.append((d >> 24) % (1 << 26) - (1 << 25))  # i32-ish Q.20 values
dot = sum(x * y for x, y in zip(a, emb))
print(f"  a = {a}")
print(f"  emb = {emb}")
print(f"  dot = {dot}")

print("== fnv1a64 ==")
for data in [b"", b"a", b"foobar"]:
    print(f"  {data!r} -> 0x{fnv1a64(data):016X}")
seq = b"".join(struct.pack("<I", t) for t in [0, 1, 65535, 8191])
print(f"  argmax-seq [0,1,65535,8191] LE -> 0x{fnv1a64(seq):016X}")

# ===========================================================================
# v0.3 sections — ROPE-I / SOFTMAX-I / ACT-I golden constants
# (docs/CIS-1_SPEC_DRAFT_v0.1.md, "v0.3 integer-attention notes").
# Independent big-int implementations of the normative procedures; the Rust
# module under test (aegis-core/src/cis_attn.rs) never produces these.
# ===========================================================================

import math

# Pinned constants: round-half-even of published 50-digit decimal expansions.
PI_1E50 = 314159265358979323846264338327950288419716939937511  # round(pi*10^50)
LN2_1E50 = 69314718055994530941723212145817656807550013436026  # round(ln2*10^50)
TWO_PI_Q62 = rne_div(2 * PI_1E50 << 62, 10**50)
PI_Q62 = rne_div(PI_1E50 << 62, 10**50)
PI_2_Q62 = rne_div(PI_1E50 << 62, 2 * 10**50)
LOG2E_Q32 = rne_div((10**50) << 32, LN2_1E50)


def exp2_chain():
    """C[k] = 2^(-2^-k) in Q0.62 by the floor-isqrt chain (normative)."""
    c = [0] * 33
    c[0] = 1 << 61
    for k in range(1, 33):
        c[k] = math.isqrt(c[k - 1] << 62)
    return c


def exp2_neg_frac(f_q32, chain):
    """2^(-f) for f in Q0.32, result Q0.62, RNE after each chain multiply."""
    acc = 1 << 62
    for k in range(1, 33):
        if (f_q32 >> (32 - k)) & 1:
            acc = rne_div(acc * chain[k], 1 << 62)
    return acc


def exp_lut():
    chain = exp2_chain()
    e = [rne_div(exp2_neg_frac(i << 22, chain), 1 << 31) for i in range(1024)]
    e.append(1 << 30)
    return e


def lut_exp2_neg(f_q32, lut):
    i = f_q32 >> 22
    r = f_q32 & ((1 << 22) - 1)
    return lut[i] - rne_div((lut[i] - lut[i + 1]) * r, 1 << 22)


def exp_neg_q31(z_q32, lut):
    y = z_q32 * LOG2E_Q32  # Q.64
    n = y >> 64
    if n >= 31:
        return 0
    f = rne_div(y & ((1 << 64) - 1), 1 << 32)
    if f == 1 << 32:
        n += 1
        f = 0
        if n >= 31:
            return 0
    return rne_div(lut_exp2_neg(f, lut), 1 << n)


def softmax_i(scores, lut):
    m = max(scores)
    e = [exp_neg_q31((m - s) << 8, lut) for s in scores]  # Q.24 -> Q.32
    S = sum(e)
    return [rne_div(x << 15, S) for x in e]


def log2_q32_f32(bits):
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    assert bits >> 31 == 0 and exp != 0xFF and exp >= 128
    e = exp - 127
    m = ((man | (1 << 23))) << 39  # Q2.62 in [2^62, 2^63)
    frac = 0
    for _ in range(32):
        m = rne_div(m * m, 1 << 62)
        frac <<= 1
        if m >= 1 << 63:
            m = rne_div(m, 2)
            frac |= 1
    return (e << 32) | frac


def sincos_q62(x):
    assert 0 <= x < TWO_PI_Q62
    sign_s, sign_c = 1, 1
    if x >= PI_Q62:
        x -= PI_Q62
        sign_s, sign_c = -1, -1
    if x >= PI_2_Q62:
        x = min(max(PI_Q62 - x, 0), PI_2_Q62)
        sign_c = -sign_c
    x2 = rne_div(x * x, 1 << 62)
    t, s = x, x
    for k in range(1, 9):
        t = rne_div(t * x2, 1 << 62)
        t = rne_div(t, (2 * k) * (2 * k + 1))
        s += -t if k & 1 else t
    t, c = 1 << 62, 1 << 62
    for k in range(1, 9):
        t = rne_div(t * x2, 1 << 62)
        t = rne_div(t, (2 * k - 1) * (2 * k))
        c += -t if k & 1 else t
    return s * sign_s, c * sign_c


def rope_table(max_seq, head_dim, base_bits):
    half = head_dim // 2
    chain = exp2_chain()
    L = log2_q32_f32(base_bits)
    inv_freq = []
    for d in range(half):
        a = rne_div(2 * d * L, head_dim)
        n, f = a >> 32, a & 0xFFFFFFFF
        assert n < 62
        inv_freq.append(rne_div(exp2_neg_frac(f, chain), 1 << n))
    cos, sin = [], []
    for pos in range(max_seq):
        for ivf in inv_freq:
            r = (pos * ivf) % TWO_PI_Q62
            s, c = sincos_q62(r)
            cos.append(max(-(1 << 30), min(1 << 30, rne_div(c, 1 << 32))))
            sin.append(max(-(1 << 30), min(1 << 30, rne_div(s, 1 << 32))))
    return cos, sin


def rope_apply(vec, pos, head_dim, cos, sin):
    half = head_dim // 2
    off = pos * half
    out = list(vec)
    for h in range(len(vec) // head_dim):
        b = h * head_dim
        for d in range(half):
            c, s = cos[off + d], sin[off + d]
            v0, v1 = vec[b + d], vec[b + d + half]
            out[b + d] = rne_div(v0 * c - v1 * s, 1 << 30)
            out[b + d + half] = rne_div(v0 * s + v1 * c, 1 << 30)
    return out


def inv_sqrt_q30(n):
    return rne_div(1 << 60, math.isqrt(n << 60))


def relu2_q20(g, u):
    if g <= 0:
        return 0
    return rne_div(rne_div(g * g, 1 << 20) * u, 1 << 20)


def silu_q20(g, u, lut):
    if g >= 0:
        t = exp_neg_q31(g << 12, lut)
        sig = rne_div(1 << 62, (1 << 31) + t)
    else:
        t = exp_neg_q31((-g) << 12, lut)
        sig = rne_div(t << 31, (1 << 31) + t)
    s = rne_div(g * sig, 1 << 31)
    return rne_div(s * u, 1 << 20)


print("== v0.3 pinned constants ==")
print(f"  TWO_PI_Q62 = {TWO_PI_Q62}")
print(f"  PI_Q62     = {PI_Q62}")
print(f"  PI_2_Q62   = {PI_2_Q62}")
print(f"  LOG2E_Q32  = {LOG2E_Q32}")

print("== exp2 chain (Q0.62) ==")
CH = exp2_chain()
for k in [1, 2, 8, 16, 32]:
    print(f"  C[{k}] = {CH[k]}")
d = FNV_OFFSET
for k in range(1, 33):
    d = fnv1a64(struct.pack("<Q", CH[k])) if False else d
# digest over C[1..=32] as u64 LE
d = fnv1a64(b"".join(struct.pack("<Q", CH[k]) for k in range(1, 33)))
print(f"  chain digest (u64 LE, k=1..32) = 0x{d:016X}")

print("== exp LUT (Q0.31) ==")
LUT = exp_lut()
for i in [0, 1, 512, 1023, 1024]:
    print(f"  E[{i}] = {LUT[i]}")
d = fnv1a64(b"".join(struct.pack("<I", LUT[i]) for i in range(1025)))
print(f"  LUT digest (u32 LE, i=0..1024) = 0x{d:016X}")

print("== log2_q32_f32 ==")
for v in [2.0, 4.0, 10000.0, 500000.0]:
    print(f"  log2({v}) -> {log2_q32_f32(f32_bits(v))}")

print("== exp_neg_q31 ==")
LN2_Q32 = rne_div(LN2_1E50 << 32, 10**50)
for z in [0, 1 << 32, 5 << 32, LN2_Q32, 22 << 32, 23 << 32, 1 << 80]:
    print(f"  exp_neg({z}) = {exp_neg_q31(z, LUT)}")
print(f"  (ln2 in Q0.32 = {LN2_Q32})")

print("== softmax_i n=8, seed 0xA1CE5EED00000008 ==")
state = 0xA1CE5EED00000008
s = []
for _ in range(8):
    state, dd = lcg_next(state)
    s.append((dd >> 30) % (1 << 26) - (1 << 25))
p = softmax_i(s, LUT)
print(f"  scores = {s}")
print(f"  probs  = {p}")
print(f"  sum    = {sum(p)}")

print("== sincos_q62 ==")
for x in [0, PI_2_Q62, PI_Q62, TWO_PI_Q62 - 1, 1 << 62, 5 << 61]:
    s, c = sincos_q62(x)
    print(f"  sincos({x}) = ({s}, {c})")

print("== rope table M7 (seq 512, head_dim 64, base 10000.0) ==")
COS, SIN = rope_table(512, 64, f32_bits(10000.0))
for (pos, dd) in [(0, 0), (1, 0), (1, 31), (100, 7), (511, 31)]:
    i = pos * 32 + dd
    print(f"  (pos {pos}, d {dd}): cos = {COS[i]}, sin = {SIN[i]}")
d = fnv1a64(b"".join(struct.pack("<i", COS[i]) + struct.pack("<i", SIN[i]) for i in range(512 * 32)))
print(f"  table digest (cos,sin i32 LE interleaved) = 0x{d:016X}")

print("== rope_apply head_dim 4, pos 1, base 10000.0 ==")
C4, S4 = rope_table(4, 4, f32_bits(10000.0))
q = rope_apply([1000000, -2000000, 3000000, 4000000], 1, 4, C4, S4)
k = rope_apply([70000, 80000, -90000, 100000], 1, 4, C4, S4)
print(f"  q -> {q}")
print(f"  k -> {k}")

print("== inv_sqrt_q30 ==")
for n in [1, 4, 64, 128]:
    print(f"  inv_sqrt_q30({n}) = {inv_sqrt_q30(n)}")

print("== relu2_q20 ==")
for (g, u) in [(-5 << 20, 3 << 20), (0, 3 << 20), (1 << 20, 1 << 20), (3 << 19, 1 << 21), (7, 1 << 20), (724, 1 << 20)]:
    print(f"  relu2({g}, {u}) = {relu2_q20(g, u)}")

print("== silu_q20 ==")
for (g, u) in [(0, 5 << 20), (1 << 20, 1 << 20), (-(1 << 20), 1 << 20), (10 << 20, 1 << 20), (-(10 << 20), 1 << 20), (-(30 << 20), 1 << 20), (3 << 19, -(1 << 20)), (-1, 1 << 30), (0, 1 << 30), (1, 1 << 30)]:
    print(f"  silu({g}, {u}) = {silu_q20(g, u, LUT)}")
