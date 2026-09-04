"""A real, lossless static rANS coder (order-0 and order-1), vectorized over
many interleaved lanes with numpy so it runs at practical speed in pure
Python. Written for E-S1 (trit census of ternary weights) — see
state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md in claudius-maximus.

This is NOT a toy: it is a real asymmetric numeral system coder (the "ryg_rans"
16-bit-renormalization design: 32-bit-ish state, one conditional renorm word
per symbol) that must satisfy an exact byte-for-byte round trip. Every call
site in trit_census.py asserts that round trip; there is no "trust the math"
shortcut here.

Vectorization trick (interleaved / striped rANS, a standard technique used by
zstd's FSE and GPU rANS implementations): split the N input symbols into K
independent lanes, each a *contiguous* chunk of the original sequence
(`symbols.reshape(K, T)`), so within a lane consecutive columns are also
consecutive in the original data — this keeps an order-1 "previous symbol"
context well-defined per lane (except at the K chunk boundaries, negligible
for K << N). Because rANS is a LIFO stack, each lane is encoded processing
its own T columns in *reverse* (t = T-1 .. 0); all K lanes are stepped
together so every operation is a numpy array op over K elements, not a
Python-level loop over N symbols. Decoding replays each lane forward
(t = 0 .. T-1), which is exactly how ryg_rans is meant to be used; the only
difference here is K independent copies run side by side.
"""

from __future__ import annotations

import numpy as np

RANS_L = 1 << 16          # lower bound of the normalized state interval
SCALE_BITS = 12           # quantized-probability precision: M = 4096
M = 1 << SCALE_BITS
RENORM_MASK = 0xFFFF      # 16-bit renormalization word


def quantize_freqs(counts: np.ndarray, total_m: int = M) -> np.ndarray:
    """Turn integer counts into frequencies summing exactly to total_m, with
    every symbol that has count > 0 guaranteed freq >= 1 (required so rANS can
    losslessly represent it). Largest-remainder rounding.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum()
    if n <= 0:
        # No data at all (empty tensor slice) — degenerate uniform table.
        freqs = np.full(len(counts), total_m // len(counts), dtype=np.int64)
        freqs[: total_m - freqs.sum()] += 1
        return freqs
    raw = counts / n * total_m
    freqs = np.floor(raw).astype(np.int64)
    # Any symbol that actually occurs must get at least 1 slot.
    nonzero = counts > 0
    freqs[nonzero & (freqs == 0)] = 1
    deficit = total_m - freqs.sum()
    if deficit > 0:
        # Hand out the remaining slots to the symbols with the largest
        # fractional remainder (largest-remainder / Hamilton apportionment).
        remainder = raw - np.floor(raw)
        remainder[~nonzero] = -1.0  # never grow a symbol that never occurs
        order = np.argsort(-remainder)
        for i in range(deficit):
            freqs[order[i % len(order)]] += 1
    elif deficit < 0:
        # Take slots away from the largest frequencies first, never below 1
        # for an occurring symbol.
        order = np.argsort(-freqs)
        i = 0
        while deficit < 0:
            idx = order[i % len(order)]
            if freqs[idx] > (1 if nonzero[idx] else 0):
                freqs[idx] -= 1
                deficit += 1
            i += 1
    assert freqs.sum() == total_m
    return freqs


def cumfreq_of(freqs: np.ndarray) -> np.ndarray:
    c = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=c[1:])
    return c


def symbol_lut_of(freqs: np.ndarray, cum: np.ndarray, total_m: int = M) -> np.ndarray:
    lut = np.zeros(total_m, dtype=np.uint16)
    for s in range(len(freqs)):
        lut[cum[s] : cum[s + 1]] = s
    return lut


def _choose_lanes(n: int, max_lanes: int = 64) -> int:
    """Lane count trades Python-loop speed for per-lane bookkeeping overhead
    (each lane needs its own 32-bit final state + word count — real cost of
    the interleaved-stream trick, not a coding-theoretic one). Capped at 64:
    on the tensor sizes here (>=10^5 weights) that keeps total bookkeeping
    well under 0.01 bits/weight while still vectorizing the inner loop 64x,
    which is what makes this run in seconds instead of hours."""
    if n <= 0:
        return 1
    k = max(1, min(max_lanes, n // 4096))
    return k


def encode_order0(symbols: np.ndarray, alphabet: int, K: int | None = None):
    """Encode a 1-D array of small non-negative ints (values < alphabet).
    Returns a dict blob sufficient for exact decode_order0."""
    n = symbols.shape[0]
    K = K or _choose_lanes(n)
    T = n // K
    used = T * K
    main = symbols[:used]
    tail = symbols[used:]  # raw remainder, < K elements

    counts = np.bincount(main, minlength=alphabet).astype(np.int64)
    freqs = quantize_freqs(counts)
    cum = cumfreq_of(freqs)

    grid = main.reshape(K, T)  # row k = contiguous original chunk k
    blob = _encode_grid(grid, freqs, cum, lambda t, g: None)
    blob.update(
        alphabet=alphabet,
        freqs=freqs,
        cum=cum,
        K=K,
        T=T,
        n=n,
        tail=tail.copy(),
        order=0,
    )
    return blob


def decode_order0(blob) -> np.ndarray:
    grid = _decode_grid(blob, lambda t, g: None)
    out = np.empty(blob["n"], dtype=np.uint8)
    out[: blob["K"] * blob["T"]] = grid.reshape(-1)
    if blob["tail"].size:
        out[blob["K"] * blob["T"] :] = blob["tail"]
    return out


def encode_order1(symbols: np.ndarray, alphabet: int, K: int | None = None):
    """Order-1 context = previous symbol in the original (flat, row-major)
    sequence; the first element of each lane's chunk uses a fixed context
    of 0 (a documented convention — see module docstring), which biases at
    most K positions out of n."""
    n = symbols.shape[0]
    K = K or _choose_lanes(n)
    T = n // K
    used = T * K
    main = symbols[:used]
    tail = symbols[used:]

    grid = main.reshape(K, T)
    ctx = np.empty_like(grid)
    ctx[:, 1:] = grid[:, :-1]
    ctx[:, 0] = 0

    joint = np.zeros((alphabet, alphabet), dtype=np.int64)
    flat_ctx = ctx.reshape(-1)
    flat_sym = grid.reshape(-1)
    np.add.at(joint, (flat_ctx, flat_sym), 1)

    freqs2 = np.stack([quantize_freqs(joint[c]) for c in range(alphabet)])
    cum2 = np.stack([cumfreq_of(freqs2[c]) for c in range(alphabet)])

    def ctx_at(t, g):
        return ctx[:, t]

    blob = _encode_grid(grid, freqs2, cum2, ctx_at, per_context=True)
    blob.update(
        alphabet=alphabet,
        freqs=freqs2,
        cum=cum2,
        K=K,
        T=T,
        n=n,
        tail=tail.copy(),
        order=1,
    )
    return blob


def decode_order1(blob) -> np.ndarray:
    K, T, alphabet = blob["K"], blob["T"], blob["alphabet"]
    freqs2, cum2 = blob["freqs"], blob["cum"]
    lut2 = np.stack([symbol_lut_of(freqs2[c], cum2[c]) for c in range(alphabet)])

    grid = np.zeros((K, T), dtype=np.uint8)
    ctx_col = np.zeros(K, dtype=np.uint8)  # context for column 0 is fixed 0

    x = blob["final_state"].astype(np.uint64).copy()
    words = blob["words"]
    lane_len = blob["lane_len"]
    ptr = np.zeros(K, dtype=np.int64)

    for t in range(T):
        # gather per-lane freq/cum rows for this lane's current context
        freq_row = freqs2[ctx_col]      # (K, alphabet)
        cum_row = cum2[ctx_col]         # (K, alphabet+1)
        slot = (x & (M - 1)).astype(np.int64)
        # symbol = which bucket slot falls into, per lane's own context table
        sym = lut2[ctx_col, slot]
        f_s = freq_row[np.arange(K), sym].astype(np.uint64)
        c_s = cum_row[np.arange(K), sym].astype(np.uint64)
        x = f_s * (x >> SCALE_BITS) + slot.astype(np.uint64) - c_s
        need = x < RANS_L
        if need.any():
            idx = np.nonzero(need)[0]
            w = words[idx, ptr[idx]].astype(np.uint64)
            x[idx] = (x[idx] << 16) | w
            ptr[idx] += 1
        grid[:, t] = sym
        ctx_col = sym

    out = np.empty(blob["n"], dtype=np.uint8)
    out[: K * T] = grid.reshape(-1)
    if blob["tail"].size:
        out[K * T :] = blob["tail"]
    return out


def _encode_grid(grid: np.ndarray, freqs, cum, ctx_fn, per_context: bool = False):
    K, T = grid.shape
    x = np.full(K, RANS_L, dtype=np.uint64)
    # Upper bound on words any single lane can emit: worst case every symbol
    # forces a renorm (true only for a maximally-skewed table); allocate
    # generously and trim.
    max_words = T + 4
    words = np.zeros((K, max_words), dtype=np.uint16)
    nwritten = np.zeros(K, dtype=np.int64)

    for t in range(T - 1, -1, -1):
        s = grid[:, t]
        if per_context:
            c_idx = ctx_fn(t, grid)
            f = freqs[c_idx, s].astype(np.uint64)
            c = cum[c_idx, s].astype(np.uint64)
        else:
            f = freqs[s].astype(np.uint64)
            c = cum[s].astype(np.uint64)
        x_max = ((RANS_L >> SCALE_BITS) << 16) * f
        emit = x >= x_max
        if emit.any():
            idx = np.nonzero(emit)[0]
            words[idx, nwritten[idx]] = (x[idx] & RENORM_MASK).astype(np.uint16)
            nwritten[idx] += 1
            x[idx] = x[idx] >> 16
        x = ((x // f) << SCALE_BITS) + (x % f) + c

    lane_len = nwritten.copy()
    max_len = int(lane_len.max()) if K else 0
    # Words were appended in emission order, which is DECREASING t (reverse
    # chronological). The decoder consumes them in INCREASING t. Reverse each
    # lane's own (variable-length) prefix in place.
    for k in range(K):
        L = lane_len[k]
        if L:
            words[k, :L] = words[k, :L][::-1]

    return dict(final_state=x.copy(), words=words[:, :max_len].copy(), lane_len=lane_len)


def _decode_grid(blob, ctx_fn):
    K, T, alphabet = blob["K"], blob["T"], blob["alphabet"]
    freqs, cum = blob["freqs"], blob["cum"]
    lut = symbol_lut_of(freqs, cum)

    grid = np.zeros((K, T), dtype=np.uint8)
    x = blob["final_state"].astype(np.uint64).copy()
    words = blob["words"]
    ptr = np.zeros(K, dtype=np.int64)

    for t in range(T):
        slot = (x & (M - 1)).astype(np.int64)
        sym = lut[slot]
        f_s = freqs[sym].astype(np.uint64)
        c_s = cum[sym].astype(np.uint64)
        x = f_s * (x >> SCALE_BITS) + slot.astype(np.uint64) - c_s
        need = x < RANS_L
        if need.any():
            idx = np.nonzero(need)[0]
            w = words[idx, ptr[idx]].astype(np.uint64)
            x[idx] = (x[idx] << 16) | w
            ptr[idx] += 1
        grid[:, t] = sym

    return grid


def coded_size_bytes(blob) -> int:
    """Total bytes the coded representation actually occupies: the per-lane
    word streams + final states + (small) tables + raw tail. This is the
    honest 'achieved bytes', not a theoretical estimate."""
    words_bytes = int(blob["lane_len"].sum()) * 2
    states_bytes = blob["K"] * 4
    if blob["order"] == 0:
        table_bytes = len(blob["freqs"]) * 2  # freqs only; cum is derived
    else:
        table_bytes = blob["freqs"].size * 2
    tail_bytes = int(np.ceil(blob["tail"].size * 2 / 8))  # 2 raw bits/symbol
    lane_len_bytes = blob["K"] * 4  # lengths must be transmitted too
    return words_bytes + states_bytes + table_bytes + tail_bytes + lane_len_bytes


def roundtrip_order0(symbols: np.ndarray, alphabet: int, K: int | None = None):
    blob = encode_order0(symbols, alphabet, K)
    back = decode_order0(blob)
    mismatches = int(np.count_nonzero(back != symbols))
    return blob, mismatches


def roundtrip_order1(symbols: np.ndarray, alphabet: int, K: int | None = None):
    blob = encode_order1(symbols, alphabet, K)
    back = decode_order1(blob)
    mismatches = int(np.count_nonzero(back != symbols))
    return blob, mismatches


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(0)
    failures = 0

    def check(name, symbols, alphabet):
        global failures
        b0, m0 = roundtrip_order0(symbols, alphabet)
        b1, m1 = roundtrip_order1(symbols, alphabet)
        n = symbols.shape[0]
        h0 = coded_size_bytes(b0) * 8 / n
        h1 = coded_size_bytes(b1) * 8 / n
        status = "OK" if (m0 == 0 and m1 == 0) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"[{status}] {name}: n={n} mism0={m0} mism1={m1} "
            f"bits/sym order0={h0:.4f} order1={h1:.4f}"
        )

    # 1. Skewed ternary, p0=0.95 (matches plan's 0.336 bits/weight point)
    n = 200_000
    p0, p1, p2 = 0.95, 0.025, 0.025
    syms = rng.choice(3, size=n, p=[p0, p1, p2]).astype(np.uint8)
    check("skewed p0=0.95", syms, 3)

    # 2. Even split, p0=1/3 (max entropy point, H=1.585)
    syms = rng.choice(3, size=n, p=[1 / 3, 1 / 3, 1 / 3]).astype(np.uint8)
    check("even 1/3", syms, 3)

    # 3. Extreme skew, p0=0.995 (near the 0.158 bit point family)
    syms = rng.choice(3, size=n, p=[0.98, 0.01, 0.01]).astype(np.uint8)
    check("skewed p0=0.98", syms, 3)

    # 4. Degenerate: all zeros (a symbol with prob 1 — edge case for freq table)
    syms = np.zeros(50_000, dtype=np.uint8)
    check("all-zero", syms, 3)

    # 5. Strong order-1 correlation: symbol repeats its predecessor 90% of the time
    syms = np.zeros(n, dtype=np.uint8)
    cur = 0
    rnd = rng.random(n)
    for i in range(n):
        if rnd[i] < 0.9:
            pass  # repeat
        else:
            cur = rng.integers(0, 3)
        syms[i] = cur
    check("order1-correlated", syms, 3)

    # 6. Odd length (exercises the raw tail path) and tiny input
    check("odd-length-tiny", rng.choice(3, size=137, p=[0.7, 0.15, 0.15]).astype(np.uint8), 3)
    check("odd-length-small", rng.choice(3, size=10_003, p=[0.6, 0.2, 0.2]).astype(np.uint8), 3)

    if failures:
        print(f"{failures} check(s) FAILED", file=sys.stderr)
        sys.exit(1)
    print("all rans_codec self-tests passed")
