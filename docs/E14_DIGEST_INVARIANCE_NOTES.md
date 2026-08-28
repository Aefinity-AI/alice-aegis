# E14 — CIS-1 op-level digest invariance to opt-level / target-cpu / LTO

Not a hardware log (Rule C append-only path is `docs/hardware_logs/`, left for
Fable to file the raw evidence there). This is a working note for that filing.

Machine: dev box, Intel i5-10210U (Comet Lake, AVX2), per CLAUDE.md.
Binary: `aegis-linux --example cis_selftest`, reference digest
`76985613c965f643` (spec §8 Tier 2, normally built with default release
profile = opt-level 3, default target-cpu, LTO off).

## Matrix (9 configs, `cargo clean -p aegis-linux --release` between each)

| RUSTFLAGS / env                                   | bin sha256 (first 8) | CIS_SELFTEST digest | match ref? |
|---|---|---|---|
| `-C opt-level=0`                                   | 8d6f4ac0 | 76985613c965f643 | yes |
| `-C opt-level=1`                                   | 4d99e413 | 76985613c965f643 | yes |
| `-C opt-level=2`                                   | 493bb3e0 | 76985613c965f643 | yes |
| `-C opt-level=3` (= default release)               | ad96adca | 76985613c965f643 | yes |
| `-C opt-level=s`                                   | 5fe44da1 | 76985613c965f643 | yes |
| `-C opt-level=3 -C target-cpu=x86-64` (baseline)    | ad96adca | 76985613c965f643 | yes |
| `-C opt-level=3 -C target-cpu=native` (host AVX2)   | 31915a1a | 76985613c965f643 | yes |
| `-C opt-level=3` + `CARGO_PROFILE_RELEASE_LTO=true` | 984bf8ca | 76985613c965f643 | yes |
| `-C opt-level=3` + `CARGO_PROFILE_RELEASE_LTO=false`| ad96adca | 76985613c965f643 | yes |

All 9/9 configs: `ALL_PASS=true`, digest `76985613c965f643` — identical to the
CI reference. Verdict: **INVARIANT**.

Note `-C opt-level=3 -C target-cpu=x86-64` and plain `-C opt-level=3` and
`LTO=false` produced byte-identical binaries (same sha256) — expected, since
`x86-64` and `LTO=false` are the implicit defaults; this is not a weakness of
the test, it just means those three rows are the same build sampled three
ways with the same result.

## Non-triviality check (codegen genuinely differs)

- `objdump -d` of opt-level=0 vs opt-level=3 binaries: **different**
  (md5 `8b8dc7ac5fc065b9ccdc5f919485a718` vs `c5725d5d884efa2dc10e1ffcbe79adc0`;
  83210 vs 74487 disassembly lines; opt0 has 3429 call sites vs 2754 at opt3 —
  consistent with opt0's un-inlined call-heavy code).
- `objdump -d` of target-cpu=x86-64 (baseline) vs target-cpu=native (host
  AVX2), both opt-level=3: **different**
  (md5 `7762997cd02070fafd50538780bf9cd1` vs `45647e3d8871b6c4a7af4bbc46e60bb2`;
  baseline binary contains **zero** ymm/vpaddd/vpmulld/vpermd instructions;
  the native binary contains AVX2 vector instructions — 213+166+... ymm0/ymm1/
  etc. register uses, `vpaddd` x23, `vpmulld` x4, `vpermd` x6).

So the invariance is not because all these builds coincidentally produced the
same machine code — they demonstrably did not — but because CIS-1's integer
reference ops are specified to be scalar/deterministic regardless of how the
compiler schedules or vectorizes the surrounding code.

Raw run log (build output + per-config binary hash + full CIS_SELFTEST
section output) is at (scratch, not in-repo):
`/tmp/claude-1000/-home-justinbrianthompson/a37ab37d-936e-4479-b31a-6b5a7265f55a/scratchpad/e14_matrix.log`
