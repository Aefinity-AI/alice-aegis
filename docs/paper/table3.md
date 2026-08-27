# Table 3 — RESEARCH_LEDGER.md rows A19–A34: row → log path

Mechanical row-by-row scrape of `program/RESEARCH_LEDGER.md` rows A19–A34
for the CIS-1 paper. One row per ledger entry (distinct from
`table3_provenance.md`, which maps individual paper *claims*, of which
there can be more than one per ledger row, to their logs). "Log path" is
the first raw-log path recorded in the ledger row's provenance field
(skipping a leading prereg/derivation `.md` document where present, e.g.
A22). Existence checked with `test -e` against the repo root on
2026-08-27.

| Row | Experiment (short) | Log path | Exists? |
|---|---|---|---|
| A19 | CIS-1 E2 gate: hybrid path PPL cost, M7 | docs/hardware_logs/cis1_e2_int_vs_float_ppl_m7_i5-10210U_crosvm_2026-08-01.log | ✓ |
| A20 | CIS-1 v0.3: full-integer path PPL cost, M7 | docs/hardware_logs/cis1_fullint_attention_ppl_m7_i5-10210U_crosvm_2026-08-01.log | ✓ |
| A21 | CIS-1 E2, BitNet-2B leg: hybrid path PPL cost | docs/hardware_logs/cis1_e2_bitnet2b_int_vs_float_ppl_i5-10210U_crosvm_2026-08-01.log | ✓ |
| A22 | MECH v2 preregistered paired redo: ring-0 unikernel vs minimal Linux decode | docs/hardware_logs/mech2_U_BOOTLOG_2026-08-01.txt | ✓ |
| A23 | Column-skip kernel candidate vs incumbent | docs/hardware_logs/mech2colskip_L_dell_BOOTLOG_2026-08-01.txt | ✓ |
| A24 | FABLE-0 gate G1: bandwidth ceiling measurement | docs/hardware_logs/mech2colskip_L_dell_BOOTLOG_2026-08-01.txt | ✓ |
| A25 | CIS-1 four-implementation digest jury (HP N4020 bare iron) | docs/hardware_logs/hp_L_BOOTLOG_2026-08-02.txt | ✓ |
| A26 | CIS-1 throughput cost by microarchitecture (Dell + HP) | docs/hardware_logs/cis_vs_float_L_dell_BOOTLOG_2026-08-05.txt | ✓ |
| A27 | AVX2 integer kernel vs float AVX2 kernel, order-controlled | docs/hardware_logs/cis_avx2_armD_ordercontrol_L_dell_BOOTLOG_2026-08-06.txt | ✓ |
| A28 | CIS-1 crosses first ISA boundary (aarch64 selftest digest) | docs/hardware_logs/cis_selftest_aarch64_github_arm_2026-08-07.log | ✓ |
| A29 | Token-level cross-ISA identity, full decode digest | docs/hardware_logs/cis_decode_token_crossisa_ci_2026-08-07.log | ✓ |
| A30 | NEON CIS-1 kernel bit-identical on real ARM silicon | docs/hardware_logs/cis_neon_tests_aarch64_github_arm_2026-08-07.log | ✓ |
| A31 | Two clean-room implementations reproduce the tier-2 digest | docs/hardware_logs/cis_cleanroom_tier2_2026-08-08.log | ✓ |
| A32 | Decode receipt verifies bit-for-bit across ISA boundary | docs/hardware_logs/witness_receipt_aarch64_ci_2026-08-08.log | ✓ |
| A33 | First physical-iron kit verification, Dell (AVX2 path) | docs/hardware_logs/dell_i5-5200U_kit_iron_verify_bootlog_2026-08-08.txt | ✓ |
| A34 | Physical-iron kit verification, HP N4020 (SSE2 scalar path) | docs/hardware_logs/hp_n4020_kit_iron_verify_bootlog_2026-08-08.txt | ✓ |
