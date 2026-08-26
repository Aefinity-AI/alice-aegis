# autoresearch — 2026-08-26T1027Z

Unattended integrity sweep. Produces NO measurements by design; see header.

## Evidence chain
```
ok   A13.bw.seq1t          10.57 GB/s     docs/hardware_logs/membw_2026-08-25_162849Z.log
ok   A13.bw.seq8t          25.28 GB/s     docs/hardware_logs/membw_2026-08-25_162849Z.log
ok   A13.bw.tern1t          0.80 GB/s     docs/hardware_logs/membw_2026-08-25_162849Z.log
ok   A3.baremetal      726238201 ticks/tok docs/hardware_logs/bitnet_baremetal_postfix_2026-07-29.log
note A4.sweep2026   kind=commit-only  (no source expected)
ok   A6.zeros.full        42.21%          aegis-core/benches/ctz_vs_simd.rs
note A8             kind=derived      (no source expected)
ok   A8.raw                3.988 J/tok    docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log
ok   B9                   16.124 PPL      docs/hardware_logs/wikitext2_full_ppl_2026-07-17_newtokenizer_run.log

0 live measured claim(s) failed evidence_check.
```

## Dead-number sweep (ev lint)
| file | dead-number uses |
|---|---|
| `docs/TECHNICAL_REPORT.md` | 2 |
| `program/RESEARCH_LEDGER.md` | 0 |
| `program/ROADMAP.md` | 0 |
| `aegis-core/src/ops.rs` | 0 |
| `README.md` | 0 |

## Runcard receipts
- 1 runcard(s) on disk
- all validate

## Unversioned evidence (would not survive a wipe)
- none — every hardware log is tracked

## Roadmap drift
- ⛔ blocked: 5   🔶 in-flight: 5   ⬜ todo: 9
- ROADMAP.md last touched: 2026-08-08
- **ROADMAP.md is 18 days stale** while work has continued; status markers are probably lying.

## Queued for a human (cannot be done unattended)
- box busiest process: 9.6% of a core
- measurements (`ev run …`) need a quiet box AND no resident agent session.
- ⛔ `ev run thread_sweep` BLOCKED: model weights absent (gitignored; lost in the 2026-08-24 reset).

```

0 packet(s) awaiting a verdict.  Nothing is blocked on the network.
```

---
**2 finding(s) needing attention.**

_No number in this report was produced by this run. Anything requiring measurement is queued above._
