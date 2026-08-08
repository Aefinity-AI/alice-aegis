# Session Protocol — how any session (local or cloud) picks up this program

The program's recurring failure mode is context loss between sessions
(API errors, network moves, OOM kills). This file is the countermeasure.
It is deliberately short; everything it points to carries the detail.

## Boot sequence (first 5 minutes of any session)

1. Read `program/ROADMAP.md` — the master list of done / in-flight / next,
   each with provenance. This is the single source of truth for "where are we."
2. Read the auto-memory index (`~/.claude/projects/-home-killboxincorporated/memory/MEMORY.md`)
   and any memory file the current task touches.
3. If working the hybrid-engine line: read `HANDOFF_2026-07-13_TERNARY_PORTFOLIO.md`.
   If working RANGER: read `ranger/research/phase1/RESULTS-phase1.md` (verdicts)
   and `ranger/research/phase2/PLAN.md`.
4. `git log --oneline -10` in the repo you're touching. Trust code, not changelogs.

## Execution discipline (non-negotiable, each has burned us)

- **Measurement integrity**: every number comes from a runtime computation in
  a log you can name. No number enters `ROADMAP.md` or `RESEARCH_LEDGER.md`
  without a provenance path. Fabricated benchmarks already cost this project
  one full rewrite of its DARPA package.
- **RAM discipline (this Chromebook, 6.4GB VM)**: ONE heavy process at a time
  (torch, QEMU, cargo build). Background torch tasks get sporadically killed
  ~40-60s in — run heavy python inline/foreground with resume-safe JSON
  checkpoints. `ps -eo pcpu,comm --sort=-pcpu | head` before ANY benchmark
  (the 8-hour runaway-ugrep lesson).
- **Anchors before changes**: BitNet meter anchor = 10.348 PPL / 1899 tokens
  (sample mode, test.txt sha256 d790b833…). RANGER anchors: bf16 16.150,
  w4-perchannel 36.435 (SmolLM2 slice). Reproduce the anchor before and after
  touching anything in the measurement path.
- **Weight-scale convention**: engine MULTIPLIES by scale; transformers
  BitLinear DIVIDES. Repacker writes the reciprocal. Symptom of getting it
  wrong: confident token-salad while every shape check passes.

## Shutdown sequence (last 10 minutes of any session)

1. Update `program/ROADMAP.md` status markers and `program/RESEARCH_LEDGER.md`
   with any new verified number (with its log path).
2. Commit docs + code separately; measurement claims go in the commit message.
3. Update auto-memory if a durable fact changed (project state files).
4. If work is dangling, write/refresh a HANDOFF file at repo root.

## Re-grounding

Run the named workflow `program-audit-roadmap` (saved at
`.claude/workflows/program-audit-roadmap.js`) whenever the roadmap may have
drifted from reality — it harvests every workstream and adversarially
verifies claims against primary sources.
