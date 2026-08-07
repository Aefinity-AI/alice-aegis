# M6 Domain Eval — design (DOMAIN LOCKED 2026-07-21; pre-registration = item freeze pending)

**DOMAIN (user decision 2026-07-21): "Offline field operations assistant"**
— two pillars, blended per user direction (display target: Hanover, NH —
CRREL/ERDC audience):

1. **Edge-systems technical assistance** — low-resource computing, hardware
   troubleshooting, Linux/boot/recovery without internet, field
   electronics/comms gear basics.
2. **Small-unit infantry field knowledge** — Ranger Handbook (TC 3-21.76)
   class doctrine: battle drills, patrol base ops, troop leading procedures,
   call-for-fire format, land navigation how-tos (azimuths, pace count,
   resection, terrain association), service creeds across branches.

**Provenance advantage:** pillar-2 corpus = US Government works (FMs/TCs/
creeds) → PUBLIC DOMAIN (17 USC §105), human-written — the cleanest rows
the provenance ledger will ever hold. Eval items are authored FRESH (about
the doctrine, never copied verbatim) → no contamination with any corpus.

**Item allocation (150):** 50 edge-systems / 100 field knowledge (≈40
tactics & battle drills, 25 land navigation, 20 call-for-fire & comms
procedures, 15 creeds & general military knowledge). Split 100 test / 50
dev as below. User = SME reviewer before freeze.

Per the critic amendment, this eval must be frozen, hashed into the ledger,
and baselined BEFORE any M6 training token is spent.

## 1. Purpose and gate

M6 fine-tunes BitNet-2B (vehicle fixed by ledger M17/M18) on 1–5M tokens of
domain + replay mix. This eval is the ONLY instrument that can declare M6 a
success. Gate (pre-registered, engine-side):

- PRIMARY: domain MC accuracy (raw) improves by **≥ +5.0 pts** over the
  BitNet-2B base model, measured ON THE ENGINE with the parity-gated `--mc`
  harness (ledger M15).
- TRIPWIRES (any one fails → rollback per MODEL_LAB M6 row): ARC-Easy n=570
  raw acc drops > 2.0 pts vs base; WT2 full-set PPL anchor (B9: 16.124)
  rises > 5%; coherence spot-check (10 fixed generation prompts) shows
  repetition/looping regression by inspection, logged.

## 2. Domain — USER DECISION REQUIRED (recommendation below)

| Option | What | Pro | Con |
|---|---|---|---|
| **A. Offline edge-systems technical assistant (RECOMMENDED)** | Low-resource computing, hardware troubleshooting, Linux/sysadmin-without-internet, field electronics, storage/boot/recovery procedures | Directly embodies the DARPA low-resource-computing story; the demo IS the pitch: an OS-less box answering the questions a field tech asks when nothing else works; eval items easy to author + verify factually | Broad-ish; needs careful item curation |
| B. Emergency/first-response field knowledge | First aid, triage, disaster comms procedures | Emotionally compelling demo | Liability/stakes framing in a federal review; harder to verify safely |
| C. ALICE self-support | The model explains/operates the very system it runs on | Smallest corpus, most self-referential demo | Too narrow to prove general specialization ability |

## 3. Instrument (fixed regardless of domain)

- **Part 1 — MC suite: 150 four-choice items**, authored fresh for this
  program (never scraped from existing benchmarks → no contamination
  concerns), split 100 held-out test / 50 dev. Same JSONL schema as
  ARC-Easy harness; scored by aegis-eval `--mc` (ids-based, acc + acc_norm,
  ties→lower). Items authored by Claude as TOOLING (permitted — items are
  an eval instrument, not training text), reviewed by user, published with
  the grant package.
- **Part 2 — generation spot-check: 10 fixed prompts** (domain tasks),
  greedy decode on engine, outputs archived per run; judged by documented
  human inspection against a 3-point rubric written BEFORE M6. No LLM
  judging in the gate (reproducibility).
- **Pre-registration:** sha256 of the frozen item files + this design doc
  recorded as a ledger row BEFORE M6 training starts. Items never edited
  after; dev items may inform mix design, test items touched only at gate
  time.
- **Baselines required pre-M6:** BitNet-2B base (needs pruned-vocab-aware
  mc_prep — 50,256-of-128,256 remap), Falcon-E-1B, Falcon-E-3B on the same
  suite (cheap: ~27k tokens × 3 engine runs, overnight class).

## 4. Training-mix implications (feeds M2 close-out)

70/30 domain/replay per MODEL_LAB. Domain text sources must be
openly-licensed and ledgered (PROVENANCE.md rules apply; no NC, generator
column mandatory). Replay = slice of the KEEP-decision SFT subsets
(m2_corpus_stats.json quantifies each). Llama-naming flow-down question only
triggers if Magpie-Ultra stays in the replay slice — decide at mix freeze.

*Draft by session 2026-07-21. On user's domain pick: author items → freeze →
hash → ledger row → baselines → THEN M6.*
