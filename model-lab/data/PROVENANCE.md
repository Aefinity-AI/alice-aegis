# Training-data provenance ledger (M2)

Rule (critic-amended, 2026-07-17): every subset carries BOTH a license AND a
generator-model column. LLM-generated data is permitted ONLY when third-party
published under an open license — the program never trains on frontier-API
output it generated itself (Anthropic ToS + DARPA provenance). Sovereignty
framing everywhere: **"own training run on openly licensed data."** NC or
unlicensed subsets are dropped, never waived. Mix stays TEXT-FORM; token
counts recorded for BOTH tokenizers.

**Counts updated 2026-07-21 from local parquets** (`m2_corpus_stats.json`,
unit m2-corpus-stats, 234s): rows EXACT (parquet metadata); token totals =
SAMPLED ESTIMATES (400 rows/subset, seed 42, mean×rows) under Falcon-E
(32,768) and Llama-3/BitNet (128,256). "F"/"L" = est. million tokens.
Metadata marked VERIFY-CARD is unconfirmed — check the dataset card before
mix freeze; never cite VERIFY-CARD rows as settled.

## SmolTalk (HuggingFaceTB/smoltalk, Apache-2.0 repo-level) — ~1,034M F total
("all" config = union of subsets: 1,098,865 rows, 1034.0 F / 906.4 L — do
not double-count)

| Subset | Rows | F(M) | L(M) | Generator | Decision |
|---|---|---|---|---|---|
| smol-magpie-ultra | 431,092 | 667.2 | 589.9 | Llama-3.1-405B-Instruct | KEEP only if Llama naming flow-down accepted (user/counsel) |
| smol-summarize | 101,428 | 54.1 | 46.8 | Qwen2.5-72B-Instruct (per card) | KEEP |
| smol-rewrite | 56,150 | 20.4 | 18.2 | Qwen2.5-72B-Instruct (per card) | KEEP |
| smol-constraints | 36,236 | 7.7 | 7.1 | Qwen2.5-72B-Instruct (per card) | KEEP |
| openhermes-100k | 100,000 | 42.9 | 37.0 | mixed GPT-4-era (OpenHermes-2.5) | DROP (source license undeclared) |
| numina-cot-100k | 111,734 | 63.3 | 56.0 | VERIFY-CARD (NuminaMath CoT) | INSPECT |
| apigen-80k | 87,521 | 54.5 | 46.6 | VERIFY-CARD (Salesforce APIGen) | INSPECT |
| self-oss-instruct | 50,661 | 18.0 | 15.0 | StarCoder2-15B self-generated | KEEP (verify license on card) |
| systemchats-30k | 35,930 | 21.6 | 18.8 | VERIFY-CARD (GPT-4-era) | INSPECT |
| metamathqa-50k | 50,000 | 13.2 | 11.1 | GPT-3.5-augmented (MetaMath) | INSPECT (MIT per upstream, verify) |
| explore-instruct-rewriting | 32,000 | 2.6 | 2.3 | VERIFY-CARD | INSPECT |
| longalign | 3,734 | 45.8 | 38.3 | VERIFY-CARD (THUDM LongAlign) | INSPECT |
| everyday-conversations | 2,379 | 0.4 | 0.4 | VERIFY-CARD (smollm-era) | KEEP-leaning (used in M5 smokes, discarded) |

## Tulu-3 SFT mixture (allenai/tulu-3-sft-mixture, ODC-BY repo-level; per-source licenses vary) — ~940M F total

| Source | Rows | F(M) | L(M) | Generator | Decision |
|---|---|---|---|---|---|
| tulu_v3.9_wildchat_100k | 100,000 | 330.5 | 233.2 | GPT-4/GPT-4-Turbo (user chats, WildChat) | INSPECT — OpenAI-generated; flag in provenance appendix |
| personahub_math_v5_regen_149960 | 149,960 | 225.6 | 192.9 | VERIFY-CARD (PersonaHub regen) | INSPECT |
| tulu_v3.9_aya_100k | 100,000 | 82.9 | 48.2 | HUMAN (Aya, multilingual) | KEEP (Apache-2.0 upstream, verify) |
| evol_codealpaca_heval_decontaminated | 107,276 | 70.2 | 57.4 | GPT-4-evolved (Evol-Instruct) | INSPECT |
| numinamath_tir_math_decontaminated | 64,312 | 51.7 | 44.2 | VERIFY-CARD | INSPECT |
| flan_v2_converted | 89,982 | 39.0 | 31.4 | HUMAN-curated academic NLP tasks | KEEP (verify FLAN licensing notes) |
| personas-math-grade | 49,980 | 25.6 | 21.8 | VERIFY-CARD | INSPECT |
| personahub_math_interm_algebra_20k | 20,000 | 23.4 | 19.8 | VERIFY-CARD | INSPECT |
| personahub_code_v2_34999 | 34,999 | 15.6 | 12.8 | VERIFY-CARD | INSPECT |
| sciriff_10k | 10,000 | 14.0 | 12.1 | VERIFY-CARD (SciRIFF) | INSPECT |
| wildjailbreak_decontaminated_50k | 50,000 | 13.8 | 12.4 | synthetic safety (AI2) | DROP-leaning (safety-tuning text skews an edge assistant; revisit) |
| open_math_2_gsm8k_50k | 50,000 | 13.1 | 10.7 | VERIFY-CARD (OpenMathInstruct-2) | INSPECT |
| personahub_ifdata_manual_seed_v3_29980 | 29,980 | 12.1 | 10.4 | VERIFY-CARD | INSPECT |
| wildguardmix synthetic 50k | 50,000 | 10.3 | 9.2 | synthetic safety (AI2) | DROP-leaning (as above) |
| no_robots_converted | 9,500 | 3.5 | 3.0 | HUMAN | **DROP — CC-BY-NC, hard exclusion** |
| oasst1_converted | 7,131 | 3.1 | 2.2 | HUMAN (OpenAssistant) | KEEP (Apache-2.0) |
| coconot_converted | 10,983 | 1.1 | 1.0 | VERIFY-CARD (AI2 noncompliance) | INSPECT |
| table_gpt_5k | 5,000 | 4.7 | 3.8 | VERIFY-CARD | INSPECT |
| tulu_hard_coded_repeated_10 | 240 | 0.0 | 0.0 | AI2 hand-written identity rows | DROP (identity text names Tulu/AI2) |

## Other corpora

| Corpus | License | Generator | Decision | Notes |
|---|---|---|---|---|
| TinyStories V2 (local, 2.2GB) | CDLA-Sharing-1.0 | GPT-4 (V2 files) | KEEP (M7/A-series pretrain; in use by twin now) | third-party published; cite generator openly |
| smoltalk2 | per-subset | per-subset | DEFERRED — scoped download only (87.7GB repo; ENOSPC lesson) | select SFT-branch subsets by name at mix time |
| Cosmopedia-v2 slice | Apache-2.0 (verify) | Mixtral-8x7B | DEFERRED | disk now ample (165GB) — no longer eviction-gated |
| WikiText-2 (test.txt) | CC-BY-SA | human | EVAL ONLY — never trained on | PPL anchor (ledger B9) |

## M6 sizing context

M6 needs only 1–5M tokens (70/30 domain/replay). The clean KEEP pool alone
(aya + flan + oasst1 + smol-trio ≈ 190M F-tokens) exceeds the replay need by
~50×; INSPECT rows are optional enrichment, not blockers. Resolve VERIFY-CARD
metadata only for subsets actually entering the frozen mix; freeze = exact
tokenization of the final mix (no sampling), hash, ledger row.

## M6 domain corpus (domain locked 2026-07-21: offline field operations assistant)

| Source | License | Generator | Decision | Notes |
|---|---|---|---|---|
| Ranger Handbook (TC 3-21.76) + related FMs/TCs (land nav, call for fire, battle drills) | PUBLIC DOMAIN (US Gov work, 17 USC §105) | HUMAN (US Army) | KEEP — pillar-2 domain text | fetch from official .mil/APD sources; record edition + URL per doc at mix freeze |
| Service creeds (Ranger/Soldier's/NCO/Sailor's/Airman's/Rifleman's) | PUBLIC DOMAIN (US Gov) | HUMAN | KEEP | tiny; include verbatim |
| Edge-systems/Linux technical text (pillar 1) | source-by-source at mix time | mixed | INSPECT at mix freeze | prefer PD/CC-BY sources; ledger rows added when selected |
