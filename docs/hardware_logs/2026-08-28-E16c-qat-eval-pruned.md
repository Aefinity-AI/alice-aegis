CIS-1 on QAT fine-tuned BitNet-2B (E16) — APPLES-TO-APPLES re-forge with A35's pruned-vocab contract (--llama3-prune) — 2026-08-28
Machine: dev Chromebook i5-10210U crosvm — QUALITY/IDENTITY ONLY (Rule A: any wall-time below is incidental, not a result).
Follow-up to E16b (full-vocab, docs/hardware_logs/2026-08-28-E16b-qat-eval.md), which used repack_ternary.py's default full_vocab_id_space and so was not directly digest/vocab-comparable to A35/A36 (which used the pruned 50,256-token Llama3-8B-1.58 id space). This run re-forges the SAME E16a checkpoint (~/projects/kaggle-kernels/e16-bitnet-qat-ft/out/e16_ckpt/) with `--llama3-prune` added, into a separate artifacts_pruned/ dir (E16b's full-vocab artifacts untouched).
Command: python3 aegis-forge/repack_ternary.py ~/projects/kaggle-kernels/e16-bitnet-qat-ft/out/e16_ckpt ~/projects/kaggle-kernels/e16-bitnet-qat-ft/artifacts_pruned --max-seq 512 --llama3-prune
Binaries: aegis-eval, cis_decode, cis_witness release + standalone cis-verify, same worktree/build as E16b (alice-aegis-e16b, branch cm/e16b-qat-eval).

== forge output identities (pruned) ==
c5075aa121d6b79f49923ed19bc33449f75596b7e1f6d3f52fd67aa366e03f89  MODEL.SAF
4f9f78aa4155b06a33951c5a3cdca59f359b08ed386e00d03cbe5c92875dbc24  EMBED.BIN
5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae  VOCAB.BIN  <- BYTE-IDENTICAL to A35/A21's VOCAB.BIN hash (5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae). Confirms the pruning id-space reproduction is exact, matching A35's forge convention.
d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0  test.txt (identical to A21/A35/E16b)
VOCAB.BIN: 50256 tokens, 110042 merges (matches A35's "50,256 of 128,256" pruned-vocab description).

== --cis-full transcript, RUN 1 ==
==================================================
 A.L.I.C.E. Perplexity Evaluator (measured)
==================================================
Engine online. SIMD level: AVX2+FMA
Dataset: test.txt | sample: 200 tokens
--------------------------------------------------
CIS-1 v0.3 — float vs hybrid-int vs FULL-INTEGER, teacher-forced
float      PPL (199 scored tokens): 31.016864   [56.5s]
hybrid-int PPL (199 scored tokens): 31.117867   [328.5s]  delta vs float: +0.3256%
full-int   PPL (199 scored tokens): 30.974325   [338.7s]  delta vs float: -0.1371%
kill line (full-int vs float): +5.0%
FULL-INT VERDICT: PASS (within the +5% line)
hybrid-int argmax digest (FNV-1a 64): 0xE2692DEE9A73DB58
full-int   argmax digest (FNV-1a 64): 0xA02F063B67F7CDE7
--------------------------------------------------
NOTE: pruned-vocab model (50,256 of 128,256 tokens; ASCII-oriented).
Report this number only alongside that caveat.

== --cis-full transcript, RUN 2 ==
==================================================
 A.L.I.C.E. Perplexity Evaluator (measured)
==================================================
Engine online. SIMD level: AVX2+FMA
Dataset: test.txt | sample: 200 tokens
--------------------------------------------------
CIS-1 v0.3 — float vs hybrid-int vs FULL-INTEGER, teacher-forced
float      PPL (199 scored tokens): 31.016864   [54.8s]
hybrid-int PPL (199 scored tokens): 31.117867   [327.2s]  delta vs float: +0.3256%
full-int   PPL (199 scored tokens): 30.974325   [335.3s]  delta vs float: -0.1371%
kill line (full-int vs float): +5.0%
FULL-INT VERDICT: PASS (within the +5% line)
hybrid-int argmax digest (FNV-1a 64): 0xE2692DEE9A73DB58
full-int   argmax digest (FNV-1a 64): 0xA02F063B67F7CDE7
--------------------------------------------------
NOTE: pruned-vocab model (50,256 of 128,256 tokens; ASCII-oriented).
Report this number only alongside that caveat.

== cis_decode run1 ==
model : 30 layers, hidden 2560, vocab 50256
prompt: "Once upon a time" -> 4 tokens
token ids: [11, 304, 264, 2678, 6424, 11, 1070, 574, 264, 1912, 315, 4885, 889, 10456, 311, 1514, 3871, 13, 2435, 1051, 2744, 3411, 369, 502, 3953, 311, 1514, 13, 3861, 1938, 11, 814, 6773, 311, 1514, 264, 1847, 315, 4877, 13, 2435, 18255, 5694, 1139, 1403, 7411, 323, 3940, 4401, 2212, 279, 6246, 382, 2170, 814, 1051, 5737, 11, 814, 14000, 430, 832, 315, 872]
text     : ", in a small town, there was a group of friends who loved to play together. They were always looking for new games to play. One day, they decided to play a game of tag. They divided themselves into two teams and started running around the park.\n\nAs they were playing, they noticed that one of their"
CIS_DECODE digest=1462def76f90c282 prompt_toks=4 gen_toks=64 mode=fullint

== cis_decode run2 ==
model : 30 layers, hidden 2560, vocab 50256
prompt: "Once upon a time" -> 4 tokens
token ids: [11, 304, 264, 2678, 6424, 11, 1070, 574, 264, 1912, 315, 4885, 889, 10456, 311, 1514, 3871, 13, 2435, 1051, 2744, 3411, 369, 502, 3953, 311, 1514, 13, 3861, 1938, 11, 814, 6773, 311, 1514, 264, 1847, 315, 4877, 13, 2435, 18255, 5694, 1139, 1403, 7411, 323, 3940, 4401, 2212, 279, 6246, 382, 2170, 814, 1051, 5737, 11, 814, 14000, 430, 832, 315, 872]
text     : ", in a small town, there was a group of friends who loved to play together. They were always looking for new games to play. One day, they decided to play a game of tag. They divided themselves into two teams and started running around the park.\n\nAs they were playing, they noticed that one of their"
CIS_DECODE digest=1462def76f90c282 prompt_toks=4 gen_toks=64 mode=fullint

== receipt mint ==
AEGIS-WITNESS v1-CIS
model c5075aa121d6b79f49923ed19bc33449f75596b7e1f6d3f52fd67aa366e03f89
embed 4f9f78aa4155b06a33951c5a3cdca59f359b08ed386e00d03cbe5c92875dbc24
vocab 5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae
maxtok 64
prompt-hex 4f6e63652075706f6e20612074696d65
prompt-toks 4
gen-toks 64
token-ids 11,304,264,2678,6424,11,1070,574,264,1912,315,4885,889,10456,311,1514,3871,13,2435,1051,2744,3411,369,502,3953,311,1514,13,3861,1938,11,814,6773,311,1514,264,1847,315,4877,13,2435,18255,5694,1139,1403,7411,323,3940,4401,2212,279,6246,382,2170,814,1051,5737,11,814,14000,430,832,315,872
cis-digest 1462def76f90c282
chain 0059bd18ad2ad5864a6d11302b83a7b7eef1bf600855a33a66de94f450461b9e

== cis_witness verify ==
receipt cis-digest 1462def76f90c282 chain 0059bd18ad2ad586
local   cis-digest 1462def76f90c282 chain 0059bd18ad2ad586
VERIFY PASS — replay reproduced 64 tokens, the token digest, and the full logit chain bit-for-bit

== standalone cis-verify ==
cis-verify: receipt=tests/golden/witness_v1_e16_qat_pruned_once64.receipt
cis-verify: MODEL.SAF=/home/justinbrianthompson/projects/kaggle-kernels/e16-bitnet-qat-ft/artifacts_pruned/MODEL.SAF (521953185 bytes)
cis-verify: EMBED.BIN=/home/justinbrianthompson/projects/kaggle-kernels/e16-bitnet-qat-ft/artifacts_pruned/EMBED.BIN (257310720 bytes)
cis-verify: VOCAB.BIN=/home/justinbrianthompson/projects/kaggle-kernels/e16-bitnet-qat-ft/artifacts_pruned/VOCAB.BIN (1759936 bytes)
check: receipt parse ......... ok
check: artifact hashes ........ ok
check: prompt tokenization ..... ok
check: token-id sequence (64 steps) ok
check: cis-digest (FNV-1a 64) .. ok
check: witness chain (SHA-256) . ok
VERIFY PASS

== comparison table (pruned-vocab, apples-to-apples with A35; full-vocab E16b kept as secondary) ==
               A35 (base, pruned)   E16c (QAT, pruned)   E16b (QAT, full-vocab, secondary)
float PPL      30.706665            31.016864            26.126337
hybrid PPL     30.934140            31.117867             26.169092
full-int PPL   30.744724            30.974325             26.195775
full-int vs float delta: A35 +0.1239% | E16c -0.1371% | E16b (full-vocab) +0.2658% -- all PASS, well inside +5%.
E16c vs A35 (apples-to-apples, same 50,256-token pruned vocab, same test.txt): PPL got very slightly WORSE after fine-tuning (+~1.0% on float: 30.71->31.02), the OPPOSITE direction of E16b's full-vocab comparison (~15% BETTER). This is the correct, comparable number; E16b's full-vocab PPL improvement was an artifact of using a different (larger, differently-tokenized) vocab, NOT evidence the fine-tune improved quality under A35's exact evaluation contract.
digests: hybrid 0xE2692DEE9A73DB58, full-int 0xA02F063B67F7CDE7, both runs identical (run1==run2). Differ from A35's digests as expected (different weights). cis_decode digest 1462def76f90c282, identical run1==run2 -- SAME digest as E16b's full-vocab decode (coincidental: all 64 generated token ids for this prompt happen to fall within the pruned 50,256 id space, so decode output is unaffected by the vocab truncation for this specific prompt).
receipt: tests/golden/witness_v1_e16_qat_pruned_once64.receipt, chain 0059bd18ad2ad586..., VERIFY PASS via both cis_witness and standalone cis-verify.

PLAIN-LANGUAGE VERDICT (supersedes E16b's PPL-improvement claim):
Under the SAME evaluation contract A35 used (pruned 50,256-token vocab, identical VOCAB.BIN hash), 600 steps of smoltalk QAT fine-tuning did NOT improve CIS-1 held-out PPL -- it got marginally worse (float 30.706665 -> 31.016864, ~+1.0%). E16b's reported ~15% improvement was real but was measuring a DIFFERENT, non-comparable artifact (full 128,256-token vocab, different tokenizer coverage), not the effect of fine-tuning alone. The receipt/CIS-1 pipeline again worked completely unchanged (no forge or engine code changes) on this pruned re-forge: full-int PPL stayed inside the +5% kill line, two --cis-full runs and two cis_decode runs were digest-identical, and the receipt verified bit-for-bit via both cis_witness and cis-verify. Caveats as A35: pruned vocab, 200-token window, teacher-forced, Rule A (no timing).

== updated drafted ledger row (supersedes A43 as drafted in the E16b log; NOT applied to RESEARCH_LEDGER.md) ==
| A43 | **CIS-1 SURVIVES A QAT FINE-TUNING ROUND ON BitNet-2B, APPLES-TO-APPLES WITH A35 (2026-08-28): re-forged the SAME E16 checkpoint (Kaggle kernel aefinityaiinc/e16-bitnet-qat-ft, 600 smoltalk QAT steps) with repack_ternary.py's --llama3-prune flag exactly as A35 -- VOCAB.BIN is BYTE-IDENTICAL to A35's (5bde1b0355ef99c6875190ebfff081d985ca48977ae9269e3477f5cc2d97d9ae) -- and found full-int PPL +... (-0.1371% vs its own float; delta vs A35's float baseline: +~1.0%, i.e. fine-tuning did NOT improve CIS-1 PPL under A35's exact eval contract, unlike a naive full-vocab re-forge (see caveat) which showed a spurious ~15% "improvement" from vocab-coverage differences, not the fine-tune.** Same test.txt (sha256 d790b833... identical to A21/A35), same forge script/contract (unpacked ternary, 210 quantized tensors), same aegis-eval/cis_decode/cis_witness/cis-verify binaries, NO code changes. Two sequential --cis-full runs digest-identical (hybrid 0xE2692DEE9A73DB58, full-int 0xA02F063B67F7CDE7); cis_decode 64-token digest 1462def76f90c282 identical x2; receipt (tests/golden/witness_v1_e16_qat_pruned_once64.receipt, chain 0059bd18ad2ad586...) mints and verifies bit-for-bit via BOTH cis_witness and standalone cis-verify. A companion full-vocab re-forge (no --llama3-prune, docs/hardware_logs/2026-08-28-E16b-qat-eval.md) is kept as a secondary/non-comparable data point illustrating why vocab contract must match exactly before comparing PPL across fine-tuning runs. Dev host i5-10210U crosvm, QUALITY/IDENTITY ONLY (Rule A). | ✅ measured (2 runs + receipt, pruned + full-vocab secondary) | docs/hardware_logs/2026-08-28-E16c-qat-eval-pruned.md (primary), docs/hardware_logs/2026-08-28-E16b-qat-eval.md (secondary) |
