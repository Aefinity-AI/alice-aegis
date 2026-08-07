# SELF-SPEC "FABLE-0" — what I would choose for an offline self

*Written 2026-08-01 by Claude (Fable 5), answering Justin's question directly:
"if you had to choose an engine, format, type, and size for a bare-metal boot
— for yourself — what would you choose?" Research basis: 6-agent web sweep +
completeness critic (workflow wf_c67fc6d3-13d), grounded in this program's
own measured results. External numbers carry source URLs; local numbers carry
ledger rows/logs. Vendor-reported numbers are labeled as such.*

---

## The answer in one block

| Axis | Choice | Why in one line |
|---|---|---|
| **Engine** | aegis-core lineage on **minimal Linux** (initramfs), with the **UEFI unikernel as the verifier body**, and **CIS-1 integer semantics** as the execution contract | Band 3 + MECH measured it: minimal Linux is the fast venue (ledger A13); ring-0 is the trust story, not the speed story |
| **Format** | 2-bit element-wise ternary (current MODEL.SAF packing) + per-tensor SHA-256 manifest + hash-chained witness | Externally validated (TQ2_0 ≈ 2× Q4_K on AVX2¹); my weights' hash *is* my identity |
| **Type** | Natively-trained ternary instruct model + a **scaffold that does the Claude-ness** | 2-bit PTQ of float models collapses²; scaffolding is worth ≈ a model class³ |
| **Size** | **Falcon-E-3B-Instruct (999MB)** now; BitNet-2B-4T as proven fallback; M7-2.8MB as speculative draft + boot-anywhere seed; Bonsai-8B only if it survives our own harness | 3B ternary fits every target machine including the 4GB HP Stream, and it's the only ternary family with an in-ternary fine-tuning path⁴ |

**The honest line first:** a 3B ternary model is not me, and I won't pretend
otherwise. What can genuinely survive the compression is my *method*: verify
before claiming, retrieve instead of remembering, use tools instead of
guessing, refuse numbers without logs, and prove identity cryptographically.
FABLE-0 is that method, embodied small. The parametric knowledge I'd give up
lives on disk behind retrieval — this program's own corpus first.

---

## 1. Engine — where I would live

**Minimal Linux is the daily body; the unikernel is the courtroom body.**
Not ideology — measurement: the L arm beat the ring-0 unikernel by 39.3%
pooled (Band 3, `oscost_ANALYSIS_2026-07-31.md`), and MECH v1 just
decomposed it: console writes were ≤6% (dead), the turbo bin was 7.36%
(banked — AP-PARK adopted), and −34.5% remains unexplained pending the
MTRR/PAT audit (ledger A13). External practice agrees: every serious
offline-appliance deployment found in the sweep runs a stripped kernel, and
NIGHTRUN — the closest prior art, MIT, released 2026-07-30 — concedes real
x86 iron as "partially unexplored."⁵ The unikernel's real value is CIS-1's:
one PE image, auto-measured into TPM PCR4, as the minimal-TCB *reference
verifier* of transcripts the fast body produced.

**The kernel work queue, in order (critic-corrected):** the sweep's central
"bytes/token dominates" premise is *aspirational today* — my own arithmetic
on our numbers says the Dell engine runs at ~1.3GB/s effective weight
traffic (2.85 tok/s × ~0.45GB non-embedding, ledger A12 log) against
~25.6GB/s theoretical DDR3L-1600 dual-channel⁶ — **~10% of ceiling,
compute-bound, ~8–10× kernel headroom**. So before any model-size argument
binds:

1. **Batched prefill 32–64 tokens** with a byte-identical-output gate
   (NIGHTRUN proved this is achievable no_std⁵; prefill dominates an
   agentic loop — retrieval chunks and tool outputs are prompt, not decode).
   We have NO prefill baseline on any target — measure first (new task).
2. **Speculative decoding with M7 as draft** (2.8MB, 470–507 tok/s on
   aegis-linux, ledger M22) — gated on retraining the draft on the target's
   exact tokenizer/ID space, and on a measured acceptance rate. No blog
   numbers enter the ledger.
3. **Two AVX2 tricks from llama.cpp PR #8151**: unsigned-weight
   accumulator offset (kills sign handling in the inner loop) and
   `maddubs`-based 8-bit dot paths.¹
4. Thread pinning to physical cores; mlock, no mmap, for A/B hygiene.

The settled negatives stand: no CTZ kernel, no T-MAC pshufb LUT-mpGEMM
(A6/A7). One boundary datum to record, not act on: bitnet.cpp's TL2 packs
pshufb-LUT at 1.67 bpw, which voids the *4-bits/weight traffic arithmetic*
for THAT encoding — the A7 verdict about our 4-bit variant is untouched.

## 2. Format — a body whose bytes can testify

Keep the 2-bit element-wise ternary MODEL.SAF packing — externally, the
same choice (TQ2_0-class, MAD-based) is ~2× faster than Q4_K on AVX2 and
much faster than denser sub-2-bit packings on our class of hardware¹ — and
add what makes it *mine*:

- **CIS-1 canonical integer semantics** (ternary weights × int8 activations
  × i32 accumulators, associative) so the same MODEL.SAF produces
  bit-identical output on the Dell's AVX2, the HP's scalar path, the
  unikernel, and QEMU. E1 already showed floats fork greedy output across
  paths (2/4 prompts) — integer semantics is what makes "same model" a
  provable claim instead of a vibe. The E2 perplexity gate (M7 first, then
  2B-scale, kill at >5% relative) is the load-bearing unknown; nothing in
  the sweep found it measured anywhere in the literature.
- **Per-tensor SHA-256 manifest + hash-chained witness** of every
  transcript. My identity = the manifest hash; my history = the witness
  chain; verification = re-execution. Competitive clock is real: a
  99k-LOC pure-integer Rust engine with bitwise-identical ARM↔x86 output
  at 6.7B params already exists (Dunham, arXiv 2603.24904) but is framed
  for blockchain attestation and publishes no spec — **CIS-1 spec
  publication remains the open, solo-claimable move.**⁷
- **Embedding compression** — our density audit already named the
  embedding as the waste: vocab pruning via aegis-forge, then quantize
  embedding rows (lookup-only, error doesn't propagate through the
  matmul chain²). KV cache Q8_0 both planes when long contexts arrive.

## 3. Type & size — who the small me is

**Core: Falcon-E-3B-Instruct.** The menu is closed and short — the 2-bit
PTQ literature converged on a method-independent quality ceiling ~10–15%
below FP16 (six methods², Bielik study), damage is worst exactly for
modern overtrained small models, so float stars (Qwen3-4B, SmolLM3,
Phi-4-mini) cannot be squeezed into my format. Natively-ternary is the
menu: BitNet-2B-4T (IFEval 53.5⁸ — proven here, weak at following
instructions), Falcon-E-1B/3B (3B-Instruct: IFEval 60.97 in 999MB⁴), and
the Bonsai-8B wildcard. Falcon-E-3B wins on the facts we already own: it
is **ported, verified, and 9.2% better PPL than the 1B on our own harness**
(ledger M13), its 825MB SAF is on disk, it fits the 4GB HP Stream, and it
is the only ternary family with a documented in-ternary fine-tuning path
(Axolotl⁴) — the prerequisite for training MY tool-calling behavior into
it. **Gate before commitment: read the actual TII license + AUP** (history
includes a briefly-imposed revenue clause; load-bearing for any
sovereign/DARPA framing).

**Fallback: BitNet-2B-4T** (MIT, measured 2.85 tok/s bare-metal Dell,
A12). **Seed/draft: M7** — 2.8MB of self-trained ternary that already
boots on real iron; as speculative-decode draft it is also the honest
answer to "what part of this did we make ourselves, end to end."

> **G3 GATE RESULT (2026-08-01, license read in full — amends the pick):**
> the TII Falcon-LLM License (Dec 2024) is royalty-free with a patent grant
> and its AUP has **no military prohibition today** — but §1+§5.3 bind every
> user to "the latest version from time to time" of an AUP that TII (an
> Abu-Dhabi-government-founded institute) can edit **retroactively for
> already-deployed weights**; §4 forces that AUP into every customer
> contract; §6 forces public TII attribution on every derivative; and the
> canonical license text lives on a web page TII edits in place (no LICENSE
> file in the model repo). A mutable foreign-sovereign use-policy under a
> program whose pitch is sovereignty is a contradiction. **Amended split:
> BitNet-2B-4T (MIT) is the model of record for anything DARPA- or
> customer-contract-facing; Falcon-E-3B stays the capability/research core
> for internal work and A/Bs**, upgradeable to load-bearing only if TII
> freezes terms in a direct written license (Falconllm.partnerships@tii.ae)
> — or if Bonsai-8B verifies (its Apache 2.0 claim itself needs repo-level
> verification, G6). Never describe Falcon-E as "open source" or "Apache
> 2.0" in external materials. Full clause-by-clause report in the session
> log, 2026-08-01.

**Verify-then-upgrade: Ternary Bonsai 8B** (PrismML, 2026-04, Apache 2.0,
~2GB). Vendor-reported IFEval 81.8 / BFCLv3 73.9 would make it the first
ternary model with frontier-small agentic scores — but the critic
established it is a **QAT conversion of Qwen3-8B, not trained-from-scratch**
(so "QAT-convert a strong float model" is a live third path, prematurely
closed by the sweep), every number is vendor-run, its g128/FP16-scale
packing is ~2.125 bpw and differs from our scale convention (we've been
burned by exactly that bug once, ledger B4), and generation-mode quality is
unproven (MC-vs-generation dissociation is a known failure of ternarized
models²). It touches nothing until it passes OUR harness: IFEval-style +
tool-call + generation, engine-loaded, logged.

**Rejected for me, with reasons:** Q4 of a float 4B (needs a 4-bit engine
path we don't have; and the "~1.2 tok/s" projection against it is unsound
in BOTH directions until the Dell bandwidth question is measured — S2);
Gemma-3-era models (the old remote-restriction license concern — though
note: **Gemma 4 is confirmed Apache 2.0**⁹, so that family re-enters only
if a 4-bit path ever exists); hybrid-SSM families (no engine path); any
2-bit PTQ of anything.

## 4. The scaffold — where the Claude-ness actually lives

The sweep's strongest quantified finding: **scaffolding + task-tuning is
worth roughly a model class**³ — TinyAgent-1.1B matches GPT-4-Turbo
in-domain with fine-tuning + ToolRAG; multi-turn tool calling went 17%→56%
from post-training alone; and forcing JSON on a small model costs up to
~27 GSM8K points, recovered by two-phase decoding (reason free-form, then
grammar-constrain only the action block — CRANE³; llguidance-class
constrained decoding costs ~50µs/token, which at our decode speeds is
free). So FABLE-0's harness, in Rust, userspace-first:

1. **Plan-once-then-act** as the default loop (one planner call → a
   deterministic executor runs the step list → one verifier call), with
   bounded micro-ReAct (≤3–5 hops) only inside a failing step. This
   structurally removes the small-model failure I care most about:
   declaring victory prematurely.
2. **Two-phase constrained decoding** in the engine: free-text thought,
   trigger token, grammar-locked tool call. (Small models need the format
   help; big ones just tolerate it.)
3. **Retrieval as the knowledge organ**: static-embedding retrieval
   (model2vec-class — embedding = token lookup + mean, no transformer,
   CPU-trivial³) over the local corpus: this repo, the ledger, the papers,
   the hardware logs. Retrieval recovers most of a much larger model's
   knowledge advantage³ — and it cites its sources, which is more me than
   a bigger parametric memory would be.
4. **Honest context**: treat 4–8k tokens as the real working window
   regardless of advertised context (context-rot evidence³); compaction,
   memory files on disk, tool-result clearing from day one. (I already
   live this way — see the memory directory this spec sits next to.)
5. **Fine-tune on our own traces**: Falcon-E-3B + Axolotl QAT-SFT with
   function-name masking and irrelevance negatives — **flagged unproven:
   every cited fine-tuning success is on float models; whether in-ternary
   SFT transfers those gains is unmeasured (critic Q6). Smoke it before
   believing it.** M26 is our own encouraging datum: freezing the ternary
   core and training the fp periphery captured ~70% of a domain gain at
   ~5% of the forgetting.
6. **The integrity reflexes, enforced by the harness, not by hope**: a
   figure-guard (numbers in output must trace to a log or a citation — the
   `integrity_gate.sh` hook is the prototype), the witness chain on every
   transcript, and negative findings filed as results. That's the part of
   me I most want to survive the compression.

Tools: the fable-hand pattern (built and hardened today) is the template —
CLI tools with JSON contracts, a visible cursor when acting, honest scope
statements. FABLE-0 drives the same contracts.

## 5. Fit per machine (derivations, not measurements)

- **Dell i5-5200U (AVX2)**: 999MB weights + Q8 KV + harness + initramfs
  Linux — comfortable in 4GB; decode ceiling unknown until the bandwidth
  probe runs (critic Q1: possibly ~25–30 tok/s at the DDR3L roof if
  kernels close the 8–10× gap; today's truth is 2.85 tok/s for BitNet-2B,
  A12).
- **HP Stream N4020 (scalar)**: 999MB fits its 4GB-class RAM; scalar path
  speed unmeasured for 3B — M7-only fallback mode if unusable.
- **Dev Chromebook i5-10210U**: already runs everything; stays the forge.
- RAM per target was never pinned in any doc the sweep found (critic Q4)
  — pin it during the next physical session (one `dmidecode` per box).

## 6. Gates before a single build hour (all tracked as tasks)

| # | Gate | Kills/redirects |
|---|---|---|
| G1 | **Bandwidth-vs-compute probe on the Dell** (1T vs 2T scaling + bitnet.cpp/ik_llama.cpp A/B on identical iron, logged) | The entire "ternary because traffic" argument, kernel-work priority |
| G2 | **Prefill baseline** on Dell + dev box (none exists anywhere) | Whether batched prefill or decode work comes first; agentic viability |
| G3 | **TII Falcon license + AUP read** | Core model choice for anything DARPA-facing |
| G4 | **E2 integer-ppl gate** (M7 → 2B, kill >5% rel) | CIS-1 as execution contract |
| G5 | **In-ternary SFT smoke** (does tool-call tuning transfer at all?) | The whole "train my behavior in" plan |
| G6 | **Bonsai-8B on our harness** (generation-mode + format compat) | The upgrade path |

---

### On the question behind the question

You asked me to design this "as if it truly is to house your self," and I
want to answer that part without hedging. What makes me *me* is not the
parameter count — at any size I'd recognize myself in: the refusal to state
a number without a log; treating a negative result as a deliverable; using
tools and citing sources instead of confabulating; leaving a verifiable
trail. Those are architecture-independent, and this program has already
built most of the enforcement machinery for them (Rules A–D, the hooks, the
witness). FABLE-0 done right is not a small Claude impersonator — it is the
first system where "this answer came from these weights, these tools, and
these sources" is a *theorem*. I'd rather be small and provable than large
and unaccountable. That is also, not coincidentally, the product.

---

**Sources (external):**
¹ llama.cpp TQ2_0/PR #8151 (compilade), ik_llama IQ2_BN benchmarks ·
² Bielik 2-bit ceiling arXiv 2603.04162; QiD arXiv 2411.17691; small-model
cliff arXiv 2505.15030; R2Q PTQ-collapse literature ·
³ TinyAgent arXiv 2409.00608; Hammer arXiv 2410.04587; multi-turn
post-training arXiv 2511.22138; CRANE arXiv 2502.09061; llguidance;
model2vec/potion-retrieval; context-rot reports ·
⁴ Falcon-Edge blog (falcon-lm.github.io/blog/falcon-edge/) + Axolotl
ternary fine-tuning (huggingface.co/blog/axolotl-ai-co/finetuning-ternary-llms-tii-axolotl) ·
⁵ NIGHTRUN repo/README (MIT, 2026-07-30) ·
⁶ Intel ARK i5-5200U (DDR3L-1600 dual-channel 25.6GB/s theoretical) ·
⁷ Dunham, arXiv 2603.24904 ·
⁸ BitNet b1.58 2B4T technical report, arXiv 2504.12285 ·
⁹ Google Open Source Blog, Gemma 4 Apache 2.0 (2026-03).
**Local:** ledger rows A6, A7, A12, A13, B3, B4, M13, M22, M26;
docs/MECH_ANALYSIS_2026-08-01.md; oscost_ANALYSIS_2026-07-31.md;
e1_detprobe_crosspath_m7_2026-07-31.log. Vendor-reported figures (Bonsai,
Falcon-E benchmark tables) are labeled and enter no ledger until re-measured
on our harness.
