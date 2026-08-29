# C2b v3: BitNet-b1.58-2B-4T logit-distillation SFT on ARC-Easy/ARC-Challenge/
# OpenBookQA (openly-licensed train splits only), export ternary-packed
# safetensors via the E16a-proven contract for aegis-forge/repack_ternary.py.
#
# Pre-registration: ~/projects/claudius-maximus/state/reports/
#   2026-08-29-COMPACT-AXIS-PREREG.md (leg C2b). Gate is falsifiable and
# evaluated OFF-Kaggle (CIS-1 pipeline on penguin), not by this kernel.
#
# PRIVATE. Rule A: no timing numbers printed or recorded anywhere in this
# kernel's stdout or output artifacts.
#
# License audit (checked 2026-08-29 via HF datasets API):
#   allenai/ai2_arc            -> cc-by-sa-4.0   (ARC-Easy + ARC-Challenge)  INCLUDED
#   allenai/openbookqa "main"  -> Apache-2.0 (per task brief; HF card field
#                                  itself reports "unknown" but upstream
#                                  AI2 release is Apache-2.0)               INCLUDED
#   allenai/sciq               -> cc-by-nc-3.0 (NonCommercial)              EXCLUDED per brief
#
# Eval holdout (M16/C2a ARC-Easy n=570 ids, from
# alice-aegis/model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42.jsonl,
# which is drawn from allenai/ai2_arc "ARC-Easy" *validation* split, size
# 570 == exact match): this kernel NEVER loads the ARC-Easy validation or
# test splits at all (only "train"), and additionally asserts, defensively,
# that none of the embedded HOLDOUT_IDS appear in any id field of any
# dataset row actually used for training.
#
# ---------------------------------------------------------------------------
# v3 MEMORY REDESIGN (v1 and v2 both OOM'd on the first backward: fp32 master
# 2.4B params ~= 9.6 GB + fp32 grads ~= 9.6 GB cannot fit on one T4 no matter
# the optimizer). v3 splits into two phases:
#   Phase 1 (teacher-only): load the teacher on cuda:1, run every train +
#     held-out example ONCE under no_grad, cache the per-answer-token top-64
#     logit values (fp16) + indices (int32) to a .pt file, then free the
#     teacher entirely (del + empty_cache) before the student is ever loaded.
#   Phase 2 (student-only): load BitNet-2B-4T with device_map="balanced"
#     (accelerate) split across cuda:0+cuda:1, fp32 master, gradient
#     checkpointing, PagedAdamW8bit (Adafactor/AdamW fallback), embed_tokens+
#     lm_head frozen. KD loss = KL between the student's log-softmax
#     restricted+renormalized to the cached top-64 indices and the cached
#     (temperature-scaled) teacher distribution over those same 64 indices +
#     CE on gold. If step 0 still OOMs, freeze the bottom 15 of 30 transformer
#     blocks and retry once (recorded as C2B_TRAINABLE=...).
# ---------------------------------------------------------------------------
import gc, hashlib, json, math, os, subprocess, sys, time, traceback

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                     "transformers", "bitsandbytes", "safetensors", "datasets",
                     "huggingface_hub", "accelerate"])
print("pip rc", r.returncode, flush=True)
if r.returncode:
    sys.exit("PIP_FAILED")

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from safetensors.torch import save_file

print("ENV torch", torch.__version__, "| transformers", transformers.__version__,
      "| gpus", torch.cuda.device_count(), flush=True)

STUDENT = "microsoft/BitNet-b1.58-2B-4T-bf16"
TEACHER_CANDIDATES = [
    ("meta-llama/Llama-3.2-3B-Instruct", "meta_gated"),
    ("unsloth/Llama-3.2-3B-Instruct", "unsloth_ungated_mirror"),
]
MAX_STEPS = 1200          # ~4096 tok/step (batch1 x grad_accum8 x seq512) -> ~4.9M tokens, inside 1-5M budget
SEQ_LEN = 512
LR = 1e-4
GRAD_ACCUM = 8            # 512 * 8 = 4096 tok/step
WARMUP_STEPS = 40
ABORT_STEP = 300
SAVE_EVERY = 200
TEMP = 2.0
CE_WEIGHT = 1.0
KD_WEIGHT = 1.0
TOPK = 64                 # v3: cached teacher top-k logits per answer token
N_FREEZE_BLOCKS_FALLBACK = 15   # of 30, only if step 0 still OOMs after v3's other fixes
OUT_CKPT = "/kaggle/working/c2b_ckpt"
os.makedirs(OUT_CKPT, exist_ok=True)

PROJECTIONS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

# --- eval holdout ids (M16/C2a ARC-Easy n=570, ai2_arc "ARC-Easy" validation
# split) -- embedded so this kernel is self-contained and can assert
# exclusion without any extra Kaggle dataset upload. UNCHANGED from v1/v2. ---
HOLDOUT_IDS = json.loads(r'''["MCAS_2000_4_6", "Mercury_7057260", "ACTAAP_2014_7_6", "Mercury_7122448", "Mercury_SC_416516", "MCAS_2016_8_3", "Mercury_7030153", "AKDE&ED_2012_8_43", "NYSEDREGENTS_2014_8_10", "Mercury_SC_LBS10938", "ACTAAP_2007_7_19", "Mercury_176873", "Mercury_SC_LBS10170", "Mercury_400978", "Mercury_7185220", "Mercury_7213378", "Mercury_7241343", "TIMSS_1995_8_J5", "Mercury_7271583", "Mercury_412415", "Mercury_SC_400175", "Mercury_7006213", "NCEOGA_2013_5_28", "Mercury_SC_407569", "MCAS_1999_8_35", "ACTAAP_2009_5_14", "Mercury_7112788", "Mercury_7137463", "Mercury_7008435", "Mercury_SC_401208", "Mercury_7057925", "OHAT_2008_5_13", "Mercury_7013073", "ACTAAP_2013_5_16", "Mercury_SC_415396", "Mercury_7179638", "ACTAAP_2011_5_2", "Mercury_7249970", "Mercury_7092418", "MCAS_2012_8_23641", "MEA_2010_8_12", "Mercury_SC_401158", "Mercury_7091928", "Mercury_7112735", "Mercury_7040863", "Mercury_7268818", "VASoL_2007_3_33", "LEAP_2001_4_10239", "Mercury_SC_400840", "AIMS_2008_8_9", "Mercury_7263305", "TIMSS_2011_8_pg139", "Mercury_7173845", "Mercury_7239208", "Mercury_7064050", "Mercury_400540", "ACTAAP_2010_7_15", "Mercury_7137673", "MCAS_2013_5_29401", "NYSEDREGENTS_2014_4_24", "Mercury_7194425", "Mercury_7004165", "Mercury_7086205", "NYSEDREGENTS_2014_8_8", "Mercury_7041948", "Mercury_7092348", "VASoL_2009_5_25", "Mercury_7212713", "Mercury_SC_406661", "NCEOGA_2013_5_51", "Mercury_7163940", "Mercury_7164658", "Mercury_7197908", "Mercury_7080448", "Mercury_7130498", "MCAS_2013_5_17", "Mercury_7136623", "Mercury_7014508", "Mercury_SC_401652", "Mercury_7069003", "TIMSS_2007_4_pg110", "Mercury_183190", "Mercury_SC_416137", "Mercury_SC_401592", "Mercury_7032288", "Mercury_7007630", "NYSEDREGENTS_2014_8_25", "Mercury_182665", "Mercury_189018", "Mercury_SC_405304", "MCAS_2004_5_12", "MCAS_2004_9_16", "Mercury_7107363", "Mercury_7083965", "Mercury_SC_416156", "Mercury_7082688", "CSZ20680", "Mercury_7228568", "Mercury_7269220", "AKDE&ED_2012_4_35", "VASoL_2007_5_26", "TIMSS_2003_8_pg96", "Mercury_7119805", "Mercury_SC_402067", "Mercury_189770", "Mercury_SC_403016", "Mercury_7044730", "Mercury_7270025", "Mercury_7252683", "Mercury_7017080", "MCAS_2011_5_17662", "Mercury_7041580", "Mercury_7204050", "Mercury_7024378", "Mercury_7251755", "Mercury_SC_LBS10682", "NYSEDREGENTS_2014_8_42", "OHAT_2011_5_20", "NYSEDREGENTS_2014_8_24", "Mercury_7084123", "Mercury_7038098", "MDSA_2010_4_7", "Mercury_7175700", "Mercury_7206448", "Mercury_7007665", "Mercury_408929", "Mercury_7094080", "Mercury_7068495", "TIMSS_2007_4_pg81", "AKDE&ED_2008_4_12", "TIMSS_2011_4_pg72", "LEAP_2011_8_10435", "Mercury_SC_407391", "Mercury_SC_401340", "NYSEDREGENTS_2014_8_21", "Mercury_178728", "Mercury_416365", "Mercury_411424", "Mercury_7015715", "Mercury_7172218", "Mercury_7094185", "Mercury_7161053", "MCAS_2002_5_12", "Mercury_7248378", "Mercury_7134698", "Mercury_7213553", "NCEOGA_2013_5_15", "Mercury_SC_400370", "Mercury_SC_401137", "Mercury_7107030", "Mercury_7018200", "Mercury_7092138", "Mercury_7139685", "NYSEDREGENTS_2014_8_2", "Mercury_7233538", "Mercury_7236128", "Mercury_182683", "Mercury_7216755", "Mercury_7013090", "Mercury_7248273", "Mercury_7187215", "Mercury_7206500", "Mercury_7141348", "Mercury_7044783", "Mercury_SC_400002", "Mercury_7227973", "Mercury_7026863", "Mercury_7068565", "NYSEDREGENTS_2014_8_39", "Mercury_SC_415696", "Mercury_7227850", "Mercury_7228515", "AIMS_2008_8_8", "Mercury_SC_405215", "NYSEDREGENTS_2014_4_8", "Mercury_7013668", "MCAS_2011_8_17695", "Mercury_SC_416526", "VASoL_2011_5_3", "Mercury_7100713", "Mercury_7122553", "Mercury_7037590", "Mercury_SC_400061", "MDSA_2011_4_8", "MEAP_2005_8_43", "Mercury_184695", "LEAP_2011_8_10436", "Mercury_SC_400989", "VASoL_2010_5_39", "VASoL_2009_5_10", "Mercury_SC_405952", "Mercury_SC_401111", "Mercury_7203245", "Mercury_7042858", "Mercury_SC_415418", "Mercury_SC_400150", "Mercury_SC_413138", "Mercury_7114940", "MDSA_2008_5_41", "MCAS_2002_8_2", "MCAS_2016_5_7", "Mercury_7154490", "Mercury_7177520", "Mercury_7241273", "Mercury_7234185", "Mercury_7074970", "LEAP__7_10350", "NYSEDREGENTS_2014_8_27", "Mercury_7245928", "Mercury_7205380", "Mercury_7072415", "Mercury_7008943", "NYSEDREGENTS_2014_8_9", "Mercury_7220238", "ACTAAP_2010_7_11", "MEAP_2005_8_34", "Mercury_7010815", "Mercury_7270410", "Mercury_7168648", "Mercury_7082793", "Mercury_7270060", "Mercury_7085190", "AKDE&ED_2012_4_4", "Mercury_7213885", "Mercury_SC_415425", "Mercury_7142030", "Mercury_183733", "Mercury_7014070", "Mercury_SC_408452", "Mercury_178990", "Mercury_7027615", "Mercury_SC_LBS10920", "NYSEDREGENTS_2014_8_40", "MDSA_2010_8_1", "VASoL_2011_5_14", "Mercury_7168315", "Mercury_SC_400201", "NAEP_2000_4_S21+2", "Mercury_7227763", "TIMSS_2011_8_pg121", "Mercury_7212520", "Mercury_7168753", "NYSEDREGENTS_2014_8_3", "Mercury_7056508", "MCAS_2012_8_23647", "Mercury_7081813", "Mercury_7107433", "Mercury_7090528", "Mercury_7008400", "NYSEDREGENTS_2014_4_3", "Mercury_7176190", "Mercury_7135433", "Mercury_7030678", "MCAS_2003_5_35", "Mercury_SC_402280", "Mercury_7223475", "Mercury_SC_416172", "Mercury_7056945", "Mercury_SC_405887", "MCAS_2013_5_29396", "Mercury_7001785", "Mercury_7012775", "Mercury_7221883", "NYSEDREGENTS_2014_4_27", "Mercury_179533", "NYSEDREGENTS_2014_8_23", "Mercury_7110933", "Mercury_7227728", "Mercury_7220500", "NYSEDREGENTS_2014_4_11", "AKDE&ED_2012_8_18", "Mercury_SC_401146", "Mercury_7080553", "Mercury_7098193", "MCAS_2000_8_6", "Mercury_7004848", "Mercury_7056753", "VASoL_2008_3_26", "Mercury_7211085", "Mercury_7007963", "Mercury_7075233", "Mercury_175648", "MCAS_2005_8_30", "MDSA_2012_8_23", "Mercury_7099015", "AKDE&ED_2012_8_31", "Mercury_7235865", "NYSEDREGENTS_2014_4_10", "TIMSS_2007_8_pg96", "Mercury_7025165", "MCAS_2013_8_29415", "NYSEDREGENTS_2014_8_11", "Mercury_7029698", "Mercury_SC_408287", "MEAP_2005_5_27", "Mercury_7213448", "Mercury_7083388", "Mercury_7014455", "Mercury_SC_400179", "Mercury_7235708", "NYSEDREGENTS_2014_8_19", "Mercury_7242918", "Mercury_7205450", "TIMSS_2007_8_pg52", "Mercury_SC_415764", "NYSEDREGENTS_2014_8_4", "Mercury_SC_408040", "Mercury_184748", "Mercury_7176155", "Mercury_SC_400004", "Mercury_7172008", "MCAS_2003_5_26", "ACTAAP_2011_5_7", "Mercury_7068828", "Mercury_7158865", "Mercury_180128", "Mercury_7263463", "Mercury_7024518", "Mercury_7084193", "NCEOGA_2013_5_45", "Mercury_7213640", "Mercury_7032673", "CSZ_2007_5_CSZ10148", "Mercury_SC_400924", "MEA_2014_5_4", "Mercury_7027318", "Mercury_SC_400054", "TIMSS_2011_4_pg76", "Mercury_7034930", "Mercury_7064663", "Mercury_7100415", "NAEP_2009_8_S10+1", "Mercury_7223073", "NYSEDREGENTS_2014_8_6", "Mercury_SC_400037", "Mercury_185203", "ACTAAP_2011_5_10", "Mercury_SC_400303", "Mercury_7140088", "Mercury_7011025", "Mercury_7201915", "Mercury_SC_405442", "Mercury_7082740", "Mercury_SC_402278", "Mercury_SC_417580", "Mercury_SC_410870", "Mercury_7211978", "Mercury_7282030", "Mercury_7008803", "Mercury_190138", "Mercury_7173513", "Mercury_SC_400391", "Mercury_179393", "Mercury_SC_405526", "MCAS_1999_4_3", "Mercury_SC_400862", "Mercury_7203508", "Mercury_7016573", "Mercury_SC_415772", "Mercury_7057540", "Mercury_417570", "Mercury_7064523", "Mercury_SC_401367", "Mercury_7090650", "NCEOGA_2013_5_48", "MDSA_2010_4_13", "Mercury_7247905", "WASL_2005_5_11", "Mercury_7188265", "Mercury_401035", "Mercury_7213080", "Mercury_7233783", "MDSA_2008_5_39", "AKDE&ED_2008_4_22", "Mercury_7026600", "Mercury_7166740", "Mercury_7123480", "Mercury_SC_413454", "TIMSS_2011_4_pg90", "Mercury_7003710", "Mercury_SC_409254", "MDSA_2010_4_24", "TIMSS_2007_4_pg35", "Mercury_7064383", "MCAS_2013_8_29420", "Mercury_7248133", "Mercury_SC_413544", "Mercury_7041825", "Mercury_177380", "Mercury_7069528", "Mercury_SC_401245", "VASoL_2011_5_19", "Mercury_SC_400206", "Mercury_7215408", "Mercury_7205538", "WASL_2004_8_16", "Mercury_7210735", "Mercury_SC_400171", "Mercury_7084508", "Mercury_7027370", "MDSA_2011_5_6", "MCAS_1998_4_25", "Mercury_7014245", "Mercury_7271653", "MDSA_2011_4_36", "Mercury_7177940", "MCAS_2005_5_27", "TIMSS_2007_8_pg105", "Mercury_SC_401178", "Mercury_7186918", "Mercury_7245140", "NYSEDREGENTS_2014_8_5", "Mercury_7012968", "Mercury_7183225", "Mercury_SC_LBS10789", "Mercury_7080500", "Mercury_7007893", "Mercury_7163748", "Mercury_7222933", "VASoL_2010_3_5", "Mercury_SC_409390", "Mercury_7252648", "Mercury_7218558", "NYSEDREGENTS_2014_4_18", "NYSEDREGENTS_2014_8_22", "Mercury_SC_400049", "Mercury_7188475", "NYSEDREGENTS_2014_4_21", "Mercury_SC_400058", "Mercury_SC_406155", "Mercury_400813", "LEAP__5_10316", "MCAS_2009_8_4", "MDSA_2010_8_17", "Mercury_7170835", "Mercury_7107398", "ACTAAP_2011_5_12", "Mercury_SC_405721", "TIMSS_2003_4_pg29", "Mercury_7103338", "Mercury_7030590", "NYSEDREGENTS_2014_8_7", "Mercury_7269063", "MEA_2010_8_14-v1", "MCAS_2004_5_26", "Mercury_SC_401256", "Mercury_7180968", "Mercury_7083948", "Mercury_SC_401622", "Mercury_188773", "Mercury_7233363", "MCAS_2009_8_18", "Mercury_7212013", "Mercury_SC_LBS10689", "MCAS_2004_9_6", "AKDE&ED_2012_8_53", "Mercury_7032550", "Mercury_7172673", "Mercury_7016783", "Mercury_409399", "Mercury_7207253", "Mercury_7227990", "Mercury_SC_402253", "Mercury_SC_400194", "Mercury_7032568", "Mercury_SC_LBS10612", "Mercury_7008330", "Mercury_7108413", "Mercury_7176085", "Mercury_7043593", "Mercury_7033705", "Mercury_7114083", "Mercury_SC_LBS10785", "Mercury_7114013", "Mercury_192903", "Mercury_7082093", "Mercury_7004708", "Mercury_SC_LBS10783", "Mercury_7037433", "Mercury_SC_400033", "NYSEDREGENTS_2014_4_25", "Mercury_SC_402287", "Mercury_SC_414364", "Mercury_7250215", "Mercury_7138898", "Mercury_7010920", "Mercury_403682", "Mercury_7026828", "Mercury_7235760", "Mercury_406772", "Mercury_7217525", "Mercury_SC_405302", "Mercury_7032708", "Mercury_7082583", "Mercury_7245875", "Mercury_7122798", "VASoL_2008_3_39", "MSA_2015_8_26", "Mercury_177013", "Mercury_SC_400185", "Mercury_SC_402159", "VASoL_2009_5_31", "Mercury_SC_401598", "MCAS_2004_8_24", "Mercury_SC_413081", "CSZ_2008_5_CSZ50776", "MCAS_1998_4_22", "Mercury_7267785", "Mercury_SC_405035", "TIMSS_2003_4_pg85", "Mercury_7032200", "Mercury_SC_408374", "LEAP_2009_8_10431", "Mercury_7082460", "Mercury_7143150", "Mercury_7103373", "Mercury_SC_407720", "MCAS_2002_8_19", "Mercury_7009940", "Mercury_7239645", "Mercury_7026268", "Mercury_7057610", "VASoL_2009_5_14", "Mercury_SC_400589", "TIMSS_1995_8_I11", "MCAS_2000_8_30", "Mercury_7171553", "Mercury_7190015", "Mercury_7032830", "AIMS_2008_4_17", "Mercury_7271618", "NCEOGA_2013_8_46", "Mercury_7084560", "MCAS_2005_8_35", "Mercury_7202388", "Mercury_7210403", "Mercury_SC_401826", "Mercury_7269028", "Mercury_SC_407784", "Mercury_7042508", "Mercury_7271443", "TIMSS_2003_8_pg98", "Mercury_7029873", "Mercury_7228323", "Mercury_7217140", "Mercury_SC_400515", "MCAS_2003_8_13", "OHAT_2009_5_4", "ACTAAP_2012_7_16", "Mercury_7156800", "VASoL_2007_3_32", "Mercury_SC_416126", "Mercury_SC_416133", "Mercury_7254520", "Mercury_7037415", "Mercury_7221533", "Mercury_SC_410965", "TIMSS_2011_8_pg106", "Mercury_406736", "Mercury_SC_400691", "NYSEDREGENTS_2014_8_18", "Mercury_404901", "Mercury_7006265", "MCAS_2006_9_16", "MCAS_2006_9_9-v1", "MCAS_2011_8_17687", "Mercury_7214480", "Mercury_7094098", "Mercury_SC_401188", "Mercury_7193078"]''')
HOLDOUT_SHA256_SORTED = "c1e07b075f3d91cfb3169908ed24d58b4bb8fc7303c312d2348bef241869ef5c"
_check = hashlib.sha256(("\n".join(sorted(HOLDOUT_IDS))).encode()).hexdigest()
print(f"holdout ids: n={len(HOLDOUT_IDS)} sha256(sorted)={_check}", flush=True)
assert _check == HOLDOUT_SHA256_SORTED, "HOLDOUT_IDS embedding corrupted"
HOLDOUT_SET = set(HOLDOUT_IDS)

with open("/kaggle/working/eval_holdout_ids.json", "w") as f:
    json.dump({"source": "alice-aegis/model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42.jsonl",
               "n": len(HOLDOUT_IDS), "sha256_sorted": HOLDOUT_SHA256_SORTED,
               "ids": HOLDOUT_IDS}, f, indent=2)

tok = AutoTokenizer.from_pretrained(STUDENT)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---------------------------------------------------------------------------
# Data: ARC-Easy train, ARC-Challenge train, OpenBookQA train. Only "train"
# splits are ever requested -- ARC-Easy validation/test are never loaded.
# UNCHANGED from v1/v2.
# ---------------------------------------------------------------------------
LETTERS = ["A", "B", "C", "D", "E"]


def arc_to_examples(ds, tag):
    out = []
    dropped_holdout = 0
    for ex in ds:
        rid = ex.get("id", "")
        if rid in HOLDOUT_SET:
            dropped_holdout += 1
            continue
        choices = ex["choices"]
        texts, labels = choices["text"], choices["label"]
        answer_key = ex["answerKey"]
        if answer_key not in labels or len(texts) < 2:
            continue
        idx = labels.index(answer_key)
        lines = [f"Question: {ex['question']}", "Choices:"]
        for j, (lab, txt) in enumerate(zip(labels, texts)):
            lines.append(f"{LETTERS[j] if j < len(LETTERS) else lab}. {txt}")
        prompt = "\n".join(lines) + "\nAnswer:"
        answer_text = f" {LETTERS[idx] if idx < len(LETTERS) else answer_key}. {texts[idx]}"
        out.append({"prompt": prompt, "answer": answer_text, "id": rid, "src": tag})
    print(f"{tag}: kept={len(out)} dropped_holdout_id_matches={dropped_holdout}", flush=True)
    assert dropped_holdout == 0, f"{tag}: train split unexpectedly contained a holdout id -- refusing to continue"
    return out


def obqa_to_examples(ds, tag):
    out = []
    dropped_holdout = 0
    for ex in ds:
        rid = ex.get("id", "")
        if rid in HOLDOUT_SET:
            dropped_holdout += 1
            continue
        choices = ex["choices"]
        texts, labels = choices["text"], choices["label"]
        answer_key = ex["answerKey"]
        if answer_key not in labels:
            continue
        idx = labels.index(answer_key)
        lines = [f"Question: {ex['question_stem']}", "Choices:"]
        for j, (lab, txt) in enumerate(zip(labels, texts)):
            lines.append(f"{LETTERS[j] if j < len(LETTERS) else lab}. {txt}")
        prompt = "\n".join(lines) + "\nAnswer:"
        answer_text = f" {LETTERS[idx] if idx < len(LETTERS) else answer_key}. {texts[idx]}"
        out.append({"prompt": prompt, "answer": answer_text, "id": rid, "src": tag})
    print(f"{tag}: kept={len(out)} dropped_holdout_id_matches={dropped_holdout}", flush=True)
    assert dropped_holdout == 0, f"{tag}: train split unexpectedly contained a holdout id -- refusing to continue"
    return out


arc_easy_train = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
arc_chal_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
obqa_train = load_dataset("allenai/openbookqa", "main", split="train")

examples = (arc_to_examples(arc_easy_train, "arc_easy_train")
            + arc_to_examples(arc_chal_train, "arc_challenge_train")
            + obqa_to_examples(obqa_train, "openbookqa_train"))
assert not (set(e["id"] for e in examples) & HOLDOUT_SET), "post-merge holdout leak"
print(f"combined training pool: {len(examples)} examples "
      f"(arc_easy+arc_challenge cc-by-sa-4.0, openbookqa apache-2.0; sciq NC excluded)", flush=True)

import random
random.seed(42)
random.shuffle(examples)
HELD_N = min(100, max(20, len(examples) // 50))
held_out = examples[:HELD_N]
train_pool = examples[HELD_N:]
print(f"train_pool={len(train_pool)} held_out(NaN/plateau-check, disjoint from eval-holdout)={len(held_out)}", flush=True)


def encode_example(ex, seq_len):
    """Tokenize prompt+answer, build labels masked to -100 outside the answer
    span (so CE and KD losses below are both restricted to answer tokens,
    matching the pre-reg's 'KL...on answer tokens + CE on gold')."""
    prompt_ids = tok(ex["prompt"], add_special_tokens=True)["input_ids"]
    answer_ids = tok(ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:seq_len]
    n_prompt = min(len(prompt_ids), len(input_ids))
    labels = [-100] * n_prompt + input_ids[n_prompt:]
    labels = labels[:len(input_ids)]
    return input_ids, labels


def make_batch(exs, seq_len, device):
    enc = [encode_example(e, seq_len) for e in exs]
    maxlen = max(len(ii) for ii, _ in enc)
    pad_id = tok.pad_token_id
    input_ids, labels, attn = [], [], []
    for ii, lb in enc:
        padlen = maxlen - len(ii)
        input_ids.append(ii + [pad_id] * padlen)
        labels.append(lb + [-100] * padlen)
        attn.append([1] * len(ii) + [0] * padlen)
    return {"input_ids": torch.tensor(input_ids, device=device),
            "labels": torch.tensor(labels, device=device),
            "attention_mask": torch.tensor(attn, device=device)}


def eval_held_out(model, held, seq_len, device):
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for e in held:
            b = make_batch([e], seq_len, device)
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
            n_tok = int((b["labels"] != -100).sum().item())
            if n_tok == 0:
                continue
            total_loss += out.loss.item() * n_tok
            total_tok += n_tok
    model.train()
    avg = total_loss / max(total_tok, 1)
    return {"avg_loss": avg, "tokens": total_tok, "ppl": math.exp(min(avg, 20.0))}


# ---------------------------------------------------------------------------
# HF_TOKEN Kaggle secret (for the gated meta-llama attempt). UNCHANGED.
# ---------------------------------------------------------------------------
try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login as hf_login
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    hf_login(token=_hf_token)
    print("HF_TOKEN secret found and logged in", flush=True)
except Exception as e:
    print(f"no HF_TOKEN Kaggle secret available ({type(e).__name__}: {e}) -- "
          f"gated meta-llama load will likely fail and fall through to the "
          f"ungated mirror", flush=True)

n_gpu = torch.cuda.device_count()
student_device = "cuda:0" if n_gpu >= 1 else "cpu"
teacher_device = "cuda:1" if n_gpu >= 2 else student_device

# ---------------------------------------------------------------------------
# PHASE 1: teacher load + one-time top-64 logit caching over every train +
# held-out example, then the teacher is freed BEFORE the student ever loads.
# ---------------------------------------------------------------------------
teacher = None
teacher_name = None
teacher_path = "plain_sft_no_teacher"
for cand, tag in TEACHER_CANDIDATES:
    try:
        print(f"attempting teacher load: {cand}", flush=True)
        cand_tok = AutoTokenizer.from_pretrained(cand)
        same_vocab = cand_tok.vocab_size == tok.vocab_size or len(cand_tok) == len(tok)
        teacher = AutoModelForCausalLM.from_pretrained(cand, dtype=torch.float16).to(teacher_device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher_name = cand
        teacher_path = f"token_level_kd:{tag}" if same_vocab else f"sequence_level_kd_fallback:{tag}"
        print(f"teacher loaded OK: {cand} same_vocab={same_vocab} path={teacher_path}", flush=True)
        break
    except Exception as e:
        print(f"teacher load FAILED for {cand}: {type(e).__name__}: {e}", flush=True)
        teacher = None

if teacher is None:
    print("no teacher accessible -- falling back to plain SFT on gold answers only", flush=True)
    teacher_path = "plain_sft_no_teacher"

print(f"C2B_TEACHER_PATH={teacher_path} teacher_name={teacher_name}", flush=True)

teacher_cache = {}
teacher_cache_path = "/kaggle/working/c2b_teacher_cache.pt"
if teacher is not None:
    print("=== Phase 1: teacher top-64 logit caching begin", flush=True)
    all_for_cache = train_pool + held_out
    n_cached, n_skipped = 0, 0
    with torch.no_grad():
        for i, ex in enumerate(all_for_cache):
            key = f"{ex['src']}::{ex['id']}"
            batch = make_batch([ex], SEQ_LEN, teacher_device)
            answer_mask = (batch["labels"] != -100)
            if answer_mask.sum() == 0:
                n_skipped += 1
                continue
            t_out = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = t_out.logits[0]
            mask1d = answer_mask[0]
            ans_logits = logits[mask1d].float()
            topv, topi = torch.topk(ans_logits, min(TOPK, ans_logits.size(-1)), dim=-1)
            teacher_cache[key] = {"topk_vals": topv.to(torch.float16).cpu(),
                                   "topk_idx": topi.to(torch.int32).cpu()}
            n_cached += 1
            del t_out, logits, ans_logits, topv, topi
            if i % 500 == 0:
                print(f"teacher cache progress: {i}/{len(all_for_cache)}", flush=True)
    torch.save(teacher_cache, teacher_cache_path)
    cache_bytes = os.path.getsize(teacher_cache_path)
    print(f"C2B_TEACHER_CACHE_WRITTEN n_cached={n_cached} n_skipped={n_skipped} "
          f"path={teacher_cache_path} bytes={cache_bytes}", flush=True)
    del teacher
    teacher = None
    gc.collect()
    torch.cuda.empty_cache()
    print("=== Phase 1 done: teacher freed, both GPUs available for Phase 2", flush=True)
else:
    print("no teacher -- skipping Phase 1 cache entirely (plain SFT, CE-only)", flush=True)

# ---------------------------------------------------------------------------
# PHASE 2: student load (device_map="balanced" across both T4s when 2 GPUs
# are present) + manual KD/SFT training loop against the CACHED top-64
# teacher logits (no live teacher forward during student training at all).
# fp32 master + fp16 AMP, same convention as E16a's fp32_amp_1024 winner.
# ---------------------------------------------------------------------------
if n_gpu >= 2:
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32, device_map="balanced")
else:
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(student_device)
student.config.use_cache = False
student.gradient_checkpointing_enable()
student.train()
input_device = student.get_input_embeddings().weight.device
print(f"student device_map={getattr(student, 'hf_device_map', None)} input_device={input_device}", flush=True)

# freeze embed_tokens + lm_head (v2 fix, kept in v3)
frozen_params = 0
for name, p in student.named_parameters():
    if "embed_tokens" in name or name.endswith("lm_head.weight"):
        p.requires_grad_(False)
        frozen_params += p.numel()
print(f"frozen embed_tokens+lm_head params: {frozen_params}", flush=True)


def build_optimizer(params):
    try:
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=LR), "bnb.PagedAdamW8bit"
    except Exception as e:
        print(f"paged_adamw_8bit unavailable ({e}), falling back to Adafactor", flush=True)
        try:
            from transformers.optimization import Adafactor
            return Adafactor(params, lr=LR, scale_parameter=False,
                              relative_step=False, warmup_init=False), "transformers.Adafactor"
        except Exception as e2:
            print(f"Adafactor unavailable ({e2}), falling back to torch AdamW", flush=True)
            return torch.optim.AdamW(params, lr=LR), "torch.AdamW"


trainable_params = [p for p in student.parameters() if p.requires_grad]
optimizer, optimizer_name = build_optimizer(trainable_params)
print(f"optimizer={optimizer_name} trainable_tensors={len(trainable_params)}", flush=True)

scaler = torch.cuda.amp.GradScaler(enabled=True)


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(MAX_STEPS - WARMUP_STEPS, 1)
    return 0.5 * LR * (1 + math.cos(math.pi * min(prog, 1.0)))


def kd_loss_from_cache(student_logits, answer_mask, ex, temp):
    """KL(student || teacher) restricted+renormalized to the cached top-64
    indices for this example's answer tokens (batch size 1 assumed)."""
    key = f"{ex['src']}::{ex['id']}"
    cached = teacher_cache.get(key)
    if cached is None:
        return None
    dev = student_logits.device
    mask1d = answer_mask[0]
    s_ans = student_logits[0][mask1d]  # [n_ans, vocab]
    topk_vals = cached["topk_vals"].to(dev).float()   # [n_cached, K]
    topk_idx = cached["topk_idx"].to(dev).long()      # [n_cached, K]
    n = min(s_ans.size(0), topk_vals.size(0))
    if n == 0:
        return None
    s_ans = s_ans[:n]
    topk_vals = topk_vals[:n]
    topk_idx = topk_idx[:n]
    s_topk = torch.gather(s_ans, 1, topk_idx)  # [n, K]
    s_logp = F.log_softmax(s_topk / temp, dim=-1)
    t_p = F.softmax(topk_vals / temp, dim=-1)
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (temp * temp)


idx_cursor = [0]
n_pool = max(len(train_pool), 1)


def run_step(step):
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    last_ce, last_kd = None, None
    for _ in range(GRAD_ACCUM):
        ex = train_pool[idx_cursor[0] % n_pool]
        idx_cursor[0] += 1
        batch = make_batch([ex], SEQ_LEN, input_device)
        answer_mask = (batch["labels"] != -100)
        if answer_mask.sum() == 0:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            out = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           labels=batch["labels"])
            ce_loss = out.loss
            loss = CE_WEIGHT * ce_loss
            kd_loss_val = None
            if teacher_cache:
                kd = kd_loss_from_cache(out.logits, answer_mask, ex, TEMP)
                if kd is not None:
                    kd_loss_val = kd.item()
                    loss = loss + KD_WEIGHT * kd
            loss = loss / GRAD_ACCUM
        scaler.scale(loss).backward()
        accum_loss += loss.item() * GRAD_ACCUM
        last_ce, last_kd = ce_loss.item(), kd_loss_val
    scaler.step(optimizer)
    scaler.update()
    return accum_loss / GRAD_ACCUM, last_ce, last_kd


ppl_before = eval_held_out(student, held_out, SEQ_LEN, input_device)
print(f"HELD_OUT_BEFORE avg_loss={ppl_before['avg_loss']:.4f} ppl={ppl_before['ppl']:.4f}", flush=True)

step_losses, held_out_track = [], []
best_held_loss = ppl_before["avg_loss"]
abort_reason = None
trainable_mode = "all_30_blocks"

# --- step 0, with the one-shot OOM-retry-by-freezing-bottom-half fallback ---
try:
    accum_loss, ce_val, kd_val = run_step(0)
except torch.cuda.OutOfMemoryError as e:
    print(f"C2B_OOM_AT_STEP0: {type(e).__name__}: {e}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    frozen_block_params = 0
    for name, p in student.named_parameters():
        if name.startswith("model.layers."):
            try:
                layer_idx = int(name.split(".")[2])
            except (IndexError, ValueError):
                continue
            if layer_idx < N_FREEZE_BLOCKS_FALLBACK and p.requires_grad:
                p.requires_grad_(False)
                frozen_block_params += p.numel()
    trainable_mode = f"top_{30 - N_FREEZE_BLOCKS_FALLBACK}_of_30_blocks"
    print(f"C2B_TRAINABLE={trainable_mode} frozen_block_params={frozen_block_params}", flush=True)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer, optimizer_name = build_optimizer(trainable_params)
    print(f"optimizer(retry)={optimizer_name} trainable_tensors={len(trainable_params)}", flush=True)
    accum_loss, ce_val, kd_val = run_step(0)  # if this OOMs again, let it propagate -> kernel fails loudly

print(f"C2B_TRAINABLE={trainable_mode}", flush=True)
step_losses.append((0, accum_loss))
print(f"step=0 loss={accum_loss:.4f} ce={ce_val} kd={kd_val if kd_val is not None else 'n/a'} "
      f"lr={lr_at(0):.2e}", flush=True)

for gi in range(n_gpu):
    dev = f"cuda:{gi}"
    alloc = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    print(f"MEM_AFTER_STEP1 device={dev} allocated_gib={alloc:.2f} "
          f"reserved_gib={reserved:.2f} max_allocated_gib={max_alloc:.2f}", flush=True)

if math.isnan(accum_loss) or math.isinf(accum_loss):
    abort_reason = "NaN/Inf loss at step 0"
    print(f"C2B_ABORT: {abort_reason}", flush=True)

# --- remaining steps ---
if abort_reason is None:
    for step in range(1, MAX_STEPS):
        accum_loss, ce_val, kd_val = run_step(step)
        if step % 10 == 0:
            print(f"step={step} loss={accum_loss:.4f} ce={ce_val} "
                  f"kd={kd_val if kd_val is not None else 'n/a'} lr={lr_at(step):.2e}", flush=True)
        step_losses.append((step, accum_loss))

        if math.isnan(accum_loss) or math.isinf(accum_loss):
            abort_reason = f"NaN/Inf loss at step {step}"
            print(f"C2B_ABORT: {abort_reason}", flush=True)
            break

        if step % 100 == 0:
            held = eval_held_out(student, held_out, SEQ_LEN, input_device)
            held_out_track.append((step, held["avg_loss"]))
            print(f"HELD_OUT step={step} avg_loss={held['avg_loss']:.4f} ppl={held['ppl']:.4f} "
                  f"best_so_far={best_held_loss:.4f}", flush=True)
            if held["avg_loss"] < best_held_loss:
                best_held_loss = held["avg_loss"]
            if step >= ABORT_STEP and best_held_loss >= ppl_before["avg_loss"]:
                abort_reason = f"no held-out improvement over baseline by step {step}"
                print(f"C2B_ABORT: {abort_reason}", flush=True)
                break

        if step % SAVE_EVERY == 0:
            ck_dir = f"/kaggle/working/c2b_ckpt_step{step}"
            os.makedirs(ck_dir, exist_ok=True)
            student.save_pretrained(ck_dir, safe_serialization=True)
            tok.save_pretrained(ck_dir)
            print(f"checkpoint saved: {ck_dir}", flush=True)

ppl_after = eval_held_out(student, held_out, SEQ_LEN, input_device)
print(f"HELD_OUT_AFTER avg_loss={ppl_after['avg_loss']:.4f} ppl={ppl_after['ppl']:.4f}", flush=True)

losses = [l for _, l in step_losses]
bad = any(math.isnan(l) or math.isinf(l) for l in losses)
train_pass = (ppl_after["avg_loss"] < ppl_before["avg_loss"]) and not bad and abort_reason is None
print(f"C2B_ABORT_REASON={abort_reason}", flush=True)
print("C2B_TRAIN_PASS" if train_pass else "C2B_TRAIN_FAIL", flush=True)

# ---------------------------------------------------------------------------
# Export: identical contract to E16a (ternary-quantize the 7 per-layer
# projections, unpacked int8 + f32 _scale siblings; norms/embed/lm_head as
# BF16). Skipped on NaN abort (weights are garbage in that case). UNCHANGED.
# ---------------------------------------------------------------------------
export_done = False
if bad:
    print("C2B_EXPORT_SKIPPED: NaN/Inf detected, weights not trustworthy", flush=True)
else:
    print("=== EXPORT begin", flush=True)
    student.eval()
    sd = student.state_dict()
    num_layers = student.config.num_hidden_layers
    out_sd = {}
    quant_report = []
    for i in range(num_layers):
        p = f"model.layers.{i}"
        for norm_name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                           "self_attn.attn_sub_norm.weight", "mlp.ffn_sub_norm.weight"):
            key = f"{p}.{norm_name}"
            if key in sd:
                out_sd[key] = sd[key].detach().to(torch.bfloat16).cpu().contiguous()
        for proj in PROJECTIONS:
            bkey = f"{p}.{proj}.bias"
            if bkey in sd:
                raise SystemExit(f"{bkey} present -- engine has no bias path, refusing export")
            wkey = f"{p}.{proj}.weight"
            w = sd[wkey].detach().to(torch.float32)
            scale = 1.0 / w.abs().mean().clamp_min(1e-12)
            q = (w * scale).round().clamp_(-1, 1)
            bad_vals = int(((q != -1) & (q != 0) & (q != 1)).sum().item())
            if bad_vals:
                raise SystemExit(f"{wkey}: {bad_vals} non-ternary values after round/clamp -- quantizer bug")
            out_sd[wkey] = q.to(torch.int8).cpu().contiguous()
            out_sd[f"{wkey}_scale"] = scale.reshape(1).to(torch.float32).cpu().contiguous()
            quant_report.append({"tensor": wkey, "scale": float(scale.item()), "shape": list(q.shape)})
        print(f"layer {i}: quantized", flush=True)

    out_sd["model.norm.weight"] = sd["model.norm.weight"].detach().to(torch.bfloat16).cpu().contiguous()
    out_sd["model.embed_tokens.weight"] = sd["model.embed_tokens.weight"].detach().to(torch.bfloat16).cpu().contiguous()
    tie = bool(getattr(student.config, "tie_word_embeddings", True))
    if not tie:
        lm_key = "lm_head.weight"
        if lm_key not in sd:
            raise SystemExit("config says untied but lm_head.weight missing from state_dict")
        out_sd[lm_key] = sd[lm_key].detach().to(torch.bfloat16).cpu().contiguous()
    print(f"tie_word_embeddings={tie}", flush=True)

    model_saf_path = os.path.join(OUT_CKPT, "model.safetensors")
    save_file(out_sd, model_saf_path, metadata={"format": "pt"})
    print(f"wrote {model_saf_path} ({os.path.getsize(model_saf_path)} bytes)", flush=True)
    student.config.save_pretrained(OUT_CKPT)
    tok.save_pretrained(OUT_CKPT)
    export_done = True
    print("C2B_EXPORT_PASS", flush=True)

# ---------------------------------------------------------------------------
# MANIFEST + sha256sums over every artifact this kernel wrote. UNCHANGED.
# ---------------------------------------------------------------------------
manifest = {"files": []}
for root, _, files in os.walk("/kaggle/working"):
    for fn in files:
        fp = os.path.join(root, fn)
        h = hashlib.sha256()
        try:
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            manifest["files"].append({"path": os.path.relpath(fp, "/kaggle/working"),
                                       "bytes": os.path.getsize(fp), "sha256": h.hexdigest()})
        except Exception as e:
            manifest["files"].append({"path": os.path.relpath(fp, "/kaggle/working"), "error": str(e)})
with open("/kaggle/working/c2b_MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"MANIFEST written: {len(manifest['files'])} files", flush=True)

result_out = {
    "student": STUDENT,
    "teacher_path": teacher_path,
    "teacher_name": teacher_name,
    "trainable_mode": trainable_mode,
    "optimizer": optimizer_name,
    "data_sources": [
        {"name": "allenai/ai2_arc:ARC-Easy", "split": "train", "license": "cc-by-sa-4.0", "n": len(arc_easy_train)},
        {"name": "allenai/ai2_arc:ARC-Challenge", "split": "train", "license": "cc-by-sa-4.0", "n": len(arc_chal_train)},
        {"name": "allenai/openbookqa:main", "split": "train", "license": "apache-2.0", "n": len(obqa_train)},
        {"name": "allenai/sciq", "excluded": True, "reason": "cc-by-nc-3.0 NonCommercial"},
    ],
    "eval_holdout_n": len(HOLDOUT_IDS),
    "eval_holdout_sha256_sorted": HOLDOUT_SHA256_SORTED,
    "combined_train_pool": len(train_pool),
    "held_out_disjoint_check": len(held_out),
    "teacher_cache_n_examples": len(teacher_cache),
    "hyperparameters": {"max_steps": MAX_STEPS, "seq_len": SEQ_LEN, "lr": LR,
                         "grad_accum": GRAD_ACCUM, "warmup_steps": WARMUP_STEPS,
                         "temperature": TEMP, "ce_weight": CE_WEIGHT, "kd_weight": KD_WEIGHT,
                         "topk": TOPK, "abort_step_check": ABORT_STEP, "save_every": SAVE_EVERY,
                         "dtype": "fp32_master_fp16_amp", "device_map": "balanced" if n_gpu >= 2 else "single"},
    "abort_reason": abort_reason,
    "held_out_before": ppl_before,
    "held_out_after": ppl_after,
    "held_out_track_every_100_steps": held_out_track,
    "losses_every_10_steps": step_losses[::10],
    "export_done": export_done,
    "export": {"ckpt_dir": OUT_CKPT, "source_packing_contract": "unpacked"} if export_done else None,
}
with open("/kaggle/working/c2b_result.json", "w") as f:
    json.dump(result_out, f, indent=2)
print("C2B_RESULT_WRITTEN", flush=True)
