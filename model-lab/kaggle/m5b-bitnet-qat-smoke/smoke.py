# M5b SMOKE v5 (ledger gate): BitNet-2B-4T bf16 master online-QAT fine-tune.
# trl REMOVED (1.8.x internals incompatible with native BitNet output);
# plain transformers Trainer = same CLM loss, stable API. Tries configs
# best->worst, reports which one fits 2xT4. SMOKE ONLY - artifact discarded.
import gc, json, math, subprocess, sys, time, traceback
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "bitsandbytes"])
print("pip rc", r.returncode, f"{time.time()-t0:.0f}s", flush=True)
if r.returncode: sys.exit("PIP_FAILED")
import torch, transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
from datasets import load_dataset
print("ENV torch", torch.__version__, "| transformers", transformers.__version__,
      "| gpus", torch.cuda.device_count(), flush=True)

MODEL = "microsoft/BitNet-b1.58-2B-4T-bf16"
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
raw = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:1000]")
raw = raw.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
              remove_columns=raw.column_names)

def tokenize(seq_len):
    return raw.map(lambda ex: tok(ex["text"], truncation=True, max_length=seq_len),
                   remove_columns=["text"])

ATTEMPTS = [
    ("fp32_amp_1024", torch.float32, True, 1024, 1e-4),
    ("fp32_amp_512",  torch.float32, True, 512,  1e-4),
    ("fp16_pure_512", torch.float16, False, 512, 5e-5),
]
result = None
for name, dtype, amp, seq, lr in ATTEMPTS:
    print(f"=== ATTEMPT {name}: dtype={dtype} amp={amp} seq={seq} lr={lr}", flush=True)
    model = trainer = None
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto")
        model.config.use_cache = False
        args = TrainingArguments(
            output_dir="/kaggle/working/m5b_out", max_steps=150,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=lr, warmup_steps=10, lr_scheduler_type="cosine",
            fp16=amp, gradient_checkpointing=True, optim="paged_adamw_8bit",
            logging_steps=5, save_strategy="no", report_to=[])
        trainer = Trainer(model=model, args=args, train_dataset=tokenize(seq),
                          data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
        trainer.train()
        hist = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
        result = {"config": name, "seq": seq, "lr": lr, "losses": hist}
        break
    except torch.cuda.OutOfMemoryError:
        print(f"OOM at {name} -> next config", flush=True)
    except Exception as e:
        if "out of memory" in str(e).lower() or "FP16 gradients" in str(e):
            print(f"{type(e).__name__} at {name}: {e} -> next config", flush=True)
        else:
            traceback.print_exc()
            sys.exit("M5B_ENV_BUG")
    finally:
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()

if not result:
    print("M5B_SMOKE_FAIL: no config fits 2xT4")
    sys.exit(1)
json.dump(result, open("/kaggle/working/m5b_result.json", "w"))
losses = [l for _, l in result["losses"]]
head, tail = sum(losses[:5])/5, sum(losses[-5:])/5
bad = any(math.isnan(l) or math.isinf(l) for l in losses)
print(f"M5B_CONFIG_USED={result['config']} | loss head5 {head:.4f} -> tail5 {tail:.4f} | nan/inf {bad}")
print("M5B_SMOKE_PASS" if (tail < head and not bad) else "M5B_SMOKE_FAIL")
