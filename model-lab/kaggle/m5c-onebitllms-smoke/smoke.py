# M5c SMOKE (ledger gate): onebitllms on Falcon-E needs the PINNED env -
# stock is dead on T4 (Triton>=3.3 dropped Turing CC7.5). Pin torch==2.6.0
# (bundles triton 3.2) + onebitllms, THEN import torch. Gate: env imports,
# BitLinear replacement works, loss falls ~100 steps, no NaN.
# SMOKE ONLY - artifact discarded.
import json, math, subprocess, sys, time
t0 = time.time()
def pip(*args):
    print("pip install", *args, flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
    print("-> rc", r.returncode, f"{time.time()-t0:.0f}s", flush=True)
    return r.returncode
# pinned env FIRST, before any torch import
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchvision", "torchaudio"])
print("removed torchvision/torchaudio (compiled against torch 2.10, poison imports under 2.6)", flush=True)
pip("torch==2.6.0") and sys.exit("TORCH_PIN_FAILED")
pip("onebitllms") and sys.exit("ONEBITLLMS_INSTALL_FAILED")
pip("transformers==4.52.4", "trl==0.18.2", "peft", "bitsandbytes") and sys.exit("PINNED_STACK_INSTALL_FAILED")

# T4 is sm_75: NO bf16 hardware. onebitllms triton kernels hardcode bf16
# (ptxas 'requires sm_80' proven in v4 run) -> patch kernels to fp16 in place.
import glob, re
patched = []
for f in glob.glob("/usr/local/lib/python3.12/dist-packages/onebitllms/**/*.py", recursive=True):
    s = open(f).read()
    if "bfloat16" in s:
        open(f, "w").write(s.replace("bfloat16", "float16"))
        patched.append(f.split("onebitllms/")[-1])
print("bf16->fp16 patched files:", patched, flush=True)

import torch, triton, transformers, trl, onebitllms
print("ENV torch", torch.__version__, "| triton", triton.__version__,
      "| transformers", transformers.__version__, "| trl", trl.__version__,
      "| onebitllms", getattr(onebitllms, "__version__", "?"),
      "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("onebitllms API:", [n for n in dir(onebitllms) if not n.startswith("_")])

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

MODEL = "tiiuae/Falcon-E-1B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, revision="prequantized")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision="prequantized", torch_dtype=torch.float16)
# BitLinear replacement - try known API names, print surface on miss
for fn_name in ("replace_linear_with_bitnet_linear", "replace_linear_layers", "prepare_bitnet_model"):
    fn = getattr(onebitllms, fn_name, None)
    if fn:
        model = fn(model)
        print("replaced linears via", fn_name)
        break
else:
    sys.exit("NO_REPLACE_FN_FOUND - see API dump above")
model = model.to("cuda")
model.config.use_cache = False
print("model ready", f"{sum(p.numel() for p in model.parameters())/1e9:.2f}B", f"{time.time()-t0:.0f}s")

ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:600]")
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)},
            remove_columns=ds.column_names)
import inspect as _insp
cfg_kw = dict(
    output_dir="/kaggle/working/m5c_out", max_steps=100,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=1e-4, warmup_steps=10, lr_scheduler_type="cosine",
    fp16=True, gradient_checkpointing=True,
    optim="paged_adamw_8bit", logging_steps=5, save_strategy="no",
    report_to=[], dataset_text_field="text")
_p = _insp.signature(SFTConfig.__init__).parameters
cfg_kw["max_seq_length" if "max_seq_length" in _p else "max_length"] = 1024
cfg = SFTConfig(**cfg_kw)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
trainer.train()

hist = [(h["step"], h["loss"]) for h in trainer.state.log_history if "loss" in h]
json.dump(hist, open("/kaggle/working/m5c_losses.json", "w"))
losses = [l for _, l in hist]
head, tail = sum(losses[:4])/4, sum(losses[-4:])/4
bad = any(math.isnan(l) or math.isinf(l) for l in losses)
print(f"loss head4 {head:.4f} -> tail4 {tail:.4f} | nan/inf: {bad}")
print("M5C_SMOKE_PASS" if (tail < head and not bad) else "M5C_SMOKE_FAIL")
