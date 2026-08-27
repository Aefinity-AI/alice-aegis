#!/usr/bin/env python3
"""Regenerate the pruned vocab.bin + embed.bin PAIR so they include the Llama-3
special tokens (which live in tokenizer.json's added_tokens, not model.vocab —
the reason the forge stripper missed them).

New id space:
  0..49999   = base BPE tokens with original id < 50000 (ids unchanged)
  50000..50255 = the 256 added special tokens, in original-id order (128000+k -> 50000+k)

embed.bin rows are sliced from the ORIGINAL Microsoft BF16 embedding table, so
row order exactly matches vocab.bin id order. vocab.bin carries the filtered
BPE merges section (ids remapped; only merges whose left/right/merged all
survive are kept, original rank order preserved)."""
import json, struct, shutil, os
from pathlib import Path

# Paths are derived relative to this script's location so the tool works
# regardless of which user/home checked out the repo. Override with env vars
# if tokenizer.json / model.safetensors live elsewhere.
SCRIPT_DIR = Path(__file__).resolve().parent      # .../alice-aegis/aegis-forge
REPO_ROOT = SCRIPT_DIR.parent                     # .../alice-aegis

TOKENIZER_JSON = os.environ.get("AEGIS_TOKENIZER_JSON", str(REPO_ROOT / "tokenizer.json"))
MODEL = os.environ.get("AEGIS_MODEL_SAFETENSORS", str(REPO_ROOT / "model.safetensors"))
OUT_DIR = os.environ.get("AEGIS_OUT_DIR", str(SCRIPT_DIR))
MAGIC = 0x564F4341  # 'ACOV'
HIDDEN = 2560
ROW_BYTES = HIDDEN * 2  # BF16

tk = json.load(open(TOKENIZER_JSON))
base_vocab = tk["model"]["vocab"]          # str -> old_id
added = tk.get("added_tokens", [])          # list of {id, content, ...}

# --- build new id space ---
base_kept = sorted(((oid, s) for s, oid in base_vocab.items() if oid < 50000))
assert len(base_kept) == 50000, f"expected 50000 base tokens, got {len(base_kept)}"
for i, (oid, _) in enumerate(base_kept):
    assert oid == i, f"base id gap at {i} (old id {oid})"

specials = sorted(((t["id"], t["content"]) for t in added))
print(f"specials: {len(specials)} (old ids {specials[0][0]}..{specials[-1][0]})")

new_tokens = [s for _, s in base_kept] + [s for _, s in specials]
old_ids    = [oid for oid, _ in base_kept] + [oid for oid, _ in specials]
n = len(new_tokens)
print(f"new vocab size: {n}")

str_to_newid = {s: i for i, s in enumerate(new_tokens)}
assert len(str_to_newid) == n, "duplicate token strings"
for t in ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
          "<|eot_id|>", "<|end_of_text|>", "ĊĊ"]:
    print(f"  {t!r}: id {str_to_newid.get(t, 'MISSING!')}")

# --- merges (base tokens only survive the id<50000 cut) ---
kept_merges = []
for m in tk["model"]["merges"]:
    p1, p2 = (m.split(" ") if isinstance(m, str) else m)
    i1, i2, im = str_to_newid.get(p1), str_to_newid.get(p2), str_to_newid.get(p1 + p2)
    if i1 is not None and i2 is not None and im is not None:
        kept_merges.append((i1, i2, im))
print(f"merges kept: {len(kept_merges)} of {len(tk['model']['merges'])}")

# --- write vocab.bin ---
vpath = os.path.join(OUT_DIR, "vocab.bin")
if os.path.exists(vpath):
    shutil.copy(vpath, vpath + ".pre_specials.bak")
with open(vpath, "wb") as f:
    f.write(struct.pack("<II", MAGIC, n))
    for s in new_tokens:
        b = s.encode("utf-8")
        f.write(struct.pack("<H", len(b)))
        f.write(b)
    f.write(struct.pack("<I", len(kept_merges)))
    for tri in kept_merges:
        f.write(struct.pack("<III", *tri))
print(f"wrote {vpath} ({os.path.getsize(vpath)} bytes)")

# --- write embed.bin (slice BF16 rows from original model) ---
with open(MODEL, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    e = hdr["model.embed_tokens.weight"]
    assert e["dtype"] == "BF16" and e["shape"] == [128256, HIDDEN]
    tensor_base = 8 + hlen + e["data_offsets"][0]
    epath = os.path.join(OUT_DIR, "embed.bin")
    if os.path.exists(epath):
        shutil.copy(epath, epath + ".pre_specials.bak")
    with open(epath, "wb") as out:
        for oid in old_ids:
            f.seek(tensor_base + oid * ROW_BYTES)
            row = f.read(ROW_BYTES)
            assert len(row) == ROW_BYTES
            out.write(row)
print(f"wrote {epath} ({os.path.getsize(epath)} bytes; expect {n * ROW_BYTES})")

# --- update config vocab_size for consistency ---
cpath = os.path.join(OUT_DIR, "aegis_pruned_config.json")
cfg = json.load(open(cpath))
cfg["vocab_size"] = n
json.dump(cfg, open(cpath, "w"), indent=2)
print(f"updated {cpath} vocab_size -> {n}")
