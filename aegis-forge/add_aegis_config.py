#!/usr/bin/env python3
"""Add a MODEL.SAF-style __metadata__["aegis_config"] block to a safetensors
file that has none, WITHOUT touching a single tensor byte.

Why this exists: aegis-core/src/model.rs's SafeTensors::metadata_field (used
by aegis-linux/examples/cis_decode.rs:51-54 and cis_witness.rs) hard-requires
__metadata__["aegis_config"] and refuses to run without it. aegis-eval's
TernaryInferenceEngine::new (aegis-core/src/inference.rs:97-116) tolerates a
missing block by falling back to the baked-in aegis-forge/aegis_pruned_config.json
with vocab_size overridden from VOCAB.BIN — so the two front doors accept
artifacts of different completeness. This script closes that gap for a file
that was produced before the metadata convention existed, by writing a NEW
file (never overwriting the input) that is byte-identical except for the
safetensors header.

Schema written (must match aegis-forge/repack_ternary.py's `aegis_config`
dict literal at repack_ternary.py:561-574 exactly — key set and types):
    num_hidden_layers      int
    hidden_size            int
    num_attention_heads    int
    num_key_value_heads    int
    intermediate_size      int
    vocab_size             int
    max_position_embeddings int
    hidden_act             "relu2" | "silu"
    rope_theta             float
    rms_norm_eps           float
    tie_word_embeddings    bool
    chat_template           "llama3" | "chatml" | "none"

Derivation policy (mirrors aegis-core/src/model.rs ModelConfig::from_json,
model.rs:56-125, and the load-time cross-checks in
aegis-core/src/inference.rs:97-160):

  Derivable from MODEL.SAF tensor shapes alone, and used directly (no config
  file consulted for these):
    - num_hidden_layers   : count of distinct `model.layers.N.` prefixes
    - hidden_size         : len(model.norm.weight) == len(*.input_layernorm.weight)
    - intermediate_size   : len(*.mlp.ffn_sub_norm.weight)
    - tie_word_embeddings : True iff no `lm_head.weight` tensor is present
                             (model.rs:400-419: lm_head is REQUIRED when untied,
                             absent when tied)
    - hidden_act signal   : presence of attn_sub_norm/ffn_sub_norm tensors in
                             every layer implies relu2/BitNet (model.rs:319-328
                             comment: SubLN is optional only for families
                             trained without it, i.e. silu/Falcon-E) — this is
                             a CROSS-CHECK against the config value, not itself
                             sufficient to fully determine hidden_act (an
                             engine kernel choice with only two legal values)

  NOT derivable from MODEL.SAF shapes (no head-count split is recoverable from
  a packed [dim_out, dim_in/4] byte count alone, and vocab/seq-len simply
  don't appear in this file's tensor set — the embedding table lives in
  EMBED.BIN, not MODEL.SAF), so pulled from the model's own
  aegis-forge/aegis_pruned_config.json (a HF-style config.json for this exact
  pruned artifact) and then CROSS-CHECKED against shapes/sibling artifacts:
    - num_attention_heads, num_key_value_heads:
        cross-check: hidden_size % num_attention_heads == 0,
                      num_attention_heads % num_key_value_heads == 0
                      (inference.rs:150-159 enforces both at load time)
        cross-check: kv_dim derived from k_proj/v_proj tensor byte count
                      (kv_dim = total_elems * 4 / hidden_size) must equal
                      num_key_value_heads * (hidden_size / num_attention_heads)
    - vocab_size:
        cross-check against EMBED.BIN: vocab_size == len(embed.bin) / (hidden_size*2)
        cross-check against VOCAB.BIN: vocab_size == token count in its 'ACOV' header
    - max_position_embeddings, rope_theta, rms_norm_eps:
        no shape signal exists; taken as-is from config json
    - chat_template:
        sniffed from the checkpoint's tokenizer_config.json jinja, using the
        exact same rule as repack_ternary.py's sniff_chat_template
        (repack_ternary.py:445-463): "<|im_start|>" -> chatml,
        "<|start_header_id|>" -> llama3, else -> none (with a warning)

Usage:
    python3 add_aegis_config.py <input.safetensors> [--config PATH]
        [--embed PATH] [--vocab-bin PATH] [--tokenizer-config PATH]

Output: <input>.cis.safetensors (suffix swap: foo.safetensors -> foo.cis.safetensors)
"""
import argparse
import json
import os
import struct
import sys

MAGIC_ACOV = 0x564F4341


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return n, hdr


def tensor_names_by_layer(hdr):
    layers = {}
    for k in hdr:
        if k == "__metadata__" or not k.startswith("model.layers."):
            continue
        idx = int(k.split(".")[2])
        layers.setdefault(idx, set()).add(k)
    return layers


def shape_len(hdr, name):
    entry = hdr.get(name)
    if entry is None:
        return None
    shape = entry["shape"]
    total = 1
    for d in shape:
        total *= d
    return total


def derive_from_shapes(hdr):
    """Values pulled straight from MODEL.SAF tensor shapes/names, no config
    file involved. Mirrors what a loader that only sees MODEL.SAF could
    possibly know."""
    layers = tensor_names_by_layer(hdr)
    if not layers:
        raise SystemExit("no model.layers.N.* tensors found — not a MODEL.SAF-shaped file")
    num_hidden_layers = max(layers) + 1
    if set(layers.keys()) != set(range(num_hidden_layers)):
        raise SystemExit(f"layer indices not contiguous 0..{num_hidden_layers - 1}: {sorted(layers)}")

    hidden_size = shape_len(hdr, "model.norm.weight")
    if hidden_size is None:
        raise SystemExit("model.norm.weight missing — cannot derive hidden_size")
    h0 = shape_len(hdr, "model.layers.0.input_layernorm.weight")
    if h0 != hidden_size:
        raise SystemExit(
            f"hidden_size mismatch: model.norm.weight={hidden_size} vs "
            f"layer0 input_layernorm={h0}"
        )

    intermediate_size = shape_len(hdr, "model.layers.0.mlp.ffn_sub_norm.weight")
    if intermediate_size is None:
        # Falcon-E-style checkpoints may lack ffn_sub_norm; fall back to the
        # packed gate_proj byte count: gate_proj is [intermediate, hidden/4].
        gate_elems = shape_len(hdr, "model.layers.0.mlp.gate_proj.weight")
        if gate_elems is None or hidden_size % 4 != 0:
            raise SystemExit("cannot derive intermediate_size from shapes")
        intermediate_size = gate_elems // (hidden_size // 4)

    tie_word_embeddings = "lm_head.weight" not in hdr

    subln_layers = 0
    for i in range(num_hidden_layers):
        p = f"model.layers.{i}"
        if f"{p}.self_attn.attn_sub_norm.weight" in hdr and f"{p}.mlp.ffn_sub_norm.weight" in hdr:
            subln_layers += 1
    if subln_layers == num_hidden_layers:
        hidden_act_signal = "relu2"
    elif subln_layers == 0:
        hidden_act_signal = "silu"
    else:
        raise SystemExit(
            f"{subln_layers}/{num_hidden_layers} layers carry SubLN tensors — "
            "inconsistent checkpoint, refusing to guess hidden_act"
        )

    # kv_dim = num_key_value_heads * head_dim, recovered from the packed
    # k_proj byte count: k_proj is [kv_dim, hidden/4] (model.rs DecoderLayer
    # loads it as a flat tensor; repack_ternary.py's PROJECTIONS table pins
    # the shape convention).
    k_elems = shape_len(hdr, "model.layers.0.self_attn.k_proj.weight")
    if k_elems is None or hidden_size % 4 != 0:
        raise SystemExit("cannot derive kv_dim from shapes")
    kv_dim = k_elems * 4 // hidden_size
    if kv_dim * (hidden_size // 4) != k_elems:
        raise SystemExit(
            f"k_proj element count {k_elems} not divisible cleanly by hidden/4 "
            f"({hidden_size // 4}) — kv_dim derivation failed"
        )

    # q_proj must be [hidden, hidden/4] — a pure hidden_size cross-check.
    q_elems = shape_len(hdr, "model.layers.0.self_attn.q_proj.weight")
    if q_elems != hidden_size * (hidden_size // 4):
        raise SystemExit(
            f"q_proj element count {q_elems} != hidden*hidden/4 "
            f"({hidden_size * (hidden_size // 4)}) — hidden_size derivation inconsistent"
        )

    return {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "tie_word_embeddings": tie_word_embeddings,
        "hidden_act_signal": hidden_act_signal,
        "kv_dim": kv_dim,
    }


def sniff_chat_template(tokenizer_config_path):
    """Exact port of repack_ternary.py's sniff_chat_template (auto branch),
    repack_ternary.py:445-463."""
    try:
        with open(tokenizer_config_path) as f:
            jinja = json.load(f).get("chat_template") or ""
    except (OSError, ValueError):
        jinja = ""
    if "<|im_start|>" in jinja:
        return "chatml"
    if "<|start_header_id|>" in jinja:
        return "llama3"
    if jinja:
        print(
            "WARNING: unrecognized chat_template jinja; storing 'none' — "
            "pass --chat-template explicitly if the engine should wrap prompts"
        )
    return "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .safetensors with no __metadata__")
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    ap.add_argument("--config", default=os.path.join(here, "aegis_pruned_config.json"),
                     help="model's own config.json-style file for fields not derivable from shapes")
    ap.add_argument("--embed", default=os.path.join(here, "embed.bin"),
                     help="EMBED.BIN, used only to cross-check vocab_size")
    ap.add_argument("--vocab-bin", default=os.path.join(here, "vocab.bin"),
                     help="VOCAB.BIN, used only to cross-check vocab_size")
    ap.add_argument("--tokenizer-config", default=os.path.join(repo_root, "tokenizer_config.json"),
                     help="tokenizer_config.json to sniff chat_template from")
    ap.add_argument("--chat-template", choices=["auto", "llama3", "chatml", "none"], default="auto")
    args = ap.parse_args()

    if not args.input.endswith(".safetensors"):
        raise SystemExit(f"expected a .safetensors input, got {args.input}")
    out_path = args.input[: -len(".safetensors")] + ".cis.safetensors"
    if os.path.abspath(out_path) == os.path.abspath(args.input):
        raise SystemExit("refusing to overwrite the input file")

    n0, hdr = read_header(args.input)
    if "__metadata__" in hdr:
        raise SystemExit(f"{args.input} already carries __metadata__ — nothing to do")

    shapes = derive_from_shapes(hdr)
    print("derived from MODEL.SAF shapes:")
    for k, v in shapes.items():
        print(f"  {k}: {v}")

    with open(args.config) as f:
        cfg = json.load(f)

    hidden_act = cfg.get("hidden_act", "relu2")
    if hidden_act not in ("relu2", "silu"):
        raise SystemExit(f"config hidden_act '{hidden_act}' has no engine kernel")
    if hidden_act != shapes["hidden_act_signal"]:
        raise SystemExit(
            f"hidden_act mismatch: config says '{hidden_act}', but SubLN tensor "
            f"presence in MODEL.SAF signals '{shapes['hidden_act_signal']}'"
        )

    num_attention_heads = int(cfg["num_attention_heads"])
    num_key_value_heads = int(cfg["num_key_value_heads"])
    hidden_size = shapes["hidden_size"]
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise SystemExit("config num_attention_heads / num_key_value_heads must be positive")
    if hidden_size % num_attention_heads != 0:
        raise SystemExit(
            f"hidden_size {hidden_size} (from shapes) not divisible by "
            f"config num_attention_heads {num_attention_heads}"
        )
    if num_attention_heads % num_key_value_heads != 0:
        raise SystemExit(
            f"config num_attention_heads {num_attention_heads} not a multiple of "
            f"num_key_value_heads {num_key_value_heads}"
        )
    head_dim = hidden_size // num_attention_heads
    kv_dim_from_config = num_key_value_heads * head_dim
    if kv_dim_from_config != shapes["kv_dim"]:
        raise SystemExit(
            f"kv_dim mismatch: config (num_key_value_heads={num_key_value_heads} * "
            f"head_dim={head_dim}) = {kv_dim_from_config}, but k_proj shape in "
            f"MODEL.SAF implies kv_dim={shapes['kv_dim']}"
        )

    vocab_size = int(cfg["vocab_size"])
    if os.path.exists(args.embed):
        embed_bytes = os.path.getsize(args.embed)
        if embed_bytes % (hidden_size * 2) != 0:
            raise SystemExit(
                f"{args.embed}: {embed_bytes} bytes not divisible by hidden_size*2 ({hidden_size * 2})"
            )
        vocab_from_embed = embed_bytes // (hidden_size * 2)
        if vocab_from_embed != vocab_size:
            raise SystemExit(
                f"vocab_size mismatch: config says {vocab_size}, {args.embed} implies "
                f"{vocab_from_embed} rows ({embed_bytes} bytes / {hidden_size * 2})"
            )
        print(f"cross-check OK: vocab_size {vocab_size} matches {args.embed} row count")
    else:
        print(f"WARNING: {args.embed} not found — skipping vocab_size cross-check against EMBED.BIN")

    if os.path.exists(args.vocab_bin):
        with open(args.vocab_bin, "rb") as f:
            magic, count = struct.unpack("<II", f.read(8))
        if magic != MAGIC_ACOV:
            raise SystemExit(f"{args.vocab_bin}: bad magic 0x{magic:08x}, expected 0x{MAGIC_ACOV:08x} ('ACOV')")
        if count != vocab_size:
            raise SystemExit(
                f"vocab_size mismatch: config says {vocab_size}, {args.vocab_bin} header says {count} tokens"
            )
        print(f"cross-check OK: vocab_size {vocab_size} matches {args.vocab_bin} token count")
    else:
        print(f"WARNING: {args.vocab_bin} not found — skipping vocab_size cross-check against VOCAB.BIN")

    max_position_embeddings = int(cfg["max_position_embeddings"])
    if max_position_embeddings <= 0:
        raise SystemExit("config max_position_embeddings must be positive")

    rope_theta = float(cfg.get("rope_theta", 500000.0))
    rms_norm_eps = float(cfg.get("rms_norm_eps", 1e-5))

    tie_word_embeddings = bool(cfg.get("tie_word_embeddings", True))
    if tie_word_embeddings != shapes["tie_word_embeddings"]:
        raise SystemExit(
            f"tie_word_embeddings mismatch: config says {tie_word_embeddings}, but "
            f"MODEL.SAF {'lacks' if shapes['tie_word_embeddings'] else 'carries'} "
            "an lm_head.weight tensor"
        )

    if args.chat_template != "auto":
        chat_template = args.chat_template
    else:
        chat_template = sniff_chat_template(args.tokenizer_config)

    aegis_config = {
        "num_hidden_layers": shapes["num_hidden_layers"],
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "intermediate_size": shapes["intermediate_size"],
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "hidden_act": hidden_act,
        "rope_theta": rope_theta,
        "rms_norm_eps": rms_norm_eps,
        "tie_word_embeddings": tie_word_embeddings,
        "chat_template": chat_template,
    }

    cfg_json = json.dumps(aegis_config, separators=(",", ":"))
    if "\\" in cfg_json:
        # Same refusal as repack_ternary.py's write_model_saf: the engine's
        # metadata parser is deliberately ASCII-only.
        raise SystemExit(
            "aegis_config needs JSON escapes (non-ASCII or quotes in values) — "
            "the engine metadata parser is ASCII-only"
        )

    new_hdr = {"__metadata__": {"aegis_config": cfg_json}}
    new_hdr.update(hdr)  # preserve every original tensor entry, in order
    new_hjson = json.dumps(new_hdr, separators=(",", ":")).encode()

    in_size = os.path.getsize(args.input)
    with open(args.input, "rb") as fin, open(out_path, "wb") as fout:
        fin.seek(8 + n0)
        fout.write(struct.pack("<Q", len(new_hjson)))
        fout.write(new_hjson)
        chunk = 1 << 24
        copied = 0
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            fout.write(buf)
            copied += len(buf)

    expect_data_bytes = in_size - 8 - n0
    if copied != expect_data_bytes:
        raise SystemExit(
            f"copied {copied} tensor-data bytes, expected {expect_data_bytes} — short read?"
        )

    print()
    print(f"wrote {out_path}")
    print(f"aegis_config: {cfg_json}")


if __name__ == "__main__":
    main()
