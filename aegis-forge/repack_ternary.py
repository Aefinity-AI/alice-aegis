#!/usr/bin/env python3
"""Repack an already-ternary HF checkpoint into the engine's artifact triple:

    MODEL.SAF   safetensors, engine tensor names, engine 2-bit packing,
                per-tensor scalar F32 weight_scale, BF16 norms (+ BF16
                lm_head.weight when untied), and the model's config JSON
                stored in __metadata__["aegis_config"]
    EMBED.BIN   BF16 rows, id order == VOCAB.BIN order
    VOCAB.BIN   'ACOV' flat vocab + BPE merges trailer

This is a REPACKER, not a transmuter: source weights must already be
ternary. Any tensor whose values are not exactly {-1, 0, +1} (after the
declared source packing is decoded) aborts the run — quantizing dense
weights here is how fifteen months got burned, and it stays refused.

Engine packing (aegis-core/src/ops.rs build_unpack_lut):
    4 weights per byte, LSB-first 2-bit fields; code 0 -> 0.0,
    1 -> +1.0, 2 -> -1.0, 3 undefined (decoded as 0.0, never emitted).
    Row-major: each output row is dim_in/4 packed bytes.

Source packings supported (--source-packing):
    unpacked    int8/float tensors already holding {-1, 0, +1}
                (what onebitllms unpacking produces for Falcon-E)
    hf1bitllm   uint8 [out/4, in] packed ALONG DIM 0 in row blocks:
                the 2-bit field at shift 2*i of byte (r, c) is output row
                i*(out/4) + r, stored as w+1 ({0,1,2} = {-1,0,+1}).
                This is the transformers pack_weights layout used by
                HF1BitLLM Llama3-8B-1.58 and microsoft BitNet packed
                checkpoints, as pinned by correct_transmute.py's
                unpack_bitnet — NOT packed along the input dim, so a
                full unpack/repack is required, never a byte relabel.

Usage:
    python3 repack_ternary.py CKPT_DIR OUT_DIR \
        [--source-packing unpacked|hf1bitllm] [--llama3-prune] \
        [--max-seq 2048]

--llama3-prune reproduces regen_vocab_embed.py's id space (base ids
< 50000 unchanged; the 256 added specials remapped 128000+k -> 50000+k)
so the Llama3-8B-1.58 port reuses the proven pruned-vocab path.
numpy is used when importable (a must for real checkpoints); tiny
tensors fall back to pure python so the self-tests run anywhere.
"""
import argparse
import json
import os
import struct
import sys

MAGIC = 0x564F4341  # 'ACOV'

try:
    import numpy as _np
except ImportError:  # tests run stdlib-only; real conversions want numpy
    _np = None


# --------------------------------------------------------------------------
# 2-bit packing primitives (T3a round-trips these)
# --------------------------------------------------------------------------

ENGINE_CODE = {0: 0, 1: 1, -1: 2}
ENGINE_VALUE = {0: 0, 1: 1, 2: -1}  # code 3 undefined; never emitted


def pack_ternary(values):
    """{-1,0,+1} ints -> engine packed bytes (4 per byte, LSB-first)."""
    if len(values) % 4 != 0:
        raise ValueError(f"ternary run of {len(values)} is not a multiple of 4")
    if _np is not None and len(values) >= 4096:
        a = _np.asarray(values, dtype=_np.int8).reshape(-1, 4)
        if bool(((a < -1) | (a > 1)).any()):
            raise ValueError("non-ternary values — this tool repacks, it does not quantize")
        codes = _np.where(a == -1, 2, a).astype(_np.uint8)
        b = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
        return b.tobytes()
    bad = set(values) - {-1, 0, 1}
    if bad:
        raise ValueError(f"non-ternary values {sorted(bad)} — this tool repacks, it does not quantize")
    out = bytearray(len(values) // 4)
    for i in range(0, len(values), 4):
        out[i // 4] = (
            ENGINE_CODE[values[i]]
            | (ENGINE_CODE[values[i + 1]] << 2)
            | (ENGINE_CODE[values[i + 2]] << 4)
            | (ENGINE_CODE[values[i + 3]] << 6)
        )
    return bytes(out)


def unpack_ternary(packed):
    """Engine packed bytes -> list of {-1,0,+1} ints."""
    out = []
    for b in packed:
        for shift in (0, 2, 4, 6):
            code = (b >> shift) & 3
            if code == 3:
                raise ValueError("code 3 encountered — corrupt or foreign packing")
            out.append(ENGINE_VALUE[code])
    return out


def unpack_hf1bitllm(raw, dim_out, dim_in):
    """hf1bitllm packed bytes -> row-major [out, in] weights in {-1, 0, +1}.

    Source layout ([out/4, in] uint8, packed along dim 0): with M = out/4,
    the 2-bit field at shift 2*i of byte (r, c) is output row i*M + r of
    column c, stored as w+1. Mirrors correct_transmute.py unpack_bitnet,
    the converter that produced the working BitNet-2B MODEL.SAF.
    """
    m = dim_out // 4
    if dim_out % 4 != 0 or len(raw) != m * dim_in:
        raise ValueError(f"hf1bitllm: {len(raw)} bytes does not fit [{dim_out}/4, {dim_in}]")
    if _np is not None and len(raw) >= 4096:
        p = _np.frombuffer(bytes(raw), dtype=_np.uint8).reshape(m, dim_in)
        out = _np.empty((dim_out, dim_in), dtype=_np.int8)
        for i in range(4):
            out[i * m:(i + 1) * m, :] = ((p >> (2 * i)) & 3).astype(_np.int8) - 1
        if (out > 1).any():
            raise ValueError("2-bit code 3 encountered — not hf1bitllm packing")
        return out.reshape(-1)
    vals = [0] * (dim_out * dim_in)
    for r in range(m):
        for c in range(dim_in):
            b = raw[r * dim_in + c]
            for i in range(4):
                code = (b >> (2 * i)) & 3
                if code == 3:
                    raise ValueError("2-bit code 3 encountered — not hf1bitllm packing")
                vals[(i * m + r) * dim_in + c] = code - 1
    return vals


# --------------------------------------------------------------------------
# dtype helpers (stdlib '<e' handles fp16; BF16 = truncated top of f32,
# matching the existing forge slicer's behavior)
# --------------------------------------------------------------------------

def to_f32_list(raw, dtype):
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw)//4}f", raw))
    if dtype == "F16":
        return list(struct.unpack(f"<{len(raw)//2}e", raw))
    if dtype == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i+2])[0] for i in range(0, len(raw), 2)]
    if dtype in ("I8", "int8"):
        return [b - 256 if b > 127 else b for b in raw]
    if dtype == "U8":
        return list(raw)
    raise ValueError(f"unsupported dtype {dtype}")


def _to_f32_np(raw, dtype):
    """to_f32_list's numpy fast path: one vectorized pass, no per-element calls."""
    if dtype == "F32":
        return _np.frombuffer(raw, dtype="<f4")
    if dtype == "F16":
        return _np.frombuffer(raw, dtype="<f2").astype(_np.float32)
    if dtype == "BF16":
        return (_np.frombuffer(raw, dtype="<u2").astype(_np.uint32) << 16).view(_np.float32)
    if dtype in ("I8", "int8"):
        return _np.frombuffer(raw, dtype=_np.int8).astype(_np.float32)
    if dtype == "U8":
        return _np.frombuffer(raw, dtype=_np.uint8).astype(_np.float32)
    raise ValueError(f"unsupported dtype {dtype}")


def f32_list_to_bf16(vals):
    if _np is not None and len(vals) >= 4096:
        a = _np.asarray(vals, dtype=_np.float32)
        return a.view(_np.uint32).astype(_np.uint32).__rshift__(16).astype("<u2").tobytes()
    out = bytearray()
    for v in vals:
        out += struct.pack("<f", v)[2:4]
    return bytes(out)


def raw_to_bf16(raw, dtype):
    """Any float dtype -> BF16 bytes. BF16 passes through untouched."""
    if dtype == "BF16":
        return bytes(raw)
    if dtype == "F32":
        if _np is not None and len(raw) >= 16384:
            return _np.frombuffer(raw, dtype="<u4").__rshift__(16).astype("<u2").tobytes()
        return b"".join(raw[i+2:i+4] for i in range(0, len(raw), 4))
    if dtype == "F16":
        if _np is not None and len(raw) >= 16384:
            f32 = _np.frombuffer(raw, dtype="<f2").astype("<f4")
            return f32.view("<u4").__rshift__(16).astype("<u2").tobytes()
        return f32_list_to_bf16(to_f32_list(raw, "F16"))
    raise ValueError(f"cannot convert dtype {dtype} to BF16")


# --------------------------------------------------------------------------
# source checkpoint access (single-file or sharded safetensors, stdlib-only,
# same manual header walk regen_vocab_embed.py uses)
# --------------------------------------------------------------------------

class SourceTensors:
    def __init__(self, ckpt_dir):
        self.dir = ckpt_dir
        index = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(ckpt_dir, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(f"no model.safetensors[.index.json] in {ckpt_dir}")
            self.weight_map = None
            self._single = "model.safetensors"
        self._headers = {}

    def _header(self, fname):
        if fname not in self._headers:
            with open(os.path.join(self.dir, fname), "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                self._headers[fname] = (json.loads(f.read(hlen)), 8 + hlen)
        return self._headers[fname]

    def names(self):
        if self.weight_map is not None:
            return list(self.weight_map)
        hdr, _ = self._header(self._single)
        return [k for k in hdr if k != "__metadata__"]

    def has(self, name):
        # dict membership, not a rebuilt name list: this runs ~9x per layer
        if self.weight_map is not None:
            return name in self.weight_map
        hdr, _ = self._header(self._single)
        return name != "__metadata__" and name in hdr

    def get(self, name):
        """-> (raw bytes, dtype, shape)"""
        fname = self.weight_map[name] if self.weight_map else self._single
        hdr, base = self._header(fname)
        ent = hdr[name]
        start, end = ent["data_offsets"]
        with open(os.path.join(self.dir, fname), "rb") as f:
            f.seek(base + start)
            raw = f.read(end - start)
        return raw, ent["dtype"], ent["shape"]


# --------------------------------------------------------------------------
# vocab + embeddings
# --------------------------------------------------------------------------

def full_vocab_id_space(tok):
    """Dense id -> token string for a full (unpruned) tokenizer. The added
    tokens overlay the base vocab; ids must come out dense or the engine's
    Vec-indexed vocab would silently shift every row after a gap."""
    base = tok["model"]["vocab"]
    entries = {oid: s for s, oid in base.items()}
    for t in tok.get("added_tokens", []):
        entries[t["id"]] = t["content"]
    n = max(entries) + 1
    missing = [i for i in range(n) if i not in entries]
    if missing:
        raise ValueError(f"vocab has id gaps (first: {missing[:5]}) — refusing to guess a row order")
    tokens = [entries[i] for i in range(n)]
    return tokens, list(range(n))


def llama3_pruned_id_space(tok):
    """regen_vocab_embed.py's id space, reproduced: base ids < 50000 keep
    their ids; the 256 added specials append in original-id order."""
    base = tok["model"]["vocab"]
    base_kept = sorted((oid, s) for s, oid in base.items() if oid < 50000)
    if len(base_kept) != 50000:
        raise ValueError(f"expected 50000 base tokens below the cut, got {len(base_kept)}")
    for i, (oid, _) in enumerate(base_kept):
        if oid != i:
            raise ValueError(f"base id gap at {i} (old id {oid})")
    specials = sorted((t["id"], t["content"]) for t in tok.get("added_tokens", []))
    tokens = [s for _, s in base_kept] + [s for _, s in specials]
    old_ids = [oid for oid, _ in base_kept] + [oid for oid, _ in specials]
    return tokens, old_ids


def write_vocab_bin(path, tokens, tok):
    str_to_id = {s: i for i, s in enumerate(tokens)}
    if len(str_to_id) != len(tokens):
        raise ValueError("duplicate token strings in vocab")
    kept = []
    for m in tok["model"].get("merges", []):
        p1, p2 = (m.split(" ") if isinstance(m, str) else m)
        i1, i2, im = str_to_id.get(p1), str_to_id.get(p2), str_to_id.get(p1 + p2)
        if i1 is not None and i2 is not None and im is not None:
            kept.append((i1, i2, im))
    with open(path, "wb") as f:
        f.write(struct.pack("<II", MAGIC, len(tokens)))
        for s in tokens:
            b = s.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(kept)))
        for tri in kept:
            f.write(struct.pack("<III", *tri))
    zero_id = sum(1 for tri in kept if 0 in tri)
    if zero_id:
        print(f"WARNING: {zero_id} merges involve token id 0 ('{tokens[0]}'); the engine "
              "loader (aegis-core/src/tokenizer.rs) currently DROPS merges containing id 0, "
              "so those pairs tokenize differently from the reference stack. Run the T2d "
              "tokenization-parity gate before trusting any cross-stack number.")
    print(f"VOCAB.BIN: {len(tokens)} tokens, {len(kept)} merges")
    return len(tokens)


def slice_rows_bf16(raw, dtype, hidden, old_ids, n_rows):
    """Row-slice a [vocab, hidden] table into BF16, preserving id order.
    Rows are sliced from the SOURCE first, then converted: with --llama3-prune
    only 39% of the table survives, so converting everything first would
    waste both the conversion work and a gigabyte-class intermediate.

    `n_rows` is the source table's row count: python slicing past the end
    yields b"" SILENTLY, which would truncate EMBED.BIN with no error and
    only surface as zero-embedding garbage on high-id tokens at runtime."""
    elem = {"F32": 4, "F16": 2, "BF16": 2}.get(dtype)
    if elem is None:
        raise ValueError(f"cannot slice dtype {dtype}")
    too_big = [oid for oid in old_ids if oid >= n_rows]
    if too_big:
        raise ValueError(
            f"tokenizer defines ids up to {max(too_big)} but the table has only "
            f"{n_rows} rows — tokenizer and checkpoint are mispaired")
    row = hidden * elem
    picked = b"".join(raw[oid * row:(oid + 1) * row] for oid in old_ids)
    return raw_to_bf16(picked, dtype)


# --------------------------------------------------------------------------
# MODEL.SAF writer
# --------------------------------------------------------------------------

def write_safetensors(path, tensors, metadata=None):
    """Serialize (name, dtype, shape, bytes) tuples (+ optional string-map
    metadata) into a safetensors file. The ONE writer in the forge tree —
    gen_synth_checkpoint.py and the self-tests reuse it, so a layout change
    cannot fork the artifact contract between the gates."""
    header = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hjson = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for _, _, _, raw in tensors:
            f.write(raw)
    return 8 + len(hjson) + offset


def write_model_saf(path, tensors, aegis_config):
    """tensors: list of (name, dtype, shape, bytes). Metadata carries the
    config the engine will parse (values in __metadata__ must be strings,
    so the JSON is embedded escaped — aegis-core unescapes it)."""
    cfg_json = json.dumps(aegis_config, separators=(",", ":"))
    if "\\" in cfg_json:
        # json.dumps escapes non-ASCII as \uXXXX; the engine's metadata
        # parser is deliberately ASCII-only and treats such configs as a
        # load error. Refuse here, where the config can still be fixed.
        raise SystemExit("aegis_config needs JSON escapes (non-ASCII or quotes "
                         "in values) — the engine metadata parser is ASCII-only")
    total = write_safetensors(path, tensors, {"aegis_config": cfg_json})
    print(f"MODEL.SAF: {len(tensors)} tensors, {total} bytes")


# --------------------------------------------------------------------------
# the repack itself
# --------------------------------------------------------------------------

PROJECTIONS = [
    ("self_attn.q_proj", "hidden", "hidden"),
    ("self_attn.k_proj", "kv", "hidden"),
    ("self_attn.v_proj", "kv", "hidden"),
    ("self_attn.o_proj", "hidden", "hidden"),
    ("mlp.gate_proj", "ffn", "hidden"),
    ("mlp.up_proj", "ffn", "hidden"),
    ("mlp.down_proj", "hidden", "ffn"),
]


def ternary_weight_bytes(src, name, packing, dim_out, dim_in):
    raw, dtype, shape = src.get(name)
    if packing == "hf1bitllm":
        if dtype != "U8":
            raise ValueError(f"{name}: expected U8 packed, got {dtype}")
        # The header shape is the one signal that distinguishes the two
        # possible packing axes (both have the same byte count) — check it.
        if shape != [dim_out // 4, dim_in]:
            raise ValueError(
                f"{name}: shape {shape}, expected [{dim_out // 4}, {dim_in}] "
                "(hf1bitllm packs along dim 0) — wrong --source-packing?")
        return pack_ternary(unpack_hf1bitllm(raw, dim_out, dim_in))
    # unpacked: any numeric dtype, values must already be exactly ternary
    if _np is not None and len(raw) >= 4096:
        a = _to_f32_np(raw, dtype)
        if a.size != dim_out * dim_in:
            raise ValueError(f"{name}: {a.size} values != {dim_out}x{dim_in} (shape {shape})")
        ints = _np.rint(a)
        if bool((ints != a).any()) or bool(((ints < -1) | (ints > 1)).any()):
            raise ValueError(
                f"{name}: non-ternary values — repack refused "
                "(quantizing dense weights is out of scope, by design)")
        return pack_ternary(ints.astype(_np.int8))
    vals = to_f32_list(raw, dtype)
    ints = []
    for v in vals:
        i = int(round(v))
        if i != v or i not in (-1, 0, 1):
            raise ValueError(
                f"{name}: value {v} is not ternary — repack refused "
                "(quantizing dense weights is out of scope, by design)")
        ints.append(i)
    if len(ints) != dim_out * dim_in:
        raise ValueError(f"{name}: {len(ints)} values != {dim_out}x{dim_in} (shape {shape})")
    return pack_ternary(ints)


def scalar_f32_scale(src, name):
    """Per-tensor weight_scale, converted to the ENGINE's convention.

    Source checkpoints served by transformers' BitLinear (Falcon-E via
    onebitllms, HF1BitLLM) store the QUANTIZATION scale and dequantize by
    DIVISION: out = (x_q @ w_q) / (act_scale * weight_scale), i.e.
    effective weights = w_q / weight_scale (onebitllms _weight_quant:
    scale = 1/mean|w|, so stored values are ~10-60).

    The engine MULTIPLIES: out[row] = dot * scale (ops.rs), the convention
    of the Microsoft BitNet-2B artifact chain (empirically pinned by the
    10.348 eval anchor and the 'Paris' generation). So the repack must
    write the RECIPROCAL of the source scale. Found the hard way: copying
    the raw scalar made every Falcon-E linear ~450x too large and produced
    confident token-salad while all shape/ternary checks passed.
    """
    raw, dtype, shape = src.get(name)
    n = 1
    for d in shape:
        n *= d
    if n != 1:
        raise ValueError(f"{name}: {shape} is not a scalar — group-wise scales are unsupported "
                         "by the engine; this checkpoint needs a different recipe")
    source_scale = to_f32_list(raw, dtype)[0]
    if source_scale <= 0.0:
        raise ValueError(f"{name}: nonpositive weight_scale {source_scale}")
    return struct.pack("<f", 1.0 / source_scale)


def sniff_chat_template(args):
    """Engine prompt convention for aegis_config. Auto = read the source's
    tokenizer_config.json jinja and match its turn markers; unrecognized
    templates fall back to 'none' with a warning (a wrong template is
    worse than no template — the Falcon-E bring-up proved a model can
    stay fluent through 'none' but every PPL/quality gate is confounded)."""
    if args.chat_template != "auto":
        return args.chat_template
    try:
        with open(os.path.join(args.ckpt_dir, "tokenizer_config.json")) as f:
            jinja = json.load(f).get("chat_template") or ""
    except (OSError, ValueError):
        jinja = ""
    if "<|im_start|>" in jinja:
        return "chatml"
    if "<|start_header_id|>" in jinja:
        return "llama3"
    if jinja:
        print("WARNING: unrecognized chat_template jinja; storing 'none' — "
              "pass --chat-template explicitly if the engine should wrap prompts")
    return "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--source-packing", choices=["unpacked", "hf1bitllm"], default="unpacked")
    ap.add_argument("--llama3-prune", action="store_true",
                    help="use the proven pruned-vocab id space (Llama3-8B-1.58)")
    ap.add_argument("--max-seq", type=int, default=2048,
                    help="engine KV/arena window; the arena is sized by this, so a 32k "
                         "source context would exhaust a 2GB-class target")
    ap.add_argument("--chat-template", choices=["auto", "llama3", "chatml", "none"],
                    default="auto",
                    help="prompt convention stored in aegis_config; auto sniffs "
                         "tokenizer_config.json's chat_template")
    args = ap.parse_args()

    with open(os.path.join(args.ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    hidden_act = cfg.get("hidden_act", "silu")
    if hidden_act not in ("silu", "relu2"):
        raise SystemExit(f"hidden_act '{hidden_act}' has no engine kernel — this is an "
                         "architecture port, not a repack; refusing")
    if cfg.get("rope_scaling"):
        print(f"WARNING: source uses rope_scaling={cfg['rope_scaling']}; the engine has none. "
              f"Within a {args.max_seq}-token window this is usually moot — measure G4a before trusting it.")

    tie = bool(cfg.get("tie_word_embeddings", True))
    hidden = cfg["hidden_size"]
    kv_dim = (hidden // cfg["num_attention_heads"]) * cfg["num_key_value_heads"]
    dims = {"hidden": hidden, "kv": kv_dim, "ffn": cfg["intermediate_size"]}

    with open(os.path.join(args.ckpt_dir, "tokenizer.json")) as f:
        tok = json.load(f)
    tokens, old_ids = (llama3_pruned_id_space(tok) if args.llama3_prune
                       else full_vocab_id_space(tok))

    os.makedirs(args.out_dir, exist_ok=True)
    vocab_size = write_vocab_bin(os.path.join(args.out_dir, "VOCAB.BIN"), tokens, tok)

    src = SourceTensors(args.ckpt_dir)

    raw, dtype, shape = src.get("model.embed_tokens.weight")
    if shape[1] != hidden:
        raise SystemExit(f"embed_tokens hidden dim {shape[1]} != config hidden_size {hidden}")
    embed = slice_rows_bf16(raw, dtype, hidden, old_ids, shape[0])
    with open(os.path.join(args.out_dir, "EMBED.BIN"), "wb") as f:
        f.write(embed)
    print(f"EMBED.BIN: {len(embed)} bytes ({vocab_size} x {hidden} BF16)")

    out_tensors = []
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            raw, dtype, _ = src.get(f"{p}.{norm}")
            out_tensors.append((f"{p}.{norm}", "BF16", [hidden], raw_to_bf16(raw, dtype)))
        for sub, width in (("self_attn.attn_sub_norm.weight", hidden),
                           ("mlp.ffn_sub_norm.weight", cfg["intermediate_size"])):
            name = f"{p}.{sub}"
            if src.has(name):  # SubLN is optional: present in BitNet, absent in Falcon-E
                raw, dtype, _ = src.get(name)
                out_tensors.append((name, "BF16", [width], raw_to_bf16(raw, dtype)))
        for proj, dk_out, dk_in in PROJECTIONS:
            bias = f"{p}.{proj}.bias"
            if src.has(bias):
                raise SystemExit(f"{bias} exists — the engine has no bias path; refusing to drop weights")
            w = f"{p}.{proj}.weight"
            dim_out, dim_in = dims[dk_out], dims[dk_in]
            packed = ternary_weight_bytes(src, w, args.source_packing, dim_out, dim_in)
            out_tensors.append((w, "U8", [dim_out, dim_in // 4], packed))
            out_tensors.append((f"{w}_scale", "F32", [1], scalar_f32_scale(src, f"{w}_scale")))
        print(f"layer {i}: repacked")

    raw, dtype, _ = src.get("model.norm.weight")
    out_tensors.append(("model.norm.weight", "BF16", [hidden], raw_to_bf16(raw, dtype)))

    if not tie:
        raw, dtype, shape = src.get("lm_head.weight")
        head = slice_rows_bf16(raw, dtype, hidden, old_ids, shape[0])
        out_tensors.append(("lm_head.weight", "BF16", [vocab_size, hidden], head))
        print(f"lm_head.weight: untied, {len(head)} bytes BF16")

    aegis_config = {
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": hidden,
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "intermediate_size": cfg["intermediate_size"],
        "vocab_size": vocab_size,
        "max_position_embeddings": min(cfg.get("max_position_embeddings", args.max_seq), args.max_seq),
        "hidden_act": hidden_act,
        "rope_theta": float(cfg.get("rope_theta", 500000.0)),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-5)),
        "tie_word_embeddings": tie,
        "chat_template": sniff_chat_template(args),
    }
    write_model_saf(os.path.join(args.out_dir, "MODEL.SAF"), out_tensors, aegis_config)
    print("Repack complete. Next gates: loader asserts (cargo test), reference "
          "parity (tests/reference_parity.rs), then G4a PPL agreement.")


if __name__ == "__main__":
    main()
