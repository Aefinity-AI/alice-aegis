#!/usr/bin/env python3
"""M3 parity comparison: engine vs transformers reference MC results on
identical token ids. PASS if |acc_engine - acc_ref| <= 5 raw points AND
|acc_norm_engine - acc_norm_ref| <= 5 raw points.

Usage: mc_compare.py <engine_results.jsonl> <ref_results.jsonl>
Exit code 0 on PASS, 1 on FAIL/mismatch.
"""
import json
import sys

GATE_POINTS = 5.0


def load(path):
    rows, summary = [], None
    for line in open(path):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("summary"):
            summary = d
        elif "id" in d:
            rows.append(d)
    if summary is None:
        raise SystemExit(f"{path}: no summary line — run incomplete")
    return rows, summary


def main():
    eng_path, ref_path = sys.argv[1], sys.argv[2]
    eng, eng_sum = load(eng_path)
    ref, ref_sum = load(ref_path)
    if len(eng) != len(ref):
        raise SystemExit(f"item count mismatch: engine {len(eng)} vs ref {len(ref)}")

    max_abs = 0.0
    max_rel = 0.0
    max_abs_id = max_rel_id = ""
    sum_abs = 0.0
    n_nll = 0
    disagree_raw = []
    disagree_norm = []
    per_item = []
    for e, r in zip(eng, ref):
        assert e["id"] == r["id"], f"id order mismatch {e['id']} vs {r['id']}"
        assert e["choice_cont_tokens"] == r["choice_cont_tokens"], e["id"]
        item_max = 0.0
        for ve, vr in zip(e["choice_nll"], r["choice_nll"]):
            d = abs(ve - vr)
            rel = d / abs(vr) if vr != 0 else float("inf")
            sum_abs += d
            n_nll += 1
            item_max = max(item_max, d)
            if d > max_abs:
                max_abs, max_abs_id = d, e["id"]
            if rel > max_rel:
                max_rel, max_rel_id = rel, e["id"]
        per_item.append((item_max, e["id"]))
        if e["pred_raw"] != r["pred_raw"]:
            disagree_raw.append(e["id"])
        if e["pred_norm"] != r["pred_norm"]:
            disagree_norm.append(e["id"])

    d_acc = abs(eng_sum["acc"] - ref_sum["acc"]) * 100
    d_norm = abs(eng_sum["acc_norm"] - ref_sum["acc_norm"]) * 100
    ok = d_acc <= GATE_POINTS and d_norm <= GATE_POINTS

    print("--- M3 parity diff table (engine vs transformers reference) ---")
    print(f"n items: {len(eng)} | per-choice NLL values compared: {n_nll}")
    print(f"per-choice NLL: max abs diff {max_abs:.6f} nats (item {max_abs_id})")
    print(f"per-choice NLL: max rel diff {max_rel*100:.3f}% (item {max_rel_id})")
    print(f"per-choice NLL: mean abs diff {sum_abs/n_nll:.6f} nats")
    top = sorted(per_item, reverse=True)[:5]
    print("largest per-item NLL diffs: " + ", ".join(f"{i}={d:.4f}" for d, i in top))
    print(f"pred_raw disagreements : {len(disagree_raw)}/{len(eng)} {disagree_raw}")
    print(f"pred_norm disagreements: {len(disagree_norm)}/{len(eng)} {disagree_norm}")
    print(
        f"acc      : engine {eng_sum['acc']:.4f} vs ref {ref_sum['acc']:.4f} "
        f"-> |delta| {d_acc:.2f} points (gate {GATE_POINTS})"
    )
    print(
        f"acc_norm : engine {eng_sum['acc_norm']:.4f} vs ref {ref_sum['acc_norm']:.4f} "
        f"-> |delta| {d_norm:.2f} points (gate {GATE_POINTS})"
    )
    print(f"=== M3 VERDICT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
