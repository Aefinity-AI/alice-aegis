#!/usr/bin/env python3
"""E17.2 helper: digest the top-1 log-probability sequence returned by
llama.cpp's /completion server endpoint (n_probs=1).

This is NOT a raw pre-softmax logit vector — llama.cpp's server exposes only
post-softmax log-probabilities of the top-n requested candidate tokens, and
there is no stock CLI or server interface that exports raw logits. See the
job's "logit-dumping-interface finding" step for the full statement.

Usage: e17_2_logprob_digest.py <threads> <server_response.json>
"""
import hashlib
import json
import sys


def main() -> int:
    threads = sys.argv[1]
    path = sys.argv[2]
    try:
        with open(path) as f:
            data = json.load(f)
        probs = data.get("completion_probabilities", [])
        top1 = [p["probs"][0]["prob"] for p in probs if p.get("probs")]
        blob = ",".join(f"{v:.17g}" for v in top1).encode()
        digest = hashlib.sha256(blob).hexdigest()
        print(
            f"threads={threads} top1-logprob-count={len(top1)} "
            f"sha256={digest}"
        )
    except Exception as e:  # noqa: BLE001 - best-effort diagnostic, not a gate
        print(f"threads={threads} logprob digest FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
