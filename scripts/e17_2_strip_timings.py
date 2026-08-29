#!/usr/bin/env python3
"""E17.2 Rule A compliance: strip the "timings" object (prompt_ms,
predicted_ms, tokens-per-second, etc.) that llama.cpp's /completion
server endpoint embeds in every response, before the JSON is persisted
to disk, hashed, or uploaded as a CI artifact. Rule A: this job must
never record or print timing numbers (CI hardware is shared/unnamed).

Usage: e17_2_strip_timings.py <server_response.json>
Edits the file in place; a missing/unparseable file is a no-op (the
caller's own error handling covers that case).
"""
import json
import sys

path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    sys.exit(0)

removed = data.pop("timings", None) is not None
with open(path, "w") as f:
    json.dump(data, f)

print(f"stripped timings key from {path}: {removed}")
