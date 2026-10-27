#!/usr/bin/env bash
# Runs inside a fresh rust:1-bookworm container: anonymous clone of the PUBLIC repo at the pinned
# main commit, then the five identity checks from .github/workflows/arm-digest.yml. Identity only:
# no timing figure is recorded (CLAUDE.md Rule A).
set -euo pipefail
PIN=d3700f0d8fd096d9428dd98a1e36ffd29874d6d4
echo "REPRO_KIT_SELFTEST start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "container image: ${IMAGE_TAG:-unknown}; host label: ${HOST_LABEL:-unknown}"
uname -m; grep -m1 -E 'model name|CPU part' /proc/cpuinfo || true
rustc --version; cargo --version; git --version
git clone -q https://github.com/Aefinity-AI/alice-aegis.git /work
cd /work && git checkout -q "$PIN" && echo "checked out $(git rev-parse HEAD)"
echo "== step 1-3: build the three aegis-linux examples"
cargo build --release --example cis_selftest --example cis_decode --example cis_witness --manifest-path aegis-linux/Cargo.toml 2>&1 | tail -3
echo "== step 1: op-level digest"
./aegis-linux/target/release/examples/cis_selftest | grep '^CIS_SELFTEST '
M=model-lab/tinybit/m7_final_gate_work/artifacts
echo "== step 2: token-level decode digest"
./aegis-linux/target/release/examples/cis_decode "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" 64 "Once upon a time" | grep '^CIS_DECODE '
echo "== step 3: golden receipt verify (in-repo cis_witness)"
./aegis-linux/target/release/examples/cis_witness verify "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" tests/golden/witness_v1_m7_once64.receipt
echo "== step 4: standalone cis-verify test suite"
cargo test --release --features std --manifest-path cis-verify/Cargo.toml -- --nocapture 2>&1 | grep -E '^CIS_SELFTEST |^test result'
echo "== step 5: standalone cis-verify CLI on the golden receipt"
cargo run -q --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- tests/golden/witness_v1_m7_once64.receipt "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" 2>&1 | grep -E '^CIS_SELFTEST |VERIFY (PASS|FAIL)'
echo "REPRO_KIT_SELFTEST end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
