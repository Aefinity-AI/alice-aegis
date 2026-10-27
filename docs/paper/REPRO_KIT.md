# Reproduction kit: CIS-1 identity artifacts

This kit lets you check three CIS-1 identity artifacts on your own machine:
two conformance digests and one golden receipt. It checks identity only.
It does not ask for timing, and you should not send any. The kit is pinned to
commit `d3700f0d8fd096d9428dd98a1e36ffd29874d6d4` on branch `main` of
`https://github.com/Aefinity-AI/alice-aegis.git`. The clone is anonymous. The
three M7 model artifacts (about 9 MB) are plain files in the repo. There is
no model download and no GPU. After the clone, the only network use is
cargo fetching one small crate (`libm`, a dependency of `aegis-core`) from
crates.io on the first build.

## Prerequisites

- Linux on x86_64 or aarch64.
- `git`.
- Stable Rust via rustup. CI uses `rustup toolchain install stable --profile minimal`.
- A C linker (for example `build-essential` or `gcc`).
- Disk: a few hundred MB for the `target/` directories.

Not needed: nightly Rust, QEMU, UEFI firmware, a USB stick, Python.

## Steps

Run these in order from a shell. The commands are the ones in
`.github/workflows/arm-digest.yml`. The kit builds the three examples in one
`cargo build` call where CI builds them one at a time.

```
git clone https://github.com/Aefinity-AI/alice-aegis.git && cd alice-aegis
git checkout d3700f0d8fd096d9428dd98a1e36ffd29874d6d4
cargo build --release --example cis_selftest --example cis_decode --example cis_witness --manifest-path aegis-linux/Cargo.toml
./aegis-linux/target/release/examples/cis_selftest | grep '^CIS_SELFTEST '
M=model-lab/tinybit/m7_final_gate_work/artifacts
./aegis-linux/target/release/examples/cis_decode "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" 64 "Once upon a time" | grep '^CIS_DECODE '
./aegis-linux/target/release/examples/cis_witness verify "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" tests/golden/witness_v1_m7_once64.receipt
cargo test --release --features std --manifest-path cis-verify/Cargo.toml -- --nocapture 2>&1 | grep -E '^CIS_SELFTEST |^test result'
cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- tests/golden/witness_v1_m7_once64.receipt "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN"
```

## Expected output

| Step | Expected line |
|---|---|
| `cis_selftest` | `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true` |
| `cis_decode` | `CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint` |
| `cis_witness verify` | a line that starts with `VERIFY PASS` (the source in `aegis-linux/examples/cis_witness.rs` continues it with a description of the replay) |
| `cargo test` in `cis-verify` | the same `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true` line, and `test result: ok` |
| `cis-verify` CLI | `VERIFY PASS` on its own line, after six `check:` lines that each end in `ok` |

Any other digest, `ALL_PASS=false`, or a line `VERIFY FAIL (<field>)` is a
divergence. A divergence is a finding we want reported. It is not a bug to
tune away, so please do not change the code to make it pass.

The `cis_witness` example prints `VERIFY FAIL` followed by a short reason if
the replay differs from the receipt. Report that line verbatim too.

## One-command container variant

This is the variant we use to test the kit ourselves. It runs the same
clone, checkout and commands inside a fresh container.

```
docker run --rm rust:1-bookworm bash -c 'set -eo pipefail; git clone https://github.com/Aefinity-AI/alice-aegis.git && cd alice-aegis && git checkout d3700f0d8fd096d9428dd98a1e36ffd29874d6d4 && cargo build --release --example cis_selftest --example cis_decode --example cis_witness --manifest-path aegis-linux/Cargo.toml && ./aegis-linux/target/release/examples/cis_selftest | grep "^CIS_SELFTEST " && M=model-lab/tinybit/m7_final_gate_work/artifacts && ./aegis-linux/target/release/examples/cis_decode "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" 64 "Once upon a time" | grep "^CIS_DECODE " && ./aegis-linux/target/release/examples/cis_witness verify "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" tests/golden/witness_v1_m7_once64.receipt && cargo test --release --features std --manifest-path cis-verify/Cargo.toml -- --nocapture 2>&1 | grep -E "^CIS_SELFTEST |^test result" && cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- tests/golden/witness_v1_m7_once64.receipt "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN"'
```

## What to report

Copy these into the issue:

- `uname -m`
- the CPU model line from `/proc/cpuinfo` (`grep -m1 'model name' /proc/cpuinfo`)
- your OS (`/etc/os-release`, the `PRETTY_NAME` line)
- `rustc --version` and `cargo --version`
- the commit hash (`git rev-parse HEAD`)
- the printed `CIS_SELFTEST`, `CIS_DECODE` and `VERIFY` lines, verbatim
- whether the machine is physical, a VM or a container

Do not send timing numbers (tokens per second, seconds, cycles). The paper
does not use reproducer timings. Rule A in `CLAUDE.md` requires a named
physical machine for every performance figure, and this kit measures nothing.

## Where to report

Open a GitHub issue on `Aefinity-AI/alice-aegis`:

- success: the "Reproduction report" template, title prefix `REPRODUCTION:`
- any mismatch: the "Divergence report" template, title prefix `DIVERGENCE:`

With the reporter's consent, reproductions are listed in the paper's
reproduction table as "architecture, CPU model, OS, result" only.

## Optional

These are not needed for a reproduction.

- On an ARM host, the NEON kernel tests:
  `cargo test --release --manifest-path aegis-core/Cargo.toml --test cis_neon_equivalence --test cis_neon_contract --test cis_neon_mechanism`
- The unikernel boot under QEMU (`cargo xtask boot-test`) is a correctness
  check only. It is not part of this kit.

## Kit self-test record

The kit was exercised on 2026-10-27 in a fresh `rust:1-bookworm` container on
hq (Dell XPS 15 9560, i7-7700HQ, Debian 13), from an anonymous clone of the
public URL at the pinned commit. The log is at
`docs/hardware_logs/repro_kit_docker_hq_2026-10-27.log`. Result: PASS. Both
digests and both `VERIFY PASS` lines matched the table above, and the
`cis-verify` suite reported `test result: ok`. The one-command container
variant above was run as written on the same day; its log is
`docs/hardware_logs/repro_kit_oneliner_hq_2026-10-27.log`. That log's header
names the `docker run` line by its position in this file at run time (line
60); a later edit to the Steps section moved it to line 62.
