# Packaging the verifier

Three ways to hand someone the `agent_trace` verifier without handing them a
Rust toolchain. All three run the same code and produce the same
`VERIFY PASS` / `VERIFY FAIL` verdict with the same exit code (0 / 1); a
receipt's trace chain is a property of the model artifacts and the receipt,
never of how the verifier was packaged.

| Form | Build | Runs on | Size class |
|---|---|---|---|
| Static binary | `scripts/build-static.sh` (musl, `x86_64` or `aarch64`) | any Linux of that ISA, no libc needed | single file, low MB |
| Container image | `docker build -f packaging/Dockerfile .` | any OCI runtime, `x86_64` | from-scratch, one file inside |
| Source | `cargo build --release --example agent_trace` in `aegis-linux/` | anywhere Rust stable builds | — |

The static binary is the one to copy to a phone (`aarch64`, via `adb push` or
Termux) or to an air-gapped box. Runtime CPU dispatch (AVX2 or scalar) is
unchanged by static linking, so one `x86_64` file serves both an AVX2 laptop and
a scalar Celeron.

`demo/agent-trace/run.sh` accepts `AEGIS_AGENT_TRACE_BIN=<path>` to run the
whole gen / verify / tamper demo against any prebuilt binary; the
`static-verifier` CI workflow does exactly that on `x86_64` and `aarch64` and
uploads both binaries as workflow artifacts.

What this does NOT change: the verifier still needs the three model artifact
files (`MODEL.SAF`, `EMBED.BIN`, `VOCAB.BIN`) whose SHA-256 fingerprints the
receipt pins. They are data, not code, and are distributed separately.
