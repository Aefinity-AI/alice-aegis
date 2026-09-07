# packaging/Dockerfile — the agent_trace verifier as a from-scratch image.
# The image contains exactly one file: the static musl binary built by
# scripts/build-static.sh. Mount the three model artifacts and the receipt
# read-only and run `verify`:
#
#   docker build -f packaging/Dockerfile -t alice-verify .
#   docker run --rm -v "$PWD/artifacts:/a:ro" -v "$PWD/receipt.txt:/r.txt:ro" \
#       alice-verify verify /a/MODEL.SAF /a/EMBED.BIN /a/VOCAB.BIN /r.txt
#
# Exit 0 = VERIFY PASS, exit 1 = VERIFY FAIL. No network, no shell, no libc.
#
# Stage 1 builds on the official Rust image (Debian) with the musl target;
# stage 2 copies the single binary into an empty root.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN rustup target add x86_64-unknown-linux-musl \
 && cd aegis-linux \
 && cargo build --release --target x86_64-unknown-linux-musl --example agent_trace

FROM scratch
COPY --from=build /src/aegis-linux/target/x86_64-unknown-linux-musl/release/examples/agent_trace /agent_trace
ENTRYPOINT ["/agent_trace"]
