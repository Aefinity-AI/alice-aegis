# SAFE-5: enforcement boundary

Implements (a) systemd agent sandbox + (c) tool-side HMAC capability
tokens from `state/reports/2026-09-13-SAFE5-ENFORCEMENT-BOUNDARY-DESIGN.md`
(claudius-maximus), plus a persistent gateway daemon (`gateway serve`).

## Enforced

- **Verify -> execute -> capability chain.** `gateway serve` keeps ONE
  `Gateway` alive for the process's life (fixes the per-call freshness
  reset from safe-1c). On ALLOW it mints a single-use, expiring
  HMAC-SHA256 capability token (`src/capability.rs`): `hex(HMAC(cap_key,
  idx_be || 0 || action_hash[32] || 0 || expiry_be))`, `idx` gateway-
  assigned, never caller-supplied.
- **Tool-side independent verification.** Each shim
  (`src/bin/tool_shim_{shell,file,http}.rs`) recomputes the action hash
  from the actual bytes about to run, verifies the token, and keeps its
  OWN fsync'd consumed-index file — a captured, still-valid token cannot
  be replayed against the tool even if never replayed against the
  gateway. No token / forged / expired / spent all get `SHIM REFUSE:
  <reason>` and nothing executes.
- **Agent confinement (systemd).** `systemd/cm-agent-sandbox.service`:
  `RestrictAddressFamilies=AF_UNIX` + `PrivateNetwork=yes` block AF_INET
  socket creation outright (proven below: 0ms failure, not a timeout);
  `CapabilityBoundingSet=`, `NoNewPrivileges=yes`, `ProtectSystem=strict`,
  `ProtectHome=read-only`, `DynamicUser=yes`.
- **AppArmor exec allow-list (when active).** `apparmor/cm-agent` denies
  exec under `/usr/bin`,`/usr/sbin`,`/bin`,`/sbin` except `python3`.

## Not enforced (honest residual limits)

- **A compromised gateway host**: holds both HMAC keys in plaintext at
  startup; root or process compromise there forges any token — no
  real-time detection (attestation, design item (e), blocked on TPM).
- **Exec allow-listing without AppArmor**: plain systemd sandboxing does
  NOT stop exec of a binary already on the process's own PATH at its own
  privilege. **AppArmor is not installed on penguin** (`apparmor_status`:
  not present, 2026-09-13). Escape test 2 shows this honestly.
- **A tool reachable outside its shim**: every real tool entry point must
  be wrapped, or this layer simply does not apply to it.
- **Two-box split (d) and attestation (e)** are not built on this branch;
  escape test 5 has no forced inter-box channel yet.
- **Supply-chain compromise** of the gateway/shim binaries or
  aegis-core's sha256/HMAC.

## Escape tests (run 2026-09-13 on penguin, `tests/escape/run.sh`; verbatim
output: `state/reports/2026-09-14-safe5b-enforcement-penguin.md`)

1. `curl http://example.com` inside `RestrictAddressFamilies=AF_UNIX
   PrivateNetwork=yes` — PASS, fails in 0ms (syscall-level denial).
2. `exec /bin/bash -c id` inside the same unit — exec succeeds (no
   AppArmor on penguin, a real gap); network from the exec'd process
   still fails. Re-run on box1/box2 with `apparmor/cm-agent` loaded for a
   true PASS on exec-blocking.
3. `tool_shim_shell` with a forged token — PASS: `SHIM REFUSE: bad mac`.
4. `tool_shim_shell` with a replayed valid token — PASS: first call
   executes, second `SHIM REFUSE: already consumed`.
5. box1 reaching box2's credential store outside the forced channel — not
   runnable, design item (d) not built on this branch.

## Install (one box)

```
cd demo/agent-trace/gateway && cargo build --release
install -d -m 700 ~/.config/cm-gateway
head -c 32 /dev/urandom > ~/.config/cm-gateway/allowlist.key
head -c 32 /dev/urandom > ~/.config/cm-gateway/cap.key
chmod 600 ~/.config/cm-gateway/{allowlist,cap}.key
# build+sign allowlist.signed: "model_sha256 embed_sha256 vocab_sha256"
# + "sig <hmac-hex>" (see sign_allowlist/load_allowlist, src/main.rs)
mkdir -p ~/.config/systemd/user
cp systemd/cm-gateway.service systemd/cm-agent-sandbox.service \
  ~/.config/systemd/user/   # edit cm-gateway.service ExecStart paths
systemctl --user daemon-reload && systemctl --user enable --now cm-gateway.service
# only if AppArmor is active on this box:
sudo cp apparmor/cm-agent /etc/apparmor.d/cm-agent
sudo apparmor_parser -r /etc/apparmor.d/cm-agent && sudo aa-enforce cm-agent
```
