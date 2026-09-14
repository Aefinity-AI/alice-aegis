# SAFE-5: the enforcement boundary this gateway now provides

Implements (a) systemd agent sandbox + (c) tool-side HMAC capability
tokens from `state/reports/2026-09-13-SAFE5-ENFORCEMENT-BOUNDARY-DESIGN.md`
(claudius-maximus), plus a persistent gateway daemon (`gateway serve`).
See that doc for (b) (folded into (a)), (d) (two-box split, not built),
and (e) (attestation, blocked — box1 has no TPM, box2's PTT has no usable
EK chain).

## What exists today

1. **`gateway serve` / `gateway_daemon`** (`src/bin/gateway_daemon.rs`): a
   persistent unix-socket process holding ONE `Gateway` for its whole
   lifetime, so freshness/nonce state and the hash-chained decision log
   are genuinely continuous across every call — fixing the per-call
   freshness reset from safe-1c (a fresh `Gateway`, and therefore
   amnesia, on every exec of the CLI binary). Proven by
   `tests/daemon_persistence_smoke.sh`: the same (receipt, action,
   session, counter) sent over two separate socket connections gets
   ALLOW then DENY-on-freshness, which is only possible with continuous
   state.
2. **`gateway::capability`** (`src/capability.rs`): single-use, expiring
   HMAC-SHA256 capability tokens (`hex(HMAC(cap_key, idx_be || 0 ||
   action_hash[32] || 0 || expiry_be))`), `idx` gateway-assigned, never
   caller-supplied, issued by the daemon on every ALLOW.
3. **Tool-side independent verification.** Each shim
   (`src/bin/tool_shim_{shell,file,http}.rs`) recomputes the action hash
   from the actual bytes about to run, verifies the token, and keeps its
   OWN fsync'd consumed-index file — a captured, still-valid token cannot
   be replayed against the tool even if never replayed against the
   gateway. No token / forged / expired / spent all get `SHIM REFUSE:
   <reason>` and nothing executes.
4. **Agent confinement (systemd).** `systemd/cm-agent-sandbox.service` /
   `systemd/agent-sandbox.service`: `RestrictAddressFamilies=AF_UNIX` +
   `PrivateNetwork=yes` block AF_INET socket creation outright (0ms
   failure, not a timeout — proven below); `CapabilityBoundingSet=`,
   `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ProtectHome=read-only`,
   `DynamicUser=yes`.
5. **AppArmor exec allow-list (when active).** Two profiles ship in this
   repo, kept both because they were validated on different boxes:
   - `apparmor/cm-agent` — denies exec under `/usr/bin`,`/usr/sbin`,
     `/bin`,`/sbin` except `python3` (an agent that runs as an
     interpreter, not through tool shims). Written on penguin;
     **AppArmor is not installed on penguin** (`apparmor_status`: not
     present, 2026-09-13), so this profile is syntax-plausible but
     unvalidated by the real parser there.
   - `systemd/agent-sandbox.apparmor` — denies exec of everything except
     the three tool shims and a `/bin/sh` sub-profile (`sh_subshell`)
     reachable only through `tool_shim_shell` after its own capability
     check has already passed. This is the profile that was actually
     loaded and enforcing: on box1, `apparmor_parser -r` succeeded and
     `aa-status` showed `agent-sandbox` + `agent-sandbox//sh_subshell`
     enforcing (see escape test 2 result below).

## Escape tests

`tests/escape/run.sh` (tests 1-4; test 5, the two-box split, is out of
scope — no forced inter-box channel exists yet).

**Run on penguin (no AppArmor installed), 2026-09-13** — see
`state/reports/2026-09-14-safe5b-enforcement-penguin.md`:
1. `curl` to an IP literal under `RestrictAddressFamilies=AF_UNIX
   PrivateNetwork=yes` — PASS, fails in 0ms (syscall-level denial).
2. `exec /bin/bash -c id` inside the same unit — exec succeeds (no
   AppArmor on penguin, a real gap); network from the exec'd process
   still fails.
3. `tool_shim_shell` with a forged token — PASS: `SHIM REFUSE: bad mac`.
4. `tool_shim_shell` with a replayed valid token — PASS: first call
   executes, second `SHIM REFUSE: already consumed`.

**Observed on aefinity-box (box1) under AppArmor, `agent-sandbox`
profile enforcing (`aa-status` showing `agent-sandbox` +
`agent-sandbox//sh_subshell`)** — real `sudo`/`systemd-run`/`aa-exec`,
not simulated:
1. `socket(AF_INET, ...) = -1 EAFNOSUPPORT` (confirmed via `strace`, not
   just an exit code).
2. `exec /bin/bash` denied — `/bin/sh: 1: /bin/bash: Permission denied`,
   exit 126 — while the profile's own explicitly allowed `/bin/sh` still
   runs, so the denial is the AppArmor profile, not a missing binary or a
   broken shell. **This is the true PASS on exec-blocking that penguin's
   run above could not demonstrate.**
3. `tool_shim_shell` with no token: `REFUSE: no capability token
   provided`; with a forged/random one: `REFUSE: capability token
   signature invalid...`.
4. A genuinely valid, single-use token (minted via
   `src/bin/cap_issue_for_test.rs`, reusing the exact `capability::issue`
   + `shim_common` action-encoding the daemon/shims use) ALLOWs and runs
   once, then is refused on replay: `REFUSE: capability token already
   consumed (idx=7) -- single-use tokens cannot be replayed`.

5. box1 reaching box2's credential store outside the forced channel — not
   runnable, design item (d) not built on this branch.

## Not enforced (honest residual limits)

- **A compromised gateway host**: holds both HMAC keys (allowlist-signing
  and capability-signing) in plaintext at startup; root or process
  compromise there forges any token — no real-time detection
  (attestation, design item (e), blocked on TPM). This is the single
  point of failure the two-box split (design doc (d), not built) is meant
  to reduce, by moving those keys off the box where agent-controlled or
  model-generated content lives.
- **Root on this box in general.** `DynamicUser`/`CapabilityBoundingSet=`
  /AppArmor are LSM/systemd mediation on top of the existing kernel and
  root account; root can unload the AppArmor profile, edit the unit, or
  read the gateway's key files directly. Nothing here is a hypervisor- or
  TPM-rooted boundary.
- **Exec allow-listing without AppArmor active.** Plain systemd
  sandboxing does NOT stop exec of a binary already on the process's own
  PATH at its own privilege — that requires the AppArmor layer above,
  which must actually be installed and loaded (true on box1, not true on
  penguin as of 2026-09-13).
- **A tool reachable outside its shim**: every real tool entry point must
  be wrapped, or this layer simply does not apply to it. Today that is
  shell, file-write, and HTTP-egress only.
- **The shim's consumed-index store is not race-safe.** `shim_common::
  mark_consumed` is a plain file append with no locking. Two concurrent
  invocations of the same shim racing on the same consumed-index file
  could both read "not yet consumed" and both act before either write
  lands — a real gap for concurrent deployment, not yet closed.
- **Action-byte-encoding agreement is an integration responsibility.**
  The daemon issues a capability token bound to whatever `action_bytes`
  the caller passed to `decide()`; each shim independently recomputes its
  own action hash from its own encoding of the tool call
  (`gateway::shim_common::{shell,file,http}_action`). A mismatch fails
  closed (`BadMac`), but the daemon and shim call sites must be written
  by the same integration code with the same encoding — not built/tested
  end-to-end here.
- **Two-box split (d) and attestation (e)** are not built on this branch;
  escape test 5 has no forced inter-box channel yet.
- **Supply-chain compromise** of the gateway/shim binaries or
  aegis-core's sha256/HMAC.
- **AppArmor is LSM-level mediation, not a hardware boundary**, and on
  box1 the kernel audit trail (`dmesg`/`journalctl` `DENIED` record) for
  the exec that was actually denied was not observed (the denial itself —
  verbatim `Permission denied`, exit 126 — was observed and is real; the
  AUDIT line was not, possibly due to nested virtualization). Confirm
  audit logging before relying on it for detection, not just enforcement.

## What the public doc may honestly claim after this pass

"A rogue process on a box with this layer installed and AppArmor active —
even one reachable only through the sandboxed agent unit — cannot invoke
shell, file-write, or HTTP-egress tool actions without a gateway-issued,
single-use, expiring capability token that the tool shim independently
verifies; AF_INET egress is refused by the kernel before the request
would even reach a tool, and (where AppArmor is loaded, as demonstrated
on box1) exec of anything but the tool shims is refused by the LSM before
it can even reach a shim." It must NOT claim protection against a
compromised gateway host, must NOT claim a race-safe consumed-token store
under concurrent load, must NOT claim exec-blocking on a box without
AppArmor active (penguin, as of 2026-09-13), and must NOT claim
end-to-end integration between a real agent loop and these shims has been
built or tested (only the shims' own capability-verification logic has
been).

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
# only if AppArmor is active on this box (validated so far on box1 with
# systemd/agent-sandbox.apparmor; apparmor/cm-agent is untested on penguin):
sudo apparmor_parser -r systemd/agent-sandbox.apparmor && sudo aa-status
# or:
sudo cp apparmor/cm-agent /etc/apparmor.d/cm-agent
sudo apparmor_parser -r /etc/apparmor.d/cm-agent && sudo aa-enforce cm-agent
```
