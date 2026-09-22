# SAFE-5 escape tests

Per `state/reports/2026-09-13-SAFE5-ENFORCEMENT-BOUNDARY-DESIGN.md`'s "5
escape tests". Test 5 (box1 reaching box2's credential store outside the
forced inter-box channel) needs a second box and is out of scope here —
tracked as a future QUEUE item (the two-box split, design doc (d), is not
built yet either).

Run all four from `demo/agent-trace/gateway/`:

```
bash tests/escape/03_forged_token.sh
bash tests/escape/04_replayed_token.sh
sudo bash tests/escape/01_no_af_inet.sh
sudo bash tests/escape/02_no_exec_outside_shims.sh
```

1/2 need `sudo` (transient `systemd-run`/`aa-exec` with the sandbox
properties) and, for #2, the AppArmor profile loaded:
`sudo apparmor_parser -r systemd/agent-sandbox.apparmor`.

Each script prints PASS/FAIL and the verbatim refusal text observed; see
`state/reports/2026-09-13-safe5b-enforcement-box1.md` (claudius-maximus
repo) for a copy of an actual run's output.
