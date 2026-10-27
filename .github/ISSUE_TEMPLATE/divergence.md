---
name: Divergence report
about: Report a mismatch while running docs/paper/REPRO_KIT.md
title: "DIVERGENCE: <arch> / <CPU model> / <which step>"
labels: divergence
---

A divergence is treated as a finding. It will be recorded in this repo's
ledger, with credit to you if you want it. Please do not change the code to
make the step pass.

- [ ] `uname -m`:
- [ ] CPU model line from `/proc/cpuinfo`:
- [ ] OS (`PRETTY_NAME` from `/etc/os-release`):
- [ ] `rustc --version`:
- [ ] `cargo --version`:
- [ ] Commit hash (`git rev-parse HEAD`):
- [ ] Machine type (physical, VM or container):
- [ ] Which step diverged:
- [ ] Full verbatim output of that step (paste below):
- [ ] The line you expected, from the Expected output table:
- [ ] Full build log attached (`cargo build` output, as a file):

Output of the diverging step:

```
```

Credit:

- [ ] Name or handle to credit in the ledger (leave blank for anonymous):
- [ ] You may list this report (arch, CPU, OS, result) in the paper.
