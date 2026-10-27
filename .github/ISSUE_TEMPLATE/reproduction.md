---
name: Reproduction report
about: Report a successful run of docs/paper/REPRO_KIT.md
title: "REPRODUCTION: <arch> / <CPU model> / <commit prefix>"
labels: reproduction
---

Thank you for running the kit. Identity artifacts only. Please do not add timing numbers.

- [ ] `uname -m`:
- [ ] CPU model line from `/proc/cpuinfo`:
- [ ] OS (`PRETTY_NAME` from `/etc/os-release`):
- [ ] `rustc --version`:
- [ ] `cargo --version`:
- [ ] Commit hash (`git rev-parse HEAD`):
- [ ] Machine type (physical, VM or container):
- [ ] `CIS_SELFTEST` line, verbatim:
- [ ] `CIS_DECODE` line, verbatim:
- [ ] `VERIFY PASS` line from `cis_witness verify`, verbatim:
- [ ] `VERIFY PASS` line from the `cis-verify` CLI, verbatim:

Consent:

- [ ] You may list this reproduction (arch, CPU, OS, result) in the paper.
