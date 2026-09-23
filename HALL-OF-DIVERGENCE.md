# Hall of Divergence

The public ledger for the falsification challenge in
[`CHALLENGE.md`](CHALLENGE.md).

**Credit only. There is no cash offer attached to this challenge.** What you
get is your name, permanently, on the record of a published negative result —
and, if you want it, co-authorship discussion on the write-up that follows.

Two things are recorded here:

- **Divergences** — a reproducible run of an unmodified conforming build that
  produces a different digest, or a witness `VERIFY FAIL`, on real hardware.
  First finder per *distinct root cause* is credited.
- **Reproductions** — a run on a machine, OS, ISA or toolchain not already in
  the table that reproduces the pinned digests exactly. Negative results for
  the challenge, positive results for the spec; both are worth publishing.

## How to be listed

Open an issue on this repo titled `DIVERGENCE:` or `REPRODUCTION:` with the
full log, `uname -a`, `/proc/cpuinfo` flags (or the ARM equivalent), the
compiler version, and the exact build commands you ran. Say how you want to be
credited — real name, handle, or affiliation. If you would rather not be
listed at all, say so and the finding is published without your name.

## Divergences

| date | who | machine / ISA | root cause | status | report |
| ---- | --- | ------------- | ---------- | ------ | ------ |
| _(none recorded yet)_ | | | | | |

## Reproductions

| date | who | machine / ISA | digests reproduced | report |
| ---- | --- | ------------- | ------------------ | ------ |
| _(none recorded from outside the project yet)_ | | | | |

In-project reproductions — the ones this challenge is asking a stranger to
break — are listed in `CHALLENGE.md` and in the CIS-2 conformance dataset:
<https://huggingface.co/datasets/aefinityAIINC/cis2-conformance>

---

A divergence that turns out to be a **spec hole** — the spec text permits two
readings that produce different bits — is the most valuable outcome here, not
the least. It means the document is not yet sufficient, and sufficiency is the
entire claim.
