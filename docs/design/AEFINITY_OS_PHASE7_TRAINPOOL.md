# AEFINITY OS phase 7 — TRAIN-POOL (design, no code)

*Draft v0.1 — 2026-09-14. Design-only scoping tick. Extends
`program/AEFINITY_OS_FLEET_DESIGN.md` §5–§9 (phase 6, POOL) without changing
any shipped behaviour there. Written against the phase 6 code actually in
the fleet — `bin/cm-os-pool` (1433 LOC, worktree
`claudius-maximus-pool` branch `cm/aefinity-pool`) and its normative spec,
`program/AEFINITY_OS_FLEET_DESIGN.md` §5 — not the spec prose alone.
Phase 6 explicitly deferred all of phase 7 (§9: "all of phase 7 TRAIN-POOL"),
citing "no embarrassingly parallel workload needs [box-to-box reduction],
and it is untestable on a 6 GB dev box without two concurrent guests." This
doc is the design for actually doing it.*

## 0. One paragraph

Phase 6's `cm-os-pool` dispatches independent, embarrassingly-parallel
**units** (`EVAL`/`VERIFY`/`MEMBW`/`CPUID` job fragments) to boxes with no
relationship between one box's work and another's — a box can finish, fail,
or be re-leased to a different box without any other box knowing or caring.
Phase 7 (TRAIN-POOL) is qualitatively different: a torch DDP training job
needs a **fixed set of participants who all start together, all agree on
addresses before the first step, and all stay up for the duration of every
step** — the opposite of phase 6's shard-and-retry model. Phase 7 therefore
does **not** extend the wire protocol or the unikernel (this is
Debian-side, per the task) — it adds a *new dispatch mode* to the host
control plane that reuses phase 6's box health-tracking and box-liveness
machinery but replaces per-unit leasing with a one-shot, all-or-nothing
rendezvous, because DDP has no notion of "re-lease this rank to a
different box mid-job" the way an EVAL shard does.

## 1. What phase 7 adds on top of phase 6

Phase 6 treats a **unit** as an opaque `JOB.TXT` body sent to *some*
eligible unikernel box on the LAN, dispatched independently, retried
independently, and reconciled after the fact by `run_id`/digest (§5 of the
fleet design; `Pool.lease_next`, `bin/cm-os-pool:606`). None of that model
survives contact with DDP:

- DDP's `torch.distributed.init_process_group` is a **barrier**: every
  rank blocks until `world_size` processes have registered with the
  rendezvous point. There is no "lease this rank to a different box on
  timeout" — if one rank never shows up, *no* rank starts.
- The workload is **Debian-side Python/torch processes**, not `JOB.TXT`
  run by the unikernel. `cm-os-pool` today only ever talks to a box's
  AEFINITY-OS TCP listener (port 4242, the §1.2 verb protocol) — it has no
  code path that launches or supervises a process on a box's Debian side
  at all. That is a **gap**, not an extension: phase 7 needs a new
  "run this shell command on box X's Debian side and stream its exit
  status back" primitive that does not exist anywhere in this fleet's
  host tooling today (`cm-os-fs`, `cm-os-job`, `cm-os-pool` all speak only
  to the unikernel listener). The closest existing thing is plain `ssh`
  from `penguin`/`aefinity-box` — phase 7's dispatcher is realistically
  "an orchestration layer over `ssh` to each box's Debian side," not a new
  wire verb.
- Once running, gradient sync between the box processes is **torch's own
  job** (NCCL/gloo `all_reduce` under the hood) — phase 7 adds *no*
  new reduction protocol. Phase 6's box-to-box reduction gap (explicitly
  named as deferred in §9) is filled by torch.distributed, not by
  anything in this repo. This is the load-bearing design choice: **phase
  7 is a thin launcher around a fixed, well-tested distributed-training
  stack, not a reimplementation of it.**

What phase 7 *reuses* from phase 6, concretely:
- `parse_boxes_for_pool` / `state/BOXES.md` — the fleet's box inventory
  (name, host, port) is the natural source of "which boxes are DDP
  candidates," even though DDP doesn't use the unikernel port at all.
- The box health state machine (`Pool.record_box_success` /
  `record_box_failure`, HEALTHY→SUSPECT→DOWN→RECOVER→HEALTHY,
  `bin/cm-os-pool:535-578`) — reused as a **pre-flight gate**: a box that
  is `SUSPECT` or `DOWN` on the unikernel side is refused as a DDP rank
  candidate even though DDP doesn't touch the unikernel, because a box
  flaky enough to be `SUSPECT` on port 4242 is not a box you want holding
  a DDP rank open for an hour (see §3, box1's 09-13/09-14 off-LAN
  incident).
- The "**hard stop before a run starts**" pattern from the artifacts guard
  (`run_pool`, `bin/cm-os-pool:979-1018`, "Pre-flight artifacts guard
  (§5: 'a hard stop before a pool run starts')") — phase 7's analogue is a
  pre-flight rendezvous check (§2) that refuses to launch any rank until
  every box has confirmed reachability and a matching torch/CUDA-or-CPU
  environment, rather than discovering a mismatch after some ranks have
  already started training.
- The plan-as-unit-of-reproducibility idea (§5 "The plan is the unit of
  reproducibility") — phase 7's plan additionally has to pin a random
  seed, `WORLD_SIZE`, and the rank→box assignment, because unlike an EVAL
  unit, DDP's numerical result is affected by which boxes hold which
  ranks (different CPUs/GPUs, different reduction order under async
  collectives).

What phase 7 **does not** reuse, because phase 6 doesn't have it: per-unit
leasing (`lease_next`), retry/backoff, and `RUNID`-based replay
deduplication are all built around independent, restartable, idempotent
units. A DDP rank is none of those — restarting one rank mid-training
either needs full DDP elastic/fault-tolerance support (torchrun's
`--rdzv-backend c10d` restart semantics) or the whole job dies. Phase 7
should **not** try to bend `lease_next`/RUNID onto ranks; it needs its own,
much simpler "all ranks up or job doesn't start" gate.

## 2. Interface phase 7's dispatcher needs from phase 6

`torch.distributed.init_process_group` (env:// init method) needs each box
process to have four things in its environment before calling it:

| var | meaning | phase 7 source |
|---|---|---|
| `MASTER_ADDR` | IP/hostname of rank 0 | pool plan or `state/BOXES.md` lookup for the box assigned rank 0 |
| `MASTER_PORT` | a free TCP port on the rank-0 box, **not** 4242 (that's the unikernel listener; conflating them would be a real bug) | a new fixed constant for phase 7, e.g. `TRAINPOOL_PORT` (TBD — needs to not collide with 4242 or the collector's 8787) |
| `RANK` | this process's global rank, `0..WORLD_SIZE-1` | assigned by the dispatcher from the plan's box order |
| `WORLD_SIZE` | total participant count | `len(plan.boxes)`, fixed for the job's lifetime (no elastic resize in scope) |

None of `MASTER_ADDR`/`MASTER_PORT`/`RANK`/`WORLD_SIZE` exist anywhere in
phase 6's plan schema (`load_plan`, `bin/cm-os-pool:89-151` — fields are
`pool_id`, `budget_s`, `units[].id/body`, `artifacts_expect`) or in the
unikernel's `JOB.TXT` grammar (§2 of the fleet design — `TOKEN`, `RUNID`,
`TAG`, `SHARD i/n`, `SEED`, `STRICT`, `CPUID`, `VERIFY`, `EVAL`, `MEMBW`,
`MECH`). **Gap**: phase 7 needs a new plan schema (not `PLAN.json`'s
`units` list — a `ddp_job` shape: `{pool_id, boxes: [name,...], budget_s,
master_port, script, args, seed}`) and a new dispatcher (`cm-ddp-launch` or
similar) that is a sibling to `cm-os-pool`, not a mode inside it, because
the two have almost no code in common below the box-health/box-inventory
layer.

Sequencing the dispatcher needs to enforce, since `init_process_group` is a
blocking barrier with no independent per-rank retry:
1. Resolve `boxes[0]`'s LAN IP from `state/BOXES.md` as `MASTER_ADDR`.
2. Confirm every box in `boxes` is `HEALTHY` on the phase-6 health table
   (reusing `Pool.eligible_boxes`-style logic) — refuse to launch if any
   is `SUSPECT`/`DOWN`.
3. SSH-launch (or equivalent — see §1's gap) rank 0 first, wait for it to
   report the rendezvous port is listening, **then** launch ranks
   1..N-1 in parallel with `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`
   set.
4. If any rank's launch fails or any rank doesn't reach
   `init_process_group` within a timeout, kill all ranks and report
   `FAILED` — there is no partial-success state for a DDP job the way
   phase 6 has `PARTIAL` for a EVAL unit's `job.N.partial=k`.

## 3. Failure modes specific to training

Training failure modes are worse than phase 6's because DDP has **no
partial-credit state**: phase 6's `job.N.partial=k>0` (fleet design §5,
"Partial results") lets an EVAL unit be scored on however many items it
got through before a budget expired. DDP has no equivalent — if any rank
drops mid-`all_reduce`, every other rank hangs on that collective (NCCL/
gloo do not fail fast by default) or the whole group dies, and any
gradient state not checkpointed is lost.

This fleet has **already produced the exact event class** that makes this
matter, not a hypothetical: `state/BLOCKERS.md` line 98 / `state/LOG.md`
around `2026-09-14 01:18` —

> `aefinity-box (box1) OFF LAN since ~2026-09-13 22:37Z (no ping on
> .20/.64, no GitHub push since 22:04Z as of 01:16Z 09-14) — not a
> transient Wi-Fi drop this time.` Resolved 04:10Z: "box1 rebooted...
> journal stops abruptly at 22:48:05Z with pkg temp 48C, no OOM/panic/
> thermal lines... sudden power loss or hard freeze, NOT Wi-Fi."

That is ~5.5 hours a box was unreachable, unattended, and the operator
(Justin) was away — exactly the situation a multi-hour DDP job would be
running through. If box1 held rank 1 of a 2-box DDP job at 22:37Z, the
whole job would have hung indefinitely (default NCCL/gloo timeouts are
often 30 min but can be configured much longer, and a hang is silent
unless something is watching) rather than failing loudly. Design
consequences:

- **No unattended multi-hour DDP job without a watchdog.** Phase 7 needs
  its own liveness poll (SSH heartbeat or a lightweight callback) with a
  **hard wall-clock timeout** shorter than "someone notices the box is
  gone," and on timeout it must kill *all* ranks, not just the dead one —
  matches phase 6's `--reboot-on-quarantine off by default` principle
  (fleet design §5): the dispatcher observes and reports, it does not try
  to power-cycle or auto-heal a box.
- **Partial epoch state is not useful without checkpointing**, which is
  out of scope for the smoke-test step (§4) and needs its own design once
  real training is in scope (§5).
- **A box dropping off-LAN mid-collective is indistinguishable from a
  slow box** from the dispatcher's vantage point until the timeout fires
  — same ambiguity phase 6 handles for EVAL via lease deadlines, but DDP
  has no cheap "just re-lease it" recovery, so the only safe action on
  timeout is to fail the whole job and report which rank/box was last
  seen, for a human to re-launch.
- **This is a two-box fleet in practice** (box1, box2, occasionally a dev
  box) — `WORLD_SIZE=2` is the realistic first target, and a single box
  going down is a **100% job failure**, not a degraded run. That materially
  changes the cost/benefit of chasing DDP fault tolerance (torchrun
  elastic mode) versus just re-running: worth flagging to Justin as an
  open question (§6) rather than assumed.

## 4. Minimal smoke-test plan (first concrete implementation step)

Before any training code: prove `torch.distributed` can form a 2-box group
over this fleet's actual LAN and produce an identical result on both
sides. Concretely, as the first phase-7 PR:

1. A ~15-line script (`scripts/trainpool/allreduce_smoke.py`, not yet
   written) that: reads `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT`
   from env, calls `init_process_group(backend="gloo")` (CPU-only —
   neither box in this fleet is confirmed to have a GPU/NCCL path; start
   with gloo, which needs none), builds a small fixed tensor (e.g.
   `torch.arange(16, dtype=torch.float64)`), calls `all_reduce(tensor,
   op=SUM)`, and prints the resulting tensor's bytes plus a sha256 of
   them.
2. Launch by hand first (plain `ssh box1 ...` / `ssh box2 ...` with the
   four env vars set manually) — **no dispatcher yet** — to validate the
   rendezvous mechanics work at all on this LAN before automating
   anything.
3. Pass criterion: both boxes print the **byte-identical** sha256 of the
   reduced tensor. This is the DDP-world analogue of phase 6's `AGREE`
   check (fleet design §5) — same "two boxes must produce the same bytes"
   discipline this whole program is built on, applied to a collective
   instead of an EVAL digest.
4. Only after that passes by hand: write the actual dispatcher
   (`cm-ddp-launch`, §2) to automate steps 1–2, with the pre-flight health
   gate from §2 step 2.
5. Only after the dispatcher passes the smoke test automated: consider a
   real training loop at all (out of scope here, §5).

This mirrors phase 6's own sequencing discipline (fleet design §10: gate
green and pasted into the PR before the next step) — smoke test before
plumbing, plumbing before any real workload.

## 5. Explicitly out of scope for phase 7

- **No unikernel training.** The unikernel (`aegis-uefi`) is a stateless
  inference worker (fleet design §0's core principle) with no `no_std`
  autograd story and no plan to build one; DDP runs entirely on Debian.
- **No new model.** This repo's training/finetuning setup, if any exists
  today, was not found by this scoping pass (`modeling_bitnet.py` at repo
  root is a HF-style model definition for CIS-1 quantized inference, not
  a training script — no `optimizer.step()`/loss-backward loop was found
  in this repo). **Model choice is deferred** to whoever scopes actual
  training — phase 7 as designed here is model-agnostic: it launches
  *some* torch script under DDP, and the smoke test (§4) doesn't need a
  model at all.
- **No GPU/NCCL path assumed.** Start CPU/gloo (§4); GPU support is a
  later decision contingent on what hardware the fleet actually has,
  which this scoping pass did not verify.
- **No elastic/fault-tolerant DDP** (torchrun `--rdzv-backend c10d` restart
  semantics, checkpoint-and-resume mid-job). Flagged as a real question
  for a 2-box fleet (§3, §6) but not designed here.
- **No new wire protocol / unikernel change.** Phase 7 is 100% Debian-side
  host tooling, per the task and per fleet design §9's own framing of
  phase 7 as deferred Debian-side work.
- **No autoscaling / box discovery beyond `state/BOXES.md`.** Rank
  assignment is a fixed, explicit list in the phase-7 plan, not
  auto-negotiated.

## 6. Open questions for Justin

1. **Is 2-box DDP worth building at all**, given §3's finding that a
   2-participant job has zero fault tolerance without elastic mode (any
   single box outage = 100% job failure), and this fleet has had a
   5.5-hour unattended box outage in the last 24 hours (§3)? Is the goal
   throughput (two boxes faster than one) or just proving the mechanism
   works, given Kaggriculture policy says "Kaggle notebooks are the
   compute, not this box" (CLAUDE.md)?
2. **What actually gets trained?** No training script/target model was
   found in this repo during this pass (§5) — is there one elsewhere
   (`alice-model*` worktrees?), or is phase 7 purely infrastructure ahead
   of a not-yet-chosen workload?
3. **CPU-only (gloo) or is there GPU hardware** on any box worth wiring
   NCCL for? Changes the smoke test and the whole cost model.
4. **New port for `MASTER_PORT`/rendezvous** — any fleet convention to
   follow (4242 is the unikernel listener, 8787 is the collector; needs a
   third, unclaimed one)?
5. **Launch mechanism**: is plain `ssh` from the dispatcher box acceptable
   as phase 7's "run this on box X's Debian side" primitive, or does this
   need something more robust (a small daemon, systemd units per box) —
   given `cm-os-pool` today has *no* Debian-side execution primitive at
   all (§1), this is a real gap to fill, not a detail.

---

*Gaps found in phase 6 relative to what phase 7 needs, summarized: (a) no
Debian-side "run a command on box X" primitive exists in any host tool
(`cm-os-fs`/`cm-os-job`/`cm-os-pool` only speak the unikernel's port-4242
protocol); (b) phase 6's plan schema and `JOB.TXT` grammar have no fields
for `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` — a new plan shape and
dispatcher are needed, not an extension of `load_plan`; (c) phase 6's
leasing/retry/RUNID-replay model is built for independent, restartable
units and does not fit DDP's all-or-nothing rendezvous — phase 7 needs its
own, simpler all-ranks-up-or-fail gate rather than reusing `lease_next`;
(d) no training script or model target exists in this repo today.*
