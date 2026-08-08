# Patch 01 — give every bare-metal row a build identity

**Defect this closes, by name:** ledger row **A12**, whose headline pair
(0.62 → 3.03 tok/s, 3,548,301,534 → 726,238,201 ticks/tok) is marked
**CONFOUNDED** in this program's own ledger because *"the arms are 2026-07-12 vs
2026-07-29 builds and 8 commits touched aegis-core/src between them, incl.
019cd81 (forward-pass change)."*

That confound was not a mistake of reasoning. It was **structurally
unavoidable**: `docs/hardware_logs/gauntlet_dataset.tsv` has 18 columns and not
one of them records a date, a commit, or a binary hash. Two rows in that file
are indistinguishable from an A/B even when they are four weeks and eight
commits apart. The file *cannot* tell you the arms differ.

Four columns fix it permanently. `runcard.py` already computes the equivalent
for the Linux side (`env_hash`); this is the no-OS half.

---

## 1. Export the build identity at compile time

`aegis-uefi/build_hardfloat.sh` — add before the `cargo build`:

```bash
# Build identity travels INTO the binary, so a BOOTLOG.TXT from a stick found in
# a drawer six months from now still names the tree that produced it. A log that
# cannot name its binary cannot be an arm of an A/B (ledger A12).
export AEGIS_BUILD_SHA="$(git -C "$(git rev-parse --show-toplevel)" rev-parse --short=12 HEAD)"
export AEGIS_BUILD_DIRTY="$(git -C "$(git rev-parse --show-toplevel)" status --porcelain | head -c1)"
export AEGIS_BUILD_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "build identity: ${AEGIS_BUILD_SHA}${AEGIS_BUILD_DIRTY:+-dirty} at ${AEGIS_BUILD_UTC}"
```

`aegis-uefi/build.rs` (create if absent) — makes the env vars visible to the
crate *and* forces a rebuild when they change:

```rust
// Build identity for the gauntlet header. Without this, a BOOTLOG.TXT row cannot
// name the binary that produced it, and two rows four weeks apart look like an
// A/B (see ledger A12: "that pair is CONFOUNDED").
fn main() {
    for k in ["AEGIS_BUILD_SHA", "AEGIS_BUILD_DIRTY", "AEGIS_BUILD_UTC"] {
        println!("cargo:rerun-if-env-changed={k}");
    }
    // Absent env (a plain `cargo build`) must be VISIBLE, not silently blank:
    // "unknown" in the log is a fact; an empty field is an ambiguity.
    println!("cargo:rustc-env=AEGIS_BUILD_SHA={}",
             std::env::var("AEGIS_BUILD_SHA").unwrap_or_else(|_| "unknown".into()));
    println!("cargo:rustc-env=AEGIS_BUILD_DIRTY={}",
             std::env::var("AEGIS_BUILD_DIRTY").unwrap_or_default());
    println!("cargo:rustc-env=AEGIS_BUILD_UTC={}",
             std::env::var("AEGIS_BUILD_UTC").unwrap_or_else(|_| "unknown".into()));
}
```

## 2. Print it in the gauntlet header

`aegis-uefi/src/main.rs`, immediately after the existing `boot_log(&mut root,
"==== GAUNTLET ====")` at line 635:

```rust
                    // RUNID binds every row below to one boot of one binary.
                    // BOOTLOG.TXT accumulates across boots (deliberately — crash
                    // forensics), so without a per-run marker the collector
                    // cannot tell which rows belong together; it currently
                    // guesses by splitting on the last "==== GAUNTLET ====".
                    boot_log(&mut root, &format!(
                        "GAUNTLET RUNID: build={}{} built={} boot_tsc={}",
                        env!("AEGIS_BUILD_SHA"),
                        if env!("AEGIS_BUILD_DIRTY").is_empty() { "" } else { "-dirty" },
                        env!("AEGIS_BUILD_UTC"),
                        unsafe { core::arch::x86_64::_rdtsc() }
                    ));
```

`boot_tsc` is the boot-unique component: the Dell has no RTC access from this
app and no network, so a TSC read at gauntlet start is the cheapest thing that
differs between two boots of the *same* binary. It makes RUNID unique without
adding a dependency.

## 3. Make repeats possible — the drift control the userspace side never had

The gauntlet's `gbench!` (main.rs:595) runs each arm **once**. `PSTATE_run1` vs
`PSTATE_run2_control` is a two-sample drift control for one arm only; the SIMD
and PREFILL arms have no repeat, so their ratios carry an unknown spread. One
line makes every arm repeatable:

```rust
                    macro_rules! gbench_n {
                        ($label:expr, $prompt:expr, $max:expr, $n:expr) => {{
                            // Round-robin at the CALL SITE, not here: interleaving
                            // arms is what defeats monotone thermal drift. This
                            // macro only adds the repeat index so the collector
                            // can compute a spread instead of assuming one.
                            for _rep in 0..$n {
                                gbench!(concat!($label, "#"), $prompt, $max);
                            }
                        }};
                    }
```

then interleave in segment 2 instead of running each arm once:

```rust
                    // seg2, drift-resistant: alternate arms, 3 rounds each.
                    for _round in 0..3 {
                        aegis_core::ops::set_force_scalar(true);
                        gbench!("SIMD_scalar", ESSAY, 30);
                        aegis_core::ops::set_force_scalar(false);
                        gbench!("SIMD_native", ESSAY, 30);
                    }
```

`collect_gauntlet.sh`'s `dticks()` uses `re.search`, which takes the **first**
match — with repeats it must take the **median**. See patch 02.

## 4. Refuse a dataset row with no build identity

`collect_gauntlet.sh`, after the existing `GAUNTLET DONE` check:

```bash
grep -qa "GAUNTLET RUNID:" "$RAW" || {
  echo "REFUSING to append: this BOOTLOG.TXT has no GAUNTLET RUNID line, so the row"
  echo "  could not name the binary that produced it. That is exactly how ledger A12's"
  echo "  headline pair became CONFOUNDED. Rebuild with build_hardfloat.sh (which now"
  echo "  exports AEGIS_BUILD_SHA) and re-run /gauntlet."
  echo "  To archive the log anyway without polluting the dataset: pass --raw-only."
  [ "${2:-}" = "--raw-only" ] || exit 1
}
```

and in the Python block, extend the header and the row:

```python
runid = re.search(r"GAUNTLET RUNID: build=(\S+) built=(\S+) boot_tsc=(\d+)", txt)
build_sha  = runid.group(1) if runid else "unknown"
built_utc  = runid.group(2) if runid else "unknown"
boot_tsc   = runid.group(3) if runid else "0"
collected  = os.popen("date -u +%Y-%m-%dT%H:%M:%SZ").read().strip()

# Four new columns, first — so a reader cannot see a number before seeing which
# binary produced it.
hdr = ("runid\tbuild_sha\tbuilt_utc\tcollected_utc\t"
       "nickname\tcpu\tsimd\t" + "\t".join(seg.keys()) +
       "\tr_simd\tr_batch\tdrift\tr_turbo\tr_ctx\n")
row = [f"{build_sha}.{boot_tsc[-8:]}", build_sha, built_utc, collected,
       nick, cpu, simd] + [str(seg[k]) for k in seg] + \
      [str(r_simd), str(r_batch), str(drift), str(r_turbo), str(r_ctx)]
```

## 5. Migrate the four existing rows honestly

The four rows already in the TSV cannot be retro-fitted with a build sha — that
information was never captured. Write `unknown` and say so in the file, rather
than back-filling a guess:

```bash
python3 - <<'PY'
import pathlib
p = pathlib.Path.home()/"docs/hardware_logs/gauntlet_dataset.tsv"
lines = p.read_text().splitlines()
new = ["# Rows collected before patch 01 (2026-07-30) carry build_sha=unknown:",
       "# their binary identity was never recorded and cannot be recovered.",
       "# Two `unknown` rows are NOT an A/B. See ledger A12.",
       "runid\tbuild_sha\tbuilt_utc\tcollected_utc\t" + lines[0]]
new += ["unknown\tunknown\tunknown\tunknown\t" + l for l in lines[1:]]
p.with_suffix(".tsv.pre-patch01").write_text("\n".join(lines) + "\n")
p.write_text("\n".join(new) + "\n")
print("migrated; original kept at", p.with_suffix(".tsv.pre-patch01").name)
PY
```

---

**Cost:** about 40 lines across five files, one rebuild, one re-run of
`/gauntlet` per machine to get the first identified rows. The migration is
lossless (original preserved).

**What it does not fix:** it cannot make the existing four rows comparable. A12
stays confounded forever; this only stops the *next* one.
