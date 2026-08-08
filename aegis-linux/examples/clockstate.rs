// Clock-state prober (Rule A, RDTSC corollary).
//
// RDTSC is NOT a cycle counter. It advances at the invariant nominal rate
// regardless of what the core clock is doing, so `ticks/token` is not
// `cycles/token`, and any figure derived from it (MAC/cycle, %-of-peak) is
// wrong by exactly the effective/nominal ratio unless that ratio is measured
// and published alongside.
//
// This emits that ratio. Two steps:
//
//   1. Calibrate the TSC's nominal rate against CLOCK_MONOTONIC. This gives
//      ticks-per-second for the invariant counter — NOT the core clock.
//   2. Run a chain of DEPENDENT integer adds. Each add has 1-cycle latency and
//      depends on its predecessor, so the chain cannot be reordered, fused, or
//      executed in parallel: it retires at exactly one op per core cycle. The
//      op count is therefore a direct core-cycle count.
//
//   core_GHz = ops * tsc_nominal_GHz / tsc_ticks
//
// Print this block at the head of every timing capture and cite it with the
// figures it rescales. This tool lived in a scratchpad and was lost to a crash;
// it is checked in so the ratio is never re-derived from memory.
//
// Usage: clockstate [chain_len] [reps]      (defaults: 120000000, 5)

use std::arch::x86_64::_rdtsc;
use std::time::Instant;

/// Executes `n` iterations of an 8-deep dependent add chain and returns the
/// final accumulator. Returns 8*n retired dependent adds.
fn dependent_add_chain(n: u64) -> u64 {
    let mut x: u64 = 1;
    let mut i: u64 = n;
    // SAFETY: pure register arithmetic on locals. No memory operands, no
    // observable side effects, and every register touched is declared to the
    // compiler as inout/lateout so it cannot assume a value survives the block.
    // `nostack` holds because the block neither pushes nor calls.
    unsafe {
        std::arch::asm!(
            "2:",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "add {x}, 1",
            "dec {i}",
            "jnz 2b",
            x = inout(reg) x,
            i = inout(reg) i,
            options(nostack),
        );
    }
    // Consume both so nothing is dead-code eliminated.
    x.wrapping_add(i)
}

/// Calibrate TSC ticks/second against CLOCK_MONOTONIC over a ~200 ms window.
fn calibrate_tsc_hz() -> f64 {
    // SAFETY: rdtsc is unprivileged and always available on x86_64; it reads a
    // counter and has no side effects.
    let t0 = Instant::now();
    let c0 = unsafe { _rdtsc() };
    while t0.elapsed().as_millis() < 200 {
        std::hint::spin_loop();
    }
    let c1 = unsafe { _rdtsc() };
    let secs = t0.elapsed().as_secs_f64();
    (c1 - c0) as f64 / secs
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let chain_len: u64 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(120_000_000);
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    // 8 adds per loop iteration.
    let iters = chain_len / 8;
    let ops = iters * 8;

    // Warm up so the first rep is not measuring the ramp out of an idle P-state.
    let _ = dependent_add_chain(iters / 4);

    let tsc_hz = calibrate_tsc_hz();
    let tsc_ghz = tsc_hz / 1e9;

    println!("--- CLOCK STATE (bound to this measurement; RDTSC ticks are NOT core cycles) ---");
    println!("    TSC nominal rate        : {tsc_ghz:.4} GHz  (calibrated vs CLOCK_MONOTONIC)");
    println!("    chain length            : {ops} dependent adds x {reps} reps");
    println!("  ");
    println!("    rep       tsc_ticks              ops     core_GHz vs_nominal");

    let mut ghz = Vec::with_capacity(reps);
    for rep in 0..reps {
        // SAFETY: see calibrate_tsc_hz.
        let c0 = unsafe { _rdtsc() };
        let sink = dependent_add_chain(iters);
        let c1 = unsafe { _rdtsc() };
        std::hint::black_box(sink);

        let ticks = c1 - c0;
        // ops retire at 1/cycle, so core_GHz = ops / wall_seconds, and
        // wall_seconds = ticks / tsc_hz.
        let core_ghz = ops as f64 * tsc_ghz / ticks as f64;
        println!(
            "    {rep}    {ticks:>14}   {ops:>14}     {core_ghz:>6.4}     {:.3}x",
            core_ghz / tsc_ghz
        );
        ghz.push(core_ghz);
    }

    let mean = ghz.iter().sum::<f64>() / ghz.len() as f64;
    let min = ghz.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = ghz.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let spread = (max - min) / mean * 100.0;

    println!("  ");
    println!("    mean {mean:.4} GHz | min {min:.4} | max {max:.4} | spread {spread:.3}%");
    println!(
        "    EFFECTIVE / NOMINAL = {:.3}  <-- the factor that rescales every derived figure",
        mean / tsc_ghz
    );
}
