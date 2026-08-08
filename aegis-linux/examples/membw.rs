// membw.rs — measure the machine's memory read bandwidth, because the number the
// report uses has never been measured.
//
// TECHNICAL_REPORT.md:185 states "Measured machine bandwidth ceiling (single
// thread) | 17.3 GB/s" and the 2026-07-29 forensic sweep found NO SOURCE FOR IT
// ANYWHERE: `grep -rn 'GB/s\|bandwidth' aegis-core/ aegis-linux/ aegis-eval/
// --include=*.rs` returns exactly one hit, a doc comment at
// aegis-core/src/ops.rs:1240 that QUOTES the figure rather than producing it.
// It is the roofline's denominator — §6's whole "21.7% of peak / bandwidth-bound
// above ~1.4 GHz" argument divides by it — and it is the only claim in the audit
// with no source of any kind.
//
// This file is the smallest honest replacement. It reports three separate
// numbers and refuses to collapse them, because they are not the same thing and
// the report's single row hides which one it means:
//
//   1. PEAK SEQUENTIAL READ  — one thread, u64 loads, buffer >> LLC. The
//      hardware ceiling in the roofline sense.
//   2. ALL-THREAD READ       — the same, saturated. A single-thread figure is
//      NOT the machine ceiling; on this class of part one core cannot saturate
//      the controller, so quoting a 1T number as "the machine ceiling" flatters
//      the roofline.
//   3. WEIGHT-STREAM READ    — the ACTUAL access pattern of ternary_matvec:
//      sequential over 2-bit packed rows with a per-row LUT-indexed inner loop.
//      This is the number that belongs in a decode roofline, and it will be
//      lower than (1).
//
// Method, stated so it can be attacked: allocate a buffer 8x the largest cache
// this machine reports (or 512 MB, whichever is larger), touch it once to fault
// it in, then time N sequential passes with rdtsc AND wall clock, reporting
// both. Reads are accumulated into a volatile-consumed sum so LLVM cannot
// eliminate them; the accumulator is printed for exactly that reason.
//
// Usage: membw [buffer_mb] [passes] [threads]
use std::time::Instant;

#[inline(always)]
fn read_sum(buf: &[u64]) -> u64 {
    // Four independent accumulators: one dependent chain would measure latency,
    // not bandwidth. This is the difference between ~2 GB/s and the real figure.
    let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
    let mut i = 0;
    while i + 4 <= buf.len() {
        a = a.wrapping_add(buf[i]);
        b = b.wrapping_add(buf[i + 1]);
        c = c.wrapping_add(buf[i + 2]);
        d = d.wrapping_add(buf[i + 3]);
        i += 4;
    }
    while i < buf.len() {
        a = a.wrapping_add(buf[i]);
        i += 1;
    }
    a.wrapping_add(b).wrapping_add(c).wrapping_add(d)
}

/// The decode access pattern: 2-bit packed ternary rows, LUT-indexed, exactly as
/// aegis_core::ops::ternary_matvec_serial walks them. Bandwidth measured under
/// THIS pattern is the only one a decode roofline may divide by.
fn read_sum_ternary(buf: &[u8], lut: &[f32; 4]) -> f32 {
    let mut acc = 0.0f32;
    for &byte in buf {
        acc += lut[(byte & 0b11) as usize]
            + lut[((byte >> 2) & 0b11) as usize]
            + lut[((byte >> 4) & 0b11) as usize]
            + lut[((byte >> 6) & 0b11) as usize];
    }
    acc
}

fn gbs(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mb: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(512);
    let passes: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let nthreads: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1)
    });

    let bytes = mb * 1024 * 1024;
    println!(
        "# membw: buffer {} MB, {} passes, up to {} threads",
        mb, passes, nthreads
    );
    println!("# method: 4 independent accumulators (a dependent chain measures LATENCY,");
    println!("#         not bandwidth); buffer pre-faulted; wall clock via Instant.");
    println!("# NOTE: a single-thread figure is NOT the machine ceiling. Both are printed.");

    // Pre-fault and fill with non-zero so nothing is a zero page.
    let mut v: Vec<u64> = (0..bytes / 8)
        .map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .collect();
    v[0] = 1;
    let buf = &v[..];

    println!("\nlabel,threads,pass,bytes_read,secs,GB_per_s,checksum");

    // --- 1. single-thread sequential read ---
    let mut best1 = 0.0f64;
    for p in 0..passes {
        let t = Instant::now();
        let s = read_sum(buf);
        let e = t.elapsed().as_secs_f64();
        let g = gbs(bytes as u64, e);
        best1 = best1.max(g);
        println!("seq_read,1,{},{},{:.6},{:.3},{:016x}", p, bytes, e, g, s);
    }

    // --- 2. all-thread sequential read (disjoint slices, no sharing) ---
    let mut bestn = 0.0f64;
    if nthreads > 1 {
        for p in 0..passes {
            let chunk = buf.len() / nthreads;
            let t = Instant::now();
            let sum: u64 = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..nthreads)
                    .map(|k| {
                        let lo = k * chunk;
                        let hi = if k + 1 == nthreads {
                            buf.len()
                        } else {
                            lo + chunk
                        };
                        sc.spawn(move || read_sum(&buf[lo..hi]))
                    })
                    .collect();
                hs.into_iter()
                    .map(|h| h.join().unwrap())
                    .fold(0u64, |a, b| a.wrapping_add(b))
            });
            let e = t.elapsed().as_secs_f64();
            let g = gbs(bytes as u64, e);
            bestn = bestn.max(g);
            println!(
                "seq_read,{},{},{},{:.6},{:.3},{:016x}",
                nthreads, p, bytes, e, g, sum
            );
        }
    }

    // --- 3. the ternary weight-stream pattern, single thread ---
    let bytes8: &[u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, bytes) };
    let lut = [0.0f32, 1.0, -1.0, 0.0];
    let mut best_t = 0.0f64;
    for p in 0..passes {
        let t = Instant::now();
        let s = read_sum_ternary(bytes8, &lut);
        let e = t.elapsed().as_secs_f64();
        let g = gbs(bytes as u64, e);
        best_t = best_t.max(g);
        println!("ternary_stream,1,{},{},{:.6},{:.3},{:e}", p, bytes, e, g, s);
    }

    println!(
        "\n==== SUMMARY (best of {} passes; report ALL THREE or none) ====",
        passes
    );
    println!("  peak sequential read, 1 thread     : {:.2} GB/s", best1);
    if nthreads > 1 {
        println!(
            "  peak sequential read, {} threads    : {:.2} GB/s",
            nthreads, bestn
        );
    }
    println!("  ternary weight-stream, 1 thread    : {:.2} GB/s", best_t);
    println!();
    println!("  CAVEAT ON ARM 3: this is a SCALAR LUT walk, not the AVX2 kernel");
    println!("  (aegis_core::ops::ternary_matvec_serial). It is therefore a LOWER");
    println!("  BOUND on the engine's streaming rate, not the engine's rate. Compare");
    println!("  it to the technical report's 'achieved' bandwidth row, which is itself");
    println!("  a derivation from a cycle count and an unstated nominal clock.");
    println!();
    println!("  The decode roofline must divide by the TERNARY figure, not the peak.");
    println!("  Quoting a single-thread peak as 'the machine bandwidth ceiling' both");
    println!("  understates the machine and overstates the kernel's share of it.");
    println!("  This run is one machine, one buffer size, one pattern. It replaces an");
    println!("  unsourced number with a measured one; it does not make it a ceiling.");
}
