// Feasibility probe #2: enumerate EVERY variable that can differ between two
// x86_64 machines running the SAME no_std binary with the SAME weights.
//
// Arm A: AVX2 reduction, MXCSR = as inherited
// Arm B: scalar reduction, MXCSR = as inherited   (reduction-order variable)
// Arm C: AVX2 reduction, MXCSR |= FTZ|DAZ         (firmware-config variable)
// Arm D: scalar reduction, MXCSR |= FTZ|DAZ
//
// If C == A bitwise for all prompts, then FTZ/DAZ is NOT a cross-machine
// confound for this engine, and (given no rcp/rsqrt approximations anywhere in
// the tree) the cross-silicon arm of the candidate is provably NULL.
extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;
use std::fs;

const FTZ: u32 = 1 << 15;
const DAZ: u32 = 1 << 6;

#[inline]
fn getcsr() -> u32 {
    unsafe { core::arch::x86_64::_mm_getcsr() }
}
#[inline]
fn setcsr(v: u32) {
    unsafe { core::arch::x86_64::_mm_setcsr(v) }
}

fn gen(engine: &mut TernaryInferenceEngine, prompt: &str, n: usize) -> Vec<String> {
    let mut toks: Vec<String> = Vec::new();
    engine.process_intent(prompt, n, |t| {
        if !t.starts_with("[SYSTEM]") && !t.contains("[PERFORMANCE]") {
            toks.push(t.to_string());
        }
    });
    toks
}

fn first_div(a: &[String], b: &[String]) -> i64 {
    let m = a.len().min(b.len());
    for i in 0..m {
        if a[i] != b[i] {
            return i as i64;
        }
    }
    if a.len() != b.len() {
        return m as i64;
    }
    -1
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::args().nth(1).expect("artifact dir");
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let model = fs::read(format!("{base}/MODEL.SAF"))?;
    let embed = fs::read(format!("{base}/EMBED.BIN"))?;
    let vocab = fs::read(format!("{base}/VOCAB.BIN"))?;

    let prompts: Vec<&str> = vec![
        "Once upon a time",
        "Lily and Ben went to the park",
        "The little dog was very",
        "One day a girl named Anna",
        "Tom had a red ball and he",
        "In a small house near the river",
        "The cat sat on the mat and",
        "Sara found a shiny rock in",
        "Every morning the boy would",
        "The big tree in the garden was",
        "Mom said it was time to",
        "A bird flew over the tall",
        "Jack wanted to build a",
        "The old man smiled and said",
        "When the rain stopped the children",
        "There was a tiny mouse who",
        "Emma opened the box and saw",
        "The sun was hot and the",
        "Two friends decided to walk to",
        "The toy truck was broken so",
        "Nina liked to paint pictures of",
        "Under the bed there was a",
        "The teacher asked the class to",
        "Sam heard a strange noise from",
        "The ice cream melted before",
        "A butterfly landed on the",
        "Ben and his sister found a map",
        "The kitten was hungry and",
        "At the beach the waves were",
        "Grandma baked a cake for",
    ];

    let inherited = getcsr();
    println!("inherited MXCSR = 0x{inherited:04X}  (FTZ={} DAZ={} RC={})",
        (inherited & FTZ) != 0, (inherited & DAZ) != 0, (inherited >> 13) & 3);
    println!("simd capability = {}", aegis_core::ops::simd_level_name());
    println!("prompts={} max_new_tokens={n}", prompts.len());

    let run = |scalar: bool, ftz: bool, p: &str| -> Vec<String> {
        let save = getcsr();
        if ftz {
            setcsr(save | FTZ | DAZ);
        } else {
            setcsr(save & !(FTZ | DAZ));
        }
        aegis_core::ops::set_force_scalar(scalar);
        let mut e = TernaryInferenceEngine::new(&embed, &model, &vocab).unwrap();
        let out = gen(&mut e, p, n);
        aegis_core::ops::set_force_scalar(false);
        setcsr(save);
        out
    };

    let mut c_ne_a = 0usize;
    let mut d_ne_b = 0usize;
    let mut b_ne_a = 0usize;
    let mut a_ne_a2 = 0usize;

    for (pi, p) in prompts.iter().enumerate() {
        let a = run(false, false, p);
        let a2 = run(false, false, p); // same-arm repeat: determinism control
        let b = run(true, false, p);
        let c = run(false, true, p);
        let d = run(true, true, p);

        let (dv_a2, dv_b, dv_c, dv_d) = (
            first_div(&a, &a2),
            first_div(&a, &b),
            first_div(&a, &c),
            first_div(&b, &d),
        );
        if dv_a2 >= 0 {
            a_ne_a2 += 1;
        }
        if dv_b >= 0 {
            b_ne_a += 1;
        }
        if dv_c >= 0 {
            c_ne_a += 1;
        }
        if dv_d >= 0 {
            d_ne_b += 1;
        }
        println!(
            "[{pi:2}] lenA={:3} | A-vs-Arepeat={:4} | A-vs-scalar={:4} | A-vs-A+FTZ/DAZ={:4} | scalar-vs-scalar+FTZ/DAZ={:4}",
            a.len(), dv_a2, dv_b, dv_c, dv_d
        );
    }

    println!("\n==== SUMMARY over {} prompts, {n} tokens ====", prompts.len());
    println!("same arm repeated (determinism control) differed : {a_ne_a2}");
    println!("reduction order changed (AVX2 -> scalar) differed : {b_ne_a}");
    println!("MXCSR FTZ|DAZ set, AVX2 path, differed           : {c_ne_a}");
    println!("MXCSR FTZ|DAZ set, scalar path, differed         : {d_ne_b}");
    Ok(())
}
