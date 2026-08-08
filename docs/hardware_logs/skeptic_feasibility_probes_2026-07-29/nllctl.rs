extern crate alloc;
use aegis_core::inference::TernaryInferenceEngine;
use std::fs;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = std::env::args().nth(1).unwrap();
    let model = fs::read(format!("{base}/MODEL.SAF"))?;
    let embed = fs::read(format!("{base}/EMBED.BIN"))?;
    let vocab = fs::read(format!("{base}/VOCAB.BIN"))?;
    let t = "Question: Which of these is a source of light? Answer: the sun";
    // CONTROL: same arm, twice. Must be bit-identical or the probe is invalid.
    let mut v = vec![];
    for scalar in [false, false, true, true, false] {
        aegis_core::ops::set_force_scalar(scalar);
        let mut e = TernaryInferenceEngine::new(&embed, &model, &vocab)?;
        let ids = e.tokenizer.encode(t);
        let p = e.calculate_perplexity(&ids);
        v.push((scalar, p));
        aegis_core::ops::set_force_scalar(false);
    }
    for (s, p) in &v { println!("force_scalar={s:5}  ppl bits=0x{:016x}  ppl={:.12}", p.to_bits(), p); }
    println!("A==A2 bitwise: {}", v[0].1.to_bits()==v[1].1.to_bits());
    println!("B==B2 bitwise: {}", v[2].1.to_bits()==v[3].1.to_bits());
    println!("A==A3 bitwise (after B ran): {}", v[0].1.to_bits()==v[4].1.to_bits());
    println!("A==B  bitwise: {}", v[0].1.to_bits()==v[2].1.to_bits());
    Ok(())
}
