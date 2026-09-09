//! toks — decode AEGIS-TRACE `toks=` fields back to text, and encode a prompt.
//!
//! A receipt records token ids, not text, so diagnosing "why did this episode
//! report `tool=no-tool`?" otherwise means guessing at what the model wrote.
//! This is a read-only diagnostic over the vocabulary alone: it loads no
//! model, runs no inference, and is not part of any receipt, verification or
//! conformance path.
//!
//!   toks decode <VOCAB.BIN> 717,489,220
//!   toks encode <VOCAB.BIN> "Q: What is 12 + 30?"
//!   toks receipt <VOCAB.BIN> <receipt>     # every step's text, in order

use aegis_core::tokenizer::AegisTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: toks decode <VOCAB.BIN> <id,id,...>\n\
             \x20      toks encode <VOCAB.BIN> \"text\"\n\
             \x20      toks receipt <VOCAB.BIN> <receipt-path>"
        );
        std::process::exit(2);
    }
    let vocab = std::fs::read(&args[2]).expect("read VOCAB.BIN");
    let tk = AegisTokenizer::new(&vocab).expect("parse VOCAB.BIN");
    match args[1].as_str() {
        "decode" => println!("{}", tk.decode(&parse_ids(&args[3]))),
        "encode" => {
            let ids = tk.encode(&args[3]);
            println!(
                "{}",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
            );
        }
        "receipt" => {
            let text = std::fs::read_to_string(&args[3]).expect("read receipt");
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("step ") {
                    let (idx, _) = rest.split_once(':').unwrap_or((rest, ""));
                    let ids = line
                        .split_whitespace()
                        .find_map(|t| t.strip_prefix("toks="))
                        .map(parse_ids)
                        .unwrap_or_default();
                    println!("step {idx}: {:?}", tk.decode(&ids));
                } else if let Some(hex) = line.strip_prefix("prompt-hex ") {
                    let bytes: Vec<u8> = (0..hex.len() / 2)
                        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap_or(b'?'))
                        .collect();
                    println!("prompt: {:?}", String::from_utf8_lossy(&bytes));
                }
            }
        }
        other => {
            eprintln!("toks: unknown mode {other:?}");
            std::process::exit(2);
        }
    }
}

fn parse_ids(s: &str) -> Vec<u32> {
    s.split(',')
        .filter(|t| !t.is_empty())
        .map(|t| t.trim().parse().expect("token id"))
        .collect()
}
