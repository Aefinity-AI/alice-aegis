//! Reproduces the golden vectors used by cis2-fp's verify3/test_ternary.c.
//!
//! Those goldens were originally produced by a throwaway, uncommitted example
//! (`ternary_golden_tmp.rs`) that called `aegis_core::ops::ternary_matvec` and
//! was then deleted. This file is the permanent, committed replacement: same
//! four cases (weights, input bits, dim_out, dim_in, scale), same call into
//! aegis-core's real ternary matvec code (no reimplementation), asserted
//! against the exact bit patterns hardcoded in test_ternary.c so the goldens
//! stay reproducible instead of resting on a deleted script.
//!
//! Run with:
//!   cargo run --release --example ternary_golden --no-default-features

use aegis_core::ops::ternary_matvec;

struct Case {
    name: &'static str,
    weights: &'static [u8],
    input_bits: &'static [u32],
    want_bits: &'static [u32],
    dim_out: usize,
    dim_in: usize,
    scale: f32,
}

const CASES: &[Case] = &[
    // == block_aligned: dim_out=8 dim_in=32 scale=1.0 seed=0xc0ffee ==
    Case {
        name: "block_aligned",
        weights: &[
            145, 90, 92, 14, 124, 16, 142, 247, 16, 66, 142, 121, 249, 196, 109, 73, 231, 62, 138,
            253, 38, 176, 121, 49, 67, 71, 207, 50, 56, 104, 98, 142, 23, 123, 195, 188, 91, 155,
            252, 231, 142, 17, 229, 248, 241, 6, 86, 18, 68, 220, 205, 80, 126, 192, 165, 146,
            121, 137, 4, 176, 233, 138, 251, 72,
        ],
        input_bits: &[
            0x40b36000, 0x3a400000, 0x3e70c000, 0x3f802000, 0xc0d7b800, 0x407c3000, 0xc0811c00,
            0x3fec6000, 0x40db6e00, 0xbeca8000, 0x40fcbe00, 0x40198c00, 0x4037bc00, 0x405dd000,
            0xc0a9cc00, 0xc0f86800, 0x400b2c00, 0xc0f56400, 0x40ed8800, 0xbefae000, 0x3fb4c800,
            0x40ebe800, 0xc0638400, 0x40e4a200, 0xc0d01600, 0xc0a6cc00, 0x40ec8600, 0xc00a4400,
            0x40882a00, 0xc0b64600, 0x40aba800, 0xc0d50200,
        ],
        want_bits: &[
            0x413cd100, 0xc08abe00, 0xc1b8b700, 0x41644000, 0xc10a0400, 0xc0d0a400, 0xc1a8b780,
            0xc0fa0000,
        ],
        dim_out: 8,
        dim_in: 32,
        scale: 1.0,
    },
    // == scalar_tail: dim_out=5 dim_in=36 scale=0.5 seed=0xdeadbeef ==
    Case {
        name: "scalar_tail",
        weights: &[
            82, 174, 225, 60, 102, 120, 64, 254, 23, 241, 224, 201, 72, 220, 117, 247, 66, 194,
            21, 63, 7, 247, 32, 244, 165, 210, 123, 197, 136, 109, 109, 153, 26, 244, 93, 191,
            238, 105, 97, 137, 110, 192, 203, 174, 73,
        ],
        input_bits: &[
            0x3fd8e800, 0xc02a5400, 0xbedfe000, 0x4066cc00, 0x40ec0000, 0x40c7d800, 0x40aa5600,
            0xc0977e00, 0x4029a400, 0x40dafa00, 0x3fbf3800, 0x3fd84000, 0xc0370800, 0xc0cdcc00,
            0xc0d0b000, 0xc031e400, 0xc0fe4000, 0x3f970800, 0x3f8c2000, 0x404e9c00, 0x40ee0000,
            0x3e9dc000, 0xc05e6400, 0x404cdc00, 0x40932200, 0x40ec8400, 0x40c65c00, 0xc00a6000,
            0xbe85c000, 0x40834a00, 0x4082cc00, 0x40d6c800, 0xc0393800, 0x40df2200, 0x40881c00,
            0x40d40c00,
        ],
        want_bits: &[0x41112780, 0x41490b00, 0x40a2d000, 0xc09ed100, 0xc0c14100],
        dim_out: 5,
        dim_in: 36,
        scale: 0.5,
    },
    // == dim_out_remainder: dim_out=6 dim_in=64 scale=1.75 seed=0xabcd1234 ==
    Case {
        name: "dim_out_remainder",
        weights: &[
            16, 26, 250, 37, 207, 6, 250, 211, 232, 237, 170, 215, 116, 254, 109, 143, 14, 128,
            221, 112, 182, 147, 222, 57, 193, 178, 131, 230, 35, 51, 93, 121, 35, 1, 133, 62, 252,
            211, 66, 192, 135, 244, 141, 82, 174, 63, 171, 214, 89, 117, 203, 88, 250, 223, 48,
            58, 162, 149, 146, 241, 20, 190, 31, 161, 96, 40, 112, 220, 239, 138, 231, 144, 203,
            90, 224, 209, 138, 29, 47, 117, 63, 169, 176, 175, 190, 127, 135, 248, 129, 126, 254,
            133, 10, 210, 193, 222,
        ],
        input_bits: &[
            0x3ffd8800, 0x40f19c00, 0x40d57e00, 0xbfdab000, 0x40138c00, 0x3eed4000, 0x407b9000,
            0xc0d6ce00, 0x3fc78000, 0x408b8200, 0x40b81400, 0x4034c800, 0x4033a000, 0xc0293c00,
            0x406d6800, 0xbe550000, 0x400d5400, 0xc022b400, 0xc06b0c00, 0xc0342400, 0x40164800,
            0x40fcb200, 0xbfa0d800, 0x4098e400, 0xc0ed2c00, 0xc0dc1000, 0x40ffb600, 0xc0ca3800,
            0xc0a45e00, 0x404f2800, 0xbeba6000, 0xbf667000, 0x3f13a000, 0x3fc54000, 0xc0219c00,
            0xbfd45000, 0xc0071000, 0x409b0000, 0x40803600, 0x3fcad800, 0x3f954800, 0xc078d000,
            0xc04ec400, 0x3fc8b800, 0xc0706c00, 0x3fd11000, 0x3f596000, 0x403c3400, 0x40de2600,
            0xc02fcc00, 0x3e794000, 0x40bb4400, 0xc0dece00, 0x4003e000, 0x40ef7200, 0x3bb00000,
            0x40f12c00, 0xc0c63200, 0xc081c400, 0xc0d64000, 0x40cd9200, 0xc0997800, 0x3f811000,
            0x40060800,
        ],
        want_bits: &[
            0x42631d90, 0x422dc810, 0x4101e580, 0x3fda2600, 0xc2603870, 0x41a76c20,
        ],
        dim_out: 6,
        dim_in: 64,
        scale: 1.75,
    },
    // == all_zero_weights: dim_out=4 dim_in=16 scale=3.0 seed=0x0 ==
    // Sanity check independent of the LUT/bit tables: code 00 must always
    // decode to 0.0 regardless of input, so output is exactly zero.
    Case {
        name: "all_zero_weights",
        weights: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        input_bits: &[
            0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000,
            0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000, 0xc1000000,
            0xc1000000, 0xc1000000,
        ],
        want_bits: &[0x00000000, 0x00000000, 0x00000000, 0x00000000],
        dim_out: 4,
        dim_in: 16,
        scale: 3.0,
    },
];

fn main() {
    let mut failures = 0usize;
    for case in CASES {
        let input: Vec<f32> = case
            .input_bits
            .iter()
            .map(|b| f32::from_bits(*b))
            .collect();
        let mut output = vec![0f32; case.dim_out];
        ternary_matvec(
            &mut output,
            &input,
            case.weights,
            case.dim_out,
            case.dim_in,
            case.scale,
        );

        print!("{}: want =", case.name);
        for w in case.want_bits {
            print!(" 0x{:08x}", w);
        }
        println!();
        print!("{}: got  =", case.name);
        for v in &output {
            print!(" 0x{:08x}", v.to_bits());
        }
        println!();

        for (i, (got, want)) in output.iter().zip(case.want_bits.iter()).enumerate() {
            if got.to_bits() != *want {
                println!(
                    "FAIL: {} row {} got 0x{:08x} want 0x{:08x}",
                    case.name,
                    i,
                    got.to_bits(),
                    want
                );
                failures += 1;
            }
        }
    }

    if failures == 0 {
        println!("ALL PASS");
    } else {
        println!("{} FAILURE(S)", failures);
        std::process::exit(1);
    }
}
