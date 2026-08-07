use aegis_core::ops::ternary_matvec;

#[test]
fn test_matvec_loop() {
    let mut output = vec![0.0f32; 8];
    let input = vec![1.0f32; 32];
    let weights = vec![0u8; 64]; // dim_out=8, dim_in=32, packed_dim_in=8, total=64
    ternary_matvec(&mut output, &input, &weights, 8, 32, 1.0);
    println!("Output: {:?}", output);
}
