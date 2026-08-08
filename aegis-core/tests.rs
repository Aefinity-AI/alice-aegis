#[test]
fn test_ternary_matvec() {
    let mut output = vec![0.0f32; 4];
    let input = vec![1.0f32; 32];
    // packed_dim_in = 32/4 = 8 bytes per row.
    // dim_out = 4 rows. Total bytes = 32.
    let weights = vec![0u8; 32];
    crate::ops::ternary_matvec(&mut output, &input, &weights, 4, 32, 1.0);
    assert_eq!(output[0], 0.0);
}
