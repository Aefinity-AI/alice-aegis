use aegis_core::sampler::Sampler;

#[test]
fn test_padding_logits_infinite_rejection() {
    // 1. Setup mock logits state, simulating vocabulary truncation and Logit Initialization Patch
    let original_vocab_size = 128_256;
    let pruned_vocab_size = 50_000;

    // Create a vector of logits, initialized to -f32::INFINITY
    let mut logits = vec![-f32::INFINITY; original_vocab_size];

    // Populate the first 50,000 valid tokens with simulated true probabilities
    for i in 0..pruned_vocab_size {
        // Arbitrary small logit values; max is set to token 42
        logits[i] = if i == 42 { 15.0 } else { -5.0 };
    }

    // Simulate a permutation/adversarial attack where padded tokens might theoretically surface
    // if left as 0.0 instead of -f32::INFINITY, because -5.0 is less than 0.0.

    // 2. Sampler Block
    let sampler = Sampler::new(0.0, 1.0); // Temperature 0.0 implies Argmax

    let selected_token = sampler.sample(&logits);

    // 3. Mathematical Rigor Validation
    assert_eq!(
        selected_token, 42,
        "Argmax failed to select the correct active token."
    );

    // Verify padded void tokens are completely unreachable
    for i in pruned_vocab_size..original_vocab_size {
        assert!(
            logits[i] == -f32::INFINITY,
            "Padding logit modified from -INFINITY!"
        );
    }

    // Verify that NO padded token is ever selected when all active tokens are negative
    let mut negative_logits = vec![-f32::INFINITY; original_vocab_size];
    for i in 0..pruned_vocab_size {
        negative_logits[i] = -10.0; // Valid tokens have very negative probability (e.g. from RMSNorm shift)
    }
    negative_logits[123] = -5.0; // The max of the negative active tokens

    let selected_negative = sampler.sample(&negative_logits);

    assert!(
        selected_negative < pruned_vocab_size as u32,
        "Sampler selected a padded void token!"
    );
    assert_eq!(
        selected_negative, 123,
        "Argmax failed on all-negative valid logits."
    );
}
