use std::collections::{BTreeMap, HashMap};

/// Executes the Vocabulary Truncation sequence.  
pub fn strip_non_ascii_vocab(
    original_vocab: &HashMap<String, u32>,
) -> (BTreeMap<String, u32>, Vec<usize>) {
    // SORT the vocabulary by original ID to guarantee row alignment!
    let mut sorted_vocab: Vec<_> = original_vocab.iter().collect();
    sorted_vocab.sort_by_key(|a| a.1);

    let mut new_vocab = BTreeMap::new();
    let mut keep_indices = Vec::new();
    let mut new_id = 0;

    for (token, &old_id) in sorted_vocab {
        // Keep the first 50000 tokens (most common BPEs) OR any special token
        if old_id < 50000 || token.starts_with("<|") || token.starts_with("<") {
            new_vocab.insert(token.clone(), new_id);
            keep_indices.push(old_id as usize);
            new_id += 1;
        }
    }

    (new_vocab, keep_indices)
}
