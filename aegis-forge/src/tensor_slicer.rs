pub fn slice_embeddings(
    raw_embeddings: &[f32],
    hidden_dim: usize,
    keep_indices: &[u32],
) -> Vec<f32> {
    let mut pruned_embeddings = Vec::with_capacity(keep_indices.len() * hidden_dim);

    for &old_id in keep_indices {
        let start_idx = (old_id as usize) * hidden_dim;
        let end_idx = start_idx + hidden_dim;

        // Copy only the rows corresponding to preserved English tokens
        pruned_embeddings.extend_from_slice(&raw_embeddings[start_idx..end_idx]);
    }

    pruned_embeddings
}
