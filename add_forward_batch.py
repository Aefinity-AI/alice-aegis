import re

with open('/home/killboxincorporated/aegis-core/src/inference.rs', 'r') as f:
    content = f.read()

forward_batch_code = """
    fn forward_batch(&mut self, batch_tokens: &[u32], seq_pos_start: usize) {
        let batch_size = batch_tokens.len();
        let emb_dim = self.pipeline.hidden_dim;
        let num_heads = 20;
        let num_kv_heads = 5;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = 6912;
        
        self.arena.batch_hidden_state[0..batch_size * emb_dim].fill(0.0);
        
        for b in 0..batch_size {
            let current_tok = batch_tokens[b];
            let start = current_tok as usize * emb_dim * 4;
            let hidden_offset = b * emb_dim;
            
            if start + emb_dim * 4 <= self.pipeline.embeddings.len() {
                for i in 0..emb_dim {
                    let offset = start + i * 4;
                    self.arena.batch_hidden_state[hidden_offset + i] += crate::ops::f32_from_bytes(
                        self.pipeline.embeddings[offset],
                        self.pipeline.embeddings[offset+1],
                        self.pipeline.embeddings[offset+2],
                        self.pipeline.embeddings[offset+3]
                    );
                }
            }
        }
        
        for (layer_idx, layer) in self.pipeline.layers.iter().enumerate() {
            self.arena.batch_norm_state[0..batch_size * emb_dim].copy_from_slice(&self.arena.batch_hidden_state[0..batch_size * emb_dim]);
            for b in 0..batch_size {
                let offset = b * emb_dim;
                crate::ops::rmsnorm(&mut self.arena.batch_norm_state[offset..offset+emb_dim], layer.input_layernorm_weight.data(), 1e-5); 
            }
            
            self.arena.batch_q[0..batch_size * emb_dim].fill(0.0);
            self.arena.batch_k[0..batch_size * (num_kv_heads * head_dim)].fill(0.0);
            self.arena.batch_v[0..batch_size * (num_kv_heads * head_dim)].fill(0.0);
            
            crate::ops::ternary_matmul(&mut self.arena.batch_q, &self.arena.batch_norm_state[0..batch_size * emb_dim], layer.q_proj.data(), batch_size, emb_dim, emb_dim, layer.q_proj_scale);
            crate::ops::ternary_matmul(&mut self.arena.batch_k, &self.arena.batch_norm_state[0..batch_size * emb_dim], layer.k_proj.data(), batch_size, num_kv_heads * head_dim, emb_dim, layer.k_proj_scale);
            crate::ops::ternary_matmul(&mut self.arena.batch_v, &self.arena.batch_norm_state[0..batch_size * emb_dim], layer.v_proj.data(), batch_size, num_kv_heads * head_dim, emb_dim, layer.v_proj_scale);
            
            for b in 0..batch_size {
                let q_offset = b * emb_dim;
                let kv_offset = b * (num_kv_heads * head_dim);
                crate::attention::apply_rope(
                    &mut self.arena.batch_q[q_offset..q_offset+emb_dim], 
                    &mut self.arena.batch_k[kv_offset..kv_offset+(num_kv_heads * head_dim)], 
                    seq_pos_start + b, num_heads, num_kv_heads, head_dim, &self.rope_cache
                );
                
                let (k_slot, v_slot) = self.kv_cache.get_layer_mut(layer_idx, seq_pos_start + b);
                k_slot.copy_from_slice(&self.arena.batch_k[kv_offset..kv_offset+(num_kv_heads * head_dim)]);
                v_slot.copy_from_slice(&self.arena.batch_v[kv_offset..kv_offset+(num_kv_heads * head_dim)]);
            }
            
            self.arena.batch_attn_out[0..batch_size * emb_dim].fill(0.0);
            
            for b in 0..batch_size {
                let q_offset = b * emb_dim;
                let attn_out_offset = b * emb_dim;
                let current_seq_pos = seq_pos_start + b;
                
                for h in 0..num_heads {
                    let kv_h = h / (num_heads / num_kv_heads);
                    for t in 0..=current_seq_pos {
                        let (k_cache, _) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += self.arena.batch_q[q_offset + h * head_dim + d] * k_cache[kv_h * head_dim + d];
                        }
                        self.arena.scores[t] = score / libm::sqrtf(head_dim as f32);
                    }
                    
                    crate::ops::softmax(&mut self.arena.scores[0..=current_seq_pos]);
                    self.arena.head_out.fill(0.0);
                    
                    for t in 0..=current_seq_pos {
                        let (_, v_cache) = self.kv_cache.get_layer_mut(layer_idx, t);
                        let w = self.arena.scores[t];
                        for d in 0..head_dim {
                            self.arena.head_out[d] += w * v_cache[kv_h * head_dim + d];
                        }
                    }
                    for d in 0..head_dim {
                        self.arena.batch_attn_out[attn_out_offset + h * head_dim + d] = self.arena.head_out[d];
                    }
                }
            }
            
            self.arena.batch_o[0..batch_size * emb_dim].fill(0.0);
            for b in 0..batch_size {
                let offset = b * emb_dim;
                crate::ops::rmsnorm(&mut self.arena.batch_attn_out[offset..offset+emb_dim], layer.attn_sub_norm.data(), 1e-5);
            }
            crate::ops::ternary_matmul(&mut self.arena.batch_o, &self.arena.batch_attn_out[0..batch_size * emb_dim], layer.o_proj.data(), batch_size, emb_dim, emb_dim, layer.o_proj_scale);
            
            for i in 0..(batch_size * emb_dim) { self.arena.batch_hidden_state[i] += self.arena.batch_o[i]; }
            
            self.arena.batch_norm_state[0..batch_size * emb_dim].copy_from_slice(&self.arena.batch_hidden_state[0..batch_size * emb_dim]);
            for b in 0..batch_size {
                let offset = b * emb_dim;
                crate::ops::rmsnorm(&mut self.arena.batch_norm_state[offset..offset+emb_dim], layer.post_attention_layernorm_weight.data(), 1e-5);
            }
            
            self.arena.batch_up[0..batch_size * intermediate_size].fill(0.0);
            self.arena.batch_gate[0..batch_size * intermediate_size].fill(0.0);
            
            crate::ops::ternary_matmul(&mut self.arena.batch_up, &self.arena.batch_norm_state[0..batch_size * emb_dim], layer.up_proj.data(), batch_size, intermediate_size, emb_dim, layer.up_proj_scale);
            crate::ops::ternary_matmul(&mut self.arena.batch_gate, &self.arena.batch_norm_state[0..batch_size * emb_dim], layer.gate_proj.data(), batch_size, intermediate_size, emb_dim, layer.gate_proj_scale);
            
            for b in 0..batch_size {
                let offset = b * intermediate_size;
                crate::ops::relu2(&mut self.arena.batch_gate[offset..offset+intermediate_size]);
            }
            for i in 0..(batch_size * intermediate_size) { self.arena.batch_up[i] *= self.arena.batch_gate[i]; }
            
            for b in 0..batch_size {
                let offset = b * intermediate_size;
                crate::ops::rmsnorm(&mut self.arena.batch_up[offset..offset+intermediate_size], layer.ffn_sub_norm.data(), 1e-5);
            }
            
            self.arena.batch_down[0..batch_size * emb_dim].fill(0.0);
            crate::ops::ternary_matmul(&mut self.arena.batch_down, &self.arena.batch_up[0..batch_size * intermediate_size], layer.down_proj.data(), batch_size, emb_dim, intermediate_size, layer.down_proj_scale);
            
            for i in 0..(batch_size * emb_dim) { self.arena.batch_hidden_state[i] += self.arena.batch_down[i]; }
        }
    }
"""

# Replace the prefill phase in process_intent with forward_batch
prefill_pattern = r"// --- PREFILL PHASE ---.*?// --- DECODE PHASE ---"
new_prefill = """// --- PREFILL PHASE ---
        self.forward_batch(&tokens, 0);
        
        let final_token_offset = (tokens.len() - 1) * emb_dim;
        self.arena.hidden_state.copy_from_slice(&self.arena.batch_hidden_state[final_token_offset..final_token_offset+emb_dim]);
        
        crate::ops::rmsnorm(&mut self.arena.hidden_state, self.pipeline.final_norm.data(), 1e-5);
        self.arena.logits.fill(0.0);
        crate::ops::f32_dot(&mut self.arena.logits, &self.arena.hidden_state, self.pipeline.embeddings, self.tokenizer.vocab.len(), emb_dim);
        
        next_token = self.sampler.argmax(&self.arena.logits);
        let decoded_word = self.tokenizer.decode(&[next_token]);
        print_cb(&decoded_word);
        generated_tokens.push(next_token);
        
        let mut total_cycles = 0;
        let mut steps = 0;
        
        // --- DECODE PHASE ---"""

content = re.sub(prefill_pattern, new_prefill, content, flags=re.DOTALL)

# Insert forward_batch before forward_step
content = content.replace("fn forward_step(&mut self", forward_batch_code + "\n    fn forward_step(&mut self")

with open('/home/killboxincorporated/aegis-core/src/inference.rs', 'w') as f:
    f.write(content)
