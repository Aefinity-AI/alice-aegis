import re

with open('/home/killboxincorporated/aegis-core/src/inference.rs', 'r') as f:
    content = f.read()

# We need to extract the layer loop and put it into a new method `forward_step`
# And rewrite process_intent.

forward_step_code = """
    fn forward_step(&mut self, current_tok: u32, seq_pos: usize) {
        let emb_dim = self.pipeline.hidden_dim;
        let num_heads = 20;
        let num_kv_heads = 5;
        let head_dim = emb_dim / num_heads;
        let intermediate_size = 6912;
        
        self.arena.hidden_state.fill(0.0);
        
        let start = current_tok as usize * emb_dim * 4;
        if start + emb_dim * 4 <= self.pipeline.embeddings.len() {
            for i in 0..emb_dim {
                let offset = start + i * 4;
                self.arena.hidden_state[i] += crate::ops::f32_from_bytes(
                    self.pipeline.embeddings[offset],
                    self.pipeline.embeddings[offset+1],
                    self.pipeline.embeddings[offset+2],
                    self.pipeline.embeddings[offset+3]
                );
            }
        }
        
        for (layer_idx, layer) in self.pipeline.layers.iter().enumerate() {
            self.arena.norm_state.copy_from_slice(&self.arena.hidden_state);
            crate::ops::rmsnorm(&mut self.arena.norm_state, layer.input_layernorm_weight.data(), 1e-5); 
            
            self.arena.q.fill(0.0); self.arena.k.fill(0.0); self.arena.v.fill(0.0);
            
            crate::ops::ternary_matvec(&mut self.arena.q, &self.arena.norm_state, layer.q_proj.data(), emb_dim, emb_dim, layer.q_proj_scale);
            crate::ops::ternary_matvec(&mut self.arena.k, &self.arena.norm_state, layer.k_proj.data(), num_kv_heads * head_dim, emb_dim, layer.k_proj_scale);
            crate::ops::ternary_matvec(&mut self.arena.v, &self.arena.norm_state, layer.v_proj.data(), num_kv_heads * head_dim, emb_dim, layer.v_proj_scale);
            
            crate::attention::apply_rope(&mut self.arena.q, &mut self.arena.k, seq_pos, num_heads, num_kv_heads, head_dim, &self.rope_cache);
            
            let (k_slot, v_slot) = self.kv_cache.get_layer_mut(layer_idx, seq_pos);
            k_slot.copy_from_slice(&self.arena.k);
            v_slot.copy_from_slice(&self.arena.v);
            
            self.arena.attn_out.fill(0.0);
            
            for h in 0..num_heads {
                let kv_h = h / (num_heads / num_kv_heads);
                for t in 0..=seq_pos {
                    let (k_cache, _) = self.kv_cache.get_layer_mut(layer_idx, t);
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += self.arena.q[h * head_dim + d] * k_cache[kv_h * head_dim + d];
                    }
                    self.arena.scores[t] = score / libm::sqrtf(head_dim as f32);
                }
                
                crate::ops::softmax(&mut self.arena.scores[0..=seq_pos]);
                self.arena.head_out.fill(0.0);
                
                for t in 0..=seq_pos {
                    let (_, v_cache) = self.kv_cache.get_layer_mut(layer_idx, t);
                    let w = self.arena.scores[t];
                    for d in 0..head_dim {
                        self.arena.head_out[d] += w * v_cache[kv_h * head_dim + d];
                    }
                }
                for d in 0..head_dim {
                    self.arena.attn_out[h * head_dim + d] = self.arena.head_out[d];
                }
            }
            
            self.arena.o.fill(0.0);
            crate::ops::rmsnorm(&mut self.arena.attn_out, layer.attn_sub_norm.data(), 1e-5);
            crate::ops::ternary_matvec(&mut self.arena.o, &self.arena.attn_out, layer.o_proj.data(), emb_dim, emb_dim, layer.o_proj_scale);
            for i in 0..emb_dim { self.arena.hidden_state[i] += self.arena.o[i]; }
            
            self.arena.norm_state.copy_from_slice(&self.arena.hidden_state);
            crate::ops::rmsnorm(&mut self.arena.norm_state, layer.post_attention_layernorm_weight.data(), 1e-5);
            
            self.arena.up.fill(0.0); self.arena.gate.fill(0.0);
            crate::ops::ternary_matvec(&mut self.arena.up, &self.arena.norm_state, layer.up_proj.data(), intermediate_size, emb_dim, layer.up_proj_scale);
            crate::ops::ternary_matvec(&mut self.arena.gate, &self.arena.norm_state, layer.gate_proj.data(), intermediate_size, emb_dim, layer.gate_proj_scale);
            
            crate::ops::relu2(&mut self.arena.gate);
            for i in 0..intermediate_size { self.arena.up[i] *= self.arena.gate[i]; }
            
            crate::ops::rmsnorm(&mut self.arena.up, layer.ffn_sub_norm.data(), 1e-5);
            
            self.arena.down.fill(0.0);
            crate::ops::ternary_matvec(&mut self.arena.down, &self.arena.up, layer.down_proj.data(), emb_dim, intermediate_size, layer.down_proj_scale);
            for i in 0..emb_dim { self.arena.hidden_state[i] += self.arena.down[i]; }
        }
    }
"""

new_process_intent = """
    pub fn process_intent(&mut self, intent: &str, mut print_cb: impl FnMut(&str)) -> String {
        let mut tokens = vec![128000]; // BOS Token
        tokens.extend(self.tokenizer.encode(intent));
        if tokens.is_empty() { return String::from("No valid tokens."); }
        
        let mut generated_tokens = Vec::new();
        let emb_dim = self.pipeline.hidden_dim;
        let mut next_token = 0;
        
        // --- PREFILL PHASE ---
        for step in 0..tokens.len() {
            self.forward_step(tokens[step], step);
            
            if step == tokens.len() - 1 {
                crate::ops::rmsnorm(&mut self.arena.hidden_state, self.pipeline.final_norm.data(), 1e-5);
                self.arena.logits.fill(0.0);
                crate::ops::f32_dot(&mut self.arena.logits, &self.arena.hidden_state, self.pipeline.embeddings, self.tokenizer.vocab.len(), emb_dim);
                
                next_token = self.sampler.argmax(&self.arena.logits);
                let decoded_word = self.tokenizer.decode(&[next_token]);
                print_cb(&decoded_word);
                generated_tokens.push(next_token);
            }
        }
        
        let mut total_cycles = 0;
        let mut steps = 0;
        
        // --- DECODE PHASE ---
        let mut step = tokens.len();
        while step < tokens.len() + 100 {
            if next_token == 0 || next_token == 2 || next_token == 128001 || next_token == 128009 { break; }
            
            let t_start = unsafe { core::arch::x86_64::_rdtsc() };
            
            self.forward_step(next_token, step);
            
            crate::ops::rmsnorm(&mut self.arena.hidden_state, self.pipeline.final_norm.data(), 1e-5);
            self.arena.logits.fill(0.0);
            crate::ops::f32_dot(&mut self.arena.logits, &self.arena.hidden_state, self.pipeline.embeddings, self.tokenizer.vocab.len(), emb_dim);
            
            for &tok in &generated_tokens {
                if (tok as usize) < self.arena.logits.len() {
                    self.arena.logits[tok as usize] -= 5.0; // Repetition penalty
                }
            }
            
            next_token = self.sampler.argmax(&self.arena.logits);
            let decoded_word = self.tokenizer.decode(&[next_token]);
            print_cb(&decoded_word);
            generated_tokens.push(next_token);
            
            let t_end = unsafe { core::arch::x86_64::_rdtsc() };
            total_cycles += t_end - t_start;
            steps += 1;
            step += 1;
        }
        
        let avg_cycles = if steps > 0 { total_cycles / steps } else { 0 };
        print_cb(&alloc::format!("\r\n\r\n[PERFORMANCE] Average Cycles/Token: {}\r\n", avg_cycles));
        
        self.tokenizer.decode(&generated_tokens)
    }
"""

start_idx = content.find("pub fn process_intent")
end_idx = content.find("}", content.rfind("avg_cycles"))
end_idx = content.find("}", end_idx + 1)
end_idx = content.find("}", end_idx + 1)
end_idx += 1

new_content = content[:start_idx] + forward_step_code + "\n" + new_process_intent + content[end_idx:]

with open('/home/killboxincorporated/aegis-core/src/inference.rs', 'w') as f:
    f.write(new_content)
