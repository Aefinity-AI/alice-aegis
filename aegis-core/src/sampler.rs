pub struct Sampler {
    temperature: f32,
    top_p: f32,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32) -> Self {
        Self { temperature, top_p }
    }

    pub fn argmax(&self, logits: &[f32]) -> u32 {
        let mut max_val = f32::MIN;
        let mut max_idx = 0;

        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        max_idx as u32
    }

    // In the future, implement top-p sampling with softmax.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            self.argmax(logits)
        } else {
            // Fallback to argmax for prototype to avoid rand dependency
            self.argmax(logits)
        }
    }
}
