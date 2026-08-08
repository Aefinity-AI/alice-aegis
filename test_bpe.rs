fn bpe_char_to_byte(c: char) -> u8 {
    let u = c as u32;
    if u < 256 {
        return u as u8;
    }
    // Reverse the GPT-2 byte_encoder logic
    let mut missing = alloc::vec::Vec::new();
    for b in 0..=255u32 {
        let is_good = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        if !is_good {
            missing.push(b as u8);
        }
    }
    
    let offset = u - 256;
    if (offset as usize) < missing.len() {
        missing[offset as usize]
    } else {
        b'?'
    }
}
