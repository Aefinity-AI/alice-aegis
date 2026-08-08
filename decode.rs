use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("/home/killboxincorporated/aegis-forge/vocab.bin").unwrap();
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x564F4341 { panic!("Invalid Vocab Magic"); }
    
    let num_tokens = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let mut tokens = Vec::new();
    
    let mut offset = 8;
    for _ in 0..num_tokens {
        let len = u16::from_le_bytes([bytes[offset], bytes[offset+1]]) as usize;
        offset += 2;
        let tok_bytes = &bytes[offset..offset+len];
        offset += len;
        
        let s = String::from_utf8_lossy(tok_bytes).into_owned();
        tokens.push(s);
    }
    
    println!("Token 271: {:?}", tokens.get(271));
    println!("Token 198: {:?}", tokens.get(198));
}
