use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("/home/killboxincorporated/qemu_mnt/vocab.bin").unwrap();
    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    
    // Simple lookup if we can parse the vocab
    println!("Loaded vocab size: {}", data.len());
}
