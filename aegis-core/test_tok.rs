fn main() {}
#[test]
fn test_decode() {
    let tok = aegis_core::tokenizer::AegisTokenizer::new(include_bytes!("../../test_usb_dir/vocab.bin"));
    println!("Decode 0: {:?}", tok.decode(&[0]));
}
