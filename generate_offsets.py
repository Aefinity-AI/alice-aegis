import os
import json
import struct

FILE_PATH = "/home/killboxincorporated/aegis-linux/aegis_model.safetensors"
OUTPUT_RUST_FILE = "/home/killboxincorporated/aegis-uefi/src/tensor_map.rs"

def generate_rust_map():
    with open(FILE_PATH, "rb") as f:
        # The first 8 bytes in a safetensors file dictate the length of the JSON header
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        # Read the JSON header
        header_json = f.read(header_size).decode('utf-8')
        header_data = json.loads(header_json)
        
        # The binary data starts exactly here
        data_start_offset = 8 + header_size
        
        # Generate the Rust file
        with open(OUTPUT_RUST_FILE, "w") as out:
            out.write("pub const TENSOR_MAP: &[(&str, usize, usize)] = &[\n")
            
            for key, meta in header_data.items():
                if key == "__metadata__":
                    continue
                
                # data_offsets inside safetensors are relative to the start of the binary data
                start = meta["data_offsets"][0]
                end = meta["data_offsets"][1]
                
                # Write to rust map
                out.write(f'    ("{key}", {start}, {end}),\n')
                
            out.write("];\n")

    print(f"Successfully generated {OUTPUT_RUST_FILE}")

if __name__ == "__main__":
    generate_rust_map()
