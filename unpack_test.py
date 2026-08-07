import os
import json
import struct
import numpy as np

with open('/home/killboxincorporated/model.safetensors', 'rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len).decode('utf-8'))
    
    q_meta = header['model.layers.0.self_attn.q_proj.weight']
    start = q_meta['data_offsets'][0]
    
    f.seek(8 + header_len + start)
    data = f.read(16)
    arr = np.frombuffer(data, dtype=np.uint8)
    
    print("Shape in JSON:", q_meta['shape'])
    for byte in arr:
        # try to unpack 2-bit values
        w0 = byte & 0b11
        w1 = (byte >> 2) & 0b11
        w2 = (byte >> 4) & 0b11
        w3 = (byte >> 6) & 0b11
        print(f"Byte: {byte:02x} -> {w0}, {w1}, {w2}, {w3}")
