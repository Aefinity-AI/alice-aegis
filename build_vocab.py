import json
import struct

def build_vocab():
    print("Building packed vocab.bin with BPE merges...")
    with open('/home/killboxincorporated/tokenizer.json', 'r') as f:
        data = json.load(f)

    vocab = data['model']['vocab']
    merges = data['model']['merges']
    added_tokens = data.get('added_tokens', [])
    for token_obj in added_tokens:
        vocab[token_obj['content']] = token_obj['id']

    # Sort vocab by value (ID)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    # Convert merges to pairs of strings
    merge_pairs = []
    for merge in merges:
        parts = merge.split(' ')
        if len(parts) == 2:
            merge_pairs.append((parts[0], parts[1]))

    with open('/home/killboxincorporated/aegis-forge/vocab.bin', 'wb') as f:
        # Magic: 'VOC\x41' (0x564F4341)
        f.write(struct.pack('<I', 0x564F4341))
        # num_tokens
        f.write(struct.pack('<I', len(sorted_vocab)))

        # tokens
        for word, _ in sorted_vocab:
            word_bytes = word.encode('utf-8')
            f.write(struct.pack('<H', len(word_bytes)))
            f.write(word_bytes)

        # num_merges
        f.write(struct.pack('<I', len(merge_pairs)))
        
        # merges (as pairs of token IDs)
        for p1, p2 in merge_pairs:
            # We look up the IDs for the string pieces.
            # But wait, merges use the exact string fragments.
            id1 = vocab.get(p1, 0)
            id2 = vocab.get(p2, 0)
            merged_str = p1 + p2
            id_merged = vocab.get(merged_str, 0)
            
            f.write(struct.pack('<III', id1, id2, id_merged))

    print("vocab.bin built successfully.")

if __name__ == "__main__":
    build_vocab()
