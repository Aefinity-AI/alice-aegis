import json

def extract_vocab():
    with open('/home/killboxincorporated/tokenizer.json', 'r') as f:
        data = json.load(f)
    
    vocab = data['model']['vocab']
    
    with open('/home/killboxincorporated/aegis-linux/vocab.json', 'w') as f:
        json.dump(vocab, f)
        
if __name__ == "__main__":
    extract_vocab()
