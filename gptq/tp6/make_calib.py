"""corpus.txt -> calib_ids.jsonl: docs tokenized with the model tokenizer, chunked to 1024 tokens with the
[gMASK]<sop> prefix the chat template puts at every sequence start (ids 154822, 154824)."""
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file('src/tokenizer.json')
docs = open('corpus.txt').read().split('\n\x00DOC\x00\n')
n = 0
with open('calib_ids.jsonl', 'w') as f:
    for d in docs:
        ids = tok.encode(d, add_special_tokens=False).ids
        for i in range(0, len(ids) - 256, 1022):
            f.write(json.dumps({'ids': [154822, 154824] + ids[i:i + 1022]}) + '\n'); n += len(ids[i:i + 1022]) + 2
print('calib tokens', n)
