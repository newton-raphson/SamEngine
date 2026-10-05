"""Downloads Tiny Shakespeare and writes byte-level tokens (vocab 256) to data/{train,val}.bin (90/10 split)."""
import os
import urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

os.makedirs(OUT, exist_ok=True)
tokens = np.frombuffer(urllib.request.urlopen(URL).read(), dtype=np.uint8)
n = int(0.9 * len(tokens))
tokens[:n].tofile(os.path.join(OUT, "train.bin"))
tokens[n:].tofile(os.path.join(OUT, "val.bin"))
print(f"{len(tokens):,} tokens -> train {n:,}, val {len(tokens) - n:,}")
