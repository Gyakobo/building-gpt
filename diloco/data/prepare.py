"""
Download TinyStories and tokenize it with the GPT-2 BPE tokenizer.

Writes data/train.bin and data/val.bin: one long array of uint16 token ids each,
with an end-of-text token between stories.

    python data/prepare.py
"""

import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_PROC = max(1, (os.cpu_count() or 2) // 2)


def main():
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("roneneldan/TinyStories")  # splits: "train", "validation"

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # 50256 marks the end of each story
        return {"ids": ids, "len": len(ids)}

    for split, name in [("train", "train"), ("validation", "val")]:
        tok = ds[split].map(
            tokenize,
            remove_columns=["text"],
            num_proc=NUM_PROC,
            desc=f"tokenizing {split}",
        )
        total = int(np.sum(tok["len"], dtype=np.uint64))
        path = os.path.join(OUT_DIR, f"{name}.bin")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))

        # write in shards so we never hold the whole dataset in RAM
        n_shards, idx = 256, 0
        for i in tqdm(range(n_shards), desc=f"writing {path}"):
            shard = tok.shard(
                num_shards=n_shards, index=i, contiguous=True
            ).with_format("numpy")
            if len(shard) == 0:
                continue
            chunk = np.concatenate(shard["ids"])
            arr[idx : idx + len(chunk)] = chunk
            idx += len(chunk)
        arr.flush()
        print(f"{name}: {total:,} tokens")


if __name__ == "__main__":  # required on Windows because of num_proc
    main()
