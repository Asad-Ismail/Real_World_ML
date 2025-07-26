import numpy as np  
from collections import Counter
import regex as re
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def pre_tokenize(chunk: str):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counter = Counter()
    for match in re.finditer(PAT, chunk):
        token = match.group(0)
        counter[token] += 1
    return counter


class BPE():

    def __init__(self, vocab_sz=1000, special_tokens=None):
        self.vocab_sz = vocab_sz
        self.special_tokens = special_tokens
        self.merges = []
        self.ids2bytes = {}
        self.bytes2ids = {}
        for i in range(256):
            self.ids2bytes[i] = bytes([i])
            self.bytes2ids[bytes([i])] = i
        for i, token in enumerate(special_tokens, start=256):
            self.ids2bytes[i] = token.encode("utf-8")
            self.bytes2ids[token.encode("utf-8")] = i

    @staticmethod
    def pre_tokenize(self, chunk: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        counter = Counter()
        for match in re.finditer(PAT, chunk):
            token = match.group(0)
            counter[token] += 1
        return counter         

    def train_tokenizer(self, pretokenized: Counter):
        for token, count in pretokenized.items():
            if token not in self.bytes2ids:
                self.bytes2ids[token] = len(self.bytes2ids)
                self.ids2bytes[len(self.ids2bytes)] = token
        for token, count in pretokenized.items():
            if token not in self.bytes2ids:
                self.bytes2ids[token] = len(self.bytes2ids)

if __name__ == "__main__":
    
    splits = 50
    vocab_size = 1000     
    SPECIAL_TOKENS = ["<|endoftext|>"]
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"

    bpe= BPE()
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, splits, "<|endoftext|>".encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            print(start, end)
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Remove special tokens from the text
            pattern = re.compile("|".join(re.escape(tok) for tok in SPECIAL_TOKENS))
            chunk = pattern.sub("", chunk)
            pretokenized=pre_tokenize(chunk)
            
            +   vocab, merges = bpe_trainer(pretokenized, vocal_size, special_tokens= SPECIAL_TOKENS)
            print(vocab)
            print(merges)
            break