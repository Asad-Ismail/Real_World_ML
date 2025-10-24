import re
from collections import Counter
from typing import List

class BPE:
    def __init__(self,vocab_size,special_tokens=["<pad>","<endoftext>"]):
        self.vocabsize=vocab_size
        ## vocab maps from bpe to int , initial vocab is 256
        self.vocab_dict = {i:bytes([i]) for i in range(256)}
        self.vocab_dict.update({i:token.encode("utf-8") for i, token in enumerate(special_tokens, 256)})
        # merges will be a list of bytes 
        self.merges =[]

    @staticmethod
    def _get_Stats(tokens:list[int])->Counter:
        pairs = Counter(zip(tokens[:-1], tokens[1:]))
        return pairs

    @staticmethod
    def _merge(tokens:list[int],pair:tuple[int,int],new_token:int)->list[int]:
        # replace all occurrences of pair with new_token
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == pair[0] and i+1 < len(tokens) and tokens[i+1] == pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens   

    def train(self,text_file:str="data/TinyStoriesV2-GPT4-valid.txt"):
        with open(text_file,"r") as f:
            text=f.read().strip()
        #pre_tokenized = re.findall(r"\s*\w+|\s*\S",text)
        tokens = list(text.encode("utf-8"))

        while len(self.vocab_dict) < self.vocabsize:
            print("Current vocabulary size:", len(self.vocab_dict))
            pairs = BPE._get_Stats(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            print(best_pair)
            # replace best token with the new token
            tokens = BPE._merge(tokens,best_pair,len(self.vocab_dict))
            first_byte = self.vocab_dict[best_pair[0]]
            second_byte = self.vocab_dict[best_pair[1]]
            self.merges.append((first_byte, second_byte))
            self.vocab_dict[len(self.vocab_dict)] = first_byte + second_byte
        print("Final vocabulary size:", len(self.vocab_dict))

    def encode(self, text: str) -> List[int]:
        tokens = list(text.encode("utf-8"))
        merge_idx = {bytepair: i for i, bytepair in enumerate(self.merges)}
        inv_dict = {v: k for k, v in self.vocab_dict.items()}

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            if not pairs:
                break
            
            pair_to_merge = min(
                pairs, 
                key=lambda pair: merge_idx.get(
                    (self.vocab_dict[pair[0]], self.vocab_dict[pair[1]]), 
                    float("inf")
                )
            )
            
            bytepair = (self.vocab_dict[pair_to_merge[0]], self.vocab_dict[pair_to_merge[1]])
            if bytepair not in merge_idx:
                break
            
            merged_token_id = inv_dict[bytepair[0] + bytepair[1]] 
            tokens = BPE._merge(tokens, pair_to_merge, merged_token_id)
        
        return tokens
    

    def decode(self,input_ids:List[int])->str:
        bytestr =b"".join([self.vocab_dict[i] for i in input_ids])
        return bytestr.decode("utf-8", errors="replace")
    



class BPEChar:
    def __init__(self,vocab_size,special_tokens=["<pad>","<endoftext>"]):
        self.vocabsize=vocab_size
        ## vocab maps from bpe to int , initial vocab is 256
        self.vocab_dict = {i:token for i, token in enumerate(special_tokens)}
        # merges will be a list of bytes 
        self.merges =[]

    @staticmethod
    def _get_Stats(tokens:list[int])->Counter:
        pairs = Counter(zip(tokens[:-1], tokens[1:]))
        return pairs

    @staticmethod
    def _merge(tokens:list[int],pair:tuple[int,int],new_token:int)->list[int]:
        # replace all occurrences of pair with new_token
        new_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == pair[0] and i+1 < len(tokens) and tokens[i+1] == pair[1]:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens   

    def train(self,text_file:str="data/TinyStoriesV2-GPT4-valid.txt"):
        with open(text_file,"r") as f:
            text=f.read().strip()
        #pre_tokenized = re.findall(r"\s*\w+|\s*\S",text)
        tokens = list(text)
        unique_tokens =set(tokens)
        for token in unique_tokens:
            self.vocab_dict[len(self.vocab_dict)] = token

        while len(self.vocab_dict) < self.vocabsize:
            print("Current vocabulary size:", len(self.vocab_dict))
            pairs = BPEChar._get_Stats(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            print(best_pair)
            # replace best token with the new token
            tokens = BPEChar._merge(tokens,best_pair,best_pair[0]+best_pair[1])
            self.merges.append((best_pair[0], best_pair[1]))
            self.vocab_dict[len(self.vocab_dict)] = best_pair[0] + best_pair[1]
        print("Final vocabulary size:", len(self.vocab_dict))

    def encode(self, text: str) -> List[int]:
        tokens = list(text)
        merge_idx = {pair: i for i, pair in enumerate(self.merges)}
        inv_dict = {v: k for k, v in self.vocab_dict.items()}

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            if not pairs:
                break
            
            pair_to_merge = min(
                pairs, 
                key=lambda pair: merge_idx.get(
                    (pair[0], pair[1]), 
                    float("inf")
                )
            )

            if pair_to_merge not in merge_idx:
                break
            
            merged_token_id = inv_dict[pair_to_merge[0] + pair_to_merge[1]] 
            tokens = BPE._merge(tokens, pair_to_merge, merged_token_id)
        
        return tokens
    

    def decode(self,input_ids:List[int])->str:
            str ="".join([self.vocab_dict[i] for i in input_ids])
            return string



if __name__ == "__main__":


    bp =BPEChar(vocab_size=250)
    bp.train()
    input_ids = bp.encode("Hello, world!")
    decoded_text = bp.decode(input_ids)
    print("Decoded text:", decoded_text)
