import os
import regex as re
from tqdm import tqdm
from pprint import pprint

class BPETokenizer():
    def __init__(self, corpus:str, special_tokens: list[str] = ["<|endoftext|>"]):
        self.vocab = {}
        for i in range(256):
            self.vocab[chr(i).encode("utf-8")] = i
        
        for sp_token in special_tokens:
            self.vocab[sp_token.encode("utf-8")] = len(self.vocab)
        
        self.corpus = []
        self.pretokenized_corpus = []
        self.merges = []

        PAT = '|'.join(map(re.escape, special_tokens))
        with open(corpus, "r", encoding="utf-8") as f:
            self.corpus = re.split(PAT, f.read())

    @staticmethod
    def pre_tokenize(text: str):
        """
        对分块之后的语料库进行预分词

        预分词之后，不考虑合并垮两个“pre-tokenized-token”的字节

        Args:
            text (str): 需要预分词的文本
        
        Returns:
            frequency (dict[str:int]): 预分词后的token列表
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return re.findall(PAT, text, re.UNICODE)
    
    def pre_tokenize_corpus(self):
        """
        对语料库进行预分词
        """
        for text in tqdm(self.corpus, desc="Pre-tokenizing corpus"):
            self.pretokenized_corpus.append(self.pre_tokenize(text))

    def train_bpe(self, maximum_vocab_size: int = 10000):
        """
        在语料库上合并BPE字节对。

        Args:
            maximum_vocab_size (int): 训练BPE的最大词汇表大小，默认为10000

        """
        appearances = {}
        prefix = {}
        suffix = {}
        
        print("Training BPE tokenizer...")
        print("Pre-calculating appearances of byte pairs...")
        maxn = 0
        maxp = None
        if len(self.vocab) < maximum_vocab_size:
            for words in self.pretokenized_corpus:
                for word in words:
                    bs = [(b, ) for b in word.encode("utf-8")]
                    for i in range(len(bs) - 1):
                        pair = (bs[i], bs[i + 1])
                        if pair not in prefix:
                            prefix[pair] = []
                        if pair not in suffix:
                            suffix[pair] = []
                        if i > 0:
                            prefix[pair].append(bs[i - 1])
                        if i < len(bs) - 2:
                            suffix[pair].append(bs[i + 2])
                        appearances[pair] = appearances.get(pair, 0) + 1
                        if appearances[pair] >= maxn:
                            maxn = appearances[pair]
                            maxp = pair
        
        pbar = tqdm(total=maximum_vocab_size - len(self.vocab), desc="Merging byte pairs")
        while len(self.vocab) < maximum_vocab_size:
            current_vocab_size = len(self.vocab)
            if maxp is not None:
                self.merges.append(maxp)
                appearances[maxp] = 0
                maxpb = maxp[0] + maxp[1]
                self.vocab[bytes(maxpb)] = len(self.vocab)
                pbar.update(len(self.vocab) - current_vocab_size) 
                
                for pre_b in prefix.get(maxp, []):
                    appearances[(pre_b, maxp[0])] = appearances.get((pre_b, maxp[0]), 0) - 1
                    appearances[(pre_b, maxpb)] = appearances.get((pre_b, maxpb), 0) + 1
                    prefix[(pre_b, maxpb)] = prefix.get((pre_b, maxp[0]), [])
                
                for suf_b in suffix.get(maxp, []):
                    appearances[(maxp[1], suf_b)] = appearances.get((maxp[1], suf_b), 0) - 1
                    appearances[(maxpb, suf_b)] = appearances.get((maxpb, suf_b), 0) + 1
                    suffix[(maxpb, suf_b)] = suffix.get((maxp[1], suf_b), [])
                
                maxn = 0
                maxp = None
                for pair in appearances:
                    if appearances[pair] >= maxn:
                        maxn = appearances[pair]
                        maxp = pair

        pbar.close()
        print("BPE training complete.")

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    tokenizer = BPETokenizer(corpus = input_path, special_tokens=special_tokens)
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=vocab_size)
    return tokenizer.vocab, tokenizer.merges

if __name__ == "__main__":
    tokenizer = BPETokenizer("data/baby_data.txt", special_tokens=["<|endoftext|>"])
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=300)
    pprint(tokenizer.vocab)
    