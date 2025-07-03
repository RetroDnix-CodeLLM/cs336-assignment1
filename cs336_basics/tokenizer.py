import os
import regex as re
from time import time
from tqdm import tqdm
from pickle import dump
from multiprocessing import Pool, cpu_count
from collections import Counter

from cs336_basics.utils import UnionFindSet, increaseD, decreaseD, appendD, removeD

class BPETokenizer():
    pretokenizePAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self, corpus:str, special_tokens: list[str] = ["<|endoftext|>"]):
        self.vocab = {}
        for i in range(256):
            self.vocab[bytes([i,])] = i
        
        for sp_token in special_tokens:
            self.vocab[sp_token.encode("utf-8")] = len(self.vocab)
        
        self.corpus = []
        self.frequency = {}
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
        return re.finditer(BPETokenizer.pretokenizePAT, text, re.UNICODE)
    
    def pre_tokenize_corpus(self):
        """
        对语料库进行预分词
        """
        for text in tqdm(self.corpus, desc="Pre-tokenizing corpus"):
            for result in BPETokenizer.pre_tokenize(text):
                token = result.group(0)
                self.frequency[token] = self.frequency.get(token, 0) + 1
        
        print(f"Pre-tokenization complete. Found {len(self.frequency)} unique tokens.")

    @staticmethod
    def _process_text_chunk(text_chunk: str):
        """
        处理单个文本块的辅助函数，用于并行处理
        
        Args:
            text (str): 需要处理的文本块
        
        Returns:
            dict: 该文本块中token的频率字典
        """
        id = text_chunk[0]
        chunk = text_chunk[1]
        frequency = Counter()
        for i, text in enumerate(chunk):
            for result in re.finditer(BPETokenizer.pretokenizePAT, text, re.UNICODE):
                token = result.group(0)
                frequency[token] += 1
            if i % 10000 == 0:
                print(f"Process {id}: {i} / {len(chunk)}")
        return frequency

    def parallel_pre_tokenize_corpus(self, num_processes=None):
        """
        对语料库进行并行预分词
        
        Args:
            num_processes (int, optional): 使用的进程数。如果为None，则使用CPU核心数
        """
        if num_processes is None:
            num_processes = cpu_count()
        
        print(f"Starting parallel pre-tokenization using {num_processes} processes...")
        
        # 初始化频率字典
        self.frequency = Counter()
        self.chunked_corpus = [self.corpus[i::num_processes] for i in range(num_processes)]
        
        # 使用多进程池处理语料库
        with Pool(num_processes) as pool:
            for result in pool.imap_unordered(BPETokenizer._process_text_chunk, enumerate(self.chunked_corpus)):
                self.frequency += result
        
        print(f"Parallel pre-tokenization complete. Found {len(self.frequency)} unique tokens.")

    def train_bpe(self, maximum_vocab_size: int = 10000):
        """
        在语料库上合并BPE字节对。

        Args:
            maximum_vocab_size (int): 训练BPE的最大词汇表大小，默认为10000

        """
        appearances = {}    # apprearance[pair]: 表示字符对pair出现的频率
        position = {}       # position[pair]: 表示字符对pair(a,b)出现的位置（使用b的首字节作为标记）
        value = {}          # value[(i,j,k)]: 表示位置(i,j,k)的字节对的值（首字节标记）
        ufs = {}
        
        maxn = 0
        maxp = None
        frequency = self.frequency

        if len(self.vocab) < maximum_vocab_size:
            pbar = tqdm(total=len(frequency), desc="BPE Training(Pre-processing)...")
            for token in frequency:
                bs = [(b, ) for b in token.encode("utf-8")]
                l = len(bs)
                s = ufs[token] = UnionFindSet(size = l)
                v = value[token] = {}
                for k in range(l):
                    v[k] = bs[k]

                    if k < l - 1:
                        pair = (bs[k], bs[k + 1])
                        increaseD(appearances, pair, frequency[token])
                        appendD(position, pair, (token, k))
                        if appearances[pair] > maxn or (appearances[pair] == maxn and pair > maxp):
                            maxn = appearances[pair]
                            maxp = pair
                pbar.update(1)
            pbar.close()
        
        pbar = tqdm(total=maximum_vocab_size - len(self.vocab), desc="BPE Training(Merging byte pairs)...")
        while len(self.vocab) < maximum_vocab_size:
            current_vocab_size = len(self.vocab)
            if maxp is not None:
                bMaxPair = maxp[0] + maxp[1]
                self.merges.append((bytes(maxp[0]), bytes(maxp[1])))
                self.vocab[bytes(bMaxPair)] = len(self.vocab)

                pbar.update(len(self.vocab) - current_vocab_size) 
                
                pos = list(position[maxp])
                ord = 0
                for token, k1 in pos:
                    s = ufs[token]
                    v = value[token]
                    k2 = k1 + s.getSize(k1)
                    sizek2 = s.getSize(k2)

                    s.union(k1, k2)
                    v[k1] = bMaxPair
                    
                    if s.has(k1 - 1):
                        k0 = s.find(k1 - 1)
                        bPre = v[k0]
                        increaseD(appearances, (bPre, bMaxPair), frequency[token])
                        appendD(position, (bPre, bMaxPair), (token, k0))
                        
                        decreaseD(appearances, (bPre, maxp[0]), frequency[token])
                        removeD(position, (bPre, maxp[0]), (token, k0))
                        if (bPre, maxp[0]) == maxp and pos.index((token, k0)) > ord:
                           pos.remove((token, k0))

                    if s.has(k2 + sizek2):
                        k3 = k2 + sizek2
                        bSuf = v[k3]
                        increaseD(appearances, (bMaxPair, bSuf), frequency[token])
                        appendD(position, (bMaxPair, bSuf), (token, k1))

                        decreaseD(appearances, (maxp[1], bSuf), frequency[token])
                        removeD(position, (maxp[1], bSuf), (token, k2))
                        if (maxp[1], bSuf) == maxp and pos.index((token, k2)) > ord:
                            pos.remove((token, k2))
                    ord += 1
                        
                appearances.pop(maxp)
                position.pop(maxp)
                v.pop(k2)

                maxn = 0
                maxp = None
                for pair in appearances:
                    if appearances[pair] > maxn or (appearances[pair] == maxn and pair > maxp):
                        maxn = appearances[pair]
                        maxp = pair

        pbar.close()
        print("BPE training complete.")

    def save_vocab(self, output_path: str | os.PathLike):
        """
        保存BPE词汇表到指定路径

        Args:
            output_path (str | os.PathLike): 输出路径
        """
        with open(output_path, "wb") as f:
            dump(self.vocab, f)
        print(f"Vocabulary saved to {output_path}")
    
    def save_merges(self, output_path: str | os.PathLike):
        """
        保存BPE合并规则到指定路径

        Args:
            output_path (str | os.PathLike): 输出路径
        """
        with open(output_path, "wb") as f:
            dump(self.merges, f)
        print(f"Merges saved to {output_path}")
    
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    tokenizer = BPETokenizer(corpus = input_path, special_tokens=special_tokens)
    s = time()
    # tokenizer.parallel_pre_tokenize_corpus(4)
    tokenizer.pre_tokenize_corpus()
    d = time()
    print(f"Pre-tokenization took {d - s:.2f} seconds.")
    tokenizer.train_bpe(maximum_vocab_size=vocab_size)
    print(f"BPE training took {time() - d:.2f} seconds.")
    reversed_vocab = {v: k for k, v in tokenizer.vocab.items()}
    return reversed_vocab, tokenizer.merges

if __name__ == "__main__":
    tokenizer = BPETokenizer("data/baby_data.txt", special_tokens=["<|endoftext|>"])
    tokenizer.parallel_pre_tokenize_corpus(4)
    print(tokenizer.frequency)
    tokenizer.train_bpe(maximum_vocab_size=300)
