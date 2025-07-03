import os
import regex as re
from time import time
from tqdm import tqdm
from pickle import dump
from multiprocessing import Pool, cpu_count

from .utils import UnionFindSet, increaseD, decreaseD, appendD, removeD

class BPETokenizer():
    def __init__(self, corpus:str, special_tokens: list[str] = ["<|endoftext|>"]):
        self.vocab = {}
        for i in range(256):
            self.vocab[bytes([i,])] = i
        
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
    
    @staticmethod
    def _pre_tokenize_wrapper(text_chunk):
        """
        多进程预分词的包装函数
        
        Args:
            text_chunk (list[str]): 文本块列表
        
        Returns:
            list[list[str]]: 预分词后的结果列表
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        results = []
        for i, text in enumerate(text_chunk[1]):
            results.append(re.findall(PAT, text, re.UNICODE))
            if i % 10000 == 0:
                print(f"Process {text_chunk[0]}: {i} / {len(text_chunk[1])}")
    
        return results

    def parallel_pre_tokenize_corpus(self, num_processes=None):
        """
        对语料库进行并行预分词
        
        Args:
            num_processes (int, optional): 使用的进程数。如果为None，则使用CPU核心数
        """
        if num_processes is None:
            num_processes = cpu_count()
        
        # 将语料库分割成块
        chunk_size = max(1, len(self.corpus) // num_processes)
        chunks = [self.corpus[i:i + chunk_size] for i in range(0, len(self.corpus), chunk_size)]
        
        print(f"Using {num_processes} processes to pre-tokenize {len(self.corpus)} texts")
        
        # 使用多进程池进行并行处理
        with Pool(processes=num_processes) as pool:
            results = list(
                pool.imap(BPETokenizer._pre_tokenize_wrapper, enumerate(chunks))
            )
        # 合并结果
        print("Merging results from all processes...")
        self.pretokenized_corpus = []
        for chunk_result in results:
            self.pretokenized_corpus.extend(chunk_result)
        
        print(f"Pre-tokenization complete.")

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

        if len(self.vocab) < maximum_vocab_size:
            pbar = tqdm(total=len(self.pretokenized_corpus), desc="BPE Training(Pre-processing)...")
            for i, words in enumerate(self.pretokenized_corpus):
                for j, word in enumerate(words):
                    # i, j 是语料库中第i行的第j个预分词token
                    bs = [(b, ) for b in word.encode("utf-8")]
                    l = len(bs)
                    s = ufs[(i, j)] = UnionFindSet(size = l)
                    v = value[(i, j)] = {}
                    for k in range(l):
                        v[k] = bs[k]

                        if k < l - 1:
                            pair = (bs[k], bs[k + 1])
                            increaseD(appearances, pair)
                            appendD(position, pair, (i, j, k))
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
                for i, j, k1 in pos:
                    s = ufs[(i, j)]
                    v = value[(i, j)]
                    k2 = k1 + s.getSize(k1)
                    sizek2 = s.getSize(k2)

                    s.union(k1, k2)
                    v[k1] = bMaxPair
                    
                    if s.has(k1 - 1):
                        k0 = s.find(k1 - 1)
                        bPre = v[k0]
                        increaseD(appearances, (bPre, bMaxPair))
                        appendD(position, (bPre, bMaxPair), (i, j, k0))
                        
                        decreaseD(appearances, (bPre, maxp[0]))
                        removeD(position, (bPre, maxp[0]), (i, j, k0))
                        if (bPre, maxp[0]) == maxp:
                            pos.remove((i, j, k0))

                    if s.has(k2 + sizek2):
                        k3 = k2 + sizek2
                        bSuf = v[k3]
                        increaseD(appearances, (bMaxPair, bSuf))
                        appendD(position, (bMaxPair, bSuf), (i, j, k1))

                        decreaseD(appearances, (maxp[1], bSuf))
                        removeD(position, (maxp[1], bSuf), (i, j, k2))
                        if (maxp[1], bSuf) == maxp:
                            pos.remove((i, j, k2))
                        
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
    
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
):
    tokenizer = BPETokenizer(corpus = input_path, special_tokens=special_tokens)
    s = time()
    tokenizer.pre_tokenize_corpus()
    d = time()
    print(f"Pre-tokenization took {d - s:.2f} seconds.")
    tokenizer.train_bpe(maximum_vocab_size=vocab_size)
    print(f"BPE training took {time() - d:.2f} seconds.")
    reversed_vocab = {v: k for k, v in tokenizer.vocab.items()}
    return reversed_vocab, tokenizer.merges

if __name__ == "__main__":
    tokenizer = BPETokenizer("data/baby_data.txt", special_tokens=["<|endoftext|>"])
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=300)
