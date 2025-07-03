from cs336_basics import BPETokenizer

data = "data/TinyStoriesV2-GPT4-train.txt"

if __name__ == "__main__":
    tokenizer = BPETokenizer(data, special_tokens=["<|endoftext|>"])
    tokenizer.parallel_pre_tokenize_corpus(4)
    tokenizer.train_bpe(maximum_vocab_size=32000)
    tokenizer.save_vocab("data/tinystories_bpe_vocab.pkl")
    print("BPE vocabulary for TinyStories saved.")