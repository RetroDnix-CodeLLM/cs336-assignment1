from cs336_basics import BPETokenizer

data = "data/tinystories.txt"

if __name__ == "__main__":
    tokenizer = BPETokenizer(data, special_tokens=["<|endoftext|>"])
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=32000)
    tokenizer.save_vocab("data/tinystories_bpe_vocab.pkl")
    print("BPE vocabulary for TinyStories saved.")