from cs336_basics import BPETokenizer

data = "data/owt_train.txt"

if __name__ == "__main__":
    tokenizer = BPETokenizer(data, special_tokens=["<|endoftext|>"])
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=32000)
    tokenizer.save_vocab("data/owt_train_vocab.pkl")
    tokenizer.save_merges("data/owt_train_merges.pkl")