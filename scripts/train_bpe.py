from cs336_basics import BPETrainer

dname = "TinyStoriesV2-GPT4-train"
data = f"data/{dname}.txt"

if __name__ == "__main__":
    tokenizer = BPETrainer(data, special_tokens=["<|endoftext|>"])
    tokenizer.pre_tokenize_corpus()
    tokenizer.train_bpe(maximum_vocab_size=10000)
    tokenizer.save_vocab(f"data/{dname}_vocab.pkl")
    tokenizer.save_merges(f"data/{dname}_merges.pkl")