import yaml
from cs336_basics.tokenizer import BPETokenizer

config = yaml.safe_load(open("config/train_tinystories.yaml", "r"))

tokenizer = BPETokenizer(
    vocab = config["tokenizer"]["vocab_file"],
    merges= config["tokenizer"]["merges_file"],
    special_tokens=config["tokenizer"]["special_tokens"]
)

tokenizer.parallel_tokenize_txt(config["dataset"]["train_file"], 4, config["dataset"]["tokenized_train_file"])
