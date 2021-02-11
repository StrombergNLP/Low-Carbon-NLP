from transformers import RobertaTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

class Tokenizer():

    def __init__(self, dataset, vocab_size=30000, min_frequency=2, special_tokens, max_len=512):
        self.tokenizer = tokenizers.ByteLevelBPETokenizer()
        self.tokenizer.train(files=dataset, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        self.tokenizer.enable_truncation(max_length=max_len)

        

