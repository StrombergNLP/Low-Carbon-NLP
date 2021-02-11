from transformers import RobertaTokenizerFast
from tokenizers.processors import BertProcessing

class RoBERTaTokenizer():

    def __init__(self, dataset, vocab_size=30000, min_frequency=2, special_tokens, max_len=512):
        
        self.tokenizer = RobertaTokenizerFast.
