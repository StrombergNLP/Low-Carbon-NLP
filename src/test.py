import os
import transformers
import sys

from transformers import pipeline

def main():
    fill_mask = pipeline(
        "fill-mask",
        model="./trained_model/pytorch_model",
        tokenizer="roberta-base"
    )

    fill_mask("Let's get together and <mask> sometime!")

if __name__ == '__main__':
    main()

