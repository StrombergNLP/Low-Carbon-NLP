import os
import transformers
import sys

from transformers import pipeline

def main():
    fill_mask = pipeline(
        "fill-mask",
        model="./trained_model",
        tokenizer="roberta-base"
    )

    print(fill_mask("The capital of France is <mask>."))

if __name__ == '__main__':
    main()

