import os
import random
import math
import torch
import transformers

from datasets import load_dataset
from transformers import EvalPrediction
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from callbacks.PerplexityCallback import PerplexityCallback
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback


def main():
    epochs = 1
    torch.cuda.device(1)
    random.seed(25565)

    dataset = load_dataset('cc_news', script_version='master')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset_reduced = dataset['train']['text'][:200]
    del dataset
    random.shuffle(dataset_reduced)

    inputs = tokenizer.batch_encode_plus(
        dataset_reduced, truncation=True, padding=True, verbose=True, max_length=512
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    config = RobertaConfig.from_pretrained('./trained_model/config.json')
    model = RobertaForMaskedLM(config)
    model.load_state_dict(torch.load('./trained_model/pytorch_model.bin'))


    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_eval_batch_size=1,
        logging_dir='./logs',
        eval_accumulation_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs['input_ids'],
        eval_dataset=inputs['input_ids'],
        data_collator=data_collator,
        callbacks=[CarbonTrackerCallback(epochs), PerplexityCallback()],
    )

    with torch.no_grad():
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        print(perplexity)
    






if __name__ == '__main__':
    main()

