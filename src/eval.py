import os
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
    dataset = load_dataset('cc_news', script_version='master')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    dataset_reduced = dataset['train']['text'][:2000]
    del dataset

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
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs['input_ids'],
        eval_dataset=inputs['input_ids'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CarbonTrackerCallback(epochs), PerplexityCallback()],
    )

    with torch.no_grad():
        trainer.evaluate()


def compute_metrics(eval_prediction: EvalPrediction):
    # Computes the perplexity
    labels = eval_prediction.label_ids
    preds = pred.predictions.argmax(-1)
    print('Labels: {}'.format(labels))
    print('Preds: {}'.format(preds))
    return {}


if __name__ == '__main__':
    main()

