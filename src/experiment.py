import os
import json

from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
from datasets import load_dataset
from model.RoBERTaModel import RoBERTaModel
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback


def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

    with open(path + '/config.json') as json_file:
        config = json.load(json_file)

        dataset = load_dataset(config['dataset'], script_version='master')
        dataset_train = dataset['train']
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

        dataset_reduced = dataset_train['text'][:128]

        inputs = tokenizer.batch_encode_plus(
            dataset_reduced, truncation=True, padding=True, verbose=True
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        model = RoBERTaModel(config['model_parameters'][0])

        epochs = config['train_epochs']

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=128,
            per_device_eval_batch_size=128,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=inputs['input_ids'],
            data_collator=data_collator,
            callbacks=[CarbonTrackerCallback(epochs)]
        )

        trainer.train()



if __name__ == '__main__':
    main()

