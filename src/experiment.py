import os
import json
import transformers

from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
from transformers import AdamW
from datasets import load_dataset
from model.RoBERTaModel import RoBERTaModel
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback
from callbacks import PerplexityCallback


def main():
    transformers.logging.set_verbosity_info()
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

    with open(path + '/config.json') as json_file:
        config = json.load(json_file)
        epochs = config['train_epochs']


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


        opt_param = config['optimizer_parameters'][0]
        optimizer = AdamW(params=model.parameters(), lr=opt_param['lr'], betas=(opt_param['beta_one'], opt_param['beta_two']), eps=opt_param['eps'], weight_decay=opt_param['weight_decay'])
        scheduler = None

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            warmup_steps=opt_param['warmup_steps'],
            weight_decay=opt_param['weight_decay'],
            logging_dir='./logs'
        )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=inputs['input_ids'],
            data_collator=data_collator,
            callbacks=[CarbonTrackerCallback(epochs)],
            optimizers=(optimizer, scheduler)
        )

        # trainer.train()

        # trainer.save_model('trained_model.json')



if __name__ == '__main__':
    main()

