import os
import json

from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from model.model import Model


def main():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))

    with open(path + '/config.json') as json_file:
        config = json.load(json_file)

        model = Model(config['model_name'])

        data = load_dataset(config['dataset'], script_version='master')
        print(data)

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
            model=model,
            args=training_args,
            train_dataset=data
        )

        # trainer.train()



if __name__ == '__main__':
    main()

