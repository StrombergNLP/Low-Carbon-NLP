import os
# import mxnet as mx

# from mlm.scorers import MLMScorerPT
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

# ctxs = [mx.gpu(0)]

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name, script_version='master')

    dataset_reduced = dataset['train']['text'][:100000]
    del dataset

    return dataset_reduced


if __name__ == '__main__':
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    # Loop through all model folders and get id, epoch & model bin
    for subdir, dirs, files in os.walk(config_path):
        for file in files:
            full_model_path = os.path.join(subdir, file)
            if full_model_path.endswith('.bin'):
                split = full_model_path.split('/')
                model_id = split[-3]
                epoch = split[-2]
                model_dir = '/' + os.path.join(*split[:-1]) # + '/'
                print(model_dir)
                #prøv igen xd
                config = RobertaConfig.from_pretrained(model_dir)
                model = RobertaForMaskedLM.from_pretrained(model_dir, config=config)
                tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
                vocab = tokenizer.get_vocab()

                tracker = CarbonTracker(epochs=1, epochs_before_pred=-1, monitor_epochs=-1, verbose=2, log_dir=model_dir)
                tracker.epoch_start()

                #########################
                # KODE TIL EVAL HERINDE #
                #########################

                tracker.epoch_end()
                tracker.stop()

                # Get energy consumed
                logs = parser.parse_all_logs(log_dir=model_dir)
                latest_log = logs[len(logs) - 1]
                energy = latest_log['actual']['energy (kWh)']

