import os
import random
import json
import transformers
import torch
import math
import sys
import time
import csv
import datetime
import uuid

from torch import nn
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
from carbontracker.tracker import CarbonTracker
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AdamW
from datasets import load_dataset
from carbontracker import parser

from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.mongoexp import MongoTrials

###########################################################################################
# Hyperopt sucks at subpackages, so we need to package callbacks and models into one file #
###########################################################################################

carbondir_path = './carbon_logs_'+ '2' + '/'
os.mkdir(carbondir_path)

class CarbonTrackerCallback(TrainerCallback):
    def __init__(self, max_epochs):
        super().__init__()
        self.tracker = CarbonTracker(epochs=max_epochs, epochs_before_pred=-1, monitor_epochs=-1, verbose=2, log_dir=carbondir_path)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.epoch_start()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.epoch_end()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.stop()


class RoBERTaModel(nn.Module):
    """
    Class for tuning any given model architecture
    """

    def __init__(
            self,
            model_parameters,
            output_hidden_states=True,
            output_logits=True,
            output_attentions=True
        ):

        super().__init__()

        self.config = RobertaConfig(
            vocab_size=model_parameters['vocab_size'],
            hidden_size=model_parameters['hidden_size'] * model_parameters['num_attention_heads'],
            num_hidden_layers=model_parameters['num_hidden_layers'],
            num_attention_heads=model_parameters['num_attention_heads'],
            intermediate_size=model_parameters['intermediate_size'],
            hidden_act=model_parameters['hidden_act'],
            hidden_dropout_prob=model_parameters['hidden_dropout_prob'],
            attention_probs_dropout_prog=model_parameters['attention_probs_dropout_prog'],
            max_position_embeddings=model_parameters['max_position_embeddings'] * 2,
            type_vocab_size=model_parameters['type_vocab_size'],
            initializer_range=model_parameters['initializer_range'],
            layer_norm_eps=model_parameters['layer_norm_eps'],
            gradient_checkpointing=model_parameters['gradient_checkpointing'],
            position_embedding_type=model_parameters['position_embedding_type'],
            use_cache=model_parameters['use_cache'],
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
            output_attentions=output_attentions
        )
        self.model = RobertaForMaskedLM(config=self.config)

        # self.tune()


    def forward(self, *model_args,**model_kwargs):
        return self.model(*model_args, **model_kwargs)


    def tune(self):
        state_dict = self.model.state_dict()
        # print(state_dict.keys())


###########################################################################################
###########################################################################################
###########################################################################################


def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name, script_version='master')

    dataset_reduced = dataset['train']['text'][:100000]
    del dataset

    return dataset_reduced


def get_dataset_from_disk(dataset_name):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    dataset = load_dataset('text', data_files=data_path + dataset_name)
    return dataset['train']['text'][:2000]


torch.cuda.device(1)
transformers.logging.set_verbosity_info()

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
now = datetime.datetime.now()
dt_string = now.strftime('%Y-%d-%m_T%H-%M-%S')
filename = dt_string + "_" + "opt_log.txt"

# dataset = get_dataset('cc_news')
dataset = get_dataset_from_disk('/cc_news_reduced.txt')

csv_name = 'file2.csv'
csv_columns = ["vocab_size","hidden_size","num_hidden_layers","num_attention_heads","intermediate_size","hidden_act","hidden_dropout_prob","attention_probs_dropout_prog", "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", "gradient_checkpointing","position_embedding_type","use_cache","energy_consumption","perplexity","energy_loss","loss","date"]
def objective(params):
    """
    Function to set up the model and train it.
    Loss function is based on energy consumption times perplexity
    """

    with open(config_path + '/config.json') as json_file:
        random.seed(25565)
        config = json.load(json_file)
        epochs = config['train_epochs']

        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

        inputs = tokenizer.batch_encode_plus(
            dataset, truncation=True, padding=True, verbose=True, max_length=config['model_parameters'][0]['max_position_embeddings']
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        model = RoBERTaModel(params)
        model.model.resize_token_embeddings(len(tokenizer))

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
            logging_dir='./logs',
            eval_accumulation_steps=10 
       )

        trainer = Trainer(
            model=model.model,
            args=training_args,
            train_dataset=inputs['input_ids'],
            eval_dataset=inputs['input_ids'],
            data_collator=data_collator,
            callbacks=[CarbonTrackerCallback(epochs)],
            optimizers=(optimizer, scheduler)
        )

        train_metrics = trainer.train()
        _, loss, metrics = train_metrics
        perplexity = math.exp(loss)

        # This is v erry cringe code
        time.sleep(60)
        print(f"Carbonpath log directory: {carbondir_path}")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f"Current working path: {dir_path}")
        print(f"CWD: {os.getcwd()}")
        logs = parser.parse_all_logs(log_dir=carbondir_path)
        print(f"Log length: {len(logs)}")
        latest_log = logs[len(logs)-1]
        print("LATEST LOG")
        print(latest_log)
        print("LATEST LOG ACTUAL")
        print(latest_log['actual'])
        energy_consumption = latest_log['actual']['energy (kWh)']
        
        energy_loss = perplexity * energy_consumption
        
        print('##################################')
        print(f'Energy Consumption: {energy_consumption}')
        print(f'Perplexity: {perplexity}')
        print(f'Energy Loss: {energy_loss}')
        print('##################################')
        

        post = params.copy()
        post['loss'] = loss
        post['perplexity'] = perplexity
        post['energy_consumption'] = energy_consumption
        post['energy_loss'] = energy_loss
        post['date'] = datetime.datetime.utcnow()
        
        with open(csv_name, 'a+') as result_file:
            writer = csv.DictWriter(result_file, fieldnames=csv_columns)
            writer.writerow(post)
        
        trainer.save_model('trained_model')
        
        with open(results_path + '/' + filename, 'a+') as log_file:
            log_file.write('###################################\n')
            log_file.write('MODEL PARAMS\n')
            log_file.write(json.dumps(params))
            log_file.write('\n')
            log_file.write(f'Perplexity: {perplexity}\n')
            log_file.write(f'Energy Consumption: {energy_consumption}\n')
            log_file.write(f'Energy Loss: {energy_loss}\n')

        return energy_loss



space = {
    'vocab_size': hp.uniformint('vocab_size', 1, 30522),
    'hidden_size': hp.uniformint('hidden_size_multiplier', 1, 100),
    'num_hidden_layers': hp.uniformint('hidden_layers', 1, 12),
    'num_attention_heads': hp.uniformint('attention_heads', 1, 18),
    'intermediate_size': hp.uniformint('intermediate_size', 1, 3072),
    'hidden_act': hp.choice('hidden_act', [
        'gelu',
        'relu',
        'silu',
        'gelu_new'
    ]),
    'hidden_dropout_prob': hp.uniform('hidden_dropout_prob', 0.1, 1),
    'attention_probs_dropout_prog': hp.uniform('attention_prob_dropout_prog', 0.1, 1),
    'max_position_embeddings': 512,
    'type_vocab_size': 1,
    'initializer_range': 0.02,
    'layer_norm_eps': 1e-12,
    'gradient_checkpointing': False,
    'position_embedding_type': hp.choice('position_embedding_type', [
        'absolute',
        'relative_key',
        'relative_key_query'
    ]),
    'use_cache': True,
}

trials = MongoTrials('mongo://root:pass123@135.181.38.74:27017/admin/jobs?authSource=admin', exp_key='test1')
best = fmin(objective,
            space=space,
            trials=trials,
            algo=tpe.suggest,
            max_evals=100)


with open(results_path + '/' + filename, 'a+') as log_file:
    log_file.write('###################################\n')
    log_file.write(f'BEST: {best}\n')
    log_file.write(f'SPACE EVAL: {space_eval(space, best)}\n')

