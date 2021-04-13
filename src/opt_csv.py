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

import pandas as pd

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
from transformers import PrinterCallback, DefaultFlowCallback
from datasets import load_dataset
from carbontracker import parser
from timeit import default_timer as timer
from model.RoBERTaModel import RoBERTaModel
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback
from callbacks.SaveModelCallback import SaveModelCallback


def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name, script_version='master')

    dataset_reduced = dataset['train']['text'][:100000]
    del dataset

    return dataset_reduced


def get_dataset_from_disk(dataset_name):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    dataset = load_dataset('text', data_files=data_path + dataset_name)
    return dataset['train']['text']


def main(params, dataset, config_path, results_path):
    model_id = params['id']
    params_file_name = sys.argv[2]
    csv_name =  'param_results_' + sys.argv[1] + '.csv'
    csv_columns = ['id','vocab_size','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','hidden_act','hidden_dropout_prob','attention_probs_dropout_prog', 'max_position_embeddings', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing','position_embedding_type','use_cache','energy_consumption','perplexity','energy_loss','loss','date', 'time']
    carbondir_path = './carbon_logs/' + 'carbon_log_id_' + str(model_id) + '/'
    
    if not os.path.exists(carbondir_path):
        os.mkdir(carbondir_path)

    print('#################################')
    print('THIS IS THE MODEL ID, THIS IS IMPORTANT')
    print(f'Model ID: {model_id}')
    print('#################################')

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
            callbacks=[
                CarbonTrackerCallback(epochs, carbondir_path, model_id, results_path, params_file_name),
                SaveModelCallback(results_path, model_id),
                DefaultFlowCallback(),
                PrinterCallback(),
            ],
            optimizers=(optimizer, scheduler)
        )

        start = timer()
        train_metrics = trainer.train()
        end = timer()
        time_spent = end - start
        _, loss, metrics = train_metrics
        perplexity = math.exp(loss)

        # This is v erry cringe code
        print(f'Carbonpath log directory: {carbondir_path}')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        logs = parser.parse_all_logs(log_dir=carbondir_path)
        print(f'Log length: {len(logs)}')
        latest_log = logs[len(logs)-1]
        print('LATEST LOG')
        print(latest_log)
        print('LATEST LOG ACTUAL')
        print(latest_log['actual'])
        energy_consumption = latest_log['actual']['energy (kWh)']
        
        energy_loss = perplexity * energy_consumption
        
        print('##################################')
        print(f'Energy Consumption: {energy_consumption}')
        print(f'Perplexity: {perplexity}')
        print(f'Energy Loss: {energy_loss}')
        print(f'Time: {time_spent}')
        print('##################################')
        

        post = params.copy()
        post['loss'] = loss
        post['perplexity'] = perplexity
        post['energy_consumption'] = energy_consumption
        post['energy_loss'] = energy_loss
        post['date'] = datetime.datetime.utcnow()
        post['time'] = time
        
        with open(results_path + '/' + csv_name, 'a+') as result_file:
            writer = csv.DictWriter(result_file, fieldnames=csv_columns)
            writer.writerow(post)

        # trainer.save_model('model_3epochs_id_' + str(params['id']))
        


if __name__ == '__main__':
    torch.cuda.device(1)
    transformers.logging.set_verbosity_info()

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
    results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    now = datetime.datetime.now()

    # dataset = get_dataset('cc_news')
    dataset = get_dataset_from_disk('/cc_news_reduced.txt')

    params_file = params_path + '/' + sys.argv[2] + '.csv'
    df = pd.read_csv(params_file)
    df.drop(['energy_consumption', 'perplexity', 'energy_loss', 'loss', 'date'], axis=1)

    models = df.transpose().to_dict()

    for model in models:
        params = models[model]
        main(params, dataset, config_path, results_path)

