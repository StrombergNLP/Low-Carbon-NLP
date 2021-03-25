import os
import random
import json
import transformers
import torch
import math

from datetime import datetime
from transformers import EvalPrediction
from transformers import Trainer, TrainingArguments
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AdamW
from datasets import load_dataset
from carbontracker import parser
from model.RoBERTaModel import RoBERTaModel
from callbacks.CarbonTrackerCallback import CarbonTrackerCallback
from callbacks.PerplexityCallback import PerplexityCallback

from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.mongoexp import MongoTrials

def get_dataset(dataset_name):
    dataset = load_dataset(dataset_name, script_version='master')

    dataset_reduced = dataset['train']['text'][:100000]
    del dataset

    return dataset_reduced

torch.cuda.device(1)
transformers.logging.set_verbosity_info()

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
now = datetime.now()
dt_string = now.strftime('%Y-%d-%m_T%H-%M-%S')
filename = dt_string + "_" + "opt_log.txt"

dataset = get_dataset(config['dataset'])

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
            callbacks=[CarbonTrackerCallback(epochs), PerplexityCallback()],
            optimizers=(optimizer, scheduler)
        )

        train_metrics = trainer.train()
        _, loss, metrics = train_metrics
        perplexity = math.exp(loss)

        # This is v erry cringe code
        logs = parser.parse_all_logs(log_dir='./carbon_logs/')
        latest_log = logs[len(logs)-1]
        energy_consumption = latest_log['actual']['energy (kWh)']
        
        energy_loss = perplexity * energy_consumption
        
        print('##################################')
        print(f'Energy Consumption: {energy_consumption}')
        print(f'Perplexity: {perplexity}')
        print(f'Energy Loss: {energy_loss}')
        print('##################################')
        

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
    'max_position_embeddings': hp.uniformint('max_position_embeddings', 1, 512),
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

trials = MongoTrials('mongodb://root:pass123@135.181.38.74:27017/admin/jobs?authSource=admin', exp_key='exp1')
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100)


with open(results_path + '/' + filename, 'a+') as log_file:
    log_file.write('###################################\n')
    log_file.write(f'BEST: {best}\n')
    log_file.write(f'SPACE EVAL: {space_eval(space, best)}\n')
