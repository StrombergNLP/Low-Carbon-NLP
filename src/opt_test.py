
import json, os
from datetime import datetime
from hyperopt import fmin, tpe, hp, Trials

results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
now = datetime.now()
dt_string = now.strftime('%Y-%d-%m_T%H-%M-%S')
filename = dt_string + "_" + "opt_log.txt"

def objective(params):
    loss = params['num_hidden_layers'] * params['num_attention_heads'] ** params['num_attention_heads']

    with open(results_path + '/' + filename, 'a+') as log_file:
        log_file.write('###################################\n')
        log_file.write('MODEL PARAMS\n')
        log_file.write(json.dumps(params))
        log_file.write('\n')
        log_file.write(f'loss: {loss}\n')


    return loss


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


trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials)

print(best)

