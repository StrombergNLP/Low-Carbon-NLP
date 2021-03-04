from hyperopt import fmin, tpe, hp, Trials

def objective(attention_heads):
    return {
        'loss': attention_heads ** 2,
    }

space = {
    'vocab_size': hp.uniform('vocab_size', 15261, 30522),
    'hidden_size': hp.uniform('hidden_size', 384, 768),
    'hidden_layers': hp.uniform('hidden_layers', 3, 12),
    'attention_heads': hp.uniform('attention_heads_x', 6, 18),
    'intermediate_size': hp.uniform('intermediate_size', 1536, 3072),
    'hidden_act': hp.choice('hidden_act', [
        {'act_type': 'gelu'},
        {'act_type': 'relu'},
        {'act_type': 'silu'},
        {'act_type': 'gelu_new'},
    ]),
    'hidden_dropout_prob': hp.normal('hidden_dropout_prob', 0.1, 0.1),
    'attention_prob_dropout_prog': hp.normal('attention_prob_dropout_prog', 0.1, 0.1),
    'max_position_embeddings': hp.uniform('max_position_embeddings', 128, 256),
    'position_embedding_type': hp.choice('position_embedding_type', [
        {'embedding_type': 'absolute'},
        {'embedding_type': 'relative_key'},
        {'embedding_type': 'relative_key_query'},
    ])
}

trials = Trials()
best = fmin(objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)

