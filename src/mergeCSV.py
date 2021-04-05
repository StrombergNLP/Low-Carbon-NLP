#"vocab_size","hidden_size","num_hidden_layers","num_attention_heads","intermediate_size","hidden_act","hidden_dropout_prob","attention_probs_dropout_prog", "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", "gradient_checkpointing","position_embedding_type","use_cache","energy_consumption","perplexity","energy_loss","loss","date"

import os, glob
import pandas as pd

#"vocab_size","hidden_size","num_hidden_layers","num_attention_heads",
# "intermediate_size","hidden_act","hidden_dropout_prob","attention_probs_dropout_prog", 
# "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", 
# "gradient_checkpointing","position_embedding_type","use_cache","energy_consumption",
# "perplexity","energy_loss","loss","date"

path = os.getcwd()
results_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
Header = ['vocab_size','hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','hidden_act','hidden_dropout_prob','attention_probs_dropout_prog', 'max_position_embeddings', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing','position_embedding_type','use_cache','energy_consumption','perplexity','energy_loss','loss','date']

def extract_all_csv():
    all_files = glob.glob(os.path.join(results_path, "param_results_*.csv"))
    return all_files

def combine_csv():
    all_files = extract_all_csv()
    all_df = []
    for f in all_files:
        df = pd.read_csv(f, sep=',', header = None)
        all_df.append(df)
    
    df_merged = pd.concat(all_df, ignore_index=True)
    df_merged.columns = Header
    df_merged.to_csv( "merged.csv")
    return print(len(all_files))
    
combine_csv()