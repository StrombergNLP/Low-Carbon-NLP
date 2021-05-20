import csv
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from graphs import correlation_ratio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from yellowbrick.cluster import KElbowVisualizer

def plot_corr_matrix(path):
    df = pd.read_csv('data/' + path + '.csv')
    df = df.drop(['max_position_embeddings', 'hidden_size', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing', 'use_cache', 'id'], axis=1)
    df['hidden_act'] = df['hidden_act'].astype('category').cat.codes
    df['position_embedding_type'] = df['position_embedding_type'].astype('category').cat.codes
    
    activation = df['hidden_act']
    embedding_type = df['position_embedding_type']
    energy_loss = df['energy_loss']
    perplexity = df['perplexity']
    energy_consumption = df['energy_consumption']

    act_loss = correlation_ratio(activation, energy_loss)
    emb_loss = correlation_ratio(embedding_type, energy_loss)
    act_perplex = correlation_ratio(activation, perplexity)
    act_energy = correlation_ratio(activation, energy_consumption)
    embedding_perplex = correlation_ratio(embedding_type, perplexity)
    embedding_energy = correlation_ratio(embedding_type, energy_consumption)

    corr = df.corr()#.loc[:'position_embedding_type', 'energy_loss':]
    corr['energy_loss']['hidden_act'] = act_loss
    corr['energy_loss']['position_embedding_type'] = emb_loss
    corr['perplexity']['hidden_act'] = act_perplex
    corr['perplexity']['position_embedding_type'] = embedding_perplex
    corr['energy_consumption']['hidden_act'] = act_energy
    corr['energy_consumption']['position_embedding_type'] = embedding_energy

    cmap = sb.diverging_palette(220, 20, as_cmap=True)
    heatmap = sb.heatmap(corr, center=0.0, cmap=cmap, annot=True)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
    fig=plt.gcf()
    plt.show()
    #plt.savefig(path + '.png')
    plt.clf()

#plot_corr_matrix('10epochs')
#plot_corr_matrix('15worst')
#plot_corr_matrix('15best')
#plot_corr_matrix('actualbase')
#plot_corr_matrix('3epochs_outlier')
plot_corr_matrix('base_without_outliers')