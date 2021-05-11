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
    df = df.drop(['max_position_embeddings', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing', 'use_cache', 'id', 'perplexity', 'energy_consumption'], axis=1)
    df['hidden_act'] = df['hidden_act'].astype('category').cat.codes
    df['position_embedding_type'] = df['position_embedding_type'].astype('category').cat.codes
    
    activation = df['hidden_act']
    embedding_type = df['position_embedding_type']
    energy_loss = df['energy_loss']

    act = correlation_ratio(activation, energy_loss)
    emb = correlation_ratio(embedding_type, energy_loss)

    corr = df.corr().loc[:'position_embedding_type', 'energy_loss':]
    corr['energy_loss']['hidden_act'] = act
    corr['energy_loss']['position_embedding_type'] = emb

    cmap = sb.diverging_palette(220, 20, as_cmap=True)
    sb.heatmap(corr, center=0.0, cmap=cmap, annot=True)
    fig=plt.gcf()
    plt.show()
    plt.savefig(path + '.png')
    plt.clf()

plot_corr_matrix('10epochs')
plot_corr_matrix('15worst')
plot_corr_matrix('15best')