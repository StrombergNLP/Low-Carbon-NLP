from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/actualbase.csv')
df = df.drop(['id', 'vocab_size', 'hidden_size','num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'actual_hidden_size', 'hidden_act', 'hidden_dropout_prob',
'attention_probs_dropout_prog', 'max_position_embeddings', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing', 'position_embedding_type',
'use_cache', 'energy_loss'], axis=1)

df = df[['perplexity', 'energy_consumption']]
df['energy_consumption'] = df['energy_consumption'] * 142.3439911

X = df.to_numpy()

clustering = DBSCAN(eps=200, min_samples=2).fit(X)
labels = clustering.labels_
print(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.ylabel("Energy Consumption Scaled")
plt.xlabel("Perplexity")
plt.show()