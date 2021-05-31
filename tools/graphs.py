import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(path):
    return pd.read_csv(path)


def identify_pareto(scores):
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    pareto_front = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                pareto_front[i] = 0
                break

    return population_ids[pareto_front]


def plot_pareto_graph(df):
    x = df['perplexity'].tolist()
    y = df['energy_consumption'].tolist()
    scores = np.array(list(zip(x, y)))
    pareto = identify_pareto(scores)
    pareto_front = scores[pareto]

    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values

    pareto_frame = df.iloc[pareto]

    x_pareto = pareto_front[:, 0]
    y_pareto = pareto_front[:, 1]

    sns.set_theme()
    for index, row in df.iterrows():
        sns.scatterplot(x=[row['perplexity']], y=[row['energy_consumption']])
    sns.lineplot(x=x_pareto, y=y_pareto, drawstyle='steps-pre')
    plt.title('Hyperopt Tuning Epoch 3')
    plt.xlabel('Perplexity (lower is better)')
    plt.ylabel('Energy Consumption (kWh)')
    plt.show()


def correlation_ratio(categories, measurements):
    '''
    Blatantly stolen from this article:
    https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    '''
    fcat, _ = pd.factorize(categories) #ADDED Comment: pandas.factorize() method helps to get the numeric representation of an array by identifying distinct values. 
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)

    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))

    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)

    return eta


def plot_correlation_heatmap(df):
    df = df.drop(['max_position_embeddings', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'gradient_checkpointing', 'use_cache', 'energy_loss', 'loss', 'id'], axis=1)
    df['hidden_act'] = df['hidden_act'].astype('category').cat.codes
    df['position_embedding_type'] = df['position_embedding_type'].astype('category').cat.codes

    # calculate correlation between categorical entries and perplexity/energy
    activation = df['hidden_act']
    embedding_type = df['position_embedding_type']
    perplexity = df['perplexity']
    energy_consumption = df['energy_consumption']

    act_perplex = correlation_ratio(activation, perplexity)
    act_energy = correlation_ratio(activation, energy_consumption)
    embedding_perplex = correlation_ratio(embedding_type, perplexity)
    embedding_energy = correlation_ratio(embedding_type, energy_consumption)

    # the .loc takes everything from the start to the given parameter as y axis, and then everything from the given parameter to the last one as x
    correlation_matrix = df.corr().loc[:'position_embedding_type', 'energy_consumption':]
    correlation_matrix['perplexity']['hidden_act'] = act_perplex
    correlation_matrix['energy_consumption']['hidden_act'] = act_energy
    correlation_matrix['perplexity']['position_embedding_type'] = embedding_perplex
    correlation_matrix['energy_consumption']['position_embedding_type'] = embedding_energy

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, center=0.0, cmap=cmap, annot=True)
    plt.show()


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    df = read_data(data_path + '/model_data_3epochs.csv')

    # plot_correlation_heatmap(df)
    plot_pareto_graph(df)

