import os, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats


def read_data(path):
    return pd.read_csv(path)

def plot_linear(df):
    sns.set_theme()
    #sns.regplot(x='energy_consumption', y='perplexity', data=df, fit_reg=True, label=True, ci=None) #<--- not scaled
    #sns.regplot(x='scaled_energy', y='scaled_ppl', data=df, fit_reg=True, label=True, ci=None) #<--- scaled
    #sns.regplot(x='perplexity', y='energy_consumption', data=df, fit_reg=True, label=True, ci=None) #<--- not scaled
    sns.regplot(x='scaled_ppl', y='scaled_energy', data=df, fit_reg=True, label=True, ci=None) #<--- scaled
    plt.show()


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    dfEnergyLoss = read_data(data_path + '/cluster2.csv').iloc[: , 1:]

    plot_linear(dfEnergyLoss)

