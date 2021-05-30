import os, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats


def read_data(path):
    return pd.read_csv(path)

def plot_energy_loss(df):
    dfnew = []
    for ind in df.index:
        d = pd.DataFrame(df.loc[ind].reset_index())
        d.columns = ['Epoch', 'Energy Loss']
        d['Model'] = ind
        d['Epoch'] = d['Epoch'].astype(int)
        dfnew.append(d)
    ddf = pd.concat(dfnew)
    sns.set_theme()
    sns.lineplot(x='Epoch', y='Energy Loss', hue='Model', data=ddf, legend=False, palette='deep') # <--- works
    plt.show()


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    dfEnergyLoss = read_data(data_path + '/energy_loss_all_epochs.csv').iloc[: , 1:]

    plot_energy_loss(dfEnergyLoss)

