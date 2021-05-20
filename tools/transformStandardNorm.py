import os, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats


def read_data(path):
    return pd.read_csv(path)

#Essentially this is the tranformed values, corresponding to the "x" values on a graph, 
#while the probaility given "x" for a normal distribution: N(0,1) is the y-axis, 
# with highest point of probability centered arround mean: 0
def transform_to_standard_norm(df):
    rows, columns = df.shape
    mean = df.mean()
    std = df.std()
    maxValues = df.max()
    minValues = df.min()

    dfnew = df.copy()

    for X in dfnew:
        x = int(X)
        dfnew[X] = dfnew[X].apply(lambda a: (a - mean[x-1])/std[x-1])

    return dfnew

#might be irrelevant, calculated the probability for all transformed values
def prop_norm(df):
    mean = 0
    std = 1
    normalDist = scipy.stats.norm(mean, std)
    dfnew = df.copy()

    for X in dfnew:
        x = int(X)
        dfnew[X] = dfnew[X].apply(lambda a: normalDist.pdf(a))

    return dfnew

def save_df_to_csv(df, name):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    save_path = data_path + '/' + name + '.csv'
    df.to_csv(save_path)


#plots  the histogram/normal distribution for either 
def make_transformed_graph_1_axis(dfx, name):
    mean = 0
    std = 1
    epoch = 1
    data = dfx[str(epoch)]

    plt.hist(data, bins=10, density=True, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mean, std)

    plt.plot(x,p, 'k', linewidth=2)
    title = "standard norm " + name
    plt.title(title)

    plt.show()

#combining data from ppl and energy for a specific epoch into one datafram with 
# 2 columns: PPL, Energy
def combine_data_epoch(dfPPL, dfEnergy, epoch):
    rows, columns = dfPPL.shape

    dfnew = dfPPL.copy()
    dfnew = dfnew.drop(dfnew.loc[:,:].columns, axis=1)

    dfnew['PPL'] = dfPPL[[epoch]]
    dfnew['Energy'] = dfEnergy[[epoch]]

    return dfnew

#param 3 graph, follow a translation and scale so both axis have mean 0 and s.d. 1
def make_transformed_combined(df, epoch):
    #sns.scatterplot(data=df, x='PPL', y='Energy', hue='PPL', palette='dark')  #with nice colours but weird legends
    sns.scatterplot(data=df, x='PPL', y='Energy')
    plt.xlabel('Perplexity, tranlated and scaled')
    plt.ylabel('Energy Consumption (kWh), tranlated and scaled')
    plt.show()


def transform(dfPPL, dfEnergy, epoch, param):
    dfTransPPL = transform_to_standard_norm(dfPPL)
    dfTransEnergy = transform_to_standard_norm(dfEnergy)
    #dfPropTransPPL = prop_norm(dfTransPPL)                 #Calculates the probability on the n(0,1) for ppl
    #dfPropTransEnergy = prop_norm(dfTransEnergy)           #Calculates the probability on the n(0,1) for energy
    #save_df_to_csv(dfTransPPL, 'PPLTransformed')           #Saves the transformed ppl in results
    #save_df_to_csv(dfTransEnergy, 'EnergyTransformed')     #Saves the transformed energy in results

    epochData = combine_data_epoch(dfTransPPL, dfTransEnergy, epoch)

    if param == 1:
        make_transformed_graph_1_axis(dfTransPPL, "PPL")
    if param == 2:
        make_transformed_graph_1_axis(dfTransEnergy, "Energy")
    if param == 3: 
        make_transformed_combined(epochData, epoch)


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    dfPPL = read_data(data_path + '/perplexity_all_epochs.csv').iloc[: , 1:]
    dfEnergy = read_data(data_path + '/energy_all_epochs.csv').iloc[: , 1:]

    #<-----------------------------------------------------------------------------------
    # changeable parameter:
    #param = 1: histogram and normDist for perplexity
    #param = 2: histogram and normDist for energy
    #param = 3: dual standard normal plot TODO
    param = 3
    epoch = '1'
    #<-----------------------------------------------------------------------------------
    transform(dfPPL, dfEnergy, epoch, param)

