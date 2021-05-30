import os, glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats

from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic


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
    
    sns.set_theme()
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
    sns.set_theme()
    sns.scatterplot(data=df, x='PPL', y='Energy', hue='PPL', palette='deep', legend=False)  #with nice colours but weird legends
    #sns.scatterplot(data=df, x='PPL', y='Energy')
    plt.xlabel('Perplexity, tranlated and scaled')
    plt.ylabel('Energy Consumption (kWh), tranlated and scaled')
    plt.title('Scaled and normalised for Epoch ' + epoch)
    plt.show()


#transform and normalise with sklearn
def transform_sklearn(dfPPL, dfEnergy):
    names = list(dfPPL)
    #atempt 3:
    dnppl = preprocessing.normalize(dfPPL, axis=0)

    dsnppl = preprocessing.scale(dnppl, axis=0)
    scaled_ppl_df = pd.DataFrame(dsnppl, columns=names)

    dnenergy = preprocessing.normalize(dfEnergy, axis=0)
    dsnenergy = preprocessing.scale(dnenergy, axis=0)
    scaled_energy_df = pd.DataFrame(dsnenergy, columns=names)

    return (scaled_ppl_df, scaled_energy_df)




def transform(dfPPL, dfEnergy, epoch, param):
    dfTransPPL = transform_to_standard_norm(dfPPL)
    dfTransEnergy = transform_to_standard_norm(dfEnergy)
    #dfPropTransPPL = prop_norm(dfTransPPL)                 #Calculates the probability on the n(0,1) for ppl
    #dfPropTransEnergy = prop_norm(dfTransEnergy)           #Calculates the probability on the n(0,1) for energy
    #save_df_to_csv(dfTransPPL, 'PPLTransformed')           #Saves the transformed ppl in results
    #save_df_to_csv(dfTransEnergy, 'EnergyTransformed')     #Saves the transformed energy in results

    
    if param == 1:
        make_transformed_graph_1_axis(dfTransPPL, "PPL")
    if param == 2:
        make_transformed_graph_1_axis(dfTransEnergy, "Energy")
    if param == 3: 
        epochData = combine_data_epoch(dfTransPPL, dfTransEnergy, epoch)
        make_transformed_combined(epochData, epoch)
    if param == 4:
        df_scaledPPL, df_scaledEnergy = transform_sklearn(dfPPL, dfEnergy)
        epochData_scaled = combine_data_epoch(df_scaledPPL, df_scaledEnergy, epoch)
        #save_df_to_csv(df_scaledPPL, 'SklearnScaledPPL')                               #<--- Saves the scaled&normalised data for ppl
        #save_df_to_csv(df_scaledEnergy, 'SklearnScaledEnergy')                         #<--- Saves the scaled&normalised data for energy
        make_transformed_combined(epochData_scaled, epoch)


if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    dfPPL = read_data(data_path + '/perplexity_all_epochs.csv').iloc[: , 1:]
    dfEnergy = read_data(data_path + '/energy_all_epochs.csv').iloc[: , 1:]

    #<-----------------------------------------------------------------------------------
    # changeable parameter:
    #param = 1: histogram and normDist for perplexity
    #param = 2: histogram and normDist for energy
    #param = 3: dual standard normal plot 
    #param = 4: sklearn norm and scale
    param = 4
    epoch = '1'
    #<-----------------------------------------------------------------------------------
    transform(dfPPL, dfEnergy, epoch, param)

