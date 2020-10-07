from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def decompose_plot():
    instrument_name = "AUD_USD"
    data = pd.read_csv("AUD_USD.csv")
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    series = data["DF Avg Rank"]
    #series = series[:90]
    print(series)
   
    result = seasonal_decompose(series, model='additive',period=12)
    # result.plot()
    # # pyplot.show()
    # fig, axes = plt.subplots()
    # ax = axes[1]
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # fig = result.plot()
    # fig.set_figheight(10)
    # fig.set_figwidth(20)  
    # plt.xticks(fontsize=0.8)

    observed = result.observed
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    df = pd.DataFrame({"observed":observed,"trend":trend, "seasonal":seasonal,"residual":residual})

    _, axes = plt.subplots(nrows=4,ncols=1, figsize=(28 , 22))
    for i, ax in enumerate(axes):
        ax = df.iloc[:,i].plot(ax=ax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(18))
        #ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        #ax.xaxis.set_minor_formatter(fmt)
        ax.set_ylabel(df.iloc[:,i].name)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=0)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        plt.xticks(fontsize=0.5)


    title = instrument_name +" Decomposition of Divergence Time series."
    plt.title(title,fontname="Times New Roman",fontweight="bold")

    fig_name = instrument_name +"_Divergence_Decomposition" + ".png" 
    plt.savefig(fig_name)

decompose_plot()

