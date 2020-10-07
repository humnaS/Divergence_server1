from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import cufflinks as cf
#import plotly.offline as py
import chart_studio.plotly as py
cf.go_offline()
init_notebook_mode(connected=True)
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

def Plot_Items():
    instrument_name = "AUD_USD"
    data = pd.read_csv("AUD_USD.csv")
    #my_fig = data[["Price","DF Avg Rank","DF Avgs Ranks"]].iplot()
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    N = 100
    x = data.index
    y = data["Price"]
    y2 = data["DF Avg Rank"]
    y3 = data["% Rank of DF Avgs"]
    
    data_plot = data[["Price","DF Avg Rank","% Rank of DF Avgs"]]
    title = instrument_name + "'s Price graph"
    # create figure and axis objects with subplots()
    fig0,ax = plt.subplots(figsize=(15, 17))
    # make a plot
    ax.plot(x, y, color="orange", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=40)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)
    # set y-axis label
    ax.set_ylabel("Price change",color="orange",fontsize=26)

    
    title = "Price change over time for " + instrument_name + "."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=22)

    # save the plot as a file
    fig0.savefig(instrument_name +'_price.png',
                format='png')


    #figures1_df = px.bar(data_plot, y="DF Avg Rank", color="% Rank of DF Avgs",title= title, barmode="group")
    
    fig1,ax = plt.subplots(figsize=(15, 17))
    # make a plot
    ax.plot(x, y2, color="blue", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=40)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)
    # set y-axis label
    ax.set_ylabel("Average Divergence Percentile Rank",color="blue",fontsize=26)

    
    title =  "Average Divergence Percentile Rank over time for " + instrument_name + "."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=22)

    # save the plot as a file
    fig1.savefig(instrument_name +'_price.png',
                format='png')


    
    fig2,ax = plt.subplots(figsize=(15, 17))
    # make a plot
    ax.plot(x, y3, color="red", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=40)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)
    # set y-axis label
    ax.set_ylabel("Percentile Rank of Average Divergence",color="red",fontsize=26)

    
    title =   "Percentile Rank of Average Divergence over time for " + instrument_name + "."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=22)

    # save the plot as a file
    fig2.savefig(instrument_name +'_price.png',
                format='png')




    # title = instrument_name + "'s Divergence Rank graph"
    # figures2_Rank = px.bar(data_plot, y="% Rank of DF Avgs", color="% Rank of DF Avgs",title= title, barmode="group")
   
    title = "Percentage Price change , Average Divergence Percentile Rank and Percentile Rank of Average Divergence over time for " + instrument_name + "."
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
                        mode='lines+markers',
                        name='Price'))
    fig.add_trace(go.Scatter(x=x, y=y2,
                        mode='lines+markers',
                        name='DF Avg Rank'))
    fig.add_trace(go.Scatter(x=x, y=y3,
                        mode='lines+markers', name='% Rank of DF Avgs'))
    fig.update_layout(
    title={
        'text': title
        })

  

  
    
   

    #figures0_price.write_image(instrument_name +"_price.png")
    #figures1_df.write_image(instrument_name +"_df.png")
    #figures2_Rank.write_image(instrument_name +"_%Rank.png")
  
    fig.write_image(instrument_name +"_price-df-rank.png")


Plot_Items()


def Plot_Multi_Axes():
    instrument_name = "AUD_USD"
    data = pd.read_csv("AUD_USD.csv")
    #my_fig = data[["Price","DF Avg Rank","DF Avgs Ranks"]].iplot()
    data = data.tail(90)
    print(len(data))
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)

    N = 100
    x = data.index
    y = data["Price"]
    y2 = data["DF Avg Rank"]
    #y3 = data["% Rank of DF Avgs"]
    
    #data_plot = data[["Price","DF Avg Rank","% Rank of DF Avgs"]]

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(14, 16))
    # make a plot
    ax.plot(x, y, color="orange", marker="o")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=10)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=70)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
    # set y-axis label
    ax.set_ylabel("Price change",color="orange",fontsize=16)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(x, y2,color="blue",marker="o")
    ax2.set_ylabel("Average Divergence Percentile Rank",color="blue",fontsize=16)

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.xaxis.set_major_formatter(years_fmt)
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    plt.setp(ax2.xaxis.get_minorticklabels(), rotation=70)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)

    title = "Price change and Average Divergence Percentile Rank over the time for "+ instrument_name +"."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=20)

    #plt.show()
    # save the plot as a file
    fig.savefig(instrument_name +'_two_different_y_axis.jpg',
                format='png',
                dpi=100,
                bbox_inches='tight')




#Plot_Multi_Axes()