import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
from datetime import datetime
import os
import glob
import csv
from pyexcel.cookbook import merge_all_to_a_book

def plot_peaks_troughs(inst_name):
    df=pd.read_csv(inst_name+".csv")
    merge_all_to_a_book(glob.glob(inst_name+".csv"), inst_name+".xlsx")
    df_new=pd.read_excel(inst_name+".xlsx")
    for i in range(len(df_new)):
        date = df.loc[i,"Date"]
        date_upd = date.split("+")[0]
        
        df.loc[i,"Date"] = date_upd
        df_new.loc[i,"Date_Time"] = datetime.strptime(df.loc[i,"Date"], '%Y-%m-%d %H:%M:%S')
    
    df_new=df_new.tail(90)
    x = np.array(df_new["Date_Time"].tolist())
    x=[i.strftime('%Y-%m-%d %H:%M:%S') for i in x]
    df_new["Date"]=x
    peak90,peak90time,trough90,trough90time=plot_and_return_peak_trough(df_new,inst_name,90)
    peak30,peak30time,trough30,trough30time=plot_and_return_peak_trough(df_new.tail(30),inst_name,30)

    peak90df=pd.DataFrame({"Date":peak90time,"Peak":peak90})
    trough90df=pd.DataFrame({"Date":trough90time,"Trough":trough90})
    peak30df=pd.DataFrame({"Date":peak30time,"Peak":peak30})
    trough30df=pd.DataFrame({"Date":trough30time,"Trough":trough30})
    peak90df.to_csv(inst_name+"peak90"+".csv")
    trough90df.to_csv(inst_name+"trough90"+".csv")
    peak30df.to_csv(inst_name+"peak30"+".csv")
    trough30df.to_csv(inst_name+"trough30"+".csv")


def plot_and_return_peak_trough(df,inst_name,num):
    dates = np.array(df["Date"].tolist())
    returns = np.array(df["DF Avg Rank"].tolist())
    minimas = (np.diff(np.sign(np.diff(returns))) > 0).nonzero()[0] + 1 
    maximas = (np.diff(np.sign(np.diff(returns))) < 0).nonzero()[0] + 1
    figure(figsize=(50, 8))
    plt.xticks(rotation=90)
    plt.xlabel('DateTime') 
    plt.ylabel('Average Divergence Percentile Rank') 
    plt.title("Peak and Trough Analysis for "+inst_name+" over time") 
    plt.plot(dates, returns)
    peak=[]
    trough=[]
    pt=[]
    tt=[]
    for minima in minimas:
        plt.plot(df.iloc[minima]["Date"], df.iloc[minima]["DF Avg Rank"], marker="v")
        tt.append(df.iloc[minima]["Date"])
        trough.append(minima)
    for maxima in maximas:
        plt.plot(df.iloc[maxima]["Date"], df.iloc[maxima]["DF Avg Rank"], marker="^")
        pt.append(df.iloc[maxima]["Date"])
        peak.append(maxima)
    plt.savefig(inst_name+str(num)+".png", bbox_inches='tight')
    return peak,pt,trough,tt


plot_peaks_troughs("AUD_CAD")