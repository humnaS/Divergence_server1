import os
import pprint
import talib
import sys
import numpy as np
from scipy import stats
import plotly.express as px
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
import cufflinks as cf
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
from itertools import cycle, islice
import statsmodels.api as sm
#import plotly.offline as py
import chart_studio.plotly as py
cf.go_offline()
init_notebook_mode(connected=True)
import plotly.graph_objects as go
from matplotlib import pyplot
import matplotlib.ticker as ticker
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pyplot import figure
import matplotlib
import matplotlib.pyplot as plt
import ppscore as pps
MEDIUM_SIZE = 10
plt.rc('axes', labelsize=MEDIUM_SIZE) 
from matplotlib.pyplot import figure
import datetime
from datetime import datetime
import glob
import csv
from pyexcel.cookbook import merge_all_to_a_book
import pandas as pd
from indicators import (adx, macd, macd_percentile, moving_average,
                        plot_agg_divergence, plot_agg_momentum, plot_data, pnl,
                        pnl_n, pnl_percentile, rsi, rsi_percentile, williams,cci_percentile)
from oanda import get_candles
specific_insts=None
plt.style.use('ggplot')

PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data/daily/v8'
)

DATA_FILES = {
    'Open': 'open.xls',
    'High': 'high.xls',
    'Low': 'low.xls',
    'Close': 'close.xls',
    'Volume': 'volume.xls',
    'Date': 'date.xls',
}

COLUMNS = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
]

SORT_COLUMN = 'adx'

# start and end period in days
START_PERIOD = -100
END_PERIOD = -1
# last day
SELECTED_PERIOD = -1
SELECTED_INSTRUMENT = 'AUD_CAD'


# OANDA API variables
INSTRUMENTS = [
    'AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD', 'AUD_SGD',
    'AUD_USD', 'BCO_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CAD_SGD',
    'CHF_HKD', 'CHF_JPY', 'CHF_ZAR', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF',
    'EUR_CZK', 'EUR_DKK', 'EUR_GBP', 'EUR_HKD', 'EUR_HUF', 'EUR_JPY',
    'EUR_NOK', 'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 'EUR_TRY',
    'EUR_USD', 'EUR_ZAR', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD',
    'GBP_JPY', 'GBP_NZD', 'GBP_PLN', 'GBP_SGD', 'GBP_USD', 'GBP_ZAR',
    'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_JPY', 'NZD_SGD',
    'NZD_USD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USB02Y_USD',
    'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'USD_CAD', 'USD_CHF',
    'USD_CNH', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR',
    'USD_JPY', 'USD_MXN', 'USD_NOK', 'USD_PLN', 'USD_SAR', 'USD_SEK',
    'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'ZAR_JPY',
]
# https://developer.oanda.com/rest-live-v20/instrument-df/#CandlestickGranularity
GRANULARITY = [
    'S5', #5 second candlesticks, minute alignment
    'S10', #10 second candlesticks, minute alignment
    'S15', # 15 second candlesticks, minute alignment
    'S30', # 30 second candlesticks, minute alignment
    'M1', #  1 minute candlesticks, minute alignment
    'M2', #  2 minute candlesticks, hour alignment
    'M4', #  4 minute candlesticks, hour alignment
    'M5', #  5 minute candlesticks, hour alignment
    'M10', # 10 minute candlesticks, hour alignment
    'M15', # 15 minute candlesticks, hour alignment
    'M30', # 30 minute candlesticks, hour alignment
    'H1', #  1 hour candlesticks, hour alignment
    'H2', #  2 hour candlesticks, day alignment
    'H3', #  3 hour candlesticks, day alignment
    'H4', #  4 hour candlesticks, day alignment
    'H6', #  6 hour candlesticks, day alignment
    'H8', #  8 hour candlesticks, day alignment
    'H12', # 12 hour candlesticks, day alignment
    'D',# 1 day candlesticks, day alignment
    'W', #  1 week candlesticks, aligned to start of week
    'M', # 1 month candlesticks, aligned to first day of the month
]

PERIODS = 500

def rank_formulation(array):
    values=[]
    for index in range(len(array)):
        value = array[index]
        compare_array = array[:index]
        rank = stats.percentileofscore(compare_array,value)
        rank = rank / 100
        values.append(rank)
    return values

def change_in_price(price):
    cprice=[]
    #cprice.append(price[0])
    #price=np.delete(price,0)
    for i in range(0,len(price)):
        if i==0:
            cprice.append(price[i])
        else:
            a=np.log(price[i]/price[i-1])
            cprice.append(a)
    return np.array(cprice)

def get_file_data():
    '''
    Read the data from the csv or excel files
    Key is olhc
    '''
    values = {}
    for key, filename in DATA_FILES.items():
        path = os.path.join(PATH, filename)
        if not os.path.exists(path):
            print(f'file {path} not found')
            continue
        if key not in values:
            values[key] = None
        if filename.endswith('.csv'):
            values[key] = pd.read_csv(path)
        elif filename.endswith('.xls'):
            values[key] = pd.read_excel(path)
    return values


def get_oanda_api(
    instruments,
    granularity='D',
    prices=['B'],
    since=None,
    until=None,
    count=PERIODS,
    daily_alignment=17,
):
    # https://developer.oanda.com/rest-live-v20/instrument-ep
    params = {
        'granularity': granularity,
        'prices': prices,
        'since': since,
        'until': until,
        'count': count,
        'dailyAlignment': daily_alignment,
        'alignmentTimezone': 'Europe/London',
    }
    return get_candles(instruments, params)


def get_ig_api(
    instruments,
    granularity='D',
    prices=['B'],
    since=None,
    until=None,
    count=PERIODS
):
    params = {
        'granularity': granularity,
        'prices': prices,
        'since': since,
        'until': until,
        'count': count
    }
    return get_candles(instruments, params)


def heatmap1(x, y, **kwargs):
    inst_name=kwargs["instrument_name"]
    del kwargs["instrument_name"]
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=30, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    ax.set_title("Correaltion of "+inst_name)
    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 


def corrplot(data, size_scale=500, marker='s',instrument_name="instrument"):
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap1(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale,
        instrument_name=instrument_name
    )

##PPS
def heatmap(df,instrument_name):
    df = df[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    ax.set_title("PPS matrix of "+instrument_name)
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
    return ax


def main():
    global specific_insts
    debug = False
    if not debug:
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 1000)
    user_input=input("Do You Want Select Instrument Manually:y or n")
    man_inst_names=None
    if user_input=="y":
        man_inst_names=input("Enter Instrument names Separated by Space:")
        man_inst_names = man_inst_names.split()

    

    instruments = INSTRUMENTS
    # # instruments = ['AUD_CAD',]
    # # instruments = ['AUD_CHF',]
    # instruments = ['AUD_CHF', 'AUD_CAD']
    # instruments = ['AUD_CAD',]
    #data = get_file_data()
    stock = get_oanda_api(instruments, granularity='D')
  

    # print (stock['AUD_CAD'])

    # stock = get_ig_api(instruments)
    # print (stock['AUD_CAD'])


    # instruments = data['Close'].columns.values
    # # Initialize all assign all instrument data to dataframes
    # stock = {}
    # for instrument in instruments:
    #     values = {}
    #     for key in COLUMNS:
    #         values[key] = data.get(key, {}).get(instrument)
    #     values['Date'] = data.get('Date').iloc[:len(values[key]), 0]
    #     stock[instrument] = pd.DataFrame(values, columns=COLUMNS)

    # print(stock[SELECTED_INSTRUMENT])
    # return
    # Calculate the MACD, RSI and Profit and Loss for all instrument paid
    # Also, Calculate the MACD, RSI and Profit and Loss percentile for all
    # instruments
    instruments_list=[]
    CCI_list=[]
    dic_for_all_cci={}

    for instrument in instruments:
        
    
        nsize = len(stock[instrument]['Close'])

        # Calculate MACD
        stock[instrument] = stock[instrument].join(
            macd(stock[instrument]['Close'])
        )


        # Calculate RSI for n = 14
        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close'])
        )

        #changeInPrice
        stock[instrument]["Change In Price"] = change_in_price(stock[instrument]["Close"].values)

    

        # Calculate Profile and Loss
        stock[instrument] = stock[instrument].join(
            pnl(stock[instrument]['Close'])
        )
        # Calculate MACD Percentile
        stock[instrument] = stock[instrument].join(
            macd_percentile(stock[instrument]['MACD'])
        )
        # Calculate RSI Percentile
        stock[instrument] = stock[instrument].join(
            rsi_percentile(stock[instrument]['RSI'])
        )
        # Calculate  Profile and Loss Percentile
        stock[instrument] = stock[instrument].join(
            pnl_percentile(stock[instrument]['Profit/Loss'])
        )



        #Calculate CCI
        high = stock[instrument]["High"].values
        close = stock[instrument]["Close"].values
        low = stock[instrument]["Low"].values
        #create instrument dataframe
        ccis=talib.CCI(high,low,close,timeperiod=14)
        #ccis=list(ccis)
        instruments_list.append(instrument)
        CCI_list.append(ccis[-1])
        dic_for_all_cci[instrument]=ccis

        stock[instrument]["CCI"]=ccis
      

        # Calculate Divergence factor 1 and 2
        stock[instrument] = stock[instrument].join(
            pd.Series(
                (
                    stock[instrument]['MACD Percentile'] + 0.1 -
                    stock[instrument]['RSI Percentile']
                ) / 2.0,
                name='Divergence Factor 1'
            )
        )
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['Divergence Factor 1'] -
                stock[instrument]['PNL Percentile'],
                name='Divergence Factor 2'
            )
        )
        
        # Calculate Divergence factor 3
        n = 19
        for i in range(nsize):
            stock[instrument].loc[i: nsize, 'Macd_20'] = (
                stock[instrument]['MACD'].iloc[i] -
                stock[instrument]['MACD'].iloc[i - n]
            )
            stock[instrument].loc[i: nsize, 'Prc_20'] = (
                (stock[instrument]['Close'].iloc[i] -
                    stock[instrument]['Close'].iloc[i - n])
            ) / stock[instrument]['Close'].iloc[i - n]
            stock[instrument].loc[i: nsize, 'Divergence Factor 3'] = (
                stock[instrument]['Macd_20'].iloc[i] /
                stock[instrument]['Close'].iloc[i]
            ) - stock[instrument]['Prc_20'].iloc[i]

        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close'], 20, name='RSI_20')
        )

        # Calculate the momentum factors
        stock[instrument] = stock[instrument].join(
            pnl_n(stock[instrument]['Close'], 10)
        )
        stock[instrument] = stock[instrument].join(
            pnl_n(stock[instrument]['Close'], 30)
        )

        stock[instrument]['Close_fwd'] = stock[instrument]['Close'].shift(-2)
        stock[instrument].loc[-1: nsize, 'Close_fwd'] = stock[instrument]['Close'].iloc[-1]
        stock[instrument].loc[-2: nsize, 'Close_fwd'] = stock[instrument]['Close'].iloc[-2]

        stock[instrument] = stock[instrument].join(
            macd(
                stock[instrument]['Close_fwd'],
                name='MACD_fwd'
            )
        )
        n = 19
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['MACD_fwd'].diff(n) - stock[instrument]['MACD'],
                name='M_MACD_CHANGE'
            )
        )

        stock[instrument] = stock[instrument].join(
            rsi(stock[instrument]['Close_fwd'], n=20, name='RSI_20_fwd')
        )
        stock[instrument] = stock[instrument].join(
            pd.Series(
                stock[instrument]['RSI_20_fwd'] - stock[instrument]['RSI_20'],
                name='M_RSI_CHANGE'
            )
        )

        # Calculate the ADX, PDI & MDI
        _adx, _pdi, _mdi = adx(stock[instrument])

        stock[instrument] = stock[instrument].join(_adx)
        stock[instrument] = stock[instrument].join(_pdi)
        stock[instrument] = stock[instrument].join(_mdi)

        # Calculate the Moving Averages: 5, 10, 20, 50, 100
        for period in [5, 10, 20, 50, 100]:
            stock[instrument] = stock[instrument].join(
                moving_average(
                    stock[instrument]['Close'],
                    period,
                    name=f'{period}MA'
                )
            )

        # Calculate the Williams PCTR
        stock[instrument] = stock[instrument].join(
            williams(stock[instrument])
        )

        # Calculate the Minmax Range
        n = 17
        for i in range(nsize):
            maxval = stock[instrument]['High'].iloc[i - n: i].max()
            minval = stock[instrument]['Low'].iloc[i - n: i].min()
            rng = abs(maxval) - abs(minval)
            # where is the last price in the range of minumimn to maximum
            pnow = stock[instrument]['Close'].iloc[i - n: i]
            if len(pnow.iloc[-1: i].values) > 0:
                whereinrng = (
                    (pnow.iloc[-1: i].values[0] - abs(minval)) / rng
                ) * 100.0
                stock[instrument].loc[i: nsize, 'MinMaxPosition'] = whereinrng
                stock[instrument].loc[i: nsize, 'High_Price(14)'] = maxval
                stock[instrument].loc[i: nsize, 'Low_Price(14)'] = minval

        stock[instrument]['Divergence factor Avg'] = (
            stock[instrument]['Divergence Factor 1'] +
            stock[instrument]['Divergence Factor 2'] +
            stock[instrument]['Divergence Factor 3']
        ) / 3.0

        stock[instrument]['Momentum Avg'] = (
            stock[instrument]['M_MACD_CHANGE'] +
            stock[instrument]['M_RSI_CHANGE'] +
            stock[instrument]['Profit/Loss_10'] +
            stock[instrument]['Profit/Loss_30']
        ) / 4.0

        df_instrument=pd.DataFrame()
        df_instrument["High"]=stock[instrument]['High']
        df_instrument["Low"]=stock[instrument]['Low']
        df_instrument["Close"]=stock[instrument]['Close']
        df_instrument["Volume"]=stock[instrument]['Volume']
        df_instrument["Price"]=stock[instrument]['Close']
        df_instrument["Change In Price"]=change_in_price(stock[instrument]['Close'].values)
        df_instrument["CCI"]=stock[instrument]['CCI']
        df_instrument["PNL Percentile"]=stock[instrument]['PNL Percentile']
        df_instrument["Divergence Factor 1"]=stock[instrument]['Divergence Factor 1']
        df_instrument["Divergence Factor 2"]=stock[instrument]['Divergence Factor 2']
        df_instrument["Divergence Factor 3"]=stock[instrument]['Divergence Factor 3']
        df_instrument["RSI"]=stock[instrument]["RSI"]
        df_instrument["MACD"]=stock[instrument]["MACD"]
        df_instrument["WPCTR"]=stock[instrument]["Williams PCTR"]
        df_instrument["pdi"]=stock[instrument]["pdi"]
        df_instrument["mdi"]=stock[instrument]["mdi"]
        df_instrument["adx"]=stock[instrument]["adx"]
        #df_instrument= df_instrument[pd.notnull(df_instrument['CCI'])]
        df_instrument=df_instrument.dropna(how="any")
        df_instrument["CCI Percentile"]=cci_percentile(df_instrument["CCI"])
        df_instrument["Divergence Factor 4"]=df_instrument["CCI Percentile"]-df_instrument["PNL Percentile"]
        df_instrument['Divergence Factor 1 Rank'] = rank_formulation(df_instrument['Divergence Factor 1'].values)
        df_instrument['Divergence Factor 2 Rank'] = rank_formulation(df_instrument['Divergence Factor 2'].values)
        df_instrument['Divergence Factor 3 Rank'] = rank_formulation(df_instrument['Divergence Factor 3'].values)
        df_instrument['Divergence Factor 4 Rank'] = rank_formulation(df_instrument['Divergence Factor 4'].values)
        df_instrument['DF Avg Rank'] = (
            df_instrument['Divergence Factor 1 Rank'] +
            df_instrument['Divergence Factor 2 Rank'] +
            df_instrument['Divergence Factor 3 Rank'] +
            df_instrument['Divergence Factor 4 Rank']
        ) / 4.0
        df_instrument["% Rank of DF Avgs"] =rank_formulation(df_instrument['DF Avg Rank'].values)
        df_instrument=df_instrument[['High','Low','Close','Volume','Price','Change In Price','Divergence Factor 1', 'Divergence Factor 2', 'Divergence Factor 3', 'Divergence Factor 4',
           'Divergence Factor 1 Rank', 'Divergence Factor 2 Rank', 'Divergence Factor 3 Rank','Divergence Factor 4 Rank',
           'DF Avg Rank', '% Rank of DF Avgs','RSI', 'MACD', 'WPCTR', 'CCI', 'CCI Percentile', 'PNL Percentile','pdi', 'mdi', 'adx',]]
        df_instrument.to_csv(instrument+".csv")
    
    ccis_df=pd.DataFrame(dic_for_all_cci)
    cci_percentile_list=[]
    dic={"Instrument":instruments_list,"CCI":CCI_list}
    new_df=pd.DataFrame(dic)
    cci_percentile_list=cci_percentile(new_df["CCI"]).to_list()
 
    #sys.exit()
    # calculate the aggregrate for each oeruod
    # calculate the Divergence_Macd_Prc_Rank
    for nrow in range(nsize):
        row = [
            stock[instrument]['Divergence Factor 3'].iloc[nrow] for
            instrument in instruments
        ]
        series = pd.Series(row).rank() / len(row)
        for i, instrument in enumerate(instruments):
            stock[instrument].loc[nrow: nsize, 'Divergence_Macd_Prc_Rank'] = series.iloc[i]

    # calculate the Divergence and Momentum average rank
    indices = [instrument for instrument in instruments]
    columns = [
        'Price',
        "Change In Price",
        'Divergence Factor 1',
        'Divergence Factor 2',
        'Divergence Factor 3',
        'Divergence Factor 1 Rank',
        'Divergence Factor 2 Rank',
        'Divergence Factor 3 Rank',
        'M_MACD_CHANGE',
        'M_RSI_CHANGE',
        'Profit/Loss_10',
        'Profit/Loss_30',
        'M_MACD_CHANGE Rank',
        'M_RSI_CHANGE Rank',
        'Profit/Loss_10 Rank',
        'Profit/Loss_30 Rank',
        'Momentum Average Rank',
        'Momentum Averages Ranks',
        'MinMaxPosition',
        'RSI',
        'WPCTR',
        'pdi', 'mdi', 'adx',
        'High_Price(14)',
        'Low_Price(14)',
        '5MA', '10MA', '20MA', '50MA', '100MA',
        "MACD",
        'PNL Percentile',
        "DF Avg Rank",
        "% Rank of DF Avgs",
    ]

    periods = []
    for i in range(nsize):

        period = []

        for instrument in instruments:

            period.append([
                stock[instrument]['Close'].iloc[i],
                stock[instrument]["Change In Price"].iloc[i],
                stock[instrument]['Divergence Factor 1'].iloc[i],
                stock[instrument]['Divergence Factor 2'].iloc[i],
                stock[instrument]['Divergence Factor 3'].iloc[i],
                None,
                None,
                None,
                stock[instrument]['M_MACD_CHANGE'].iloc[i],
                stock[instrument]['M_RSI_CHANGE'].iloc[i],
                stock[instrument]['Profit/Loss_10'].iloc[i],
                stock[instrument]['Profit/Loss_30'].iloc[i],
                None,
                None,
                None,
                None,
                None,
                None,
                stock[instrument]['MinMaxPosition'].iloc[i],
                stock[instrument]['RSI'].iloc[i],
                stock[instrument]['Williams PCTR'].iloc[i],
                stock[instrument]['pdi'].iloc[i],
                stock[instrument]['mdi'].iloc[i],
                stock[instrument]['adx'].iloc[i],
                stock[instrument]['High_Price(14)'].iloc[i],
                stock[instrument]['Low_Price(14)'].iloc[i],
                stock[instrument]['5MA'].iloc[i],
                stock[instrument]['10MA'].iloc[i],
                stock[instrument]['20MA'].iloc[i],
                stock[instrument]['50MA'].iloc[i],
                stock[instrument]['100MA'].iloc[i],
                stock[instrument]["MACD"].iloc[i],
                stock[instrument]['PNL Percentile'].iloc[i],
                None,
                None,
            ])
        df = pd.DataFrame(data=period, index=indices, columns=columns)
        df['Divergence Factor 1 Rank'] =rank_formulation(df["Divergence Factor 1"].values)
        df['Divergence Factor 2 Rank'] = rank_formulation(df["Divergence Factor 2"].values)
        df['Divergence Factor 3 Rank'] = rank_formulation(df["Divergence Factor 3"].values)
        df['M_MACD_CHANGE Rank'] = rank_formulation(df['M_MACD_CHANGE'].values)
        df['M_RSI_CHANGE Rank'] = rank_formulation(df['M_RSI_CHANGE'].values)
        df['Profit/Loss_10 Rank'] = rank_formulation(df['Profit/Loss_10'].values)
        df['Profit/Loss_30 Rank'] = rank_formulation(df['Profit/Loss_30'].values)
        df['Momentum Average Rank'] = (
            df['M_MACD_CHANGE Rank'] +
            df['M_RSI_CHANGE Rank'] +
            df['Profit/Loss_10 Rank'] +
            df['Profit/Loss_30 Rank']
        ) / 4.0
        df['Momentum Averages Ranks'] = rank_formulation(df['Momentum Average Rank'].values)

        #df.to_excel("target_data.xlsx")
        periods.append(df)
    pnl_percentile_nparaay=np.array(df["PNL Percentile"].values)
    cci_percentile_nparray=cci_percentile_list
    divergent_factor_4=cci_percentile_nparray-pnl_percentile_nparaay
    df["CCI"]=CCI_list
    df["CCI Percentile"]=cci_percentile_list
    df["Divergence Factor 4"]=divergent_factor_4
    df['Divergence Factor 4 Rank'] = rank_formulation(df['Divergence Factor 1'].values)
    df['DF Avg Rank'] = (
            df['Divergence Factor 1 Rank'] +
            df['Divergence Factor 2 Rank'] +
            df['Divergence Factor 3 Rank'] +
            df['Divergence Factor 4 Rank']
        ) / 4.0
    df["% Rank of DF Avgs"] = rank_formulation(df['DF Avg Rank'].values)
    df=df[['Price', 'Change In Price','Divergence Factor 1', 'Divergence Factor 2', 'Divergence Factor 3', 'Divergence Factor 4',
           'Divergence Factor 1 Rank', 'Divergence Factor 2 Rank', 'Divergence Factor 3 Rank','Divergence Factor 4 Rank',
           'DF Avg Rank', '% Rank of DF Avgs', 'M_MACD_CHANGE', 'M_RSI_CHANGE', 'Profit/Loss_10', 'Profit/Loss_30', 
           'M_MACD_CHANGE Rank', 'M_RSI_CHANGE Rank', 'Profit/Loss_10 Rank', 'Profit/Loss_30 Rank', 'Momentum Average Rank', 
           'Momentum Averages Ranks', 'MinMaxPosition', 'RSI', 'MACD', 'WPCTR', 'CCI', 'CCI Percentile', 'PNL Percentile',
           'pdi', 'mdi', 'adx', 'High_Price(14)', 'Low_Price(14)', '5MA', '10MA', '20MA', '50MA', '100MA']]
    df.to_excel("target_data.xlsx")
    df.sort_values(by="% Rank of DF Avgs",inplace=True)
    df.to_excel("ordered_target_data.xlsx")
    top5,last5=instrument_selection_rules(df)
    specific_insts=top5+last5
    print(specific_insts)
   
    Graph_Plots_For_Individual_Instrument(specific_insts,False)
    if man_inst_names is not None:
        Graph_Plots_For_Individual_Instrument(man_inst_names,True)
    
    
    '''
    if SELECTED_INSTRUMENT in INSTRUMENTS:
        print(stock[SELECTED_INSTRUMENT])
        print(
            periods[SELECTED_PERIOD].sort_values(by=[SORT_COLUMN], ascending=True)
        )
        plot_agg_divergence(
            stock, periods, SELECTED_INSTRUMENT, START_PERIOD, END_PERIOD
        )
        plot_agg_momentum(
            stock, periods, SELECTED_INSTRUMENT, START_PERIOD, END_PERIOD
        )
        plot_data(stock, SELECTED_INSTRUMENT, START_PERIOD, END_PERIOD)
    '''




def instrument_selection_rules(df):
    top5=df.head(5)
    last5=df.tail(5)
    top5_indicies = top5[(top5['RSI']<30)].index.tolist()
    last5_indicies=last5[(last5['RSI']>70)].index.tolist()
    return top5_indicies,last5_indicies

############################################################# kand shuru yhn se
def Plot_Items(instrument_name,flag):
    data = pd.read_csv(instrument_name+".csv")
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    N = 100
    x = data.index
    y = data["Change In Price"]
    y2 = data["DF Avg Rank"]
    y3 = data["% Rank of DF Avgs"]
    
    data_plot = data[["Change In Price","DF Avg Rank","% Rank of DF Avgs"]]
    title = instrument_name + "'s Change In Price graph"
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
    ax.set_ylabel("Change In Price",color="orange",fontsize=26)

    
    title = "Price change over time for " + instrument_name + "."
    plt.title(title,fontname="Times New Roman",fontweight="bold", fontsize=22)

    # save the plot as a file
    if flag==True:
        fig0.savefig("man_select_inst"+"\\"+instrument_name +'_price.png',
                format='png')
    else:
        fig0.savefig("rule_select_inst"+"\\"+instrument_name +'_price.png',
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
    if flag==True:
        fig1.savefig("man_select_inst"+"\\"+instrument_name +'_dfavg.png',
                format='png')
    else:
        fig1.savefig("rule_select_inst"+"\\"+instrument_name +'_dfavg.png',
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
    if flag==True:
        fig2.savefig("man_select_inst"+"\\"+instrument_name +'_percentile_df.png',
                format='png')
    else:
        fig2.savefig("rule_select_inst"+"\\"+instrument_name +'_percentile_df.png',
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

    if flag==True:
        fig.write_image("man_select_inst"+"\\"+instrument_name +"_price-df-rank.png")
    else:
        fig.write_image("rule_select_inst"+"\\"+instrument_name +"_price-df-rank.png")
    
    #save_image(instrument_name +"_price-df-rank.png")
    #total 4 images

def Plot_Multi_Axes(instrument_name,flag):
    #instrument_name = "AUD_USD"
    data = pd.read_csv(instrument_name + ".csv")
    data = data.tail(90)
    print(len(data))
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)

    N = 100
    x = data.index
    y = data["Change In Price"]
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
    ax.set_ylabel("Change In Price",color="orange",fontsize=16)

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
    if flag==True:
        fig.savefig("man_select_inst"+"\\"+instrument_name +'_two_different_y_axis.jpg',
                format='png',
                dpi=100,
                bbox_inches='tight')
    else:
        fig.savefig("rule_select_inst"+"\\"+instrument_name +'_two_different_y_axis.jpg',
                format='png',
                dpi=100,
                bbox_inches='tight')
    



def decompose_plot(instrument_name,flag):
    data = pd.read_csv(instrument_name + ".csv")
    data = data.tail(90)
    data.set_index('Date',inplace=True)
    data.index = data.index.map(str)
    series = data["DF Avg Rank"]
    
    result = seasonal_decompose(series, model='additive',period=12)
    

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
    if flag==True:
        plt.savefig("man_select_inst"+"\\"+fig_name)
    else:
        plt.savefig("rule_select_inst"+"\\"+fig_name)



def Divergence_Plots():
    data = pd.read_excel("ordered_target_data.xlsx")

    print(data.head())
    # Making a copy of data frame and dropping all the null values
    df_copy = data.copy()
    df_copy = df_copy.dropna(axis = 1)
    df_copy = df_copy.dropna()

    print(len(df_copy))
    
    X = df_copy[["CCI","RSI","MACD","WPCTR","pdi","mdi","adx","Divergence Factor 1","Divergence Factor 2","Divergence Factor 3","Divergence Factor 4"]]
    y = df_copy["DF Avg Rank"]

    print(len(X),len(y))
    # Embedded method
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)

    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    dic={"imp_var":X.columns,"score":reg.coef_}
    print(dic)
    dff=pd.DataFrame(dic)
    #df.drop(df[df.score<=0].index, inplace=True)
    dff = dff[dff.score>0]
    print(dff)
    dff.to_csv()


    sys.exit()
    fig,ax = plt.subplots(1)
    sns.set()
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (10, 14)
    matplotlib.rcParams['ytick.labelsize'] = 3
    matplotlib.rcParams['axes.labelsize']  = 3
    matplotlib.rcParams['legend.fontsize']  = 1

    plt.rc('axes', titlesize=12)
    plt.yticks(fontsize=9.5)
    my_colors = list(islice(cycle(['orange','b', 'r', 'g', 'y', 'k','m']), None, len(df_copy)))
    ax = imp_coef.plot(kind = "barh", stacked=True, color=my_colors, width=0.91,align='edge')
    ax.yaxis.label.set_size(3)

    title = "Feature importance using the Lasso Model"
    plt.title(title)
    fig_name = "EmbaddedMethod" + ".png" 
    plt.savefig(fig_name)


def Divergence_Plots_For_Single_Instrument(instrument_name,flag):
    
    data = pd.read_csv(instrument_name + ".csv")

    # Making a copy of data frame and dropping all the null values
    df_copy = data.copy()
    df_copy = df_copy.dropna(axis = 1)
    df_copy = df_copy.dropna()

    print(len(df_copy))
    
    X = df_copy[["CCI","RSI","MACD","WPCTR","pdi","mdi","adx","Divergence Factor 1","Divergence Factor 2","Divergence Factor 3","Divergence Factor 4"]]
    y = df_copy["DF Avg Rank"]

    print(len(X),len(y))
    # Embedded method
    reg = LassoCV()
    reg.fit(X, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = X.columns)

    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


    fig,ax = plt.subplots(1)
    sns.set()
    imp_coef = coef.sort_values()
    matplotlib.rcParams['figure.figsize'] = (10, 14)
    matplotlib.rcParams['ytick.labelsize'] = 3
    matplotlib.rcParams['axes.labelsize']  = 3
    matplotlib.rcParams['legend.fontsize']  = 1

    plt.rc('axes', titlesize=12)
    plt.yticks(fontsize=9.5)
    my_colors = list(islice(cycle(['orange','b', 'r', 'g', 'y', 'k','m']), None, len(df_copy)))
    ax = imp_coef.plot(kind = "barh", stacked=True, color=my_colors, width=0.91,align='edge')
    ax.yaxis.label.set_size(3)

    title = f"Feature importance of {instrument_name} using the Lasso Model"
    plt.title(title)
    fig_name =  instrument_name + "_EmbaddedMethod" + ".png" 
    if flag==True:
        plt.savefig("man_select_inst"+"\\"+fig_name)
    else:
        plt.savefig("rule_select_inst"+"\\"+fig_name)


def plot_peaks_troughs(inst_name,flag):
    df=pd.read_csv(inst_name+".csv")
    merge_all_to_a_book(glob.glob(inst_name+".csv"), inst_name+".xlsx")
    df_new=pd.read_excel(inst_name+".xlsx")
    for i in range(len(df_new)):
        date = df.loc[i,"Date"]
        date_upd = date.split("+")[0]
        
        df.loc[i,"Date"] = date_upd
        df_new.loc[i,"Date_Time"] = datetime.strptime(df.loc[i,"Date"], '%Y-%m-%d %H:%M:%S')
    
    df_new=df_new.tail(90) #get last 90 sample
    x = np.array(df_new["Date_Time"].tolist())
    x=[i.strftime('%Y-%m-%d %H:%M:%S') for i in x]
    print("instrument name",inst_name)
    
    df_new["Date"]=x
    peak90,peak90time,trough90,trough90time=plot_and_return_peak_trough(df_new,inst_name,90,flag)
    peak30,peak30time,trough30,trough30time=plot_and_return_peak_trough(df_new.tail(30),inst_name,30,flag)

    peak90df=pd.DataFrame({"Date":peak90time,"Peak":peak90})
    trough90df=pd.DataFrame({"Date":trough90time,"Trough":trough90})
    peak30df=pd.DataFrame({"Date":peak30time,"Peak":peak30})
    trough30df=pd.DataFrame({"Date":trough30time,"Trough":trough30})
    if flag==True:
        peak90df.to_csv("man_select_inst"+"\\"+inst_name+"peak90"+".csv")
        trough90df.to_csv("man_select_inst"+"\\"+inst_name+"trough90"+".csv")
        peak30df.to_csv("man_select_inst"+"\\"+inst_name+"peak30"+".csv")
        trough30df.to_csv("man_select_inst"+"\\"+inst_name+"trough30"+".csv")
    else:
        peak90df.to_csv("rule_select_inst"+"\\"+inst_name+"peak90"+".csv")
        trough90df.to_csv("rule_select_inst"+"\\"+inst_name+"trough90"+".csv")
        peak30df.to_csv("rule_select_inst"+"\\"+inst_name+"peak30"+".csv")
        trough30df.to_csv("rule_select_inst"+"\\"+inst_name+"trough30"+".csv")



def plot_and_return_peak_trough(df,inst_name,num,flag):
    dates = np.array(df["Date"].tolist())
    returns = np.array(df["DF Avg Rank"].tolist())
    minimas = (np.diff(np.sign(np.diff(returns))) > 0).nonzero()[0] + 1 
    maximas = (np.diff(np.sign(np.diff(returns))) < 0).nonzero()[0] + 1
    figure(figsize=(30, 8))
    plt.xticks(rotation=60)
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
    if flag==True:
        plt.savefig("man_select_inst"+"\\"+inst_name+str(num)+".png", bbox_inches='tight')
    else:
        plt.savefig("rule_select_inst"+"\\"+inst_name+str(num)+".png", bbox_inches='tight')
    return peak,pt,trough,tt

def Graph_Plots_For_Individual_Instrument(instrument_lst,flag):
    for instrument in instrument_lst:
        Plot_Items(instrument,flag)
        decompose_plot(instrument,flag)
        Divergence_Plots_For_Single_Instrument(instrument,flag)
        plot_peaks_troughs(instrument,flag)
        Plot_Multi_Axes(instrument,flag)
        #better corr graph
        data=pd.read_csv(instrument+".csv",index_col="Date")
        plt.figure(figsize=(15, 10))
        corrplot(data.corr(), size_scale=500,instrument_name=instrument)
        if flag==True:
            plt.savefig("man_select_inst"+"\\"+"BetterCorr_"+instrument+".png")
        else:
            plt.savefig("rule_select_inst"+"\\"+"BetterCorr_"+instrument+".png")
        #PPS
        matrix = pps.matrix(data)
        plt.figure(figsize=(18, 15))
        heatmap(matrix,instrument)
        if flag==True:
            plt.savefig("man_select_inst"+"\\"+"PPS_"+instrument+".png")
        else:
            plt.savefig("rule_select_inst"+"\\"+"PPS_"+instrument+".png")

    Divergence_Plots()
    
if __name__ == '__main__':
    main()


