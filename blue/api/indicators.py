import datetime

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator

plt.style.use('ggplot')
np.seterr(divide='ignore', invalid='ignore')


def pnl(df, name='Profit/Loss'):
    return pd.Series(
        (df - df[0]) * 1000,
        name=name
    )


def pnl_n(df, n=10, name='Profit/Loss'):
    return pd.Series(
        (df - df.shift(n-1)) / (df * .0001),
        name=f'{name}_{n}'
    )


def moving_average(df, n, name='Moving Average'):
    """Calculate the moving average for the given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    return pd.Series(
        df.rolling(n, min_periods=n).mean(),
        name=name
    )


def exponential_moving_average(df, n, name='EMA'):
    """Calculate the EMA

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    return pd.Series(
        df.ewm(span=n, min_periods=n).mean(),
        name=name
    )


def macd(df, n_fast=12, n_slow=26, signal=9, name='MACD'):
    """Calculate MACD, MACD Signal and MACD difference

    :param df: pandas.DataFrame
    :param n_fast:
    :param n_slow:
    :return: pandas.DataFrame
    """
    '''
    MATLAB:
    f = xlsread('/Users/tolu/fxtrade/data2/close.xls');
    macd(f(1:end,1))
    ema26p = movavg(f(1:end,1),'exponential',26);
    ema12p = movavg(f(1:end,1),'exponential',12);
    '''
    emafast = pd.Series(
        df.ewm(adjust=False, span=n_fast, min_periods=n_slow).mean()
    )
    emaslow = pd.Series(
        df.ewm(adjust=False, span=n_slow, min_periods=n_slow).mean()
    )
    return pd.Series(
        emafast - emaslow,
        name=name
    )


def rsi(df, n=14, name='RSI'):
    """Calculate Relative Strength Index(RSI) for given data.

    MATLAB:
    f = xlsread('/Users/tolu/fxtrade/data2/close.xls');
    New_datax = f(1:end,1)
    Table_rsi = rsindex(New_datax);
    Table_rsi_rank = tiedrank(Table_rsi)/(length(Table_rsi)-14);

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    delta = df.diff()
    d_up, d_down = delta.copy(), delta.copy()
    d_up[d_up < 0] = 0
    d_down[d_down > 0] = 0
    rol_up = d_up.rolling(n).mean()
    rol_down = d_down.rolling(n).mean().abs()
    rs = rol_up / rol_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, name=name)


def macd_percentile(df, name='MACD Percentile', n=26):
    return pd.Series(
        df.rank() / (len(df) - n),
        name=name
    )

def cci_percentile(df, n=26):
    return (df.rank() / (len(df) - n))

def rsi_percentile(df, name='RSI Percentile', n=14):
    return pd.Series(
        df.rank() / (len(df) - n),
        name=name
    )


def pnl_percentile(df, name='PNL Percentile'):
    return pd.Series(
        df.rank() / len(df),
        name=name
    )


def williams(df, lbp=14, fillna=False, name='Williams PCTR'):
    # highest high over lookback period lbp
    hh = df['High'].rolling(lbp).max()
    # lowest low over lookback period lbp
    ll = df['Low'].rolling(lbp).min()
    wr = -100 * (hh - df['Close']) / (hh - ll)
    if fillna:
        wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    return pd.Series(wr, name=name)


def adx(df, n=14, fillna=False):
    """Calculate the Average Directional Movement Index for given data.

    :param df: pandas.DataFrame
    :param n:
    :param n_ADX:
    :return: pandas.DataFrame
    """
    cs = df['Close'].shift(1)
    pdm = df['High'].combine(
        cs,
        lambda x1,
        x2: max(x1, x2) if np.isnan(x1) == False and np.isnan(x2) == False else np.nan
    )
    pdn = df['Low'].combine(
        cs,
        lambda x1,
        x2: min(x1, x2) if np.isnan(x1)==False and np.isnan(x2) == False else np.nan
    )
    tr = pdm - pdn

    trs_initial = np.zeros(n - 1)
    trs = np.zeros(len(df['Close']) - (n - 1))
    trs[0] = tr.dropna()[0:n].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs) - 1):
        trs[i] = trs[i - 1] - (trs[i - 1]/float(n)) + tr[n + i]

    up = df['High'] - df['High'].shift(1)
    dn = df['Low'].shift(1) - df['Low']
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(df['Close']) - (n - 1))
    dip_mio[0] = pos.dropna()[0:n].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio) - 1):
        dip_mio[i] = dip_mio[i - 1] - (dip_mio[i - 1] / float(n)) + pos[n + i]

    din_mio = np.zeros(len(df['Close']) - (n - 1))
    din_mio[0] = neg.dropna()[0:n].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio) - 1):
        din_mio[i] = din_mio[i - 1] - (din_mio[i - 1] / float(n)) + neg[n + i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        if trs[i] == 0:
            continue
        dip[i] = 100.0 * (dip_mio[i] / trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        if trs[i] == 0:
            continue
        din[i] = 100.0 * (din_mio[i] / trs[i])

    dx = 100.0 * np.abs((dip - din) / (dip + din))

    dip = np.roll(dip, 1)
    din = np.roll(din, 1)

    adx = np.zeros(len(trs))
    adx[n] = dx[0:n].mean()

    for i in range(n + 1, len(adx)):
        adx[i] = ((adx[i - 1] * (n - 1)) + dx[i - 1]) / float(n)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=df['Close'].index)

    pdi = np.concatenate((trs_initial, dip), axis=0)
    pdi = pd.Series(data=pdi, index=df['Close'].index)

    mdi = np.concatenate((trs_initial, din), axis=0)
    mdi = pd.Series(data=mdi, index=df['Close'].index)

    if fillna:
        adx = adx.replace([np.inf, -np.inf], np.nan).fillna(20)

    return (
        pd.Series(adx, name='adx'),
        pd.Series(pdi, name='pdi'),
        pd.Series(mdi, name='mdi'),
    )


def plot_date(df, column, title, ylabel=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    min_date = None
    max_date = None

    dates = [dateutil.parser.parse(d) for d in df['Date']]
    ax.plot_date(
        x=dates,
        y=df[column],
        fmt='-',
        label='Date',
        tz=None,
        xdate=True,
        ydate=False,
        linewidth=0.5
    )

    # establish the date range for the data
    if min_date:
        min_date = min(min_date, min(dates))
    else:
        min_date = min(dates)
    if max_date:
        max_date = max(max_date, max(dates))
    else:
        max_date = max(dates)

    # give a bit of space at each end of the plot - aesthetics
    span = max_date - min_date
    extra = int(span.days * 0.03) * datetime.timedelta(days=1)
    ax.set_xlim([min_date - extra, max_date + extra])

    # format the x tick marks
    ax.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))

    # grid, legend and yLabel
    ax.grid(True)
    ax.legend(loc='best', prop={'size': 'x-small'})

    # x label is always the date
    ax.set_xlabel('Date')
    if ylabel:
        ax.set_ylabel(ylabel)

    # heading
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(pad=2.5)

    # footnote
    footnote = 'charles.fxtrade,com'
    fig.text(
        0.99, 0.01, footnote, ha='right', va='bottom', fontsize=8,
        color='#999999'
    )
    plt.show(block=True)


def plot_data(df, instrument, start_period=None, end_period=None):
    # plot the data for the selected instrument for divergence factor 1

    # dates = [
    #     dateutil.parser.parse(d) for d in df[instrument]['Date'][
    #         start_period:end_period
    #     ]
    # ]
    dates = [
        d for d in df[instrument].index.date[
            start_period:end_period
        ]
    ]
    min_date = None
    max_date = None
    # establish the date range for the data
    if min_date:
        min_date = min(min_date, min(dates))
    else:
        min_date = min(dates)
    if max_date:
        max_date = max(max_date, max(dates))
    else:
        max_date = max(dates)

    # give a bit of space at each end of the plot - aesthetics
    span = max_date - min_date
    extra = int(span.days * 0.03) * datetime.timedelta(days=1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([min_date - extra, max_date + extra])
    # format the x tick marks
    ax1.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax1.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))
    # grid, legend and yLabel
    ax1.grid(True)
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_title(f'{instrument} - Divergence Factor 1/Close')
    ax1.set_ylabel('Divergence factor', color=color)
    ax1.plot_date(
        dates,
        df[instrument]['Divergence Factor 1'][start_period:end_period],
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Close price', color=color)
    ax2.plot_date(
        dates,
        df[instrument]['Close'][start_period:end_period],
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{instrument}-divergence-factor-1-close-price.png')

    # plot the data for the selected instrument for divergence factor 2
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([min_date - extra, max_date + extra])
    # format the x tick marks
    ax1.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax1.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))
    # grid, legend and yLabel
    ax1.grid(True)
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_title(f'{instrument} - Divergence Factor 2/Close')
    ax1.set_ylabel('Divergence factor', color=color)
    ax1.plot_date(
        dates,
        df[instrument]['Divergence Factor 2'][start_period:end_period],
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Close price', color=color)
    ax2.plot_date(
        dates,
        df[instrument]['Close'][start_period:end_period],
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{instrument}-divergence-factor-2-close-price.png')

    # plot the data for the selected instrument for momentum factors
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([min_date - extra, max_date + extra])
    # format the x tick marks
    ax1.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax1.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))
    # grid, legend and yLabel
    ax1.grid(True)
    # color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_title('Momentum factors')
    # ax1.set_ylabel('Divergence factor', color=color)
    ax1.plot_date(
        dates,
        df[instrument]['M_MACD_CHANGE'][start_period:end_period],
        color='tab:red',
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.plot_date(
        dates,
        df[instrument]['M_RSI_CHANGE'][start_period:end_period],
        color='tab:green',
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.plot_date(
        dates,
        df[instrument]['Profit/Loss_10'][start_period:end_period],
        color='tab:blue',
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.plot_date(
        dates,
        df[instrument]['Profit/Loss_30'][start_period:end_period],
        color='tab:orange',
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(f'{instrument}-momentum-factors.png')

    plt.show()



def plot_agg_divergence(
    df, periods, instrument, start_period=None, end_period=None
):
    values = []
    for period in periods[start_period:end_period]:
        values.append(period['DF Avg Rank'].loc[instrument])
    # dates = [
    #     dateutil.parser.parse(d) for d in df[instrument]['Date'][
    #         start_period:end_period
    #     ]
    # ]
    dates = [
        d for d in df[instrument].index.date[
            start_period:end_period
        ]
    ]

    min_date = None
    max_date = None
    # establish the date range for the data
    if min_date:
        min_date = min(min_date, min(dates))
    else:
        min_date = min(dates)
    if max_date:
        max_date = max(max_date, max(dates))
    else:
        max_date = max(dates)

    # give a bit of space at each end of the plot - aesthetics
    span = max_date - min_date
    extra = int(span.days * 0.03) * datetime.timedelta(days=1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([min_date - extra, max_date + extra])
    # format the x tick marks
    ax1.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax1.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))
    # grid, legend and yLabel
    ax1.grid(True)
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_title(f'{instrument} - Divergence average and ranks')
    ax1.set_ylabel('DF Avg Rank', color=color)
    ax1.plot_date(
        dates,
        values,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.tick_params(axis='y', labelcolor=color)

    values = []
    for period in periods[start_period:end_period]:
        values.append(period['% Rank of DF Avgs'].loc[instrument])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('% Rank of DF Avgs', color=color)
    ax2.plot_date(
        dates,
        values,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax2.tick_params(axis='y', labelcolor=color)

    values2 = []
    for period in periods[start_period:end_period]:
        values2.append(period['Price'].loc[instrument])
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.spines['right'].set_position(('axes', 1.2))
    color = 'tab:green'
    ax3.set_ylabel('Price', color=color)
    ax3.plot_date(
        dates,
        values2,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_agg_momentum(
    df, periods, instrument, start_period=None, end_period=None
):

    values = []
    for period in periods[start_period:end_period]:
        values.append(period['Momentum Average Rank'].loc[instrument])
    # dates = [
    #     dateutil.parser.parse(d) for d in df[instrument]['Date'][
    #         start_period:end_period
    #     ]
    # ]
    dates = [
        d for d in df[instrument].index.date[
            start_period:end_period
        ]
    ]
    min_date = None
    max_date = None
    # establish the date range for the data
    if min_date:
        min_date = min(min_date, min(dates))
    else:
        min_date = min(dates)
    if max_date:
        max_date = max(max_date, max(dates))
    else:
        max_date = max(dates)

    # give a bit of space at each end of the plot - aesthetics
    span = max_date - min_date
    extra = int(span.days * 0.03) * datetime.timedelta(days=1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlim([min_date - extra, max_date + extra])
    # format the x tick marks
    ax1.xaxis.set_major_formatter(DateFormatter('%Y/%m/%d'))
    ax1.xaxis.set_minor_formatter(DateFormatter('\n%b'))
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_minor_locator(MonthLocator(bymonthday=1, interval=5))
    # grid, legend and yLabel
    ax1.grid(True)
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_title(f'{instrument} - Momentum average and ranks')
    ax1.set_ylabel('Momentum Average Rank', color=color)
    ax1.plot_date(
        dates,
        values,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax1.tick_params(axis='y', labelcolor=color)

    values = []
    for period in periods[start_period:end_period]:
        values.append(period['Momentum Averages Ranks'].loc[instrument])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Momentum Averages Ranks', color=color)
    ax2.plot_date(
        dates,
        values,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax2.tick_params(axis='y', labelcolor=color)
    values2 = []
    for period in periods[start_period:end_period]:
        values2.append(period['Price'].loc[instrument])
    ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax3.spines['right'].set_position(('axes', 1.2))
    color = 'tab:green'
    ax3.set_ylabel('Price', color=color)
    ax3.plot_date(
        dates,
        values2,
        color=color,
        linewidth=0.5,
        ydate=False,
        fmt='-',
        tz=None,
        xdate=True,
    )
    ax3.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
