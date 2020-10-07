import click
import pandas as pd
from fxtrade import GRANULARITY, INSTRUMENTS, PERIODS, get_oanda_api
from indicators import (adx, macd, macd_percentile, moving_average, pnl, pnl_n,
                        pnl_percentile, rsi, rsi_percentile, williams)


@click.command()
@click.option('--instrument', '-i', help='Currency pair', required=True)
@click.option('--periods', '-p', help='Periods', default=PERIODS, type=int)
@click.option('--granularity', '-g', help='Instrument Granularity', default='D')
@click.option(
    '--daily_alignment',
    '-d',
    help='daily alignment https://developer.oanda.com/rest-live-v20/instrument-ep/',
    default=17,
)
@click.option('--outfile', '-o', help='Output file', default='outfile.csv')
def main(instrument, periods, granularity, outfile, daily_alignment):

    print(f'Periods        : {periods}')
    print(f'Instrument     : {instrument}')
    print(f'Granularity    : {granularity}')
    print(f'daily alignment: {daily_alignment}')
    print(f'Outfile        : {outfile}')

    if instrument.upper() not in INSTRUMENTS:
        raise Exception(f'Invalid instrument {instrument}')

    if granularity.upper() not in GRANULARITY:
        raise Exception(f'Invalid granularity {granularity}')

    instrument = instrument.upper()
    granularity = granularity.upper()

    debug = True
    if not debug:
        pd.set_option('display.max_columns', 50)
        # pd.set_option('display.max_rows', 500000)
        pd.set_option('display.width', 1000)

    stock = get_oanda_api(
        [instrument],
        granularity=granularity,
        count=periods,
        daily_alignment=daily_alignment,
    )

    nsize = len(stock[instrument]['Close'])

    # Calculate MACD
    stock[instrument] = stock[instrument].join(
        macd(stock[instrument]['Close'])
    )

    # Calculate RSI for n = 14
    stock[instrument] = stock[instrument].join(
        rsi(stock[instrument]['Close'])
    )
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

    headers = [
        'Close',
        'adx',
        'pdi',
        'mdi',
        'MACD',
        'RSI',
        'Divergence Factor 1',
        'Divergence Factor 2',
        'Divergence Factor 3',
    ]
    stock[instrument].to_csv(
        outfile,
        columns=headers,
        mode='w',
        sep=',',
        date_format='%d-%b-%Y %r',
    )
    # print(stock[instrument][['Close', 'adx', 'mdi', 'pdi']])


if __name__ == '__main__':
    main()
