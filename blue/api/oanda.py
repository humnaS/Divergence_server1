from collections import OrderedDict

import oandapyV20.endpoints.instruments as v20instruments
import oandapyV20.endpoints.pricing as pricing
import pandas as pd
from environs import Env
from oandapyV20 import API

env = Env()

#ACCOUNT_ID = env.str('OANDA_ACCOUNT_ID')
# ACCESS_TOKEN = env.str('OANDA_ACCESS_TOKEN')
# ACCOUNT_ID = '001-004-391907-001'
# ACCESS_TOKEN = 'c6c7915905af083a09f8cdde6e2bfa14-5e59f1f1208440e6c897353fd17e219c'
# ACCOUNT_ID = '101-004-1683826-001'
# ACCESS_TOKEN = 'c7f4c988c1d40271e8de001b490fc4e9-41315f8cc08fc259344e79d633df01a1'

ACCOUNT_ID = '101-004-1683826-005'
ACCESS_TOKEN = 'f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148'


def DataFrameFactory(r, colmap=None, conv=None):
    def convrec(r, m):
        """convrec - convert OANDA candle record.

        return array of values, dynamically constructed, corresponding
        with config in mapping m.
        """
        v = []
        for keys in [x.split(":") for x in m.keys()]:
            _v = r.get(keys[0])
            for k in keys[1:]:
                _v = _v.get(k)
            v.append(_v)
        return v

    record_converter = convrec if conv is None else conv
    column_map_ohlcv = OrderedDict(
        [
            ('time', 'Date'),
            ('mid:o', 'Open'),
            ('mid:h', 'High'),
            ('mid:l', 'Low'),
            ('mid:c', 'Close'),
            ('volume', 'Volume')
        ]
    )
    cmap = column_map_ohlcv if colmap is None else colmap
    df = pd.DataFrame(
        [
            list(
                record_converter(rec, cmap)
            ) for rec in r.get('candles')
        ]
    )
    df.columns = list(cmap.values())
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
    del df['Date']
    df = df.apply(pd.to_numeric)  # OANDA returns string values: make all numeric
    return df


def PriceDataFrameFactory(r, colmap=None, conv=None):
    def convrec(r, m):
        """convrec - convert OANDA candle record.

        return array of values, dynamically constructed, corresponding
        with config in mapping m.
        """
        v = []
        for keys in [x.split(":") for x in m.keys()]:
            _v = r.get(keys[0])
            for k in keys[1:]:
                _v = _v.get(k)
            v.append(_v)
        return v

    record_converter = convrec if conv is None else conv
    column_map_ohlcv = OrderedDict(
        [
            ('time', 'Date'),
            ('closeoutAsk', 'Close'),
        ]
    )
    cmap = column_map_ohlcv if colmap is None else colmap
    df = pd.DataFrame(
        [
            list(
                record_converter(rec, cmap)
            ) for rec in r.get('prices')
        ]
    )
    df.columns = list(cmap.values())
    df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
    del df['Date']
    df = df.apply(pd.to_numeric)  # OANDA returns string values: make all numeric
    return df


def get_price(params):
    api = API(access_token=ACCESS_TOKEN)
    try:
        request = pricing.PricingInfo(accountID=ACCOUNT_ID, params=params)
        response = api.request(request)
    except Exception as err:
        print(f'Error: {err}')
        raise
    return response


def get_candles(instruments, params):
    df = {}
    api = API(access_token=ACCESS_TOKEN)
    for instrument in instruments:
        try:
            request = v20instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
            api.request(request)
        except Exception as err:
            print(f'Error: {err}')
            raise
        else:
            df.update({
                instrument: DataFrameFactory(request.response)
            })
            # if params.get('granularity') not in ['d', 'D', 'w', 'W', 'm', 'M']:
            #     price_infp = get_price(
            #         params={
            #             'instruments': instrument,
            #             'prices': ['B']
            #         }
            #     )
            #     df_price = PriceDataFrameFactory(price_infp)
            #     df[instrument] = pd.concat([df[instrument], df_price], sort=True)
    return df
