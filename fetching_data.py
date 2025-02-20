import yfinance as yf
import pandas as pd


def fetch_commodity_data(ticker, start='2020-01-01', end='2025-02-19'):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True
    )
    return data


def fetch_multiple_commodities(commodity_dict, start='2020-01-01', end='2025-02-19'):
    dict_commos = {}
    for name, ticker in commodity_dict.items():
        dict_commos[name] = fetch_commodity_data(ticker, start, end)
    return dict_commos
