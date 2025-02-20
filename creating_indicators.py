import pandas as pd
import numpy as np


def calculate_moving_average(data, window=20):
    # Calculating an asymetric moving average :
    # for the first 19 indicators, we will use the available data and for the rest we will use the window size.
    data['SMA'] = data['Close'].rolling(window=window, min_periods=1).mean()
    return data


def calculate_exponential_moving_average(data, window=20):
    # Calculating an asymetric moving average :
    # for the first 19 indicators, we will use the available data and for the rest we will use the window size.
    data['EMA'] = data['Close'].ewm(span=window, adjust=False).mean()
    return data
