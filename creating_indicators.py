import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None


def calculate_moving_average(data, window=20):
    # Calculating an asymetric moving average :
    # for the first 19 indicators, we will use the available data and for the rest we will use the window size.
    data['SMA_'+str(window)] = data['Close'].rolling(
        window=window, min_periods=1).mean()
    return data


def calculate_exponential_moving_average(data, window=20):
    # Calculating an asymetric moving average :
    # for the first 19 indicators, we will use the available data and for the rest we will use the window size.
    data['EMA_'+str(window)] = data['Close'].ewm(span=window,
                                                 adjust=False).mean()
    return data


def local_smoothing_moving_average(data, window=20):
    # calculating a local linear smoothing moving average with a polyfit of degree 1
    data["lSMA_"+str(window)] = data["Close"].rolling(window=window, min_periods=2).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x)-1) + np.polyfit(np.arange(len(x)), x, 1)[1])
    # Using replace method to replace nan values with the first value of the close price
    data["lSMA_"+str(window)] = np.where(data["lSMA_"+str(window)
                                              ].isnull(), data["Close"], data["lSMA_"+str(window)])
    return data


def MACD(data):
    # calculating the MACD
    data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean(
    ) - data["Close"].ewm(span=26, adjust=False).mean()
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["Histogram"] = data["MACD"] - \
        data["Signal_Line"]  # MACD - Signal Line
    return data


def RSI(data):
    # calculating the RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
    RS = gain / loss
    data["RSI"] = 100 - (100 / (1 + RS))
    # adding some thresholds
    data["Overbought"] = 70
    data["Oversold"] = 30
    return data


def enrich_signal(data):
    parameters = {"EMA": [9, 21, 55], "SMA": [
        5, 50, 200], "LSMA": [5, 50, 200]}
    for indicator, values in parameters.items():
        for value in values:
            if indicator == "EMA":
                data = calculate_exponential_moving_average(data, value)
            elif indicator == "SMA":
                data = calculate_moving_average(data, value)
            elif indicator == "LSMA":
                data = local_smoothing_moving_average(data, value)
    data["TR"] = abs(data["High"] - data["Low"])
    data["ATR"] = data["TR"].ewm(span=14, adjust=False).mean()
    data['Log_Volume'] = np.log1p(data['Volume'])
    data = MACD(data)
    data = RSI(data)
    return data
# test enrich singla function
