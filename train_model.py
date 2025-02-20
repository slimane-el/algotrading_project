import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def train_model(data):
    # target variable : is close(t+1) - close(t)
    data["target"] = data["Close"].diff(-1)
    # features : all the indicators
    features = [col for col in data.columns if col not in ["target", "Date"]]
    # splitting the data
    X = data[features]
    y = data["target"]
    tscv = TimeSeriesSplit(n_splits=5, test_size=30, gap=7)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # scaling the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # training the model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        # evaluating the model
        print(model.score(X_test, y_test))
