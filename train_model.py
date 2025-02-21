import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def train_model(data):
    # target variable : is close(t+1) - close(t)
    data["target"] = data["Close"].diff(-1)
    data['RSI'] = data["RSI"].bfill()
    # features : all the indicators
    features = [col for col in data.columns if col not in [('target', '')]]
    # splitting the data
    X = data[features].to_numpy()
    # drop last row of X and y
    X = X[:-1]
    y = data["target"]
    y = y[:-1]
    # crossvalidation
    tscv = TimeSeriesSplit(n_splits=5, test_size=150)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # scaling the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # training the model
        model = XGBRegressor()
        model.fit(X_train, y_train)
        # print score
        print("R2 score : ", r2_score(y_test, model.predict(X_test)))
        print("MSE : ", mean_squared_error(y_test, model.predict(X_test)))
    final_model = LinearRegression()
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    final_model.fit(X, y)
    return final_model, scaler
    # training model
