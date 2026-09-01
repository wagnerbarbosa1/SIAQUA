import pandas as pd
import numpy as np
import optuna

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

from mlforecast import MLForecast
from mlforecast.auto import AutoMLForecast, AutoRandomForest, AutoLightGBM, AutoXGBoost

df = pd.read_csv("../../../data/joinville_processed.csv")
df = df.drop(columns="unique_id")

class ExpandingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range(self.trainw, self.n_samples - self.horizon + 1, self.step): #+1 só porque o range exclui o limite superior
            #           k é o conjunto de treinamento
            trainidxs = slice(0, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs

class SlidingWindow:
    def __init__(self, n_samples, trainw, step, horizon):
        self.n_samples = n_samples
        self.trainw = trainw
        self.step = step
        self.horizon = horizon

    def split(self):
        for k in range (self.trainw, self.n_samples - self.horizon + 1, self.step):
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.horizon)

            yield trainidxs, testidxs

expanding_window = ExpandingWindow(
    n_samples=len(df),
    trainw=385,
    step=1,
    horizon=2
)


forecast = MLForecast(
    models=[AutoRandomForest, AutoLightGBM, AutoXGBoost],
    freq="D",
    date_features=["timestamp", "day"],
    num_threads=-1
)

for i,(trainidxs, testidxs) in enumerate(expanding_window.split()):
    df_train = df[trainidxs].drop(columns="water_produced")
    df_test = df[testidxs].drop(columns="water_produced")

    y_train = df.iloc[trainidxs]["water_produced"]
    y_test = df.iloc[trainidxs]["water_produced"]

    forecast.fit(df_train, target_col=y_train)
    pred = forecast.predict(df_test)