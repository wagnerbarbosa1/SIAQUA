import pandas as pd
import numpy as np
from matplotlib import pyplot
from siaqua.visualization.plots import linear_graph
from siaqua.io.csv import read_csv, save_dataframe

df = pd.read_csv("/home/wagner-barbosa/Documents/IC/output_models/output_chronos/chronos_forecast/results_chronos_2026-09-15-23-21-03.csv")

linear_graph(df, "timestamp", "yhat0", compare_cols=["ytrue0"])
