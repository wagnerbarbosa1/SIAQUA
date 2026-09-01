"""
Pipeline de validacao expanding window para previsao de producao de agua
utilizando o modelo moirai2 2.5 (200M, PyTorch) com covariaveis dinamicas.
"""

import os
import datetime

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
 
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


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

def load_data():
    # parse_dates=True só afeta o índice do DataFrame, não a coluna "timestamp" -- sem isso
    # ela ficava como string/object em vez de datetime, o que pode confundir o PandasDataset.
    df = pd.read_csv("/home/wagner-barbosa/Documents/IC/zero_shot_models/data/joinville_processed.csv", parse_dates=["timestamp"])
    if df.isna().any().any():
        raise ValueError("O DataFrame contém valores nulos.")
    return df


df = load_data()

covariate_cols = [
    #"is_holiday",
    #"precipitation_median",
  #  "temperature_2m_previsao_mean",
   # "shortwave_radiation_previsao_std",
  #  "volume_trend_7d",
  #  "dayofweek_sin",
  #  "dayofweek_cos",
   # "season_sin",
   # "season_cos",
]

df["water_produced"] = df["water_produced"].astype("float32")
df[covariate_cols] = df[covariate_cols].astype("float32")

dynamic_real_cols = [
    "is_holiday",
    #"temperature_2m_previsao_mean",
    #"shortwave_radiation_previsao_std",
    "dayofweek_sin",
    "dayofweek_cos",
    "season_sin",
    "season_cos",
]

past_dynamic_real_cols = ["precipitation_median", "volume_trend_7d"] 

feat_cols = dynamic_real_cols + past_dynamic_real_cols
print(df[feat_cols].dtypes)
df[feat_cols] = df[feat_cols].astype("float32")

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw= 385, #o tamanho da janela inicial
    step=1, #passo
    horizon=2 #horizonte de previsao
)

#torch.set_float32_matmul_precision("high") #melhora o desempenho de multiplicação de matriz em troca de perda de precisão

print("Loading Moirai2")
moirai_module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

resultsEW_val = dict(timestamp=[],  ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split()):

    full_idxs = slice(trainidxs.start, testidxs.stop)
    context_df = df[full_idxs]

    ds = PandasDataset(
        context_df,
        target="water_produced",
        timestamp="timestamp",
        freq="D",
        past_feat_dynamic_real= past_dynamic_real_cols,
        feat_dynamic_real= dynamic_real_cols,
    )

#Não é possível modificar o tamanho dos patchs do moirai2
    model = Moirai2Forecast(
            module=moirai_module,
            prediction_length= expanding_window_test.horizon,
            context_length=385,
            target_dim=1,
            feat_dynamic_real_dim= ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim= ds.num_past_feat_dynamic_real,
    )

    _, test_template = split(ds, offset=-expanding_window_test.horizon) #test_template é um template que guarda o historico passado e o horizonte futuro
    #com uma "linha" de recorte do offset.

    test_data = test_template.generate_instances( #gera um tensor dos dados passados e futuros conforme o template do test_template
    prediction_length=expanding_window_test.horizon,
    windows=1,
    )

    predictor = model.create_predictor(batch_size=32)
    forecasts = list(predictor.predict(test_data.input))
    forecast = forecasts[0]
    median = forecast.quantile(0.5)

#Organização das previsões e timestamps
    y_t = df[testidxs]

    resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[0])

    resultsEW_val["ytrue0"].append(y_t["water_produced"].iloc[0]) 
    resultsEW_val["ytrue1"].append(y_t["water_produced"].iloc[1])
    resultsEW_val["yhat0"].append(median[0])
    resultsEW_val["yhat1"].append(median[1])

    print(f"Iteração Window: {i}")


df_result = pd.DataFrame(resultsEW_val)

print("\nTotal de linhas em df_result:", len(df_result))
print("NaNs por coluna:\n", df_result[["ytrue0", "yhat0", "ytrue1", "yhat1"]].isna().sum())
 
n_nan = df_result[["yhat0", "yhat1"]].isna().sum().sum()
if n_nan > 0:
    raise ValueError(f"AVISO: {n_nan} previsões vieram com NaN")
 
# Métricas Horizonte 0
rmse0 = round(np.sqrt(mean_squared_error(df_result["ytrue0"], df_result["yhat0"])), 3)
mae0 = round(mean_absolute_error(df_result["ytrue0"], df_result["yhat0"]), 3)
mape0 = round(mean_absolute_percentage_error(df_result["ytrue0"], df_result["yhat0"]), 3)
r20 = round(r2_score(df_result["ytrue0"], df_result["yhat0"]), 3)
 
# Métricas Horizonte 1
rmse1 = round(np.sqrt(mean_squared_error(df_result["ytrue1"], df_result["yhat1"])), 3)
mae1 = round(mean_absolute_error(df_result["ytrue1"], df_result["yhat1"]), 3)
mape1 = round(mean_absolute_percentage_error(df_result["ytrue1"], df_result["yhat1"]), 3)
r21 = round(r2_score(df_result["ytrue1"], df_result["yhat1"]), 3)
 
print("RMSE 0 : ", rmse0)
print("R2 0 : ", r20)
print("MAE 0 : ", mae0)
print("MAPE 0 : ", mape0)
 
print("\nRMSE 1 : ", rmse1)
print("R2 1 : ", r21)
print("MAE 1 : ", mae1)
print("MAPE 1 : ", mape1)
 
fig, ax = plt.subplots(figsize=(12, 6))
plt.style.use('default')
 
ax.plot(df_result["timestamp"], df_result["ytrue0"], color='#d3d3d3', alpha=0.8, linewidth=1.0, label='Water_produced')
 
ax.plot(df_result["timestamp"], df_result["yhat0"], color="#2554b9", linewidth=1.3, alpha=0.6, label='Moirai')
 
plt.title('Moirai x Water_produced',
          fontsize=16, loc='left', pad=20, color='#333333')
 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
 
ax.tick_params(axis='both', colors='#666666')
 
plt.legend(frameon=False, loc='upper left', fontsize=9)
 
plt.grid(axis='y', linestyle='--', alpha=0.3)
 
plt.tight_layout()
 
output_models = "output_models"
if not os.path.exists(output_models):
    os.makedirs(output_models)

output_moirai2 = "output_moirai2"
output_models_moirai2 = os.path.join(output_models, output_moirai2)
if not os.path.exists(output_models_moirai2):
    os.makedirs(output_models_moirai2)

output_moirai2_graph = "moirai2_graph"
output_models_moirai2_graph = os.path.join(output_models_moirai2, output_moirai2_graph)

if not os.path.exists(output_models_moirai2_graph):
    os.makedirs(output_models_moirai2_graph)

current_time = datetime.datetime.now()

timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

graphs_file = os.path.join(output_models_moirai2_graph, f"graph_moirai2_{timestamp}")

plt.savefig(graphs_file)


output_metrics = "moirai2_metrics_directory"
output_models_moirai2_metrics = os.path.join(output_models_moirai2, output_metrics)
if not os.path.exists(output_models_moirai2_metrics):
    os.makedirs(output_models_moirai2_metrics)

metrics_file = os.path.join(output_models_moirai2_metrics, f"metrics_moirai2_{timestamp}.csv")

with open(metrics_file, "w") as f: #with garante que eu abra e feche o arquivo
    f.write("model,rmse0,mae0,mape0,r20,rmse1,mae1,mape1,r21,split\n")

# Salva as metricas
with open(metrics_file, "a") as f:
    f.write(f"moirai2-EW,{rmse0},{mae0},{mape0},{r20},{rmse1},{mae1},{mape1},{r21},test\nEW-Parameters, step: {expanding_window_test.step}, " \
            f"horizon: {expanding_window_test.horizon}, trainw_start: {expanding_window_test.trainw}, n_samples: {expanding_window_test.n_samples}")

# CSV com ytrue e yhat
output_dir2 = "moirai2_forecast"
output_models_moirai2_forecast = os.path.join(output_models_moirai2, output_dir2)

if not os.path.exists(output_models_moirai2_forecast):
    os.makedirs(output_models_moirai2_forecast)

values_file = os.path.join(output_models_moirai2_forecast, f"results_moirai2_{timestamp}.csv")

df_result.to_csv(values_file, index=False)