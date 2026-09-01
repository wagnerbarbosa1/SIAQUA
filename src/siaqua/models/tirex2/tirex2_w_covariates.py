import pandas as pd
from tirex2 import TimeseriesType, load_model
import numpy as np
import torch
import os
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt



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

df = pd.read_csv("/home/wagner-barbosa/Documents/TCC/zero_shot_models/data/processed.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Configuração das colunas ---
target_col = "water_produced"

# colunas conhecidas apenas no passado (mesma lógica do script original,
# que retirava "volume_trend_7d" do future_df usado pelo tirex2-2)

past_only_cols = ["volume_trend_7d"]
# todas as demais colunas de covariáveis (exceto timestamp/unique_id/target/past_only)
# são tratadas como covariáveis conhecidas no futuro (future-known)

non_covariate_cols = ["timestamp", "unique_id", target_col] + past_only_cols
future_cols = [c for c in df.columns if c not in non_covariate_cols]

print("Target:", target_col)
print("Past-only covariates:", past_only_cols)
print("Future-known covariates:", future_cols)

pipeline = load_model(
    "NX-AI/TiRex-2",
    device="cuda",
)

# níveis de quantis nativos do modelo e índice da mediana (0.5)
quantile_levels = [round(float(q), 6) for q in pipeline.quantiles]
if 0.5 in quantile_levels:
    median_idx = quantile_levels.index(0.5)
else:
    median_idx = int(np.argmin(np.abs(np.array(quantile_levels) - 0.5)))

expanding_window_test = ExpandingWindow(
    n_samples=len(df),
    trainw= 385, #o tamanho da janela inicial
    step=1, #passo
    horizon=2 #horizonte de previsao
)

resultsEW_val = dict(timestamp=[], ytrue0=[], ytrue1=[], yhat0=[], yhat1=[])
print("Start Expanding Window")

for i, (trainidxs, testidxs) in enumerate(expanding_window_test.split()):
    # Dados de treino (contexto)
    context_df = df[trainidxs]

    # Dados de validação
    y_t = df[testidxs]

    # janela completa (contexto + horizonte), necessária para as future_covariates,
    # que precisam cobrir T + H passos
    full_df = df.iloc[trainidxs.start:testidxs.stop]

    # target: shape [V_t, T]
    target_tensor = torch.tensor(
        context_df[target_col].to_numpy(dtype=np.float32)
    ).unsqueeze(0)

    # past_covariates: shape [V_p, T] (conhecidas só até o presente)
    if past_only_cols:
        past_cov_tensor = torch.tensor(
            context_df[past_only_cols].to_numpy(dtype=np.float32).T
        )
    else:
        past_cov_tensor = None

    # future_covariates: shape [V_f, >=T+H] (conhecidas com antecedência)
    if future_cols:
        future_cov_tensor = torch.tensor(
            full_df[future_cols].to_numpy(dtype=np.float32).T
        )
    else:
        future_cov_tensor = None

    ts = TimeseriesType(
        target=target_tensor,
        past_covariates=past_cov_tensor,
        future_covariates=future_cov_tensor,
    )

    forecast = pipeline.forecast(
        [ts],
        prediction_length=expanding_window_test.horizon,
        output_type="numpy",
    )[0]  # shape [V_t, Q, H]

    median = forecast[0, median_idx, :]  # shape [H]

    ytrue = y_t[target_col]

    resultsEW_val["ytrue0"].append(ytrue.iloc[0]) #append adiciona os valores em blocos (tanto faz se é string, macaco, banana)
    resultsEW_val["ytrue1"].append(ytrue.iloc[1])
    resultsEW_val["yhat0"].append(median[0])
    resultsEW_val["yhat1"].append(median[1])
    resultsEW_val["timestamp"].append(y_t["timestamp"].iloc[0])

    print(i)


df_result = pd.DataFrame(resultsEW_val)
print("NaNs por coluna:\n", df_result[["ytrue0", "yhat0", "ytrue1", "yhat1"]].isna().sum())
df_result.to_csv("teste.csv", index=False)

#df_result["timestamp"] = df["timestamp"].iloc[expanding_window_test.trainw:] #aplicar logs para verificação de dimensionalidade do dataset
#df_result["timestamp"] = pd.to_datetime(df_result["timestamp"])

rmse0 = round(np.sqrt(mean_squared_error(df_result["ytrue0"], df_result["yhat0"])), 4)
mae0 = round(mean_absolute_error(df_result["ytrue0"], df_result["yhat0"]), 4)
mape0 = round(mean_absolute_percentage_error(df_result["ytrue0"], df_result["yhat0"]), 4)
r20 = round(r2_score(df_result["ytrue0"], df_result["yhat0"]), 4)

rmse1 = round(np.sqrt(mean_squared_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna())), 4)
mae1 = round(mean_absolute_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)
mape1 = round(mean_absolute_percentage_error(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)
r21 = round(r2_score(df_result["ytrue1"].dropna(), df_result["yhat1"].dropna()), 4)

print("RMSE : ", rmse0)
print("R2 : ", r20)
print("MAE : ", mae0)
print("MAPE : ", mape0)

print("\nRMSE : ", rmse1)
print("R2 : ", r21)
print("MAE : ", mae1)
print("MAPE : ", mape1)

fig, ax = plt.subplots(figsize=(12, 6))
plt.style.use('default')

ax.plot(df_result["timestamp"], df_result["ytrue0"], color='#d3d3d3', alpha = 0.8, linewidth=1.0, label='Water_produced')

ax.plot(df_result["timestamp"], df_result["yhat0"], color="#2554b9", linewidth=1.3, alpha=0.6, label='TiRex22')

plt.title('TiRex22 x Water_produced', 
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

output_tirex2 = "output_tirex2"
output_models_tirex2 = os.path.join(output_models, output_tirex2)
if not os.path.exists(output_models_tirex2):
    os.makedirs(output_models_tirex2)

output_tirex2_graph = "tirex2_graph"
output_models_tirex2_graph = os.path.join(output_models_tirex2, output_tirex2_graph)

if not os.path.exists(output_models_tirex2_graph):
    os.makedirs(output_models_tirex2_graph)

current_time = datetime.datetime.now()

timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")

graphs_file = os.path.join(output_models_tirex2_graph, f"graph_tirex2_{timestamp}")

plt.savefig(graphs_file)


output_metrics = "tirex2_metrics_directory"
output_models_tirex2_metrics = os.path.join(output_models_tirex2, output_metrics)
if not os.path.exists(output_models_tirex2_metrics):
    os.makedirs(output_models_tirex2_metrics)

metrics_file = os.path.join(output_models_tirex2_metrics, f"metrics_tirex2_{timestamp}.csv")

with open(metrics_file, "w") as f: #with garante que eu abra e feche o arquivo
    f.write("model,rmse0,mae0,mape0,r20,rmse1,mae1,mape1,r21,split\n")

# Salva as metricas
with open(metrics_file, "a") as f:
    f.write(f"tirex2-EW,{rmse0},{mae0},{mape0},{r20},{rmse1},{mae1},{mape1},{r21},test\nEW-Parameters, step: {expanding_window_test.step}, " \
            f"horizon: {expanding_window_test.horizon}, trainw_start: {expanding_window_test.trainw}, n_samples: {expanding_window_test.n_samples}")

# CSV com ytrue e yhat
output_dir2 = "tirex2_forecast"
output_models_tirex2_forecast = os.path.join(output_models_tirex2, output_dir2)

if not os.path.exists(output_models_tirex2_forecast):
    os.makedirs(output_models_tirex2_forecast)

values_file = os.path.join(output_models_tirex2_forecast, f"results_tirex2_{timestamp}.csv")

df_result.to_csv(values_file, index=False)