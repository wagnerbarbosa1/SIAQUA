import pandas as pd
import numpy as np

feature_blocks = {}

def add_cyclical_block(df, col_name, max_val, blocks_dict):
    if col_name in df.columns:
        df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / max_val)
        df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / max_val)
        blocks_dict[col_name] = [f'{col_name}_sin', f'{col_name}_cos']
    return df

def column_repeat(y : pd.DataFrame, x : pd.DataFrame):
    repeated_columns = y.columns.intersection(x.columns)
    return x.drop(columns=repeated_columns)

def merge_diff_granularities(water_data : pd.DataFrame, features : pd.DataFrame, time_name_df1 : str, time_name_df2 : str):
    df = pd.merge_asof(
        water_data,
        features,
        left_on=time_name_df1, #coluna timestamp do dataset
        right_on=time_name_df2,
        direction="backward" #atribui os dados do left data a cada valor do right data correspondente ou menor
    )
    return df

#Todos os datasets começam em 2023-10

df_5min = pd.read_csv("/home/wagner-barbosa/Documents/IC/siaqua/data/interim/Nelson_Brandao_tratado_5min_com_dias.csv")
#df_1hour = pd.read_csv("/home/wagner-barbosa/Documents/IC/siaqua/data/raw/23_06_10-25_06_30-request-open-meteo.csv")
df_1day = pd.read_csv("/home/wagner-barbosa/Documents/IC/siaqua/data/interim/dataset_all_features_engineering.csv")

df_5min["datetime"] = (pd.to_datetime(df_5min["datetime"], utc=True))
#df_1hour["date"] = pd.to_datetime(df_1hour["date"], utc=True)
df_1day["timestamp"] = (pd.to_datetime(df_1day["timestamp"], utc=True))

df_5min = df_5min.sort_values("datetime")
#df_1hour = df_1hour.sort_values("date")
df_1day = df_1day.sort_values("timestamp")

df_5min["hour"] = (df_5min["datetime"].dt.hour)
df_5min["day_5min"] = (df_5min["datetime"].dt.hour * 12 + df_5min["datetime"].dt.minute // 5)

df_5min = add_cyclical_block(df_5min, 'hour', 24, feature_blocks)
df_5min = add_cyclical_block(df_5min, 'day_5min', 288, feature_blocks)

#print(df_5min.head())

# Decomposição de senos e cossenos
# Formula: sen(2pi * hora_do_dia / 24 [periodo da variavel])

#----------- Save dataset 5min - features 1 day ----------------------
df = merge_diff_granularities(water_data=df_5min, features=df_1day, time_name_df1="datetime", time_name_df2="timestamp")
df = df.drop(columns={"water_produced", "timestamp", "volume_lag_1", "volume_lag_7", "volume_trend_7d"})
df = df.rename(columns={"datetime" : "timestamp", "vazao" : "water_produced"})
df.to_csv("interim_caj_5m1d.csv", index=False)

"""
# --------------- Save dataset 5min - features 1 hour -------------
df_relevant_features = df_1day[["is_weekend", "is_holiday", "is_carnival", "is_school_recess", "month_sin", "month_cos", 
                               "dayofyear_sin","dayofyear_cos","dayofweek_sin","dayofweek_cos","season_sin","season_cos", "timestamp"]]

df = merge_diff_granularities(water_data=df_5min, features=df_relevant_features, time_name_df1= "datetime", time_name_df2="timestamp")

df = df.rename(columns={"vazao" : "water_produced"})
#df = df.drop(columns={"timestamp"})

df = merge_diff_granularities(water_data=df, features=df_1hour, time_name_df1= "datetime", time_name_df2="date")
#df = df.drop(columns={"date", "hour", "day_5min"})

df.to_csv("caj_merged_5min_1hour_features.csv", index=False)
"""