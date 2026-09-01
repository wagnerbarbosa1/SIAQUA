import pandas as pd
import numpy as np

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

df_5min = pd.read_csv("/home/wagner-barbosa/Documents/IC/zero_shot_models/src/preprocessing/joinville_processed_5min.csv")
df_1hour = pd.read_csv("/home/wagner-barbosa/Documents/IC/zero_shot_models/src/preprocessing/test_23_06_10-25_06_30-request-open-meteo.csv")
df_1day = pd.read_csv("/home/wagner-barbosa/Documents/IC/zero_shot_models/src/preprocessing/dataset_all_features_engineering.csv")

df_5min["datetime"] = (pd.to_datetime(df_5min["datetime"], utc=True))
df_1hour["date"] = pd.to_datetime(df_1hour["date"], utc=True)
df_1day["timestamp"] = (pd.to_datetime(df_1day["timestamp"], utc=True))

df_5min = df_5min.sort_values("datetime")
df_1hour = df_1hour.sort_values("date")
df_1day = df_1day.sort_values("timestamp")

# Decomposição de senos e cossenos
# Formula: sen(2pi * hora_do_dia / 24 [periodo da variavel])

#----------- Save dataset 5min - features 1 day ----------------------
#df = merge_diff_granularities(water_data=df_5min, features=df_1day)
#df = df.drop(columns="volume")
#df = df.rename(columns={"datetime" : "timestamp", "vazao" : "water_produced"})
#df.to_csv("df_merged_5min_1day_features.csv", index=False)


# --------------- Save dataset 5min - features 1 hour -------------
df_relevant_features = df_1day[["is_weekend", "is_holiday", "is_carnival", "is_school_recess", "month_sin", "month_cos", 
                               "dayofyear_sin","dayofyear_cos","dayofweek_sin","dayofweek_cos","season_sin","season_cos", "timestamp"]]

df = merge_diff_granularities(water_data=df_5min, features=df_relevant_features, time_name_df1= "datetime", time_name_df2="timestamp")

df = df.rename(columns={"vazao" : "water_produced"})
df = df.drop(columns={"timestamp"})

df = merge_diff_granularities(water_data=df, features=df_1hour, time_name_df1= "datetime", time_name_df2="date")
df = df.drop(columns={"date"})

df.to_csv("caj_merged_5min_1hour_features.csv", index=False)
