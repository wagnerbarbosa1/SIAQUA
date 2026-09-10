import pandas as pd

#Private Functions
def _get_granularity(df : pd.DataFrame, timestamp : str):
    timestamps = pd.to_datetime(df[timestamp], utc=True)
    granularity = timestamps.diff().mode()[0]
    return granularity


#Public Functions
def verify_timestamp(df : pd.DataFrame, timestamp : str):
    df[timestamp] = pd.to_datetime(df[timestamp], utc=True)
    df = df.sort_values(timestamp)

    granularity = _get_granularity(df, timestamp)

    expected = pd.date_range(
        start = df["timestamp"].min(),
        end = df["timestamp"].max(),
        freq=granularity
    )

    missing = expected.difference(df[timestamp])

    return df, missing
