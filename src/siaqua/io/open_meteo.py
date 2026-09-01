import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

#Coordenadas de joinville
latitude = -26.30
longitude = -48.84

cache_session = requests_cache.CachedSession('.cache', expire_after= 3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session= retry_session)

weather_features = ["temperature_2m", "precipitation", "relative_humidity_2m", "dew_point_2m", "cloud_cover", "shortwave_radiation"]

url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude" : latitude,
    "longitude" : longitude,
    "hourly": weather_features,
    "start_date" : "2016-01-01",
    "end_date" : "2019-12-31"
}

responses = openmeteo.weather_api(url, params=params)

hourly = responses[0].Hourly()

hourly_data = {
    "date" : pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
    )
}

for i, k in enumerate(weather_features):
    hourly_data[k] = hourly.Variables(i).ValuesAsNumpy()
    

hourly_dataframe = pd.DataFrame(data = hourly_data)

hourly_dataframe.to_csv("zero_shot_models/src/preprocessing/16_01-19_12_request_open_meteo.csv", index=False)