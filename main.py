from siaqua.preprocessing import verify_inconsistent
from siaqua.io import csv

TYPE_DATA = "interim"
FILENAME = "brute_5min_1day_features.csv"
TIMESTAMP = "timestamp"

df = csv.read_csv(filename=FILENAME, data_type=TYPE_DATA)
df = verify_inconsistent.verify_timestamp(df, TIMESTAMP)



print(df)


