from pandas._libs.tslibs import Timestamp
from infrastructure import haversine_distance
from pandas.core.frame import DataFrame
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/NYCTaxiFares.csv')

print(df.head())
print(df['fare_amount'])

print(df.columns)
df['dist_km'] = haversine_distance(
    df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

print(df.columns)
print(df.head())

obj = df['pickup_datetime'][0]
print(type(obj))
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime('%a')

df.head()
df.describe()
df.info()

df.to_csv('Results/ProcessedNYCTaxiFares.csv') 