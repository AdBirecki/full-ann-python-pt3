import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/NYCTaxiFares.csv')

print(df.head())
print(df.describe())

print(df['fare_amount'].describe())