from TabularModel import TabularModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('Results/ProcessedNYCTaxiFares.csv')
isCuda = torch.cuda.is_available()
print(f'CUDA available: {isCuda}')

cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
             'dropoff_latitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount']

for cat in cat_cols:
    df[cat] = df[cat].astype('category')

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], axis=1)
cats = torch.tensor(cats, dtype=torch.int64).cuda()

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float).cuda()

y = torch.tensor(df[y_col].values, dtype=torch.float).cuda()

cat_szs = [len(df[cat].cat.categories) for cat in cat_cols]
emb_szs = [(size, min(50, (size + 1)//2)) for size in cat_szs]

torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200, 100], p=0.4)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000
test_size = int(batch_size*0.2)

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]

y_train = y[:batch_size-test_size].cuda()
y_test = y[batch_size-test_size:batch_size].cuda()

start_time = time.time()
epochs = 300

losses = []

for i in range(300):
    i += 1

    y_pred = model(cat_train, con_train).cuda()
    loss = torch.sqrt(criterion(y_pred, y_train)).cuda()
    losses.append(loss)

    if i % 10 == 1:
        print(f'epoch: {i} loss is: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


duration = time.time() - start_time
print(f'training took {duration/60} minutes')
