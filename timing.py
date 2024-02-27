"""
Timing pycox : CPU versus GPU
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchtuples as tt
from pycox import datasets
from pycox.evaluation import EvalSurv
from pycox.models import LogisticHazard
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

device_name = "cuda"

if device_name == "cuda":
    assert torch.cuda.is_available()
    print("GPU is available and will be used")
elif device_name == "cpu":
    print("CPU will be used")
else:
    raise ValueError(f'Device name "{device_name}"not recognized.')

# device = torch.device("cuda")
device = torch.device(device_name)

# Parameters
batch_size = 256
epochs = 100
num_durations = 10
num_nodes = [32, 32]
batch_norm = True
dropout = 0.1
callbacks = [tt.cb.EarlyStopping()]

# We also set some seeds to make this reproducable.
# Note that on gpu, there is still some randomness. (Paul) Why ?
np.random.seed(1234)
_ = torch.manual_seed(123)

# Dataset preprocessing
df = datasets.metabric.read_df()
df_val = df.sample(frac=0.2)
df_train = df.drop(df_val.index)

cols_standardize = ["x0", "x1", "x2", "x3", "x8"]
cols_leave = ["x4", "x5", "x6", "x7"]
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]
x_mapper = DataFrameMapper(standardize + leave)
x_train = x_mapper.fit_transform(df_train).astype("float32")
x_val = x_mapper.transform(df_val).astype("float32")

labtrans = LogisticHazard.label_transform(num_durations)
get_target = lambda df: (df["duration"].values, df["event"].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)
in_features = x_train.shape[1]
out_features = labtrans.out_features
net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
model = LogisticHazard(
    net, tt.optim.Adam(0.01), duration_index=labtrans.cuts, device=device
)

# Training
t0 = time.time()
log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
t1 = time.time()
print(f"Training took {t1 - t0}")

# Prediction
surv = model.predict_surv_df(x_train)
surv = model.predict_surv_df(x_val)
