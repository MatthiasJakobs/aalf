import skorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tsx.datasets.monash import load_m4_daily_bench, load_monash
from tsx.datasets import windowing
from tsx.models.forecaster import NLinear
from tsx.models.forecaster.model_zoo import get_1d_cnn
from tsx.models import NeuralNetRegressor
from seedpy import fixedseed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def get_simple(L, H):
    with fixedseed(torch, 921172):
        #model = NLinear(L, H, 1)
        model = nn.Linear(L, H)
    return model

def get_complex(L, H, n_filters=32):
    with fixedseed(torch, 493817):
        model = nn.Sequential(

            nn.Conv1d(1, n_filters, 3, padding='same'),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),

            nn.Conv1d(n_filters, 2*n_filters, 3, padding='same'),
            nn.BatchNorm1d(2*n_filters),
            nn.ReLU(),

            nn.Conv1d(2 * n_filters, 3*n_filters, 3, padding='same'),
            nn.BatchNorm1d(3*n_filters),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(L * 3 * n_filters, H)

        )
    return model

def main():
    ### Australian electricity demands
    L = 10
    j = 0

    X, horizons = load_monash('australian_electricity_demand', return_horizon=True)
    X = X['series_value']
    H = horizons[0]
    log = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log = run_experiment(log, x, L, H, j, verbose=True)

    log = pd.DataFrame(log)
    log.index.rename('dataset_names', inplace=True)
    print(log)
    log.to_csv('results/australian_electricity_demand.csv')


    ### M4 Subset
    L = 10
    H = 1
    j = 0

    X, horizons = load_m4_daily_bench(return_horizon=True)
    log = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log = run_experiment(log, x, L, horizons[i], j)

    log = pd.DataFrame(log)
    log.index.rename('dataset_names', inplace=True)
    log.to_csv('results/m4.csv')

    ### KDD Cup
    L = 10
    H = 1
    j = 0

    X, horizons = load_monash('kdd_cup_nomissing', return_horizon=True)
    X = X['series_value']
    log = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log = run_experiment(log, x, L, horizons[i], j, lr=2e-4, verbose=True)

    log = pd.DataFrame(log)
    log.index.rename('dataset_names', inplace=True)
    log.to_csv('results/kdd.csv')

def run_experiment(log, X, L, H, j, lr=1e-3, verbose=False):
    print('ts length', X.shape)

    # Split and normalize data
    end_train = int(len(X) * 0.5)
    end_val = end_train + int(len(X) * 0.25)
    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    mu = np.mean(X_train)
    std = np.std(X_train)

    X = (X - mu) / std

    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    # Instead of forecasting t+1, forecast t+j
    x_train, y_train = windowing(X_train, L=L, H=H)
    x_val, y_val = windowing(X_val, L=L, H=H)
    x_test, y_test = windowing(X_test, L=L, H=H)
    y_train = y_train[..., -1:]
    y_val = y_val[..., -1:]
    y_test = y_test[..., -1:]
    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)
    print(x_test.shape, y_test.shape)

    x_train = np.expand_dims(x_train.astype(np.float32), 1)
    x_val = np.expand_dims(x_val.astype(np.float32), 1)
    x_test = np.expand_dims(x_test.astype(np.float32), 1)
    y_train = y_train.reshape(-1, 1).astype(np.float32)
    y_val = y_val.reshape(-1, 1).astype(np.float32)
    y_test = y_test.reshape(-1, 1).astype(np.float32)

    # Train base models
    val_ds = Dataset(x_val, y_val)

    f_i = LinearRegression()
    f_i.fit(x_train.squeeze(), y_train)
    preds_i = f_i.predict(x_test.squeeze()).squeeze()
    loss_i = mean_squared_error(preds_i, y_test.squeeze())


    # Random Forests
    f_c = RandomForestRegressor(n_estimators=16, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_16 = mean_squared_error(preds_c, y_test.squeeze())

    f_c = RandomForestRegressor(n_estimators=32, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_32 = mean_squared_error(preds_c, y_test.squeeze())

    f_c = RandomForestRegressor(n_estimators=64, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_64 = mean_squared_error(preds_c, y_test.squeeze())

    f_c = RandomForestRegressor(n_estimators=128, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_128 = mean_squared_error(preds_c, y_test.squeeze())

    '''

    f_c = RandomForestRegressor(n_estimators=256, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_256 = mean_squared_error(preds_c, y_test.squeeze())

    f_c = RandomForestRegressor(n_estimators=512, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_512 = mean_squared_error(preds_c, y_test.squeeze())

    f_c = RandomForestRegressor(n_estimators=1024, random_state=129281, n_jobs=-1)
    f_c.fit(x_train.squeeze(), y_train.squeeze())
    preds_c = f_c.predict(x_test.squeeze()).squeeze()
    loss_c_1024 = mean_squared_error(preds_c, y_test.squeeze())
    '''

    loss_lv = mean_squared_error(x_test[..., -1:].squeeze(), y_test.squeeze())
    loss_mean = mean_squared_error(x_test.mean(axis=-1).reshape(-1, 1).squeeze(), y_test.squeeze()) 

    log.append({'lv_test': loss_lv, 'mean_test': loss_mean, 'linear_test': loss_i, 'complex_test_16': loss_c_16, 'complex_test_32': loss_c_32, 'complex_test_64': loss_c_64, 'complex_test_128': loss_c_128})

    return log


if __name__ == '__main__':
    main()
