import skorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    L = 48
    H = 1
    j = 0

    #X = load_m4_daily_bench()[100]
    X = load_monash('australian_electricity_demand')['series_value']
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        run_experiment(x, L, H, j)


def run_experiment(X, L, H, j, verbose=False):
    print('ts length', X.shape)

    # Split and normalize data
    end_train = int(len(X) * 0.5)
    end_val = end_train + int(len(X) * 0.25)
    X_train = X[:end_train]
    X_val = X[end_train:end_val]
    X_test = X[end_val:]

    mu = np.mean(X_train)
    std = np.std(X_train)

    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std
    X_test = (X_test - mu) / std

    # Instead of forecasting t+1, forecast t+j
    x_train, y_train = windowing(X_train, L=L, H=H+j)
    x_val, y_val = windowing(X_val, L=L, H=H+j)
    x_test, y_test = windowing(X_test, L=L, H=H+j)
    y_train = y_train[..., j:]
    y_val = y_val[..., j:]
    y_test = y_test[..., j:]

    x_train = np.expand_dims(x_train.astype(np.float32), 1)
    x_val = np.expand_dims(x_val.astype(np.float32), 1)
    x_test = np.expand_dims(x_test.astype(np.float32), 1)
    y_train = y_train.reshape(-1, H).astype(np.float32)
    y_val = y_val.reshape(-1, H).astype(np.float32)
    y_test = y_test.reshape(-1, H).astype(np.float32)

    # Train base models
    print('train shape', x_train.shape, y_train.shape)
    print('val shape', x_val.shape, y_val.shape)
    print('test shape', x_test.shape, y_test.shape)

    simple_hp = {
        'random_state': 192937,
        'max_epochs': 2000,
        'device': None,
        'lr': 1e-2,
        'batch_size': 2048,
        'callbacks': [skorch.callbacks.EarlyStopping(load_best=True)],
    }
    f_i = LinearRegression()
    f_i.fit(x_train.squeeze(), y_train)
    # f_i = NeuralNetRegressor(get_simple(L, H), verbose=verbose, **simple_hp)
    # f_i.fit(x_train.squeeze(), y_train)
    preds_i = f_i.predict(x_val.squeeze()).squeeze()
    loss_i = mean_squared_error(preds_i, y_val.squeeze())

    val_ds = Dataset(x_val, y_val)
    complex_hp = {
        'random_state': 192937,
        'max_epochs': 10000,
        #'max_epochs': 2000,
        'device': None,
        'lr': 1e-3,
        'batch_size': 1024,
        'callbacks': [skorch.callbacks.EarlyStopping(load_best=True)],
        'train_split': predefined_split(val_ds)
    }

    f_c = NeuralNetRegressor(get_complex(L, H), verbose=verbose, **complex_hp)
    f_c.fit(x_train, y_train)
    preds_c = f_c.predict(x_val).squeeze()
    loss_c = mean_squared_error(preds_c, y_val.squeeze())

    loss_lv = mean_squared_error(x_val[..., -1:].repeat(H, 2).squeeze(), y_val.squeeze())
    loss_mean = mean_squared_error(x_val.mean(axis=-1).reshape(-1, 1).repeat(H, 1).squeeze(), y_val.squeeze()) 

    print('--- validation ---')
    print('last value', loss_lv)
    print('mean value', loss_mean)
    print('linear', loss_i)
    print('neural net', loss_c)

    preds_i = f_i.predict(x_test.squeeze()).squeeze()
    loss_i = mean_squared_error(preds_i, y_test.squeeze())
    preds_c = f_c.predict(x_test).squeeze()
    loss_c = mean_squared_error(preds_c, y_test.squeeze())
    loss_lv = mean_squared_error(x_test[..., -1:].repeat(H, 2).squeeze(), y_test.squeeze())
    loss_mean = mean_squared_error(x_test.mean(axis=-1).reshape(-1, 1).repeat(H, 1).squeeze(), y_test.squeeze()) 

    print('--- test ---')
    print('last value', loss_lv)
    print('mean value', loss_mean)
    print('linear', loss_i)
    print('neural net', loss_c)


if __name__ == '__main__':
    main()
