import skorch
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
from sklearn.model_selection import RandomizedSearchCV
from seedpy import fixedseed
from os import makedirs

from cdd_plots import create_cdd

class Ensemble:

    def __init__(self, base_estimator, N, *args, **kwargs):
        self.estimators = [base_estimator(*args, **kwargs) for _ in range(N)]

    def fit(self, *args, **kwargs):
        for estimator in self.estimators:
            estimator.fit(*args, **kwargs)

    def predict(self, X):
        preds = []
        for estimator in self.estimators:
            preds.append(estimator.predict(X).reshape(-1, 1))
        
        return np.median(np.concatenate(preds, axis=-1), axis=-1).squeeze()

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

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
    L = 10

    '''
    ### Australian electricity demands
    X, horizons = load_monash('australian_electricity_demand', return_horizon=True)
    X = X['series_value']
    H = horizons[0]
    log_val = []
    log_test = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'australian_electricity_demand', i, x, L, H, verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/australian_electricity_demand_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/australian_electricity_demand_test.csv')

    create_cdd('australian_electricity_demand')

    ### KDD Cup
    X, horizons = load_monash('kdd_cup_nomissing', return_horizon=True)
    X = X['series_value']
    log_val = []
    log_test = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'kdd', i, x, L, horizons[i], verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/kdd_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/kdd_test.csv')

    create_cdd('kdd')
    '''

    ### weather
    X, horizons = load_monash('weather', return_horizon=True)
    X = X['series_value']
    log_val = []
    log_test = []
    for i, x in enumerate(X):
        print('-'*30, i, '-'*30)
        log_val, log_test = run_experiment(log_val, log_test, 'weather', i, x, L, horizons[i], verbose=True)

    log_val = pd.DataFrame(log_val)
    log_val.index.rename('dataset_names', inplace=True)
    log_val.to_csv('results/weather_val.csv')
    log_test = pd.DataFrame(log_test)
    log_test.index.rename('dataset_names', inplace=True)
    log_test.to_csv('results/weather_test.csv')

    create_cdd('weather')

def run_experiment(log_val, log_test, ds_name, ds_index, X, L, H, lr=1e-3, verbose=False):
    print(H, 'ts length', X.shape)
    makedirs(f'models/{ds_name}/{ds_index}', exist_ok=True)

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

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Train base models
    try:
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'rb') as f:
            f_i = pickle.load(f)
    except Exception:
        f_i = LinearRegression()
        f_i.fit(x_train, y_train)
        with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'wb') as f:
            pickle.dump(f_i, f)

    loss_i_val = rmse(f_i.predict(x_val), y_val)
    loss_i_test = rmse(f_i.predict(x_test), y_test)

    # Random Forests
    try:
        with open(f'models/{ds_name}/{ds_index}/rf.pickle', 'rb') as f:
            f_c = pickle.load(f)
    except Exception:
        f_c = RandomForestRegressor(n_estimators=128, max_depth=8, random_state=12345, n_jobs=-1)
        f_c.fit(x_train, y_train.squeeze())
        with open(f'models/{ds_name}/{ds_index}/rf.pickle', 'wb') as f:
            pickle.dump(f_c, f)
        
    loss_rf_val = rmse(f_c.predict(x_val).squeeze(), y_val)
    loss_rf_test = rmse(f_c.predict(x_test).squeeze(), y_test)

    # Neural net
    try:
        with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'rb') as f:
            f_c = pickle.load(f)
    except Exception:
        with fixedseed(np, 20231103):
            f_c = Ensemble(MLPRegressor, 10, (28,), learning_rate_init=lr, max_iter=500)
            f_c.fit(x_train, y_train.squeeze())
        with open(f'models/{ds_name}/{ds_index}/nn.pickle', 'wb') as f:
            pickle.dump(f_c, f)

    preds_val = f_c.predict(x_val)
    preds_test = f_c.predict(x_test)
    loss_nn_val = rmse(preds_val, y_val)
    loss_nn_test = rmse(preds_test, y_test)

    # Baselines
    loss_lv_val = rmse(x_val[..., -1:], y_val)
    loss_lv_test = rmse(x_test[..., -1:], y_test)
    loss_mean_val = rmse(x_val.mean(axis=-1).reshape(-1, 1), y_val) 
    loss_mean_test = rmse(x_test.mean(axis=-1).reshape(-1, 1), y_test) 

    log_val.append({'lv': loss_lv_val, 'mean': loss_mean_val, 'linear': loss_i_val, 'nn': loss_nn_val, 'rf': loss_rf_val})
    log_test.append({'lv': loss_lv_test, 'mean': loss_mean_test, 'linear': loss_i_test, 'nn': loss_nn_test, 'rf': loss_rf_test})

    return log_val, log_test


if __name__ == '__main__':
    main()
