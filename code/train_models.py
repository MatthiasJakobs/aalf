import pandas as pd
import pickle
import torch
import numpy as np
from utils import rmse
from sklearn.linear_model import LinearRegression
from preprocessing import load_global_data, load_local_data
from seedpy import fixedseed
from config import DATASET_HYPERPARAMETERS, DEEPAR_HYPERPARAMETERS, FCN_HYPERPARAMETERS, CNN_HYPERPARAMETERS, ALL_DATASETS
from os import makedirs
from tsx.utils import string_to_randomstate
from models import DeepAR, DeepVAR, FCNN, GlobalTorchDataset, CNN

def fit_cnn(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']
    n_channels = dsh.get('n_channels', 4)

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=True)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)
    
    with fixedseed([torch, np], seed=random_state):
        cnn = CNN(L, n_channels=n_channels, **CNN_HYPERPARAMETERS[ds_name])
        cnn.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/cnn.pickle', 'wb') as _f:
        pickle.dump(cnn, _f)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        _X_test = np.swapaxes(_X_test, 1, 2).astype(np.float32)
        test_preds = cnn.predict(_X_test).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('cnn', np.mean(losses))

def fit_fcnn(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']
    n_channels = dsh.get('n_channels', 1)

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=True)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)
    
    with fixedseed([torch, np], seed=random_state):
        mlp = FCNN(L, n_channels=n_channels, **FCN_HYPERPARAMETERS[ds_name])
        mlp.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/fcnn.pickle', 'wb') as _f:
        pickle.dump(mlp, _f)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        batch_size = _X_test.shape[0]
        _X_test = _X_test.reshape(batch_size, -1).astype(np.float32)
        _y_test = _y_test.reshape(batch_size, -1)
        test_preds = mlp.predict(_X_test).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('fcnn', np.mean(losses))

def fit_deepar(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']
    n_channels = dsh.get('n_channels', 1)

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=False)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)

    with fixedseed([np, torch], seed=random_state):
        if n_channels == 1:
            DAR = DeepAR(n_channel=4, **DEEPAR_HYPERPARAMETERS[ds_name])
        else:
            DAR = DeepVAR(random_state=random_state, **DEEPAR_HYPERPARAMETERS[ds_name])
        DAR.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/deepar.pickle', 'wb') as _f:
        pickle.dump(DAR, _f)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        n_channels = _X_test.shape[-1]-3 if len(_X_test.shape) == 3 else 1
        _y_test = _y_test.reshape(-1, n_channels)
        test_preds = DAR.predict(_X_test.astype(np.float32)).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('deepar', np.mean(losses))

def main():
    for ds_name in ALL_DATASETS:
        fit_fcnn(ds_name)
        fit_deepar(ds_name)
        fit_cnn(ds_name)

if __name__ == '__main__':
    main()
