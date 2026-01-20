import pickle
import torch
import numpy as np
from seedpy import fixedseed
from config import DATASET_HYPERPARAMETERS, DEEPAR_HYPERPARAMETERS, FCN_HYPERPARAMETERS, CNN_HYPERPARAMETERS
from os import makedirs
from tsx.utils import string_to_randomstate
from models import DeepAR, DeepVAR, FCNN, GlobalTorchDataset, CNN
from os.path import exists

def fit_cnn(ds_name):
    makedirs(f'models/{ds_name}/', exist_ok=True)
    if exists(f'models/{ds_name}/cnn.pickle'):
        return
    print('Fit cnn on', ds_name)
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
    
    with fixedseed([torch, np], seed=random_state):
        cnn = CNN(L, n_channels=n_channels, **CNN_HYPERPARAMETERS[ds_name])
        cnn.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/cnn.pickle', 'wb') as _f:
        pickle.dump(cnn, _f)

def fit_fcnn(ds_name):
    makedirs(f'models/{ds_name}/', exist_ok=True)
    if exists(f'models/{ds_name}/fcnn.pickle'):
        return
    print('Fit fcnn on', ds_name)
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
    
    with fixedseed([torch, np], seed=random_state):
        mlp = FCNN(L, n_channels=n_channels, **FCN_HYPERPARAMETERS[ds_name])
        mlp.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/fcnn.pickle', 'wb') as _f:
        pickle.dump(mlp, _f)

def fit_deepar(ds_name):
    makedirs(f'models/{ds_name}/', exist_ok=True)
    if exists(f'models/{ds_name}/deepar.pickle'):
        return
    print('Fit deepar on', ds_name)
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

    with fixedseed([np, torch], seed=random_state):
        if n_channels == 1:
            DAR = DeepAR(n_channel=4, **DEEPAR_HYPERPARAMETERS[ds_name])
        else:
            DAR = DeepVAR(random_state=random_state, **DEEPAR_HYPERPARAMETERS[ds_name])
        DAR.fit(ds_train, ds_val, verbose=True)

    with open(f'models/{ds_name}/deepar.pickle', 'wb') as _f:
        pickle.dump(DAR, _f)

def main():
    ALL_DATASETS = ['australian_electricity_demand', 'nn5_daily_nomissing', 'weather', 'pedestrian_counts', 'kdd_cup_nomissing', 'solar_10_minutes']
    for ds_name in ALL_DATASETS:
        fit_fcnn(ds_name)
        fit_deepar(ds_name)
        fit_cnn(ds_name)

if __name__ == '__main__':
    main()
