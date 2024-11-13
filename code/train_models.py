import pickle
import torch
import numpy as np
from utils import rmse
from sklearn.linear_model import LinearRegression
from preprocessing import load_global_data, load_local_data
from seedpy import fixedseed
from config import DATASET_HYPERPARAMETERS, DEEPAR_HYPERPARAMETERS, FCN_HYPERPARAMETERS
from os import makedirs
from tsx.utils import string_to_randomstate
from models import DeepAR, FCNN, GlobalTorchDataset

def fit_fcnn(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=True)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)
    
    with fixedseed([torch, np], seed=random_state):
        mlp = FCNN(L, **FCN_HYPERPARAMETERS[ds_name])
        mlp.fit(ds_train, ds_val, verbose=True)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        _X_test = _X_test.reshape(_X_test.shape[0], -1).astype(np.float32)
        test_preds = mlp.predict(_X_test).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('fcnn', np.mean(losses))

    with open(f'models/{ds_name}/fcnn.pickle', 'wb') as _f:
        pickle.dump(mlp, _f)


def fit_deepar(ds_name):
    random_state = string_to_randomstate(ds_name, return_seed=True)
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    ds_train = GlobalTorchDataset(ds_name, freq, L, 1, split='train', return_X_y=False)

    # Get some data for validation
    rng = np.random.RandomState(random_state)
    val_indices = rng.binomial(n=1, p=0.1, size=len(ds_train))

    ds_val = torch.utils.data.Subset(ds_train, np.where(val_indices)[0])
    ds_train = torch.utils.data.Subset(ds_train, np.where(np.logical_not(val_indices))[0])

    makedirs(f'models/{ds_name}/', exist_ok=True)

    with fixedseed([np, torch], seed=random_state):
        DAR = DeepAR(n_channel=4, **DEEPAR_HYPERPARAMETERS[ds_name])
        DAR.fit(ds_train, ds_val, verbose=True)

    (_, _), (_, _), (X_test, y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    losses = []
    for _X_test, _y_test in zip(X_test, y_test):
        test_preds = DAR.predict(_X_test.astype(np.float32)).reshape(_y_test.shape)
        loss = rmse(test_preds, _y_test)
        losses.append(loss)
    print('deepar', np.mean(losses))

    with open(f'models/{ds_name}/deepar.pickle', 'wb') as _f:
        pickle.dump(DAR, _f)

def evaluate_models(ds_name):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    (local_X_train, local_y_train), (_, _), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1)
    (_, _), (_, _), (global_X_test, global_y_test) = load_global_data(ds_name, L=L, H=1, freq=freq)

    print('---')
    print(ds_name)

    for m_name in ['linear', 'fcnn', 'deepar']:

        if m_name == 'linear':
            losses = []
            for ds_index in range(len(local_X_train)):
                m = LinearRegression()
                m.fit(local_X_train[ds_index], local_y_train[ds_index])
                test_preds = m.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape)
                loss = rmse(test_preds, local_y_test[ds_index])
                losses.append(loss)
            print(f'{m_name}: {np.mean(losses):.3f}')
            continue
        elif m_name == 'deepar':
            with open(f'models/{ds_name}/deepar.pickle', 'rb') as f:
                m = pickle.load(f).to('cpu')
                m.device = 'cpu'
                m.lstm.flatten_parameters()
        elif m_name == 'fcnn':
            with open(f'models/{ds_name}/fcnn.pickle', 'rb') as f:
                m = pickle.load(f)
                m.device = 'cpu'
                m.model = m.model.to('cpu')
        else:
            raise NotImplementedError('Unknown model', m_name)
        losses = []
        for X_test, y_test in zip(global_X_test, global_y_test):
            if m_name == 'fcnn':
                X_test = X_test.reshape(X_test.shape[0], -1)
            test_preds = m.predict(X_test).reshape(y_test.shape)
            loss = rmse(test_preds, y_test)
            losses.append(loss)
        print(f'{m_name}: {np.mean(losses):.3f}')


def main():
    fit_deepar('weather')
    fit_deepar('nn5_daily_nomissing')
    fit_deepar('australian_electricity_demand')
    fit_deepar('pedestrian_counts')

    fit_fcnn('weather')
    fit_fcnn('nn5_daily_nomissing')
    fit_fcnn('australian_electricity_demand')
    fit_fcnn('pedestrian_counts')

    evaluate_models('weather')
    evaluate_models('nn5_daily_nomissing')
    evaluate_models('australian_electricity_demand')
    evaluate_models('pedestrian_counts')

if __name__ == '__main__':
    main()