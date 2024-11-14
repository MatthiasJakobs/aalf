import tqdm
import numpy as np
import pandas as pd
from tsx.datasets.monash import load_monash
from tsx.datasets import split_proportion, windowing
from scipy.stats import zscore

def generate_covariates(length, freq, start_date=None):
    def _zscore(X):
        if np.all(X[0] == X):
            return np.zeros((len(X)))
        return zscore(X)

    if start_date is None:
        start_date = '1970-01-01'

    time_series = pd.date_range(start=start_date, periods=length, freq=freq)

    weekday = _zscore(time_series.weekday.to_numpy())
    hour = _zscore(time_series.hour.to_numpy())
    month = _zscore(time_series.month.to_numpy())
    covariates = np.stack([weekday, hour, month]).T

    return covariates


def _load_data(ds_name, return_start_dates=False):
    ds = load_monash(ds_name)
    Xs = ds['series_value']
    indices = range(len(Xs))
    if 'start_timestamp' in ds.keys():
        start_dates = ds['start_timestamp'].tolist()
    else:
        start_dates = [pd.Timestamp('1980-1-1') for _ in indices]

    X_train = []
    X_val = []
    X_test = []

    for X in Xs:
        x_train, x_val, x_test = split_proportion(X.to_numpy(), (0.8, 0.1, 0.1))
        mu, std = np.mean(x_train), np.std(x_train)

        X_train.append((x_train - mu) / std)
        X_val.append((x_val - mu) / std)
        X_test.append((x_test - mu) / std)

    if return_start_dates:
        return X_train, X_val, X_test, start_dates
    else:
        return X_train, X_val, X_test

def load_local_data(ds_name, L, H, verbose=True):
    _x_train = []
    _x_val = []
    _x_test = []
    _y_train = []
    _y_val = []
    _y_test = []

    X_train, X_val, X_test = _load_data(ds_name)
    for ds_index in tqdm.trange(len(X_train), desc=f'[{ds_name}] get local data', disable=(not verbose)):
        x_train, y_train = windowing(X_train[ds_index], L=L, H=H)
        x_val, y_val = windowing(X_val[ds_index], L=L, H=H)
        x_test, y_test = windowing(X_test[ds_index], L=L, H=H)

        _x_train.append(x_train)
        _x_val.append(x_val)
        _x_test.append(x_test)

        _y_train.append(y_train)
        _y_val.append(y_val)
        _y_test.append(y_test)


    return (_x_train, _y_train), (_x_val, _y_val), (_x_test, _y_test)

def load_global_data(ds_name, L, H, freq, verbose=True):
    _X_train, _X_val, _X_test, start_dates = _load_data(ds_name, return_start_dates=True)

    train_data_X = []
    val_data_X = []
    test_data_X = []
    train_data_y = []
    val_data_y = []
    test_data_y = []

    for ds_index in tqdm.trange(len(_X_train), desc=f'[{ds_name}] get global data', disable=(not verbose)):
        # Create entire TS again for covariate generation
        X_train = _X_train[ds_index]
        X_val = _X_val[ds_index]
        X_test = _X_test[ds_index]
        n_steps = len(X_train) + len(X_val) + len(X_test)
        covariates = generate_covariates(n_steps, freq, start_date=start_dates[ds_index])

        train_length = len(X_train)
        val_length = len(X_val)

        C_train = covariates[:train_length]
        C_val = covariates[train_length:train_length+val_length]
        C_test = covariates[train_length+val_length:]
        
        x_train, y_train = windowing(X_train, L=L, H=H)
        x_val, y_val = windowing(X_val, L=L, H=H)
        x_test, y_test = windowing(X_test, L=L, H=H)
        c_train, _ = windowing(C_train, L=L, H=H)
        c_val, _ = windowing(C_val, L=L, H=H)
        c_test, _ = windowing(C_test, L=L, H=H)

        x_train = x_train[..., None]
        x_val = x_val[..., None]
        x_test = x_test[..., None]
        x_train = np.concatenate([x_train, c_train], axis=-1)
        x_val = np.concatenate([x_val, c_val], axis=-1)
        x_test = np.concatenate([x_test, c_test], axis=-1)

        train_data_X.append(x_train)
        train_data_y.append(y_train)
        val_data_X.append(x_val)
        val_data_y.append(y_val)
        test_data_X.append(x_test)
        test_data_y.append(y_test)

    X_train = np.concatenate(train_data_X)
    y_train = np.concatenate(train_data_y)

    return (X_train, y_train), (val_data_X, val_data_y), (test_data_X, test_data_y)

def main():
    from config import ALL_DATASETS
    total = 0
    for ds_name in ALL_DATASETS:
        ds = load_monash(ds_name)
        Xs = ds['series_value']
        lengths = [len(X) for X in Xs]
        total += len(Xs)
        print(ds_name, 'n_series', len(Xs), 'min_length', min(lengths), 'max_length', max(lengths), 'mean_length', np.mean(lengths))
    print('---')
    print('Total', total)

if __name__ == '__main__':
    main()
