import tqdm
import numpy as np
import pandas as pd
from tsx.datasets.monash import load_monash
from tsx.datasets import split_proportion, windowing
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from config import MULTIVARIATE_DATASETS

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
    Xs = [x.to_numpy() for x in ds['series_value']]
    indices = range(len(Xs))

    if ds_name == 'solar_10_minutes': 
        # Resample to hourly
        Xs = [x.reshape(-1, 6).mean(axis=1).to_numpy() for x in ds['series_value']]

    if 'start_timestamp' in ds.keys():
        start_dates = ds['start_timestamp'].tolist()
    else:
        start_dates = [pd.Timestamp('1980-1-1') for _ in indices]

    is_multivariate = ds_name in MULTIVARIATE_DATASETS

    if is_multivariate:    
        Xs = np.stack(Xs).T[None, ...]
        indices = [0]
        start_dates = [start_dates[0]]

    X_train = []
    X_val = []
    X_test = []

    for ds_index, X in enumerate(Xs):
        x_train, x_val, x_test = split_proportion(X, (0.8, 0.1, 0.1))
        
        if not is_multivariate:
            if (x_train[0] == x_train).all() or (x_val[0] == x_val).all() or (x_test[0] == x_test).all():
                continue
            x_train = x_train.reshape(-1, 1)
            x_val = x_val.reshape(-1, 1)
            x_test = x_test.reshape(-1, 1)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).squeeze()
        x_val = scaler.transform(x_val).squeeze()
        x_test = scaler.transform(x_test).squeeze()

        X_train.append(x_train)
        X_val.append(x_val)
        X_test.append(x_test)

    if return_start_dates:
        return X_train, X_val, X_test, start_dates
    else:
        return X_train, X_val, X_test

def load_local_data(ds_name, L, H, return_split=None, verbose=True):
    _x_train = []
    _x_val = []
    _x_test = []
    _y_train = []
    _y_val = []
    _y_test = []

    if return_split is None:
        return_split = ['train', 'val', 'test']
    else:
        if isinstance(return_split, str):
            return_split = [return_split]

    X_train, X_val, X_test = _load_data(ds_name)
    for ds_index in tqdm.trange(len(X_train), desc=f'[{ds_name}] get local data', disable=(not verbose)):
        val_length = len(X_val[ds_index])
        if val_length <= L+H:
            continue
        if 'train' in return_split:
            x_train, y_train = windowing(X_train[ds_index], L=L, H=H)
            _x_train.append(x_train)
            _y_train.append(y_train)
        if 'val' in return_split:
            x_val, y_val = windowing(X_val[ds_index], L=L, H=H)
            _x_val.append(x_val)
            _y_val.append(y_val)
        if 'test' in return_split:
            x_test, y_test = windowing(X_test[ds_index], L=L, H=H)
            _x_test.append(x_test)
            _y_test.append(y_test)

    return (_x_train, _y_train), (_x_val, _y_val), (_x_test, _y_test)

def load_global_data(ds_name, L, H, freq):
    _X_train, _X_val, _X_test, start_dates = _load_data(ds_name, return_start_dates=True)

    for ds_index in range(len(_X_train)):
        # Create entire TS again for covariate generation
        X_train = _X_train[ds_index]
        X_val = _X_val[ds_index]
        X_test = _X_test[ds_index]
        n_steps = len(X_train) + len(X_val) + len(X_test)
        covariates = generate_covariates(n_steps, freq, start_date=start_dates[ds_index])

        train_length = len(X_train)
        val_length = len(X_val)

        if val_length <= L+H:
            continue

        C_train = covariates[:train_length]
        x_train, y_train = windowing(X_train, L=L, H=H)
        c_train, _ = windowing(C_train, L=L, H=H)
        x_train = np.atleast_3d(x_train)
        x_train = np.concatenate([x_train, c_train], axis=-1)

        C_val = covariates[train_length:train_length+val_length]
        x_val, y_val = windowing(X_val, L=L, H=H)
        c_val, _ = windowing(C_val, L=L, H=H)
        x_val = np.atleast_3d(x_val)
        x_val = np.concatenate([x_val, c_val], axis=-1)

        C_test = covariates[train_length+val_length:]
        x_test, y_test = windowing(X_test, L=L, H=H)
        c_test, _ = windowing(C_test, L=L, H=H)
        x_test = np.atleast_3d(x_test)
        x_test = np.concatenate([x_test, c_test], axis=-1)

        yield (x_train, y_train, x_val, y_val, x_test, y_test)

def _get_last_errors(fint_preds_val, fcomp_preds_val, y_val, fint_preds_test, fcomp_preds_test, y_test):
    last_preds_fint = np.concatenate([fint_preds_val[-1].reshape(-1), fint_preds_test[:-1].reshape(-1)])
    last_preds_fcomp = np.concatenate([fcomp_preds_val[-1].reshape(-1), fcomp_preds_test[:-1].reshape(-1)])
    y_true = np.concatenate([y_val[-1].reshape(-1), y_test[:-1].reshape(-1)])
    return (last_preds_fint-y_true)**2 - (last_preds_fcomp-y_true)**2

def create_selector_features(X_train, y_train, X_test, y_test, train_preds, test_preds):
    #pred_difference = (test_preds[1] - test_preds[0]).reshape(-1, 1)
    pred_difference = (test_preds[0] - test_preds[1]).reshape(-1, 1)
    error_difference = _get_last_errors(train_preds[1], train_preds[0], y_train, test_preds[1], test_preds[0], y_test).reshape(-1, 1)
    statistics = np.concatenate([
        np.mean(X_test, axis=1, keepdims=True),
        np.min(X_test, axis=1, keepdims=True),
        np.max(X_test, axis=1, keepdims=True),
    ], axis=-1)

    return np.concatenate([X_test, pred_difference, error_difference, statistics], axis=-1)

def main():
    from config import ALL_DATASETS
    total = 0
    for ds_name in ALL_DATASETS:
        X_train, X_val, X_test = _load_data(ds_name)
        n_series = len(X_train)
        total += n_series

        lengths = [len(x_train)+len(x_val)+len(x_test) for (x_train, x_val, x_test) in zip(X_train, X_val, X_test)]
        print(ds_name, 'n_series', n_series, 'min_length', min(lengths), 'max_length', max(lengths), 'mean_length', np.mean(lengths))
    print('---')
    print('Total', total)

if __name__ == '__main__':
    main()
