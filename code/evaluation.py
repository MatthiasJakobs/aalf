import numpy as np
import pickle 
import pandas as pd

from main import MedianPredictionEnsemble
from datasets import load_dataset
from tsx.datasets import windowing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
from cdd_plots import DATASET_DICT_SMALL
from os import system, remove
from os.path import basename, exists, dirname, join

def load_models(ds_name, ds_index):
    with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'rb') as f:
        f_i = pickle.load(f)
    
    f_c = MedianPredictionEnsemble.load_model(ds_name, ds_index)

    return f_i, f_c

def preprocess_data(X, L, H):
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
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    y_train = y_train[..., -1:]
    y_val = y_val[..., -1:]
    y_test = y_test[..., -1:]

    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main(ds_names, methods):
    ds_fraction = 0.3
    rng = np.random.RandomState(20240103)

    L = 10
    results = { method_name: {'Dataset': [], 'MAE': [], 'MSE': []} for method_name in methods}

    for ds_name in ds_names:
        _X, horizons, indices = load_dataset(ds_name)

        # Get fraction
        indices = rng.choice(indices, size=int(len(indices)*ds_fraction), replace=False)

        mses = {method_name: 0 for method_name in methods}
        maes = {method_name: 0 for method_name in methods}

        for ds_index in indices:
            (_, _), (_, _), (_, y_test) = preprocess_data(_X[ds_index], L, horizons[ds_index])
            y_test = y_test.reshape(-1)

            for method_name in methods:

                preds = np.load(f'preds/{ds_name}/{ds_index}/{method_name}.npy')
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                mses[method_name] += (mse / len(indices))
                maes[method_name] += (mae / len(indices))

        for method_name in methods:
            results[method_name]['Dataset'].append(DATASET_DICT_SMALL[ds_name])
            results[method_name]['MSE'].append(mses[method_name])
            results[method_name]['MAE'].append(maes[method_name])

    print(results)
    results = { k: pd.DataFrame(v).set_index('Dataset') for k, v in results.items() }
    df = pd.concat(results.values(), axis=1, keys=results.keys())
    return df

def highlight_min(s, to_highlight):
    indices = []
    for th in to_highlight:
        smallest_index = np.where(s == np.min(s[th].to_numpy()))[0][0]
        indices.append(smallest_index)
    return ['textbf:--rwrap;' if idx in indices else None for idx in range(len(s)) ]

def make_pretty(styler, to_highlight, save_path, transpose=False):
    if transpose:
        hide_axis=1
        highlight_axis=0
    else:
        hide_axis=0
        highlight_axis=1

    styler.apply(highlight_min, axis=highlight_axis, to_highlight=to_highlight)
    styler.format(precision=3)
    styler.hide(axis=hide_axis, names=True)

    if transpose:
        styler.to_latex(save_path, hrules=True)
    else:
        styler.to_latex(save_path, multicol_align='c', hrules=True)
    
    return styler


def plot_table(df, save_path, methods, transpose=False):

    metrics = ['MSE', 'MAE']
    
    to_highlight = [list(product(methods, [metric])) for metric in metrics]
    if transpose:
        df = df.T
    df.style.pipe(make_pretty, save_path=save_path, to_highlight=to_highlight, transpose=transpose)

if __name__ == '__main__':

    methods = ['lin', 'nn', 'v11_0.7', 'v11_0.8', 'v11_0.9']
    ds_names = ['pedestrian_counts', 'web_traffic', 'kdd_cup_nomissing', 'weather' ]

    EVAL_PATH = 'results/eval.pickle'
    TABLE_PATH = 'results/table.tex'

    if not exists(EVAL_PATH):
        print()
        results = main(ds_names, methods)
        results.to_pickle(EVAL_PATH)
    else:
        results = pd.read_pickle(EVAL_PATH)

    plot_table(results, TABLE_PATH, methods, transpose=False)