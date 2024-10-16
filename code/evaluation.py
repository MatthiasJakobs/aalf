import tqdm
import numpy as np
import pickle 
import pandas as pd
import matplotlib.pyplot as plt

from models import MedianPredictionEnsemble
from datasets import load_dataset
from tsx.datasets import windowing, split_proportion
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
from cdd_plots import DATASET_DICT_SMALL
from cdd_plots import TREATMENT_DICT
from os import system, remove
from os.path import basename, exists, dirname, join
from shutil import which, move
from subprocess import run
from utils import smse as standardized_mean_squared_error

ALL_METRICS = ['RMSE', 'MAE', 'MSE']

def load_models(ds_name, ds_index):
    with open(f'models/{ds_name}/{ds_index}/linear.pickle', 'rb') as f:
        f_i = pickle.load(f)
    
    f_c = MedianPredictionEnsemble.load_model(ds_name, ds_index)

    return f_i, f_c

def preprocess_data(X, L, H):
    # Split and normalize data
    X_train, _ = split_proportion(X, (0.5, 0.5))
    X = (X - np.mean(X)) / np.std(X)
    X_train, X_val, X_test = split_proportion(X, (0.5, 0.25, 0.25))

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
    ds_fraction = 1
    rng = np.random.RandomState(20240103)

    L = 10
    results = { method_name: {'Dataset': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'SMSE': []} for method_name in methods}

    for ds_name in ds_names:
        _X, horizons, indices = load_dataset(ds_name)

        # Get fraction
        indices = rng.choice(indices, size=int(len(indices)*ds_fraction), replace=False)

        mses = {method_name: 0 for method_name in methods}
        maes = {method_name: 0 for method_name in methods}
        rmses = {method_name: 0 for method_name in methods}
        smses = {method_name: 0 for method_name in methods}

        for ds_index in tqdm.tqdm(indices, desc=ds_name):
            (_, _), (_, _), (_, y_test) = preprocess_data(_X[ds_index], L, horizons[ds_index])
            y_test = y_test.reshape(-1)

            for method_name in methods:

                preds = np.load(f'preds/{ds_name}/{ds_index}/{method_name}.npy')
                mse = mean_squared_error(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                rmse = mean_squared_error(y_test, preds, squared=False)
                smse = standardized_mean_squared_error(y_test, preds)
                mses[method_name] += (mse / len(indices))
                maes[method_name] += (mae / len(indices))
                rmses[method_name] += (rmse / len(indices))
                smses[method_name] += (smse / len(indices))

        for method_name in methods:
            results[method_name]['Dataset'].append(DATASET_DICT_SMALL[ds_name])
            results[method_name]['MSE'].append(mses[method_name])
            results[method_name]['MAE'].append(maes[method_name])
            results[method_name]['RMSE'].append(rmses[method_name])
            results[method_name]['SMSE'].append(smses[method_name])

    print(results)
    results = { k: pd.DataFrame(v).set_index('Dataset') for k, v in results.items() }
    df = pd.concat(results.values(), axis=1, keys=results.keys())
    return df

def highlight_min(s, to_highlight):
    smallest_indices = []
    second_smallest_indices = []
    for th in to_highlight:
        s_sorted = np.sort(s[th])
        smallest_index = np.where(s == s_sorted[0])[0][0]
        second_smallest_index = np.where(s == s_sorted[1])[0][0]
        smallest_indices.append(smallest_index)
        second_smallest_indices.append(second_smallest_index)

    to_return = []
    for idx in range(len(s)):
        if idx in smallest_indices:
            to_return.append('textbf:--rwrap;')
        elif idx in second_smallest_indices:
            to_return.append('underline:--rwrap;')
        else:
            to_return.append(None)
    return to_return

def make_pretty(styler, to_highlight, save_path, transpose=False, hide_metrics=False):
    if transpose:
        hide_axis=1
        highlight_axis=0
    else:
        hide_axis=0
        highlight_axis=1

    styler.apply(highlight_min, axis=highlight_axis, to_highlight=to_highlight)
    styler.format(precision=3)
    styler.hide(axis=hide_axis, names=True)

    if hide_metrics:
        styler.hide(axis=highlight_axis, level=1)

    if transpose:
        tex = styler.to_latex(hrules=True)
    else:
        tex = styler.to_latex(multicol_align='c', hrules=True)

    # Use tabular*
    tex_list = tex.splitlines()
    tex_list = ['\\begin{tabular*}{\\linewidth}{@{\extracolsep{\\fill}} l'+ 'r'*len(to_highlight[0]) + '}'] + tex_list[1:-1] + ['\\end{tabular*}']
    tex = '\n'.join(tex_list)

    with open(save_path, 'w+') as _f:
        _f.write(tex)
    
    return styler


def plot_table(df, save_path, methods, transpose=False):

    metrics = ['RMSE']

    # Drop metrics not used 
    columns_to_keep = [(level_0, level_1) for level_0, level_1 in df.columns if level_1 in metrics]
    df = df[columns_to_keep]

    # Drop methods not used 
    columns_to_keep = [(level_0, level_1) for level_0, level_1 in df.columns if level_0 in methods]
    df = df[columns_to_keep]

    # Rename methods
    methods = [TREATMENT_DICT.get(t_name, t_name) for t_name in methods]
    df.columns = pd.MultiIndex.from_tuples([(TREATMENT_DICT.get(t_name, t_name), m_name) for (t_name, m_name) in df.columns])
    print(df.columns)
    
    to_highlight = [list(product(methods, [metric])) for metric in metrics]
    if transpose:
        df = df.T
    df.style.pipe(make_pretty, save_path=save_path, to_highlight=to_highlight, transpose=transpose, hide_metrics=len(metrics)==1)

def render_table_as_pdf(TABLE_PATH):
    latex_installed = which('pdflatex') is not None
    if latex_installed:
        # Load output file againg
        with open(TABLE_PATH, 'r') as _f:
            tex = [line.strip() for line in _f.readlines()]
        # Build a standalone latex file
        standalone = [
            r'\documentclass[border=5pt]{standalone}',
            r'\usepackage{booktabs}',
            r'\usepackage{amssymb}',
            r'\usepackage{amsmath}',
            r'\begin{document}',
            r'\setlength\tabcolsep{0pt}',
            *tex,
            r'\end{document}',
        ]
        tex = '\n'.join(standalone)
        print(basename(TABLE_PATH))
        print(dirname(TABLE_PATH))
        with open('tmp.tex', 'w') as _f:
            _f.write(tex)
        run(['pdflatex', 'tmp.tex'])
        remove('tmp.tex')
        remove('tmp.log')
        remove('tmp.aux')

        new_path = join(dirname(TABLE_PATH), basename(TABLE_PATH).replace('tex', 'pdf'))
        print(new_path)
        move('tmp.pdf', new_path)

if __name__ == '__main__':

    methods = ['v12_0.5', 'v12_0.9', 'v12_0.95', 'v12_0.99', 'oms', 'knnroc', 'ade', 'dets']
    full_methods = ['MeanValue', 'LastValue', 'v12_0.5', 'v12_0.6', 'v12_0.7', 'v12_0.8', 'v12_0.9', 'v12_0.95', 'v12_0.99', 'oms', 'knnroc', 'ade', 'dets', 'lin', 'nn' ]
    ds_names = ['pedestrian_counts', 'web_traffic', 'kdd_cup_nomissing', 'weather' ]

    EVAL_PATH = 'results/eval.pickle'
    TABLE_PATH = 'results/metrics_table.tex'
    FULL_TABLE_PATH = 'results/metrics_table_all.tex'

    if not exists(EVAL_PATH):
        print('recreate', EVAL_PATH)
        results = main(ds_names, full_methods)
        results.to_pickle(EVAL_PATH)
    else:
        results = pd.read_pickle(EVAL_PATH)

    print(results)

    plot_table(results, TABLE_PATH, methods, transpose=False)
    render_table_as_pdf(TABLE_PATH)

    # Plot long table for all results
    plot_table(results, FULL_TABLE_PATH, full_methods, transpose=True)
    render_table_as_pdf(FULL_TABLE_PATH)
