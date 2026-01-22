import pickle
import torch
import numpy as np
import pandas as pd
import io
import tqdm

from sklearn.linear_model import LinearRegression
from config import DATASET_HYPERPARAMETERS
from utils import highlight_min_multicolumn, format_significant, rmse, smape
from preprocessing import load_local_data, load_global_data, _load_data
from os import makedirs
from os.path import exists
from critdd import Diagrams
from config import ALL_DATASETS, DS_MAP, MODEL_MAP, LOSS_MAP
from itertools import product
from joblib import Parallel, delayed
from models import AutoARIMA, AutoETS
import warnings

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        else: return super().find_class(module, name)

def save_results(ds_name, model_name, train_preds, val_preds, test_preds, rmse_values, smape_values):
    # Check if output file exists, create otherwise
    if exists(f'preds/{ds_name}.pickle'):
        with open(f'preds/{ds_name}.pickle', 'rb') as f:
            predictions = pickle.load(f)
    else:
        predictions = { 'train': {}, 'val': {}, 'test': {} }

    predictions['train'][model_name] = train_preds
    predictions['val'][model_name] = val_preds
    predictions['test'][model_name] = test_preds

    with open(f'preds/{ds_name}.pickle', 'wb') as f:
        pickle.dump(predictions, f)

    if exists(f'results/basemodel_losses/{ds_name}.csv'):
        df = pd.read_csv(f'results/basemodel_losses/{ds_name}.csv', index_col=0)
    else:
        df = pd.DataFrame()

    df[f'{model_name}_rmse'] = rmse_values
    df[f'{model_name}_smape'] = smape_values
    df.to_csv(f'results/basemodel_losses/{ds_name}.csv')

def evaluate_autoets(ds_name):
    def _process_single(X_train, local_X_train, local_y_train, local_X_val, local_y_val, local_X_test, local_y_test):
        ets = AutoETS(seasonal=L)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*overflow encountered.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*failed to converge.*', category=UserWarning)
            ets.fit(X_train)

            train_preds_ets = ets.predict(local_X_train, local_y_train)
            val_preds_ets = ets.predict(local_X_val, local_y_val)
            test_preds_ets = ets.predict(local_X_test, local_y_test)

        return train_preds_ets, val_preds_ets, test_preds_ets, rmse(test_preds_ets.squeeze(), local_y_test.squeeze()), smape(test_preds_ets.squeeze(), local_y_test.squeeze())

    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']

    X_train, X_val, X_test = _load_data(ds_name, return_start_dates=False)
    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=True)

    n_ts = len(local_X_train)
    n_jobs = min(n_ts, 6)  # number of threads

    results_list = Parallel(n_jobs=n_jobs)(delayed(_process_single)(
            X_train[ds_index], local_X_train[ds_index], local_y_train[ds_index], local_X_val[ds_index], local_y_val[ds_index], local_X_test[ds_index], local_y_test[ds_index]
        )
        for ds_index in tqdm.trange(len(local_X_train), desc='Run AutoETS')
    )
    #results_list = [_process_single(ds_index, X_train, local_X_train, local_y_train, local_X_val, local_y_val, local_X_test, local_y_test) for ds_index in tqdm.trange(len(local_X_train), desc='Run AutoETS')]

    all_train_predictions = [res[0] for res in results_list]
    all_val_predictions = [res[1] for res in results_list]
    all_test_predictions = [res[2] for res in results_list]
    all_rmse = [res[3] for res in results_list]
    all_smape = [res[4] for res in results_list]
    save_results(ds_name, 'autoets', all_train_predictions, all_val_predictions, all_test_predictions, all_rmse, all_smape)

def evaluate_autoarima(ds_name):
    def _process_single(X_train, local_X_train, local_y_train, local_X_val, local_y_val, local_X_test, local_y_test):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Non-invertible starting MA parameters found.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*Non-stationary starting autoregressive parameters found.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*Maximum Likelihood optimization failed to converge.*', category=UserWarning)

            arima = AutoARIMA(max_d=1, max_p=5, max_q=5, n_runs=10, random_state=1234)

            # The series are too long for AutoARIMA to handle (on our hardware, at least)
            x = X_train[-5000:]
            arima.fit(x)
            train_preds_arima = arima.predict(local_X_train, local_y_train)
            val_preds_arima = arima.predict(local_X_val, local_y_val)
            test_preds_arima = arima.predict(local_X_test, local_y_test)

        return train_preds_arima, val_preds_arima, test_preds_arima, rmse(test_preds_arima.squeeze(), local_y_test.squeeze()), smape(test_preds_arima.squeeze(), local_y_test.squeeze())

    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']

    X_train, _, _ = _load_data(ds_name, return_start_dates=False)

    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=True)

    n_ts = len(local_X_train)
    n_jobs = min(n_ts, 6)  # number of threads

    results_list = Parallel(n_jobs=n_jobs)(delayed(_process_single)(
            X_train[ds_index], local_X_train[ds_index], local_y_train[ds_index],
            local_X_val[ds_index], local_y_val[ds_index], local_X_test[ds_index], local_y_test[ds_index],
        )
        for ds_index in tqdm.trange(len(local_X_train), desc='Run AutoARIMA')
    )
    #results_list = [_process_single(ds_index, X_train, local_X_train, local_y_train, local_X_val, local_y_val, local_X_test, local_y_test) for ds_index in tqdm.trange(len(local_X_train), desc='Run AutoARIMA')]

    all_train_predictions = [res[0] for res in results_list]
    all_val_predictions = [res[1] for res in results_list]
    all_test_predictions = [res[2] for res in results_list]
    all_rmse = [res[3] for res in results_list]
    all_smape = [res[4] for res in results_list]
    save_results(ds_name, 'autoarima', all_train_predictions, all_val_predictions, all_test_predictions, all_rmse, all_smape)

def evaluate_linear(ds_name):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']

    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=False)

    all_train_predictions = []
    all_val_predictions = []
    all_test_predictions = []
    all_rmse = []
    all_smape = []
    for ds_index in tqdm.trange(len(local_X_train), desc='Run Linear'):
        lin = LinearRegression()
        lin.fit(local_X_train[ds_index], local_y_train[ds_index])
        train_preds_lin = lin.predict(local_X_train[ds_index]).reshape(local_y_train[ds_index].shape).squeeze()
        val_preds_lin = lin.predict(local_X_val[ds_index]).reshape(local_y_val[ds_index].shape).squeeze()
        test_preds_lin = lin.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape).squeeze()

        all_train_predictions.append(train_preds_lin)
        all_val_predictions.append(val_preds_lin)
        all_test_predictions.append(test_preds_lin)
        all_rmse.append(rmse(test_preds_lin.squeeze(), local_y_test[ds_index].squeeze()))
        all_smape.append(smape(test_preds_lin.squeeze(), local_y_test[ds_index].squeeze()))

    save_results(ds_name, 'linear', all_train_predictions, all_val_predictions, all_test_predictions, all_rmse, all_smape)

def evaluate_global_models(ds_name):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']
    loss_fn_names = ['rmse', 'smape']
    model_names = ['fcnn', 'deepar', 'cnn']

    all_loss_values = {f'{m_name}_{loss_name}': [] for m_name, loss_name in product(model_names, loss_fn_names)}
    all_train_predictions = {m_name: [] for m_name in model_names + ['y']}
    all_val_predictions = {m_name: [] for m_name in model_names + ['y']}
    all_test_predictions = {m_name: [] for m_name in model_names + ['y']}

    X_train, _, _ = _load_data(ds_name, return_start_dates=False)
    n_ts = len(X_train)
    del X_train

    # Load global models
    with open(f'models/{ds_name}/deepar.pickle', 'rb') as f:
        deepar = CPU_Unpickler(f).load().to('cpu')
        deepar.device = 'cpu'
        deepar.lstm.flatten_parameters()

    with open(f'models/{ds_name}/fcnn.pickle', 'rb') as f:
        fcnn = CPU_Unpickler(f).load()
        fcnn.device = 'cpu'
        fcnn.model = fcnn.model.to('cpu')

    with open(f'models/{ds_name}/cnn.pickle', 'rb') as f:
        cnn = CPU_Unpickler(f).load()
        cnn.device = 'cpu'
        cnn.model = cnn.model.to('cpu')

    for _X_train, _y_train, _X_val, _y_val, _X_test, _y_test in tqdm.tqdm(load_global_data(ds_name, L=L, H=1, freq=freq), total=n_ts, desc='process global models'):
        for m_name, m in zip(['fcnn', 'deepar', 'cnn'], [fcnn, deepar, cnn]):
            if m_name == 'fcnn':
                X_train = _X_train.reshape(_X_train.shape[0], -1)
                y_train = _y_train.reshape(_y_train.shape[0], -1)
                X_val = _X_val.reshape(_X_val.shape[0], -1)
                y_val = _y_val.reshape(_y_val.shape[0], -1)
                X_test = _X_test.reshape(_X_test.shape[0], -1)
                y_test = _y_test.reshape(_y_test.shape[0], -1)
            elif m_name == 'cnn':
                X_train = np.swapaxes(_X_train, 1, 2)
                X_val = np.swapaxes(_X_val, 1, 2)
                X_test = np.swapaxes(_X_test, 1, 2)
            elif m_name == 'deepar':
                X_train = _X_train
                X_val = _X_val
                X_test = _X_test
                y_train = _y_train
                y_val = _y_val
                y_test = _y_test

            train_preds = m.predict(X_train).reshape(y_train.shape)
            val_preds = m.predict(X_val).reshape(y_val.shape)
            test_preds = m.predict(X_test).reshape(y_test.shape)
            all_train_predictions[m_name].append(train_preds.squeeze())
            all_val_predictions[m_name].append(val_preds.squeeze())
            all_test_predictions[m_name].append(test_preds.squeeze())

            for loss_fn in loss_fn_names:
                loss = eval(loss_fn)(test_preds.squeeze(), y_test.squeeze())
                all_loss_values[f'{m_name}_{loss_fn}'].append(loss)

    for m_name in model_names:
        save_results(ds_name, m_name, all_train_predictions[m_name], all_val_predictions[m_name], all_test_predictions[m_name], all_loss_values[f'{m_name}_rmse'], all_loss_values[f'{m_name}_smape'])

def generate_cdd_plot(loss_fn='rmse'):
    datasets_to_evaluate = ALL_DATASETS
    dfs = []
    models = ['linear', 'fcnn', 'deepar', 'cnn']
    full_model_names = [f'{model_name}_{loss_fn}' for model_name in models]

    for ds_name in datasets_to_evaluate:
        df = pd.read_csv(f'results/basemodel_losses/{ds_name}.csv', index_col=0)
        # Add loss function name to index and remap
        df = df[full_model_names]
        df = df.rename({full_model_name: small_model_name for full_model_name, small_model_name in zip(full_model_names, models)}, axis=1)
        dfs.append(df.to_numpy())

    two_dimensional_diagram = Diagrams(
        dfs,
        diagram_names=[DS_MAP[ds_name].replace(' ', r'\\') for ds_name in datasets_to_evaluate],
        treatment_names=[MODEL_MAP[model_name] for model_name in models],
        maximize_outcome=False,
    )
    makedirs('plots', exist_ok=True)
    two_dimensional_diagram.to_file(
        'plots/cdd_single_models.tex',
        preamble = "\n".join([ # colors are defined before \begin{document}
            #fr"\definecolor{{color1}}{{HTML}}{{{COLORS.blue}}}",
        ]),
        axis_options = { # style the plot
            "cycle list": ",".join([ # define the markers for treatments
                r"{color1,mark=x,very thick,mark options={scale=2}}",
                r"{color3,mark=x,very thick,mark options={scale=2}}",
                r"{color2,mark=x,very thick,mark options={scale=2}}",
                r"{color4,mark=x,very thick,mark options={scale=2}}",
            ]),
            'width': '360', # should be 372 but axis labels are not considered
            'height': r'0.9*\axisdefaultheight',
            'xticklabel style': r'font=\fontsize{8}{8}\selectfont',
            'yticklabel style': r'font=\fontsize{8}{8}\selectfont, align=right',
            'legend style': r'at={(0.98, 0.7)}, font=\fontsize{8}{8}\selectfont,/tikz/every odd column/.append style={column sep=.25em}',
        },
    )

def generate_latex_table():
    datasets_to_evaluate = ALL_DATASETS
    dfs = []
    loss_function_names = ['RMSE', 'SMAPE']

    for ds_name in datasets_to_evaluate:
        df = pd.read_csv(f'results/basemodel_losses/{ds_name}.csv', index_col=0)
        # Take mean per loss
        df = df.mean(axis=0)
        # Split name to multiindex
        multiindex_dict = {(MODEL_MAP[k.split('_')[0]], LOSS_MAP[k.split('_')[1]]): v for k, v in df.to_dict().items() }
        df = pd.DataFrame([multiindex_dict], index=[DS_MAP[ds_name]])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        dfs.append(df)

    df = pd.concat(dfs)

    # Style table
    df = df.apply(format_significant, axis=1)
    df = df.apply(highlight_min_multicolumn, metric_names=loss_function_names, axis=1)

    #df = df.transpose()

    latex_output = df.to_latex(
        multirow=False,
        escape=False, 
        multicolumn_format='l',
        #column_format=r'llp{2cm}p{1.0cm}p{1.0cm}p{1.3cm}p{1.0cm}p{1.0cm}',
        column_format=r'p{2.5cm}p{0.75cm}p{0.85cm}p{0.75cm}p{0.85cm}p{0.75cm}p{0.85cm}p{0.75cm}p{0.85cm}',
    )
    with open(f'plots/single_results.tex', 'w') as f:
        f.write(latex_output)

def main():
    # for ds_name in ALL_DATASETS:
    #     evaluate_models(ds_name, verbose=True)
    # generate_cdd_plot(loss_fn='rmse')
    # generate_latex_table()
    for ds_name in ALL_DATASETS[2:]:
        evaluate_autoarima(ds_name)
        evaluate_autoets(ds_name)

if __name__ == '__main__':
    main()
