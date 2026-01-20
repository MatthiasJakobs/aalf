import pickle
import torch
import numpy as np
import pandas as pd
import io
import tqdm

from sklearn.linear_model import LinearRegression
from config import DATASET_HYPERPARAMETERS
from utils import highlight_min_multicolumn, format_significant
from preprocessing import load_local_data, load_global_data
from os import makedirs
from os.path import exists
from critdd import Diagrams
from config import ALL_DATASETS, DS_MAP, MODEL_MAP, LOSS_MAP
from itertools import product

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        else: return super().find_class(module, name)

def evaluate_models(ds_name, verbose=False):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']
    loss_fn_names = ['rmse', 'smape']
    model_names = ['linear', 'fcnn', 'deepar', 'cnn']

    all_loss_values = {f'{m_name}_{loss_name}': [] for m_name, loss_name in product(model_names, loss_fn_names)}
    all_train_predictions = {m_name: [] for m_name in model_names + ['y']}
    all_val_predictions = {m_name: [] for m_name in model_names + ['y']}
    all_test_predictions = {m_name: [] for m_name in model_names + ['y']}

    # Linear model
    (local_X_train, local_y_train), (local_X_val, local_y_val), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'val', 'test'], verbose=verbose)
    for ds_index in tqdm.trange(len(local_X_train), desc='process local models'):
        m = LinearRegression()
        m.fit(local_X_train[ds_index], local_y_train[ds_index])

        train_preds = m.predict(local_X_train[ds_index]).reshape(local_y_train[ds_index].shape)
        val_preds = m.predict(local_X_val[ds_index]).reshape(local_y_val[ds_index].shape)
        test_preds = m.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape)

        all_train_predictions['linear'].append(train_preds.squeeze())
        all_train_predictions['y'].append(local_y_train[ds_index].squeeze())
        all_val_predictions['linear'].append(val_preds.squeeze())
        all_val_predictions['y'].append(local_y_val[ds_index].squeeze())
        all_test_predictions['linear'].append(test_preds.squeeze())
        all_test_predictions['y'].append(local_y_test[ds_index].squeeze())

        for loss_fn in loss_fn_names:
            loss = eval(loss_fn)(test_preds.squeeze(), local_y_test[ds_index].squeeze())
            all_loss_values[f'linear_{loss_fn}'].append(loss)

    n_ts = len(local_X_train)
    del local_X_train, local_y_train, local_X_val, local_y_val, local_X_test, local_y_test

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

    makedirs('preds', exist_ok=True)
    all_predictions = {
        'train': all_train_predictions,
        'val': all_val_predictions,
        'test': all_test_predictions,
    }
    with open(f'preds/{ds_name}.pickle', 'wb') as f:
        pickle.dump(all_predictions, f)

    df = pd.DataFrame({k: v for k, v in all_loss_values.items() if len(v) > 0})
    if verbose:
        print(df.mean())
        print('---'*30)
    
    makedirs('results', exist_ok=True)
    makedirs('results/basemodel_losses', exist_ok=True)
    df.to_csv(f'results/basemodel_losses/{ds_name}.csv')
    return df

def generate_cdd_plot(loss_fn='rmse'):
    datasets_to_evaluate = ALL_DATASETS
    dfs = []
    models = ['linear', 'fcnn', 'deepar', 'cnn']
    full_model_names = [f'{model_name}_{loss_fn}' for model_name in models]

    for ds_name in datasets_to_evaluate:
        if not exists(f'results/basemodel_losses/{ds_name}.csv'):
            df = evaluate_models(ds_name, verbose=True)
        else:
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
        if not exists(f'results/basemodel_losses/{ds_name}.csv'):
            df = evaluate_models(ds_name, verbose=True)
        else:
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
    for ds_name in ALL_DATASETS:
        evaluate_models(ds_name, verbose=True)
    generate_cdd_plot(loss_fn='rmse')
    generate_latex_table()

if __name__ == '__main__':
    main()
