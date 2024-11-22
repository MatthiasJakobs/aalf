import pickle
import torch
import numpy as np
import pandas as pd
import io

from sklearn.linear_model import LinearRegression
from config import DATASET_HYPERPARAMETERS
from utils import rmse, smape
from preprocessing import load_local_data, load_global_data
from os import makedirs
from os.path import exists
from critdd import Diagrams
from config import ALL_DATASETS, DS_MAP, MODEL_MAP, LOSS_MAP
from itertools import product
from models import MultivariateLinearModel


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
    model_names = ['linear', 'fcnn', 'deepar']

    (local_X_train, local_y_train), (_, _), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, return_split=['train', 'test'], verbose=verbose)
    (_, _), (_, _), (global_X_test, global_y_test) = load_global_data(ds_name, L=L, H=1, freq=freq, return_split=['test'], verbose=verbose)

    all_loss_values = {f'{m_name}_{loss_name}': [] for m_name, loss_name in product(model_names, loss_fn_names)}
    all_predictions = {m_name: [] for m_name in model_names + ['y']}

    for m_name in model_names:
        if m_name == 'linear':
            for ds_index in range(len(local_X_train)):
                m = MultivariateLinearModel()
                m.fit(local_X_train[ds_index], local_y_train[ds_index])
                test_preds = m.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape)
                all_predictions[m_name].append(test_preds.squeeze())
                all_predictions['y'].append(local_y_test[ds_index].squeeze())
                for loss_fn in loss_fn_names:
                    loss = eval(loss_fn)(test_preds.squeeze(), local_y_test[ds_index].squeeze())
                    all_loss_values[f'linear_{loss_fn}'].append(loss)
            continue
        elif m_name == 'deepar':
            try:
                with open(f'models/{ds_name}/deepar.pickle', 'rb') as f:
                    m = CPU_Unpickler(f).load().to('cpu')
                    m.device = 'cpu'
                    m.lstm.flatten_parameters()
            except FileNotFoundError:
                continue
        elif m_name == 'fcnn':
            try:
                with open(f'models/{ds_name}/fcnn.pickle', 'rb') as f:
                    m = CPU_Unpickler(f).load()
                    m.device = 'cpu'
                    m.model = m.model.to('cpu')
            except FileNotFoundError:
                continue
        else:
            raise NotImplementedError('Unknown model', m_name)
        for X_test, y_test in zip(global_X_test, global_y_test):
            if m_name == 'fcnn':
                X_test = X_test.reshape(X_test.shape[0], -1)
                y_test = y_test.reshape(y_test.shape[0], -1)
            test_preds = m.predict(X_test).reshape(y_test.shape)
            all_predictions[m_name].append(test_preds.squeeze())
            for loss_fn in loss_fn_names:
                loss = eval(loss_fn)(test_preds.squeeze(), y_test.squeeze())
                all_loss_values[f'{m_name}_{loss_fn}'].append(loss)

    makedirs('preds', exist_ok=True)
    with open(f'preds/{ds_name}.pickle', 'wb') as f:
        pickle.dump(all_predictions, f)

    df = pd.DataFrame({k: v for k, v in all_loss_values.items() if len(v) > 0})
    if verbose:
        print(df.mean())
        print('---'*30)
    
    makedirs('results', exist_ok=True)
    df.to_csv(f'results/{ds_name}.csv')
    return df

def generate_cdd_plot(loss_fn='rmse'):
    datasets_to_evaluate = ALL_DATASETS
    dfs = []
    models = ['linear', 'fcnn', 'deepar']
    full_model_names = [f'{model_name}_{loss_fn}' for model_name in models]

    for ds_name in datasets_to_evaluate:
        if not exists(f'results/{ds_name}.csv'):
            df = evaluate_models(ds_name, verbose=True)
        else:
            df = pd.read_csv(f'results/{ds_name}.csv', index_col=0)
        # Add loss function name to index and remap
        df = df[full_model_names]
        df = df.rename({full_model_name: small_model_name for full_model_name, small_model_name in zip(full_model_names, models)}, axis=1)
        dfs.append(df.to_numpy())

    two_dimensional_diagram = Diagrams(
        dfs,
        diagram_names=[DS_MAP[ds_name] for ds_name in datasets_to_evaluate],
        treatment_names=[MODEL_MAP[model_name] for model_name in models],
        maximize_outcome=False,
    )
    makedirs('plots', exist_ok=True)
    two_dimensional_diagram.to_file(
        'plots/2d_example.tex',
        axis_options={
            'width': '372.0pt',
            'title': loss_fn
        },
    )

def generate_latex_table():
    datasets_to_evaluate = ALL_DATASETS
    dfs = []
    models = ['linear', 'fcnn', 'deepar']
    #loss_function_names = ['rmse', 'smape']
    #full_model_names = [f'{model_name}_{loss_fn}' for model_name in models]

    for ds_name in datasets_to_evaluate:
        if not exists(f'results/{ds_name}.csv'):
            df = evaluate_models(ds_name, verbose=True)
        else:
            df = pd.read_csv(f'results/{ds_name}.csv', index_col=0)
        # Take mean per loss
        df = df.mean(axis=0)
        # Split name to multiindex
        multiindex_dict = {(MODEL_MAP[k.split('_')[0]], LOSS_MAP[k.split('_')[1]]): v for k, v in df.to_dict().items() }
        df = pd.DataFrame([multiindex_dict], index=[DS_MAP[ds_name]])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        dfs.append(df)

    print(pd.concat(dfs))

def main():
    # generate_cdd_plot(loss_fn='rmse')
    for ds_name in ['fred_md']:
        evaluate_models(ds_name, verbose=True)
    #generate_latex_table()

if __name__ == '__main__':
    main()
