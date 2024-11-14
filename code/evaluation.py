import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from config import DATASET_HYPERPARAMETERS
from utils import rmse
from preprocessing import load_local_data, load_global_data
from os import makedirs

def evaluate_models(ds_name, verbose=False):
    dsh = DATASET_HYPERPARAMETERS[ds_name]
    L = dsh['L']
    freq = dsh['freq']

    (local_X_train, local_y_train), (_, _), (local_X_test, local_y_test) = load_local_data(ds_name, L=L, H=1, verbose=verbose)
    (_, _), (_, _), (global_X_test, global_y_test) = load_global_data(ds_name, L=L, H=1, freq=freq, verbose=verbose)

    all_loss_values = {'linear': [], 'fcnn': [], 'deepar': []}

    for m_name in ['linear', 'fcnn', 'deepar']:

        if m_name == 'linear':
            for ds_index in range(len(local_X_train)):
                m = LinearRegression()
                m.fit(local_X_train[ds_index], local_y_train[ds_index])
                test_preds = m.predict(local_X_test[ds_index]).reshape(local_y_test[ds_index].shape)
                loss = rmse(test_preds, local_y_test[ds_index])
                all_loss_values['linear'].append(loss)
            continue
        elif m_name == 'deepar':
            try:
                with open(f'models/{ds_name}/deepar.pickle', 'rb') as f:
                    m = pickle.load(f).to('cpu')
                    m.device = 'cpu'
                    m.lstm.flatten_parameters()
            except FileNotFoundError:
                continue
        elif m_name == 'fcnn':
            try:
                with open(f'models/{ds_name}/fcnn.pickle', 'rb') as f:
                    m = pickle.load(f)
                    m.device = 'cpu'
                    m.model = m.model.to('cpu')
            except FileNotFoundError:
                continue
        else:
            raise NotImplementedError('Unknown model', m_name)
        for X_test, y_test in zip(global_X_test, global_y_test):
            if m_name == 'fcnn':
                X_test = X_test.reshape(X_test.shape[0], -1)
            test_preds = m.predict(X_test).reshape(y_test.shape)
            loss = rmse(test_preds, y_test)
            all_loss_values[m_name].append(loss)

    df = pd.DataFrame({k: v for k, v in all_loss_values.items() if len(v) > 0})
    if verbose:
        print(df.mean())
        print('---'*30)
    return df


def main():
    from critdd import Diagrams
    datasets_to_evaluate = ['pedestrian_counts', 'nn5_daily_nomissing']
    dfs = []
    models = ['linear', 'fcnn', 'deepar']

    DS_MAP = {
        'pedestrian_counts': 'Pedestrian Counts',
        'nn5_daily_nomissing': 'NN5 (Daily)',
    }
    MODEL_MAP = {
        'linear': 'Linear',
        'fcnn': 'FCNN',
        'deepar': 'DeepAR',
    }

    for ds_name in datasets_to_evaluate:
        df = evaluate_models(ds_name, verbose=False)
        dfs.append(df[models].to_numpy())

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
        },
    )

if __name__ == '__main__':
    main()
