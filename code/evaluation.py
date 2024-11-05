import torch
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np
import pickle
from gluonts.mx.model.predictor import Predictor
#from gluonts.model.predictor import Predictor, ParallelizedPredictor
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from config import Ls, tsx_to_gluon, all_gluon_models, all_datasets
from preprocessing import get_train_data
from pathlib import Path
from tsx.datasets import windowing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from os import makedirs

class Predictor:

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            out = self.model(X).squeeze().numpy()

        return out

def rmse(*args):
    return mean_squared_error(*args, squared=False)

def mse(*args):
    return mean_squared_error(*args)

def mae(*args):
    return mean_absolute_error(*args)

def smape(y_true, y_pred):
    nom = np.abs(y_pred - y_true)
    denom = (np.abs(y_pred) + np.abs(y_true)) / 2
    N = len(y_true)
    return 100/N * (nom/denom).sum()

def load_pickled(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def load_global_model(model_name, ds_name):
    if model_name not in ['fcn']:
        raise NotImplementedError('Cannot load unknown model', model_name)
    path = f'models/global/{ds_name}/{model_name}.pth'

    if model_name == 'fcn':
        L = Ls[ds_name]
        model = nn.Sequential(
            nn.Linear(L, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        model.load_state_dict(torch.load(path, weights_only=True))

    # Wrap in predictor
    model = Predictor(model)
    return model

def get_val_preds(ds_name):
    ds = get_dataset(tsx_to_gluon[ds_name])
    _, X_val, _ = get_train_data(ds)
    L = Ls[ds_name]
    H = 1

    # Load models
    global_models = {model_name: load_global_model(model_name, ds_name) for model_name in ['fcn']}
    local_models = {model_name: load_pickled(f'models/local/{ds_name}/{model_name}.pickle') for model_name in ['linear']}

    makedirs(f'preds/', exist_ok=True)

    ds_predictions = {m_name: [] for m_name in (global_models | local_models).keys()}
    for val_idx in tqdm.trange(len(X_val)):

        X = X_val[val_idx]
        x, _ = windowing(X, L=L, H=H)

        # Apply global models
        for gm_name, predictor in global_models.items():
            # preds = evaluate_on_windows(predictor, x, ds.metadata.freq).squeeze().reshape(-1)
            preds = predictor.predict(x)
            ds_predictions[gm_name].append(preds)

        # Apply local models
        for lm_name, model_list in local_models.items():
            model = model_list[val_idx]
            preds = model.predict(x).squeeze().reshape(-1)
            ds_predictions[lm_name].append(preds)

    with open(f'preds/{ds_name}.pickle', 'wb') as f:
        pickle.dump(ds_predictions, f)

def evaluate_test(ds_name, relative_error='linear'):
    ds = get_dataset(tsx_to_gluon[ds_name])
    _, _, X_test = get_train_data(ds, gluon=False)
    L = Ls[ds_name]
    H = 1

    loss_functions = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'SMAPE': smape,
    }

    # Load models
    global_models = {model_name: load_global_model(model_name, ds_name) for model_name in ['fcn']}
    local_models = {model_name: load_pickled(f'models/local/{ds_name}/{model_name}.pickle') for model_name in ['linear']}

    test_losses = []
    for test_idx in tqdm.trange(len(X_test), desc=f'{ds_name} - generate table'):
        losses = {}

        X = X_test[test_idx]
        if len(X) < L + H:
            continue
        x, y = windowing(X, L=L, H=H)
        if len(X) == L+H:
            x = x.reshape(1, -1)
            y = y.reshape(1)

        # Apply global models
        for gm_name, predictor in global_models.items():
            preds = predictor.predict(x).squeeze()
            if len(preds.shape) == 0:
                preds = preds.reshape(1)
            for loss_fn_name, loss_fn in loss_functions.items():
                losses[(gm_name, loss_fn_name)] = loss_fn(y, preds)

        # Apply local models
        for lm_name, model_list in local_models.items():
            model = model_list[test_idx]
            preds = model.predict(x).squeeze()
            if len(preds.shape) == 0:
                preds = preds.reshape(1)
            for loss_fn_name, loss_fn in loss_functions.items():
                losses[(lm_name, loss_fn_name)] = loss_fn(y, preds)

        if relative_error is not None:
            divider = {loss_name: value for (model_name, loss_name), value in losses.items() if model_name == relative_error}
            losses = {(model_name, loss_name): value / divider[loss_name] for (model_name, loss_name), value in losses.items()}
        test_losses.append(losses)

    df = pd.DataFrame(test_losses)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    return df

def get_overall_table(datasets):
    dfs = [evaluate_test(ds_name).mean() for ds_name in datasets]
    dfs = [pd.DataFrame([df], index=[ds_name]) for ds_name, df in zip(datasets, dfs)]
    for df in dfs:
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    final_df = pd.concat(dfs)
    print(final_df)

if __name__ == '__main__':
    #get_val_preds('weather')
    for ds_name in all_datasets:
        get_val_preds(ds_name)
    #get_overall_table(all_datasets)