import torch
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np
import pickle
#from gluonts.model.predictor import Predictor, ParallelizedPredictor
from gluonts.dataset.repository import get_dataset
from config import Ls, tsx_to_gluon
from preprocessing import get_train_data
from tsx.datasets import windowing
from tsx.utils import get_device
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sklearn.linear_model import LinearRegression
from os import makedirs
from chronos import ChronosPipeline

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
    return root_mean_squared_error(*args)

def mse(*args):
    return mean_squared_error(*args)

def mae(*args):
    return mean_absolute_error(*args)

def smape(y_true, y_pred):
    return MeanAbsolutePercentageError(symmetric=True)(y_true, y_pred)

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
    X_train, _, X_test = get_train_data(ds)
    L = Ls[ds_name]
    H = 1

    mus = [np.mean(X) for X in X_train]
    stds = [np.std(X) for X in X_train]

    n_skipped = 0

    loss_functions = {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'SMAPE': smape,
    }

    models = ['linear', 'chronos']

    test_losses = []
    for test_idx in tqdm.trange(len(X_test), desc=f'{ds_name} - generate table'):
        losses = {}

        #X = (X_test[test_idx] - mus[test_idx])/stds[test_idx]
        X = X_test[test_idx]
        if len(X) < L + H:
            n_skipped += 1
            continue
        y_true = X[L:]

        # Load predictions
        try:
            for model_name in models:
                preds = np.load(f'preds/{ds_name}/{model_name}/test/{test_idx}.npy')
                for loss_fn_name, loss_fn in loss_functions.items():
                    losses[(model_name, loss_fn_name)] = loss_fn(y_true, preds)
        except FileNotFoundError:
            n_skipped += 1
            continue

        if relative_error is not None:
            divider = {loss_name: value for (model_name, loss_name), value in losses.items() if model_name == relative_error}
            losses = {(model_name, loss_name): value / divider[loss_name] for (model_name, loss_name), value in losses.items()}

        test_losses.append(losses)

    df = pd.DataFrame(test_losses)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    print(f'{ds_name} skipped {n_skipped}')

    return df

def get_overall_table(datasets, relative_error=None):
    dfs = [evaluate_test(ds_name, relative_error=relative_error).mean() for ds_name in datasets]
    dfs = [pd.DataFrame([df], index=[ds_name]) for ds_name, df in zip(datasets, dfs)]
    for df in dfs:
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    final_df = pd.concat(dfs)
    print(final_df)

def get_linear_predictions(ds_name):
    # Get data
    ds = get_dataset(tsx_to_gluon[ds_name])
    X_train, X_val, X_test = get_train_data(ds)
    L = Ls[ds_name]
    H = 1

    mus = [np.mean(X) for X in X_train]
    stds = [np.std(X) for X in X_train]

    makedirs(f'preds/{ds_name}/linear/val', exist_ok=True)
    makedirs(f'preds/{ds_name}/linear/test', exist_ok=True)

    for ds_index in tqdm.trange(len(X_train), desc=f'[{ds_name}] linear predictions'):
        try:
            x_train, y_train = windowing(X_train[ds_index], L=L, H=H)
            x_val, y_val = windowing(X_val[ds_index], L=L, H=H)
            x_test, y_test = windowing(X_test[ds_index], L=L, H=H)
            if len(x_val.shape) <= 1:
                x_val = x_val.reshape(1, -1)
                y_val = y_val.reshape(H)
            if len(x_test.shape) <= 1:
                x_test = x_test.reshape(1, -1)
                y_test = y_test.reshape(H)
        except RuntimeError:
            continue

        lin = LinearRegression()
        lin.fit(x_train, y_train)
        val_preds = lin.predict(x_val).reshape(-1)
        test_preds = lin.predict(x_test).reshape(-1)
        np.save(f'preds/{ds_name}/linear/val/{ds_index}.npy', val_preds)
        np.save(f'preds/{ds_name}/linear/test/{ds_index}.npy', test_preds)

def get_chronos_predictions(ds_name, n_samples=5):
    # Get data
    ds = get_dataset(tsx_to_gluon[ds_name])
    X_train, X_val, X_test = get_train_data(ds)
    L = Ls[ds_name]
    H = 1

    mus = [np.mean(X) for X in X_train]
    stds = [np.std(X) for X in X_train]

    # Instantiate model
    device = get_device()
    pipeline = ChronosPipeline.from_pretrained(
        'amazon/chronos-t5-base',
        device_map=device, 
        torch_dtype=torch.bfloat16,
    )

    makedirs(f'preds/{ds_name}/chronos/val', exist_ok=True)
    makedirs(f'preds/{ds_name}/chronos/test', exist_ok=True)

    for ds_idx, X in tqdm.tqdm(enumerate(X_val), desc=f'[{ds_name}] chronos val', total=len(X_val)):
        try:
            x, _ = windowing(X, L=L, H=H)
        except RuntimeError:
            continue
        x = torch.from_numpy(x).float()
        preds = pipeline.predict(context=x, prediction_length=H, num_samples=n_samples).mean(axis=1).reshape(-1)
        np.save(f'preds/{ds_name}/chronos/val/{ds_idx}.npy', preds)

    for ds_idx, X in tqdm.tqdm(enumerate(X_test), desc=f'[{ds_name}] chronos test', total=len(X_test)):
        try:
            x, _ = windowing(X, L=L, H=H)
        except RuntimeError:
            continue
        x = torch.from_numpy(x).float()
        preds = pipeline.predict(context=x, prediction_length=H, num_samples=n_samples).mean(axis=1).reshape(-1)
        np.save(f'preds/{ds_name}/chronos/test/{ds_idx}.npy', preds)
        
if __name__ == '__main__':
    get_chronos_predictions('nn5_daily')
    get_chronos_predictions('tourism_monthly')
    # get_chronos_predictions('weather')
    # get_chronos_predictions('dominick')

    get_linear_predictions('nn5_daily')
    get_linear_predictions('tourism_monthly')
    # get_linear_predictions('weather')
    # get_linear_predictions('dominick')

    get_overall_table(['nn5_daily', 'tourism_monthly'])

    #get_val_preds('weather')
    # for ds_name in all_datasets:
    #     get_val_preds(ds_name)
    #get_overall_table(all_datasets)